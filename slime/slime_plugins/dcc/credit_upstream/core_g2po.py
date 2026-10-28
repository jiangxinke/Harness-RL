# Vendored from Nala-YN/G2PO, commit b8ffaf56ad241ce73507908e6ed30209cafd324d.
# Source: g2po/core_g2po.py. Only the annotation-only verl import is replaced.
# Licensed under Apache-2.0: https://www.apache.org/licenses/LICENSE-2.0
# The complete upstream LICENSE, NOTICE and original source are preserved in
# third_party/g2po-source.json.
# Upstream NOTICE: Copyright 2023-2024 Bytedance Ltd. and/or its affiliates;
# Copyright 2025 Nanyang Technological University (NTU), Singapore and the
# verl-agent (GiGPO) team.
# Distributed on an AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.

import numpy as np
import torch
from collections import defaultdict
from typing import Any as DataProto

SUCCESS_REWARD = 10.0


def episode_norm_reward(token_level_rewards: torch.Tensor,
                        response_mask: torch.Tensor,
                        index: np.array,
                        traj_index: np.array,
                        epsilon: float = 1e-6,
                        remove_std: bool = True,
                        compute_mean_std_cross_steps: bool = True,
                        ):
    """Normalize episode-level rewards by subtracting the per-prompt mean.

    For each prompt (identified by index), collects all trajectory scores and
    normalizes them. When compute_mean_std_cross_steps is False, only the first
    occurrence of each (index, traj_index) pair contributes to the statistics.
    When remove_std is True, only the mean is subtracted; otherwise z-score
    normalization is applied.
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        # Collect scores per prompt, optionally deduplicating by trajectory.
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))

        # Compute per-prompt statistics.
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        # Normalize each score by its prompt's statistics.
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

        episode_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask

    return episode_advantages


def compute_group_aggregation_values(batch: DataProto) -> torch.Tensor:
    """Assign group IDs by anchor observation and compute discounted group scores.

    Steps within trajectories are grouped by their anchor observations.
    A discounted sum (gamma=0.95) of episode rewards is accumulated per group,
    then averaged across members. Also computes next_group_idx:
      -1 for terminal success states, -2 for terminal failure states,
      or the group index of the next step within the same trajectory.
    """
    episode_rewards = batch.non_tensor_batch['episode_rewards']
    response_tensor = batch.batch["responses"]
    anchor_obs = batch.non_tensor_batch["anchor_obs"]
    traj_index = batch.non_tensor_batch["traj_uid"]
    index = batch.non_tensor_batch["uid"]
    bs = response_tensor.shape[0]
    unique_index = np.unique(index)
    scores = np.zeros(bs)
    steps = np.zeros(bs)
    group_idx = np.ones(bs, dtype=int) * (-1)
    next_group_idx = np.ones(bs, dtype=int)
    for idx in unique_index:
        indices = np.where(index == idx)[0]
        unique_traj_index = np.unique(traj_index[indices])
        groups = defaultdict(list)
        obs2idx = {}
        group_count = 0

        # Assign group IDs: same anchor observation -> same group.
        for t_idx in unique_traj_index:
            traj_indices = np.where(traj_index == t_idx)[0]
            for step_count, traj_idx in enumerate(traj_indices):
                steps[traj_idx] = step_count
                obs = anchor_obs[traj_idx]
                if obs2idx.get(obs) is None:
                    obs2idx[obs] = group_count
                    group_idx[traj_idx] = group_count
                    groups[obs2idx[obs]].append(traj_idx)
                    group_count += 1
                else:
                    group_idx[traj_idx] = obs2idx[obs]
                    groups[obs2idx[obs]].append(traj_idx)

        group_score = np.zeros(len(groups))

        # Compute discounted group scores and next_group_idx.
        for t_idx in unique_traj_index:
            traj_indices = np.where(traj_index == t_idx)[0]
            traj_episode_rewards = episode_rewards[traj_indices]
            episode_reward = traj_episode_rewards[0]

            for step_count, traj_idx in enumerate(traj_indices):
                # Link to next step's group, or mark terminal state.
                if step_count + 1 < len(traj_indices):
                    next_group_idx[traj_idx] = group_idx[traj_indices[step_count + 1]]
                else:
                    if episode_rewards[traj_idx] == SUCCESS_REWARD:
                        next_group_idx[traj_idx] = -1  # terminal success
                    else:
                        next_group_idx[traj_idx] = -2  # terminal failure

                group_score[group_idx[traj_idx]] += (0.95 ** (len(traj_indices) - step_count)) * episode_reward

        # Average group scores across members.
        for groupnum in range(group_count):
            for traj_idx in groups[groupnum]:
                scores[traj_idx] = group_score[groupnum] / len(groups[groupnum])

    return scores, group_idx, next_group_idx


def compute_step_level_advantage(
                        step_rewards: torch.Tensor,
                        response_mask: torch.Tensor,
                        is_action_valid: np.array,
                        index: np.array,
                        group_idx: np.array,
                        next_group_idx: np.array,
                        epsilon: float = 1e-6,
                        mode: str = "mean_std_norm",
                        invalid_action_penalty: float = 0.1,
                        ):
    """Compute step-level advantages via group-based comparison.

    Within each prompt, steps in the same group (same anchor observation) are
    compared. The advantage has two components:
    1. Next-group score: the reward of the group this step leads to, normalized
       across all steps within the same group. Invalid actions are penalized.
    2. Score gain: the improvement (next_group_score - current_group_score),
       normalized across all steps in the prompt.

    Terminal groups: -1 maps to SUCCESS_REWARD, -2 maps to 0.
    """
    action_invalids = np.array(1 - is_action_valid, dtype=float)
    response_length = response_mask.shape[-1]
    bs = response_mask.shape[0]
    unique_index = np.unique(index)
    scores = torch.zeros(bs)

    for idx in unique_index:
        indices = np.where(index == idx)[0]
        groups = defaultdict(list)
        groupnum2score = {}

        # Map each group to its step reward (verified consistent across members).
        for traj_idx in indices:
            groupnum = group_idx[traj_idx]
            groups[groupnum].append(traj_idx)
            if groupnum2score.get(groupnum) is None:
                groupnum2score[groupnum] = step_rewards[traj_idx].item()
            else:
                assert groupnum2score[groupnum] == step_rewards[traj_idx].item(), \
                    f"Step reward mismatch in group {groupnum} at traj idx {traj_idx}"

        # Terminal group sentinel values.
        groupnum2score[-2] = 0.0
        groupnum2score[-1] = SUCCESS_REWARD

        # Component 1: normalize next-group scores within each group.
        for groupnum in groups:
            group_indices = groups[groupnum]
            next_group_scores = [groupnum2score[next_group_idx[gi]] for gi in group_indices]
            group_action_invalids = action_invalids[group_indices]
            next_group_scores -= group_action_invalids * invalid_action_penalty
            if len(next_group_scores) > 1:
                scores_mean = torch.mean(torch.tensor(next_group_scores))
                scores_std = torch.std(torch.tensor(next_group_scores))
                for traj_idx, score in zip(groups[groupnum], next_group_scores):
                    if mode == "mean_std_norm":
                        scores[traj_idx] = (score - scores_mean) / (scores_std + epsilon)
                    else:
                        scores[traj_idx] = (score - scores_mean)

        # Component 2: normalize score gains across all steps in the prompt.
        score_gains = []
        for indice in indices:
            group_score = groupnum2score[group_idx[indice]]
            next_group_score = groupnum2score[next_group_idx[indice]]
            score_gain = next_group_score - group_score
            score_gains.append(score_gain)
        score_gain_mean = torch.mean(torch.tensor(score_gains))
        score_gain_std = torch.std(torch.tensor(score_gains))
        for traj_idx, score_gain in zip(indices, score_gains):
            if mode == "mean_std_norm":
                scores[traj_idx] += (score_gain - score_gain_mean) / (score_gain_std + epsilon)
            else:
                scores[traj_idx] += score_gain - score_gain_mean

    step_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
    return step_advantages


def compute_g2po_outcome_advantage(token_level_rewards: torch.Tensor,
                                    step_rewards: np.array,
                                    response_mask: torch.Tensor,
                                    index: np.array,
                                    traj_index: np.array,
                                    group_idx: np.array,
                                    next_group_idx: np.array,
                                    is_action_valid: np.array,
                                    epsilon: float = 1e-6,
                                    step_advantage_w: float = 1.0,
                                    mode: str = "mean_norm",
                                   ):
    """Combine episode-level and step-level advantages for G2PO training.

    Computes the episode advantage via episode_norm_reward, the step advantage
    via compute_step_level_advantage, and returns their weighted sum.
    step_advantage_w controls the relative weight of the step-level signal.
    """
    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    episode_advantages = episode_norm_reward(token_level_rewards, response_mask, index, traj_index, epsilon, remove_std)

    step_advantages = compute_step_level_advantage(step_rewards, response_mask, is_action_valid, index, group_idx, next_group_idx, epsilon, mode)

    scores = episode_advantages + step_advantage_w * step_advantages
    return scores, scores
