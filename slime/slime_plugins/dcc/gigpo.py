"""GiGPO's published computation applied to recorded CUAD decisions."""
from __future__ import annotations

import contextlib
import io
import math
from types import SimpleNamespace

import numpy as np
import torch

from .credit_adapter import build_credit_visits, unpack_scores
from .credit_upstream import core_gigpo as upstream


def compute_advantages(visits, *, gamma=.95, weight=1.0,
                       mode="mean_std_norm", similarity_threshold=.9,
                       invalid_penalty=.1):
    if (not 0 < gamma <= 1 or not math.isfinite(weight) or weight < 0
            or not 0 < similarity_threshold <= 1 or not math.isfinite(invalid_penalty)
            or invalid_penalty < 0 or mode not in {"mean_norm", "mean_std_norm"}):
        raise ValueError("Invalid GiGPO advantage configuration")
    if not visits:
        return [], {"rows": [], "groups": []}
    trajectories = {}
    for visit in visits:
        trajectories.setdefault((visit.task_id, visit.episode), []).append(visit)
    for trajectory in trajectories.values():
        if ([v.step for v in trajectory] != list(range(len(trajectory)))
                or [v.terminal for v in trajectory] != [False] * (len(trajectory) - 1) + [True]
                or len({(v.success, v.task_reward) for v in trajectory}) != 1
                or trajectory[0].task_reward != 10.0 * trajectory[0].success
                or any(a.next_observation != b.observation for a, b in zip(trajectory, trajectory[1:]))):
            raise ValueError("GiGPO requires ordered, complete, consistent binary-reward trajectories")
    task_keys = np.array([v.task_id for v in visits], dtype=object)
    trajectory_keys = np.array([f"{v.task_id}:{v.episode}" for v in visits], dtype=object)
    mask = torch.ones((len(visits), 1), dtype=torch.float32)
    invalid = torch.tensor([not v.is_action_valid for v in visits], dtype=torch.float32)
    episode_scores = torch.tensor([v.task_reward for v in visits], dtype=torch.float32)
    raw_rewards = np.array([v.task_reward if v.terminal else 0.0 for v in visits], dtype=np.float32)
    data = SimpleNamespace(batch={"input_ids": mask}, non_tensor_batch={
        "rewards": raw_rewards, "traj_uid": trajectory_keys,
        "active_masks": np.ones(len(visits), dtype=np.float32),
    })
    discounted = upstream.compute_step_discounted_returns(data, gamma)
    # The author trainer applies this after discounted returns are computed.
    episode_scores = episode_scores - invalid_penalty * invalid
    local_scores = discounted - invalid_penalty * invalid
    remove_std = mode == "mean_norm"
    episode = upstream.episode_norm_reward(episode_scores[:, None], mask,
        task_keys, trajectory_keys, remove_std=remove_std).flatten()
    # Final-only and tool-enabled decisions have different admissible actions.
    # Keep that environment distinction outside approximate text matching.
    phase_keys = np.array([v.task_id + ":" + v.observation.split("\n", 1)[0]
                           for v in visits], dtype=object)
    observations = np.array([v.observation for v in visits], dtype=object)
    with contextlib.redirect_stdout(io.StringIO()):
        groups = upstream.build_step_group(observations, phase_keys,
            enable_similarity=similarity_threshold < 1,
            similarity_thresh=similarity_threshold)
    local = upstream.step_norm_reward(local_scores, mask, groups,
                                     remove_std=remove_std).flatten()
    combined = episode + weight * local
    group_ids = {key: i for i, key in enumerate(dict.fromkeys(groups))}
    rows = [dict(episode=v.episode, step=v.step, group=group_ids[groups[i]],
        task_id=v.task_id, is_action_valid=v.is_action_valid,
        task_reward=v.task_reward, discounted_return=float(discounted[i]),
        episode_score=float(episode_scores[i]), local_score=float(local_scores[i]),
        episode_advantage=float(episode[i]), local_advantage=float(local[i]),
        advantage=float(combined[i]), observation=v.observation,
        prompt_hash=v.prompt_hash, evidence_hash=v.evidence_hash)
        for i, v in enumerate(visits)]
    return combined.tolist(), dict(rows=rows,
        groups=[[i for i, key in enumerate(groups) if key == group]
                for group in group_ids],
        upstream_commit="20bd331bdbc9026a5668e11362178e10ab7400c8",
        mode=mode, invalid_penalty=invalid_penalty,
        similarity_threshold=similarity_threshold, gamma=gamma, weight=weight)


def step_advantages(results, *, tokenizer, gamma=.95, weight=1.0,
                    similarity_threshold=.9, invalid_penalty=.1,
                    mode="mean_std_norm"):
    visits, diagnostics = build_credit_visits(results, tokenizer=tokenizer)
    scores, components = compute_advantages(visits, gamma=gamma, weight=weight,
        similarity_threshold=similarity_threshold, invalid_penalty=invalid_penalty, mode=mode)
    return unpack_scores(results, visits, scores), {**diagnostics, **components}
