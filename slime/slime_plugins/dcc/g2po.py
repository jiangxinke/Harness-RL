"""Adapt exact-observation G2PO credit to DCC's recorded policy decisions."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from types import SimpleNamespace

import numpy as np
import torch

from .credit_adapter import CreditVisit
from .credit_numerics import zero_constant_groups
from .credit_upstream import core_g2po


UPSTREAM_COMMIT = "b8ffaf56ad241ce73507908e6ed30209cafd324d"


def _validated_order(visits):
    trajectories = defaultdict(list)
    for index, visit in enumerate(visits):
        if not isinstance(visit.task_id, str) or not visit.task_id:
            raise ValueError("G2PO visits require a nonempty task_id")
        if not isinstance(visit.episode, int) or not isinstance(visit.step, int) or visit.step < 0:
            raise ValueError("G2PO episode and step indices must be integers, with step >= 0")
        if not isinstance(visit.observation, str) or not isinstance(visit.next_observation, str):
            raise ValueError("G2PO observations must be strings")
        if visit.is_action_valid not in (False, True) or visit.terminal not in (False, True):
            raise ValueError("G2PO action-valid and terminal flags must be boolean")
        if visit.success not in (False, True) or visit.task_reward != (10.0 if visit.success else 0.0):
            raise ValueError("G2PO requires binary terminal rewards: success=10, failure=0")
        trajectories[(visit.task_id, visit.episode)].append((visit.step, index))
    ordered = []
    task_counts = defaultdict(int)
    for (task, _episode), indices in sorted(trajectories.items()):
        indices.sort()
        if [step for step, _ in indices] != list(range(len(indices))):
            raise ValueError("G2PO requires complete trajectories with consecutive steps from zero")
        trajectory = [visits[index] for _, index in indices]
        if any(bool(v.terminal) != (i == len(trajectory) - 1) for i, v in enumerate(trajectory)):
            raise ValueError("G2PO requires exactly one terminal visit, at the end of each trajectory")
        if len({(v.task_reward, v.success) for v in trajectory}) != 1:
            raise ValueError("G2PO task outcome must be consistent across a trajectory")
        if any(a.next_observation != b.observation for a, b in zip(trajectory, trajectory[1:])):
            raise ValueError("G2PO next_observation must match the next decision's observation")
        ordered.extend(index for _, index in indices)
        task_counts[task] += len(indices)
    if any(count < 2 for count in task_counts.values()):
        # The upstream edge standard deviation is undefined for one visit.
        raise ValueError("G2PO needs at least two recorded visits per task")
    return ordered


def compute_advantages(
    visits: list[CreditVisit], *, weight=1.0, mode="mean_std_norm", invalid_penalty=0.1,
):
    """Return upstream credit and separately auditable episode, node and edge terms."""
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("G2PO step weight must be finite and nonnegative")
    if not math.isfinite(invalid_penalty) or invalid_penalty < 0:
        raise ValueError("G2PO invalid-action penalty must be finite and nonnegative")
    if mode not in ("mean_norm", "mean_std_norm"):
        raise ValueError("G2PO normalization must be mean_norm or mean_std_norm")
    metadata = dict(method="g2po", upstream_commit=UPSTREAM_COMMIT, gamma=0.95,
                    weight=weight, mode=mode, invalid_penalty=invalid_penalty,
                    episode_weighting="visits", std_ddof=1, epsilon=1e-6,
                    grouping="exact_current_observation", reward_scale="binary_10_0")
    if not visits:
        return [], {**metadata, "rows": [], "tasks": [], "nodes": []}
    order = _validated_order(visits)
    records = [visits[index] for index in order]
    count = len(records)
    uid = np.array([v.task_id for v in records])
    # Upstream looks up trajectories without filtering their task IDs.
    trajectory_ids = {key: f"trajectory-{i}" for i, key in enumerate(
        dict.fromkeys((v.task_id, v.episode) for v in records))}
    traj_uid = np.array([trajectory_ids[(v.task_id, v.episode)] for v in records])
    rewards = np.array([v.task_reward for v in records], dtype=np.float64)
    valid = np.array([v.is_action_valid for v in records], dtype=np.int64)
    batch = SimpleNamespace(
        batch={"responses": torch.zeros((count, 1), dtype=torch.float32)},
        non_tensor_batch=dict(uid=uid, traj_uid=traj_uid, episode_rewards=rewards,
                              anchor_obs=np.array([v.observation for v in records])),
    )
    values, groups, next_groups = core_g2po.compute_group_aggregation_values(batch)
    raw_episode = rewards - invalid_penalty * (1 - valid)
    mask = torch.ones((count, 1), dtype=torch.float32)
    episode_normalization_scores = torch.tensor(raw_episode, dtype=torch.float32)
    episode = core_g2po.episode_norm_reward(
        episode_normalization_scores[:, None], mask, uid, traj_uid,
        remove_std=mode == "mean_norm",
    )[:, 0]
    step = core_g2po.compute_step_level_advantage(
        values, mask, valid, uid, groups, next_groups, mode=mode,
        invalid_action_penalty=invalid_penalty,
    )[:, 0]
    combined = episode + weight * step

    node = torch.zeros(count, dtype=torch.float32)
    edge = torch.zeros(count, dtype=torch.float32)
    next_values = np.zeros(count)
    raw_node = np.zeros(count)
    raw_edge = np.zeros(count)
    task_diagnostics, node_diagnostics = [], []
    for task in np.unique(uid):
        indices = np.where(uid == task)[0]
        members = defaultdict(list)
        node_values = {-1: 10.0, -2: 0.0}
        for i in indices:
            members[int(groups[i])].append(i)
            node_values[int(groups[i])] = float(values[i])
        for group, group_indices in members.items():
            target_values = np.array([node_values[int(next_groups[i])] for i in group_indices])
            local_raw = target_values - invalid_penalty * (1 - valid[group_indices])
            next_values[group_indices] = target_values
            raw_node[group_indices] = local_raw
            if len(group_indices) > 1:
                tensor = torch.tensor(local_raw)
                local = tensor - tensor.mean()
                if mode == "mean_std_norm":
                    local /= tensor.std() + 1e-6
                node[group_indices] = local.to(torch.float32)
            node_diagnostics.append(dict(
                task_id=str(task), node=group, value=node_values[group], visits=len(group_indices),
                trajectories=len({records[i].episode for i in group_indices}),
                observation_hash=hashlib.sha256(records[group_indices[0]].observation.encode()).hexdigest(),
                member_indices=[order[i] for i in group_indices],
            ))
        gains = [node_values[int(next_groups[i])] - node_values[int(groups[i])] for i in indices]
        raw_edge[indices] = gains
        tensor = torch.tensor(gains)
        global_advantage = tensor - tensor.mean()
        if mode == "mean_std_norm":
            global_advantage /= tensor.std() + 1e-6
        edge[indices] = global_advantage
        task_diagnostics.append(dict(
            task_id=str(task), node_count=len(members), edge_visits=len(indices),
            unique_node_pairs=len({(int(groups[i]), int(next_groups[i])) for i in indices}),
            self_loop_visits=sum(int(groups[i] == next_groups[i]) for i in indices),
            success_terminals=sum(int(next_groups[i] == -1) for i in indices),
            failure_terminals=sum(int(next_groups[i] == -2) for i in indices),
        ))
    # Verify the decomposition before the explicit constant-group correction.
    torch.testing.assert_close(step, node + edge, rtol=1e-6, atol=1e-6)
    original_episode, original_node, original_edge = episode.clone(), node.clone(), edge.clone()
    original_step, original_combined = step.clone(), combined.clone()
    episode, episode_fixes = zero_constant_groups(episode_normalization_scores, uid, episode)
    node, node_fixes = zero_constant_groups(
        raw_node, [(str(task), int(group)) for task, group in zip(uid, groups)], node,
    )
    # The author's edge normalization constructs a float32 tensor from gains.
    edge_normalization_scores = torch.tensor(raw_edge, dtype=torch.float32)
    edge, edge_fixes = zero_constant_groups(edge_normalization_scores, uid, edge)
    step = node + edge
    combined = episode + weight * step
    if not torch.isfinite(combined).all():
        raise ValueError("G2PO produced nonfinite advantages")
    lengths = defaultdict(int)
    for v in records:
        lengths[(v.task_id, v.episode)] += 1
    scores, rows = [0.0] * count, [None] * count
    for i, original_index in enumerate(order):
        v = records[i]
        scores[original_index] = float(combined[i])
        rows[original_index] = dict(
            task_id=v.task_id, episode=v.episode, step=v.step,
            node=int(groups[i]), next_node=int(next_groups[i]), terminal=bool(v.terminal),
            success=bool(v.success), task_reward=float(v.task_reward), is_action_valid=bool(v.is_action_valid),
            observation_hash=hashlib.sha256(v.observation.encode()).hexdigest(),
            discounted_return=0.95 ** (lengths[(v.task_id, v.episode)] - v.step) * v.task_reward,
            node_value=float(values[i]), next_value=float(next_values[i]),
            episode_raw_score=float(raw_episode[i]), node_raw_score=float(raw_node[i]),
            episode_normalization_score=float(episode_normalization_scores[i]),
            edge_normalization_score=float(edge_normalization_scores[i]),
            edge_raw_gain=float(raw_edge[i]), episode_advantage=float(episode[i]),
            node_advantage=float(node[i]), edge_advantage=float(edge[i]),
            local_advantage=float(step[i]), advantage=scores[original_index],
            upstream_episode_advantage=float(original_episode[i]),
            upstream_node_advantage=float(original_node[i]),
            upstream_edge_advantage=float(original_edge[i]),
            upstream_local_advantage=float(original_step[i]),
            upstream_advantage=float(original_combined[i]),
        )
    corrections = {
        name: [{**fix, "indices": [order[i] for i in fix["indices"]]} for fix in fixes]
        for name, fixes in (("episode", episode_fixes), ("node", node_fixes), ("edge", edge_fixes))
    }
    return scores, {**metadata, "rows": rows, "nodes": node_diagnostics, "tasks": task_diagnostics,
                    "numeric_guard": "exact_constant_groups_zero_v1",
                    "normalization_corrections": corrections,
                    "normalization_correction_index_basis": "returned rows / original input visits"}
