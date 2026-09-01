"""Reward normalization helpers for rollouts that fan out into Samples.

The training backend permits one logical rollout (episode) to emit multiple
Samples.  Reward statistics must therefore operate on unique rollout ids and
only then broadcast the resulting advantage to the rollout's Samples; treating
every Sample as an observation biases GRPO toward episodes with more segments.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch


def normalize_group_rewards_by_rollout(
    samples: list[Any],
    raw_rewards: list[float],
    *,
    n_samples_per_prompt: int,
    normalize_std: bool,
) -> list[float]:
    """Normalize rewards per prompt while counting each rollout once.

    ``group_index`` identifies the prompt/GRPO group and ``rollout_id``
    identifies one sampled episode.  Multiple Samples with the same pair are
    trajectory segments (for example Central plus sub-agent trajectories), so
    they receive the same normalized reward and do not change group statistics.

    Legacy/custom inputs without these ids retain the ordinary fixed-size
    grouping behavior: each Sample is treated as a distinct rollout and groups
    are formed in chunks of ``n_samples_per_prompt``.
    """
    if len(samples) != len(raw_rewards):
        raise ValueError(f"samples/rewards length mismatch: {len(samples)} != {len(raw_rewards)}")
    if not samples:
        return []

    chunk_size = max(int(n_samples_per_prompt), 1)
    groups: OrderedDict[object, list[int]] = OrderedDict()
    for position, sample in enumerate(samples):
        group_index = getattr(sample, "group_index", None)
        group_key = ("group", group_index) if group_index is not None else ("legacy", position // chunk_size)
        groups.setdefault(group_key, []).append(position)

    normalized = [0.0] * len(samples)
    for positions in groups.values():
        rollout_positions: OrderedDict[object, list[int]] = OrderedDict()
        for position in positions:
            rollout_id = getattr(samples[position], "rollout_id", None)
            # An absent rollout id means an ordinary one-Sample rollout.  Never
            # collapse all ``None`` ids into a single episode.
            rollout_key = ("rollout", rollout_id) if rollout_id is not None else ("sample", position)
            rollout_positions.setdefault(rollout_key, []).append(position)

        episode_rewards: list[float] = []
        for rollout_key, sibling_positions in rollout_positions.items():
            sibling_rewards = [float(raw_rewards[position]) for position in sibling_positions]
            if any(reward != sibling_rewards[0] for reward in sibling_rewards[1:]):
                raise ValueError(
                    f"Samples sharing {rollout_key!r} have inconsistent episode rewards: {sibling_rewards}"
                )
            episode_rewards.append(sibling_rewards[0])

        rewards = torch.tensor(episode_rewards, dtype=torch.float)
        rewards = rewards - rewards.mean()
        # torch.std's unbiased estimate is undefined for a singleton group.
        # Its only centered advantage is exactly zero.
        if normalize_std and rewards.numel() > 1:
            rewards = rewards / (rewards.std() + 1e-6)

        for advantage, sibling_positions in zip(rewards.tolist(), rollout_positions.values(), strict=True):
            for position in sibling_positions:
                normalized[position] = advantage

    return normalized
