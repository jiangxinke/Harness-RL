"""Optional OLE selection through slime's rollout and post-advantage hooks.

Ported from ../OLE/verl/verl/trainer/ppo/ray_trainer.py: score each prompt
group by mean(reward) + k * population_std(reward), reject scores strictly
below the previous batch's threshold, then update the quantile threshold.
Only policy advantages are zeroed; rewards, returns, loss masks and CAPO
routes keep their original meaning.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from typing import Any

import numpy as np

from .config import TrainingConfig

_STATE_KEY = "stackplanner_ole"


def _flatten(samples: list[Any]) -> Iterator[Any]:
    for sample in samples:
        if isinstance(sample, list):
            yield from _flatten(sample)
        else:
            yield sample


def select_prompt_groups(
    args: Any,
    config: TrainingConfig,
    samples: list[Any],
    state: dict[str, Any],
    *,
    rollout_id: int,
) -> dict[str, float | int]:
    """Annotate every segment, counting each valid episode exactly once.

    ``state`` lives in the global data source's checkpointed metadata. The
    nested sample shape and all existing training fields are preserved.
    Infrastructure failures and episodes without actor tokens do not influence
    the score distribution or the next threshold.
    """
    if not config.ole_enabled:
        return {}

    flat_samples = list(_flatten(samples))
    episodes: dict[tuple[int, int], list[Any]] = defaultdict(list)
    for sample in flat_samples:
        group = getattr(sample, "group_index", None)
        episode = getattr(sample, "rollout_id", None)
        if group is None or episode is None:
            raise ValueError("OLE requires group_index and rollout_id on every episode segment")
        episodes[(group, episode)].append(sample)

    group_rewards: dict[int, list[float]] = defaultdict(list)
    valid_episodes: set[tuple[int, int]] = set()
    for key, siblings in episodes.items():
        eligible = []
        for sample in siblings:
            if sample.remove_sample or (sample.metadata or {}).get("infrastructure_failure"):
                continue
            has_actor_tokens = sample.response_length > 0 if sample.loss_mask is None else any(sample.loss_mask)
            if has_actor_tokens:
                eligible.append(sample)
        if not eligible:
            continue
        rewards = [float(sample.get_reward_value(args)) for sample in eligible]
        if not all(np.isfinite(value) for value in rewards):
            raise ValueError("OLE requires finite episode rewards")
        if any(value != rewards[0] for value in rewards[1:]):
            raise ValueError("OLE received inconsistent rewards for segments of one episode")
        group_rewards[key[0]].append(rewards[0])
        valid_episodes.add(key)

    scores = {group: float(np.mean(rewards) + config.ole_k * np.std(rewards)) for group, rewards in group_rewards.items()}
    threshold = float(state.get("threshold", config.ole_init_threshold))
    # Match OLE's float32 quantile input and population (ddof=0) deviation.
    next_threshold = threshold
    if scores:
        score_values = np.array(list(scores.values()), dtype=np.float32)
        next_threshold = float(np.quantile(score_values, 1 - config.ole_threshold_percent))
    rejected = {group for group, score in scores.items() if score < threshold}
    for (group, _), siblings in episodes.items():
        for sample in siblings:
            sample.train_metadata = {
                **(sample.train_metadata or {}),
                "ole": {
                    "selected": group not in rejected,
                    "score": scores.get(group),
                    "threshold": threshold,
                },
            }

    discarded = sum(group in rejected for group, _ in valid_episodes)
    discarded_total = int(state.get("discarded_total", 0)) + discarded
    state.update(threshold=next_threshold, discarded_total=discarded_total, rollout_id=rollout_id)
    return {
        "ole/threshold": threshold,
        "ole/next_threshold": next_threshold,
        "ole/groups": len(scores),
        "ole/discarded_groups": len(rejected),
        "ole/episodes": len(valid_episodes),
        "ole/discarded_episodes": discarded,
        "ole/discarded_total": discarded_total,
    }


def generate_rollout(args: Any, rollout_id: int, data_source: Any, evaluation: bool = False) -> Any:
    """Wrap the standard generator so scoring precedes debug saves and CAPO."""
    from slime.rollout.sglang_rollout import generate_rollout as base_generate

    result = base_generate(args, rollout_id, data_source, evaluation=evaluation)
    if evaluation:
        return result

    from .generate import _load_training_config

    config = _load_training_config()
    if config.ole_enabled:
        state = data_source.metadata.setdefault(_STATE_KEY, {})
        metrics = select_prompt_groups(args, config, result.samples, state, rollout_id=rollout_id)
        result.metrics = {**(result.metrics or {}), **metrics}
    return result


def apply_selection_to_advantages(args: Any, rollout_id: int, rollout_data: dict[str, Any]) -> None:
    """Zero rejected advantages after estimator/whitening, on each DP shard.

    Use new tensors: GRPO's returns can alias advantages. Mutating tensors
    in place would also change critic targets. Retaining every sample keeps
    the batch schedule and per-rollout loss denominator unchanged.
    """
    del args, rollout_id
    advantages = rollout_data.get("advantages")
    if advantages is None:  # Intermediate pipeline stages have no advantages.
        return
    metadata = rollout_data.get("metadata")
    if metadata is None or len(metadata) != len(advantages):
        raise ValueError("OLE requires selection metadata aligned with advantages")
    selected = []
    for item in metadata:
        decision = item.get("ole", {}).get("selected") if isinstance(item, dict) else None
        if not isinstance(decision, bool):
            raise ValueError("OLE selection metadata is missing; replay must use OLE-annotated rollouts")
        selected.append(decision)

    import torch

    rollout_data["advantages"] = [advantage if keep else torch.zeros_like(advantage) for advantage, keep in zip(advantages, selected, strict=True)]
