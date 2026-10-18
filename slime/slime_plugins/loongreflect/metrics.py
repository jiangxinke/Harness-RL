"""Per-outer-step metrics for Reflect tree training."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import torch

from .self_distill import low_variance_reverse_kl

logger = logging.getLogger(__name__)

_ACTION_TAGS = ("think", "search", "reflect", "backtrack")


def summarize_tree_for_training(tree: dict[str, Any]) -> dict[str, Any]:
    """Return compact, non-privileged metrics for one rollout tree.

    Tag counts use unique model turns and their raw model output. Harness-added
    repair nodes are therefore not mislabeled as model-generated actions.
    """
    counts = {tag: 0 for tag in _ACTION_TAGS}
    seen_turns: set[int] = set()
    answer_leaks = 0
    for node in tree.get("nodes", []):
        metadata = node.get("metadata") or {}
        turn_id = metadata.get("model_turn_id")
        if node.get("kind") != "think" or not isinstance(turn_id, int):
            continue
        if turn_id in seen_turns:
            continue
        seen_turns.add(turn_id)
        raw = str(metadata.get("raw_output", ""))
        for tag in _ACTION_TAGS:
            counts[tag] += len(re.findall(rf"<{tag}>", raw, flags=re.IGNORECASE))
        if metadata.get("reflect_answer_leak"):
            answer_leaks += 1

    termination_reason = str(tree.get("termination_reason") or "")
    return {
        **{f"{tag}_count": counts[tag] for tag in _ACTION_TAGS},
        "output_tokens": int(tree.get("output_tokens", 0)),
        "model_turns": int(tree.get("model_turns", len(seen_turns))),
        "completed": int(tree.get("status") == "completed"),
        "truncated": int(termination_reason.startswith(("max_steps_reached", "max_output_tokens_reached"))),
        "answer_leak_count": answer_leaks,
        "termination_reason": termination_reason or None,
    }


def _as_float_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().float()
    return torch.as_tensor(value, dtype=torch.float32)


def _whole_tree_denominator(
    rollout_data: dict[str, Any],
    key: str,
    indices: list[int],
    *,
    local_total: float,
) -> float:
    """Return the pre-partition whole-tree denominator for one DP fragment.

    A tree's compact node samples may be split across data-parallel ranks. The
    rollout manager broadcasts the same whole-tree mask total alongside every
    compact sample, allowing each rank to report an additive fraction of the
    tree mean. Legacy/replay fixtures without the field retain local behavior.
    """
    values = rollout_data.get(key)
    if values is None:
        return local_total
    denominators = {float(_as_float_tensor(values[index]).item()) for index in indices}
    if len(denominators) != 1:
        raise ValueError(f"inconsistent {key} for rollout fragment: {sorted(denominators)}")
    denominator = denominators.pop()
    if denominator < local_total:
        raise ValueError(f"{key}={denominator} is smaller than local total {local_total}")
    return denominator


def compute_local_step_totals(args: Any, rollout_data: dict[str, Any]) -> dict[str, float]:
    """Compute additive per-DP-rank totals before distributed reduction."""
    result = {
        "trees": 0.0,
        "think_count": 0.0,
        "search_count": 0.0,
        "reflect_count": 0.0,
        "backtrack_count": 0.0,
        "output_tokens": 0.0,
        "model_turns": 0.0,
        "completed": 0.0,
        "truncated": 0.0,
        "answer_leak_count": 0.0,
        "reward_trees": 0.0,
        "other_reward_tree_means": 0.0,
        "rollout_reward_tree_means": 0.0,
        "distill_trees": 0.0,
        "distill_reward_tree_means": 0.0,
        "reflect_tokens": 0.0,
        "trainable_tokens": 0.0,
    }

    rollout_ids = rollout_data["rollout_ids"]
    groups: dict[int, list[int]] = {}
    for index, rollout_id in enumerate(rollout_ids):
        if isinstance(rollout_id, torch.Tensor):
            rollout_id = int(rollout_id.item())
        groups.setdefault(int(rollout_id), []).append(index)

    metadata = rollout_data.get("metadata") or [None] * len(rollout_ids)
    for indices in groups.values():
        tree_metrics = next(
            (item.get("reflect_tree_metrics") for item in (metadata[index] for index in indices) if isinstance(item, dict) and isinstance(item.get("reflect_tree_metrics"), dict)),
            None,
        )
        if tree_metrics is not None:
            result["trees"] += 1.0
            for key in (
                "think_count",
                "search_count",
                "reflect_count",
                "backtrack_count",
                "output_tokens",
                "model_turns",
                "completed",
                "truncated",
                "answer_leak_count",
            ):
                result[key] += float(tree_metrics.get(key, 0.0))

        mask = torch.cat([_as_float_tensor(rollout_data["loss_masks"][index]) for index in indices])
        rewards = torch.cat([_as_float_tensor(rollout_data["position_rewards"][index]) for index in indices])
        local_trainable = float(mask.sum().item())
        result["trainable_tokens"] += local_trainable
        whole_tree_trainable = _whole_tree_denominator(
            rollout_data,
            "rollout_mask_sums",
            indices,
            local_total=local_trainable,
        )
        if whole_tree_trainable <= 0:
            continue
        # This is an additive fragment contribution, not necessarily a full
        # local mean. Summing it over all DP ranks reconstructs exactly one
        # token-weighted mean for the tree.
        other_reward_contribution = float((rewards * mask).sum().item() / whole_tree_trainable)
        result["other_reward_tree_means"] += other_reward_contribution
        # The tree summary exists on exactly one compact sample globally, so it
        # is also the ownership marker for equal-weight tree denominators.
        owns_tree = tree_metrics is not None
        if owns_tree and whole_tree_trainable > 0:
            result["reward_trees"] += 1.0

        reflect_mask = torch.cat([_as_float_tensor(rollout_data["reflect_masks"][index]) for index in indices])
        local_reflect_tokens = float(reflect_mask.sum().item())
        result["reflect_tokens"] += local_reflect_tokens
        whole_tree_reflect_tokens = _whole_tree_denominator(
            rollout_data,
            "rollout_reflect_mask_sums",
            indices,
            local_total=local_reflect_tokens,
        )
        distill_reward_contribution = 0.0
        if local_reflect_tokens > 0 and whole_tree_reflect_tokens > 0:
            student = torch.cat([_as_float_tensor(rollout_data["log_probs"][index]) for index in indices])
            teacher = torch.cat([_as_float_tensor(rollout_data["teacher_log_probs"][index]) for index in indices])
            reverse_kl = low_variance_reverse_kl(
                student,
                teacher,
                clip=float(getattr(args, "reflect_distill_kl_clip", 10.0)),
            )
            distill_reward_contribution = -float((reverse_kl * reflect_mask).sum().item() / whole_tree_reflect_tokens)
            result["distill_reward_tree_means"] += distill_reward_contribution
        if owns_tree and whole_tree_reflect_tokens > 0:
            result["distill_trees"] += 1.0

        mode = os.environ.get("REFLECT_TRAIN_MODE", getattr(args, "reflect_train_mode", "joint"))
        if mode in {"joint", "distill_only"}:
            distill_weight = float(getattr(args, "reflect_distill_weight", 0.5))
        elif mode == "fast_slow":
            distill_weight = float(getattr(args, "reflect_fast_distill_weight", 1.0))
        elif mode == "policy_only":
            distill_weight = 0.0
        else:
            raise ValueError(f"unknown reflect_train_mode: {mode!r}")
        policy_weight = 0.0 if mode == "distill_only" else 1.0
        # Reward-equivalent diagnostic for the signals actually optimized by
        # this ablation. Fast-Slow update multiplicity is reported separately
        # and intentionally is not multiplied into this per-rollout measure.
        result["rollout_reward_tree_means"] += policy_weight * other_reward_contribution + distill_weight * distill_reward_contribution

    return result


def finalize_step_metrics(rollout_id: int, args: Any, totals: dict[str, float]) -> dict[str, Any]:
    trees = max(totals["trees"], 1.0)
    reward_trees = max(totals["reward_trees"], 1.0)
    distill_trees = totals["distill_trees"]
    mode = os.environ.get("REFLECT_TRAIN_MODE", getattr(args, "reflect_train_mode", "joint"))
    return {
        "step": int(rollout_id),
        "train_mode": mode,
        "num_rollout_trees": int(totals["trees"]),
        "think_count_mean": totals["think_count"] / trees,
        "search_count_mean": totals["search_count"] / trees,
        "reflect_count_mean": totals["reflect_count"] / trees,
        "backtrack_count_mean": totals["backtrack_count"] / trees,
        "output_tokens_mean": totals["output_tokens"] / trees,
        "model_turns_mean": totals["model_turns"] / trees,
        "rollout_reward_mean": totals["rollout_reward_tree_means"] / reward_trees,
        "distill_reward_mean": (totals["distill_reward_tree_means"] / distill_trees if distill_trees else 0.0),
        "other_reward_mean": totals["other_reward_tree_means"] / reward_trees,
        "reflect_rollout_fraction": distill_trees / trees,
        "completion_rate": totals["completed"] / trees,
        "truncation_rate": totals["truncated"] / trees,
        "answer_leak_count": int(totals["answer_leak_count"]),
        "trainable_tokens": int(totals["trainable_tokens"]),
        "reflect_tokens": int(totals["reflect_tokens"]),
        "fast_reflect_updates": (int(getattr(args, "reflect_fast_updates_per_step", 3)) if mode == "fast_slow" else 0),
        "verifier_updates": 1 if mode == "fast_slow" else 0,
    }


def append_step_metrics(path: str | Path, metrics: dict[str, Any]) -> None:
    """Atomically upsert one outer-step row.

    A guardian retry resumes from the last checkpoint and may legitimately
    recompute rows written after that checkpoint. Replacing by ``step`` keeps
    the metrics file contiguous and audit-safe across those retries.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows: dict[int, dict[str, Any]] = {}
    if target.is_file():
        for line in target.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            step = row.get("step")
            if not isinstance(step, int) or step < 0:
                raise ValueError(f"invalid existing reflect metrics row: {row!r}")
            rows[step] = row
    step = metrics.get("step")
    if not isinstance(step, int) or step < 0:
        raise ValueError(f"invalid reflect metrics step: {step!r}")
    rows[step] = metrics
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o640,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for index in sorted(rows):
                handle.write(json.dumps(rows[index], ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def log_reflect_step_metrics(rollout_id: int, args: Any, rollout_data: dict[str, Any]) -> dict[str, Any] | None:
    """Reduce one metrics record over DP ranks and append it as JSONL."""
    import torch.distributed as dist
    from megatron.core import mpu

    if mpu.get_tensor_model_parallel_rank() != 0 or not mpu.is_pipeline_last_stage():
        return None

    local = compute_local_step_totals(args, rollout_data)
    dp_size = mpu.get_data_parallel_world_size(with_context_parallel=True)
    source = mpu.get_data_parallel_src_rank(with_context_parallel=True)
    group = mpu.get_data_parallel_group_gloo(with_context_parallel=True)
    if dist.get_rank() == source:
        gathered: list[dict[str, float] | None] = [None] * dp_size
        dist.gather_object(local, gathered, dst=source, group=group)
        totals = {key: sum(float(item[key]) for item in gathered if item is not None) for key in local}
        metrics = finalize_step_metrics(rollout_id, args, totals)
        path = os.environ.get("REFLECT_METRICS_PATH") or getattr(args, "reflect_metrics_path", None)
        if path:
            append_step_metrics(path, metrics)
        logger.info("reflect step metrics %s: %s", rollout_id, metrics)
        from slime.utils import logging_utils

        log_dict = {f"reflect/{key}": value for key, value in metrics.items() if isinstance(value, (int, float)) and key != "step"}
        log_dict["reflect/step"] = int(rollout_id)
        logging_utils.log(args, log_dict, step_key="reflect/step")
        return metrics

    dist.gather_object(local, None, dst=source, group=group)
    return None
