"""Pure helpers for StackPlanner role/capability rollout metrics."""

from __future__ import annotations

from typing import Any


def compute_stackplanner_sample_metrics(samples: list[Any]) -> dict[str, int]:
    role_token_counts: dict[str, int] = {}
    capability_token_counts: dict[str, int] = {}
    infrastructure_failures = 0
    removed_samples = 0
    for sample in samples:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        role = str(metadata.get("role") or "unknown")
        mask = list(sample.loss_mask or [1] * sample.response_length)
        role_token_counts[role] = role_token_counts.get(role, 0) + sum(
            bool(value) for value in mask
        )
        training_metadata = (
            sample.train_metadata if isinstance(sample.train_metadata, dict) else {}
        )
        labels = training_metadata.get("capability_labels")
        if isinstance(labels, list) and len(labels) == sample.response_length:
            for enabled, capability in zip(mask, labels, strict=True):
                if enabled and capability:
                    key = str(capability)
                    capability_token_counts[key] = (
                        capability_token_counts.get(key, 0) + 1
                    )
        infrastructure_failures += int("infrastructure_failure" in metadata)
        removed_samples += int(bool(sample.remove_sample))

    metrics = {
        f"role_tokens/{role}": count
        for role, count in sorted(role_token_counts.items())
    }
    metrics.update(
        {
            f"capability_tokens/{capability}": count
            for capability, count in sorted(capability_token_counts.items())
        }
    )
    metrics["infrastructure_failure_count"] = infrastructure_failures
    metrics["removed_sample_count"] = removed_samples
    return metrics
