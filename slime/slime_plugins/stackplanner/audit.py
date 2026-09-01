"""Completion checks for a StackPlanner training run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


class AuditError(RuntimeError):
    """Raised when persisted metrics and checkpoints do not prove completion."""


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AuditError(f"training metrics are unreadable: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AuditError(
                f"invalid metrics JSON on line {line_number}: {exc}"
            ) from exc
        if isinstance(record, dict):
            records.append(record)
    return records


def _successful_steps(records: list[dict[str, Any]]) -> tuple[set[int], int]:
    steps: list[int] = []
    for record in records:
        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            continue
        if metrics.get("train/optimizer_update_successful") != 1:
            continue
        step = metrics.get("train/step")
        if isinstance(step, int) and step >= 0:
            steps.append(step)
    return set(steps), len(steps) - len(set(steps))


def audit_run(run_dir: str | Path, expected_updates: int | None = None) -> dict[str, Any]:
    """Verify optimizer-update metrics and the matching Megatron checkpoint."""

    run_path = Path(run_dir).resolve()
    if expected_updates is None:
        try:
            config = json.loads(
                (run_path / "resolved_config.yaml").read_text(encoding="utf-8")
            )
            expected_updates = int(config["train_steps"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise AuditError(f"cannot determine expected optimizer updates: {exc}") from exc
    if expected_updates < 1:
        raise AuditError("expected optimizer updates must be positive")

    records = _load_records(run_path / "metrics/metrics.jsonl")
    successful_steps, duplicates = _successful_steps(records)
    if len(successful_steps) != expected_updates:
        raise AuditError(
            "metrics do not prove exactly "
            f"{expected_updates} successful optimizer updates; "
            f"found {len(successful_steps)}"
        )

    checkpoint_root = run_path / "checkpoints"
    tracker = checkpoint_root / "latest_checkpointed_iteration.txt"
    try:
        iteration = int(tracker.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as exc:
        raise AuditError(f"checkpoint tracker is unreadable: {exc}") from exc
    checkpoint_dir = checkpoint_root / f"iter_{iteration:07d}"
    if not checkpoint_dir.is_dir():
        raise AuditError(f"tracked checkpoint directory is missing: {checkpoint_dir}")
    common_path = checkpoint_dir / "common.pt"
    try:
        state = torch.load(common_path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise AuditError(f"checkpoint state is unreadable: {exc}") from exc
    if not isinstance(state, dict):
        raise AuditError("checkpoint state must be a mapping")
    if state.get("iteration") != iteration:
        raise AuditError("checkpoint iteration does not match its tracker")
    if iteration != max(successful_steps):
        raise AuditError(
            "checkpoint iteration does not match the latest successful optimizer update"
        )
    if not isinstance(state.get("optimizer"), dict):
        raise AuditError("checkpoint does not contain optimizer state")
    scheduler = state.get("opt_param_scheduler")
    if not isinstance(scheduler, dict) or int(scheduler.get("num_steps", -1)) < expected_updates:
        raise AuditError("checkpoint scheduler does not cover the expected updates")

    return {
        "completion_verified": True,
        "successful_optimizer_updates": len(successful_steps),
        "deduplicated_metric_records": duplicates,
        "checkpoint": {
            "iteration": iteration,
            "path": str(checkpoint_dir),
        },
    }
