from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from slime_plugins.stackplanner.audit import AuditError, audit_run


def _make_run(tmp_path: Path, updates: int = 3) -> Path:
    run = tmp_path / "run"
    (run / "metrics").mkdir(parents=True)
    (run / "resolved_config.yaml").write_text(
        json.dumps({"train_steps": updates}), encoding="utf-8"
    )
    records = [
        {
            "metrics": {
                "train/step": step,
                "train/optimizer_update_successful": 1,
            }
        }
        for step in range(updates)
    ]
    records.append(records[-1])
    (run / "metrics/metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    checkpoint = run / "checkpoints"
    iteration = updates - 1
    state_dir = checkpoint / f"iter_{iteration:07d}"
    state_dir.mkdir(parents=True)
    (checkpoint / "latest_checkpointed_iteration.txt").write_text(
        str(iteration), encoding="utf-8"
    )
    torch.save(
        {
            "iteration": iteration,
            "optimizer": {"state": {}},
            "opt_param_scheduler": {"num_steps": updates},
        },
        state_dir / "common.pt",
    )
    return run


def test_audit_proves_deduplicated_updates_and_checkpoint(tmp_path: Path) -> None:
    report = audit_run(_make_run(tmp_path))

    assert report["completion_verified"] is True
    assert report["successful_optimizer_updates"] == 3
    assert report["deduplicated_metric_records"] == 1
    assert report["checkpoint"]["iteration"] == 2


def test_audit_rejects_missing_successful_update(tmp_path: Path) -> None:
    run = _make_run(tmp_path)
    lines = (run / "metrics/metrics.jsonl").read_text().splitlines()
    (run / "metrics/metrics.jsonl").write_text("\n".join(lines[:-2]) + "\n")

    with pytest.raises(AuditError, match="do not prove exactly"):
        audit_run(run)


def test_audit_rejects_checkpoint_that_lags_metrics(tmp_path: Path) -> None:
    run = _make_run(tmp_path)
    (run / "checkpoints/latest_checkpointed_iteration.txt").write_text("1")

    with pytest.raises(AuditError, match="tracked checkpoint directory is missing"):
        audit_run(run)
