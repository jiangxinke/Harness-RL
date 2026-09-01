from __future__ import annotations

import json
import multiprocessing
from slime.utils.local_metrics import append_local_metrics


def test_log_always_appends_strict_local_jsonl(monkeypatch, tmp_path):
    path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("SP_METRICS_JSONL", str(path))
    monkeypatch.setenv("SP_TRAINING_RUN_ID", "run-a")
    append_local_metrics(
        {"train/step": 7, "train/loss": 0.25, "train/grad_norm": float("nan")},
        "train/step",
    )

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["run_id"] == "run-a"
    assert record["step_key"] == "train/step"
    assert record["metrics"] == {
        "train/grad_norm": "nan",
        "train/loss": 0.25,
        "train/step": 7,
    }


def _append_many(path: str, worker: int) -> None:
    import os

    os.environ["SP_METRICS_JSONL"] = path
    for index in range(25):
        append_local_metrics({"train/step": worker * 100 + index}, "train/step")


def test_local_metrics_are_process_safe_json_lines(tmp_path):
    path = tmp_path / "metrics.jsonl"
    processes = [multiprocessing.Process(target=_append_many, args=(str(path), worker)) for worker in range(4)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 100
