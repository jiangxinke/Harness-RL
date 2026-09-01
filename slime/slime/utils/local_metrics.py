"""Strict append-only local metric records independent of W&B/TensorBoard."""

from __future__ import annotations

import fcntl
import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

_METRICS_LOCK = threading.Lock()


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def append_local_metrics(metrics: dict[str, Any], step_key: str) -> None:
    """Append one metric group to ``SP_METRICS_JSONL`` when configured."""

    destination = os.environ.get("SP_METRICS_JSONL")
    if not destination:
        return
    record = {
        "recorded_at_unix": time.time(),
        "pid": os.getpid(),
        "step_key": step_key,
        "run_id": os.environ.get("SP_TRAINING_RUN_ID", ""),
        "metrics": {key: _json_value(value) for key, value in metrics.items()},
    }
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()
    with _METRICS_LOCK, path.open("ab", buffering=0) as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(encoded)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
