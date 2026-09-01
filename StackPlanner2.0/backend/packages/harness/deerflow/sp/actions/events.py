"""Lightweight SP action event helpers."""

from __future__ import annotations

from typing import Any

from deerflow.sp.memory.entry import utc_now_iso


def make_sp_event(event_type: str, *, action_id: str | None = None, run_id: str | None = None, **payload: Any) -> dict[str, Any]:
    return {
        "event_type": event_type,
        "ts": utc_now_iso(),
        "action_id": action_id,
        "run_id": run_id,
        "payload": payload,
    }
