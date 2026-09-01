"""Adapter from SP action events to DeerFlow's RunEventStore."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol


class _RunEventStoreProtocol(Protocol):
    async def put_batch(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Write normalized run events."""


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    return repr(value)


def normalize_sp_event_for_store(event: Mapping[str, Any], *, thread_id: str, run_id: str | None = None) -> dict[str, Any]:
    """Convert one SP event dict into a DR2 RunEventStore put payload."""
    payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
    resolved_run_id = str(event.get("run_id") or run_id or "")
    if not resolved_run_id:
        raise ValueError("SP run event persistence requires run_id")
    metadata = {
        "source": "stackplanner",
        "action_id": event.get("action_id"),
        "payload": _json_safe(payload),
    }
    task_id = payload.get("task_id") if isinstance(payload, Mapping) else None
    if task_id is not None:
        metadata["task_id"] = str(task_id)
    return {
        "thread_id": thread_id,
        "run_id": resolved_run_id,
        "event_type": str(event.get("event_type") or "sp.event"),
        "category": "trace",
        "content": {
            "action_id": event.get("action_id"),
            "payload": _json_safe(payload),
        },
        "metadata": metadata,
        "created_at": event.get("ts"),
    }


class SPRunEventAdapter:
    """Persist SP HandlerResult/ActionLoop events through DR2 RunEventStore."""

    def __init__(self, event_store: _RunEventStoreProtocol):
        self._event_store = event_store

    async def write_events(
        self,
        events: Iterable[Mapping[str, Any]],
        *,
        thread_id: str,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        records = [normalize_sp_event_for_store(event, thread_id=thread_id, run_id=run_id) for event in events]
        if not records:
            return []
        return await self._event_store.put_batch(records)
