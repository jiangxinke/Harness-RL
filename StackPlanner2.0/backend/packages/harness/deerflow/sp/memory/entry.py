"""Structured task-memory entries for StackPlanner orchestration.

The SP migration stores control-flow memory as plain JSON-serializable data so
DeerFlow's ThreadState/checkpointer remains the persistence boundary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

EntryDict = dict[str, Any]

MAX_ENTRY_CONTENT_CHARS = 2000
MAX_FAILURE_NOTE_CHARS = 1000
MAX_METADATA_JSON_CHARS = 8000
MAX_METADATA_PREVIEW_CHARS = 2000


def _bounded_text(value: Any, *, max_chars: int) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    suffix = "...<truncated>"
    return f"{text[: max_chars - len(suffix)]}{suffix}"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)

    content = getattr(value, "content", None)
    if isinstance(content, str):
        return {
            "type": value.__class__.__name__,
            "content": content,
            "additional_kwargs": _json_safe(getattr(value, "additional_kwargs", {})),
        }
    return repr(value)


def _bounded_metadata(value: Any) -> EntryDict:
    normalized = _json_safe(value) or {}
    if not isinstance(normalized, dict):
        normalized = {"value": normalized}
    serialized = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    if len(serialized) <= MAX_METADATA_JSON_CHARS:
        return normalized

    preserved_keys = {
        "action_id",
        "interaction_id",
        "legacy_action",
        "legacy_agent_type",
        "memory_query",
        "status",
        "target_agent",
        "task_id",
    }
    preserved = {key: normalized[key] for key in preserved_keys if key in normalized}
    return {
        **preserved,
        "_truncated": True,
        "_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "_preview": _bounded_text(serialized, max_chars=MAX_METADATA_PREVIEW_CHARS),
    }


def _priority_from_legacy(action: str, agent_type: str | None, result: Any) -> str:
    if action == "human_feedback" or agent_type == "human":
        return "critical"
    if isinstance(result, dict) and str(result.get("priority", "")).upper() == "HIGHEST":
        return "critical"
    return "normal"


def _status_from_priority(priority: str) -> str:
    return "pinned" if priority == "critical" else "active"


def _actor_from_legacy(agent_type: str | None) -> str:
    if agent_type in {None, "", "central_agent"}:
        return "central"
    return str(agent_type)


def _action_from_legacy(action: str, result: Any) -> str:
    if action == "human_feedback":
        return "feedback"
    if action == "delegate" and result is not None:
        return "observe"
    return action or "observe"


@dataclass(slots=True)
class StackMemoryEntry:
    """One SP control-flow memory event stored inside DeerFlow ThreadState."""

    id: str = field(default_factory=lambda: f"spmem_{uuid4().hex}")
    ts: str = field(default_factory=utc_now_iso)
    thread_id: str | None = None
    run_id: str | None = None
    actor: str = "central"
    action: str = "think"
    content: str = ""
    result_ref: str | None = None
    priority: str = "normal"
    stage: str | None = None
    status: str = "active"
    parent_ids: list[str] = field(default_factory=list)
    failure_note: str | None = None
    promotion_candidate: bool = False
    metadata: EntryDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = str(self.id or f"spmem_{uuid4().hex}")
        self.ts = str(self.ts or utc_now_iso())
        self.actor = str(self.actor or "central")
        self.action = str(self.action or "think")
        self.content = _bounded_text(self.content, max_chars=MAX_ENTRY_CONTENT_CHARS)
        self.priority = str(self.priority or "normal")
        self.status = str(self.status or "active")
        self.parent_ids = [str(parent_id) for parent_id in self.parent_ids]
        self.failure_note = _bounded_text(self.failure_note, max_chars=MAX_FAILURE_NOTE_CHARS) if self.failure_note is not None else None
        self.metadata = _bounded_metadata(self.metadata)

    @classmethod
    def from_dict(cls, data: EntryDict) -> StackMemoryEntry:
        """Restore a new-format entry or normalize a legacy SP stack entry."""
        if "timestamp" in data or "agent_type" in data or "result" in data:
            return cls.from_legacy_dict(data)

        return cls(
            id=str(data.get("id") or data.get("entry_id") or f"spmem_{uuid4().hex}"),
            ts=str(data.get("ts") or data.get("timestamp") or utc_now_iso()),
            thread_id=data.get("thread_id"),
            run_id=data.get("run_id"),
            actor=str(data.get("actor") or "central"),
            action=str(data.get("action") or "think"),
            content=str(data.get("content") or ""),
            result_ref=data.get("result_ref"),
            priority=str(data.get("priority") or "normal"),
            stage=data.get("stage"),
            status=str(data.get("status") or "active"),
            parent_ids=list(data.get("parent_ids") or []),
            failure_note=data.get("failure_note"),
            promotion_candidate=bool(data.get("promotion_candidate", False)),
            metadata=_json_safe(data.get("metadata") or {}),
        )

    @classmethod
    def from_legacy_dict(cls, data: EntryDict, *, thread_id: str | None = None, run_id: str | None = None) -> StackMemoryEntry:
        """Convert StackPlanner's old MemoryStackEntry dict into SP-on-DR2 shape."""
        legacy_action = str(data.get("action") or "")
        legacy_agent_type = data.get("agent_type")
        legacy_result = data.get("result")
        priority = _priority_from_legacy(legacy_action, legacy_agent_type, legacy_result)

        return cls(
            ts=str(data.get("timestamp") or utc_now_iso()),
            thread_id=thread_id,
            run_id=run_id,
            actor=_actor_from_legacy(legacy_agent_type),
            action=_action_from_legacy(legacy_action, legacy_result),
            content=str(data.get("content") or ""),
            priority=priority,
            status=_status_from_priority(priority),
            metadata={
                "legacy_action": legacy_action,
                "legacy_agent_type": legacy_agent_type,
                "legacy_result": _json_safe(legacy_result),
            },
        )

    def to_dict(self) -> EntryDict:
        return {
            "id": self.id,
            "ts": self.ts,
            "thread_id": self.thread_id,
            "run_id": self.run_id,
            "actor": self.actor,
            "action": self.action,
            "content": self.content,
            "result_ref": self.result_ref,
            "priority": self.priority,
            "stage": self.stage,
            "status": self.status,
            "parent_ids": list(self.parent_ids),
            "failure_note": self.failure_note,
            "promotion_candidate": self.promotion_candidate,
            "metadata": _json_safe(self.metadata),
        }
