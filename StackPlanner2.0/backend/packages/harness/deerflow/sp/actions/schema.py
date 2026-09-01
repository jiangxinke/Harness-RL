"""Schema for SP CentralAgent actions and handler results."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from deerflow.sp.memory import StackMemoryEntry

HandlerNextStep = Literal["continue", "interrupt", "finish", "error_recoverable", "error_fatal"]


class ActionValidationError(ValueError):
    """Raised when CentralAgent output fails action schema validation."""


class ActionType(StrEnum):
    THINK = "THINK"
    DELEGATE = "DELEGATE"
    RECALL_MEMORY = "RECALL_MEMORY"
    REFLECT = "REFLECT"
    REVISE = "REVISE"
    BACKTRACK = "BACKTRACK"
    REPLAN = "REPLAN"
    SUMMARIZE = "SUMMARIZE"
    ASK_HUMAN = "ASK_HUMAN"
    FINISH = "FINISH"


ALLOWED_DELEGATE_AGENTS = frozenset({"researcher", "coder", "reporter", "outline", "perception"})
ALLOWED_STAGES = frozenset({"perception", "planning", "research", "implementation", "reporting", "revision", "verification", "finished"})
STAGE_ALIASES = {
    "analysis": "perception",
    "recognition": "perception",
    "thinking": "perception",
    "clarification": "perception",
    "delegation": "implementation",
    "delegation-attempt": "implementation",
    "execution": "implementation",
    "file-inspection": "perception",
    "scope-assessment": "perception",
    "perception-and-scope": "perception",
    "task-intake": "perception",
    "perceive": "perception",
    "plan": "planning",
    "outline": "planning",
    "data-collection": "research",
    "retrieval": "research",
    "reinquiry": "research",
    "information-gathering": "research",
    "information gathering": "research",
    "parallel-delegation": "implementation",
    "implement": "implementation",
    "coder": "implementation",
    "researcher": "research",
    "report": "reporting",
    "reporter": "reporting",
    "synthesis": "reporting",
    "revise": "revision",
    "backtrack": "revision",
    "verify": "verification",
    "finish": "finished",
    "done": "finished",
}
ALLOWED_PRIORITIES = frozenset({"critical", "high", "normal", "low"})
ALLOWED_BACKTRACK_TARGET_TYPES = frozenset({"entry", "stage", "artifact_version", "delegation"})
ALLOWED_ROLLBACK_SCOPES = frozenset({"memory_only", "artifact_refs", "delegation", "stage", "full_working_state"})


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    return repr(value)


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ActionValidationError(f"{field_name} must be a non-empty string")
    return value.strip()


def _normalize_action_type(value: Any) -> ActionType:
    if isinstance(value, ActionType):
        return value
    if not isinstance(value, str):
        raise ActionValidationError("action_type must be a string")
    normalized = value.strip().upper()
    try:
        return ActionType(normalized)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in ActionType)
        raise ActionValidationError(f"Unsupported action_type {value!r}; allowed: {allowed}") from exc


def _normalize_stage(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ActionValidationError("stage must be a non-empty string when provided")
    normalized = value.strip().lower().replace("_", "-")
    return STAGE_ALIASES.get(normalized, normalized)


def _normalize_expected_output(value: Any) -> str | None:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, dict | list):
        return json.dumps(
            _json_safe(value),
            ensure_ascii=False,
            sort_keys=True,
        )
    raise ActionValidationError("expected_output must be a string, JSON object, or JSON array when provided")


def _stable_idempotency_key(action_id: str, action_type: ActionType, task: str | None, metadata: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "action_id": action_id,
            "action_type": action_type.value,
            "task": task,
            "metadata": metadata,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return f"spidem_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


@dataclass(slots=True)
class SPAction:
    """Validated action emitted by SP CentralAgent."""

    action_id: str
    action_type: ActionType
    idempotency_key: str
    reason: str
    target_agent: str | None = None
    task: str | None = None
    input_refs: list[str] = field(default_factory=list)
    expected_output: str | None = None
    stage: str | None = None
    priority: str = "normal"
    requires_human: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SPAction:
        if not isinstance(data, dict):
            raise ActionValidationError("Action payload must be a dict")

        action_type = _normalize_action_type(data.get("action_type") or data.get("type") or data.get("action"))
        action_id = _require_text(data.get("action_id"), "action_id")
        reason = _require_text(data.get("reason"), "reason")
        metadata = _json_safe(data.get("metadata") or {})
        if not isinstance(metadata, dict):
            raise ActionValidationError("metadata must be a JSON object")
        task = data.get("task")
        if task is not None:
            task = _require_text(task, "task")
        idempotency_key = data.get("idempotency_key")
        if idempotency_key is None:
            idempotency_key = _stable_idempotency_key(action_id, action_type, task, metadata)
        idempotency_key = _require_text(idempotency_key, "idempotency_key")

        target_agent = data.get("target_agent")
        if isinstance(target_agent, str) and not target_agent.strip():
            target_agent = None
        elif target_agent is not None:
            target_agent = _require_text(target_agent, "target_agent")
        input_refs = data.get("input_refs") or []
        if not isinstance(input_refs, list):
            raise ActionValidationError("input_refs must be a list")

        action = cls(
            action_id=action_id,
            action_type=action_type,
            idempotency_key=idempotency_key,
            reason=reason,
            target_agent=target_agent,
            task=task,
            input_refs=[str(item) for item in input_refs],
            expected_output=data.get("expected_output"),
            stage=_normalize_stage(data.get("stage")),
            priority=str(data.get("priority") or "normal"),
            requires_human=bool(data.get("requires_human", False)),
            metadata=metadata,
        )
        action.validate()
        return action

    @classmethod
    def create(
        cls,
        action_type: ActionType | str,
        *,
        reason: str,
        action_id: str | None = None,
        idempotency_key: str | None = None,
        **kwargs: Any,
    ) -> SPAction:
        action_type_value = _normalize_action_type(action_type)
        action_id = action_id or f"spact_{uuid4().hex}"
        payload = {
            "action_id": action_id,
            "action_type": action_type_value.value,
            "idempotency_key": idempotency_key,
            "reason": reason,
            **kwargs,
        }
        return cls.from_dict(payload)

    def validate(self) -> None:
        # Normalize direct dataclass construction as well as from_dict/create.
        self.stage = _normalize_stage(self.stage)
        self.expected_output = _normalize_expected_output(self.expected_output)
        if isinstance(self.target_agent, str) and not self.target_agent.strip():
            self.target_agent = None
        if self.priority not in ALLOWED_PRIORITIES:
            allowed = ", ".join(sorted(ALLOWED_PRIORITIES))
            raise ActionValidationError(f"Unsupported priority {self.priority!r}; allowed: {allowed}")
        if self.stage is not None and self.stage not in ALLOWED_STAGES:
            allowed = ", ".join(sorted(ALLOWED_STAGES))
            raise ActionValidationError(f"Unsupported stage {self.stage!r}; allowed: {allowed}")
        if self.action_type == ActionType.DELEGATE:
            target_agent = _require_text(self.target_agent, "target_agent")
            if target_agent not in ALLOWED_DELEGATE_AGENTS:
                allowed = ", ".join(sorted(ALLOWED_DELEGATE_AGENTS))
                raise ActionValidationError(f"Unsupported target_agent {target_agent!r}; allowed: {allowed}")
            _require_text(self.task, "task")
        elif self.action_type == ActionType.RECALL_MEMORY:
            query = self.metadata.get("memory_query") or self.task
            _require_text(query, "metadata.memory_query or task")
        elif self.action_type == ActionType.REFLECT:
            target_entry_ids = self.metadata.get("target_entry_ids")
            if target_entry_ids is not None:
                if not isinstance(target_entry_ids, list) or not target_entry_ids:
                    raise ActionValidationError("metadata.target_entry_ids must be a non-empty list when provided")
                for entry_id in target_entry_ids:
                    _require_text(entry_id, "metadata.target_entry_ids[]")
        elif self.action_type == ActionType.REVISE:
            target_entry_ids = self.metadata.get("target_entry_ids")
            if not isinstance(target_entry_ids, list) or not target_entry_ids:
                raise ActionValidationError("metadata.target_entry_ids must be a non-empty list")
            for entry_id in target_entry_ids:
                _require_text(entry_id, "metadata.target_entry_ids[]")
            _require_text(self.task or self.metadata.get("correction"), "task or metadata.correction")
        elif self.action_type == ActionType.BACKTRACK:
            target_type = _require_text(self.metadata.get("backtrack_target_type"), "metadata.backtrack_target_type")
            _require_text(self.metadata.get("backtrack_target_id"), "metadata.backtrack_target_id")
            if target_type not in ALLOWED_BACKTRACK_TARGET_TYPES:
                allowed = ", ".join(sorted(ALLOWED_BACKTRACK_TARGET_TYPES))
                raise ActionValidationError(f"Unsupported backtrack_target_type {target_type!r}; allowed: {allowed}")
            rollback_scope = str(self.metadata.get("rollback_scope") or "memory_only")
            if rollback_scope not in ALLOWED_ROLLBACK_SCOPES:
                allowed = ", ".join(sorted(ALLOWED_ROLLBACK_SCOPES))
                raise ActionValidationError(f"Unsupported rollback_scope {rollback_scope!r}; allowed: {allowed}")
            if self.metadata.get("preserve_artifacts") is False:
                raise ActionValidationError("BACKTRACK cannot delete artifact history; metadata.preserve_artifacts must not be false")
        elif self.action_type == ActionType.SUMMARIZE:
            _require_text(self.task or self.metadata.get("summary"), "task or metadata.summary")
        elif self.action_type == ActionType.ASK_HUMAN:
            _require_text(self.task or self.metadata.get("question"), "task or metadata.question")
        elif self.action_type == ActionType.FINISH:
            _require_text(self.task, "task")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "idempotency_key": self.idempotency_key,
            "reason": self.reason,
            "target_agent": self.target_agent,
            "task": self.task,
            "input_refs": list(self.input_refs),
            "expected_output": self.expected_output,
            "stage": self.stage,
            "priority": self.priority,
            "requires_human": self.requires_human,
            "metadata": _json_safe(self.metadata),
        }


@dataclass(slots=True)
class HandlerResult:
    """Uniform result returned by every SP action handler."""

    next_step: HandlerNextStep
    state_update: dict[str, Any] = field(default_factory=dict)
    memory_entries: list[StackMemoryEntry] = field(default_factory=list)
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    run_events: list[dict[str, Any]] = field(default_factory=list)
    idempotency_key: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "next_step": self.next_step,
            "state_update": _json_safe(self.state_update),
            "memory_entries": [entry.to_dict() for entry in self.memory_entries],
            "artifact_refs": _json_safe(self.artifact_refs),
            "run_events": _json_safe(self.run_events),
            "idempotency_key": self.idempotency_key,
            "error": self.error,
        }
