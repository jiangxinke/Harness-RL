"""Bounded context assembly shared by SP action handlers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from deerflow.sp.actions.handlers.base import HandlerContext
from deerflow.sp.memory import StackMemoryEntry
from deerflow.utils.messages import message_to_text

MAX_HANDLER_MEMORY_ENTRIES = 24
MAX_HANDLER_PINNED_ENTRIES = 12
MAX_HANDLER_USER_INPUT_CHARS = 2400
MAX_HANDLER_ARTIFACT_HISTORY = 12
MAX_HANDLER_ENTRY_CONTENT_CHARS = 900
MAX_HANDLER_FAILURE_NOTE_CHARS = 240
MAX_HANDLER_REQUIREMENT_ENTRIES = 8
MAX_HANDLER_REQUIREMENT_CHARS = 4000
MAX_HANDLER_REQUIREMENT_TOTAL_CHARS = 12_000


def _bounded_text(value: Any, *, max_chars: int) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    if len(text) <= max_chars:
        return text
    suffix = "...<truncated>"
    return f"{text[: max_chars - len(suffix)]}{suffix}"


def _entry_context(entry: StackMemoryEntry) -> dict[str, Any]:
    """Keep handler context useful without forwarding unbounded entry metadata."""
    return {
        "id": entry.id,
        "actor": entry.actor,
        "action": entry.action,
        "content": _bounded_text(
            entry.content,
            max_chars=MAX_HANDLER_ENTRY_CONTENT_CHARS,
        ),
        "result_ref": entry.result_ref,
        "priority": entry.priority,
        "stage": entry.stage,
        "status": entry.status,
        "failure_note": _bounded_text(
            entry.failure_note,
            max_chars=MAX_HANDLER_FAILURE_NOTE_CHARS,
        ),
    }


def _bounded_task_memory(context: HandlerContext) -> dict[str, Any]:
    pinned = context.stack.get_pinned_entries()[-MAX_HANDLER_PINNED_ENTRIES:]
    pinned_ids = {entry.id for entry in pinned}
    active = [entry for entry in context.stack.get_active_entries() if entry.id not in pinned_ids]
    remaining = max(MAX_HANDLER_MEMORY_ENTRIES - len(pinned), 0)
    if context.run_id:
        current = [entry for entry in active if entry.run_id == context.run_id]
        carry_candidates = [entry for entry in active if entry.run_id != context.run_id and entry.action in {"summarize", "user_request", "revise"}]
        current_limit = min(len(current), remaining)
        carryover_limit = max(remaining - current_limit, 0)
        carryover = carry_candidates[-carryover_limit:] if carryover_limit else []
        recent = [*carryover, *(current[-current_limit:] if current_limit else [])]
    else:
        recent = active[-remaining:] if remaining else []
    selected = [*pinned, *recent]
    return {
        "version": 1,
        "entry_count": len(context.stack.entries),
        "selected_entry_count": len(selected),
        "entries": [_entry_context(entry) for entry in selected],
    }


def _bounded_artifact_refs(value: Any, *, current_run_id: str | None) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None

    def belongs_to_current_run(item: Any) -> bool:
        if not isinstance(item, Mapping) or not current_run_id:
            return True
        ref_run_id = item.get("run_id")
        return ref_run_id is None or str(ref_run_id) == current_run_id

    refs = {str(key): item for key, item in value.items() if key != "_history" and belongs_to_current_run(item)}
    history = value.get("_history")
    if isinstance(history, list):
        current_history = [item for item in history if belongs_to_current_run(item)]
        if current_history:
            refs["_history"] = current_history[-MAX_HANDLER_ARTIFACT_HISTORY:]
    return refs or None


def _visible_user_inputs(state: Mapping[str, Any]) -> list[str]:
    messages = state.get("messages")
    if not isinstance(messages, list):
        return []
    values: list[str] = []
    for message in messages:
        message_type = getattr(message, "type", None)
        if message_type is None and isinstance(message, Mapping):
            message_type = message.get("type") or message.get("role")
        if message_type not in {"human", "user"}:
            continue
        additional_kwargs = getattr(message, "additional_kwargs", None)
        if additional_kwargs is None and isinstance(message, Mapping):
            additional_kwargs = message.get("additional_kwargs")
        if isinstance(additional_kwargs, Mapping) and additional_kwargs.get("hide_from_ui") is True:
            continue
        text = " ".join(message_to_text(message).split())
        if text:
            values.append(text[:MAX_HANDLER_USER_INPUT_CHARS])
    return values


def _mandatory_requirements(context: HandlerContext) -> dict[str, Any] | None:
    """Preserve authoritative human constraints outside compact task memory."""
    eligible = [entry for entry in context.stack.entries if entry.action == "feedback" and entry.status in {"active", "pinned"} and (not context.run_id or entry.run_id in {None, context.run_id} or entry.status == "pinned")][
        -MAX_HANDLER_REQUIREMENT_ENTRIES:
    ]
    if not eligible:
        return None

    remaining = MAX_HANDLER_REQUIREMENT_TOTAL_CHARS
    feedback: list[dict[str, Any]] = []
    for entry in eligible:
        if remaining <= 0:
            break
        raw = str(entry.content).strip()
        limit = min(MAX_HANDLER_REQUIREMENT_CHARS, remaining)
        truncated = len(raw) > limit
        content = raw[:limit]
        feedback.append(
            {
                "entry_id": entry.id,
                "content": content,
                "priority": entry.priority,
                "status": entry.status,
                "run_id": entry.run_id,
                "truncated": truncated,
            }
        )
        remaining -= len(content)
    return {
        "human_feedback": feedback,
        "authoritative": True,
        "truncated": len(feedback) < len(eligible) or any(item["truncated"] for item in feedback),
    }


def build_handler_context_refs(context: HandlerContext) -> dict[str, Any]:
    """Build the precise, bounded context passed to DR2 subagents."""
    refs: dict[str, Any] = {"task_memory": _bounded_task_memory(context)}
    artifact_refs = _bounded_artifact_refs(
        context.state.get("sp_current_artifact_refs"),
        current_run_id=context.run_id,
    )
    if artifact_refs is not None:
        refs["artifact_refs"] = artifact_refs
    pending_human = context.state.get("sp_pending_human_interaction")
    if isinstance(pending_human, Mapping):
        refs["pending_human_interaction"] = dict(pending_human)
    mandatory_requirements = _mandatory_requirements(context)
    if mandatory_requirements is not None:
        refs["mandatory_requirements"] = mandatory_requirements
    current_stage = context.state.get("sp_current_stage")
    if current_stage:
        refs["current_stage"] = str(current_stage)
    user_inputs = _visible_user_inputs(context.state)
    if user_inputs:
        # The latest visible user message is the authoritative task for this
        # run. The first message in a long-lived thread is historical context,
        # not the delegated task's "original query".
        refs["original_query"] = user_inputs[-1]
        refs["latest_user_input"] = user_inputs[-1]
        if len(user_inputs) > 1:
            refs["thread_first_user_input"] = user_inputs[0]
    return refs
