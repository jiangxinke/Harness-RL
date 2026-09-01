"""BACKTRACK action handler."""

from __future__ import annotations

from typing import Any

from deerflow.sp.actions.events import make_sp_event
from deerflow.sp.actions.handlers.base import HandlerContext
from deerflow.sp.actions.schema import HandlerResult, SPAction


def _entry_ids_after_target(context: HandlerContext, target_type: str, target_id: str) -> list[str]:
    entries = context.stack.entries
    target_idx = None
    for idx, entry in enumerate(entries):
        if target_type == "entry" and entry.id == target_id:
            target_idx = idx
        elif target_type == "stage" and entry.stage == target_id:
            target_idx = idx
    if target_idx is None:
        return []
    return [entry.id for entry in entries[target_idx + 1 :] if entry.status == "active"]


def _artifact_ref_for_version(state_refs: dict[str, Any], target_id: str) -> dict[str, Any]:
    history = state_refs.get("_history")
    if not isinstance(history, list):
        return state_refs
    for ref in reversed(history):
        if not isinstance(ref, dict):
            continue
        if ref.get("artifact_id") == target_id or str(ref.get("version")) == target_id:
            refs = dict(state_refs)
            artifact_type = ref.get("type")
            if not artifact_type:
                return state_refs
            family = _artifact_version_family(str(artifact_type))
            for key, current in list(refs.items()):
                if key == "_history" or key not in family or not isinstance(current, dict):
                    continue
                refs[key] = {**current, "is_current": current.get("artifact_id") == ref.get("artifact_id")}
            selected = {**ref, "is_current": True}
            refs[str(artifact_type)] = selected
            refs["_history"] = [
                {
                    **item,
                    "is_current": item.get("artifact_id") == ref.get("artifact_id"),
                }
                if isinstance(item, dict) and item.get("type") in family
                else item
                for item in history
            ]
            return refs
    return state_refs


def _artifact_version_family(artifact_type: str) -> set[str]:
    if artifact_type in {"report", "report_revision", "final_report"}:
        return {"report", "report_revision", "final_report"}
    return {artifact_type}


def _artifact_type_for_target(refs: dict[str, Any], target_id: str) -> str | None:
    history = refs.get("_history")
    if not isinstance(history, list):
        return None
    for ref in reversed(history):
        if not isinstance(ref, dict):
            continue
        if ref.get("artifact_id") == target_id or str(ref.get("version")) == target_id:
            return str(ref.get("type")) if ref.get("type") else None
    return None


def _current_artifact_id(refs: dict[str, Any], artifact_type: str | None) -> str | None:
    if artifact_type is None:
        return None
    family = _artifact_version_family(artifact_type)
    candidates = [value for key, value in refs.items() if key != "_history" and isinstance(value, dict) and value.get("type") in family and value.get("is_current", True)]
    if not candidates:
        return None
    current = max(candidates, key=lambda value: int(value.get("version") or 0))
    return str(current["artifact_id"]) if current.get("artifact_id") else None


class BacktrackHandler:
    def handle(self, action: SPAction, context: HandlerContext) -> HandlerResult:
        target_type = str(action.metadata["backtrack_target_type"])
        target_id = str(action.metadata["backtrack_target_id"])
        reason = str(action.metadata.get("reason") or action.reason)
        source_entry_ids = [str(entry_id) for entry_id in action.metadata.get("source_entry_ids", [])]
        if not source_entry_ids and target_type in {"entry", "stage"}:
            source_entry_ids = _entry_ids_after_target(context, target_type, target_id)

        entry = context.stack.mark_backtracked(
            source_entry_ids,
            reason,
            thread_id=context.thread_id,
            run_id=context.run_id,
            stage=action.stage,
            priority=action.priority,
            metadata={"action_id": action.action_id, **action.metadata},
        )

        state_update: dict[str, Any] = {"sp_last_run_summary": reason}
        rollback_scope = str(action.metadata.get("rollback_scope") or "memory_only")
        if action.stage:
            state_update["sp_current_stage"] = action.stage
        if rollback_scope in {"delegation", "full_working_state"} or target_type == "delegation":
            state_update["sp_active_delegate_id"] = None
        if rollback_scope in {"artifact_refs", "full_working_state"} or target_type == "artifact_version":
            refs = context.state.get("sp_current_artifact_refs") or {}
            if isinstance(refs, dict):
                restored_refs = _artifact_ref_for_version(refs, target_id)
                state_update["sp_current_artifact_refs"] = restored_refs

        events = [
            make_sp_event(
                "sp.delegate.backtracked" if target_type == "delegation" else "sp.memory.backtracked",
                action_id=action.action_id,
                run_id=context.run_id,
                target_type=target_type,
                target_id=target_id,
                source_entry_ids=source_entry_ids,
            )
        ]
        restored_refs = state_update.get("sp_current_artifact_refs")
        if isinstance(restored_refs, dict):
            artifact_type = _artifact_type_for_target(refs, target_id)
            previous_artifact_id = _current_artifact_id(refs, artifact_type)
            current_artifact_id = _current_artifact_id(restored_refs, artifact_type)
            if current_artifact_id and current_artifact_id != previous_artifact_id:
                events.append(
                    make_sp_event(
                        "sp.artifact.current_changed",
                        action_id=action.action_id,
                        run_id=context.run_id,
                        previous_artifact_id=previous_artifact_id,
                        current_artifact_id=current_artifact_id,
                        reason="backtrack",
                    )
                )

        return HandlerResult(
            next_step="continue",
            state_update=state_update,
            memory_entries=[entry],
            idempotency_key=action.idempotency_key,
            run_events=events,
        )
