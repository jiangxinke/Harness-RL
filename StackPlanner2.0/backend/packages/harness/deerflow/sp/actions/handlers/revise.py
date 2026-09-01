"""REVISE action handler for central-memory self-correction."""

from __future__ import annotations

from deerflow.sp.actions.events import make_sp_event
from deerflow.sp.actions.handlers.base import HandlerContext
from deerflow.sp.actions.schema import HandlerResult, SPAction


class ReviseHandler:
    """Replace erroneous active task-memory entries with a corrected entry."""

    def handle(self, action: SPAction, context: HandlerContext) -> HandlerResult:
        target_entry_ids = [str(entry_id) for entry_id in action.metadata["target_entry_ids"]]
        correction = str(action.task or action.metadata.get("correction") or "").strip()
        reason = str(action.metadata.get("revision_reason") or action.reason)
        entry = context.stack.revise(
            target_entry_ids,
            correction,
            reason,
            thread_id=context.thread_id,
            run_id=context.run_id,
            stage=action.stage,
            priority=action.priority,
            metadata={
                "action_id": action.action_id,
                "revision_reason": reason,
                "target_entry_ids": target_entry_ids,
                **action.metadata,
            },
        )
        state_update = {"sp_last_run_summary": correction}
        if action.stage:
            state_update["sp_current_stage"] = action.stage
        return HandlerResult(
            next_step="continue",
            state_update=state_update,
            memory_entries=[entry],
            idempotency_key=action.idempotency_key,
            run_events=[
                make_sp_event(
                    "sp.memory.revised",
                    action_id=action.action_id,
                    run_id=context.run_id,
                    target_entry_ids=target_entry_ids,
                    replacement_entry_id=entry.id,
                    reason=reason,
                )
            ],
        )
