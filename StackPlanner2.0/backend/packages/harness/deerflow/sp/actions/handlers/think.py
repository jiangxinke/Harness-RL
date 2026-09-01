"""THINK action handler."""

from __future__ import annotations

from deerflow.sp.actions.handlers.base import HandlerContext
from deerflow.sp.actions.schema import HandlerResult, SPAction


class ThinkHandler:
    def handle(self, action: SPAction, context: HandlerContext) -> HandlerResult:
        content = action.task or action.reason
        entry = context.stack.append_think(
            content,
            thread_id=context.thread_id,
            run_id=context.run_id,
            stage=action.stage,
            priority=action.priority,
            metadata={"action_id": action.action_id, **action.metadata},
        )
        state_update = {}
        if action.stage:
            state_update["sp_current_stage"] = action.stage
        return HandlerResult(next_step="continue", state_update=state_update, memory_entries=[entry], idempotency_key=action.idempotency_key)
