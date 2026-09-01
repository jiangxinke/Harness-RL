"""ASK_HUMAN action handler."""

from __future__ import annotations

from deerflow.sp.actions.events import make_sp_event
from deerflow.sp.actions.handlers.base import HandlerContext
from deerflow.sp.actions.schema import HandlerResult, SPAction
from deerflow.sp.hitl import SPHumanInteraction
from deerflow.sp.memory import StackMemoryEntry


class AskHumanHandler:
    def handle(self, action: SPAction, context: HandlerContext) -> HandlerResult:
        question = str(action.metadata.get("question") or action.task)
        entry = context.stack.append(
            StackMemoryEntry(
                thread_id=context.thread_id,
                run_id=context.run_id,
                actor="central",
                action="ask_human",
                content=question,
                priority=action.priority,
                stage=action.stage,
                metadata={"action_id": action.action_id, **action.metadata},
            )
        )
        artifact_refs = action.metadata.get("artifact_refs")
        if not isinstance(artifact_refs, dict):
            artifact_refs = context.state.get("sp_current_artifact_refs") if isinstance(context.state.get("sp_current_artifact_refs"), dict) else {}
        interaction = SPHumanInteraction(
            interaction_type=str(action.metadata.get("interaction_type") or "clarification"),
            question=question,
            artifact_refs=artifact_refs,
            created_by_entry_id=entry.id,
            metadata={"action_id": action.action_id, **action.metadata},
        )
        state_update = {
            "sp_pending_human_interaction": interaction.to_dict(),
        }
        if action.stage:
            state_update["sp_current_stage"] = action.stage
        return HandlerResult(
            next_step="interrupt",
            state_update=state_update,
            memory_entries=[entry],
            idempotency_key=action.idempotency_key,
            run_events=[
                make_sp_event(
                    "sp.human.requested",
                    action_id=action.action_id,
                    run_id=context.run_id,
                    interaction_id=interaction.interaction_id,
                    interaction_type=interaction.interaction_type,
                )
            ],
        )
