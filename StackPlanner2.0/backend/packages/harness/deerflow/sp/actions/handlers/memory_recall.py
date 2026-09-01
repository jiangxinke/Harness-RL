"""RECALL_MEMORY action handler."""

from __future__ import annotations

from typing import Any

from deerflow.sp.actions.events import make_sp_event
from deerflow.sp.actions.handlers.base import HandlerContext
from deerflow.sp.actions.handlers.context import build_handler_context_refs
from deerflow.sp.actions.schema import HandlerResult, SPAction
from deerflow.sp.memory import StackMemoryEntry, normalize_memory_recall_result
from deerflow.sp.subagents import SPSubagentExecutorProtocol, SPSubagentResult, SPSubagentTask

MEMORY_RECALLER_AGENT = "memory_recaller"


def _build_recall_task(query: str, action: SPAction) -> str:
    lines = [
        "Recall relevant long-term memory for the StackPlanner CentralAgent.",
        "Use DeerFlow's existing memory context and memory tools available to this subagent.",
        "Do not write, update, promote, or mutate long-term memory.",
        f"Query: {query}",
        'Return compact JSON: {"summary": str, "items": [{"content": str, "source": str, "score": number, "scope": str, "memory_id": str}]}',
    ]
    if action.expected_output:
        lines.append(f"Expected output: {action.expected_output}")
    return "\n".join(lines)


class MemoryRecallHandler:
    """Route memory recall through a memory_recaller subagent, never direct storage."""

    def __init__(self, *, executor: SPSubagentExecutorProtocol | None = None, subagent_type: str = MEMORY_RECALLER_AGENT) -> None:
        self._executor = executor
        self._subagent_type = subagent_type

    def handle(self, action: SPAction, context: HandlerContext) -> HandlerResult:
        if self._executor is None:
            return HandlerResult(
                next_step="error_recoverable",
                idempotency_key=action.idempotency_key,
                error="RECALL_MEMORY requires a memory_recaller subagent executor",
            )

        query = str(action.metadata.get("memory_query") or action.task)
        request_entry = context.stack.append_memory_recall(
            f"Recall long-term memory: {query}",
            thread_id=context.thread_id,
            run_id=context.run_id,
            stage=action.stage,
            priority=action.priority,
            metadata={
                "action_id": action.action_id,
                "memory_query": query,
                "dry_run_promotion": True,
                **action.metadata,
            },
        )
        task = SPSubagentTask(
            action_id=action.action_id,
            subagent_type=self._subagent_type,
            task=_build_recall_task(query, action),
            description=action.reason,
            input_refs=list(action.input_refs),
            expected_output=action.expected_output or "read-only normalized memory recall",
            context_refs=build_handler_context_refs(context),
            thread_id=context.thread_id,
            run_id=context.run_id,
            metadata={
                "memory_query": query,
                "dry_run_promotion": True,
                **action.metadata,
            },
        )

        result = self._executor.execute(task)
        if result.is_success:
            return self._handle_success(action, context, request_entry, query, result)
        return self._handle_failure(action, context, request_entry, result)

    def _handle_success(
        self,
        action: SPAction,
        context: HandlerContext,
        request_entry: StackMemoryEntry,
        query: str,
        result: SPSubagentResult,
    ) -> HandlerResult:
        recall = normalize_memory_recall_result(query, result)
        recall_entry = context.stack.append_memory_recall(
            recall.to_memory_content(),
            actor=self._subagent_type,
            thread_id=context.thread_id,
            run_id=context.run_id,
            stage=action.stage,
            priority=action.priority,
            result_ref=result.task_id,
            parent_ids=[request_entry.id],
            metadata={
                "action_id": action.action_id,
                "task_id": result.task_id,
                "stop_reason": result.stop_reason,
                "memory_query": query,
                "dry_run_promotion": True,
                "recall": recall.to_dict(),
            },
        )
        state_update: dict[str, Any] = {}
        if action.stage:
            state_update["sp_current_stage"] = action.stage
        return HandlerResult(
            next_step="continue",
            state_update=state_update,
            memory_entries=[request_entry, recall_entry],
            idempotency_key=action.idempotency_key,
            run_events=[
                make_sp_event("sp.memory.recall.started", action_id=action.action_id, run_id=context.run_id, query=query),
                make_sp_event(
                    "sp.memory.recall.completed",
                    action_id=action.action_id,
                    run_id=context.run_id,
                    task_id=result.task_id,
                    item_count=len(recall.items),
                    dry_run_promotion=True,
                ),
            ],
        )

    def _handle_failure(
        self,
        action: SPAction,
        context: HandlerContext,
        request_entry: StackMemoryEntry,
        result: SPSubagentResult,
    ) -> HandlerResult:
        error = result.error or f"memory_recaller failed with status {result.status.value}"
        error_entry = context.stack.append(
            StackMemoryEntry(
                thread_id=context.thread_id,
                run_id=context.run_id,
                actor=self._subagent_type,
                action="error",
                content=error,
                result_ref=result.task_id,
                priority="high",
                stage=action.stage,
                parent_ids=[request_entry.id],
                failure_note=error,
                metadata={"action_id": action.action_id, "status": result.status.value},
            )
        )
        return HandlerResult(
            next_step="error_recoverable",
            memory_entries=[request_entry, error_entry],
            idempotency_key=action.idempotency_key,
            error=error,
            run_events=[
                make_sp_event("sp.memory.recall.started", action_id=action.action_id, run_id=context.run_id),
                make_sp_event("sp.memory.recall.failed", action_id=action.action_id, run_id=context.run_id, task_id=result.task_id, error=error),
            ],
        )
