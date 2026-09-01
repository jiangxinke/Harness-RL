"""Middleware for SP short-term task memory.

This middleware restores StackPlanner task memory from DeerFlow ThreadState,
keeps the serialized stack normalized for checkpointing, and injects a bounded
request-only prompt context for the CentralAgent. It does not register with the
default Lead Agent chain; SP orchestration should opt into it explicitly.
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, override

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.config import get_config
from langgraph.runtime import Runtime

from deerflow.agents.thread_state import ThreadState
from deerflow.sp.memory import StackMemoryEntry, TaskMemoryStack
from deerflow.sp.prompt import PromptContextBuilder
from deerflow.utils.messages import message_to_text

SP_TASK_CONTEXT_MESSAGE_NAME = "sp_task_memory_context"
SP_TASK_CONTEXT_KWARG = "sp_task_memory_context"
AUTO_SUMMARY_MAX_CHARS = 1100
AUTO_SUMMARY_ENTRY_MAX_CHARS = 160
DEBUG_TASK_CONTEXT_MAX_CHARS = 24000


def _record_debug_task_context(
    runtime: Runtime | None,
    *,
    context: str,
    stack: TaskMemoryStack,
    current_stage: Any,
) -> None:
    """Persist the exact task-memory block sent to CentralAgent in debug mode."""

    runtime_context = getattr(runtime, "context", None)
    if not isinstance(runtime_context, Mapping) or not bool(runtime_context.get("debug_trace_enabled")):
        return
    journal = runtime_context.get("__run_journal")
    record = getattr(journal, "record_custom_event", None)
    if not callable(record):
        return
    active_entries = stack.get_active_entries()
    record(
        "sp.central.context",
        content={
            "prompt_context": context[:DEBUG_TASK_CONTEXT_MAX_CHARS],
            "prompt_context_chars": len(context),
            "prompt_context_truncated": len(context) > DEBUG_TASK_CONTEXT_MAX_CHARS,
            "prompt_context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
            "current_stage": current_stage,
            "active_memory_entry_ids": [entry.id for entry in active_entries],
            "active_memory_entry_count": len(active_entries),
        },
        metadata={
            "source": "stackplanner",
            "run_id": _run_id(runtime),
            "debug_trace": True,
        },
    )


def _automatic_fallback_summary(
    stack: TaskMemoryStack,
    source_entry_ids: list[str],
) -> str:
    """Build a bounded extractive checkpoint without inventing task facts."""
    source_set = set(source_entry_ids)
    lines = ["Task-memory checkpoint (automatic fallback):"]
    for entry in stack.entries:
        if entry.id not in source_set:
            continue
        content = " ".join(str(entry.content).split())
        if len(content) > AUTO_SUMMARY_ENTRY_MAX_CHARS:
            content = f"{content[: AUTO_SUMMARY_ENTRY_MAX_CHARS - 15]}...<truncated>"
        line = f"- {entry.action}: {content}"
        if len("\n".join([*lines, line])) > AUTO_SUMMARY_MAX_CHARS:
            break
        lines.append(line)
    return "\n".join(lines)


def _record_auto_summary_event(
    runtime: Runtime,
    *,
    source_entry_ids: list[str],
    summary_entry_id: str,
) -> None:
    context = getattr(runtime, "context", None)
    journal = context.get("__run_journal") if isinstance(context, Mapping) else None
    record = getattr(journal, "record_custom_event", None)
    if not callable(record):
        return
    record(
        "sp.memory.auto_summarized",
        content={
            "summary_entry_id": summary_entry_id,
            "source_entry_ids": source_entry_ids,
            "source_entry_count": len(source_entry_ids),
        },
        metadata={
            "source": "stackplanner",
            "run_id": _run_id(runtime),
            "automatic_fallback": True,
        },
    )


class TaskMemoryMiddleware(AgentMiddleware[ThreadState]):
    """Restore and inject SP task memory without creating a new runtime store."""

    state_schema = ThreadState

    def __init__(
        self,
        *,
        context_builder: PromptContextBuilder | None = None,
        max_stack_entries: int = 50,
        max_active_entries: int = 24,
        max_active_chars: int = 12000,
        inject_context: bool = True,
        manage_run_boundary: bool = True,
    ) -> None:
        super().__init__()
        self._context_builder = context_builder or PromptContextBuilder()
        self._max_stack_entries = max_stack_entries
        self._max_active_entries = max_active_entries
        self._max_active_chars = max_active_chars
        self._inject_context = inject_context
        self._manage_run_boundary = manage_run_boundary

    @override
    def before_agent(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        stack = self._restore_stack(state, runtime)
        update: dict[str, Any] = {}
        fresh_user_turn = _is_fresh_runtime(runtime)
        if fresh_user_turn:
            # A user explicitly started over after an abnormal run. Keep only
            # durable short-term continuity and human authority; abandoned raw
            # execution steps must not steer the new task.
            stack = TaskMemoryStack(
                (entry for entry in stack.entries if entry.status == "pinned" or entry.priority == "critical" or (entry.status == "active" and entry.action in {"summarize", "finish"})),
                max_size=stack.max_size,
                max_history_size=stack.max_history_size,
            )

        if self._manage_run_boundary:
            update.update(self._new_run_update(state, runtime, stack, fresh_user_turn=fresh_user_turn))

        if stack.is_empty() and not self._has_raw_stack(state) and not update:
            return None
        stack.prune(max_entries=self._max_active_entries, max_chars=self._max_active_chars)
        payload = stack.to_dict()
        if payload != state.get("sp_task_memory"):
            update["sp_task_memory"] = payload
        return update or None

    @override
    async def abefore_agent(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        return self.before_agent(state, runtime)

    @override
    def after_agent(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        stack = self._restore_stack(state, runtime)
        if stack.is_empty() and not self._has_raw_stack(state):
            return None
        run_id = _run_id(runtime)
        auto_summarized = False
        source_entry_ids = stack.select_summarization_source_ids()
        if source_entry_ids and stack.can_summarize(current_run_id=run_id):
            summary = stack.condense(
                source_entry_ids,
                _automatic_fallback_summary(stack, source_entry_ids),
                actor="central",
                thread_id=_thread_id(runtime),
                run_id=run_id,
                stage=state.get("sp_current_stage"),
                metadata={
                    "automatic_fallback": True,
                    "source_entry_count": len(source_entry_ids),
                },
            )
            _record_auto_summary_event(
                runtime,
                source_entry_ids=source_entry_ids,
                summary_entry_id=summary.id,
            )
            auto_summarized = True
        stack.prune(max_entries=self._max_active_entries, max_chars=self._max_active_chars)
        payload = stack.to_dict()
        if payload == state.get("sp_task_memory") and not auto_summarized:
            return None
        update: dict[str, Any] = {"sp_task_memory": payload}
        if auto_summarized:
            update["sp_summarize_committed_run_id"] = run_id
        return update

    @override
    async def aafter_agent(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        return self.after_agent(state, runtime)

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        return handler(self._augment_request(request))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        return await handler(self._augment_request(request))

    def _augment_request(self, request: ModelRequest) -> ModelRequest:
        messages = getattr(request, "messages", None)
        if not isinstance(messages, list):
            return request
        if _is_fresh_user_turn(request):
            # Drop the abandoned tool-call transcript, but still inject the
            # compact SP task context retained for the new user turn.
            request = request.override(messages=_isolate_fresh_turn_messages(messages))
            messages = getattr(request, "messages", None)
            if not isinstance(messages, list):
                return request
        if not self._inject_context:
            return request
        if _has_task_context_message(messages):
            # Request-only context should never persist, but older checkpoints
            # and custom callers may contain one. Replace it with a fresh
            # snapshot instead of trusting stale pre-summary memory.
            request = request.override(messages=[message for message in messages if getattr(message, "name", None) != SP_TASK_CONTEXT_MESSAGE_NAME])
            messages = getattr(request, "messages", None)
            if not isinstance(messages, list):
                return request

        runtime = getattr(request, "runtime", None)
        state = _request_state(request)
        stack = self._restore_stack(state, runtime)
        if stack.is_empty() and not _has_context_refs(state):
            return request

        context = self._context_builder.build(
            stack,
            current_stage=state.get("sp_current_stage"),
            active_delegate_id=state.get("sp_active_delegate_id"),
            pending_human_interaction=_mapping_or_none(state.get("sp_pending_human_interaction")),
            artifact_refs=_mapping_or_none(state.get("sp_current_artifact_refs")),
            report_version=state.get("sp_current_report_version"),
            current_run_id=_run_id(runtime),
        )
        _record_debug_task_context(
            runtime,
            context=context,
            stack=stack,
            current_stage=state.get("sp_current_stage"),
        )
        context_message = HumanMessage(
            content=context,
            name=SP_TASK_CONTEXT_MESSAGE_NAME,
            additional_kwargs={"hide_from_ui": True, SP_TASK_CONTEXT_KWARG: True},
        )
        return request.override(messages=_insert_before_last_human(messages, context_message))

    def _restore_stack(self, state: Mapping[str, Any] | None, runtime: Runtime | None) -> TaskMemoryStack:
        state = state or {}
        thread_id = _thread_id(runtime)
        run_id = _run_id(runtime)
        raw = self._raw_stack_payload(state)
        stack = TaskMemoryStack.from_dict(raw, thread_id=thread_id, run_id=run_id, max_size=self._max_stack_entries)
        for entry in stack.entries:
            if thread_id and entry.thread_id is None:
                entry.thread_id = thread_id
            if run_id and entry.run_id is None:
                entry.run_id = run_id
        return stack

    def _new_run_update(
        self,
        state: Mapping[str, Any],
        runtime: Runtime,
        stack: TaskMemoryStack,
        *,
        fresh_user_turn: bool,
    ) -> dict[str, Any]:
        run_id = _run_id(runtime)
        if not run_id or str(state.get("sp_loop_run_id") or "") == run_id:
            return {}

        pending_human = state.get("sp_pending_human_interaction")
        continuing_human_interaction = isinstance(pending_human, Mapping) and not fresh_user_turn
        if not continuing_human_interaction:
            latest_user = _latest_visible_user_message(state)
            if latest_user is not None:
                source_message_id = str(getattr(latest_user, "id", None) or "")
                already_recorded = any(entry.action == "user_request" and entry.run_id == run_id and str(entry.metadata.get("source_message_id") or "") == source_message_id for entry in stack.entries)
                if not already_recorded:
                    stack.append(
                        StackMemoryEntry(
                            thread_id=_thread_id(runtime),
                            run_id=run_id,
                            actor="human",
                            action="user_request",
                            content=message_to_text(latest_user),
                            priority="high",
                            stage="perception",
                            metadata={
                                "source_message_id": source_message_id or None,
                                "run_boundary": True,
                            },
                        )
                    )

        update: dict[str, Any] = {
            "sp_loop_run_id": run_id,
            "sp_loop_iteration": 0,
            "sp_decision_attempts": 0,
            "sp_last_handler_result": None,
            "sp_last_action_id": None,
            "sp_last_idempotency_key": None,
            "sp_current_action": None,
            "sp_current_action_id": None,
            "sp_active_delegate_id": None,
            "sp_last_run_summary": None,
            "sp_last_final_artifact_ref": None,
            "sp_summarize_committed_run_id": None,
        }
        if not continuing_human_interaction:
            update["sp_current_stage"] = "perception"
            update["sp_current_report_version"] = None
        if fresh_user_turn:
            update["sp_pending_human_interaction"] = None
            update["sp_current_artifact_refs"] = None
        return update

    @staticmethod
    def _raw_stack_payload(state: Mapping[str, Any]) -> Any:
        if "sp_task_memory" in state:
            return state.get("sp_task_memory")
        return state.get("memory_stack")

    @staticmethod
    def _has_raw_stack(state: Mapping[str, Any] | None) -> bool:
        return bool(state is not None and ("sp_task_memory" in state or "memory_stack" in state))


def _request_state(request: ModelRequest) -> Mapping[str, Any]:
    state = getattr(request, "state", None)
    if isinstance(state, Mapping):
        return state
    runtime = getattr(request, "runtime", None)
    runtime_state = getattr(runtime, "state", None)
    if isinstance(runtime_state, Mapping):
        return runtime_state
    return {}


def _mapping_or_none(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _has_context_refs(state: Mapping[str, Any]) -> bool:
    return any(
        state.get(key)
        for key in (
            "sp_current_stage",
            "sp_active_delegate_id",
            "sp_pending_human_interaction",
            "sp_current_artifact_refs",
            "sp_current_report_version",
        )
    )


def _has_task_context_message(messages: list[BaseMessage]) -> bool:
    return any(getattr(message, "name", None) == SP_TASK_CONTEXT_MESSAGE_NAME for message in messages)


def _is_fresh_user_turn(request: ModelRequest) -> bool:
    runtime = getattr(request, "runtime", None)
    return _is_fresh_runtime(runtime)


def _is_fresh_runtime(runtime: Runtime | None) -> bool:
    context = getattr(runtime, "context", None) if runtime is not None else None
    return isinstance(context, Mapping) and bool(context.get("fresh_user_turn_after_terminal"))


def _latest_visible_user_message(state: Mapping[str, Any]) -> HumanMessage | None:
    messages = state.get("messages")
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if not isinstance(message, HumanMessage):
            continue
        if getattr(message, "name", None) == SP_TASK_CONTEXT_MESSAGE_NAME:
            continue
        additional_kwargs = getattr(message, "additional_kwargs", None)
        if isinstance(additional_kwargs, Mapping) and additional_kwargs.get("hide_from_ui") is True:
            continue
        if message_to_text(message).strip():
            return message
    return None


def _isolate_fresh_turn_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Keep the current prompt while dropping an abandoned tool-call history."""
    latest_human = next(
        (message for message in reversed(messages) if isinstance(message, HumanMessage) and getattr(message, "name", None) != SP_TASK_CONTEXT_MESSAGE_NAME),
        None,
    )
    if latest_human is None:
        return messages
    system_messages = [message for message in messages if isinstance(message, SystemMessage)]
    return [*system_messages, latest_human]


def _insert_before_last_human(messages: list[BaseMessage], context_message: HumanMessage) -> list[BaseMessage]:
    for idx in reversed(range(len(messages))):
        if isinstance(messages[idx], HumanMessage):
            return [*messages[:idx], context_message, *messages[idx:]]
    return [context_message, *messages]


def _thread_id(runtime: Runtime | None) -> str | None:
    context = getattr(runtime, "context", None) if runtime is not None else None
    if isinstance(context, Mapping) and context.get("thread_id"):
        return str(context["thread_id"])
    try:
        config = get_config()
    except RuntimeError:
        return None
    thread_id = config.get("configurable", {}).get("thread_id")
    return str(thread_id) if thread_id else None


def _run_id(runtime: Runtime | None) -> str | None:
    context = getattr(runtime, "context", None) if runtime is not None else None
    if isinstance(context, Mapping) and context.get("run_id"):
        return str(context["run_id"])
    try:
        config = get_config()
    except RuntimeError:
        return None
    run_id = config.get("metadata", {}).get("run_id") or config.get("configurable", {}).get("run_id")
    return str(run_id) if run_id else None
