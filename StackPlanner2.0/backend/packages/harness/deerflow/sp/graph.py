"""Checkpointable StackPlanner orchestration graph running on DR2 state."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from typing import Any, Literal, Protocol

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from deerflow.agents.human_input import read_human_input_response
from deerflow.agents.middlewares.title_middleware import TitleMiddleware
from deerflow.agents.thread_state import ThreadState
from deerflow.sp.actions import ActionRouter, ActionValidationError, SPAction, build_default_action_router
from deerflow.sp.actions.events import make_sp_event
from deerflow.sp.hitl import record_human_feedback
from deerflow.sp.loop import CentralActionDecider, CentralDecisionRequest
from deerflow.sp.memory import MemoryCandidateExtractor, MemoryPromotionHook, StackMemoryEntry, TaskMemoryStack
from deerflow.sp.middlewares import TaskMemoryMiddleware
from deerflow.sp.prompt import PromptContextBuilder
from deerflow.sp.subagents import SPSubagentExecutorProtocol
from deerflow.utils.messages import message_to_text

logger = logging.getLogger(__name__)

DEFAULT_SP_MAX_ITERATIONS = 20
STACKPLANNER_HUMAN_INPUT_SOURCE = "stackplanner"
DEBUG_DECISION_CONTEXT_MAX_CHARS = 24000


class SPExecutorProvider(Protocol):
    """Build a state-bound DR2 subagent adapter for one action execution."""

    def __call__(self, state: Mapping[str, Any], runtime: Runtime) -> SPSubagentExecutorProtocol:
        """Return the executor using this graph run's DR2 runtime context."""


RouteName = Literal["central_decide", "execute_action", "human_request", "final_response", "final_error"]
HumanRequestRoute = Literal["end", "final_error"]


def _runtime_context(runtime: Runtime | None) -> Mapping[str, Any]:
    context = getattr(runtime, "context", None)
    return context if isinstance(context, Mapping) else {}


def _runtime_id(runtime: Runtime | None, key: str) -> str | None:
    value = _runtime_context(runtime).get(key)
    return str(value) if value else None


def _emit_events(runtime: Runtime | None, events: list[Mapping[str, Any]]) -> None:
    journal = _runtime_context(runtime).get("__run_journal")
    record = getattr(journal, "record_custom_event", None)
    if not callable(record):
        return
    for event in events:
        try:
            payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
            record(
                str(event.get("event_type") or "sp.event"),
                content={
                    "action_id": event.get("action_id"),
                    "payload": dict(payload),
                    "occurred_at": event.get("ts"),
                },
                metadata={
                    "source": "stackplanner",
                    "action_id": event.get("action_id"),
                    "run_id": event.get("run_id"),
                    "occurred_at": event.get("ts"),
                },
            )
        except Exception:
            logger.warning("Failed to record StackPlanner run event", exc_info=True)


def _human_input_request(message: Any) -> Mapping[str, Any] | None:
    if not isinstance(message, ToolMessage):
        return None
    artifact = getattr(message, "artifact", None)
    if not isinstance(artifact, Mapping):
        return None
    request = artifact.get("human_input")
    return request if isinstance(request, Mapping) else None


def _pending_feedback_value(state: Mapping[str, Any], pending: Mapping[str, Any]) -> str | None:
    messages = state.get("messages")
    if not isinstance(messages, list):
        return None
    request_id = str(pending.get("interaction_id") or "")
    request_index: int | None = None
    for index, message in enumerate(messages):
        request = _human_input_request(message)
        if request and str(request.get("request_id") or "") == request_id:
            request_index = index

    for message in reversed(messages[(request_index + 1) if request_index is not None else 0 :]):
        if not isinstance(message, HumanMessage):
            continue
        response = read_human_input_response(message.additional_kwargs)
        if response is not None and (not request_id or response["request_id"] == request_id):
            return response["value"].strip()
        if request_index is not None:
            fallback = message_to_text(message).strip()
            if fallback:
                return fallback
    return None


def _context_request(
    state: Mapping[str, Any],
    *,
    context_builder: PromptContextBuilder,
    system_prompt: str,
    decision_context: str,
    iteration: int,
    thread_id: str | None,
    run_id: str | None,
) -> CentralDecisionRequest:
    stack = TaskMemoryStack.from_dict(state.get("sp_task_memory"), thread_id=thread_id, run_id=run_id)
    pending = state.get("sp_pending_human_interaction")
    artifact_refs = state.get("sp_current_artifact_refs")
    task_context = context_builder.build(
        stack,
        current_stage=state.get("sp_current_stage"),
        active_delegate_id=state.get("sp_active_delegate_id"),
        pending_human_interaction=pending if isinstance(pending, Mapping) else None,
        artifact_refs=artifact_refs if isinstance(artifact_refs, Mapping) else None,
        report_version=state.get("sp_current_report_version"),
        current_run_id=run_id,
    )
    if decision_context:
        task_context = f"{decision_context}\n\n{task_context}"
    return CentralDecisionRequest(
        system_prompt=system_prompt,
        task_context=task_context,
        state=state,
        iteration=iteration,
        thread_id=thread_id,
        run_id=run_id,
    )


def _bounded_error_entry(
    error: str,
    *,
    thread_id: str | None,
    run_id: str | None,
    stage: str | None,
    metadata: dict[str, Any] | None = None,
) -> StackMemoryEntry:
    return StackMemoryEntry(
        thread_id=thread_id,
        run_id=run_id,
        actor="central",
        action="error",
        content=error,
        priority="high",
        stage=stage,
        failure_note=error,
        metadata=metadata or {},
    )


def _route_from_last_result(state: Mapping[str, Any]) -> RouteName:
    result = state.get("sp_last_handler_result")
    next_step = str(result.get("next_step") or "error_fatal") if isinstance(result, Mapping) else "error_fatal"
    if next_step in {"continue", "error_recoverable"}:
        return "central_decide"
    if next_step == "interrupt":
        return "human_request"
    if next_step == "finish":
        return "final_response"
    return "final_error"


def _validation_route(state: Mapping[str, Any]) -> Literal["execute_action", "route_next"]:
    return "execute_action" if isinstance(state.get("sp_current_action"), Mapping) else "route_next"


def _human_request_route(state: Mapping[str, Any]) -> HumanRequestRoute:
    return "end" if isinstance(state.get("sp_pending_human_interaction"), Mapping) else "final_error"


def _human_input_payload(pending: Mapping[str, Any]) -> dict[str, Any]:
    metadata = pending.get("metadata") if isinstance(pending.get("metadata"), Mapping) else {}
    raw_options = metadata.get("options")
    options = [str(option) for option in raw_options] if isinstance(raw_options, list) else []
    payload: dict[str, Any] = {
        "version": 1,
        "kind": "human_input_request",
        "source": STACKPLANNER_HUMAN_INPUT_SOURCE,
        "request_id": str(pending.get("interaction_id") or ""),
        "clarification_type": str(pending.get("interaction_type") or "clarification"),
        "title": str(metadata.get("title") or "StackPlanner needs your input"),
        "question": str(pending.get("question") or "Please provide the requested feedback."),
        "input_mode": "choice_with_other" if options else "free_text",
    }
    context = metadata.get("context")
    if context is not None:
        payload["context"] = str(context)
    if options:
        payload["options"] = [{"id": f"option-{index}", "label": option, "value": option} for index, option in enumerate(options, 1)]
    return payload


def _final_artifact_line(state: Mapping[str, Any], *, run_id: str | None = None) -> str | None:
    refs = state.get("sp_current_artifact_refs")
    if not isinstance(refs, Mapping):
        return None
    raw_memory = state.get("sp_task_memory")
    entries = raw_memory.get("entries") if isinstance(raw_memory, Mapping) else None
    has_current_feedback = bool(run_id and isinstance(entries, list) and any(isinstance(entry, Mapping) and entry.get("action") == "feedback" and entry.get("run_id") == run_id for entry in entries))
    state_run_id = state.get("sp_loop_run_id")
    for key in ("final_report", "report_revision", "report", "generated_file", "outline"):
        ref = refs.get(key)
        if not isinstance(ref, Mapping):
            continue
        explicit_ref = state.get("sp_last_final_artifact_ref")
        if run_id and state_run_id and not has_current_feedback and ref.get("run_id") != run_id and explicit_ref not in {ref.get("artifact_id"), ref.get("artifact_url"), ref.get("virtual_path")}:
            continue
        label = str(ref.get("summary") or key.replace("_", " ").title())
        url = ref.get("artifact_url")
        path = ref.get("virtual_path")
        if url:
            return f"Artifact: [{label}]({url})"
        if path:
            return f"Artifact: `{path}`"
    return None


def create_sp_agent_graph(
    *,
    decider: CentralActionDecider,
    system_prompt: str,
    decision_context: str = "",
    context_builder: PromptContextBuilder | None = None,
    router: ActionRouter | None = None,
    executor_provider: SPExecutorProvider | None = None,
    task_memory_middleware: TaskMemoryMiddleware | None = None,
    memory_candidate_extractor: MemoryCandidateExtractor | None = None,
    memory_promotion_hook: MemoryPromotionHook | None = None,
    title_middleware: TitleMiddleware | None = None,
    max_iterations: int = DEFAULT_SP_MAX_ITERATIONS,
):
    """Create the checkpointable SP policy graph consumed by the DR2 RunWorker."""
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    builder = StateGraph(ThreadState, context_schema=dict)
    prompt_builder = context_builder or PromptContextBuilder()
    task_memory = task_memory_middleware or TaskMemoryMiddleware(
        context_builder=prompt_builder,
        inject_context=False,
        manage_run_boundary=False,
    )
    candidate_extractor = memory_candidate_extractor or MemoryCandidateExtractor()
    promotion_hook = memory_promotion_hook or MemoryPromotionHook(dry_run=True)

    def title_update(state: ThreadState, runtime: Runtime, new_messages: list[Any]) -> dict[str, Any]:
        if title_middleware is None:
            return {}
        existing_messages = state.get("messages")
        effective_messages = [*(existing_messages if isinstance(existing_messages, list) else []), *new_messages]
        try:
            update = title_middleware.after_model({**state, "messages": effective_messages}, runtime)
        except Exception:
            logger.warning("Failed to generate StackPlanner thread title", exc_info=True)
            return {}
        return dict(update) if isinstance(update, Mapping) else {}

    def prepare_context(state: ThreadState, runtime: Runtime) -> dict[str, Any]:
        thread_id = _runtime_id(runtime, "thread_id")
        run_id = _runtime_id(runtime, "run_id")
        before_stack = TaskMemoryStack.from_dict(
            state.get("sp_task_memory"),
            thread_id=thread_id,
            run_id=run_id,
        )
        before_status = {entry.id: entry.status for entry in before_stack.entries}
        normalized_memory = task_memory.before_agent(state, runtime) or {}
        stack = TaskMemoryStack.from_dict(
            normalized_memory.get("sp_task_memory", state.get("sp_task_memory")),
            thread_id=thread_id,
            run_id=run_id,
        )
        fresh_user_turn = bool(_runtime_context(runtime).get("fresh_user_turn_after_terminal"))
        if fresh_user_turn:
            stack = TaskMemoryStack(
                (entry for entry in stack.entries if entry.status == "pinned" or entry.priority == "critical" or (entry.status == "active" and entry.action in {"summarize", "finish"})),
                max_size=stack.max_size,
            )
        update: dict[str, Any] = {
            "sp_task_memory": stack.to_dict(),
            "sp_current_action": None,
            "sp_current_action_id": None,
            "sp_max_loop_iterations": max_iterations,
        }
        if fresh_user_turn:
            update.update(
                {
                    "sp_pending_human_interaction": None,
                    "sp_current_artifact_refs": None,
                    "sp_current_report_version": None,
                }
            )
        is_new_run = bool(run_id and state.get("sp_loop_run_id") != run_id)
        if is_new_run:
            update.update(
                {
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
                }
            )

        effective_state = {**state, **update}
        pending = effective_state.get("sp_pending_human_interaction")
        events = [make_sp_event("sp.loop.context_prepared", run_id=run_id)]
        pruned_entry_ids = [entry.id for entry in stack.entries if entry.status == "pruned" and before_status.get(entry.id) in {"active", "pinned"}]
        if pruned_entry_ids:
            events.append(
                make_sp_event(
                    "sp.memory.pruned",
                    run_id=run_id,
                    entry_ids=pruned_entry_ids,
                    entry_count=len(pruned_entry_ids),
                )
            )
        if isinstance(pending, Mapping):
            feedback = _pending_feedback_value(effective_state, pending)
            if feedback:
                feedback_result = record_human_feedback(
                    effective_state,
                    feedback,
                    thread_id=thread_id,
                    run_id=run_id,
                )
                update.update(feedback_result.state_update)
                events.extend(feedback_result.run_events)
        if is_new_run and not any(event.get("event_type") == "sp.human.feedback_received" for event in events):
            update.update(
                {
                    "sp_current_stage": "perception",
                    "sp_current_report_version": None,
                }
            )
        _emit_events(runtime, events)
        return update

    def central_decide(state: ThreadState, runtime: Runtime) -> dict[str, Any]:
        thread_id = _runtime_id(runtime, "thread_id")
        run_id = _runtime_id(runtime, "run_id")
        iteration = int(state.get("sp_loop_iteration") or 0) + 1
        if iteration > max_iterations:
            error = f"SP action loop exceeded max iterations: {max_iterations}"
            event = make_sp_event("sp.loop.max_iterations_exceeded", run_id=run_id, error=error)
            _emit_events(runtime, [event])
            return {
                "sp_current_action": None,
                "sp_last_handler_result": {"next_step": "error_fatal", "error": error},
            }

        request = _context_request(
            state,
            context_builder=prompt_builder,
            system_prompt=system_prompt,
            decision_context=decision_context,
            iteration=iteration,
            thread_id=thread_id,
            run_id=run_id,
        )
        attempts = int(state.get("sp_decision_attempts") or 0) + 1
        try:
            raw_action = decider.decide(request)
            action_payload = raw_action.to_dict() if isinstance(raw_action, SPAction) else dict(raw_action)
        except Exception as exc:
            error = f"CentralAgent decision failed: {exc}"
            next_step = "error_fatal" if iteration >= max_iterations else "error_recoverable"
            stack = TaskMemoryStack.from_dict(state.get("sp_task_memory"), thread_id=thread_id, run_id=run_id)
            stack.append(
                _bounded_error_entry(
                    error,
                    thread_id=thread_id,
                    run_id=run_id,
                    stage=state.get("sp_current_stage"),
                    metadata={"decision_attempt": attempts},
                )
            )
            event = make_sp_event("sp.central.failed", run_id=run_id, iteration=iteration, error=error)
            _emit_events(runtime, [event])
            return {
                "sp_task_memory": stack.to_dict(),
                "sp_current_action": None,
                "sp_loop_iteration": iteration,
                "sp_decision_attempts": attempts,
                "sp_last_handler_result": {"next_step": next_step, "error": error},
            }

        decision_payload: dict[str, Any] = {
            "iteration": iteration,
            "action_id": action_payload.get("action_id"),
            "action_type": action_payload.get("action_type"),
            "reason": action_payload.get("reason"),
            "task": action_payload.get("task"),
            "target_agent": action_payload.get("target_agent"),
            "stage": action_payload.get("stage"),
            "expected_output": action_payload.get("expected_output"),
            "input_refs": action_payload.get("input_refs"),
        }
        if bool(_runtime_context(runtime).get("debug_trace_enabled")):
            # ``task_context`` is the exact working-context block handed to the
            # CentralAgent. Keeping a bounded copy makes memory-use failures
            # diagnosable without recording provider-private reasoning.
            prompt_context = str(request.task_context or "")
            decision_payload.update(
                {
                    "prompt_context": prompt_context[:DEBUG_DECISION_CONTEXT_MAX_CHARS],
                    "prompt_context_chars": len(prompt_context),
                    "prompt_context_truncated": len(prompt_context) > DEBUG_DECISION_CONTEXT_MAX_CHARS,
                    "prompt_context_sha256": hashlib.sha256(prompt_context.encode("utf-8")).hexdigest(),
                }
            )
        _emit_events(runtime, [make_sp_event("sp.central.decided", run_id=run_id, **decision_payload)])
        return {
            "sp_current_action": action_payload,
            "sp_current_action_id": action_payload.get("action_id"),
            "sp_decision_attempts": attempts,
        }

    def validate_action(state: ThreadState, runtime: Runtime) -> dict[str, Any]:
        raw_action = state.get("sp_current_action")
        run_id = _runtime_id(runtime, "run_id")
        if raw_action is None and isinstance(state.get("sp_last_handler_result"), Mapping):
            return {}
        try:
            action = SPAction.from_dict(dict(raw_action or {}))
        except (ActionValidationError, TypeError, ValueError) as exc:
            iteration = int(state.get("sp_loop_iteration") or 0) + 1
            next_step = "error_fatal" if iteration >= max_iterations else "error_recoverable"
            error = str(exc)
            event = make_sp_event("sp.action.validation_failed", run_id=run_id, error=error)
            _emit_events(runtime, [event])
            return {
                "sp_current_action": None,
                "sp_current_action_id": None,
                "sp_loop_iteration": iteration,
                "sp_last_handler_result": {"next_step": next_step, "error": error},
            }

        update: dict[str, Any] = {
            "sp_current_action": action.to_dict(),
            "sp_current_action_id": action.action_id,
        }
        if action.action_type.value == "DELEGATE":
            update["sp_active_delegate_id"] = action.action_id
        return update

    def execute_action(state: ThreadState, runtime: Runtime) -> dict[str, Any]:
        thread_id = _runtime_id(runtime, "thread_id")
        run_id = _runtime_id(runtime, "run_id")
        action_payload = state.get("sp_current_action")
        action_router = router
        try:
            if action_router is None:
                executor = executor_provider(state, runtime) if executor_provider is not None else None
                action_router = build_default_action_router(
                    delegate_executor=executor,
                    memory_recall_executor=executor,
                )
            result = action_router.execute(
                dict(action_payload or {}),
                state=state,
                thread_id=thread_id,
                run_id=run_id,
            )
        except Exception as exc:
            error = f"SP action execution failed: {exc}"
            stack = TaskMemoryStack.from_dict(state.get("sp_task_memory"), thread_id=thread_id, run_id=run_id)
            stack.append(
                _bounded_error_entry(
                    error,
                    thread_id=thread_id,
                    run_id=run_id,
                    stage=state.get("sp_current_stage"),
                    metadata={"action_id": state.get("sp_current_action_id")},
                )
            )
            iteration = int(state.get("sp_loop_iteration") or 0) + 1
            next_step = "error_fatal" if iteration >= max_iterations else "error_recoverable"
            events = [make_sp_event("sp.handler.failed", action_id=state.get("sp_current_action_id"), run_id=run_id, error=error)]
            _emit_events(runtime, events)
            return {
                "sp_task_memory": stack.to_dict(),
                "sp_current_action": None,
                "sp_current_action_id": None,
                "sp_active_delegate_id": None,
                "sp_loop_iteration": iteration,
                "sp_last_handler_result": {"next_step": next_step, "error": error},
            }

        _emit_events(runtime, result.run_events)
        return {**result.state_update, "sp_current_action": None}

    def route_next(state: ThreadState, runtime: Runtime) -> dict[str, Any]:
        result = state.get("sp_last_handler_result")
        if not isinstance(result, Mapping):
            return {"sp_last_handler_result": {"next_step": "error_fatal", "error": "Missing SP handler result"}}
        next_step = str(result.get("next_step") or "error_fatal")
        iteration = int(state.get("sp_loop_iteration") or 0)
        if next_step in {"continue", "error_recoverable"} and iteration >= max_iterations:
            error = f"SP action loop exceeded max iterations: {iteration}/{max_iterations}"
            _emit_events(runtime, [make_sp_event("sp.loop.max_iterations_exceeded", run_id=_runtime_id(runtime, "run_id"), error=error)])
            return {"sp_last_handler_result": {**dict(result), "next_step": "error_fatal", "error": error}}
        _emit_events(
            runtime,
            [
                make_sp_event(
                    "sp.loop.route_next",
                    action_id=result.get("action_id"),
                    run_id=_runtime_id(runtime, "run_id"),
                    next_step=next_step,
                )
            ],
        )
        return {}

    def human_request(state: ThreadState, runtime: Runtime) -> dict[str, Any]:
        pending = state.get("sp_pending_human_interaction")
        if not isinstance(pending, Mapping):
            return {
                "sp_last_handler_result": {
                    "next_step": "error_fatal",
                    "error": "SP interrupt requested without pending human interaction",
                }
            }
        payload = _human_input_payload(pending)
        action_id = str(state.get("sp_last_action_id") or pending.get("interaction_id") or "sp-human")
        ai_message = AIMessage(
            id=f"sp-human-action:{action_id}",
            content="",
            tool_calls=[
                {
                    "id": action_id,
                    "name": "ask_clarification",
                    "args": {"question": payload["question"]},
                    "type": "tool_call",
                }
            ],
        )
        tool_message = ToolMessage(
            id=f"sp-human-request:{payload['request_id']}",
            content=str(payload["question"]),
            tool_call_id=action_id,
            name="ask_clarification",
            artifact={"human_input": payload},
        )
        _emit_events(
            runtime,
            [
                make_sp_event(
                    "sp.human.interrupted",
                    action_id=action_id,
                    run_id=_runtime_id(runtime, "run_id"),
                    interaction_id=payload["request_id"],
                )
            ],
        )
        messages = [ai_message, tool_message]
        return {"messages": messages, **title_update(state, runtime, messages)}

    def final_response(state: ThreadState, runtime: Runtime) -> dict[str, Any]:
        content = str(state.get("sp_last_run_summary") or "Task completed.").strip()
        artifact_line = _final_artifact_line(state, run_id=_runtime_id(runtime, "run_id"))
        if artifact_line and artifact_line not in content:
            content = f"{content}\n\n{artifact_line}"
        action_id = str(state.get("sp_last_action_id") or "finish")
        message = AIMessage(
            id=f"sp-final:{action_id}",
            content=content,
            additional_kwargs={
                "stackplanner": {
                    "action_id": action_id,
                    "stage": state.get("sp_current_stage"),
                }
            },
        )
        stack = TaskMemoryStack.from_dict(
            state.get("sp_task_memory"),
            thread_id=_runtime_id(runtime, "thread_id"),
            run_id=_runtime_id(runtime, "run_id"),
        )
        refs = state.get("sp_current_artifact_refs")
        candidates = candidate_extractor.extract(
            stack,
            artifact_refs=refs if isinstance(refs, Mapping) else None,
        )
        promotion = promotion_hook.promote(candidates)
        decision_summaries = [
            {
                "candidate_id": decision.candidate.candidate_id,
                "kind": decision.candidate.kind,
                "scope": decision.candidate.scope,
                "status": decision.status,
                "reason": decision.reason,
                "content_preview": decision.candidate.content[:400],
            }
            for decision in promotion.decisions
        ]
        _emit_events(
            runtime,
            [
                make_sp_event(
                    "sp.memory.promotion.dry_run",
                    action_id=action_id,
                    run_id=_runtime_id(runtime, "run_id"),
                    candidate_count=len(candidates),
                    approved_count=promotion.approved_count,
                    written_count=promotion.written_count,
                    decisions=decision_summaries,
                ),
                make_sp_event(
                    "sp.loop.completed",
                    action_id=action_id,
                    run_id=_runtime_id(runtime, "run_id"),
                    next_step="finish",
                ),
            ],
        )
        return {
            "messages": [message],
            "sp_current_action": None,
            "sp_current_action_id": None,
            **title_update(state, runtime, [message]),
        }

    def final_error(state: ThreadState, runtime: Runtime) -> dict[str, Any]:
        result = state.get("sp_last_handler_result")
        error = str(result.get("error") or "StackPlanner stopped before completing the task.") if isinstance(result, Mapping) else "StackPlanner stopped before completing the task."
        run_id = _runtime_id(runtime, "run_id")
        message = AIMessage(
            id=f"sp-error:{run_id or 'unknown'}",
            content=f"StackPlanner could not complete this run: {error}",
            additional_kwargs={"stackplanner": {"status": "error_fatal"}},
        )
        _emit_events(runtime, [make_sp_event("sp.loop.completed", run_id=run_id, next_step="error_fatal", error=error)])
        return {"messages": [message], "sp_current_action": None, "sp_current_action_id": None, "sp_active_delegate_id": None}

    builder.add_node("prepare_context", prepare_context)
    builder.add_node("central_decide", central_decide)
    builder.add_node("validate_action", validate_action)
    builder.add_node("execute_action", execute_action)
    builder.add_node("route_next", route_next)
    builder.add_node("human_request", human_request)
    builder.add_node("final_response", final_response)
    builder.add_node("final_error", final_error)

    builder.add_edge(START, "prepare_context")
    builder.add_edge("prepare_context", "central_decide")
    builder.add_edge("central_decide", "validate_action")
    builder.add_conditional_edges(
        "validate_action",
        _validation_route,
        {"execute_action": "execute_action", "route_next": "route_next"},
    )
    builder.add_edge("execute_action", "route_next")
    builder.add_conditional_edges(
        "route_next",
        _route_from_last_result,
        {
            "central_decide": "central_decide",
            "human_request": "human_request",
            "final_response": "final_response",
            "final_error": "final_error",
        },
    )
    builder.add_conditional_edges(
        "human_request",
        _human_request_route,
        {"end": END, "final_error": "final_error"},
    )
    builder.add_edge("final_response", END)
    builder.add_edge("final_error", END)
    return builder.compile(name="stackplanner")
