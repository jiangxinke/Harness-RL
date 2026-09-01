"""Recoverable SP CentralAgent action loop."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from deerflow.sp.actions import ActionRouter, HandlerNextStep, HandlerResult, SPAction, build_default_action_router
from deerflow.sp.actions.events import make_sp_event
from deerflow.sp.central.prompt import CENTRAL_AGENT_ACTION_PROMPT
from deerflow.sp.memory import TaskMemoryStack
from deerflow.sp.prompt import PromptContextBuilder

DEFAULT_ACTION_LOOP_MAX_ITERATIONS = 20


@dataclass(slots=True)
class CentralDecisionRequest:
    """Context passed to the SP CentralAgent decision policy."""

    system_prompt: str
    task_context: str
    state: Mapping[str, Any]
    iteration: int
    thread_id: str | None = None
    run_id: str | None = None


class CentralActionDecider(Protocol):
    """Decision boundary for SP CentralAgent implementations."""

    def decide(self, request: CentralDecisionRequest) -> SPAction | dict[str, Any]:
        """Return the next structured SP action."""


@dataclass(slots=True)
class ActionLoopResult:
    """Terminal result of one SP action-loop run."""

    next_step: HandlerNextStep
    state_update: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    run_events: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    last_handler_result: HandlerResult | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "next_step": self.next_step,
            "state_update": self.state_update,
            "state": self.state,
            "run_events": self.run_events,
            "iterations": self.iterations,
            "last_handler_result": self.last_handler_result.to_dict() if self.last_handler_result else None,
            "error": self.error,
        }


class ActionLoop:
    """Run CentralAgent decisions through ActionRouter until a terminal step."""

    def __init__(
        self,
        *,
        decider: CentralActionDecider,
        router: ActionRouter | None = None,
        context_builder: PromptContextBuilder | None = None,
        max_iterations: int = DEFAULT_ACTION_LOOP_MAX_ITERATIONS,
        system_prompt: str = CENTRAL_AGENT_ACTION_PROMPT,
    ) -> None:
        self._decider = decider
        self._router = router or build_default_action_router()
        self._context_builder = context_builder or PromptContextBuilder()
        self._max_iterations = max(1, int(max_iterations))
        self._system_prompt = system_prompt

    def run(
        self,
        state: Mapping[str, Any] | None = None,
        *,
        thread_id: str | None = None,
        run_id: str | None = None,
    ) -> ActionLoopResult:
        working_state = dict(state or {})
        working_state.setdefault("sp_max_loop_iterations", self._max_iterations)
        accumulated_update: dict[str, Any] = {"sp_max_loop_iterations": working_state["sp_max_loop_iterations"]}
        events = [make_sp_event("sp.loop.started", run_id=run_id, max_iterations=self._max_iterations)]

        last_result: HandlerResult | None = None
        for iteration in range(1, self._max_iterations + 1):
            request = self.prepare_context(working_state, iteration=iteration, thread_id=thread_id, run_id=run_id)
            events.append(make_sp_event("sp.loop.context_prepared", run_id=run_id, iteration=iteration, context_chars=len(request.task_context)))

            try:
                action_input = self._decider.decide(request)
            except Exception as exc:
                error = f"CentralAgent decision failed: {exc}"
                events.append(make_sp_event("sp.central.failed", run_id=run_id, iteration=iteration, error=error))
                return ActionLoopResult(
                    next_step="error_fatal",
                    state_update=accumulated_update,
                    state=working_state,
                    run_events=events,
                    iterations=iteration,
                    last_handler_result=last_result,
                    error=error,
                )

            events.append(make_sp_event("sp.central.decided", run_id=run_id, iteration=iteration))
            handler_result = self._router.execute(action_input, state=working_state, thread_id=thread_id, run_id=run_id)
            last_result = handler_result
            events.extend(handler_result.run_events)
            if handler_result.state_update:
                working_state.update(handler_result.state_update)
                accumulated_update.update(handler_result.state_update)

            events.append(make_sp_event("sp.loop.route_next", run_id=run_id, iteration=iteration, next_step=handler_result.next_step))
            if handler_result.next_step in {"interrupt", "finish", "error_fatal"}:
                events.append(make_sp_event("sp.loop.completed", run_id=run_id, iteration=iteration, next_step=handler_result.next_step))
                return ActionLoopResult(
                    next_step=handler_result.next_step,
                    state_update=accumulated_update,
                    state=working_state,
                    run_events=events,
                    iterations=iteration,
                    last_handler_result=handler_result,
                    error=handler_result.error,
                )

        error = f"SP action loop exceeded max iterations: {self._max_iterations}"
        events.append(make_sp_event("sp.loop.max_iterations_exceeded", run_id=run_id, max_iterations=self._max_iterations, error=error))
        accumulated_update["sp_last_handler_result"] = {
            "next_step": "error_fatal",
            "error": error,
        }
        working_state.update(accumulated_update)
        return ActionLoopResult(
            next_step="error_fatal",
            state_update=accumulated_update,
            state=working_state,
            run_events=events,
            iterations=self._max_iterations,
            last_handler_result=last_result,
            error=error,
        )

    def prepare_context(
        self,
        state: Mapping[str, Any],
        *,
        iteration: int,
        thread_id: str | None = None,
        run_id: str | None = None,
    ) -> CentralDecisionRequest:
        stack = TaskMemoryStack.from_dict(state.get("sp_task_memory"), thread_id=thread_id, run_id=run_id)
        pending_human = state.get("sp_pending_human_interaction") if isinstance(state.get("sp_pending_human_interaction"), Mapping) else None
        artifact_refs = state.get("sp_current_artifact_refs") if isinstance(state.get("sp_current_artifact_refs"), Mapping) else None
        context = self._context_builder.build(
            stack,
            current_stage=state.get("sp_current_stage"),
            active_delegate_id=state.get("sp_active_delegate_id"),
            pending_human_interaction=pending_human,
            artifact_refs=artifact_refs,
            report_version=state.get("sp_current_report_version"),
            current_run_id=run_id,
        )
        return CentralDecisionRequest(
            system_prompt=self._system_prompt,
            task_context=context,
            state=state,
            iteration=iteration,
            thread_id=thread_id,
            run_id=run_id,
        )
