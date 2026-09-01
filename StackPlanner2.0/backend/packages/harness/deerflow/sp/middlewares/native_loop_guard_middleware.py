"""A hard model-turn cap for StackPlanner's native LangChain agent loop."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import hook_config
from langchain_core.messages import AIMessage


class SPNativeLoopGuardMiddleware(AgentMiddleware[AgentState]):
    """Count every Central model attempt and stop before attempt ``max + 1``.

    ActionRouter already limits accepted ``sp_*`` actions, but malformed or
    undefined tool calls never reach it. Counting model attempts closes that
    gap and gives callers an explicit policy termination instead of relying on
    LangGraph's broad recursion guard.
    """

    state_schema = AgentState

    def __init__(self, *, max_iterations: int) -> None:
        super().__init__()
        if int(max_iterations) <= 0:
            raise ValueError("max_iterations must be positive")
        self.max_iterations = int(max_iterations)

    def before_agent(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        context = getattr(runtime, "context", None)
        run_id = str(context.get("run_id") or "") if isinstance(context, Mapping) else ""
        update: dict[str, Any] = {"sp_max_loop_iterations": self.max_iterations}
        state_run_id = str(state.get("sp_loop_run_id") or "")
        if run_id and state_run_id != run_id:
            update["sp_decision_attempts"] = 0
        return update

    async def abefore_agent(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        return self.before_agent(state, runtime)

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        result = state.get("sp_last_handler_result") if isinstance(state, Mapping) else None
        if isinstance(result, Mapping) and result.get("next_step") == "finish":
            return None
        attempts = int(state.get("sp_decision_attempts") or 0)
        if attempts < self.max_iterations:
            return None

        context = getattr(runtime, "context", None)
        run_id = str(context.get("run_id") or "") if isinstance(context, Mapping) else ""
        error = f"StackPlanner policy did not finish within {self.max_iterations} model iterations."
        message = AIMessage(
            id=f"sp-loop-capped:{run_id or 'unknown'}",
            content=error,
            additional_kwargs={
                "stackplanner": {
                    "status": "loop_capped",
                    "attempts": attempts,
                    "max_iterations": self.max_iterations,
                }
            },
        )
        return {
            "jump_to": "end",
            "messages": [message],
            "sp_last_handler_result": {
                "next_step": "error_fatal",
                "error": error,
                "termination": "loop_capped",
                "loop_iteration": attempts,
                "run_id": run_id or None,
            },
        }

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        return self.before_model(state, runtime)

    def after_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        del runtime
        return {"sp_decision_attempts": int(state.get("sp_decision_attempts") or 0) + 1}

    async def aafter_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        return self.after_model(state, runtime)
