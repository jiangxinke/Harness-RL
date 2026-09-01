"""Expose only canonical SP tools while retaining one hidden Qwen alias."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse

SP_QWEN_TOOL_ALIASES = {"sp_thought": "sp_think"}


class SPQwenActionNormalizationMiddleware(AgentMiddleware[AgentState]):
    """Hide compatibility aliases from model schemas.

    Alias tools remain registered with LangGraph so a sampled Qwen near-match
    can execute without rewriting its provider message. Filtering them from
    every model binding means the policy is trained against only the canonical
    vocabulary, while slime retains the exact original token IDs, logprobs,
    and replay history. Unknown near-matches are not registered and fail closed.
    """

    state_schema = AgentState

    @staticmethod
    def _filter_request(request: ModelRequest) -> ModelRequest:
        active = [tool for tool in request.tools if getattr(tool, "name", None) not in SP_QWEN_TOOL_ALIASES]
        return request.override(tools=active)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        return handler(self._filter_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        return await handler(self._filter_request(request))
