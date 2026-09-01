"""Bound the evidence-gathering loop used by Search-R1 inference."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import HumanMessage, ToolMessage


class SPSearchInferenceMiddleware(AgentMiddleware[AgentState]):
    """Require initial retrieval and stop exposing search after a small budget.

    Search-R1 questions are external-evidence tasks.  Letting a small policy
    answer before seeing a retrieval result makes model-size comparisons mostly
    measure parametric recall, while leaving the tool exposed indefinitely
    recreates the search loops observed in failed StackPlanner trajectories.
    """

    state_schema = AgentState

    def __init__(self, *, min_searches: int = 2, max_searches: int = 3) -> None:
        if min_searches < 1 or max_searches < min_searches:
            raise ValueError("search limits must satisfy 1 <= min <= max")
        self.min_searches = min_searches
        self.max_searches = max_searches

    @staticmethod
    def _search_count(request: ModelRequest) -> int:
        messages = list(request.messages)
        start = 0
        for index, message in enumerate(messages):
            if isinstance(message, HumanMessage):
                start = index + 1
        return sum(
            isinstance(message, ToolMessage) and str(message.name or "") == "web_search"
            for message in messages[start:]
        )

    def _prepare(self, request: ModelRequest) -> ModelRequest:
        search_count = self._search_count(request)
        if search_count < self.min_searches:
            return request.override(tool_choice="required")
        if search_count >= self.max_searches:
            return request.override(tools=[], tool_choice="none")
        return request.override(tool_choice="auto")

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        return handler(self._prepare(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        return await handler(self._prepare(request))
