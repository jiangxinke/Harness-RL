"""Training-only bootstrap for the Central action-choice capability route."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import HumanMessage, ToolMessage

_TASK_CONTEXT_MESSAGE_NAME = "sp_task_memory_context"


def _is_visible_user_message(message: object) -> bool:
    if not isinstance(message, HumanMessage):
        return False
    if getattr(message, "name", None) == _TASK_CONTEXT_MESSAGE_NAME:
        return False
    additional_kwargs = getattr(message, "additional_kwargs", None)
    return not (isinstance(additional_kwargs, Mapping) and additional_kwargs.get("hide_from_ui") is True)


class SPTrainingRouteBootstrapMiddleware(AgentMiddleware[AgentState]):
    """Require one initial ``sp_delegate`` call in a training episode.

    The middleware narrows the first Central model request to the single
    canonical delegation tool. As soon as the current visible user turn has a
    delegation result, it stops modifying requests and the production control
    policy resumes unchanged. The runtime factory only installs this
    middleware when an explicit training model factory is present.
    """

    state_schema = AgentState

    @staticmethod
    def _delegate_seen_in_current_turn(request: ModelRequest) -> bool:
        messages = list(request.messages)
        start = 0
        for index, message in enumerate(messages):
            if _is_visible_user_message(message):
                start = index + 1
        return any(isinstance(message, ToolMessage) and str(message.name or "") == "sp_delegate" for message in messages[start:])

    def _prepare(self, request: ModelRequest) -> ModelRequest:
        if self._delegate_seen_in_current_turn(request):
            return request
        delegate_tools = [tool for tool in request.tools if getattr(tool, "name", None) == "sp_delegate"]
        if len(delegate_tools) != 1:
            raise RuntimeError("training route bootstrap requires exactly one sp_delegate tool")
        return request.override(tools=delegate_tools, tool_choice="required")

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
