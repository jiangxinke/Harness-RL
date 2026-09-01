from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from deerflow.sp.middlewares.training_route_bootstrap_middleware import (
    SPTrainingRouteBootstrapMiddleware,
)


def _tool(name: str):
    return SimpleNamespace(name=name)


def _request(messages, tools):
    return SimpleNamespace(
        messages=messages,
        tools=tools,
        override=lambda **kwargs: kwargs,
    )


def test_training_route_bootstrap_requires_only_delegate_before_first_delegate():
    middleware = SPTrainingRouteBootstrapMiddleware()
    request = _request(
        [HumanMessage(content="question")],
        [_tool("sp_think"), _tool("sp_delegate"), _tool("sp_finish")],
    )

    prepared = middleware._prepare(request)

    assert prepared["tool_choice"] == "required"
    assert [tool.name for tool in prepared["tools"]] == ["sp_delegate"]


def test_training_route_bootstrap_stops_intervening_after_delegate_result():
    middleware = SPTrainingRouteBootstrapMiddleware()
    tools = [_tool("sp_think"), _tool("sp_delegate"), _tool("sp_finish")]
    request = _request(
        [
            HumanMessage(content="question"),
            ToolMessage(
                content="delegated evidence",
                tool_call_id="delegate-1",
                name="sp_delegate",
            ),
            HumanMessage(
                content="hidden task context",
                name="sp_task_memory_context",
                additional_kwargs={"hide_from_ui": True},
            ),
        ],
        tools,
    )

    assert middleware._prepare(request) is request


def test_training_route_bootstrap_is_scoped_to_latest_visible_user_turn():
    middleware = SPTrainingRouteBootstrapMiddleware()
    request = _request(
        [
            HumanMessage(content="old question"),
            ToolMessage(
                content="old evidence",
                tool_call_id="delegate-old",
                name="sp_delegate",
            ),
            HumanMessage(content="new question"),
        ],
        [_tool("sp_delegate"), _tool("sp_finish")],
    )

    prepared = middleware._prepare(request)

    assert prepared["tool_choice"] == "required"
    assert [tool.name for tool in prepared["tools"]] == ["sp_delegate"]


def test_training_route_bootstrap_fails_closed_without_delegate_tool():
    middleware = SPTrainingRouteBootstrapMiddleware()
    request = _request(
        [HumanMessage(content="question")],
        [_tool("sp_think"), _tool("sp_finish")],
    )

    with pytest.raises(RuntimeError, match="exactly one sp_delegate"):
        middleware._prepare(request)
