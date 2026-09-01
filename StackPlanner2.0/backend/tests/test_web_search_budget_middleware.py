"""Tests for deterministic web-search limits in lead and subagent runtimes."""

import asyncio
import json
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deerflow.agents.middlewares.web_search_budget_middleware import (
    MAX_WEB_SEARCHES_PER_TURN,
    WebSearchBudgetMiddleware,
)


def _call(call_id: str, query: str) -> dict:
    return {
        "id": call_id,
        "name": "web_search",
        "args": {"query": query},
        "type": "tool_call",
    }


def _request(call_id: str, query: str, messages):
    return SimpleNamespace(
        tool_call=_call(call_id, query),
        state={"messages": messages},
    )


def _error_code(message: ToolMessage) -> str:
    return json.loads(str(message.content))["error"]


def test_reuses_completed_identical_search_without_calling_provider():
    middleware = WebSearchBudgetMiddleware()
    messages = [
        HumanMessage(content="Research the topic"),
        AIMessage(content="", tool_calls=[_call("old", "OpenAI official")]),
        ToolMessage(content='{"results":[{"title":"OpenAI"}]}', tool_call_id="old", name="web_search"),
        AIMessage(content="", tool_calls=[_call("new", "  openai   OFFICIAL ")]),
    ]

    result = middleware.wrap_tool_call(
        _request("new", "  openai   OFFICIAL ", messages),
        lambda _request: (_ for _ in ()).throw(AssertionError("provider must not run")),
    )

    assert result.tool_call_id == "new"
    assert result.additional_kwargs["web_search_cache_hit"] is True
    assert result.additional_kwargs["source_tool_call_id"] == "old"


def test_stops_after_two_consecutive_failed_searches():
    middleware = WebSearchBudgetMiddleware()
    messages = [
        HumanMessage(content="Research the topic"),
        AIMessage(content="", tool_calls=[_call("one", "query one")]),
        ToolMessage(content='{"error":"No results found"}', tool_call_id="one", name="web_search"),
        AIMessage(content="", tool_calls=[_call("two", "query two")]),
        ToolMessage(content='{"error":"WEB_SEARCH_UNAVAILABLE"}', tool_call_id="two", name="web_search"),
        AIMessage(content="", tool_calls=[_call("three", "query three")]),
    ]

    result = middleware.wrap_tool_call(
        _request("three", "query three", messages),
        lambda _request: (_ for _ in ()).throw(AssertionError("provider must not run")),
    )

    assert _error_code(result) == "WEB_SEARCH_ATTEMPTS_EXHAUSTED"
    assert result.status == "error"


def test_stops_after_total_search_budget():
    middleware = WebSearchBudgetMiddleware()
    messages = [HumanMessage(content="Research the topic")]
    for index in range(MAX_WEB_SEARCHES_PER_TURN):
        call_id = f"old-{index}"
        messages.extend(
            [
                AIMessage(content="", tool_calls=[_call(call_id, f"query {index}")]),
                ToolMessage(content='{"results":[{"title":"ok"}]}', tool_call_id=call_id, name="web_search"),
            ]
        )
    messages.append(AIMessage(content="", tool_calls=[_call("new", "one more query")]))

    result = middleware.wrap_tool_call(
        _request("new", "one more query", messages),
        lambda _request: (_ for _ in ()).throw(AssertionError("provider must not run")),
    )

    assert _error_code(result) == "WEB_SEARCH_BUDGET_EXHAUSTED"


def test_skips_duplicate_and_excess_parallel_searches_in_same_model_turn():
    middleware = WebSearchBudgetMiddleware()
    calls = [
        _call("one", "same"),
        _call("two", "same"),
        _call("three", "third"),
        _call("four", "fourth"),
        _call("five", "fifth"),
    ]
    messages = [HumanMessage(content="Research"), AIMessage(content="", tool_calls=calls)]

    duplicate = middleware.wrap_tool_call(
        _request("two", "same", messages),
        lambda _request: (_ for _ in ()).throw(AssertionError("provider must not run")),
    )
    excess = middleware.wrap_tool_call(
        _request("five", "fifth", messages),
        lambda _request: (_ for _ in ()).throw(AssertionError("provider must not run")),
    )

    assert _error_code(duplicate) == "WEB_SEARCH_DUPLICATE_SKIPPED"
    assert duplicate.status == "success"
    assert _error_code(excess) == "WEB_SEARCH_PARALLEL_LIMIT"


def test_allows_normal_search_in_sync_and_async_paths():
    middleware = WebSearchBudgetMiddleware()
    request = _request(
        "new",
        "fresh query",
        [HumanMessage(content="Research"), AIMessage(content="", tool_calls=[_call("new", "fresh query")])],
    )
    expected = ToolMessage(content="ok", tool_call_id="new", name="web_search")

    assert middleware.wrap_tool_call(request, lambda _request: expected) is expected

    async def run():
        async def handler(_request):
            return expected

        return await middleware.awrap_tool_call(request, handler)

    assert asyncio.run(run()) is expected
