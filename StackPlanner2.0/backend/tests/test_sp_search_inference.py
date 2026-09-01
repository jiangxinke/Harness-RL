from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from deerflow.sp.middlewares.search_inference_middleware import (
    SPSearchInferenceMiddleware,
)


def _request(messages):
    return SimpleNamespace(
        messages=messages,
        override=lambda **kwargs: kwargs,
    )


def test_search_inference_requires_first_search_then_allows_synthesis():
    middleware = SPSearchInferenceMiddleware(min_searches=1, max_searches=3)

    assert middleware._prepare(_request([HumanMessage(content="question")]))[
        "tool_choice"
    ] == "required"
    assert middleware._prepare(
        _request(
            [
                HumanMessage(content="question"),
                ToolMessage(
                    content="evidence",
                    tool_call_id="search-1",
                    name="web_search",
                ),
            ]
        )
    )["tool_choice"] == "auto"


def test_search_inference_removes_tool_at_budget_for_concise_answer():
    middleware = SPSearchInferenceMiddleware(min_searches=1, max_searches=2)
    request = _request(
        [
            HumanMessage(content="question"),
            ToolMessage(content="one", tool_call_id="search-1", name="web_search"),
            ToolMessage(content="two", tool_call_id="search-2", name="web_search"),
        ]
    )

    prepared = middleware._prepare(request)

    assert prepared["tool_choice"] == "none"
    assert prepared["tools"] == []


def test_search_inference_rejects_zero_budget():
    with pytest.raises(ValueError, match="1 <= min <= max"):
        SPSearchInferenceMiddleware(max_searches=0)


def test_search_inference_requires_second_search_for_verification():
    middleware = SPSearchInferenceMiddleware(min_searches=2, max_searches=3)
    request = _request(
        [
            HumanMessage(content="question"),
            ToolMessage(content="one", tool_call_id="search-1", name="web_search"),
        ]
    )

    assert middleware._prepare(request)["tool_choice"] == "required"
