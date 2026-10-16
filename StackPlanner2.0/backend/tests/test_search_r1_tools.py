"""Tests for the local Search-R1 retriever tool."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from deerflow.community.search_r1 import tools


@pytest.mark.asyncio
async def test_web_search_returns_ddg_compatible_results() -> None:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "result": [
            [
                {
                    "document": {
                        "id": "42",
                        "title": "Ada Lovelace",
                        "text": "English mathematician and early programmer.",
                    },
                    "score": 0.875,
                }
            ]
        ]
    }

    with (
        patch.object(tools, "_get_tool_config", return_value={"base_url": "http://node-2:8000", "max_results": 3}),
        patch("deerflow.community.search_r1.tools.httpx.AsyncClient") as client_cls,
    ):
        client_cls.return_value.__aenter__.return_value.post = AsyncMock(return_value=response)
        raw = await tools.web_search_tool.ainvoke({"query": "Ada Lovelace"})

    payload = json.loads(raw)
    assert payload["query"] == "Ada Lovelace"
    assert payload["total_results"] == 1
    assert payload["results"] == [
        {
            "title": "Ada Lovelace",
            "url": "https://en.wikipedia.org/wiki/Ada_Lovelace",
            "content": "English mathematician and early programmer.",
            "score": 0.875,
        }
    ]
    call = client_cls.return_value.__aenter__.return_value.post.call_args
    assert call.args == ("http://node-2:8000/retrieve",)
    assert call.kwargs["json"] == {"queries": ["Ada Lovelace"], "topk": 3, "return_scores": True}


@pytest.mark.asyncio
async def test_web_search_supports_plain_documents_and_contents() -> None:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"result": [[{"contents": "Alan Turing\nComputer scientist"}]]}

    with (
        patch.object(tools, "_get_tool_config", return_value={}),
        patch("deerflow.community.search_r1.tools.httpx.AsyncClient") as client_cls,
    ):
        client_cls.return_value.__aenter__.return_value.post = AsyncMock(return_value=response)
        raw = await tools.web_search_tool.ainvoke({"query": "Alan Turing", "max_results": 2})

    result = json.loads(raw)["results"][0]
    assert result["title"] == "Alan Turing"
    assert result["content"] == "Computer scientist"
    assert result["url"] == "https://en.wikipedia.org/wiki/Alan_Turing"


@pytest.mark.asyncio
async def test_web_search_reports_retryable_backend_failure() -> None:
    with (
        patch.object(tools, "_get_tool_config", return_value={}),
        patch("deerflow.community.search_r1.tools.httpx.AsyncClient") as client_cls,
    ):
        client_cls.return_value.__aenter__.return_value.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
        raw = await tools.web_search_tool.ainvoke({"query": "test"})

    payload = json.loads(raw)
    assert payload["error"] == "WEB_SEARCH_UNAVAILABLE"
    assert payload["retryable"] is True
    assert payload["query"] == "test"


@pytest.mark.asyncio
async def test_web_search_rejects_malformed_retriever_payload() -> None:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"result": []}

    with (
        patch.object(tools, "_get_tool_config", return_value={}),
        patch("deerflow.community.search_r1.tools.httpx.AsyncClient") as client_cls,
    ):
        client_cls.return_value.__aenter__.return_value.post = AsyncMock(return_value=response)
        raw = await tools.web_search_tool.ainvoke({"query": "test"})

    payload = json.loads(raw)
    assert payload["error"] == "WEB_SEARCH_INVALID_RESPONSE"
    assert payload["retryable"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["connection", "invalid_json", "invalid_shape", "invalid_root"])
async def test_retryable_failures_are_recorded_in_episode_context(failure) -> None:
    response = MagicMock()
    if failure == "invalid_json":
        response.json.side_effect = json.JSONDecodeError("invalid JSON", "oops", 0)
    elif failure == "invalid_root":
        response.json.return_value = []
    else:
        response.json.return_value = {"result": []}
    failures = []
    token = tools.SEARCH_R1_FAILURES.set(failures)
    try:
        with (
            patch.object(tools, "_get_tool_config", return_value={}),
            patch.object(tools.httpx, "AsyncClient") as client_cls,
        ):
            post = AsyncMock(return_value=response)
            if failure == "connection":
                post.side_effect = httpx.ConnectError("retriever offline")
            client_cls.return_value.__aenter__.return_value.post = post
            payload = json.loads(await tools.web_search_tool.ainvoke({"query": "test"}))
        assert len(failures) == 1
        assert failures[0]["kind"] == payload["error"]
        assert failures[0]["detail"]
    finally:
        tools.SEARCH_R1_FAILURES.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("hits", [[], [{"title": "Ada", "text": "mathematician"}]])
async def test_success_and_empty_results_do_not_mark_infrastructure_failure(hits) -> None:
    response = MagicMock()
    response.json.return_value = {"result": [hits]}
    failures = []
    token = tools.SEARCH_R1_FAILURES.set(failures)
    try:
        with (
            patch.object(tools, "_get_tool_config", return_value={}),
            patch.object(tools.httpx, "AsyncClient") as client_cls,
        ):
            client_cls.return_value.__aenter__.return_value.post = AsyncMock(return_value=response)
            await tools.web_search_tool.ainvoke({"query": "test"})
        assert failures == []
    finally:
        tools.SEARCH_R1_FAILURES.reset(token)


@pytest.mark.asyncio
async def test_failure_context_is_bounded_and_propagates_to_specialist_thread() -> None:
    failures = []
    token = tools.SEARCH_R1_FAILURES.set(failures)
    try:

        def specialist():
            for _ in range(20):
                tools._record_infrastructure_failure("WEB_SEARCH_UNAVAILABLE", RuntimeError("x" * 1000))

        await asyncio.to_thread(specialist)
        assert len(failures) == 16
        assert all(len(failure["detail"]) <= 300 for failure in failures)
    finally:
        tools.SEARCH_R1_FAILURES.reset(token)
