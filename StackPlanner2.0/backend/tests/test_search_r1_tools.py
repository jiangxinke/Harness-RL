"""Tests for the local Search-R1 retriever tool."""

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
