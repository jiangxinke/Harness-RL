"""Expose a Search-R1 retrieval server through DeerFlow's ``web_search`` contract."""

import json
import logging
from contextvars import ContextVar
from typing import Any
from urllib.parse import quote

import httpx
from langchain.tools import tool

from deerflow.config import get_app_config

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_MAX_RESULTS = 5
DEFAULT_TIMEOUT = 30.0

# The episode runner binds a fresh list. Context propagation also reaches
# delegated specialists whose tool messages never enter Central's state.
SEARCH_R1_FAILURES: ContextVar[list[dict[str, str]] | None] = ContextVar("search_r1_episode_failures", default=None)


def _record_infrastructure_failure(kind: str, exc: BaseException) -> None:
    failures = SEARCH_R1_FAILURES.get()
    if failures is not None and len(failures) < 16:
        failures.append({"kind": kind, "detail": repr(exc)[:300]})


def _get_tool_config(tool_name: str = "web_search") -> dict[str, Any]:
    config = get_app_config().get_tool_config(tool_name)
    if config is None or config.model_extra is None:
        return {}
    return dict(config.model_extra)


def _document_text(document: dict[str, Any]) -> tuple[str, str]:
    title = str(document.get("title") or "").strip()
    text = str(document.get("text") or document.get("content") or "").strip()
    contents = str(document.get("contents") or "").strip()
    if contents:
        first, separator, remainder = contents.partition("\n")
        if not title:
            title = first.strip().strip('"')
        if not text:
            text = remainder.strip() if separator else contents
    return title or "Wikipedia passage", text


def _wikipedia_url(document: dict[str, Any], title: str) -> str:
    explicit = str(document.get("url") or document.get("href") or "").strip()
    if explicit.startswith(("http://", "https://")):
        return explicit
    return f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'), safe='()_-')}"


def _normalize_hit(hit: Any) -> dict[str, Any]:
    score = None
    document = hit
    if isinstance(hit, dict) and "document" in hit:
        document = hit["document"]
        score = hit.get("score")
    if not isinstance(document, dict):
        raise ValueError("retriever hit is not a document object")

    title, content = _document_text(document)
    normalized = {
        "title": title,
        "url": _wikipedia_url(document, title),
        "content": content,
    }
    if score is not None:
        normalized["score"] = score
    return normalized


@tool("web_search", parse_docstring=True)
async def web_search_tool(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> str:
    """Search a local Wikipedia passage index for factual evidence.

    Args:
        query: Search keywords describing the needed factual evidence.
        max_results: Maximum number of passages to return.
    """
    config = _get_tool_config()
    base_url = str(config.get("base_url", DEFAULT_BASE_URL)).rstrip("/")
    max_results = int(config.get("max_results", max_results))
    timeout = float(config.get("timeout", DEFAULT_TIMEOUT))

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{base_url}/retrieve",
                json={"queries": [query], "topk": max_results, "return_scores": True},
            )
            response.raise_for_status()
            payload = response.json()

        if not isinstance(payload, dict):
            raise ValueError("retriever response must be an object")
        result_batches = payload.get("result")
        if not isinstance(result_batches, list) or len(result_batches) != 1:
            raise ValueError("retriever response must contain exactly one result batch")
        hits = result_batches[0]
        if not isinstance(hits, list):
            raise ValueError("retriever result batch must be a list")
        results = [_normalize_hit(hit) for hit in hits[:max_results]]
    except (httpx.HTTPError, TimeoutError) as exc:
        _record_infrastructure_failure("WEB_SEARCH_UNAVAILABLE", exc)
        logger.error("Search-R1 request failed: %s", exc)
        return json.dumps(
            {"error": "WEB_SEARCH_UNAVAILABLE", "message": str(exc), "query": query, "retryable": True},
            ensure_ascii=False,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        _record_infrastructure_failure("WEB_SEARCH_INVALID_RESPONSE", exc)
        logger.error("Search-R1 returned an invalid response: %s", exc)
        return json.dumps(
            {"error": "WEB_SEARCH_INVALID_RESPONSE", "message": str(exc), "query": query, "retryable": True},
            ensure_ascii=False,
        )

    if not results:
        return json.dumps({"error": "No results found", "query": query}, ensure_ascii=False)
    return json.dumps(
        {"query": query, "total_results": len(results), "results": results},
        indent=2,
        ensure_ascii=False,
    )
