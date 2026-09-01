"""Bound and deduplicate web search inside one agent turn.

The CentralAgent can tell a delegated researcher to stop after repeated search
failures, but that instruction arrives only after the subagent has returned.
This middleware enforces the budget where searches actually execute, so a
broken provider or an over-eager model cannot spend hundreds of seconds issuing
near-identical calls before the parent regains control.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, override

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

WEB_SEARCH_TOOL_NAME = "web_search"
MAX_WEB_SEARCHES_PER_TURN = 8
MAX_PARALLEL_WEB_SEARCHES = 4
MAX_CONSECUTIVE_WEB_SEARCH_FAILURES = 2


def _real_user_turn_start(messages: list[Any]) -> int:
    """Return the latest visible user-message index, or the history start."""
    latest = -1
    for index, message in enumerate(messages):
        if not isinstance(message, HumanMessage):
            continue
        if (message.additional_kwargs or {}).get("hide_from_ui") is True:
            continue
        latest = index
    return latest + 1


def _normalized_query(tool_call: Mapping[str, Any]) -> str:
    args = tool_call.get("args")
    if not isinstance(args, Mapping):
        return ""
    query = args.get("query")
    return " ".join(str(query or "").casefold().split())


def _is_failed_search(message: ToolMessage) -> bool:
    if getattr(message, "status", None) == "error":
        return True
    content = message.content
    text = content if isinstance(content, str) else json.dumps(content, default=str, ensure_ascii=False)
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        payload = None
    if isinstance(payload, Mapping) and payload.get("error"):
        return True
    lowered = text.casefold()
    return any(
        marker in lowered
        for marker in (
            "no results found",
            "web_search_unavailable",
            "web_search_attempts_exhausted",
            "web_search_budget_exhausted",
        )
    )


def _completed_searches(messages: list[Any]) -> list[tuple[str, ToolMessage]]:
    """Pair completed search ToolMessages with their original normalized query."""
    query_by_call_id: dict[str, str] = {}
    completed: list[tuple[str, ToolMessage]] = []
    for message in messages[_real_user_turn_start(messages) :]:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                if str(call.get("name") or "") != WEB_SEARCH_TOOL_NAME:
                    continue
                call_id = str(call.get("id") or "")
                if call_id:
                    query_by_call_id[call_id] = _normalized_query(call)
            continue
        if not isinstance(message, ToolMessage) or str(message.name or "") != WEB_SEARCH_TOOL_NAME:
            continue
        call_id = str(message.tool_call_id or "")
        completed.append((query_by_call_id.get(call_id, ""), message))
    return completed


def _current_parallel_searches(messages: list[Any]) -> list[tuple[str, str]]:
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        return [(str(call.get("id") or ""), _normalized_query(call)) for call in message.tool_calls if str(call.get("name") or "") == WEB_SEARCH_TOOL_NAME]
    return []


def _guard_message(request: ToolCallRequest, *, code: str, message: str, status: str = "error") -> ToolMessage:
    return ToolMessage(
        content=json.dumps(
            {
                "error": code,
                "message": message,
                "retryable": False,
            },
            ensure_ascii=False,
        ),
        tool_call_id=str(request.tool_call.get("id") or "web-search-guard"),
        name=WEB_SEARCH_TOOL_NAME,
        status=status,
    )


class WebSearchBudgetMiddleware(AgentMiddleware):
    """Cache exact searches and stop failed or excessive search loops."""

    @staticmethod
    def _short_circuit(request: ToolCallRequest) -> ToolMessage | None:
        if str(request.tool_call.get("name") or "") != WEB_SEARCH_TOOL_NAME:
            return None

        messages = list(request.state.get("messages", []))
        query = _normalized_query(request.tool_call)
        current_id = str(request.tool_call.get("id") or "")
        parallel = _current_parallel_searches(messages)
        current_index = next(
            (index for index, (call_id, _) in enumerate(parallel) if call_id == current_id),
            None,
        )
        if current_index is not None:
            if query and any(previous_query == query for _, previous_query in parallel[:current_index]):
                return _guard_message(
                    request,
                    code="WEB_SEARCH_DUPLICATE_SKIPPED",
                    message="An identical search is already running in this tool batch. Use that sibling result.",
                    status="success",
                )
            if current_index >= MAX_PARALLEL_WEB_SEARCHES:
                return _guard_message(
                    request,
                    code="WEB_SEARCH_PARALLEL_LIMIT",
                    message=("This turn already contains enough parallel searches. Analyze the first results before issuing more queries."),
                )

        completed = _completed_searches(messages)
        if query:
            for previous_query, previous in reversed(completed):
                if previous_query != query or _is_failed_search(previous):
                    continue
                return ToolMessage(
                    content=previous.content,
                    tool_call_id=current_id or "web-search-cache",
                    name=WEB_SEARCH_TOOL_NAME,
                    status=getattr(previous, "status", "success"),
                    artifact=getattr(previous, "artifact", None),
                    additional_kwargs={
                        **dict(previous.additional_kwargs or {}),
                        "web_search_cache_hit": True,
                        "source_tool_call_id": previous.tool_call_id,
                    },
                )

        consecutive_failures = 0
        for _, previous in reversed(completed):
            if not _is_failed_search(previous):
                break
            consecutive_failures += 1
        if consecutive_failures >= MAX_CONSECUTIVE_WEB_SEARCH_FAILURES:
            return _guard_message(
                request,
                code="WEB_SEARCH_ATTEMPTS_EXHAUSTED",
                message=("web_search failed repeatedly in this turn. Stop retrying, use available/local evidence, and report that live search is unavailable."),
            )
        if len(completed) >= MAX_WEB_SEARCHES_PER_TURN:
            return _guard_message(
                request,
                code="WEB_SEARCH_BUDGET_EXHAUSTED",
                message=("The per-turn web-search budget is exhausted. Stop searching and synthesize the evidence already collected."),
            )
        return None

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        return self._short_circuit(request) or handler(request)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        short_circuit = self._short_circuit(request)
        return short_circuit if short_circuit is not None else await handler(request)
