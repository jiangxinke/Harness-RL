"""Keep uploaded-document questions local until the file has been inspected.

The upload middleware exposes file metadata and a converted Markdown path in the
prompt.  This guard is the deterministic counterpart to that instruction: it
prevents a model from spending a long time searching the web for a document it
has not read yet.  After one local inspection, explicit external lookups remain
available.
"""

from collections.abc import Awaitable, Callable
from typing import Any, override

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

_REMOTE_SEARCH_TOOLS = frozenset({"web_search", "web_fetch", "image_search"})
_LOCAL_INSPECTION_TOOLS = frozenset({"read_file", "grep", "glob", "ls"})
_MAX_REMOTE_LOOKUPS_PER_UPLOAD_TURN = 3

_BLOCKED_SEARCH_MESSAGE = (
    "Local uploaded-file inspection is required before external search. Use `read_file` on the uploaded Markdown path (or `grep` the uploads directory) first. Do not search the web for details that may already be present in the upload."
)

_SEARCH_CAP_MESSAGE = "The uploaded document has already been inspected and the external-search cap for this turn is reached. Stop searching and answer from the local file plus the evidence already collected."


def _has_uploaded_file_in_latest_turn(messages: list[Any]) -> bool:
    """Return whether the latest human turn contains uploaded-file context."""
    latest_human_index = -1
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            latest_human_index = index
    if latest_human_index < 0:
        return False

    latest_human = messages[latest_human_index]
    content = getattr(latest_human, "content", "")
    kwargs = getattr(latest_human, "additional_kwargs", {}) or {}
    return (isinstance(kwargs.get("files"), list) and bool(kwargs["files"])) or (isinstance(content, str) and "<uploaded_files>" in content)


def _has_local_inspection_after_latest_turn(messages: list[Any]) -> bool:
    latest_human_index = -1
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            latest_human_index = index
    if latest_human_index < 0:
        return False

    return any(getattr(message, "type", None) == "tool" and getattr(message, "name", None) in _LOCAL_INSPECTION_TOOLS for message in messages[latest_human_index + 1 :])


def _remote_lookup_count_after_latest_turn(messages: list[Any]) -> int:
    latest_human_index = -1
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            latest_human_index = index
    if latest_human_index < 0:
        return 0

    return sum(getattr(message, "type", None) == "tool" and getattr(message, "name", None) in _REMOTE_SEARCH_TOOLS for message in messages[latest_human_index + 1 :])


class UploadedFileSearchGuardMiddleware(AgentMiddleware):
    """Require one local file inspection before remote search in upload turns."""

    @staticmethod
    def _should_block(request: ToolCallRequest) -> bool:
        tool_name = request.tool_call.get("name")
        if tool_name not in _REMOTE_SEARCH_TOOLS:
            return False
        messages = list(request.state.get("messages", []))
        if not _has_uploaded_file_in_latest_turn(messages):
            return False
        return not _has_local_inspection_after_latest_turn(messages) or _remote_lookup_count_after_latest_turn(messages) >= _MAX_REMOTE_LOOKUPS_PER_UPLOAD_TURN

    @staticmethod
    def _blocked_message(request: ToolCallRequest) -> str:
        messages = list(request.state.get("messages", []))
        if _remote_lookup_count_after_latest_turn(messages) >= _MAX_REMOTE_LOOKUPS_PER_UPLOAD_TURN:
            return _SEARCH_CAP_MESSAGE
        return _BLOCKED_SEARCH_MESSAGE

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        if self._should_block(request):
            return ToolMessage(
                content=self._blocked_message(request),
                tool_call_id=str(request.tool_call.get("id") or "uploaded-file-search-guard"),
                name=str(request.tool_call.get("name") or "web_search"),
            )
        return handler(request)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        if self._should_block(request):
            return ToolMessage(
                content=self._blocked_message(request),
                tool_call_id=str(request.tool_call.get("id") or "uploaded-file-search-guard"),
                name=str(request.tool_call.get("name") or "web_search"),
            )
        return await handler(request)
