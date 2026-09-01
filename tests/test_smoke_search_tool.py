from __future__ import annotations

import asyncio

from scripts import smoke_search_tool


def test_invoke_search_uses_async_tool_interface(monkeypatch) -> None:
    calls: list[dict[str, str]] = []

    class FakeSearchTool:
        async def ainvoke(self, payload: dict[str, str]) -> str:
            calls.append(payload)
            return '{"results": [{"url": "https://example.test"}]}'

    monkeypatch.setattr(smoke_search_tool, "web_search_tool", FakeSearchTool())

    result = asyncio.run(smoke_search_tool.invoke_search("locked query"))

    assert result == '{"results": [{"url": "https://example.test"}]}'
    assert calls == [{"query": "locked query"}]
