"""Successful reads retain their evidence while revealing duplicate windows."""
import asyncio

from slime_plugins.dcc.tools import WebTools


def test_repeat_keeps_source_text_and_navigation():
    async def run():
        tools = WebTools(max_page_chars=4)
        tools.pages["https://example.org/table"] = "abcdefgh"
        first = await tools.fetch_url("https://example.org/table")
        repeated = await tools.fetch_url("https://example.org/table#section")
        next_window = await tools.fetch_url("https://example.org/table", start=4)
        again = await tools.fetch_url("https://example.org/table")
        assert "read_progress" not in first
        assert repeated["text"] == first["text"] == "abcd"
        assert repeated["next_start"] == first["next_start"] == 4
        assert repeated["read_progress"] == {
            "new_text": False, "previous_identical_reads": 1,
            "first_read": {"url": "https://example.org/table", "start": 0, "end": 4},
        }
        assert "error" not in repeated
        assert "read_progress" not in next_window
        assert next_window["text"] == "efgh"
        assert again["read_progress"]["previous_identical_reads"] == 2
        assert tools.citation_validity([{"citations": ["https://example.org/table"]}]) == 1
    asyncio.run(run())


def test_same_body_from_different_urls_is_reported_without_aliasing_citations():
    async def run():
        tools = WebTools()
        urls = ["https://example.org/app", "https://example.org/app?format=json"]
        for url in urls:
            tools.pages[url] = "Public application shell"
        first = await tools.fetch_url(urls[0])
        second = await tools.fetch_url(urls[1])
        assert second["text"] == first["text"]
        assert second["url"] == urls[1]
        assert second["read_progress"]["first_read"]["url"] == urls[0]
        assert set(tools.sources) == set(urls)
        assert "error" not in second
    asyncio.run(run())


def test_progress_cache_is_episode_local_and_bounded():
    async def run():
        tools = WebTools()
        for index in range(129):
            url = f"https://example.org/{index}"
            tools.pages[url] = f"different source {index}"
            result = await tools.fetch_url(url)
            assert "read_progress" not in result
        assert len(tools._read_windows) == 128
        assert "different source 0" not in tools._read_windows
        assert not WebTools()._read_windows
        assert "read_progress" not in await tools.fetch_url("https://example.org/0")
    asyncio.run(run())
