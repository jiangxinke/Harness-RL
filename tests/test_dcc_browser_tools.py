"""Optional DOM reads preserve evidence boundaries, fallback and continuation."""
import asyncio

import pytest

from slime_plugins.dcc import tools
from slime_plugins.dcc.episode import _WebEvidence

URL = "https://example.org/app"
API = "https://example.org/api/collection"


class Reader:
    def __init__(self, outcome=None):
        self.calls = []
        self.outcome = outcome

    async def read(self, url):
        self.calls.append(url)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome if self.outcome is not None else {
            "text": "Visible application data",
            "outgoing_links": [{"title": "Table", "url": API}],
            "observed_json_urls": [{"url": API, "status": 200, "bytes": 42}],
            "browser_runtime": {"version": "test-visible-dom-v1", "network": {"requests": 2}},
            "truncated": False, "total_characters": 24,
            "final_url": url,
        }


def downloaded(monkeypatch, text="HTTP application shell", content_type="text/html"):
    monkeypatch.setattr(tools, "_download_public", lambda url: (text.encode(), content_type, "utf-8"))


def test_dom_text_has_stable_offsets_and_does_not_approve_observed_api(monkeypatch):
    downloaded(monkeypatch)

    async def run():
        reader = Reader()
        web = tools.WebTools(browser_reader=reader, max_page_chars=7)
        first = await web.fetch_url(URL)
        second = await web.fetch_url(URL, start=first["next_start"])
        assert first["text"] == "Visible" and second["text"] == " applic"
        assert reader.calls == [URL]
        assert first["browser_read"]["method"] == "visible_dom"
        assert first["browser_read"]["observed_json_urls"][0]["url"] == API
        evidence = _WebEvidence()
        evidence.observe("fetch_url", first)
        assert URL in evidence.fetched_sources and API not in evidence.fetched_sources
        assert API not in web.pages and API not in web.sources
        assert first["outgoing_links"][0]["url"] == API
    asyncio.run(run())


@pytest.mark.parametrize("outcome", [RuntimeError("sensitive browser details"), {"error": "timeout"}, {"text": ""}, {"text": "incomplete result"}])
def test_failed_or_malformed_browser_result_keeps_http_source(monkeypatch, outcome):
    downloaded(monkeypatch)
    result = asyncio.run(tools.WebTools(browser_reader=Reader(outcome)).fetch_url(URL))
    assert result["text"] == "HTTP application shell"
    assert result["browser_read"]["method"] == "http_fallback"
    assert "sensitive browser details" not in str(result)


@pytest.mark.parametrize("content_type", ["text/csv", "application/json"])
def test_non_html_reads_do_not_launch_browser(monkeypatch, content_type):
    downloaded(monkeypatch, '{"data":[1,2]}', content_type)
    reader = Reader()
    result = asyncio.run(tools.WebTools(browser_reader=reader).fetch_url(URL))
    assert "text" in result and "browser_read" not in result and not reader.calls


def test_http_challenge_and_http_error_do_not_launch_browser(monkeypatch):
    downloaded(monkeypatch, "Access denied")
    reader = Reader()
    first = asyncio.run(tools.WebTools(browser_reader=reader).fetch_url(URL))
    assert "error" in first and not reader.calls

    def blocked(url):
        raise tools._HTTPRetrievalError(403)

    monkeypatch.setattr(tools, "_download_public", blocked)
    second = asyncio.run(tools.WebTools(browser_reader=reader).fetch_url(URL))
    assert second["http_status"] == 403 and not reader.calls


def test_default_reader_keeps_http_text_and_no_browser_metadata(monkeypatch):
    downloaded(monkeypatch)
    result = asyncio.run(tools.WebTools().fetch_url(URL))
    assert result["text"] == "HTTP application shell"
    assert "browser_read" not in result
