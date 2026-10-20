"""Recover real web observations without fabricating source evidence."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

from slime_plugins.dcc import search_provider, tools

SOURCE = "https://example.org/source"


def test_html_fallback_preserves_table_cell_and_row_boundaries(monkeypatch):
    monkeypatch.setitem(sys.modules, "trafilatura", SimpleNamespace(extract=lambda *args, **kwargs: None))
    markup = b"<html><table><tr><th>Entity</th><th>Country</th></tr><tr><td>Alpha</td><td>Canada</td></tr><tr><td>Beta</td><td>Japan</td></tr></table></html>"
    text = tools.extract_page(markup, "text/html")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    assert lines == ["| Entity | Country", "| Alpha | Canada", "| Beta | Japan"]


@pytest.mark.parametrize("body,content_type", [
    (b'<?xml version="1.0"?><record><name>Alpha</name></record>', "application/save-to-disk"),
    (b'{"name":"Alpha"}', "application/octet-stream"),
    (b'{"name":"Alpha"}', "application/ld+json"),
    (b'<record><name>Alpha</name></record>', "application/atom+xml"),
])
def test_structured_public_data_downloads_remain_readable(body, content_type):
    assert "Alpha" in tools.extract_page(body, content_type)


@pytest.mark.parametrize("body", [b"\x00\x01\xff", b"<?xml broken", b"{incomplete"])
def test_generic_binary_or_invalid_attachment_is_not_a_successful_page(body):
    with pytest.raises(ValueError, match="unsupported page content type"):
        tools.extract_page(body, "application/save-to-disk")


@pytest.mark.parametrize("body", [b"", b"   ", b"Just a moment. Verify you are human before continuing."])
def test_empty_or_challenge_page_is_not_cached_as_successful_evidence(monkeypatch, body):
    bodies = iter([body, b"Alpha is a verified entity in Canada."])
    calls = []

    def download(url):
        calls.append(url)
        return next(bodies), "text/plain", "utf-8"

    monkeypatch.setattr(tools, "_download_public", download)
    web = tools.WebTools()

    async def scenario():
        first = await web.fetch_url(SOURCE)
        assert "error" in first and SOURCE not in web.pages
        second = await web.fetch_url(SOURCE)
        assert second["text"] == "Alpha is a verified entity in Canada."

    asyncio.run(scenario())
    assert calls == [SOURCE, SOURCE]


def test_fragment_reads_share_one_page_and_past_end_read_is_actionable(monkeypatch):
    calls = []

    def download(url):
        calls.append(url)
        return b"Alpha Beta", "text/plain", "utf-8"

    monkeypatch.setattr(tools, "_download_public", download)
    web = tools.WebTools(max_page_chars=6)

    async def scenario():
        first = await web.fetch_url(SOURCE + "#first")
        assert first["text"] == "Alpha " and first["next_start"] == 6
        second = await web.fetch_url(SOURCE + "#second", start=first["next_start"])
        assert second["text"] == "Beta" and second["next_start"] is None
        end = await web.fetch_url(SOURCE, start=second["end"])
        assert "error" in end and end["total_characters"] == 10
        assert "start=0" in end["error"]

    asyncio.run(scenario())
    assert calls == [SOURCE]


def test_transient_public_index_error_recovers_without_five_minute_blackout(monkeypatch):
    now = [100.0]
    search_provider._CACHE.clear()
    search_provider._PUBLIC_COOLDOWN_UNTIL.clear()
    search_provider._PUBLIC_NEXT_REQUEST.clear()
    monkeypatch.setattr(search_provider.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(search_provider.time, "sleep", lambda _: None)
    calls = []

    def request(url, **kwargs):
        calls.append(url)
        if len(calls) == 1:
            raise tools.InfrastructureError("HTTP retrieval failed (TimeoutError)")
        return f'<h3><a href="{SOURCE}">Alpha source</a></h3>'

    monkeypatch.setattr(tools, "_request", request)
    with pytest.raises(tools.InfrastructureError, match="TimeoutError"):
        search_provider.public_html_search("first", 3, "seznam")
    now[0] = 105.0
    with pytest.raises(tools.InfrastructureError, match="cooldown"):
        search_provider.public_html_search("second", 3, "seznam")
    now[0] = 111.0
    hits = search_provider.public_html_search("third", 3, "seznam")
    assert hits[0]["url"] == SOURCE and len(calls) == 2
    search_provider._CACHE.clear()
    search_provider._PUBLIC_COOLDOWN_UNTIL.clear()
    search_provider._PUBLIC_NEXT_REQUEST.clear()
