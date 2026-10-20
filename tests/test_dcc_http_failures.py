"""HTTP failures guide new source discovery without becoming read evidence."""

from __future__ import annotations

import asyncio
import http.client
import io
import json
import traceback
import urllib.error
from email.message import Message
from types import SimpleNamespace

import pytest

from slime_plugins.dcc import search_provider, tools
from slime_plugins.dcc.episode import _WebEvidence

SOURCE = "https://example.org/missing.csv"
PRIVATE = "private-response-value-must-not-escape"
_ORIGINAL_PUBLIC_URL = tools._public_url


@pytest.fixture(autouse=True)
def offline_fetch(monkeypatch):
    monkeypatch.setattr(search_provider, "wikipedia_page", lambda _: None)
    monkeypatch.setattr(tools, "_public_url", lambda url: url)


def http_error(status):
    return urllib.error.HTTPError(SOURCE, status, PRIVATE, {"Set-Cookie": PRIVATE, "X-Private": PRIVATE}, io.BytesIO(PRIVATE.encode()))


def opener(monkeypatch, outcomes):
    outcomes, calls = iter(outcomes), []

    class Response:
        def __init__(self, body):
            self.body = body
            self.headers = Message()
            self.headers["Content-Type"] = "text/plain; charset=utf-8"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit):
            return self.body[:limit]

    def open_request(request, **kwargs):
        calls.append(request.full_url)
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return Response(outcome)

    monkeypatch.setattr(tools.urllib.request, "build_opener", lambda *args: SimpleNamespace(open=open_request))
    return calls


@pytest.mark.parametrize("status,retryable", [(404, False), (410, False), (403, False), (408, True), (425, True), (429, True), (500, True), (502, True), (503, True), (504, True)])
def test_http_status_is_preserved_without_private_response_details(monkeypatch, status, retryable):
    calls = opener(monkeypatch, [http_error(status)])
    with pytest.raises(tools.InfrastructureError) as captured:
        tools._download_public(SOURCE)
    error = captured.value
    assert error.http_status == status and error.retryable is retryable
    assert f"HTTP {status}" in str(error) and PRIVATE not in str(error)
    assert PRIVATE not in "".join(traceback.format_exception(error))
    assert calls == [SOURCE]


@pytest.mark.parametrize("status,retryable", [(404, False), (429, True), (503, True)])
def test_api_reads_share_safe_http_status_diagnostics(monkeypatch, status, retryable):
    def fail(*args, **kwargs):
        raise http_error(status)

    monkeypatch.setattr(tools.urllib.request, "urlopen", fail)
    with pytest.raises(tools.InfrastructureError) as captured:
        tools._request(SOURCE, headers={"Authorization": PRIVATE})
    assert captured.value.http_status == status and captured.value.retryable is retryable
    assert PRIVATE not in str(captured.value)
    assert PRIVATE not in "".join(traceback.format_exception(captured.value))


@pytest.mark.parametrize("status", [404, 410])
def test_missing_resource_is_cached_across_windows_but_is_never_evidence(monkeypatch, status):
    calls = opener(monkeypatch, [http_error(status)])
    web, evidence = tools.WebTools(), _WebEvidence()

    async def scenario():
        first = await web.fetch_url(SOURCE + "#first")
        assert first["http_status"] == status and first["retryable"] is False and first["cached"] is False
        assert "web_search" in first["error"] and "outgoing_links" in first["error"] and "guessing" in first["error"]
        evidence.observe("fetch_url", first)
        first["error"] = "caller mutation"
        second = await web.fetch_url(SOURCE + "#other", start=100)
        assert second["cached"] is True and second["error"] != "caller mutation"
        assert PRIVATE not in json.dumps(second) and "text" not in second
        evidence.observe("fetch_url", second)

    asyncio.run(scenario())
    assert calls == [SOURCE]
    assert evidence.fetches == 0 and not evidence.fetched_sources
    assert not web.pages and not web.sources


@pytest.mark.parametrize("status", [403, 408, 429, 500, 502, 503, 504])
def test_retryable_and_other_nonpermanent_statuses_can_recover(monkeypatch, status):
    calls = opener(monkeypatch, [http_error(status), b"A real recovered source."])
    web = tools.WebTools()

    async def scenario():
        failed = await web.fetch_url(SOURCE)
        assert failed["http_status"] == status and failed["cached"] is False
        assert failed["retryable"] is (status != 403)
        assert await web.fetch_url(SOURCE) == {
            "url": SOURCE, "text": "A real recovered source.", "start": 0, "end": 24,
            "total_characters": 24, "truncated": False, "next_start": None,
        }

    asyncio.run(scenario())
    assert calls == [SOURCE, SOURCE] and not web._permanent_http_failures


@pytest.mark.parametrize("error", [TimeoutError(PRIVATE), urllib.error.URLError(PRIVATE), OSError(PRIVATE),
                                  http.client.RemoteDisconnected(PRIVATE), http.client.BadStatusLine(PRIVATE),
                                  http.client.IncompleteRead(b"private-response-value-must-not-escape", 100)])
def test_transport_failure_keeps_retryability_without_leaking_exception_text(monkeypatch, error):
    calls = opener(monkeypatch, [error, b"Recovered."])
    web = tools.WebTools()

    async def scenario():
        failed = await web.fetch_url(SOURCE)
        assert failed["http_status"] is None and failed["retryable"] is True
        assert PRIVATE not in json.dumps(failed)
        assert (await web.fetch_url(SOURCE))["text"] == "Recovered."

    asyncio.run(scenario())
    assert calls == [SOURCE, SOURCE]


@pytest.mark.parametrize("url", ["https://example.org/path space", "https://example.org/path\nnext",
                                "https://example.org/path\tnext", "https://example.org/path\x7fnext"])
def test_malformed_url_is_rejected_before_dns_and_cannot_become_evidence(monkeypatch, url):
    def unexpected_dns(*args, **kwargs):
        raise AssertionError("invalid URL must not resolve or fetch")

    monkeypatch.setattr(tools, "_public_url", _ORIGINAL_PUBLIC_URL)
    monkeypatch.setattr(tools.socket, "getaddrinfo", unexpected_dns)
    web, evidence = tools.WebTools(), _WebEvidence()
    result = asyncio.run(web.fetch_url(url))
    assert "error" in result and "text" not in result and "percent-encoded" in result["error"]
    evidence.observe("fetch_url", result)
    assert evidence.fetches == 0 and not web.pages and not web.sources


def test_stdlib_invalid_path_exception_is_a_recoverable_tool_error(monkeypatch):
    def invalid_path(request, **kwargs):
        # putrequest validates before opening a socket: exercise the real
        # stdlib failure from the crashed writer without a network request.
        connection = http.client.HTTPConnection("example.org")
        connection.putrequest("GET", "/en/country/saudi arabia")

    monkeypatch.setattr(tools.urllib.request, "build_opener", lambda *args: SimpleNamespace(open=invalid_path))
    result = asyncio.run(tools.WebTools().fetch_url(SOURCE))
    assert result == {"url": SOURCE, "error": "Malformed HTTP URL. Use an observed link with spaces percent-encoded."}


def test_unicode_encoding_error_is_a_safe_tool_observation(monkeypatch):
    opener(monkeypatch, [UnicodeEncodeError("ascii", PRIVATE, 0, 1, PRIVATE)])
    result = asyncio.run(tools.WebTools().fetch_url(SOURCE))
    assert "percent-encoded" in result["error"] and PRIVATE not in json.dumps(result)
    assert "text" not in result


@pytest.mark.parametrize("error", [http.client.InvalidURL(PRIVATE), http.client.RemoteDisconnected(PRIVATE),
                                  UnicodeEncodeError("ascii", PRIVATE, 0, 1, PRIVATE)])
def test_search_http_protocol_errors_remain_safe_infrastructure_failures(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(tools.urllib.request, "urlopen", fail)
    with pytest.raises(tools.InfrastructureError) as captured:
        tools._request(SOURCE)
    assert captured.value.http_status is None and captured.value.retryable is True
    assert PRIVATE not in "".join(traceback.format_exception(captured.value))


def test_failure_cache_is_bounded_episode_local_and_keeps_query_strings(monkeypatch):
    monkeypatch.setattr(tools, "_HTTP_FAILURE_CACHE_SIZE", 2)
    calls = opener(monkeypatch, [http_error(404)] * 6)
    web = tools.WebTools()

    async def scenario():
        await web.fetch_url(SOURCE + "?version=1")
        await web.fetch_url(SOURCE + "?version=2")
        assert (await web.fetch_url(SOURCE + "?version=1"))["cached"] is True
        await web.fetch_url(SOURCE + "?version=3")
        assert len(web._permanent_http_failures) == 2
        assert (await web.fetch_url(SOURCE + "?version=2"))["cached"] is False
        assert (await tools.WebTools().fetch_url(SOURCE + "?version=1"))["cached"] is False

    asyncio.run(scenario())
    assert calls == [SOURCE + "?version=1", SOURCE + "?version=2", SOURCE + "?version=3", SOURCE + "?version=2", SOURCE + "?version=1"]


def test_a_different_observed_link_can_be_read_after_a_permanent_failure(monkeypatch):
    calls = opener(monkeypatch, [http_error(404), b"number,name\n1,Verified record\n"])
    web = tools.WebTools()
    corrected = "https://example.org/actual-export.csv"

    async def scenario():
        assert (await web.fetch_url(SOURCE))["http_status"] == 404
        result = await web.fetch_url(corrected)
        assert result["text"] == "number,name\n1,Verified record\n" and "error" not in result

    asyncio.run(scenario())
    assert calls == [SOURCE, corrected] and corrected in web.pages
