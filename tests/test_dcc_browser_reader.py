"""Optional browser semantics, public network controls and bounded metadata."""
from __future__ import annotations

import asyncio
import builtins
from dataclasses import replace
import ipaddress
import json
import os
from types import SimpleNamespace
import urllib.parse

import pytest

from slime_plugins.dcc.browser_reader import (
    BROWSER_READER_VERSION, PLAYWRIGHT_VERSION, BrowserLimits, BrowserReader,
    browser_reader_configuration, validate_browser_reader_runtime,
)

URL = "https://fixture.example/page"
PRIVATE = "private-body-and-cookie-value"


def public_url(url):
    hostname = urllib.parse.urlsplit(url).hostname
    try:
        if not ipaddress.ip_address(hostname).is_global:
            raise ValueError("private address")
    except ValueError as exc:
        if str(exc) == "private address":
            raise
    return url


def runtime():
    return {"version": BROWSER_READER_VERSION, "browser_version": "141.0.7390.37",
            "playwright_version": PLAYWRIGHT_VERSION, "cpu_only": True,
            "configuration": browser_reader_configuration(enabled=True)}


class Response:
    def __init__(self, body=b"<html>visible</html>", status=200, headers=None):
        self.status, self.data = status, body
        self.headers = {"content-type": "text/html", **(headers or {})}
        self.disposed = False

    async def body(self):
        return self.data

    async def dispose(self):
        self.disposed = True


class Route:
    def __init__(self, url=URL, method="GET", kind="document", response=None):
        self.request = SimpleNamespace(url=url, method=method, resource_type=kind)
        self.response = Response() if response is None else response
        self.fetched, self.fulfilled, self.aborted = [], None, False

    async def fetch(self, **kwargs):
        self.fetched.append(kwargs)
        return self.response

    async def fulfill(self, **kwargs):
        self.fulfilled = kwargs

    async def abort(self):
        self.aborted = True


class Page:
    def __init__(self, context):
        self.context, self.url = context, URL
        self.events = {}

    def on(self, name, callback):
        self.events[name] = callback

    async def goto(self, url, **kwargs):
        self.url = url
        routes = self.context.routes
        if self.context.concurrent:
            await asyncio.gather(*(self.context.handler(route) for route in routes))
        else:
            for route in routes:
                await self.context.handler(route)
        if routes[0].aborted:
            raise RuntimeError("Navigation blocked")
        return SimpleNamespace(status=routes[0].fulfilled["status"])

    async def wait_for_timeout(self, ms):
        return None

    async def evaluate(self, source, limits):
        return {"title": "Fixture", "text": "Visible DOM text", "total_characters": 16,
                "truncated": False, "outgoing_links": list(self.context.links),
                "link_candidates_truncated": False}


class Context:
    def __init__(self, routes, links=(), concurrent=False):
        self.routes, self.links, self.concurrent = routes, links, concurrent
        self.closed, self.events = False, {}

    async def route(self, pattern, handler):
        assert pattern == "**/*"
        self.handler = handler

    async def route_web_socket(self, pattern, handler):
        self.websocket = (pattern, handler)

    async def new_page(self):
        self.page = Page(self)
        return self.page

    def on(self, event, handler):
        self.events[event] = handler

    async def close(self):
        self.closed = True


class Browser:
    def __init__(self, contexts):
        self.contexts, self.settings, self.closed = contexts, [], False

    async def new_context(self, **kwargs):
        self.settings.append(kwargs)
        return self.contexts[len(self.settings) - 1]

    async def close(self):
        self.closed = True


def scenario(routes, *, limits=None, links=(), concurrent=False):
    context = Context(routes, links=links, concurrent=concurrent)
    reader = BrowserReader(enabled=True, limits=limits or replace(BrowserLimits(), settle_ms=0, idle_ms=0), url_validator=public_url)
    reader._browser = Browser([context])
    reader._runtime = {key: value for key, value in runtime().items() if key != "configuration"}
    result = asyncio.run(reader.read(URL))
    return reader, context, result


def test_disabled_reader_and_static_metadata_do_not_import_playwright(monkeypatch):
    real_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        assert not name.startswith("playwright")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    reader = BrowserReader()
    assert reader.configuration() == browser_reader_configuration()
    assert reader.runtime is None
    async def run():
        assert await reader.preflight() is None
        assert "disabled" in (await reader.read(URL))["error"]
        await reader.close()
    asyncio.run(run())
    assert validate_browser_reader_runtime(runtime()) == runtime()
    configuration = reader.configuration()
    configuration["limits"]["concurrency"] = 99
    assert reader.configuration()["limits"]["concurrency"] == 2


@pytest.mark.parametrize("changes", [{"concurrency": 0}, {"max_requests": True}, {"settle_ms": -1},
    {"total_timeout_seconds": float("nan")}, {"max_link_candidates": 1}])
def test_invalid_budgets_rejected(changes):
    with pytest.raises(ValueError):
        BrowserLimits(**changes)


@pytest.mark.parametrize("changes", [{"extra": 1}, {"cpu_only": 1}, {"browser_version": ""},
    {"playwright_version": "1.55.0"}, {"version": "different"}])
def test_enabled_metadata_rejects_incomplete_or_inaccurate_runtime(changes):
    value = runtime(); value.update(changes)
    with pytest.raises(ValueError):
        validate_browser_reader_runtime(value)


def test_enabled_metadata_requires_exact_default_configuration():
    for key, value in [("enabled", False), ("enabled", 1), ("limits", {"concurrency": 2})]:
        metadata = runtime(); metadata["configuration"][key] = value
        with pytest.raises(ValueError):
            validate_browser_reader_runtime(metadata)
    metadata = runtime(); metadata["configuration"]["limits"]["concurrency"] = 2.0
    with pytest.raises(ValueError):
        validate_browser_reader_runtime(metadata)


def test_start_fails_before_importing_wrong_playwright(monkeypatch):
    monkeypatch.setattr("slime_plugins.dcc.browser_reader.importlib.metadata.version", lambda name: "1.55.0")
    async def run():
        with pytest.raises(RuntimeError, match="requires Playwright 1.56.0"):
            await BrowserReader(enabled=True).preflight()
    asyncio.run(run())


@pytest.mark.parametrize("url", ["file:///etc/passwd", "https://user:secret@fixture.example/", "https://@fixture.example/", "http://127.0.0.1/", "http://169.254.169.254/", "http://[::1]/"])
def test_unsafe_navigation_never_starts_browser(url):
    reader = BrowserReader(enabled=True, url_validator=public_url)
    async def should_not_start():
        pytest.fail("unsafe navigation started browser")
    reader._start = should_not_start
    assert "error" in asyncio.run(reader.read(url))


@pytest.mark.parametrize("method,kind,url", [("POST", "fetch", URL), ("PUT", "xhr", URL),
    ("GET", "image", URL), ("GET", "font", URL), ("GET", "media", URL),
    ("GET", "fetch", "http://127.0.0.1/secrets"), ("GET", "fetch", "https://user@fixture.example/secrets")])
def test_subrequests_are_validated_and_unsafe_requests_blocked(method, kind, url):
    route = Route(url, method, kind)
    _, context, result = scenario([Route(), route])
    assert result["text"] == "Visible DOM text"
    assert route.aborted and not route.fetched and route.fulfilled is None
    assert context.closed
    assert context.websocket[0] == "**/*"
    assert context.page.events["download"]


def test_head_allowed_fresh_contexts_shared_browser_and_decoded_bytes():
    main, head = Route(), Route(URL + "/head", "HEAD", "fetch", Response(b"", headers={"content-encoding": "gzip", "content-length": "0", "connection": "keep-alive"}))
    reader, context, result = scenario([main, head])
    assert "text" in result and head.fulfilled is not None
    assert not {"content-encoding", "content-length", "connection"} & set(head.fulfilled["headers"])
    assert head.fetched == [{"max_redirects": 0, "max_retries": 0, "timeout": 15000}]
    assert head.response.disposed
    second = Context([Route()]); reader._browser.contexts.append(second)
    async def run():
        assert "text" in await reader.read(URL)
        assert len(reader._browser.settings) == 2
        assert all(settings == {"service_workers": "block", "accept_downloads": False} for settings in reader._browser.settings)
        browser = reader._browser
        await reader.close()
        assert browser.closed and second.closed
        assert "error" in await reader.read(URL)
    asyncio.run(run())


@pytest.mark.parametrize("response", [Response(status=302, headers={"location": "http://127.0.0.1/secrets"}), Response(headers={"content-disposition": "attachment; filename=secret.txt"})])
def test_redirects_and_attachments_are_not_forwarded(response):
    route = Route(response=response)
    _, _, result = scenario([route])
    assert "text" not in result and "error" in result
    assert route.aborted and not route.fulfilled and response.disposed
    assert route.fetched[0]["max_redirects"] == 0


@pytest.mark.parametrize("status", [403, 404, 500])
def test_failed_http_status_is_not_visible_evidence(status):
    _, _, result = scenario([Route(response=Response(status=status))])
    assert result["http_status"] == status and "text" not in result


@pytest.mark.parametrize("budget,expected", [("max_requests", "request_limit_reached"), ("max_responses", "response_limit_reached")])
def test_concurrent_request_and_response_budgets(budget, expected):
    routes = [Route(), Route(URL + "/one", "GET", "fetch"), Route(URL + "/two", "GET", "fetch")]
    _, _, result = scenario(routes, limits=replace(BrowserLimits(), settle_ms=0, idle_ms=0, **{budget: 2}), concurrent=True)
    network = result["browser_runtime"]["network"]
    assert network[expected]
    assert sum(bool(route.fetched) for route in routes) == 2
    assert routes[2].aborted


@pytest.mark.parametrize("limits", [{"max_response_bytes": 25}, {"max_total_response_bytes": 30}])
def test_response_byte_budgets(limits):
    main, oversized = Route(), Route(URL + "/large", "GET", "fetch", Response(b"x" * 40))
    _, _, result = scenario([main, oversized], limits=replace(BrowserLimits(), settle_ms=0, idle_ms=0, **limits))
    assert oversized.aborted and oversized.fulfilled is None
    assert result["browser_runtime"]["network"]["response_bytes_limit_reached"]


def test_json_collection_metadata_is_only_actual_successful_get_navigation():
    def api(path, body, status=200, method="GET"):
        return Route(URL + path, method, "fetch", Response(json.dumps(body).encode(), status, {"content-type": "application/json", "set-cookie": PRIVATE}))
    good = api("/collection", [{"secret": PRIVATE}, {"secret": PRIVATE}])
    routes = [Route(), good, api("/object", {"a": PRIVATE}), api("/error", [{}, {}], 500), api("/post", [{}, {}], method="POST"), api("/second", [{}, {}])]
    _, _, result = scenario(routes, limits=replace(BrowserLimits(), settle_ms=0, idle_ms=0, max_json_urls=1))
    assert result["observed_json_urls"] == [{"url": good.request.url, "status": 200, "bytes": len(good.response.data), "navigation_only": True, "note": result["observed_json_urls_note"]}]
    assert "explicitly fetch" in result["observed_json_urls_note"]
    assert PRIVATE not in json.dumps(result)
    assert result["observed_json_urls_truncated"]
    assert result["browser_runtime"]["network"]["json_navigation_truncated"]


def test_json_inspection_and_link_truncation_flags_are_separate():
    api = Route(URL + "/collection", "GET", "xhr", Response(b"[{},{},{}]", headers={"content-type": "application/json"}))
    links = [{"title": "one", "url": URL + "/1"}, {"title": "two", "url": URL + "/2"}, {"title": "private", "url": "http://127.0.0.1/"}]
    _, _, result = scenario([Route(), api], links=links, limits=replace(BrowserLimits(), settle_ms=0, idle_ms=0, max_json_bytes=4, max_links=1))
    assert result["browser_runtime"]["network"]["json_body_inspection_truncated"]
    assert result["observed_json_urls"] == [] and not result["observed_json_urls_truncated"]
    assert len(result["outgoing_links"]) == 1 and result["outgoing_links_truncated"]
    assert "fetch them" in result["outgoing_links_note"]


def test_timeout_and_cancellation_preserve_no_evidence_and_close_context():
    reader = BrowserReader(enabled=True, limits=replace(BrowserLimits(), total_timeout_seconds=0.01))
    async def hangs(url):
        await asyncio.Future()
    reader._read = hangs
    assert asyncio.run(reader.read(URL))["error_type"] == "TimeoutError"
    async def cancelled():
        task = asyncio.create_task(reader.read(URL)); await asyncio.sleep(0); task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(cancelled())


def test_cancellation_closes_fresh_context(monkeypatch):
    context = Context([Route()])
    reader = BrowserReader(enabled=True, url_validator=public_url)
    reader._browser = Browser([context])
    entered = asyncio.Event()
    async def hang(*args, **kwargs):
        entered.set()
        await asyncio.Future()
    monkeypatch.setattr(Page, "goto", hang)
    async def run():
        task = asyncio.create_task(reader.read(URL))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert context.closed
    asyncio.run(run())


def test_websocket_and_download_callbacks_close_without_network():
    _, context, _ = scenario([Route()])
    closed = []
    async def close():
        closed.append(True)
    async def run():
        context.websocket[1](SimpleNamespace(close=close))
        context.page.events["download"](SimpleNamespace(cancel=close))
        await asyncio.sleep(0)
        assert closed == [True, True]
    asyncio.run(run())


def test_close_has_a_total_cleanup_timeout():
    reader = BrowserReader(enabled=True, limits=replace(BrowserLimits(), cleanup_timeout_seconds=0.01))
    stopped = []
    async def hang():
        await asyncio.Future()
    async def stop():
        stopped.append(True)
    reader._browser = SimpleNamespace(close=hang)
    reader._playwright = SimpleNamespace(stop=stop)
    async def run():
        await asyncio.wait_for(reader.close(), timeout=0.2)
        assert reader._closed and reader._browser is None and reader._playwright is None
        assert stopped == [True]
    asyncio.run(run())


@pytest.mark.skipif(os.environ.get("DCC_BROWSER_READER_TEST_CHROMIUM") != "1", reason="explicit optional CPU Chromium semantic test")
def test_real_chromium_visible_dom_links_and_dynamic_json():
    pytest.importorskip("playwright.async_api")
    html = b'''<!doctype html><html><head><title>Real fixture</title><style>.hidden {display:none} /* style-marker */</style></head>
    <body><p>Visible contract text.</p><p hidden>hidden-marker</p><p class="hidden">css-hidden-marker</p>
    <a href="/visible">Visible source</a><a href="/visible">Duplicate source</a><a hidden href="/hidden">Hidden link</a>
    <a href="http://127.0.0.1/secrets">Private</a><a href="https://user:pass@fixture.example/credentials">Credentials</a>
    <script>const ignored='script-marker'; fetch('/collection'); fetch('/post',{method:'POST',body:'blocked'});
    setTimeout(()=>{document.body.appendChild(document.createTextNode(' Dynamic visible result.'));},10);</script></body></html>'''
    async def run():
        reader = BrowserReader(enabled=True, url_validator=public_url, limits=replace(BrowserLimits(), settle_ms=150, idle_ms=0))
        fetched = []
        async def fake_fetch(route):
            fetched.append((route.request.url, route.request.method))
            if route.request.url.endswith("/collection"):
                return Response(b'[{"private":"never-in-text"},{"private":"never-in-text"}]', headers={"content-type": "application/json"})
            return Response(html)
        reader._fetch_route = fake_fetch
        try:
            metadata = await reader.preflight()
            assert metadata["playwright_version"] == PLAYWRIGHT_VERSION and metadata["cpu_only"]
            result = await reader.read(URL)
            assert "error" not in result, result
            assert "Visible contract text." in result["text"] and "Dynamic visible result." in result["text"]
            for absent in ("hidden-marker", "css-hidden-marker", "style-marker", "script-marker", "never-in-text"):
                assert absent not in result["text"]
            assert result["outgoing_links"] == [{"title": "Visible source", "url": "https://fixture.example/visible"}]
            assert result["observed_json_urls"][0]["url"] == "https://fixture.example/collection"
            assert all(method == "GET" for _, method in fetched)
            assert result["browser_runtime"]["network"]["blocked_requests"] >= 1
            reader.limits = replace(reader.limits, max_text_chars=7, max_links=1, max_link_candidates=1)
            clipped = await reader.read(URL)
            assert len(clipped["text"]) == 7 and clipped["truncated"]
        finally:
            await reader.close()
    asyncio.run(run())
