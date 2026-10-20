"""Explicitly enabled CPU browser reads of public, successful HTML documents.

Importing this module, constructing a disabled reader and declaring its config
do not import Playwright. Browser failures return an error observation so the
owner can retain its ordinary HTTP result. Network JSON is navigation metadata
only: it must be fetched explicitly before its contents are read or cited.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import importlib.metadata
import json
import math
import urllib.parse

BROWSER_READER_VERSION = "public-visible-dom-get-navigation-v1"
PLAYWRIGHT_VERSION = "1.56.0"
_NAVIGATION_NOTE = "Observed successful public GET JSON collection response. This is navigation metadata only; explicitly fetch this URL before using its contents or citing it."
_LINK_NOTE = "Visible links from the rendered public page. Linked pages have not been read; fetch them before using their contents or citing them."


@dataclass(frozen=True)
class BrowserLimits:
    concurrency: int = 2
    total_timeout_seconds: float = 45.0
    cleanup_timeout_seconds: float = 3.0
    navigation_ms: int = 20000
    request_timeout_ms: int = 15000
    settle_ms: int = 4000
    idle_ms: int = 3000
    max_requests: int = 150
    max_responses: int = 128
    max_response_bytes: int = 8_000_000
    max_total_response_bytes: int = 32_000_000
    max_text_chars: int = 200_000
    max_links: int = 256
    max_link_candidates: int = 1024
    max_json_bytes: int = 500_000
    max_json_urls: int = 12

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name.endswith("_seconds"):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Browser {name} must be finite and positive")
            elif type(value) is not int or value < (0 if name in {"settle_ms", "idle_ms"} else 1):
                raise ValueError(f"Invalid browser {name}")
        if self.max_link_candidates < self.max_links:
            raise ValueError("Browser link candidate budget must cover the returned link budget")


def browser_reader_configuration(enabled=False, limits=None):
    if type(enabled) is not bool:
        raise ValueError("browser reader enabled must be a boolean")
    limits = BrowserLimits() if limits is None else limits
    if not isinstance(limits, BrowserLimits):
        raise TypeError("limits must be BrowserLimits")
    return {"enabled": enabled, "version": BROWSER_READER_VERSION,
            "limits": asdict(limits), "network": "public HTTP(S) GET/HEAD; redirects rejected",
            "json_navigation_only": True}


def validate_browser_reader_runtime(value):
    """Validate portable enabled-run metadata without importing Playwright."""
    expected_keys = {"version", "browser_version", "playwright_version", "cpu_only", "configuration"}
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("Invalid browser reader runtime fields")
    if value["version"] != BROWSER_READER_VERSION or value["cpu_only"] is not True:
        raise ValueError("Invalid browser reader version or CPU setting")
    if value["playwright_version"] != PLAYWRIGHT_VERSION:
        raise ValueError(f"Browser reader requires Playwright {PLAYWRIGHT_VERSION}")
    if not isinstance(value["browser_version"], str) or not value["browser_version"].strip():
        raise ValueError("Missing browser reader browser version")
    try:
        configuration = json.dumps(value["configuration"], sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid browser reader configuration") from exc
    if configuration != json.dumps(browser_reader_configuration(enabled=True), sort_keys=True):
        raise ValueError("Browser reader runtime requires its complete default enabled configuration")
    return value


def _http_url(url):
    if not isinstance(url, str):
        raise ValueError("Expected a public HTTP(S) URL")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise ValueError("Expected public HTTP(S) without host credentials")
    return url


def _public_url(url):
    # Late import avoids a cycle when the HTTP tools explicitly inject a reader.
    from .tools import _public_url as validate
    return validate(_http_url(url))


def _has_collection(value, depth=0):
    """Bounded structural filter; no provider schemas, labels or endpoint rules."""
    if depth > 5:
        return False
    if isinstance(value, list):
        if len(value) >= 2 and all(isinstance(item, dict) for item in value[:20]):
            return True
        return any(_has_collection(item, depth + 1) for item in value[:40])
    if isinstance(value, dict):
        return any(_has_collection(item, depth + 1) for item in list(value.values())[:80])
    return False


_VISIBLE_DOM = r"""limits => {
 const text=document.body ? document.body.innerText : '';
 const links=[],seen=new Set();let candidatesTruncated=false;
 for(const a of document.querySelectorAll('a[href]')) {
  if(a.closest('[hidden],[aria-hidden="true"]')||!a.getClientRects().length)continue;
  const style=getComputedStyle(a);
  if(style.display==='none'||style.visibility==='hidden')continue;
  let url;try {url=new URL(a.href)} catch {continue}
  if(!['http:','https:'].includes(url.protocol)||url.username||url.password||seen.has(url.href))continue;
  seen.add(url.href);
  if(links.length>=limits.max_link_candidates){candidatesTruncated=true;break}
  links.push({title:a.innerText.trim().slice(0,200),url:url.href});
 }
 return {title:document.title.slice(0,500),text:text.slice(0,limits.max_text_chars),
   total_characters:text.length,truncated:text.length>limits.max_text_chars,
   outgoing_links:links,link_candidates_truncated:candidatesTruncated};
}"""


class BrowserReader:
    def __init__(self, *, enabled=False, limits=None, url_validator=None):
        browser_reader_configuration(enabled, limits)
        self.enabled = enabled
        self.limits = BrowserLimits() if limits is None else limits
        self.url_validator = _public_url if url_validator is None else url_validator
        self._semaphore = asyncio.Semaphore(self.limits.concurrency)
        self._start_lock = asyncio.Lock()
        self._playwright = self._browser = None
        self._runtime = None
        self._closed = False

    def configuration(self):
        return browser_reader_configuration(self.enabled, self.limits)

    @property
    def runtime(self):
        if self._runtime is None:
            return None
        # The owner can persist this exact, path-free runtime in its manifest.
        return {**self._runtime, "configuration": self.configuration()}

    async def start(self):
        """Explicit CPU preflight; disabled readers do not import Playwright."""
        if not self.enabled:
            return None
        await asyncio.wait_for(self._start(), timeout=self.limits.total_timeout_seconds)
        return self.runtime

    async def preflight(self):
        return await self.start()

    async def _validate(self, url):
        _http_url(url)
        await asyncio.to_thread(self.url_validator, url)

    async def _start(self):
        async with self._start_lock:
            if self._closed:
                raise RuntimeError("Browser reader is closed")
            if self._browser is None:
                try:
                    package_version = importlib.metadata.version("playwright")
                except importlib.metadata.PackageNotFoundError as exc:
                    raise RuntimeError(f"Optional browser reader requires Playwright {PLAYWRIGHT_VERSION}") from exc
                if package_version != PLAYWRIGHT_VERSION:
                    raise RuntimeError(f"Optional browser reader requires Playwright {PLAYWRIGHT_VERSION}; found {package_version}")
                from playwright.async_api import async_playwright
                self._playwright = await async_playwright().start()
                try:
                    self._browser = await self._playwright.chromium.launch(headless=True,
                        args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"])
                    self._runtime = {"version": BROWSER_READER_VERSION,
                                     "browser_version": self._browser.version,
                                     "playwright_version": package_version, "cpu_only": True}
                except BaseException:
                    await self._playwright.stop()
                    self._playwright = None
                    raise

    async def _fetch_route(self, route):
        # Playwright's route callback runs only for the FIRST URL of a server
        # redirect chain. Never use continue_() or automatic redirect following.
        return await route.fetch(max_redirects=0, max_retries=0,
                                 timeout=self.limits.request_timeout_ms)

    async def read(self, url):
        if not self.enabled:
            return {"url": url, "error": "Optional browser reader is disabled", "http_status": None}
        try:
            # Includes queue wait, validation, browser startup, rendering and
            # metadata capture. Cancellation still permits bounded cleanup.
            return await asyncio.wait_for(self._read(url), timeout=self.limits.total_timeout_seconds)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return {"url": url, "error": "Optional browser read failed; retain the ordinary HTTP result",
                    "error_type": type(exc).__name__, "http_status": None}

    async def _read(self, url):
        await self._validate(url)
        async with self._semaphore:
            await self._start()
            context = await self._browser.new_context(service_workers="block", accept_downloads=False)
            limits = self.limits
            state = {"requests": 0, "responses": 0, "response_slots": 0, "response_bytes": 0,
                     "blocked_requests": 0, "redirects_rejected": 0,
                     "request_limit_reached": False, "response_limit_reached": False,
                     "response_bytes_limit_reached": False, "json_navigation_truncated": False,
                     "json_body_inspection_truncated": False}
            observed = []; observed_urls = set(); route_tasks = set(); auxiliary_tasks = set()
            validated_link_hosts = set()

            def auxiliary(coroutine):
                task = asyncio.create_task(coroutine); auxiliary_tasks.add(task)

            async def route_handler(route):
                task = asyncio.current_task(); route_tasks.add(task)
                request = route.request; response = None
                state["requests"] += 1
                try:
                    if state["requests"] > limits.max_requests:
                        state["request_limit_reached"] = True
                        raise ValueError("request budget")
                    if state["response_slots"] >= limits.max_responses:
                        state["response_limit_reached"] = True
                        raise ValueError("response budget")
                    if request.method not in {"GET", "HEAD"} or request.resource_type in {"image", "font", "media"}:
                        raise ValueError("blocked request kind")
                    await self._validate(request.url)
                    validated_link_hosts.add(urllib.parse.urlsplit(request.url).netloc)
                    if state["response_slots"] >= limits.max_responses:
                        state["response_limit_reached"] = True
                        raise ValueError("response budget")
                    state["response_slots"] += 1
                    response = await self._fetch_route(route)
                    state["responses"] += 1
                    if state["responses"] > limits.max_responses:
                        state["response_limit_reached"] = True
                        raise ValueError("response budget")
                    status = response.status
                    if 300 <= status < 400:
                        state["redirects_rejected"] += 1
                        raise ValueError("browser redirects are not followed")
                    headers = response.headers
                    if headers.get("content-disposition", "").lower().lstrip().startswith("attachment"):
                        raise ValueError("downloads disabled")
                    length = headers.get("content-length", "")
                    if length.isdigit() and int(length) > limits.max_response_bytes:
                        state["response_bytes_limit_reached"] = True
                        raise ValueError("response byte budget")
                    body = await response.body()
                    if len(body) > limits.max_response_bytes or state["response_bytes"] + len(body) > limits.max_total_response_bytes:
                        state["response_bytes_limit_reached"] = True
                        raise ValueError("response byte budget")
                    state["response_bytes"] += len(body)
                    ct = headers.get("content-type", "").partition(";")[0].strip().lower()
                    if request.method == "GET" and request.resource_type in {"xhr", "fetch"} and status == 200 and (ct == "application/json" or ct.endswith("+json")):
                        if len(body) > limits.max_json_bytes:
                            state["json_body_inspection_truncated"] = True
                        else:
                            try:
                                collection = _has_collection(json.loads(body))
                            except (ValueError, RecursionError):
                                collection = False
                            if collection and request.url not in observed_urls:
                                if len(observed) >= limits.max_json_urls:
                                    state["json_navigation_truncated"] = True
                                else:
                                    observed_urls.add(request.url)
                                    observed.append({"url": request.url, "status": 200, "bytes": len(body),
                                                     "navigation_only": True, "note": _NAVIGATION_NOTE})
                    # APIResponse.body() is decoded. Do not forward compressed
                    # transfer framing around decoded bytes to the renderer.
                    forwarded = {key: value for key, value in headers.items()
                                 if key.lower() not in {"content-encoding", "content-length", "transfer-encoding", "connection"}}
                    await route.fulfill(status=status, headers=forwarded, body=body)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    state["blocked_requests"] += 1
                    try:
                        await route.abort()
                    except Exception:
                        pass
                finally:
                    if response is not None:
                        try:
                            await response.dispose()
                        except Exception:
                            pass
                    route_tasks.discard(task)

            result = None
            try:
                await context.route("**/*", route_handler)
                await context.route_web_socket("**/*", lambda ws: auxiliary(ws.close()))
                page = await context.new_page()
                page.on("download", lambda download: auxiliary(download.cancel()))
                context.on("page", lambda popup: auxiliary(popup.close()))
                response = await page.goto(url, wait_until="domcontentloaded", timeout=limits.navigation_ms)
                status = response.status if response else None
                if status is None or not 200 <= status < 300:
                    return {"url": url, "error": "Browser navigation did not return a successful public document", "http_status": status}
                await self._validate(page.url)
                await page.wait_for_timeout(limits.settle_ms)
                if limits.idle_ms:
                    from playwright.async_api import TimeoutError as BrowserTimeoutError
                    try:
                        await page.wait_for_load_state("networkidle", timeout=limits.idle_ms)
                    except BrowserTimeoutError:
                        pass
                result = await page.evaluate(_VISIBLE_DOM, {"max_text_chars": limits.max_text_chars,
                                                            "max_link_candidates": limits.max_link_candidates})
                if not result["text"].strip():
                    return {"url": url, "error": "Browser returned no visible text; check browser/font readiness", "http_status": status}
                link_semaphore = asyncio.Semaphore(8)
                async def public_link(link):
                    async with link_semaphore:
                        try:
                            _http_url(link["url"])
                            key = urllib.parse.urlsplit(link["url"]).netloc
                            if key not in validated_link_hosts:
                                await self._validate(link["url"])
                                validated_link_hosts.add(key)
                            return link
                        except Exception:
                            return None
                links = [link for link in await asyncio.gather(*[public_link(link) for link in result["outgoing_links"]]) if link is not None]
                result["outgoing_links"] = links[:limits.max_links]
                result["outgoing_links_truncated"] = len(links) > limits.max_links
                result["outgoing_links_note"] = _LINK_NOTE
                result.update(url=url, final_url=page.url, http_status=status,
                              observed_json_urls=list(observed), observed_json_urls_note=_NAVIGATION_NOTE,
                              observed_json_urls_truncated=state["json_navigation_truncated"],
                              browser_runtime={**self.runtime, "network": dict(state)})
                return result
            finally:
                try:
                    await asyncio.wait_for(context.close(), timeout=limits.cleanup_timeout_seconds)
                except Exception:
                    pass
                pending = {task for task in route_tasks | auxiliary_tasks if not task.done()}
                for task in pending:
                    task.cancel()
                if pending:
                    try:
                        await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=limits.cleanup_timeout_seconds)
                    except Exception:
                        pass
                if result is not None and "browser_runtime" in result:
                    # Closing a context can finish/abort outstanding requests.
                    # Persist their final counters rather than an earlier copy.
                    result["browser_runtime"]["network"] = dict(state)

    async def close(self):
        self._closed = True
        browser, playwright = self._browser, self._playwright
        self._browser = self._playwright = None

        async def shutdown():
            try:
                if browser is not None:
                    await browser.close()
            finally:
                if playwright is not None:
                    await playwright.stop()

        try:
            await asyncio.wait_for(shutdown(), timeout=self.limits.cleanup_timeout_seconds)
        except asyncio.TimeoutError:
            # An optional reader must not hang an owner's final cleanup.
            pass
