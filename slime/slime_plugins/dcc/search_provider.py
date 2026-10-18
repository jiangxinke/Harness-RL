"""Public search providers with explicit coverage, bounded caches and cooldowns."""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.parse
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

_CACHE: OrderedDict[tuple[str, str, int], tuple[float, list[dict[str, str]]]] = OrderedDict()
_CACHE_LOCK = threading.Lock()
_DDG_LOCK = threading.Lock()
_DDG_COOLDOWN_UNTIL = 0.0
_DDG_NEXT_REQUEST = 0.0
_CACHE_SECONDS = 3600
_PUBLIC_LOCKS = {backend: threading.Lock() for backend in ("seznam", "baidu_mobile")}
_PUBLIC_COOLDOWN_UNTIL: dict[str, float] = {}
_PUBLIC_NEXT_REQUEST: dict[str, float] = {}
_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}


def _chinese(query: str) -> bool:
    return any("\u4e00" <= character <= "\u9fff" for character in query)


def _cached(backend: str, query: str, count: int) -> list[dict[str, str]] | None:
    key = (backend, query, count)
    with _CACHE_LOCK:
        value = _CACHE.get(key)
        if value and time.monotonic() - value[0] < _CACHE_SECONDS:
            _CACHE.move_to_end(key)
            return [hit.copy() for hit in value[1]]
        _CACHE.pop(key, None)
    return None


def _remember(backend: str, query: str, count: int, hits: list[dict[str, str]]) -> list[dict[str, str]]:
    with _CACHE_LOCK:
        _CACHE[(backend, query, count)] = (time.monotonic(), [hit.copy() for hit in hits])
        while len(_CACHE) > 1024:
            _CACHE.popitem(last=False)
    return hits


def wikipedia_search(query: str, count: int) -> list[dict[str, str]]:
    from .tools import InfrastructureError, _request, visible_text

    cached = _cached("wikipedia", query, count)
    if cached is not None:
        return cached
    language = "zh" if _chinese(query) else "en"
    endpoint = f"https://{language}.wikipedia.org/w/api.php?" + urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": count, "utf8": "1"})
    try:
        payload = json.loads(_request(endpoint))
        if "error" in payload:
            raise ValueError("MediaWiki API returned an error")
        rows = payload["query"]["search"]
        if not isinstance(rows, list):
            raise ValueError("MediaWiki search results must be a list")
        hits = [{"title": row["title"], "url": f"https://{language}.wikipedia.org/wiki/" + urllib.parse.quote(row["title"].replace(" ", "_"), safe="()_"), "snippet": visible_text(row.get("snippet", ""))} for row in rows[:count]]
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise InfrastructureError("Wikipedia search returned an invalid response") from exc
    return _remember("wikipedia", query, count, hits)


def wikipedia_page(url: str) -> str | None:
    """Read article HTML through the official API, retaining enumeration tables."""

    from .tools import InfrastructureError, _request, visible_text

    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password or parsed.hostname not in {"en.wikipedia.org", "zh.wikipedia.org"} or not parsed.path.startswith("/wiki/"):
        return None
    title = urllib.parse.unquote(parsed.path[len("/wiki/") :]).replace("_", " ")
    # TextExtracts deliberately removes tables. WideSearch questions often ask
    # for the entities in those tables, so fetch parsed article HTML instead.
    endpoint = f"https://{parsed.hostname}/w/api.php?" + urllib.parse.urlencode({"action": "parse", "prop": "text", "redirects": "1", "page": title, "format": "json", "formatversion": "2", "utf8": "1"})
    try:
        payload = json.loads(_request(endpoint))
        article = payload["parse"]
        markup = article["text"]
        if not isinstance(markup, str):
            raise ValueError("Wikipedia article HTML must be a string")
        # The parse API already returns article content without site chrome.
        # Generic article extraction can remove table headers and captions;
        # retain every visible cell and row from this article-only HTML.
        text = visible_text(markup)
        if not text.strip():
            raise ValueError("Wikipedia article is missing")
        return str(article.get("title", title)) + "\n" + text
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise InfrastructureError("Wikipedia article text is unavailable") from exc


def _duckduckgo_search(query: str, count: int, search: Callable[[], list[dict[str, str]]]) -> list[dict[str, str]]:
    from .tools import InfrastructureError

    global _DDG_COOLDOWN_UNTIL, _DDG_NEXT_REQUEST
    cached = _cached("duckduckgo_lite", query, count)
    if cached is not None:
        return cached
    with _DDG_LOCK:
        # Check again after waiting for another episode's query.
        cached = _cached("duckduckgo_lite", query, count)
        if cached is not None:
            return cached
        now = time.monotonic()
        if now < _DDG_COOLDOWN_UNTIL:
            raise InfrastructureError("DuckDuckGo is in a 300-second process-local cooldown")
        delay = _DDG_NEXT_REQUEST - now
        if delay > 0:
            time.sleep(min(delay, 1.0))
        _DDG_NEXT_REQUEST = time.monotonic() + 1.0
        try:
            hits = search()
        except InfrastructureError:
            _DDG_COOLDOWN_UNTIL = time.monotonic() + 300
            raise
        return _remember("duckduckgo_lite", query, count, hits)


def search_with_fallback(query: str, count: int, search: Callable[[], list[dict[str, str]]]) -> dict[str, Any]:
    from .tools import InfrastructureError

    try:
        hits = _duckduckgo_search(query, count, search)
        if hits:
            return {"results": hits, "backend": "duckduckgo_lite", "search_scope": "web", "fallback_reason": None}
        reason = "DuckDuckGo returned no search results"
    except InfrastructureError as exc:
        reason = str(exc)
    return {"results": wikipedia_search(query, count), "backend": "wikipedia", "search_scope": "encyclopedia", "fallback_reason": reason, "coverage_note": "This is encyclopedia search, not a general web index. Adjust queries or use a configured web-search API for broader coverage."}


def _parse_public_results(raw: str, backend: str, count: int) -> list[dict[str, str]]:
    from .tools import InfrastructureError, visible_text

    try:
        from lxml import etree, html
    except ImportError as exc:
        raise InfrastructureError("Public HTML search requires configs/dcc/requirements.txt") from exc
    try:
        document = html.fromstring(raw)
    except (ValueError, etree.ParserError) as exc:
        raise InfrastructureError(f"{backend} returned invalid search HTML") from exc
    # Normal pages may contain CAPTCHA code in scripts. Inspect visible page
    # text only; never echo a challenge page, which can contain client details.
    for element in document.xpath("//script | //style | //noscript"):
        element.drop_tree()
    text = visible_text(html.tostring(document, encoding="unicode"))
    if re.search(r"unusual traffic|verify (?:that )?you are human|not a robot|百度安全验证|请输入验证码|podezřel[áéý]? aktivita|ověřte,? že", text, re.IGNORECASE):
        raise InfrastructureError(f"{backend} challenged this request")

    def clean(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    elements = document.xpath('//*[@data-log]') if backend == "baidu_mobile" else document.xpath("//h2/a | //h3/a | //a[.//h3]")
    for element in elements:
        if backend == "baidu_mobile":
            try:
                metadata = json.loads(element.get("data-log", ""))
            except (ValueError, TypeError):
                continue
            if not isinstance(metadata, dict):
                continue
            url = metadata.get("mu", "")
            headings = element.xpath('.//h3 | .//h2 | .//*[contains(@class,"c-title") or contains(@class,"cosc-title")]')
            title = clean(" ".join(headings[0].itertext())) if headings else ""
            snippet = clean(" ".join(element.itertext()))
        else:
            url = element.get("href", "")
            title = clean(" ".join(element.itertext()))
            parent = element.getparent()
            if parent is not None and parent.getparent() is not None:
                parent = parent.getparent()
            snippet = clean(" ".join(parent.itertext())) if parent is not None else title
        if not isinstance(url, str):
            continue
        try:
            parsed = urllib.parse.urlsplit(url)
        except ValueError:
            continue
        host = parsed.hostname or ""
        if parsed.scheme not in {"https", "http"} or not host or parsed.username or parsed.password or url in seen:
            continue
        # Related-query widgets contain synthetic data-log.mu hostnames; they
        # are navigation, not indexed pages. Preserve genuine Baidu articles.
        if host == "search.seznam.cz" or host.endswith(".recommend_list.baidu.com") or (host == "ikan.baidu.com" and parsed.path == "/pedia/"):
            continue
        if not title:
            title = snippet[:240]
        if not title:
            continue
        seen.add(url)
        hits.append({"title": title[:240], "url": url, "snippet": snippet[:650]})
        if len(hits) >= count:
            break
    return hits


def public_html_search(query: str, count: int, backend: str) -> list[dict[str, str]]:
    """Search public HTML endpoints without following provider tracking links."""
    from .tools import InfrastructureError, _request

    if backend not in _PUBLIC_LOCKS:
        raise ValueError(f"unknown public HTML search backend: {backend}")
    cached = _cached(backend, query, count)
    if cached is not None:
        return cached
    with _PUBLIC_LOCKS[backend]:
        cached = _cached(backend, query, count)
        if cached is not None:
            return cached
        now = time.monotonic()
        if now < _PUBLIC_COOLDOWN_UNTIL.get(backend, 0.0):
            raise InfrastructureError(f"{backend} is in a 300-second process-local cooldown")
        delay = _PUBLIC_NEXT_REQUEST.get(backend, 0.0) - now
        if delay > 0:
            time.sleep(min(delay, 0.5))
        _PUBLIC_NEXT_REQUEST[backend] = time.monotonic() + 0.5
        endpoint, parameter = ("https://m.baidu.com/s", "word") if backend == "baidu_mobile" else ("https://search.seznam.cz/", "q")
        try:
            raw = _request(endpoint + "?" + urllib.parse.urlencode({parameter: query}), headers=_BROWSER_HEADERS)
            hits = _parse_public_results(raw, backend, count)
        except InfrastructureError:
            _PUBLIC_COOLDOWN_UNTIL[backend] = time.monotonic() + 300
            raise
        return _remember(backend, query, count, hits)


def public_web_search(query: str, count: int, search: Callable[[], list[dict[str, str]]]) -> dict[str, Any]:
    """Try general web indexes before an explicitly marked encyclopedia search."""
    from .tools import InfrastructureError

    providers: list[tuple[str, Callable[[], list[dict[str, str]]]]] = [("duckduckgo_lite", lambda: _duckduckgo_search(query, count, search))]
    # Seznam's public index returned relevant English results but no Chinese
    # coverage in live multi-query validation. Route Chinese to Baidu directly.
    if not _chinese(query):
        providers.append(("seznam", lambda: public_html_search(query, count, "seznam")))
    providers.extend([("baidu_mobile", lambda: public_html_search(query, count, "baidu_mobile")), ("wikipedia", lambda: wikipedia_search(query, count))])
    reasons: list[str] = []
    last_empty: dict[str, Any] | None = None
    for backend, provider in providers:
        try:
            hits = provider()
        except InfrastructureError as exc:
            # Fixed labels avoid leaking response text, endpoint addresses or
            # client details into policy observations and experiment reports.
            message = str(exc).lower()
            reason = "cooldown" if "cooldown" in message else "challenge" if "challeng" in message or "captcha" in message else "provider error"
            reasons.append(f"{backend}: {reason}")
            continue
        result: dict[str, Any] = {"results": hits, "backend": backend, "search_scope": "encyclopedia" if backend == "wikipedia" else "web", "fallback_reason": "; ".join(reasons) or None}
        if backend == "wikipedia":
            result["coverage_note"] = "This is encyclopedia search, not a general web index."
        if hits:
            return result
        last_empty = result
        reasons.append(f"{backend}: no search results")
    if last_empty is not None:
        # An empty but valid index response remains a policy observation even
        # if subsequent fallback providers are unavailable.
        last_empty["fallback_reason"] = "; ".join(reasons)
        return last_empty
    raise InfrastructureError("All public search providers are unavailable")
