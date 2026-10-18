"""Public search fallback with explicit encyclopedia coverage and rate control."""

from __future__ import annotations

import json
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
    language = "zh" if any("\u4e00" <= character <= "\u9fff" for character in query) else "en"
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
