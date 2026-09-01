"""
Web Search Tool - Search the web using DuckDuckGo (no API key required).
"""

import json
import logging
import os
import socket
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from threading import Lock
from urllib.parse import urlencode, urlparse
from urllib.request import ProxyHandler, Request, build_opener

from langchain.tools import tool

from deerflow.config import get_app_config

logger = logging.getLogger(__name__)
_PROXY_ENV_NAMES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
_PROXY_CHECK_TIMEOUT = 0.25
_proxy_env_lock = Lock()


class WebSearchUnavailableError(RuntimeError):
    """Raised when the search backend is unreachable rather than simply empty."""


def _local_proxy_is_unavailable(value: str) -> bool:
    try:
        parsed = urlparse(value)
        host = (parsed.hostname or "").lower()
        port = parsed.port
    except ValueError:
        return False
    if host not in {"127.0.0.1", "localhost", "::1"} or port is None:
        return False
    try:
        with socket.create_connection((host, port), timeout=_PROXY_CHECK_TIMEOUT):
            return False
    except OSError:
        return True


@contextmanager
def _bypass_unavailable_local_proxies():
    """Temporarily bypass dead loopback proxies without breaking live proxies."""
    unavailable = [(name, os.environ[name]) for name in _PROXY_ENV_NAMES if os.environ.get(name) and _local_proxy_is_unavailable(os.environ[name])]
    if not unavailable:
        yield
        return
    with _proxy_env_lock:
        try:
            for name, _ in unavailable:
                os.environ.pop(name, None)
            yield
        finally:
            for name, value in unavailable:
                os.environ[name] = value


DEFAULT_BACKEND = "auto"
DEFAULT_REGION = "wt-wt"
DEFAULT_SAFESEARCH = "moderate"
DEFAULT_TIMEOUT = 30
DEFAULT_WIKIPEDIA_REGION = "us-en"

WIKIPEDIA_BACKENDS = {"auto", "all", "wikipedia"}
WIKIPEDIA_LANGUAGE_ALIASES = {
    "jp": "ja",
    "kr": "ko",
    "tzh": "zh",
    "wt": "en",
}


def _normalize_backend(backend: str | list[str] | tuple[str, ...] | None) -> str:
    if backend is None:
        return DEFAULT_BACKEND
    if isinstance(backend, (list, tuple)):
        return ",".join(str(part).strip() for part in backend if str(part).strip()) or DEFAULT_BACKEND
    return str(backend).strip() or DEFAULT_BACKEND


def _normalize_setting(value: str | None, default: str) -> str:
    return str(value).strip() if value else default


def _normalize_proxies(value: str | list[str] | tuple[str, ...] | None) -> list[str | None]:
    """Return ordered DDGS proxy candidates, preserving direct mode as ``None``."""
    if value is None:
        return [None]
    if isinstance(value, str):
        normalized = value.strip()
        return [normalized] if normalized else [None]
    normalized = [str(item).strip() for item in value if str(item).strip()]
    return normalized or [None]


def _search_bing_rss(
    query: str,
    max_results: int,
    proxies: str | list[str] | tuple[str, ...] | None,
    timeout: int,
) -> list[dict]:
    """No-key fallback for when DDGS providers are blocked or rate-limited."""
    is_chinese = _contains_codepoint(query, ((0x3400, 0x9FFF),))
    params = {
        "q": query,
        "format": "rss",
        "count": max_results,
        "setlang": "zh-hans" if is_chinese else "en-US",
        "cc": "CN" if is_chinese else "US",
    }
    url = f"https://www.bing.com/search?{urlencode(params)}"
    request = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; DeerFlowWebSearch/1.0)"},
    )
    last_error: Exception | None = None

    for proxy in _normalize_proxies(proxies):
        if proxy and _local_proxy_is_unavailable(proxy):
            last_error = ConnectionError(f"Configured web-search proxy is unavailable: {proxy}")
            continue
        proxy_map = {"http": proxy, "https": proxy} if proxy else {}
        opener = build_opener(ProxyHandler(proxy_map))
        try:
            with opener.open(request, timeout=max(1, int(timeout))) as response:
                payload = response.read(1024 * 1024)
            root = ET.fromstring(payload)
            results = []
            for item in root.findall("./channel/item"):
                title = (item.findtext("title") or "").strip()
                href = (item.findtext("link") or "").strip()
                body = (item.findtext("description") or "").strip()
                if title and href.startswith(("http://", "https://")):
                    results.append({"title": title, "href": href, "body": body})
                if len(results) >= max_results:
                    break
            if results:
                logger.info("Bing RSS fallback returned %d result(s)", len(results))
                return results
        except Exception as exc:
            last_error = exc
            logger.warning("Bing RSS fallback failed via proxy %s: %s", proxy or "direct", exc)

    if last_error is not None:
        logger.error("All Bing RSS fallback attempts failed: %s", last_error)
    return []


def _backend_includes_wikipedia(backend: str | list[str] | tuple[str, ...] | None) -> bool:
    backend = _normalize_backend(backend)
    return any(part.strip().lower() in WIKIPEDIA_BACKENDS for part in backend.split(","))


def _contains_codepoint(query: str, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= ord(char) <= end for char in query for start, end in ranges)


def _infer_wikipedia_region(query: str) -> str:
    """Pick a valid Wikipedia language region when DDGS' worldwide region is used."""
    if _contains_codepoint(query, ((0x3040, 0x30FF), (0x31F0, 0x31FF))):
        return "jp-ja"
    if _contains_codepoint(query, ((0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F))):
        return "kr-ko"
    if _contains_codepoint(query, ((0x3400, 0x9FFF),)):
        return "cn-zh"
    if _contains_codepoint(query, ((0x0400, 0x04FF),)):
        return "ru-ru"
    if _contains_codepoint(query, ((0x0370, 0x03FF),)):
        return "gr-el"
    if _contains_codepoint(query, ((0x0590, 0x05FF),)):
        return "il-he"
    if _contains_codepoint(query, ((0x0600, 0x06FF),)):
        return "xa-ar"
    return DEFAULT_WIKIPEDIA_REGION


def _resolve_ddgs_region(query: str, region: str | None, backend: str | list[str] | tuple[str, ...] | None) -> str:
    """
    DDGS' wikipedia engine treats the second part of region as a Wikipedia
    subdomain. Its default worldwide region, wt-wt, becomes wt.wikipedia.org.
    """
    normalized_region = _normalize_setting(region, DEFAULT_REGION).lower()
    if not _backend_includes_wikipedia(backend):
        return normalized_region

    if normalized_region == DEFAULT_REGION:
        return _infer_wikipedia_region(query)

    if "-" not in normalized_region:
        return DEFAULT_WIKIPEDIA_REGION

    country, language = normalized_region.split("-", 1)
    return f"{country}-{WIKIPEDIA_LANGUAGE_ALIASES.get(language, language)}"


def _search_text(
    query: str,
    max_results: int = 5,
    region: str | None = DEFAULT_REGION,
    safesearch: str | None = DEFAULT_SAFESEARCH,
    backend: str | list[str] | tuple[str, ...] | None = DEFAULT_BACKEND,
    proxies: str | list[str] | tuple[str, ...] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    rss_fallback: bool = False,
) -> list[dict]:
    """
    Execute text search using DuckDuckGo.

    Args:
        query: Search keywords
        max_results: Maximum number of results
        region: Search region
        safesearch: Safe search level
        backend: DDGS backend(s), e.g. "auto", "duckduckgo", or "duckduckgo,brave"
        proxies: Ordered proxy candidates passed explicitly to DDGS
        timeout: Timeout in seconds for each proxy attempt
        rss_fallback: Fall back to Bing's no-key RSS search endpoint

    Returns:
        List of search results
    """
    try:
        from ddgs import DDGS
    except ImportError:
        logger.error("ddgs library not installed. Run: pip install ddgs")
        return []

    normalized_backend = _normalize_backend(backend)
    normalized_safesearch = _normalize_setting(safesearch, DEFAULT_SAFESEARCH)
    effective_region = _resolve_ddgs_region(query, region, normalized_backend)
    proxy_candidates = _normalize_proxies(proxies)
    last_error: Exception | None = None
    saw_empty_response = False

    for proxy in proxy_candidates:
        if proxy and _local_proxy_is_unavailable(proxy):
            last_error = ConnectionError(f"Configured web-search proxy is unavailable: {proxy}")
            logger.warning("Skipping unavailable web-search proxy: %s", proxy)
            continue

        try:
            with _bypass_unavailable_local_proxies():
                # DDGS intentionally does not read HTTP_PROXY/HTTPS_PROXY. The
                # proxy must be passed explicitly (or supplied as DDGS_PROXY).
                ddgs = DDGS(proxy=proxy, timeout=max(1, int(timeout)))
                results = ddgs.text(
                    query,
                    region=effective_region,
                    safesearch=normalized_safesearch,
                    max_results=max_results,
                    backend=normalized_backend,
                )
            if results:
                return list(results)
            saw_empty_response = True
        except Exception as exc:
            if "no results found" in str(exc).lower():
                saw_empty_response = True
                logger.info("Search returned no results via proxy %s; trying fallback", proxy or "direct")
                continue
            last_error = exc
            logger.warning("Search attempt failed via proxy %s: %s", proxy or "direct", exc)

    if rss_fallback:
        fallback_results = _search_bing_rss(query, max_results, proxies, timeout)
        if fallback_results:
            return fallback_results
    if saw_empty_response:
        return []
    if last_error is not None:
        logger.error("Failed to search web: %s", last_error)
        raise WebSearchUnavailableError(str(last_error)) from last_error
    return []


@tool("web_search", parse_docstring=True)
def web_search_tool(
    query: str,
    max_results: int = 5,
) -> str:
    """Search the web for information. Use this tool to find current information, news, articles, and facts from the internet.

    Args:
        query: Search keywords describing what you want to find. Be specific for better results.
        max_results: Maximum number of results to return. Default is 5.
    """
    config = get_app_config().get_tool_config("web_search")
    region = DEFAULT_REGION
    safesearch = DEFAULT_SAFESEARCH
    backend = DEFAULT_BACKEND
    proxies = None
    timeout = DEFAULT_TIMEOUT
    rss_fallback = False

    if config is not None:
        # Override tool call defaults from config if set.
        max_results = config.model_extra.get("max_results", max_results)
        region = config.model_extra.get("region", region)
        safesearch = config.model_extra.get("safesearch", safesearch)
        backend = config.model_extra.get("backend", backend)
        proxies = config.model_extra.get("proxies", config.model_extra.get("proxy", proxies))
        timeout = config.model_extra.get("timeout", timeout)
        rss_fallback = bool(config.model_extra.get("rss_fallback", rss_fallback))

    try:
        results = _search_text(
            query=query,
            max_results=max_results,
            region=region,
            safesearch=safesearch,
            backend=backend,
            proxies=proxies,
            timeout=timeout,
            rss_fallback=rss_fallback,
        )
    except WebSearchUnavailableError as exc:
        return json.dumps(
            {"error": "WEB_SEARCH_UNAVAILABLE", "message": str(exc), "query": query, "retryable": True},
            ensure_ascii=False,
        )

    if not results:
        return json.dumps({"error": "No results found", "query": query}, ensure_ascii=False)

    normalized_results = [
        {
            "title": r.get("title", ""),
            "url": r.get("href", r.get("link", "")),
            "content": r.get("body", r.get("snippet", "")),
        }
        for r in results
    ]

    output = {
        "query": query,
        "total_results": len(normalized_results),
        "results": normalized_results,
    }

    return json.dumps(output, indent=2, ensure_ascii=False)
