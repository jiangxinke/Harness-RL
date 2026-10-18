"""Bounded contract retrieval and genuine HTTP web tools.

Credentials and deployment addresses come from environment variables. No web
credentials, server paths, retrieved pages, or cache contents belong in Git.
"""

from __future__ import annotations

import asyncio
import ipaddress
import io
import json
import math
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Any


class InfrastructureError(RuntimeError):
    """An external service failed; the episode must not become a policy target."""


def _terms(text: str) -> list[str]:
    stop = {"the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "for", "that", "this", "any", "under", "contract", "agreement", "highlight", "parts", "related", "whether", "which", "what", "does", "shall", "with", "as", "be", "by", "from", "on", "it"}
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 1 and token not in stop]


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    start: int
    end: int
    text: str


class DocumentTools:
    def __init__(self, document: str, *, chunk_chars: int = 1600, overlap_chars: int = 200) -> None:
        if chunk_chars <= overlap_chars or overlap_chars < 0:
            raise ValueError("chunk size must exceed a nonnegative overlap")
        self.document = document
        self.chunks: list[DocumentChunk] = []
        start = 0
        while start < len(document):
            end = min(start + chunk_chars, len(document))
            if end < len(document):
                boundary = max(document.rfind("\n", start + chunk_chars // 2, end), document.rfind(". ", start + chunk_chars // 2, end))
                if boundary > start:
                    end = boundary + 1
            self.chunks.append(DocumentChunk(start, end, document[start:end]))
            if end == len(document):
                break
            start = max(start + 1, end - overlap_chars)
        self.counts = [Counter(_terms(chunk.text)) for chunk in self.chunks]
        self.average_length = sum(sum(count.values()) for count in self.counts) / max(len(self.counts), 1)
        self.document_frequency = Counter(term for count in self.counts for term in count)

    def search_document(self, query: str, *, top_k: int = 4) -> dict[str, Any]:
        query_terms = set(_terms(query))
        ranked: list[tuple[float, int]] = []
        for index, counts in enumerate(self.counts):
            score = 0.0
            length = sum(counts.values())
            for term in query_terms:
                frequency = counts[term]
                if frequency:
                    inverse = math.log(1 + (len(self.counts) - self.document_frequency[term] + 0.5) / (self.document_frequency[term] + 0.5))
                    score += inverse * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * length / max(self.average_length, 1)))
            ranked.append((score, index))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        hits = [{"start": self.chunks[index].start, "end": self.chunks[index].end, "text": self.chunks[index].text, "score": round(score, 4)} for score, index in ranked[: max(1, min(int(top_k), 8))]]
        return {"document_length": len(self.document), "hits": hits}

    def read_document(self, start: int = 0, end: int | None = None) -> dict[str, Any]:
        start = max(0, int(start))
        end = min(len(self.document), int(end) if end is not None else start + 8000, start + 12000)
        if start >= len(self.document) or end <= start:
            return {"error": "invalid document character range", "document_length": len(self.document)}
        return {"start": start, "end": end, "text": self.document[start:end], "document_length": len(self.document)}


class _VisibleText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self.hidden += 1
        if tag in {"p", "div", "br", "li", "tr", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


def visible_text(html: str) -> str:
    parser = _VisibleText()
    parser.feed(html)
    return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n\n", unescape(" ".join(parser.parts)))).strip()


class _DuckDuckGoResults(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hits: list[dict[str, str]] = []
        self.capture: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()
        if tag == "a" and any(name in classes for name in ("result-link", "result__a")):
            target = unescape(attributes.get("href") or "")
            if target.startswith("//"):
                target = "https:" + target
            parsed = urllib.parse.urlsplit(target)
            target = urllib.parse.parse_qs(parsed.query).get("uddg", [target])[0]
            self.hits.append({"title": "", "url": target, "snippet": ""})
            self.capture = "title"
        elif tag in {"td", "a", "div"} and any(name in classes for name in ("result-snippet", "result__snippet")):
            self.capture = "snippet"

    def handle_endtag(self, tag: str) -> None:
        if tag in {"a", "td", "div"}:
            self.capture = None

    def handle_data(self, data: str) -> None:
        if self.capture and self.hits:
            self.hits[-1][self.capture] += data


def _public_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("web tools require an HTTP(S) URL without credentials")
    for record in socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM):
        if not ipaddress.ip_address(record[4][0]).is_global:
            raise ValueError("web tools only fetch public internet hosts")
    return url


def _request(url: str, *, payload: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> str:
    # Search service endpoints are explicitly configured by the operator and
    # may be local. User-provided fetch_url targets are validated separately.
    request_headers = {"User-Agent": "Mozilla/5.0 (compatible; DCCResearch/1.0)", **(headers or {})}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        request_headers.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(url, data=data, headers=request_headers)
    try:
        with urllib.request.urlopen(request, timeout=float(os.environ.get("DCC_WEB_TIMEOUT", "30"))) as response:
            return response.read(2_000_000).decode(response.headers.get_content_charset() or "utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise InfrastructureError(f"HTTP retrieval failed ({type(exc).__name__})") from exc


class _PublicRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        _public_url(new_url)
        return super().redirect_request(request, fp, code, message, headers, new_url)


def _download_public(url: str) -> tuple[bytes, str, str]:
    _public_url(url)
    opener = urllib.request.build_opener(_PublicRedirects())
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; DCCResearch/1.0)"})
    try:
        with opener.open(request, timeout=float(os.environ.get("DCC_WEB_TIMEOUT", "30"))) as response:
            content = response.read(5_000_000)
            return content, response.headers.get_content_type(), response.headers.get_content_charset() or "utf-8"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise InfrastructureError(f"HTTP retrieval failed ({type(exc).__name__})") from exc


def extract_page(content: bytes, content_type: str, encoding: str = "utf-8") -> str:
    if content_type == "application/pdf" or content.startswith(b"%PDF-"):
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ValueError("PDF reader unavailable; install configs/dcc/requirements.txt") from exc
        try:
            reader = PdfReader(io.BytesIO(content))
            return "\n\n".join(page.extract_text() or "" for page in reader.pages[:100])
        except Exception as exc:
            raise ValueError("could not extract text from this PDF") from exc
    try:
        text = content.decode(encoding, errors="replace")
    except LookupError:
        text = content.decode("utf-8", errors="replace")
    if "html" in content_type or "<html" in text[:1000].lower():
        try:
            import trafilatura

            extracted = trafilatura.extract(text, include_tables=True, favor_recall=True)
            if extracted:
                return extracted
        except ImportError:
            pass
        except Exception:
            # Unsupported page structures still have an HTML text fallback.
            pass
        return visible_text(text)
    if content_type.startswith("text/") or content_type in {"application/json", "application/xml"}:
        return text
    raise ValueError(f"unsupported page content type: {content_type}")


class WebTools:
    """Search using DuckDuckGo, Serper, Brave, SearXNG, or Bing RSS."""

    def __init__(self, *, backend: str | None = None, max_page_chars: int = 14000) -> None:
        self.backend = backend or os.environ.get("DCC_SEARCH_BACKEND", "duckduckgo_lite")
        self.max_page_chars = max_page_chars
        self.sources: dict[str, dict[str, str]] = {}
        self.pages: dict[str, str] = {}
        self._semaphore = asyncio.Semaphore(8)

    async def web_search(self, query: str, *, count: int = 8) -> dict[str, Any]:
        query = str(query).strip()[:1000]
        if not query:
            return {"error": "query must be nonempty"}
        count = max(1, min(int(count), 10))
        async with self._semaphore:
            hits = await asyncio.to_thread(self._search, query, count)
        for hit in hits:
            self.sources[hit["url"]] = hit
        return {"query": query, "results": hits, "backend": self.backend}

    def _search(self, query: str, count: int) -> list[dict[str, str]]:
        if self.backend == "duckduckgo_lite":
            raw = _request("https://lite.duckduckgo.com/lite/?" + urllib.parse.urlencode({"q": query}))
            if "anomaly-modal" in raw or "challenge-form" in raw:
                raise InfrastructureError("DuckDuckGo challenged this request; use a configured search API")
            parser = _DuckDuckGoResults()
            parser.feed(raw)
            hits = parser.hits
        elif self.backend == "serper":
            key = os.environ.get("SERPER_API_KEY")
            if not key:
                raise InfrastructureError("SERPER_API_KEY is required for serper")
            data = json.loads(_request("https://google.serper.dev/search", payload={"q": query, "num": count}, headers={"X-API-KEY": key}))
            hits = [{"title": item.get("title", ""), "url": item["link"], "snippet": item.get("snippet", "")} for item in data.get("organic", [])]
        elif self.backend == "brave":
            key = os.environ.get("BRAVE_SEARCH_API_KEY")
            if not key:
                raise InfrastructureError("BRAVE_SEARCH_API_KEY is required for brave")
            url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode({"q": query, "count": count})
            data = json.loads(_request(url, headers={"X-Subscription-Token": key}))
            hits = [{"title": item.get("title", ""), "url": item["url"], "snippet": item.get("description", "")} for item in data.get("web", {}).get("results", [])]
        elif self.backend == "searxng":
            endpoint = os.environ.get("DCC_SEARCH_URL", "").rstrip("/")
            if not endpoint:
                raise InfrastructureError("DCC_SEARCH_URL is required for searxng")
            data = json.loads(_request(endpoint + "/search?" + urllib.parse.urlencode({"q": query, "format": "json"})))
            hits = [{"title": item.get("title", ""), "url": item["url"], "snippet": item.get("content", "")} for item in data.get("results", [])]
        elif self.backend == "bing_rss":
            raw = _request("https://www.bing.com/search?" + urllib.parse.urlencode({"q": query, "format": "rss", "count": count}))
            try:
                root = ET.fromstring(raw)
            except ET.ParseError as exc:
                raise InfrastructureError("search endpoint returned invalid RSS") from exc
            hits = [{"title": item.findtext("title", ""), "url": item.findtext("link", ""), "snippet": visible_text(item.findtext("description", ""))} for item in root.findall("./channel/item")]
        else:
            raise ValueError(f"unknown search backend: {self.backend}")
        return [item for item in hits if item["url"].startswith(("https://", "http://"))][:count]

    async def fetch_url(self, url: str, start: int = 0) -> dict[str, Any]:
        if url not in self.pages:
            try:
                async with self._semaphore:
                    raw, content_type, encoding = await asyncio.to_thread(_download_public, url)
                self.pages[url] = await asyncio.to_thread(extract_page, raw, content_type, encoding)
            except (InfrastructureError, ValueError, socket.gaierror) as exc:
                # Dead/blocked pages are normal web-agent observations. An
                # unavailable search backend, in contrast, aborts the episode.
                return {"url": url, "error": str(exc)}
        text = self.pages[url]
        start = max(0, int(start))
        end = min(len(text), start + self.max_page_chars)
        result = {"url": url, "text": text[start:end], "start": start, "end": end, "total_characters": len(text), "truncated": end < len(text), "next_start": end if end < len(text) else None}
        self.sources.setdefault(url, {"title": "", "url": url, "snippet": text[:300]})
        return result

    def citation_validity(self, answer: Any) -> float:
        if not isinstance(answer, list) or not answer:
            return 0.0
        citations = [url for row in answer if isinstance(row, dict) for url in row.get("citations", []) if isinstance(url, str)]
        return sum(url in self.sources for url in citations) / len(citations) if citations else 0.0
