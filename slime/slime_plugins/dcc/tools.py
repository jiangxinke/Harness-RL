"""Bounded contract retrieval and genuine HTTP web tools.

Credentials and deployment addresses come from environment variables. No web
credentials, server paths, retrieved pages, or cache contents belong in Git.
"""

from __future__ import annotations

import asyncio
import csv
import http.client
import ipaddress
import io
import json
import math
import os
import posixpath
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, OrderedDict
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Any

WEB_READER_VERSION = "bounded-http-explicit-size-v9"

_HTTP_FAILURE_CACHE_SIZE = 128

_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_XLSX_MAX_MEMBERS = 1000
_XLSX_MAX_MEMBER_BYTES = 8_000_000
_XLSX_MAX_TOTAL_BYTES = 25_000_000
_XLSX_MAX_SHEETS = 32
_XLSX_MAX_ROWS = 10_000
_XLSX_MAX_COLUMNS = 256
_XLSX_MAX_CELLS = 100_000
_XLSX_MAX_TEXT_CHARS = 1_000_000
_XLSX_NAV_NAME_CHARS = 128
_XLSX_NAV_PREVIEW_CHARS = 256
_XLSX_NAV_TOTAL_PREVIEW_CHARS = 4096


class InfrastructureError(RuntimeError):
    """An external service failed; the episode must not become a policy target."""


class _HTTPRetrievalError(InfrastructureError):
    """Safe fetch diagnostics without response bodies, reasons, or headers."""

    def __init__(self, http_status: int | None, *, cause_type: str = "network error") -> None:
        self.http_status = http_status
        self.retryable = http_status is None or http_status in {408, 425, 429} or 500 <= http_status < 600
        if http_status in {404, 410}:
            message = (
                f"HTTP {http_status}: this resource is unavailable. Use a real URL returned by web_search "
                "or outgoing_links instead of guessing an endpoint; search for the source index or documented export."
            )
        elif http_status is None:
            message = f"HTTP retrieval failed ({cause_type}). Retry later or use another public source."
        elif self.retryable:
            message = f"HTTP {http_status}: temporary source failure. Retry later or use another public source."
        else:
            message = f"HTTP {http_status}: source request was rejected. Use another public source or an observed link."
        super().__init__(message)


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
        if tag in {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "caption"}:
            self.parts.append("\n")
        elif tag in {"td", "th"} and not self.hidden:
            # Preserve column boundaries when the article extractor falls
            # back to visible HTML, instead of merging names and attributes.
            self.parts.append(" | ")

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


def _outgoing_url(base: str, href: str) -> str | None:
    """Filter link metadata without DNS; actual reads use _public_url again."""
    href = href.strip()
    if not href or href.startswith("#") or len(href) > 2048:
        return None
    try:
        url = urllib.parse.urldefrag(urllib.parse.urljoin(base, href))[0]
        parsed = urllib.parse.urlsplit(url)
        host = (parsed.hostname or "").rstrip(".").lower()
        port = parsed.port
        if (parsed.scheme not in {"http", "https"} or not host or parsed.username is not None
                or parsed.password is not None or len(url) > 2048 or any(character.isspace() or ord(character) < 32 for character in url)):
            return None
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            if ("." not in host or host.endswith((".localhost", ".local", ".localdomain", ".internal", ".lan", ".home", ".invalid", ".test"))
                    or re.fullmatch(r"(?:0x[\da-f]+|\d+)(?:\.(?:0x[\da-f]+|\d+))*", host, re.IGNORECASE)):
                return None
        else:
            if not address.is_global:
                return None
        netloc = f"[{host}]" if ":" in host else host
        if port is not None:
            netloc += f":{port}"
        return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, ""))
    except (ValueError, UnicodeError):
        return None


def _swagger_spec_urls(script: str) -> list[str]:
    """Recognize literal top-level widget metadata without evaluating JS."""
    if len(script) > 65536:
        return []
    token_pattern = r'''(?P<comment>//[^\r\n]*|/\*.*?\*/)|(?P<string>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`)|(?P<name>[A-Za-z_$][\w$]*)|(?P<symbol>[^\s])'''
    tokens = [(match.lastgroup, match.group()) for match in re.finditer(token_pattern, script, re.DOTALL)
              if match.lastgroup != "comment"]
    output = []
    widgets = 0
    for index, token in enumerate(tokens[:-3]):
        if token != ("name", "SwaggerUIBundle") or [value for _, value in tokens[index + 1:index + 3]] != ["(", "{"]:
            continue
        widgets += 1
        if widgets > 8:
            break
        stack = ["{"]
        cursor = index + 3
        while cursor < len(tokens) and stack:
            kind, value = tokens[cursor]
            if (stack == ["{"] and tokens[cursor - 1][1] in {"{", ","}
                    and (kind == "name" and value == "url" or kind == "string" and value in {'"url"', "'url'"})
                    and cursor + 3 < len(tokens)):
                colon, target, delimiter = tokens[cursor + 1:cursor + 4]
                if (colon[1] == ":" and target[0] == "string" and target[1][0] in {'"', "'"}
                        and "\\" not in target[1] and delimiter[1] in {",", "}"}):
                    output.append(target[1][1:-1])
            if kind == "symbol":
                if value in {"{", "[", "("}:
                    stack.append(value)
                elif value in {"}", "]", ")"}:
                    if stack[-1] != {"}": "{", "]": "[", ")": "("}[value]:
                        break
                    stack.pop()
            cursor += 1
    return output


class _PageLinks(HTMLParser):
    """Bounded anchor metadata; downloads cannot be crowded out by navigation."""

    def __init__(self, url: str) -> None:
        super().__init__()
        self.url = url
        self.base: str | None = None
        self.hidden = 0
        self.anchor: dict[str, Any] | None = None
        self.script: list[str] | None = None
        self.script_chars = 0
        self.downloads: dict[str, tuple[str, str]] = {}
        self.navigation: dict[str, tuple[str, str]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "script" and not self.hidden and not attributes.get("src") and (attributes.get("type") or "").lower() in {"", "text/javascript", "application/javascript", "module"}:
            self.script = []
            self.script_chars = 0
        if tag in {"script", "style", "noscript", "svg"}:
            self.hidden += 1
        if self.hidden:
            return
        if tag == "base" and self.base is None:
            self.base = _outgoing_url(self.url, attributes.get("href") or "")
        elif tag in {"redoc", "rapi-doc"}:
            self._remember(attributes.get("spec-url") or "", "OpenAPI specification", True)
        elif tag == "link" and "alternate" in (attributes.get("rel") or "").lower().split():
            mime = (attributes.get("type") or "").partition(";")[0].strip().lower()
            if mime in {"application/json", "application/xml", "text/xml"} or mime.endswith(("+json", "+xml")):
                self._remember(attributes.get("href") or "", "Alternate machine-readable source", True)
        elif tag == "a":
            self._finish_anchor()
            href = (attributes.get("href") or "").strip()
            if href and len(href) <= 2048:
                priority = "download" in attributes or bool(re.search(
                    r"\.(?:csv|tsv|json|xml|pdf|xlsx?)(?:$|[?#])|[?&](?:format|output|type)=(?:csv|tsv|json|xml|pdf)(?:&|$)|/(?:api|downloads?)(?:[/?.#]|$)",
                    href, re.IGNORECASE,
                ))
                self.anchor = {"href": href, "text": "", "fallback": (attributes.get("title") or attributes.get("aria-label") or "")[:160], "priority": priority}
        elif tag == "img" and self.anchor is not None:
            self.handle_data(attributes.get("alt") or "")

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self.script is not None:
            for url in _swagger_spec_urls("".join(self.script)):
                self._remember(url, "OpenAPI specification", True)
            self.script = None
        if tag == "a":
            self._finish_anchor()
        if tag in {"script", "style", "noscript", "svg"}:
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data: str) -> None:
        if self.script is not None:
            self.script_chars += len(data)
            if self.script_chars <= 65536:
                self.script.append(data)
            else:
                self.script = None
        if self.anchor is not None and not self.hidden:
            remaining = 160 - len(self.anchor["text"])
            self.anchor["text"] += data[:remaining]

    def _finish_anchor(self) -> None:
        if self.anchor is None:
            return
        anchor, self.anchor = self.anchor, None
        self._remember(anchor["href"], " ".join((anchor["text"] or anchor["fallback"]).split())[:160], anchor["priority"])

    def _remember(self, href: str, title: str, priority: bool) -> None:
        url = _outgoing_url(self.base or self.url, href)
        if url is None:
            return
        pool = self.downloads if priority else self.navigation
        if len(pool) < 32 or url in pool:
            pool.setdefault(url, (href, title[:160]))

    def links(self) -> list[dict[str, str]]:
        self._finish_anchor()
        output: dict[str, dict[str, str]] = {}
        for href, title in (*self.downloads.values(), *self.navigation.values()):
            url = _outgoing_url(self.base or self.url, href)
            if url is not None:
                output.setdefault(url, {"title": title or url[:160], "url": url})
            if len(output) == 32:
                break
        return list(output.values())


def _decode_page(content: bytes, encoding: str) -> str:
    try:
        return content.decode(encoding, errors="replace")
    except LookupError:
        return content.decode("utf-8", errors="replace")


def _is_html(content_type: str, text: str) -> bool:
    mime = content_type.partition(";")[0].strip().lower()
    return mime in {"text/html", "application/xhtml+xml"} or (
        mime in {"", "text/plain", "application/octet-stream", "application/save-to-disk"}
        and bool(re.match(r"\s*(?:<!doctype\s+html\b|<html\b)", text.lstrip("\ufeff")[:1000], re.IGNORECASE))
    )


def _page_links(content: bytes, content_type: str, encoding: str, url: str) -> list[dict[str, str]] | None:
    if content.startswith(b"%PDF-"):
        return None
    text = _decode_page(content, encoding)
    if not _is_html(content_type, text):
        return None
    parser = _PageLinks(url)
    parser.feed(text)
    parser.close()
    return parser.links()


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
    # urlsplit strips some controls; reject them before parsing, and report a
    # malformed request as an ordinary tool observation rather than crashing
    # the whole evaluation writer in http.client._validate_path.
    if any(ord(character) <= 32 or ord(character) == 127 for character in url):
        raise ValueError("URL contains spaces or control characters. Use an observed link with spaces percent-encoded.")
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
    except urllib.error.HTTPError as exc:
        raise _HTTPRetrievalError(exc.code) from None
    except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException, UnicodeError) as exc:
        raise _HTTPRetrievalError(None, cause_type=type(exc).__name__) from None


class _PublicRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        _public_url(new_url)
        return super().redirect_request(request, fp, code, message, headers, new_url)


def _download_public(url: str) -> tuple[bytes, str, str]:
    _public_url(url)
    opener = urllib.request.build_opener(_PublicRedirects())
    request = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/pdf,text/plain;q=0.9,*/*;q=0.5",
        "Accept-Language": "en-US,en;q=0.9,zh;q=0.8",
    })
    try:
        with opener.open(request, timeout=float(os.environ.get("DCC_WEB_TIMEOUT", "30"))) as response:
            content = response.read(5_000_001)
            if len(content) > 5_000_000:
                raise ValueError("Source download exceeds the 5000000-byte limit; use a smaller source or export. No truncated source was published.")
            return content, response.headers.get_content_type(), response.headers.get_content_charset() or "utf-8"
    except urllib.error.HTTPError as exc:
        # HTTPError also subclasses URLError; preserve only the safe status.
        raise _HTTPRetrievalError(exc.code) from None
    except http.client.InvalidURL:
        raise ValueError("Malformed HTTP URL. Use an observed link with spaces percent-encoded.") from None
    except UnicodeError:
        raise ValueError("URL could not be encoded for HTTP. Use a correctly percent-encoded observed link.") from None
    except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
        raise _HTTPRetrievalError(None, cause_type=type(exc).__name__) from None


def _xlsx_page(content: bytes, *, sections: list[dict[str, Any]] | None = None) -> str | None:
    """Read bounded OpenXML cell values; never extract files or follow links."""
    if len(content) > 5_000_000:
        raise ValueError("XLSX download exceeds the byte limit")
    # Check the directory count before ZipFile allocates an object per member.
    end = content.rfind(b"PK\x05\x06", max(0, len(content) - 65557))
    if end < 0 or end + 22 > len(content):
        raise ValueError("could not read this XLSX archive")
    if int.from_bytes(content[end + 10:end + 12], "little") > _XLSX_MAX_MEMBERS:
        raise ValueError("XLSX archive exceeds the member limit")
    directory_size = int.from_bytes(content[end + 12:end + 16], "little")
    position = int.from_bytes(content[end + 16:end + 20], "little")
    if position + directory_size != end:
        raise ValueError("invalid or unsupported XLSX archive directory")
    member_count = 0
    while position < end:
        if position + 46 > end or content[position:position + 4] != b"PK\x01\x02":
            raise ValueError("invalid XLSX archive directory")
        member_count += 1
        if member_count > _XLSX_MAX_MEMBERS:
            raise ValueError("XLSX archive exceeds the member limit")
        position += 46 + sum(int.from_bytes(content[position + offset:position + offset + 2], "little") for offset in (28, 30, 32))
    if position != end or member_count != int.from_bytes(content[end + 10:end + 12], "little"):
        raise ValueError("invalid XLSX archive member count")
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            members = archive.infolist()
            names = {member.filename for member in members}
            if "xl/workbook.xml" not in names:
                return None
            if len(members) > _XLSX_MAX_MEMBERS or len(names) != len(members):
                raise ValueError("XLSX archive has too many or duplicate members")
            if (any(member.file_size > _XLSX_MAX_MEMBER_BYTES or member.flag_bits & 1 for member in members)
                    or sum(member.file_size for member in members) > _XLSX_MAX_TOTAL_BYTES):
                raise ValueError("XLSX archive exceeds the uncompressed byte limits or is encrypted")

            def xml(path: str):
                raw = archive.read(path)
                if len(raw) > _XLSX_MAX_MEMBER_BYTES or re.search(br"<!\s*(?:DOCTYPE|ENTITY)\b", raw.replace(b"\x00", b""), re.IGNORECASE):
                    raise ValueError("XLSX XML exceeds limits or declares entities")
                return ET.fromstring(raw)

            def local(element):
                return element.tag.rsplit("}", 1)[-1]

            def children(element, name):
                return [child for child in element if local(child) == name]

            def string(element):
                # Phonetic annotations are not part of the displayed value.
                return "".join((part.text or "") for child in element
                               for part in ([child] if local(child) == "t" else children(child, "t") if local(child) == "r" else []))

            workbook = xml("xl/workbook.xml")
            if local(workbook) != "workbook":
                raise ValueError("invalid XLSX workbook metadata")
            sheet_groups = children(workbook, "sheets")
            sheets = children(sheet_groups[0], "sheet") if len(sheet_groups) == 1 else []
            if not sheets or len(sheets) > _XLSX_MAX_SHEETS:
                raise ValueError("XLSX workbook has no sheets or exceeds the sheet limit")
            relationships = {}
            for relation in xml("xl/_rels/workbook.xml.rels"):
                if local(relation) != "Relationship":
                    continue
                key = relation.get("Id")
                if key in relationships:
                    raise ValueError("duplicate XLSX workbook relationship")
                relationships[key] = relation
            shared = []
            if "xl/sharedStrings.xml" in names:
                root = xml("xl/sharedStrings.xml")
                if local(root) != "sst":
                    raise ValueError("invalid XLSX shared strings")
                shared = [string(item) for item in children(root, "si")]
                if len(shared) > _XLSX_MAX_CELLS:
                    raise ValueError("XLSX shared strings exceed the cell limit")
            output = io.StringIO()
            output.write("XLSX workbook; formula cells contain cached values only.\n")
            writer = csv.writer(output, lineterminator="\n")
            cell_count = 0
            navigation = []
            preview_budget = _XLSX_NAV_TOTAL_PREVIEW_CHARS
            for sheet in sheets:
                relation_id = next((value for key, value in sheet.attrib.items() if key.rsplit("}", 1)[-1] == "id"), None)
                relation = relationships.get(relation_id)
                if relation is None or not relation.get("Type", "").endswith("/worksheet"):
                    raise ValueError("XLSX sheet has no worksheet relationship")
                target = relation.get("Target", "")
                if (relation.get("TargetMode", "").lower() == "external" or not target or "\\" in target
                        or urllib.parse.urlsplit(target).scheme or "\x00" in target):
                    raise ValueError("external or invalid XLSX worksheet relationship")
                path = posixpath.normpath(target.lstrip("/") if target.startswith("/") else posixpath.join("xl", target))
                if not path.startswith("xl/") or path not in names:
                    raise ValueError("invalid XLSX worksheet path")
                worksheet = xml(path)
                if local(worksheet) != "worksheet":
                    raise ValueError("invalid XLSX worksheet XML")
                name = sheet.get("name", "")
                section = {"name": name[:_XLSX_NAV_NAME_CHARS], "start": output.tell(),
                           "preview_row": None, "preview": ""}
                if len(name) > _XLSX_NAV_NAME_CHARS:
                    section["name_truncated"] = True
                output.write("\nWorksheet: " + json.dumps(name, ensure_ascii=False) + "\n")
                groups = children(worksheet, "sheetData")
                if len(groups) > 1:
                    raise ValueError("invalid XLSX sheet data")
                previous_row = 0
                for row in children(groups[0], "row") if groups else []:
                    row_number = int(row.get("r", previous_row + 1))
                    if row_number <= previous_row or row_number > _XLSX_MAX_ROWS:
                        raise ValueError("XLSX row index is invalid or exceeds the row limit")
                    values = []
                    for cell in children(row, "c"):
                        cell_count += 1
                        if cell_count > _XLSX_MAX_CELLS:
                            raise ValueError("XLSX workbook exceeds the cell limit")
                        coordinate = cell.get("r")
                        column = len(values) + 1
                        if coordinate is not None:
                            match = re.fullmatch(r"([A-Za-z]{1,3})([1-9][0-9]*)", coordinate)
                            if match is None or int(match[2]) != row_number:
                                raise ValueError("invalid XLSX cell coordinate")
                            column = 0
                            for letter in match[1].upper():
                                column = column * 26 + ord(letter) - ord("A") + 1
                        if column <= len(values) or column > _XLSX_MAX_COLUMNS:
                            raise ValueError("XLSX column index is invalid or exceeds the column limit")
                        nodes = children(cell, "v")
                        value = nodes[0].text or "" if nodes else ""
                        kind = cell.get("t", "n")
                        if kind == "s" and value:
                            index = int(value)
                            if index < 0 or index >= len(shared):
                                raise ValueError("invalid XLSX shared string reference")
                            value = shared[index]
                        elif kind == "inlineStr":
                            inline = children(cell, "is")
                            value = string(inline[0]) if inline else ""
                        elif kind == "b" and value:
                            if value not in {"0", "1"}:
                                raise ValueError("invalid XLSX boolean cell")
                            value = "TRUE" if value == "1" else "FALSE"
                        values.extend([""] * (column - len(values) - 1))
                        values.append(value)
                    for _ in range(row_number - previous_row - 1):
                        writer.writerow([])
                    row_start = output.tell()
                    writer.writerow(values)
                    if section["preview_row"] is None and any(value.strip() for value in values):
                        # A literal first nonempty row, not an inferred header.
                        # Record positions while writing; sheet names and cells
                        # may themselves contain marker-looking text/newlines.
                        row_text = output.getvalue()[row_start:-1]
                        limit = min(_XLSX_NAV_PREVIEW_CHARS, preview_budget)
                        section["preview_row"] = row_number
                        section["preview"] = row_text[:limit]
                        if len(row_text) > limit:
                            section["preview_truncated"] = True
                        preview_budget -= len(section["preview"])
                    previous_row = row_number
                    if output.tell() > _XLSX_MAX_TEXT_CHARS:
                        raise ValueError("XLSX extracted text exceeds the character limit")
                section["end"] = output.tell()
                navigation.append(section)
            if output.tell() > _XLSX_MAX_TEXT_CHARS:
                raise ValueError("XLSX extracted text exceeds the character limit")
            if sections is not None:
                sections.extend(navigation)
            return output.getvalue()
    except (zipfile.BadZipFile, KeyError, ET.ParseError, RuntimeError, OSError, NotImplementedError) as exc:
        raise ValueError("could not read this XLSX workbook") from exc


def extract_page(content: bytes, content_type: str, encoding: str = "utf-8", *,
                 xlsx_sections: list[dict[str, Any]] | None = None,
                 archive_sections: list[dict[str, Any]] | None = None) -> str:
    if (content_type.partition(";")[0].strip().lower() in {_XLSX_MIME, "application/zip", "application/x-zip-compressed"}
            or content.startswith((b"PK\x03\x04", b"PK\x05\x06"))):
        extracted = _xlsx_page(content, sections=xlsx_sections)
        if extracted is not None:
            return extracted
        from .web_archive import text_zip_page

        return text_zip_page(content, sections=archive_sections)
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
    text = _decode_page(content, encoding)
    if _is_html(content_type, text):
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
    if content_type.startswith("text/") or content_type in {"application/json", "application/xml"} or content_type.endswith(("+json", "+xml")):
        return text
    if content_type in {"application/octet-stream", "application/save-to-disk"}:
        # Public data portals sometimes label their XML/JSON downloads as a
        # generic attachment. Validate the format before exposing it as text;
        # decoding an arbitrary binary download is not a successful page read.
        stripped = text.lstrip("\ufeff \t\r\n")
        if stripped.startswith("<?xml") and "<!DOCTYPE" not in stripped:
            try:
                ET.fromstring(stripped)
            except ET.ParseError:
                pass
            else:
                return text
        elif stripped.startswith(("{", "[")):
            try:
                json.loads(stripped)
            except ValueError:
                pass
            else:
                return text
    raise ValueError(f"unsupported page content type: {content_type}")


class WebTools:
    """Search public indexes or an operator-configured web-search API."""

    def __init__(self, *, backend: str | None = None, max_page_chars: int = 14000, browser_reader=None, web_cached_rows=False) -> None:
        from .web_rows import web_cached_rows_configuration

        self.web_cached_rows = web_cached_rows_configuration(web_cached_rows)["enabled"]
        self.backend = backend or os.environ.get("DCC_SEARCH_BACKEND", "public_web")
        self.max_page_chars = max_page_chars
        self.sources: dict[str, dict[str, str]] = {}
        self.pages: dict[str, str] = {}
        self.page_links: dict[str, list[dict[str, str]]] = {}
        self.page_worksheets: dict[str, list[dict[str, Any]]] = {}
        self.page_archives: dict[str, list[dict[str, Any]]] = {}
        self.browser_reader = browser_reader
        self.page_browser: dict[str, dict[str, Any]] = {}
        self._permanent_http_failures: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._read_windows: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._semaphore = asyncio.Semaphore(8)

    async def web_search(self, query: str, *, count: int = 8) -> dict[str, Any]:
        query = str(query).strip()[:1000]
        if not query:
            return {"error": "query must be nonempty"}
        from .search_query import prepare_search_query

        original_query = query
        query, scope = prepare_search_query(query)
        count = max(1, min(int(count), 10))
        async with self._semaphore:
            if self.backend == "public_web":
                from .search_provider import public_web_search

                information = await asyncio.to_thread(public_web_search, query, count, lambda: self._search(query, count, backend="duckduckgo_lite"))
            elif self.backend in {"auto", "duckduckgo_wikipedia"}:
                from .search_provider import search_with_fallback

                information = await asyncio.to_thread(search_with_fallback, query, count, lambda: self._search(query, count, backend="duckduckgo_lite"))
            elif self.backend == "wikipedia":
                from .search_provider import wikipedia_search

                hits = await asyncio.to_thread(wikipedia_search, query, count)
                information = {"results": hits, "backend": "wikipedia", "search_scope": "encyclopedia", "fallback_reason": None, "coverage_note": "Encyclopedia search only; this is not a general web index."}
            else:
                hits = await asyncio.to_thread(self._search, query, count)
                information = {"results": hits, "backend": self.backend, "search_scope": "web", "fallback_reason": None}
        if scope is not None:
            information["results"] = scope.filter(information["results"])
            information["query_scope"] = scope.metadata()
        if query != original_query:
            information["original_query"] = original_query
        for hit in information["results"]:
            self.sources[hit["url"]] = hit
        return {"query": query, "configured_backend": self.backend, **information}

    def _search(self, query: str, count: int, *, backend: str | None = None) -> list[dict[str, str]]:
        backend = backend or self.backend
        if backend == "duckduckgo_lite":
            raw = _request("https://lite.duckduckgo.com/lite/?" + urllib.parse.urlencode({"q": query}))
            if "anomaly-modal" in raw or "challenge-form" in raw:
                raise InfrastructureError("DuckDuckGo challenged this request; use a configured search API")
            parser = _DuckDuckGoResults()
            parser.feed(raw)
            hits = parser.hits
        elif backend in {"seznam", "baidu_mobile"}:
            from .search_provider import public_html_search

            hits = public_html_search(query, count, backend)
        elif backend == "serper":
            key = os.environ.get("SERPER_API_KEY")
            if not key:
                raise InfrastructureError("SERPER_API_KEY is required for serper")
            data = json.loads(_request("https://google.serper.dev/search", payload={"q": query, "num": count}, headers={"X-API-KEY": key}))
            hits = [{"title": item.get("title", ""), "url": item["link"], "snippet": item.get("snippet", "")} for item in data.get("organic", [])]
        elif backend == "brave":
            key = os.environ.get("BRAVE_SEARCH_API_KEY")
            if not key:
                raise InfrastructureError("BRAVE_SEARCH_API_KEY is required for brave")
            url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode({"q": query, "count": count})
            data = json.loads(_request(url, headers={"X-Subscription-Token": key}))
            hits = [{"title": item.get("title", ""), "url": item["url"], "snippet": item.get("description", "")} for item in data.get("web", {}).get("results", [])]
        elif backend == "searxng":
            endpoint = os.environ.get("DCC_SEARCH_URL", "").rstrip("/")
            if not endpoint:
                raise InfrastructureError("DCC_SEARCH_URL is required for searxng")
            data = json.loads(_request(endpoint + "/search?" + urllib.parse.urlencode({"q": query, "format": "json"})))
            hits = [{"title": item.get("title", ""), "url": item["url"], "snippet": item.get("content", "")} for item in data.get("results", [])]
        elif backend == "bing_rss":
            raw = _request("https://www.bing.com/search?" + urllib.parse.urlencode({"q": query, "format": "rss", "count": count}))
            try:
                root = ET.fromstring(raw)
            except ET.ParseError as exc:
                raise InfrastructureError("search endpoint returned invalid RSS") from exc
            hits = [{"title": item.findtext("title", ""), "url": item.findtext("link", ""), "snippet": visible_text(item.findtext("description", ""))} for item in root.findall("./channel/item")]
        else:
            raise ValueError(f"unknown search backend: {backend}")
        return [item for item in hits if item["url"].startswith(("https://", "http://"))][:count]

    def _fetch_cached_rows(self, url, request):
        from .web_rows import select_cached_rows

        try:
            if not self.web_cached_rows:
                raise ValueError("Cached row selection is disabled")
            if url not in self.pages:
                raise ValueError("Source is not in this episode's downloaded cache; fetch the URL normally first")
            view = select_cached_rows(self.pages[url], request, worksheets=self.page_worksheets.get(url, ()),
                                      archives=self.page_archives.get(url, ()))
            candidates = view.pop("records")
            view.update(records=[], returned_records=0, next_after=None, truncated=False)
            result = {"url": url, "row_view": view, "total_characters": len(self.pages[url])}
            # Stage read telemetry so an over-budget or malformed result cannot
            # mark unseen records read. A literal row may occur more than once.
            staged = OrderedDict((key, value.copy()) for key, value in self._read_windows.items())
            limit = request.get("limit", 32)
            for index, original in enumerate(candidates):
                if len(view["records"]) >= limit:
                    break
                row = original.copy()
                window = row["text"]
                previous = staged.get(window)
                row["read_progress"] = {"new_text": previous is None,
                                         "previous_identical_reads": previous["count"] if previous else 0}
                if previous is not None:
                    row["read_progress"]["first_read"] = {key: previous[key] for key in ("url", "start", "end")}
                if not view["records"]:
                    result.update(text=window, start=row["start"], end=row["end"])
                view["records"].append(row)
                view.update(returned_records=len(view["records"]),
                            truncated=index + 1 < len(candidates),
                            next_after=row["record_index"] + 1 if index + 1 < len(candidates) else None)
                if len(json.dumps(result, ensure_ascii=False)) > self.max_page_chars:
                    view["records"].pop()
                    if not view["records"]:
                        raise ValueError("One literal record with provenance exceeds the original observation character budget")
                    previous_row = view["records"][-1]
                    view.update(returned_records=len(view["records"]), truncated=True,
                                next_after=previous_row["record_index"] + 1)
                    break
                if previous is not None:
                    previous["count"] += 1
                    staged.move_to_end(window)
                else:
                    staged[window] = {"url": url, "start": row["start"], "end": row["end"], "count": 1}
                    while len(staged) > 128:
                        staged.popitem(last=False)
            if len(json.dumps(result, ensure_ascii=False)) > self.max_page_chars:
                raise ValueError("Selector metadata exceeds the original observation character budget")
            self._read_windows = staged
            if view["records"]:
                self.sources.setdefault(url, {"title": "", "url": url, "snippet": self.pages[url][:300]})
            return result
        except (ValueError, RecursionError) as exc:
            result = {"url": url, "error": str(exc)}
            if len(json.dumps(result, ensure_ascii=False)) > self.max_page_chars:
                result = {"error": "Cached row selection failed within the observation budget"}
            return result

    async def fetch_url(self, url: str, start: int = 0, select_rows=None) -> dict[str, Any]:
        if not isinstance(url, str):
            return {"error": "url must be an HTTP(S) string"}
        url = url.strip()
        try:
            url = urllib.parse.urldefrag(url)[0]
        except ValueError:
            return {"url": url, "error": "url is malformed"}
        if select_rows is not None:
            if type(start) is not int or start != 0:
                return {"error": "select_rows uses its own after cursor; do not combine it with a text start offset"}
            return self._fetch_cached_rows(url, select_rows)
        cached_failure = self._permanent_http_failures.get(url)
        if cached_failure is not None:
            self._permanent_http_failures.move_to_end(url)
            return {**cached_failure, "cached": True}
        if url not in self.pages:
            try:
                from .search_provider import wikipedia_page

                article = await asyncio.to_thread(wikipedia_page, url)
                links = None
                worksheets = []
                archive_members = []
                if article is not None:
                    text = article
                else:
                    async with self._semaphore:
                        raw, content_type, encoding = await asyncio.to_thread(_download_public, url)
                    text = await asyncio.to_thread(extract_page, raw, content_type, encoding, xlsx_sections=worksheets, archive_sections=archive_members)
                    links = await asyncio.to_thread(_page_links, raw, content_type, encoding, url)
                    # Rendering is explicitly injected by the evaluation owner.
                    # HTTP failures and access challenges never trigger it. Keep
                    # the successful HTTP source if the optional renderer fails.
                    html = content_type.split(";", 1)[0].strip().lower() in {"text/html", "application/xhtml+xml"}
                    challenge = len(text) < 4000 and re.search(
                        r"verify (?:that )?you are human|verify (?:that )?you(?: are|'re|’re) not a robot|enable javascript and cookies|checking your browser|^\s*(?:access denied|just a moment|百度安全验证|请输入验证码)(?:\b|\s|[.!：:])",
                        text[:1200], re.IGNORECASE,
                    )
                    if self.browser_reader is not None and html and not challenge:
                        metadata = {"method": "http_fallback"}
                        try:
                            rendered = await self.browser_reader.read(url)
                            visible = rendered.get("text") if isinstance(rendered, dict) else None
                            if (not isinstance(rendered, dict) or rendered.get("error")
                                    or not isinstance(visible, str) or not visible.strip()):
                                metadata["note"] = "Browser rendering supplied no usable visible text; retained the successful HTTP read."
                            else:
                                metadata = {
                                    "method": "visible_dom",
                                    "browser_reader_version": rendered["browser_runtime"]["version"],
                                    "source_text_truncated": rendered["truncated"],
                                    "source_total_text_characters": rendered["total_characters"],
                                    "final_url": rendered["final_url"],
                                    "outgoing_links_truncated": rendered.get("outgoing_links_truncated", False),
                                    "link_candidates_truncated": rendered.get("link_candidates_truncated", False),
                                    "observed_json_urls": rendered.get("observed_json_urls", []),
                                    "observed_json_urls_truncated": rendered.get("observed_json_urls_truncated", False),
                                    "network": rendered["browser_runtime"].get("network", {}),
                                    "observed_json_note": "These are navigation URLs from actual browser GET responses. Their JSON contents are not included in this observation. Fetch a URL explicitly before using its contents or citing it.",
                                }
                                text, links = visible, rendered.get("outgoing_links", [])
                        except Exception as exc:
                            metadata.update(error_type=type(exc).__name__,
                                            note="Browser rendering failed; retained the successful HTTP read.")
                        self.page_browser[url] = metadata
                # PDF extractors can emit unpaired UTF-16 surrogates in a
                # Python str. Rust tokenizers reject those strings. Clean at
                # this external-page boundary before caching, slicing, or
                # exposing text so observations and continuation offsets agree.
                text = text.encode("utf-8", errors="replace").decode("utf-8")
                if not text.strip():
                    return {"url": url, "error": "Source returned no readable text. Fetch another search result or a different format of this source."}
                # A 200 status can still be an access challenge. Do not count
                # that page as factual evidence or encourage citing it.
                if len(text) < 4000 and re.search(
                    r"verify (?:that )?you are human|verify (?:that )?you(?: are|'re|’re) not a robot|enable javascript and cookies|checking your browser|^\s*(?:access denied|just a moment|百度安全验证|请输入验证码)(?:\b|\s|[.!：:])",
                    text[:1200], re.IGNORECASE,
                ):
                    return {"url": url, "error": "Source returned an access challenge, not article text. Fetch a different public source."}
                self.pages[url] = text
                if links is not None:
                    self.page_links[url] = links
                if worksheets:
                    self.page_worksheets[url] = worksheets
                if archive_members:
                    self.page_archives[url] = archive_members
            except _HTTPRetrievalError as exc:
                observation = {"url": url, "error": str(exc), "http_status": exc.http_status,
                               "retryable": exc.retryable, "cached": False}
                if exc.http_status in {404, 410}:
                    # WebTools is episode-local. Keep exact URLs (including
                    # query strings), and never turn failed reads into evidence.
                    self._permanent_http_failures[url] = observation.copy()
                    while len(self._permanent_http_failures) > _HTTP_FAILURE_CACHE_SIZE:
                        self._permanent_http_failures.popitem(last=False)
                return observation
            except (InfrastructureError, ValueError, socket.gaierror) as exc:
                # Dead/blocked pages are normal web-agent observations. An
                # unavailable search backend, in contrast, aborts the episode.
                return {"url": url, "error": str(exc)}
        text = self.pages[url]
        start = max(0, int(start))
        if start >= len(text):
            return {"url": url, "error": "Read offset is beyond this page. Use start=0 to reread it or fetch another source.", "total_characters": len(text)}
        end = min(len(text), start + self.max_page_chars)
        result = {"url": url, "text": text[start:end], "start": start, "end": end, "total_characters": len(text), "truncated": end < len(text), "next_start": end if end < len(text) else None}
        if url in self.page_browser:
            result["browser_read"] = self.page_browser[url]
        if url in self.page_links:
            result["outgoing_links"] = self.page_links[url]
            result["outgoing_links_note"] = (
                "Visible links from the rendered page, subject to browser_read truncation flags. Linked pages have not been read; fetch them before using their contents or citing them."
                if self.page_browser.get(url, {}).get("method") == "visible_dom" else
                "Links from the entire HTML page, including beyond this text window. Linked pages have not been read; fetch them before using their contents or citing them."
            )
        if url in self.page_worksheets:
            result["worksheets"] = self.page_worksheets[url]
            result["worksheets_note"] = "Source-wide XLSX navigation, including beyond this text window. Each start/end bounds a worksheet in the extracted text; preview is only its first nonempty CSV row, not an inferred header. Fetch this URL with start set to a worksheet's start to read its contents before using them as evidence. Names/previews may be explicitly marked truncated."
        if url in self.page_archives:
            result["archive_members"] = self.page_archives[url]
            result["archive_members_note"] = (
                "Source-wide bounded ZIP directory, including beyond this text window. Readable text members "
                "have start/end character offsets in this source; content_start follows the member label. Byte sizes come from the ZIP directory. "
                "Preview is a literal bounded prefix, only for navigation. Fetch this same URL with start set "
                "to a member's start to read it before using its contents as evidence. Directory and unsupported "
                "members have no text offsets and have not been read. Names/previews may be marked truncated."
            )
        window = result["text"]
        previous = self._read_windows.get(window)
        if previous is not None:
            result["read_progress"] = {
                "new_text": False, "previous_identical_reads": previous["count"],
                "first_read": {key: previous[key] for key in ("url", "start", "end")},
            }
            result["read_progress_note"] = (
                "This exact text window was already read; this call adds no source text. "
                "Use the saved evidence to answer. If information is still missing, change the web_search "
                "query or follow actual document, API, export, or outgoing links. Use next_start or a "
                "worksheet or archive-member offset to read an unseen section. Changing URL parameters can return the same page."
            )
            previous["count"] += 1
            self._read_windows.move_to_end(window)
        else:
            self._read_windows[window] = {"url": url, "start": start, "end": end, "count": 1}
            while len(self._read_windows) > 128:
                self._read_windows.popitem(last=False)
        self.sources.setdefault(url, {"title": "", "url": url, "snippet": text[:300]})
        return result

    def citation_validity(self, answer: Any) -> float:
        if not isinstance(answer, list) or not answer:
            return 0.0
        citations = [url for row in answer if isinstance(row, dict) for url in row.get("citations", []) if isinstance(url, str)]
        def observed(url: str) -> bool:
            try:
                return urllib.parse.urldefrag(url.strip())[0] in self.sources
            except ValueError:
                return False
        return sum(observed(url) for url in citations) / len(citations) if citations else 0.0
