"""Generic downloads need bounded validated table syntax before becoming evidence."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import sys
from email.message import Message
from types import SimpleNamespace

import pytest

from slime_plugins.dcc import search_provider, tools
from slime_plugins.dcc.episode import _WebEvidence

SOURCE = "https://example.org/download?id=public-table"
GENERIC_TYPES = ("application/octet-stream", "application/save-to-disk")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(search_provider, "wikipedia_page", lambda _: None)
    monkeypatch.setitem(sys.modules, "trafilatura", SimpleNamespace(extract=lambda *args, **kwargs: None))


@pytest.mark.parametrize("content_type", GENERIC_TYPES)
def test_generic_csv_regression_preserves_public_table(content_type):
    body = b"name,country\nAlpha,Canada\n"
    assert tools.extract_page(body, content_type) == body.decode()


@pytest.mark.parametrize("body", [
    b"name,country\nAlpha,Canada\n",
    b"name\tcountry\nAlpha\tCanada\n",
    b"name,2023,2024\nAlpha,12,14\n",
    b"\xef\xbb\xbfname,country\r\nAlpha,Canada\r\n",
    b"name,country\rAlpha,Canada\r",
    b"name,country\nAlpha,Canada",
    b'"name","country"\n"Alpha","Canada"\n',
    b'name,note\nAlpha,"has a comma, and a ""quote"""\n',
    b'name,note\nAlpha,"a first line\nand another line"\n',
    b"name,optional,value\nAlpha,,12\nBeta,,13\n",
    "名称,国家\n阿尔法,加拿大\n".encode(),
    b"name,note\nAlpha,\n",
    b'name,note\nAlpha,"a\ttab"\n',
    b"name,code\tstatus\nAlpha,12\tactive\n",
])
@pytest.mark.parametrize("content_type", GENERIC_TYPES)
def test_valid_utf8_tables_preserve_original_text(body, content_type):
    assert tools.extract_page(body, content_type, "unsupported-charset") == body.decode("utf-8")


@pytest.mark.parametrize("body", [
    b"",
    b"name,country",
    b"name,country\n",
    b"Only ordinary text.\nAnother plain sentence.\n",
    b"Hello, reader.\nThis is, ordinary prose.\n",
    b"Plain text\nMore plain text\n",
    b"name;country\nAlpha;Canada\n",
    b"name|country\nAlpha|Canada\n",
    b"name\nAlpha\n",
    b"name,country\nAlpha,Canada,extra\n",
    b"name,country\nAlpha\n",
    b"name,country\n\nAlpha,Canada\n",
    b",country\nAlpha,Canada\n",
    b" \t,country\nAlpha,Canada\n",
    b"name,name\nAlpha,Canada\n",
    b"name, NAME \nAlpha,Canada\n",
    b"2023,2024\n12,14\n",
    b'name,"country\nlabel"\nAlpha,Canada\n',
    b'name,"\ncountry"\nAlpha,Canada\n',
    b"name,country\n,\n",
    b'name,country\n"Alpha,Canada\n',
    b'name,country\nAl"pha,Canada\n',
    b'name,country\n"Alpha"x,Canada\n',
    b'name,country\n"Alpha" ,Canada\n',
    b'name,country\n"Alpha",Canada\n"Beta"x,France\n',
    b"name,country\nAlpha,Canad\xff\n",
    b"\xff\xfeN\x00a\x00m\x00e\x00",
    b"name,country\nAlpha,Canad\x00a\n",
    b"name,country\nAlpha,Canad\x01a\n",
    b"name,country\nAlpha,Canad\x0ba\n",
    b"name,country\nAlpha,Canad\x0ca\n",
    b"name,country\nAlpha,Canad\x1fa\n",
    b"name,country\nAlpha,Canad\x7fa\n",
    "name,country\nAlpha,Canad\u0085a\n".encode(),
    "name,country\nAlpha,Canad\u009fa\n".encode(),
    b"\x89PNG\r\n\x1a\nname,country\nAlpha,Canada\n",
    b"name,country\nAlpha,Canada\n\n",
])
@pytest.mark.parametrize("content_type", GENERIC_TYPES)
def test_unvalidated_downloads_are_rejected(body, content_type):
    with pytest.raises(ValueError, match="unsupported page content type"):
        tools.extract_page(body, content_type, "latin-1")


def table(rows):
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(rows)
    return output.getvalue().encode()


@pytest.mark.parametrize("limit,body,accepted", [
    ("_DELIMITED_MAX_BYTES", b"name,value\nAlpha,12\n", 20),
    ("_DELIMITED_MAX_TEXT_CHARS", b"name,value\nAlpha,12\n", 20),
    ("_DELIMITED_MAX_ROWS", b"name,value\nAlpha,12\nBeta,13\n", 3),
    ("_DELIMITED_MAX_COLUMNS", b"name,2023,2024\nAlpha,12,13\n", 3),
    ("_DELIMITED_MAX_CELLS", b"name,2023,2024\nAlpha,12,13\n", 6),
    ("_DELIMITED_MAX_FIELD_CHARS", b"name,val\nAlpha,12345\n", 5),
    ("_DELIMITED_MAX_HEADER_CHARS", b"name,value\nAlpha,12\n", 5),
])
def test_work_limits_accept_boundary_and_reject_overflow(monkeypatch, limit, body, accepted):
    if limit.endswith(("BYTES", "TEXT_CHARS")):
        accepted = len(body)
    monkeypatch.setattr(tools, limit, accepted)
    assert tools.extract_page(body, GENERIC_TYPES[0]) == body.decode()
    monkeypatch.setattr(tools, limit, accepted - 1)
    with pytest.raises(ValueError, match="unsupported"):
        tools.extract_page(body, GENERIC_TYPES[0])


@pytest.mark.parametrize("body", [
    b"name,value\nAlpha,12\n" + b"x" * 5_000_000,
    b"name,value\nAlpha," + b"x" * 1_000_000,
    table([["name", "value"], *([["Alpha", 12]] * 10_000)]),
    table([["name", *map(str, range(256))], ["Alpha", *(["1"] * 256)]]),
    table([["name", *map(str, range(99))], *([["Alpha", *(["1"] * 99)]] * 1000)]),
    table([["name", "value"], ["Alpha", "x" * 32_769]]),
    table([["name", "x" * 129], ["Alpha", "12"]]),
    table([["name", "value" + " " * 124], ["Alpha", "12"]]),
])
def test_production_work_bounds_reject_without_partial_source(body):
    with pytest.raises(ValueError, match="unsupported"):
        tools.extract_page(body, GENERIC_TYPES[0])


def test_quoted_field_limit_counts_unescaped_chars_and_embedded_newlines(monkeypatch):
    monkeypatch.setattr(tools, "_DELIMITED_MAX_FIELD_CHARS", 5)
    assert tools.extract_page(b'name,note\nAlpha,"a""b\nc"\n', GENERIC_TYPES[0])
    with pytest.raises(ValueError, match="unsupported"):
        tools.extract_page(b'name,note\nAlpha,"a""b\ncd"\n', GENERIC_TYPES[0])


def test_sniffing_does_not_mutate_process_global_csv_limit():
    before = csv.field_size_limit()
    assert tools.extract_page(b"name,country\nAlpha,Canada\n", GENERIC_TYPES[0])
    assert csv.field_size_limit() == before


@pytest.mark.parametrize("content_type,text", [
    ("application/octet-stream", '{"name":"Alpha","value":12}'),
    ("application/save-to-disk", '[{"name":"Alpha","value":12}]'),
    ("application/octet-stream", '<?xml version="1.0"?><data><name>Alpha</name></data>'),
    ("application/save-to-disk", '<?xml version="1.0"?><data><name>Alpha</name></data>'),
    ("text/csv", "This text/csv path keeps its prior permissive behavior."),
    ("text/plain", "Plain text is already an explicitly supported format."),
    ("application/json", '{"name":"Alpha"}'),
    ("application/xml", "<data>Alpha</data>"),
])
def test_existing_explicit_text_xml_json_paths_are_unchanged(content_type, text):
    assert tools.extract_page(text.encode(), content_type) == text


@pytest.mark.parametrize("content_type", GENERIC_TYPES)
def test_existing_generic_html_sniff_is_unchanged(content_type):
    body = b"<html><body><p>Public page.</p><a href='table.csv'>Table</a></body></html>"
    assert tools.extract_page(body, content_type) == "Public page. Table"


def test_existing_pdf_magic_dispatch_is_unchanged(monkeypatch):
    pages = [SimpleNamespace(extract_text=lambda: "PDF page")]
    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=lambda _: SimpleNamespace(pages=pages)))
    assert tools.extract_page(b"%PDF-synthetic-dispatch", GENERIC_TYPES[0]) == "PDF page"


def test_extract_without_http_still_enforces_new_sniff_byte_bound(monkeypatch):
    monkeypatch.setattr(tools, "_DELIMITED_MAX_BYTES", 1)
    with pytest.raises(ValueError, match="unsupported"):
        tools.extract_page(b"name,country\nAlpha,Canada\n", GENERIC_TYPES[0])


@pytest.mark.parametrize("url", [SOURCE, "https://example.org/misleading.csv", "https://example.org/no-suffix"])
@pytest.mark.parametrize("content_type", GENERIC_TYPES)
def test_real_async_fetch_paginates_caches_and_publishes_only_successful_reads(monkeypatch, url, content_type):
    body = b"name,country\nAlpha,Canada\nBeta,France\n"
    calls = []

    def download(requested):
        calls.append(requested)
        return body, content_type, "latin-1"

    monkeypatch.setattr(tools, "_download_public", download)
    web, evidence = tools.WebTools(max_page_chars=11), _WebEvidence()
    assert web.citation_validity([{"citations": [url]}]) == 0

    async def scenario():
        parts = []
        start = 0
        while True:
            result = await web.fetch_url(url + "#ignored", start=start)
            evidence.observe("fetch_url", result)
            assert "error" not in result
            assert result["start"] == start
            assert result["end"] == start + len(result["text"])
            assert result["total_characters"] == len(body)
            for key in ("outgoing_links", "outgoing_links_note", "worksheets", "archive_members"):
                assert key not in result
            parts.append(result["text"])
            if result["next_start"] is None:
                assert not result["truncated"]
                break
            assert result["truncated"]
            start = result["next_start"]
        assert "".join(parts) == body.decode()
        first_again = await web.fetch_url(url)
        assert first_again["text"] == body.decode()[:11]

    asyncio.run(scenario())
    assert calls == [url]
    assert web.pages == {url: body.decode()}
    assert url in web.sources and url in evidence.fetched_sources
    assert web.citation_validity([{"citations": [url]}]) == 1
    assert not web.page_links and not web.page_worksheets and not web.page_archives


@pytest.mark.parametrize("body", [
    b"name,country\nAlpha,Canad\xff\n",
    b"name,country",
    b"name,country\nAlpha,Canada,extra\n",
    b"name,country\nAlpha,Canad\x00a\n",
    b"Some ordinary prose.\nAnother sentence.\n",
])
def test_real_async_fetch_rejection_never_publishes_evidence_even_with_csv_suffix(monkeypatch, body):
    url = "https://example.org/data.csv"
    monkeypatch.setattr(tools, "_download_public", lambda _: (body, GENERIC_TYPES[0], "latin-1"))
    web, evidence = tools.WebTools(), _WebEvidence()
    result = asyncio.run(web.fetch_url(url))
    evidence.observe("fetch_url", result)
    assert "error" in result and "text" not in result
    assert not web.pages and not web.sources and not web.page_links
    assert evidence.fetches == 0 and not evidence.fetched_sources
    assert web.citation_validity([{"citations": [url]}]) == 0


def test_real_async_http_boundary_oversize_does_not_publish_partial_table(monkeypatch):
    body = b"name,country\nAlpha,Canada\n" + b" " * 5_000_000
    response = io.BytesIO(body)
    response.headers = Message()
    response.headers["Content-Type"] = GENERIC_TYPES[0]
    monkeypatch.setattr(tools, "_public_url", lambda url: url)
    monkeypatch.setattr(tools.urllib.request, "build_opener", lambda *args: SimpleNamespace(open=lambda *args, **kwargs: response))
    web, evidence = tools.WebTools(), _WebEvidence()
    result = asyncio.run(web.fetch_url(SOURCE))
    evidence.observe("fetch_url", result)
    assert "exceeds" in result["error"]
    assert "text" not in result and "outgoing_links" not in result
    assert not web.pages and not web.sources
    assert evidence.fetches == 0 and not evidence.fetched_sources


def test_real_async_http_boundary_reads_generic_csv_from_response_headers(monkeypatch):
    body = b"name,country\nAlpha,Canada\n"
    response = io.BytesIO(body)
    response.headers = Message()
    response.headers["Content-Type"] = "application/octet-stream; charset=iso-8859-1"
    calls = []

    def open_request(request, **kwargs):
        calls.append(request.full_url)
        return response

    monkeypatch.setattr(tools, "_public_url", lambda url: url)
    monkeypatch.setattr(tools.urllib.request, "build_opener", lambda *args: SimpleNamespace(open=open_request))
    web = tools.WebTools()
    result = asyncio.run(web.fetch_url(SOURCE))
    assert result["text"] == body.decode() and "outgoing_links" not in result
    assert calls == [SOURCE] and web.pages == {SOURCE: body.decode()}
