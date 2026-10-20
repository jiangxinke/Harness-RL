"""HTML links are useful navigation metadata, never unread source evidence."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import sys
import zipfile
from html import escape
from types import SimpleNamespace

import pytest

from slime_plugins.dcc import Task, search_provider, tools
from slime_plugins.dcc.episode import _WebEvidence

SOURCE = "https://example.org/registry/index.html"


@pytest.fixture(autouse=True)
def offline_reader(monkeypatch):
    monkeypatch.setattr(search_provider, "wikipedia_page", lambda url: None)
    monkeypatch.setitem(sys.modules, "trafilatura", SimpleNamespace(extract=lambda *args, **kwargs: None))


def read_links(markup, url=SOURCE):
    return tools._page_links(markup.encode(), "text/html", "utf-8", url)


def test_html_links_resolve_base_relative_urls_titles_and_fragments_without_dns(monkeypatch):
    def no_dns(*args, **kwargs):
        pytest.fail("Extracting link metadata must not resolve DNS")

    monkeypatch.setattr(tools.socket, "getaddrinfo", no_dns)
    result = read_links('''<html><head><base href="../data/"></head><body>
        <a href="records.csv#all">All <strong>registry</strong> records</a>
        <a href="https://example.org/data/records.csv#other">Duplicate</a>
        <a href="/legal" title="Legal information"></a>
        <a href="//cdn.example.org/export" download><img alt="Download archive"></a>
        <a href="help" aria-label="Reader help"></a>
        <a href="bare"></a>
        <base href="https://ignored.example.org/">
        <script><a href="hidden.csv">Hidden</a></script>
        <style><a href="hidden2.csv">Hidden</a></style>
        <noscript><a href="hidden3.csv">Hidden</a></noscript>
        <svg><a href="hidden4.csv">Hidden</a></svg>
        </body></html>''')
    assert result == [
        {"title": "All registry records", "url": "https://example.org/data/records.csv"},
        {"title": "Download archive", "url": "https://cdn.example.org/export"},
        {"title": "Legal information", "url": "https://example.org/legal"},
        {"title": "Reader help", "url": "https://example.org/data/help"},
        {"title": "https://example.org/data/bare", "url": "https://example.org/data/bare"},
    ]


def test_relative_links_use_base_even_when_it_appears_later():
    result = read_links('<html><a href="records.csv">Records</a><base href="https://data.example.org/files/"></html>')
    assert result == [{"title": "Records", "url": "https://data.example.org/files/records.csv"}]


def test_unsafe_urls_and_duplicate_fragments_cannot_crowd_out_safe_downloads():
    unsafe = [
        "#local", "javascript:alert(1)", "data:text/csv,private", "mailto:admin@example.org",
        "file:///tmp/private.csv", "https://user:secret@example.org/private.csv",
        "http://localhost/private.csv", "http://service/private.csv", "http://node.local/private.csv",
        "http://node.localhost/private.csv", "http://node.internal/private.csv",
        "http://127.0.0.1/private.csv", "http://10.0.0.1/private.csv", "http://169.254.169.254/private.csv",
        "http://[::1]/private.csv", "http://[fc00::1]/private.csv", "http://[::ffff:127.0.0.1]/private.csv",
        "http://2130706433/private.csv", "http://0x7f000001/private.csv", "http://0177.0.0.1/private.csv",
        "https://example.org:invalid/private.csv", "https://bad host.example.org/private.csv",
        "https://example.org/" + "x" * 2050,
    ]
    markup = "<html>" + "".join(f'<a href="{escape(href, quote=True)}" download>Unsafe</a>' for href in unsafe * 3)
    markup += "".join(f'<a href="records.csv#row{index}">Duplicate CSV</a>' for index in range(100))
    markup += '<a href="late.json">Late dataset</a></html>'
    assert read_links(markup) == [
        {"title": "Duplicate CSV", "url": "https://example.org/registry/records.csv"},
        {"title": "Late dataset", "url": "https://example.org/registry/late.json"},
    ]


def test_late_data_links_beat_navigation_and_metadata_stays_bounded():
    navigation = "".join(f'<a href="/nav/{index}">Navigation {index}</a>' for index in range(200))
    downloads = [
        ("data.csv", "CSV"), ("data.json", "JSON"), ("report.pdf", "PDF"),
        ("/api?resource=registry", "API"), ("/download?id=registry", "Download"),
        ("export?format=csv", "Formatted CSV"),
    ]
    markup = "<html>" + navigation
    markup += "".join(f'<a href="{escape(href, quote=True)}">{title}</a>' for href, title in downloads)
    markup += '<a href="archive" download>' + "T" * 300 + "</a></html>"
    result = read_links(markup)
    assert len(result) == 32
    assert [item["title"] for item in result[:6]] == [title for _, title in downloads]
    assert result[6] == {"title": "T" * 160, "url": "https://example.org/registry/archive"}
    assert result[7] == {"title": "Navigation 0", "url": "https://example.org/nav/0"}
    assert all(len(item["title"]) <= 160 and len(item["url"]) <= 2048 for item in result)


def test_public_ipv4_ipv6_and_full_url_length_limit():
    prefix = "https://example.org/"
    exact = prefix + "x" * (2048 - len(prefix))
    result = read_links(f'''<html><a href="{exact}">Maximum length</a>
        <a href="{exact}x">Too long</a><a href="http://8.8.8.8/dns">IPv4</a>
        <a href="https://[2606:4700:4700::1111]:443/dns">IPv6</a></html>''')
    assert [item["url"] for item in result] == [exact, "http://8.8.8.8/dns", "https://[2606:4700:4700::1111]:443/dns"]


def test_paginated_html_keeps_whole_page_links_out_of_body_and_downloads_once(monkeypatch):
    prefix = "0123456789" * 6
    markup = f'<html>{prefix}<a href="records.csv">Registry CSV</a></html>'
    calls = []

    def download(url):
        calls.append(url)
        return markup.encode(), "text/html", "utf-8"

    monkeypatch.setattr(tools, "_download_public", download)
    web = tools.WebTools(max_page_chars=12)

    async def scenario():
        first = await web.fetch_url(SOURCE + "#top")
        second = await web.fetch_url(SOURCE, start=first["next_start"])
        assert first["text"] == prefix[:12] and second["text"] == prefix[12:24]
        assert first["start"] == 0 and first["end"] == second["start"] == 12 and second["end"] == 24
        assert first["next_start"] == 12 and second["next_start"] == 24
        assert first["truncated"] and second["truncated"]
        assert first["total_characters"] == second["total_characters"] == len(prefix + " Registry CSV")
        assert first["outgoing_links"] == second["outgoing_links"] == [
            {"title": "Registry CSV", "url": "https://example.org/registry/records.csv"},
        ]
        assert "entire HTML page" in first["outgoing_links_note"]
        assert "have not been read" in first["outgoing_links_note"]

    asyncio.run(scenario())
    assert calls == [SOURCE]
    assert web.pages == {SOURCE: prefix + " Registry CSV"}
    assert list(web.sources) == [SOURCE]


@pytest.mark.parametrize("content_type,text", [
    ("text/plain", 'Plain text with <a href="https://other.example.org/data.csv">markup</a>'),
    ("application/json", json.dumps({"body": '<html><a href="https://other.example.org/data.csv">markup</a></html>'})),
    ("application/ld+json", json.dumps({"body": "<html>embedded fragment</html>"})),
    ("application/octet-stream", json.dumps({"body": "<html>embedded fragment</html>"})),
    ("application/xml", '<record><html><a href="https://other.example.org/data.csv">markup</a></html></record>'),
])
def test_non_html_documents_keep_text_and_do_not_create_html_link_metadata(monkeypatch, content_type, text):
    monkeypatch.setattr(tools, "_download_public", lambda url: (text.encode(), content_type, "utf-8"))
    web = tools.WebTools()
    result = asyncio.run(web.fetch_url(SOURCE))
    assert result["text"] == text
    assert "outgoing_links" not in result and "outgoing_links_note" not in result
    assert web.page_links == {}


@pytest.mark.parametrize("content_type", ["text/html", "application/xhtml+xml", "text/plain", "application/octet-stream"])
def test_html_mime_and_leading_document_sniffing_discover_links(content_type):
    markup = '\ufeff  <!doctype html><html><a href="data.csv">Data</a></html>'
    result = tools._page_links(markup.encode(), content_type, "unknown-charset", SOURCE)
    assert result == [{"title": "Data", "url": "https://example.org/registry/data.csv"}]


def test_discovered_links_become_citable_only_after_the_target_is_read(monkeypatch):
    dataset = "https://example.org/registry/records.csv"
    calls = []

    def download(url):
        calls.append(url)
        if url == SOURCE:
            return b'<html>Registry listing.<a href="records.csv">CSV data</a></html>', "text/html", "utf-8"
        assert url == dataset
        return b"name,country\nAlpha,Canada\n", "text/csv", "utf-8"

    monkeypatch.setattr(tools, "_download_public", download)
    web = tools.WebTools()
    evidence = _WebEvidence()
    task = Task("ws", "widesearch", "validation", "Find Alpha", gold=[{"name": "Alpha"}], metadata={"evaluation": {"required": ["name"]}})
    answer = [{"name": "Alpha", "citations": [dataset]}]
    evidence.observe("web_search", {"results": [{"url": SOURCE}]})

    async def scenario():
        listing = await web.fetch_url(SOURCE)
        evidence.observe("fetch_url", listing)
        assert listing["outgoing_links"][0]["url"] == dataset
        assert dataset not in web.sources and dataset not in web.pages
        assert dataset not in evidence.sources and dataset not in evidence.fetched_sources
        assert web.citation_validity(answer) == 0
        assert evidence.reject_reasons(task, answer) == ["unobserved_citation_url"]
        data = await web.fetch_url(dataset)
        evidence.observe("fetch_url", data)
        assert data["text"] == "name,country\nAlpha,Canada\n"
        assert "outgoing_links" not in data
        assert web.citation_validity(answer) == 1
        assert evidence.reject_reasons(task, answer) == []

    asyncio.run(scenario())
    assert calls == [SOURCE, dataset]


def test_actual_fetch_still_rejects_a_discovered_hostname_resolving_to_private_ip(monkeypatch):
    dataset = "https://data.example.org/records.csv"
    assert read_links(f'<html><a href="{dataset}">Data</a></html>')[0]["url"] == dataset
    monkeypatch.setattr(tools.socket, "getaddrinfo", lambda *args, **kwargs: [(None, None, None, None, ("127.0.0.1", 443))])
    web = tools.WebTools()
    result = asyncio.run(web.fetch_url(dataset))
    assert "public internet hosts" in result["error"]
    assert web.pages == {} and web.sources == {}


def xlsx_workbook(sheets, *, shared="", extra=None, relationship=None):
    """Build ordinary OpenXML packages without a spreadsheet dependency."""
    def attribute(value):
        return escape(value, quote=True).replace("\n", "&#10;").replace("\r", "&#13;").replace("\t", "&#9;")

    main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    office = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    package = "http://schemas.openxmlformats.org/package/2006/relationships"
    parts = {
        "[Content_Types].xml": '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="xml" ContentType="application/xml"/></Types>',
        "xl/workbook.xml": f'<workbook xmlns="{main}" xmlns:r="{office}"><sheets>'
        + "".join(f'<sheet name="{attribute(name)}" sheetId="{index}" r:id="rId{index}"/>'
                  for index, (name, _) in enumerate(sheets, 1)) + '</sheets></workbook>',
        "xl/_rels/workbook.xml.rels": f'<Relationships xmlns="{package}">'
        + (relationship if relationship is not None else "".join(
            f'<Relationship Id="rId{index}" Type="{office}/worksheet" Target="worksheets/sheet{index}.xml"/>'
            for index in range(1, len(sheets) + 1))) + '</Relationships>',
    }
    parts.update({f"xl/worksheets/sheet{index}.xml": f'<worksheet xmlns="{main}"><sheetData>{rows}</sheetData></worksheet>'
                  for index, (_, rows) in enumerate(sheets, 1)})
    if shared:
        parts["xl/sharedStrings.xml"] = f'<sst xmlns="{main}">{shared}</sst>'
    parts.update(extra or {})
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in parts.items():
            archive.writestr(name, data)
    return stream.getvalue()


@pytest.mark.parametrize("mime", [tools._XLSX_MIME, tools._XLSX_MIME + "; charset=binary", "application/octet-stream", "application/save-to-disk"])
def test_xlsx_preserves_shared_inline_sparse_boolean_error_and_cached_formula_values(mime):
    rows = '''<row r="1"><c r="A1" t="s"><v>0</v></c>
        <c r="B1" t="inlineStr"><is><r><t>Second</t></r><r><t> column</t></r></is></c><c r="D1"><v>4</v></c></row>
        <row r="3"><c r="A3" t="b"><v>1</v></c><c r="B3" t="b"><v>0</v></c>
        <c r="C3"><f>1+1</f><v>2</v></c><c r="D3" t="str"><f>UNSAFE()</f><v>cached</v></c>
        <c r="E3"><f>NO_CACHE()</f></c><c r="F3" t="e"><v>#N/A</v></c>
        <c r="G3" t="inlineStr"><is><t>comma,quote&quot;\nnewline</t></is></c></row>
        <row r="4"><c r="A4" t="s"><v>1</v></c></row>'''
    shared = '<si><t>Header</t></si><si><r><t>Rich </t></r><rPh sb="0" eb="4"><t>phonetic</t></rPh><r><t>text</t></r></si>'
    workbook = xlsx_workbook([('First "sheet"', rows), ("Second sheet", '<row><c t="inlineStr"><is><t>Other</t></is></c></row>')], shared=shared,
                             extra={"xl/vbaProject.bin": b"must never execute", "xl/externalLinks/link1.xml": '<externalLink href="http://127.0.0.1/private"/>'})
    text = tools.extract_page(workbook, mime)
    assert 'Worksheet: "First \\"sheet\\""' in text
    first, second = text.split('Worksheet: "First \\"sheet\\""\n')[1].split('\nWorksheet: "Second sheet"\n')
    assert list(csv.reader(io.StringIO(first))) == [
        ["Header", "Second column", "", "4"], [],
        ["TRUE", "FALSE", "2", "cached", "", "#N/A", 'comma,quote"\nnewline'], ["Rich text"],
    ]
    assert list(csv.reader(io.StringIO(second))) == [["Other"]]
    assert "UNSAFE" not in text and "NO_CACHE" not in text and "must never execute" not in text and "phonetic" not in text
    assert tools._page_links(workbook, mime, "utf-8", SOURCE) is None


def test_xlsx_fetch_keeps_normal_page_pagination_and_only_the_workbook_source(monkeypatch):
    workbook = xlsx_workbook([("Names", '<row><c t="inlineStr"><is><t>Alpha</t></is></c></row>')])
    downloads = []

    def download(url):
        downloads.append(url)
        return workbook, "application/octet-stream", "utf-8"

    monkeypatch.setattr(tools, "_download_public", download)
    web = tools.WebTools(max_page_chars=30)
    expected = tools.extract_page(workbook, "application/octet-stream")

    async def scenario():
        pages = []
        start = 0
        while True:
            page = await web.fetch_url(SOURCE, start=start)
            assert "outgoing_links" not in page and "error" not in page
            assert page["worksheets"] == [{"name": "Names", "start": expected.index('\nWorksheet: "Names"'), "end": len(expected),
                                             "preview_row": 1, "preview": "Alpha"}]
            pages.append(page["text"])
            if page["next_start"] is None:
                break
            start = page["next_start"]
        assert "".join(pages) == expected
        assert list(web.sources) == [SOURCE]

    asyncio.run(scenario())
    assert downloads == [SOURCE]


def test_xlsx_navigation_offsets_handle_escaped_names_and_marker_looking_cells(monkeypatch):
    names = ['Notes "quoted" & \\name', 'Data\nWorksheet: "forged"\n中文', "Empty"]
    note = "Notes\nWorksheet: \"fake data\"\nnot a worksheet"
    rows = '<row r="2"/><row r="4"><c r="B4" t="inlineStr"><is><t>' + escape(note) + '</t></is></c></row>'
    data = '<row r="3"><c r="A3" t="inlineStr"><is><t>Name</t></is></c><c r="C3" t="inlineStr"><is><t>Value</t></is></c></row>'
    data += '<row r="4"><c r="A4" t="inlineStr"><is><t>Alpha</t></is></c><c r="C4"><f>2+3</f><v>5</v></c></row>'
    workbook = xlsx_workbook([(names[0], rows), (names[1], data), (names[2], "")])
    sections = []
    original = tools.extract_page(workbook, tools._XLSX_MIME)
    text = tools.extract_page(workbook, tools._XLSX_MIME, xlsx_sections=sections)
    assert text == original
    assert [section["name"] for section in sections] == names
    assert len(sections) == 3 and sections[-1]["end"] == len(text)
    assert sections[0]["end"] == sections[1]["start"]
    assert sections[1]["end"] == sections[2]["start"]
    for section, (_, expected_rows) in zip(sections, [(names[0], rows), (names[1], data), (names[2], "")]):
        block = text[section["start"]:section["end"]]
        marker = "\nWorksheet: " + json.dumps(section["name"], ensure_ascii=False) + "\n"
        assert block.startswith(marker)
        actual_rows = list(csv.reader(io.StringIO(block[len(marker):])))
        if section["preview_row"] is not None:
            assert list(csv.reader(io.StringIO(section["preview"]))) == [actual_rows[section["preview_row"] - 1]]
        else:
            assert not actual_rows and not expected_rows and section["preview"] == ""
    assert sections[0]["preview_row"] == 4 and sections[1]["preview_row"] == 3
    assert "2+3" not in text and "Alpha,,5" in text

    downloads = []
    monkeypatch.setattr(tools, "_download_public", lambda url: (downloads.append(url) or workbook, tools._XLSX_MIME, "utf-8"))
    web = tools.WebTools(max_page_chars=48)

    async def navigate():
        first = await web.fetch_url(SOURCE)
        assert first["worksheets"] == sections
        assert "Alpha" not in first["text"]
        assert "source-wide" in first["worksheets_note"].lower()
        target = first["worksheets"][1]
        jumped = await web.fetch_url(SOURCE, start=target["start"])
        assert jumped["text"] == original[target["start"]:target["start"] + web.max_page_chars]
        assert jumped["start"] == target["start"] and jumped["worksheets"] == sections
        assert list(web.sources) == [SOURCE]

    asyncio.run(navigate())
    assert downloads == [SOURCE]


def test_xlsx_navigation_bounds_names_and_previews_without_changing_text(monkeypatch):
    monkeypatch.setattr(tools, "_XLSX_NAV_NAME_CHARS", 8)
    monkeypatch.setattr(tools, "_XLSX_NAV_PREVIEW_CHARS", 6)
    monkeypatch.setattr(tools, "_XLSX_NAV_TOTAL_PREVIEW_CHARS", 10)
    rows = '<row r="3"><c t="inlineStr"><is><t>Long original cell</t></is></c></row>'
    workbook = xlsx_workbook([(f"Long worksheet {index}", rows) for index in range(3)])
    sections = []
    text = tools.extract_page(workbook, tools._XLSX_MIME, xlsx_sections=sections)
    assert text == tools.extract_page(workbook, tools._XLSX_MIME)
    assert [section["preview"] for section in sections] == ["Long o", "Long", ""]
    assert all(section["name"] == "Long wor" and section["name_truncated"] for section in sections)
    assert all(section["preview_row"] == 3 and section["preview_truncated"] for section in sections)
    assert text.count("Long original cell") == 3 and '"Long worksheet 2"' in text


def test_xlsx_failure_does_not_publish_partial_navigation():
    workbook = xlsx_workbook([("Good", '<row><c><v>1</v></c></row>'),
                              ("Bad", '<row r="1"><c r="A2"><v>2</v></c></row>')])
    sections = []
    with pytest.raises(ValueError, match="coordinate"):
        tools.extract_page(workbook, tools._XLSX_MIME, xlsx_sections=sections)
    assert sections == []


def test_html_marker_text_does_not_create_xlsx_navigation(monkeypatch):
    html = b'<html>Worksheet: "Fake"<p>name,value</p></html>'
    monkeypatch.setattr(tools, "_download_public", lambda url: (html, "text/html", "utf-8"))
    web = tools.WebTools()
    result = asyncio.run(web.fetch_url(SOURCE))
    assert "worksheets" not in result and "worksheets_note" not in result
    assert web.page_worksheets == {}


@pytest.mark.parametrize("limit,value,rows,sheet_count,expected", [
    ("_XLSX_MAX_MEMBERS", 2, '<row><c><v>1</v></c></row>', 1, "member limit"),
    ("_XLSX_MAX_MEMBER_BYTES", 10, '<row><c><v>1</v></c></row>', 1, "uncompressed byte limits"),
    ("_XLSX_MAX_TOTAL_BYTES", 10, '<row><c><v>1</v></c></row>', 1, "uncompressed byte limits"),
    ("_XLSX_MAX_SHEETS", 1, '<row><c><v>1</v></c></row>', 2, "sheet limit"),
    ("_XLSX_MAX_ROWS", 2, '<row r="3"><c r="A3"><v>1</v></c></row>', 1, "row limit"),
    ("_XLSX_MAX_COLUMNS", 2, '<row r="1"><c r="C1"><v>1</v></c></row>', 1, "column limit"),
    ("_XLSX_MAX_CELLS", 1, '<row><c><v>1</v></c><c><v>2</v></c></row>', 1, "cell limit"),
    ("_XLSX_MAX_TEXT_CHARS", 10, '', 1, "character limit"),
])
def test_xlsx_bounds_fail_without_partial_success(monkeypatch, limit, value, rows, sheet_count, expected):
    workbook = xlsx_workbook([(f"Sheet {index}", rows) for index in range(sheet_count)])
    monkeypatch.setattr(tools, limit, value)
    with pytest.raises(ValueError, match=expected):
        tools.extract_page(workbook, "application/octet-stream")


@pytest.mark.parametrize("rows,shared,expected", [
    ('<row><c t="s"><v>-1</v></c></row>', '<si><t>Value</t></si>', "shared string reference"),
    ('<row><c t="s"><v>2</v></c></row>', '<si><t>Value</t></si>', "shared string reference"),
    ('<row><c t="b"><v>2</v></c></row>', '', "boolean"),
    ('<row r="1"><c r="A2"><v>1</v></c></row>', '', "coordinate"),
    ('<row r="1"><c r="A1"><v>1</v></c><c r="A1"><v>2</v></c></row>', '', "column index"),
    ('<row r="2"/><row r="1"/>', '', "row index"),
    ('<row>', '', "could not read"),
])
def test_xlsx_malformed_cells_are_not_successful_evidence(rows, shared, expected):
    workbook = xlsx_workbook([("Sheet", rows)], shared=shared)
    with pytest.raises(ValueError, match=expected):
        tools.extract_page(workbook, tools._XLSX_MIME)


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
def test_xlsx_rejects_entity_declarations_in_xml(encoding):
    workbook = xlsx_workbook([("Sheet", "")], extra={"xl/sharedStrings.xml":
        ('<?xml version="1.0" encoding="' + encoding + '"?><!DOCTYPE sst [<!ENTITY x "expanded">]><sst><si><t>&x;</t></si></sst>').encode(encoding)})
    with pytest.raises(ValueError, match="declares entities"):
        tools.extract_page(workbook, tools._XLSX_MIME)


@pytest.mark.parametrize("target,mode", [("https://example.org/private", "External"), ("../../private.xml", ""), ("worksheets\\sheet1.xml", "")])
def test_xlsx_never_follows_external_or_escaping_worksheet_relationships(monkeypatch, target, mode):
    def forbidden(*args, **kwargs):
        pytest.fail("XLSX extraction must not use a network or extract archive files")

    monkeypatch.setattr(tools.urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(zipfile.ZipFile, "extract", forbidden)
    monkeypatch.setattr(zipfile.ZipFile, "extractall", forbidden)
    relationship = f'<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="{target}" TargetMode="{mode}"/>'
    with pytest.raises(ValueError, match="relationship|path"):
        tools.extract_page(xlsx_workbook([("Sheet", "")], relationship=relationship), "application/octet-stream")


def test_unknown_zip_truncated_xlsx_and_forged_mime_are_rejected():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("records.bin", b"\x00\xff\x01")
    with pytest.raises(ValueError, match="no supported readable text"):
        tools.extract_page(stream.getvalue(), "application/octet-stream")
    with pytest.raises(ValueError, match="XLSX"):
        tools.extract_page(b"not a workbook", tools._XLSX_MIME)
    workbook = xlsx_workbook([("Sheet", "")])
    with pytest.raises(ValueError, match="XLSX"):
        tools.extract_page(workbook[:-30], tools._XLSX_MIME)


def test_xlsx_directory_count_is_checked_before_zipfile_allocation(monkeypatch):
    workbook = bytearray(xlsx_workbook([("Sheet", "")]))
    end = workbook.rfind(b"PK\x05\x06")
    # A forged directory count cannot bypass the actual member-count bound.
    workbook[end + 10:end + 12] = (1).to_bytes(2, "little")
    monkeypatch.setattr(tools, "_XLSX_MAX_MEMBERS", 2)
    monkeypatch.setattr(zipfile, "ZipFile", lambda *args, **kwargs: pytest.fail("Must reject before ZipFile allocation"))
    with pytest.raises(ValueError, match="member limit"):
        tools.extract_page(bytes(workbook), tools._XLSX_MIME)


def test_standard_static_api_links_resolve_deduplicate_and_remain_navigation_metadata(monkeypatch):
    def no_dns(*args, **kwargs):
        pytest.fail("API link metadata must not resolve or fetch URLs")

    monkeypatch.setattr(tools.socket, "getaddrinfo", no_dns)
    html = '''<html><base href="https://docs.example.org/data/">
        <script>window.ui = SwaggerUIBundle({dom_id: '#app', url: 'openapi.json', presets: []});</script>
        <redoc spec-url="openapi.json"></redoc><rapi-doc spec-url="other.json"></rapi-doc>
        <link rel="alternate" type="application/ld+json; charset=utf-8" href="records.json">
        <link rel="alternate" type="application/xml" href="records.xml">
        <link rel="stylesheet" type="application/json" href="ignored.json">
        <link rel="alternate" type="text/html" href="ignored.html"></html>'''
    assert read_links(html) == [
        {"title": "OpenAPI specification", "url": "https://docs.example.org/data/openapi.json"},
        {"title": "OpenAPI specification", "url": "https://docs.example.org/data/other.json"},
        {"title": "Alternate machine-readable source", "url": "https://docs.example.org/data/records.json"},
        {"title": "Alternate machine-readable source", "url": "https://docs.example.org/data/records.xml"},
    ]


def test_static_swagger_metadata_accepts_the_observed_standard_documentation_fragment():
    html = '''<html><script>
        window.onload = function() {
          var ui = SwaggerUIBundle({
            url: "/openapi/checklistbank.json",
            dom_id: '#swagger-ui',
            deepLinking: true,
            presets: [SwaggerUIBundle.presets.apis, SwaggerUIStandalonePreset]
          });
        };
        </script></html>'''
    assert read_links(html, "https://docs.example.org/en/api/species") == [
        {"title": "OpenAPI specification", "url": "https://docs.example.org/openapi/checklistbank.json"},
    ]


@pytest.mark.parametrize("script", [
    '''// SwaggerUIBundle({url: '/comment.json'});''',
    '''/* SwaggerUIBundle({url: '/comment.json'}); */''',
    '''const text = "SwaggerUIBundle({url: '/string.json'})";''',
    "const text = `SwaggerUIBundle({url: '/template.json'})`;",
    '''SwaggerUIBundle({url: window.configuration});''',
    '''SwaggerUIBundle({url: '/prefix' + '/dynamic.json'});''',
    '''SwaggerUIBundle({url: `/dynamic/${name}.json`});''',
    '''SwaggerUIBundle({config: {url: '/nested.json'}});''',
    '''fetch('/unrelated.json'); var url = '/unrelated.json';''',
    '''SwaggerUIBundle({url: '/escaped\\u002fpath.json'});''',
])
def test_uncertain_script_fragments_do_not_create_api_links(script):
    assert read_links('<html><script>' + script + '</script></html>') == []


def test_hidden_oversized_and_data_scripts_cannot_produce_api_links():
    declaration = "SwaggerUIBundle({url:'/hidden.json'});"
    html = f'''<html><noscript><script>{declaration}</script><redoc spec-url="hidden.json"></redoc></noscript>
        <script type="application/json">{declaration}</script><script src="external.js">{declaration}</script>
        <script>{" " * 65537}{declaration}</script></html>'''
    assert read_links(html) == []


@pytest.mark.parametrize("target", ["http://127.0.0.1/private", "http://service.local/private", "javascript:alert(1)", "https://user:secret@example.org/private"])
def test_api_spec_metadata_uses_existing_unsafe_url_filter(target):
    assert read_links(f'''<html><redoc spec-url="{escape(target, quote=True)}"></redoc>
        <script>SwaggerUIBundle({{url:'{target}'}});</script></html>''') == []


def test_many_static_api_links_keep_existing_link_count_bound():
    html = '<html>' + ''.join(f'<redoc spec-url="schema{index}.json"></redoc>' for index in range(100)) + '</html>'
    assert len(read_links(html)) == 32


def test_discovered_api_spec_is_not_answer_evidence_before_data_is_fetched(monkeypatch):
    spec_url = "https://example.org/api/openapi.json"
    html = f'<html>API documentation<script>SwaggerUIBundle({{url:"{spec_url}"}});</script></html>'
    monkeypatch.setattr(tools, "_download_public", lambda url: (html.encode(), "text/html", "utf-8"))
    web = tools.WebTools()
    evidence = _WebEvidence()
    evidence.observe("web_search", {"results": [{"url": SOURCE}]})
    page = asyncio.run(web.fetch_url(SOURCE))
    evidence.observe("fetch_url", page)
    assert page["outgoing_links"] == [{"title": "OpenAPI specification", "url": spec_url}]
    assert spec_url not in web.pages and spec_url not in web.sources
    assert spec_url not in evidence.sources and spec_url not in evidence.fetched_sources
    assert spec_url not in page["text"]
    assert web.citation_validity([{"name": "Alpha", "citations": [spec_url]}]) == 0


@pytest.mark.parametrize("size", [5_000_000, 5_000_001, 5_000_200])
def test_http_download_never_publishes_a_silently_truncated_source(monkeypatch, size):
    from email.message import Message
    payload = b"a" * size
    headers = Message()
    headers["Content-Type"] = "text/csv; charset=utf-8"
    response = io.BytesIO(payload)
    response.headers = headers
    monkeypatch.setattr(tools, "_public_url", lambda url: url)
    monkeypatch.setattr(tools.urllib.request, "build_opener", lambda *args: SimpleNamespace(open=lambda *args, **kwargs: response))
    if size > 5_000_000:
        with pytest.raises(ValueError, match="exceeds.*byte limit"):
            tools._download_public(SOURCE)
    else:
        assert tools._download_public(SOURCE) == (payload, "text/csv", "utf-8")


def test_oversized_download_does_not_publish_read_evidence_or_navigation(monkeypatch):
    from email.message import Message
    response = io.BytesIO(b"a" * 5_000_001)
    response.headers = Message()
    monkeypatch.setattr(tools, "_public_url", lambda url: url)
    monkeypatch.setattr(tools.urllib.request, "build_opener", lambda *args: SimpleNamespace(open=lambda *args, **kwargs: response))
    reader = tools.WebTools(backend="duckduckgo_lite")
    observation = asyncio.run(reader.fetch_url(SOURCE))
    evidence = _WebEvidence()
    evidence.observe("fetch_url", observation)
    assert "exceeds" in observation["error"]
    assert "text" not in observation and "worksheets" not in observation and "archive_members" not in observation
    assert not reader.pages and not evidence.fetched_sources
