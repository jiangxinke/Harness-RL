"""Portable ZIP exports preserve evidence/navigation and reject hostile archives."""
import asyncio
import io
import json
import stat
import struct
import zipfile

import pytest

from slime_plugins.dcc import tools, web_archive


def zipped(*members, compression=zipfile.ZIP_DEFLATED):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=compression) as archive:
        for name, content in members:
            archive.writestr(name, content)
    return stream.getvalue()


def read(data, mime="application/octet-stream", sections=None):
    return tools.extract_page(data, mime, archive_sections=sections)


@pytest.mark.parametrize("mime", ["application/zip", "application/x-zip-compressed", "application/octet-stream", "text/plain"])
def test_csv_xml_export_preserves_literal_rows_and_member_positions(mime):
    csv = 'name,code,note\r\n"Alp,ha",AL,"line1\nline2"\r\n中😀,ZH,\r\n'
    xml = '<?xml version="1.0"?><rows><row id="1"><name>Beta &amp; Co</name></row></rows>'
    sections = []
    data = zipped(("records.csv", csv), ("metadata.xml", xml))
    text = read(data, mime, sections)
    assert csv in text and xml in text
    assert [s["name"] for s in sections] == ["records.csv", "metadata.xml"]
    for section, literal in zip(sections, [csv, xml]):
        assert section["status"] == "readable"
        assert text[section["start"]:section["content_start"]].startswith('\nArchive member: ')
        assert text[section["content_start"]:section["content_end"]] == literal
        assert text[section["content_end"]:section["end"]] == "\n"
        assert section["preview"] == literal
        assert not section["preview_truncated"]
    assert text == read(data, mime)


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16", "utf-32"])
def test_bom_text_encoding_and_unicode_offsets(encoding):
    literal = 'name\n中文😀\n'
    sections = []
    text = read(zipped(("rows.csv", literal.encode(encoding))), sections=sections)
    assert text[sections[0]["content_start"]:sections[0]["content_end"]] == literal


def test_explicit_xml_legacy_encoding_preserves_values():
    xml = '<?xml version="1.0" encoding="windows-1252"?><row>Café</row>'
    assert xml in read(zipped(("rows.xml", xml.encode("cp1252"))))


def test_mixed_directory_skips_unsupported_files_without_claiming_read():
    sections = []
    text = read(zipped(("data/", b""), ("data/rows.tsv", "name\tvalue\nAlpha\t1\n"),
                      ("notes.txt", "literal notes"), ("image.png", b"\x89PNG\r\n\x1a\n"),
                      ("data/rows.json", '[{"name":"Beta"}]')), sections=sections)
    assert [s["status"] for s in sections] == ["directory", "readable", "readable", "unsupported_file_type", "readable"]
    assert not any(key in sections[0] or key in sections[3] for key in ["start", "end", "preview"])
    assert "PNG" not in text
    assert json.loads(text[sections[4]["content_start"]:sections[4]["content_end"]]) == [{"name": "Beta"}]


@pytest.mark.parametrize("name", ["../rows.csv", "/rows.csv", "C:/rows.csv", "dir\\rows.csv", "dir/../rows.csv",
                                   "dir/./rows.csv", "dir//rows.csv", "rows\n.csv", "rows\u202e.csv"])
def test_unsafe_member_names_are_rejected_atomically(name):
    sections = []
    with pytest.raises(ValueError, match="unsafe member name"):
        read(zipped(("good.csv", "name\nAlpha\n"), (name, "bad")), sections=sections)
    assert sections == []


def test_duplicate_member_names_are_rejected():
    with pytest.warns(UserWarning, match="Duplicate name"):
        data = zipped(("rows.csv", "name\nA\n"), ("rows.csv", "name\nB\n"))
    with pytest.raises(ValueError, match="duplicate members"):
        read(data)


@pytest.mark.parametrize("mode", [stat.S_IFLNK | 0o777, stat.S_IFIFO | 0o600, stat.S_IFSOCK | 0o600])
def test_symbolic_links_and_special_members_are_rejected(mode):
    member = zipfile.ZipInfo("rows.csv")
    member.create_system = 3
    member.external_attr = mode << 16
    with pytest.raises(ValueError, match="symbolic link or special"):
        read(zipped((member, "target.csv")))


@pytest.mark.parametrize("name,payload", [("inner.zip", b"text"), ("inner.tar.gz", b"text"), ("inner.xlsx", b"text"),
                                           ("inner.csv", b"PK\x03\x04bad"), ("inner.csv", b"\x1f\x8bbad")])
def test_nested_archives_are_rejected_by_name_and_magic(name, payload):
    with pytest.raises(ValueError, match="nested ZIP/archive"):
        read(zipped((name, payload)))


@pytest.mark.parametrize("name,payload,match", [
    ("rows.csv", b"\x00\x01binary", "binary control"),
    ("rows.csv", b"name\n\xff\n", "text encoding"),
    ("rows.xml", b"<broken>", "not well-formed"),
    ("rows.xml", b'<!DOCTYPE root [<!ENTITY a "value">]><root>&a;</root>', "entity"),
    ("rows.json", b"[{broken]", "not well-formed"),
])
def test_invalid_member_content_keeps_member_and_cause(name, payload, match):
    sections = []
    with pytest.raises(ValueError, match=match) as error:
        read(zipped(("good.csv", "name\nAlpha\n"), (name, payload)), sections=sections)
    assert name in str(error.value) and error.value.__cause__ is not None
    assert sections == []


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16", "utf-32"])
def test_entity_declarations_are_rejected_after_decoding(encoding):
    xml = '<!DOCTYPE root [<!ENTITY a "value">]><root>&a;</root>'
    with pytest.raises(ValueError, match="entity"):
        read(zipped(("rows.xml", xml.encode(encoding))))


@pytest.mark.parametrize("attribute,value,match", [
    ("_MAX_DOWNLOAD_BYTES", 20, "download.*byte limit"),
    ("_MAX_MEMBERS", 1, "member limit"),
    ("_MAX_MEMBER_BYTES", 4, "member byte limit"),
    ("_MAX_TOTAL_BYTES", 10, "total expanded byte"),
    ("_MAX_TEXT_CHARS", 10, "character limit"),
])
def test_archive_limits_reject_without_partial_navigation(monkeypatch, attribute, value, match):
    data = zipped(("rows.csv", "name\nAlpha\n"), ("notes.txt", "notes"))
    monkeypatch.setattr(web_archive, attribute, value)
    sections = []
    with pytest.raises(ValueError, match=match):
        read(data, sections=sections)
    assert sections == []


def test_compression_bomb_is_rejected_before_opening_member(monkeypatch):
    data = zipped(("rows.csv", "A" * 500000))
    monkeypatch.setattr(zipfile.ZipFile, "open", lambda *args, **kwargs: pytest.fail("Must reject before decompression"))
    with pytest.raises(ValueError, match="compression-ratio"):
        read(data)


def test_directory_count_is_verified_before_allocation(monkeypatch):
    data = bytearray(zipped(("rows.csv", "name\nA\n"), ("notes.txt", "text")))
    end = data.rfind(b"PK\x05\x06")
    struct.pack_into("<H", data, end + 8, 1)
    struct.pack_into("<H", data, end + 10, 1)
    monkeypatch.setattr(zipfile, "ZipFile", lambda *args, **kwargs: pytest.fail("Must reject before ZipFile allocation"))
    with pytest.raises(ValueError, match="member count"):
        web_archive.text_zip_page(bytes(data))


def test_preallocation_limit_cannot_be_bypassed_by_forged_count(monkeypatch):
    data = bytearray(zipped(("a.csv", "name\nA\n"), ("b.csv", "name\nB\n")))
    end = data.rfind(b"PK\x05\x06")
    struct.pack_into("<H", data, end + 8, 1)
    struct.pack_into("<H", data, end + 10, 1)
    monkeypatch.setattr(web_archive, "_MAX_MEMBERS", 1)
    monkeypatch.setattr(zipfile, "ZipFile", lambda *args, **kwargs: pytest.fail("Must reject before ZipFile allocation"))
    with pytest.raises(ValueError, match="member limit"):
        web_archive.text_zip_page(bytes(data))


@pytest.mark.parametrize("offset,value,match", [(4, 1, "split"), (6, 1, "split"), (10, 65535, "ZIP64")])
def test_split_and_zip64_directory_metadata_rejected(offset, value, match):
    data = bytearray(zipped(("rows.csv", "name\nAlpha\n")))
    end = data.rfind(b"PK\x05\x06")
    struct.pack_into("<H", data, end + offset, value)
    with pytest.raises(ValueError, match=match):
        web_archive.text_zip_page(bytes(data))


def test_preview_and_names_bounded_without_changing_literal_member_text(monkeypatch):
    monkeypatch.setattr(web_archive, "_NAV_NAME_CHARS", 5)
    monkeypatch.setattr(web_archive, "_NAV_PREVIEW_CHARS", 10)
    monkeypatch.setattr(web_archive, "_NAV_TOTAL_PREVIEW_CHARS", 15)
    sections = []
    literal = 'name\nArchive member: "fake.csv"\n中😀\n'
    data = zipped(("original1.csv", literal), ("original2.csv", literal))
    text = read(data, sections=sections)
    assert all(s["name"] == "origi" and s["name_truncated"] for s in sections)
    assert [len(s["preview"]) for s in sections] == [10, 5]
    assert all(s["preview_truncated"] for s in sections)
    assert text == read(data)
    for section in sections:
        assert text[section["content_start"]:section["content_end"]] == literal


def test_no_extraction_or_network_and_fetch_windows_remain_ordinary_evidence(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("ZIP extraction must never write members, resolve URLs, or fetch their contents")
    monkeypatch.setattr(zipfile.ZipFile, "extract", forbidden)
    monkeypatch.setattr(zipfile.ZipFile, "extractall", forbidden)
    monkeypatch.setattr(tools.socket, "getaddrinfo", forbidden)
    from slime_plugins.dcc import search_provider
    monkeypatch.setattr(search_provider, "wikipedia_page", lambda url: None)
    data = zipped(("metadata.xml", "<title>Source metadata</title>"), ("records.csv", "name\n中文😀\n"))
    downloads = []
    def download(url):
        downloads.append(url)
        return data, "application/zip", "utf-8"
    monkeypatch.setattr(tools, "_download_public", download)
    async def run():
        url = "https://example.org/export.zip"
        web = tools.WebTools(max_page_chars=30)
        initial = await web.fetch_url(url)
        assert len(initial["text"]) == 30
        sections = initial["archive_members"]
        assert sections[1]["start"] > initial["end"]
        assert "Source-wide" in initial["archive_members_note"] and "only for navigation" in initial["archive_members_note"]
        jumped = await web.fetch_url(url, start=sections[1]["start"])
        assert jumped["start"] == sections[1]["start"]
        assert jumped["archive_members"] == sections
        assert jumped["text"] == web.pages[url][jumped["start"]:jumped["end"]]
        assert set(web.sources) == {url} and downloads == [url]
        assert not web.page_worksheets
        repeated = await web.fetch_url(url)
        assert repeated["read_progress"]["new_text"] is False
        assert repeated["archive_members"] == sections
    asyncio.run(run())


def test_failed_archive_never_becomes_source_or_cached_page(monkeypatch):
    from slime_plugins.dcc import search_provider
    monkeypatch.setattr(search_provider, "wikipedia_page", lambda url: None)
    monkeypatch.setattr(tools, "_download_public", lambda url: (zipped(("rows.csv", b"\x00bad")), "application/zip", "utf-8"))
    async def run():
        web = tools.WebTools()
        result = await web.fetch_url("https://example.org/export.zip")
        assert "error" in result and "rows.csv" in result["error"]
        assert not web.pages and not web.page_archives and not web.sources
    asyncio.run(run())


@pytest.mark.parametrize("data", [zipped(("image.bin", b"\x00binary")), zipped(("rows.csv", "")), zipped()])
def test_no_readable_member_is_a_failed_read(data):
    with pytest.raises(ValueError, match="no supported readable text"):
        read(data, "application/zip")


def test_embedded_nul_in_actual_zip_filename_rejected():
    # ZipFile.writestr truncates NUL names before writing; forge actual bytes.
    data = zipped(("rowsX.csv", "name\nAlpha\n")).replace(b"rowsX.csv", b"rows\x00.csv")
    with pytest.raises(ValueError, match="unsafe member name"):
        read(data)


def test_encrypted_flag_rejected_before_open(monkeypatch):
    data = bytearray(zipped(("rows.csv", "name\nAlpha\n")))
    local = data.index(b"PK\x03\x04")
    central = data.index(b"PK\x01\x02")
    struct.pack_into("<H", data, local + 6, 1)
    struct.pack_into("<H", data, central + 8, 1)
    monkeypatch.setattr(zipfile.ZipFile, "open", lambda *args, **kwargs: pytest.fail("Must reject before reading"))
    with pytest.raises(ValueError, match="encrypted"):
        read(bytes(data))


def test_unsupported_compression_rejected():
    with pytest.raises(ValueError, match="unsupported compression"):
        read(zipped(("rows.csv", "name\nAlpha\n"), compression=zipfile.ZIP_BZIP2))


def test_bad_crc_is_not_successful_evidence():
    data = bytearray(zipped(("rows.csv", "name\nAlpha\n"), compression=zipfile.ZIP_STORED))
    location = data.index(b"Alpha")
    data[location] = ord("B")
    with pytest.raises(ValueError, match="rows.csv") as error:
        read(bytes(data))
    assert isinstance(error.value.__cause__, zipfile.BadZipFile)
