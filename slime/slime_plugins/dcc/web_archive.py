"""Bounded in-memory reading of public ZIP text exports.

No archive member is extracted to disk, executed, fetched, or recursively opened.
The existing OpenXML workbook reader remains responsible for XLSX files.
"""

from __future__ import annotations

import io
import json
import re
import stat
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from pathlib import PurePosixPath
from typing import Any

_MAX_DOWNLOAD_BYTES = 5_000_000
_MAX_MEMBERS = 128
_MAX_MEMBER_BYTES = 8_000_000
_MAX_TOTAL_BYTES = 25_000_000
_MAX_COMPRESSION_RATIO = 200
_MAX_TEXT_CHARS = 1_000_000
_MAX_NAME_CHARS = 1024
_NAV_NAME_CHARS = 256
_NAV_PREVIEW_CHARS = 256
_NAV_TOTAL_PREVIEW_CHARS = 4096
WEB_ZIP_MEMBER_FALLBACK_VERSION = "bounded-zip-character-member-fallback-v1"


def web_zip_member_fallback_configuration(enabled=False):
    if type(enabled) is not bool:
        raise ValueError("web_zip_member_fallback must be a boolean")
    return {"enabled": enabled,
            "version": WEB_ZIP_MEMBER_FALLBACK_VERSION if enabled else None}


class _MemberTextCharacterLimit(ValueError):
    """Only the decoded-text character cap can opt in to member skipping."""


_TEXT_FORMATS = {".csv": "csv", ".tsv": "tsv", ".xml": "xml", ".json": "json", ".txt": "text"}
_ARCHIVE_SUFFIXES = {
    ".zip", ".zipx", ".gz", ".gzip", ".tgz", ".bz2", ".tbz", ".tbz2", ".xz", ".txz", ".zst", ".tzst",
    ".7z", ".rar", ".tar", ".jar", ".apk", ".xlsx", ".xlsm", ".docx", ".pptx", ".odt", ".ods", ".odp",
}
_ARCHIVE_MAGIC = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08", b"\x1f\x8b", b"BZh", b"\xfd7zXZ\x00",
                  b"7z\xbc\xaf\x27\x1c", b"Rar!\x1a\x07", b"\x28\xb5\x2f\xfd")


def _directory(content: bytes) -> None:
    """Bound the central-directory allocation before constructing ZipFile."""
    if len(content) > _MAX_DOWNLOAD_BYTES:
        raise ValueError("ZIP download exceeds the byte limit")
    end = content.rfind(b"PK\x05\x06", max(0, len(content) - 65557))
    if end < 0 or end + 22 > len(content):
        raise ValueError("invalid ZIP end-of-directory record")
    value = lambda offset, size=2: int.from_bytes(content[end + offset:end + offset + size], "little")
    count, directory_size, position = value(10), value(12, 4), value(16, 4)
    if value(4) or value(6) or value(8) != count or count == 65535 or directory_size == 0xffffffff or position == 0xffffffff:
        raise ValueError("split and ZIP64 archives are unsupported")
    if end + 22 + value(20) != len(content) or position + directory_size != end:
        raise ValueError("invalid ZIP central-directory bounds")
    if count > _MAX_MEMBERS:
        raise ValueError("ZIP archive exceeds the member limit")
    actual = 0
    while position < end:
        if position + 46 > end or content[position:position + 4] != b"PK\x01\x02":
            raise ValueError("invalid ZIP central-directory entry")
        actual += 1
        if actual > _MAX_MEMBERS:
            raise ValueError("ZIP archive exceeds the member limit")
        position += 46 + sum(int.from_bytes(content[position + offset:position + offset + 2], "little") for offset in (28, 30, 32))
    if position != end or actual != count:
        raise ValueError("invalid ZIP central-directory member count")


def _safe_member(member: zipfile.ZipInfo) -> None:
    name = member.filename
    parts = name[:-1].split("/") if name.endswith("/") else name.split("/")
    # ZipInfo truncates a NUL-containing filename; inspect its original too.
    if (not name or member.orig_filename != name or len(name) > _MAX_NAME_CHARS or "\\" in name or ":" in name
            or name.startswith("/") or any(part in {"", ".", ".."} for part in parts)
            or any(unicodedata.category(character).startswith("C") for character in name)):
        raise ValueError("ZIP archive contains an unsafe member name")
    kind = stat.S_IFMT(member.external_attr >> 16)
    if kind not in {0, stat.S_IFREG, stat.S_IFDIR} or (kind != 0 and (kind == stat.S_IFDIR) != member.is_dir()):
        raise ValueError("ZIP archive contains a symbolic link or special member")
    if member.flag_bits & 0x41:
        raise ValueError("encrypted ZIP members are unsupported")
    if member.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
        raise ValueError("ZIP member uses an unsupported compression method")
    if member.file_size > _MAX_MEMBER_BYTES:
        raise ValueError("ZIP archive exceeds the member byte limit")
    if member.file_size > _MAX_COMPRESSION_RATIO * max(member.compress_size, 1):
        raise ValueError("ZIP archive exceeds the compression-ratio limit")
    if any(suffix.lower() in _ARCHIVE_SUFFIXES for suffix in PurePosixPath(name).suffixes):
        raise ValueError("nested ZIP/archive members are unsupported")
    if member.is_dir() and member.file_size:
        raise ValueError("ZIP directory member has nonempty content")


def _decode_text(raw: bytes, format: str, *, member_fallback: bool = False) -> str:
    if raw.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        encoding = "utf-32"
    elif raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        encoding = "utf-16"
    else:
        encoding = "utf-8-sig"
        if format == "xml":
            declaration = re.match(br'''\s*<\?xml\b[^>]*\bencoding\s*=\s*["']([A-Za-z0-9._-]+)["']''', raw[:256], re.IGNORECASE)
            if declaration:
                encoding = declaration[1].decode("ascii").lower()
                if encoding not in {"utf-8", "utf8", "ascii", "us-ascii", "iso-8859-1", "iso8859-1", "latin1", "windows-1252", "cp1252"}:
                    raise ValueError("unsupported XML text encoding; use UTF-8 or a Unicode BOM")
    try:
        text = raw.decode(encoding, errors="strict")
    except UnicodeError as exc:
        raise ValueError("ZIP member has invalid or unsupported text encoding") from exc
    if any(ord(character) < 32 and character not in "\t\n\r" for character in text):
        raise ValueError("ZIP member contains binary control bytes")
    over_character_limit = len(text) > _MAX_TEXT_CHARS
    if over_character_limit and not member_fallback:
        raise ValueError("ZIP member text exceeds the character limit")
    if format == "xml":
        if re.search(r"<!\s*(?:DOCTYPE|ENTITY)\b", text, re.IGNORECASE):
            raise ValueError("ZIP XML member declares a document type or entity")
        try:
            ET.fromstring(text)
        except ET.ParseError as exc:
            raise ValueError("ZIP XML member is not well-formed") from exc
    elif format == "json":
        try:
            json.loads(text)
        except (ValueError, RecursionError) as exc:
            raise ValueError("ZIP JSON member is not well-formed") from exc
    # In fallback mode, a text cap cannot mask a structured-format error.
    # Validation remains bounded by the unchanged member/total byte limits.
    if over_character_limit:
        raise _MemberTextCharacterLimit("ZIP member text exceeds the character limit")
    # Match the external-page Unicode boundary before recording offsets.
    return text.encode("utf-8", errors="replace").decode("utf-8")


def text_zip_page(content: bytes, *, sections: list[dict[str, Any]] | None = None,
                  member_fallback: bool = False) -> str:
    """Return literal bounded text and atomic source-wide member navigation.

    The opt-in skips only whole members exceeding text character budgets. It
    never converts other reader or archive validation failures into success.
    """
    web_zip_member_fallback_configuration(member_fallback)
    _directory(content)
    navigation: list[dict[str, Any]] = []
    output = io.StringIO()
    output.write("ZIP text export; member contents are decoded literal text.\n")
    preview_budget = _NAV_TOTAL_PREVIEW_CHARS
    readable = 0
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            members = archive.infolist()
            if len(members) > _MAX_MEMBERS or len({member.filename for member in members}) != len(members):
                raise ValueError("ZIP archive contains too many or duplicate members")
            if sum(member.file_size for member in members) > _MAX_TOTAL_BYTES:
                raise ValueError("ZIP archive exceeds the total expanded byte limit")
            for member in members:
                _safe_member(member)
            for index, member in enumerate(members):
                format = _TEXT_FORMATS.get(PurePosixPath(member.filename).suffix.lower())
                section: dict[str, Any] = {
                    "index": index, "name": member.filename[:_NAV_NAME_CHARS],
                    "uncompressed_bytes": member.file_size, "compressed_bytes": member.compress_size,
                    "status": "directory" if member.is_dir() else "unsupported_file_type", "format": format,
                }
                if len(member.filename) > _NAV_NAME_CHARS:
                    section["name_truncated"] = True
                navigation.append(section)
                if member.is_dir() or format is None:
                    continue
                try:
                    with archive.open(member) as stream:
                        raw = stream.read(_MAX_MEMBER_BYTES + 1)
                    if len(raw) > _MAX_MEMBER_BYTES or len(raw) != member.file_size:
                        raise ValueError("ZIP member expanded size is inconsistent or exceeds the byte limit")
                    if raw.startswith(_ARCHIVE_MAGIC) or raw[257:262] == b"ustar":
                        raise ValueError("nested ZIP/archive content is unsupported")
                    text = _decode_text(raw, format, member_fallback=member_fallback)
                except _MemberTextCharacterLimit:
                    section.update(status="unexposed_text_limit",
                                   reason="member_text_character_limit", text_exposed=False)
                    continue
                except (zipfile.BadZipFile, UnicodeError, ValueError, OSError, RuntimeError, NotImplementedError) as exc:
                    raise ValueError(f"ZIP member {member.filename!r}: {exc}") from exc
                label = "\nArchive member: " + json.dumps(member.filename, ensure_ascii=False) + f" ({format})\n"
                if member_fallback and output.tell() + len(label) + len(text) + 1 > _MAX_TEXT_CHARS:
                    section.update(status="unexposed_text_limit",
                                   reason="archive_text_character_limit", text_exposed=False)
                    continue
                section.update(status="readable", start=output.tell())
                output.write(label)
                section["content_start"] = output.tell()
                output.write(text)
                section["content_end"] = output.tell()
                output.write("\n")
                section["end"] = output.tell()
                limit = min(_NAV_PREVIEW_CHARS, preview_budget)
                section["preview"] = text[:limit]
                section["preview_truncated"] = len(text) > limit
                preview_budget -= len(section["preview"])
                if output.tell() > _MAX_TEXT_CHARS:
                    raise ValueError("ZIP extracted text exceeds the character limit")
                readable += bool(text.strip())
    except (zipfile.BadZipFile, OSError, RuntimeError, NotImplementedError) as exc:
        raise ValueError("could not read this ZIP text export") from exc
    if not readable:
        raise ValueError("ZIP archive contains no supported readable text members (CSV, TSV, XML, JSON, TXT)")
    if sections is not None:
        sections.extend(navigation)
    return output.getvalue()
