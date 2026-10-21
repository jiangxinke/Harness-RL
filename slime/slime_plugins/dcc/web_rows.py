"""Optional bounded selection of literal records in an already downloaded source.

This module has no network, model, benchmark, or reference-answer access. Source
offsets always address the cached extracted text, including worksheet/member
labels. Numeric cells retain their original JSON spelling.
"""
from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

WEB_CACHED_ROWS_VERSION = "cached-literal-record-windows-v3"
WEB_CACHED_ROWS_LIMITS = {"source_chars": 5_000_000, "records": 100_000, "nodes": 300_000,
                          "depth": 32, "columns": 32, "predicates": 8, "returned_rows": 64}
WEB_CACHED_ROWS_INSTRUCTIONS = (
    '\nFor structured tables, prefer cached row selection to scanning successive text windows. '
    'First fetch the source URL normally. Once its actual header and table location are known, '
    'call fetch_url on the SAME URL with select_rows to retrieve the requested entities and '
    'columns, even when their rows lie beyond the current text window. This filters the full '
    'bounded downloaded cache without another network request. A truncated text window does '
    'not mean the downloaded cache ends there. '
    'For a ZIP or workbook, use the exact observed archive_member or worksheet name. If the '
    'header is not visible, first read that member or worksheet at its observed start; set '
    'header_row to the zero-based CSV record containing column names, counting title, preamble '
    'and blank records from the start of that table section. Request only the exact identity and answer columns needed, '
    'including the requested year columns of a wide table. Use an eq predicate on an observed '
    'entity/code column to locate each requested entity. where predicates are ANDed, so use '
    'separate select_rows calls for alternative entity values; do not AND two different values '
    'of the same column. Set after=next_after only to continue the same cached selection, '
    'without a text start offset. '
    'Keep reasoning focused on the next source operation; do not copy long CSV rows, enumerate '
    'unrelated values or continue a numeric sequence in thought. Inspect returned literal cells '
    'and provenance before answering. If selection fails or the table structure is unsupported, '
    'inspect the diagnostic and correct the arguments or continue reading/documented API access; '
    'selection is a preference, not a restriction. It adds no source facts or completeness proof. '
    'The optional fetch_url argument select_rows accepts format '
    '(csv, tsv or json), exact columns, optional where predicates, limit (1..64), and after '
    '(zero-based data-record cursor). CSV/TSV use an explicit zero-based header_row (default 0), '
    'exact header names and optional exact worksheet or archive_member from observed metadata. '
    'CSV/TSV reject empty header names by default. Explicitly set '
    'csv_trailing_empty_columns=verified_empty_padding only to verify and exclude a suffix of '
    'empty header names whose cells are strictly empty in every cached data record. The full '
    'bounded table must retain the original record width; inner empty names, duplicate names, '
    'nonempty padding cells and malformed records still fail. Spaces, null and 0 are not empty. '
    'Literal source text and offsets, including the padding, are preserved. '
    'JSON uses an explicit RFC 6901 path to an array (default empty pointer for a root array), '
    'and columns/predicate column are RFC 6901 pointers relative to each record, e.g. /name. '
    'Predicates are ANDed: eq/ne compare exact strings, booleans or null without coercion; '
    'contains/starts_with/lex_ge/lex_le require strings; num_eq/num_ge/num_le explicitly compare '
    'finite decimal spellings, and exists tests presence without a value. Numeric source cells '
    'default to strict errors; explicitly set numeric_source_policy=exclude_uncomparable to '
    'exclude present source scalars that cannot be compared as finite decimals, without '
    'interpreting ranges or guessing values. Invalid predicate arguments still fail. '
    'Exclusion does not prove a record fails an original task condition; if '
    'uncomparable_records_after is positive, inspect those source records separately. '
    'The opt-in reports independent per-predicate counts; predicates remain ANDed, so individual '
    'matches need not share a matching record. Counts are navigation metadata, not row evidence. '
    'Use ne to exclude an exact scalar value. '
    'No regex, joins, '
    'guessed columns, remote pagination or inferred filters. Each returned row is an actual '
    'literal source record with its own start/end/text and selected cells. The primary text '
    'window covers only the first returned row; other rows are separate observed windows. '
    'Headers, selector metadata and empty results establish no data row. Missing cells are '
    'marked absent, never filled. next_after continues this cached selection only; it does not '
    'prove the remote dataset is complete. Check every original public condition yourself.'
)


def web_cached_rows_configuration(enabled=False):
    if type(enabled) is not bool:
        raise ValueError("web_cached_rows must be a boolean")
    configuration = {"enabled": enabled, "version": WEB_CACHED_ROWS_VERSION if enabled else None,
            "scope": "structured-web-evaluation" if enabled else None,
            "cache": "episode-downloaded-extracted-text-only" if enabled else None,
            "output": "literal-record-windows-with-cell-provenance" if enabled else None,
            "budget": "original-max-page-chars-entire-json-observation" if enabled else None,
            "limits": dict(WEB_CACHED_ROWS_LIMITS) if enabled else None}
    if enabled:
        configuration["csv_trailing_empty_columns"] = {
            "default": "reject", "supported": ["reject", "verified_empty_padding"],
            "formats": ["csv", "tsv"],
            "validation": "full-bounded-table-original-width-and-strict-empty-tail-cells",
            "source_text_and_offsets": "unchanged"}
    return configuration


def validate_web_cached_rows_records(records):
    from .web_structured import web_structured_actions_configuration, web_structured_configuration

    values = [record.get("web_cached_rows", web_cached_rows_configuration(False)) for record in records]
    for value in values:
        if (not isinstance(value, dict) or type(value.get("enabled")) is not bool
                or value != web_cached_rows_configuration(value["enabled"])
                or value["enabled"] and (not isinstance(value.get("limits"), dict)
                    or any(type(v) is not int for v in value["limits"].values()))):
            raise ValueError("Web cached rows require complete supported configuration")
    if any(value != values[0] for value in values[1:]):
        raise ValueError("Web cached rows differ between manifest, runtime and protocol")
    if values[0]["enabled"] and any(
            record.get("web_structured_final") != web_structured_configuration(True)
            or record.get("web_structured_actions") != web_structured_actions_configuration(True)
            or record.get("inference_backend", record.get("backend")) != "vllm"
            for record in records):
        raise ValueError("Web cached rows require matching structured Web actions/final and vLLM")
    return values[0]


def select_rows_schema():
    return {"type": "object", "properties": {
        "format": {"type": "string", "enum": ["csv", "tsv", "json"]},
        "columns": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 32},
        "where": {"type": "array", "maxItems": 8, "items": {"type": "object", "properties": {
            "column": {"type": "string"},
            "op": {"type": "string", "enum": ["eq", "ne", "contains", "starts_with", "lex_ge", "lex_le",
                                                   "num_eq", "num_ge", "num_le", "exists"]},
            "value": {"type": ["string", "boolean", "null"]},
        }, "required": ["column", "op"], "additionalProperties": False}},
        "path": {"type": "string"}, "worksheet": {"type": "string"},
        "numeric_source_policy": {"type": "string", "enum": ["error", "exclude_uncomparable"]},
        "csv_trailing_empty_columns": {"type": "string", "enum": ["reject", "verified_empty_padding"]},
        "archive_member": {"type": "string"}, "header_row": {"type": "integer", "minimum": 0},
        "after": {"type": "integer", "minimum": 0},
        "limit": {"type": "integer", "minimum": 1, "maximum": 64},
    }, "required": ["format", "columns"], "additionalProperties": False}


_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")
_JSON_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_MISSING = object()
_UNCOMPARABLE = object()


def _pointer(value):
    if not isinstance(value, str) or len(value) > 2048 or value and not value.startswith("/"):
        raise ValueError("JSON paths must be explicit RFC 6901 pointers")
    parts = value.split("/")[1:]
    if any(re.search(r"~(?:[^01]|$)", part) for part in parts):
        raise ValueError("Invalid RFC 6901 pointer escape")
    return [part.replace("~1", "/").replace("~0", "~") for part in parts]


def _request(request):
    allowed = set(select_rows_schema()["properties"])
    if not isinstance(request, dict) or set(request) - allowed:
        raise ValueError("select_rows requires only the documented arguments")
    # JSON escapes in arguments can decode to unpaired surrogate code points.
    # Reject them rather than returning observations a tokenizer cannot encode.
    def safe_string(value):
        return isinstance(value, str) and not any(0xD800 <= ord(c) <= 0xDFFF for c in value)
    kind, columns = request.get("format"), request.get("columns")
    if not isinstance(kind, str) or kind not in {"csv", "tsv", "json"}:
        raise ValueError("select_rows format must be csv, tsv or json")
    numeric_source_policy = request.get("numeric_source_policy", "error")
    if (not isinstance(numeric_source_policy, str)
            or numeric_source_policy not in {"error", "exclude_uncomparable"}):
        raise ValueError("numeric_source_policy must be error or exclude_uncomparable")
    trailing_empty_columns = request.get("csv_trailing_empty_columns", "reject")
    if (not isinstance(trailing_empty_columns, str)
            or trailing_empty_columns not in {"reject", "verified_empty_padding"}):
        raise ValueError("csv_trailing_empty_columns must be reject or verified_empty_padding")
    if (not isinstance(columns, list) or not 1 <= len(columns) <= WEB_CACHED_ROWS_LIMITS["columns"]
            or any(not safe_string(c) or len(c) > 2048 for c in columns) or len(set(columns)) != len(columns)):
        raise ValueError("select_rows requires distinct exact columns within the column limit")
    for key, minimum, maximum in (("after", 0, 100_000), ("header_row", 0, 100_000), ("limit", 1, 64)):
        value = request.get(key, 32 if key == "limit" else 0)
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"select_rows {key} is outside its integer bounds")
    if "worksheet" in request and "archive_member" in request:
        raise ValueError("Select one worksheet or one archive member, not both")
    for key in ("worksheet", "archive_member"):
        if key in request and (not safe_string(request[key]) or not request[key] or len(request[key]) > 1024):
            raise ValueError(f"select_rows {key} must be an exact nonempty observed name")
    if kind == "json":
        if "csv_trailing_empty_columns" in request:
            raise ValueError("JSON selection does not accept a CSV/TSV header padding policy")
        if "header_row" in request or "worksheet" in request:
            raise ValueError("JSON selection does not accept a CSV header or worksheet")
        if not safe_string(request.get("path", "")):
            raise ValueError("JSON path requires encodable exact text")
        _pointer(request.get("path", ""))
        for column in columns:
            _pointer(column)
    elif "path" in request:
        raise ValueError("CSV/TSV selection does not accept a JSON path")
    predicates = request.get("where", [])
    if not isinstance(predicates, list) or len(predicates) > WEB_CACHED_ROWS_LIMITS["predicates"]:
        raise ValueError("select_rows where exceeds the predicate limit")
    ops = set(select_rows_schema()["properties"]["where"]["items"]["properties"]["op"]["enum"])
    for predicate in predicates:
        if (not isinstance(predicate, dict) or set(predicate) - {"column", "op", "value"}
                or not safe_string(predicate.get("column")) or len(predicate["column"]) > 2048
                or not isinstance(predicate.get("op"), str) or predicate["op"] not in ops):
            raise ValueError("Invalid exact-column select_rows predicate")
        if kind == "json":
            _pointer(predicate["column"])
        op = predicate["op"]
        if isinstance(predicate.get("value"), str) and not safe_string(predicate["value"]):
            raise ValueError("Predicate value requires encodable exact text")
        if op == "exists":
            if "value" in predicate:
                raise ValueError("exists does not accept a value")
        elif "value" not in predicate or not (predicate["value"] is None or type(predicate["value"]) in {str, bool}):
            raise ValueError("Predicate requires an exact string, boolean or null value")
        elif op not in {"eq", "ne"} and not isinstance(predicate["value"], str):
            raise ValueError("String and explicit numeric predicates require string values")
        if op.startswith("num_"):
            _decimal(predicate["value"])
    return kind, columns, predicates


def _decimal(value):
    if not isinstance(value, str) or len(value) > 256 or _NUMBER.fullmatch(value) is None:
        raise ValueError("Explicit numeric comparisons require finite JSON decimal spellings")
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("Invalid explicit decimal value") from exc
    if not number.is_finite() or abs(number.as_tuple().exponent) > 10000:
        raise ValueError("Explicit decimal exceeds finite exponent bounds")
    return number


@dataclass(slots=True)
class _Node:
    start: int
    end: int
    kind: str
    value: object


class _JSON:
    """Strict bounded parser retaining source spans and exact numeric lexemes."""
    def __init__(self, text):
        self.text, self.pos, self.nodes = text, 0, 0

    def _space(self):
        while self.pos < len(self.text) and self.text[self.pos] in " \t\r\n":
            self.pos += 1

    def parse(self, depth=0):
        self._space()
        start = self.pos
        self.nodes += 1
        if depth > WEB_CACHED_ROWS_LIMITS["depth"] or self.nodes > WEB_CACHED_ROWS_LIMITS["nodes"]:
            raise ValueError("JSON structure exceeds bounded depth or node limits")
        if start >= len(self.text):
            raise ValueError("Incomplete JSON source")
        character = self.text[start]
        if character in "[{":
            self.pos += 1
            self._space()
            object_kind = character == "{"
            values = {} if object_kind else []
            close = "}" if object_kind else "]"
            if self.pos < len(self.text) and self.text[self.pos] == close:
                self.pos += 1
                return _Node(start, self.pos, "object" if object_kind else "array", values)
            while True:
                if object_kind:
                    key = self.parse(depth + 1)
                    if key.kind != "string" or key.value in values:
                        raise ValueError("JSON objects require distinct string keys; duplicate keys are unsupported")
                    self._space()
                    if self.pos >= len(self.text) or self.text[self.pos] != ":":
                        raise ValueError("JSON object is missing a colon")
                    self.pos += 1
                    values[key.value] = self.parse(depth + 1)
                else:
                    values.append(self.parse(depth + 1))
                self._space()
                if self.pos >= len(self.text):
                    raise ValueError("Incomplete JSON source")
                delimiter = self.text[self.pos]
                self.pos += 1
                if delimiter == close:
                    break
                if delimiter != ",":
                    raise ValueError("Invalid JSON record delimiter")
            return _Node(start, self.pos, "object" if object_kind else "array", values)
        if character == '"':
            try:
                value, self.pos = json.JSONDecoder().raw_decode(self.text, start)
            except (ValueError, RecursionError) as exc:
                raise ValueError("Invalid JSON string") from exc
            if any(0xD800 <= ord(c) <= 0xDFFF for c in value):
                raise ValueError("JSON string contains an unpaired Unicode surrogate")
            return _Node(start, self.pos, "string", value)
        for literal, kind, value in (("true", "boolean", True), ("false", "boolean", False), ("null", "null", None)):
            if self.text.startswith(literal, start):
                self.pos += len(literal)
                return _Node(start, self.pos, kind, value)
        match = _JSON_NUMBER.match(self.text, start)
        if match is None:
            raise ValueError("Invalid JSON scalar; nonfinite values are unsupported")
        self.pos = match.end()
        return _Node(start, self.pos, "number", match[0])

    def document(self):
        if self.text.startswith("\ufeff"):
            self.pos = 1
        node = self.parse()
        self._space()
        if self.pos != len(self.text):
            raise ValueError("Trailing or malformed JSON source")
        return node


def _resolve(node, pointer):
    for part in _pointer(pointer):
        if node.kind == "object":
            node = node.value.get(part, _MISSING)
        elif node.kind == "array" and re.fullmatch(r"0|[1-9][0-9]*", part):
            index = int(part) if len(part) < 12 else len(node.value)
            node = node.value[index] if index < len(node.value) else _MISSING
        else:
            return _MISSING
        if node is _MISSING:
            return node
    return node


def _cell(node):
    if node is _MISSING:
        return {"present": False}
    if node.kind not in {"string", "number", "boolean", "null"}:
        raise ValueError("Selected cells and predicate operands must be scalar; use an exact child pointer")
    return {"present": True, "kind": node.kind, "value": node.value}


def _matches(cell, predicate, *, numeric_source_policy="error"):
    if predicate["op"] == "exists":
        return cell["present"]
    if not cell["present"]:
        return False
    op, value, actual = predicate["op"], predicate["value"], cell["value"]
    if op in {"eq", "ne"}:
        if cell["kind"] == "number":
            raise ValueError("JSON numeric cells require an explicit num_* comparison")
        equal = type(actual) is type(value) and actual == value
        return equal if op == "eq" else not equal
    if op.startswith("num_"):
        # Selector arguments remain strict, independent of the source policy.
        # _request validates these before parsing any source, even an empty one.
        right = _decimal(value)
        try:
            if cell["kind"] not in {"string", "number"}:
                raise ValueError("Explicit numeric predicate encountered a non-numeric cell kind")
            left = _decimal(actual)
        except ValueError:
            if numeric_source_policy == "exclude_uncomparable":
                return _UNCOMPARABLE
            raise
        return left == right if op == "num_eq" else left >= right if op == "num_ge" else left <= right
    if cell["kind"] != "string":
        raise ValueError("String predicate encountered a non-string source cell")
    return (value in actual if op == "contains" else actual.startswith(value) if op == "starts_with"
            else actual >= value if op == "lex_ge" else actual <= value)


class _Lines:
    def __init__(self, text):
        self.stream = io.StringIO(text, newline="")
    def __iter__(self):
        return self
    def __next__(self):
        line = self.stream.readline()
        if not line:
            raise StopIteration
        return line


def _csv_records(text, kind, header_index, *, trailing_empty_columns="reject"):
    lines = _Lines(text)
    reader = csv.reader(lines, delimiter="," if kind == "csv" else "\t", strict=True)
    header, records, header_span, source_width = None, [], None, None
    try:
        for index in range(WEB_CACHED_ROWS_LIMITS["records"] + header_index + 2):
            start = lines.stream.tell()
            try:
                values = next(reader)
            except StopIteration:
                break
            end = lines.stream.tell()
            if index < header_index:
                continue
            if index == header_index:
                source_width = len(values)
                content_width = source_width
                if trailing_empty_columns == "verified_empty_padding":
                    while content_width and values[content_width - 1] == "":
                        content_width -= 1
                content_header = values[:content_width]
                if (not content_header or source_width > 4096 or any(not c for c in content_header)
                        or len(set(content_header)) != len(content_header)):
                    raise ValueError("CSV header must contain distinct nonempty exact column names")
                header, header_span = content_header, (start, end)
            else:
                if len(records) >= WEB_CACHED_ROWS_LIMITS["records"]:
                    raise ValueError("CSV source exceeds the bounded data-record limit")
                if len(values) != source_width:
                    raise ValueError("CSV record width differs from the explicit header; no cells were filled")
                if any(value != "" for value in values[len(header):]):
                    raise ValueError("CSV trailing empty header columns contain a nonempty data cell")
                records.append((start, end, dict(zip(header, values))))
    except csv.Error as exc:
        raise ValueError("CSV source is malformed or exceeds the parser's field limit") from exc
    if header is None:
        raise ValueError("Explicit CSV header_row is beyond this cached source")
    return header, header_span, records


def _section(text, request, worksheets, archives):
    base, end = 0, len(text)
    if worksheets:
        if request["format"] != "csv":
            raise ValueError("Observed workbook sections use the existing CSV extraction format")
        if "worksheet" not in request:
            raise ValueError("Workbook selection requires an explicit observed worksheet name")
        matches = [item for item in worksheets if item.get("name") == request["worksheet"]]
        if len(matches) != 1 or matches[0].get("name_truncated"):
            raise ValueError("Worksheet name is missing, ambiguous or truncated")
        item = matches[0]
        label = "\nWorksheet: " + json.dumps(item["name"], ensure_ascii=False) + "\n"
        base, end = item.get("start"), item.get("end")
        if type(base) is not int or type(end) is not int or not 0 <= base < end <= len(text) or not text.startswith(label, base):
            raise ValueError("Worksheet boundaries do not match the cached source")
        base += len(label)
    elif "worksheet" in request:
        raise ValueError("This cached source has no observed worksheet boundaries")
    if archives:
        if "archive_member" not in request:
            raise ValueError("Archive selection requires an explicit observed archive_member")
        matches = [item for item in archives if item.get("name") == request["archive_member"]]
        if (len(matches) != 1 or matches[0].get("name_truncated") or matches[0].get("status") != "readable"
                or matches[0].get("format") != request["format"]):
            raise ValueError("Archive member is missing, ambiguous, unreadable or has a different format")
        base, end = matches[0].get("content_start"), matches[0].get("content_end")
        if type(base) is not int or type(end) is not int or not 0 <= base <= end <= len(text):
            raise ValueError("Archive member boundaries do not match the cached source")
    elif "archive_member" in request:
        raise ValueError("This cached source has no observed archive-member boundaries")
    return base, end


def select_cached_rows(text, request, *, worksheets=(), archives=()):
    """Validate the full bounded table, then expose only actual matching records.

    Returned cells are exact decoded strings/scalars; number values are source
    lexemes, never floats. Each record's literal text remains the primary proof.
    The caller applies the original observation-size bound and read telemetry.
    """
    kind, columns, predicates = _request(request)
    numeric_source_policy = request.get("numeric_source_policy", "error")
    diagnostics = numeric_source_policy == "exclude_uncomparable"
    match_counts, uncomparable_counts = [0] * len(predicates), [0] * len(predicates)
    uncomparable_records = 0
    if not isinstance(text, str) or len(text) > WEB_CACHED_ROWS_LIMITS["source_chars"]:
        raise ValueError("Cached source exceeds the bounded selector character limit")
    base, end = _section(text, request, worksheets, archives)
    source = text[base:end]
    bom = 1 if source.startswith("\ufeff") and kind != "json" else 0
    if bom:
        base += 1
        source = source[1:]
    header = None
    if kind == "json":
        root = _JSON(source).document()
        array = _resolve(root, request.get("path", ""))
        if array is _MISSING or array.kind != "array":
            raise ValueError("Explicit JSON path must resolve to an actual array")
        if len(array.value) > WEB_CACHED_ROWS_LIMITS["records"]:
            raise ValueError("JSON source exceeds the bounded record limit")
        rows = [(node.start, node.end, node) for node in array.value]
        def cell(record, column):
            node = _resolve(record, column)
            value = _cell(node)
            if value["present"]:
                value.update(start=base + node.start, end=base + node.end)
            return value
    else:
        headers, span, rows = _csv_records(source, kind, request.get("header_row", 0),
                                         trailing_empty_columns=request.get("csv_trailing_empty_columns", "reject"))
        requested = set(columns) | {predicate["column"] for predicate in predicates}
        if requested - set(headers):
            raise ValueError("Requested exact CSV column is absent from the explicit header")
        header = {"start": base + span[0], "end": base + span[1], "text": source[slice(*span)],
                  "role": "column-navigation-only;not-a-data-record"}
        def cell(record, column):
            return {"present": True, "kind": "string", "value": record[column]}
    selected, matching = [], 0
    after = request.get("after", 0)
    if after > len(rows):
        raise ValueError("Cached selection after cursor is beyond this table")
    for index, (start, end, record) in enumerate(rows):
        # Validate every requested scalar, including before cursor and rejected
        # rows. A later malformed row must not make a partial table look valid.
        cells = {column: cell(record, column) for column in columns}
        checks = [_matches(cell(record, predicate["column"]), predicate,
                           numeric_source_policy=numeric_source_policy) for predicate in predicates]
        if index >= after and diagnostics:
            for position, check in enumerate(checks):
                match_counts[position] += check is True
                uncomparable_counts[position] += check is _UNCOMPARABLE
            uncomparable_records += any(check is _UNCOMPARABLE for check in checks)
        if index >= after and all(check is True for check in checks):
            matching += 1
            if len(selected) <= request.get("limit", 32):
                selected.append({"record_index": index, "start": base + start, "end": base + end,
                                 "text": source[start:end], "cells": cells})
    result = {"version": WEB_CACHED_ROWS_VERSION, "kind": "literal-record-windows", "format": kind,
            "source_start": base, "source_end": base + len(source), "columns": list(columns),
            "after": after, "candidate_records": len(rows), "matching_records_after": matching,
            "header": header, "records": selected,
            "note": "Each record is a separate actually observed literal source window. Cells are exact decoded source scalars; numeric values retain their JSON lexeme. Absent fields are not filled. Selection completeness applies only to this cached source, never to a remote dataset or undocumented API pagination."}
    if diagnostics:
        result.update(numeric_source_policy=numeric_source_policy,
                      uncomparable_records_after=uncomparable_records,
                      predicate_diagnostics={
                          "scope": "cached-records-at-or-after-cursor", "semantics": "AND",
                          "items": [{"predicate_index": index, "match_count": match_counts[index],
                                     "uncomparable_count": uncomparable_counts[index]}
                                    for index in range(len(predicates))],
                          "note": "Independent counts need not share a matching record. Uncomparable counts cover present numeric operands without an accepted finite decimal spelling; ranges are not interpreted, missing cells remain nonmatches. Counts are navigation metadata, not observed row evidence or remote completeness."})
    if kind != "json" and request.get("csv_trailing_empty_columns") == "verified_empty_padding":
        result["csv_trailing_empty_columns"] = "verified_empty_padding"
    return result
