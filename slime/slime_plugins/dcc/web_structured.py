"""Optional Web final schemas derived from public columns and read sources only."""

from __future__ import annotations

import inspect
import json
from copy import deepcopy

WEB_FINAL_SCHEMA_VERSION = "web-final-nonempty-row-read-citations-v2"
WEB_ACTION_SCHEMA_VERSION = "web-single-action-search-first-v1"
WEB_MULTI_CITATION_VERSION = "web-multi-read-citations-v1"


def web_multi_citation_configuration(enabled=False):
    """Opt in without changing the legacy schema or its recorded version."""
    if type(enabled) is not bool:
        raise ValueError("web_multi_citation must be a boolean")
    return {"enabled": enabled, "version": WEB_MULTI_CITATION_VERSION if enabled else None,
            "max_citations": 4 if enabled else 1}


def validate_web_multi_citation_records(records):
    values = [record.get("web_multi_citation", web_multi_citation_configuration(False))
              for record in records]
    if not values:
        raise ValueError("Web multi-citation requires configuration records")
    for value in values:
        if (not isinstance(value, dict) or type(value.get("enabled")) is not bool
                or value != web_multi_citation_configuration(value["enabled"])):
            raise ValueError("Web multi-citation requires complete supported configuration")
    if any(value != values[0] for value in values[1:]):
        raise ValueError("Web multi-citation differs between manifest, runtime and protocol")
    if values[0]["enabled"] and any(
            record.get("web_structured_final") != web_structured_configuration(True)
            or record.get("web_structured_actions") != web_structured_actions_configuration(True)
            or record.get("inference_backend", record.get("backend")) != "vllm"
            for record in records):
        raise ValueError("Web multi-citation requires matching structured Web actions/final and vLLM")
    return values[0]


def web_structured_configuration(enabled=False):
    if type(enabled) is not bool:
        raise ValueError("web_structured_final must be a boolean")
    return {"enabled": enabled, "schema_version": WEB_FINAL_SCHEMA_VERSION if enabled else None,
            "backend": "xgrammar" if enabled else None}


def web_structured_actions_configuration(enabled=False):
    if type(enabled) is not bool:
        raise ValueError("web_structured_actions must be a boolean")
    return {"enabled": enabled, "schema_version": WEB_ACTION_SCHEMA_VERSION if enabled else None,
            "backend": "xgrammar" if enabled else None}


def web_final_schema(columns, read_urls, *, multi_citation=False):
    if (not isinstance(columns, list) or not columns or
            any(not isinstance(column, str) or not column or column == "citations" for column in columns)):
        raise ValueError("Structured Web final requires declared nonempty public answer columns")
    columns = list(dict.fromkeys(columns))
    if not isinstance(read_urls, (list, tuple, set, frozenset)) or any(
            not isinstance(url, str) or not url for url in read_urls):
        raise ValueError("Structured Web citations require actual read URL strings")
    urls = sorted(set(read_urls))
    properties = {column: {"type": ["string", "number", "boolean", "null"]} for column in columns}
    properties["citations"] = {"type": "array", "minItems": 1,
                               "maxItems": web_multi_citation_configuration(multi_citation)["max_citations"],
                               "items": {"type": "string", **({"enum": urls} if urls else {})}}
    row = {"type": "object", "properties": properties,
           "required": [*columns, "citations"], "additionalProperties": False}
    # Match the existing evidence guard: a row needs at least one actual cell.
    # Other cells can remain unknown, and the empty table remains an abstention.
    # Expand complete object alternatives instead of intersecting properties
    # with anyOf: the installed xgrammar backend must enforce the same schema.
    alternatives = []
    for column in columns:
        alternative = deepcopy(row)
        alternative["properties"][column] = {
            "anyOf": [{"type": "string", "minLength": 1},
                      {"type": ["number", "boolean"]}],
        }
        alternatives.append(alternative)
    answer = {"type": "array", "items": {"anyOf": alternatives}}
    if not urls:
        # Do not fabricate a citation or use an invalid empty JSON-schema enum.
        answer["maxItems"] = 0
    return {"type": "object", "properties": {"answer": answer},
            "required": ["answer"], "additionalProperties": False}


def web_action_schema(columns, read_urls, *, successful_search, cached_rows=False, multi_citation=False):
    """One action per turn; only a real successful search opens other branches.

    Columns and citation URLs come from the same public/observed inputs as the
    final schema. Fetch URLs remain unrestricted strings; WebTools validates
    public network access when the model actually requests a read.
    """
    if type(successful_search) is not bool:
        raise ValueError("successful_search must be a boolean")
    if type(cached_rows) is not bool:
        raise ValueError("cached_rows must be a boolean")
    final = web_final_schema(columns, read_urls, multi_citation=multi_citation)
    search = {
        "type": "object", "properties": {
            "tool": {"const": "web_search"},
            "arguments": {"type": "object", "properties": {
                "query": {"type": "string"},
                "count": {"type": "integer", "minimum": 1, "maximum": 10},
            }, "required": ["query"], "additionalProperties": False},
        }, "required": ["tool", "arguments"], "additionalProperties": False,
    }
    if not successful_search:
        return search
    fetch = {
        "type": "object", "properties": {
            "tool": {"const": "fetch_url"},
            "arguments": {"type": "object", "properties": {
                "url": {"type": "string"},
                "start": {"type": "integer", "minimum": 0},
            }, "required": ["url"], "additionalProperties": False},
        }, "required": ["tool", "arguments"], "additionalProperties": False,
    }
    if cached_rows:
        from .web_rows import select_rows_schema

        fetch["properties"]["arguments"]["properties"]["select_rows"] = select_rows_schema()
    return {"anyOf": [search, fetch, final]}


def validate_web_structured_schema(schema):
    """Reject backend-unsupported exact keys instead of silently rewriting them."""
    try:
        import xgrammar

        xgrammar.Grammar.from_json_schema(json.dumps(schema))
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("Installed xgrammar cannot compile this Web final JSON schema; "
                         "public column keys and source URLs must remain exact") from exc


def preflight_web_structured(*, actions=False, explicit_review=False, cached_rows=False, multi_citation=False):
    """Check installed API/backend and compile a schema without creating an engine."""
    if type(cached_rows) is not bool or cached_rows and not actions:
        raise ValueError("Cached rows preflight requires structured actions")
    if type(explicit_review) is not bool or explicit_review and not actions:
        raise ValueError("Web explicit review preflight requires structured actions")
    try:
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        if "structured_outputs" not in inspect.signature(SamplingParams).parameters:
            raise ValueError("SamplingParams lacks structured_outputs")
        for columns, urls in ((["name"], []), (["name"], ["https://example.org/source"]),
                              (["中文"], ["https://example.org/source?name=%22"])):
            schemas = [web_final_schema(columns, urls, multi_citation=multi_citation)]
            if actions:
                schemas.extend(web_action_schema(columns, urls, successful_search=state, cached_rows=cached_rows,
                                                 multi_citation=multi_citation)
                               for state in (False, True))
            if explicit_review:
                from .web_review import web_review_schema

                schemas = [web_review_schema(schema) for schema in schemas]
            for schema in schemas:
                StructuredOutputsParams(json=schema, disable_fallback=True)
                validate_web_structured_schema(schema)
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("--web-structured-final requires vLLM StructuredOutputsParams and a working xgrammar JSON-schema backend") from exc
