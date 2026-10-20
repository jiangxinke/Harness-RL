"""Optional Web final schemas derived from public columns and read sources only."""

from __future__ import annotations

import inspect
import json
from copy import deepcopy

WEB_FINAL_SCHEMA_VERSION = "web-final-nonempty-row-read-citations-v2"
WEB_ACTION_SCHEMA_VERSION = "web-single-action-search-first-v1"


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


def web_final_schema(columns, read_urls):
    if (not isinstance(columns, list) or not columns or
            any(not isinstance(column, str) or not column or column == "citations" for column in columns)):
        raise ValueError("Structured Web final requires declared nonempty public answer columns")
    columns = list(dict.fromkeys(columns))
    if not isinstance(read_urls, (list, tuple, set, frozenset)) or any(
            not isinstance(url, str) or not url for url in read_urls):
        raise ValueError("Structured Web citations require actual read URL strings")
    urls = sorted(set(read_urls))
    properties = {column: {"type": ["string", "number", "boolean", "null"]} for column in columns}
    properties["citations"] = {"type": "array", "minItems": 1, "maxItems": 1,
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


def web_action_schema(columns, read_urls, *, successful_search, cached_rows=False):
    """One action per turn; only a real successful search opens other branches.

    Columns and citation URLs come from the same public/observed inputs as the
    final schema. Fetch URLs remain unrestricted strings; WebTools validates
    public network access when the model actually requests a read.
    """
    if type(successful_search) is not bool:
        raise ValueError("successful_search must be a boolean")
    if type(cached_rows) is not bool:
        raise ValueError("cached_rows must be a boolean")
    final = web_final_schema(columns, read_urls)
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


def preflight_web_structured(*, actions=False, explicit_review=False, cached_rows=False):
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
            schemas = [web_final_schema(columns, urls)]
            if actions:
                schemas.extend(web_action_schema(columns, urls, successful_search=state, cached_rows=cached_rows)
                               for state in (False, True))
            if explicit_review:
                from .web_review import web_review_schema

                schemas = [web_review_schema(schema) for schema in schemas]
            for schema in schemas:
                StructuredOutputsParams(json=schema, disable_fallback=True)
                validate_web_structured_schema(schema)
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("--web-structured-final requires vLLM StructuredOutputsParams and a working xgrammar JSON-schema backend") from exc
