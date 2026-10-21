"""Exact public CUAD/Web schema admission for the native HF runtime.

Web columns and read-URL enums are recovered only to round-trip through the
existing public schema constructors. Their constraints and the pinned review
and whitespace grammars are never replaced by a looser backend schema.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json

from .cuad_structured import cuad_backend_schema, is_cuad_structured_schema
from .web_review import web_review_schema
from .web_review_decoder import is_web_review_schema, web_review_grammar
from .web_structured import validate_web_structured_schema, web_action_schema, web_final_schema
from .web_whitespace import web_bounded_whitespace_grammar


def _same_schema(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _web_review_mode(schema):
    reviewed = is_web_review_schema(schema)
    bare = deepcopy(schema)
    if reviewed:
        branches = bare["anyOf"] if set(bare) == {"anyOf"} else [bare]
        for branch in branches:
            branch["properties"] = {key: value for key, value in branch["properties"].items()
                                    if key not in {"plan", "checks"}}
            branch["required"] = branch["required"][2:]
    wrap = web_review_schema if reviewed else lambda value: value
    # Before a successful search the public schema has no columns/read URLs.
    if _same_schema(schema, wrap(web_action_schema(["column"], [], successful_search=False))):
        return reviewed
    branches = bare["anyOf"] if set(bare) == {"anyOf"} else [bare]
    final = branches[-1]
    row = final["properties"]["answer"]["items"]["anyOf"][0]
    columns = [key for key in row["properties"] if key != "citations"]
    citations = row["properties"]["citations"]
    if type(citations["maxItems"]) is not int or citations["maxItems"] not in {1, 4}:
        raise ValueError("Unsupported Web citation limit")
    urls = citations["items"].get("enum", [])
    multi = citations["maxItems"] == 4
    expected = [web_final_schema(columns, urls, multi_citation=multi)]
    expected.extend(web_action_schema(columns, urls, successful_search=True,
                    cached_rows=cached, multi_citation=multi) for cached in (False, True))
    if not any(_same_schema(schema, wrap(candidate)) for candidate in expected):
        raise ValueError("Web schema differs from the public constructors")
    return reviewed


def native_structured_schema_kind(schema):
    """Fail closed before model access; recognize only exact public schemas."""
    if is_cuad_structured_schema(schema):
        return "cuad"
    try:
        kind = "web-review" if _web_review_mode(schema) else "web"
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
        raise ValueError("HF native runtime supports exact public CUAD/Web schemas only") from error
    _validate_web_backend_schema(json.dumps(schema, ensure_ascii=False, allow_nan=False))
    return kind


@lru_cache(maxsize=128)
def _validate_web_backend_schema(serialized):
    # Match the existing Web backend: unsupported exact keys fail before a
    # thought is sampled, rather than being silently renamed or discarded.
    validate_web_structured_schema(json.loads(serialized))


def compile_native_structured_schema(compiler, schema, *, web_bounded_whitespace=False):
    """Reuse the existing backend policy, retaining original-schema validation."""
    kind = native_structured_schema_kind(schema)
    if kind == "cuad":
        return compiler.compile_json_schema(cuad_backend_schema(schema))
    if web_bounded_whitespace:
        return compiler.compile_grammar(web_bounded_whitespace_grammar(
            schema, explicit_review=kind == "web-review"))
    if kind == "web-review":
        return compiler.compile_grammar(web_review_grammar(schema))
    return compiler.compile_json_schema(schema)
