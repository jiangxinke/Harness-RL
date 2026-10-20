"""Equivalent finite review-string grammar for the pinned xgrammar converter.

Only the converter's bounded review-string production changes. Tool, answer,
row, citation, key order and character rules retain their generated language.
The converter currently rejects JSON escapes in bounded strings; this module
preserves that limitation rather than silently widening the accepted language.
"""

from __future__ import annotations

import importlib.metadata
import json
from functools import lru_cache

WEB_REVIEW_DECODER_VERSION = "web-review-finite-right-recursive-v1"
WEB_REVIEW_XGRAMMAR_VERSION = "0.1.25"
_REVIEW_FIELD = {"type": "string", "minLength": 1, "maxLength": 240}
_PREFIX = "dcc_review_finite_"
# Filled from, and pinned to, the independently checked 0.1.25 converter.
_EXPECTED_STANDALONE = 'string ::= (("\\"" string_repeat_1{0, 236} string_repeat_2 string_repeat_3 string_repeat_4 string_repeat_5 "\\""))\nroot ::= ((string))\nstring_repeat_1 ::= (([^\\"\\\\\\r\\n])) (=(string_repeat_2 string_repeat_3 string_repeat_4 string_repeat_5))\nstring_repeat_2 ::= ("" | ([^\\"\\\\\\r\\n])) (=(string_repeat_3 string_repeat_4 string_repeat_5))\nstring_repeat_3 ::= ("" | ([^\\"\\\\\\r\\n])) (=(string_repeat_4 string_repeat_5))\nstring_repeat_4 ::= ("" | ([^\\"\\\\\\r\\n])) (=(string_repeat_5))\nstring_repeat_5 ::= (([^\\"\\\\\\r\\n])) (=("\\""))\n'


def web_review_decoder_configuration():
    """Check the installed converter before an engine can be launched."""
    actual = importlib.metadata.version("xgrammar")
    if actual != WEB_REVIEW_XGRAMMAR_VERSION:
        raise ValueError("Bounded review decoding requires xgrammar==" + WEB_REVIEW_XGRAMMAR_VERSION)
    return {"version": WEB_REVIEW_DECODER_VERSION,
            "format": "ebnf", "algorithm": "finite-right-recursive-string",
            "xgrammar_version": actual, "min_chars": 1, "max_chars": 240,
            "character_language": "unchanged pinned JSON-schema converter production",
            "disable_fallback": True}


def is_web_review_schema(schema):
    """Recognize only the exact top-level wrapper produced by web_review_schema.

    An ordinary schema returns False. A partially present or changed review
    wrapper raises instead of being routed through a different decoder.
    """
    branches = schema.get("anyOf") if isinstance(schema, dict) and set(schema) == {"anyOf"} else [schema]
    if not isinstance(branches, list) or not branches:
        raise ValueError("Expected a nonempty Web action schema")
    present = [isinstance(branch, dict) and isinstance(branch.get("properties"), dict)
               and bool(set(branch["properties"]) & {"plan", "checks"}) for branch in branches]
    if not any(present):
        return False
    for branch in branches:
        properties = branch.get("properties") if isinstance(branch, dict) else None
        if (not isinstance(properties, dict) or list(properties)[:2] != ["plan", "checks"]
                or branch.get("required", [])[:2] != ["plan", "checks"]
                or any(properties.get(key) != _REVIEW_FIELD for key in ("plan", "checks"))):
            raise ValueError("Unsupported review wrapper; preserve exact plan/checks bounds and order")
    return True


def web_review_grammar(schema):
    """Return an equivalent grammar, failing closed on converter/schema changes."""
    web_review_decoder_configuration()
    if not is_web_review_schema(schema):
        raise ValueError("Finite review decoding requires the exact review schema")
    return _cached_review_grammar(json.dumps(schema, ensure_ascii=False, separators=(",", ":")))


@lru_cache(maxsize=128)
def _cached_review_grammar(serialized_schema):
    import xgrammar

    standalone = str(xgrammar.Grammar.from_json_schema(_REVIEW_FIELD))
    if standalone != _EXPECTED_STANDALONE:
        raise ValueError("Pinned xgrammar bounded-string converter shape has changed")
    grammar = str(xgrammar.Grammar.from_json_schema(serialized_schema))
    lines = grammar.splitlines()
    expected = standalone.splitlines()
    bounded = expected[0]
    helpers = [line for line in expected if line.startswith("string_repeat_")]
    if (len(helpers) != 5 or lines.count(bounded) != 1
            or any(lines.count(line) != 1 for line in helpers)
            or _PREFIX in grammar):
        raise ValueError("Unexpected bounded review grammar naming, shape or prefix collision")
    # Copy the pinned converter's character class, including its escape limits.
    character = helpers[-1].split("::= ", 1)[1].split(" (=", 1)[0]
    replacement = 'string ::= ("\\\"" ' + _PREFIX + 'char ' + _PREFIX + 'tail_239 "\\\"")'
    lines[lines.index(bounded)] = replacement
    additions = [_PREFIX + "char ::= " + character, _PREFIX + 'tail_0 ::= ""']
    additions.extend(_PREFIX + f'tail_{count} ::= ("" | (' + _PREFIX + "char "
                     + _PREFIX + f"tail_{count - 1}))" for count in range(1, 240))
    return "\n".join([*lines, *additions]) + "\n"


def preflight_web_review_decoder(*, cached_rows=False, multi_citation=False):
    """Validate API/converter compatibility without constructing a model engine."""
    import xgrammar
    from vllm.sampling_params import StructuredOutputsParams
    from .web_review import web_review_schema
    from .web_structured import web_action_schema, web_final_schema

    configuration = web_review_decoder_configuration()
    for columns, urls in ((["name"], []), (["name"], ["https://example.org/source"]),
                          (["中文"], ["https://example.org/source?name=%22"])):
        schemas = [web_final_schema(columns, urls, multi_citation=multi_citation)]
        schemas.extend(web_action_schema(columns, urls, successful_search=state, cached_rows=cached_rows, multi_citation=multi_citation) for state in (False, True))
        for schema in schemas:
            grammar = web_review_grammar(web_review_schema(schema))
            xgrammar.Grammar.from_ebnf(grammar)
            StructuredOutputsParams(grammar=grammar, disable_fallback=True)
    return configuration
