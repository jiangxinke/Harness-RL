"""Optional finite insignificant JSON whitespace for structured Web decoding.

The switch changes formatting, never parsed JSON values or the Web schema and
its guards. It preserves the pinned converter's existing space/LF/TAB language
and finite-review string rules, including their existing escape limitations.
"""

from __future__ import annotations

import importlib.metadata
import json
from functools import lru_cache

WEB_WHITESPACE_VERSION = "web-json-bounded-whitespace-v1"
WEB_WHITESPACE_XGRAMMAR_VERSION = "0.1.25"
WEB_WHITESPACE_MAX_CHARS = 8
_PREFIX = "dcc_json_ws_"
_WHITESPACE_CLASS = r"[ \n\t]"


def web_bounded_whitespace_configuration(enabled=False):
    """Portable provenance validation does not require an installed GPU runtime."""
    if type(enabled) is not bool:
        raise ValueError("web_bounded_whitespace must be a boolean")
    return {"version": WEB_WHITESPACE_VERSION, "enabled": enabled,
            "max_whitespace_chars": WEB_WHITESPACE_MAX_CHARS,
            "whitespace_characters": ["space", "LF", "TAB"],
            "format": "finite-right-recursive-ebnf", "xgrammar_version": WEB_WHITESPACE_XGRAMMAR_VERSION,
            "json_values_and_schema": "unchanged", "disable_fallback": True}


def validate_web_whitespace_records(records):
    """Reject omitted, changed or mixed formatting provenance across a run."""
    from .web_structured import web_structured_configuration

    values = [record.get("web_bounded_whitespace", web_bounded_whitespace_configuration(False))
              for record in records]
    for value in values:
        if (not isinstance(value, dict) or type(value.get("enabled")) is not bool
                or value != web_bounded_whitespace_configuration(value["enabled"])):
            raise ValueError("Web bounded whitespace requires complete supported configuration")
    if any(value != values[0] for value in values[1:]):
        raise ValueError("Web bounded whitespace differs between saved generation records")
    if values[0]["enabled"] and any(
            record.get("web_structured_final") != web_structured_configuration(True) for record in records):
        raise ValueError("Web bounded whitespace requires matching structured Web final configuration")


def _require_converter():
    if importlib.metadata.version("xgrammar") != WEB_WHITESPACE_XGRAMMAR_VERSION:
        raise ValueError("Bounded JSON whitespace requires xgrammar==" + WEB_WHITESPACE_XGRAMMAR_VERSION)


def _bound_whitespace(grammar):
    """Rewrite grammar whitespace atoms, never quoted strings or other classes.

    A shared finite recursive rule avoids the converter's per-site bounded
    repetition expansion and its lookahead helper production proliferation.
    """
    if _PREFIX in grammar:
        raise ValueError("Bounded JSON whitespace grammar prefix collision")
    output = []
    index = 0
    replaced = 0
    while index < len(grammar):
        begin = index
        character = grammar[index]
        if character == '"':
            # Copy an entire EBNF literal, including escaped quotes/classes.
            index += 1
            while index < len(grammar):
                if grammar[index] == "\\":
                    index += 2
                elif grammar[index] == '"':
                    index += 1
                    break
                else:
                    index += 1
            else:
                raise ValueError("Unterminated generated grammar literal")
            output.append(grammar[begin:index])
        elif character == "[":
            index += 1
            while index < len(grammar):
                if grammar[index] == "\\":
                    index += 2
                elif grammar[index] == "]":
                    index += 1
                    break
                else:
                    index += 1
            else:
                raise ValueError("Unterminated generated grammar character class")
            atom = grammar[begin:index]
            if atom == _WHITESPACE_CLASS and grammar[index:index + 1] == "*":
                output.append(_PREFIX + str(WEB_WHITESPACE_MAX_CHARS))
                replaced += 1
                index += 1
            else:
                whitespace_only = atom[1:-1].replace(r"\n", "").replace(r"\t", "").replace(r"\r", "").strip(" ") == ""
                if whitespace_only and grammar[index:index + 1] == "*":
                    raise ValueError("Pinned JSON-schema converter whitespace class has changed")
                output.append(atom)
        else:
            output.append(character)
            index += 1
    if not replaced:
        raise ValueError("Pinned JSON-schema converter whitespace shape has changed")
    additions = [_PREFIX + '0 ::= ""']
    additions.extend(_PREFIX + f'{count} ::= ("" | (' + _WHITESPACE_CLASS + ' '
                     + _PREFIX + f'{count - 1}))' for count in range(1, WEB_WHITESPACE_MAX_CHARS + 1))
    return "".join(output).rstrip("\n") + "\n" + "\n".join(additions) + "\n"


def web_bounded_whitespace_grammar(schema, *, explicit_review=False):
    """Bound only generated inter-token JSON whitespace; validate first."""
    from .web_review_decoder import is_web_review_schema
    from .web_structured import validate_web_structured_schema

    if type(explicit_review) is not bool:
        raise ValueError("explicit_review must be a boolean")
    validate_web_structured_schema(schema)
    if is_web_review_schema(schema) != explicit_review:
        raise ValueError("Bounded whitespace requires matching explicit-review provenance")
    _require_converter()
    return _cached_grammar(json.dumps(schema, ensure_ascii=False, separators=(",", ":")), explicit_review)


@lru_cache(maxsize=128)
def _cached_grammar(serialized_schema, explicit_review):
    import xgrammar
    from .web_review_decoder import web_review_grammar

    schema = json.loads(serialized_schema)
    grammar = (web_review_grammar(schema) if explicit_review
               else str(xgrammar.Grammar.from_json_schema(serialized_schema)))
    return _bound_whitespace(grammar)


def preflight_web_bounded_whitespace(*, explicit_review=False, cached_rows=False, multi_citation=False):
    """Check converter, grammar and vLLM parameter compatibility without a model."""
    import xgrammar
    from vllm.sampling_params import StructuredOutputsParams
    from .web_review import web_review_schema
    from .web_structured import web_action_schema, web_final_schema

    _require_converter()
    for columns, urls in ((["name"], []), (["name"], ["https://example.org/source"]),
                          (["中文"], ["https://example.org/source?name=%22"])):
        schemas = [web_final_schema(columns, urls, multi_citation=multi_citation)]
        schemas.extend(web_action_schema(columns, urls, successful_search=state, cached_rows=cached_rows, multi_citation=multi_citation) for state in (False, True))
        for schema in schemas:
            if explicit_review:
                schema = web_review_schema(schema)
            grammar = web_bounded_whitespace_grammar(schema, explicit_review=explicit_review)
            xgrammar.Grammar.from_ebnf(grammar)
            StructuredOutputsParams(grammar=grammar, disable_fallback=True)
    return web_bounded_whitespace_configuration(True)
