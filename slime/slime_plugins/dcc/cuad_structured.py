"""Optional canonical CUAD actions derived only from the public harness protocol.

No task, document, metadata, gold answer or generated target is an input. The
original parser/answer validator and document tools remain authoritative. The
structured path intentionally generates the canonical span/confidence objects
requested by _system_prompt; the legacy parser continues accepting older forms.
"""

from copy import deepcopy

CUAD_STRUCTURED_SCHEMA_VERSION = "cuad-public-document-actions-span-confidence-review-v1"
# The original final control says at most 40 words, without a character cap.
# Anchors matter: JSON Schema patterns normally search instead of full-match.
_BRIEF_REVIEW = {"type": "string", "pattern": r"^\s*\S+(?:\s+\S+){0,39}\s*$"}
_NONBLANK_TEXT = {"type": "string", "pattern": r"[\s\S]*\S[\s\S]*"}


def cuad_structured_configuration(enabled=False):
    if type(enabled) is not bool:
        raise ValueError("cuad_structured_actions must be a boolean")
    return {"enabled": enabled,
            "schema_version": CUAD_STRUCTURED_SCHEMA_VERSION if enabled else None,
            "backend": "xgrammar" if enabled else None,
            "final_review_max_words": 40 if enabled else None}


def _answer_schema(*, final_review):
    span = {"type": "object", "properties": {
        "text": deepcopy(_NONBLANK_TEXT),
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    }, "required": ["text", "confidence"], "additionalProperties": False}
    properties = {}
    if final_review:
        properties.update(plan=deepcopy(_BRIEF_REVIEW), checks=deepcopy(_BRIEF_REVIEW))
    properties["answer"] = {"type": "array", "items": span}
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def cuad_final_schema():
    """Last-turn plan/checks/answer; empty answer is a permitted abstention."""
    return _answer_schema(final_review=True)


def cuad_action_schema():
    """One public document tool or an early canonical final answer.

    Search query is required, top_k is optional and defaults to 4 at the original
    tool. Read start/end are optional, defaulting to 0/None. Integer clamping,
    document bounds, the 12000-character cap and errors remain with DocumentTools;
    no answer-dependent range or target-derived restriction is introduced.
    """
    search = {"type": "object", "properties": {
        "tool": {"const": "search_document"},
        "arguments": {"type": "object", "properties": {
            "query": {"type": "string"}, "top_k": {"type": "integer"},
        }, "required": ["query"], "additionalProperties": False},
    }, "required": ["tool", "arguments"], "additionalProperties": False}
    read = {"type": "object", "properties": {
        "tool": {"const": "read_document"},
        "arguments": {"type": "object", "properties": {
            "start": {"type": "integer"}, "end": {"type": ["integer", "null"]},
        }, "additionalProperties": False},
    }, "required": ["tool", "arguments"], "additionalProperties": False}
    return {"anyOf": [search, read, _answer_schema(final_review=False)]}


def is_cuad_structured_schema(schema):
    return schema == cuad_action_schema() or schema == cuad_final_schema()


def cuad_structured_options(schema):
    """Keep CUAD separate from the Web-only 240-character review decoder.

    vLLM/xgrammar grammar is followed by the native policy's complete strict JSON
    and JSON Schema validation. In particular that final validation enforces the
    numeric confidence bounds even if a backend converter overaccepts a number.
    """
    if not is_cuad_structured_schema(schema):
        raise ValueError("Expected the exact public CUAD action/final schema")
    import xgrammar
    from vllm.sampling_params import StructuredOutputsParams

    xgrammar.Grammar.from_json_schema(schema)
    return {"structured_outputs": StructuredOutputsParams(json=schema, disable_fallback=True)}


def preflight_cuad_structured():
    """Check installed schema/backend API without creating a model engine."""
    for schema in (cuad_action_schema(), cuad_final_schema()):
        cuad_structured_options(schema)
