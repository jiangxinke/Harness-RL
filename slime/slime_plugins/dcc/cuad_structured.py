"""Optional canonical CUAD actions derived only from the public harness protocol.

No task, document, metadata, gold answer or generated target is an input. The
original parser/answer validator and document tools remain authoritative. The
structured path intentionally generates the canonical span/confidence objects
requested by _system_prompt; the legacy parser continues accepting older forms.
"""

from copy import deepcopy

CUAD_STRUCTURED_SCHEMA_VERSION = "cuad-public-document-actions-span-confidence-review-v1"
CUAD_BACKEND_SCHEMA_VERSION = "cuad-structural-pattern-postvalidation-v1"
# The original final control says at most 40 words, without a character cap.
# Anchors matter: JSON Schema patterns normally search instead of full-match.
_BRIEF_REVIEW = {"type": "string", "pattern": r"^\s*\S+(?:\s+\S+){0,39}\s*$"}
_NONBLANK_TEXT = {"type": "string", "pattern": r"[\s\S]*\S[\s\S]*"}


def cuad_structured_configuration(enabled=False):
    if type(enabled) is not bool:
        raise ValueError("cuad_structured_actions must be a boolean")
    result = {"enabled": enabled,
              "schema_version": CUAD_STRUCTURED_SCHEMA_VERSION if enabled else None,
              "backend": "xgrammar" if enabled else None,
              "final_review_max_words": 40 if enabled else None}
    if enabled:
        result.update(backend_schema_version=CUAD_BACKEND_SCHEMA_VERSION,
                      backend_schema_policy="structural-schema-without-string-patterns",
                      backend_schema_matches_public=False,
                      original_public_schema_postvalidation=True)
    return result


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


def cuad_backend_schema(schema):
    """Lower the exact public schema to a JSON-safe backend structural schema.

    xgrammar 0.1.25 lowers the public string patterns to raw character classes
    inside JSON quotes, allowing controls and swallowing structural boundaries.
    Keep its ordinary JSON string recognizer by omitting only those patterns.
    This is a physical backend schema change: nonblank text and the 40-word
    reviews remain enforced by the unchanged original public schema validator.
    """
    if not is_cuad_structured_schema(schema):
        raise ValueError("Expected the exact public CUAD action/final schema")

    def lower(value):
        if isinstance(value, dict):
            return {key: lower(item) for key, item in value.items() if key != "pattern"}
        if isinstance(value, list):
            return [lower(item) for item in value]
        return deepcopy(value)

    return lower(schema)


def cuad_structured_options(schema):
    """Compile structural JSON constraints, then validate the public schema.

    The backend receives a structural schema without string patterns. The native
    policy still applies complete strict JSON parsing and the exact original
    public JSON Schema, including patterns and numeric confidence bounds.
    """
    backend_schema = cuad_backend_schema(schema)
    import xgrammar
    from vllm.sampling_params import StructuredOutputsParams

    xgrammar.Grammar.from_json_schema(backend_schema)
    return {"structured_outputs": StructuredOutputsParams(json=backend_schema, disable_fallback=True)}


def preflight_cuad_structured():
    """Check installed schema/backend API without creating a model engine."""
    for schema in (cuad_action_schema(), cuad_final_schema()):
        cuad_structured_options(schema)
