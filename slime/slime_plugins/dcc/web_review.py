"""Optional bounded public-task review before each structured Web action."""
from copy import deepcopy

WEB_EXPLICIT_REVIEW_VERSION = "web-explicit-plan-checks-v1"
WEB_EXPLICIT_REVIEW_MAX_CHARS = 240
WEB_EXPLICIT_REVIEW_INSTRUCTIONS = (
    '\nBefore every Web tool call or answer, include top-level JSON strings "plan" '
    'and "checks", each 1 to 240 characters. Write keys in this order: "plan", '
    '"checks", then "tool" and "arguments", or "answer". In plan, restate the '
    'original question\'s entity, every filter, unique-key and missing-field '
    'condition, without adding conditions. In checks, briefly identify which '
    'conditions the actually read source evidence establishes, what is still '
    'unverified, and whether another read is needed. Before answering, check '
    'every row against the original conditions, exact required columns, '
    'duplicate key, missing-field rule and one supporting read URL; stop after '
    'the supported finite set is exhausted. Distinguish successfully read '
    'source windows from a target source or window not yet read. These review '
    'strings are not source evidence. Do not '
    'invent values, a target row count, URLs or claims of verification. Return '
    'one JSON object with no text outside it.'
)


def web_explicit_review_configuration(enabled=False):
    if type(enabled) is not bool:
        raise ValueError("web_explicit_review must be a boolean")
    return {"enabled": enabled,
            "schema_version": WEB_EXPLICIT_REVIEW_VERSION if enabled else None,
            "prompt_version": WEB_EXPLICIT_REVIEW_VERSION if enabled else None,
            "max_chars": WEB_EXPLICIT_REVIEW_MAX_CHARS if enabled else None,
            "scope": "every-structured-web-action" if enabled else None,
            "backend": "xgrammar" if enabled else None}


def web_review_schema(schema):
    """Wrap action objects, preserving every existing tool/row restriction."""
    schema = deepcopy(schema)
    if set(schema) == {"anyOf"}:
        schema["anyOf"] = [web_review_schema(branch) for branch in schema["anyOf"]]
        return schema
    if schema.get("type") != "object" or not isinstance(schema.get("properties"), dict):
        raise ValueError("Web review requires an explicit action-object schema")
    if set(schema["properties"]) & {"plan", "checks"}:
        raise ValueError("Web action already contains review fields")
    field = {"type": "string", "minLength": 1, "maxLength": WEB_EXPLICIT_REVIEW_MAX_CHARS}
    schema["properties"] = {"plan": dict(field), "checks": dict(field), **schema["properties"]}
    schema["required"] = ["plan", "checks", *schema["required"]]
    return schema


def web_review_final_control(message):
    """Keep the existing final control while making its top-level shape exact."""
    result = dict(message)
    result["content"] = (result["content"]
        .replace('{"answer":[...]}', '{"plan":"...","checks":"...","answer":[...]}')
        .replace('{"answer":[]}', '{"plan":"...","checks":"...","answer":[]}')
        + WEB_EXPLICIT_REVIEW_INSTRUCTIONS)
    return result
