"""Optional generic source-discovery instructions, never answer evidence."""
from __future__ import annotations

WEB_DATA_DISCOVERY_VERSION = "public-web-data-discovery-guidance-v1"
WEB_DATA_DISCOVERY_PREFIX = "\nData discovery guidance (controller instructions; not source evidence):\n"
WEB_DATA_DISCOVERY_GUIDANCE = WEB_DATA_DISCOVERY_PREFIX + (
    'For tasks requiring structured or enumerated records, separate source discovery from filtering. '
    'Start with a concise query for the authority and entity/data type; do not require CSV or put every '
    'field and filter into the discovery query unless the user requires that format. Inspect the returned '
    'titles, domains and snippets. If they are unrelated, materially reformulate rather than only '
    'rearranging the same words. If a source lacks required fields, identify that gap and search for '
    'entity metadata, documentation, an API or a structured export that can supply it.'
    "\n\n"
    'Read relevant authoritative documentation and actual data. Use only formats and request parameters '
    'supported by documentation or observed source links. If a response reports an invalid format or '
    'another API error, treat it as a failed data request, inspect the diagnostic and correct the request '
    'or source; it does not verify any answer row. For paginated APIs, distinguish a character window of '
    "one response from another response page: next_start only continues the same URL's extracted text; "
    'use documented API parameters to request further pages.'
    "\n\n"
    'Keep every original question condition, exact column, unique-key and missing-field rule. Verify '
    'returned records themselves; a query, snippet, documentation example, unread link or error is not '
    'evidence for an answer row. Keep supported rows, never force a nonempty answer, and reserve the '
    'existing final turn. This guidance adds no turns, data, URLs or evidence exceptions.'
)


def web_data_discovery_configuration(enabled=False):
    if type(enabled) is not bool:
        raise ValueError("web_data_discovery_guidance must be a boolean")
    return {"enabled": enabled,
            "version": WEB_DATA_DISCOVERY_VERSION if enabled else None,
            "scope": "structured-web-evaluation" if enabled else None,
            "placement": "system-once" if enabled else None}


def validate_web_data_discovery_records(records):
    from .web_structured import web_structured_actions_configuration, web_structured_configuration

    values = [record.get("web_data_discovery_guidance", web_data_discovery_configuration(False))
              for record in records]
    if not values:
        raise ValueError("Web data discovery requires configuration records")
    for value in values:
        if (not isinstance(value, dict) or type(value.get("enabled")) is not bool
                or value != web_data_discovery_configuration(value["enabled"])):
            raise ValueError("Web data discovery requires complete supported configuration")
    if any(value != values[0] for value in values[1:]):
        raise ValueError("Web data discovery differs between manifest, runtime and protocol")
    if values[0]["enabled"] and any(
            record.get("web_structured_final") != web_structured_configuration(True)
            or record.get("web_structured_actions") != web_structured_actions_configuration(True)
            or record.get("inference_backend", record.get("backend")) != "vllm"
            for record in records):
        raise ValueError("Web data discovery requires matching structured Web actions/final and vLLM")
    return values[0]


def validate_web_data_discovery_messages(messages):
    """Attest the declared opt-in against the saved public system instructions."""
    if (not isinstance(messages, list) or not messages or not isinstance(messages[0], dict)
            or messages[0].get("role") != "system"
            or not isinstance(messages[0].get("content"), str)
            or messages[0]["content"].count(WEB_DATA_DISCOVERY_PREFIX) != 1
            or not messages[0]["content"].endswith(WEB_DATA_DISCOVERY_GUIDANCE)):
        raise ValueError("Web data discovery is missing or duplicated in the saved system instructions")
