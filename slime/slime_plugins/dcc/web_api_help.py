"""Navigation diagnostics for actual public API responses; never answer evidence."""
from __future__ import annotations

import json
from urllib.parse import urlsplit

WORLD_BANK_METADATA_HELP_VERSION = "world-bank-indicator-metadata-help-v1"
_MAX_DIAGNOSTIC_TEXT_CHARS = 16_384
_OBSERVATION_API_TEMPLATE = (
    "https://api.worldbank.org/v2/country/{COUNTRY_IDS}/indicator/{INDICATOR_ID}"
    "?format=json&date={START}:{END}&per_page={PAGE_SIZE}"
)


def worldbank_metadata_diagnostic(observation):
    """Recognize a complete, actually returned indicator-definition JSON window.

    No request, URL rewrite, hidden cache read, row extraction or evidence
    registration occurs here. Ambiguous, partial and mixed responses get no hint.
    """
    if (not isinstance(observation, dict) or observation.get("error")
            or observation.get("start") != 0 or observation.get("truncated") is not False):
        return None
    url, text = observation.get("url"), observation.get("text")
    if (not isinstance(url, str) or not isinstance(text, str)
            or not 0 < len(text) <= _MAX_DIAGNOSTIC_TEXT_CHARS):
        return None
    try:
        parsed = urlsplit(url)
        if (parsed.scheme not in {"http", "https"} or parsed.hostname != "api.worldbank.org"
                or parsed.username is not None or parsed.password is not None):
            return None
        payload = json.loads(text)
    except (ValueError, RecursionError):
        return None
    if (not isinstance(payload, list) or len(payload) != 2
            or not isinstance(payload[0], dict)
            or not {"page", "pages", "total"}.issubset(payload[0])
            or not isinstance(payload[1], list) or not payload[1]):
        return None
    records = payload[1]
    if not all(
        isinstance(row, dict)
        and isinstance(row.get("id"), str)
        and isinstance(row.get("name"), str)
        and isinstance(row.get("source"), dict)
        and isinstance(row.get("sourceNote"), str)
        and not {"country", "date", "value"}.intersection(row)
        for row in records
    ):
        return None
    return {
        "version": WORLD_BANK_METADATA_HELP_VERSION,
        "kind": "indicator_metadata_not_time_series_observations",
        "note": (
            "This fetched JSON contains indicator definitions, not time-series observation rows: "
            "its records have no country/date/value fields. Changing only country or date query "
            "parameters on an indicator-definition endpoint does not establish the requested values. "
            "The public observation API puts country before indicator in the path. Replace the "
            "template placeholders using the public task and documentation, explicitly fetch that "
            "request, then verify actual country/date/value records and complete pagination. "
            "The template has not been fetched and is navigation help only. Neither it nor this "
            "metadata response establishes any requested time-series value."
        ),
        "public_observation_api_template": _OBSERVATION_API_TEMPLATE,
        "navigation_only": True,
        "template_fetched": False,
    }
