"""API navigation help must neither invent observations nor grant source evidence."""

import asyncio
import copy
import json

import pytest

from slime_plugins.dcc import tools
from slime_plugins.dcc.web_api_help import worldbank_metadata_diagnostic

URL = "https://api.worldbank.org/v2/indicator/EXAMPLE.INDICATOR?format=json"
DEFINITION = {
    "id": "EXAMPLE.INDICATOR", "name": "Example indicator",
    "source": {"id": "2", "value": "Example dataset"}, "sourceNote": "A definition, not observations.",
}
DATA = {"country": {"id": "ZZ", "value": "Example"}, "date": "2001", "value": 1.25}


def observation(records=None, **changes):
    text = json.dumps([{"page": 1, "pages": 1, "total": 1}, [DEFINITION] if records is None else records])
    return {"url": URL, "text": text, "start": 0, "truncated": False, **changes}


def test_complete_indicator_metadata_gets_navigation_help_without_mutation():
    original = observation()
    before = copy.deepcopy(original)
    hint = worldbank_metadata_diagnostic(original)
    assert original == before
    assert hint["kind"] == "indicator_metadata_not_time_series_observations"
    assert hint["navigation_only"] is True and hint["template_fetched"] is False
    assert "/country/{COUNTRY_IDS}/indicator/{INDICATOR_ID}?" in hint["public_observation_api_template"]
    assert "EXAMPLE.INDICATOR" not in hint["public_observation_api_template"]
    assert not {"sources", "results", "answer", "citations", "value"}.intersection(hint)


@pytest.mark.parametrize("records", [[DATA], [DEFINITION, DATA], []])
def test_observations_and_mixed_responses_are_not_indicator_definitions(records):
    assert worldbank_metadata_diagnostic(observation(records)) is None


@pytest.mark.parametrize("field", ["country", "date", "value"])
def test_definition_with_an_observation_field_is_ambiguous(field):
    assert worldbank_metadata_diagnostic(observation([{**DEFINITION, field: None}])) is None


@pytest.mark.parametrize("changes", [
    {"start": 1}, {"truncated": True}, {"error": "HTTP 404"},
    {"text": "<html>Indicator</html>"}, {"text": "[malformed"}, {"text": "{}"},
    {"text": "[{}, []]"}, {"text": " " * 16385},
    {"url": "https://api.worldbank.org.example/metadata"},
    {"url": "https://example.test/api.worldbank.org/metadata"},
    {"url": "https://user:pass@api.worldbank.org/v2/indicator/X"},
])
def test_partial_failed_malformed_or_foreign_response_gets_no_hint(changes):
    assert worldbank_metadata_diagnostic(observation(**changes)) is None


def test_fetch_metadata_keeps_original_page_and_does_not_fetch_or_register_template(monkeypatch):
    def no_download(*args, **kwargs):
        pytest.fail("A cached response diagnostic must not make a network request")

    monkeypatch.setattr(tools, "_download_public", no_download)
    web = tools.WebTools()
    text = observation()["text"]
    web.pages[URL] = text
    result = asyncio.run(web.fetch_url(URL))
    hint = result.pop("api_diagnostic")
    assert result == {"url": URL, "text": text, "start": 0, "end": len(text),
                      "total_characters": len(text), "truncated": False, "next_start": None}
    template = hint["public_observation_api_template"]
    assert web.pages == {URL: text} and set(web.sources) == {URL}
    assert template not in web.sources and template not in web.pages
    assert web.citation_validity([{"citations": [template]}]) == 0
    assert web.citation_validity([{"citations": [URL]}]) == 1


@pytest.mark.parametrize("records,window", [([DATA], None), ([DEFINITION], 100)])
def test_regular_data_and_partial_windows_keep_the_same_observation(records, window):
    text = observation(records)["text"]
    web = tools.WebTools(**({"max_text_window_chars": window} if window else {}))
    web.pages[URL] = text
    result = asyncio.run(web.fetch_url(URL))
    end = min(len(text), window or len(text))
    assert result == {"url": URL, "text": text[:end], "start": 0, "end": end,
                      "total_characters": len(text), "truncated": end < len(text),
                      "next_start": end if end < len(text) else None}
