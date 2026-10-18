from __future__ import annotations

import pytest

import asyncio
import json

from slime_plugins.dcc import search_provider
from slime_plugins.dcc.tools import InfrastructureError


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    search_provider._CACHE.clear()
    monkeypatch.setattr(search_provider, "_DDG_COOLDOWN_UNTIL", 0.0)
    monkeypatch.setattr(search_provider, "_DDG_NEXT_REQUEST", 0.0)


def test_successful_web_result_has_no_hidden_fallback():
    hits = [{"title": "Example", "url": "https://example.org", "snippet": "source"}]
    result = search_provider.search_with_fallback("example", 3, lambda: hits)
    assert result["backend"] == "duckduckgo_lite"
    assert result["fallback_reason"] is None
    assert result["search_scope"] == "web"


def test_challenge_starts_cooldown_and_uses_explicit_wikipedia_fallback(monkeypatch):
    calls = []
    monkeypatch.setattr(
        search_provider,
        "wikipedia_search",
        lambda query, count: [
            {
                "title": query,
                "url": "https://en.wikipedia.org/wiki/Example",
                "snippet": "source",
            }
        ],
    )

    def challenge():
        calls.append("ddg")
        raise InfrastructureError("DuckDuckGo challenged this request")

    first = search_provider.search_with_fallback("first", 3, challenge)
    second = search_provider.search_with_fallback("second", 3, challenge)
    assert calls == ["ddg"]
    assert first["backend"] == second["backend"] == "wikipedia"
    assert "challenged" in first["fallback_reason"]
    assert "cooldown" in second["fallback_reason"]
    assert first["search_scope"] == "encyclopedia"


def test_successful_queries_use_bounded_process_cache():
    calls = []

    def search():
        calls.append("ddg")
        return [{"title": "Example", "url": "https://example.org", "snippet": "source"}]

    first = search_provider.search_with_fallback("same", 3, search)
    first["results"][0]["title"] = "mutated"
    second = search_provider.search_with_fallback("same", 3, search)
    assert calls == ["ddg"]
    assert second["results"][0]["title"] == "Example"


def test_empty_wikipedia_results_are_a_valid_observation(monkeypatch):
    monkeypatch.setattr(search_provider, "wikipedia_search", lambda query, count: [])
    result = search_provider.search_with_fallback("no results", 3, lambda: [])
    assert result["results"] == []
    assert result["backend"] == "wikipedia"


def test_encyclopedia_outage_propagates_and_does_not_become_empty_search(monkeypatch):
    def unavailable(query, count):
        raise InfrastructureError("Wikipedia unavailable")

    monkeypatch.setattr(search_provider, "wikipedia_search", unavailable)
    with pytest.raises(InfrastructureError, match="Wikipedia unavailable"):
        search_provider.search_with_fallback("query", 3, lambda: [])


def test_web_tool_returns_actual_fallback_backend_to_policy(monkeypatch):
    from slime_plugins.dcc.tools import WebTools

    def fail(self, query, count, *, backend=None):
        raise InfrastructureError("DDG challenge")

    monkeypatch.setattr(WebTools, "_search", fail)
    monkeypatch.setattr(
        search_provider,
        "wikipedia_search",
        lambda query, count: [
            {
                "title": "Source",
                "url": "https://en.wikipedia.org/wiki/Source",
                "snippet": "Evidence",
            }
        ],
    )
    result = asyncio.run(WebTools(backend="duckduckgo_wikipedia").web_search("query"))
    assert result["backend"] == "wikipedia"
    assert result["configured_backend"] == "duckduckgo_wikipedia"
    assert result["search_scope"] == "encyclopedia"
    assert result["fallback_reason"] == "DDG challenge"


def test_wikipedia_search_keeps_real_article_links_and_cleans_snippets(monkeypatch):
    from slime_plugins.dcc import tools

    urls = []

    def request(url):
        urls.append(url)
        return json.dumps(
            {
                "query": {
                    "search": [
                        {
                            "title": "北京大学",
                            "snippet": '<span class="searchmatch">北京大学</span>是一所大学',
                        }
                    ]
                }
            }
        )

    monkeypatch.setattr(tools, "_request", request)
    results = search_provider.wikipedia_search("北京大学", 3)
    assert urls[0].startswith("https://zh.wikipedia.org/w/api.php?")
    assert results[0]["url"].startswith("https://zh.wikipedia.org/wiki/")
    assert "<span" not in results[0]["snippet"]


def test_wikipedia_reader_retains_table_entities_and_rejects_missing_pages(monkeypatch):
    from slime_plugins.dcc import tools

    urls = []

    def article(url):
        urls.append(url)
        return json.dumps(
            {
                "parse": {
                    "title": "Source",
                    "text": "<p>Actual article text</p><table><tr><th>Award</th><th>Winner</th></tr><tr><td>Best Actor</td><td>Cillian Murphy</td></tr></table>",
                }
            }
        )

    monkeypatch.setattr(
        tools,
        "_request",
        article,
    )
    text = search_provider.wikipedia_page("https://en.wikipedia.org/wiki/Source")
    assert "Actual article text" in text
    assert "Best Actor" in text and "Cillian Murphy" in text
    assert "action=parse" in urls[0] and "prop=text" in urls[0]
    monkeypatch.setattr(
        tools,
        "_request",
        lambda url: json.dumps({"error": {"code": "missingtitle"}}),
    )
    with pytest.raises(InfrastructureError, match="unavailable"):
        search_provider.wikipedia_page("https://en.wikipedia.org/wiki/Missing")
