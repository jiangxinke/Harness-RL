from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from slime_plugins.dcc import search_provider
from slime_plugins.dcc.tools import InfrastructureError


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    search_provider._CACHE.clear()
    monkeypatch.setattr(search_provider, "_DDG_COOLDOWN_UNTIL", 0.0)
    monkeypatch.setattr(search_provider, "_DDG_NEXT_REQUEST", 0.0)
    search_provider._PUBLIC_COOLDOWN_UNTIL.clear()
    search_provider._PUBLIC_NEXT_REQUEST.clear()


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


def test_public_web_prefers_ddg_without_contacting_fallbacks(monkeypatch):
    def unexpected(*args):
        pytest.fail("fallback must not run after a successful primary search")

    monkeypatch.setattr(search_provider, "public_html_search", unexpected)
    monkeypatch.setattr(search_provider, "wikipedia_search", unexpected)
    hits = [{"title": "Source", "url": "https://example.org", "snippet": "Evidence"}]
    result = search_provider.public_web_search("example", 3, lambda: hits)
    assert result["backend"] == "duckduckgo_lite"
    assert result["fallback_reason"] is None


@pytest.mark.parametrize(
    ("query", "expected"),
    [("example topic", ["seznam", "baidu_mobile"]), ("中文查询", ["baidu_mobile"])],
)
def test_public_web_tries_real_indexes_before_wikipedia(monkeypatch, query, expected):
    calls = []

    def html_search(query, count, backend):
        calls.append(backend)
        if backend == "seznam":
            return []
        return [{"title": "Source", "url": "https://example.org", "snippet": "Evidence"}]

    def unavailable():
        raise InfrastructureError("challenge with response details that must stay private")

    monkeypatch.setattr(search_provider, "public_html_search", html_search)
    result = search_provider.public_web_search(query, 3, unavailable)
    assert calls == expected
    assert result["backend"] == "baidu_mobile"
    assert result["search_scope"] == "web"
    assert "challenge" in result["fallback_reason"]
    assert "private" not in result["fallback_reason"]


def test_public_web_uses_seznam_results_without_contacting_baidu(monkeypatch):
    calls = []

    def html_search(query, count, backend):
        calls.append(backend)
        return [{"title": "Source", "url": "https://example.org", "snippet": "Evidence"}]

    monkeypatch.setattr(search_provider, "public_html_search", html_search)
    result = search_provider.public_web_search("example", 3, lambda: [])
    assert calls == ["seznam"]
    assert result["backend"] == "seznam"


def test_public_web_labels_final_encyclopedia_scope(monkeypatch):
    monkeypatch.setattr(search_provider, "public_html_search", lambda *args: [])
    monkeypatch.setattr(search_provider, "wikipedia_search", lambda *args: [])
    result = search_provider.public_web_search("example", 3, lambda: [])
    assert result["results"] == []
    assert result["backend"] == "wikipedia"
    assert result["search_scope"] == "encyclopedia"
    assert "encyclopedia" in result["coverage_note"]


def test_public_web_valid_empty_survives_later_provider_outages(monkeypatch):
    def unavailable(*args):
        raise InfrastructureError("service unavailable")

    monkeypatch.setattr(search_provider, "public_html_search", unavailable)
    monkeypatch.setattr(search_provider, "wikipedia_search", unavailable)
    result = search_provider.public_web_search("example", 3, lambda: [])
    assert result["results"] == []
    assert result["backend"] == "duckduckgo_lite"
    assert result["search_scope"] == "web"


def test_public_web_all_outages_raise_without_response_details(monkeypatch):
    def unavailable(*args):
        raise InfrastructureError("provider failed at private.invalid/path")

    monkeypatch.setattr(search_provider, "public_html_search", unavailable)
    monkeypatch.setattr(search_provider, "wikipedia_search", unavailable)
    with pytest.raises(InfrastructureError, match="All public search providers are unavailable") as caught:
        search_provider.public_web_search("example", 3, unavailable)
    assert "private.invalid" not in str(caught.value)


def test_baidu_html_extracts_direct_sources_and_excludes_navigation():
    raw = """<html><body>
    <div data-log='{"mu":"https://example.org/article"}'>
      <h3>Actual <em>source</em></h3>Relevant evidence.
      <a href="https://m.baidu.com/tc?tracking=opaque">read</a></div>
    <div data-log='{"mu":"https://example.org/article"}'><h3>Duplicate</h3></div>
    <div data-log='{"mu":"http://1.recommend_list.baidu.com"}'><h3>Related queries</h3></div>
    <div data-log='["not-a-dictionary"]'><h3>Invalid</h3></div>
    <div data-log='{"mu":"javascript:bad()"}'><h3>Invalid</h3></div>
    <div data-log='not-json'>Invalid</div>
    </body></html>"""
    hits = search_provider._parse_public_results(raw, "baidu_mobile", 8)
    assert len(hits) == 1
    assert hits[0]["url"] == "https://example.org/article"
    assert hits[0]["title"] == "Actual source"
    assert "Relevant evidence" in hits[0]["snippet"]


def test_seznam_html_keeps_source_headings_and_ignores_captcha_scripts():
    raw = """<html><body><script>captcha = true; verify you are human</script>
    <div><h3><a href="https://example.org/article">Source title</a></h3>
    Evidence snippet.</div>
    <h3><a href="https://example.org/article">Duplicate</a></h3>
    <h2><a href="https://search.seznam.cz/?q=next">Next query</a></h2>
    </body></html>"""
    hits = search_provider._parse_public_results(raw, "seznam", 8)
    assert len(hits) == 1
    assert hits[0]["url"] == "https://example.org/article"
    assert "Evidence snippet" in hits[0]["snippet"]


@pytest.mark.parametrize("backend", ["seznam", "baidu_mobile"])
def test_public_html_challenge_triggers_cooldown_without_leaking_page(monkeypatch, backend):
    from slime_plugins.dcc import tools

    calls = []

    def challenge(url, **kwargs):
        calls.append(url)
        return "<html><h1>Verify you are human</h1>private client details</html>"

    monkeypatch.setattr(tools, "_request", challenge)
    with pytest.raises(InfrastructureError, match="challenged") as caught:
        search_provider.public_html_search("first", 3, backend)
    assert "private client" not in str(caught.value)
    with pytest.raises(InfrastructureError, match="cooldown"):
        search_provider.public_html_search("second", 3, backend)
    assert len(calls) == 1


@pytest.mark.parametrize("backend", ["seznam", "baidu_mobile"])
def test_public_html_cache_is_copied_and_reused_across_threads(monkeypatch, backend):
    from slime_plugins.dcc import tools

    calls = []

    def request(url, **kwargs):
        calls.append(url)
        return '<div data-log=\'{"mu":"https://example.org/article"}\'><h3><a href="https://example.org/article">Source</a></h3>Evidence</div>'

    monkeypatch.setattr(tools, "_request", request)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: search_provider.public_html_search("same", 3, backend), range(8)))
    assert len(calls) == 1
    results[0][0]["title"] = "mutated"
    assert all(rows[0]["title"] == "Source" for rows in results[1:])


def test_web_tool_defaults_to_public_web_and_exposes_actual_backend(monkeypatch):
    from slime_plugins.dcc.tools import WebTools

    monkeypatch.delenv("DCC_SEARCH_BACKEND", raising=False)
    monkeypatch.setattr(WebTools, "_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        search_provider, "public_html_search", lambda *args: [{"title": "Source", "url": "https://example.org", "snippet": "Evidence"}]
    )
    result = asyncio.run(WebTools().web_search("example"))
    assert result["configured_backend"] == "public_web"
    assert result["backend"] == "seznam"
    assert result["search_scope"] == "web"
