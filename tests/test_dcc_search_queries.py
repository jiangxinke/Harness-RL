from __future__ import annotations

import asyncio

import pytest

from slime_plugins.dcc import search_provider
from slime_plugins.dcc.search_query import prepare_search_query
from slime_plugins.dcc.tools import WebTools


@pytest.fixture(autouse=True)
def reset_search_cache(monkeypatch):
    search_provider._CACHE.clear()
    monkeypatch.setattr(search_provider, "_DDG_COOLDOWN_UNTIL", 0.0)
    monkeypatch.setattr(search_provider, "_DDG_NEXT_REQUEST", 0.0)


@pytest.mark.parametrize("query", [
    "ordinary public documentation", "  preserve  ordinary spacing  ",
    "email researcher@example.org", "version 3.14 and 1.2.3", "report.pdf data.json table.csv archive.zip",
    "report.final numpy.ndarray source.py README.md", 'literal "example.org"',
    "https://user:secret@example.org/api", "site:localhost documentation",
])
def test_unscoped_or_ambiguous_query_is_byte_identical(query):
    assert prepare_search_query(query) == (query, None)


@pytest.mark.parametrize(("query", "expected"), [
    ("API docs example.org", "API docs site:example.org"),
    ("API docs domain:EXAMPLE.ORG", "API docs site:example.org"),
    ("API docs https://example.org/api?format=json#docs", "API docs site:example.org/api"),
    ("site:https://example.org/api?format=json#docs reference", "site:example.org/api reference"),
    ("site:bücher.de reference", "site:xn--bcher-kva.de reference"),
    ("domain:example.dev reference", "site:example.dev reference"),
])
def test_one_explicit_host_or_url_becomes_generic_site_operator(query, expected):
    actual, scope = prepare_search_query(query)
    assert actual == expected and scope is not None
    assert prepare_search_query(actual) == (actual, scope)


@pytest.mark.parametrize("query", [
    "site:example.org OR site:example.net docs", "site:example.org site:example.net docs",
    "site:example.org | site:example.net", "(site:example.org) docs",
    '"site:example.org" docs', "domain:example.org OR example.net docs",
    "example.org example.net docs", "site:example.org AND docs",
])
def test_complex_or_multiple_scope_retains_provider_semantics(query):
    assert prepare_search_query(query) == (query, None)


def test_scope_respects_host_boundary_subdomains_path_and_exclusions():
    query, scope = prepare_search_query("domain:example.org/api -site:private.example.org docs")
    assert query == "site:example.org/api -site:private.example.org docs"
    urls = ["https://example.org/api", "https://docs.example.org/api/v1?q=1#part",
            "https://example.org/apiculture", "https://notexample.org/api", "https://example.org.evil.net/api",
            "https://private.example.org/api/v1", "https://example.org@evil.net/api", "file://example.org/api"]
    hits = [{"url": url} for url in urls]
    assert scope.filter(hits) == hits[:2]


def test_unicode_and_punycode_hosts_match_same_source():
    _, scope = prepare_search_query("site:bücher.de docs")
    hits = [{"url": "https://bücher.de/api"}, {"url": "https://xn--bcher-kva.de/api"},
            {"url": "https://xn--bcher-kva.de.evil.org/api"}]
    assert scope.filter(hits) == hits[:2]


def test_negative_only_scope_excludes_without_requiring_an_included_host():
    query, scope = prepare_search_query("docs -site:example.org -domain:example.net")
    assert query == "docs -site:example.org -site:example.net"
    hits = [{"url": "https://example.org/a"}, {"url": "https://example.net/a"}, {"url": "https://other.org/a"}]
    assert scope.filter(hits) == hits[2:]


def test_public_fallback_does_not_accept_nonempty_out_of_scope_primary(monkeypatch):
    calls = []
    def fallback(query, count, backend):
        calls.append((query, backend))
        return [{"title": "Docs", "url": "https://docs.example.org/api", "snippet": "Public docs"}]
    monkeypatch.setattr(search_provider, "public_html_search", fallback)
    result = search_provider.public_web_search("site:example.org API", 3,
        lambda: [{"title": "Unrelated", "url": "https://unrelated.net/", "snippet": "Unrelated"}])
    assert result["backend"] == "seznam" and result["results"][0]["url"] == "https://docs.example.org/api"
    assert calls == [("site:example.org API", "seznam")]


def test_site_filter_does_not_expose_out_of_scope_encyclopedia_rows(monkeypatch):
    monkeypatch.setattr(search_provider, "public_html_search", lambda *args: [])
    monkeypatch.setattr(search_provider, "wikipedia_search", lambda *args: [{"url": "https://en.wikipedia.org/wiki/Example"}])
    result = search_provider.public_web_search("site:example.org docs", 3, lambda: [])
    assert result["results"] == []


def test_complex_or_query_does_not_filter_out_other_allowed_domain(monkeypatch):
    hits = [{"url": "https://example.org/docs"}, {"url": "https://example.net/docs"}]
    result = search_provider.public_web_search("site:example.org OR site:example.net docs", 3, lambda: hits)
    assert result["results"] == hits


def test_web_tool_rewrites_only_public_explicit_host_and_registers_filtered_sources(monkeypatch):
    calls = []
    def search(self, query, count, **kwargs):
        calls.append(query)
        return [{"title": "Docs", "url": "https://example.org/api", "snippet": "Docs"},
                {"title": "Other", "url": "https://example.org.evil.net/api", "snippet": "Other"}]
    monkeypatch.setattr(WebTools, "_search", search)
    tools = WebTools(backend="seznam")
    result = asyncio.run(tools.web_search("API docs domain:example.org"))
    assert calls == ["API docs site:example.org"]
    assert result["original_query"] == "API docs domain:example.org"
    assert result["query_scope"] == {"site": "example.org", "excluded_sites": []}
    assert list(tools.sources) == ["https://example.org/api"]


def test_web_tool_ordinary_query_observation_stays_exactly_compatible(monkeypatch):
    hits = [{"title": "Docs", "url": "https://example.org/api", "snippet": "Docs"}]
    monkeypatch.setattr(WebTools, "_search", lambda *args, **kwargs: hits)
    result = asyncio.run(WebTools(backend="seznam").web_search("ordinary documentation"))
    assert result == {"query": "ordinary documentation", "configured_backend": "seznam", "results": hits,
                      "backend": "seznam", "search_scope": "web", "fallback_reason": None}
