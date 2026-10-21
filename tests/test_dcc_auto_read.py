from __future__ import annotations

import asyncio
import json
import urllib.parse

import pytest
from slime_plugins.dcc import EpisodeRunner, PolicyCompletion, Task, tools
from slime_plugins.dcc.evaluation import evaluate_results

URLS = [f"https://example.org/source-{index}" for index in range(8)]
SEARCH = {"tool": "web_search", "arguments": {"query": "Alpha entities"}}


def final(url=URLS[0]):
    return {"answer": [{"name": "Alpha", "citations": [url]}]}


def wide():
    return Task("synthetic-wide", "widesearch", "test", "Find Alpha", gold=[{"name": "Alpha"}], metadata={
        "evaluation": {"required": ["name"], "unique_columns": ["name"],
                       "eval_pipeline": {"name": {"metric": ["exact_match"], "preprocess": ["norm_str"]}}},
        "hidden_auto_urls": [URLS[-1]],
    })


@pytest.fixture
def transport(monkeypatch):
    state = {"results": URLS[:3], "errors": set(), "empty": set(), "queries": {}, "calls": [], "pages": {}}

    def request(url, **kwargs):
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["q"][0]
        state["calls"].append(("search", query))
        urls = state["queries"].get(query, state["results"])
        return "".join(f'<a class="result-link" href="{source}">Source</a>' for source in urls)

    def download(url):
        state["calls"].append(("fetch", url))
        if url in state["errors"]:
            raise tools.InfrastructureError("source retrieval failed")
        if url in state["pages"]:
            return state["pages"][url].encode(), "text/html", "utf-8"
        text = "" if url in state["empty"] else "Alpha authoritative page evidence. " * 4
        return text.encode(), "text/plain", "utf-8"

    monkeypatch.setattr(tools, "_request", request)
    monkeypatch.setattr(tools, "_download_public", download)
    return state


def run(actions, *, auto=1, item=None, default=False, follow_docs=None):
    iterator, prompts = iter(actions), []

    async def policy(messages, **kwargs):
        prompts.append(messages)
        return PolicyCompletion(json.dumps(next(iterator)), [1], [2], [-0.1])

    options = {} if default else {"auto_read_sources": auto}
    if follow_docs is not None:
        options["auto_read_follow_docs"] = follow_docs
    runner = EpisodeRunner(max_steps=len(actions), web_tools_factory=lambda: tools.WebTools(backend="duckduckgo_lite", max_page_chars=40), **options)
    return asyncio.run(runner.run(item or wide(), policy, evaluation=True)), prompts


def observation(prompt):
    message = next(message for message in reversed(prompt) if message["content"].startswith("Tool observation (source material):"))
    return json.loads(message["content"].split("\n", 1)[1].split("\nRemaining turns:", 1)[0])


@pytest.mark.parametrize("value", [-1, 3, 1.0, True, None])
def test_auto_read_rejects_invalid_configuration(value):
    with pytest.raises(ValueError, match="integer between 0 and 2"):
        EpisodeRunner(auto_read_sources=value)


def test_auto_search_fetches_real_page_and_passes_guard_in_two_turns(transport):
    result, prompts = run([SEARCH, final()])
    assert result.status == "completed"
    assert transport["calls"] == [("search", "Alpha entities"), ("fetch", URLS[0])]
    assert len(result.steps) == 2 and len(result.trainable_steps) == 2
    assert result.metrics["tool_calls"] == 2
    guard = result.metrics["evidence_guard"]
    assert guard["accepted"] and guard["successful_searches"] == guard["successful_fetches"] == 1
    search, fetch = result.tool_events
    assert search["tool"] == "web_search" and "auto_read_sources" not in search["observation"]
    assert fetch["automatic"] is True and fetch["trigger"] == "web_search"
    assert fetch["arguments"] == {"url": URLS[0], "start": 0}
    seen = observation(prompts[1])
    assert seen["auto_read_sources"] == [fetch["observation"]]
    assert "Alpha authoritative" in seen["auto_read_sources"][0]["text"]
    assert seen["auto_read_sources"][0]["next_start"] == 40
    assert "start=next_start" in seen["auto_read_note"]
    assert seen["auto_read_attempts_remaining"] == 3
    assert URLS[-1] not in [value for kind, value in transport["calls"] if kind == "fetch"]


def test_default_off_exactly_preserves_observations_events_and_results(transport):
    actions = [SEARCH, {"tool": "fetch_url", "arguments": {"url": URLS[0]}}, final()]
    implicit, left = run(actions, default=True)
    explicit, right = run(actions, auto=0)
    assert implicit.to_dict() == explicit.to_dict() and left == right
    assert all("automatic" not in event for event in implicit.tool_events)
    assert "auto_read_sources" not in observation(left[1])


def test_search_order_per_search_limit_and_total_four_attempt_budget(transport):
    transport["queries"] = {"first": URLS[:3], "second": URLS[2:5], "third": URLS[4:]}
    actions = [{"tool": "web_search", "arguments": {"query": query}} for query in ("first", "second", "third")]
    result, prompts = run([*actions, final()], auto=2)
    automatic = [event for event in result.tool_events if event.get("automatic")]
    assert [event["arguments"]["url"] for event in automatic] == URLS[:4]
    assert [len(observation(prompt)["auto_read_sources"]) for prompt in prompts[1:]] == [2, 2, 0]
    assert observation(prompts[-1])["auto_read_attempts_remaining"] == 0
    assert result.metrics["tool_calls"] == 7 and result.status == "completed"


def test_duplicate_urls_and_already_successful_manual_reads_are_skipped(transport):
    transport["results"] = [URLS[0], URLS[0], URLS[1], URLS[1], URLS[2]]
    actions = [{"tool": "fetch_url", "arguments": {"url": URLS[0]}}, SEARCH, SEARCH, final()]
    result, _ = run(actions, auto=2)
    automatic = [event for event in result.tool_events if event.get("automatic")]
    assert [event["arguments"]["url"] for event in automatic] == URLS[1:3]
    assert result.metrics["evidence_guard"]["successful_fetches"] == 3


def test_manual_fetch_can_continue_automatically_read_page(transport):
    actions = [SEARCH, {"tool": "fetch_url", "arguments": {"url": URLS[0], "start": 40}}, final()]
    result, _ = run(actions)
    assert result.status == "completed"
    assert result.tool_events[1]["automatic"] is True
    assert "automatic" not in result.tool_events[2]
    assert result.tool_events[2]["observation"]["start"] == 40
    assert result.metrics["evidence_guard"]["successful_fetches"] == 2


def test_failed_auto_fetch_is_recorded_then_next_returned_url_is_read(transport):
    transport["errors"] = {URLS[0]}
    result, prompts = run([SEARCH, final(URLS[1])], auto=2)
    assert result.status == "completed"
    assert result.tool_events[1]["observation"] == {"url": URLS[0], "error": "source retrieval failed"}
    assert result.metrics["evidence_guard"]["successful_fetches"] == 2
    seen = observation(prompts[1])
    assert "error" in seen["auto_read_sources"][0] and "text" in seen["auto_read_sources"][1]
    assert len(seen["auto_read_sources"]) == 3
    assert seen["auto_read_successful_sources"] == 2
    assert seen["auto_read_attempts_remaining"] == 1


def test_blocked_source_is_a_real_reader_error_and_never_evidence(transport, monkeypatch):
    def blocked(url):
        raise ValueError("web tools only fetch public internet hosts")

    monkeypatch.setattr(tools, "_download_public", blocked)
    result, _ = run([SEARCH, final()])
    assert result.status == "evidence_failure"
    assert result.tool_events[1]["observation"]["error"] == "web tools only fetch public internet hosts"
    assert result.metrics["evidence_guard"]["successful_fetches"] == 0


@pytest.mark.parametrize("failure", [asyncio.CancelledError, RuntimeError, TypeError])
def test_cancellation_and_programming_errors_propagate(transport, monkeypatch, failure):
    async def failed(self, url, start=0):
        raise failure("stop")

    monkeypatch.setattr(tools.WebTools, "fetch_url", failed)
    with pytest.raises(failure, match="stop"):
        run([SEARCH, final()])


@pytest.mark.parametrize("failure", ["errors", "empty"])
def test_failed_or_empty_pages_keep_full_evaluation_denominator(transport, failure):
    transport["results"] = URLS[:2]
    transport[failure] = set(URLS[:2])
    result, _ = run([SEARCH, SEARCH, final()], auto=2)
    assert result.status == "evidence_failure" and result.reward == 0
    assert len([event for event in result.tool_events if event.get("automatic")]) == 2
    assert result.metrics["evidence_guard"]["successful_fetches"] == 0
    assert len(result.trainable_steps) == 3
    report = evaluate_results([wide()], [result.to_dict()])
    assert report["complete"] and report["widesearch"]["metric_denominator"] == 1
    assert report["widesearch"]["score"] == 0


def test_empty_search_and_terminal_search_never_auto_fetch(transport):
    transport["results"] = []
    result, _ = run([SEARCH, final()], auto=2)
    assert result.status == "evidence_failure"
    assert not any(kind == "fetch" for kind, _ in transport["calls"])
    transport["calls"].clear()
    result, _ = run([SEARCH], auto=2)
    assert result.status == "evidence_failure" and transport["calls"] == []
    assert "unavailable on the final turn" in result.tool_events[0]["observation"]["error"]


def test_cuad_results_and_prompts_do_not_change_when_auto_read_is_enabled(transport):
    item = Task("synthetic-contract", "cuad", "test", "Find clause", document="Exact clause", gold=["Exact clause"])
    actions = [{"tool": "search_document", "arguments": {"query": "clause"}}, {"answer": [{"text": "Exact clause", "confidence": 1}]}]
    baseline, left = run(actions, auto=0, item=item)
    enabled, right = run(actions, auto=2, item=item)
    assert baseline.to_dict() == enabled.to_dict() and left == right
    assert enabled.reward == 1 and transport["calls"] == []


@pytest.mark.parametrize("failure", ["errors", "empty"])
def test_unreadable_results_do_not_exhaust_success_target_or_exceed_total_attempts(transport, failure):
    transport["results"] = URLS
    transport[failure] = set(URLS[:3])
    result, prompts = run([SEARCH, final(URLS[3])], auto=2)
    seen = observation(prompts[1])
    assert [event["arguments"]["url"] for event in result.tool_events if event.get("automatic")] == URLS[:4]
    assert seen["auto_read_successful_sources"] == 1
    assert seen["auto_read_attempts_remaining"] == 0
    assert result.status == "completed"
    assert result.metrics["evidence_guard"]["successful_fetches"] == 1


def portal(*links):
    return '<html><body><p>Alpha public catalog source and navigation.</p>' + ''.join(
        f'<a href="{url}">{title}</a>' for url, title in links) + '</body></html>'


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_follow_docs_requires_an_explicit_boolean(value):
    with pytest.raises(ValueError, match="auto_read_follow_docs must be a boolean"):
        EpisodeRunner(auto_read_follow_docs=value)


def test_follow_docs_default_off_preserves_actual_sources_observations_and_events(transport):
    docs = "https://example.org/help"
    transport["pages"][URLS[0]] = portal((docs, "Help"))
    actions = [SEARCH, final()]
    implicit, left = run(actions, auto=2)
    explicit, right = run(actions, auto=2, follow_docs=False)
    assert implicit.to_dict() == explicit.to_dict() and left == right
    assert [value for kind, value in transport["calls"] if kind == "fetch"] == URLS[:2] * 2


def test_follow_docs_reads_observed_help_before_next_result_without_extra_turns(transport):
    docs = "https://example.org/catalog/service"
    transport["pages"][URLS[0]] = portal((docs, "Help"))
    transport["pages"][docs] = portal(("https://example.org/nested-api", "API documentation"))
    result, prompts = run([SEARCH, final(docs)], auto=2, follow_docs=True)
    assert [value for kind, value in transport["calls"] if kind == "fetch"] == [URLS[0], docs]
    assert result.status == "completed" and len(result.steps) == 2
    assert result.metrics["evidence_guard"]["successful_fetches"] == 2
    followup = result.tool_events[-1]
    assert followup["trigger"] == "observed_documentation_link" and followup["parent_url"] == URLS[0]
    assert followup["arguments"] == {"url": docs, "start": 0}
    assert observation(prompts[-1])["auto_read_attempts_remaining"] == 2


def test_follow_docs_filters_origin_self_and_already_read_fragment_aliases(transport):
    prior, docs = "https://example.org/prior-api", "https://example.org/documentation"
    transport["pages"][URLS[0]] = portal(
        (prior + "#section", "API"), (URLS[0] + "#help", "Help"),
        ("https://different.example.org/help", "Help"), ("http://example.org/help", "Help"),
        ("https://example.org:444/help", "Help"), ("https://user@example.org/help", "Help"),
        (docs, "Documentation"), (docs, "Documentation"))
    actions = [{"tool": "fetch_url", "arguments": {"url": prior}}, SEARCH, SEARCH, final(docs)]
    result, _ = run(actions, auto=2, follow_docs=True)
    automatic = [event["arguments"]["url"] for event in result.tool_events if event.get("automatic")]
    assert automatic == [URLS[0], docs, URLS[1], URLS[2]]
    assert len(automatic) == 4 and result.status == "completed"


def test_failed_followup_consumes_original_episode_cap_and_is_not_evidence(transport):
    docs = "https://example.org/help"
    transport["pages"][URLS[0]] = portal((docs, "Help"))
    transport["results"] = URLS
    transport["errors"] = {docs, *URLS[1:]}
    result, prompts = run([SEARCH, SEARCH, final()], auto=2, follow_docs=True)
    automatic = [event for event in result.tool_events if event.get("automatic")]
    assert [event["arguments"]["url"] for event in automatic] == [URLS[0], docs, URLS[1], URLS[2]]
    assert automatic[1]["observation"].get("error")
    assert observation(prompts[1])["auto_read_attempts_remaining"] == 0
    assert observation(prompts[2])["auto_read_sources"] == []
    assert result.metrics["evidence_guard"]["successful_fetches"] == 1


@pytest.mark.parametrize("auto", [0, 1])
def test_follow_docs_does_not_raise_configured_success_target(transport, auto):
    transport["pages"][URLS[0]] = portal(("https://example.org/help", "Help"))
    result, _ = run([SEARCH, final()], auto=auto, follow_docs=True)
    assert len([event for event in result.tool_events if event.get("automatic")]) == auto
