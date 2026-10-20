"""CPU contracts for optional observed-source navigation; no network or model."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest

from slime_plugins.dcc import EpisodeRunner, Task
from slime_plugins.dcc.episode import _system_prompt
from slime_plugins.dcc.web_navigation import (WEB_NAVIGATION_INSTRUCTIONS, WEB_NAVIGATION_LIMITS,
    WEB_NAVIGATION_PREFIX, WebNavigationLedger, web_navigation_configuration)
from slime_plugins.dcc.web_structured import web_structured_actions_configuration, web_structured_configuration

ROOT = Path(__file__).parents[1]
URL = "https://fixture.example/source"
OTHER = "https://fixture.example/docs/data.csv"
CONFIG = web_navigation_configuration(True)
REVIEW = runpy.run_path(str(ROOT / "tests/test_dcc_web_explicit_review.py"))
Policy, task = REVIEW["Policy"], REVIEW["task"]


def event(tool, observation, **arguments):
    return {"tool": tool, "arguments": arguments, "observation": observation}


def payload(ledger):
    rendered = ledger.render()
    assert len(rendered) <= WEB_NAVIGATION_LIMITS["max_chars"]
    return json.loads(rendered.removeprefix(WEB_NAVIGATION_PREFIX))


def located(url=URL, start=0, text="alpha", **metadata):
    return {"url": url, "text": text, "start": start, "end": start + len(text), **metadata}


def test_frontier_uses_only_actual_metadata_and_never_body_urls_gold_or_previews():
    ledger = WebNavigationLedger()
    events = [event("web_search", {"results": [{"url": URL, "title": "source"}]}),
              event("fetch_url", located(text="https://body.example/not-a-candidate", next_start=40,
                   outgoing_links=[{"url": OTHER, "title": "docs"}],
                   worksheets=[{"name": "Sheet", "start": 70, "preview": "GOLD_NEVER_IN_LEDGER"}],
                   archive_members=[{"name": "table.csv", "status": "readable", "start": 100},
                                    {"name": "directory/", "status": "directory", "start": 120},
                                    {"name": "file.bin", "status": "unsupported", "start": 130}]))]
    ledger.observe_events(events)
    data = payload(ledger)
    candidates = {(row["url"], row["start"]): row for row in data["frontier"]}
    assert set(candidates) == {(URL, 0), (OTHER, 0), (URL, 40), (URL, 70), (URL, 100)}
    assert candidates[(URL, 0)]["state"] == "read"
    assert candidates[(URL, 40)]["state"] == "unread"
    assert data["frontier"][0]["url"] == OTHER
    assert "GOLD_NEVER_IN_LEDGER" not in ledger.render()
    assert "body.example" not in ledger.render()
    assert candidates[(URL, 100)]["kind"] == "archive-member"
    # Incremental observation never duplicates actual events or reads.
    before = ledger.render()
    ledger.observe_events(events)
    assert ledger.render() == before


def test_windows_errors_and_constructed_urls_remain_distinct_with_rereads_allowed():
    ledger = WebNavigationLedger()
    api = "https://constructed.example/api/items?offset=10"
    events = [event("web_search", {"results": [{"url": URL}]}),
              event("fetch_url", {"url": URL, "error": "rate limited"}, url=URL, start=0),
              event("fetch_url", located(api, 20)), event("fetch_url", located(api, 20)),
              event("fetch_url", located(URL, 0)), event("fetch_url", located(URL, 5))]
    ledger.observe_events(events)
    data = payload(ledger)
    assert data["frontier"][0]["state"] == "read"
    assert data["recent_errors"] == [{"tool": "fetch_url", "url": URL, "error": "Tool reported an error",
                                      "http_status": None, "retryable": None}]
    assert next(w for w in data["recent_windows"] if w["url"] == api)["reads"] == 2
    assert {w["start"] for w in data["recent_windows"] if w["url"] == URL} == {0, 5}
    failed = WebNavigationLedger()
    failed.observe_events(events[:2])
    assert payload(failed)["frontier"][0]["state"] == "failed"
    assert payload(failed)["recent_windows"] == []


@pytest.mark.parametrize("start,end", [(True, 5), (-1, 4), (0, 99), (None, None)])
def test_unlocated_or_invalid_actual_ranges_never_mark_an_observed_target_read(start, end):
    ledger = WebNavigationLedger()
    ledger.observe_events([event("web_search", {"results": [{"url": URL}]}),
        event("fetch_url", {"url": URL, "text": "alpha", "start": start, "end": end})])
    data = payload(ledger)
    assert data["frontier"][0]["state"] == "unread"
    assert data["recent_windows"][0]["start"] is None


def test_bounded_storage_render_and_full_actionable_urls_with_explicit_omissions():
    ledger = WebNavigationLedger()
    urls = ["https://fixture.example/" + str(i) + "/" + "a" * 1700 for i in range(80)]
    ledger.observe_events([event("web_search", {"results": [{"url": u, "title": "t" * 1000} for u in urls]})] +
        [event("fetch_url", located(u)) for u in urls] +
        [event("fetch_url", {"url": u, "error": "e" * 1000}) for u in urls])
    assert len(ledger.frontier) <= WEB_NAVIGATION_LIMITS["max_frontier"]
    assert len(ledger.windows) == WEB_NAVIGATION_LIMITS["max_windows"]
    assert len(ledger.errors) == WEB_NAVIGATION_LIMITS["max_errors"]
    data = payload(ledger)
    assert all(value > 0 for value in data["evicted_records"].values())
    assert any(value > 0 for value in data["omitted_from_snapshot"].values())
    assert all(row["url"] in urls for field in ["frontier", "recent_windows", "recent_errors"] for row in data[field])
    ledger.observe_events([*([{}] * ledger.events_seen), event("web_search", {"results": [{"url": "https://fixture.example/" + "x" * 3000}]})])
    assert len(ledger.frontier) <= WEB_NAVIGATION_LIMITS["max_frontier"]


@pytest.mark.parametrize("invalid", [1, "true", None])
def test_switch_requires_a_real_bool(invalid):
    with pytest.raises(ValueError, match="boolean"):
        web_navigation_configuration(invalid)


class Web:
    async def web_search(self, query):
        return {"results": [{"url": URL, "title": "public table"}, {"url": OTHER, "title": "docs"}]}
    async def fetch_url(self, url, start=0):
        return located(url, start, text="Alpha", next_start=5 if start == 0 else None)
    def citation_validity(self, rows):
        return 1.


def runner(**kwargs):
    return EpisodeRunner(web_tools_factory=Web, web_structured_final=True, web_structured_actions=True,
                         web_navigation_ledger=True, **kwargs)


@pytest.mark.parametrize("review", [False, True])
def test_actual_runner_latest_snapshot_final_control_order_and_original_evidence_guard(review):
    values = [{"tool": "web_search", "arguments": {"query": "synthetic"}},
              {"tool": "fetch_url", "arguments": {"url": URL}},
              {"tool": "fetch_url", "arguments": {"url": URL, "start": 5}},
              {"answer": [{"name": "Alpha", "citations": [URL]}]}]
    if review:
        values = [REVIEW["reviewed"](value) for value in values]
    policy = Policy(values)
    result = asyncio.run(runner(max_steps=4, web_explicit_review=review).run(task(), policy, evaluation=True))
    assert result.status == "completed"
    assert len(result.metrics["web_navigation_ledger"]) == len(policy.calls) == 4
    for index, (_, messages, _) in enumerate(policy.calls):
        ledgers = [m for m in messages if m["content"].startswith(WEB_NAVIGATION_PREFIX)]
        assert len(ledgers) == 1
        assert ledgers[0]["content"] == result.metrics["web_navigation_ledger"][index]["content"]
    assert WEB_NAVIGATION_INSTRUCTIONS in policy.calls[0][1][0]["content"]
    assert policy.calls[-1][1][-2]["content"].startswith(WEB_NAVIGATION_PREFIX)
    assert "Final answer now" in policy.calls[-1][1][-1]["content"]
    assert not any(m["content"].startswith(WEB_NAVIGATION_PREFIX) for m in result.messages)
    assert "SYNTHETIC_NOT_FOR_PROMPT" not in json.dumps(policy.calls)
    assert len(json.loads(policy.calls[-1][1][-2]["content"].removeprefix(WEB_NAVIGATION_PREFIX))["recent_windows"]) == 2
    assert all("/not-allowed" not in str(call[2]) for call in policy.calls)


def test_automatic_actual_events_are_included_without_reprocessing_search_nested_observations():
    policy = Policy([{"tool": "web_search", "arguments": {"query": "synthetic"}},
                     {"answer": [{"name": "Alpha", "citations": [URL]}]}])
    result = asyncio.run(runner(max_steps=2, auto_read_sources=1).run(task(), policy, evaluation=True))
    assert result.status == "completed"
    data = json.loads(result.metrics["web_navigation_ledger"][-1]["content"].removeprefix(WEB_NAVIGATION_PREFIX))
    assert len(data["recent_windows"]) == 1 and data["recent_windows"][0]["reads"] == 1


def test_unread_candidate_never_becomes_citation_evidence():
    policy = Policy([{"tool": "web_search", "arguments": {"query": "synthetic"}},
                     {"tool": "fetch_url", "arguments": {"url": URL}},
                     {"answer": [{"name": "Alpha", "citations": [OTHER]}]}], validate=False)
    result = asyncio.run(runner(max_steps=3).run(task(), policy, evaluation=True))
    assert result.status == "evidence_failure"
    assert OTHER not in policy.calls[-1][2]["json_schema"]["properties"]["answer"]["items"]["anyOf"][0]["properties"]["citations"]["items"]["enum"]


def test_constructed_url_and_necessary_reread_are_not_restricted():
    api = "https://constructed.example/items?offset=50"
    policy = Policy([{"tool": "web_search", "arguments": {"query": "synthetic"}},
                     {"tool": "fetch_url", "arguments": {"url": api}},
                     {"tool": "fetch_url", "arguments": {"url": api}},
                     {"answer": [{"name": "Alpha", "citations": [api]}]}])
    result = asyncio.run(runner(max_steps=4).run(task(), policy, evaluation=True))
    assert result.status == "completed"
    data = json.loads(result.metrics["web_navigation_ledger"][-1]["content"].removeprefix(WEB_NAVIGATION_PREFIX))
    assert data["recent_windows"][0]["reads"] == 2


@pytest.mark.parametrize("benchmark,evaluation,enabled", [("cuad", True, True), ("widesearch", False, True), ("widesearch", True, False)])
def test_default_cuad_and_training_keep_original_messages(benchmark, evaluation, enabled):
    target = task() if benchmark == "widesearch" else Task("contract", "cuad", "validation", "Find clause", document="Synthetic", gold=[])
    policy = Policy([{"answer": []}], validate=False)
    result = asyncio.run(EpisodeRunner(max_steps=1, web_tools_factory=Web,
        web_structured_final=True, web_structured_actions=True, web_navigation_ledger=enabled).run(target, policy, evaluation=evaluation))
    assert policy.calls[0][1][0]["content"] == _system_prompt(target, 1)
    assert WEB_NAVIGATION_INSTRUCTIONS not in json.dumps(policy.calls)
    assert "web_navigation_ledger" not in result.metrics


@pytest.mark.parametrize("damage", [None, "manifest", "runtime", "protocol", "version", "limit", "extra", "bool", "actions", "mode"])
def test_paired_protocol_requires_complete_matching_config_and_dependencies(tmp_path, damage):
    helpers = runpy.run_path(str(ROOT / "tests/test_dcc_compare_runs.py"))
    values = helpers["pair"](tmp_path, [helpers["web"]("synthetic")])
    for run, declaration in zip(values[:2], values[2:], strict=True):
        for record in (run["manifest"], run["runtime"], declaration["common"]["generation"]):
            record.update(web_navigation_ledger=deepcopy(CONFIG), web_structured_final=web_structured_configuration(True),
                          web_structured_actions=web_structured_actions_configuration(True))
    if damage in {"manifest", "runtime"}:
        values[0][damage].pop("web_navigation_ledger")
    elif damage == "protocol":
        values[2]["common"]["generation"].pop("web_navigation_ledger")
    elif damage in {"version", "extra", "bool"}:
        key, value = {"version": ("version", "unsupported"), "extra": ("extra", True), "bool": ("enabled", 1)}[damage]
        values[0]["manifest"]["web_navigation_ledger"][key] = value
    elif damage == "limit":
        values[0]["manifest"]["web_navigation_ledger"]["limits"]["max_chars"] += 1
    elif damage == "actions":
        values[0]["manifest"].pop("web_structured_actions")
    elif damage == "mode":
        for record in (values[1]["manifest"], values[1]["runtime"], values[3]["common"]["generation"]):
            record.pop("web_navigation_ledger")
    if damage:
        with pytest.raises(ValueError, match="[Nn]avigation|Saved generation|[Pp]rotocol"):
            helpers["COMPARE"]["validate_protocols"](*values)
    else:
        helpers["COMPARE"]["validate_protocols"](*values)


@pytest.mark.parametrize("damage", [None, "runtime", "manifest", "version", "limits", "bool", "actions", "backend"])
def test_score_only_requires_full_runtime_match(tmp_path, damage):
    helpers = runpy.run_path(str(ROOT / "tests/test_dcc_model_scoring.py"))
    tasks = helpers["logical_predictions"](tmp_path)
    for path in [*tmp_path.glob("manifest_rank*.json"), *tmp_path.glob("inference_runtime*.json")]:
        record = json.loads(path.read_text())
        record.update(web_navigation_ledger=deepcopy(CONFIG), web_structured_final=web_structured_configuration(True),
                      web_structured_actions=web_structured_actions_configuration(True))
        if damage == "runtime" and path.name.startswith("inference_runtime") or damage == "manifest" and path.name.startswith("manifest"):
            record.pop("web_navigation_ledger")
        elif damage == "version":
            record["web_navigation_ledger"]["version"] = "unsupported"
        elif damage == "limits":
            record["web_navigation_ledger"]["limits"]["max_windows"] = 33
        elif damage == "bool":
            record["web_navigation_ledger"]["enabled"] = 1
        elif damage == "actions":
            record.pop("web_structured_actions")
        elif damage == "backend":
            record["inference_backend" if path.name.startswith("manifest") else "backend"] = "hf"
        path.write_text(json.dumps(record))
    if damage:
        with pytest.raises(ValueError, match="Web navigation ledger"):
            helpers["cli"]()["_score_predictions"](tmp_path, tasks)
    else:
        _, manifest, runtime = helpers["cli"]()["_score_predictions"](tmp_path, tasks)
        assert manifest["web_navigation_ledger"] == runtime["web_navigation_ledger"] == CONFIG


@pytest.mark.parametrize("enabled,bounded_whitespace", [(False, False), (True, False), (True, True)])
def test_evaluator_metadata_loader_and_resume_follow_actual_mode(monkeypatch, tmp_path, enabled, bounded_whitespace):
    helpers = runpy.run_path(str(ROOT / "tests/test_dcc_model_scoring.py"))
    policy = Policy([{"answer": []}], validate=False)
    policy.provenance = {"backend": "vllm", "model": "example/model"}
    async def close():
        pass
    policy.close = close
    if enabled:
        policy.provenance.update(web_navigation_ledger=deepcopy(CONFIG), web_structured_final=web_structured_configuration(True),
                                 web_structured_actions=web_structured_actions_configuration(True))
    if bounded_whitespace:
        from slime_plugins.dcc.web_whitespace import web_bounded_whitespace_configuration
        policy.provenance["web_bounded_whitespace"] = web_bounded_whitespace_configuration(True)
    evaluate, args = helpers["stub_evaluation"](monkeypatch, tmp_path, policy, [task()])
    args.web_bounded_whitespace = bounded_whitespace
    args.max_steps = 1
    seen = []
    def load(*pos, **kwargs):
        seen.append(kwargs)
        return policy
    monkeypatch.setitem(sys.modules, "slime_plugins.dcc.vllm_policy", SimpleNamespace(load_vllm_policy=load))
    if enabled:
        args.web_navigation_ledger = args.web_structured_final = args.web_structured_actions = True
    asyncio.run(evaluate(args))
    for name in ["manifest_rank0.json", "inference_runtime.json"]:
        record = json.loads((tmp_path / name).read_text())
        if enabled:
            assert record["web_navigation_ledger"] == CONFIG
        else:
            assert "web_navigation_ledger" not in record
    assert (seen[0].get("web_navigation_ledger") is True) is enabled
    assert (seen[0].get("web_bounded_whitespace") is True) is bounded_whitespace
    if enabled:
        args.web_navigation_ledger = False
        with pytest.raises(ValueError, match="manifest|settings|configuration"):
            asyncio.run(evaluate(args))


def test_loader_attests_actual_mode_using_cpu_runtime(monkeypatch):
    helpers = runpy.run_path(str(ROOT / "tests/test_dcc_vllm_policy.py"))
    helpers["runtime"].__wrapped__(monkeypatch)
    import transformers
    from slime_plugins.dcc import vllm_policy
    monkeypatch.setattr(vllm_policy, "preflight_web_structured", lambda **kwargs: None)
    monkeypatch.setitem(sys.modules, "vllm.config", SimpleNamespace(StructuredOutputsConfig=lambda **kwargs: SimpleNamespace(**kwargs)))
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **kw: SimpleNamespace(model_type="qwen3_moe"))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: helpers["Tokenizer"]())
    monkeypatch.setattr(transformers.GenerationConfig, "from_pretrained", lambda *a, **kw: SimpleNamespace(eos_token_id=1))
    monkeypatch.setattr(vllm_policy, "preflight_web_bounded_whitespace", lambda **kwargs: None)
    for enabled, bounded_whitespace in [(False, False), (True, False), (True, True)]:
        policy = vllm_policy.load_vllm_policy("synthetic/model", web_structured_final=True,
                                            web_structured_actions=True, web_navigation_ledger=enabled,
                                            web_bounded_whitespace=bounded_whitespace)
        assert policy.web_bounded_whitespace is bounded_whitespace
        if enabled:
            assert policy.provenance["web_navigation_ledger"] == CONFIG
        else:
            assert "web_navigation_ledger" not in policy.provenance


def test_dependencies_fail_before_model_loading():
    evaluate = runpy.run_path(str(ROOT / "scripts/dcc_model_evaluate.py"))["evaluate"]
    with pytest.raises(ValueError, match="web-navigation-ledger requires"):
        asyncio.run(evaluate(SimpleNamespace(web_navigation_ledger=True)))
    with pytest.raises(ValueError, match="requires structured Web actions"):
        EpisodeRunner(web_navigation_ledger=True)
    from slime_plugins.dcc.vllm_policy import load_vllm_policy
    with pytest.raises(ValueError, match="requires structured Web actions"):
        load_vllm_policy("synthetic/model", web_navigation_ledger=True)
