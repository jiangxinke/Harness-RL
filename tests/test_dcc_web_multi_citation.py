"""Offline public-source contracts; no model, network or benchmark labels."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import jsonschema
import pytest

from slime_plugins.dcc import EpisodeRunner, Task
from slime_plugins.dcc.episode import _WebEvidence
from slime_plugins.dcc.web_review import web_review_schema
from slime_plugins.dcc.web_structured import (
    web_action_schema, web_final_schema, web_multi_citation_configuration,
    validate_web_multi_citation_records, web_structured_actions_configuration,
    web_structured_configuration,
)

ROOT = Path(__file__).parents[1]
DOC, DATA = "https://example.org/docs", "https://example.org/data.json"
URLS = [DOC, DATA, "https://example.org/other", "https://example.org/last"]
CONFIG = web_multi_citation_configuration(True)
REVIEW = runpy.run_path(str(ROOT / "tests/test_dcc_web_explicit_review.py"))
Policy, task = REVIEW["Policy"], REVIEW["task"]


def configure(record):
    record.update(web_multi_citation=deepcopy(CONFIG),
        web_structured_final=web_structured_configuration(True),
        web_structured_actions=web_structured_actions_configuration(True))


class Web:
    async def web_search(self, query):
        return {"results": [{"url": DOC}, {"url": DATA}]}
    async def fetch_url(self, url, start=0):
        return {"url": url, "text": "Observed official source", "start": start, "end": start + 24}
    def citation_validity(self, rows):
        return 1.0


def actions(citations=(DOC, DATA), *, review=False):
    values = [{"tool": "web_search", "arguments": {"query": "official records"}},
        {"tool": "fetch_url", "arguments": {"url": DOC}},
        {"tool": "fetch_url", "arguments": {"url": DATA}},
        {"answer": [{"name": "Observed entity", "citations": list(citations)}]}]
    return [{"plan": "Find sources", "checks": "Read supporting evidence", **v} for v in values] if review else values


@pytest.mark.parametrize("invalid", [1, 0, "true", None, [], {}])
def test_flag_is_boolean(invalid):
    with pytest.raises(ValueError, match="boolean"):
        web_multi_citation_configuration(invalid)


def test_default_schema_bytes_match_legacy_review_and_action_fixtures():
    fixtures = json.loads((ROOT / "tests/fixtures/dcc_web_legacy_citation_schemas.json").read_text())
    for f in fixtures:
        for options in ({}, {"multi_citation": False}):
            schema = (web_final_schema(["name", "date"], [DOC, DATA], **options)
                if f["state"] == "final" else web_action_schema(["name", "date"], [DOC, DATA],
                    successful_search=f["state"] == "ready", **options))
            if f["review"]: schema = web_review_schema(schema)
            assert json.dumps(schema, ensure_ascii=False, separators=(",", ":")) == f["schema_json"]


@pytest.mark.parametrize("citations,accepted", [
    ([DOC], True), ([DOC, DATA], True), (URLS, True), (URLS + [DOC], False),
    ([], False), ([DOC, "https://example.org/unread"], False),
])
@pytest.mark.parametrize("review", [False, True])
def test_actual_json_and_bounded_review_grammars_enforce_four_read_urls(citations, accepted, review):
    import xgrammar
    from xgrammar.testing import _is_grammar_accept_string
    from slime_plugins.dcc.web_review_decoder import web_review_grammar
    from slime_plugins.dcc.web_whitespace import web_bounded_whitespace_grammar

    schema = web_final_schema(["name"], URLS, multi_citation=True)
    value = {"answer": [{"name": "Observed", "citations": citations}]}
    if review:
        schema = web_review_schema(schema)
        value = {"plan": "Read sources", "checks": "Verify fields", **value}
    if accepted: jsonschema.validate(value, schema)
    else:
        with pytest.raises(jsonschema.ValidationError): jsonschema.validate(value, schema)
    grammars = [xgrammar.Grammar.from_json_schema(schema), xgrammar.Grammar.from_ebnf(
        web_bounded_whitespace_grammar(schema, explicit_review=review))]
    if review: grammars.append(xgrammar.Grammar.from_ebnf(web_review_grammar(schema)))
    encoded = json.dumps(value, ensure_ascii=False).replace("/", "\\/")
    for grammar in grammars:
        assert _is_grammar_accept_string(grammar, encoded) is accepted


@pytest.mark.parametrize("review", [False, True])
def test_runner_uses_identical_cap_on_actions_review_and_final(review):
    policy = Policy(actions(review=review))
    result = asyncio.run(EpisodeRunner(max_steps=4, web_tools_factory=Web,
        web_structured_final=True, web_structured_actions=True, web_explicit_review=review,
        web_multi_citation=True).run(task(), policy, evaluation=True))
    assert result.status == "completed" and result.answer[0]["citations"] == [DOC, DATA]
    assert len(result.steps) == 4 and len(result.tool_events) == 3
    for index, (_, _, options) in enumerate(policy.calls):
        schema = (web_final_schema(["name"], [DOC, DATA], multi_citation=True) if index == 3
            else web_action_schema(["name"], [DOC, DATA][:max(0, index - 1)],
                successful_search=index > 0, multi_citation=True))
        if review: schema = web_review_schema(schema)
        assert options["json_schema"] == schema


def test_default_off_preserves_requests_and_evidence_guard_rejects_any_unread_url():
    policies = []
    for options in ({}, {"web_multi_citation": False}):
        policy = Policy(actions((DOC,)))
        result = asyncio.run(EpisodeRunner(max_steps=4, web_tools_factory=Web,
            web_structured_final=True, web_structured_actions=True, **options).run(task(), policy, evaluation=True))
        assert result.status == "completed"
        policies.append(policy)
    assert policies[0].calls == policies[1].calls
    evidence = _WebEvidence()
    evidence.observe("web_search", {"results": [{"url": DOC}, {"url": DATA}]})
    evidence.observe("fetch_url", {"url": DOC, "text": "Observed docs"})
    assert evidence.reject_reasons(task(), [{"name": "Observed", "citations": [DOC, DATA]}]) == ["unread_citation_url"]
    evidence.observe("fetch_url", {"url": DATA, "error": "Access challenge"})
    assert evidence.fetches == 1
    assert evidence.reject_reasons(task(), [{"name": "Observed", "citations": [DOC, DATA]}]) == ["unread_citation_url"]
    assert "empty_answer" in evidence.reject_reasons(task(), [])


@pytest.mark.parametrize("damage", [None, "manifest", "runtime", "protocol", "version", "cap", "boolean", "extra", "actions", "backend", "mode"])
def test_pair_protocol_rejects_mixed_or_unsupported_cap(tmp_path, damage):
    helpers = runpy.run_path(str(ROOT / "tests/test_dcc_compare_runs.py"))
    pair = helpers["pair"](tmp_path, [helpers["web"]("invented")])
    for run, declaration in zip(pair[:2], pair[2:], strict=True):
        for record in (run["manifest"], run["runtime"], declaration["common"]["generation"]): configure(record)
    if damage in {"manifest", "runtime"}: pair[0][damage].pop("web_multi_citation")
    elif damage == "protocol": pair[2]["common"]["generation"].pop("web_multi_citation")
    elif damage in {"version", "cap", "boolean", "extra"}:
        key, value = {"version": ("version", "unknown"), "cap": ("max_citations", 5),
            "boolean": ("enabled", 1), "extra": ("extra", True)}[damage]
        pair[0]["manifest"]["web_multi_citation"][key] = value
    elif damage == "actions": pair[0]["manifest"].pop("web_structured_actions")
    elif damage == "backend": pair[0]["runtime"]["backend"] = "hf"
    elif damage == "mode":
        for record in (pair[1]["manifest"], pair[1]["runtime"], pair[3]["common"]["generation"]): record.pop("web_multi_citation")
    if damage:
        with pytest.raises(ValueError, match="[Mm]ulti-citation|Saved generation|[Pp]rotocol"):
            helpers["COMPARE"]["validate_protocols"](*pair)
    else: helpers["COMPARE"]["validate_protocols"](*pair)


@pytest.mark.parametrize("enabled", [False, True])
def test_evaluator_records_and_resumes_exact_cap(monkeypatch, tmp_path, enabled):
    helpers = runpy.run_path(str(ROOT / "tests/test_dcc_model_scoring.py"))
    policy = Policy([{"answer": []}], validate=False)
    policy.provenance = {"backend": "vllm", "model": "example/model"}
    async def close(): pass
    policy.close = close
    if enabled: configure(policy.provenance)
    evaluate, args = helpers["stub_evaluation"](monkeypatch, tmp_path, policy, [task()])
    args.web_multi_citation, args.max_steps = enabled, 1
    if enabled: args.web_structured_final = args.web_structured_actions = True
    seen = []
    def load(*a, **kwargs): seen.append(kwargs); return policy
    monkeypatch.setitem(sys.modules, "slime_plugins.dcc.vllm_policy", SimpleNamespace(load_vllm_policy=load))
    asyncio.run(evaluate(args))
    for name in ["manifest_rank0.json", "inference_runtime.json"]:
        record = json.loads((tmp_path / name).read_text())
        if enabled: assert record["web_multi_citation"] == CONFIG
        else: assert "web_multi_citation" not in record
    assert (seen[0].get("web_multi_citation") is True) is enabled
    asyncio.run(evaluate(args)); assert len(seen) == 1
    if enabled:
        runtime = json.loads((tmp_path / "inference_runtime.json").read_text())
        runtime["web_multi_citation"]["max_citations"] = 1
        (tmp_path / "inference_runtime.json").write_text(json.dumps(runtime))
        with pytest.raises(ValueError, match="[Mm]ulti-citation"): asyncio.run(evaluate(args))
        assert len(seen) == 1


def test_native_sidecar_reconstructs_exact_multicitation_schema(tmp_path):
    helpers = runpy.run_path(str(ROOT / "tests/test_dcc_native_thinking.py"))
    tokenizer, engine, policy = helpers["make_policy"](tmp_path, values=actions())
    public_task = task()
    async def run():
        bound = policy.bind_task(public_task.id, 0)
        result = await EpisodeRunner(max_steps=4, max_new_tokens=policy.budget.total_tokens,
            web_tools_factory=Web, web_structured_final=True, web_structured_actions=True,
            web_multi_citation=True).run(public_task, bound, evaluation=True)
        record = result.to_dict(); record["native_thinking_traces"] = bound.references
        await policy.close()
        return record
    record = asyncio.run(run())
    assert record["status"] == "completed"
    manifest, runtime = helpers["declarations"](4, budget=policy.budget)
    for declaration in (manifest, runtime): configure(declaration); declaration["backend"] = "vllm"
    validate = helpers["validate_native_sidecars"]
    assert validate(tmp_path, {public_task.id: record}, manifest, runtime,
        {public_task.id: 0}, [public_task], tokenizer=tokenizer)
    runtime["web_multi_citation"]["max_citations"] = 1
    with pytest.raises(ValueError, match="[Mm]ulti-citation"):
        validate(tmp_path, {public_task.id: record}, manifest, runtime,
            {public_task.id: 0}, [public_task], tokenizer=tokenizer)


def test_real_loader_attests_cap_with_cpu_mocked_engine(monkeypatch):
    helpers = runpy.run_path(str(ROOT / "tests/test_dcc_vllm_policy.py"))
    helpers["runtime"].__wrapped__(monkeypatch)
    import transformers
    from slime_plugins.dcc import vllm_policy
    calls = []
    monkeypatch.setattr(vllm_policy, "preflight_web_structured", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setitem(sys.modules, "vllm.config", SimpleNamespace(StructuredOutputsConfig=lambda **kwargs: SimpleNamespace(**kwargs)))
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **kw: SimpleNamespace(model_type="qwen3_moe"))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: helpers["Tokenizer"]())
    monkeypatch.setattr(transformers.GenerationConfig, "from_pretrained", lambda *a, **kw: SimpleNamespace(eos_token_id=1))
    for enabled in (False, True):
        policy = vllm_policy.load_vllm_policy("invented/model", web_structured_final=True,
            web_structured_actions=True, web_multi_citation=enabled)
        assert policy.provenance.get("web_multi_citation") == (CONFIG if enabled else None)
        assert calls[-1] == ({"actions": True, "multi_citation": True} if enabled else {"actions": True})


@pytest.mark.parametrize("damage", [None, "manifest", "runtime", "cap", "version"])
def test_score_only_validates_saved_cap_before_scoring(tmp_path, damage):
    helpers = runpy.run_path(str(ROOT / "tests/test_dcc_model_scoring.py"))
    tasks = helpers["logical_predictions"](tmp_path)
    for path in [*tmp_path.glob("manifest_rank*.json"), *tmp_path.glob("inference_runtime*.json")]:
        record = json.loads(path.read_text()); configure(record)
        if damage == "manifest" and path.name.startswith("manifest"): record.pop("web_multi_citation")
        elif damage == "runtime" and path.name.startswith("inference_runtime"): record.pop("web_multi_citation")
        elif damage == "cap": record["web_multi_citation"]["max_citations"] = 2
        elif damage == "version": record["web_multi_citation"]["version"] = "unknown"
        path.write_text(json.dumps(record))
    if damage:
        with pytest.raises(ValueError, match="[Mm]ulti-citation"):
            helpers["cli"]()["_score_predictions"](tmp_path, tasks)
    else:
        saved, manifest, runtime = helpers["cli"]()["_score_predictions"](tmp_path, tasks)
        assert set(saved) == {t.id for t in tasks}
        assert manifest["web_multi_citation"] == runtime["web_multi_citation"] == CONFIG


def test_dependencies_fail_before_model_or_data_loading():
    evaluate = runpy.run_path(str(ROOT / "scripts/dcc_model_evaluate.py"))["evaluate"]
    with pytest.raises(ValueError, match="web-multi-citation requires"):
        asyncio.run(evaluate(SimpleNamespace(web_multi_citation=True)))
    with pytest.raises(ValueError, match="requires structured Web actions"):
        EpisodeRunner(web_multi_citation=True)
    from slime_plugins.dcc.vllm_policy import load_vllm_policy
    with pytest.raises(ValueError, match="requires structured Web actions"):
        load_vllm_policy("invented/model", web_multi_citation=True)
    assert validate_web_multi_citation_records(({}, {})) == web_multi_citation_configuration(False)


@pytest.mark.parametrize("spelling", ["you're", "you’re", "you are"])
@pytest.mark.parametrize("render", [False, True])
def test_robot_challenge_never_counts_as_read_evidence(monkeypatch, spelling, render):
    from slime_plugins.dcc import tools
    text = ("JavaScript is disabled\nIn order to continue, we need to verify that " + spelling
        + " not a robot. This requires JavaScript. Enable JavaScript and then reload the page.")
    if spelling == "you're": assert len(text) == 157
    monkeypatch.setattr(tools, "_download_public", lambda url: (text.encode(), "text/html", "utf-8"))
    class Browser:
        async def read(self, url):
            raise AssertionError("Access challenge must not trigger browser rendering")
    web, evidence = tools.WebTools(browser_reader=Browser() if render else None), _WebEvidence()
    observation = asyncio.run(web.fetch_url(DOC))
    evidence.observe("fetch_url", observation)
    assert "access challenge" in observation["error"] and "text" not in observation
    assert evidence.fetches == 0 and not evidence.fetched_sources and not web.pages


def test_robotics_article_is_still_readable(monkeypatch):
    from slime_plugins.dcc import tools
    text = "Robotics research: this autonomous robot uses JavaScript. Verify the measured latency."
    monkeypatch.setattr(tools, "_download_public", lambda url: (text.encode(), "text/plain", "utf-8"))
    observation = asyncio.run(tools.WebTools().fetch_url(DOC))
    evidence = _WebEvidence(); evidence.observe("fetch_url", observation)
    assert observation["text"] == text and evidence.fetches == 1
