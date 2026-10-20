"""Synthetic CPU contracts for opt-in source discovery; no live data or URLs."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest

from slime_plugins.dcc import EpisodeRunner, Task
from slime_plugins.dcc.web_data_discovery import (
    WEB_DATA_DISCOVERY_GUIDANCE, WEB_DATA_DISCOVERY_PREFIX,
    validate_web_data_discovery_messages, validate_web_data_discovery_records,
    web_data_discovery_configuration,
)
from slime_plugins.dcc.web_structured import (
    web_structured_actions_configuration, web_structured_configuration,
)

ROOT = Path(__file__).parents[1]
REVIEW = runpy.run_path(str(ROOT / "tests/test_dcc_web_explicit_review.py"))
NAV = runpy.run_path(str(ROOT / "tests/test_dcc_web_navigation.py"))
Policy, Web, task = REVIEW["Policy"], NAV["Web"], REVIEW["task"]
URL = NAV["URL"]
CONFIG = web_data_discovery_configuration(True)


def runner(enabled=False, **kwargs):
    return EpisodeRunner(web_tools_factory=Web, web_structured_final=True,
        web_structured_actions=True, web_data_discovery_guidance=enabled, **kwargs)


def configure(record):
    record.update(web_data_discovery_guidance=deepcopy(CONFIG),
        web_structured_final=web_structured_configuration(True),
        web_structured_actions=web_structured_actions_configuration(True))


@pytest.mark.parametrize("invalid", [1, 0, "true", None, [], {}])
def test_switch_is_boolean(invalid):
    with pytest.raises(ValueError, match="boolean"):
        web_data_discovery_configuration(invalid)


def test_default_and_explicit_off_keep_original_prompt_and_optin_is_once_per_request():
    actions = [{"tool": "web_search", "arguments": {"query": "invented metadata"}},
        {"tool": "fetch_url", "arguments": {"url": URL}},
        {"tool": "fetch_url", "arguments": {"url": URL, "start": 5}},
        {"answer": [{"name": "Alpha", "citations": [URL]}]}]
    policies, results = [], []
    for enabled in [None, False, True]:
        policy = Policy(actions)
        options = {} if enabled is None else {"web_data_discovery_guidance": enabled}
        result = asyncio.run(EpisodeRunner(max_steps=4, max_new_tokens=73,
            web_tools_factory=Web, web_structured_final=True, web_structured_actions=True,
            web_explicit_review=False, web_navigation_ledger=True,
            web_research_checkpoints=True, **options).run(task(), policy, evaluation=True))
        assert result.status == "completed"
        policies.append(policy); results.append(result)
    assert policies[0].calls == policies[1].calls
    assert results[0].to_dict() == results[1].to_dict()
    assert results[0].tool_events == results[2].tool_events
    assert results[0].answer == results[2].answer
    assert results[0].status == results[2].status
    assert len(policies[2].calls) == 4
    for before, after in zip(policies[0].calls, policies[2].calls, strict=True):
        assert before[2] == after[2]  # same schema, sampling and token budget
        assert after[1][0]["content"] == before[1][0]["content"] + WEB_DATA_DISCOVERY_GUIDANCE
        assert after[1][1:] == before[1][1:]
        assert after[1][0]["content"].count(WEB_DATA_DISCOVERY_PREFIX) == 1
        validate_web_data_discovery_messages(after[1])
    assert "SYNTHETIC_NOT_FOR_PROMPT" not in json.dumps(policies[2].calls)


@pytest.mark.parametrize("evaluation,benchmark", [(False, "widesearch"), (True, "cuad")])
def test_training_and_cuad_requests_are_unchanged(evaluation, benchmark):
    public_task = task() if benchmark == "widesearch" else Task(
        "invented-contract", "cuad", "validation", "Extract clauses", document="A clause.", gold=[])
    policies = []
    for enabled in [False, True]:
        policy = Policy([{"answer": []}], validate=False)
        result = asyncio.run(runner(enabled, max_steps=1, initial_chunks=0).run(
            public_task, policy, evaluation=evaluation, score_train=False))
        policies.append(policy)
        assert WEB_DATA_DISCOVERY_PREFIX not in json.dumps(result.messages)
    assert policies[0].calls == policies[1].calls


def test_errors_and_repeated_abstention_are_preserved_not_promoted_to_evidence():
    class ErrorWeb(Web):
        async def fetch_url(self, url, start=0):
            return {"url": url, "error": "Invented format rejection; use documented parameters."}
    actions = [{"tool": "web_search", "arguments": {"query": "invented CSV fields"}},
        {"tool": "fetch_url", "arguments": {"url": URL}},
        {"answer": []}, {"answer": []}]
    results = []
    for enabled in [False, True]:
        result = asyncio.run(EpisodeRunner(max_steps=4, web_tools_factory=ErrorWeb,
            web_structured_final=True, web_structured_actions=True,
            web_data_discovery_guidance=enabled).run(task(), Policy(actions), evaluation=True))
        assert result.status == "evidence_failure" and "empty_answer" in result.detail
        assert len(result.steps) == 4 and len(result.tool_events) == 2
        assert any("Invented format rejection" in m["content"] for m in result.messages)
        results.append(result)
    assert results[0].tool_events == results[1].tool_events


@pytest.mark.parametrize("damage", ["missing", "duplicate", "truncated"])
def test_saved_optin_must_actually_appear_once(damage):
    content = "Original instructions" + WEB_DATA_DISCOVERY_GUIDANCE
    if damage == "missing": content = "Original instructions"
    if damage == "duplicate": content += WEB_DATA_DISCOVERY_GUIDANCE
    if damage == "truncated": content = content[:-1]
    with pytest.raises(ValueError, match="saved system instructions"):
        validate_web_data_discovery_messages([{"role": "system", "content": content}])


@pytest.mark.parametrize("damage", [None, "manifest", "runtime", "protocol", "version", "extra", "bool", "scope", "placement", "actions", "backend", "mode"])
def test_comparator_rejects_mixed_or_unsupported_declarations(tmp_path, damage):
    helpers = runpy.run_path(str(ROOT / "tests/test_dcc_compare_runs.py"))
    pair = helpers["pair"](tmp_path, [helpers["web"]("invented")])
    for run, declaration in zip(pair[:2], pair[2:], strict=True):
        for record in (run["manifest"], run["runtime"], declaration["common"]["generation"]):
            configure(record)
    if damage in {"manifest", "runtime"}: pair[0][damage].pop("web_data_discovery_guidance")
    elif damage == "protocol": pair[2]["common"]["generation"].pop("web_data_discovery_guidance")
    elif damage in {"version", "extra", "bool", "scope", "placement"}:
        key, value = {"version": ("version", "unknown"), "extra": ("extra", True),
            "bool": ("enabled", 1), "scope": ("scope", "training"),
            "placement": ("placement", "every-turn")}[damage]
        pair[0]["manifest"]["web_data_discovery_guidance"][key] = value
    elif damage == "actions": pair[0]["manifest"].pop("web_structured_actions")
    elif damage == "backend": pair[0]["runtime"]["backend"] = "hf"
    elif damage == "mode":
        for record in (pair[1]["manifest"], pair[1]["runtime"], pair[3]["common"]["generation"]):
            record.pop("web_data_discovery_guidance")
    if damage:
        with pytest.raises(ValueError, match="[Dd]iscovery|Saved generation|[Pp]rotocol"):
            helpers["COMPARE"]["validate_protocols"](*pair)
    else: helpers["COMPARE"]["validate_protocols"](*pair)


@pytest.mark.parametrize("enabled", [False, True])
def test_evaluator_records_and_resumes_exact_configuration(monkeypatch, tmp_path, enabled):
    helpers = runpy.run_path(str(ROOT / "tests/test_dcc_model_scoring.py"))
    policy = Policy([{"answer": []}], validate=False)
    policy.provenance = {"backend": "vllm", "model": "example/model"}
    async def close(): pass
    policy.close = close
    if enabled: configure(policy.provenance)
    evaluate, args = helpers["stub_evaluation"](monkeypatch, tmp_path, policy, [task()])
    args.web_data_discovery_guidance = enabled
    args.max_steps = 1
    if enabled: args.web_structured_final = args.web_structured_actions = True
    seen = []
    def load(*a, **kwargs): seen.append(kwargs); return policy
    monkeypatch.setitem(sys.modules, "slime_plugins.dcc.vllm_policy", SimpleNamespace(load_vllm_policy=load))
    asyncio.run(evaluate(args))
    for filename in ["manifest_rank0.json", "inference_runtime.json"]:
        record = json.loads((tmp_path / filename).read_text())
        if enabled: assert record["web_data_discovery_guidance"] == CONFIG
        else: assert "web_data_discovery_guidance" not in record
    assert (seen[0].get("web_data_discovery_guidance") is True) is enabled
    asyncio.run(evaluate(args))
    assert len(seen) == 1
    if enabled:
        runtime = json.loads((tmp_path / "inference_runtime.json").read_text())
        runtime["web_data_discovery_guidance"]["version"] = "unknown"
        (tmp_path / "inference_runtime.json").write_text(json.dumps(runtime))
        with pytest.raises(ValueError, match="Web data discovery"):
            asyncio.run(evaluate(args))
        assert len(seen) == 1


def test_real_loader_attests_flag_with_cpu_mocked_engine(monkeypatch):
    helpers = runpy.run_path(str(ROOT / "tests/test_dcc_vllm_policy.py"))
    helpers["runtime"].__wrapped__(monkeypatch)
    import transformers
    from slime_plugins.dcc import vllm_policy
    monkeypatch.setattr(vllm_policy, "preflight_web_structured", lambda **kwargs: None)
    monkeypatch.setitem(sys.modules, "vllm.config", SimpleNamespace(StructuredOutputsConfig=lambda **kwargs: SimpleNamespace(**kwargs)))
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **kw: SimpleNamespace(model_type="qwen3_moe"))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: helpers["Tokenizer"]())
    monkeypatch.setattr(transformers.GenerationConfig, "from_pretrained", lambda *a, **kw: SimpleNamespace(eos_token_id=1))
    for enabled in [False, True]:
        policy = vllm_policy.load_vllm_policy("invented/model", web_structured_final=True,
            web_structured_actions=True, web_data_discovery_guidance=enabled)
        assert policy.provenance.get("web_data_discovery_guidance") == (CONFIG if enabled else None)


def test_native_sidecar_attests_actual_enabled_prompt_and_metadata(tmp_path):
    helpers = runpy.run_path(str(ROOT / "tests/test_dcc_native_thinking.py"))
    values = [{"tool": "web_search", "arguments": {"query": "invented fields"}},
        {"tool": "fetch_url", "arguments": {"url": URL}},
        {"answer": [{"name": "Alpha", "citations": [URL]}]}]
    tokenizer, engine, policy = helpers["make_policy"](tmp_path, values=values)
    public_task = task()
    async def run():
        bound = policy.bind_task(public_task.id, 0)
        result = await runner(True, max_steps=3, max_new_tokens=policy.budget.total_tokens).run(
            public_task, bound, evaluation=True)
        record = result.to_dict(); record["native_thinking_traces"] = bound.references
        await policy.close()
        return record
    record = asyncio.run(run())
    manifest, runtime = helpers["declarations"](3, budget=policy.budget)
    for declaration in (manifest, runtime):
        configure(declaration); declaration["backend"] = "vllm"
    validate = helpers["validate_native_sidecars"]
    assert validate(tmp_path, {public_task.id: record}, manifest, runtime,
        {public_task.id: 0}, [public_task], tokenizer=tokenizer)
    damaged = deepcopy(record)
    damaged["messages"][0]["content"] = damaged["messages"][0]["content"].removesuffix(WEB_DATA_DISCOVERY_GUIDANCE)
    with pytest.raises(ValueError, match="saved system instructions"):
        validate(tmp_path, {public_task.id: damaged}, manifest, runtime,
            {public_task.id: 0}, [public_task], tokenizer=tokenizer)


def test_dependencies_fail_before_model_or_data_loading_and_absence_is_off():
    evaluate = runpy.run_path(str(ROOT / "scripts/dcc_model_evaluate.py"))["evaluate"]
    with pytest.raises(ValueError, match="web-data-discovery-guidance requires"):
        asyncio.run(evaluate(SimpleNamespace(web_data_discovery_guidance=True)))
    with pytest.raises(ValueError, match="requires structured Web actions"):
        EpisodeRunner(web_data_discovery_guidance=True)
    from slime_plugins.dcc.vllm_policy import load_vllm_policy
    with pytest.raises(ValueError, match="requires structured Web actions"):
        load_vllm_policy("invented/model", web_data_discovery_guidance=True)
    assert validate_web_data_discovery_records(({}, {})) == web_data_discovery_configuration(False)
    assert "http" not in WEB_DATA_DISCOVERY_GUIDANCE
    assert "worldbank" not in WEB_DATA_DISCOVERY_GUIDANCE.lower()
    assert "gbif" not in WEB_DATA_DISCOVERY_GUIDANCE.lower()


@pytest.mark.parametrize("damage", [None, "manifest", "runtime", "version", "missing", "duplicate", "truncated"])
def test_score_only_attests_metadata_and_actual_saved_web_prompt(tmp_path, damage):
    helpers = runpy.run_path(str(ROOT / "tests/test_dcc_model_scoring.py"))
    synthetic_tasks = helpers["logical_predictions"](tmp_path)
    public_tasks = [Task(t.id, "widesearch", "validation", "Invented entity fields",
                         metadata={"answer_columns": ["name"]}) for t in synthetic_tasks]
    for path in [*tmp_path.glob("manifest_rank*.json"), *tmp_path.glob("inference_runtime*.json")]:
        record = json.loads(path.read_text()); configure(record)
        if damage == "manifest" and path.name.startswith("manifest"):
            record.pop("web_data_discovery_guidance")
        elif damage == "runtime" and path.name.startswith("inference_runtime"):
            record.pop("web_data_discovery_guidance")
        elif damage == "version":
            record["web_data_discovery_guidance"]["version"] = "unsupported"
        path.write_text(json.dumps(record))
    for path in tmp_path.glob("predictions_rank*.jsonl"):
        record = json.loads(path.read_text())
        content = "Original synthetic instructions" + WEB_DATA_DISCOVERY_GUIDANCE
        if damage == "missing": content = "Original synthetic instructions"
        elif damage == "duplicate": content += WEB_DATA_DISCOVERY_GUIDANCE
        elif damage == "truncated": content = content[:-1]
        record["messages"] = [{"role": "system", "content": content}]
        path.write_text(json.dumps(record) + "\n")
    if damage:
        with pytest.raises(ValueError, match="[Dd]iscovery"):
            helpers["cli"]()["_score_predictions"](tmp_path, public_tasks)
    else:
        saved, manifest, runtime = helpers["cli"]()["_score_predictions"](tmp_path, public_tasks)
        assert set(saved) == {t.id for t in public_tasks}
        assert manifest["web_data_discovery_guidance"] == runtime["web_data_discovery_guidance"] == CONFIG


def test_native_metadata_rejected_before_optional_engine_and_trace_loading(tmp_path):
    from slime_plugins.dcc.native_thinking_eval import validate_native_sidecars
    manifest = {"inference_backend": "vllm"}; configure(manifest)
    runtime = {"backend": "vllm"}; configure(runtime)
    runtime["web_data_discovery_guidance"]["placement"] = "every-turn"
    with pytest.raises(ValueError, match="[Dd]iscovery"):
        validate_native_sidecars(tmp_path, {}, manifest, runtime, {}, [])
