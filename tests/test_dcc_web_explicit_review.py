"""Synthetic CPU schema/runner/provenance checks; never loads a model."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest

from slime_plugins.dcc import EpisodeRunner, PolicyCompletion, Task
from slime_plugins.dcc.episode import _WebEvidence, _system_prompt, parse_action
from slime_plugins.dcc.web_review import (WEB_EXPLICIT_REVIEW_INSTRUCTIONS,
    WEB_EXPLICIT_REVIEW_MAX_CHARS, WEB_EXPLICIT_REVIEW_VERSION,
    web_explicit_review_configuration, web_review_schema)
from slime_plugins.dcc.web_structured import (preflight_web_structured, web_action_schema,
    web_final_schema, web_structured_actions_configuration, web_structured_configuration)

ROOT = Path(__file__).parents[1]
URL = "https://fixture.example/read"
CONFIG = web_explicit_review_configuration(True)


def reviewed(value):
    return {"plan": "Use every original filter and unique key; omit incomplete rows if required.",
            "checks": "Use actually read support, exact columns and unique rows; stop after verified rows.", **value}


def task():
    return Task("synthetic", "widesearch", "validation", "Find matching entries; require every field and unique names.",
        gold=[{"name": "SYNTHETIC_NOT_FOR_PROMPT"}], metadata={"evaluation": {"required": ["name"]}})


class Web:
    async def web_search(self, query):
        return {"results": [{"url": URL, "title": "Synthetic public source"}]}
    async def fetch_url(self, url):
        return {"url": url, "text": "Synthetic source says Alpha", "start": 0, "end": 27}
    def citation_validity(self, rows):
        return 1.


class Policy:
    def __init__(self, values, validate=True):
        self.values, self.calls, self.validate = iter(values), [], validate
    async def __call__(self, messages, **kwargs):
        self.calls.append(("ordinary", deepcopy(messages), kwargs))
        return PolicyCompletion(json.dumps(next(self.values)), [1], [2])
    async def generate_structured(self, messages, **kwargs):
        self.calls.append(("structured", deepcopy(messages), kwargs))
        value = next(self.values)
        if self.validate:
            pytest.importorskip("jsonschema").validate(value, kwargs["json_schema"])
        return PolicyCompletion(json.dumps(value), [1], [2])


@pytest.mark.parametrize("invalid", [1, "true", None])
def test_review_switch_requires_real_boolean(invalid):
    with pytest.raises(ValueError, match="boolean"):
        web_explicit_review_configuration(invalid)


@pytest.mark.parametrize("state,urls", [(False, []), (True, []), (True, [URL])])
def test_review_wraps_only_top_level_actions_and_preserves_every_base_constraint(state, urls):
    original = web_action_schema(["plan", "中文"], urls, successful_search=state)
    snapshot = deepcopy(original)
    wrapped = web_review_schema(original)
    branches = wrapped.get("anyOf", [wrapped])
    stripped = []
    for branch in branches:
        assert list(branch["properties"])[:2] == ["plan", "checks"]
        assert branch["required"][:2] == ["plan", "checks"]
        assert branch["properties"]["plan"] == {"type": "string", "minLength": 1, "maxLength": 240}
        stripped.append({**branch, "properties": {k: v for k, v in branch["properties"].items() if k not in {"plan", "checks"}},
                         "required": branch["required"][2:]})
    assert ({"anyOf": stripped} if "anyOf" in wrapped else stripped[0]) == original
    assert original == snapshot


@pytest.mark.parametrize("damage", ["missing", "empty", "long", "integer", "extra", "mixed"])
def test_schema_rejects_unbounded_missing_or_ambiguous_review_actions(damage):
    value = reviewed({"tool": "web_search", "arguments": {"query": "synthetic"}})
    if damage == "missing":
        del value["plan"]
    elif damage == "empty":
        value["checks"] = ""
    elif damage == "long":
        value["plan"] = "中" * 241
    elif damage == "integer":
        value["plan"] = 1
    elif damage == "extra":
        value["explanation"] = "not allowed"
    else:
        value["answer"] = []
    with pytest.raises(pytest.importorskip("jsonschema").ValidationError):
        pytest.importorskip("jsonschema").validate(value, web_review_schema(web_action_schema(["name"], [], successful_search=False)))


def test_real_xgrammar_enforces_review_length_and_order_with_exact_citation():
    xgrammar = pytest.importorskip("xgrammar")
    from xgrammar.testing import _is_grammar_accept_string
    grammar = xgrammar.Grammar.from_json_schema(web_review_schema(web_final_schema(["name"], [URL])))
    for text, accepted in [("a", True), ("中" * 240, True), ("a" * 241, False), ("", False)]:
        value = {"plan": text, "checks": "Observed support", "answer": [{"name": "Alpha", "citations": [URL]}]}
        encoded = json.dumps(value, ensure_ascii=False).replace("/", "\\/")
        assert _is_grammar_accept_string(grammar, encoded) is accepted
    value = {"answer": [], "plan": "Review", "checks": "Review"}
    assert not _is_grammar_accept_string(grammar, json.dumps(value))


@pytest.mark.parametrize("searched,read", [(False, False), (True, False), (True, True)])
def test_real_xgrammar_preserves_reviewed_search_read_and_final_state_gates(searched, read):
    xgrammar = pytest.importorskip("xgrammar")
    from xgrammar.testing import _is_grammar_accept_string
    grammar = xgrammar.Grammar.from_json_schema(web_review_schema(
        web_action_schema(["name"], [URL] if read else [], successful_search=searched)))
    for action, accepted in [({"tool": "web_search", "arguments": {"query": "synthetic"}}, True),
                             ({"tool": "fetch_url", "arguments": {"url": URL}}, searched),
                             ({"answer": [{"name": "Alpha", "citations": [URL]}]}, searched and read)]:
        encoded = json.dumps(reviewed(action)).replace("/", "\\/")
        assert _is_grammar_accept_string(grammar, encoded) is accepted


def test_every_web_action_and_final_get_review_from_actual_search_and_read_states():
    policy = Policy([reviewed({"tool": "web_search", "arguments": {"query": "synthetic"}}),
                     reviewed({"tool": "fetch_url", "arguments": {"url": URL}}),
                     reviewed({"answer": [{"name": "Alpha", "citations": [URL]}]})])
    runner = EpisodeRunner(max_steps=3, web_tools_factory=Web, web_structured_final=True,
                          web_structured_actions=True, web_explicit_review=True)
    result = asyncio.run(runner.run(task(), policy, evaluation=True))
    assert result.status == "completed" and len(result.tool_events) == 2
    assert [call[0] for call in policy.calls] == ["structured"] * 3
    assert "anyOf" not in policy.calls[0][2]["json_schema"]
    assert policy.calls[1][2]["json_schema"]["anyOf"][-1]["properties"]["answer"]["maxItems"] == 0
    final = policy.calls[-1]
    assert final[2]["json_schema"] == web_review_schema(web_final_schema(["name"], [URL]))
    assert '"plan":"...","checks":"...","answer"' in final[1][-1]["content"]
    assert 'object {"answer"' not in final[1][-1]["content"]
    assert WEB_EXPLICIT_REVIEW_INSTRUCTIONS in policy.calls[0][1][0]["content"]
    assert "SYNTHETIC_NOT_FOR_PROMPT" not in json.dumps(policy.calls)


@pytest.mark.parametrize("kind", ["no_read", "unread_citation", "empty"])
def test_review_text_never_supplies_evidence_or_overrides_existing_guard(kind):
    values = [{"tool": "web_search", "arguments": {"query": "synthetic"}}]
    if kind != "no_read":
        values.append({"tool": "fetch_url", "arguments": {"url": URL}})
    rows = [] if kind == "empty" else [{"name": "Alpha", "citations": [URL if kind == "no_read" else URL + "/unread"]}]
    values.append({"answer": rows})
    # Deliberately bypass the fake grammar to exercise the unchanged guard.
    policy = Policy([reviewed(value) for value in values], validate=False)
    result = asyncio.run(EpisodeRunner(max_steps=len(values), web_tools_factory=Web,
        web_structured_final=True, web_structured_actions=True, web_explicit_review=True).run(task(), policy, evaluation=True))
    assert result.status == "evidence_failure"


@pytest.mark.parametrize("benchmark,evaluation,enabled", [("cuad", True, True), ("widesearch", False, True), ("widesearch", True, False)])
def test_cuad_training_and_default_prompts_keep_original_behavior(benchmark, evaluation, enabled):
    target = task() if benchmark == "widesearch" else Task("contract", "cuad", "validation", "Find clause", document="Synthetic", gold=[])
    policy = Policy([{"answer": []}], validate=False)
    runner = EpisodeRunner(max_steps=1, web_tools_factory=Web, web_structured_final=True,
                          web_structured_actions=True, web_explicit_review=enabled)
    asyncio.run(runner.run(target, policy, evaluation=evaluation))
    assert policy.calls[0][1][0]["content"] == _system_prompt(target, 1)
    assert WEB_EXPLICIT_REVIEW_INSTRUCTIONS not in json.dumps(policy.calls)
    if benchmark == "cuad" or not evaluation:
        assert policy.calls[0][0] == "ordinary"


def test_existing_parser_ignores_review_when_dispatching_tool_and_final():
    for value in [{"tool": "web_search", "arguments": {"query": "synthetic"}}, {"answer": []}]:
        wrapped = reviewed(value)
        assert parse_action(json.dumps(wrapped), widesearch_rows=True) == wrapped


def test_preflight_compiles_every_reviewed_action_state_without_engine(monkeypatch):
    calls = []
    def sampling(*, structured_outputs=None):
        pass
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=sampling))
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", SimpleNamespace(StructuredOutputsParams=lambda **kwargs: None))
    from slime_plugins.dcc import web_structured
    monkeypatch.setattr(web_structured, "validate_web_structured_schema", lambda schema: calls.append(schema))
    preflight_web_structured(actions=True, explicit_review=True)
    assert len(calls) == 9
    for schema in calls:
        for branch in schema.get("anyOf", [schema]):
            assert branch["required"][:2] == ["plan", "checks"]
    with pytest.raises(ValueError, match="requires structured actions"):
        preflight_web_structured(explicit_review=True)


def test_loader_preflights_reviewed_grammar_and_attests_mode_with_cpu_doubles(monkeypatch):
    helpers = runpy.run_path(str(ROOT / "tests/test_dcc_vllm_policy.py"))
    helpers["runtime"].__wrapped__(monkeypatch)
    import transformers
    from slime_plugins.dcc import vllm_policy
    calls = []
    monkeypatch.setattr(vllm_policy, "preflight_web_structured", lambda **kwargs: calls.append(kwargs))
    decoder_configuration = {"version": "web-review-finite-right-recursive-v1",
                             "xgrammar_version": "0.1.25"}
    decoder_preflight_calls = []
    monkeypatch.setattr(vllm_policy, "preflight_web_review_decoder",
                        lambda: decoder_preflight_calls.append(True) or decoder_configuration)
    monkeypatch.setitem(sys.modules, "vllm.config", SimpleNamespace(StructuredOutputsConfig=lambda **kwargs: SimpleNamespace(**kwargs)))
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **kw: SimpleNamespace(model_type="qwen3_moe"))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: helpers["Tokenizer"]())
    monkeypatch.setattr(transformers.GenerationConfig, "from_pretrained", lambda *a, **kw: SimpleNamespace(eos_token_id=1))
    policy = vllm_policy.load_vllm_policy("synthetic/model", web_structured_final=True,
                                        web_structured_actions=True, web_explicit_review=True)
    assert calls == [{"actions": True, "explicit_review": True}]
    assert policy.provenance["web_explicit_review"] == CONFIG
    assert decoder_preflight_calls == [True]
    assert policy.provenance["web_review_decoder"] == decoder_configuration
    with pytest.raises(ValueError, match="requires structured Web actions"):
        vllm_policy.load_vllm_policy("synthetic/model", web_explicit_review=True)


@pytest.mark.parametrize("damage", [None, "manifest", "runtime", "protocol", "version", "prompt", "limit", "extra", "bool", "actions"])
def test_paired_protocol_requires_complete_matching_review_and_dependencies(tmp_path, damage):
    helpers = runpy.run_path(str(ROOT / "tests/test_dcc_compare_runs.py"))
    values = helpers["pair"](tmp_path, [helpers["web"]("synthetic")])
    for run, declaration in zip(values[:2], values[2:], strict=True):
        for record in (run["manifest"], run["runtime"], declaration["common"]["generation"]):
            record.update(web_explicit_review=deepcopy(CONFIG), web_structured_final=web_structured_configuration(True),
                          web_structured_actions=web_structured_actions_configuration(True))
    if damage in {"manifest", "runtime"}:
        values[0][damage].pop("web_explicit_review")
    elif damage == "protocol":
        values[2]["common"]["generation"].pop("web_explicit_review")
    elif damage in {"version", "prompt", "limit", "extra", "bool"}:
        key, value = {"version": ("schema_version", "unsupported"), "prompt": ("prompt_version", "unsupported"),
                      "limit": ("max_chars", 241), "extra": ("extra", True), "bool": ("enabled", 1)}[damage]
        values[0]["manifest"]["web_explicit_review"][key] = value
    elif damage == "actions":
        values[0]["manifest"].pop("web_structured_actions")
    if damage:
        with pytest.raises(ValueError, match="[Ww]eb explicit review"):
            helpers["COMPARE"]["validate_protocols"](*values)
    else:
        helpers["COMPARE"]["validate_protocols"](*values)


def test_scoring_rejects_missing_review_runtime(tmp_path):
    helpers = runpy.run_path(str(ROOT / "tests/test_dcc_model_scoring.py"))
    tasks = helpers["logical_predictions"](tmp_path)
    for path in tmp_path.glob("manifest_rank*.json"):
        record = json.loads(path.read_text())
        record["web_explicit_review"] = deepcopy(CONFIG)
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="[Ww]eb explicit review"):
        helpers["cli"]()["_score_predictions"](tmp_path, tasks)


@pytest.mark.parametrize("enabled", [False, True])
def test_evaluation_metadata_loader_and_resume_match_exact_optional_mode(monkeypatch, tmp_path, enabled):
    helpers = runpy.run_path(str(ROOT / "tests/test_dcc_model_scoring.py"))
    policy = Policy([reviewed({"answer": []})] if enabled else [{"answer": []}], validate=False)
    policy.provenance = {"backend": "vllm", "model": "example/model"}
    async def close():
        pass
    policy.close = close
    if enabled:
        policy.provenance.update(web_explicit_review=deepcopy(CONFIG), web_structured_final=web_structured_configuration(True),
                                 web_structured_actions=web_structured_actions_configuration(True))
    evaluate, args = helpers["stub_evaluation"](monkeypatch, tmp_path, policy, [task()])
    args.max_steps = 1
    seen = []
    def load(*pos, **kwargs):
        seen.append(kwargs)
        return policy
    monkeypatch.setitem(sys.modules, "slime_plugins.dcc.vllm_policy", SimpleNamespace(load_vllm_policy=load))
    if enabled:
        args.web_explicit_review = args.web_structured_final = args.web_structured_actions = True
    asyncio.run(evaluate(args))
    for name in ["manifest_rank0.json", "inference_runtime.json"]:
        record = json.loads((tmp_path / name).read_text())
        if enabled:
            assert record["web_explicit_review"] == CONFIG
        else:
            assert "web_explicit_review" not in record
    assert (seen[0].get("web_explicit_review") is True) is enabled
    if enabled:
        args.web_explicit_review = False
        with pytest.raises(ValueError, match="manifest|settings|configuration"):
            asyncio.run(evaluate(args))


def test_review_dependencies_fail_before_policy_loading():
    evaluate = runpy.run_path(str(ROOT / "scripts/dcc_model_evaluate.py"))["evaluate"]
    with pytest.raises(ValueError, match="web-explicit-review requires"):
        asyncio.run(evaluate(SimpleNamespace(web_explicit_review=True)))
    with pytest.raises(ValueError, match="requires structured Web actions"):
        EpisodeRunner(web_explicit_review=True)
