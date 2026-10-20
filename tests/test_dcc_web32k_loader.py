"""Semantic CPU fixtures: real validators and native code, synthetic engine only.

No GPU, weights, benchmark files, network, collection or held-out examples.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import asyncio
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SOURCE / "scripts"), str(SOURCE / "slime")]
import dcc_native_sft as N
import dcc_review_training as R
from slime_plugins.dcc.prompt_history import PromptHistoryConfig, fit_prompt_history
from slime_plugins.dcc.native_budget import NativeBudget

NATIVE = runpy.run_path(str(SOURCE / "tests/test_dcc_native_thinking.py"))
REVIEW = runpy.run_path(str(SOURCE / "tests/test_dcc_review_training.py"))
CUAD = runpy.run_path(str(SOURCE / "tests/test_dcc_native_train_data_fixes.py"))


def web_profile():
    return N.WebSchemaProfile(20, True, False, True, R.WEB32K_PROFILE)


def public_web_record():
    """Two complete 20k source reads; only the newest must fit the 32k prompt."""
    class LargeTools(REVIEW["Tools"]):
        async def fetch_url(self, url):
            text = REVIEW["TEXT"] + " public-context" * 1500
            return {"url": url, "text": text, "start": 0, "end": len(text)}

    from slime_plugins.dcc.data import Task
    task = Task("web32k-public-fixture", "widesearch", "train", "Find the name in the public list.",
                metadata={"evaluation": {"required": ["Name"]}})
    policy = REVIEW["Policy"]([REVIEW["action"]("web_search"), REVIEW["action"]("fetch_url"),
                              REVIEW["action"]("fetch_url"), REVIEW["action"]()])
    runner = REVIEW["C"].make_runner("widesearch", N.NativeSchemaProfile(web_profile()),
                                    timeout_seconds=5, web_tools_factory=LargeTools)
    result = asyncio.run(runner.run(task, policy, evaluation=True, score_train=False))
    assert result.status == "completed", result.detail
    messages, options, value = policy.calls[-1]
    return dict(source="independent_curriculum", split="train", task_id=task.id,
                stage="final", messages=messages, completion=json.dumps(value)), options["json_schema"], 3


def produce(tmp_path, monkeypatch, record, schema, turn, *, managed, source="f" * 40, seed=42):
    cls = NATIVE["FakeTokenizer"]
    monkeypatch.setattr(cls, "__len__", lambda self: 160000, raising=False)
    monkeypatch.setattr(cls, "chat_template", "synthetic-template", raising=False)
    tmp_path.mkdir(parents=True, exist_ok=True)
    budget = NativeBudget(16384, "thinking")
    tokenizer, engine, original = NATIVE["make_policy"](tmp_path,
        values=[json.loads(record["completion"])], budget=budget)
    config = PromptHistoryConfig(managed)
    base = original.base
    base.allow_truncation = False
    base.max_prompt_tokens = 32768 if managed else 16384
    base.max_model_len = base.max_prompt_tokens + budget.total_tokens
    base.seed = seed
    receipt = deepcopy(NATIVE["RECEIPT"])
    receipt["source_commit"] = source
    native = type(original)(base, source_receipt=receipt, trace_sink=original.trace_sink,
                            budget=budget, prompt_history_config=config)
    trace_id = "fixture-" + record["task_id"]

    async def generate():
        try:
            await native.generate_structured(record["messages"], max_new_tokens=budget.total_tokens,
                temperature=0.0, json_schema=schema, trace_metadata=dict(trace_id=trace_id,
                    task_id=record["task_id"], rank=0, turn_index=turn))
        finally:
            await native.close()

    asyncio.run(generate())
    identity = N.NativeIdentity("synthetic-derivative", (151643, 151645), source,
        base.max_prompt_tokens, base.max_model_len, seed, budget.thought_tokens,
        budget.thought_sampling_profile, config.to_dict() if managed else None)
    bundle = dict(contract=N.CONTRACT, record=deepcopy(record), identity=identity.manifest(tokenizer),
        collector=dict(split="train", task_input="public-train-task-without-gold",
            gold_visible_to_policy=False, recorded_messages=True, synthetic_thought=False,
            inserted_close=False, inserted_eos=False, episode_id="fixture-episode-" + record["task_id"],
            turn=turn, rank=0, trace_id=trace_id), native_trace=deepcopy(native.traces[0]))
    return tokenizer, identity, bundle, engine


def fixture(tmp_path, monkeypatch):
    record, schema, turn = public_web_record()
    tokenizer, identity, bundle, engine = produce(tmp_path, monkeypatch, record, schema, turn, managed=True)
    return tokenizer, identity, bundle, schema, engine


def trusted_profile(web, cuad=None):
    result = dict(version=R.WEB32K_PROFILE, identity=deepcopy(web), collection_overlay_id="synthetic-web32k-overlay",
                  web=dict(max_steps=20, explicit_review=True, cached_rows=False, navigation_ledger=True))
    if cuad is not None:
        from slime_plugins.dcc.cuad_structured import CUAD_STRUCTURED_SCHEMA_VERSION
        result.update(cuad=dict(max_steps=4, structured_actions=True, schema_version=CUAD_STRUCTURED_SCHEMA_VERSION),
                      cuad_identity=deepcopy(cuad), cuad_identity_differences=sorted(
                          k for k in set(web) | set(cuad) if web.get(k) != cuad.get(k)))
    return result


def test_effective_native_ids_replay_and_full_history_schema_audit(tmp_path, monkeypatch):
    tokenizer, identity, bundle, schema, engine = fixture(tmp_path, monkeypatch)
    before = deepcopy(bundle)
    trace = bundle["native_trace"]
    assert trace["prompt_history"]["compacted"] is True
    assert trace["prompt_history"]["original_prompt_tokens"] > 32768
    assert trace["prompt_history"]["actual_prompt_tokens"] <= 32768
    assert trace["prompt_history_original_messages"] == bundle["record"]["messages"]
    assert web_profile().original_schema(bundle["record"], bundle["collector"]) == schema
    example = N.validate_native_record(bundle, tokenizer, expected_identity=identity,
                                      original_schema=schema, web_schema_profile=web_profile())
    assert example.prompt == engine.calls[0][0]["prompt_token_ids"] == trace["original_prompt_ids"]
    assert example.completion == trace["completion_ids"]
    assert len(engine.calls) == 2 and bundle == before


@pytest.mark.parametrize("damage", ["history", "original_history", "indices", "configuration", "notice",
                                    "url", "prompt", "count", "missing_report", "preflight"])
def test_replay_rejects_tampered_context_receipt_and_nontrainable_preflight(tmp_path, monkeypatch, damage):
    tokenizer, identity, bundle, schema, _ = fixture(tmp_path, monkeypatch)
    trace = bundle["native_trace"]
    if damage == "history": bundle["record"]["messages"][3]["content"] += " forged"
    elif damage == "original_history": trace["prompt_history_original_messages"][1]["content"] += " forged"
    elif damage == "indices": trace["prompt_history"]["retained_message_indices"] = [0, 1]
    elif damage == "configuration": trace["prompt_history"]["configuration"]["keep_recent_turns"] = 2
    elif damage == "notice": trace["prompt_history"]["notice"]["content"] += " forged"
    elif damage == "url": trace["prompt_history"]["omitted_observed_urls"][0]["url"] = "https://forged.example/"
    elif damage == "prompt": trace["original_prompt_ids"][0] += 1
    elif damage == "count": trace["prompt_history"]["original_prompt_tokens"] += 1
    elif damage == "missing_report": del trace["prompt_history"]
    else:
        trace.update(status="preflight_failure", failure_reason="minimum_complete_context_exceeds_budget",
                     generation_submitted=False, phases=[])
    with pytest.raises(ValueError):
        N.validate_native_record(bundle, tokenizer, expected_identity=identity,
                                 original_schema=schema, web_schema_profile=web_profile())


@pytest.mark.parametrize("damage", ["legacy_context", "extra", "cuad_without_identity", "unlisted", "model", "web16", "disabled"])
def test_new_context_and_domain_policy_never_smuggled_into_old_profiles(tmp_path, monkeypatch, damage):
    tokenizer, identity, bundle, schema, _ = fixture(tmp_path, monkeypatch)
    old = replace(identity, source_commit="e" * 40, generation_seed=58,
                  max_prompt_tokens=16384, max_model_len=36864, prompt_history=None).manifest(tokenizer)
    profile = trusted_profile(identity.manifest(tokenizer), old)
    if damage == "legacy_context": profile = dict(version=R.LEGACY_PROFILE, identity=profile["identity"],
        web=dict(max_steps=8, explicit_review=False, cached_rows=False))
    elif damage == "extra": profile["unapproved"] = True
    elif damage == "cuad_without_identity": del profile["cuad_identity"]
    elif damage == "unlisted": profile["cuad_identity_differences"].remove("source_commit")
    elif damage == "model": profile["cuad_identity"]["derivative_id"] = "other"
    elif damage == "web16": profile["identity"]["max_prompt_tokens"] = 16384
    else: profile["identity"]["prompt_history"] = PromptHistoryConfig(False).to_dict()
    with pytest.raises(ValueError): R.approved_web_profile(profile, N.WebSchemaProfile)


def mixed_files(tmp_path, monkeypatch):
    tokenizer, web_identity, web_bundle, schema, _ = fixture(tmp_path / "web", monkeypatch)
    cuad_record = CUAD["direct_cuad"]()
    # At turn zero the approved CUAD action schema has only the answer field;
    # 40-word plan/checks controls apply only to its actual final-budget turn.
    cuad_record["completion"] = json.dumps({"answer": json.loads(cuad_record["completion"])["answer"]})
    from slime_plugins.dcc.cuad_structured import cuad_action_schema
    _, cuad_identity, cuad_bundle, _ = produce(tmp_path / "cuad", monkeypatch, cuad_record,
        cuad_action_schema(), 0, managed=False, source="e" * 40, seed=58)
    web_path, cuad_path = tmp_path / "web.jsonl", tmp_path / "cuad.jsonl"
    web_path.write_text(json.dumps(web_bundle) + "\n")
    cuad_path.write_text(json.dumps(cuad_bundle) + "\n")
    profile = trusted_profile(web_identity.manifest(tokenizer), cuad_identity.manifest(tokenizer))
    return tokenizer, web_identity, cuad_identity, web_bundle, cuad_bundle, web_path, cuad_path, profile


def test_mixed_trusted_file_identities_preserve_old_cuad_bytes_and_fail_budget_overflow(tmp_path, monkeypatch):
    tokenizer, web, cuad, wb, cb, web_path, cuad_path, profile = mixed_files(tmp_path, monkeypatch)
    before = cuad_path.read_bytes()
    root = tmp_path / "trusted-source"; root.mkdir()
    (root / ".dcc-native-source.json").write_text(json.dumps(NATIVE["RECEIPT"]))
    (root / ".dcc-training-code-overlay.json").write_text(json.dumps(dict(
        overlay_id=profile["collection_overlay_id"], explicit_training_profile=R.WEB32K_PROFILE,
        baseline_source_unchanged=False)))
    path = tmp_path / "profile.json"; path.write_text(json.dumps(profile))
    args = SimpleNamespace(adapter=None, cuad_train_data=None, native_expected_source=web.source_commit,
        native_train_profile=str(path), cuad_tool_records=str(cuad_path), records=str(web_path), model="unused")
    data = N.prepare_native_training(args, source_root=root, tokenizer=tokenizer)
    assert len(data.records) == len(data.cuad_records) == 1
    assert data.cuad_identity == cuad and data.identity == web
    assert data.configuration["domain_identities"]["cuad_train"] == cb["identity"]
    assert data.configuration["cuad_identity_differences"] == profile["cuad_identity_differences"]
    skips = Counter()
    for record in data.records + data.cuad_records:
        ex = data.encode_record(record, tokenizer, max_prompt=32768, max_response=20480,
                                batch_tokens=212992, skips=skips)
        assert ex.completion == (wb if record["source"] == "independent_curriculum" else cb)["native_trace"]["completion_ids"]
    with pytest.raises(ValueError, match="refusing to skip"):
        data.encode_record(data.records[0], tokenizer, max_prompt=1, max_response=20480,
                           batch_tokens=212992, skips=skips)
    assert skips == Counter() and cuad_path.read_bytes() == before


@pytest.mark.parametrize("damage", ["wrong_domain", "row_identity", "undeclared_differences", "different_model"])
def test_loader_uses_trusted_file_domain_and_rejects_identity_mixing(tmp_path, monkeypatch, damage):
    tokenizer, web, cuad, wb, cb, web_path, cuad_path, profile = mixed_files(tmp_path, monkeypatch)
    diffs = profile["cuad_identity_differences"]
    if damage == "wrong_domain": web_path, cuad_path = cuad_path, web_path
    elif damage == "row_identity":
        wb["identity"] = cb["identity"]; web_path.write_text(json.dumps(wb) + "\n")
    elif damage == "undeclared_differences": diffs = ["max_prompt_tokens"]
    else: cuad = replace(cuad, derivative_id="unapproved-model")
    schema_profile = N.NativeSchemaProfile(web_profile(), N.CuadSchemaProfile(**profile["cuad"]))
    with pytest.raises(ValueError):
        N.load_native_training_data(web_path, tokenizer, expected_identity=web,
            source_receipt=NATIVE["RECEIPT"], schema_resolver=schema_profile.original_schema,
            cuad_path=cuad_path, expected_cuad_identity=cuad, approved_cuad_identity_differences=diffs,
            strict_training_budgets=True)


def test_old_default_manifests_unchanged_and_new_context_requires_exact_declaration(tmp_path, monkeypatch):
    tokenizer, identity, _, _, _ = fixture(tmp_path, monkeypatch)
    old = replace(identity, max_prompt_tokens=16384, max_model_len=36864, prompt_history=None)
    manifest = old.manifest(tokenizer)
    assert "prompt_history" not in manifest
    assert N.identity_from_manifest(manifest, old.source_commit) == old
    before = deepcopy(manifest)
    R.approved_web_profile(dict(version=R.REVIEW20_PROFILE, identity=manifest,
        collection_overlay_id="synthetic-old-overlay", web=dict(max_steps=20,
            explicit_review=True, cached_rows=False, navigation_ledger=True)), N.WebSchemaProfile)
    assert manifest == before
    malformed = deepcopy(identity.prompt_history); malformed["unapproved"] = True
    with pytest.raises(ValueError): replace(identity, prompt_history=malformed).manifest(tokenizer)


def test_full_history_replay_remains_strict_after_effective_prompt_compaction(tmp_path, monkeypatch):
    _, _, bundle, schema, _ = fixture(tmp_path, monkeypatch)
    record = deepcopy(bundle["record"])
    # An omitted historical observation still belongs to complete schema/grounding audit.
    record["messages"][3]["content"] = record["messages"][3]["content"].replace("Remaining turns: 19", "Remaining turns: 3")
    with pytest.raises(ValueError): web_profile().original_schema(record, bundle["collector"])
