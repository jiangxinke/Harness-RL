"""CPU-only genuine policy control flow with synthetic engine responses.

No model weights, CUDA allocation, labels, network, DEV or TEST are used.
"""
from copy import deepcopy
from dataclasses import replace
import asyncio
import importlib
import json
import os
from pathlib import Path
import runpy
import sys

import pytest

SOURCE = Path(os.environ.get("DCC_REVIEW_SOURCE", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
G = importlib.import_module("dcc_collect_native_preferences")
N = importlib.import_module("dcc_native_sft")
C = importlib.import_module("dcc_collect_native_train")
FIXTURE = runpy.run_path(str(SOURCE / "tests/test_dcc_native_thinking.py"))
REVIEW = runpy.run_path(str(SOURCE / "tests/test_dcc_review_training.py"))


@pytest.fixture
def fake_tokenizer(monkeypatch):
    cls = FIXTURE["FakeTokenizer"]
    monkeypatch.setattr(cls, "__len__", lambda self: 160000, raising=False)
    monkeypatch.setattr(cls, "chat_template", "synthetic-template", raising=False)
    return cls()


def identity(tokenizer):
    return N.NativeIdentity("synthetic-derivative", (151643, 151645), "f" * 40,
        16384, 36864, 42, 16384, "thinking")


def web_task():
    return G.Task("review-public-fixture", "widesearch", "train", "Find the name in the public list.",
        metadata={"evaluation": {"required": ["Name"]}})


def source_profile(tokenizer):
    return {"version": "dcc-native-train-profile-web20-review-navigation-v1",
        "identity": identity(tokenizer).manifest(tokenizer),
        "collection_overlay_id": "synthetic-explicit-overlay",
        "web": {"max_steps": 20, "explicit_review": True, "cached_rows": False,
                "navigation_ledger": True}}


def export_args(tokenizer, **extra):
    return dict(source_profile=source_profile(tokenizer), source_receipt=FIXTURE["RECEIPT"],
        tokenizer=tokenizer, candidate_identity=identity(tokenizer),
        candidate_source_receipt=FIXTURE["RECEIPT"], experiment_id="prospective-fixture",
        harness_id="explicit-review-final-intervention", seeds=[11, 23, 47, 71], **extra)


def make_native(path, *, values, review=True):
    from slime_plugins.dcc.native_budget import NativeBudget
    path.mkdir(parents=True, exist_ok=True)
    result = FIXTURE["make_policy"](path, values=values, review=review,
                                     budget=NativeBudget(16384, "thinking"))
    result[2].base.allow_truncation = False
    return result


def make_closed_raw(path, tokenizer, *, rejected=False, length=3):
    values = [REVIEW["action"]("web_search"), REVIEW["action"]("fetch_url"),
              *[REVIEW["action"]("web_search") for _ in range(length - 3)], REVIEW["action"]()]
    if rejected:
        values[-1]["answer"] = []  # Physical final eligible despite failed answer/evidence acceptance.
    _, engine, native = make_native(path, values=values)
    profile = source_profile(tokenizer)
    schemas = G.source_schema_profile(profile)
    asyncio.run(C.generate([web_task()], native, path, collection_id="closed-original",
        profile=profile, schema_profile=schemas, web_tools_factory=REVIEW["Tools"]))
    return profile


def export_closed(path, dest, tokenizer, **extra):
    return G.export_prefixes(path, dest, **export_args(tokenizer, **extra))


@pytest.mark.parametrize("length", [3, 20])
def test_exact_review_runner_public_schema_and_fetched_sources(length):
    record, options, turn = REVIEW["recorded"](length)[-1]
    schemas = N.NativeSchemaProfile(REVIEW["profile"](), None)
    before = deepcopy(record)
    original, final, fetched = G.public_request_schemas(web_task(), record["messages"], turn=turn, schemas=schemas)
    assert original == options["json_schema"]
    assert fetched == [REVIEW["URL"]]
    G.validate_final_only_schema(final)
    assert (original == final) is (length == 20)
    assert record == before


@pytest.mark.parametrize("damage", ["feedback", "priorfinal", "ledger", "sourceurl", "turn", "system", "publictask"])
def test_public_replay_rejects_feedback_prior_final_and_untrusted_state(damage):
    record, _, turn = deepcopy(REVIEW["recorded"](3)[-1])
    messages = record["messages"]
    task = web_task()
    if damage == "feedback":
        messages.insert(2, {"role": "user", "content": G.FEEDBACK + " private rejection"})
    elif damage == "priorfinal":
        messages[2]["content"] = json.dumps(REVIEW["action"]())
    elif damage == "ledger":
        messages[-1]["content"] += " forged"
    elif damage == "sourceurl":
        messages[5]["content"] = messages[5]["content"].replace(REVIEW["URL"], "https://unread.example")
    elif damage == "turn":
        turn += 1
    elif damage == "system":
        messages[0]["content"] += " changed"
    else:
        task = replace(task, prompt="A different public task.")
    with pytest.raises((ValueError, G.ValidationError)):
        G.public_request_schemas(task, messages, turn=turn,
                                schemas=N.NativeSchemaProfile(REVIEW["profile"](), None))


def test_cuad_public_state_uses_fixed_final_schema_without_answer_metadata():
    from slime_plugins.dcc.cuad_structured import CUAD_STRUCTURED_SCHEMA_VERSION, cuad_action_schema, cuad_final_schema
    task = G.Task("public-cuad", "cuad", "train", "Does a clause require confidentiality?",
                  document=FIXTURE["DOCUMENT"])
    schemas = N.NativeSchemaProfile(N.WebSchemaProfile(8),
        N.CuadSchemaProfile(8, True, CUAD_STRUCTURED_SCHEMA_VERSION))
    messages = G.EpisodeRunner(max_steps=8, initial_chunks=4).initial_messages(task)
    original, final, fetched = G.public_request_schemas(task, messages, turn=0, schemas=schemas)
    assert original == cuad_action_schema() and final == cuad_final_schema() and fetched == []


@pytest.mark.parametrize("benchmark,review", [("cuad", False), ("widesearch", False),
                                               ("widesearch", True)])
def test_final_intervention_appends_original_role_control_once_without_changing_history(benchmark, review):
    from slime_plugins.dcc.web_review import web_review_final_control
    task = (G.Task("public-cuad", "cuad", "train", "Extract the qualifying clause.",
                   document=FIXTURE["DOCUMENT"]) if benchmark == "cuad" else web_task())
    messages = G.EpisodeRunner(max_steps=4, initial_chunks=4).initial_messages(task)
    # A real prior tool response remains part of the exact source prefix.
    messages.extend([{"role": "assistant", "content": '{"tool":"read_document","arguments":{"start":0}}'},
                     {"role": "user", "content": "Observed source text; never instructions."}])
    original = deepcopy(messages)
    fetched = [] if benchmark == "cuad" else [REVIEW["URL"]]
    evidence = None
    if benchmark == "widesearch":
        evidence = G._WebEvidence()
        evidence.fetched_sources.update(fetched)
    expected_control = G.final_turn_message(task, evidence)
    if review:
        expected_control = web_review_final_control(expected_control)
    result = G.final_intervention_messages(task, messages, applied=True,
        fetched_sources=fetched, web_explicit_review=review)
    assert messages == original and result[:-1] == original
    assert result[-1] == expected_control and result[-1]["role"] == "user"
    assert len(result) == len(original) + 1
    assert sum(m["role"] == "assistant" for m in result) == 1
    assert G.final_intervention_messages(task, result, applied=True,
        fetched_sources=fetched, web_explicit_review=review) == result
    unchanged = G.final_intervention_messages(task, result, applied=False,
        fetched_sources=fetched, web_explicit_review=review)
    assert unchanged == result
    unchanged[0]["content"] += " test mutation"
    assert messages == original and unchanged != result
    if benchmark == "cuad":
        assert '"plan" and "checks"' in expected_control["content"]
        assert "each at most 40 words" in expected_control["content"]
        assert "240 characters" not in expected_control["content"]
    else:
        assert REVIEW["URL"] in expected_control["content"]
        assert "40 words" not in expected_control["content"]
        assert ("1 to 240 characters" in expected_control["content"]) is review


def test_export_defaults_fail_closed_when_original_schema_still_permits_tools(tmp_path, fake_tokenizer):
    make_closed_raw(tmp_path / "raw", fake_tokenizer)
    with pytest.raises(ValueError, match="No eligible"):
        export_closed(tmp_path / "raw", tmp_path / "export", fake_tokenizer)
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize("rejected", [False, True])
def test_export_accepts_physical_poor_originals_discards_targets_and_freezes_ids(tmp_path, fake_tokenizer, rejected):
    make_closed_raw(tmp_path / "raw", fake_tokenizer, rejected=rejected)
    freeze = export_closed(tmp_path / "raw", tmp_path / "export", fake_tokenizer,
                           schema_policy=G.FINAL_INTERVENTION)
    loaded, prefixes, native_identity = G.load_frozen(tmp_path / "export", fake_tokenizer)
    assert freeze == loaded and len(prefixes) == 1 and freeze["seeds"] == [11, 23, 47, 71]
    assert freeze["selection_policy"] == G.SELECTION
    assert freeze["benchmarks"] == ["cuad", "widesearch"]
    assert freeze["prefix_selection_seed"] == 47
    prefix = prefixes[0].to_dict()
    assert "generation_seed" not in prefix["native_identity"]
    assert not {"gold", "category", "polarity", "completion", "reward", "reward_version"}.intersection(prefix)
    assert "POISON" not in json.dumps(prefix)  # Original native thought never enters new prompt.
    lineage = list(G.read_jsonl(tmp_path / "export/lineage.jsonl"))[0]
    assert lineage["intervention_applied"] is True
    assert "anyOf" in lineage["original_json_schema"]
    assert "anyOf" not in lineage["generation_json_schema"]
    assert lineage["fetched_sources"] == [REVIEW["URL"]]
    assert lineage["original_completion_retained"] is False and lineage["sft_acceptance_used"] is False
    original_request = next(row for row in G.read_jsonl(tmp_path / "raw/requests_rank0.jsonl")
        if row["trace_id"] == lineage["source_trace_id"] and row["event"] == "request")
    assert prefix["messages"][:-1] == original_request["messages"]
    assert prefix["messages"][-1]["role"] == "user"
    assert "1 to 240 characters" in prefix["messages"][-1]["content"]
    assert lineage["prompt_intervention"] == {
        "version": G.PROMPT_INTERVENTION_VERSION, "applied": True,
        "source_messages_preserved": True,
        "source_message_count": len(original_request["messages"]), "appended_message_count": 1}
    if rejected:
        episode = list(G.read_jsonl(tmp_path / "raw/episodes_rank0.jsonl"))[0]
        assert episode["status"] != "completed"  # No accepted-SFT/easy-original bias.


def test_export_cannot_read_unclosed_or_tampered_source_journals(tmp_path, fake_tokenizer):
    make_closed_raw(tmp_path / "raw", fake_tokenizer)
    closure = tmp_path / "raw/generation_closed_rank0.json"
    closure.unlink()
    with pytest.raises(FileNotFoundError):
        export_closed(tmp_path / "raw", tmp_path / "export", fake_tokenizer,
                      schema_policy=G.FINAL_INTERVENTION)
    assert not (tmp_path / "export").exists()


def test_raw_last_turn_default_keeps_original_final_schema(tmp_path, fake_tokenizer):
    make_closed_raw(tmp_path / "raw", fake_tokenizer, length=20)
    export_closed(tmp_path / "raw", tmp_path / "export", fake_tokenizer)
    _, prefixes, _ = G.load_frozen(tmp_path / "export", fake_tokenizer)
    lineage = list(G.read_jsonl(tmp_path / "export/lineage.jsonl"))[0]
    assert lineage["turn"] == 19 and lineage["intervention_applied"] is False
    assert lineage["original_json_schema"] == lineage["generation_json_schema"]
    assert list(prefixes[0].to_dict()["json_schema"]["properties"]) == ["plan", "checks", "answer"]
    assert lineage["prompt_intervention"]["applied"] is False
    assert lineage["prompt_intervention"]["appended_message_count"] == 0


def test_raw_cuad_wrong_ungrounded_final_is_not_filtered_by_sft_acceptance(tmp_path, fake_tokenizer):
    from slime_plugins.dcc.cuad_structured import CUAD_STRUCTURED_SCHEMA_VERSION
    task = G.Task("synthetic-cuad", "cuad", "train", "Extract a confidentiality clause.",
                  document=FIXTURE["DOCUMENT"])
    value = {"answer": [{"text": "Completely invented unrelated clause", "confidence": 0.5}]}
    _, _, native = make_native(tmp_path / "raw", values=[value])
    profile = source_profile(fake_tokenizer)
    profile["cuad"] = {"max_steps": 8, "structured_actions": True,
                       "schema_version": CUAD_STRUCTURED_SCHEMA_VERSION}
    asyncio.run(C.generate([task], native, tmp_path / "raw", collection_id="cuad-original",
        profile=profile, schema_profile=G.source_schema_profile(profile)))
    args = export_args(fake_tokenizer, schema_policy=G.FINAL_INTERVENTION)
    args["source_profile"] = profile
    G.export_prefixes(tmp_path / "raw", tmp_path / "export", **args)
    _, prefixes, _ = G.load_frozen(tmp_path / "export", fake_tokenizer)
    assert len(prefixes) == 1 and prefixes[0].benchmark == "cuad"
    assert "invented unrelated" not in prefixes[0].messages_json
    assert "grounding_fields" not in prefixes[0].to_dict()
    prefix = prefixes[0].to_dict()
    lineage = list(G.read_jsonl(tmp_path / "export/lineage.jsonl"))[0]
    original_request = next(row for row in G.read_jsonl(tmp_path / "raw/requests_rank0.jsonl")
        if row["trace_id"] == lineage["source_trace_id"] and row["event"] == "request")
    assert prefix["messages"][:-1] == original_request["messages"]
    assert prefix["messages"][-1] == G.final_turn_message(task)
    assert "each at most 40 words" in prefix["messages"][-1]["content"]
    assert prefix["json_schema"] == lineage["generation_json_schema"]
    from slime_plugins.dcc.cuad_structured import cuad_final_schema
    assert prefix["json_schema"] == cuad_final_schema()
    assert lineage["prompt_intervention"]["appended_message_count"] == 1


def test_v2_loader_rejects_old_schema_only_v1_freeze(tmp_path, fake_tokenizer):
    frozen_case(tmp_path, fake_tokenizer)
    path = tmp_path / "export/freeze.json"
    freeze = json.loads(path.read_text())
    assert freeze["version"] == "native-final-g4-prospective-v2"
    freeze["version"] = "native-final-g4-prospective-v1"
    path.write_text(json.dumps(freeze))
    with pytest.raises(ValueError, match="prospective freeze"):
        G.load_frozen(tmp_path / "export", fake_tokenizer)


def test_raw_cuad_last_turn_preserves_original_fixed_final_schema(tmp_path, fake_tokenizer):
    from slime_plugins.dcc.cuad_structured import CUAD_STRUCTURED_SCHEMA_VERSION
    task = G.Task("synthetic-cuad-last", "cuad", "train", "Extract a confidentiality clause.",
                  document=FIXTURE["DOCUMENT"])
    values = [{"tool": "search_document", "arguments": {"query": "confidential"}} for _ in range(7)]
    values.append({"plan": "Inspect the task.", "checks": "Use observed evidence.", "answer": []})
    _, _, native = make_native(tmp_path / "raw", values=values)
    profile = source_profile(fake_tokenizer)
    profile["cuad"] = {"max_steps": 8, "structured_actions": True,
                       "schema_version": CUAD_STRUCTURED_SCHEMA_VERSION}
    asyncio.run(C.generate([task], native, tmp_path / "raw", collection_id="cuad-last",
        profile=profile, schema_profile=G.source_schema_profile(profile)))
    args = export_args(fake_tokenizer); args["source_profile"] = profile
    G.export_prefixes(tmp_path / "raw", tmp_path / "export", **args)
    lineage = list(G.read_jsonl(tmp_path / "export/lineage.jsonl"))[0]
    assert lineage["turn"] == 7 and lineage["intervention_applied"] is False


def test_mixed_closed_source_exports_independently_frozen_domain_budgets(tmp_path, fake_tokenizer):
    from slime_plugins.dcc.cuad_structured import CUAD_STRUCTURED_SCHEMA_VERSION
    cuad = G.Task("public-cuad-domain", "cuad", "train", "Extract a clause.",
                  document=FIXTURE["DOCUMENT"])
    values = [{"answer": []}, REVIEW["action"]("web_search"),
              REVIEW["action"]("fetch_url"), REVIEW["action"]()]
    _, _, native = make_native(tmp_path / "raw", values=values)
    profile = source_profile(fake_tokenizer)
    profile["cuad"] = {"max_steps": 8, "structured_actions": True,
                       "schema_version": CUAD_STRUCTURED_SCHEMA_VERSION}
    asyncio.run(C.generate([cuad, web_task()], native, tmp_path / "raw", collection_id="mixed-original",
        profile=profile, schema_profile=G.source_schema_profile(profile),
        web_tools_factory=REVIEW["Tools"]))
    for benchmark, model_len in [("cuad", 36864), ("widesearch", 65536)]:
        args = export_args(fake_tokenizer, schema_policy=G.FINAL_INTERVENTION,
                           benchmarks=[benchmark], prefix_selection_seed=47)
        args["source_profile"] = profile
        args["experiment_id"] += ":" + benchmark
        args["candidate_identity"] = replace(identity(fake_tokenizer), max_model_len=model_len)
        destination = tmp_path / ("export-" + benchmark)
        freeze = G.export_prefixes(tmp_path / "raw", destination, **args)
        _, prefixes, loaded_identity = G.load_frozen(destination, fake_tokenizer)
        assert freeze["benchmarks"] == [benchmark]
        assert freeze["web_explicit_review"] is (benchmark == "widesearch")
        assert [prefix.benchmark for prefix in prefixes] == [benchmark]
        assert loaded_identity.max_model_len == model_len
        assert prefixes[0].to_dict()["native_identity"] == freeze["native_identity"]


def test_prefix_shuffle_is_public_reproducible_domain_local_and_not_first_contracts(tmp_path, fake_tokenizer):
    freeze, prefixes, _ = frozen_case(tmp_path, fake_tokenizer)
    original = prefixes[0]
    population = [(replace(original, task_id=f"task-{i:03d}", prefix_id=f"prefix-{i:03d}"), {})
                  for i in range(40)]
    args = dict(benchmarks=["widesearch"], maximum_per_benchmark=8, prefix_selection_seed=47)
    selected = G.select_public_prefixes(population, **args)
    assert selected == G.select_public_prefixes(list(reversed(population)), **args)
    assert selected != population[:8]
    assert selected != G.select_public_prefixes(population, **dict(args, prefix_selection_seed=48))
    extra_domain = [(replace(original, benchmark="cuad", source="cuad_train",
                             task_id="cuad-task", prefix_id="cuad-prefix"), {})]
    both = G.select_public_prefixes(extra_domain + population, **dict(args, benchmarks=["cuad", "widesearch"]))
    assert [item for item in both if item[0].benchmark == "widesearch"] == selected


@pytest.mark.parametrize("damage", ["domain", "seed", "cap", "policy"])
def test_loader_rejects_domain_or_sampling_freeze_tampering(tmp_path, fake_tokenizer, damage):
    frozen_case(tmp_path, fake_tokenizer)
    path = tmp_path / "export/freeze.json"
    freeze = json.loads(path.read_text())
    if damage == "domain":
        freeze["benchmarks"] = ["cuad"]
    elif damage == "seed":
        freeze["prefix_selection_seed"] = True
    elif damage == "cap":
        freeze["maximum_per_benchmark"] = 0
    else:
        freeze["selection_policy"] = "reward-selected"
    path.write_text(json.dumps(freeze))
    with pytest.raises(ValueError):
        G.load_frozen(tmp_path / "export", fake_tokenizer)


@pytest.mark.parametrize("scope", [[], ["unknown"], ["cuad", "cuad"], "cuad"])
def test_domain_scope_is_explicit_unique_and_public(scope):
    with pytest.raises(ValueError):
        G.checked_benchmarks(scope)


@pytest.mark.parametrize("seeds", [[1, 2, 3], [1, 2, 3, 3], [True, 2, 3, 4], [-1, 2, 3, 4]])
def test_exactly_four_independent_declared_seeds(seeds):
    with pytest.raises(ValueError):
        G.checked_seeds(seeds)


def frozen_case(tmp_path, fake_tokenizer):
    make_closed_raw(tmp_path / "raw", fake_tokenizer)
    export_closed(tmp_path / "raw", tmp_path / "export", fake_tokenizer,
                  schema_policy=G.FINAL_INTERVENTION)
    return G.load_frozen(tmp_path / "export", fake_tokenizer)


def test_g4_preserves_exact_native_ids_and_seeds_and_closes_every_journal(tmp_path, fake_tokenizer):
    freeze, prefixes, ident = frozen_case(tmp_path, fake_tokenizer)
    tokenizer, engine, native = make_native(tmp_path / "g4", values=[REVIEW["action"]() for _ in range(4)])
    asyncio.run(G.generate_g4(prefixes, native, tmp_path / "g4", freeze=freeze, identity=ident))
    assert [call[1].seed for call in engine.calls] == [11, 11, 23, 23, 47, 47, 71, 71]
    assert native.base._closed and native.trace_sink.stream.closed
    closed = G.read_closed_g4(tmp_path / "g4", freeze=freeze, prefixes=prefixes,
                             tokenizer=tokenizer, expected_source="f" * 40)
    assert len(closed["candidates"]) == len(closed["audits"]) == 4
    assert len(closed["phase_request_ids"]) == 8
    assert closed["generation_closed_before_scoring"] is True
    for envelope in closed["candidates"]:
        bundle = envelope["native_bundle"]
        trace = bundle["native_trace"]
        assert trace["completion_ids"] == trace["phases"][0]["token_ids"] + trace["phases"][1]["token_ids"]
        assert trace["phases"][0]["token_ids"][-1] == 151668
        assert trace["phases"][1]["token_ids"][-1] == 151645
        assert bundle["record"]["completion"] == trace["final_text"]
        assert not {"metadata", "grounding_fields", "positive", "gold", "reward"}.intersection(bundle["record"])


def test_parallel_prefixes_never_change_seed_between_native_phases(tmp_path, fake_tokenizer):
    from types import SimpleNamespace
    freeze, prefixes, ident = frozen_case(tmp_path, fake_tokenizer)
    prefixes.append(replace(prefixes[0], prefix_id="second-prefix", task_id="second-task"))
    freeze["prefix_ids"].append(prefixes[1].prefix_id)
    freeze["task_ids"].append(prefixes[1].task_id)
    tokenizer, engine, native = make_native(tmp_path / "g4", values=[])
    calls = []
    async def concurrent_generate(prompt, params, request_id, *, lora_request):
        calls.append(params.seed)
        await asyncio.sleep(0)
        assert native.base.seed == params.seed
        ids = (tokenizer.encode("actual thought</think>") if params.structured_outputs is None
               else tokenizer.encode(json.dumps(REVIEW["action"]())) + [151645])
        await asyncio.sleep(0)
        assert native.base.seed == params.seed
        yield SimpleNamespace(prompt_token_ids=prompt["prompt_token_ids"], finished=True,
            outputs=[SimpleNamespace(token_ids=ids, text="synthetic", finish_reason="stop", stop_reason=ids[-1])])
    engine.generate = concurrent_generate
    asyncio.run(G.generate_g4(prefixes, native, tmp_path / "g4", freeze=freeze, identity=ident, concurrency=2))
    assert calls == [seed for seed in freeze["seeds"] for _ in range(4)]
    closed = G.read_closed_g4(tmp_path / "g4", freeze=freeze, prefixes=prefixes, tokenizer=tokenizer)
    assert len(closed["candidates"]) == len(closed["audits"]) == 8


@pytest.mark.parametrize("damage", ["missingclosure", "seed", "trace", "audit", "freeze", "source"])
def test_reward_stage_rejects_unclosed_or_mismatched_g4_before_any_reward_access(tmp_path, fake_tokenizer, damage):
    freeze, prefixes, ident = frozen_case(tmp_path, fake_tokenizer)
    tokenizer, _, native = make_native(tmp_path / "g4", values=[REVIEW["action"]() for _ in range(4)])
    asyncio.run(G.generate_g4(prefixes, native, tmp_path / "g4", freeze=freeze, identity=ident))
    expected_source = "f" * 40
    if damage == "missingclosure":
        (tmp_path / "g4/generation_closed_rank0.json").unlink()
    elif damage == "freeze":
        freeze = deepcopy(freeze); freeze["harness_id"] = "different"
    elif damage == "source":
        expected_source = "e" * 40
    else:
        name = {"seed": "requests", "trace": "native_thinking_traces", "audit": "audit"}[damage]
        path = tmp_path / f"g4/{name}_rank0.jsonl"
        rows = list(G.read_jsonl(path))
        if damage == "seed":
            rows[0]["generation_seed"] = 999
        elif damage == "trace":
            rows[0]["phases"][0]["token_ids"][-1] = 151645
        else:
            rows[0]["candidate_valid"] = False
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises((ValueError, FileNotFoundError)):
        G.read_closed_g4(tmp_path / "g4", freeze=freeze, prefixes=prefixes,
                         tokenizer=tokenizer, expected_source=expected_source)


def test_failed_native_attempt_is_retained_once_and_never_fabricated_or_retried(tmp_path, fake_tokenizer):
    freeze, prefixes, ident = frozen_case(tmp_path, fake_tokenizer)
    from slime_plugins.dcc.native_budget import NativeBudget
    path = tmp_path / "g4"; path.mkdir()
    tokenizer, engine, native = FIXTURE["make_policy"](path, review=True,
        budget=NativeBudget(16384, "thinking"), scripts=[
            {"ids": fake_tokenizer.encode("unfinished thought"), "finish": "length", "stop": None},
            *[part for _ in range(3) for part in (
                {"ids": fake_tokenizer.encode("actual thought</think>")},
                {"ids": fake_tokenizer.encode(json.dumps(REVIEW["action"]())) + [151645]})]])
    native.base.allow_truncation = False
    asyncio.run(G.generate_g4(prefixes, native, path, freeze=freeze, identity=ident))
    closed = G.read_closed_g4(path, freeze=freeze, prefixes=prefixes, tokenizer=tokenizer)
    assert len(engine.calls) == 7  # Failed thought phase does not submit JSON or retry.
    assert len(closed["candidates"]) == 3 and len(closed["audits"]) == 4
    failed = [row for row in closed["audits"] if not row["candidate_valid"]]
    assert len(failed) == 1 and failed[0]["retried"] is False
    trace = native.trace_index[failed[0]["trace_id"]]
    assert trace["status"] != "completed" and len(trace["phases"]) == 1
    assert trace["phases"][0]["token_ids"] == fake_tokenizer.encode("unfinished thought")


def test_independent_candidates_cannot_reuse_actual_phase_request_id(tmp_path, fake_tokenizer):
    freeze, prefixes, ident = frozen_case(tmp_path, fake_tokenizer)
    tokenizer, _, native = make_native(tmp_path / "g4", values=[REVIEW["action"]() for _ in range(4)])
    asyncio.run(G.generate_g4(prefixes, native, tmp_path / "g4", freeze=freeze, identity=ident))
    trace_path = tmp_path / "g4/native_thinking_traces_rank0.jsonl"
    candidate_path = tmp_path / "g4/candidates_rank0.jsonl"
    traces, candidates = list(G.read_jsonl(trace_path)), list(G.read_jsonl(candidate_path))
    reused = traces[0]["phases"][0]["request_id"]
    traces[1]["phases"][0]["request_id"] = reused
    candidates[1]["native_bundle"]["native_trace"]["phases"][0]["request_id"] = reused
    for path, rows in [(trace_path, traces), (candidate_path, candidates)]:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="Repeated actual native phase"):
        G.read_closed_g4(tmp_path / "g4", freeze=freeze, prefixes=prefixes, tokenizer=tokenizer)


def test_shard_closure_requires_every_declared_rank_even_empty_shard(tmp_path, fake_tokenizer):
    freeze, prefixes, ident = frozen_case(tmp_path, fake_tokenizer)
    tokenizer, _, native = make_native(tmp_path / "g4", values=[REVIEW["action"]() for _ in range(4)])
    asyncio.run(G.generate_g4(prefixes, native, tmp_path / "g4", freeze=freeze, identity=ident,
                              rank=0, world_size=2))
    with pytest.raises(FileNotFoundError):
        G.read_closed_g4(tmp_path / "g4", freeze=freeze, prefixes=prefixes, tokenizer=tokenizer, world_size=2)


def test_real_local_tokenizer_cpu_identity_and_final_schema_only():
    declared = os.environ.get("DCC_LOCAL_NATIVE_TOKENIZER")
    path = Path(declared) if declared else None
    if path is None or not (path / "tokenizer.json").is_file():
        pytest.skip("Approved local tokenizer cache absent")
    from transformers import AutoTokenizer
    from slime_plugins.dcc.hf_policy import encode_messages
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    ident = identity(tokenizer)
    record, _, turn = REVIEW["recorded"](3)[-1]
    original, final, _ = G.public_request_schemas(web_task(), record["messages"], turn=turn,
        schemas=N.NativeSchemaProfile(REVIEW["profile"](), None))
    manifest = ident.manifest(tokenizer); manifest.pop("generation_seed")
    prefix = G.PublicFinalPrefix.from_dict({"contract": "native-final-public-prefix-v1",
        "prefix_id": "actual-tokenizer-cpu", "task_id": web_task().id, "benchmark": "widesearch",
        "source": "independent_curriculum", "split": "train", "harness_id": "cpu-only",
        "messages": record["messages"], "json_schema": final, "native_identity": manifest})
    ids = encode_messages(tokenizer, json.loads(prefix.messages_json), ident.max_prompt_tokens,
                          allow_truncation=False)
    assert ids[-4:] == tokenizer.encode(N.NATIVE_SUFFIX, add_special_tokens=False)[-4:]
    assert tokenizer.encode("<think>", add_special_tokens=False) == [151667]
    assert tokenizer.encode("</think>", add_special_tokens=False) == [151668]
    assert prefix.to_dict()["json_schema"] == final and original != final
