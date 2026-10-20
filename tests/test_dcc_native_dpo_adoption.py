"""Synthetic CPU DPO adapter adoption, actual LoRA save and native validators.

NF4 metadata below is explicitly synthetic. These tests do not load quantized
weights or claim a production NF4/GPU validation. Native fake-engine values and
tasks come only from the existing invented CPU protocol fixtures.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "scripts"),
                str(Path(__file__).resolve().parents[1] / "slime")]

import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from dcc_native_dpo import ReferenceIdentity
from dcc_native_dpo_train import QUANTIZATION_ID, merge_training_collections, run_dpo_loop
import dcc_native_dpo_train as TRAIN
from slime_plugins.dcc.native_dpo_adapter import (
    DPO_APPROVAL, DPO_METHOD, validate_dpo_approval, validate_dpo_sidecar_binding,
)
from slime_plugins.dcc.native_lora import validate_native_adapter
from slime_plugins.dcc.native_budget import NativeBudget
from slime_plugins.dcc.native_thinking_eval import NativeThinkingPolicy, native_options
from slime_plugins.dcc.eval_sampling import EvaluationSampling

_SPEC = importlib.util.spec_from_file_location("native_dpo_loop_fixture",
    Path(__file__).with_name("test_dcc_native_dpo_train.py"))
_LOOP = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_LOOP)


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def adopted(tmp_path):
    value = _LOOP.model()
    tokenizer, collection, _, _, _ = _LOOP._FIXTURE._physical_fixture()
    collection = replace(collection, derivative_id="synthetic-base", source_commit="d" * 40,
        thought_sampling_profile="thinking")
    receipt = {"version": "dcc-native-inference-source-v1", "source_commit": collection.source_commit,
        "base_source_commit": _LOOP._FIXTURE.SFT.BASE_SOURCE_COMMIT,
        "protocol": _LOOP._FIXTURE.SFT.PROTOCOL, "frozen": True}
    manifest = collection.manifest(tokenizer)
    manifest.pop("generation_seed")
    metadata = {"experiment_id": "synthetic-experiment", "harness_id": "synthetic-harness",
        "collection_source_commit": collection.source_commit, "collection_source_receipt": receipt,
        "collection_native_identity": manifest, "reward_version": "synthetic-TRAIN-reward-v1",
        "schema_policy": "final-decision-only-v1", "generation_world_size": 1,
        "generation_closed_before_scoring": True, "attempt_count": 4, "valid_candidates": 4,
        "eligible_pairs": 1, "omitted_prefixes": {}, "training_sources": ["cuad_train"],
        "pair_selection": "one-best-worst-per-frozen-prefix; reward-gap-strictly-greater-than-0.05",
        "training_source": {"version": "dcc-offline-native-dpo-training-source-v1", "source_commit": "e" * 40,
            "tracked_runtime_clean": True, "training_entrypoint": "dcc_native_dpo_train.py"},
        "quantization": {"mode": "nf4", "transformers_version": "4.57.6", "expert_4bit_coverage": 1.,
            "expert_parameters": 96, "expert_4bit_parameters": 96, "expert_4bit_matrices": 12}}
    metadata.update(frozen_prefix_ids=["synthetic-prefix-0"], frozen_task_ids=["synthetic-task-0"],
        attempt_trace_ids=[f"trace-{i}" for i in range(1, 5)],
        phase_request_ids=[f"phase-{i}" for i in range(1, 9)], selected_pairs=[{
            "prefix_id": "synthetic-prefix-0", "chosen_trace_id": "trace-1", "rejected_trace_id": "trace-2",
            "chosen_seed": 1, "rejected_seed": 2}])
    metadata["public_freeze"] = {"labels_loaded": False, "native_identity": manifest, "source_receipt": receipt,
        **{key: metadata[key] for key in ("experiment_id", "harness_id", "schema_policy")},
        "prefix_ids": metadata["frozen_prefix_ids"], "task_ids": metadata["frozen_task_ids"]}
    training_source, report = metadata.pop("training_source"), metadata.pop("quantization")
    _, merged = merge_training_collections([("synthetic-cuad", _LOOP.pairs(1), metadata)])
    merged.update(training_source=training_source, quantization=report)
    identity = ReferenceIdentity("synthetic-base", "synthetic-tokenizer", QUANTIZATION_ID, "base")
    output = tmp_path / "DPO"
    run_dpo_loop(value, _LOOP.SavedTokenizer(), _LOOP.pairs(1), identity,
                 output=output, steps=1, pairs_per_step=1, config=merged)
    adapter = output / "final"
    cfg = json.loads((adapter / "training_config.json").read_text())
    approval = {"version": DPO_APPROVAL, "training_configuration": cfg, "checkpoint_optimizer_steps": 1}
    evaluation_receipt = dict(receipt, source_commit="f" * 40)
    options = {"source_receipt": evaluation_receipt, "approved_derivative_id": "synthetic-base",
        "model_config": value.config.to_dict(), "evaluation_protocol": NativeBudget(8192, "thinking").protocol,
        "approved_dpo": approval}
    return adapter, approval, options


def test_actual_saved_tiny_dpo_adapter_requires_explicit_independent_approval(adopted):
    adapter, approval, options = adopted
    metadata, binding = validate_native_adapter(adapter, **options)
    assert metadata["config"]["r"] == 16 and binding["tensor_pairs"] == 3
    assert binding["training_method"] == DPO_METHOD
    assert binding["training_source"]["source_commit"] == "e" * 40
    assert binding["collection_source_receipts"][0]["source_receipt"]["source_commit"] == "d" * 40
    assert binding["checkpoint_optimizer_steps"] == 1
    assert binding["evaluation_source_receipt"]["source_commit"] == "f" * 40
    assert validate_dpo_sidecar_binding(binding, approval,
        derivative_id="synthetic-base", source_receipt=options["source_receipt"],
        evaluation_protocol=options["evaluation_protocol"]) == binding
    no_dpo = {key: value for key, value in options.items() if key != "approved_dpo"}
    with pytest.raises(ValueError, match="TRAIN configuration"):
        validate_native_adapter(adapter, **no_dpo)
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_native_adapter(adapter, **dict(options, approved_training_protocol=options["evaluation_protocol"]))
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("damage", ["reference", "reward", "pair", "source", "collection_source", "warm",
    "SFT_masquerade", "official_source", "quantization", "checkpoint", "shape", "extra_tensor", "sidecar_extra"])
def test_dpo_adoption_rejects_metadata_or_original_header_tampering(adopted, damage):
    adapter, approval, options = adopted
    cfg = approval["training_configuration"]
    if damage == "reference": cfg["reference"]["mode"] = "initial_adapter"
    elif damage == "reward": cfg["collections"][0]["reward_version"] = "unapproved-reward"
    elif damage == "pair": cfg["minimum_reward_gap"] = 0.
    elif damage == "source": cfg["training_source"]["tracked_runtime_clean"] = False
    elif damage == "collection_source": cfg["collections"][0]["collection_source_commit"] = "a" * 40
    elif damage == "warm": cfg["warm_start"] = True
    elif damage == "SFT_masquerade": cfg["native_thinking"] = {"validated_records": 1}
    elif damage == "official_source": cfg["collections"][0]["training_sources"] = ["official_widesearch"]
    elif damage == "quantization": cfg["quantization"]["expert_4bit_coverage"] = .5
    elif damage == "checkpoint": dump(adapter / "checkpoint_complete.json", {"complete": False, "optimizer_steps": 1})
    elif damage in {"shape", "extra_tensor"}:
        from safetensors.torch import load_file, save_file
        tensors = load_file(str(adapter / "adapter_model.safetensors"))
        if damage == "shape":
            name = next(iter(tensors))
            tensors[name] = tensors[name][:-1]
        else:
            tensors["unapproved.weight"] = torch.zeros(1)
        save_file(tensors, str(adapter / "adapter_model.safetensors"))
    elif damage == "sidecar_extra":
        _, binding = validate_native_adapter(adapter, **options)
        binding["unapproved"] = True
        with pytest.raises(ValueError, match="binding differs"):
            validate_dpo_sidecar_binding(binding, approval, derivative_id="synthetic-base",
                source_receipt=options["source_receipt"], evaluation_protocol=options["evaluation_protocol"])
        return
    with pytest.raises(ValueError):
        validate_native_adapter(adapter, **options)


def test_actual_native_wrapper_and_preflight_accept_only_explicit_dpo(adopted, tmp_path, monkeypatch):
    native = runpy.run_path(str(Path(__file__).with_name("test_dcc_native_thinking.py")))
    adapter, approval, options = adopted
    fixture_dir = tmp_path / "native-wrapper"; fixture_dir.mkdir()
    _, engine, old = native["make_policy"](fixture_dir, values=[{"plan": "Read", "checks": "None", "answer": []}],
        budget=NativeBudget(8192, "thinking"))
    metadata, _ = validate_native_adapter(adapter, **options)
    base = old.base
    base.lora_request = SimpleNamespace(lora_path=str(adapter))
    base.provenance["adapter"] = metadata
    engine.vllm_config.model_config = SimpleNamespace(hf_config=SimpleNamespace(to_dict=lambda: options["model_config"]))
    adopted_wrapper = NativeThinkingPolicy(base, source_receipt=options["source_receipt"], trace_sink=old.trace_sink,
        budget=NativeBudget(8192, "thinking"), approved_derivative_id="synthetic-base", approved_dpo=approval)
    assert adopted_wrapper.provenance["native_lora_binding"]["training_method"] == DPO_METHOD
    assert not engine.calls
    with pytest.raises(ValueError, match="explicit approval"):
        NativeThinkingPolicy(base, source_receipt=options["source_receipt"], trace_sink=old.trace_sink,
            budget=NativeBudget(8192, "thinking"), approved_derivative_id="synthetic-base")
    approval_path = tmp_path / "dpo-approval.json"; dump(approval_path, approval)
    source_path = tmp_path / ".dcc-native-source.json"; dump(source_path, options["source_receipt"])
    from transformers import AutoConfig
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(to_dict=lambda: options["model_config"]))
    args = SimpleNamespace(native_thinking=True, native_thought_tokens=8192, native_thought_sampling="thinking",
        native_source_receipt=str(source_path), native_lora_derivative_id="synthetic-base",
        native_lora_dpo_approval=str(approval_path), adapter=str(adapter), inference_backend="vllm",
        vllm_async=True, web_structured_final=True, web_structured_actions=True,
        max_new_tokens=12288, model="synthetic-base")
    assert native_options(args, [native["cuad_task"]()], EvaluationSampling(), tmp_path)["protocol"] == options["evaluation_protocol"]
    args.native_thinking = False
    with pytest.raises(ValueError):
        native_options(args, [native["cuad_task"]()], EvaluationSampling(), tmp_path)
    old.trace_sink.close()


def test_actual_native_sidecars_and_paired_comparison_distinguish_dpo(adopted, tmp_path, monkeypatch):
    fixture = runpy.run_path(str(Path(__file__).with_name("test_dcc_native_compare.py")))
    compare = fixture["COMPARE"]
    pair_dir = tmp_path / "pair"; pair_dir.mkdir()
    task, (base, trained, bp, tp) = fixture["native_pair"](pair_dir, monkeypatch)
    adapter, approval, options = adopted
    metadata, binding = validate_native_adapter(adapter, **options)
    dump(Path(trained["manifest"]["model"]) / "config.json", options["model_config"])
    trained["manifest"].update(adapter=str(adapter), adapter_identity=metadata,
        native_lora_derivative_id="synthetic-base", native_lora_dpo_approval=approval)
    trained["manifest"].pop("native_lora_training_approval")
    trained["runtime"].update(adapter=metadata, native_lora_binding=binding)
    tp["policy"]["adapter"] = str(adapter)
    trace_path = trained["output"] / "native_thinking_traces_rank0.jsonl"
    traces = [json.loads(line) for line in trace_path.read_text().splitlines()]
    for trace in traces:
        trace["native_lora_binding"] = binding
    trace_path.write_text("".join(json.dumps(trace) + "\n" for trace in traces))
    report_path = trained["output"] / "scores_strict.json"
    report = json.loads(report_path.read_text())
    report.update(generation_manifest=trained["manifest"], inference_runtime=trained["runtime"])
    dump(report_path, report)
    report = compare["compare"]([task], base, trained, bp, tp, resamples=100)
    record = report["native_policy_bindings"]["trained"]
    assert record["native_lora_training_approval"] is None
    assert record["native_lora_dpo_approval"] == approval
    assert record["native_lora_binding"]["training_method"] == DPO_METHOD
    trained["manifest"]["native_lora_dpo_approval"]["training_configuration"]["collections"][0]["reward_version"] = "unapproved"
    with pytest.raises(ValueError):
        compare["validate_protocols"](base, trained, bp, tp)


def test_real_step8_checkpoint_keeps_configured_total16_and_requires_exact_step(adopted, tmp_path):
    _, approval, options = adopted
    value = _LOOP.model()
    cfg = approval["training_configuration"]
    identity = ReferenceIdentity(**cfg["reference"])
    output = tmp_path / "sixteen-updates"
    run_dpo_loop(value, _LOOP.SavedTokenizer(), _LOOP.pairs(1), identity,
        output=output, steps=16, pairs_per_step=1, save_every=8, config=cfg)
    step8, final = output / "step_0008", output / "final"
    actual = json.loads((step8 / "training_config.json").read_text())
    assert actual["steps"] == 16
    approved8 = {"version": DPO_APPROVAL, "training_configuration": actual, "checkpoint_optimizer_steps": 8}
    _, binding = validate_native_adapter(step8, **dict(options, approved_dpo=approved8))
    assert binding["checkpoint_optimizer_steps"] == 8
    assert binding["caller_approved_dpo"]["training_configuration"]["steps"] == 16
    with pytest.raises(ValueError, match="approved actual optimizer step"):
        validate_native_adapter(final, **dict(options, approved_dpo=approved8))
    wrong = dict(approved8, checkpoint_optimizer_steps=16)
    with pytest.raises(ValueError, match="approved actual optimizer step"):
        validate_native_adapter(step8, **dict(options, approved_dpo=wrong))


def _two_collection_fixture(approval):
    cuad = deepcopy(approval["training_configuration"]["collections"][0])
    cuad["collection_id"] = "cuad-16k"
    cuad["collection_native_identity"].update(max_prompt_tokens=16384, max_model_len=24576)
    cuad["public_freeze"]["native_identity"] = deepcopy(cuad["collection_native_identity"])
    web = deepcopy(cuad)
    web.update(collection_id="web-32k", experiment_id="web-experiment", harness_id="web-harness",
        reward_version="synthetic-WEB-reward-v1", training_sources=["independent_curriculum"],
        frozen_prefix_ids=["web-prefix-0"], frozen_task_ids=["web-task-0"],
        attempt_trace_ids=[f"web-trace-{i}" for i in range(1, 5)],
        phase_request_ids=[f"web-phase-{i}" for i in range(1, 9)], selected_pairs=[{
            "prefix_id": "web-prefix-0", "chosen_trace_id": "web-trace-1", "rejected_trace_id": "web-trace-2",
            "chosen_seed": 1, "rejected_seed": 2}])
    web["collection_native_identity"].update(max_prompt_tokens=32768, max_model_len=40960)
    web["public_freeze"].update(native_identity=deepcopy(web["collection_native_identity"]),
        prefix_ids=web["frozen_prefix_ids"], task_ids=web["frozen_task_ids"],
        **{key: web[key] for key in ("experiment_id", "harness_id")})
    original = _LOOP.pairs(1)[0]
    prefix = replace(original.chosen.prefix, prefix_id="web-prefix-0", task_id="web-task-0",
        benchmark="widesearch", source="independent_curriculum")
    web_pair = type(original)(replace(original.chosen, prefix=prefix, trace_id="web-trace-1"),
                              replace(original.rejected, prefix=prefix, trace_id="web-trace-2"))
    return [("cuad-16k", [original], cuad), ("web-32k", [web_pair], web)]


def test_explicit_domain_collections_keep_separate_contexts_and_provenance(adopted):
    _, approval, _ = adopted
    collections = _two_collection_fixture(approval)
    rows, metadata = merge_training_collections(collections)
    assert len(rows) == 2
    assert [p["collection_native_identity"]["max_prompt_tokens"] for p in metadata["collections"]] == [16384, 32768]
    assert [p["reward_version"] for p in metadata["collections"]] == ["synthetic-TRAIN-reward-v1", "synthetic-WEB-reward-v1"]
    merged_approval = deepcopy(approval)
    merged_approval["training_configuration"].update(metadata)
    validate_dpo_approval(merged_approval)
    bad = deepcopy(collections)
    bad[1][2]["frozen_task_ids"] = bad[0][2]["frozen_task_ids"]
    with pytest.raises(ValueError, match="Reused public"):
        merge_training_collections(bad)
    bad = deepcopy(collections)
    bad[1][2]["collection_native_identity"]["derivative_id"] = "different-base"
    with pytest.raises(ValueError, match="base/tokenizer"):
        merge_training_collections(bad)


def test_all_collection_closures_precede_any_reward_access(adopted, tmp_path, monkeypatch):
    _, approval, _ = adopted
    collections = _two_collection_fixture(approval)
    manifest = {"version": TRAIN.COLLECTIONS_CONTRACT, "base_derivative_id": "synthetic-base",
        "tokenizer_id": "synthetic-tokenizer", "collections": [{"collection_id": cid,
            "approved_freeze": cid, "generation_dir": "synthetic-generation", "generation_world_size": 8,
            "train_rewards": "synthetic-reward-file", "expected_source": "d" * 40,
            "reward_version": metadata["reward_version"]} for cid, _, metadata in collections]}
    path = tmp_path / "approved-collections.json"; dump(path, manifest)
    args = SimpleNamespace(approved_collections=str(path), approved_freeze=None, generation_dir=None,
        train_rewards=None, expected_source=None, reward_version=None,
        base_derivative_id="synthetic-base", tokenizer_id="synthetic-tokenizer")
    events = []
    indexed = {cid: (rows, metadata) for cid, rows, metadata in collections}
    prepared = {cid: (metadata["public_freeze"], [], None, {
        "attempts": [{"trace_id": trace} for trace in metadata["attempt_trace_ids"]],
        "phase_request_ids": metadata["phase_request_ids"]}) for cid, _, metadata in collections}
    def close(local, tokenizer):
        events.append("closure:" + local.approved_freeze)
        return prepared[local.approved_freeze]
    def reward(local, tokenizer, closed):
        assert closed == prepared[local.approved_freeze]
        events.append("reward:" + local.approved_freeze)
        return indexed[local.approved_freeze]
    monkeypatch.setattr(TRAIN, "read_approved_collection", close)
    monkeypatch.setattr(TRAIN, "pairs_from_closed_collection", reward)
    rows, metadata = TRAIN.load_training_pairs(args, object())
    assert events == ["closure:cuad-16k", "closure:web-32k", "reward:cuad-16k", "reward:web-32k"]
    assert len(rows) == 2 and str(tmp_path) not in json.dumps(metadata)
    events.clear()
    def fail_second(local, tokenizer):
        events.append("closure:" + local.approved_freeze)
        if local.approved_freeze == "web-32k":
            raise ValueError("second collection missing rank closure")
        return prepared[local.approved_freeze]
    monkeypatch.setattr(TRAIN, "read_approved_collection", fail_second)
    with pytest.raises(ValueError, match="second collection missing"):
        TRAIN.load_training_pairs(args, object())
    assert events == ["closure:cuad-16k", "closure:web-32k"]


@pytest.mark.parametrize("damage", ["prefix", "task", "attempt", "phase", "base", "collection"])
def test_all_collection_ownership_precedes_any_reward_access(adopted, tmp_path, monkeypatch, damage):
    _, approval, _ = adopted
    collections = _two_collection_fixture(approval)
    manifest = {"version": TRAIN.COLLECTIONS_CONTRACT, "base_derivative_id": "synthetic-base",
        "tokenizer_id": "synthetic-tokenizer", "collections": [{"collection_id": cid,
            "approved_freeze": cid, "generation_dir": "synthetic-generation", "generation_world_size": 8,
            "train_rewards": "synthetic-reward-file", "expected_source": "d" * 40,
            "reward_version": metadata["reward_version"]} for cid, _, metadata in collections]}
    prepared = {cid: (deepcopy(metadata["public_freeze"]), [], None, {
        "attempts": [{"trace_id": trace} for trace in metadata["attempt_trace_ids"]],
        "phase_request_ids": list(metadata["phase_request_ids"])}) for cid, _, metadata in collections}
    cuad, web = prepared["cuad-16k"], prepared["web-32k"]
    if damage == "prefix": web[0]["prefix_ids"] = cuad[0]["prefix_ids"]
    elif damage == "task": web[0]["task_ids"] = cuad[0]["task_ids"]
    elif damage == "attempt": web[3]["attempts"][0] = cuad[3]["attempts"][0]
    elif damage == "phase": web[3]["phase_request_ids"][0] = cuad[3]["phase_request_ids"][0]
    elif damage == "base": web[0]["native_identity"]["derivative_id"] = "different-base"
    elif damage == "collection": manifest["collections"][1]["collection_id"] = "cuad-16k"
    path = tmp_path / "approved-collections.json"; dump(path, manifest)
    args = SimpleNamespace(approved_collections=str(path), approved_freeze=None, generation_dir=None,
        train_rewards=None, expected_source=None, reward_version=None,
        base_derivative_id="synthetic-base", tokenizer_id="synthetic-tokenizer")
    events = []
    def close(local, tokenizer):
        events.append("closure:" + local.approved_freeze)
        return prepared[local.approved_freeze]
    def forbidden_reward(*args, **kwargs):
        events.append("REWARD_OPENED")
        raise AssertionError("Any reward access before all collection ownership is validated")
    monkeypatch.setattr(TRAIN, "read_approved_collection", close)
    monkeypatch.setattr(TRAIN, "pairs_from_closed_collection", forbidden_reward)
    monkeypatch.setattr(TRAIN, "read_train_rewards", forbidden_reward)
    with pytest.raises(ValueError, match="Reused public|base/tokenizer|Duplicate approved"):
        TRAIN.load_training_pairs(args, object())
    assert events == ["closure:cuad-16k", "closure:web-32k"]


def test_malformed_selected_pair_is_rejected_as_validation_error(adopted):
    _, approval, _ = adopted
    approval["training_configuration"]["collections"][0]["selected_pairs"] = [None]
    with pytest.raises(ValueError, match="selected pair ownership"):
        validate_dpo_approval(approval)
