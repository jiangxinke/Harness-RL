"""Synthetic CPU coverage for native paired-policy audit exemptions.

Uses the existing native fake-engine fixture and the production sidecar/adapter
validators. No real tokenizer, pretrained weights, inference engine or GPU loads.
"""
from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest

COMPARE = runpy.run_path(str(Path(__file__).parents[1] / "scripts/dcc_compare_runs.py"))
NATIVE = runpy.run_path(str(Path(__file__).with_name("test_dcc_native_thinking.py")))
LEGACY = runpy.run_path(str(Path(__file__).with_name("test_dcc_compare_runs.py")))


def dump(path, value):
    path.write_text(json.dumps(value))


def native_pair(tmp_path, monkeypatch, *, failed=False, training_budget=None):
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    from slime_plugins.dcc.eval_sampling import EvaluationSampling
    from slime_plugins.dcc.native_budget import NativeBudget, approved_training_receipt
    from slime_plugins.dcc.native_lora import validate_native_adapter
    from slime_plugins.dcc.native_thinking_eval import MODEL_REVISION, PROTOCOL
    from slime_plugins.dcc.web_review import web_explicit_review_configuration
    from slime_plugins.dcc.web_structured import web_structured_actions_configuration, web_structured_configuration
    from slime_plugins.dcc.evaluation import evaluate_results

    model, adapter = tmp_path / "model", tmp_path / "adapter"
    model.mkdir(); adapter.mkdir()
    model_config = {"hidden_size": 4, "num_hidden_layers": 1, "num_attention_heads": 2,
                    "num_key_value_heads": 1, "head_dim": 2}
    dump(model / "config.json", model_config)
    approval = {"source_commit": "e" * 40, "protocol": PROTOCOL if training_budget is None else training_budget.protocol}
    receipt = approved_training_receipt(NATIVE["RECEIPT"], **approval)
    derivative = "synthetic/native-attention-r2"
    dump(adapter / "adapter_config.json", {"peft_type": "LORA", "r": 2, "target_modules": ["q_proj"]})
    save_file({"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(2, 4),
               "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(4, 2)},
              str(adapter / "adapter_model.safetensors"))
    train_config = {"steps": 1, "lora_rank": 2, "native_thinking": {
        "enabled": True, "protocol": approval["protocol"], "source_receipt": receipt, "derivative_id": derivative,
        "model_revision": MODEL_REVISION, "all_files_validated_before_model_load": True,
        "warm_start": False, "raw_cuad_auxiliary": False, "validated_records": 1,
        "validated_cuad_records": 0}}
    if training_budget is not None and not training_budget.legacy:
        from slime_plugins.dcc.native_thinking_eval import native_configuration
        train_config["native_thinking"]["rollout_native_thinking"] = native_configuration(receipt, budget=training_budget)
    dump(adapter / "training_config.json", train_config)
    budget = NativeBudget(8192, "thinking")
    metadata, binding = validate_native_adapter(str(adapter), source_receipt=NATIVE["RECEIPT"],
        approved_derivative_id=derivative, model_config=model_config,
        approved_training_source_receipt=receipt, approved_training_protocol=approval["protocol"],
        evaluation_protocol=budget.protocol)
    tokenizer = NATIVE["FakeTokenizer"]()
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    task = NATIVE["cuad_task"](); task.metadata["contract_id"] = "synthetic-contract"
    runs, protocols = [], []
    for name, adapted in (("base", False), ("trained", True)):
        output = tmp_path / name; output.mkdir()
        kwargs = {"scripts": [{"ids": tokenizer.encode("unfinished"), "finish": "length", "stop": None}]} if failed else {
            "values": [{"plan": "Read the public contract", "checks": "No matching span", "answer": []}]}
        _, _, _, _, row, manifest, runtime, _ = NATIVE["run_cuad"](output, budget=budget, **kwargs)
        manifest.update(model=str(model), adapter=str(adapter) if adapted else None,
            adapter_identity=metadata if adapted else None, data=["synthetic-tasks.jsonl"], eval_num_shards=1,
            initial_chunks=4, limit=None, inference_backend="vllm", quantization="none", timeout_seconds=1800)
        manifest["inference_parameters"]["batch_size"] = 2
        runtime.update(model=str(model), backend="vllm", adapter=metadata if adapted else None,
            evaluation_sampling=EvaluationSampling().to_dict(), async_engine=True)
        declaration = LEGACY["protocol"](str(adapter) if adapted else None)
        generation = declaration["common"]["generation"]
        generation.update(max_prompt_tokens=16384, max_new_tokens=budget.total_tokens,
            sampling=EvaluationSampling().to_dict(), native_thinking=deepcopy(manifest["native_thinking"]))
        for record in (manifest, runtime, generation):
            record["web_explicit_review"] = web_explicit_review_configuration(False)
            record["web_structured_final"] = web_structured_configuration(True)
            record["web_structured_actions"] = web_structured_actions_configuration(True)
        if adapted:
            manifest.update(native_lora_derivative_id=derivative, native_lora_training_approval=approval)
            runtime["native_lora_binding"] = binding
            trace_path = output / "native_thinking_traces_rank0.jsonl"
            traces = [json.loads(line) for line in trace_path.read_text().splitlines()]
            for trace in traces: trace["native_lora_binding"] = binding
            trace_path.write_text("".join(json.dumps(trace) + "\n" for trace in traces))
        dump(output / "manifest_rank0.json", manifest)
        dump(output / "inference_runtime.json", runtime)
        dump(output / "rank0.done", {"tasks": 1, "world_size": 1})
        (output / "predictions_rank0.jsonl").write_text(json.dumps(row) + "\n")
        report = evaluate_results([task], [row])
        report.update(generation_manifest=manifest, inference_runtime=runtime,
                      policy_quantization="none", policy_inference_backend="vllm")
        dump(output / "scores_strict.json", report)
        runs.append(COMPARE["load_run"](output, [task])); protocols.append(declaration)
    return task, (*runs, *protocols)


def test_budgeted_native_training_is_bound_independently_of_evaluation(tmp_path, monkeypatch):
    from slime_plugins.dcc.native_budget import NativeBudget, BUDGET_PROTOCOL
    task, values = native_pair(tmp_path, monkeypatch, training_budget=NativeBudget(16384, "thinking"))
    report = COMPARE["compare"]([task], *values, resamples=100)
    binding = report["native_policy_bindings"]["trained"]
    assert binding["native_lora_binding"]["protocol"] == BUDGET_PROTOCOL
    assert binding["native_lora_training_approval"]["protocol"] == BUDGET_PROTOCOL
    assert binding["native_lora_binding"]["training_rollout_native_thinking"]["thought_max_tokens"] == 16384
    path = Path(values[1]["manifest"]["adapter"]) / "training_config.json"
    config = json.loads(path.read_text())
    config["native_thinking"]["rollout_native_thinking"]["thought_sampling"]["temperature"] = 0
    dump(path, config)
    with pytest.raises(ValueError, match="rollout differs"):
        COMPARE["validate_protocols"](*values)


def test_otherwise_valid_training_budget_change_cannot_reuse_saved_binding(tmp_path, monkeypatch):
    from slime_plugins.dcc.native_budget import NativeBudget
    from slime_plugins.dcc.native_thinking_eval import native_configuration
    _, values = native_pair(tmp_path, monkeypatch, training_budget=NativeBudget(16384, "thinking"))
    path = Path(values[1]["manifest"]["adapter"]) / "training_config.json"
    config = json.loads(path.read_text())
    native = config["native_thinking"]
    native["rollout_native_thinking"] = native_configuration(native["source_receipt"], budget=NativeBudget(8192, "greedy"))
    dump(path, config)
    with pytest.raises(ValueError, match="metadata/binding"):
        COMPARE["validate_protocols"](*values)


@pytest.mark.parametrize("failed", [False, True])
def test_native_v3_pairs_distinct_actual_policies_and_retains_failure_denominator(tmp_path, monkeypatch, failed):
    task, values = native_pair(tmp_path, monkeypatch, failed=failed)
    original = deepcopy([value["manifest"] for value in values[:2]])
    report = COMPARE["compare"]([task], *values, resamples=100)
    assert report["tasks"] == report["benchmarks"]["cuad"]["tasks"] == 1
    assert not report["eligible_for_improvement_claim"]
    assert [value["manifest"] for value in values[:2]] == original
    binding = report["native_policy_bindings"]
    assert binding["baseline"]["adapter_identity"] is None
    assert binding["trained"]["native_lora_training_approval"]["source_commit"] == "e" * 40
    assert binding["trained"]["native_lora_binding"]["evaluation_source_receipt"]["source_commit"] == "f" * 40
    if failed:
        assert report["runs"]["trained"]["status_counts"] == {"model_protocol_failure": 1}
        assert report["benchmarks"]["cuad"]["metrics"]["micro_f1"]["trained"] == 0


@pytest.mark.parametrize("damage", [
    "undeclared_native", "mixed_native", "mixed_protocol", "different_source", "different_sampler",
    "different_budget", "different_common", "runtime_source", "base_binding", "base_identity",
    "base_runtime_adapter", "derivative", "training_approval", "binding_source", "binding_rank",
    "binding_extra", "actual_metadata", "actual_train_source", "actual_header", "strict_report",
    "strict_status", "strict_coverage", "missing_trace", "tampered_trace", "unknown_runtime",
])
def test_native_pairing_rejects_forged_bindings_and_harness_mismatches(tmp_path, monkeypatch, damage):
    task, values = native_pair(tmp_path, monkeypatch)
    base, trained, bp, tp = values
    m, r, g = trained["manifest"], trained["runtime"], tp["common"]["generation"]
    if damage == "undeclared_native": g.pop("native_thinking")
    elif damage == "mixed_native":
        for rec in (base["manifest"], base["runtime"], bp["common"]["generation"]): rec.pop("native_thinking")
    elif damage == "mixed_protocol": m["native_thinking"]["protocol"] = NATIVE["PROTOCOL"]
    elif damage == "different_source":
        for rec in (m, r, g): rec["native_thinking"]["source_receipt"]["source_commit"] = "d" * 40
    elif damage == "different_sampler":
        for rec in (m, r, g): rec["native_thinking"]["thought_sampling"]["temperature"] = 0.5
    elif damage == "different_budget":
        for rec in (m, r, g): rec["native_thinking"]["thought_max_tokens"] = 16384
    elif damage == "different_common": g["native_thinking"] = deepcopy(g["native_thinking"]); g["native_thinking"]["retry"] = True
    elif damage == "runtime_source": r["source_commit"] = "a" * 40
    elif damage == "base_binding": base["runtime"]["native_lora_binding"] = deepcopy(r["native_lora_binding"])
    elif damage == "base_identity": base["manifest"]["adapter_identity"] = deepcopy(m["adapter_identity"])
    elif damage == "base_runtime_adapter": base["runtime"]["adapter"] = deepcopy(r["adapter"])
    elif damage == "derivative": m["native_lora_derivative_id"] = "forged"
    elif damage == "training_approval": m["native_lora_training_approval"]["source_commit"] = "d" * 40
    elif damage == "binding_source": r["native_lora_binding"]["training_source_receipt"]["source_commit"] = "d" * 40
    elif damage == "binding_rank": r["native_lora_binding"]["lora_rank"] = 3
    elif damage == "binding_extra": r["native_lora_binding"]["unverified"] = True
    elif damage == "actual_metadata":
        m["adapter_identity"]["weights_bytes"] += 1; r["adapter"]["weights_bytes"] += 1
    elif damage == "actual_train_source":
        p = Path(m["adapter"]) / "training_config.json"; config = json.loads(p.read_text())
        config["native_thinking"]["source_receipt"]["source_commit"] = "d" * 40; dump(p, config)
    elif damage == "actual_header":
        import torch
        from safetensors.torch import save_file
        save_file({"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(2, 3),
                   "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(4, 2)},
                  str(Path(m["adapter"]) / "adapter_model.safetensors"))
    elif damage.startswith("strict_"):
        p = trained["output"] / "scores_strict.json"; report = json.loads(p.read_text())
        if damage == "strict_report": report["generation_manifest"]["seed"] += 1
        elif damage == "strict_status": report["status_counts"] = {"evidence_failure": 1}
        else: report["prediction_coverage"] = 0
        dump(p, report)
    elif damage == "missing_trace": (trained["output"] / "native_thinking_traces_rank0.jsonl").unlink()
    elif damage == "tampered_trace":
        p = trained["output"] / "native_thinking_traces_rank0.jsonl"; trace = json.loads(p.read_text())
        trace["phases"][0]["sampling"]["temperature"] = 0; p.write_text(json.dumps(trace) + "\n")
    elif damage == "unknown_runtime": r["unverified_runtime_change"] = True
    if not damage.startswith("strict_"):
        # Keep scorer provenance matched so metadata/header/trace guards, rather
        # than an earlier stale-report check, must reject these corruptions.
        for run in values[:2]:
            path = run["output"] / "scores_strict.json"
            saved = json.loads(path.read_text())
            saved.update(generation_manifest=run["manifest"], inference_runtime=run["runtime"])
            dump(path, saved)
    with pytest.raises((ValueError, FileNotFoundError)):
        COMPARE["validate_protocols"](*values)


def test_non_native_policy_cannot_use_native_exemptions(tmp_path):
    values = LEGACY["pair"](tmp_path, [LEGACY["contract"]("a", "A", [])])
    values[1]["manifest"]["native_lora_training_approval"] = {"source_commit": "e" * 40}
    with pytest.raises(ValueError, match="Native policy bindings"):
        COMPARE["validate_protocols"](*values)


@pytest.mark.parametrize("difference", ["source", "sampler", "budget"])
def test_individually_valid_native_v3_runs_still_require_identical_harness(tmp_path, monkeypatch, difference):
    from slime_plugins.dcc.eval_sampling import EvaluationSampling
    from slime_plugins.dcc.native_budget import NativeBudget, effective_sampling
    from slime_plugins.dcc.native_thinking_eval import native_configuration
    task, values = native_pair(tmp_path, monkeypatch)
    run, declaration = values[1], values[3]
    manifest, runtime = run["manifest"], run["runtime"]
    generation = declaration["common"]["generation"]
    budget = NativeBudget(16384 if difference == "budget" else 8192,
                          "greedy" if difference == "sampler" else "thinking")
    receipt = deepcopy(NATIVE["RECEIPT"])
    if difference == "source": receipt["source_commit"] = "d" * 40
    config = native_configuration(receipt, budget=budget)
    for record in (manifest, runtime, generation): record["native_thinking"] = deepcopy(config)
    for record in (manifest, generation): record["max_new_tokens"] = budget.total_tokens
    context = manifest["max_prompt_tokens"] + budget.total_tokens
    manifest["inference_parameters"]["max_model_len"] = runtime["max_model_len"] = context
    runtime["source_commit"] = receipt["source_commit"]
    runtime["native_lora_binding"]["evaluation_source_receipt"] = deepcopy(receipt)
    path = run["output"] / "native_thinking_traces_rank0.jsonl"
    traces = [json.loads(line) for line in path.read_text().splitlines()]
    for trace in traces:
        trace.update(native_thinking=deepcopy(config), source_commit=receipt["source_commit"],
                     native_lora_binding=deepcopy(runtime["native_lora_binding"]),
                     thought_max_tokens=budget.thought_tokens, declared_max_new_tokens=budget.total_tokens)
        trace["phases"][0].update(max_tokens=budget.thought_tokens,
            sampling=budget.thought_sampling.to_dict(), effective_sampling=effective_sampling(budget.thought_sampling))
    path.write_text("".join(json.dumps(trace) + "\n" for trace in traces))
    path = run["output"] / "scores_strict.json"; report = json.loads(path.read_text())
    report.update(generation_manifest=manifest, inference_runtime=runtime); dump(path, report)
    assert COMPARE["_validate_native_pair_run"](run, generation, "lora") is True
    with pytest.raises(ValueError, match="Paired protocols differ"):
        COMPARE["validate_protocols"](*values)
