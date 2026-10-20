#!/usr/bin/env python3
"""Compare two fixed DCC runs with a paired cluster bootstrap (CPU only).

Reads existing manifest_rank*.json, predictions_rank*.jsonl, rank*.done and,
for WideSearch, scores_strict.json or scores_local_judge.json. Same-shard
retries follow dcc_model_evaluate.py: the last outcome wins and is reported.

Both protocol JSON files must have schema=dcc_paired_run_protocol_v1,
identical nonempty common entries base_model/harness/generation/retrieval/
scoring, and policy={kind: base|lora, adapter: null|saved_adapter_path}.
common.generation records backend, quantization, batch_size, temperature,
timeout_seconds, max_steps, max_new_tokens, max_prompt_tokens, initial_chunks
and seed. For WideSearch, common.scoring.widesearch_protocol names the saved
evaluation_protocol and table_input_protocol binds CSV ingestion semantics.
common.retrieval includes retrieval_context and, for
WideSearch, search_backend/search_endpoint/web_timeout_seconds/auto_read_sources.
A saved retrieval_profile must also be declared. Saved timeouts are checked;
missing timeouts in legacy manifests remain operator declarations. Protocols
attest provenance, not historical executable identity. Explicit evaluation
sampling must be fully declared and match the saved manifest and runtime;
legacy runs without such metadata require greedy temperature=0.

CUAD resamples entire contracts and recomputes official AUPR/micro F1.
WideSearch resamples tasks using the same frozen scorer's per-task results.
All non-completed episodes remain in the denominator with empty/zero outputs.
Judge failures are rejected; infrastructure episodes make a comparison
ineligible for an improvement claim. No checkpoint selection is performed.
NumPy is used only for CPU resampling. Explicit native comparisons additionally
import the original CPU validators and inspect local tokenizer/config/LoRA headers;
no model weights or inference engine are loaded.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "slime"))

from slime_plugins.dcc.data import load_tasks
from slime_plugins.dcc.browser_reader import validate_browser_reader_runtime
from slime_plugins.dcc.eval_sampling import EvaluationSampling
from slime_plugins.dcc.metrics import cuad_counts, cuad_predictions
from slime_plugins.dcc.web_whitespace import (
    validate_web_whitespace_records, web_bounded_whitespace_configuration,
)
from slime_plugins.dcc.web_review import web_explicit_review_configuration
from slime_plugins.dcc.web_navigation import web_navigation_configuration, validate_web_navigation_records
from slime_plugins.dcc.web_rows import validate_web_cached_rows_records
from slime_plugins.dcc.web_research import validate_web_research_records
from slime_plugins.dcc.web_structured import web_structured_actions_configuration, web_structured_configuration

STATUSES = {"completed", "invalid_answer", "max_steps", "evidence_failure", "infrastructure_error"}
WIDE_METRICS = ("score", "precision_by_row", "recall_by_row", "f1_by_row",
                "precision_by_item", "recall_by_item", "f1_by_item")
THRESHOLDS = tuple(value / 100 for value in range(99, 0, -1)) + (0.001, 0.0)
PRIMARY_METRICS = {"cuad": "aupr", "widesearch": "f1_by_row"}


def _invalid_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def _json(text):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(text, parse_constant=_invalid_constant, object_pairs_hook=unique_object)


def _read(path):
    value = _json(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {Path(path).name}")
    return value


def _rank(path, prefix):
    suffix = path.stem.removeprefix(prefix)
    if not suffix.isdecimal():
        raise ValueError(f"Invalid shard filename: {path.name}")
    return int(suffix)


def load_run(output, tasks):
    """Validate persisted shard ownership/completion without loading a model."""
    output = Path(output)
    manifests = {_rank(path, "manifest_rank"): _read(path)
                 for path in output.glob("manifest_rank*.json")}
    if not manifests:
        raise ValueError("Missing generation manifests")
    manifest = next(iter(manifests.values()))
    world = manifest.get("world_size")
    if type(world) is not int or world < 1 or set(manifests) != set(range(world)):
        raise ValueError("Generation manifests do not cover the configured world")
    if "eval_num_shards" in manifest and manifest["eval_num_shards"] != world:
        raise ValueError("Logical shard count differs from world_size")
    if any(value != manifest for value in manifests.values()):
        raise ValueError("Mixed generation manifests within a run")
    expected = {task.id for task in tasks}
    if len(expected) != len(tasks):
        raise ValueError("Duplicate evaluation task IDs")
    paths = {_rank(path, "predictions_rank"): path
             for path in output.glob("predictions_rank*.jsonl")}
    if set(paths) != set(range(world)):
        raise ValueError("Prediction shards do not cover the configured world")
    saved, owners, retries = {}, {}, Counter()
    for rank, path in sorted(paths.items()):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = _json(line)
                if not isinstance(row, dict) or not isinstance(row.get("task_id"), str) or not row["task_id"]:
                    raise ValueError("Prediction requires a nonempty task_id")
                task_id = row["task_id"]
                if task_id in owners:
                    if owners[task_id] != rank:
                        raise ValueError("Duplicate prediction across shards")
                    retries[task_id] += 1
                owners[task_id], saved[task_id] = rank, row
        assigned = {task.id for task in tasks[rank::world]}
        if {key for key, owner in owners.items() if owner == rank} != assigned:
            raise ValueError("Prediction task coverage or logical shard ownership differs")
        marker = _read(output / f"rank{rank}.done")
        if marker != {"tasks": len(assigned), "world_size": world}:
            raise ValueError("Shard completion marker differs from task ownership")
    if set(saved) != expected:
        raise ValueError("Prediction task sets differ from evaluation tasks")
    for task in tasks:
        row = saved[task.id]
        statuses = STATUSES | ({"model_protocol_failure"} if "native_thinking" in manifest else set())
        if row.get("status") not in statuses:
            raise ValueError("Prediction has an unknown or missing episode status")
        for key, expected_value in (("benchmark", task.benchmark), ("split", task.split)):
            if key in row and row[key] != expected_value:
                raise ValueError(f"Prediction {key} differs from evaluation task")
        if row["status"] == "completed" and not isinstance(row.get("answer"), list):
            raise ValueError("Completed prediction requires an answer list")
    runtime_paths = list(output.glob("inference_runtime*.json"))
    runtimes = [_read(path) for path in runtime_paths]
    runtime = runtimes[0] if runtimes else None
    if any(value != runtime for value in runtimes):
        raise ValueError("Mixed inference runtimes within a run")
    if manifest.get("inference_backend", "hf") == "vllm":
        for rank in range(world):
            if not tasks[rank::world]:
                continue
            name = f"inference_runtime_rank{rank}.json" if world > 1 else "inference_runtime.json"
            if name not in {path.name for path in runtime_paths}:
                raise ValueError("Missing inference runtime for a nonempty shard")
        if runtime is None or runtime.get("model") != manifest.get("model") or runtime.get("backend") != "vllm":
            raise ValueError("Inference runtime does not identify the manifest model/backend")
        if "adapter_identity" in manifest and runtime.get("adapter") != manifest["adapter_identity"]:
            raise ValueError("Inference runtime adapter differs from generation manifest")
    return {"output": output, "manifest": manifest, "runtime": runtime, "predictions": saved,
            "_native_context": (owners, tasks) if "native_thinking" in manifest else None,
            "retry_records": sum(retries.values()), "retried_tasks": len(retries),
            "status_counts": dict(Counter(row["status"] for row in saved.values()))}



def _validate_native_pair_run(run, generation, kind):
    """Grant policy-field exemptions only after original native/artifact checks."""
    manifest, runtime = run["manifest"], run["runtime"] or {}
    records = (manifest, runtime, generation)
    if not any("native_thinking" in record for record in records):
        if (any(key in manifest for key in ("native_lora_derivative_id", "native_lora_training_approval"))
                or "native_lora_binding" in runtime):
            raise ValueError("Native policy bindings require the native generation protocol")
        return False
    config = manifest.get("native_thinking")
    if (not isinstance(config, dict) or runtime.get("native_thinking") != config
            or generation.get("native_thinking") != config):
        raise ValueError("Native configuration differs between protocol, manifest and runtime")
    from slime_plugins.dcc.native_budget import BUDGET_PROTOCOL, approved_training_receipt
    from slime_plugins.dcc.native_thinking_eval import validate_native_sidecars
    context = run.get("_native_context")
    if context is None:
        raise ValueError("Native pairing requires the original saved task/trace ownership")
    owners, tasks = context
    validate_native_sidecars(run["output"], run["predictions"], manifest, runtime, owners, tasks)
    # The strict scorer must have consumed these exact manifests and predictions.
    report = _read(run["output"] / "scores_strict.json")
    if (report.get("generation_manifest") != manifest or report.get("inference_runtime") != runtime
            or report.get("tasks") != len(run["predictions"]) or report.get("prediction_coverage") != 1
            or report.get("status_counts") != run["status_counts"]
            or report.get("policy_quantization") != manifest.get("quantization", "none")
            or report.get("policy_inference_backend") != manifest.get("inference_backend", "hf")
            or (report.get("complete") is not True and not run["status_counts"].get("infrastructure_error"))):
        raise ValueError("Native strict scorer report differs from saved generation/coverage")
    if kind == "base":
        if manifest.get("adapter_identity") is not None or runtime.get("adapter") is not None:
            raise ValueError("Base native policy cannot claim an adapter identity")
    else:
        from slime_plugins.dcc.native_lora import validate_native_adapter
        approval = manifest.get("native_lora_training_approval") or {}
        receipt = approved_training_receipt(config["source_receipt"],
            approval.get("source_commit"), approval.get("protocol"))
        metadata, binding = validate_native_adapter(manifest["adapter"],
            source_receipt=config["source_receipt"],
            approved_derivative_id=manifest.get("native_lora_derivative_id"),
            model_config=_read(Path(manifest["model"]) / "config.json"),
            approved_training_source_receipt=receipt,
            approved_training_protocol=approval.get("protocol"), evaluation_protocol=config["protocol"])
        from slime_plugins.dcc.adapter_identity import saved_adapter_matches
        if (not saved_adapter_matches(manifest.get("adapter_identity"), metadata)
                or manifest.get("adapter_identity") != runtime.get("adapter")
                or binding != runtime.get("native_lora_binding")):
            raise ValueError("Native paired policy differs from actual local adapter metadata/binding")
    return config["protocol"] == BUDGET_PROTOCOL


def validate_protocols(baseline, trained, baseline_protocol, trained_protocol):
    protocols = (baseline_protocol, trained_protocol)
    native_v3 = []
    for protocol, run, kind in zip(protocols, (baseline, trained), ("base", "lora"), strict=True):
        if protocol.get("schema") != "dcc_paired_run_protocol_v1":
            raise ValueError("A dcc_paired_run_protocol_v1 declaration is required for each run")
        common, policy = protocol.get("common"), protocol.get("policy")
        if not isinstance(common, dict) or any(not common.get(key) for key in
                ("base_model", "harness", "generation", "retrieval", "scoring")):
            raise ValueError("Protocol must identify base_model, harness, generation, retrieval and scoring")
        if not isinstance(policy, dict) or policy.get("kind") != kind or "adapter" not in policy:
            raise ValueError("Policies must identify the unadapted base and a trained LoRA")
        adapter = run["manifest"].get("adapter")
        if policy["adapter"] != adapter or (kind == "base" and adapter is not None) or (
                kind == "lora" and (not isinstance(adapter, str) or not adapter)):
            raise ValueError("Protocol policy adapter differs from saved generation")
        generation = common["generation"]
        required = ("max_steps", "max_new_tokens", "max_prompt_tokens", "initial_chunks", "seed",
                    "timeout_seconds", "temperature", "backend", "quantization", "batch_size")
        if not isinstance(generation, dict) or any(key not in generation for key in required):
            raise ValueError("Protocol generation must include all budgets, seed, timeout, backend and batching")
        native_v3.append(_validate_native_pair_run(run, generation, kind))
        validate_web_navigation_records((run["manifest"], run["runtime"] or {}, generation))
        validate_web_research_records((run["manifest"], run["runtime"] or {}, generation))
        validate_web_cached_rows_records((run["manifest"], run["runtime"] or {}, generation))
        review_records = (run["manifest"], run["runtime"] or {}, generation)
        validate_web_whitespace_records(review_records)
        review_values = [record.get("web_explicit_review", web_explicit_review_configuration(False))
                         for record in review_records]
        for value in review_values:
            if (not isinstance(value, dict) or type(value.get("enabled")) is not bool
                    or value != web_explicit_review_configuration(value["enabled"])):
                raise ValueError("Web explicit review requires complete supported configuration")
        if any(value != review_values[0] for value in review_values[1:]):
            raise ValueError("Web explicit review differs between protocol, manifest and runtime")
        if review_values[0]["enabled"] and any(
                record.get("web_structured_final") != web_structured_configuration(True)
                or record.get("web_structured_actions") != web_structured_actions_configuration(True)
                for record in review_records):
            raise ValueError("Web explicit review requires matching structured Web actions and final")
        browser_records = (run["manifest"], run["runtime"] or {}, generation)
        if any("web_browser_reader" in record for record in browser_records):
            browser_values = [validate_browser_reader_runtime(record.get("web_browser_reader"))
                              for record in browser_records]
            if any(value != browser_values[0] for value in browser_values[1:]):
                raise ValueError("Web browser reader differs between protocol, manifest and inference runtime")
            if generation["backend"] != "vllm" or (run["runtime"] or {}).get("backend") != "vllm":
                raise ValueError("Web browser reader requires the vLLM evaluation backend")
        disabled_structured = web_structured_configuration(False)
        structured_values = [
            run["manifest"].get("web_structured_final", disabled_structured),
            (run["runtime"] or {}).get("web_structured_final", disabled_structured),
            generation.get("web_structured_final", disabled_structured),
        ]
        for value in structured_values:
            if (not isinstance(value, dict) or type(value.get("enabled")) is not bool
                    or value != web_structured_configuration(value["enabled"])):
                raise ValueError("Structured Web final must declare the complete supported configuration")
        if any(value != structured_values[0] for value in structured_values[1:]):
            raise ValueError("Structured Web final differs between protocol, manifest and inference runtime")
        if structured_values[0]["enabled"] and generation["backend"] != "vllm":
            raise ValueError("Structured Web final requires the vLLM backend")
        disabled_actions = web_structured_actions_configuration(False)
        action_values = [
            run["manifest"].get("web_structured_actions", disabled_actions),
            (run["runtime"] or {}).get("web_structured_actions", disabled_actions),
            generation.get("web_structured_actions", disabled_actions),
        ]
        for value in action_values:
            if (not isinstance(value, dict) or type(value.get("enabled")) is not bool
                    or value != web_structured_actions_configuration(value["enabled"])):
                raise ValueError("Structured Web actions must declare the complete supported configuration")
        if any(value != action_values[0] for value in action_values[1:]):
            raise ValueError("Structured Web actions differ between protocol, manifest and inference runtime")
        if action_values[0]["enabled"] and (
                not structured_values[0]["enabled"] or generation["backend"] != "vllm"):
            raise ValueError("Structured Web actions require structured final and the vLLM backend")
        for key in ("max_steps", "max_new_tokens", "max_prompt_tokens", "initial_chunks", "batch_size"):
            if type(generation[key]) is not int or generation[key] < 1:
                raise ValueError("Protocol token/turn/batch budgets must be positive integers")
        if type(generation["seed"]) is not int or generation["seed"] < 0:
            raise ValueError("Protocol seed must be a nonnegative integer")
        timeout = generation["timeout_seconds"]
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Protocol timeout must be finite and positive")
        saved_sampling = run["manifest"].get("evaluation_sampling")
        declared_temperature = generation["temperature"]
        if (isinstance(declared_temperature, bool) or not isinstance(declared_temperature, (int, float))
                or not math.isfinite(declared_temperature) or not 0 <= declared_temperature <= 2):
            raise ValueError("Protocol evaluation sampling temperature must be finite and between 0 and 2")
        if saved_sampling is None:
            if isinstance(generation["temperature"], bool) or generation["temperature"] != 0:
                raise ValueError("Legacy paired evaluation requires greedy temperature=0")
            if "sampling" in generation:
                if generation["sampling"] != EvaluationSampling().to_dict():
                    raise ValueError("Legacy paired evaluation cannot attest nondefault sampling")
        else:
            if not isinstance(saved_sampling, dict) or set(saved_sampling) != set(EvaluationSampling().to_dict()):
                raise ValueError("Saved evaluation sampling must declare all supported parameters")
            declared_sampling = EvaluationSampling(**saved_sampling).to_dict()
            protocol_sampling = generation.get("sampling")
            if not isinstance(protocol_sampling, dict) or set(protocol_sampling) != set(declared_sampling):
                raise ValueError("Protocol evaluation sampling must declare all supported parameters")
            protocol_sampling = EvaluationSampling(**protocol_sampling).to_dict()
            if protocol_sampling != declared_sampling or generation["temperature"] != declared_sampling["temperature"]:
                raise ValueError("Protocol evaluation sampling differs from saved manifest")
            if (run["runtime"] is None or run["runtime"].get("evaluation_sampling") != declared_sampling
                    or generation["backend"] != "vllm"):
                raise ValueError("Inference runtime does not attest the saved evaluation sampling")
        for key in ("max_steps", "max_new_tokens", "max_prompt_tokens", "initial_chunks", "seed"):
            if generation[key] != run["manifest"].get(key):
                raise ValueError(f"Protocol generation {key} differs from saved manifest")
        for key, value in (("backend", run["manifest"].get("inference_backend", "hf")),
                           ("quantization", run["manifest"].get("quantization", "none"))):
            if generation[key] != value:
                raise ValueError(f"Protocol generation {key} differs from saved manifest")
        saved_batch = run["manifest"].get("inference_parameters", {}).get("batch_size")
        if saved_batch is not None and saved_batch != generation["batch_size"]:
            raise ValueError("Protocol batch size differs from saved manifest")
        if "timeout_seconds" in run["manifest"] and timeout != run["manifest"]["timeout_seconds"]:
            raise ValueError("Protocol timeout_seconds differs from saved manifest")
        retrieval = common["retrieval"]
        if not isinstance(retrieval, dict) or "retrieval_context" not in retrieval:
            raise ValueError("Protocol retrieval must include retrieval_context")
        context = retrieval["retrieval_context"]
        if not isinstance(context, dict) or set(context) != {"context_chars", "max_context_chars"} or (
                type(context["context_chars"]) is not int or context["context_chars"] < 0 or
                type(context["max_context_chars"]) is not int or context["max_context_chars"] < 1):
            raise ValueError("Protocol retrieval_context must declare valid context limits")
        saved_context = run["manifest"].get("retrieval_context", {"context_chars": 0, "max_context_chars": 16000})
        if context != saved_context:
            raise ValueError("Protocol retrieval_context differs from saved manifest/defaults")
        if retrieval.get("retrieval_profile") != run["manifest"].get("retrieval_profile"):
            raise ValueError("Protocol retrieval_profile differs from saved manifest")
        if any(row.get("benchmark") == "widesearch" for row in run["predictions"].values()) or "search_backend" in run["manifest"]:
            web_keys = ("search_backend", "search_endpoint", "web_timeout_seconds", "auto_read_sources")
            if any(key not in retrieval for key in web_keys):
                raise ValueError("Protocol retrieval must declare web timeout, search backend/endpoint and auto-read")
            web_timeout = retrieval["web_timeout_seconds"]
            if isinstance(web_timeout, bool) or not isinstance(web_timeout, (int, float)) or not math.isfinite(web_timeout) or web_timeout <= 0:
                raise ValueError("Protocol web timeout must be finite and positive")
            if not isinstance(retrieval["search_backend"], str) or not retrieval["search_backend"] or (
                    retrieval["search_endpoint"] is not None and not isinstance(retrieval["search_endpoint"], str)):
                raise ValueError("Protocol search backend/endpoint must identify the configured search service")
            if type(retrieval["auto_read_sources"]) is not int or not 0 <= retrieval["auto_read_sources"] <= 2:
                raise ValueError("Protocol auto_read_sources must be an integer between 0 and 2")
            for key in web_keys:
                if key in run["manifest"] and retrieval[key] != run["manifest"][key]:
                    raise ValueError(f"Protocol {key} differs from saved manifest")
            if retrieval["auto_read_sources"] != run["manifest"].get("auto_read_sources", 0):
                raise ValueError("Protocol auto_read_sources differs from saved manifest/defaults")
    if baseline_protocol["common"] != trained_protocol["common"]:
        raise ValueError("Paired protocols differ in base model, harness, budget, retrieval or scoring")
    exclude = {"adapter", "adapter_identity"}
    runtime_exclude = {"adapter"}
    if all(native_v3):
        exclude.update({"native_lora_derivative_id", "native_lora_training_approval"})
        runtime_exclude.add("native_lora_binding")
    manifests = [{key: value for key, value in run["manifest"].items() if key not in exclude}
                 for run in (baseline, trained)]
    if manifests[0] != manifests[1]:
        raise ValueError("Saved generation manifests differ beyond the adapter")
    runtimes = [None if run["runtime"] is None else
                {key: value for key, value in run["runtime"].items() if key not in runtime_exclude}
                for run in (baseline, trained)]
    if runtimes[0] != runtimes[1]:
        raise ValueError("Inference runtime differs beyond the adapter")
    return baseline_protocol["common"]


def cuad_cluster_counts(tasks, predictions):
    import numpy as np

    contract_ids = [task.metadata.get("contract_id") for task in tasks]
    if any(not isinstance(group, str) or not group.strip() for group in contract_ids):
        raise ValueError("CUAD bootstrap requires a nonempty metadata.contract_id for every task")
    groups = sorted(set(contract_ids))
    index = {group: number for number, group in enumerate(groups)}
    counts = np.zeros((len(groups), len(THRESHOLDS) + 1, 3), dtype=np.int64)
    for task in tasks:
        gold = [] if task.gold is None else task.gold
        if not isinstance(gold, list) or any(not isinstance(span, str) or not span for span in gold):
            raise ValueError("CUAD gold must be a list of nonempty spans")
        row = predictions[task.id]
        nbest = cuad_predictions(row.get("answer", []) if row["status"] == "completed" else [])
        memo = {}
        for slot, threshold in enumerate((*THRESHOLDS, None)):
            spans = tuple(item["text"] for item in nbest if threshold is None or item["probability"] > threshold)
            if spans not in memo:
                memo[spans] = cuad_counts(task.id, gold, list(spans))
            counts[index[task.metadata["contract_id"]], slot] += memo[spans]
    return groups, counts


def cuad_aggregate(counts):
    """Vectorized official threshold curve; preserve builtin max's NaN order."""
    import numpy as np

    counts = np.asarray(counts)
    if counts.ndim == 2:
        counts = counts[None, ...]
    tp, fp, fn = (counts[:, :-1, column] for column in range(3))
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.concatenate((np.ones((len(counts), 1)), tp / (tp + fp)), axis=1)
        recall = np.concatenate((np.zeros((len(counts), 1)), tp / (tp + fn)), axis=1)
    for position in range(precision.shape[1] - 2, -1, -1):
        # max(next, current) returns next when current or next is NaN.
        precision[:, position] = np.where(precision[:, position] > precision[:, position + 1],
                                          precision[:, position], precision[:, position + 1])
    area = np.sum(np.diff(recall, axis=1) * (precision[:, 1:] + precision[:, :-1]) / 2, axis=1)
    area = np.where(np.isnan(area), 0.0, area)
    totals = counts[:, -1]
    denominator = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
    f1 = np.divide(2 * totals[:, 0], denominator, out=np.zeros(len(counts)), where=denominator != 0)
    return {"aupr": area, "micro_f1": f1}


def load_wide_scores(run, tasks, score_name):
    report = _read(run["output"] / score_name)
    wide = report.get("widesearch", {})
    if report.get("generation_manifest") != run["manifest"] or (
            run["runtime"] is not None and report.get("inference_runtime") != run["runtime"]):
        raise ValueError("Scorer provenance differs from saved generation")
    if report.get("tasks") != len(run["predictions"]) or report.get("prediction_coverage") != 1:
        raise ValueError("Scorer report has incomplete prediction coverage")
    if report.get("status_counts") != run["status_counts"]:
        raise ValueError("Scorer status counts differ from saved predictions")
    if report.get("policy_quantization") != run["manifest"].get("quantization", "none") or report.get("policy_inference_backend") != run["manifest"].get("inference_backend", "hf"):
        raise ValueError("Scorer policy precision/backend differs from saved generation")
    if wide.get("tasks") != len(tasks) or wide.get("metric_denominator") != len(tasks) or wide.get("judge_failures") != 0:
        raise ValueError("WideSearch scores must include every task with zero judge failures")
    if not report.get("complete") and not run["status_counts"].get("infrastructure_error"):
        raise ValueError("Scorer report is incomplete")
    protocol = wide.get("evaluation_protocol")
    if protocol not in {"strict_exact_proxy", "official_protocol_custom_judge"}:
        raise ValueError("Missing or unsupported WideSearch scoring protocol")
    provenance = {key: wide.get(key) for key in ("evaluation_protocol", "alignment_protocol", "table_input_protocol", "judge_model")}
    if not isinstance(provenance["table_input_protocol"], str) or not provenance["table_input_protocol"]:
        raise ValueError("WideSearch scores require table_input_protocol provenance")
    provenance.update({key: report.get(key) for key in ("judge_configuration", "judge_quantization")})
    if protocol == "official_protocol_custom_judge" and (not provenance["judge_model"] or
            not provenance["alignment_protocol"] or not provenance["judge_configuration"]):
        raise ValueError("Semantic scores require complete frozen judge provenance")
    if protocol == "official_protocol_custom_judge" and (report.get("judge_configuration_complete") is not True or wide.get("alignment_protocol_complete") is not True):
        raise ValueError("Semantic scores contain incomplete judge/alignment provenance")
    rows = wide.get("per_task")
    if not isinstance(rows, list):
        raise ValueError("WideSearch report requires per_task scores")
    saved = {}
    overrides = 0
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("task_id"), str) or row["task_id"] in saved:
            raise ValueError("Duplicate or missing WideSearch score task ID")
        if "judge_error" in row:
            raise ValueError("Unresolved WideSearch judge error")
        if row.get("alignment_protocol") != provenance["alignment_protocol"]:
            raise ValueError("Mixed WideSearch alignment protocols")
        if row.get("table_input_protocol") != provenance["table_input_protocol"]:
            raise ValueError("Mixed WideSearch table input protocols")
        values = []
        for key in WIDE_METRICS:
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("WideSearch per-task metrics must be finite and in [0, 1]")
            values.append(float(value))
        saved[row["task_id"]] = values
    if set(saved) != {task.id for task in tasks}:
        raise ValueError("WideSearch score task sets differ from evaluation tasks")
    for position, key in enumerate(WIDE_METRICS):
        reported = wide.get(key)
        observed = sum(values[position] for values in saved.values()) / len(tasks)
        if isinstance(reported, bool) or not isinstance(reported, (int, float)) or not math.isfinite(reported) or not math.isclose(reported, observed, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("WideSearch aggregate differs from its complete per-task scores")
    for task in tasks:
        if run["predictions"][task.id]["status"] != "completed":
            overrides += int(any(saved[task.id]))
            saved[task.id] = [0.0] * len(WIDE_METRICS)
    return [saved[task.id] for task in sorted(tasks, key=lambda task: task.id)], provenance, overrides


def _summarize(base, trained, differences, confidence):
    import numpy as np

    alpha = (1 - confidence) / 2
    low, high = np.quantile(differences, (alpha, 1 - alpha), method="linear")
    return {"baseline": float(base), "trained": float(trained), "delta": float(trained - base),
            "delta_ci": [float(low), float(high)], "ci_excludes_zero_positive": bool(low > 0)}


def compare(tasks, baseline, trained, baseline_protocol, trained_protocol, *, score_name="scores_strict.json",
            resamples=10000, seed=47, confidence=0.95):
    import numpy as np

    if type(resamples) is not int or resamples < 100 or type(seed) is not int or seed < 0:
        raise ValueError("Use at least 100 bootstrap resamples and a nonnegative integer seed")
    if isinstance(confidence, bool) or not isinstance(confidence, (float, int)) or not 0 < confidence < 1:
        raise ValueError("Confidence level must be in (0, 1)")
    if not tasks or any(task.split == "train" for task in tasks) or len({task.id for task in tasks}) != len(tasks) or len({task.split for task in tasks}) != 1:
        raise ValueError("Comparison requires nonempty unique held-out evaluation tasks")
    common = validate_protocols(baseline, trained, baseline_protocol, trained_protocol)
    expected_ids = {task.id for task in tasks}
    if any(set(run["predictions"]) != expected_ids for run in (baseline, trained)):
        raise ValueError("Paired prediction task sets differ from evaluation tasks")
    rng = np.random.default_rng(seed)
    benchmarks = {}
    for benchmark in ("cuad", "widesearch"):
        subset = [task for task in tasks if task.benchmark == benchmark]
        if not subset:
            continue
        if benchmark == "cuad":
            groups, base_counts = cuad_cluster_counts(subset, baseline["predictions"])
            train_groups, trained_counts = cuad_cluster_counts(subset, trained["predictions"])
            assert groups == train_groups
            units = len(groups)
            base_values, trained_values = cuad_aggregate(base_counts.sum(axis=0)), cuad_aggregate(trained_counts.sum(axis=0))
            draws = {key: [] for key in base_values}
            for start in range(0, resamples, 256):
                weights = rng.multinomial(units, np.full(units, 1 / units), size=min(256, resamples - start))
                left = cuad_aggregate((weights @ base_counts.reshape(units, -1)).reshape(-1, len(THRESHOLDS) + 1, 3))
                right = cuad_aggregate((weights @ trained_counts.reshape(units, -1)).reshape(-1, len(THRESHOLDS) + 1, 3))
                for key in draws:
                    draws[key].extend(right[key] - left[key])
            metrics = {key: _summarize(base_values[key][0], trained_values[key][0], draws[key], confidence)
                       for key in draws}
            benchmarks[benchmark] = {"tasks": len(subset), "resampling_unit": "contract", "units": units,
                                      "metrics": metrics, "confidence_source": "model_generated_span_confidence"}
        else:
            left, left_provenance, left_overrides = load_wide_scores(baseline, subset, score_name)
            right, right_provenance, right_overrides = load_wide_scores(trained, subset, score_name)
            if left_provenance != right_provenance:
                raise ValueError("WideSearch scorer/judge protocols differ between paired runs")
            if not isinstance(common["scoring"], dict) or common["scoring"].get("widesearch_protocol") != left_provenance["evaluation_protocol"]:
                raise ValueError("Declared WideSearch scoring protocol differs from saved scores")
            if common["scoring"].get("table_input_protocol") != left_provenance["table_input_protocol"]:
                raise ValueError("Declared WideSearch table input protocol differs from saved scores")
            if "judge_configuration" in common["scoring"] and common["scoring"]["judge_configuration"] != left_provenance["judge_configuration"]:
                raise ValueError("Declared judge configuration differs from saved scores")
            left, right = np.asarray(left), np.asarray(right)
            units = len(subset)
            difference = right - left
            samples = []
            for start in range(0, resamples, 256):
                weights = rng.multinomial(units, np.full(units, 1 / units), size=min(256, resamples - start))
                samples.append(weights @ difference / units)
            samples = np.concatenate(samples)
            metrics = {key: _summarize(left[:, position].mean(), right[:, position].mean(), samples[:, position], confidence)
                       for position, key in enumerate(WIDE_METRICS)}
            benchmarks[benchmark] = {"tasks": len(subset), "resampling_unit": "task", "units": units,
                                      "metrics": metrics, "scoring": left_provenance,
                                      "nonzero_failed_score_overrides": {"baseline": left_overrides, "trained": right_overrides}}
    infrastructure = {name: run["status_counts"].get("infrastructure_error", 0)
                      for name, run in (("baseline", baseline), ("trained", trained))}
    eligible = not any(infrastructure.values()) and all(value["units"] >= 2 for value in benchmarks.values())
    for name, result in benchmarks.items():
        primary = result["metrics"][PRIMARY_METRICS[name]]
        result.update(primary_metric=PRIMARY_METRICS[name],
                      primary_improvement_supported=eligible and primary["ci_excludes_zero_positive"])
    report = {"schema": "dcc_paired_comparison_v1", "complete": True, "tasks": len(tasks),
            "common_protocol": common, "protocol_evidence": "Supplied common declarations plus saved generation/runtime/scorer consistency; no historical code or model bytes are verified.",
            "legacy_timeout_declarations": {
                name: [key for key in ("timeout_seconds", "web_timeout_seconds")
                       if key not in run["manifest"] and (key == "timeout_seconds" or any(task.benchmark == "widesearch" for task in tasks))]
                for name, run in (("baseline", baseline), ("trained", trained))},
            "bootstrap": {"method": "paired percentile cluster bootstrap", "resamples": resamples, "seed": seed,
                          "confidence": confidence, "intervals": "marginal; no multiple-comparison adjustment"},
            "failure_policy": "Every non-completed episode has an empty CUAD answer or zero WideSearch score; all tasks remain in the denominator.",
            "infrastructure_errors": infrastructure, "eligible_for_improvement_claim": eligible,
            "all_primary_improvements_supported": eligible and all(value["primary_improvement_supported"] for value in benchmarks.values()),
            "runs": {name: {key: run[key] for key in ("retry_records", "retried_tasks", "status_counts")}
                     for name, run in (("baseline", baseline), ("trained", trained))},
            "benchmarks": benchmarks,
            "inference_scope": "Conditional on fixed task, retrieval/source and scorer artifacts. A live-web protocol cannot establish source determinism. No checkpoint selection or fresh-test claim is made."}
    if "native_thinking" in common["generation"]:
        report["native_policy_bindings"] = {name: {
            "adapter_identity": run["manifest"].get("adapter_identity"),
            "native_lora_derivative_id": run["manifest"].get("native_lora_derivative_id"),
            "native_lora_training_approval": run["manifest"].get("native_lora_training_approval"),
            "native_lora_binding": (run["runtime"] or {}).get("native_lora_binding"),
        } for name, run in (("baseline", baseline), ("trained", trained))}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline-output", type=Path, required=True)
    parser.add_argument("--trained-output", type=Path, required=True)
    parser.add_argument("--baseline-protocol", type=Path, required=True)
    parser.add_argument("--trained-protocol", type=Path, required=True)
    parser.add_argument("--wide-score-name", choices=("scores_strict.json", "scores_local_judge.json"), default="scores_strict.json")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        tasks = [task for path in args.data for task in load_tasks(path)]
        result = compare(tasks, load_run(args.baseline_output, tasks), load_run(args.trained_output, tasks),
                         _read(args.baseline_protocol), _read(args.trained_protocol), score_name=args.wide_score_name,
                         resamples=args.resamples, seed=args.seed, confidence=args.confidence)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"tasks": result["tasks"], "eligible_for_improvement_claim": result["eligible_for_improvement_claim"],
                      "all_primary_improvements_supported": result["all_primary_improvements_supported"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
