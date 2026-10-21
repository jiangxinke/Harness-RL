#!/usr/bin/env python3
"""Matched CUAD evaluation for the independent native HF/NF4 online policy.

Run base and adapter in separate output directories with identical arguments.
Each task gets its own deterministic seed, independent of sharding. Generate
one shard per GPU, then use --score-only to score all closed shards together.
This protocol must not be merged with existing vLLM native benchmark results.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import random
import re
import sys
import time
import zlib

sys.path[:0] = [str(Path(__file__).resolve().parent),
                str(Path(__file__).resolve().parents[1] / "slime")]

import torch

from dcc_online_grpo import make_runner, validate_source
from slime_plugins.dcc import load_tasks
from slime_plugins.dcc.cuad_reasoning import cuad_concise_reasoning_configuration
from slime_plugins.dcc.web_auto_read import auto_read_follow_docs_configuration
from slime_plugins.dcc.durable_journal import OrderedDurableJSONLWriter, close_durable_journals
from slime_plugins.dcc.evaluation import evaluate_results
from slime_plugins.dcc.hf_native_policy import NativeHFPolicy, PROTOCOL
from slime_plugins.dcc.hf_policy import load_policy_model
from slime_plugins.dcc.hf_compute import (COMPUTE_PROFILES, adapter_compute_profile,
    compute_profile_configuration, precision_observation, validate_compute_profile, validate_model_compute_profile)
from slime_plugins.dcc.metrics import cuad_task_metrics
from slime_plugins.dcc.online_grpo import ALGORITHM

VERSION = "dcc-online-matched-HF-CUAD-evaluation-v1"
SCORE_FIELDS = {"f1", "precision", "recall", "no_answer_correct", "tp", "fp", "fn",
                "verbatim_span_fraction"}


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n"):
                raise ValueError("Prediction journal ends with an incomplete record")
            rows.append(json.loads(line))
    return rows


def task_seed(seed, task_id):
    """Stable public task seed; no task labels or Python hash randomization."""
    return (seed + zlib.crc32(task_id.encode("utf-8"))) % (2**32)


def public_task_identity(tasks):
    """One compact canonical identity of public inputs, never task labels.

    Repeated CUAD documents are hashed once; this does not inspect model files.
    """
    documents, rows = {}, []
    for task in tasks:
        if task.document not in documents:
            documents[task.document] = hashlib.blake2b(task.document.encode("utf-8"), digest_size=16).hexdigest()
        rows.append([task.id, task.benchmark, task.split, task.prompt, documents[task.document]])
    canonical = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2b(canonical, digest_size=16).hexdigest()


def adapter_condition(adapter):
    if not adapter:
        return dict(adapter=None, policy="base")
    root = Path(adapter)
    config = (json.loads((root / "training_config.json").read_text(encoding="utf-8"))
              if (root / "training_config.json").is_file() else {})
    checkpoint = (json.loads((root / "checkpoint_complete.json").read_text(encoding="utf-8"))
                  if (root / "checkpoint_complete.json").is_file() else {})
    training = dict(compute_profile=adapter_compute_profile(adapter), precision=config.get("precision"),
        algorithm=config.get("algorithm"), training_source_commit=config.get("source_commit"),
        rollout_iterations=config.get("rollout_iterations"), optimizer_updates=config.get("optimizer_updates"),
        checkpoint_complete=checkpoint.get("complete") is True,
        checkpoint_optimizer_steps=checkpoint.get("optimizer_steps"))
    updates, iterations = training["optimizer_updates"], training["rollout_iterations"]
    recorded = (training["algorithm"] == ALGORITHM
        and isinstance(training["training_source_commit"], str)
        and re.fullmatch(r"[0-9a-f]{40}", training["training_source_commit"]) is not None
        and type(updates) is int and updates > 0 and type(iterations) is int and iterations > 0
        and training["checkpoint_complete"] and training["checkpoint_optimizer_steps"] == updates
        and config.get("sft_auxiliary") is False and config.get("offline_preferences") is False)
    # Evaluation source identity lives in comparison.source_commit. Training
    # uses its own frozen commit; those two commits need not be equal.
    return dict(adapter=adapter, policy="adapter", training=training,
                online_policy_gradient_updates_recorded=recorded)


def manifest(args, tasks):
    compute = validate_compute_profile(getattr(args, "compute_profile", "legacy"), adapter=args.adapter)
    profile = {"thinking": dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20),
               "greedy": dict(do_sample=False, temperature=1.0, top_p=1.0, top_k=0)}[args.profile]
    retrieval = (json.loads(Path(args.retrieval_profile).read_text(encoding="utf-8"))
                 if args.retrieval_profile else None)
    comparison = dict(source_commit=args.source_commit, protocol=PROTOCOL,
        runtime="independent-HF; separate from vLLM native v3 and Slime/Megatron",
        model=args.model, precision=dict(quantization=args.quantization,
            load_dtype="bfloat16", nf4_compute_dtype="bfloat16" if args.quantization == "nf4" else None,
            kbit_preparation=args.quantization == "nf4", inference_loader="hf_policy.load_policy_model",
            compute_profile=compute),
        thought_sampling_profile=args.profile, thought_sampling=profile,
        json_sampling="greedy-neutral-xgrammar-structural; public-schema-postvalidation",
        seed=args.seed, seed_method="(seed + crc32(utf8(task_id))) mod 2**32; reset per episode",
        max_steps=args.max_steps, thought_tokens=args.thought_tokens, json_tokens=args.json_tokens,
        max_prompt_tokens=args.max_prompt_tokens, episode_timeout=args.episode_timeout,
        initial_chunks=args.initial_chunks, retrieval_profile=retrieval,
        retrieval_context_chars=args.retrieval_context_chars,
        retrieval_max_context_chars=args.retrieval_max_context_chars,
        retrieval_task_bound_expansion=args.retrieval_task_bound_expansion,
        auto_read_sources=args.auto_read_sources, cuad_structured_actions=True,
        auto_read_follow_docs=auto_read_follow_docs_configuration(getattr(args, "auto_read_follow_docs", False)),
        cuad_concise_reasoning=cuad_concise_reasoning_configuration(getattr(args, "cuad_concise_reasoning", False)),
        episodes_per_task=1,
        allow_prompt_truncation=False, full_task_ids=[task.id for task in tasks],
        public_task_identity=public_task_identity(tasks),
        original_splits=[task.split for task in tasks],
        rollout_labels_removed=True, rollout_scored=False,
        failure_policy="persist every episode; failures remain in scoring task set; no retries")
    packages = {}
    for name in ("torch", "transformers", "peft", "bitsandbytes", "xgrammar"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return dict(version=VERSION, comparison=comparison,
                condition=adapter_condition(args.adapter),
                packages=packages, shards=args.shards, shard_index=args.shard_index,
                assigned_task_ids=[task.id for task in tasks[args.shard_index::args.shards]])


async def generate(args, tasks):
    """Persist label-free predictions; no real-gold scoring happens here."""
    configuration = manifest(args, tasks)
    rank = args.shard_index
    assigned = tasks[rank::args.shards]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / f"config_rank{rank}.json", configuration)
    writers, policy, failure = [], None, None
    started = time.monotonic()
    try:
        predictions = OrderedDurableJSONLWriter(args.output_dir / f"predictions_rank{rank}.jsonl")
        writers.append(predictions)
        traces = OrderedDurableJSONLWriter(args.output_dir / f"native_traces_rank{rank}.jsonl")
        writers.append(traces)
        model, tokenizer = load_policy_model(args.model, adapter=args.adapter,
            trainable=False, rank=args.device_index, quantization=args.quantization,
            compute_profile=getattr(args, "compute_profile", "legacy"))
        validate_model_compute_profile(model, getattr(args, "compute_profile", "legacy"))
        model.eval()
        runner = make_runner(args)
        policy = NativeHFPolicy(model, tokenizer, thought_tokens=args.thought_tokens,
            json_tokens=args.json_tokens, max_prompt_tokens=args.max_prompt_tokens,
            sampling_profile=args.profile,
            trace_sink=lambda row: traces.write_line(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"))
        for task in assigned:
            seed = task_seed(args.seed, task.id)
            random.seed(seed)
            torch.manual_seed(seed)
            # Keep the true split, but never send its labels to the runner.
            # Existing CUAD runner computes placeholder metrics on held-out
            # splits even with score_train=False; discard those below.
            result = await runner.run(replace(task, gold=None), policy.bind_task(task.id, rank),
                                      evaluation=True, score_train=False)
            await policy.close()  # Drain model access before the next task seed.
            row = result.to_dict(include_tokens=True)
            row.pop("reward", None)
            row["metrics"] = {key: value for key, value in row["metrics"].items() if key not in SCORE_FIELDS}
            row.update(task_seed=seed, rollout_labels_removed=True, rollout_scored=False)
            predictions.write_line(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        await policy.close()
        close_durable_journals(writers)
        saved = read_jsonl(args.output_dir / f"predictions_rank{rank}.jsonl")
        if [row["task_id"] for row in saved] != [task.id for task in assigned] or predictions.count != len(assigned):
            raise ValueError("Closed prediction journal does not exactly cover its assigned shard")
        write_json(args.output_dir / f"rank{rank}.done.json", dict(
            generation_complete=True, version=VERSION, source_commit=args.source_commit,
            shard_index=rank, tasks=len(assigned), traces=traces.count,
            seconds=time.monotonic() - started,
            precision=precision_observation(model),
            journals_closed=True, rollout_scored=False))
    except BaseException as error:
        failure = error
        raise
    finally:
        if policy is not None:
            try:
                await policy.close()
            except BaseException as error:
                if failure is None:
                    failure = error
                else:
                    failure.add_note(f"Policy cleanup: {type(error).__name__}: {error}")
        close_durable_journals(writers, failure=failure)
        if failure is not None and sys.exc_info()[0] is None:
            raise failure
    return configuration


def score(args, tasks):
    """Score every persisted episode only after all shard journals are closed."""
    expected = manifest(args, tasks)
    for template in ("config_rank{}.json", "predictions_rank{}.jsonl",
                     "native_traces_rank{}.jsonl", "rank{}.done.json"):
        actual = {path.name for path in args.output_dir.glob(template.format("*"))}
        if actual != {template.format(rank) for rank in range(args.shards)}:
            raise ValueError("Missing or extra evaluation shard files")
    saved, receipts, generation_packages, generation_precision = [], [], None, None
    for rank in range(args.shards):
        config = json.loads((args.output_dir / f"config_rank{rank}.json").read_text(encoding="utf-8"))
        done = json.loads((args.output_dir / f"rank{rank}.done.json").read_text(encoding="utf-8"))
        assigned = tasks[rank::args.shards]
        if generation_packages is None:
            generation_packages = config["packages"]
            generation_precision = done.get("precision")
        if (any(config[key] != expected[key] for key in ("version", "comparison", "condition", "shards"))
                or config["packages"] != generation_packages
                or config["shard_index"] != rank
                or config["assigned_task_ids"] != [task.id for task in assigned]
                or done.get("generation_complete") is not True or done.get("journals_closed") is not True
                or done.get("source_commit") != args.source_commit or done.get("shard_index") != rank
                or done.get("version") != VERSION or done.get("rollout_scored") is not False
                or not isinstance(done.get("precision"), dict) or done["precision"] != generation_precision
                or done["precision"].get("compute_profile") != expected["comparison"]["precision"]["compute_profile"]
                or done["precision"].get("compute_context", {}).get("autocast_enabled")
                    != expected["comparison"]["precision"]["compute_profile"]["autocast"]
                or done["precision"].get("compute_context", {}).get("autocast_dtype")
                    != expected["comparison"]["precision"]["compute_profile"]["autocast_dtype"]
                or done.get("tasks") != len(assigned)):
            raise ValueError("Evaluation shards differ in configuration, ownership or completed coverage")
        rows = read_jsonl(args.output_dir / f"predictions_rank{rank}.jsonl")
        if ([row.get("task_id") for row in rows] != [task.id for task in assigned]
                or any(row.get("split") != task.split or row.get("benchmark") != "cuad"
                       or row.get("task_seed") != task_seed(args.seed, task.id)
                       or row.get("rollout_labels_removed") is not True
                       or row.get("rollout_scored") is not False
                       or "reward" in row or SCORE_FIELDS.intersection(row.get("metrics", {}))
                       or row.get("status") not in {"completed", "max_steps", "invalid_answer", "evidence_failure",
                                                    "model_protocol_failure", "infrastructure_error", "context_budget_failure"}
                       for task, row in zip(assigned, rows, strict=True))):
            raise ValueError("Persisted predictions do not match the complete assigned task set")
        # Only completed answers can receive credit. Failed episodes still
        # occupy their original task position in every scoring denominator.
        saved.extend(dict(row, answer=row.get("answer", []) if row.get("status") == "completed" else []) for row in rows)
        receipts.append(done)
    report = evaluate_results(tasks, saved)
    by_id = {row["task_id"]: row for row in saved}
    macro = sum(cuad_task_metrics(task.id, task.gold, by_id[task.id]["answer"])["f1"]
                if by_id[task.id]["status"] == "completed" else 0.0 for task in tasks) / len(tasks)
    report.update(version=VERSION, comparison=expected["comparison"], condition=expected["condition"],
                  packages=generation_packages, scorer_packages=expected["packages"],
                  precision=generation_precision,
                  generation_complete=True, scored_after_journals_closed=True,
                  failure_zero_macro_f1=macro, metric_task_denominator=len(tasks), shard_receipts=receipts)
    # Labels enter only this post-generation scoring phase. Bind both sides to
    # the same references without putting held-out labels into rollout metadata.
    references = json.dumps([[task.id, task.gold] for task in tasks], ensure_ascii=False,
                            separators=(",", ":")).encode("utf-8")
    report["scoring_reference_identity"] = hashlib.blake2b(references, digest_size=16).hexdigest()
    if args.adapter:
        training_precision = expected["condition"]["training"]["precision"]
        if training_precision is not None and training_precision != generation_precision:
            raise ValueError("Adapter evaluation precision differs from its recorded training precision")
        if args.base_output_dir is None:
            raise ValueError("Adapter scoring requires --base-output-dir for an actual matched base run")
        baseline = json.loads((args.base_output_dir / "report.json").read_text(encoding="utf-8"))
        if (baseline.get("version") != VERSION or baseline.get("comparison") != report["comparison"]
                or baseline.get("condition") != {"adapter": None, "policy": "base"}
                or baseline.get("packages") != generation_packages
                or baseline.get("precision", {}).get("base_parameter_dtypes") != generation_precision["base_parameter_dtypes"]
                or baseline.get("precision", {}).get("quantization_report") != generation_precision["quantization_report"]
                or baseline.get("precision", {}).get("compute_profile") != generation_precision["compute_profile"]
                or baseline.get("precision", {}).get("compute_context") != generation_precision["compute_context"]
                or baseline.get("generation_complete") is not True or baseline.get("scored_after_journals_closed") is not True
                or baseline.get("prediction_coverage") != 1 or baseline.get("tasks") != len(tasks)
                or baseline.get("scoring_reference_identity") != report["scoring_reference_identity"]
                or baseline.get("metric_task_denominator") != len(tasks)):
            raise ValueError("Base and adapter differ in generation, retrieval, seeds, precision or task coverage")
        report["matched_base"] = dict(output=str(args.base_output_dir),
            complete=bool(report["complete"] and baseline.get("complete")),
            failure_zero_macro_f1_delta=report["failure_zero_macro_f1"] - baseline["failure_zero_macro_f1"],
            micro_f1_delta=report["cuad"]["micro_f1"] - baseline["cuad"]["micro_f1"],
            benchmark_improvement_claimed=False)
    write_json(args.output_dir / "report.json", report)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", required=True)
    result.add_argument("--data", required=True, type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--source-commit", required=True)
    result.add_argument("--adapter", help="Omit for the actual base model; supply an online final adapter to compare")
    result.add_argument("--base-output-dir", type=Path,
                        help="Required for adapter scoring; completed matched base report (generation may run in parallel)")
    result.add_argument("--split", choices=("train", "validation", "test"))
    result.add_argument("--profile", choices=("thinking", "greedy"), default="thinking")
    result.add_argument("--quantization", choices=("nf4", "none"), default="nf4")
    result.add_argument("--compute-profile", choices=COMPUTE_PROFILES, default="legacy")
    result.add_argument("--shard-index", type=int, default=0)
    result.add_argument("--shards", type=int, default=1)
    result.add_argument("--device-index", type=int, default=0)
    result.add_argument("--seed", type=int, default=47)
    result.add_argument("--thought-tokens", type=int, default=4096)
    result.add_argument("--json-tokens", type=int, default=4096)
    result.add_argument("--max-prompt-tokens", type=int, default=16384)
    result.add_argument("--max-steps", type=int, default=2)
    result.add_argument("--episode-timeout", type=float, default=7200)
    result.add_argument("--initial-chunks", type=int, default=4)
    result.add_argument("--retrieval-profile")
    result.add_argument("--retrieval-context-chars", type=int, default=1200)
    result.add_argument("--retrieval-max-context-chars", type=int, default=12000)
    result.add_argument("--retrieval-task-bound-expansion", action="store_true")
    result.add_argument("--cuad-concise-reasoning", action="store_true",
                        help="Opt in to concise CUAD source-offset reasoning; must match base and adapter evaluation")
    result.add_argument("--auto-read-sources", type=int, choices=(0, 1, 2), default=2)
    result.add_argument("--auto-read-follow-docs", action="store_true",
                        help="Prioritize one observed same-origin documentation link within existing automatic-read budgets; must match base and adapter")
    result.add_argument("--web-text-window-chars", type=int,
                        help="Optional positive fetch_url body-window limit; must match base and adapter Web evaluation")
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--score-only", action="store_true", help="Merge and score every closed shard without loading a model")
    mode.add_argument("--generate-only", action="store_true",
                      help="Generate and close journals without scoring, including with a single shard")
    return result


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    if (not re.fullmatch(r"[0-9a-f]{40}", args.source_commit)
            or any(value < 1 for value in (args.shards, args.thought_tokens, args.json_tokens,
                   args.max_prompt_tokens, args.max_steps, args.retrieval_context_chars,
                   args.retrieval_max_context_chars))
            or not 0 <= args.shard_index < args.shards or args.device_index < 0 or args.initial_chunks < 0
            or (args.web_text_window_chars is not None and args.web_text_window_chars < 1)
            or not math.isfinite(args.episode_timeout) or args.episode_timeout <= 0):
        arguments.error("Invalid source identity, shard, device or episode budget")
    validate_source(args.source_commit)
    tasks = load_tasks(args.data, split=args.split)
    if (not tasks or args.shards > len(tasks) or any(task.benchmark != "cuad" for task in tasks)
            or any(not isinstance(task.gold, list) or any(not isinstance(span, str) for span in task.gold) for task in tasks)):
        arguments.error("A nonempty labeled CUAD task set with at least one task per shard is required")
    if args.score_only:
        report = score(args, tasks)
        print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    else:
        asyncio.run(generate(args, tasks))
        if args.shards == 1 and not args.generate_only:
            report = score(args, tasks)
            print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
        else:
            print(json.dumps(dict(generation_complete=True, scoring_complete=False, shard_index=args.shard_index,
                                  next_step="Run --score-only after every shard is complete")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
