#!/usr/bin/env python3
"""Matched native-HF Web evaluation with strict row/item F1 proxy scores.

Accept prepared independent Web DEV or WideSearch task JSONL. Metadata records
dataset provenance; the `widesearch` backend field does not establish official
WideSearch200 coverage. No GPT-4.1 semantic judge is used. TEST is for final
comparison, never automatic harness selection or training.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
import urllib.request

sys.path[:0] = [str(Path(__file__).resolve().parent),
                str(Path(__file__).resolve().parents[1] / "slime")]

import torch

from dcc_online_evaluate import adapter_condition, parser as cuad_parser, task_seed
from dcc_online_grpo import make_runner, validate_source
from slime_plugins.dcc import load_tasks
from slime_plugins.dcc.adapter_identity import adapter_content_identity
from slime_plugins.dcc.durable_journal import OrderedDurableJSONLWriter, close_durable_journals
from slime_plugins.dcc.episode import EPISODE_PROMPT_VERSION, WIDESEARCH_EVIDENCE_GUARD_VERSION
from slime_plugins.dcc.hf_compute import precision_observation, validate_compute_profile, validate_model_compute_profile
from slime_plugins.dcc.hf_native_policy import NativeHFPolicy, PROTOCOL
from slime_plugins.dcc.hf_policy import load_policy_model, PROMPT_ENCODING_VERSION
from slime_plugins.dcc.tools import WEB_READER_VERSION
from slime_plugins.dcc.web_hf_evaluation import (STATUSES, OwnedEvaluationPolicy, canonical,
    dataset_identity, identity, load_public_tasks, prediction_row, public_task, read_jsonl,
    task_identity, verify_native_ownership)
from slime_plugins.dcc.widesearch_metrics import evaluate_widesearch

VERSION = "dcc-online-matched-HF-Web-proxy-evaluation-v1"
METRICS = ("precision_by_row", "recall_by_row", "f1_by_row", "precision_by_item", "recall_by_item", "f1_by_item")
JOURNALS = ("predictions", "native_traces", "request_owners")


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def web_tool_configuration(runner):
    """Bind the actual public-tool implementation and effective safe options.

    Source identities cover fixed fetch/search/parser limits that are not
    operator options. Never retain endpoint/proxy credentials or API keys.
    A matched environment does not imply frozen live search responses.
    """
    from slime_plugins.dcc import tools, search_provider, search_query

    web = runner.web_tools_factory()
    timeout = float(os.environ.get("DCC_WEB_TIMEOUT", "30"))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("DCC_WEB_TIMEOUT must be finite and positive")
    return dict(factory=f"{type(web).__module__}.{type(web).__qualname__}",
        implementation_identities={module.__name__: identity(inspect.getsource(module))
            for module in (tools, search_provider, search_query)},
        search_backend=web.backend, web_timeout_seconds=timeout,
        search_endpoint_identity=identity(os.environ.get("DCC_SEARCH_URL", "").rstrip("/")),
        effective_urllib_proxies_identity=identity(urllib.request.getproxies()),
        tls_environment_identities={key: identity(os.environ.get(key))
            for key in ("SSL_CERT_FILE", "SSL_CERT_DIR")},
        api_key_configured=bool(os.environ.get("SERPER_API_KEY" if web.backend == "serper"
            else "BRAVE_SEARCH_API_KEY")) if web.backend in {"serper", "brave"} else None,
        max_page_chars=web.max_page_chars, max_text_window_chars=web.max_text_window_chars,
        browser_reader=None if web.browser_reader is None
            else f"{type(web.browser_reader).__module__}.{type(web.browser_reader).__qualname__}",
        search_query_max_chars=1000,
        search_count_default=inspect.signature(web.web_search).parameters["count"].default,
        search_count_bounds=[1, 10], request_max_response_bytes=2_000_000,
        public_download_max_bytes=5_000_000, episode_tool_concurrency=8,
        search_headers=dict(search_provider._BROWSER_HEADERS),
        fetch_accept_language="en-US,en;q=0.9,zh;q=0.8",
        public_provider_order=dict(english=["duckduckgo_lite", "seznam", "baidu_mobile", "wikipedia"],
                                   chinese=["duckduckgo_lite", "baidu_mobile", "wikipedia"]),
        wikipedia_language="zh for CJK query; otherwise en",
        search_cache_seconds=search_provider._CACHE_SECONDS,
        search_cache_entries=1024, cache_and_cooldown_scope="process-local; live state not frozen",
        provider_min_interval_seconds=dict(duckduckgo_lite=1., public_html=.5),
        provider_cooldown_seconds=dict(challenge=300., transient_error=10.),
        extraction_limits={key: value for key, value in vars(tools).items()
            if key.startswith(("_DELIMITED_MAX_", "_XLSX_MAX_"))})


def runner_configuration(args):
    """Read actual shared runner settings, including complete versioned guards."""
    from slime_plugins.dcc.cuad_reasoning import cuad_concise_reasoning_configuration
    from slime_plugins.dcc.web_structured import (web_structured_actions_configuration,
        web_structured_configuration, web_multi_citation_configuration)
    from slime_plugins.dcc.web_review import web_explicit_review_configuration
    from slime_plugins.dcc.web_navigation import web_navigation_configuration
    from slime_plugins.dcc.web_research import web_research_configuration
    from slime_plugins.dcc.web_data_discovery import web_data_discovery_configuration
    from slime_plugins.dcc.web_rows import web_cached_rows_configuration
    from slime_plugins.dcc.web_whitespace import web_bounded_whitespace_configuration
    from slime_plugins.dcc.web_auto_read import auto_read_follow_docs_configuration

    runner = make_runner(args)
    scalar = ("max_steps", "max_new_tokens", "temperature", "evaluation_temperature", "initial_chunks",
              "timeout_seconds", "auto_read_sources", "retrieval_context_chars", "retrieval_max_context_chars",
              "retrieval_task_bound_expansion", "cuad_structured_actions")
    return {"shared_factory": "dcc_online_grpo.make_runner", "episode_prompt": EPISODE_PROMPT_VERSION,
            "cuad_concise_reasoning": cuad_concise_reasoning_configuration(runner.cuad_concise_reasoning),
            "auto_read_follow_docs": auto_read_follow_docs_configuration(runner.auto_read_follow_docs),
            "prompt_encoding": PROMPT_ENCODING_VERSION, "evidence_guard": WIDESEARCH_EVIDENCE_GUARD_VERSION,
            "web_reader": WEB_READER_VERSION, **{key: getattr(runner, key) for key in scalar},
            "retrieval_profile": asdict(runner.retrieval_profile) if runner.retrieval_profile else None,
            "web_structured_actions": web_structured_actions_configuration(runner.web_structured_actions),
            "web_structured_final": web_structured_configuration(runner.web_structured_final),
            "web_explicit_review": web_explicit_review_configuration(runner.web_explicit_review),
            "web_navigation_ledger": web_navigation_configuration(runner.web_navigation_ledger),
            "web_research_checkpoints": web_research_configuration(runner.web_research_checkpoints),
            "web_data_discovery_guidance": web_data_discovery_configuration(runner.web_data_discovery_guidance),
            "web_multi_citation": web_multi_citation_configuration(runner.web_multi_citation),
            "web_cached_rows": web_cached_rows_configuration(runner.web_cached_rows),
            "web_bounded_whitespace": web_bounded_whitespace_configuration(False),
            "web_tools": web_tool_configuration(runner)}


def manifest(args, tasks):
    from importlib.metadata import PackageNotFoundError, version
    compute = validate_compute_profile(args.compute_profile, adapter=args.adapter)
    sampling = {"raw": dict(do_sample=True, temperature=1., top_p=1., top_k=0),
                "thinking": dict(do_sample=True, temperature=.6, top_p=.95, top_k=20),
                "greedy": dict(do_sample=False, temperature=1., top_p=1., top_k=0)}[args.profile]
    packages = {}
    for name in ("torch", "transformers", "peft", "bitsandbytes", "xgrammar", "pandas",
                 "pypdf", "trafilatura", "lxml"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return dict(version=VERSION, comparison=dict(source_commit=args.source_commit, protocol=PROTOCOL,
        runtime="independent native HF; separate from vLLM native and Slime/Megatron",
        model=args.model, quantization=args.quantization, compute_profile=compute,
        thought_sampling_profile=args.profile, thought_sampling=sampling,
        json_sampling="greedy-neutral-xgrammar-structural; unchanged public schema postvalidation",
        thought_tokens=args.thought_tokens, json_tokens=args.json_tokens,
        max_prompt_tokens=args.max_prompt_tokens, allow_prompt_truncation=False,
        seed=args.seed, seed_method="(seed + crc32(utf8(task_id))) mod 2**32; reset per episode",
        episodes_per_task=1, harness=runner_configuration(args),
        full_task_ids=[task.id for task in tasks], public_task_identity=task_identity(tasks),
        dataset=dataset_identity(tasks), rollout_labels_removed=True, rollout_scored=False,
        failure_policy="persist every outcome; no retries; all noncompleted tasks score zero in full denominator",
        metric="strict row/item F1 proxy; no GPT-4.1 semantic judge",
        test_policy="final comparison only; no test-driven selection or training"),
        condition=adapter_condition(args.adapter), adapter_identity=adapter_content_identity(args.adapter),
        packages=packages, shards=args.shards, shard_index=args.shard_index,
        assigned_task_ids=[task.id for task in tasks[args.shard_index::args.shards]])


def runtime_identity(model, tokenizer, policy):
    config = model.config.to_dict()
    # Config/revision/tokenizer provenance is small. Never hash all base weights.
    for key in ("_name_or_path", "transformers_version"):
        config.pop(key, None)
    return dict(model_config_identity=identity(config), model_revision=getattr(model.config, "_commit_hash", None),
        tokenizer_class=type(tokenizer).__name__, tokenizer_size=len(tokenizer),
        tokenizer_chat_template_identity=identity(tokenizer.chat_template),
        native_suffix_ids=list(policy.suffix_ids), eos_ids=list(policy.eos_ids),
        pad_token_id=tokenizer.pad_token_id)


async def generate(args, tasks):
    """Generate without retained labels; close all three journals before done."""
    if any(task.gold is not None for task in tasks):
        raise ValueError("Generation accepts only label-free public input tasks")
    configuration = manifest(args, tasks)
    rank, assigned = args.shard_index, tasks[args.shard_index::args.shards]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / f"config_rank{rank}.json", configuration)
    writers, policy, failure = {}, None, None
    started = time.monotonic()
    try:
        for name in JOURNALS:
            writers[name] = OrderedDurableJSONLWriter(args.output_dir / f"{name}_rank{rank}.jsonl")
        def enqueue(name, row):
            writers[name].write_line(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        model, tokenizer = load_policy_model(args.model, adapter=args.adapter, trainable=False,
            rank=args.device_index, quantization=args.quantization, compute_profile=args.compute_profile)
        validate_model_compute_profile(model, args.compute_profile)
        model.eval()
        policy = NativeHFPolicy(model, tokenizer, thought_tokens=args.thought_tokens,
            json_tokens=args.json_tokens, max_prompt_tokens=args.max_prompt_tokens,
            sampling_profile=args.profile, trace_sink=lambda row: enqueue("native_traces", row))
        runtime = runtime_identity(model, tokenizer, policy)
        runner = make_runner(args)
        for task in assigned:
            seed = task_seed(args.seed, task.id)
            random.seed(seed)
            torch.manual_seed(seed)
            owner = OwnedEvaluationPolicy(policy.bind_task(task.id, rank), task_id=task.id, rank=rank,
                                          sink=lambda row: enqueue("request_owners", row))
            result = await runner.run(public_task(task), owner, evaluation=True, score_train=False)
            await policy.close()
            enqueue("predictions", prediction_row(result, seed=seed))
        await policy.close()
        close_durable_journals(writers.values())
        saved = {name: read_jsonl(args.output_dir / f"{name}_rank{rank}.jsonl") for name in JOURNALS}
        if any(len(saved[name]) != writer.count for name, writer in writers.items()):
            raise ValueError("Closed evaluation journal count differs from exact persisted records")
        ownership = verify_native_ownership(assigned, saved["predictions"], saved["request_owners"],
            saved["native_traces"], rank=rank, profile=args.profile, eos_ids=policy.eos_ids, tokenizer=tokenizer)
        write_json(args.output_dir / f"rank{rank}.done.json", dict(version=VERSION,
            source_commit=args.source_commit, shard_index=rank, tasks=len(assigned),
            generation_complete=True, policy_drained=True, journals_closed=True,
            native_token_encoding_verified=True, native_ownership=ownership,
            closed_journals={name: writer.count for name, writer in writers.items()},
            rollout_labels_removed=True, rollout_scored=False, runtime=runtime,
            precision=precision_observation(model), seconds=time.monotonic() - started))
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
        close_durable_journals(writers.values(), failure=failure)
        if failure is not None and sys.exc_info()[0] is None:
            raise failure
    return configuration


def load_closed_run(args, tasks):
    """Require every configured shard and exact trace/step coverage before score."""
    expected = manifest(args, tasks)
    for template in ("config_rank{}.json", "rank{}.done.json", *(name + "_rank{}.jsonl" for name in JOURNALS)):
        if {path.name for path in args.output_dir.glob(template.format("*"))} != {template.format(rank) for rank in range(args.shards)}:
            raise ValueError("Missing or extra closed evaluation shard files")
    saved, receipts, packages, runtime, precision = [], [], None, None, None
    for rank in range(args.shards):
        config = json.loads((args.output_dir / f"config_rank{rank}.json").read_text())
        done = json.loads((args.output_dir / f"rank{rank}.done.json").read_text())
        assigned = tasks[rank::args.shards]
        if packages is None:
            packages, runtime, precision = config.get("packages"), done.get("runtime"), done.get("precision")
        if (any(config.get(key) != expected[key] for key in ("version", "comparison", "condition", "adapter_identity", "shards"))
                or config.get("packages") != packages or config.get("shard_index") != rank
                or config.get("assigned_task_ids") != [task.id for task in assigned]
                or done.get("version") != VERSION or done.get("source_commit") != args.source_commit
                or done.get("shard_index") != rank or done.get("tasks") != len(assigned)
                or any(done.get(key) is not True for key in ("generation_complete", "policy_drained", "journals_closed", "native_token_encoding_verified", "rollout_labels_removed"))
                or done.get("rollout_scored") is not False or not isinstance(runtime, dict)
                or done.get("runtime") != runtime or not isinstance(precision, dict)
                or done.get("precision") != precision or precision.get("compute_profile") != expected["comparison"]["compute_profile"]
                or precision.get("compute_context", {}).get("autocast_enabled") != expected["comparison"]["compute_profile"]["autocast"]
                or precision.get("compute_context", {}).get("autocast_dtype") != expected["comparison"]["compute_profile"]["autocast_dtype"]):
            raise ValueError("Evaluation configuration, precision, runtime or closure differs across shards")
        rows = {name: read_jsonl(args.output_dir / f"{name}_rank{rank}.jsonl") for name in JOURNALS}
        if done.get("closed_journals") != {name: len(value) for name, value in rows.items()}:
            raise ValueError("Closed journal record counts changed")
        ownership = verify_native_ownership(assigned, rows["predictions"], rows["request_owners"], rows["native_traces"],
            rank=rank, profile=args.profile, eos_ids=runtime.get("eos_ids"))
        if ownership != done.get("native_ownership") or any(row.get("task_seed") != task_seed(args.seed, task.id)
                for task, row in zip(assigned, rows["predictions"], strict=True)):
            raise ValueError("Native ownership receipt or per-task seeds changed")
        saved.extend(rows["predictions"])
        receipts.append(done)
    return expected, saved, receipts, packages, runtime, precision


def score_predictions(tasks, predictions):
    """Every failure is zero and remains in every metric denominator."""
    by_id = {row["task_id"]: row for row in predictions}
    per_task = []
    for task in tasks:
        if not isinstance(task.gold, list) or any(not isinstance(row, dict) for row in task.gold):
            raise ValueError("Post-generation scoring requires original Web reference rows")
        evaluation = task.metadata.get("evaluation")
        if not isinstance(evaluation, dict) or any(key not in evaluation for key in ("required", "unique_columns", "eval_pipeline")):
            raise ValueError("Original Web evaluation schema required for proxy scoring")
        row = by_id[task.id]
        if row["status"] == "completed":
            metrics = evaluate_widesearch(row["answer"], task.gold, evaluation, judge=None)
            values = {key: metrics[key] for key in METRICS}
            exact = metrics["score"]
        else:
            values, exact = dict.fromkeys(METRICS, 0.), 0.
        per_task.append(dict(task_id=task.id, status=row["status"], **values, all_cells_exact_proxy=exact))
    return dict(tasks=len(tasks), metric_denominator=len(tasks), evaluation_protocol="strict_exact_proxy",
        metric_name="strict row/item F1 proxy", judge_model=None, official_gpt41_semantic_score=None,
        **{key: sum(row[key] for row in per_task) / len(tasks) for key in METRICS},
        all_cells_exact_proxy=sum(row["all_cells_exact_proxy"] for row in per_task) / len(tasks), per_task=per_task)


def scoring_reference_identity(tasks):
    return identity([[task.id, task.gold, task.metadata["evaluation"]] for task in tasks])


def score(args, tasks):
    configuration, saved, receipts, packages, runtime, precision = load_closed_run(args, tasks)
    reference_identity = scoring_reference_identity(tasks)
    baseline = None
    if args.adapter:
        training_precision = configuration["condition"].get("training", {}).get("precision")
        if training_precision is not None and training_precision != precision:
            raise ValueError("Adapter evaluation precision differs from its recorded training precision")
        if args.base_output_dir is None:
            raise ValueError("Adapter scoring requires an actual matched --base-output-dir")
        from copy import copy
        base_args = copy(args)
        base_args.output_dir, base_args.adapter = args.base_output_dir, None
        baseline = json.loads((args.base_output_dir / "report.json").read_text())
        # Base and adapter can have different shard counts; derive base coverage
        # from its actual config, never compare a re-declared imaginary base.
        base_args.shards = json.loads((args.base_output_dir / "config_rank0.json").read_text())["shards"]
        base_config, base_rows, _, base_packages, base_runtime, base_precision = load_closed_run(base_args, tasks)
        if (base_config["comparison"] != configuration["comparison"] or packages != base_packages
                or runtime != base_runtime or any(precision.get(key) != base_precision.get(key)
                    for key in ("base_parameter_dtypes", "quantization_report", "compute_profile", "compute_context"))
                or baseline.get("version") != VERSION or baseline.get("condition") != {"adapter": None, "policy": "base"}
                or baseline.get("comparison") != base_config["comparison"] or baseline.get("packages") != base_packages
                or baseline.get("runtime") != base_runtime or baseline.get("precision") != base_precision
                or baseline.get("generation_complete") is not True or baseline.get("scored_after_journals_closed") is not True
                or baseline.get("scoring_reference_identity") != reference_identity
                or baseline.get("prediction_coverage") != 1 or baseline.get("metric_task_denominator") != len(tasks)
                or baseline.get("tasks") != len(tasks) or baseline.get("widesearch") != score_predictions(tasks, base_rows)):
            raise ValueError("Base and adapter differ in model, precision, seeds, full harness, tasks or references")
    # Reference metrics are reached only after current AND matched-base shard
    # closure validation. A missing baseline never triggers partial scoring.
    metrics = score_predictions(tasks, saved)
    statuses = dict(Counter(row["status"] for row in saved))
    report = dict(version=VERSION, comparison=configuration["comparison"], condition=configuration["condition"],
        adapter_identity=configuration["adapter_identity"], packages=packages,
        scorer_packages=configuration["packages"], runtime=runtime, precision=precision,
        tasks=len(tasks), prediction_coverage=1., status_counts=statuses,
        generation_complete=True, scored_after_journals_closed=True, metric_task_denominator=len(tasks),
        scoring_reference_identity=reference_identity, widesearch=metrics,
        complete=not (statuses.get("infrastructure_error", 0) or statuses.get("context_budget_failure", 0)),
        all_tasks_succeeded=all(row["status"] == "completed" for row in saved), shard_receipts=receipts)
    if baseline is not None:
        report["matched_base"] = dict(output=str(args.base_output_dir),
            f1_by_row_delta=metrics["f1_by_row"] - baseline["widesearch"]["f1_by_row"],
            f1_by_item_delta=metrics["f1_by_item"] - baseline["widesearch"]["f1_by_item"],
            benchmark_improvement_claimed=False, official_semantic_score_claimed=False)
    write_json(args.output_dir / "report.json", report)
    return report


def parser():
    result = cuad_parser()
    result.description = __doc__
    # Native Web tasks need search/read/review/final turns. All values remain
    # explicit in the comparison manifest and must match base and adapter.
    result.set_defaults(max_steps=20)
    profile = next(action for action in result._actions if action.dest == "profile")
    profile.choices = ("raw", "thinking", "greedy")
    return result


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    if (not re.fullmatch(r"[0-9a-f]{40}", args.source_commit)
            or any(value < 1 for value in (args.shards, args.thought_tokens, args.json_tokens,
                   args.max_prompt_tokens, args.max_steps, args.retrieval_context_chars, args.retrieval_max_context_chars))
            or not 0 <= args.shard_index < args.shards or args.device_index < 0 or args.initial_chunks < 0
            or (args.web_text_window_chars is not None and args.web_text_window_chars < 1)
            or not math.isfinite(args.episode_timeout) or args.episode_timeout <= 0
            or args.split == "train"):
        arguments.error("Valid source/shard/budgets and held-out validation/test split required")
    validate_source(args.source_commit)
    tasks = load_public_tasks(args.data, split=args.split)
    if args.shards > len(tasks):
        arguments.error("At least one task per shard is required")
    if not args.score_only:
        asyncio.run(generate(args, tasks))
        if args.shards > 1 or args.generate_only:
            print(json.dumps(dict(generation_complete=True, scoring_complete=False, shard_index=args.shard_index,
                next_step="Run --score-only after every shard is complete")), flush=True)
            return 0
    # Reload original references only after all generation has finished locally.
    # score() validates all shard closures before calling any reference scorer.
    references = load_tasks(args.data, split=args.split)
    report = score(args, references)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
