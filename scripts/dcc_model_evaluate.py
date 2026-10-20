#!/usr/bin/env python3
"""Shard local GPU benchmark episodes and score persisted predictions.

Launch HF replicas with torchrun, or use explicit evaluation shards for
independent vLLM engines. Re-running resumes saved predictions; infrastructure
failures can be retried explicitly. Dataset labels are consulted only by the
separate scoring pass after policy episodes finish.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
import time
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "slime"))

from slime_plugins.dcc import EpisodeRunner, load_tasks
from slime_plugins.dcc.browser_reader import BrowserReader, validate_browser_reader_runtime
from slime_plugins.dcc.evaluation import evaluate_results
from slime_plugins.dcc.episode import EPISODE_PROMPT_VERSION, WIDESEARCH_EVIDENCE_GUARD_VERSION
from slime_plugins.dcc.episode_diagnostics import terminal_summary, token_accounting
from slime_plugins.dcc.hf_judge import HFBatchedJudge
from slime_plugins.dcc.hf_policy import PROMPT_ENCODING_VERSION, HFPolicy, load_policy_model
from slime_plugins.dcc.tools import WEB_READER_VERSION, WebTools
from slime_plugins.dcc.retrieval import load_retrieval_profile
from slime_plugins.dcc.widesearch_metrics import ALIGNMENT_PROTOCOL, extract_table
from slime_plugins.dcc.web_whitespace import (
    validate_web_whitespace_records, web_bounded_whitespace_configuration,
)
from slime_plugins.dcc.web_review import web_explicit_review_configuration
from slime_plugins.dcc.web_navigation import web_navigation_configuration, validate_web_navigation_records
from slime_plugins.dcc.web_rows import web_cached_rows_configuration, validate_web_cached_rows_records
from slime_plugins.dcc.web_research import web_research_configuration, validate_web_research_records
from slime_plugins.dcc.web_structured import web_structured_actions_configuration, web_structured_configuration


def _evaluation_sampling(args, backend):
    from slime_plugins.dcc.eval_sampling import EvaluationSampling

    defaults = EvaluationSampling().to_dict()
    value = EvaluationSampling(**{
        key: getattr(args, "eval_" + key, default) for key, default in defaults.items()
    })
    if backend != "vllm" and value != EvaluationSampling():
        raise ValueError("Nondefault --eval sampling requires --inference-backend vllm")
    return value


def _evaluation_shard(args, *, generation):
    index = getattr(args, "eval_shard_index", None)
    count = getattr(args, "eval_num_shards", None)
    if index is None and count is None:
        return None
    if index is None or count is None:
        raise ValueError("Provide both --eval-shard-index and --eval-num-shards")
    if count < 1 or not 0 <= index < count:
        raise ValueError("Evaluation shards require 0 <= shard index < a positive shard count")
    if (not generation or getattr(args, "inference_backend", "hf") != "vllm"
            or any(getattr(args, name, False) for name in ("score_only", "local_judge", "merge_judge"))):
        raise ValueError("Explicit evaluation shards require vLLM generation without scoring or judge flags")
    return index, count


def _check_generation_manifests(output, manifest):
    names = set()
    for path in output.glob("manifest_rank*.json"):
        names.add(path.name)
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Evaluation configuration changed; use a new output directory")
    return names


def _adapter_identity(adapter):
    from slime_plugins.dcc.adapter_identity import adapter_content_identity

    return adapter_content_identity(adapter)


def _saved_runtime(output):
    runtimes = [json.loads(path.read_text()) for path in sorted(output.glob("inference_runtime*.json"))]
    if runtimes and any(value != runtimes[0] for value in runtimes[1:]):
        raise ValueError("Inference runtime or adapter differs across evaluation shards")
    return runtimes[0] if runtimes else None


def _check_saved_adapter(adapter, runtime):
    if runtime is not None and "adapter" in runtime:
        previous = runtime["adapter"]
        current = _adapter_identity(adapter)
        if (previous is None) != (current is None) or (
            previous is not None and any(current.get(key) != value for key, value in previous.items())
        ):
            raise ValueError("Adapter checkpoint identity changed; use a new output directory")


def _score_predictions(output, tasks, *, require_ownership=False):
    manifests = {int(path.stem.removeprefix("manifest_rank")): json.loads(path.read_text())
                 for path in output.glob("manifest_rank*.json")}
    manifest = next(iter(manifests.values()), {})
    if any(value != manifest for value in manifests.values()):
        raise ValueError("Cannot score predictions with mixed generation manifests")
    saved, owners = {}, {}
    for file in sorted(output.glob("predictions_rank*.jsonl")):
        rank = int(file.stem.removeprefix("predictions_rank"))
        for line in file.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                task_id = record["task_id"]
                if task_id in owners and owners[task_id] != rank:
                    raise ValueError(f"Prediction {task_id} appears in multiple shards")
                if "native_thinking" in manifest and task_id in saved:
                    raise ValueError("Native predictions cannot contain retries or duplicate records")
                # Repeated records within one shard are explicit retries;
                # retain its latest outcome without hiding cross-shard overlap.
                owners[task_id], saved[task_id] = rank, record
    expected = {task.id for task in tasks}
    if len(expected) != len(tasks):
        raise ValueError("duplicate evaluation task IDs")
    if set(saved) != expected:
        raise ValueError(
            f"Coverage mismatch: {len(expected - set(saved))} missing, {len(set(saved) - expected)} extra predictions"
        )
    runtime = _saved_runtime(output)
    validate_web_navigation_records((manifest, runtime or {}))
    validate_web_research_records((manifest, runtime or {}))
    validate_web_cached_rows_records((manifest, runtime or {}))
    review_records = (manifest, runtime or {})
    validate_web_whitespace_records(review_records)
    review_values = [record.get("web_explicit_review", web_explicit_review_configuration(False))
                     for record in review_records]
    for value in review_values:
        if (not isinstance(value, dict) or type(value.get("enabled")) is not bool
                or value != web_explicit_review_configuration(value["enabled"])):
            raise ValueError("Web explicit review requires complete supported configuration")
    if review_values[0] != review_values[1]:
        raise ValueError("Web explicit review differs between manifest and runtime")
    if review_values[0]["enabled"] and any(
            record.get("web_structured_final") != web_structured_configuration(True)
            or record.get("web_structured_actions") != web_structured_actions_configuration(True)
            for record in review_records):
        raise ValueError("Web explicit review requires matching structured Web actions and final")
    browser_records = (manifest, runtime or {})
    if any("web_browser_reader" in record for record in browser_records):
        configurations = [validate_browser_reader_runtime(record.get("web_browser_reader"))
                          for record in browser_records]
        if configurations[0] != configurations[1]:
            raise ValueError("Web browser reader differs between manifest and inference runtime")
        if manifest.get("inference_backend") != "vllm" or (runtime or {}).get("backend") != "vllm":
            raise ValueError("Web browser reader requires the vLLM evaluation backend")
    if "web_structured_actions" in manifest or runtime is not None and "web_structured_actions" in runtime:
        action_records = (manifest, runtime or {})
        configurations = [value.get("web_structured_actions", web_structured_actions_configuration(False))
                          for value in action_records]
        for value in configurations:
            if (not isinstance(value, dict) or type(value.get("enabled")) is not bool
                    or value != web_structured_actions_configuration(value["enabled"])):
                raise ValueError("Structured Web actions must declare the complete supported configuration")
        if configurations[0] != configurations[1]:
            raise ValueError("Structured Web actions differ between manifest and inference runtime")
        if configurations[0]["enabled"] and (
                manifest.get("inference_backend") != "vllm" or (runtime or {}).get("backend") != "vllm"
                or any(value.get("web_structured_final") != web_structured_configuration(True)
                       for value in action_records)):
            raise ValueError("Structured Web actions require vLLM and matching structured final configuration")
    if "native_thinking" in manifest or runtime is not None and "native_thinking" in runtime:
        from slime_plugins.dcc.native_thinking_eval import validate_native_sidecars
        validate_native_sidecars(output, saved, manifest, runtime, owners, tasks)
    if require_ownership and not isinstance(manifest.get("model"), str):
        raise ValueError("Balanced judge scheduling requires recorded generation model provenance")
    logical_shards = "eval_num_shards" in manifest
    if logical_shards or require_ownership:
        output_names = {path.name for path in output.iterdir()}
        count = manifest.get("eval_num_shards", manifest.get("world_size"))
        if type(count) is not int or count < 1 or manifest.get("world_size") != count or set(manifests) != set(range(count)):
            raise ValueError("Logical shard manifests do not cover the configured world")
        statuses = {"completed", "invalid_answer", "max_steps", "evidence_failure", "infrastructure_error"}
        if "native_thinking" in manifest:
            statuses.add("model_protocol_failure")
        for rank in range(count):
            assigned = {task.id for task in tasks[rank::count]}
            if {task_id for task_id, owner in owners.items() if owner == rank} != assigned:
                raise ValueError("Prediction task ownership differs from its logical shard")
            marker = output / f"rank{rank}.done"
            if marker.name not in output_names or json.loads(marker.read_text()) != {"tasks": len(assigned), "world_size": count}:
                raise ValueError(f"Evaluation shard {rank} has no matching completion marker")
            runtime_path = output / (f"inference_runtime_rank{rank}.json" if count > 1 else "inference_runtime.json")
            if logical_shards and assigned and runtime_path.name not in output_names:
                raise ValueError(f"Evaluation shard {rank} has no inference runtime record")
        if any(record.get("status") not in statuses for record in saved.values()):
            raise ValueError("Predictions contain a missing or unknown episode status")
        if logical_shards and (runtime is None or runtime.get("backend") != "vllm" or runtime.get("model") != manifest["model"]):
            raise ValueError("Inference runtime does not identify the manifest policy")
        if "adapter_identity" in manifest and (runtime is None or runtime.get("adapter") != manifest["adapter_identity"]):
            raise ValueError("Inference runtime adapter differs from its generation manifest")
    return saved, manifest, runtime


def _judge_task_schedule(tasks, saved, world, strategy):
    """Assign scoring work from persisted answers without consulting labels."""
    if type(world) is not int or world < 1:
        raise ValueError("Judge scheduling requires a positive shard count")
    if strategy not in ("round_robin", "balanced"):
        raise ValueError("Unknown judge scheduling strategy")
    expected = {task.id for task in tasks}
    if len(expected) != len(tasks) or not expected <= saved.keys():
        raise ValueError("Judge scheduling requires unique tasks with saved predictions")
    if strategy == "round_robin":
        return [tasks[rank::world] for rank in range(world)], None

    def cost(task):
        answer = saved[task.id].get("answer")
        try:
            table = extract_table(answer)
        except (TypeError, ValueError):
            table = None
        if table is None or table.empty:
            return 128
        characters = len(answer) if isinstance(answer, str) else len(json.dumps(answer, ensure_ascii=False))
        return 128 * (1 + table.shape[0] * table.shape[1]) + characters

    costs = [cost(task) for task in tasks]
    capacities = [len(tasks) // world + (rank < len(tasks) % world) for rank in range(world)]
    assigned, loads = [[] for _ in range(world)], [0] * world
    for index in sorted(range(len(tasks)), key=lambda index: (-costs[index], index)):
        rank = min((rank for rank in range(world) if len(assigned[rank]) < capacities[rank]), key=lambda rank: (loads[rank], rank))
        assigned[rank].append(tasks[index])
        loads[rank] += costs[index]
    plan = {
        "version": "answer_lpt_count_cap_v1", "strategy": strategy, "world_size": world,
        "prediction_only": True, "reference_or_scores_used": False,
        "cost_formula": "128*(1+table_rows*table_columns)+answer_characters; empty/invalid/no table:128",
        "tie_breaking": "original task order for equal cost; lowest rank for equal load",
        "capacities": capacities, "estimated_cost_units_by_rank": loads,
        "task_ids_by_rank": [[task.id for task in shard] for shard in assigned],
    }
    return assigned, plan


def _check_judge_schedule(shards, tasks, saved, strategy):
    """Check the saved scoring assignment without changing generation ownership."""
    plans = [shard.get("judge_task_schedule") for _, shard in shards]
    if strategy == "round_robin" and all(plan is None for plan in plans):
        return None  # Legacy round-robin reports keep their existing behavior.
    if strategy != "balanced" or not plans or any(not isinstance(plan, dict) for plan in plans):
        raise ValueError("Judge scheduling differs across shards or merge settings")
    world = plans[0].get("world_size")
    assigned, expected_plan = _judge_task_schedule(tasks, saved, world, strategy)
    if len(shards) != world or {rank for rank, _ in shards} != set(range(world)):
        raise ValueError("Judge scheduling shards do not cover the configured world")
    for rank, shard in shards:
        if shard.get("judge_task_schedule") != expected_plan or type(shard.get("judge_shard_index")) is not int or shard["judge_shard_index"] != rank:
            raise ValueError("Judge scheduling differs across shards or saved predictions")
        actual = [record["task_id"] for record in shard["widesearch"]["per_task"]]
        if actual != [task.id for task in assigned[rank]]:
            raise ValueError("Judge task ownership differs from its scoring schedule")
    return expected_plan


async def _finish_evaluation(policy, stream, workers, failure):
    # Stop episodes before draining the policy or closing their output stream.
    # Some episodes may finish cancellation handlers and persist a result.
    for worker in workers:
        if not worker.done():
            worker.cancel()
    cleanup_errors = []
    if workers:
        try:
            await asyncio.gather(*workers, return_exceptions=True)
        except BaseException as error:
            cleanup_errors.append(error)
    if policy is not None:
        try:
            await policy.close()
        except BaseException as error:
            cleanup_errors.append(error)
    if stream is not None:
        try:
            stream.close()
        except BaseException as error:
            cleanup_errors.append(error)
    if cleanup_errors:
        primary = failure if failure is not None else cleanup_errors.pop(0)
        for error in cleanup_errors:
            primary.add_note(f"Evaluation cleanup also failed: {type(error).__name__}: {error}")
        if failure is None:
            raise primary


async def evaluate(args):
    enabled = getattr(args, "web_browser_reader", False)
    if type(enabled) is not bool:
        raise ValueError("--web-browser-reader must be a boolean")
    if enabled and getattr(args, "inference_backend", "hf") != "vllm":
        raise ValueError("--web-browser-reader requires --inference-backend vllm")
    browser_reader = BrowserReader(enabled=True) if enabled else None
    failure = None
    try:
        if browser_reader is not None:
            # Explicit CPU preflight occurs before publishing any generation
            # manifest or loading a GPU policy. The process shares one reader.
            validate_browser_reader_runtime(await browser_reader.start())
        marker, completion = await _evaluate(args, browser_reader=browser_reader)
    except BaseException as error:
        failure = error
        raise
    finally:
        if browser_reader is not None:
            try:
                await browser_reader.close()
            except BaseException as error:
                if failure is None:
                    raise
                failure.add_note(f"Browser cleanup also failed: {type(error).__name__}: {error}")
    # Cleanup failures must not certify the shard as successfully completed.
    marker.write_text(json.dumps(completion))


async def _evaluate(args, *, browser_reader=None):
    import torch

    if getattr(args, "judge_scheduling", "round_robin") != "round_robin":
        raise ValueError("Balanced judge scheduling requires --score-only with --local-judge or --merge-judge")
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    backend = getattr(args, "inference_backend", "hf")
    quantization = getattr(args, "quantization", "none")
    sampling = _evaluation_sampling(args, backend)
    structured_final = getattr(args, "web_structured_final", False)
    structured = web_structured_configuration(structured_final)
    structured_actions = getattr(args, "web_structured_actions", False)
    actions = web_structured_actions_configuration(structured_actions)
    navigation_ledger = getattr(args, "web_navigation_ledger", False)
    navigation = web_navigation_configuration(navigation_ledger)
    if navigation_ledger and not (structured_final and structured_actions and backend == "vllm"):
        raise ValueError("--web-navigation-ledger requires --web-structured-final, --web-structured-actions and vLLM")
    research_checkpoints = getattr(args, "web_research_checkpoints", False)
    research = web_research_configuration(research_checkpoints)
    if research_checkpoints and not (structured_final and structured_actions and backend == "vllm"):
        raise ValueError("--web-research-checkpoints requires --web-structured-final, --web-structured-actions and vLLM")
    cached_rows_enabled = getattr(args, "web_cached_rows", False)
    cached_rows = web_cached_rows_configuration(cached_rows_enabled)
    if cached_rows_enabled and not (structured_final and structured_actions and backend == "vllm"):
        raise ValueError("--web-cached-rows requires --web-structured-final, --web-structured-actions and vLLM")
    explicit_review = getattr(args, "web_explicit_review", False)
    review = web_explicit_review_configuration(explicit_review)
    bounded_whitespace = getattr(args, "web_bounded_whitespace", False)
    whitespace = web_bounded_whitespace_configuration(bounded_whitespace)
    if bounded_whitespace and not (structured_final and backend == "vllm"):
        raise ValueError("--web-bounded-whitespace requires --web-structured-final and vLLM")
    if explicit_review and not (structured_final and structured_actions and backend == "vllm"):
        raise ValueError("--web-explicit-review requires --web-structured-final, --web-structured-actions and vLLM")
    if structured_actions and not structured_final:
        raise ValueError("--web-structured-actions requires --web-structured-final")
    if structured_actions and backend != "vllm":
        raise ValueError("--web-structured-actions requires --inference-backend vllm")
    if structured_final and backend != "vllm":
        raise ValueError("--web-structured-final requires --inference-backend vllm")
    if getattr(args, "vllm_async", False) and backend != "vllm":
        raise ValueError("--vllm-async requires --inference-backend vllm")
    if backend == "vllm" and (world != 1 or local != 0 or rank != 0):
        raise ValueError("Use one evaluation process for vLLM; configure tensor parallelism instead of torchrun replicas")
    if backend == "vllm" and quantization not in {"none", "fp8"}:
        raise ValueError("The vLLM evaluation backend supports BF16 or explicit FP8 base weights")
    if backend != "vllm" and quantization == "fp8":
        raise ValueError("FP8 generation requires --inference-backend vllm")
    shard = _evaluation_shard(args, generation=True)
    if shard is not None:
        rank, world = shard
    torch.manual_seed(args.seed + rank)
    tasks = [task for path in args.data for task in load_tasks(path)]
    if any(task.split == "train" for task in tasks):
        raise ValueError("Use held-out validation or test tasks for evaluation")
    if args.limit:
        tasks = tasks[: args.limit]
    native = None
    if (getattr(args, "native_thinking", False) or getattr(args, "native_source_receipt", None) is not None
            or getattr(args, "native_lora_derivative_id", None) is not None
            or getattr(args, "native_thought_tokens", 4096) != 4096
            or getattr(args, "native_thought_sampling", "greedy") != "greedy"
            or getattr(args, "native_lora_training_source", None) is not None
            or getattr(args, "native_lora_training_protocol", None) is not None):
        from slime_plugins.dcc.native_thinking_eval import native_options
        native = native_options(args, tasks, sampling, Path(__file__).resolve().parents[1])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    retrieval_profile = load_retrieval_profile(getattr(args, "retrieval_profile", None))
    retrieval_context_chars = getattr(args, "retrieval_context_chars", 0)
    retrieval_max_context_chars = getattr(args, "retrieval_max_context_chars", 16000)
    if type(retrieval_context_chars) is not int or retrieval_context_chars < 0:
        raise ValueError("retrieval_context_chars must be a nonnegative integer")
    if type(retrieval_max_context_chars) is not int or retrieval_max_context_chars < 1:
        raise ValueError("retrieval_max_context_chars must be a positive integer")
    manifest = {
        key: getattr(args, key)
        for key in (
            "model",
            "adapter",
            "data",
            "max_steps",
            "max_new_tokens",
            "max_prompt_tokens",
            "initial_chunks",
            "seed",
            "limit",
        )
    }
    manifest["world_size"] = world
    if native is not None:
        manifest["native_thinking"] = native
        if args.adapter is not None:
            manifest["native_lora_derivative_id"] = args.native_lora_derivative_id
            if getattr(args, "native_lora_training_source", None) is not None:
                manifest["native_lora_training_approval"] = {
                    "source_commit": args.native_lora_training_source,
                    "protocol": args.native_lora_training_protocol}
        occupied = {path.name for path in output.iterdir()}
        reserved = {f"manifest_rank{rank}.json", f"predictions_rank{rank}.jsonl", f"rank{rank}.done",
                    f"native_thinking_traces_rank{rank}.jsonl", f"inference_runtime_rank{rank}.json"}
        if world == 1:
            reserved.add("inference_runtime.json")
        if occupied & reserved:
            raise ValueError("Native generation is fresh-only; use a new output directory")
    browser_runtime = None if browser_reader is None else validate_browser_reader_runtime(browser_reader.runtime)
    if browser_runtime is not None:
        manifest["web_browser_reader"] = browser_runtime
    if structured_final:
        manifest["web_structured_final"] = structured
    if structured_actions:
        manifest["web_structured_actions"] = actions
    if explicit_review:
        manifest["web_explicit_review"] = review
    if bounded_whitespace:
        manifest["web_bounded_whitespace"] = whitespace
    if navigation_ledger:
        manifest["web_navigation_ledger"] = navigation
    if research_checkpoints:
        manifest["web_research_checkpoints"] = research
    if cached_rows_enabled:
        manifest["web_cached_rows"] = cached_rows
    manifest["episode_prompt_version"] = EPISODE_PROMPT_VERSION
    manifest["prompt_encoding_version"] = PROMPT_ENCODING_VERSION
    manifest["timeout_seconds"] = getattr(args, "timeout", 1800)
    if shard is not None:
        manifest["eval_num_shards"] = world
        manifest["adapter_identity"] = _adapter_identity(args.adapter)
    if retrieval_profile is not None:
        # Profiles contain tuples in memory, while JSON persists them as lists.
        # Compare the same representation when resuming an unchanged profile.
        manifest["retrieval_profile"] = json.loads(
            json.dumps(dataclasses.asdict(retrieval_profile))
        )
    if retrieval_context_chars or retrieval_max_context_chars != 16000:
        manifest["retrieval_context"] = {
            "context_chars": retrieval_context_chars,
            "max_context_chars": retrieval_max_context_chars,
        }
    if quantization != "none":
        # Preserve existing BF16 manifests and reject changing precision when
        # resuming a quantized run. Quantized scores need a separate directory.
        manifest["quantization"] = quantization
    if quantization == "fp8":
        from slime_plugins.dcc.fp8_attention_guard import QUANTIZATION, VALIDATION_LAYOUT
        from slime_plugins.dcc.fp8_runtime import runtime_identity
        manifest.update(fp8_contract=dict(QUANTIZATION), fp8_validation=dict(VALIDATION_LAYOUT),
                        runtime_code_identity=runtime_identity())
    if backend != "hf":
        manifest["inference_backend"] = backend
        manifest["evaluation_sampling"] = sampling.to_dict()
        manifest["inference_parameters"] = {
            "tensor_parallel_size": args.vllm_tensor_parallel_size,
            "gpu_memory_utilization": args.vllm_memory_utilization,
            "max_num_batched_tokens": args.vllm_batched_tokens,
            "max_model_len": args.max_prompt_tokens + args.max_new_tokens,
            "enforce_eager": args.vllm_enforce_eager,
            "batch_size": args.batch_size,
        }
        if getattr(args, "vllm_async", False):
            manifest["inference_parameters"]["async_engine"] = True
    if any(task.benchmark == "widesearch" for task in tasks):
        manifest["search_backend"] = WebTools().backend
        manifest["search_endpoint"] = os.environ.get("DCC_SEARCH_URL")
        manifest["web_timeout_seconds"] = float(os.environ.get("DCC_WEB_TIMEOUT", "30"))
        manifest["evidence_guard_version"] = WIDESEARCH_EVIDENCE_GUARD_VERSION
        manifest["web_reader_version"] = WEB_READER_VERSION
        auto_read_sources = getattr(args, "auto_read_sources", 0)
        if auto_read_sources:
            manifest["auto_read_sources"] = auto_read_sources
    manifest_path = output / f"manifest_rank{rank}.json"
    if manifest_path.name not in _check_generation_manifests(output, manifest):
        # Other independently launched shards must never read partial JSON.
        temporary = output / f".manifest_rank{rank}.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(manifest_path)
    _check_generation_manifests(output, manifest)
    # Directory enumeration avoids slow negative metadata lookups on some
    # remote filesystems. Only this rank's pre-existing files control resume.
    output_names = {path.name for path in output.iterdir()}
    file = output / f"predictions_rank{rank}.jsonl"
    done = {}
    if file.name in output_names:
        for line in file.read_text().splitlines():
            if line.strip():
                value = json.loads(line)
                done[value["task_id"]] = value
    assigned = tasks[rank::world]
    todo = [
        task
        for task in assigned
        if task.id not in done
        or (
            args.retry_infrastructure
            and done[task.id]["status"] == "infrastructure_error"
        )
    ]
    if todo:
        # A previous successful attempt must not certify a failed retry.
        marker = output / f"rank{rank}.done"
        if marker.name in output_names:
            marker.unlink(missing_ok=True)
    if backend == "vllm":
        previous_runtime = _saved_runtime(output)
        if previous_runtime is not None:
            validate_web_research_records((manifest, previous_runtime))
            validate_web_cached_rows_records((manifest, previous_runtime))
        _check_saved_adapter(args.adapter, previous_runtime)
        if previous_runtime is not None and previous_runtime.get("web_browser_reader") != browser_runtime:
            raise ValueError("Web browser reader changed; use a new output directory")
    if todo:
        policy, stream, workers, failure = None, None, [], None
        try:
            if backend == "vllm":
                from slime_plugins.dcc.vllm_policy import load_vllm_policy

                async_options = {"asynchronous": True} if getattr(args, "vllm_async", False) else {}
                if structured_final:
                    async_options["web_structured_final"] = True
                if structured_actions:
                    async_options["web_structured_actions"] = True
                if explicit_review:
                    async_options["web_explicit_review"] = True
                if navigation_ledger:
                    async_options["web_navigation_ledger"] = True
                if research_checkpoints:
                    async_options["web_research_checkpoints"] = True
                if cached_rows_enabled:
                    async_options["web_cached_rows"] = True
                if bounded_whitespace:
                    async_options["web_bounded_whitespace"] = True
                if quantization == "fp8":
                    async_options.update(quantization="fp8",
                        fp8_validation_dir=str((output / f"fp8_validation_rank{rank}").absolute()))
                policy = load_vllm_policy(
                    args.model, adapter=args.adapter, batch_size=args.batch_size,
                    max_prompt_tokens=args.max_prompt_tokens,
                    max_model_len=args.max_prompt_tokens + args.max_new_tokens,
                    tensor_parallel_size=args.vllm_tensor_parallel_size,
                    gpu_memory_utilization=args.vllm_memory_utilization,
                    max_num_batched_tokens=args.vllm_batched_tokens,
                    enforce_eager=args.vllm_enforce_eager, seed=args.seed,
                    evaluation_sampling=sampling, **async_options,
                )
                if quantization == "fp8":
                    from slime_plugins.dcc.fp8_attention_guard import validate_worker_receipts
                    validate_worker_receipts(output / f"fp8_validation_rank{rank}", adapter=args.adapter)
                if native is not None:
                    from slime_plugins.dcc.native_thinking_eval import NativeThinkingPolicy, NativeTraceJournal, budget_from_configuration
                    journal = NativeTraceJournal(output / f"native_thinking_traces_rank{rank}.jsonl")
                    try:
                        policy = NativeThinkingPolicy(policy, source_receipt=native["source_receipt"], trace_sink=journal,
                            approved_derivative_id=getattr(args, "native_lora_derivative_id", None),
                            budget=budget_from_configuration(native),
                            approved_training_source=getattr(args, "native_lora_training_source", None),
                            approved_training_protocol=getattr(args, "native_lora_training_protocol", None))
                    except BaseException:
                        journal.close()
                        raise
                runtime_file = f"inference_runtime_rank{rank}.json" if world > 1 else "inference_runtime.json"
                provenance = dict(policy.provenance)
                if browser_runtime is not None:
                    provenance["web_browser_reader"] = browser_runtime
                identity = _adapter_identity(args.adapter)
                if shard is not None:
                    if identity != manifest["adapter_identity"]:
                        raise ValueError("Adapter checkpoint changed while loading the policy")
                provenance["adapter"] = identity
                validate_web_research_records((manifest, provenance))
                validate_web_cached_rows_records((manifest, provenance))
                if previous_runtime is not None and provenance != previous_runtime:
                    raise ValueError("Inference runtime or adapter changed; use a new output directory")
                # Publish complete JSON before another worker audits it.
                temporary = output / f".{runtime_file}.{os.getpid()}.tmp"
                temporary.write_text(json.dumps(provenance, indent=2) + "\n")
                temporary.replace(output / runtime_file)
                _saved_runtime(output)
            else:
                model, tokenizer = load_policy_model(
                    args.model, adapter=args.adapter, rank=local, quantization=quantization
                )
                policy = HFPolicy(
                    model, tokenizer, batch_size=args.batch_size,
                    max_prompt_tokens=args.max_prompt_tokens,
                )
                if hasattr(model, "dcc_quantization_report"):
                    (output / f"quantization_rank{rank}.json").write_text(
                        json.dumps(model.dcc_quantization_report, indent=2) + "\n"
                    )
            runner = EpisodeRunner(
                max_steps=args.max_steps,
                max_new_tokens=args.max_new_tokens,
                initial_chunks=args.initial_chunks,
                timeout_seconds=args.timeout,
                retrieval_profile=retrieval_profile,
                auto_read_sources=getattr(args, "auto_read_sources", 0),
                retrieval_context_chars=retrieval_context_chars,
                retrieval_max_context_chars=retrieval_max_context_chars,
                evaluation_temperature=sampling.temperature,
                web_structured_final=structured_final,
                web_structured_actions=structured_actions,
                **({"cuad_structured_actions": True} if native is not None else {}),
                **({"web_explicit_review": True} if explicit_review else {}),
                **({"web_navigation_ledger": True} if navigation_ledger else {}),
                **({"web_research_checkpoints": True} if research_checkpoints else {}),
                **({"web_cached_rows": True} if cached_rows_enabled else {}),
                **({"web_tools_factory": lambda: WebTools(browser_reader=browser_reader)}
                   if browser_reader is not None else {}),
            )
            semaphore = asyncio.Semaphore(args.batch_size * 2)
            stream = file.open("a", buffering=1)
            start = time.monotonic()

            async def one(task):
                async with semaphore:
                    episode_start = time.monotonic()
                    episode_policy = policy.bind_task(task.id, rank) if native is not None else policy
                    result = await runner.run(task, episode_policy, evaluation=True)
                    value = result.to_dict(include_tokens=args.save_tokens)
                    if native is not None:
                        value["native_thinking_traces"] = episode_policy.references
                    value["episode_latency_seconds"] = time.monotonic() - episode_start
                    token_usage = {
                        "prompt": sum(len(step.prompt_ids) for step in result.steps),
                        "completion": sum(len(step.completion_ids) for step in result.steps),
                    }
                    value["token_usage"] = token_usage
                    value["token_accounting"] = token_accounting(
                        result, native_references=episode_policy.references if native is not None else None,
                    )
                    value["terminal_summary"] = terminal_summary(result, max_steps=args.max_steps)
                    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
                    print(
                        json.dumps(
                            {
                                "rank": rank,
                                "task": task.id,
                                "status": result.status,
                                "turns": len(result.steps),
                                "token_usage": token_usage,
                                "token_accounting": value["token_accounting"],
                                "terminal_summary": value["terminal_summary"],
                                "episode_latency_seconds": value["episode_latency_seconds"],
                                "elapsed": time.monotonic() - start,
                            }
                        ),
                        flush=True,
                    )

            workers = [asyncio.create_task(one(task)) for task in todo]
            await asyncio.gather(*workers)
        except BaseException as error:
            failure = error
            raise
        finally:
            await _finish_evaluation(policy, stream, workers, failure)
    return output / f"rank{rank}.done", {"tasks": len(assigned), "world_size": world}


def _prediction_setting(output, key, default):
    """Read execution provenance from saved generation, not scoring flags."""
    values = {
        json.loads(path.read_text()).get(key, default)
        for path in output.glob("manifest_rank*.json")
    }
    if len(values) > 1:
        raise ValueError(f"Cannot score predictions with mixed policy {key}")
    return next(iter(values), "unrecorded")


def _prediction_quantization(output):
    return _prediction_setting(output, "quantization", "none")


def _judge_configuration(args):
    quantization = getattr(args, "quantization", "none")
    packages = ["torch", "transformers", "peft", "pycountry"]
    if quantization == "nf4":
        packages.append("bitsandbytes")
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return {
        "model": args.model, "model_name": args.judge_name, "adapter": None,
        "quantization": quantization, "context": args.judge_context,
        "max_new_tokens": args.judge_tokens, "batch_size": args.judge_batch_size,
        "batch_tokens": args.judge_batch_tokens, "workers": args.judge_workers,
        "allow_truncation": False, "runtime_versions": versions,
        "alignment_protocol": ALIGNMENT_PROTOCOL,
    }


def _judge_alignment_protocol(shard):
    """Audit explicit prompt versions without assigning a version to legacy data."""
    wide = shard["widesearch"]
    configuration = shard.get("judge_configuration")
    if configuration is not None and not isinstance(configuration, dict):
        raise ValueError("Judge configuration must be an object")
    sources = [wide, configuration or {}] + wide["per_task"]
    declared = [value["alignment_protocol"] for value in sources if value.get("alignment_protocol") is not None]
    if any(not isinstance(value, str) or not value for value in declared):
        raise ValueError("Judge alignment protocol must be an explicit nonempty version")
    versions = set(declared)
    if len(versions) > 1:
        raise ValueError("Judge alignment protocols differ within a scoring shard")
    return versions, len(declared) == len(sources)


def _table_input_protocol(report):
    """Read CSV adapter versions without upgrading historical score reports."""
    wide = report.get("widesearch", {})
    sources = [wide] + wide.get("per_task", [])
    declared = [value["table_input_protocol"] for value in sources if value.get("table_input_protocol") is not None]
    if any(not isinstance(value, str) or not value for value in declared):
        raise ValueError("Table input protocol must be an explicit nonempty version")
    versions = set(declared)
    if len(versions) > 1:
        raise ValueError("Table input protocols differ within a scoring report")
    return versions, len(declared) == len(sources)


def _check_judge_generation(report, manifest, runtime):
    for key, value in (("generation_manifest", manifest), ("inference_runtime", runtime)):
        if key in report and report[key] != value:
            raise ValueError("Judge report generation provenance differs from saved predictions")


def score(args):
    _evaluation_shard(args, generation=False)
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    tasks = [task for path in args.data for task in load_tasks(path)]
    if args.limit:
        tasks = tasks[: args.limit]
    output = Path(args.output_dir)
    scheduling = getattr(args, "judge_scheduling", "round_robin")
    if scheduling == "balanced" and not (getattr(args, "local_judge", False) or args.merge_judge):
        raise ValueError("Balanced judge scheduling requires --local-judge or --merge-judge")
    saved, generation_manifest, runtime = _score_predictions(output, tasks, require_ownership=scheduling == "balanced")
    policy_quantization = _prediction_quantization(output)
    policy_backend = _prediction_setting(output, "inference_backend", "hf")
    if args.merge_judge:
        report = json.loads((output / "scores_strict.json").read_text())
        _check_judge_generation(report, generation_manifest, runtime)
        records, judge_shards = [], []
        configurations, shard_count = [], 0
        alignment_versions, alignment_legacy_shards = set(), 0
        table_versions, table_strict_complete = _table_input_protocol(report)
        table_legacy_shards = 0
        judge_quantization = getattr(args, "quantization", "none")
        for path in sorted(output.glob("scores_judge_rank*.json")):
            shard = json.loads(path.read_text())
            judge_shards.append((int(path.stem.removeprefix("scores_judge_rank")), shard))
            shard_count += 1
            _check_judge_generation(shard, generation_manifest, runtime)
            if shard.get("judge_quantization", "none") != judge_quantization:
                raise ValueError("Judge quantization differs across scoring shards or merge settings")
            if shard.get("policy_quantization", policy_quantization) != policy_quantization:
                raise ValueError("Judge report policy quantization differs from prediction manifests")
            if shard.get("policy_inference_backend", policy_backend) != policy_backend:
                raise ValueError("Judge report policy backend differs from prediction manifests")
            if shard["widesearch"].get("judge_model", args.judge_name) != args.judge_name:
                raise ValueError("Judge model differs across scoring shards or merge settings")
            versions, alignment_complete = _judge_alignment_protocol(shard)
            alignment_versions.update(versions)
            alignment_legacy_shards += not alignment_complete
            if len(alignment_versions) > 1:
                raise ValueError("Judge alignment protocols differ across scoring shards")
            versions, table_complete = _table_input_protocol(shard)
            table_versions.update(versions)
            table_legacy_shards += not table_complete
            if len(table_versions) > 1:
                raise ValueError("Table input protocols differ across strict and judge scoring reports")
            configuration = shard.get("judge_configuration")
            if configuration is not None:
                if (configuration.get("model_name") != args.judge_name
                        or configuration.get("quantization") != judge_quantization
                        or configuration.get("adapter") is not None):
                    raise ValueError("Judge configuration differs from the frozen merge judge")
                comparable = {key: value for key, value in configuration.items() if key != "alignment_protocol"}
                previous = {key: value for key, value in configurations[0].items() if key != "alignment_protocol"} if configurations else None
                if previous is not None and comparable != previous:
                    raise ValueError("Judge configurations differ across scoring shards")
                configurations.append(configuration)
            records.extend(shard["widesearch"]["per_task"])
        expected_wide = {task.id for task in tasks if task.benchmark == "widesearch"}
        if (
            len(records) != len(expected_wide)
            or {value["task_id"] for value in records} != expected_wide
        ):
            raise ValueError(
                "Sharded judge reports must cover every WideSearch task exactly once"
            )
        schedule = _check_judge_schedule(judge_shards, [task for task in tasks if task.benchmark == "widesearch"], saved, scheduling)
        good = [value for value in records if "judge_error" not in value]
        keys = (
            "score",
            "precision_by_row",
            "recall_by_row",
            "f1_by_row",
            "precision_by_item",
            "recall_by_item",
            "f1_by_item",
        )
        search_coverage = report["widesearch"].get("search_coverage")
        report["widesearch"] = {
            "tasks": len(records),
            "metric_denominator": len(good),
            "judge_failures": len(records) - len(good),
            "evaluation_protocol": "official_protocol_custom_judge",
            "alignment_protocol": next(iter(alignment_versions)) if alignment_versions and not alignment_legacy_shards else None,
            "alignment_protocol_complete": alignment_legacy_shards == 0,
            "alignment_protocols_observed": sorted(alignment_versions),
            "alignment_protocol_legacy_shards": alignment_legacy_shards,
            "table_input_protocol": next(iter(table_versions)) if table_versions and table_strict_complete and not table_legacy_shards else None,
            "table_input_protocol_complete": table_strict_complete and table_legacy_shards == 0,
            "table_input_protocols_observed": sorted(table_versions),
            "table_input_protocol_legacy_shards": table_legacy_shards,
            "table_input_protocol_strict_complete": table_strict_complete,
            "judge_model": args.judge_name,
            **{
                key: sum(value[key] for value in good) / len(good) if good else None
                for key in keys
            },
            "per_task": records,
        }
        if search_coverage is not None:
            report["widesearch"]["search_coverage"] = search_coverage
        report["complete"] = report["complete"] and len(good) == len(records)
        report["policy_quantization"] = policy_quantization
        report["policy_inference_backend"] = policy_backend
        report["judge_quantization"] = judge_quantization
        # Legacy shards remain mergeable, with their unavailable configuration
        # evidence explicit rather than inferred from current CLI defaults.
        report["judge_configuration_complete"] = len(configurations) == shard_count and alignment_legacy_shards == 0
        if configurations:
            report["judge_configuration"] = dict(configurations[0])
            report["judge_configuration"]["alignment_protocol"] = report["widesearch"]["alignment_protocol"]
        if schedule is not None:
            report["judge_task_schedule"] = schedule
        (output / "scores_local_judge.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        print(
            json.dumps(
                {
                    key: value
                    for key, value in report["widesearch"].items()
                    if key != "per_task"
                },
                indent=2,
            )
        )
        return
    judge, schedule = None, None
    if args.local_judge:
        # Frozen base judge, after all policy predictions are persisted.
        assignments, schedule = _judge_task_schedule([task for task in tasks if task.benchmark == "widesearch"], saved, world, scheduling)
        if not 0 <= rank < world:
            raise ValueError("Judge rank is outside the scoring world")
        tasks = assignments[rank]
        saved = {task.id: saved[task.id] for task in tasks}
        model, tokenizer = load_policy_model(
            args.model, adapter=None, trainable=False, rank=local,
            quantization=getattr(args, "quantization", "none")
        )
        policy = HFPolicy(
            model,
            tokenizer,
            max_prompt_tokens=args.judge_context,
            allow_truncation=False,
        )

        judge = HFBatchedJudge(
            policy,
            tokenizer,
            model=args.judge_name,
            max_new_tokens=args.judge_tokens,
            max_batch=args.judge_batch_size,
            max_batch_tokens=args.judge_batch_tokens,
        )
    try:
        report = evaluate_results(
            tasks,
            saved.values(),
            judge=judge,
            workers=args.judge_workers if judge else 1,
        )
    finally:
        if judge:
            judge.close()
    report["generation"] = {
        key: generation_manifest.get(key, getattr(args, key))
        for key in (
            "max_steps",
            "max_new_tokens",
            "max_prompt_tokens",
            "initial_chunks",
            "seed",
        )
    }
    report["model"] = args.model_name
    report["generation_manifest"] = generation_manifest
    if runtime is not None:
        report["inference_runtime"] = runtime
    report["policy_quantization"] = policy_quantization
    report["policy_inference_backend"] = policy_backend
    if judge:
        report["judge_quantization"] = getattr(args, "quantization", "none")
        report["judge_configuration"] = _judge_configuration(args)
        if schedule is not None:
            report["judge_task_schedule"] = schedule
            report["judge_shard_index"] = rank
    (
        output / (f"scores_judge_rank{rank}.json" if judge else "scores_strict.json")
    ).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                key: (
                    {k: v for k, v in value.items() if k != "per_task"}
                    if isinstance(value, dict)
                    else value
                )
                for key, value in report.items()
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--adapter")
    parser.add_argument("--native-lora-derivative-id",
                        help="Caller-approved actual derivative ID; required with native LoRA evaluation")
    parser.add_argument("--quantization", choices=("none", "nf4", "fp8"), default="none")
    parser.add_argument("--retrieval-profile")
    parser.add_argument("--retrieval-context-chars", type=int, default=0)
    parser.add_argument("--retrieval-max-context-chars", type=int, default=16000)
    parser.add_argument("--auto-read-sources", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--inference-backend", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-memory-utilization", type=float, default=0.88)
    parser.add_argument("--vllm-batched-tokens", type=int, default=8192)
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--vllm-async", action="store_true",
                        help="Use native per-request vLLM scheduling and abort timed-out GPU requests")
    parser.add_argument("--native-thinking", action="store_true",
                        help="Opt into native Thinking2507 two-stage CUAD/Web evaluation:4096 thought+4096 JSON;greedy/neutral;fresh output only")
    parser.add_argument("--native-thought-tokens", type=int, choices=(4096, 8192, 16384), default=4096,
                        help="Explicit native thought budget; JSON remains 4096; nondefault opts into v3")
    parser.add_argument("--native-thought-sampling", choices=("greedy", "thinking"), default="greedy",
                        help="Thought only: thinking uses temperature=.6,top_p=.95,top_k=20; JSON stays neutral greedy")
    parser.add_argument("--native-lora-training-source",
                        help="Caller-approved actual 40-character native adapter training source, independently from evaluation")
    parser.add_argument("--native-lora-training-protocol",
                        help="Caller-approved actual adapter training protocol: native v2 or explicit budgeted v3")
    parser.add_argument("--native-source-receipt",
                        help="New frozen source root .dcc-native-source.json receipt;requires --native-thinking")
    parser.add_argument("--web-structured-final", action="store_true",
                        help="Constrain only final-budget Web evaluation JSON to public columns and actually read URLs (vLLM/xgrammar)")
    parser.add_argument("--web-bounded-whitespace", action="store_true",
                        help="Bound insignificant JSON whitespace to 8 characters between tokens (structured Web/vLLM only; opt-in)")
    parser.add_argument("--web-navigation-ledger", action="store_true",
                        help="Add bounded navigation/read-window advice from observed Web metadata (requires structured actions/final and vLLM)")
    parser.add_argument("--web-research-checkpoints", action="store_true",
                        help="Add request-only public-column and remaining-turn research guidance (requires structured actions/final and vLLM)")
    parser.add_argument("--web-cached-rows", action="store_true",
                        help="Enable bounded literal-record selection over already downloaded sources (requires structured actions/final and vLLM)")
    parser.add_argument("--web-explicit-review", action="store_true",
                        help="Add bounded public-task plan/checks before every structured Web action (requires structured actions/final and vLLM)")
    parser.add_argument("--web-structured-actions", action="store_true",
                        help="Constrain every Web evaluation turn to one JSON action and require a successful search first (requires --web-structured-final and vLLM/xgrammar)")
    parser.add_argument("--web-browser-reader", action="store_true",
                        help="Optionally render successful public HTML on CPU with a shared bounded browser (vLLM evaluation only)")
    parser.add_argument("--eval-shard-index", type=int, help="Zero-based data shard for an independent vLLM engine")
    parser.add_argument("--eval-num-shards", type=int, help="Total independent vLLM evaluation data shards")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    parser.add_argument("--initial-chunks", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-temperature", type=float, default=0.0,
                        help="Explicit vLLM evaluation sampling; training rollouts are unchanged")
    parser.add_argument("--eval-top-p", type=float, default=1.0)
    parser.add_argument("--eval-top-k", type=int, default=-1)
    parser.add_argument("--eval-min-p", type=float, default=0.0)
    parser.add_argument("--eval-presence-penalty", type=float, default=0.0)
    parser.add_argument("--eval-frequency-penalty", type=float, default=0.0)
    parser.add_argument("--eval-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--save-tokens", action="store_true")
    parser.add_argument("--retry-infrastructure", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--local-judge", action="store_true")
    parser.add_argument("--merge-judge", action="store_true")
    parser.add_argument("--judge-name", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--judge-tokens", type=int, default=4096)
    parser.add_argument("--judge-context", type=int, default=65536)
    parser.add_argument("--judge-workers", type=int, default=4)
    parser.add_argument("--judge-scheduling", choices=("round_robin", "balanced"), default="round_robin",
                        help="Optional answer-only LPT scheduling for score-only judge replicas; keep generation shards unchanged")
    parser.add_argument("--judge-batch-size", type=int, default=4)
    parser.add_argument("--judge-batch-tokens", type=int, default=32768)
    args = parser.parse_args()
    if args.score_only:
        score(args)
    else:
        asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
