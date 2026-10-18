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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "slime"))

from slime_plugins.dcc import EpisodeRunner, load_tasks
from slime_plugins.dcc.evaluation import evaluate_results
from slime_plugins.dcc.episode import WIDESEARCH_EVIDENCE_GUARD_VERSION
from slime_plugins.dcc.hf_judge import HFBatchedJudge
from slime_plugins.dcc.hf_policy import HFPolicy, load_policy_model
from slime_plugins.dcc.tools import WebTools
from slime_plugins.dcc.retrieval import load_retrieval_profile


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
    for path in output.glob("manifest_rank*.json"):
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Evaluation configuration changed; use a new output directory")


def _adapter_identity(adapter):
    if adapter is None:
        return None
    path = Path(adapter).absolute()
    weights = path / "adapter_model.safetensors"
    stat = weights.stat()
    return {
        "path": str(path), "config": json.loads((path / "adapter_config.json").read_text()),
        "weights_file": weights.name, "weights_bytes": stat.st_size,
        "weights_mtime_ns": stat.st_mtime_ns, "weights_ctime_ns": stat.st_ctime_ns,
        "weights_inode": stat.st_ino, "weights_device": stat.st_dev,
    }


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


def _score_predictions(output, tasks):
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
    if "eval_num_shards" in manifest:
        count = manifest["eval_num_shards"]
        if type(count) is not int or count < 1 or manifest.get("world_size") != count or set(manifests) != set(range(count)):
            raise ValueError("Logical shard manifests do not cover the configured world")
        statuses = {"completed", "invalid_answer", "max_steps", "evidence_failure", "infrastructure_error"}
        for rank in range(count):
            assigned = {task.id for task in tasks[rank::count]}
            if {task_id for task_id, owner in owners.items() if owner == rank} != assigned:
                raise ValueError("Prediction task ownership differs from its logical shard")
            marker = output / f"rank{rank}.done"
            if not marker.exists() or json.loads(marker.read_text()) != {"tasks": len(assigned), "world_size": count}:
                raise ValueError(f"Evaluation shard {rank} has no matching completion marker")
            runtime_path = output / (f"inference_runtime_rank{rank}.json" if count > 1 else "inference_runtime.json")
            if assigned and not runtime_path.exists():
                raise ValueError(f"Evaluation shard {rank} has no inference runtime record")
        if any(record.get("status") not in statuses for record in saved.values()):
            raise ValueError("Predictions contain a missing or unknown episode status")
        if runtime is None or runtime.get("backend") != "vllm" or runtime.get("model") != manifest["model"]:
            raise ValueError("Inference runtime does not identify the manifest policy")
        if "adapter_identity" in manifest and runtime.get("adapter") != manifest["adapter_identity"]:
            raise ValueError("Inference runtime adapter differs from its generation manifest")
    return saved, manifest, runtime


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
    import torch

    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    backend = getattr(args, "inference_backend", "hf")
    quantization = getattr(args, "quantization", "none")
    if backend == "vllm" and (world != 1 or local != 0 or rank != 0):
        raise ValueError("Use one evaluation process for vLLM; configure tensor parallelism instead of torchrun replicas")
    if backend == "vllm" and quantization != "none":
        raise ValueError("The vLLM evaluation backend currently requires BF16 base weights")
    shard = _evaluation_shard(args, generation=True)
    if shard is not None:
        rank, world = shard
    torch.manual_seed(args.seed + rank)
    tasks = [task for path in args.data for task in load_tasks(path)]
    if any(task.split == "train" for task in tasks):
        raise ValueError("Use held-out validation or test tasks for evaluation")
    if args.limit:
        tasks = tasks[: args.limit]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    retrieval_profile = load_retrieval_profile(getattr(args, "retrieval_profile", None))
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
    if shard is not None:
        manifest["eval_num_shards"] = world
        manifest["adapter_identity"] = _adapter_identity(args.adapter)
    if retrieval_profile is not None:
        # Profiles contain tuples in memory, while JSON persists them as lists.
        # Compare the same representation when resuming an unchanged profile.
        manifest["retrieval_profile"] = json.loads(
            json.dumps(dataclasses.asdict(retrieval_profile))
        )
    if quantization != "none":
        # Preserve existing BF16 manifests and reject changing precision when
        # resuming a quantized run. Quantized scores need a separate directory.
        manifest["quantization"] = quantization
    if backend != "hf":
        manifest["inference_backend"] = backend
        manifest["inference_parameters"] = {
            "tensor_parallel_size": args.vllm_tensor_parallel_size,
            "gpu_memory_utilization": args.vllm_memory_utilization,
            "max_num_batched_tokens": args.vllm_batched_tokens,
            "max_model_len": args.max_prompt_tokens + args.max_new_tokens,
            "enforce_eager": args.vllm_enforce_eager,
            "batch_size": args.batch_size,
        }
    if any(task.benchmark == "widesearch" for task in tasks):
        manifest["search_backend"] = WebTools().backend
        manifest["search_endpoint"] = os.environ.get("DCC_SEARCH_URL")
        manifest["evidence_guard_version"] = WIDESEARCH_EVIDENCE_GUARD_VERSION
        auto_read_sources = getattr(args, "auto_read_sources", 0)
        if auto_read_sources:
            manifest["auto_read_sources"] = auto_read_sources
    manifest_path = output / f"manifest_rank{rank}.json"
    _check_generation_manifests(output, manifest)
    if not manifest_path.exists():
        # Other independently launched shards must never read partial JSON.
        temporary = output / f".manifest_rank{rank}.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(manifest_path)
    _check_generation_manifests(output, manifest)
    file = output / f"predictions_rank{rank}.jsonl"
    done = {}
    if file.exists():
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
        (output / f"rank{rank}.done").unlink(missing_ok=True)
    if backend == "vllm":
        previous_runtime = _saved_runtime(output)
        _check_saved_adapter(args.adapter, previous_runtime)
    if todo:
        policy, stream, workers, failure = None, None, [], None
        try:
            if backend == "vllm":
                from slime_plugins.dcc.vllm_policy import load_vllm_policy

                policy = load_vllm_policy(
                    args.model, adapter=args.adapter, batch_size=args.batch_size,
                    max_prompt_tokens=args.max_prompt_tokens,
                    max_model_len=args.max_prompt_tokens + args.max_new_tokens,
                    tensor_parallel_size=args.vllm_tensor_parallel_size,
                    gpu_memory_utilization=args.vllm_memory_utilization,
                    max_num_batched_tokens=args.vllm_batched_tokens,
                    enforce_eager=args.vllm_enforce_eager, seed=args.seed,
                )
                runtime_file = f"inference_runtime_rank{rank}.json" if world > 1 else "inference_runtime.json"
                provenance = dict(policy.provenance)
                identity = _adapter_identity(args.adapter)
                if shard is not None:
                    if identity != manifest["adapter_identity"]:
                        raise ValueError("Adapter checkpoint changed while loading the policy")
                provenance["adapter"] = identity
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
            )
            semaphore = asyncio.Semaphore(args.batch_size * 2)
            stream = file.open("a", buffering=1)
            start = time.monotonic()

            async def one(task):
                async with semaphore:
                    result = await runner.run(task, policy, evaluation=True)
                    value = result.to_dict(include_tokens=args.save_tokens)
                    token_usage = {
                        "prompt": sum(len(step.prompt_ids) for step in result.steps),
                        "completion": sum(len(step.completion_ids) for step in result.steps),
                    }
                    value["token_usage"] = token_usage
                    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
                    print(
                        json.dumps(
                            {
                                "rank": rank,
                                "task": task.id,
                                "status": result.status,
                                "turns": len(result.steps),
                                "token_usage": token_usage,
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
    (output / f"rank{rank}.done").write_text(
        json.dumps({"tasks": len(assigned), "world_size": world})
    )


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


def score(args):
    _evaluation_shard(args, generation=False)
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    tasks = [task for path in args.data for task in load_tasks(path)]
    if args.limit:
        tasks = tasks[: args.limit]
    output = Path(args.output_dir)
    saved, generation_manifest, runtime = _score_predictions(output, tasks)
    policy_quantization = _prediction_quantization(output)
    policy_backend = _prediction_setting(output, "inference_backend", "hf")
    if args.merge_judge:
        report = json.loads((output / "scores_strict.json").read_text())
        records = []
        judge_quantization = getattr(args, "quantization", "none")
        for path in sorted(output.glob("scores_judge_rank*.json")):
            shard = json.loads(path.read_text())
            if shard.get("judge_quantization", "none") != judge_quantization:
                raise ValueError("Judge quantization differs across scoring shards or merge settings")
            if shard.get("policy_quantization", policy_quantization) != policy_quantization:
                raise ValueError("Judge report policy quantization differs from prediction manifests")
            if shard.get("policy_inference_backend", policy_backend) != policy_backend:
                raise ValueError("Judge report policy backend differs from prediction manifests")
            records.extend(shard["widesearch"]["per_task"])
        expected_wide = {task.id for task in tasks if task.benchmark == "widesearch"}
        if (
            len(records) != len(expected_wide)
            or {value["task_id"] for value in records} != expected_wide
        ):
            raise ValueError(
                "Sharded judge reports must cover every WideSearch task exactly once"
            )
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
    judge = None
    if args.local_judge:
        # Frozen base judge, after all policy predictions are persisted.
        tasks = [task for task in tasks if task.benchmark == "widesearch"][rank::world]
        saved = {task.id: saved[task.id] for task in tasks}
        model, tokenizer = load_policy_model(
            args.model, rank=local, quantization=getattr(args, "quantization", "none")
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
    parser.add_argument("--quantization", choices=("none", "nf4"), default="none")
    parser.add_argument("--retrieval-profile")
    parser.add_argument("--auto-read-sources", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--inference-backend", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-memory-utilization", type=float, default=0.88)
    parser.add_argument("--vllm-batched-tokens", type=int, default=8192)
    parser.add_argument("--vllm-enforce-eager", action="store_true")
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
    parser.add_argument("--judge-batch-size", type=int, default=4)
    parser.add_argument("--judge-batch-tokens", type=int, default=32768)
    args = parser.parse_args()
    if args.score_only:
        score(args)
    else:
        asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
