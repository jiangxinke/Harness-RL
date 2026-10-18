#!/usr/bin/env python3
"""Shard local GPU benchmark episodes and score persisted predictions.

Launch with torchrun for one model replica per GPU. Re-running resumes saved
predictions; infrastructure failures can be retried explicitly. Dataset labels
are consulted only by the separate scoring pass after policy episodes finish.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "slime"))

from slime_plugins.dcc import EpisodeRunner, load_tasks
from slime_plugins.dcc.evaluation import evaluate_results
from slime_plugins.dcc.hf_judge import HFBatchedJudge
from slime_plugins.dcc.hf_policy import HFPolicy, load_policy_model


async def evaluate(args):
    import torch

    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.manual_seed(args.seed + rank)
    tasks = [task for path in args.data for task in load_tasks(path)]
    if any(task.split == "train" for task in tasks):
        raise ValueError("Use held-out validation or test tasks for evaluation")
    if args.limit:
        tasks = tasks[: args.limit]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
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
    manifest_path = output / f"manifest_rank{rank}.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Evaluation configuration changed; use a new output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
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
        model, tokenizer = load_policy_model(
            args.model, adapter=args.adapter, rank=local
        )
        policy = HFPolicy(
            model,
            tokenizer,
            batch_size=args.batch_size,
            max_prompt_tokens=args.max_prompt_tokens,
        )
        runner = EpisodeRunner(
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
            initial_chunks=args.initial_chunks,
            timeout_seconds=args.timeout,
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

        await asyncio.gather(*(one(task) for task in todo))
        await policy.close()
        stream.close()
    (output / f"rank{rank}.done").write_text(
        json.dumps({"tasks": len(assigned), "world_size": world})
    )


def score(args):
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    tasks = [task for path in args.data for task in load_tasks(path)]
    if args.limit:
        tasks = tasks[: args.limit]
    output = Path(args.output_dir)
    saved = {}
    for file in sorted(output.glob("predictions_rank*.jsonl")):
        for line in file.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                saved[record["task_id"]] = record
    expected = {task.id for task in tasks}
    if set(saved) != expected:
        raise ValueError(
            f"Coverage mismatch: {len(expected - set(saved))} missing, {len(set(saved) - expected)} extra predictions"
        )
    if args.merge_judge:
        report = json.loads((output / "scores_strict.json").read_text())
        records = []
        for path in sorted(output.glob("scores_judge_rank*.json")):
            records.extend(json.loads(path.read_text())["widesearch"]["per_task"])
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
        model, tokenizer = load_policy_model(args.model, rank=local)
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
        key: getattr(args, key)
        for key in (
            "max_steps",
            "max_new_tokens",
            "max_prompt_tokens",
            "initial_chunks",
            "seed",
        )
    }
    report["model"] = args.model_name
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
