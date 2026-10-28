#!/usr/bin/env python3
"""Record untrained or trained CUAD trajectories with the audited protocol."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "slime"))

import torch
from slime_plugins.dcc import EpisodeRunner, load_tasks
from slime_plugins.dcc.hf_policy import HFPolicy, load_policy_model


async def evaluate(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    tasks = load_tasks(args.data)
    if args.limit:
        tasks = tasks[:args.limit]
    if not tasks or any(t.benchmark != "cuad" for t in tasks):
        raise ValueError("Evaluation requires CUAD tasks")
    model, tokenizer = load_policy_model(args.model, adapter=args.adapter, trainable=False)
    policy = HFPolicy(model, tokenizer, batch_size=args.repeats,
                      max_prompt_tokens=args.max_prompt_tokens, allow_truncation=False,
                      raw_sampling=args.temperature == 1.0)
    runner = EpisodeRunner(max_steps=args.max_steps, initial_chunks=args.initial_chunks,
        max_new_tokens=args.max_new_tokens, evaluation_temperature=args.temperature,
        cuad_protocol=args.cuad_protocol, retrieval_context_chars=0)
    summary = []
    started = time.monotonic()
    with (out / "rollouts.jsonl").open("x", encoding="utf-8") as stream:
        for task in tasks:
            results = await asyncio.gather(*(runner.run(task, policy, evaluation=True)
                                            for _ in range(args.repeats)))
            for result in results:
                stream.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
            stream.flush()
            row = dict(task=task.id, rewards=[r.reward for r in results],
                f1=[r.metrics.get("f1") for r in results],
                statuses=[r.status for r in results], decisions=[len(r.steps) for r in results])
            summary.append(row)
            print(json.dumps(row), flush=True)
    await policy.close()
    (out / "summary.json").write_text(json.dumps({
        "settings": vars(args), "training": False, "tasks": summary,
        "rollouts": len(tasks) * args.repeats, "seconds": time.monotonic() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--cuad-protocol", choices=("cuad-audited-v1", "cuad-audited-v2"), default="cuad-audited-v2")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=147)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-prompt-tokens", type=int, default=24576)
    parser.add_argument("--initial-chunks", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1 or args.initial_chunks < 1:
        parser.error("Positive repeat and initial excerpt counts are required")
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
