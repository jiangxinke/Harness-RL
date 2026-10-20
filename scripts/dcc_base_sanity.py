#!/usr/bin/env python3
"""Check basic instruction following on an unadapted base before benchmarking."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "slime"))


async def run(args):
    started = time.monotonic()
    if args.backend == "hf":
        from slime_plugins.dcc.hf_policy import HFPolicy, load_policy_model
        model, tokenizer = load_policy_model(args.model)
        policy = HFPolicy(model, tokenizer, batch_size=1, max_prompt_tokens=1024)
    else:
        from slime_plugins.dcc.vllm_policy import load_vllm_policy
        policy = load_vllm_policy(args.model, batch_size=args.batch_size, max_prompt_tokens=1024,
            max_model_len=args.max_model_len, max_num_batched_tokens=args.batched_tokens,
            enforce_eager=args.vllm_enforce_eager,
            asynchronous=args.backend == "vllm-async", gpu_memory_utilization=.86)
    report = {
        "model": args.model, "adapter": None, "backend": args.backend,
        "runtime": getattr(policy, "provenance", None),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers")},
        "cases": [],
    }
    try:
        for prompt, expected in (
            ("Return exactly the word READY.", "READY"),
            ("Return exactly the number 2.", "2"),
            ("请只输出数字3。", "3"),
            ("What is 2 + 2? Return only the digit.", "4"),
        ):
            result = await policy([{"role": "user", "content": prompt}], max_new_tokens=32, temperature=0)
            report["cases"].append({"prompt": prompt, "expected": expected, "text": result.text,
                "correct": result.text.strip() == expected, "prompt_ids": result.prompt_ids,
                "completion_ids": result.completion_ids, "finish_reason": result.finish_reason})
    finally:
        await policy.close()
    report["accepted"] = all(case["correct"] for case in report["cases"])
    report["elapsed_seconds"] = time.monotonic() - started
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=("hf", "vllm", "vllm-async"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vllm-enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--batched-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    raise SystemExit(asyncio.run(run(parser.parse_args())))
