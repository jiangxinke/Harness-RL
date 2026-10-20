#!/usr/bin/env python3
"""Exercise native asynchronous base generation and cancellation on one GPU."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "slime"))

from slime_plugins.dcc.vllm_policy import load_vllm_policy


async def run(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    policy = load_vllm_policy(
        args.model, batch_size=4, max_prompt_tokens=1024, max_model_len=5120,
        max_num_batched_tokens=4096, asynchronous=True,
        gpu_memory_utilization=args.memory_utilization, seed=42,
        enforce_eager=args.vllm_enforce_eager,
    )
    report = {"runtime": policy.provenance, "checks": {}}
    def prompt(text):
        return [{"role": "user", "content": text}]
    try:
        first = await policy(prompt("Return exactly the word READY."), max_new_tokens=16, temperature=0)
        report["first"] = {"text": first.text, "prompt_tokens": len(first.prompt_ids),
                           "completion_tokens": len(first.completion_ids), "finish_reason": first.finish_reason}
        report["checks"]["base_generates_nonempty_output"] = bool(first.text.strip()) and bool(first.completion_ids)
        report["checks"]["base_follows_ready_instruction"] = first.text.strip() == "READY"
        try:
            await asyncio.wait_for(policy(prompt(
                "Write a very long list of integers starting at 1, one integer per line, and continue until 10000."
            ), max_new_tokens=4096, temperature=0), timeout=.05)
        except TimeoutError:
            report["checks"]["long_request_cancelled"] = True
        else:
            report["checks"]["long_request_cancelled"] = False
        report["checks"]["cancelled_request_removed"] = not policy._active_requests
        responses = await asyncio.wait_for(asyncio.gather(*[
            policy(prompt(text), max_new_tokens=32, temperature=0)
            for text in ("Return exactly the number 2.", "请只输出数字3。")
        ]), timeout=60)
        report["after_cancellation"] = [r.text for r in responses]
        report["checks"]["later_concurrent_requests_succeed"] = all(r.text.strip() for r in responses)
        report["checks"]["concurrent_requests_follow_instructions"] = [r.text.strip() for r in responses] == ["2", "3"]
        report["checks"]["all_requests_drained"] = not policy._active_requests
    finally:
        await policy.close()
    report["elapsed_seconds"] = time.monotonic() - start
    report["accepted"] = all(report["checks"].values())
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--memory-utilization", type=float, default=.88)
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    raise SystemExit(asyncio.run(run(parser.parse_args())))
