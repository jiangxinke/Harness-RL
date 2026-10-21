#!/usr/bin/env python3
"""Generate DCC episodes through an OpenAI endpoint and/or score saved outputs."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "slime"))

from slime_plugins.dcc import EpisodeRunner, PolicyCompletion, load_tasks  # noqa: E402
from slime_plugins.dcc.evaluation import evaluate_results  # noqa: E402
from slime_plugins.dcc.tools import InfrastructureError  # noqa: E402
from slime_plugins.dcc.widesearch_metrics import OpenAIJudge  # noqa: E402


class OpenAIPolicy:
    """Inference-only adapter; exact trainable tokens come from the HF policy."""

    def __init__(self, base_url: str, model: str, *, timeout: int = 300) -> None:
        self.base_url, self.model, self.timeout = base_url.rstrip("/"), model, timeout

    async def __call__(self, messages, *, max_new_tokens, temperature):
        def request():
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            req = Request(
                self.base_url + "/chat/completions",
                data=json.dumps(payload).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer "
                    + os.environ.get("DCC_MODEL_API_KEY", "EMPTY"),
                },
            )
            try:
                with urlopen(req, timeout=self.timeout) as response:
                    choice = json.load(response)["choices"][0]
                return PolicyCompletion(
                    choice["message"]["content"],
                    finish_reason=choice.get("finish_reason", "stop"),
                )
            except Exception as exc:
                raise InfrastructureError("model endpoint request failed") from exc

        return await asyncio.to_thread(request)


def read_predictions(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]


async def generate(args, tasks) -> None:
    saved = read_predictions(args.predictions)
    done = {row["task_id"] for row in saved}
    pending = [task for task in tasks if task.id not in done]
    policy = OpenAIPolicy(args.base_url, args.model)
    runner = EpisodeRunner(
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        initial_chunks=args.initial_chunks,
        timeout_seconds=args.episode_timeout,
    )
    semaphore, lock = asyncio.Semaphore(args.concurrency), asyncio.Lock()
    args.predictions.parent.mkdir(parents=True, exist_ok=True)

    async def one(task):
        async with semaphore:
            result = await runner.run(task, policy, evaluation=True)
        async with lock:
            with args.predictions.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        result.to_dict(include_tokens=False),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
            print(
                json.dumps(
                    {
                        "task_id": task.id,
                        "status": result.status,
                        "model_calls": len(result.steps),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    await asyncio.gather(*(one(task) for task in pending))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--base-url",
        help="Generate missing predictions from this OpenAI-compatible endpoint",
    )
    parser.add_argument("--model")
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--initial-chunks", type=int, default=4)
    parser.add_argument("--episode-timeout", type=int, default=900)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--judge-base-url")
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-workers", type=int, default=4)
    args = parser.parse_args()
    if bool(args.base_url) != bool(args.model):
        parser.error("--base-url and --model must be supplied together")
    if bool(args.judge_base_url) != bool(args.judge_model):
        parser.error("--judge-base-url and --judge-model must be supplied together")
    tasks = load_tasks(args.tasks)
    if args.base_url:
        asyncio.run(generate(args, tasks))
    judge = (
        OpenAIJudge(
            args.judge_base_url,
            args.judge_model,
            os.environ.get("DCC_JUDGE_API_KEY", "EMPTY"),
        )
        if args.judge_base_url
        else None
    )
    report = evaluate_results(
        tasks,
        read_predictions(args.predictions),
        judge=judge,
        workers=args.judge_workers,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        key: {k: v for k, v in value.items() if k != "per_task"}
        if isinstance(value, dict)
        else value
        for key, value in report.items()
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
