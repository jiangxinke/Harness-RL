#!/usr/bin/env python3
"""Prepare independent public-registry training tasks and real tool teachers."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "slime"))

from slime_plugins.dcc import EpisodeRunner, PolicyCompletion
from slime_plugins.dcc.tools import WebTools
from slime_plugins.dcc.web_curriculum import (
    PUBLIC_TABLES, REGISTRIES, public_table_tasks, registry_tasks, verify_curriculum_rows,
)


async def teacher(task):
    """Capture actual search/read observations before supervising a final table."""
    if task.split != "train" or task.metadata.get("source") != "independent_curriculum":
        raise ValueError("Teacher generation accepts independent training tasks only")
    records = []
    url = task.metadata["source_url"]
    fetched = False
    next_start = None
    search_attempts = 0
    base_query = task.metadata["teacher_query"]
    queries = list(dict.fromkeys([base_query, base_query.split(" site:", 1)[0],
                                 base_query.split(" site:", 1)[0].replace(" CSV", "")]))

    async def policy(messages, **_):
        nonlocal fetched, next_start, search_attempts
        if not records:
            action = {"tool": "web_search", "arguments": {"query": queries[search_attempts], "count": 5}}
            search_attempts += 1
            stage = "search"
        else:
            if "Tool observation (source material):\n" not in messages[-1]["content"]:
                raise RuntimeError("Teacher answer rejected by the live evidence/schema guard")
            observation_text = messages[-1]["content"].split("Tool observation (source material):\n", 1)[-1]
            observation_text = observation_text.rsplit("\nRemaining turns:", 1)[0]
            observation = json.loads(observation_text)
            if "error" in observation:
                raise RuntimeError("Teacher source/tool unavailable; refusing a fabricated trajectory")
            if "results" in observation and not observation["results"]:
                if search_attempts >= len(queries):
                    raise RuntimeError("Teacher search returned no observed source URLs")
                action = {"tool": "web_search", "arguments": {"query": queries[search_attempts], "count": 5}}
                search_attempts += 1
                stage = "search"
            elif "text" in observation:
                fetched = bool(observation["text"].strip())
                next_start = observation.get("next_start")
                action = None
            else:
                action = None
            if action is not None:
                pass
            elif not fetched or next_start is not None:
                # Keep a fixed source-only overlap so a table cell crossing a
                # page boundary remains visible as one exact observed field.
                start = max(0, next_start - 1000) if next_start else 0
                action = {"tool": "fetch_url", "arguments": {"url": url, "start": start}}
                stage = "fetch"
            else:
                # Labels were created from this registry's actual CSV export.
                # Verify every field is visible in the real fetched source.
                verify_curriculum_rows(task, web.pages[url])
                source_text = " ".join(web.pages[url].split()).replace('""', '"')
                if any(" ".join(str(value).split()) not in source_text for row in task.gold for value in row.values()):
                    raise ValueError("Curriculum gold is no longer supported by the live registry")
                action = {"answer": [{**row, "citations": [url]} for row in task.gold]}
                stage = "final"
        text = json.dumps(action, ensure_ascii=False)
        records.append({"task_id": task.id, "source": "independent_curriculum", "split": task.split,
                        "benchmark": "widesearch", "curriculum_version": task.metadata["curriculum_version"],
                        "source_family": task.metadata.get("source_family", "iana-" + task.metadata.get("registry", "unknown")),
                        "language": task.metadata.get("language"), "answer_rows": len(task.gold),
                        "stage": stage, "messages": [dict(message) for message in messages], "completion": text, "source_urls": [url],
                        "grounding_fields": [str(value) for row in task.gold for value in row.values()] if stage == "final" else []})
        return PolicyCompletion(text)

    web = WebTools()
    # Larger independent exports require more than the tiny IANA sources.
    # All pages are real tool observations; token budgets are checked again
    # when a segment is selected for training.
    result = await EpisodeRunner(max_steps=24, max_new_tokens=8192, web_tools_factory=lambda: web).run(task, policy, evaluation=True)
    if result.status != "completed" or result.metrics.get("citation_validity") != 1.0 or result.reward != 1.0:
        raise RuntimeError(f"Teacher trajectory failed validation: {task.id} ({result.status})")
    return records, result.to_dict(include_tokens=False)


def balanced_training_tasks(tasks, *, limit=None, seed=42):
    """Cover source families, languages and list lengths before repeating them."""
    grouped = defaultdict(list)
    for task in sorted(tasks, key=lambda task: task.id):
        if task.split != "train":
            continue
        size = len(task.gold)
        key = (task.metadata.get("source_family", "iana-" + task.metadata.get("registry", "unknown")),
               task.metadata.get("language"), "small" if size <= 5 else "medium" if size <= 15 else "large")
        grouped[key].append(task)
    rng = random.Random(seed)
    for rows in grouped.values():
        rng.shuffle(rows)
    chosen = []
    while grouped and (limit is None or len(chosen) < limit):
        for key in sorted(grouped):
            chosen.append(grouped[key].pop())
            if not grouped[key]:
                del grouped[key]
            if limit is not None and len(chosen) >= limit:
                break
    return chosen


async def prepare(args):
    if args.workers < 1 or (args.teacher_limit is not None and args.teacher_limit < 1):
        raise ValueError("workers and any teacher limit must be positive")
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    raw = root / "sources"
    raw.mkdir(exist_ok=True)
    tasks = []
    mode = getattr(args, "source_mode", "mixed")
    sources = ([*REGISTRIES] if mode in {"iana", "mixed"} else []) + ([*PUBLIC_TABLES] if mode in {"public", "mixed"} else [])
    for registry in sources:
        is_registry = registry in REGISTRIES
        path = raw / f"{registry.key}.{'csv' if is_registry or registry.url.endswith('/CSV') else 'json'}"
        if not path.exists():
            if getattr(args, "no_download", False):
                raise FileNotFoundError(f"Missing cached source: {registry.key}")
            request = Request(registry.url, headers={"User-Agent": "Harness-RL-independent-curriculum/2.0"})
            with urlopen(request, timeout=30) as response:
                body = response.read(2_000_001)
                if len(body) > 2_000_000:
                    raise ValueError(f"Public source exceeded its size limit: {registry.key}")
                path.write_bytes(body)
        builder = registry_tasks if is_registry else public_table_tasks
        tasks.extend(builder(registry, path.read_text(encoding="utf-8-sig")))
    for split in ("train", "validation"):
        (root / f"{split}.jsonl").write_text("".join(json.dumps(task.to_dict(), ensure_ascii=False) + "\n" for task in tasks if task.split == split))
    manifest = {"source": "independent_curriculum", "version": "independent-public-tables-v3", "source_mode": mode,
                "teacher_training_split": "train", "validation_used_for_supervision": False,
                "prepared_at": datetime.now(timezone.utc).isoformat(),
                "counts": {split: sum(task.split == split for task in tasks) for split in ("train", "validation")},
                "families": {split: dict(Counter(task.metadata.get("source_family", "iana") for task in tasks if task.split == split))
                             for split in ("train", "validation")},
                "list_sizes": {split: dict(sorted(Counter(len(task.gold) for task in tasks if task.split == split).items()))
                               for split in ("train", "validation")},
                "registries": [{"key": source.key, "url": source.url, "split": source.split} for source in sources]}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.teachers:
        semaphore = asyncio.Semaphore(args.workers)

        async def one(task):
            async with semaphore:
                return await teacher(task)

        # Validation references never become supervised training records.
        training = balanced_training_tasks(tasks, limit=args.teacher_limit)
        values = await asyncio.gather(*(one(task) for task in training), return_exceptions=True)
        failures = [{"task_id": task.id, "error": type(value).__name__, "reason": str(value)}
                    for task, value in zip(training, values, strict=True) if isinstance(value, BaseException)]
        results = [value for value in values if not isinstance(value, BaseException)]
        (root / "teacher_train.jsonl").write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for records, _ in results for record in records))
        (root / "teacher_episodes.jsonl").write_text("".join(json.dumps(episode, ensure_ascii=False) + "\n" for _, episode in results))
        (root / "teacher_failures.jsonl").write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in failures))
        manifest["teacher_episodes"] = len(results)
        manifest["teacher_segments"] = sum(len(records) for records, _ in results)
        manifest["teacher_attempted"] = len(training)
        manifest["teacher_failures"] = len(failures)
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if not results:
            raise RuntimeError("No valid grounded teacher trajectories were produced")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teachers", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--teacher-limit", type=int)
    parser.add_argument("--source-mode", choices=("iana", "public", "mixed"), default="mixed")
    parser.add_argument("--no-download", action="store_true", help="Require cached original source exports")
    asyncio.run(prepare(parser.parse_args()))
