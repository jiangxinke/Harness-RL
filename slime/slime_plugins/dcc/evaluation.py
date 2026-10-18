"""Score saved predictions with explicit coverage and judge provenance."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .data import Task
from .metrics import evaluate_cuad
from .widesearch_metrics import Judge, JudgeError, evaluate_widesearch


def _search_coverage(tasks: list[Task], saved: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Summarize recorded search observations without echoing URLs or errors."""
    known_backends = {"public_web", "seznam", "baidu_mobile", "duckduckgo_lite", "duckduckgo_wikipedia", "wikipedia", "serper", "brave", "searxng", "bing_rss", "auto"}
    backend_counts: Counter[str] = Counter()
    configured_counts: Counter[str] = Counter()
    scope_counts: Counter[str] = Counter()
    fallback_counts: Counter[str] = Counter()
    searched: set[str] = set()
    web: set[str] = set()
    encyclopedia: set[str] = set()
    for task in tasks:
        events = saved.get(task.id, {}).get("tool_events", [])
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict) or event.get("tool") != "web_search":
                continue
            observation = event.get("observation")
            observation = observation if isinstance(observation, dict) else {}
            searched.add(task.id)
            backend = observation.get("backend")
            configured = observation.get("configured_backend")
            scope = observation.get("search_scope")
            backend_counts[backend if isinstance(backend, str) and backend in known_backends else "unknown"] += 1
            configured_counts[configured if isinstance(configured, str) and configured in known_backends else "unknown"] += 1
            scope = scope if scope in ("web", "encyclopedia") else "unknown"
            scope_counts[scope] += 1
            if scope == "web":
                web.add(task.id)
            elif scope == "encyclopedia":
                encyclopedia.add(task.id)
            reason = observation.get("fallback_reason")
            if reason:
                # Raw exception strings may contain operator-configured server
                # details. Export only fixed reason categories.
                reason = reason.lower() if isinstance(reason, str) else ""
                category = "cooldown" if "cooldown" in reason else "challenge" if "challeng" in reason or "captcha" in reason else "no_results" if "no search results" in reason else "provider_error"
                fallback_counts[category] += 1
    calls = sum(scope_counts.values())
    return {
        "tasks": len(tasks),
        "tasks_with_search": len(searched),
        "tasks_without_search": len(tasks) - len(searched),
        "tasks_with_web_search": len(web),
        "tasks_with_encyclopedia_search": len(encyclopedia),
        "tasks_with_encyclopedia_search_fraction": len(encyclopedia) / len(tasks),
        "search_calls": calls,
        "backend_counts": dict(backend_counts),
        "configured_backend_counts": dict(configured_counts),
        "scope_counts": dict(scope_counts),
        "encyclopedia_search_fraction": scope_counts["encyclopedia"] / calls if calls else None,
        "fallback_calls": sum(fallback_counts.values()),
        "fallback_reason_counts": dict(fallback_counts),
    }


def evaluate_results(tasks: Iterable[Task], predictions: Iterable[dict[str, Any]], *, judge: Judge | None = None, workers: int = 1) -> dict[str, Any]:
    tasks = list(tasks)
    if not tasks:
        raise ValueError("evaluation requires at least one task")
    saved: dict[str, dict[str, Any]] = {}
    for record in predictions:
        task_id = str(record.get("task_id", record.get("id", "")))
        if not task_id or task_id in saved:
            raise ValueError(f"missing or duplicate prediction id: {task_id}")
        saved[task_id] = record
    expected = {task.id for task in tasks}
    if len(expected) != len(tasks):
        raise ValueError("duplicate evaluation task IDs")
    if saved.keys() - expected:
        raise ValueError("prediction IDs outside evaluation task set")
    status = Counter(saved.get(task.id, {}).get("status", "missing") for task in tasks)
    report: dict[str, Any] = {
        "tasks": len(tasks),
        "prediction_coverage": len(saved) / len(tasks) if tasks else 0.0,
        "status_counts": dict(status),
        "complete": len(saved) == len(tasks) and not status["infrastructure_error"],
    }
    cuad = [task for task in tasks if task.benchmark == "cuad"]
    if cuad:
        report["cuad"] = evaluate_cuad({task.id: task.gold or [] for task in cuad}, {task.id: saved.get(task.id, {}).get("answer", []) for task in cuad})
    wide = [task for task in tasks if task.benchmark == "widesearch"]
    if wide:

        def score(task: Task) -> dict[str, Any]:
            if saved.get(task.id, {}).get("status") == "evidence_failure":
                # Rejected finals are model failures. Keep them in every
                # denominator without judging unsupported or empty answers.
                return {"task_id": task.id, **dict.fromkeys(("score", "precision_by_row", "recall_by_row", "f1_by_row", "precision_by_item", "recall_by_item", "f1_by_item"), 0.0)}
            try:
                metrics = evaluate_widesearch(saved.get(task.id, {}).get("answer", []), task.gold or [], task.metadata["evaluation"], judge=judge)
                return {"task_id": task.id, **metrics}
            except JudgeError as exc:
                return {"task_id": task.id, "judge_error": str(exc)}

        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            per_task = list(executor.map(score, wide))
        successful = [value for value in per_task if "judge_error" not in value]
        keys = ["score", "precision_by_row", "recall_by_row", "f1_by_row", "precision_by_item", "recall_by_item", "f1_by_item"]
        report["widesearch"] = {
            "tasks": len(wide),
            "metric_denominator": len(successful),
            "judge_failures": len(per_task) - len(successful),
            "evaluation_protocol": "official_protocol_custom_judge" if judge else "strict_exact_proxy",
            "judge_model": getattr(judge, "model", None),
            "search_coverage": _search_coverage(wide, saved),
            **{key: sum(value[key] for value in successful) / len(successful) if successful else None for key in keys},
            "per_task": per_task,
        }
        if len(successful) != len(wide):
            report["complete"] = False
    return report
