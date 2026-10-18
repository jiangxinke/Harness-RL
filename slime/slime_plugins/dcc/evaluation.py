"""Score saved predictions with explicit coverage and judge provenance."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .data import Task
from .metrics import evaluate_cuad
from .widesearch_metrics import Judge, JudgeError, evaluate_widesearch


def evaluate_results(tasks: Iterable[Task], predictions: Iterable[dict[str, Any]], *, judge: Judge | None = None, workers: int = 1) -> dict[str, Any]:
    tasks = list(tasks)
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
            **{key: sum(value[key] for value in successful) / len(successful) if successful else None for key in keys},
            "per_task": per_task,
        }
        if len(successful) != len(wide):
            report["complete"] = False
    return report
