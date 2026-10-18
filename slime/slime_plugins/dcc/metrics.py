"""CUAD's official span matching and precision/recall curve, without sklearn.

Matching follows TheAtticusProject/cuad/evaluate.py: word-set Jaccard >= .5,
with its additional substring rule for Parties questions. Confidence thresholds
and the monotone precision envelope match that implementation. Model-generated
span confidences are reported as such; they are not classifier probabilities.
"""

from __future__ import annotations

import math
from typing import Any


def cuad_jaccard(reference: str, prediction: str) -> float:
    def words(value: str) -> set[str]:
        for char in ".,;:":
            value = value.replace(char, "")
        return set(value.lower().replace("/", " ").split(" "))

    left, right = words(reference), words(prediction)
    return len(left & right) / len(left | right)


def cuad_predictions(answer: Any) -> list[dict[str, Any]]:
    """Normalize the public answer schema without silently accepting prose."""

    if not isinstance(answer, list):
        return []
    predictions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in answer:
        if isinstance(item, str):
            text, confidence = item.strip(), 1.0
        elif isinstance(item, dict):
            text, confidence = str(item.get("text", "")).strip(), item.get("confidence", 1.0)
        else:
            continue
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            continue
        if text and text not in seen and math.isfinite(confidence) and 0 <= confidence <= 1:
            predictions.append({"text": text, "probability": confidence})
            seen.add(text)
    return predictions


def cuad_counts(task_id: str, gold: list[str], predictions: list[str]) -> tuple[int, int, int]:
    def matches(reference: str, prediction: str) -> bool:
        return cuad_jaccard(reference, prediction) >= 0.5 or ("Parties" in task_id and reference in prediction)

    tp = sum(any(matches(reference, pred) for pred in predictions) for reference in gold)
    fp = sum(not any(matches(reference, pred) for reference in gold) for pred in predictions)
    return tp, fp, len(gold) - tp


def cuad_task_metrics(task_id: str, gold: list[str], answer: Any) -> dict[str, float]:
    predictions = [item["text"] for item in cuad_predictions(answer)]
    tp, fp, fn = cuad_counts(task_id, gold, predictions)
    # The no-answer success is useful for RL but does not create an official TP.
    precision = tp / (tp + fp) if tp + fp else float(not gold)
    recall = tp / (tp + fn) if tp + fn else float(not predictions)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "precision": precision, "recall": recall, "no_answer_correct": float(not gold and not predictions), "tp": float(tp), "fp": float(fp), "fn": float(fn)}


def evaluate_cuad(gold: dict[str, list[str]], answers: dict[str, Any]) -> dict[str, float | int | str]:
    if gold.keys() != answers.keys():
        raise ValueError("CUAD prediction IDs must exactly match evaluation IDs")
    nbest = {key: cuad_predictions(value) for key, value in answers.items()}
    precisions, recalls = [1.0], [0.0]
    thresholds = [value / 100 for value in range(99, 0, -1)] + [0.001, 0.0]
    for threshold in thresholds:
        tp = fp = fn = 0
        for task_id, references in gold.items():
            predictions = [item["text"] for item in nbest[task_id] if item["probability"] > threshold]
            counts = cuad_counts(task_id, references, predictions)
            tp, fp, fn = tp + counts[0], fp + counts[1], fn + counts[2]
        precisions.append(tp / (tp + fp) if tp + fp else float("nan"))
        recalls.append(tp / (tp + fn) if tp + fn else float("nan"))
    envelope = precisions[:]
    # Builtin max order intentionally matches official process_precisions.
    for index in range(len(envelope) - 2, -1, -1):
        envelope[index] = max(envelope[index + 1], envelope[index])
    area = sum((recalls[i] - recalls[i - 1]) * (envelope[i] + envelope[i - 1]) / 2 for i in range(1, len(recalls)))
    tp = fp = fn = 0
    for task_id, references in gold.items():
        counts = cuad_counts(task_id, references, [item["text"] for item in nbest[task_id]])
        tp, fp, fn = tp + counts[0], fp + counts[1], fn + counts[2]
    precision, recall = tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0

    def at_recall(target: float) -> float:
        return next((value for value, recall_value in zip(envelope, recalls, strict=True) if recall_value >= target), 0.0)

    return {
        "tasks": len(gold),
        "aupr": 0.0 if math.isnan(area) else area,
        "prec_at_80_recall": at_recall(0.8),
        "prec_at_90_recall": at_recall(0.9),
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "confidence_source": "model_generated_span_confidence",
    }
