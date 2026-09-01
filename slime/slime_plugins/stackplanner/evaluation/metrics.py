"""Deterministic answer recall and per-dataset/overall aggregation."""

from __future__ import annotations

import collections
import re
import string
import unicodedata
from typing import Any
from collections.abc import Iterable

_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)


def normalize_answer(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = "".join(" " if (char in string.punctuation or unicodedata.category(char).startswith("P")) else char for char in value)
    value = _ARTICLES.sub(" ", value)
    return " ".join(value.split())


def _token_recall(prediction: str, reference: str) -> float:
    gold = normalize_answer(reference).split()
    if not gold:
        return float(not normalize_answer(prediction))
    predicted = collections.Counter(normalize_answer(prediction).split())
    expected = collections.Counter(gold)
    overlap = sum(min(predicted[token], count) for token, count in expected.items())
    return overlap / len(gold)


def answer_recall(prediction: str, references: Iterable[str]) -> float:
    values = [_token_recall(prediction, reference) for reference in references]
    if not values:
        raise ValueError("at least one reference answer is required")
    return max(values)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    judged = [row for row in rows if row.get("judge_correct") is not None]
    failed = [row for row in rows if row.get("judge_error")]
    recalls = [float(row["answer_recall"]) for row in rows]
    return {
        "samples": len(rows),
        "judge_successes": len(judged),
        "judge_failures": len(failed),
        "judge_coverage": len(judged) / len(rows) if rows else 0.0,
        "judge_accuracy": (sum(bool(row["judge_correct"]) for row in judged) / len(judged)) if judged else None,
        "answer_recall": sum(recalls) / len(recalls) if recalls else None,
    }


def build_report(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    per_dataset = {name: _summary(items) for name, items in sorted(grouped.items())}
    overall_micro = _summary(rows)
    metric_names = ("judge_accuracy", "answer_recall")
    overall_macro = {name: (sum(float(values[name]) for values in per_dataset.values() if values[name] is not None) / sum(values[name] is not None for values in per_dataset.values())) if any(values[name] is not None for values in per_dataset.values()) else None for name in metric_names}
    return {
        "complete": bool(rows) and overall_micro["judge_failures"] == 0,
        "per_dataset": per_dataset,
        "overall_micro": overall_micro,
        "overall_macro": overall_macro,
    }
