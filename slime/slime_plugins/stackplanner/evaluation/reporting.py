"""Join model predictions with the frozen manifest and evaluate them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..data.sampling import EVALUATION_SOURCES
from .judge import JudgeClient
from .metrics import answer_recall, build_report


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain JSON objects")
    return rows


def evaluate_files(manifest_path: str | Path, predictions_path: str | Path, output_dir: str | Path, judge: JudgeClient) -> dict[str, Any]:
    manifest = _read_jsonl(manifest_path)
    counts = {name: 0 for name in EVALUATION_SOURCES}
    for row in manifest:
        dataset = row.get("dataset")
        if dataset not in counts:
            raise ValueError(f"unexpected evaluation dataset: {dataset!r}")
        counts[str(dataset)] += 1
    if any(count != 90 for count in counts.values()) or len(manifest) != 630:
        raise ValueError(f"evaluation manifest must contain exactly 7 x 90 = 630 rows; got {counts}")
    prediction_rows = _read_jsonl(predictions_path)
    predictions: dict[str, str] = {}
    for row in prediction_rows:
        sample_id, prediction = row.get("sample_id"), row.get("prediction")
        if not isinstance(sample_id, str) or not isinstance(prediction, str):
            raise ValueError("each prediction needs string sample_id and prediction fields")
        if sample_id in predictions:
            raise ValueError(f"duplicate prediction: {sample_id}")
        predictions[sample_id] = prediction
    manifest_ids = {str(row["sample_id"]) for row in manifest}
    if set(predictions) != manifest_ids:
        missing = sorted(manifest_ids - set(predictions))
        extra = sorted(set(predictions) - manifest_ids)
        raise ValueError(f"prediction/manifest mismatch: missing={missing[:5]}, extra={extra[:5]}")
    results = []
    for row in manifest:
        prediction = predictions[str(row["sample_id"])]
        result = {**row, "prediction": prediction, "answer_recall": answer_recall(prediction, row["answers"])}
        try:
            judgment = judge.judge(question=row["question"], references=row["answers"], prediction=prediction)
            result.update({"judge_correct": judgment.correct, "judge_reason": judgment.reason, "judge_error": None})
        except (RuntimeError, ValueError) as exc:
            result.update({"judge_correct": None, "judge_reason": None, "judge_error": str(exc)})
        results.append(result)
    report = build_report(results)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return report
