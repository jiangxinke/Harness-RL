#!/usr/bin/env python3
"""Judge a complete prediction set with durable, restart-safe progress.

Each judge attempt is appended and fsynced. Restarts retain successful IDs and
retry only missing judgments. Final results and their summary are published
atomically only after all 630 locked samples have a valid strict-JSON judgment.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SLIME_SOURCE = str(ROOT / "slime")
sys.path.insert(0, SLIME_SOURCE)

from slime_plugins.stackplanner.evaluation.judge import JudgeClient  # noqa: E402
from slime_plugins.stackplanner.evaluation.metrics import (  # noqa: E402
    answer_recall,
    build_report,
)

sys.path.remove(SLIME_SOURCE)

EXPECTED_DATASETS = {
    "2WikiMultiHopQA",
    "HotpotQA",
    "Bamboogle",
    "FRAMES",
    "MuSiQue",
    "NQ",
    "TriviaQA",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def load_evaluation_inputs(
    manifest_path: Path,
    predictions_path: Path,
    *,
    samples_per_dataset: int = 90,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if samples_per_dataset < 1:
        raise ValueError("samples_per_dataset must be positive")
    manifest = _read_jsonl(manifest_path)
    counts = Counter(str(row.get("dataset")) for row in manifest)
    if (
        len(manifest) != len(EXPECTED_DATASETS) * samples_per_dataset
        or set(counts) != EXPECTED_DATASETS
        or any(counts[name] != samples_per_dataset for name in EXPECTED_DATASETS)
    ):
        raise ValueError(
            "evaluation manifest must contain exactly 7 x "
            f"{samples_per_dataset}: {counts}"
        )
    manifest_ids: list[str] = []
    for row in manifest:
        sample_id = row.get("sample_id")
        question = row.get("question")
        answers = row.get("answers")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("every manifest row needs a non-empty sample_id")
        if sample_id in manifest_ids:
            raise ValueError(f"duplicate manifest sample_id: {sample_id}")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"manifest sample {sample_id} has no question")
        if (
            not isinstance(answers, list)
            or not answers
            or not all(isinstance(answer, str) for answer in answers)
        ):
            raise ValueError(f"manifest sample {sample_id} has invalid answers")
        manifest_ids.append(sample_id)

    predictions: dict[str, str] = {}
    for row in _read_jsonl(predictions_path):
        sample_id = row.get("sample_id")
        prediction = row.get("prediction")
        if (
            not isinstance(sample_id, str)
            or not isinstance(prediction, str)
            or not prediction.strip()
        ):
            raise ValueError("every prediction needs non-empty string fields")
        if sample_id in predictions:
            raise ValueError(f"duplicate prediction sample_id: {sample_id}")
        predictions[sample_id] = prediction
    if set(predictions) != set(manifest_ids):
        missing = sorted(set(manifest_ids) - set(predictions))
        extra = sorted(set(predictions) - set(manifest_ids))
        raise ValueError(
            f"prediction/manifest mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    return manifest, predictions


def successful_judgments(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        if (
            row.get("status") == "succeeded"
            and isinstance(row.get("sample_id"), str)
            and isinstance(row.get("judge_correct"), bool)
            and isinstance(row.get("judge_reason"), str)
        ):
            completed[str(row["sample_id"])] = row
    return completed


def _attempt_counts(path: Path) -> defaultdict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    if path.is_file():
        for row in _read_jsonl(path):
            sample_id = row.get("sample_id")
            attempt = row.get("attempt")
            if isinstance(sample_id, str) and isinstance(attempt, int):
                counts[sample_id] = max(counts[sample_id], attempt)
    return counts


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validate_resume_identity(
    metadata_path: Path, records_path: Path, identity: Mapping[str, Any]
) -> None:
    if not records_path.exists():
        return
    if not metadata_path.is_file():
        raise ValueError(
            f"cannot trust existing judge records without metadata: {records_path}"
        )
    try:
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read judge metadata: {metadata_path}") from exc
    mismatches = {
        key: {"previous": previous.get(key), "requested": value}
        for key, value in identity.items()
        if previous.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "existing judge records have a different run identity: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def materialize_results(
    manifest: list[dict[str, Any]],
    predictions: Mapping[str, str],
    completed: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
    *,
    expected_samples: int = 630,
) -> dict[str, Any]:
    manifest_ids = [str(row["sample_id"]) for row in manifest]
    if set(completed) != set(manifest_ids):
        missing = [
            sample_id for sample_id in manifest_ids if sample_id not in completed
        ]
        raise ValueError(f"cannot publish incomplete judgments: missing={missing[:10]}")
    results: list[dict[str, Any]] = []
    for row in manifest:
        sample_id = str(row["sample_id"])
        prediction = predictions[sample_id]
        judgment = completed[sample_id]
        results.append(
            {
                **row,
                "prediction": prediction,
                "answer_recall": answer_recall(prediction, row["answers"]),
                "judge_correct": bool(judgment["judge_correct"]),
                "judge_reason": str(judgment["judge_reason"]),
                "judge_error": None,
            }
        )
    report = build_report(results)
    if (
        not report["complete"]
        or report["overall_micro"]["judge_successes"] != expected_samples
    ):
        raise ValueError("complete judge records did not produce a complete report")
    temporary = output_dir / f".results.jsonl.tmp.{os.getpid()}"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_dir / "results.jsonl")
    _atomic_json(output_dir / "summary.json", report)
    return report


def _endpoint_models(base_url: str, timeout: float = 30) -> list[str]:
    request = urllib.request.Request(base_url.rstrip("/") + "/models")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
        return [str(item["id"]) for item in payload["data"]]
    except (
        urllib.error.URLError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
    ) as exc:
        raise RuntimeError(f"judge endpoint is not healthy: {base_url}: {exc}") from exc


async def _run(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.resolve()
    predictions_path = args.predictions.resolve()
    manifest, predictions = load_evaluation_inputs(
        manifest_path,
        predictions_path,
        samples_per_dataset=args.samples_per_dataset,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "judge_records.jsonl"
    metadata_path = args.output_dir / "judge_metadata.json"
    endpoint_models = await asyncio.to_thread(_endpoint_models, args.judge_base_url)
    if args.judge_model not in endpoint_models:
        raise RuntimeError(
            f"judge endpoint does not serve {args.judge_model!r}: {endpoint_models}"
        )
    identity = {
        "judge_model": args.judge_model,
        "judge_protocol": JudgeClient.PROTOCOL,
        "manifest_sha256": _sha256(manifest_path),
        "predictions_sha256": _sha256(predictions_path),
        "samples_per_dataset": args.samples_per_dataset,
        "temperature": 0.0,
    }
    _validate_resume_identity(metadata_path, records_path, identity)
    completed = successful_judgments(records_path)
    completed = {key: value for key, value in completed.items() if key in predictions}
    attempts = _attempt_counts(records_path)
    started_at = time.time()
    metadata: dict[str, Any] = {
        **identity,
        "completed_samples": len(completed),
        "endpoint_models_reported": endpoint_models,
        "judge_base_url": args.judge_base_url,
        "manifest": str(manifest_path),
        "predictions": str(predictions_path),
        "started_at_unix": started_at,
        "status": "running",
        "target_samples": len(manifest),
    }
    _atomic_json(metadata_path, metadata)

    judge = JudgeClient(
        args.judge_base_url,
        model=args.judge_model,
        api_key=os.environ.get(args.api_key_env, ""),
        timeout=args.timeout,
    )
    append_lock = asyncio.Lock()
    completed_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(args.concurrency)
    queue = [row for row in manifest if str(row["sample_id"]) not in completed]
    if args.max_new:
        queue = queue[: args.max_new]

    async def append_record(record: Mapping[str, Any]) -> None:
        rendered = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        async with append_lock:
            with records_path.open("a", encoding="utf-8") as handle:
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())

    async def worker(row: Mapping[str, Any]) -> None:
        sample_id = str(row["sample_id"])
        async with semaphore:
            for local_attempt in range(1, args.max_attempts + 1):
                attempt = attempts[sample_id] + local_attempt
                attempt_started = time.time()
                try:
                    judgment = await asyncio.to_thread(
                        judge.judge,
                        question=str(row["question"]),
                        references=list(row["answers"]),
                        prediction=predictions[sample_id],
                    )
                    record = {
                        "attempt": attempt,
                        "elapsed_seconds": time.time() - attempt_started,
                        "judge_correct": judgment.correct,
                        "judge_reason": judgment.reason,
                        "sample_id": sample_id,
                        "status": "succeeded",
                        "time_unix": time.time(),
                    }
                    await append_record(record)
                    async with completed_lock:
                        completed[sample_id] = record
                    return
                except Exception as exc:
                    await append_record(
                        {
                            "attempt": attempt,
                            "elapsed_seconds": time.time() - attempt_started,
                            "error": f"{type(exc).__name__}: {exc}",
                            "sample_id": sample_id,
                            "status": "failed",
                            "time_unix": time.time(),
                        }
                    )
                    if local_attempt < args.max_attempts:
                        await asyncio.sleep(min(30.0, 2.0**local_attempt))

    await asyncio.gather(*(worker(row) for row in queue))
    metadata["completed_samples"] = len(completed)
    metadata["elapsed_seconds"] = time.time() - started_at
    if len(completed) == len(manifest):
        report = materialize_results(
            manifest,
            predictions,
            completed,
            args.output_dir,
            expected_samples=len(manifest),
        )
        metadata["results_sha256"] = _sha256(args.output_dir / "results.jsonl")
        metadata["summary"] = report
        metadata["status"] = "complete"
        metadata["finished_at_unix"] = time.time()
        _atomic_json(metadata_path, metadata)
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    metadata["missing_samples"] = [
        str(row["sample_id"])
        for row in manifest
        if str(row["sample_id"]) not in completed
    ]
    metadata["status"] = "partial"
    _atomic_json(metadata_path, metadata)
    return 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judge-base-url", default="http://127.0.0.1:18002/v1")
    parser.add_argument("--judge-model", default="Qwen2.5-3B-Instruct")
    parser.add_argument("--api-key-env", default="JUDGE_API_KEY")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-new", type=int, default=0)
    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=90,
        help="expected balanced manifest size; formal evaluation keeps the default 90",
    )
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if (
        args.concurrency < 1
        or args.max_attempts < 1
        or args.max_new < 0
        or args.samples_per_dataset < 1
    ):
        parser.error(
            "concurrency/max-attempts/samples-per-dataset must be positive "
            "and max-new nonnegative"
        )
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
