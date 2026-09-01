"""Deterministic 7 x 90 evaluation-manifest construction."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any
from collections.abc import Iterable

EVALUATION_DATASET = "inclusionAI/ASearcher-test-data"
EVALUATION_REVISION = "46678e7f483a84d7437d575629e3464b33e30505"
EVALUATION_SOURCES = {
    "2WikiMultiHopQA": "2WikiMultihopQA_rand1000/test.jsonl",
    "HotpotQA": "HotpotQA_rand1000/test.jsonl",
    "Bamboogle": "Bamboogle/test.jsonl",
    "FRAMES": "frames/test.json",
    "MuSiQue": "Musique_rand1000/test.jsonl",
    "NQ": "NQ_rand1000/test.jsonl",
    "TriviaQA": "TriviaQA_rand1000/test.jsonl",
}
EVALUATION_SOURCE_SHA256 = {
    "2WikiMultiHopQA": "5e06ba3e90566b37f7f6282fde8327d18164f8c63b2743c9c150998fa4a09546",
    "HotpotQA": "f20ebb273fe5fad27e113f8a353c1f8fbc9548d7eeeb8d6c5e7b16867d3033ac",
    "Bamboogle": "60b732ed98c9906f24d98aae67b9280302f8584c1a65d001b55e1a0aab1b5caf",
    "FRAMES": "660b24542df9ea4e84248be59042a3d82ce6cd6203279a3c659a991e01b97f63",
    "MuSiQue": "64973914d7c06f7aa6ee0aa12d2fc286c2d15c9d088f794fde4f59ddb41ebd9b",
    "NQ": "849a4eaf27e3a9b698889012cf0da529a228965866a799ccda2171da2638889e",
    "TriviaQA": "493da3d77be23340047e2a157b7dc6df1c81e3bc93c296f4f7159f9a9783dc5d",
}
EVALUATION_MANIFEST_SHA256 = "e7f7cbc702a103d684850db006daf06fa27867e6be0ac4b29ec51f58cf9ac46d"


def download_source(dataset: str, relative_path: str, destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size:
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest == EVALUATION_SOURCE_SHA256[dataset]:
            return destination
    url = f"https://huggingface.co/datasets/{EVALUATION_DATASET}/resolve/{EVALUATION_REVISION}/{relative_path}?download=true"
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, temporary)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if digest != EVALUATION_SOURCE_SHA256[dataset]:
            raise ValueError(f"unexpected {dataset} SHA-256: {digest}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        raw_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            raw_rows = raw
        elif isinstance(raw, dict):
            raw_rows = raw.get("data") or raw.get("examples") or raw.get("items")
        else:
            raw_rows = None
    if not isinstance(raw_rows, list) or not all(isinstance(row, dict) for row in raw_rows):
        raise ValueError(f"{path} does not contain a list/JSONL of objects")
    return raw_rows


def _answers(value: Any) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = []
        for item in value:
            if isinstance(item, str):
                values.append(item)
            elif isinstance(item, dict):
                candidate = item.get("text") or item.get("answer")
                if isinstance(candidate, str):
                    values.append(candidate)
    elif isinstance(value, dict):
        return _answers(value.get("text") or value.get("answer") or value.get("aliases"))
    else:
        values = []
    return list(dict.fromkeys(item.strip() for item in values if item.strip()))


def normalize_rows(dataset: str, rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for index, row in enumerate(rows):
        question = row.get("Question") if dataset == "FRAMES" else row.get("question")
        answers = _answers(row.get("answers") if dataset == "FRAMES" else row.get("answer"))
        if not isinstance(question, str) or not question.strip() or not answers:
            raise ValueError(f"{dataset} row {index} is missing question/answer")
        identity = row.get("id") or row.get("_id") or row.get("qid")
        if identity is None:
            identity = hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:24]
        normalized.append(
            {
                "dataset": dataset,
                "sample_id": f"{dataset}:{identity}",
                "question": question.strip(),
                "answers": answers,
                "source_dataset": EVALUATION_DATASET,
                "source_revision": EVALUATION_REVISION,
                "source_file": EVALUATION_SOURCES[dataset],
            }
        )
    return normalized


def stable_sample(rows: Iterable[dict[str, Any]], *, dataset: str, count: int, seed: int) -> list[dict[str, Any]]:
    rows = list(rows)
    if len(rows) < count:
        raise ValueError(f"{dataset} has {len(rows)} rows; {count} required")

    def rank(row: dict[str, Any]) -> str:
        value = f"{seed}\0{dataset}\0{row['sample_id']}".encode()
        return hashlib.sha256(value).hexdigest()

    return sorted(rows, key=lambda row: (rank(row), row["sample_id"]))[:count]


def prepare_evaluation_manifest(source_root: str | Path, output: str | Path, *, count: int = 90, seed: int = 20260811, download: bool = True) -> dict[str, Any]:
    source_root, output = Path(source_root), Path(output)
    selected: list[dict[str, Any]] = []
    sources: dict[str, dict[str, Any]] = {}
    for dataset, relative_path in EVALUATION_SOURCES.items():
        source = source_root / relative_path
        if download:
            download_source(dataset, relative_path, source)
        normalized = normalize_rows(dataset, read_rows(source))
        dataset_rows = stable_sample(normalized, dataset=dataset, count=count, seed=seed)
        selected.extend(dataset_rows)
        sources[dataset] = {
            "path": relative_path,
            "source_rows": len(normalized),
            "selected_rows": len(dataset_rows),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    # The published manifest lock was produced with CRLF JSONL records.  Write
    # the byte contract explicitly so checkout/platform newline settings cannot
    # change the digest while leaving the logical 630 rows unchanged.
    output.write_bytes("".join(json.dumps(row, ensure_ascii=False) + "\r\n" for row in selected).encode("utf-8"))
    manifest = {
        "source_dataset": EVALUATION_DATASET,
        "source_revision": EVALUATION_REVISION,
        "seed": seed,
        "samples_per_dataset": count,
        "dataset_count": len(EVALUATION_SOURCES),
        "sample_count": len(selected),
        "sources": sources,
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if download and count == 90 and seed == 20260811 and manifest["output_sha256"] != EVALUATION_MANIFEST_SHA256:
        raise RuntimeError(f"evaluation manifest does not match the committed lock: {manifest['output_sha256']}")
    return manifest
