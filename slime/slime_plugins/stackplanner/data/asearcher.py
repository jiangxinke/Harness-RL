"""Download and convert the pinned ASearcher training corpus."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any
from collections.abc import Iterable

TRAIN_DATASET = "inclusionAI/ASearcher-train-data"
TRAIN_REVISION = "2e1b758bd53a36c28cbe1d49776af35f95a9a4b0"
TRAIN_FILE = "ASearcher-LRM-35k.jsonl"
TRAIN_SHA256 = "971d59a51b6c604b5b5eff5dcdaadf5d4624bc376cf610995d27e5017f660bfb"
TRAIN_SIZE = 13_278_900
TRAIN_ROWS = 35_054
CONVERTED_TRAIN_SHA256 = "356773b1096df2aa2a36084efd2adff9a8a1197c7cf3c641f0699b8f79602b65"


def _download_url(dataset: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{filename}?download=true"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_training_file(path: str | Path) -> None:
    path = Path(path)
    if path.stat().st_size != TRAIN_SIZE:
        raise ValueError(f"unexpected ASearcher size: {path.stat().st_size} != {TRAIN_SIZE}")
    digest = sha256_file(path)
    if digest != TRAIN_SHA256:
        raise ValueError(f"unexpected ASearcher SHA-256: {digest}")


def download_training_file(destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == TRAIN_SIZE and sha256_file(destination) == TRAIN_SHA256:
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        urllib.request.urlretrieve(
            _download_url(TRAIN_DATASET, TRAIN_REVISION, TRAIN_FILE),
            temporary,
        )
        verify_training_file(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def convert_rows(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for index, row in enumerate(rows):
        question = row.get("question")
        answer = row.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"row {index} has no non-empty question")
        if answer is None or (isinstance(answer, str) and not answer.strip()):
            raise ValueError(f"row {index} has no answer")
        source_id = str(row.get("id") or row.get("_id") or index)
        yield {
            "prompt": question.strip(),
            "label": answer,
            "metadata": {
                "dataset": "ASearcher-LRM-35k",
                "source": TRAIN_DATASET,
                "source_revision": TRAIN_REVISION,
                "source_file": TRAIN_FILE,
                "source_id": source_id,
            },
        }


def convert_training_file(source: str | Path, output: str | Path) -> int:
    source, output = Path(source), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    count = 0
    try:
        with source.open(encoding="utf-8") as input_handle, temporary.open("w", encoding="utf-8", newline="\n") as output_handle:
            for line_number, line in enumerate(input_handle, 1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError(f"line {line_number} is not a JSON object")
                converted = next(iter(convert_rows([raw])))
                converted["metadata"]["source_id"] = str(raw.get("id") or raw.get("_id") or line_number - 1)
                output_handle.write(json.dumps(converted, ensure_ascii=False) + "\n")
                count += 1
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return count
