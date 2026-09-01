#!/usr/bin/env python3
"""Derive a deterministic balanced quick-evaluation manifest from the lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_evaluation_predictions import load_manifest  # noqa: E402
from slime_plugins.stackplanner.data.sampling import (  # noqa: E402
    EVALUATION_MANIFEST_SHA256,
    EVALUATION_SOURCES,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_quick_manifest(
    source: Path,
    output: Path,
    *,
    samples_per_dataset: int = 10,
    source_samples_per_dataset: int = 90,
    seed: int = 20260813,
    expected_source_sha256: str = EVALUATION_MANIFEST_SHA256,
) -> dict[str, Any]:
    if (
        samples_per_dataset < 1
        or source_samples_per_dataset < 1
        or samples_per_dataset > source_samples_per_dataset
    ):
        raise ValueError(
            "samples_per_dataset must be positive and no greater than "
            "source_samples_per_dataset"
        )
    actual_source_sha256 = _sha256(source)
    if actual_source_sha256 != expected_source_sha256:
        raise ValueError(
            "source evaluation manifest digest mismatch: "
            f"expected {expected_source_sha256}, got {actual_source_sha256}"
        )
    rows = load_manifest(source, samples_per_dataset=source_samples_per_dataset)
    by_dataset: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset"])].append(row)

    def rank(row: dict[str, Any]) -> tuple[str, str]:
        sample_id = str(row["sample_id"])
        digest = hashlib.sha256(f"{seed}\0{sample_id}".encode()).hexdigest()
        return digest, sample_id

    selected = [
        row
        for dataset in EVALUATION_SOURCES
        for row in sorted(by_dataset[dataset], key=rank)[:samples_per_dataset]
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
        newline="\n",
    )
    metadata = {
        "dataset_count": len(EVALUATION_SOURCES),
        "output_sha256": _sha256(output),
        "sample_count": len(selected),
        "samples_per_dataset": samples_per_dataset,
        "source_samples_per_dataset": source_samples_per_dataset,
        "seed": seed,
        "source_manifest": str(source.resolve()),
        "source_manifest_sha256": actual_source_sha256,
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-dataset", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    metadata = build_quick_manifest(
        args.source,
        args.output,
        samples_per_dataset=args.samples_per_dataset,
        seed=args.seed,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
