#!/usr/bin/env python3
"""Materialize and hash Search-R1's split FAISS index and corpus."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
from pathlib import Path

CHUNK_SIZE = 16 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _concatenate(parts: list[Path], destination: Path) -> None:
    expected_size = sum(part.stat().st_size for part in parts)
    if destination.is_file() and destination.stat().st_size == expected_size:
        return
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with temporary.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=CHUNK_SIZE)
        output.flush()
        os.fsync(output.fileno())
    if temporary.stat().st_size != expected_size:
        raise RuntimeError(
            f"combined index has {temporary.stat().st_size} bytes; expected {expected_size}"
        )
    temporary.replace(destination)


def _decompress(source: Path, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > source.stat().st_size:
        return
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with gzip.open(source, "rb") as compressed, temporary.open("wb") as output:
        shutil.copyfileobj(compressed, output, length=CHUNK_SIZE)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--keep-parts", action="store_true")
    args = parser.parse_args()

    asset_dir = args.asset_dir.resolve()
    parts = [asset_dir / "part_aa", asset_dir / "part_ab"]
    compressed_corpus = asset_dir / "wiki-18.jsonl.gz"
    required = [*parts, compressed_corpus]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Search-R1 assets: {missing}")

    index = asset_dir / "e5_Flat.index"
    corpus = asset_dir / "wiki-18.jsonl"
    _concatenate(parts, index)
    _decompress(compressed_corpus, corpus)

    files = [index, corpus]
    manifest = {
        "schema_version": 1,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in files
        },
        "sources": {
            "index": "hf://datasets/PeterJinGo/wiki-18-e5-index/{part_aa,part_ab}",
            "corpus": "hf://datasets/PeterJinGo/wiki-18-corpus/wiki-18.jsonl.gz",
        },
    }
    manifest_path = asset_dir / "asset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if not args.keep_parts:
        for path in required:
            path.unlink()

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
