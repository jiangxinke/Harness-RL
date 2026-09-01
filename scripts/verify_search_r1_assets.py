#!/usr/bin/env python3
"""Verify immutable Search-R1 index assets against their published manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_assets(asset_root: Path, manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Search-R1 asset manifest must use schema_version 1")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Search-R1 asset manifest has no files")

    verified: list[dict[str, Any]] = []
    for name, identity in sorted(files.items()):
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError(f"unsafe Search-R1 asset name: {name!r}")
        if not isinstance(identity, dict):
            raise ValueError(f"invalid identity for Search-R1 asset: {name}")
        expected_bytes = identity.get("bytes")
        expected_sha256 = identity.get("sha256")
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise ValueError(f"invalid byte count for Search-R1 asset: {name}")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError(f"invalid SHA-256 for Search-R1 asset: {name}")

        path = asset_root / name
        if not path.is_file():
            raise FileNotFoundError(f"Search-R1 asset is missing: {path}")
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"Search-R1 asset size mismatch for {path}: "
                f"expected {expected_bytes}, got {actual_bytes}"
            )
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Search-R1 asset SHA-256 mismatch for {path}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        verified.append(
            {"path": str(path), "bytes": actual_bytes, "sha256": actual_sha256}
        )
    return verified


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    for identity in verify_assets(args.asset_root, args.manifest):
        print(
            "verified_search_r1_asset="
            f"{identity['path']} bytes={identity['bytes']} sha256={identity['sha256']}"
        )


if __name__ == "__main__":
    main()
