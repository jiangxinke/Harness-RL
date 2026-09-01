#!/usr/bin/env python3
"""Verify a Hugging Face checkpoint against a committed shard identity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(model_dir: Path, identity_path: Path) -> dict[str, object]:
    model_dir = model_dir.resolve()
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity.get("schema_version") != 1:
        raise ValueError("unsupported model identity schema")
    actual_lines: list[str] = []
    actual_files: list[dict[str, object]] = []
    for expected in identity["files"]:
        relative = expected["relative_path"]
        candidate = (model_dir / relative).resolve()
        if candidate.parent != model_dir or not candidate.is_file():
            raise ValueError(f"missing or unsafe checkpoint shard: {relative}")
        size = candidate.stat().st_size
        digest = _sha256(candidate)
        if size != expected["byte_size"] or digest != expected["sha256"]:
            raise ValueError(
                f"checkpoint shard mismatch: {relative}, bytes={size}, sha256={digest}"
            )
        actual_lines.append(f"{relative}\t{size}\t{digest}\n")
        actual_files.append(
            {"relative_path": relative, "byte_size": size, "sha256": digest}
        )
    aggregate = hashlib.sha256("".join(actual_lines).encode()).hexdigest()
    if aggregate != identity["base_checkpoint_sha256"]:
        raise ValueError(
            "aggregate checkpoint digest mismatch: "
            f"expected {identity['base_checkpoint_sha256']}, got {aggregate}"
        )
    return {
        "verified": True,
        "model_dir": str(model_dir),
        "model_id": identity["model_id"],
        "base_checkpoint_sha256": aggregate,
        "files": actual_files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.model_dir, args.identity)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
