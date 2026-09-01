#!/usr/bin/env python3
"""Build the pinned deterministic 630-question evaluation manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SLIME_SOURCE = str(ROOT / "slime")
sys.path.insert(0, SLIME_SOURCE)

from slime_plugins.stackplanner.data.sampling import (  # noqa: E402
    prepare_evaluation_manifest,
)

sys.path.remove(SLIME_SOURCE)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root", type=Path, default=ROOT / "data/raw/asearcher_test"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/evaluation_630.jsonl"
    )
    parser.add_argument("--count", type=int, default=90)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    manifest = prepare_evaluation_manifest(
        args.source_root,
        args.output,
        count=args.count,
        seed=args.seed,
        download=not args.no_download,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
