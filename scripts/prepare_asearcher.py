#!/usr/bin/env python3
"""Download the pinned ASearcher train file and convert it for slime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SLIME_SOURCE = str(ROOT / "slime")
sys.path.insert(0, SLIME_SOURCE)

from slime_plugins.stackplanner.data.asearcher import (  # noqa: E402
    CONVERTED_TRAIN_SHA256,
    TRAIN_FILE,
    TRAIN_ROWS,
    convert_training_file,
    download_training_file,
    sha256_file,
    verify_training_file,
)

sys.path.remove(SLIME_SOURCE)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir", type=Path, default=ROOT / "data/raw/asearcher_train"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/asearcher_train.jsonl"
    )
    parser.add_argument(
        "--source", type=Path, help="use an existing source file instead of downloading"
    )
    args = parser.parse_args()
    source = args.source or download_training_file(args.cache_dir / TRAIN_FILE)
    verify_training_file(source)
    count = convert_training_file(source, args.output)
    digest = sha256_file(args.output)
    if count != TRAIN_ROWS or digest != CONVERTED_TRAIN_SHA256:
        raise RuntimeError(
            f"converted data does not match the lock: rows={count}, sha256={digest}"
        )
    print(f"wrote {count} training rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
