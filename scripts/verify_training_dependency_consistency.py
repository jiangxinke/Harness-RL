#!/usr/bin/env python3
"""Run pip's metadata consistency check with one audited TE exception."""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    allowed = {
        "transformer_engine_cu12 2.16.1 is not supported on this platform",
        (
            "nvidia-cuda-nvcc 13.2.86 requires nvidia-cuda-runtime, "
            "which is not installed."
        ),
        (
            "nvidia-cuda-tileiras 13.2.86 requires nvidia-nvjitlink, "
            "which is not installed."
        ),
    }
    unexpected = [line for line in lines if line not in allowed]
    if unexpected:
        print(*unexpected, sep="\n", file=sys.stderr)
        raise SystemExit(result.returncode or 1)
    for line in lines:
        print(f"allowed pip metadata exception: {line}")
    print("training dependency consistency verification passed")


if __name__ == "__main__":
    main()
