"""Portable LoongReflect entry point; ordinary Harness-RL training stays opt-in."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "slime"))

from slime_plugins.loongreflect.launcher import main

if __name__ == "__main__":
    raise SystemExit(main())
