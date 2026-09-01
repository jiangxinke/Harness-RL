#!/usr/bin/env python3
"""Thin repository-root entry point for the slime StackPlanner plugin."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "slime"))

from slime_plugins.stackplanner.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
