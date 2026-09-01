"""Run one live production web_search call."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS_SOURCE = str(ROOT / "StackPlanner2.0/backend/packages/harness")
sys.path.insert(0, HARNESS_SOURCE)

from deerflow.community.search_r1.tools import web_search_tool  # noqa: E402

sys.path.remove(HARNESS_SOURCE)


async def invoke_search(query: str) -> str:
    """Invoke the production async search tool through its supported API."""

    return await web_search_tool.ainvoke({"query": query})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="current capital of France")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    result = asyncio.run(invoke_search(args.query))
    record = {
        "query": args.query,
        "wall_seconds": round(time.monotonic() - started, 3),
        "result": result,
        "success": "http://" in result or "https://" in result,
    }
    text = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if record["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
