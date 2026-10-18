"""Fit an external CUAD query-expansion profile using training rows only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "slime"))

from slime_plugins.dcc.data import load_tasks
from slime_plugins.dcc.retrieval import fit_retrieval_profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-terms", type=int, default=10)
    parser.add_argument("--min-examples", type=int, default=3)
    parser.add_argument("--min-fraction", type=float, default=0.08)
    parser.add_argument("--rrf-constant", type=float, default=60.0)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    tasks = [task for path in args.data for task in load_tasks(path)]
    profile = fit_retrieval_profile(
        tasks, max_terms=args.max_terms, min_examples=args.min_examples,
        min_fraction=args.min_fraction, rrf_constant=args.rrf_constant,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(profile, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"training": profile["training"], "questions": len(profile["questions"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
