"""CPU-only sampling and label-cue audit for frozen public TRAIN agendas."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

from dcc_prepare_cuad_public_agendas import distribution, load_trainer


def length_auc(features, key):
    values = sorted((row[key], bool(row["positive"])) for row in features)
    positives = sum(label for _, label in values)
    negatives = len(values) - positives
    if not positives or not negatives:
        return None
    wins, negatives_before, position = 0.0, 0, 0
    while position < len(values):
        end = position
        while end < len(values) and values[end][0] == values[position][0]:
            end += 1
        group = values[position:end]
        p = sum(label for _, label in group)
        n = len(group) - p
        wins += p * (negatives_before + 0.5 * n)
        negatives_before += n
        position = end
    return wins / (positives * negatives)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuad-records", type=Path, required=True)
    parser.add_argument("--web-records", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--feature-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=range(8), action="append",
                        help="Inspect selected actual rank(s); omitted checks all eight")
    parser.add_argument("--cuad-tool-partial-stages", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    trainer = load_trainer()
    web = trainer.load_records(args.web_records)
    cuad = trainer.load_cuad_records(args.cuad_records)
    by_id = {row["task_id"]: row["metadata"]["positive"] for row in cuad}
    ids = sorted(by_id)
    feature_rows = json.loads(args.feature_audit.read_text())
    final_rows = [row for row in feature_rows if row["final_record_kept"]]
    all_groups, windows = Counter(), defaultdict(Counter)
    report = {"schema": "public-agendas-cpu-preflight-v1", "gpu_used": False,
              "model_weights_loaded": False, "train_only": True,
              "official_evaluation_read": False, "objective_unchanged": True,
              "cuad_tool_partial_stages": args.cuad_tool_partial_stages,
              "sampler_recipe_unchanged": not args.cuad_tool_partial_stages,
              "world_size": 8, "steps": 128, "samples_per_step": 4,
              "web_fraction": 0.5, "negative_fraction": 1 / 3, "ranks": [],
              "selected_ranks": args.rank if args.rank is not None else list(range(8)),
              "final_state_feature_audit": {
                  "scope": "Exploratory TRAIN label correlations; no independent validation claim",
                  "retained_final_agendas": len(final_rows),
                  "read_length_label_auc": length_auc(final_rows, "read_chars"),
                  "initial_hit_count_label_auc": length_auc(final_rows, "initial_hits"),
                  "read_chars": {label: distribution([row["read_chars"] for row in final_rows if row["positive"] == positive])
                                 for label, positive in [("positive", True), ("negative", False)]},
                  "schedule_by_label": {label: dict(Counter((row["variant"] + "/" + row["geometry"]) for row in final_rows if row["positive"] == positive))
                                        for label, positive in [("positive", True), ("negative", False)]}}}
    for rank in report["selected_ranks"]:
        try:
            sampler = trainer.MixedSampler(web, [], tokenizer, trainer.EpisodeRunner(max_steps=4, initial_chunks=8),
                rank=rank, world=8, seed=47, web_fraction=0.5, negative_fraction=1 / 3,
                max_prompt=16384, max_response=4096, batch_tokens=32768,
                cuad_records=cuad, cuad_tool_fraction=1.0,
                cuad_tool_partial_stages=args.cuad_tool_partial_stages)
            assert not sampler.skips, dict(sampler.skips)
            assigned = set(ids[rank::8])
            eligible = Counter()
            for positive, categories in sampler.cuad_tools.items():
                for stages in categories.values():
                    assert set(stages) == {"search", "read", "final"} or (
                        args.cuad_tool_partial_stages and positive and set(stages) == {"search", "read"})
                    for stage, examples in stages.items():
                        for example in examples:
                            assert example.task_id in assigned
                            eligible[f"{stage}_{positive}"] += 1
            counts, batch_sizes, unique_ids = Counter(), Counter(), set()
            for step in range(1, 129):
                examples = trainer.sample_batch(sampler, 4)
                assert examples and sum(example.tokens for example in examples) <= 32768
                batch_sizes[len(examples)] += 1
                for example in examples:
                    label = f"_{'positive' if by_id[example.task_id] else 'negative'}" if example.stage.startswith("cuad_tool") else ""
                    stage = example.stage + label
                    assert example.stage not in {"cuad_positive", "cuad_negative"}
                    counts[stage] += 1
                    all_groups[stage] += 1
                    unique_ids.add(example.task_id)
                    for checkpoint in [32, 64, 128]:
                        if step <= checkpoint:
                            windows[checkpoint][stage] += 1
            assert not sampler.skips
            item = {"rank": rank, "status": "passed", "eligible": dict(eligible),
                    "sampled_stages": dict(counts), "batch_sizes": dict(batch_sizes),
                    "unique_sampled_tasks": len(unique_ids), "skips": dict(sampler.skips),
                    "missing_stages": sampler.cuad_missing_stages,
                    "positive_categories": len(sampler.cuad_tools[True]),
                    "negative_categories": len(sampler.cuad_tools[False])}
        except (ValueError, AssertionError) as error:
            item = {"rank": rank, "status": "failed", "error": str(error)}
        report["ranks"].append(item)
        print(json.dumps(item), flush=True)
    report["status"] = "passed" if all(row["status"] == "passed" for row in report["ranks"]) else "failed"
    report["sampled_stages"] = dict(all_groups)
    report["checkpoint_windows"] = {str(k): dict(v) for k, v in windows.items()}
    report["total_sampled_examples"] = sum(all_groups.values())
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"event": "preflight_complete", "status": report["status"],
                      "total_examples": report["total_sampled_examples"],
                      "feature_audit": report["final_state_feature_audit"]}), flush=True)
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
