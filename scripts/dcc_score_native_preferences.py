#!/usr/bin/env python3
"""Offline TRAIN-only rewards after every original/G4 journal has closed.

This scorer never generates, trains, selects prefixes, repairs candidates,
changes the benchmark scorers or reads held-out labels. Runtime locations are
caller inputs; published provenance contains only public collection identities.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sys

sys.path[:0] = [str(Path(__file__).resolve().parent),
                str(Path(__file__).resolve().parents[1] / "slime")]

from dcc_collect_native_preferences import (
    FINAL_INTERVENTION, OBSERVATION, PROMPT_INTERVENTION_VERSION,
    VERSION as GENERATION_VERSION, final_intervention_messages, load_frozen,
    public_request_schemas, read_closed_g4, source_schema_profile,
)
from dcc_collect_native_train import (
    Journal, load_public_train, load_train_labels, read_closed, read_jsonl,
    require, web_train_evaluation, write_fresh_json,
)
from dcc_native_dpo_train import public_id, validate_prepared_collection_ownership
from dcc_native_sft import _strict_json
from slime_plugins.dcc.data import Task
from slime_plugins.dcc.episode import _WebEvidence
from slime_plugins.dcc.metrics import cuad_predictions, cuad_task_metrics
from slime_plugins.dcc.widesearch_metrics import evaluate_widesearch

VERSION = "native-final-train-reward-inputs-v1"
RECEIPT = "native-final-train-reward-receipt-v1"
REWARD_VERSIONS = {
    "cuad": "cuad-native-final-existing-f1-verbatim-v1",
    "widesearch": "web-native-final-row75-item25-original-profile-evidence-v1",
}
ROW_KEYS = {"collection_id", "approved_freeze", "generation_dir", "generation_world_size",
            "expected_source", "original_source_dir", "public_train", "train_labels", "reward_version"}


@dataclass(frozen=True)
class ClosedCollection:
    specification: dict
    freeze: dict
    prefixes: list
    identity: object
    generation: dict
    tasks: dict
    evidence: dict


def _original_binding(row, freeze, prefixes, *, source_cache, public_cache):
    approval = _strict_json((Path(row["approved_freeze"]) / "export_source_approval.json").read_text())
    require(isinstance(approval, dict) and set(approval) == {"version", "source_profile", "source_receipt",
            "source_world_size", "source_journals_closed", "labels_loaded", "source_completion_or_reward_retained"}
        and approval["version"] == GENERATION_VERSION and approval["source_journals_closed"] is True
        and approval["labels_loaded"] is False and approval["source_completion_or_reward_retained"] is False
        and type(approval["source_world_size"]) is int and approval["source_world_size"] > 0,
        "Independently approved original closed-source export provenance required")
    source_key = (row["original_source_dir"], json.dumps(approval, sort_keys=True))
    if source_key not in source_cache:
        source_cache[source_key] = read_closed(row["original_source_dir"], profile=approval["source_profile"],
            receipt=approval["source_receipt"], world_size=approval["source_world_size"])
    episodes, requests, _ = source_cache[source_key]
    episodes = {episode["episode_id"]: episode for episode in episodes}
    require(len(episodes) == len(source_cache[source_key][0]), "Duplicate original source episode identity")
    if row["public_train"] not in public_cache:
        public_cache[row["public_train"]] = {task.id: task for task in load_public_train(row["public_train"])}
    public = public_cache[row["public_train"]]
    lineage = list(read_jsonl(Path(row["approved_freeze"]) / "lineage.jsonl"))
    require(len(lineage) == len(prefixes) and [item["prefix_id"] for item in lineage] == freeze["prefix_ids"],
        "Original closed-source lineage differs from frozen public prefix order")
    schemas, tasks, guards = source_schema_profile(approval["source_profile"]), {}, {}
    for prefix, item in zip(prefixes, lineage):
        require(item["task_id"] == prefix.task_id and item["source_journals_closed"] is True
            and item["original_completion_retained"] is False and item["original_reward_used"] is False
            and item["sft_acceptance_used"] is False and item["schema_policy"] == freeze["schema_policy"],
            "Original public task lineage cannot depend on original answer/reward acceptance")
        episode = episodes[item["source_episode_id"]]
        require(item["source_trace_id"] in episode["trace_ids"]
            and episode["task_id"] == prefix.task_id and episode["benchmark"] == prefix.benchmark,
            "Original source episode does not own the frozen prefix")
        task = Task.from_dict(episode["public_task"])
        require(prefix.task_id in public and public[prefix.task_id].to_dict() == task.to_dict(),
            "Public TRAIN task differs from the actual original closed collection task")
        request, _ = requests[item["source_trace_id"]]
        messages = _strict_json(prefix.messages_json)
        source_messages = request["messages"]
        require(request["turn_index"] == item["turn"], "Original source turn lineage differs")
        original, final, fetched = public_request_schemas(task, source_messages,
            turn=item["turn"], schemas=schemas)
        require(original == request["options"]["json_schema"] == item["original_json_schema"]
            and final == item["generation_json_schema"] == _strict_json(prefix.schema_json)
            and fetched == item["fetched_sources"], "Original public schema/evidence lineage differs")
        intervention = original != final
        require(item.get("intervention_applied") is intervention
            and (not intervention or freeze["schema_policy"] == FINAL_INTERVENTION),
            "Original source final intervention lineage differs")
        expected_messages = final_intervention_messages(task, source_messages,
            applied=intervention, fetched_sources=fetched,
            web_explicit_review=schemas.web.explicit_review)
        prompt_intervention = item.get("prompt_intervention")
        require(isinstance(prompt_intervention, dict) and prompt_intervention == {
            "version": PROMPT_INTERVENTION_VERSION, "applied": intervention,
            "source_messages_preserved": True, "source_message_count": len(source_messages),
            "appended_message_count": len(expected_messages) - len(source_messages)}
            and prompt_intervention["applied"] is intervention
            and prompt_intervention["source_messages_preserved"] is True
            and type(prompt_intervention["source_message_count"]) is int
            and type(prompt_intervention["appended_message_count"]) is int,
            "Original source final prompt intervention lineage differs")
        require(messages == expected_messages,
            "Frozen public history differs from its actual original closed request and final control")
        tasks[prefix.task_id] = task
        if task.benchmark == "widesearch":
            guard = _WebEvidence()
            # public_request_schemas already proves exact paired action/observation
            # serialization, turn counters, allowed tools and absence of old finals.
            for index, message in enumerate(source_messages):
                if message["role"] != "assistant":
                    continue
                action = _strict_json(message["content"])
                observed = source_messages[index + 1]["content"]
                payload = observed[len(OBSERVATION):].rsplit("\nRemaining turns: ", 1)[0]
                guard.observe(action["tool"], _strict_json(payload))
            require(sorted(guard.fetched_sources) == fetched, "Actual historical evidence replay differs")
            guards[prefix.prefix_id] = guard
    return tasks, guards


def prepare_collections(manifest, tokenizer):
    """No label function/file access occurs until this whole function succeeds."""
    require(isinstance(manifest, dict) and set(manifest) == {"version", "base_derivative_id", "tokenizer_id", "collections"}
        and manifest["version"] == VERSION and isinstance(manifest["collections"], list)
        and bool(manifest["collections"]), "Explicit TRAIN reward collection manifest required")
    public_id(manifest["base_derivative_id"]); public_id(manifest["tokenizer_id"])
    prepared, source_cache, public_cache = [], {}, {}
    for row in manifest["collections"]:
        require(isinstance(row, dict) and set(row) == ROW_KEYS
            and type(row["generation_world_size"]) is int and row["generation_world_size"] > 0
            and isinstance(row["expected_source"], str) and re.fullmatch(r"[0-9a-f]{40}", row["expected_source"]),
            "Explicit separately approved original/G4/public/TRAIN-label inputs required")
        public_id(row["collection_id"])
        freeze, prefixes, identity = load_frozen(row["approved_freeze"], tokenizer)
        require(identity.derivative_id == manifest["base_derivative_id"], "Approved reward base differs from collection")
        domains = {prefix.benchmark for prefix in prefixes}
        require(len(domains) == 1 and row["reward_version"] == REWARD_VERSIONS[next(iter(domains))],
            "Each separately frozen domain requires its explicit fixed reward version")
        closed = read_closed_g4(row["generation_dir"], freeze=freeze, prefixes=prefixes, tokenizer=tokenizer,
            world_size=row["generation_world_size"], expected_source=row["expected_source"])
        require(closed.get("generation_closed_before_scoring") is True, "Closed whole G4 generation required")
        tasks, evidence = _original_binding(row, freeze, prefixes, source_cache=source_cache, public_cache=public_cache)
        prepared.append(ClosedCollection(row, freeze, prefixes, identity, closed, tasks, evidence))
    validate_prepared_collection_ownership([(item.specification["collection_id"], None,
        (item.freeze, item.prefixes, item.identity, item.generation)) for item in prepared])
    return prepared


def cuad_reward(task, label, answer):
    metrics = cuad_task_metrics(task.id, label["gold"], answer)
    spans = cuad_predictions(answer)
    normalized_document = " ".join(task.document.split())
    grounded = [" ".join(item["text"].split()) in normalized_document for item in spans]
    grounding = sum(grounded) / len(grounded) if grounded else 1.0
    return float(metrics["f1"]) * grounding, {"cuad_f1": float(metrics["f1"]),
        "verbatim_span_fraction": grounding, "prediction_spans": len(spans),
        "negative_gold": not bool(label["gold"])}


def web_reward(task, label, answer, guard):
    evaluation = web_train_evaluation(label, task.to_dict())
    reasons = guard.reject_reasons(task, answer)
    compound = evaluation["unique_columns"] == evaluation["required"]
    if reasons:
        return 0.0, {"evidence_guard_rejections": reasons, "f1_by_row": None, "f1_by_item": None,
            "required_only_compound_key": compound, "evaluation_protocol": "evidence_guard_rejected"}
    score = evaluate_widesearch(answer, label["gold"], evaluation)
    require(score["evaluation_protocol"] == "strict_exact_proxy",
        "Offline TRAIN reward requires the existing deterministic exact scoring profile")
    row, item = float(score["f1_by_row"]), float(score["f1_by_item"])
    return .75 * row + .25 * item, {"evidence_guard_rejections": [], "f1_by_row": row, "f1_by_item": item,
        "required_only_compound_key": compound, "evaluation_protocol": score["evaluation_protocol"],
        "scoring_message": score.get("msg", ""),
        **{key: score[key] for key in ("num_pred_rows", "num_gold_rows") if key in score}}


def score_prepared_collections(prepared):
    """This is the first possible TRAIN label access after all public validation."""
    labels = [(item, load_train_labels(item.specification["train_labels"],
        {prefix.task_id: prefix.benchmark for prefix in item.prefixes})) for item in prepared]
    results = []
    for item, indexed in labels:
        prefixes = {prefix.prefix_id: prefix for prefix in item.prefixes}
        receipts, audit = [], []
        for envelope in item.generation["candidates"]:
            prefix = prefixes[envelope["prefix_id"]]
            bundle = envelope["native_bundle"]
            require(bundle["record"]["task_id"] == prefix.task_id, "Closed candidate public task identity differs")
            answer = _strict_json(bundle["native_trace"]["final_text"])["answer"]
            task, label = item.tasks[prefix.task_id], indexed[prefix.task_id]
            if prefix.benchmark == "cuad":
                reward, components = cuad_reward(task, label, answer)
            else:
                reward, components = web_reward(task, label, answer, item.evidence[prefix.prefix_id])
            require(type(reward) in (float, int) and math.isfinite(reward) and 0 <= reward <= 1,
                "Existing TRAIN score produced an invalid reward")
            receipt = {"contract": RECEIPT, "task_id": prefix.task_id,
                "trace_id": bundle["collector"]["trace_id"], "prefix_id": prefix.prefix_id,
                "harness_id": prefix.harness_id, "reward_version": item.specification["reward_version"],
                "value": reward, "generation_closed_before_scoring": True, "split": "train"}
            receipts.append(receipt)
            audit.append({"task_id": prefix.task_id, "prefix_id": prefix.prefix_id,
                "trace_id": receipt["trace_id"], "reward_version": receipt["reward_version"],
                "reward": reward, "components": components, "candidate_rewritten": False})
        require(len({row["trace_id"] for row in receipts}) == len(receipts)
            == len(item.generation["candidates"]), "Reward receipts must exactly cover valid physical candidates")
        values = [row["value"] for row in receipts]
        counts = {"prefixes": len(item.prefixes), "attempts": len(item.generation["attempts"]),
            "valid_candidates": len(receipts), "failed_attempts": len(item.generation["attempts"]) - len(receipts),
            "rewarded_candidates": len(receipts), "zero_reward": sum(value == 0 for value in values),
            "positive_reward": sum(value > 0 for value in values), "reward_sum": sum(values),
            "reward_mean": sum(values) / len(values) if values else None,
            "row_item_equal_candidates": sum(row["components"].get("f1_by_row") is not None
                and row["components"].get("f1_by_row") == row["components"].get("f1_by_item") for row in audit),
            "guard_rejected_candidates": sum(bool(row["components"].get("evidence_guard_rejections")) for row in audit)}
        results.append((item, receipts, audit, counts))
    return results


def publish_rewards(results, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    profiles = []
    for item, receipts, audit, counts in results:
        # IDs may use portable slash namespaces; never use them as disk paths.
        output = destination / f"collection_{len(profiles):03d}"
        output.mkdir()
        for filename, rows in (("receipts.jsonl", receipts), ("audit.jsonl", audit)):
            journal = Journal(output / filename)
            try:
                for row in rows: journal.write(row)
            finally:
                journal.close()
        profile = {"collection_id": item.specification["collection_id"],
            "reward_version": item.specification["reward_version"], "public_freeze": item.freeze,
            "generation_closed_before_scoring": True, "original_source_closed_before_scoring": True,
            "all_collection_ownership_checked_before_labels": True, "counts": counts,
            "receipt_file": f"{output.name}/receipts.jsonl"}
        write_fresh_json(output / "reward_complete.json", dict(profile, complete=True))
        profiles.append(profile)
    totals = {key: sum(profile["counts"][key] for profile in profiles) for key in (
        "prefixes", "attempts", "valid_candidates", "failed_attempts", "rewarded_candidates", "zero_reward",
        "positive_reward", "reward_sum", "row_item_equal_candidates", "guard_rejected_candidates")}
    write_fresh_json(destination / "reward_complete.json", {"version": VERSION, "complete": True,
        "collections": profiles, "fullcounts": totals, "heldout_labels_loaded": False})
    return profiles


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-collections", required=True)
    parser.add_argument("--tokenizer", required=True, help="Actual frozen local tokenizer; no model weights loaded")
    parser.add_argument("--output-dir", required=True, help="Fresh private artifact directory")
    args = parser.parse_args(argv)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    manifest = _strict_json(Path(args.approved_collections).read_text())
    prepared = prepare_collections(manifest, tokenizer)
    publish_rewards(score_prepared_collections(prepared), args.output_dir)


if __name__ == "__main__":
    main()
