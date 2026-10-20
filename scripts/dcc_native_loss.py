"""Opt-in native SFT mixture and detached, rank-local loss observations.

No generation, target editing, extra model calls, or distributed collectives.
The scalar download for logging batches the already-required loss values with
phase observations, replacing the legacy per-example loss.item() downloads.
"""
from __future__ import annotations

from collections import defaultdict

import torch

MIXTURE_LOSS_CONFIGURATION = {
    "version": "native-completion-json-mixture-sft-v1",
    "completion_weight": 0.8,
    "additional_json_weight": 0.2,
    "phase_boundaries": "validated-real-native-phase-ids",
    "example_weighting": "global-equal-example",
}
LOSS_TELEMETRY_CONFIGURATION = {
    "version": "native-per-example-phase-nll-v1",
    "aggregation_scope": "local-rank; sum group numerators/counts across rank logs for global means",
    "grouping": ["domain", "polarity", "stage"],
    "denominator": "examples, not tokens or rank means",
    "transport": "one batched detached CPU value transfer per local step; no extra collective",
    "additional_model_forward_or_backward": False,
}
_NLL_FIELDS = ("thought_nll", "json_nll", "completion_nll", "objective_nll")


def native_mixture_components(logprobs, example):
    """Return per-phase means on the complete, validated native token vector."""
    lengths = example.native_phase_lengths
    if (not isinstance(lengths, tuple) or len(lengths) != 2
            or any(type(length) is not int or length < 1 for length in lengths)
            or sum(lengths) != len(example.completion)
            or logprobs.ndim != 1 or logprobs.numel() != sum(lengths)):
        raise ValueError("Native mixture SFT requires validated native phase lengths")
    thought_length, _ = lengths
    if example.completion[thought_length - 1] != 151668:
        raise ValueError("Native thought phase must retain its actual closing ID")
    thought_nll = -logprobs[:thought_length].mean()
    json_nll = -logprobs[thought_length:].mean()
    completion_nll = -logprobs.mean()
    return thought_nll, json_nll, completion_nll


def native_completion_json_mixture_nll(logprobs, example):
    _, json_nll, completion_nll = native_mixture_components(logprobs, example)
    return 0.8 * completion_nll + 0.2 * json_nll


def native_sample_metadata(records, cuad_records):
    """Use already-validated TRAIN metadata; never infer polarity from output."""
    result = {}
    for domain, rows in (("web", records), ("cuad", cuad_records)):
        for record in rows:
            stage = record["stage"]
            polarity = "not_applicable"
            if domain == "cuad":
                positive = record.get("metadata", {}).get("positive")
                if type(positive) is not bool:
                    raise ValueError("Native CUAD loss telemetry requires recorded TRAIN polarity")
                polarity = "positive" if positive else "negative"
                stage = "cuad_tool_" + stage
            key = (record["task_id"], stage)
            value = {"domain": domain, "polarity": polarity, "stage": stage}
            if key in result and result[key] != value:
                raise ValueError("Conflicting native TRAIN loss telemetry metadata")
            result[key] = value
    return result


def merge_native_loss_summaries(previous, current):
    """CPU sums preserve exact example denominators, including unequal ranks."""
    grouped = defaultdict(lambda: {"examples": 0, **{field + "_sum": 0.0 for field in _NLL_FIELDS}})
    for summary in (previous, current):
        if summary is None:
            continue
        for row in summary["groups"]:
            key = (row["domain"], row["polarity"], row["stage"])
            target = grouped[key]
            target["examples"] += row["examples"]
            for field in _NLL_FIELDS:
                target[field + "_sum"] += row[field + "_sum"]
    rows = []
    for key, value in sorted(grouped.items()):
        row = dict(zip(("domain", "polarity", "stage"), key), **value)
        for field in _NLL_FIELDS:
            row[field + "_mean"] = row[field + "_sum"] / row["examples"]
        rows.append(row)
    return {"version": "native-per-example-phase-nll-v1", "aggregation_scope": "local-rank",
            "example_count": sum(row["examples"] for row in rows), "groups": rows}


def backward_native_mixture_batch(model, examples, *, logprob_fn, telemetry=None, sample_metadata=None):
    """One original forward/backward per example, followed by one value transfer."""
    if not examples:
        raise ValueError("Cannot train an empty token batch")
    labels = None
    if telemetry is not None:
        if sample_metadata is None:
            raise ValueError("Native loss telemetry requires validated TRAIN sample metadata")
        labels = []
        for example in examples:
            key = (example.task_id, example.stage)
            if key not in sample_metadata:
                raise ValueError("Native sample absent from validated TRAIN telemetry metadata")
            labels.append(sample_metadata[key])
    observed = []
    for example in examples:
        logprobs = logprob_fn(model, example.prompt, example.completion)
        thought_nll, json_nll, completion_nll = native_mixture_components(logprobs, example)
        objective = 0.8 * completion_nll + 0.2 * json_nll
        loss = objective / len(examples)
        # Keep the existing per-example finite-loss guard before backward.
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite supervised loss; refusing adapter update")
        loss.backward()
        values = (thought_nll, json_nll, completion_nll, objective) if telemetry is not None else (objective,)
        observed.append(torch.stack([value.detach() for value in values]))
    # No telemetry item()/CPU transfer inside the example loop. This replaces
    # the old loss.item() calls, rather than adding a second value download.
    downloaded = torch.stack(observed).to(device="cpu").tolist()
    total = sum(row[-1] for row in downloaded) / len(examples)
    if telemetry is not None:
        rows, groups = [], []
        for index, (example, label, values) in enumerate(zip(examples, labels, downloaded)):
            metrics = dict(zip(_NLL_FIELDS, values))
            rows.append({"sample_index": index, **label,
                         "thought_tokens": example.native_phase_lengths[0],
                         "json_tokens": example.native_phase_lengths[1], **metrics})
            groups.append({**label, "examples": 1,
                           **{field + "_sum": metrics[field] for field in _NLL_FIELDS}})
        telemetry.update({"version": "native-per-example-phase-nll-v1", "per_example": rows,
                          "summary": merge_native_loss_summaries(None, {"groups": groups})})
    return total
