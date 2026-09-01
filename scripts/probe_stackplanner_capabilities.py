"""Probe StackPlanner capability localization on exact production actor tokens.

The script is intended to be launched with ``torchrun``. Every rank loads one
frozen copy of the target model and evaluates a disjoint set of production
trajectory records. Rank zero aggregates representation, normalized hidden
gradient, causal intervention, held-out, and multi-seed evidence into the soft
layer map consumed by experiment C.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SLIME_SOURCE = str(ROOT / "slime")
sys.path.insert(0, SLIME_SOURCE)

from slime_plugins.stackplanner.capability.all_agent_output import (  # noqa: E402
    ALL_AGENT_OUTPUT_CAPABILITIES,
    ALL_AGENT_OUTPUT_TAXONOMY_VERSION,
)
from slime_plugins.stackplanner.capability.all_agent_output import (  # noqa: E402
    annotate_turn as annotate_all_agent_output_turn,
)
from slime_plugins.stackplanner.capability.annotator import (  # noqa: E402
    annotate_turn as annotate_central_turn,
)
from slime_plugins.stackplanner.capability.taxonomy import (  # noqa: E402
    CAPABILITIES as CENTRAL_CAPABILITIES,
)
from slime_plugins.stackplanner.capability.taxonomy import (  # noqa: E402
    CAPABILITY_TAXONOMY_VERSION as CENTRAL_TAXONOMY_VERSION,
)
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

sys.path.remove(SLIME_SOURCE)

PROBE_SCHEMA_VERSION = 1
CAPABILITIES = CENTRAL_CAPABILITIES
CAPABILITY_TAXONOMY_VERSION = CENTRAL_TAXONOMY_VERSION
ANNOTATE_TURN = annotate_central_turn


@dataclass(frozen=True)
class ProbeRecord:
    record_id: str
    tokens: list[int]
    response_length: int
    loss_mask: list[int]
    labels: list[str]
    reward: float
    source: str
    role: str = "central"

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset(
            label
            for enabled, label in zip(self.loss_mask, self.labels, strict=True)
            if enabled and label in CAPABILITIES
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split(record_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{record_id}".encode()).digest()
    return "heldout" if int.from_bytes(digest[:4], "big") % 5 == 0 else "train"


def _load_records(
    paths: list[Path], tokenizer: Any, min_reward: float
) -> tuple[list[ProbeRecord], list[dict[str, str]], set[str]]:
    records: list[ProbeRecord] = []
    rejected: list[dict[str, str]] = []
    source_actor_roles: set[str] = set()
    for path in paths:
        payload = torch.load(path, weights_only=False, map_location="cpu")
        for index, sample in enumerate(payload.get("samples", [])):
            try:
                metadata = sample.get("metadata") or {}
                train_metadata = sample.get("train_metadata") or {}
                role = str(metadata.get("role") or "central").strip().lower()
                reward = float(sample["reward"] or 0.0)
                response_length = int(sample["response_length"] or 0)
                tokens = [int(value) for value in sample["tokens"]]
                loss_mask = [int(bool(value)) for value in sample["loss_mask"]]
                if reward <= min_reward or response_length < 1 or not any(loss_mask):
                    continue
                if len(loss_mask) != response_length or response_length > len(tokens):
                    raise ValueError("misaligned response tokens/loss mask")
                response_ids = tokens[-response_length:]
                raw_output = tokenizer.decode(
                    response_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                labels = train_metadata.get("capability_labels")
                actor_roles = train_metadata.get("actor_roles")
                if (
                    CAPABILITY_TAXONOMY_VERSION == ALL_AGENT_OUTPUT_TAXONOMY_VERSION
                    and isinstance(labels, list)
                    and isinstance(actor_roles, list)
                ):
                    if (
                        len(labels) != response_length
                        or len(actor_roles) != response_length
                        or any(
                            enabled and actor_role != role
                            for enabled, actor_role in zip(
                                loss_mask, actor_roles, strict=True
                            )
                        )
                    ):
                        raise ValueError(
                            "stored all-agent labels or explicit actor roles are misaligned"
                        )
                else:
                    labels = ANNOTATE_TURN(
                        tokenizer=tokenizer,
                        output_ids=response_ids,
                        raw_output=raw_output,
                        parsed=None,
                        tools_schema=[],
                        annotation_context={"role": role},
                    )["capability_labels"]
                identity = hashlib.sha256(
                    json.dumps(
                        {
                            "source_sha256": _sha256(path),
                            "sample_index": index,
                            "tokens": tokens,
                            "response_length": response_length,
                        },
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                record = ProbeRecord(
                    record_id=identity,
                    tokens=tokens,
                    response_length=response_length,
                    loss_mask=loss_mask,
                    labels=labels,
                    reward=reward,
                    source=f"{path}:{index}",
                    role=role,
                )
                if record.capabilities:
                    records.append(record)
                    source_actor_roles.add(role)
            except (KeyError, TypeError, ValueError) as exc:
                rejected.append({"source": f"{path}:{index}", "reason": str(exc)})
    if not records:
        raise RuntimeError("no positive-reward production records were probeable")
    return records, rejected, source_actor_roles


def _select_records(
    records: list[ProbeRecord], seeds: tuple[int, ...], quota: int
) -> dict[tuple[int, str], list[ProbeRecord]]:
    selected: dict[tuple[int, str], list[ProbeRecord]] = {}
    source_roles = {record.role for record in records}
    for seed in seeds:
        ordered = sorted(
            records,
            key=lambda item: hashlib.sha256(
                f"{seed}:order:{item.record_id}".encode()
            ).digest(),
        )
        for split in ("train", "heldout"):
            counts = {capability: 0 for capability in CAPABILITIES}
            roles: set[str] = set()
            chosen: list[ProbeRecord] = []
            for record in ordered:
                if _split(record.record_id, seed) != split:
                    continue
                useful = [
                    capability
                    for capability in record.capabilities
                    if counts[capability] < quota
                ]
                if not useful and record.role in roles:
                    continue
                chosen.append(record)
                roles.add(record.role)
                for capability in useful:
                    counts[capability] += 1
                if min(counts.values()) >= quota and roles == source_roles:
                    break
            missing = {key: value for key, value in counts.items() if value < quota}
            missing_roles = sorted(source_roles - roles)
            if missing or missing_roles:
                raise RuntimeError(
                    f"insufficient {split} production coverage for seed {seed}: "
                    f"capabilities={missing}, actor_roles={missing_roles}; "
                    f"candidates={len(records)}"
                )
            selected[(seed, split)] = chosen
    return selected


def _target_positions(
    record: ProbeRecord, capability: str, max_sequence_length: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    start = max(0, len(record.tokens) - max_sequence_length)
    input_ids = torch.tensor(record.tokens[start:], dtype=torch.long)
    response_start = len(record.tokens) - record.response_length
    positions = [
        response_start + offset - start
        for offset, (enabled, label) in enumerate(
            zip(record.loss_mask, record.labels, strict=True)
        )
        if enabled and label == capability and response_start + offset - start > 0
    ]
    if not positions:
        raise ValueError(
            f"record {record.record_id} has no retained target for {capability}"
        )
    targets = input_ids[positions]
    predictors = torch.tensor([value - 1 for value in positions], dtype=torch.long)
    return input_ids, predictors, targets


def _activation_probe(
    model: Any, record: ProbeRecord, max_sequence_length: int, device: torch.device
) -> dict[str, dict[str, Any]]:
    input_ids = torch.tensor(
        record.tokens[-max_sequence_length:], dtype=torch.long, device=device
    ).unsqueeze(0)
    response_start = len(input_ids[0]) - record.response_length
    positions: dict[str, torch.Tensor] = {}
    for capability in record.capabilities:
        values = [
            response_start + offset
            for offset, (enabled, label) in enumerate(
                zip(record.loss_mask, record.labels, strict=True)
            )
            if enabled and label == capability and response_start + offset >= 0
        ]
        if values:
            positions[capability] = torch.tensor(values, device=device)
    output: dict[str, dict[str, Any]] = {
        capability: {"centroid": [], "positive_fraction": [], "mean_abs": []}
        for capability in positions
    }
    handles = []

    def hook(_module: Any, _inputs: Any, layer_output: Any) -> None:
        hidden = layer_output[0] if isinstance(layer_output, tuple) else layer_output
        for capability, indices in positions.items():
            selected = hidden[0].index_select(0, indices).float()
            output[capability]["centroid"].append(selected.mean(0).cpu())
            output[capability]["positive_fraction"].append(
                float((selected > 0).float().mean())
            )
            output[capability]["mean_abs"].append(float(selected.abs().mean()))

    for layer in model.model.layers:
        handles.append(layer.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            model.model(input_ids=input_ids, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    return output


def _gradient_probe(
    model: Any,
    record: ProbeRecord,
    capability: str,
    max_sequence_length: int,
    device: torch.device,
) -> list[float]:
    input_ids, predictors, targets = _target_positions(
        record, capability, max_sequence_length
    )
    input_ids = input_ids.to(device).unsqueeze(0)
    predictors = predictors.to(device)
    targets = targets.to(device)
    embeddings = model.model.embed_tokens(input_ids).detach().requires_grad_(True)
    captured: list[torch.Tensor] = []
    handles = []

    def hook(_module: Any, _inputs: Any, layer_output: Any) -> None:
        hidden = layer_output[0] if isinstance(layer_output, tuple) else layer_output
        hidden.retain_grad()
        captured.append(hidden)

    for layer in model.model.layers:
        handles.append(layer.register_forward_hook(hook))
    try:
        output = model.model(
            inputs_embeds=embeddings, use_cache=False, return_dict=True
        ).last_hidden_state
        logits = model.lm_head(output[0].index_select(0, predictors)).float()
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        norms = []
        for hidden in captured:
            gradient = hidden.grad[0].index_select(0, predictors).float()
            norms.append(float(torch.linalg.vector_norm(gradient, dim=-1).mean()))
        return norms
    finally:
        for handle in handles:
            handle.remove()
        embeddings.grad = None


def _target_nll(
    model: Any,
    input_ids: torch.Tensor,
    predictors: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    output = model.model(input_ids=input_ids, use_cache=False, return_dict=True)
    logits = model.lm_head(
        output.last_hidden_state[0].index_select(0, predictors)
    ).float()
    return float(F.cross_entropy(logits, targets))


def _intervention_probe(
    model: Any,
    record: ProbeRecord,
    capability: str,
    max_sequence_length: int,
    strength: float,
    device: torch.device,
) -> dict[str, Any]:
    input_ids, predictors, targets = _target_positions(
        record, capability, max_sequence_length
    )
    input_ids = input_ids.to(device).unsqueeze(0)
    predictors = predictors.to(device)
    targets = targets.to(device)
    with torch.inference_mode():
        baseline = _target_nll(model, input_ids, predictors, targets)
        deltas = []
        for layer in model.model.layers:

            def intervene(_module: Any, _inputs: Any, layer_output: Any) -> Any:
                hidden = (
                    layer_output[0] if isinstance(layer_output, tuple) else layer_output
                )
                modified = hidden.clone()
                modified[0, predictors] *= 1.0 - strength
                if isinstance(layer_output, tuple):
                    return (modified, *layer_output[1:])
                return modified

            handle = layer.register_forward_hook(intervene)
            try:
                perturbed = _target_nll(model, input_ids, predictors, targets)
            finally:
                handle.remove()
            deltas.append(perturbed - baseline)
    return {
        "baseline_nll": baseline,
        "delta_nll": deltas,
        "intervention_strength": strength,
    }


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.all():
        raise ValueError("probe metric contains a non-finite value")
    low, high = float(values.min()), float(values.max())
    if math.isclose(low, high, rel_tol=1e-9, abs_tol=1e-12):
        return np.ones_like(values)
    return (values - low) / (high - low)


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        output = np.empty_like(order, dtype=np.float64)
        output[order] = np.arange(len(values), dtype=np.float64)
        return output

    left_ranks, right_ranks = ranks(left), ranks(right)
    if np.std(left_ranks) == 0 or np.std(right_ranks) == 0:
        return 1.0 if np.allclose(left, right) else 0.0
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _aggregate(
    partials: list[dict[str, Any]],
    num_layers: int,
    capability_token_fractions: dict[str, float] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, float]], dict[str, Any]]:
    activations = [item for part in partials for item in part["activations"]]
    gradients = [item for part in partials for item in part["gradients"]]
    interventions = [item for part in partials for item in part["interventions"]]
    profile_keys = sorted({(item["seed"], item["split"]) for item in activations})
    profiles: dict[tuple[int, str, str], dict[str, np.ndarray]] = {}
    for seed, split in profile_keys:
        centroids: dict[str, np.ndarray] = {}
        for capability in CAPABILITIES:
            matching = [
                np.stack(item["centroid"])
                for item in activations
                if item["seed"] == seed
                and item["split"] == split
                and item["capability"] == capability
            ]
            if not matching:
                raise RuntimeError(
                    f"missing representation evidence: {seed}/{split}/{capability}"
                )
            centroids[capability] = np.mean(matching, axis=0)
        for capability in CAPABILITIES:
            other = np.mean(
                [value for name, value in centroids.items() if name != capability],
                axis=0,
            )
            representation = np.linalg.norm(
                centroids[capability] - other, axis=-1
            ) / math.sqrt(centroids[capability].shape[-1])
            gradient_matches = [
                np.asarray(item["gradient_norm"])
                for item in gradients
                if item["seed"] == seed
                and item["split"] == split
                and item["capability"] == capability
            ]
            gradient = (
                np.mean(gradient_matches, axis=0)
                if gradient_matches
                else np.zeros(num_layers)
            )
            intervention_matches = [
                np.abs(np.asarray(item["delta_nll"]))
                for item in interventions
                if item["split"] == split and item["capability"] == capability
            ]
            intervention = (
                np.mean(intervention_matches, axis=0)
                if intervention_matches
                else np.zeros(num_layers)
            )
            components = [0.35 * _normalize(representation)]
            component_weight = 0.35
            if np.any(gradient):
                components.append(0.35 * _normalize(gradient))
                component_weight += 0.35
            if np.any(intervention):
                components.append(0.30 * _normalize(intervention))
                component_weight += 0.30
            composite = sum(components) / component_weight
            profiles[(seed, split, capability)] = {
                "representation": representation,
                "gradient": gradient,
                "intervention": intervention,
                "composite": composite,
            }

    averaged: dict[str, np.ndarray] = {}
    for capability in CAPABILITIES:
        matches = [
            metrics["composite"]
            for (seed, split, name), metrics in profiles.items()
            if name == capability
        ]
        averaged[capability] = np.mean(matches, axis=0)

    fractions = capability_token_fractions or {
        capability: 1.0 / len(CAPABILITIES) for capability in CAPABILITIES
    }
    if set(fractions) != set(CAPABILITIES) or not math.isclose(
        sum(fractions.values()), 1.0, rel_tol=1e-6, abs_tol=1e-9
    ):
        raise ValueError(
            "capability token fractions must cover the taxonomy and sum to one"
        )
    normalized_profiles = {
        capability: _normalize(values) for capability, values in averaged.items()
    }
    weights: dict[str, dict[str, float]] = {}
    for layer_index in range(num_layers):
        raw = {
            capability: float(normalized_profiles[capability][layer_index])
            for capability in CAPABILITIES
        }
        weighted_center = sum(
            fractions[capability] * raw[capability] for capability in CAPABILITIES
        )
        residuals = {
            capability: raw[capability] - weighted_center for capability in CAPABILITIES
        }
        max_residual = max(abs(value) for value in residuals.values())
        if max_residual > 0:
            residuals = {
                capability: value / max_residual
                for capability, value in residuals.items()
            }
            # Recenter after max normalization to keep the token-weighted mean
            # exactly at the baseline gradient scale.
            residual_center = sum(
                fractions[capability] * residuals[capability]
                for capability in CAPABILITIES
            )
            residuals = {
                capability: value - residual_center
                for capability, value in residuals.items()
            }
        weights[f"layers.{layer_index}"] = {
            capability: float(min(1.2, max(0.8, 1.0 + 0.2 * residuals[capability])))
            for capability in CAPABILITIES
        }

    seed_correlations = []
    train_heldout_correlations = []
    seeds = sorted({key[0] for key in profiles})
    for capability in CAPABILITIES:
        for split in ("train", "heldout"):
            for index, seed in enumerate(seeds):
                for other_seed in seeds[index + 1 :]:
                    seed_correlations.append(
                        _rank_correlation(
                            profiles[(seed, split, capability)]["composite"],
                            profiles[(other_seed, split, capability)]["composite"],
                        )
                    )
        for seed in seeds:
            train_heldout_correlations.append(
                _rank_correlation(
                    profiles[(seed, "train", capability)]["composite"],
                    profiles[(seed, "heldout", capability)]["composite"],
                )
            )
    stability = {
        "multi_seed_spearman_mean": float(np.mean(seed_correlations)),
        "multi_seed_spearman_min": float(np.min(seed_correlations)),
        "train_heldout_spearman_mean": float(np.mean(train_heldout_correlations)),
        "train_heldout_spearman_min": float(np.min(train_heldout_correlations)),
    }
    checks = {
        "all_capabilities_covered": all(
            any(item["capability"] == capability for item in activations)
            for capability in CAPABILITIES
        ),
        "gradient_signal_nonzero": all(
            any(
                item["capability"] == capability and max(item["gradient_norm"]) > 0
                for item in gradients
            )
            for capability in CAPABILITIES
        ),
        "causal_signal_nonzero": all(
            any(
                item["capability"] == capability
                and max(abs(value) for value in item["delta_nll"]) > 1e-7
                for item in interventions
            )
            for capability in CAPABILITIES
        ),
        "multi_seed_mean_nonnegative": stability["multi_seed_spearman_mean"] >= 0.0,
        "heldout_mean_nonnegative": stability["train_heldout_spearman_mean"] >= 0.0,
    }
    stability["checks"] = checks
    stability["passed"] = all(checks.values())
    raw = {
        "activations": activations,
        "gradients": gradients,
        "interventions": interventions,
        "profiles": {
            f"{seed}/{split}/{capability}": {
                metric: values.tolist() for metric, values in metrics.items()
            }
            for (seed, split, capability), metrics in profiles.items()
        },
    }
    averaged_json = {
        capability: {
            "composite": averaged[capability].tolist(),
            "weight": [
                weights[f"layers.{index}"][capability] for index in range(num_layers)
            ],
        }
        for capability in CAPABILITIES
    }
    return (
        raw,
        weights,
        {
            "profiles": averaged_json,
            "stability": stability,
            "routing_calibration": {
                "method": "token-fraction-centered-residual-v1",
                "capability_token_fractions": fractions,
                "baseline_weight": 1.0,
                "maximum_absolute_residual": 0.2,
            },
        },
    )


def _write_svg(path: Path, weights: dict[str, dict[str, float]]) -> None:
    cell_w, cell_h, left, top = 20, 34, 220, 60
    width = left + len(weights) * cell_w + 40
    height = top + len(CAPABILITIES) * cell_h + 80
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="28" font-family="sans-serif" font-size="18" font-weight="bold">Qwen2.5 capability-to-parameter-layer routing</text>',
    ]
    for index in range(len(weights)):
        if index % 5 == 0 or index == len(weights) - 1:
            lines.append(
                f'<text x="{left + index * cell_w + 4}" y="{top - 8}" font-family="monospace" font-size="10">{index}</text>'
            )
    for row, capability in enumerate(CAPABILITIES):
        y = top + row * cell_h
        lines.append(
            f'<text x="12" y="{y + 21}" font-family="sans-serif" font-size="13">{capability}</text>'
        )
        for layer in range(len(weights)):
            value = weights[f"layers.{layer}"][capability]
            blue = int(250 - 165 * value)
            red = int(248 - 210 * value)
            green = int(250 - 105 * value)
            lines.append(
                f'<rect x="{left + layer * cell_w}" y="{y}" width="{cell_w - 1}" height="{cell_h - 2}" fill="rgb({red},{green},{blue})"><title>{capability}, layer {layer}: {value:.6f}</title></rect>'
            )
    lines.extend(
        [
            f'<text x="{left}" y="{height - 28}" font-family="sans-serif" font-size="12">Transformer parameter layer →</text>',
            f'<text x="{width - 250}" y="{height - 28}" font-family="sans-serif" font-size="11">light = shared update; dark = capability-local update</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_outputs(
    args: argparse.Namespace,
    records: list[ProbeRecord],
    rejected: list[dict[str, str]],
    selected: dict[tuple[int, str], list[ProbeRecord]],
    partials: list[dict[str, Any]],
    source_actor_roles: set[str],
) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    capability_token_counts = {
        capability: sum(
            enabled and label == capability
            for record in records
            for enabled, label in zip(record.loss_mask, record.labels, strict=True)
        )
        for capability in CAPABILITIES
    }
    total_capability_tokens = sum(capability_token_counts.values())
    if total_capability_tokens <= 0 or any(
        count <= 0 for count in capability_token_counts.values()
    ):
        raise RuntimeError(
            "probe corpus does not contain trainable tokens for every capability"
        )
    capability_token_fractions = {
        capability: count / total_capability_tokens
        for capability, count in capability_token_counts.items()
    }
    raw, weights, summary = _aggregate(
        partials, args.num_layers, capability_token_fractions
    )
    raw_path = output / "raw_probe_metrics.pt"
    torch.save(raw, raw_path)
    layer_map = {
        "model_id": str(Path(args.model).resolve()),
        "checkpoint_revision": args.model_digest,
        "num_hidden_layers": args.num_layers,
        "taxonomy_version": CAPABILITY_TAXONOMY_VERSION,
        "weights": weights,
    }
    (output / "layer_map.json").write_text(
        json.dumps(layer_map, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    taxonomy = {
        "version": CAPABILITY_TAXONOMY_VERSION,
        "capabilities": list(CAPABILITIES),
        "annotation": "exact SGLang served-token semantic partition",
    }
    (output / "taxonomy.json").write_text(
        json.dumps(taxonomy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "probe_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "layer_profiles.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("capability", "layer", "composite", "soft_weight"))
        for capability, profile in summary["profiles"].items():
            for layer, (composite, weight) in enumerate(
                zip(profile["composite"], profile["weight"], strict=True)
            ):
                writer.writerow((capability, layer, composite, weight))
    _write_svg(output / "parameter_decoupling.svg", weights)

    source_paths = sorted({Path(record.source.rsplit(":", 1)[0]) for record in records})
    selected_counts = {
        f"{seed}/{split}": {
            capability: sum(
                capability in record.capabilities for record in selected_records
            )
            for capability in CAPABILITIES
        }
        for (seed, split), selected_records in selected.items()
    }
    report = [
        "# Qwen2.5 StackPlanner capability probe",
        "",
        "This probe uses positive-reward production StackPlanner actor segments and exact served token IDs. It combines layer-wise representation separation, normalized hidden-gradient magnitude, and causal hidden-state interventions.",
        "",
        f"- Candidate production records: {len(records)}",
        f"- Seeds: {', '.join(map(str, args.seeds))}",
        f"- Stability gate passed: {summary['stability']['passed']}",
        f"- Multi-seed mean Spearman: {summary['stability']['multi_seed_spearman_mean']:.6f}",
        f"- Train/held-out mean Spearman: {summary['stability']['train_heldout_spearman_mean']:.6f}",
        "- Layer-map weights are bounded around 1.0 as token-fraction-centered residuals; routing is soft and preserves the corpus-weighted baseline gradient scale.",
        "",
        "The SVG is the requested parameter-decoupling graph. Raw tensors and all derived files are hash-locked by `probe_manifest.json`.",
    ]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    files = {}
    for path in sorted(output.iterdir()):
        if (
            path.name == "probe_manifest.json"
            or path.name.startswith("partial_rank_")
            or not path.is_file()
        ):
            continue
        files[path.name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    manifest = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "model_id": str(Path(args.model).resolve()),
        "model_digest": args.model_digest,
        "taxonomy_version": CAPABILITY_TAXONOMY_VERSION,
        "num_hidden_layers": args.num_layers,
        "seeds": list(args.seeds),
        "heldout_fraction": 0.2,
        "positive_reward_only": True,
        "minimum_reward_exclusive": args.min_reward,
        "max_sequence_length": args.max_sequence_length,
        "representation_samples_per_capability_split_seed": args.samples_per_capability,
        "source_rollouts": {
            str(path): {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in source_paths
        },
        "candidate_record_count": len(records),
        "capability_token_counts": capability_token_counts,
        "routing_calibration": summary["routing_calibration"],
        "rejected_records": rejected,
        "selected_counts": selected_counts,
        "evidence_counts": {
            key: sum(len(part[key]) for part in partials)
            for key in ("activations", "gradients", "interventions")
        },
        "stability": summary["stability"],
        "actor_role_coverage": {
            "source_roles": sorted(source_actor_roles),
            "selected_roles": sorted(
                {record.role for chosen in selected.values() for record in chosen}
            ),
            "all_source_roles_covered": source_actor_roles
            <= {record.role for chosen in selected.values() for record in chosen},
        },
        "files": files,
    }
    (output / "probe_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-digest", required=True)
    parser.add_argument("--rollout-glob", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=(42, 43, 44)
    )
    parser.add_argument("--samples-per-capability", type=int, default=3)
    parser.add_argument("--max-sequence-length", type=int, default=8192)
    parser.add_argument("--intervention-strength", type=float, default=0.10)
    parser.add_argument("--min-reward", type=float, default=0.0)
    parser.add_argument("--num-layers", type=int, default=36)
    parser.add_argument(
        "--taxonomy",
        choices=("central-capability", "all-agent-action-output"),
        default="central-capability",
    )
    return parser.parse_args()


def main() -> None:
    global CAPABILITIES, CAPABILITY_TAXONOMY_VERSION, ANNOTATE_TURN
    args = _parse_args()
    if args.taxonomy == "all-agent-action-output":
        CAPABILITIES = ALL_AGENT_OUTPUT_CAPABILITIES
        CAPABILITY_TAXONOMY_VERSION = ALL_AGENT_OUTPUT_TAXONOMY_VERSION
        ANNOTATE_TURN = annotate_all_agent_output_turn
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size > 1:
        dist.init_process_group(backend="gloo")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, use_fast=True
    )
    paths = [Path(value) for value in sorted(glob.glob(args.rollout_glob))]
    if not paths:
        raise FileNotFoundError(f"rollout glob matched nothing: {args.rollout_glob}")
    records, rejected, source_actor_roles = _load_records(
        paths, tokenizer, args.min_reward
    )
    selected = _select_records(records, tuple(args.seeds), args.samples_per_capability)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if len(model.model.layers) != args.num_layers:
        raise RuntimeError(
            f"model has {len(model.model.layers)} layers, expected {args.num_layers}"
        )

    activation_tasks = [
        (seed, split, record)
        for (seed, split), chosen in selected.items()
        for record in chosen
    ]
    gradient_tasks = []
    for (seed, split), chosen in selected.items():
        for capability in CAPABILITIES:
            matches = [record for record in chosen if capability in record.capabilities]
            gradient_tasks.append((seed, split, capability, matches[0]))
    intervention_tasks = []
    primary_seed = args.seeds[0]
    for split in ("train", "heldout"):
        chosen = selected[(primary_seed, split)]
        for capability in CAPABILITIES:
            matches = [record for record in chosen if capability in record.capabilities]
            intervention_tasks.append((primary_seed, split, capability, matches[0]))

    partial: dict[str, list[Any]] = {
        "activations": [],
        "gradients": [],
        "interventions": [],
    }
    for task_index, (seed, split, record) in enumerate(activation_tasks):
        if task_index % world_size != rank:
            continue
        values = _activation_probe(model, record, args.max_sequence_length, device)
        for capability, metrics in values.items():
            partial["activations"].append(
                {
                    "seed": seed,
                    "split": split,
                    "record_id": record.record_id,
                    "capability": capability,
                    **metrics,
                }
            )
    for task_index, (seed, split, capability, record) in enumerate(gradient_tasks):
        if task_index % world_size != rank:
            continue
        partial["gradients"].append(
            {
                "seed": seed,
                "split": split,
                "record_id": record.record_id,
                "capability": capability,
                "gradient_norm": _gradient_probe(
                    model, record, capability, args.max_sequence_length, device
                ),
            }
        )
    for task_index, (seed, split, capability, record) in enumerate(intervention_tasks):
        if task_index % world_size != rank:
            continue
        partial["interventions"].append(
            {
                "seed": seed,
                "split": split,
                "record_id": record.record_id,
                "capability": capability,
                **_intervention_probe(
                    model,
                    record,
                    capability,
                    args.max_sequence_length,
                    args.intervention_strength,
                    device,
                ),
            }
        )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    partial_path = output / f"partial_rank_{rank}.pt"
    torch.save(partial, partial_path)
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        partials = [
            torch.load(
                output / f"partial_rank_{index}.pt",
                weights_only=False,
                map_location="cpu",
            )
            for index in range(world_size)
        ]
        for path in output.glob("partial_rank_*.pt"):
            path.unlink()
        _write_outputs(args, records, rejected, selected, partials, source_actor_roles)
        print(json.dumps({"output_dir": str(output), "records": len(records)}))
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
