from __future__ import annotations

import copy

import numpy as np
import torch

from scripts.probe_stackplanner_capabilities import (
    ProbeRecord,
    _aggregate,
    _load_records,
    _normalize,
    _rank_correlation,
    _select_records,
    _split,
)


class _DecodeOnlyTokenizer:
    def decode(self, _ids, **_kwargs):
        return "unused"


def test_probe_split_is_deterministic_and_seeded():
    first = [_split(f"record-{index}", 7) for index in range(100)]
    assert first == [_split(f"record-{index}", 7) for index in range(100)]
    assert first != [_split(f"record-{index}", 8) for index in range(100)]
    assert {"train", "heldout"} <= set(first)


def test_probe_normalization_is_bounded_and_constant_safe():
    assert np.allclose(_normalize(np.array([2.0, 4.0, 6.0])), [0.0, 0.5, 1.0])
    assert np.allclose(_normalize(np.array([3.0, 3.0])), [1.0, 1.0])


def test_probe_rank_correlation_tracks_layer_order():
    ascending = np.array([1.0, 2.0, 3.0, 4.0])
    assert _rank_correlation(ascending, ascending * 2) == 1.0
    assert _rank_correlation(ascending, ascending[::-1]) == -1.0


def test_two_route_map_is_token_fraction_centered_around_baseline():
    import scripts.probe_stackplanner_capabilities as probe

    original = probe.CAPABILITIES
    probe.CAPABILITIES = ("action_choice", "action_output")
    partials = [
        {
            "activations": [
                {
                    "seed": seed,
                    "split": split,
                    "capability": capability,
                    "centroid": [
                        [float(layer + offset), float(layer - offset)]
                        for layer in range(3)
                    ],
                }
                for seed in (1, 2, 3)
                for split in ("train", "heldout")
                for capability, offset in (
                    ("action_choice", 1.0),
                    ("action_output", -1.0),
                )
            ],
            "gradients": [
                {
                    "seed": seed,
                    "split": split,
                    "capability": capability,
                    "gradient_norm": gradient,
                }
                for seed in (1, 2, 3)
                for split in ("train", "heldout")
                for capability, gradient in (
                    ("action_choice", [1.0, 2.0, 4.0]),
                    ("action_output", [4.0, 2.0, 1.0]),
                )
            ],
            "interventions": [
                {
                    "seed": 1,
                    "split": split,
                    "capability": capability,
                    "delta_nll": delta,
                }
                for split in ("train", "heldout")
                for capability, delta in (
                    ("action_choice", [0.1, 0.2, 0.4]),
                    ("action_output", [0.4, 0.2, 0.1]),
                )
            ],
        }
    ]
    fractions = {"action_choice": 0.1, "action_output": 0.9}
    try:
        _raw, weights, summary = _aggregate(partials, 3, fractions)
    finally:
        probe.CAPABILITIES = original

    for layer in weights.values():
        assert 0.8 <= min(layer.values()) <= max(layer.values()) <= 1.2
        assert all(0.8 <= value <= 1.2 for value in layer.values())
        assert np.isclose(sum(fractions[name] * layer[name] for name in fractions), 1.0)
    assert summary["routing_calibration"]["baseline_weight"] == 1.0


def test_probe_selection_covers_every_positive_actor_role():
    import scripts.probe_stackplanner_capabilities as probe

    original = probe.CAPABILITIES
    probe.CAPABILITIES = ("action_choice", "action_output")
    try:
        records = []
        for role in ("central", "researcher"):
            for index in range(200):
                labels = list(probe.CAPABILITIES)
                records.append(
                    ProbeRecord(
                        record_id=f"{role}-{index}",
                        tokens=[1, 2],
                        response_length=2,
                        loss_mask=[1, 1],
                        labels=labels,
                        reward=1.0,
                        source=f"source:{index}",
                        role=role,
                    )
                )
        selected = _select_records(records, (7,), 1)
        assert {record.role for record in selected[(7, "train")]} == {
            "central",
            "researcher",
        }
        assert {record.role for record in selected[(7, "heldout")]} == {
            "central",
            "researcher",
        }
    finally:
        probe.CAPABILITIES = original


def test_all_agent_output_loader_allows_role_metadata_gaps_only_on_masked_tokens(
    tmp_path,
):
    import scripts.probe_stackplanner_capabilities as probe

    original_capabilities = probe.CAPABILITIES
    original_version = probe.CAPABILITY_TAXONOMY_VERSION
    probe.CAPABILITIES = ("action_choice", "action_output")
    probe.CAPABILITY_TAXONOMY_VERSION = probe.ALL_AGENT_OUTPUT_TAXONOMY_VERSION
    try:
        source = tmp_path / "rollout.pt"
        sample = {
            "metadata": {"role": "researcher"},
            "reward": 1.0,
            "response_length": 3,
            "tokens": [10, 11, 12],
            "loss_mask": [0, 1, 1],
            "train_metadata": {
                "capability_labels": [None, "action_choice", "action_output"],
                "actor_roles": [None, "researcher", "researcher"],
            },
        }
        torch.save({"samples": [sample]}, source)
        records, rejected, roles = _load_records(
            [source], _DecodeOnlyTokenizer(), min_reward=0.0
        )
        assert len(records) == 1
        assert rejected == []
        assert roles == {"researcher"}

        good_sample = copy.deepcopy(sample)
        sample["train_metadata"]["actor_roles"][1] = "central"
        torch.save({"samples": [good_sample, sample]}, source)
        records, rejected, roles = _load_records(
            [source], _DecodeOnlyTokenizer(), min_reward=0.0
        )
        assert len(records) == 1
        assert roles == {"researcher"}
        assert rejected and "misaligned" in rejected[0]["reason"]
    finally:
        probe.CAPABILITIES = original_capabilities
        probe.CAPABILITY_TAXONOMY_VERSION = original_version
