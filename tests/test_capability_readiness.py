from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from slime_plugins.stackplanner.capability.readiness import (
    validate_probe_artifacts,
)
from slime_plugins.stackplanner.capability.taxonomy import (
    CAPABILITIES,
    CAPABILITY_TAXONOMY_VERSION,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    model = tmp_path / "model"
    model.mkdir()
    layer_map = tmp_path / "layer_map.json"
    layer_map.write_text(
        json.dumps(
            {
                "model_id": str(model.resolve()),
                "checkpoint_revision": "digest",
                "num_hidden_layers": 2,
                "taxonomy_version": CAPABILITY_TAXONOMY_VERSION,
                "weights": {
                    f"layers.{index}": {name: 0.5 for name in CAPABILITIES}
                    for index in range(2)
                },
            }
        )
    )
    taxonomy = tmp_path / "taxonomy.json"
    taxonomy.write_text(
        json.dumps(
            {
                "version": CAPABILITY_TAXONOMY_VERSION,
                "capabilities": list(CAPABILITIES),
            }
        )
    )
    required = [
        "raw_probe_metrics.pt",
        "probe_summary.json",
        "layer_profiles.csv",
        "parameter_decoupling.svg",
        "REPORT.md",
    ]
    for name in required:
        (tmp_path / name).write_text(name)
    closed = required + ["layer_map.json", "taxonomy.json"]
    manifest = tmp_path / "probe_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": str(model.resolve()),
                "model_digest": "digest",
                "taxonomy_version": CAPABILITY_TAXONOMY_VERSION,
                "num_hidden_layers": 2,
                "seeds": [1, 2, 3],
                "positive_reward_only": True,
                "evidence_counts": {
                    "activations": 1,
                    "gradients": 1,
                    "interventions": 1,
                },
                "stability": {
                    "passed": True,
                    "checks": {
                        "all_capabilities_covered": True,
                        "gradient_signal_nonzero": True,
                        "causal_signal_nonzero": True,
                        "multi_seed_mean_nonnegative": True,
                        "heldout_mean_nonnegative": True,
                    },
                },
                "files": {
                    name: {
                        "sha256": _digest(tmp_path / name),
                        "bytes": (tmp_path / name).stat().st_size,
                    }
                    for name in closed
                },
            }
        )
    )
    return manifest, layer_map, taxonomy


def test_capability_probe_readiness_accepts_complete_hash_locked_evidence(
    tmp_path: Path,
):
    manifest, layer_map, taxonomy = _artifacts(tmp_path)
    result = validate_probe_artifacts(
        manifest_path=manifest,
        layer_map_path=layer_map,
        taxonomy_path=taxonomy,
        model_id=tmp_path / "model",
        model_digest="digest",
    )
    assert result["stability"]["passed"] is True


def test_capability_probe_readiness_rejects_tampered_artifact(tmp_path: Path):
    manifest, layer_map, taxonomy = _artifacts(tmp_path)
    (tmp_path / "layer_profiles.csv").write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_probe_artifacts(
            manifest_path=manifest,
            layer_map_path=layer_map,
            taxonomy_path=taxonomy,
            model_id=tmp_path / "model",
            model_digest="digest",
        )


def test_capability_probe_readiness_accepts_same_digest_at_node_local_root(
    tmp_path: Path,
):
    manifest, layer_map, taxonomy = _artifacts(tmp_path)
    node_local_model = tmp_path / "other-node" / "model"
    node_local_model.mkdir(parents=True)
    result = validate_probe_artifacts(
        manifest_path=manifest,
        layer_map_path=layer_map,
        taxonomy_path=taxonomy,
        model_id=node_local_model,
        model_digest="digest",
    )
    assert result["model_digest"] == "digest"


def test_capability_probe_readiness_rejects_different_model_name(
    tmp_path: Path,
):
    manifest, layer_map, taxonomy = _artifacts(tmp_path)
    other_model = tmp_path / "other-node" / "different-model"
    other_model.mkdir(parents=True)
    with pytest.raises(ValueError, match="probe/model mismatch"):
        validate_probe_artifacts(
            manifest_path=manifest,
            layer_map_path=layer_map,
            taxonomy_path=taxonomy,
            model_id=other_model,
            model_digest="digest",
        )
