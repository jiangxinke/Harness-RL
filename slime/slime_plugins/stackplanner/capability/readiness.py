"""Fail-closed validation of target-model capability probe artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .layer_map import LayerMap, same_hash_locked_model_id
from .taxonomy import CAPABILITIES, CAPABILITY_TAXONOMY_VERSION


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_probe_artifacts(
    *,
    manifest_path: str | Path,
    layer_map_path: str | Path,
    taxonomy_path: str | Path,
    model_id: str,
    model_digest: str,
    expected_capabilities: tuple[str, ...] = CAPABILITIES,
    expected_taxonomy_version: str = CAPABILITY_TAXONOMY_VERSION,
    require_actor_role_coverage: bool = False,
) -> dict[str, Any]:
    """Validate artifact identity, evidence coverage, and every locked file."""

    manifest_file = Path(manifest_path).resolve()
    layer_map_file = Path(layer_map_path).resolve()
    taxonomy_file = Path(taxonomy_path).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported capability probe manifest schema")
    expected_identity = (str(Path(model_id).resolve()), model_digest)
    actual_identity = (manifest.get("model_id"), manifest.get("model_digest"))
    if not isinstance(actual_identity[0], str) or not same_hash_locked_model_id(actual_identity[0], expected_identity[0]) or actual_identity[1] != expected_identity[1]:
        raise ValueError(f"capability probe/model mismatch: manifest={actual_identity!r}, runtime={expected_identity!r}")
    if manifest.get("taxonomy_version") != expected_taxonomy_version:
        raise ValueError("capability probe taxonomy version mismatch")
    if len(manifest.get("seeds", [])) < 3:
        raise ValueError("capability probe requires at least three seeds")
    if not manifest.get("positive_reward_only"):
        raise ValueError("capability probe must use only positive-reward production records")
    evidence = manifest.get("evidence_counts", {})
    if any(int(evidence.get(key, 0)) <= 0 for key in ("activations", "gradients", "interventions")):
        raise ValueError("capability probe is missing representation, gradient, or intervention evidence")
    stability = manifest.get("stability", {})
    if not stability.get("passed") or not all(
        stability.get("checks", {}).get(key)
        for key in (
            "all_capabilities_covered",
            "gradient_signal_nonzero",
            "causal_signal_nonzero",
            "multi_seed_mean_nonnegative",
            "heldout_mean_nonnegative",
        )
    ):
        raise ValueError("capability probe stability gate did not pass")
    if require_actor_role_coverage:
        coverage = manifest.get("actor_role_coverage", {})
        source_roles = coverage.get("source_roles")
        selected_roles = coverage.get("selected_roles")
        if not coverage.get("all_source_roles_covered") or not isinstance(source_roles, list) or not source_roles or "central" not in {str(role).lower() for role in source_roles} or set(source_roles) != set(selected_roles or []):
            raise ValueError("all-agent probe does not cover every actor role in its production source")

    files = manifest.get("files", {})
    if not isinstance(files, dict) or not files:
        raise ValueError("capability probe manifest has no file closure")
    for name, expected in files.items():
        path = manifest_file.parent / name
        if not path.is_file():
            raise ValueError(f"capability probe artifact is missing: {path}")
        actual = {"sha256": _sha256(path), "bytes": path.stat().st_size}
        if actual != expected:
            raise ValueError(f"capability probe artifact hash mismatch for {name}: expected={expected!r}, actual={actual!r}")
    for required in (
        "raw_probe_metrics.pt",
        "probe_summary.json",
        "layer_profiles.csv",
        "parameter_decoupling.svg",
        "REPORT.md",
    ):
        if required not in files:
            raise ValueError(f"capability probe closure is missing {required}")

    taxonomy = json.loads(taxonomy_file.read_text(encoding="utf-8"))
    if taxonomy.get("version") != expected_taxonomy_version or tuple(taxonomy.get("capabilities", [])) != expected_capabilities:
        raise ValueError("capability taxonomy artifact does not match the runtime")
    layer_map = LayerMap.load(layer_map_file)
    layer_map.assert_model(
        model_id=expected_identity[0],
        revision=model_digest,
        num_hidden_layers=int(manifest["num_hidden_layers"]),
    )
    for layer in layer_map.weights.values():
        if set(layer) != set(expected_capabilities):
            raise ValueError("capability layer map does not cover the runtime taxonomy")
    return manifest
