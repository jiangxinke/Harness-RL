"""Readiness gate for evidence-backed decoupled training."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from .all_agent_output import (
    ALL_AGENT_OUTPUT_CAPABILITIES,
    ALL_AGENT_OUTPUT_TAXONOMY_VERSION,
)
from .readiness import validate_probe_artifacts
from .taxonomy import CAPABILITIES, CAPABILITY_TAXONOMY_VERSION

if TYPE_CHECKING:
    from ..config import TrainingConfig

CAPABILITY_RUNTIME_IMPLEMENTED = True


def require_capability_runtime(config: TrainingConfig) -> None:
    if config.experiment.value == "B" and config.capability_probe_bootstrap:
        if config.capability_annotator_function != "slime_plugins.stackplanner.capability.all_agent_output.annotate_turn":
            raise ValueError("experiment B probe bootstrap requires the exact two-route action/output annotator")
        return
    required = {
        "capability_taxonomy_path": config.capability_taxonomy_path,
        "capability_probe_manifest": config.capability_probe_manifest,
        "capability_layer_map": config.capability_layer_map,
        "capability_annotator_function": config.capability_annotator_function,
        "capability_loss_function": config.capability_loss_function,
        "capability_gradient_router_function": config.capability_gradient_router_function,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"experiment {config.experiment.value} is missing: " + ", ".join(missing))
    if not CAPABILITY_RUNTIME_IMPLEMENTED:
        raise ValueError(f"experiment {config.experiment.value} capability runtime is disabled")
    try:
        validate_probe_artifacts(
            manifest_path=config.capability_probe_manifest,
            layer_map_path=config.capability_layer_map,
            taxonomy_path=config.capability_taxonomy_path,
            model_id=config.hf_checkpoint,
            model_digest=config.base_checkpoint_sha256,
            expected_capabilities=(ALL_AGENT_OUTPUT_CAPABILITIES if config.experiment.value == "B" else CAPABILITIES),
            expected_taxonomy_version=(ALL_AGENT_OUTPUT_TAXONOMY_VERSION if config.experiment.value == "B" else CAPABILITY_TAXONOMY_VERSION),
            require_actor_role_coverage=config.experiment.value == "B",
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"experiment {config.experiment.value} capability evidence is unreadable: {exc}") from exc
