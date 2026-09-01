"""Fan exact-token capability masks out into ordinary slime samples."""

from __future__ import annotations

import copy
from typing import Any

from .runtime_taxonomy import active_capabilities


def fan_out_capability_samples(args: Any, samples: list[Any]) -> list[Any]:
    """Create one mask-subset sample per active decoupling capability.

    All children preserve the logical rollout id and reward. The masks
    are mutually exclusive and their union is exactly the original actor loss
    mask.  Empty capability children are retained so every rollout has a
    deterministic fan-out shape across data-parallel ranks.
    """

    del args
    capabilities = active_capabilities()
    transformed: list[Any] = []
    for sample in samples:
        metadata = dict(getattr(sample, "train_metadata", None) or {})
        labels = metadata.get("capability_labels")
        original_mask = list(getattr(sample, "loss_mask", None) or [])
        response_length = int(getattr(sample, "response_length", 0) or 0)
        if not isinstance(labels, list) or len(labels) != response_length:
            raise ValueError(f"experiment C requires one exact capability label per response token: labels={len(labels) if isinstance(labels, list) else 'missing'}, response_length={response_length}")
        if len(original_mask) != response_length:
            raise ValueError("capability fan-out received a misaligned loss mask")
        # Replayed actor output and environment observations can remain in the
        # response sequence with loss_mask=0.  Their metadata is deliberately
        # cleared during trajectory realignment, so only trainable tokens must
        # carry a capability label.
        unknown = sorted({str(label) for enabled, label in zip(original_mask, labels, strict=True) if enabled and label not in capabilities})
        if unknown:
            raise ValueError(f"unknown capability labels: {unknown}")

        reconstructed = [0] * response_length
        for capability in capabilities:
            child = copy.deepcopy(sample)
            child.loss_mask = [int(bool(enabled) and label == capability) for enabled, label in zip(original_mask, labels, strict=True)]
            reconstructed = [current + enabled for current, enabled in zip(reconstructed, child.loss_mask, strict=True)]
            child.train_metadata = {
                **metadata,
                "capability": capability,
                "microbatch_group": capability,
                "capability_labels": labels,
                "original_actor_loss_mask": original_mask,
            }
            transformed.append(child)
        if reconstructed != [int(bool(value)) for value in original_mask]:
            raise AssertionError("capability masks do not partition the actor loss mask")
    return transformed
