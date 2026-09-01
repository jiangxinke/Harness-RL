"""Versioned layer-map schema used by the future experiment C runtime."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any


def same_hash_locked_model_id(left: str | Path, right: str | Path) -> bool:
    """Compare locations for an already hash-locked model.

    A probe and a training run may use different mount points for the same
    checkpoint. The digest is validated separately, so matching the resolved
    path or the checkpoint directory name is sufficient.
    """

    left_path = Path(left).resolve()
    right_path = Path(right).resolve()
    return left_path == right_path or left_path.name == right_path.name


@dataclasses.dataclass(frozen=True, slots=True)
class LayerMap:
    model_id: str
    checkpoint_revision: str
    num_hidden_layers: int
    taxonomy_version: str
    weights: dict[str, dict[str, float]]

    @classmethod
    def load(cls, path: str | Path) -> LayerMap:
        raw: Any = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("layer map must be a JSON object")
        result = cls(**raw)
        result.validate()
        return result

    def validate(self) -> None:
        if self.num_hidden_layers < 1 or not self.weights:
            raise ValueError("layer map must contain model layers and weights")
        for layer, abilities in self.weights.items():
            if not layer.startswith("layers.") or not layer.removeprefix("layers.").isdigit():
                raise ValueError(f"invalid layer key: {layer!r}")
            if not abilities:
                raise ValueError(f"layer {layer!r} has no capability weights")
            for capability, weight in abilities.items():
                if not 0.0 <= float(weight) <= 1.2:
                    raise ValueError(f"invalid weight for {layer}/{capability}: {weight}")
        expected_layers = {f"layers.{index}" for index in range(self.num_hidden_layers)}
        if set(self.weights) != expected_layers:
            raise ValueError(f"layer map coverage mismatch: got={sorted(self.weights)}, expected={sorted(expected_layers)}")

    def weight(self, layer_index: int, capability: str) -> float:
        layer = f"layers.{layer_index}"
        try:
            return float(self.weights[layer][capability])
        except KeyError as exc:
            raise ValueError(f"layer map has no weight for {layer}/{capability}") from exc

    def assert_model(self, *, model_id: str, revision: str, num_hidden_layers: int) -> None:
        actual = (str(Path(model_id).resolve()), revision, num_hidden_layers)
        expected = (str(Path(self.model_id).resolve()), self.checkpoint_revision, self.num_hidden_layers)
        if not same_hash_locked_model_id(actual[0], expected[0]) or actual[1:] != expected[1:]:
            raise ValueError(f"layer map/model mismatch: map={expected!r}, runtime={actual!r}")
