from __future__ import annotations

import json
from pathlib import Path

import pytest

from slime_plugins.stackplanner.capability.layer_map import LayerMap


def test_layer_map_validates_weights_and_model_identity(tmp_path: Path):
    path = tmp_path / "map.json"
    path.write_text(
        json.dumps(
            {
                "model_id": "model",
                "checkpoint_revision": "rev",
                "num_hidden_layers": 2,
                "taxonomy_version": "v1",
                "weights": {
                    "layers.0": {"action": 0.8, "shared": 0.2},
                    "layers.1": {"action": 0.2, "shared": 0.8},
                },
            }
        ),
        encoding="utf-8",
    )
    layer_map = LayerMap.load(path)
    layer_map.assert_model(model_id="model", revision="rev", num_hidden_layers=2)
    layer_map.assert_model(
        model_id=tmp_path / "node-local" / "model",
        revision="rev",
        num_hidden_layers=2,
    )
    with pytest.raises(ValueError, match="mismatch"):
        layer_map.assert_model(model_id="other", revision="rev", num_hidden_layers=2)


def test_layer_map_accepts_bounded_positive_residual_route(tmp_path: Path):
    path = tmp_path / "map.json"
    path.write_text(
        json.dumps(
            {
                "model_id": "model",
                "checkpoint_revision": "rev",
                "num_hidden_layers": 1,
                "taxonomy_version": "v2",
                "weights": {
                    "layers.0": {
                        "action_choice": 1.2,
                        "action_output": 0.8,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    layer_map = LayerMap.load(path)
    assert layer_map.weight(0, "action_choice") == 1.2
