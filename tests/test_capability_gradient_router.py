from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from slime_plugins.stackplanner.capability.gradient_router import (
    _call_with_parameter_gradient_scale,
    _patch_layer,
    attach_microbatch_capability,
    install_gradient_router,
)
from slime_plugins.stackplanner.capability.layer_map import LayerMap


def _gradient(scale: float, microbatches: tuple[float, ...] = (2.0,)) -> torch.Tensor:
    layer = torch.nn.Linear(2, 1, bias=False)
    layer.weight.data.fill_(1.0)
    layer.weight.grad = None
    for value in microbatches:
        inputs = torch.full((1, 2), value)
        output = _call_with_parameter_gradient_scale(
            layer, layer.forward, scale, (inputs,), {}
        )
        output.sum().backward()
    return layer.weight.grad.detach().clone()


def test_zero_weight_layer_has_zero_effective_gradient():
    assert torch.equal(_gradient(0.0), torch.zeros((1, 2)))


def test_execution_capability_can_update_a_different_soft_layer():
    assert torch.equal(_gradient(0.25), torch.full((1, 2), 0.5))
    assert torch.equal(_gradient(0.75), torch.full((1, 2), 1.5))


def test_all_shared_mapping_matches_normal_backward():
    assert torch.equal(_gradient(1.0), torch.full((1, 2), 2.0))


def test_gradient_accumulation_preserves_route_scaling():
    assert torch.equal(_gradient(0.5, (2.0, 4.0)), torch.full((1, 2), 3.0))


def test_gradient_clipping_sees_already_routed_gradients():
    grad = _gradient(0.5, (3.0,))
    parameter = torch.nn.Parameter(torch.zeros_like(grad))
    parameter.grad = grad.clone()
    torch.nn.utils.clip_grad_norm_([parameter], max_norm=1.0)
    assert torch.linalg.vector_norm(parameter.grad) <= 1.000001


def test_route_uses_shared_storage_and_preserves_parameter_attributes():
    layer = torch.nn.Linear(2, 1, bias=False)
    layer.weight.tensor_model_parallel = True
    original_forward = layer.forward
    observed = {}

    def inspect_forward(inputs):
        routed = layer._parameters["weight"]
        observed["same_storage"] = routed.data_ptr() == layer.weight.data_ptr()
        observed["attribute"] = routed.tensor_model_parallel
        observed["is_intermediate"] = routed.grad_fn is not None
        return original_forward(inputs)

    _call_with_parameter_gradient_scale(
        layer, inspect_forward, 0.5, (torch.ones(1, 2),), {}
    ).sum().backward()
    assert observed == {
        "same_storage": True,
        "attribute": True,
        "is_intermediate": True,
    }
    assert isinstance(layer.weight, torch.nn.Parameter)
    assert layer._parameters["weight"] is layer.weight
    assert torch.equal(layer.weight.grad, torch.full((1, 2), 0.5))


def test_route_restores_parameter_registration_after_forward_failure():
    layer = torch.nn.Linear(2, 1, bias=False)
    original = layer.weight

    def fail(_inputs):
        raise RuntimeError("forward failed")

    with pytest.raises(RuntimeError, match="forward failed"):
        _call_with_parameter_gradient_scale(layer, fail, 0.5, (torch.ones(1, 2),), {})
    assert layer._parameters["weight"] is original
    assert isinstance(layer.weight, torch.nn.Parameter)


def test_bfloat16_route_scales_gradient_without_changing_forward_values():
    layer = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    layer.weight.data.fill_(1)
    inputs = torch.full((1, 2), 2, dtype=torch.bfloat16)
    output = _call_with_parameter_gradient_scale(
        layer, layer.forward, 0.25, (inputs,), {}
    )
    assert torch.equal(output, torch.full((1, 1), 4, dtype=torch.bfloat16))
    output.sum().backward()
    assert torch.equal(layer.weight.grad, torch.full((1, 2), 0.5, dtype=torch.bfloat16))


def test_router_rejects_fused_gradient_accumulation():
    args = SimpleNamespace(gradient_accumulation_fusion=True)
    with pytest.raises(RuntimeError, match="no-gradient-accumulation-fusion"):
        install_gradient_router(args, 0, 0, [], None, None)


def test_router_canonicalizes_symlinked_model_identity(tmp_path, monkeypatch):
    model = tmp_path / "canonical-model"
    model.mkdir()
    alias = tmp_path / "model-alias"
    alias.symlink_to(model, target_is_directory=True)
    layer_map = tmp_path / "layer-map.json"
    layer_map.write_text(
        '{"model_id": '
        + json.dumps(str(model.resolve()))
        + ', "checkpoint_revision": "digest", "num_hidden_layers": 1,'
        ' "taxonomy_version": "test", "weights": {'
        ' "layers.0": {"action_choice": 1.0}}}'
    )
    monkeypatch.setenv("SP_CAPABILITY_LAYER_MAP", str(layer_map))
    monkeypatch.setenv("SP_BASE_CHECKPOINT_SHA256", "digest")
    args = SimpleNamespace(
        gradient_accumulation_fusion=False,
        num_layers=1,
        hf_checkpoint=str(alias),
    )
    # No discovered local layer is valid for a pipeline shard, so the only
    # assertion exercised here is model identity after canonicalization.
    with pytest.raises(RuntimeError, match="layer discovery mismatch"):
        install_gradient_router(args, 0, 0, [], None, None)


def test_microbatch_router_accepts_b_taxonomy(monkeypatch):
    monkeypatch.setenv("SP_DECOUPLING_SCOPE", "all-agent-action-output")
    packed = SimpleNamespace()
    batch = {
        "metadata": [{"capability": "action_output"}],
        "packed_seq_params": packed,
    }
    attach_microbatch_capability(SimpleNamespace(), batch, None)
    assert packed.sp_capability == "action_output"


class _RoutableLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_number = 1
        self.self_attention = torch.nn.Identity()
        self.mlp = torch.nn.Linear(1, 1, bias=False)
        self.mlp.weight.data.fill_(1.0)

    def forward(self, inputs, *, packed_seq_params=None):
        del packed_seq_params
        return self.mlp(inputs)


def _two_route_map() -> LayerMap:
    return LayerMap(
        model_id="model",
        checkpoint_revision="digest",
        num_hidden_layers=1,
        taxonomy_version="stackplanner-central-action-output-v2",
        weights={
            "layers.0": {
                "action_choice": 0.5,
                "action_output": 1.0,
            }
        },
    )


def test_router_allows_unlabeled_reference_forward_without_scaling(monkeypatch):
    monkeypatch.setenv("SP_DECOUPLING_SCOPE", "central-capability")
    layer = _RoutableLayer()
    _patch_layer(layer, 0, _two_route_map())

    layer(torch.ones(1, 1), packed_seq_params=SimpleNamespace()).sum().backward()

    assert torch.equal(layer.mlp.weight.grad, torch.ones(1, 1))


def test_router_scales_explicit_valid_capability(monkeypatch):
    monkeypatch.setenv("SP_DECOUPLING_SCOPE", "central-capability")
    layer = _RoutableLayer()
    _patch_layer(layer, 0, _two_route_map())
    packed = SimpleNamespace(sp_capability="action_choice")

    layer(torch.ones(1, 1), packed_seq_params=packed).sum().backward()

    assert torch.equal(layer.mlp.weight.grad, torch.full((1, 1), 0.5))


def test_router_rejects_explicit_invalid_capability(monkeypatch):
    monkeypatch.setenv("SP_DECOUPLING_SCOPE", "central-capability")
    layer = _RoutableLayer()
    _patch_layer(layer, 0, _two_route_map())
    packed = SimpleNamespace(sp_capability="not-a-capability")

    with pytest.raises(
        RuntimeError, match="invalid homogeneous micro-batch capability"
    ):
        layer(torch.ones(1, 1), packed_seq_params=packed)


def test_router_supports_training_reference_training_sequence(monkeypatch):
    monkeypatch.setenv("SP_DECOUPLING_SCOPE", "central-capability")
    layer = _RoutableLayer()
    _patch_layer(layer, 0, _two_route_map())
    inputs = torch.ones(1, 1)

    layer(
        inputs,
        packed_seq_params=SimpleNamespace(sp_capability="action_choice"),
    ).sum().backward()
    layer(inputs, packed_seq_params=SimpleNamespace()).sum().backward()
    layer(
        inputs,
        packed_seq_params=SimpleNamespace(sp_capability="action_output"),
    ).sum().backward()

    assert torch.equal(layer.mlp.weight.grad, torch.full((1, 1), 2.5))


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (None, "missing training metadata"),
        ([], "missing training metadata"),
        (
            [
                {"capability": "action_choice"},
                {"capability": "action_output"},
            ],
            "requires homogeneous capability micro-batches",
        ),
        ([{"capability": "not-a-capability"}], "invalid micro-batch capability"),
    ],
)
def test_training_microbatch_hook_remains_fail_closed(monkeypatch, metadata, message):
    monkeypatch.setenv("SP_DECOUPLING_SCOPE", "central-capability")
    batch = {"metadata": metadata, "packed_seq_params": SimpleNamespace()}

    with pytest.raises(RuntimeError, match=message):
        attach_microbatch_capability(SimpleNamespace(), batch, None)
