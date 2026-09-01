"""Layer-wise soft capability gradient routing inside slime/Megatron."""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import torch

from .layer_map import LayerMap
from .runtime_taxonomy import active_capabilities

_ROUTER_MARKER = "_stackplanner_capability_router_installed"


class _ScaleParameterGradient(torch.autograd.Function):
    """Identity in forward and a scalar multiplier in backward.

    The returned tensor is a view of the original parameter storage, so routing
    does not retain a parameter-sized arithmetic result for every micro-batch.
    The leaf parameter remains the optimizer/DDP-owned object and receives the
    scaled gradient after the custom backward node.
    """

    @staticmethod
    def forward(ctx: Any, parameter: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return parameter.view_as(parameter)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return gradient.mul(ctx.scale), None


def _routed_parameter_view(parameter: torch.nn.Parameter, scale: float) -> torch.Tensor:
    routed = _ScaleParameterGradient.apply(parameter, scale)
    # Megatron/Transformer Engine attach sharding, sequence-parallel and
    # optimizer metadata directly to Parameters.  The temporary tensor must
    # expose those attributes during forward even though ownership and gradient
    # accumulation stay on the original leaf Parameter.
    routed.__dict__.update(parameter.__dict__)
    return routed


def _layer_index(module: torch.nn.Module) -> int | None:
    number = getattr(module, "layer_number", None)
    if not isinstance(number, int):
        return None
    if not hasattr(module, "self_attention") or not hasattr(module, "mlp"):
        return None
    # Megatron transformer layers are numbered from one globally.
    return number - 1


def _capability_from_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
    """Return an explicit training capability, or ``None`` for plain forwards.

    The router remains installed while slime computes reference log-probabilities
    and performs other inference-only forwards.  Those calls deliberately do not
    carry StackPlanner training metadata and must use the unmodified parameters.
    A call that *does* explicitly carry ``sp_capability`` is still validated
    fail-closed so malformed routed training cannot silently bypass the router.
    """

    packed = kwargs.get("packed_seq_params")
    if packed is None:
        packed = next(
            (value for value in args if hasattr(value, "sp_capability")),
            None,
        )
    if packed is None or not hasattr(packed, "sp_capability"):
        return None
    capability = getattr(packed, "sp_capability", None)
    if capability not in active_capabilities():
        raise RuntimeError(f"capability-routed layer forward carries an invalid homogeneous micro-batch capability: {capability!r}")
    return capability


def _call_with_parameter_gradient_scale(
    module: torch.nn.Module,
    original_forward: Any,
    scale: float,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    if scale == 1.0:
        return original_forward(*args, **kwargs)

    replaced: list[tuple[torch.nn.Module, str, torch.nn.Parameter]] = []
    try:
        for child in module.modules():
            for name, parameter in tuple(child._parameters.items()):
                if parameter is None or not parameter.requires_grad:
                    continue
                # Same storage and numeric value in forward, but the custom
                # autograd identity scales the gradient reaching the leaf.
                routed = _routed_parameter_view(parameter, scale)
                child._parameters[name] = routed
                replaced.append((child, name, parameter))
        return original_forward(*args, **kwargs)
    finally:
        for child, name, parameter in reversed(replaced):
            child._parameters[name] = parameter


def _patch_layer(module: torch.nn.Module, layer_index: int, layer_map: LayerMap) -> None:
    original_forward = module.forward

    @functools.wraps(original_forward)
    def routed_forward(*args: Any, **kwargs: Any) -> Any:
        capability = _capability_from_call(args, kwargs)
        if capability is None:
            return original_forward(*args, **kwargs)
        scale = layer_map.weight(layer_index, capability)
        return _call_with_parameter_gradient_scale(module, original_forward, scale, args, kwargs)

    module.forward = routed_forward  # type: ignore[method-assign]
    setattr(module, _ROUTER_MARKER, True)


def install_gradient_router(
    args: Any,
    rollout_id: int,
    step_id: int,
    model: list[Any],
    optimizer: Any,
    opt_param_scheduler: Any,
) -> None:
    """Install idempotent layer wrappers before the first train step."""

    del rollout_id, step_id, optimizer, opt_param_scheduler
    if bool(getattr(args, "gradient_accumulation_fusion", False)):
        raise RuntimeError("experiment C requires --no-gradient-accumulation-fusion because fused weight-gradient kernels bypass autograd parameter routing")
    path = os.environ.get("SP_CAPABILITY_LAYER_MAP")
    if not path:
        raise RuntimeError("SP_CAPABILITY_LAYER_MAP is required for experiment C")
    layer_map = LayerMap.load(Path(path))
    layer_map.assert_model(
        # Probe manifests store the canonical target path.  Resolve the runtime
        # path too so a node-local symlink to the exact same immutable model
        # cannot produce a false identity mismatch after preflight passed.
        model_id=str(Path(getattr(args, "hf_checkpoint", "")).resolve()),
        revision=str(os.environ.get("SP_BASE_CHECKPOINT_SHA256", "")),
        num_hidden_layers=int(args.num_layers),
    )
    seen: set[int] = set()
    for chunk in model:
        for module in chunk.modules():
            layer_index = _layer_index(module)
            if layer_index is None:
                continue
            seen.add(layer_index)
            if not getattr(module, _ROUTER_MARKER, False):
                _patch_layer(module, layer_index, layer_map)
    expected = set(range(int(args.num_layers)))
    if not seen or not seen <= expected:
        raise RuntimeError(f"capability router layer discovery mismatch: seen={sorted(seen)}, valid_global_layers={sorted(expected)}")


def attach_microbatch_capability(args: Any, batch: dict[str, Any], model: Any) -> None:
    """Attach one homogeneous sample capability to PackedSeqParams."""

    del args, model
    metadata = batch.get("metadata")
    if not isinstance(metadata, list) or not metadata:
        raise RuntimeError("capability-routed micro-batch is missing training metadata")
    capabilities = {item.get("capability") for item in metadata if isinstance(item, dict)}
    if len(capabilities) != 1:
        raise RuntimeError(f"experiment C requires homogeneous capability micro-batches; got {sorted(str(value) for value in capabilities)}")
    capability = capabilities.pop()
    if capability not in active_capabilities():
        raise RuntimeError(f"invalid micro-batch capability: {capability!r}")
    batch["packed_seq_params"].sp_capability = capability
