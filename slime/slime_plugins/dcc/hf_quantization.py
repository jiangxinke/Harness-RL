"""Opt-in full-expert NF4 loading for the unfused Qwen3 MoE runtime.

Keep this module lazy: the default BF16/grouped-mm runtime does not need
bitsandbytes, and newer Transformers fuse experts into unsupported parameters.
"""

from __future__ import annotations

import math
import re

import torch

_EXPERT_LINEAR = re.compile(r"(?:^|\.)experts\.\d+\.(?:gate_proj|up_proj|down_proj)$")
_TRANSFORMERS_VERSION = "4.57.6"


def nf4_load_kwargs():
    import transformers
    from transformers import BitsAndBytesConfig

    if transformers.__version__ != _TRANSFORMERS_VERSION:
        raise ValueError(
            "Full-expert NF4 requires the separate DCC quantized environment "
            f"with transformers=={_TRANSFORMERS_VERSION}; newer fused MoE "
            "parameters are not covered by bitsandbytes. See docs/dcc_training.md."
        )
    return {
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            # Keep routing and the output head unquantized. This exact leaf
            # name does not exclude expert gate_proj modules.
            llm_int8_skip_modules=["lm_head", "gate"],
        )
    }


def quantization_report(model) -> dict:
    """Count logical base weights and require every expert matrix to be NF4.

    Packed tensor numel is not an original parameter count. Quantization scales
    are reported separately from registered parameter storage; neither number
    includes activations, KV cache, CUDA workspaces or allocator reservations.
    Call before adding adapters so the logical total describes the base model.
    """
    import bitsandbytes as bnb

    config = model.config
    if config.model_type != "qwen3_moe":
        raise ValueError("DCC full-expert NF4 currently supports qwen3_moe only")
    sparse_layers = sum(
        layer not in config.mlp_only_layers
        and (layer + 1) % config.decoder_sparse_step == 0
        for layer in range(config.num_hidden_layers)
    )
    expected_expert_matrices = sparse_layers * config.num_experts * 3
    expected_expert_parameters = (
        sparse_layers * config.num_experts * 3 * config.hidden_size * config.moe_intermediate_size
    )
    if not expected_expert_matrices:
        raise ValueError("Full-expert NF4 requires at least one sparse MoE layer")

    quantized_weights, metadata_tensors = {}, {}

    def count_state(state):
        if state is None:
            return
        for field in ("absmax", "code", "offset"):
            tensor = getattr(state, field, None)
            if isinstance(tensor, torch.Tensor):
                metadata_tensors[id(tensor)] = tensor.numel() * tensor.element_size()
        count_state(getattr(state, "state2", None))

    expert_parameters, expert_matrices = 0, 0
    for name, module in model.named_modules():
        expert = bool(_EXPERT_LINEAR.search(name))
        if isinstance(module, bnb.nn.Linear4bit):
            weight = module.weight
            state = getattr(weight, "quant_state", None)
            if state is None or getattr(state, "quant_type", None) != "nf4":
                raise ValueError(f"Uninitialized or non-NF4 quantized matrix: {name}")
            logical = math.prod(state.shape)
            if logical != module.in_features * module.out_features:
                raise ValueError(f"Unexpected packed matrix shape: {name}")
            quantized_weights[id(weight)] = logical
            count_state(state)
            if expert:
                expert_matrices += 1
                expert_parameters += logical
        elif expert:
            raise ValueError(f"Expert matrix escaped NF4 quantization: {name}")
    if expert_matrices != expected_expert_matrices or expert_parameters != expected_expert_parameters:
        raise ValueError(
            "Incomplete expert quantization: "
            f"{expert_matrices}/{expected_expert_matrices} matrices, "
            f"{expert_parameters}/{expected_expert_parameters} logical parameters"
        )

    unquantized, fp32, storage = 0, 0, 0
    for parameter in model.parameters():
        storage += parameter.numel() * parameter.element_size()
        if id(parameter) not in quantized_weights:
            unquantized += parameter.numel()
            if parameter.dtype == torch.float32:
                fp32 += parameter.numel()
    quantized = sum(quantized_weights.values())
    total = quantized + unquantized
    return {
        "mode": "nf4",
        "transformers_version": _TRANSFORMERS_VERSION,
        "logical_base_parameters": total,
        "logical_4bit_parameters": quantized,
        "logical_unquantized_parameters": unquantized,
        "expert_parameters": expected_expert_parameters,
        "expert_4bit_parameters": expert_parameters,
        "expert_4bit_matrices": expert_matrices,
        "expert_4bit_coverage": expert_parameters / expected_expert_parameters,
        "fp32_unquantized_parameters": fp32,
        "fp32_fraction_of_logical_parameters": fp32 / total,
        "parameter_storage_bytes": storage,
        "quantization_metadata_bytes": sum(metadata_tensors.values()),
    }
