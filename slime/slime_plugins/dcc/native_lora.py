"""Explicit native LoRA evaluation binding; no model/tensor loading or engine.

Uses the existing attention-LoRA metadata checker plus safetensors header shapes.
No new hash pass, generation protocol, sampling or token budget is introduced.
"""
from __future__ import annotations
import json
from pathlib import Path
import re

from .vllm_policy import _adapter_metadata


def validate_native_adapter(adapter, *, source_receipt, approved_derivative_id, model_config,
                            approved_training_source_receipt=None, approved_training_protocol=None,
                            evaluation_protocol=None, approved_dpo=None):
    if approved_dpo is not None:
        if approved_training_source_receipt is not None or approved_training_protocol is not None:
            raise ValueError("DPO approval and accepted-SFT training approval are mutually exclusive")
        from .native_dpo_adapter import validate_native_dpo_adapter
        return validate_native_dpo_adapter(adapter, source_receipt=source_receipt,
            approved_derivative_id=approved_derivative_id, model_config=model_config,
            evaluation_protocol=evaluation_protocol, approved_dpo=approved_dpo)
    if not isinstance(approved_derivative_id, str) or not approved_derivative_id.strip():
        raise ValueError("Native LoRA requires a separately approved derivative artifact ID")
    metadata = _adapter_metadata(adapter)
    if metadata is None:
        raise ValueError("An actual local native LoRA checkpoint is required")
    path = Path(metadata["path"])
    from .native_thinking_eval import MODEL_REVISION, PROTOCOL, _strict_json, validate_source_receipt
    from .native_budget import LEGACY_PROTOCOL, BUDGET_PROTOCOL, budget_from_configuration
    explicit_training = approved_training_source_receipt is not None or approved_training_protocol is not None
    if explicit_training:
        if approved_training_source_receipt is None or approved_training_protocol not in {LEGACY_PROTOCOL, BUDGET_PROTOCOL}:
            raise ValueError("Explicit actual native training source/supported protocol required together")
        training_receipt = validate_source_receipt(approved_training_source_receipt)
    else:
        training_receipt = source_receipt
    training_protocol = approved_training_protocol if explicit_training else PROTOCOL
    config = _strict_json((path / "training_config.json").read_text())
    native = config.get("native_thinking")
    if (not isinstance(native, dict) or native.get("enabled") is not True
            or native.get("protocol") != training_protocol or native.get("source_receipt") != training_receipt
            or native.get("derivative_id") != approved_derivative_id
            or native.get("model_revision") != MODEL_REVISION
            or native.get("all_files_validated_before_model_load") is not True
            or native.get("warm_start") is not False or native.get("raw_cuad_auxiliary") is not False
            or any(type(native.get(key)) is not int or native[key] < 0
                   for key in ("validated_records", "validated_cuad_records"))
            or native["validated_records"] + native["validated_cuad_records"] < 1
            or type(config.get("steps")) is not int or config["steps"] < 1
            or config.get("lora_rank") != metadata["config"]["r"]):
        raise ValueError("Actual adapter TRAIN configuration differs from approved native source/derivative")
    if training_protocol == BUDGET_PROTOCOL:
        # These original TRAIN exports predate the CUAD inference backend fix.
        # Validate their exact saved declaration without relabelling TRAIN.
        if training_receipt["source_commit"] in {
                "76b57e227b943fb34a446226fa67cc1420f40e52",
                "cf89170a00eb7c25bf5de038558c479228a87587"}:
            from .legacy_native_training_configuration import native_configuration
        else:
            from .native_thinking_eval import native_configuration
        rollout = native.get("rollout_native_thinking")
        if not isinstance(rollout, dict):
            raise ValueError("Budgeted native TRAIN requires its actual rollout configuration")
        budget = budget_from_configuration(rollout)
        if (budget.protocol != training_protocol
                or rollout != native_configuration(training_receipt, budget=budget)):
            raise ValueError("Actual budgeted TRAIN rollout differs from approved training source/protocol")
    rank, tensor_pairs = validate_native_attention_headers(path, metadata, model_config)
    # Recheck the existing lightweight identity after reading headers/config.
    if _adapter_metadata(adapter) != metadata:
        raise ValueError("Adapter changed during native preflight")
    binding = {"derivative_id": approved_derivative_id, "training_source_receipt": training_receipt,
               "protocol": training_protocol, "model_revision": MODEL_REVISION, "lora_rank": rank,
               "tensor_pairs": tensor_pairs, "checked_actual_attention_shapes": True}
    if training_protocol == BUDGET_PROTOCOL:
        # Bind the actual saved TRAIN budget as well as its source/protocol.
        # Approval of the source/protocol alone is not approval of each budget;
        # the frozen training plan and update audit establish that separately.
        binding["training_rollout_native_thinking"] = native["rollout_native_thinking"]
    if explicit_training:
        if evaluation_protocol not in {LEGACY_PROTOCOL, BUDGET_PROTOCOL}:
            raise ValueError("The actual evaluation protocol must be declared")
        binding.update(evaluation_source_receipt=validate_source_receipt(source_receipt),
            evaluation_protocol=evaluation_protocol,
            caller_approved_training={"source_commit": training_receipt["source_commit"], "protocol": training_protocol})
    return metadata, binding

def validate_native_attention_headers(path, metadata, model_config):
    """Original standard attention shape/dtype/completeness guards, shared."""
    if not isinstance(model_config, dict):
        raise ValueError("Actual local base model configuration is required")
    hidden, layers = model_config.get("hidden_size"), model_config.get("num_hidden_layers")
    heads, kv_heads = model_config.get("num_attention_heads"), model_config.get("num_key_value_heads")
    if any(type(value) is not int or value < 1 for value in (hidden, layers, heads, kv_heads)):
        raise ValueError("Actual base attention dimensions are invalid")
    dimension = model_config.get("head_dim", hidden // heads)
    if type(dimension) is not int or dimension < 1:
        raise ValueError("Actual base head dimension is invalid")
    shapes = {"q_proj": (hidden, heads*dimension), "k_proj": (hidden, kv_heads*dimension),
              "v_proj": (hidden, kv_heads*dimension), "o_proj": (heads*dimension, hidden)}
    targets, rank = set(metadata["config"]["target_modules"]), metadata["config"]["r"]
    from safetensors import safe_open
    found = set()
    # get_slice().get_shape() reads headers; no pretrained/adapter tensors loaded.
    with safe_open(str(path / "adapter_model.safetensors"), framework="pt", device="cpu") as stream:
        for key in stream.keys():
            match = re.fullmatch(r"base_model\.model\.model\.layers\.(\d+)\.self_attn\.([qkvo]_proj)\.lora_([AB])\.weight", key)
            if not match:
                raise ValueError("Unexpected adapter tensor outside standard attention LoRA")
            layer, target, side = int(match[1]), match[2], match[3]
            if layer >= layers or target not in targets:
                raise ValueError("Adapter target/layer differs from actual base configuration")
            inputs, outputs = shapes[target]
            header = stream.get_slice(key)
            if header.get_dtype() not in {"F16", "BF16", "F32"}:
                raise ValueError("Native LoRA weights must use a floating-point adapter dtype")
            if list(header.get_shape()) != ([rank, inputs] if side == "A" else [outputs, rank]):
                raise ValueError("Adapter shape differs from actual base attention dimensions")
            found.add((layer,target,side))
    expected = {(layer,target,side) for layer in range(layers) for target in targets for side in ("A","B")}
    if found != expected:
        raise ValueError("Native LoRA checkpoint is empty or missing configured attention pairs")
    return rank, len(found) // 2
