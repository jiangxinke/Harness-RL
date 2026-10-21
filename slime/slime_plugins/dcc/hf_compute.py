"""Explicit shared HF compute contexts; base/LoRA storage stays unchanged."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import json
from pathlib import Path

import torch

LEGACY = "legacy"
BF16 = "bf16-autocast-v1"
COMPUTE_PROFILES = (LEGACY, BF16)
VERSION = "dcc-hf-compute-profile-v1"


def compute_profile_configuration(profile=LEGACY):
    if not isinstance(profile, str) or profile not in COMPUTE_PROFILES:
        raise ValueError("Unknown HF compute profile")
    return {"version": VERSION, "profile": profile,
            "autocast": profile == BF16, "autocast_dtype": "torch.bfloat16" if profile == BF16 else None,
            "parameter_storage": "unchanged-PEFT-preparation-and-FP32-LoRA-master-weights",
            "log_softmax_dtype": "torch.float32"}


def _checked_configuration(value):
    if (not isinstance(value, dict) or type(value.get("autocast")) is not bool
            or value != compute_profile_configuration(value.get("profile"))):
        raise ValueError("Incomplete or changed HF compute profile configuration")
    return dict(value)


def adapter_compute_profile(adapter):
    """Old adapters without a profile retain their original legacy context."""
    path = Path(adapter) / "training_config.json"
    if not path.is_file():
        return compute_profile_configuration()
    config = json.loads(path.read_text(encoding="utf-8"))
    arguments = config.get("arguments", {})
    declared = config.get("compute_profile", compute_profile_configuration())
    declared = _checked_configuration(declared)
    if ("compute_profile" in arguments
            and compute_profile_configuration(arguments["compute_profile"]) != declared):
        raise ValueError("Adapter training arguments and compute profile disagree")
    precision = config.get("precision")
    if declared["profile"] != LEGACY and precision is None:
        raise ValueError("BF16 adapter is missing its recorded training precision")
    if precision is not None and (not isinstance(precision, dict) or precision.get("compute_profile") != declared):
        raise ValueError("Adapter training precision and compute profile disagree")
    return declared


def validate_compute_profile(profile=LEGACY, *, adapter=None):
    requested = compute_profile_configuration(profile)
    if adapter and adapter_compute_profile(adapter) != requested:
        raise ValueError("Requested HF compute profile differs from adapter training profile")
    return requested


def model_compute_profile(model):
    return _checked_configuration(getattr(model, "dcc_compute_profile", compute_profile_configuration()))


def validate_model_compute_profile(model, profile=LEGACY):
    if model_compute_profile(model) != compute_profile_configuration(profile):
        raise ValueError("Loaded model and requested HF compute profile disagree")


def _device_type(model):
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    device_type = torch.device(device).type
    if device_type not in {"cpu", "cuda"}:
        raise ValueError("HF compute profiles support explicit CPU/CUDA runtimes only")
    return device_type


@contextmanager
def compute_context(model):
    """Use the same context for sampling, old/current replay and evaluation.

    The observation describes the actual active autocast context, not a claim
    that every operator or tensor is BF16. FP32 log-softmax stays outside the
    model computation. Checkpoint recomputation preserves PyTorch's context.
    """
    config, device_type = model_compute_profile(model), _device_type(model)
    context = (torch.autocast(device_type=device_type, dtype=torch.bfloat16)
               if config["autocast"] else nullcontext())
    with context:
        observed = {"device_type": device_type,
                    "autocast_enabled": torch.is_autocast_enabled(device_type),
                    "autocast_dtype": str(torch.get_autocast_dtype(device_type))
                                      if torch.is_autocast_enabled(device_type) else None}
        if (observed["autocast_enabled"] != config["autocast"]
                or observed["autocast_dtype"] != config["autocast_dtype"]):
            raise ValueError("Active autocast context differs from the HF compute profile")
        previous = getattr(model, "dcc_compute_context_observation", observed)
        if previous != observed:
            raise ValueError("HF compute context changed during this model's runtime")
        model.dcc_compute_context_observation = observed
        yield


def configure_compute_profile(model, profile=LEGACY):
    model.dcc_compute_profile = compute_profile_configuration(profile)
    with compute_context(model):
        pass


def precision_observation(model):
    """Observe registered storage dtypes and the actual selected context."""
    base, adapter = {}, {}
    for name, parameter in model.named_parameters():
        counts = adapter if "lora_" in name else base
        dtype = str(parameter.dtype)
        counts[dtype] = counts.get(dtype, 0) + parameter.numel()
    with compute_context(model):
        pass
    return dict(base_parameter_dtypes=base, adapter_parameter_dtypes=adapter,
                quantization_report=getattr(model, "dcc_quantization_report", None),
                compute_profile=model_compute_profile(model),
                compute_context=dict(model.dcc_compute_context_observation))


def validate_adapter_precision(model, adapter):
    """New checkpoints must reload the precision they actually trained with."""
    path = Path(adapter) / "training_config.json"
    config = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    expected = config.get("precision")
    if expected is None:
        if model_compute_profile(model)["profile"] != LEGACY:
            raise ValueError("BF16 adapter is missing its recorded training precision")
        return
    if expected != precision_observation(model):
        raise ValueError("Reloaded adapter precision differs from its training precision")
