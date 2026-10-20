"""Portable identity for the optional FP8-Marlin attention-LoRA runtime."""
from copy import deepcopy

IDENTITY = {
    "schema": "dcc-fp8-attention-runtime-v1",
    "implementation": "qwen3-moe-fp8-marlin-attention-lora-bf16-routers-v1",
    "quantization": "fp8",
    "weight_format": "FP8 E4M3 per-tensor online from BF16",
    "quantization_kernel": "Marlin W8A16",
    "activation_dtype": "bfloat16",
    "router_dtype": "bfloat16",
    "router_quantization": "none",
    "lora_scope": "attention projections only; fused experts excluded",
    "worker_extension": "slime_plugins.dcc.fp8_attention_worker.FP8AttentionWorker",
    "vllm_version": "0.11.2",
    "compute_capability": [8, 0],
    "tensor_parallel_size": 1,
}


def runtime_identity():
    return deepcopy(IDENTITY)
