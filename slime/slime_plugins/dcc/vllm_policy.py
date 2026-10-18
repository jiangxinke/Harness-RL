"""Optional inference-only vLLM policy with the HF harness prompt encoding.

Importing this module does not import vLLM or start an engine. The existing HF
trainer remains the source of trainable rollout probabilities.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path

from .episode import PolicyCompletion
from .hf_policy import HFPolicy, encode_messages


class VLLMPolicy(HFPolicy):
    """Reuse HF's serialized asynchronous batch queue for greedy evaluation."""

    def __init__(self, engine, tokenizer, *, batch_size=8, max_prompt_tokens=16384,
                 max_model_len=24576, seed=42, lora_request=None, eos_token_ids=None,
                 allow_truncation=True, provenance=None):
        if batch_size < 1 or max_prompt_tokens < 1 or max_model_len <= max_prompt_tokens:
            raise ValueError("Positive batch/prompt budgets and room for completion are required")
        super().__init__(engine, tokenizer, batch_size=batch_size,
                         max_prompt_tokens=max_prompt_tokens, allow_truncation=allow_truncation)
        self.max_model_len, self.seed = max_model_len, seed
        self.lora_request = lora_request
        self.eos_token_ids = sorted(set(eos_token_ids or [tokenizer.eos_token_id]) - {None})
        self.provenance = dict(provenance or {})
        self._engine_lock = threading.Lock()
        self._closed = False

    def generate(self, messages_batch, max_new_tokens, temperature):
        from vllm import SamplingParams

        if self._closed:
            raise RuntimeError("The vLLM engine has been closed")
        if temperature != 0:
            raise ValueError("VLLMPolicy supports greedy evaluation only; use HFPolicy for training")
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if not messages_batch:
            return []
        prompts = [encode_messages(self.tokenizer, messages, self.max_prompt_tokens,
                                   allow_truncation=self.allow_truncation)
                   for messages in messages_batch]
        if any(len(prompt) + max_new_tokens > self.max_model_len for prompt in prompts):
            raise ValueError("Prompt plus completion budget exceeds max_model_len; refusing to shorten it")
        sampling = SamplingParams(
            n=1, temperature=0.0, top_p=1.0, top_k=-1,
            presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0,
            max_tokens=max_new_tokens, seed=self.seed, ignore_eos=False,
            stop_token_ids=self.eos_token_ids, skip_special_tokens=True,
        )
        with self._engine_lock:
            responses = self.model.generate(
                [{"prompt_token_ids": prompt} for prompt in prompts], sampling,
                lora_request=self.lora_request, use_tqdm=False,
            )
        if len(responses) != len(prompts):
            raise RuntimeError("vLLM returned the wrong number of requests")
        result = []
        for prompt, response in zip(prompts, responses, strict=True):
            if list(response.prompt_token_ids) != prompt or len(response.outputs) != 1:
                raise RuntimeError("vLLM response ordering or token provenance does not match its input")
            output = response.outputs[0]
            if output.finish_reason not in {"stop", "length"}:
                raise RuntimeError(f"vLLM request did not complete: {output.finish_reason}")
            tokens = list(output.token_ids)
            result.append(PolicyCompletion(
                self.tokenizer.decode(tokens, skip_special_tokens=True), prompt, tokens,
                finish_reason=output.finish_reason,
            ))
        return result


    async def close(self):
        await super().close()

        def shutdown():
            with self._engine_lock:
                if self._closed:
                    return
                self._closed = True
                # vLLM 0.11.2 V1's EngineCoreClient.shutdown() finalizes its
                # background resources and engine/worker subprocesses.
                core = getattr(getattr(self.model, "llm_engine", None), "engine_core", None)
                if core is not None:
                    core.shutdown()

        await asyncio.to_thread(shutdown)


def _validate_launch_environment(tensor_parallel_size):
    import torch

    for name, expected in (("WORLD_SIZE", 1), ("RANK", 0), ("LOCAL_RANK", 0)):
        if int(os.environ.get(name, expected)) != expected:
            raise ValueError("vLLM requires one launcher process; do not use multi-rank torchrun. "
                             "Select its GPUs with CUDA_VISIBLE_DEVICES and use tensor_parallel_size.")
    if torch.distributed.is_initialized():
        raise ValueError("Create the vLLM backend in a fresh process without an existing process group")
    visible = torch.cuda.device_count()
    if visible != tensor_parallel_size:
        raise ValueError(f"Select exactly {tensor_parallel_size} visible GPUs for this engine "
                         f"using CUDA_VISIBLE_DEVICES; found {visible}")


@contextmanager
def _engine_environment():
    # Pinned V1 uni/mp executors assign ranks and a fresh TCP rendezvous
    # explicitly. External-launcher environment must not leak into children.
    keys = {"RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
            "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"}
    keys.update(key for key in os.environ if key.startswith("TORCHELASTIC_"))
    forced = {"VLLM_USE_V1": "1", "VLLM_ENABLE_V1_MULTIPROCESSING": "1",
              "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
    saved = {key: os.environ.get(key) for key in keys | forced.keys()}
    try:
        for key in keys:
            os.environ.pop(key, None)
        os.environ.update(forced)
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _adapter_metadata(adapter):
    if adapter is None:
        return None
    path = Path(adapter)
    config = json.loads((path / "adapter_config.json").read_text())
    targets = config.get("target_modules")
    if (config.get("peft_type") != "LORA" or not isinstance(targets, list)
            or not targets or not set(targets) <= {"q_proj", "k_proj", "v_proj", "o_proj"}):
        raise ValueError("The vLLM backend requires a local attention-only PEFT LoRA adapter")
    rank = config.get("r")
    if type(rank) is not int or rank < 1 or rank > 64:
        raise ValueError("Supported attention LoRA ranks are 1 through 64")
    if config.get("bias", "none") != "none" or config.get("modules_to_save"):
        raise ValueError("LoRA bias and modules_to_save are unsupported by this backend")
    unsupported = ("use_dora", "use_rslora", "rank_pattern", "alpha_pattern",
                   "fan_in_fan_out", "lora_bias", "target_parameters",
                   "layer_replication", "alora_invocation_tokens", "use_qalora")
    if any(config.get(key) for key in unsupported):
        raise ValueError("The vLLM backend requires standard LoRA without DoRA, rsLoRA or per-layer variants")
    weights = path / "adapter_model.safetensors"
    if not weights.is_file():
        raise ValueError("A local adapter_model.safetensors checkpoint is required")
    return {"path": str(path.resolve()), "config": config,
            "weights_file": weights.name, "weights_bytes": weights.stat().st_size,
            "weights_mtime_ns": weights.stat().st_mtime_ns}


def load_vllm_policy(model_name, *, adapter=None, batch_size=8, max_prompt_tokens=16384,
                     max_model_len=24576, tensor_parallel_size=1,
                     gpu_memory_utilization=0.88, max_num_batched_tokens=8192,
                     seed=42, revision=None, tokenizer_revision=None, enforce_eager=False):
    """Load a BF16 vLLM engine without changing the default HF runtime.

    Save ``policy.provenance`` alongside prediction manifests in a new external
    output directory. A backend change must never resume an existing HF run.
    GPU ownership and CUDA_VISIBLE_DEVICES are managed by the caller.
    """
    _validate_launch_environment(tensor_parallel_size)

    import transformers
    import vllm
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    if vllm.__version__ != "0.11.2":
        raise ValueError("Use the separate requirements-vllm.txt environment with vllm==0.11.2")
    if batch_size < 1 or max_prompt_tokens < 1 or max_model_len <= max_prompt_tokens:
        raise ValueError("Positive batch/prompt budgets and room for completion are required")
    if tensor_parallel_size < 1 or not 0 < gpu_memory_utilization < 1 or max_num_batched_tokens < 1:
        raise ValueError("Invalid vLLM parallelism, memory fraction or token batch budget")
    config = AutoConfig.from_pretrained(model_name, revision=revision)
    if config.model_type != "qwen3_moe" or getattr(config, "quantization_config", None):
        raise ValueError("This backend currently requires an unquantized Qwen3 MoE base checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=tokenizer_revision or revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    generation = GenerationConfig.from_pretrained(model_name, revision=revision)
    eos = generation.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    eos = sorted(set(eos + [tokenizer.eos_token_id]) - {None})
    metadata = _adapter_metadata(adapter)
    lora = LoRARequest("dcc", 1, metadata["path"]) if metadata else None
    rank = metadata["config"]["r"] if metadata else 8
    max_lora_rank = next(value for value in (8, 16, 32, 64) if value >= rank)
    engine_kwargs = dict(
        model=model_name, tokenizer=model_name, revision=revision,
        tokenizer_revision=tokenizer_revision or revision, dtype="bfloat16",
        quantization=None, kv_cache_dtype="auto", tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
        max_num_seqs=batch_size, max_num_batched_tokens=max_num_batched_tokens,
        enable_chunked_prefill=True, enable_prefix_caching=True, seed=seed,
        generation_config="vllm", enforce_eager=enforce_eager,
        distributed_executor_backend="uni" if tensor_parallel_size == 1 else "mp",
        enable_lora=bool(lora), max_lora_rank=max_lora_rank, max_loras=1,
        swap_space=0, cpu_offload_gb=0,
    )
    with _engine_environment():
        engine = LLM(**engine_kwargs)
    provenance = {
        "backend": "vllm", "vllm_version": vllm.__version__,
        "torch_version": importlib.metadata.version("torch"),
        "transformers_version": transformers.__version__,
        "model": model_name, "revision": revision,
        "resolved_config_revision": getattr(config, "_commit_hash", None),
        "tokenizer_revision": tokenizer_revision or revision,
        "adapter": metadata, "dtype": "bfloat16", "quantization": "none",
        "kv_cache_dtype": "auto (bfloat16)", "seed": seed,
        "tensor_parallel_size": tensor_parallel_size, "max_model_len": max_model_len,
        "launcher_world_size": 1, "worker_start_method": "spawn",
        "distributed_executor_backend": engine_kwargs["distributed_executor_backend"],
        "max_prompt_tokens": max_prompt_tokens, "batch_size": batch_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_num_batched_tokens": max_num_batched_tokens,
        "enable_chunked_prefill": True, "enable_prefix_caching": True,
        "enforce_eager": enforce_eager, "eos_token_ids": eos,
        "sampling": "greedy; neutral penalties; explicit EOS; generation_config=vllm",
        "prompt_encoding": "HF encode_messages; enable_thinking=False",
    }
    return VLLMPolicy(engine, tokenizer, batch_size=batch_size,
                      max_prompt_tokens=max_prompt_tokens, max_model_len=max_model_len,
                      seed=seed, lora_request=lora, eos_token_ids=eos, provenance=provenance)
