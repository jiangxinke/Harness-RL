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
import uuid
from contextlib import contextmanager, nullcontext
from pathlib import Path

from .episode import PolicyCompletion
from .eval_sampling import EvaluationSampling
from .hf_policy import PROMPT_ENCODING_VERSION, HFPolicy, encode_messages
from .web_structured import (
    preflight_web_structured,
    validate_web_structured_schema,
    web_structured_actions_configuration,
    web_structured_configuration, web_multi_citation_configuration,
)

from .web_whitespace import (
    preflight_web_bounded_whitespace, web_bounded_whitespace_configuration,
    web_bounded_whitespace_grammar,
)
from .web_review import web_explicit_review_configuration
from .web_navigation import web_navigation_configuration
from .web_rows import web_cached_rows_configuration
from .web_research import web_research_configuration
from .web_data_discovery import web_data_discovery_configuration
from .web_review_decoder import (
    is_web_review_schema,
    preflight_web_review_decoder,
    web_review_grammar,
)


class VLLMPolicy(HFPolicy):
    """Reuse HF's serialized asynchronous batch queue for inference evaluation."""

    def __init__(self, engine, tokenizer, *, batch_size=8, max_prompt_tokens=16384,
                 max_model_len=24576, seed=42, lora_request=None, eos_token_ids=None,
                 allow_truncation=True, provenance=None, evaluation_sampling=None,
                 web_structured_final=False, web_structured_actions=False,
                 web_bounded_whitespace=False):
        if batch_size < 1 or max_prompt_tokens < 1 or max_model_len <= max_prompt_tokens:
            raise ValueError("Positive batch/prompt budgets and room for completion are required")
        super().__init__(engine, tokenizer, batch_size=batch_size,
                         max_prompt_tokens=max_prompt_tokens, allow_truncation=allow_truncation)
        self.max_model_len, self.seed = max_model_len, seed
        self.lora_request = lora_request
        self.eos_token_ids = sorted(set(eos_token_ids or [tokenizer.eos_token_id]) - {None})
        self.provenance = dict(provenance or {})
        self.evaluation_sampling = EvaluationSampling() if evaluation_sampling is None else evaluation_sampling
        if not isinstance(self.evaluation_sampling, EvaluationSampling):
            raise TypeError("evaluation_sampling must be an EvaluationSampling")
        self.web_structured_final = web_structured_configuration(web_structured_final)["enabled"]
        self.web_structured_actions = web_structured_actions_configuration(web_structured_actions)["enabled"]
        if self.web_structured_actions and not self.web_structured_final:
            raise ValueError("web_structured_actions requires web_structured_final=True")
        self.web_bounded_whitespace = web_bounded_whitespace_configuration(web_bounded_whitespace)["enabled"]
        if self.web_bounded_whitespace and not self.web_structured_final:
            raise ValueError("web_bounded_whitespace requires web_structured_final=True")
        if self.web_bounded_whitespace:
            configuration = web_bounded_whitespace_configuration(True)
            if self.provenance.get("web_bounded_whitespace", configuration) != configuration:
                raise ValueError("web_bounded_whitespace provenance does not match its configuration")
            self.provenance["web_bounded_whitespace"] = configuration
        elif self.provenance.get("web_bounded_whitespace", {}).get("enabled"):
            raise ValueError("web_bounded_whitespace provenance requires explicit opt-in")
        self._engine_lock = threading.Lock()
        self._closed = False

    def _sampling_options(self, temperature):
        if isinstance(temperature, bool) or temperature != self.evaluation_sampling.temperature:
            raise ValueError("vLLM defaults to greedy evaluation only; nonzero sampling must match "
                             "an explicit EvaluationSampling configuration; use HFPolicy for training")
        return self.evaluation_sampling.to_dict()

    def _structured_options(self, json_schema):
        if json_schema is None:
            return {}
        if not self.web_structured_final:
            raise ValueError("Structured generation requires web_structured_final=True")
        validate_web_structured_schema(json_schema)
        from vllm.sampling_params import StructuredOutputsParams

        review = self.provenance.get("web_explicit_review", {})
        reviewed = review.get("enabled") is True and is_web_review_schema(json_schema)
        if self.web_bounded_whitespace:
            return {"structured_outputs": StructuredOutputsParams(
                grammar=web_bounded_whitespace_grammar(json_schema, explicit_review=reviewed),
                disable_fallback=True,
            )}
        if reviewed:
            return {"structured_outputs": StructuredOutputsParams(
                grammar=web_review_grammar(json_schema), disable_fallback=True,
            )}
        return {"structured_outputs": StructuredOutputsParams(json=json_schema, disable_fallback=True)}

    async def generate_structured(self, messages, *, json_schema, max_new_tokens, temperature):
        results = await asyncio.to_thread(
            self.generate, [messages], max_new_tokens, temperature, json_schema=json_schema,
        )
        return results[0]

    def generate(self, messages_batch, max_new_tokens, temperature, *, json_schema=None):
        from vllm import SamplingParams

        if self._closed:
            raise RuntimeError("The vLLM engine has been closed")
        sampling_options = self._sampling_options(temperature)
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
            n=1, **sampling_options,
            max_tokens=max_new_tokens, seed=self.seed, ignore_eos=False,
            stop_token_ids=self.eos_token_ids, skip_special_tokens=True,
            **self._structured_options(json_schema),
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


class AsyncVLLMPolicy(VLLMPolicy):
    """Native per-request V1 scheduling and cancellation on one event loop."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._active_requests = set()
        self._owner_loop = None

    async def __call__(self, messages, *, max_new_tokens, temperature):
        return await self._generate(messages, max_new_tokens=max_new_tokens, temperature=temperature)

    async def generate_structured(self, messages, *, json_schema, max_new_tokens, temperature):
        return await self._generate(messages, max_new_tokens=max_new_tokens,
                                    temperature=temperature, json_schema=json_schema)

    async def _generate(self, messages, *, max_new_tokens, temperature, json_schema=None):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        if self._closed:
            raise RuntimeError("The vLLM engine has been closed")
        loop = asyncio.get_running_loop()
        if self._owner_loop is None:
            self._owner_loop = loop
        if loop is not self._owner_loop:
            raise RuntimeError("Async vLLM requests must use their owning event loop")
        sampling_options = self._sampling_options(temperature)
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        prompt = encode_messages(self.tokenizer, messages, self.max_prompt_tokens,
                                 allow_truncation=self.allow_truncation)
        if len(prompt) + max_new_tokens > self.max_model_len:
            raise ValueError("Prompt plus completion budget exceeds max_model_len; refusing to shorten it")
        params = SamplingParams(
            n=1, **sampling_options,
            max_tokens=max_new_tokens, seed=self.seed, ignore_eos=False,
            stop_token_ids=self.eos_token_ids, skip_special_tokens=True,
            output_kind=RequestOutputKind.FINAL_ONLY,
            **self._structured_options(json_schema),
        )
        request_id = uuid.uuid4().hex
        self._active_requests.add(request_id)
        completed, response = False, None
        try:
            async for item in self.model.generate(
                {"prompt_token_ids": prompt}, params, request_id,
                lora_request=self.lora_request,
            ):
                response = item
            if response is None or list(response.prompt_token_ids) != prompt or len(response.outputs) != 1:
                raise RuntimeError("vLLM response token provenance does not match its request")
            output = response.outputs[0]
            if output.finish_reason not in {"stop", "length"}:
                raise RuntimeError(f"vLLM request did not complete: {output.finish_reason}")
            tokens = list(output.token_ids)
            completed = True
            return PolicyCompletion(
                self.tokenizer.decode(tokens, skip_special_tokens=True), prompt, tokens,
                finish_reason=output.finish_reason,
            )
        finally:
            try:
                if not completed:
                    # AsyncLLM owns both the output processor and async engine
                    # socket here. Never abort synchronous LLM from a thread.
                    await self.model.abort(request_id)
            finally:
                self._active_requests.discard(request_id)

    def generate(self, *args, **kwargs):
        raise RuntimeError("Use await policy(...) with the native async vLLM backend")

    async def close(self):
        if self._closed:
            return
        if self._owner_loop is not None and asyncio.get_running_loop() is not self._owner_loop:
            raise RuntimeError("Async vLLM shutdown must use its owning event loop")
        self._closed = True
        try:
            if self._active_requests:
                await self.model.abort(list(self._active_requests))
        finally:
            self.model.shutdown()


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
    # Keep the caller's symlink location while making worker paths independent
    # of their current directory. Do not expose a mount's physical target.
    return {"path": str(path.absolute()), "config": config,
            "weights_file": weights.name, "weights_bytes": weights.stat().st_size,
            "weights_mtime_ns": weights.stat().st_mtime_ns}


def load_vllm_policy(model_name, *, adapter=None, batch_size=8, max_prompt_tokens=16384,
                     max_model_len=24576, tensor_parallel_size=1,
                     gpu_memory_utilization=0.88, max_num_batched_tokens=8192,
                     seed=42, revision=None, tokenizer_revision=None, enforce_eager=False,
                     asynchronous=False, evaluation_sampling=None, web_structured_final=False,
                     web_structured_actions=False, web_explicit_review=False,
                     web_bounded_whitespace=False, web_navigation_ledger=False, web_research_checkpoints=False, web_data_discovery_guidance=False, web_multi_citation=False, web_cached_rows=False,
                     quantization="none", fp8_validation_dir=None):
    """Load BF16 or explicit FP8 vLLM inference without changing the HF trainer.

    Save ``policy.provenance`` alongside prediction manifests in a new external
    output directory. A backend change must never resume an existing HF run.
    GPU ownership and CUDA_VISIBLE_DEVICES are managed by the caller.
    """
    if quantization not in {"none", "fp8"}:
        raise ValueError("The vLLM backend supports quantization='none' or 'fp8'")
    if quantization == "fp8" and tensor_parallel_size != 1:
        raise ValueError("The FP8-Marlin runtime requires tensor_parallel_size=1")
    evaluation_sampling = EvaluationSampling() if evaluation_sampling is None else evaluation_sampling
    if not isinstance(evaluation_sampling, EvaluationSampling):
        raise TypeError("evaluation_sampling must be an EvaluationSampling")
    structured = web_structured_configuration(web_structured_final)
    actions = web_structured_actions_configuration(web_structured_actions)
    review = web_explicit_review_configuration(web_explicit_review)
    navigation = web_navigation_configuration(web_navigation_ledger)
    if web_navigation_ledger and not (web_structured_actions and web_structured_final):
        raise ValueError("web_navigation_ledger requires structured Web actions and final")
    multi_citation = web_multi_citation_configuration(web_multi_citation)
    if web_multi_citation and not (web_structured_actions and web_structured_final):
        raise ValueError("web_multi_citation requires structured Web actions and final")
    discovery = web_data_discovery_configuration(web_data_discovery_guidance)
    if web_data_discovery_guidance and not (web_structured_actions and web_structured_final):
        raise ValueError("web_data_discovery_guidance requires structured Web actions and final")
    research = web_research_configuration(web_research_checkpoints)
    if web_research_checkpoints and not (web_structured_actions and web_structured_final):
        raise ValueError("web_research_checkpoints requires structured Web actions and final")
    cached_rows = web_cached_rows_configuration(web_cached_rows)
    if web_cached_rows and not (web_structured_actions and web_structured_final):
        raise ValueError("web_cached_rows requires structured Web actions and final")
    cached_options = {"cached_rows": True} if web_cached_rows else {}
    if web_multi_citation:
        cached_options["multi_citation"] = True
    whitespace = web_bounded_whitespace_configuration(web_bounded_whitespace)
    if web_bounded_whitespace and not web_structured_final:
        raise ValueError("web_bounded_whitespace requires web_structured_final=True")
    if web_explicit_review and not (web_structured_actions and web_structured_final):
        raise ValueError("web_explicit_review requires structured Web actions and final")
    if web_structured_actions and not web_structured_final:
        raise ValueError("web_structured_actions requires web_structured_final=True")
    if web_explicit_review:
        preflight_web_structured(actions=True, explicit_review=True, **cached_options)
        review_decoder = preflight_web_review_decoder(**cached_options)
    elif web_structured_actions:
        preflight_web_structured(actions=True, **cached_options)
    elif web_structured_final:
        preflight_web_structured()
    if web_bounded_whitespace:
        preflight_web_bounded_whitespace(explicit_review=web_explicit_review, **cached_options)
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
    if quantization == "fp8":
        from .fp8_attention_guard import validate_model_config
        validate_model_config(config.to_dict())
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
        quantization=None if quantization == "none" else "fp8",
        kv_cache_dtype="auto", tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
        max_num_seqs=batch_size, max_num_batched_tokens=max_num_batched_tokens,
        enable_chunked_prefill=True, enable_prefix_caching=True, seed=seed,
        generation_config="vllm", enforce_eager=enforce_eager,
        distributed_executor_backend="uni" if tensor_parallel_size == 1 else "mp",
        enable_lora=bool(lora), max_lora_rank=max_lora_rank, max_loras=1, lora_dtype="auto",
        swap_space=0, cpu_offload_gb=0,
    )
    if web_structured_final:
        from vllm.config import StructuredOutputsConfig

        engine_kwargs["structured_outputs_config"] = StructuredOutputsConfig(
            backend="xgrammar", disable_fallback=True,
        )
    binding = nullcontext()
    if quantization == "fp8":
        from .fp8_attention_guard import WORKER_CLASS, worker_binding
        engine_kwargs["worker_cls"] = WORKER_CLASS
        binding = worker_binding(adapter=metadata["path"] if metadata else None,
                                 model_config=config.to_dict(), audit_dir=fp8_validation_dir)
    with binding, _engine_environment():
        if asynchronous:
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.v1.engine.async_llm import AsyncLLM

            engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
                **engine_kwargs, disable_log_stats=True,
            ))
        else:
            engine = LLM(**engine_kwargs)
    provenance = {
        "backend": "vllm", "vllm_version": vllm.__version__,
        "torch_version": importlib.metadata.version("torch"),
        "transformers_version": transformers.__version__,
        "model": model_name, "revision": revision,
        "resolved_config_revision": getattr(config, "_commit_hash", None),
        "tokenizer_revision": tokenizer_revision or revision,
        "adapter": metadata, "dtype": "bfloat16", "quantization": quantization,
        "lora_dtype": "auto (bfloat16)",
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
        "evaluation_sampling": evaluation_sampling.to_dict(),
        "prompt_encoding": "HF encode_messages; enable_thinking=False",
        "prompt_encoding_version": PROMPT_ENCODING_VERSION,
    }
    if quantization == "fp8":
        from .fp8_attention_guard import QUANTIZATION, VALIDATION_LAYOUT
        from .fp8_runtime import runtime_identity
        provenance.update(fp8_contract=dict(QUANTIZATION), fp8_validation=dict(VALIDATION_LAYOUT),
                          runtime_code_identity=runtime_identity())
    if evaluation_sampling != EvaluationSampling():
        provenance["sampling"] = "explicit evaluation sampling; explicit EOS; generation_config=vllm"
    if web_structured_final:
        provenance["web_structured_final"] = structured
    if web_structured_actions:
        provenance["web_structured_actions"] = actions
    if web_explicit_review:
        provenance["web_explicit_review"] = review
        provenance["web_review_decoder"] = review_decoder
    if web_bounded_whitespace:
        provenance["web_bounded_whitespace"] = whitespace
    if web_navigation_ledger:
        provenance["web_navigation_ledger"] = navigation
    if web_research_checkpoints:
        provenance["web_research_checkpoints"] = research
    if web_data_discovery_guidance:
        provenance["web_data_discovery_guidance"] = discovery
    if web_multi_citation:
        provenance["web_multi_citation"] = multi_citation
    if web_cached_rows:
        provenance["web_cached_rows"] = cached_rows
    if asynchronous:
        provenance.update(async_engine=True, request_completion_delivery="per-request",
                          timeout_cancellation="native AsyncLLM.abort on owner event loop")
    policy_class = AsyncVLLMPolicy if asynchronous else VLLMPolicy
    return policy_class(engine, tokenizer, batch_size=batch_size,
                      max_prompt_tokens=max_prompt_tokens, max_model_len=max_model_len,
                      seed=seed, lora_request=lora, eos_token_ids=eos, provenance=provenance,
                      evaluation_sampling=evaluation_sampling,
                      web_structured_final=web_structured_final,
                      web_structured_actions=web_structured_actions,
                      web_bounded_whitespace=web_bounded_whitespace)
