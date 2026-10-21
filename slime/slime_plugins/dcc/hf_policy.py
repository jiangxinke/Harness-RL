"""Local Transformers policy with batched generation and exact rollout tokens.

This is an optional, independent LoRA runtime. Importing the DCC harness does
not import Transformers or change the existing Slime/Megatron trainer.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict

import torch

from .episode import PolicyCompletion


PROMPT_ENCODING_VERSION = "message-boundary-source-excerpts-v2"
_SOURCE_PREFIXES = (
    "Initial contract retrieval (source material):\n",
    "Tool observation (source material):\n",
)
_SOURCE_OMISSION = "\n[Source content omitted to fit context.]\n"


def _chat_tokens(tokenizer, messages, *, generation_prompt=True):
    tokens = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=generation_prompt,
        enable_thinking=False,
    )
    return list(tokens["input_ids"] if hasattr(tokens, "keys") else tokens)


def _source_excerpt(content, characters, *, initial):
    # Runner-added controls follow the source body. They must survive even when
    # one observation is too large; never obtain them from a sliced token tail.
    control_start = len(content)
    for marker in ("\nRemaining turns:", "\nFinal turn:"):
        position = content.rfind(marker)
        if position >= 0:
            control_start = min(control_start, position)
    source, controls = content[:control_start], content[control_start:]
    prefix = next((value for value in _SOURCE_PREFIXES if source.startswith(value)), "")
    body = source[len(prefix):]
    characters = min(characters, len(body))
    if initial:
        # Initial CUAD hits are ranked: keep the highest-ranked source first.
        excerpt = body[:characters]
    else:
        head = (characters + 1) // 2
        tail = characters // 2
        excerpt = body[:head] + _SOURCE_OMISSION + (body[-tail:] if tail else "")
    if initial:
        excerpt += _SOURCE_OMISSION
    return prefix + excerpt + controls


def encode_messages(tokenizer, messages: list[dict[str, str]], max_prompt_tokens: int, *, allow_truncation: bool = True) -> list[int]:
    """Keep the task and recent messages, re-templating any source excerpt.

    At most one boundary source message is shortened. No assistant completion,
    system instruction, task, role delimiter, or final-turn control is sliced.
    Excerpts use observed source text only and do not inspect task labels.
    """
    if type(max_prompt_tokens) is not int or max_prompt_tokens < 1:
        raise ValueError("A positive integer prompt budget is required")
    tokens = _chat_tokens(tokenizer, messages)
    if len(tokens) <= max_prompt_tokens:
        return tokens
    if not allow_truncation:
        raise ValueError("Judge input exceeds its context budget; refusing to truncate evidence")
    anchors = messages[:2]
    retained = _chat_tokens(tokenizer, anchors)
    if len(retained) > max_prompt_tokens or len(messages) <= 2:
        raise ValueError("System prompt and question exceed the configured prompt budget")

    recent = []
    for index in range(len(messages) - 1, 1, -1):
        message = messages[index]
        candidate = _chat_tokens(tokenizer, anchors + [message] + recent)
        if len(candidate) <= max_prompt_tokens:
            recent.insert(0, message)
            retained = candidate
            continue
        source_message = message["role"] == "tool" or (
            message["role"] == "user" and (
                index == 2 or message["content"].startswith(_SOURCE_PREFIXES)
            )
        )
        if source_message:
            # Search character boundaries, then render the complete message.
            # Unicode and chat special tokens are never cut through raw IDs.
            initial = index == 2 and (
                len(messages) == 3 or message["content"].startswith(_SOURCE_PREFIXES[0])
            )
            low, high, fitted = 0, len(message["content"]), None
            while low <= high:
                size = (low + high) // 2
                excerpt = {**message, "content": _source_excerpt(message["content"], size, initial=initial)}
                candidate = _chat_tokens(tokenizer, anchors + [excerpt] + recent)
                if len(candidate) <= max_prompt_tokens:
                    fitted = candidate
                    low = size + 1
                else:
                    high = size - 1
            if fitted is not None:
                return fitted
        if recent:
            return retained
        raise ValueError("Latest source observation and its controls cannot fit the prompt budget")
    return retained


class HFPolicy:
    def __init__(self, model, tokenizer, *, batch_size: int = 4, max_prompt_tokens: int = 4096, allow_truncation: bool = True, raw_sampling: bool = False):
        self.model, self.tokenizer = model, tokenizer
        self.batch_size, self.max_prompt_tokens = batch_size, max_prompt_tokens
        self.allow_truncation = allow_truncation
        self.raw_sampling = raw_sampling
        self.queue: asyncio.Queue = asyncio.Queue()
        self.worker: asyncio.Task | None = None

    async def __call__(self, messages, *, max_new_tokens: int, temperature: float) -> PolicyCompletion:
        if self.worker is None:
            self.worker = asyncio.create_task(self._work())
        future = asyncio.get_running_loop().create_future()
        await self.queue.put((messages, max_new_tokens, temperature, future))
        return await future

    async def close(self):
        if self.worker is not None:
            await self.queue.put(None)
            await self.worker
            self.worker = None

    async def _work(self):
        while True:
            first = await self.queue.get()
            if first is None:
                return
            await asyncio.sleep(0.02)
            requests, stop = [first], False
            while len(requests) < self.batch_size and not self.queue.empty():
                item = self.queue.get_nowait()
                if item is None:
                    stop = True
                    break
                requests.append(item)
            groups: dict[tuple[int, float], list] = defaultdict(list)
            for request in requests:
                if not request[3].done():
                    groups[(request[1], request[2])].append(request)
            for (limit, temperature), group in groups.items():
                try:
                    # The worker serializes model access. Tools remain free to
                    # perform network I/O while CUDA generation runs in a thread.
                    values = await asyncio.to_thread(self.generate, [item[0] for item in group], limit, temperature)
                    for request, value in zip(group, values, strict=True):
                        if not request[3].done():
                            request[3].set_result(value)
                except Exception as exc:
                    for request in group:
                        if not request[3].done():
                            request[3].set_exception(exc)
            if stop:
                return

    @torch.inference_mode()
    def generate(self, messages_batch, max_new_tokens: int, temperature: float) -> list[PolicyCompletion]:
        if self.raw_sampling and temperature > 0 and temperature != 1.0:
            raise ValueError("Raw policy sampling requires temperature=1")
        prompts = [encode_messages(self.tokenizer, messages, self.max_prompt_tokens, allow_truncation=self.allow_truncation) for messages in messages_batch]
        width = max(map(len, prompts))
        pad = self.tokenizer.pad_token_id
        ids = torch.tensor([[pad] * (width - len(p)) + p for p in prompts], device=self.model.device)
        attention = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=ids.device)
        self.model.eval()
        options = {
            "do_sample": temperature > 0,
            "temperature": temperature if temperature > 0 else None,
            "top_p": 0.95 if temperature > 0 else None,
            "top_k": 20 if temperature > 0 else None,
        }
        if self.raw_sampling and temperature > 0:
            options.update(temperature=1.0, top_p=1.0, top_k=0)
            # Explicit kwargs override inherited generation_config values,
            # including None-valued controls in Transformers 5. A fresh
            # GenerationConfig alone would inherit the model's penalties.
            neutral = {
                "top_h": None, "min_p": None, "typical_p": 1.0,
                "epsilon_cutoff": 0.0, "eta_cutoff": 0.0,
                "repetition_penalty": 1.0, "encoder_repetition_penalty": 1.0,
                "no_repeat_ngram_size": 0, "encoder_no_repeat_ngram_size": 0,
                "bad_words_ids": None, "min_length": 0, "min_new_tokens": 0,
                "forced_bos_token_id": None, "forced_eos_token_id": None,
                "suppress_tokens": None, "begin_suppress_tokens": None,
                "sequence_bias": None, "guidance_scale": None,
                "exponential_decay_length_penalty": None,
                "watermarking_config": None, "renormalize_logits": False,
                "remove_invalid_values": False, "num_beams": 1,
                "num_beam_groups": 1, "diversity_penalty": 0.0,
                "constraints": None, "force_words_ids": None,
                "penalty_alpha": None, "num_return_sequences": 1,
                "return_dict_in_generate": False,
            }
            options.update({key: value for key, value in neutral.items() if hasattr(self.model.generation_config, key)})
        output = self.model.generate(
            input_ids=ids,
            attention_mask=attention,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad,
            use_cache=True,
            **options,
        )
        eos = self.model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos or [])
        values = []
        for prompt, row in zip(prompts, output[:, width:].tolist(), strict=True):
            end = next((i + 1 for i, value in enumerate(row) if value in eos), len(row))
            completion = row[:end]
            values.append(
                PolicyCompletion(
                    self.tokenizer.decode(completion, skip_special_tokens=True),
                    prompt,
                    completion,
                    finish_reason="stop" if completion and completion[-1] in eos else "length",
                )
            )
        return values


def load_policy_model(model_name: str, *, adapter: str | None = None, trainable: bool = False, rank: int = 0, lora_rank: int = 8, quantization: str = "none", compute_profile: str = "legacy"):
    from .hf_compute import configure_compute_profile, validate_compute_profile, validate_adapter_precision
    validate_compute_profile(compute_profile, adapter=adapter)
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if quantization not in {"none", "nf4"}:
        raise ValueError("quantization must be 'none' or 'nf4'")
    load_kwargs = {}
    if quantization == "nf4":
        from .hf_quantization import nf4_load_kwargs

        load_kwargs = nf4_load_kwargs()
    else:
        config = AutoConfig.from_pretrained(model_name)
        # Transformers 4.57 uses unfused expert ModuleLists and forwards this
        # newer keyword to the model constructor, where it raises TypeError.
        # Keep grouped-mm on versions exposing its explicit configuration API.
        if getattr(config, "num_experts", 0) > 0 and hasattr(config, "_experts_implementation"):
            load_kwargs["experts_implementation"] = "grouped_mm"
    torch.cuda.set_device(rank)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map={"": rank},
        attn_implementation="sdpa",
        **load_kwargs,
    )
    report = None
    if quantization == "nf4":
        from .hf_quantization import quantization_report

        # Validate before PEFT preparation: a fused, unquantized expert would
        # otherwise be upcast to FP32 and could cause a large memory spike.
        quantization_report(model)
        from peft import prepare_model_for_kbit_training

        # Also preserve these base dtypes for inference so an adapter reload
        # evaluates the same precision used when it was trained.
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=trainable,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        report = quantization_report(model)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
        # A compatible adapter can warm-start a derived base. Persist the base
        # actually loaded here, rather than the adapter's historical source.
        for adapter_config in model.peft_config.values():
            adapter_config.base_model_name_or_path = model_name
    elif trainable:
        model = get_peft_model(
            model,
            LoraConfig(
                r=lora_rank,
                lora_alpha=2 * lora_rank,
                lora_dropout=0.0,
                target_modules=["q_proj", "v_proj", "o_proj"],
                task_type="CAUSAL_LM",
            ),
        )
    if trainable:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if report is not None:
        model.dcc_quantization_report = report
    configure_compute_profile(model, compute_profile)
    if adapter:
        validate_adapter_precision(model, adapter)
    return model, tokenizer


def completion_logprobs(model, prompt_ids: list[int], completion_ids: list[int]) -> torch.Tensor:
    """Only assistant response positions contribute to the loss.

    Compute the head only on response positions to avoid a prompt-length by
    vocabulary allocation. Log-softmax is float32; no sampling-warper scores
    are substituted for raw model probabilities.
    """
    ids = torch.tensor([prompt_ids + completion_ids], device=model.device)
    length = len(completion_ids)
    from .hf_compute import compute_context
    with compute_context(model):
        output = model(input_ids=ids, use_cache=False, logits_to_keep=length + 1)
    logits = output.logits[0, :-1].float()
    target = ids[0, -length:]
    return logits.gather(-1, target.unsqueeze(-1)).squeeze(-1) - logits.logsumexp(-1)
