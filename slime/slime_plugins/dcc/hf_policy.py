"""Local Transformers policy with batched generation and exact rollout tokens.

This is an optional, independent LoRA runtime. Importing the DCC harness does
not import Transformers or change the existing Slime/Megatron trainer.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict

import torch

from .episode import PolicyCompletion


def encode_messages(tokenizer, messages: list[dict[str, str]], max_prompt_tokens: int, *, allow_truncation: bool = True) -> list[int]:
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    if hasattr(tokens, "keys"):
        tokens = tokens["input_ids"]
    if len(tokens) > max_prompt_tokens:
        if not allow_truncation:
            raise ValueError("Judge input exceeds its context budget; refusing to truncate evidence")
        # Preserve the full system contract and original question. Keep the
        # highest-ranked initial excerpts, then recent observations on later
        # calls. Source text can be trimmed; the question must not disappear.
        header = tokenizer.apply_chat_template(messages[:2], tokenize=True, add_generation_prompt=False, enable_thinking=False)
        if hasattr(header, "keys"):
            header = header["input_ids"]
        prefix = len(header)
        if prefix + 128 > max_prompt_tokens:
            raise ValueError("System prompt and question exceed the configured prompt budget")
        remaining = max_prompt_tokens - prefix
        if len(messages) <= 3:
            tokens = tokens[: max_prompt_tokens - 32] + tokens[-32:]
        else:
            tokens = tokens[:prefix] + tokens[-remaining:]
    return tokens


class HFPolicy:
    def __init__(self, model, tokenizer, *, batch_size: int = 4, max_prompt_tokens: int = 4096, allow_truncation: bool = True):
        self.model, self.tokenizer = model, tokenizer
        self.batch_size, self.max_prompt_tokens = batch_size, max_prompt_tokens
        self.allow_truncation = allow_truncation
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
        prompts = [encode_messages(self.tokenizer, messages, self.max_prompt_tokens, allow_truncation=self.allow_truncation) for messages in messages_batch]
        width = max(map(len, prompts))
        pad = self.tokenizer.pad_token_id
        ids = torch.tensor([[pad] * (width - len(p)) + p for p in prompts], device=self.model.device)
        attention = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=ids.device)
        self.model.eval()
        output = self.model.generate(
            input_ids=ids,
            attention_mask=attention,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=0.95 if temperature > 0 else None,
            top_k=20 if temperature > 0 else None,
            pad_token_id=pad,
            use_cache=True,
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


def load_policy_model(model_name: str, *, adapter: str | None = None, trainable: bool = False, rank: int = 0, lora_rank: int = 8):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
        experts_implementation="grouped_mm",
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
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
    return model, tokenizer


def completion_logprobs(model, prompt_ids: list[int], completion_ids: list[int]) -> torch.Tensor:
    """Only assistant response positions contribute to the loss.

    Compute the head only on response positions to avoid a prompt-length by
    vocabulary allocation. Log-softmax is float32; no sampling-warper scores
    are substituted for raw model probabilities.
    """
    ids = torch.tensor([prompt_ids + completion_ids], device=model.device)
    length = len(completion_ids)
    output = model(input_ids=ids, use_cache=False, logits_to_keep=length + 1)
    logits = output.logits[0, :-1].float()
    target = ids[0, -length:]
    return logits.gather(-1, target.unsqueeze(-1)).squeeze(-1) - logits.logsumexp(-1)
