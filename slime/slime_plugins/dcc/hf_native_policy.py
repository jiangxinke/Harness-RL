"""Opt-in HF native thought sampling plus deterministic structural JSON.

This is a separate HF/NF4 on-policy runtime. It is not the existing vLLM native
evaluation protocol. Only real stage-one token IDs receive a policy gradient;
stage-two greedy grammar decisions carry zero policy-loss weight.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
import json
import uuid

import torch

from .hf_native_structured import compile_native_structured_schema, native_structured_schema_kind
from .web_whitespace import web_bounded_whitespace_configuration
from .episode import ModelProtocolFailure, PolicyCompletion
from .hf_policy import _chat_tokens, encode_messages
from .hf_compute import compute_context, model_compute_profile
from .prompt_history import PromptHistoryBudgetExceeded

PROTOCOL = "hf-native-two-stage-thought-only-policy-v1"
OPEN_THINK, END_THINK = 151667, 151668
NATIVE_SUFFIX = "<|im_start|>assistant\n<think>\n"


@dataclass(slots=True)
class NativeHFCompletion(PolicyCompletion):
    thought_token_count: int = 0
    phase_lengths: tuple[int, int] = (0, 0)
    native_status: str = "completed"


def _strict_json(text):
    def reject_constant(value):
        raise ValueError("Non-finite JSON constant: " + value)

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON object key")
            result[key] = value
        return result

    return json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_object)


class NativeHFPolicy:
    @staticmethod
    def validate_tokenizer(tokenizer):
        if tokenizer.encode('<think>', add_special_tokens=False) != [OPEN_THINK] or \
                tokenizer.encode('</think>', add_special_tokens=False) != [END_THINK]:
            raise ValueError("Native Thinking opening/closing token IDs differ")
        suffix = tokenizer.encode(NATIVE_SUFFIX, add_special_tokens=False)
        if not suffix or suffix.count(OPEN_THINK) != 1 or END_THINK in suffix:
            raise ValueError("A real native open-thinking suffix is required")
        return {"native_suffix_ids": list(suffix), "open_think_id": OPEN_THINK, "end_think_id": END_THINK}

    def __init__(self, model, tokenizer, *, thought_tokens=8192, json_tokens=4096,
                 max_prompt_tokens=16384, max_model_len=None, batch_size=4,
                 sampling_profile="raw", allow_truncation=False, trace_sink=None,
                 web_bounded_whitespace=False):
        from transformers import GenerationConfig
        import xgrammar as xgr

        if sampling_profile not in {"raw", "greedy", "thinking"}:
            raise ValueError("Thought sampling must explicitly be raw, greedy or thinking")
        if any(type(value) is not int or value <= 0
               for value in (thought_tokens, json_tokens, max_prompt_tokens, batch_size)):
            raise ValueError("Positive integer token and batch budgets required")
        self.model, self.tokenizer = model, tokenizer
        self.thought_tokens, self.json_tokens = thought_tokens, json_tokens
        self.max_prompt_tokens, self.batch_size = max_prompt_tokens, batch_size
        self.sampling_profile, self.allow_truncation = sampling_profile, allow_truncation
        self.trace_sink = trace_sink
        self.web_bounded_whitespace = web_bounded_whitespace_configuration(web_bounded_whitespace)["enabled"]
        self.max_model_len = max_model_len or model.config.max_position_embeddings
        if self.max_prompt_tokens + thought_tokens + json_tokens > self.max_model_len:
            raise ValueError("Reserve the full prompt plus both phase budgets")
        self.suffix_ids = self.validate_tokenizer(tokenizer)["native_suffix_ids"]
        eos = model.generation_config.eos_token_id
        self.eos_ids = [eos] if type(eos) is int else list(eos or [])
        if not self.eos_ids or END_THINK in self.eos_ids or tokenizer.eos_token_id not in self.eos_ids:
            raise ValueError("Non-thinking EOS IDs including tokenizer primary EOS required")
        if tokenizer.pad_token_id is None:
            raise ValueError("An explicit padding token is required")
        self.generation_config = GenerationConfig
        self.compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(
            tokenizer, vocab_size=model.config.vocab_size, stop_token_ids=self.eos_ids), max_threads=1)
        self.grammars = {}
        self.queue = asyncio.Queue()
        self.worker = None
        self.provenance = {"protocol": PROTOCOL, "thought_sampling_profile": sampling_profile,
            "thought_tokens": thought_tokens, "json_tokens": json_tokens,
            "json_sampling": "greedy-neutral-xgrammar-structural; public-schema-postvalidation",
            "policy_gradient": "real-stage-one-IDs-only; deterministic-JSON-mask-zero",
            "failure_policy": "retain-actual-thought-IDs-with-reward-zero; no-forced-close-or-retry",
            "backend": "hf", "supported_benchmark": "cuad-and-widesearch",
            "web_bounded_whitespace": web_bounded_whitespace_configuration(web_bounded_whitespace),
            "web_schema_policy": "existing-public-schema-and-review-decoder; original-schema-postvalidation",
            "compute_profile": model_compute_profile(model),
            "separate_from_vllm_native_v3": True}

    async def __call__(self, *args, **kwargs):
        raise ValueError("HF native policy requires generate_structured and an explicit public schema")

    async def generate_structured(self, messages, *, json_schema, max_new_tokens, temperature, _task_id=None):
        native_structured_schema_kind(json_schema)
        if max_new_tokens != self.thought_tokens + self.json_tokens:
            raise ValueError("Runner must reserve exactly the two explicit phase budgets")
        if self.sampling_profile == "raw" and temperature != 1.0:
            raise ValueError("Raw on-policy thought sampling requires runner temperature=1")
        if self.worker is None:
            self.worker = asyncio.create_task(self._work())
        future = asyncio.get_running_loop().create_future()
        await self.queue.put((messages, json_schema, _task_id, future))
        return await future

    def bind_task(self, task_id, rank=0):
        return _TaskPolicy(self, task_id, rank)

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
            await asyncio.sleep(0.01)
            requests, stop = [first], False
            while len(requests) < self.batch_size and not self.queue.empty():
                item = self.queue.get_nowait()
                if item is None:
                    stop = True
                    break
                requests.append(item)
            groups = defaultdict(list)
            for request in requests:
                if not request[3].done():
                    groups[json.dumps(request[1], sort_keys=True)].append(request)
            for group in groups.values():
                try:
                    values = await asyncio.to_thread(self._generate_batch,
                        [item[0] for item in group], group[0][1], [item[2] for item in group])
                    for request, value in zip(group, values, strict=True):
                        if not request[3].done():
                            if isinstance(value, Exception):
                                request[3].set_exception(value)
                            else:
                                request[3].set_result(value)
                except Exception as error:
                    for request in group:
                        if not request[3].done():
                            request[3].set_exception(error)
            if stop:
                return

    def _phase(self, prompts, *, phase, grammar=None):
        from xgrammar.contrib.hf import LogitsProcessor
        from transformers import LogitsProcessorList

        width = max(map(len, prompts))
        pad = self.tokenizer.pad_token_id
        ids = torch.tensor([[pad] * (width - len(p)) + p for p in prompts], device=self.model.device)
        attention = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=ids.device)
        raw = phase == "thought" and self.sampling_profile == "raw"
        thinking = phase == "thought" and self.sampling_profile == "thinking"
        stop_ids = list(self.eos_ids) + ([END_THINK] if phase == "thought" else [])
        config = self.generation_config(do_sample=raw or thinking,
            temperature=0.6 if thinking else 1.0, top_p=0.95 if thinking else 1.0,
            top_k=20 if thinking else 0, eos_token_id=stop_ids, pad_token_id=pad,
            max_new_tokens=self.thought_tokens if phase == "thought" else self.json_tokens,
            use_cache=True, return_dict_in_generate=False)
        # Explicit False prevents Transformers 4.57 from restoring the model's
        # penalties/top-p/forced tokens into otherwise neutral default fields.
        with compute_context(self.model):
            output = self.model.generate(input_ids=ids, attention_mask=attention,
                generation_config=config, use_model_defaults=False,
                logits_processor=LogitsProcessorList([LogitsProcessor(grammar)] if grammar is not None else []))
        rows = []
        for row in output[:, width:].tolist():
            end = next((i + 1 for i, value in enumerate(row) if value in stop_ids), len(row))
            rows.append(row[:end])
        return rows

    @torch.inference_mode()
    def _generate_batch(self, messages_batch, json_schema, task_ids=None):
        native_structured_schema_kind(json_schema)
        prompts, selected, results = [], [], [None] * len(messages_batch)
        for i, messages in enumerate(messages_batch):
            full = _chat_tokens(self.tokenizer, messages)
            if len(full) > self.max_prompt_tokens and not self.allow_truncation:
                results[i] = PromptHistoryBudgetExceeded("full_context_exceeds_native_HF_limit",
                    diagnostics={"original_prompt_tokens": len(full), "minimum_prompt_tokens": len(full),
                                 "effective_prompt_limit": self.max_prompt_tokens}, original_prompt_ids=full)
                continue
            prompt = encode_messages(self.tokenizer, messages, self.max_prompt_tokens,
                                     allow_truncation=self.allow_truncation)
            if prompt[-len(self.suffix_ids):] != self.suffix_ids:
                raise ValueError("Real encoded prompts must end in the native open-thinking suffix")
            prompts.append(prompt)
            selected.append(i)
        if selected:
            values = self._generate_encoded_batch(prompts, json_schema,
                [task_ids[i] for i in selected] if task_ids is not None else None)
            for i, value in zip(selected, values, strict=True):
                results[i] = value
        return results

    def _generate_encoded_batch(self, prompts, json_schema, task_ids):
        from jsonschema import Draft202012Validator
        self.model.eval()
        thoughts = self._phase(prompts, phase="thought")
        valid = [i for i, tokens in enumerate(thoughts) if tokens and tokens[-1] == END_THINK
                 and tokens.count(END_THINK) == 1 and not set(tokens).intersection(self.eos_ids)]
        finals = [[] for _ in prompts]
        if valid:
            key = json.dumps(json_schema, sort_keys=True)
            if key not in self.grammars:
                self.grammars[key] = compile_native_structured_schema(self.compiler, json_schema,
                    web_bounded_whitespace=self.web_bounded_whitespace)
            generated = self._phase([prompts[i] + thoughts[i] for i in valid], phase="json", grammar=self.grammars[key])
            for i, tokens in zip(valid, generated, strict=True):
                finals[i] = tokens
        results = []
        for i, (prompt, thought, final) in enumerate(zip(prompts, thoughts, finals, strict=True)):
            reason, text = None, ""
            if i not in valid:
                reason = "thought_did_not_stop_on_real_close"
            elif not final or final[-1] not in self.eos_ids or set(final[:-1]).intersection(self.eos_ids):
                reason = "json_did_not_finish_with_real_eos"
            else:
                text = self.tokenizer.decode(final[:-1], skip_special_tokens=False)
                try:
                    Draft202012Validator(json_schema).validate(_strict_json(text))
                except Exception:
                    reason, text = "json_not_complete_or_schema_valid", ""
            status = reason or "completed"
            terminal_ids = self.eos_ids if final else self.eos_ids + [END_THINK]
            observed = final or thought
            completion = NativeHFCompletion(text=text, prompt_ids=list(prompt), completion_ids=thought + final,
                finish_reason="stop" if observed and observed[-1] in terminal_ids else "length",
                thought_token_count=len(thought), phase_lengths=(len(thought), len(final)), native_status=status)
            trace_id = uuid.uuid4().hex
            if self.trace_sink is not None:
                self.trace_sink({"protocol": PROTOCOL, "trace_id": trace_id,
                    "task_id": task_ids[i] if task_ids else None, "status": status,
                    "prompt_ids": list(prompt), "thought_ids": thought, "json_ids": final,
                    "thought_token_count": len(thought), "final_text": text,
                    "thought_sampling_profile": self.sampling_profile,
                    "json_public_schema": json_schema})
            results.append(ModelProtocolFailure(reason, completion=completion, protocol=PROTOCOL, trace_id=trace_id)
                           if reason is not None else completion)
        return results


class _TaskPolicy:
    def __init__(self, parent, task_id, rank):
        self.parent, self.task_id, self.rank = parent, task_id, rank

    async def __call__(self, *args, **kwargs):
        return await self.parent(*args, **kwargs)

    async def generate_structured(self, *args, **kwargs):
        return await self.parent.generate_structured(*args, _task_id=self.task_id, **kwargs)
