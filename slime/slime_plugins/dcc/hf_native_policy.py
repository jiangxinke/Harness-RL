"""Opt-in HF native thought sampling plus structural JSON.

This is a separate HF/NF4 on-policy runtime. It is not the existing vLLM native
evaluation protocol. By default only actual thoughts receive a policy gradient
and JSON is greedy. Joint CUAD mode samples JSON too and retains its actual
post-grammar behavior scores; it requires grammar-normalized replay.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import inspect
import json
import threading
import time
import uuid

import torch

from .hf_native_structured import compile_native_structured_schema, native_structured_schema_kind
from .web_whitespace import web_bounded_whitespace_configuration
from .episode import ModelProtocolFailure, PolicyCompletion
from .hf_policy import _chat_tokens, encode_messages
from .hf_compute import compute_context, model_compute_profile
from .native_hf_context import NativeHFPromptBudgetExceeded
from .native_behavior_logprobs import BEHAVIOR_LOGPROB_SOURCE, StreamingRawThoughtLogprobs
from .prompt_history import PromptHistoryConfig, PromptHistoryBudgetExceeded
from .web_hf_prompt_history import (native_web_history_configuration, native_web_history_fields,
    native_web_history_plan)
from .web_train_prompt_history import (FAILURE_FIELD, web_train_history_budgets,
    web_train_history_configuration, web_train_history_failure_fields,
    web_train_history_fields, web_train_history_plan)

PROTOCOL = "hf-native-two-stage-thought-only-policy-v1"
JOINT_PROTOCOL = "hf-native-two-stage-joint-masked-JSON-policy-v1"
WEB_THOUGHT_BUDGET_CLOSE_PROTOCOL = "hf-native-web-thought-budget-close-evaluation-v1"
OPEN_THINK, END_THINK = 151667, 151668
NATIVE_SUFFIX = "<|im_start|>assistant\n<think>\n"
CANCELLED_BEFORE_JSON = "request_cancelled_before_json"


class NativeHFRequestExpiredBeforeSubmission(RuntimeError):
    """A cancelled queued request has no model tokens or native trace."""


class NativeHFRequestControl:
    """Coordinate owner cancellation with queue submission and phase boundaries."""
    def __init__(self):
        self._expired = threading.Event()
        self._lock = threading.Lock()
        self.started = False

    def expire(self):
        with self._lock:
            self._expired.set()

    @property
    def expired(self):
        return self._expired.is_set()

    def try_start(self):
        # The worker calls this before to_thread, without yielding the eventloop.
        with self._lock:
            if self._expired.is_set():
                return False
            self.started = True
            return True

    def try_start_json(self):
        # Cancellation and this boundary run on different threads. Once this
        # check succeeds, that JSON phase belongs to the work that must drain.
        with self._lock:
            return not self._expired.is_set()


def web_thought_budget_close_configuration(thought_tokens):
    """Explicit identity of the optional Web evaluation decoding policy."""
    if type(thought_tokens) is not int or thought_tokens < 1:
        raise ValueError("Positive integer thought budget required")
    return dict(version="web-thought-budget-close-evaluation-v1", enabled=True,
        evaluation_only=True, thought_tokens=thought_tokens, forced_eos_token_id=END_THINK,
        forced_position=thought_tokens - 1, position_indexing="zero-based-generated-thought", extra_tokens=0)


def _explicit_generation_kwargs(config):
    """Keep native generation independent of checkpoint decoding defaults.

    Inspect the HF implementation: PEFT's wrapper accepts arbitrary kwargs.
    HF 5 removed use_model_defaults and fills None config entries from the
    checkpoint. Passing all public entries as kwargs overrides that fallback,
    including disabled optional processors whose explicit value is None.
    """
    from transformers.generation.utils import GenerationMixin

    if "use_model_defaults" in inspect.signature(GenerationMixin.generate).parameters:
        return {"generation_config": config, "use_model_defaults": False}
    return {key: value for key, value in config.to_dict().items()
            if not key.startswith("_") and key != "transformers_version"}


@dataclass(slots=True)
class NativeHFCompletion(PolicyCompletion):
    thought_token_count: int = 0
    phase_lengths: tuple[int, int] = (0, 0)
    native_status: str = "completed"
    thought_logprobs: list[float] | None = None
    thought_logprobs_source: str | None = None
    trace_id: str | None = None


@dataclass(slots=True)
class JointNativeHFCompletion(NativeHFCompletion):
    """Only the opt-in policy adds JSON sampling fields to serialized outputs."""
    json_logprobs: list[float] | None = None
    json_logprobs_source: str | None = None
    json_public_schema: dict | None = None
    json_stop_ids: list[int] | None = None


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
    # Preserve lightweight legacy decoder seams that bypass __init__. Real
    # joint runtimes always override both values explicitly in __init__.
    joint_json_policy = False
    web_thought_budget_close = False
    prompt_history_config = PromptHistoryConfig()
    web_train_prompt_history_config = PromptHistoryConfig()
    protocol = PROTOCOL

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
                 web_bounded_whitespace=False, capture_thought_logprobs=False,
                 mixed_schema_batching=False, joint_json_policy=False, web_thought_budget_close=False,
                 prompt_history_config=None, web_train_prompt_history_config=None):
        from transformers import GenerationConfig
        import xgrammar as xgr

        if sampling_profile not in {"raw", "greedy", "thinking"}:
            raise ValueError("Thought sampling must explicitly be raw, greedy or thinking")
        self.prompt_history_config = (PromptHistoryConfig() if prompt_history_config is None
                                      else prompt_history_config)
        if not isinstance(self.prompt_history_config, PromptHistoryConfig):
            raise ValueError("An explicit PromptHistoryConfig is required")
        if self.prompt_history_config.enabled and (capture_thought_logprobs or joint_json_policy or allow_truncation):
            raise ValueError("Native HF whole-turn history is Web evaluation-only; training capture, joint JSON and truncation are unsupported")
        self.web_train_prompt_history_config = (PromptHistoryConfig() if web_train_prompt_history_config is None
                                               else web_train_prompt_history_config)
        if not isinstance(self.web_train_prompt_history_config, PromptHistoryConfig):
            raise ValueError("An explicit Web TRAIN PromptHistoryConfig is required")
        if self.web_train_prompt_history_config.enabled and (not capture_thought_logprobs
                or sampling_profile != "raw" or joint_json_policy or allow_truncation
                or web_thought_budget_close or self.prompt_history_config.enabled):
            raise ValueError("Web TRAIN history requires raw behavior capture, thought-only loss and no truncation, forced close or evaluation history")
        if any(type(value) is not int or value <= 0
               for value in (thought_tokens, json_tokens, max_prompt_tokens, batch_size)):
            raise ValueError("Positive integer token and batch budgets required")
        if type(web_thought_budget_close) is not bool or (web_thought_budget_close and
                (capture_thought_logprobs or joint_json_policy)):
            raise ValueError("Web thought budget closing is evaluation-only; behavior capture and joint JSON are unsupported")
        if type(capture_thought_logprobs) is not bool or (capture_thought_logprobs and sampling_profile != "raw"):
            raise ValueError("Actual behavior scores require explicit raw thought sampling")
        if type(mixed_schema_batching) is not bool:
            raise ValueError("Mixed-schema batching must be an explicit boolean")
        if type(joint_json_policy) is not bool or (joint_json_policy and
                (sampling_profile != "raw" or not capture_thought_logprobs or web_bounded_whitespace)):
            raise ValueError("Joint CUAD JSON policy requires captured raw thoughts and unchanged CUAD grammar")
        self.joint_json_policy = joint_json_policy
        self.web_thought_budget_close = web_thought_budget_close
        self.protocol = JOINT_PROTOCOL if joint_json_policy else PROTOCOL
        self.capture_thought_logprobs = capture_thought_logprobs
        self.mixed_schema_batching = mixed_schema_batching
        self._last_thought_logprobs = None
        self._last_json_logprobs = None
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
        self.provenance = {"protocol": self.protocol, "thought_sampling_profile": sampling_profile,
            "thought_tokens": thought_tokens, "json_tokens": json_tokens,
            "thought_logprobs_source": BEHAVIOR_LOGPROB_SOURCE if capture_thought_logprobs else None,
            "json_sampling": "greedy-neutral-xgrammar-structural; public-schema-postvalidation",
            "policy_gradient": "real-stage-one-IDs-only; deterministic-JSON-mask-zero",
            "failure_policy": "retain-actual-thought-IDs-with-reward-zero; no-forced-close-or-retry",
            "backend": "hf", "supported_benchmark": "cuad-and-widesearch",
            "mixed_schema_batching": mixed_schema_batching,
            "web_bounded_whitespace": web_bounded_whitespace_configuration(web_bounded_whitespace),
            "web_schema_policy": "existing-public-schema-and-review-decoder; original-schema-postvalidation",
            "compute_profile": model_compute_profile(model),
            "separate_from_vllm_native_v3": True}
        if joint_json_policy:
            from .native_joint_grpo import JSON_BEHAVIOR_SOURCE
            self.provenance.update(json_sampling="multinomial-T1-top_p1-top_k0-xgrammar",
                json_logprobs_source=JSON_BEHAVIOR_SOURCE,
                policy_gradient="sampled-raw-thought-and-grammar-normalized-JSON",
                failure_policy="retain-actual-sampled-thought-and-JSON-IDs-with-reward-zero; no-forced-close-or-retry",
                supported_benchmark="cuad")
        if web_thought_budget_close:
            self.protocol = WEB_THOUGHT_BUDGET_CLOSE_PROTOCOL
            self.provenance.update(protocol=self.protocol,
                web_thought_budget_close=web_thought_budget_close_configuration(thought_tokens),
                policy_gradient="disabled-evaluation-only",
                failure_policy="retain-original-outcomes; last-thought-position-singleton-close; no-appended-token-or-retry",
                supported_benchmark="widesearch")
        if self.prompt_history_config.enabled:
            self.provenance.update(native_web_prompt_history=native_web_history_configuration(self.prompt_history_config),
                policy_gradient="disabled-evaluation-only", supported_benchmark="widesearch")
        if self.web_train_prompt_history_config.enabled:
            self.provenance["native_web_train_prompt_history"] = web_train_history_configuration(
                self.web_train_prompt_history_config)

    async def __call__(self, *args, **kwargs):
        raise ValueError("HF native policy requires generate_structured and an explicit public schema")

    def new_request_control(self):
        return NativeHFRequestControl()

    async def generate_structured(self, messages, *, json_schema, max_new_tokens, temperature, _task_id=None,
                                  _request_control=None):
        if _request_control is not None and not isinstance(_request_control, NativeHFRequestControl):
            raise ValueError("An explicit native request control is required")
        kind = native_structured_schema_kind(json_schema)
        if self.joint_json_policy and kind != "cuad":
            raise ValueError("Joint JSON policy currently supports CUAD only")
        if max_new_tokens != self.thought_tokens + self.json_tokens:
            raise ValueError("Runner must reserve exactly the two explicit phase budgets")
        if self.sampling_profile == "raw" and temperature != 1.0:
            raise ValueError("Raw on-policy thought sampling requires runner temperature=1")
        if self.worker is None:
            self.worker = asyncio.create_task(self._work())
        future = asyncio.get_running_loop().create_future()
        await self.queue.put((messages, json_schema, _task_id, future, _request_control))
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
                    key = None if self.mixed_schema_batching else json.dumps(request[1], sort_keys=True)
                    groups[key].append(request)
            for group in groups.values():
                # An earlier schema group may run past a sibling deadline.
                # Do not start GPU work for requests already cancelled while
                # waiting; running work still follows the existing drain path.
                active = []
                for request in group:
                    if request[3].done():
                        continue
                    control = request[4]
                    if control is not None and not control.try_start():
                        request[3].set_exception(NativeHFRequestExpiredBeforeSubmission(
                            "Native request expired before model submission"))
                    else:
                        active.append(request)
                group = active
                if not group:
                    continue
                try:
                    schemas = [item[1] for item in group] if self.mixed_schema_batching else group[0][1]
                    controls = [item[4] for item in group]
                    values = await asyncio.to_thread(self._generate_batch,
                        [item[0] for item in group], schemas, [item[2] for item in group],
                        **({"request_controls": controls} if any(c is not None for c in controls) else {}))
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
        sampled_json = phase == "json" and self.joint_json_policy
        stop_ids = list(self.eos_ids) + ([END_THINK] if phase == "thought" else [])
        config = self.generation_config(do_sample=raw or thinking or sampled_json,
            temperature=0.6 if thinking else 1.0, top_p=0.95 if thinking else 1.0,
            top_k=20 if thinking else 0, eos_token_id=stop_ids, pad_token_id=pad,
            max_new_tokens=self.thought_tokens if phase == "thought" else self.json_tokens,
            num_beams=1, num_return_sequences=1, min_length=0, early_stopping=False,
            repetition_penalty=1.0, encoder_repetition_penalty=1.0, length_penalty=1.0,
            no_repeat_ngram_size=0, encoder_no_repeat_ngram_size=0,
            typical_p=1.0, epsilon_cutoff=0.0, eta_cutoff=0.0,
            renormalize_logits=False, remove_invalid_values=False, token_healing=False,
            output_attentions=False, output_hidden_states=False, output_scores=False,
            disable_compile=False,
            use_cache=True, return_dict_in_generate=False)
        if phase == "thought" and self.web_thought_budget_close:
            # HF masks the last in-budget sampling position to this singleton.
            # generate selects and returns the real token; nothing is appended.
            config.forced_eos_token_id = END_THINK
        tracker = StreamingRawThoughtLogprobs() if raw and self.capture_thought_logprobs else None
        processors = [LogitsProcessor(grammar)] if grammar is not None else []
        if sampled_json:
            from .native_joint_grpo import StreamingMaskedJSONLogprobs
            if grammar is None:
                raise ValueError("Sampled JSON requires its exact compiled public schema")
            tracker = StreamingMaskedJSONLogprobs()
            processors.append(tracker)
        if tracker is not None:
            if processors and not sampled_json:
                raise ValueError("Raw behavior tracking cannot include grammar processors")
            if not sampled_json:
                processors.append(tracker)
        with compute_context(self.model):
            output = self.model.generate(input_ids=ids, attention_mask=attention,
                **_explicit_generation_kwargs(config),
                logits_processor=LogitsProcessorList(processors))
        rows = []
        for row in output[:, width:].tolist():
            end = next((i + 1 for i, value in enumerate(row) if value in stop_ids), len(row))
            rows.append(row[:end])
        if phase == "thought":
            self._last_thought_logprobs = tracker.finish(rows) if tracker is not None else None
        elif sampled_json:
            self._last_json_logprobs = tracker.finish(rows)
        return rows

    @torch.inference_mode()
    def _generate_batch(self, messages_batch, json_schema, task_ids=None, *, request_controls=None):
        if request_controls is not None and (len(request_controls) != len(messages_batch)
                or any(c is not None and not isinstance(c, NativeHFRequestControl) for c in request_controls)):
            raise ValueError("Every native request requires its own cancellation control")
        schemas = self._batch_schemas(json_schema, len(messages_batch))
        prompts, selected, results = [], [], [None] * len(messages_batch)
        history_metadata = []
        for i, messages in enumerate(messages_batch):
            if self.prompt_history_config.enabled:
                try:
                    plan = native_web_history_plan(self.tokenizer, messages, config=self.prompt_history_config,
                        max_prompt_tokens=self.max_prompt_tokens, max_model_len=self.max_model_len,
                        thought_tokens=self.thought_tokens, json_tokens=self.json_tokens)
                except PromptHistoryBudgetExceeded as failure:
                    failure.native_web_prompt_history_failure = native_web_history_fields(
                        messages, failure.diagnostics, self.prompt_history_config)
                    failure.native_web_prompt_history_failure["original_prompt_ids"] = list(failure.original_prompt_ids)
                    failure.native_web_prompt_history_failure["generation_submitted"] = False
                    results[i] = failure
                    continue
                prompts.append(plan.prompt_ids)
                selected.append(i)
                history_metadata.append(native_web_history_fields(messages, plan.report, self.prompt_history_config))
                continue
            history_failure = None
            if (self.web_train_prompt_history_config.enabled
                    and native_structured_schema_kind(schemas[i]) in {"web", "web-review"}):
                try:
                    plan = web_train_history_plan(self.tokenizer, messages,
                        config=self.web_train_prompt_history_config, **web_train_history_budgets(self))
                except PromptHistoryBudgetExceeded as failure:
                    # Keep the original full-request exception and its exact old
                    # preflight trace. Additional failed-plan proof is owned
                    # separately and recomputed before the group may be scored.
                    history_failure = web_train_history_failure_fields(
                        messages, failure, self.web_train_prompt_history_config)
                else:
                    prompts.append(plan.prompt_ids)
                    selected.append(i)
                    history_metadata.append(web_train_history_fields(
                        messages, plan.report, self.web_train_prompt_history_config))
                    continue
            full = _chat_tokens(self.tokenizer, messages)
            if len(full) > self.max_prompt_tokens and not self.allow_truncation:
                failure = NativeHFPromptBudgetExceeded(full, limit=self.max_prompt_tokens,
                    task_id=task_ids[i] if task_ids is not None else None,
                    thought_tokens=self.thought_tokens, json_tokens=self.json_tokens,
                    sampling_profile=self.sampling_profile,
                    behavior_source=BEHAVIOR_LOGPROB_SOURCE if self.capture_thought_logprobs else None)
                if history_failure is not None:
                    setattr(failure, FAILURE_FIELD, history_failure)
                # This additional diagnostic belongs only to raw TRAIN capture.
                # Evaluation sinks retain their completion-only trace counts.
                if self.capture_thought_logprobs and self.trace_sink is not None:
                    self.trace_sink(dict(failure.preflight))
                results[i] = failure
                continue
            if history_failure is not None:
                raise ValueError("Failed Web TRAIN history must preserve an overflowing original full request")
            prompt = encode_messages(self.tokenizer, messages, self.max_prompt_tokens,
                                     allow_truncation=self.allow_truncation)
            if prompt[-len(self.suffix_ids):] != self.suffix_ids:
                raise ValueError("Real encoded prompts must end in the native open-thinking suffix")
            prompts.append(prompt)
            selected.append(i)
            history_metadata.append(None)
        if selected:
            selected_schemas = [schemas[i] for i in selected] if isinstance(json_schema, list) else json_schema
            values = self._generate_encoded_batch(prompts, selected_schemas,
                [task_ids[i] for i in selected] if task_ids is not None else None,
                **({"history_metadata": history_metadata} if self.prompt_history_config.enabled
                   or self.web_train_prompt_history_config.enabled else {}),
                **({"request_controls": [request_controls[i] for i in selected]}
                   if request_controls is not None else {}))
            for i, value in zip(selected, values, strict=True):
                results[i] = value
        return results

    def _batch_schemas(self, json_schema, count):
        if isinstance(json_schema, list):
            if not self.mixed_schema_batching or len(json_schema) != count:
                raise ValueError("Opt-in mixed-schema batch needs exactly one public schema per request")
            schemas = json_schema
        else:
            schemas = [json_schema] * count
        for schema in schemas:
            kind = native_structured_schema_kind(schema)
            if self.prompt_history_config.enabled and kind not in {"web", "web-review"}:
                raise ValueError("Native HF whole-turn history supports exact Web evaluation schemas only")
            if self.web_thought_budget_close and kind not in {"web", "web-review"}:
                raise ValueError("Web thought budget closing supports exact Web evaluation schemas only")
            if self.joint_json_policy and kind != "cuad":
                raise ValueError("Joint JSON policy currently supports CUAD only")
        return schemas

    def _generate_encoded_batch(self, prompts, json_schema, task_ids, *, history_metadata=None,
                                request_controls=None):
        from jsonschema import Draft202012Validator
        if history_metadata is not None and len(history_metadata) != len(prompts):
            raise ValueError("Every compacted native request requires its own history provenance")
        schemas = self._batch_schemas(json_schema, len(prompts))
        self.model.eval()
        thought_started = time.monotonic()
        thoughts = self._phase(prompts, phase="thought")
        thought_seconds = time.monotonic() - thought_started
        thought_logprobs = self._last_thought_logprobs
        valid = [i for i, tokens in enumerate(thoughts) if tokens and tokens[-1] == END_THINK
                 and tokens.count(END_THINK) == 1 and not set(tokens).intersection(self.eos_ids)]
        cancelled_before_json = set()
        finals = [[] for _ in prompts]
        json_logprobs = [[] for _ in prompts] if self.joint_json_policy else None
        json_seconds = None
        if valid:
            grammars = []
            for i in valid:
                key = json.dumps(schemas[i], sort_keys=True)
                if key not in self.grammars:
                    self.grammars[key] = compile_native_structured_schema(self.compiler, schemas[i],
                        web_bounded_whitespace=self.web_bounded_whitespace)
                grammars.append(self.grammars[key])
            if request_controls is not None:
                selected = []
                kept_grammars = []
                for i, compiled in zip(valid, grammars, strict=True):
                    control = request_controls[i]
                    if control is not None and not control.try_start_json():
                        cancelled_before_json.add(i)
                    else:
                        selected.append(i)
                        kept_grammars.append(compiled)
                valid, grammars = selected, kept_grammars
        if valid:
            json_started = time.monotonic()
            # xgrammar owns a separate matcher per row. Never union schemas:
            # each answer keeps only its own columns, read URLs and citations.
            grammar = grammars if isinstance(json_schema, list) else grammars[0]
            generated = self._phase([prompts[i] + thoughts[i] for i in valid], phase="json", grammar=grammar)
            json_seconds = time.monotonic() - json_started
            for i, tokens in zip(valid, generated, strict=True):
                finals[i] = tokens
            if self.joint_json_policy:
                for i, values in zip(valid, self._last_json_logprobs, strict=True):
                    json_logprobs[i] = values
        results = []
        for i, (prompt, thought, final) in enumerate(zip(prompts, thoughts, finals, strict=True)):
            reason, text = None, ""
            if i in cancelled_before_json:
                reason = CANCELLED_BEFORE_JSON
            elif i not in valid:
                reason = "thought_did_not_stop_on_real_close"
            elif not final or final[-1] not in self.eos_ids or set(final[:-1]).intersection(self.eos_ids):
                reason = "json_did_not_finish_with_real_eos"
            else:
                text = self.tokenizer.decode(final[:-1], skip_special_tokens=False)
                try:
                    Draft202012Validator(schemas[i]).validate(_strict_json(text))
                except Exception:
                    reason, text = "json_not_complete_or_schema_valid", ""
            status = reason or "completed"
            terminal_ids = self.eos_ids if final else self.eos_ids + [END_THINK]
            observed = final or thought
            extra = {}
            completion_type = NativeHFCompletion
            if self.joint_json_policy:
                from .native_joint_grpo import JSON_BEHAVIOR_SOURCE
                completion_type = JointNativeHFCompletion
                extra = dict(json_logprobs=json_logprobs[i],
                    json_logprobs_source=JSON_BEHAVIOR_SOURCE if final else None,
                    json_public_schema=deepcopy(schemas[i]), json_stop_ids=list(self.eos_ids))
            completion = completion_type(text=text, prompt_ids=list(prompt), completion_ids=thought + final,
                finish_reason="stop" if observed and observed[-1] in terminal_ids else "length",
                thought_token_count=len(thought), phase_lengths=(len(thought), len(final)), native_status=status,
                thought_logprobs=thought_logprobs[i] if thought_logprobs is not None else None,
                thought_logprobs_source=BEHAVIOR_LOGPROB_SOURCE if thought_logprobs is not None else None, **extra)
            trace_id = uuid.uuid4().hex
            completion.trace_id = trace_id
            if self.trace_sink is not None:
                trace = {"protocol": self.protocol, "trace_id": trace_id,
                    "task_id": task_ids[i] if task_ids else None, "status": status,
                    "prompt_ids": list(prompt), "thought_ids": thought, "json_ids": final,
                    "thought_token_count": len(thought), "final_text": text,
                    "thought_logprobs": completion.thought_logprobs,
                    "thought_logprobs_source": completion.thought_logprobs_source,
                    "batch_phase_elapsed_seconds": {"thought": thought_seconds,
                        "json": json_seconds if i in valid else None},
                    "thought_sampling_profile": self.sampling_profile,
                    "json_public_schema": schemas[i]}
                if self.web_thought_budget_close:
                    trace.update(web_thought_budget_close=web_thought_budget_close_configuration(self.thought_tokens),
                        forced_thought_positions=([self.thought_tokens - 1]
                            if len(thought) == self.thought_tokens and thought[-1] == END_THINK else []))
                if history_metadata is not None and history_metadata[i] is not None:
                    trace.update(deepcopy(history_metadata[i]))
                if self.joint_json_policy:
                    trace.update(json_logprobs=completion.json_logprobs,
                        json_logprobs_source=completion.json_logprobs_source,
                        json_stop_ids=completion.json_stop_ids,
                        json_sampling_profile="multinomial-T1-top_p1-top_k0-xgrammar")
                self.trace_sink(trace)
            results.append(ModelProtocolFailure(reason, completion=completion, protocol=self.protocol, trace_id=trace_id)
                           if reason is not None else completion)
        return results


class _TaskPolicy:
    def __init__(self, parent, task_id, rank):
        self.parent, self.task_id, self.rank = parent, task_id, rank

    async def __call__(self, *args, **kwargs):
        return await self.parent(*args, **kwargs)

    def new_request_control(self):
        return self.parent.new_request_control()

    async def generate_structured(self, *args, **kwargs):
        return await self.parent.generate_structured(*args, _task_id=self.task_id, **kwargs)
