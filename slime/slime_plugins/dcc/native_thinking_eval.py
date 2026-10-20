"""Default-OFF native Thinking2507 CUAD/Web evaluation candidate.

Importing this module starts no engine. The original Web schemas, tools,
evidence guards, and scorers retain responsibility for actions and quality.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import re
from pathlib import Path
import uuid

from .episode import ModelProtocolFailure, PolicyCompletion
from .cuad_structured import (cuad_action_schema, cuad_final_schema,
    cuad_structured_configuration, cuad_structured_options, is_cuad_structured_schema,
    preflight_cuad_structured)
from .eval_sampling import EvaluationSampling
from .native_budget import NativeBudget, budget_from_configuration, approved_training_receipt, effective_sampling
from .hf_policy import PROMPT_ENCODING_VERSION, encode_messages
from .vllm_policy import AsyncVLLMPolicy
from .prompt_history import (PromptHistoryConfig, PromptHistoryBudgetExceeded,
                            fit_prompt_history, validate_prompt_history_records, prompt_history_json_equal)
from .native_phase_timing import NativePhaseTimer, configuration as phase_timing_configuration


BASE_SOURCE_COMMIT = "5bed6188feaaf6c79fac53eead30cfbad83b0beb"
MODEL_REVISION = "144afc2f379b542fdd4e85a1fcd5e1f79112d95d"
PROTOCOL = "native-thinking2507-real-close-two-stage-4096-4096-v2"
OPEN_THINK = 151667
END_THINK = 151668
THOUGHT_TOKENS = 4096
JSON_TOKENS = 4096
TOTAL_TOKENS = THOUGHT_TOKENS + JSON_TOKENS
NATIVE_SUFFIX = "<|im_start|>assistant\n<think>\n"


class NativeThinkingFailure(ModelProtocolFailure):
    """A nonretryable model/protocol failure with the observed trajectory.

    ``completion.text`` is empty. The trace and completion IDs retain observed
    tokens; callers must preserve the failed attempt instead of resubmitting it.
    EpisodeRunner persists this as a terminal model_protocol_failure.
    """

    def __init__(self, reason, *, completion, trace):
        super().__init__(reason, completion=completion, protocol=trace["protocol"], trace_id=trace["trace_id"])
        self.reason = reason
        self.completion = completion
        self.trace = trace


def _strict_json(text):
    def nonfinite(value):
        raise ValueError("Non-finite JSON constant: " + value)

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON object key")
            result[key] = value
        return result

    return json.loads(text, parse_constant=nonfinite, object_pairs_hook=unique_object)


def _stage2_eos_valid(phase, eos_token_ids, primary_eos_token_id):
    tokens = phase["token_ids"]
    if (phase["finish_reason"] != "stop" or not tokens
            or tokens[-1] not in eos_token_ids
            or any(token in eos_token_ids for token in tokens[:-1])):
        return False
    stop_reason = phase["stop_reason"]
    if stop_reason is None:
        return tokens[-1] == primary_eos_token_id
    return type(stop_reason) is int and stop_reason == tokens[-1]


class NativeThinkingPolicy:
    """Compose the frozen async runtime; preserve a single real token path.

    Stage one has no structured-output parameters. Stage two receives the exact
    original prompt plus all stage-one IDs, including the real closing token.
    Only stage-two JSON text is exposed to an action parser. Defaults retain
    v2 greedy decoding. Explicit v3 profiles change only thought budget/sampling;
    JSON remains greedy with neutral penalties and the original schema.
    """

    def __init__(self, base, *, source_receipt, trace_sink, approved_derivative_id=None,
                 budget=None, approved_training_source=None, approved_training_protocol=None,
                 prompt_history_config=None, phase_timing=False, approved_dpo=None):
        timing_configuration = phase_timing_configuration(phase_timing)
        self.phase_timing = phase_timing
        self.prompt_history_config = (PromptHistoryConfig() if prompt_history_config is None
                                      else prompt_history_config)
        if not isinstance(self.prompt_history_config, PromptHistoryConfig):
            raise TypeError("An explicit PromptHistoryConfig is required")
        if self.prompt_history_config.enabled and base.allow_truncation:
            raise ValueError("Whole-turn history requires allow_truncation=False; source slicing stays disabled")
        self.budget = NativeBudget() if budget is None else budget
        if not isinstance(self.budget, NativeBudget):
            raise TypeError("An explicit NativeBudget is required")
        if approved_dpo is not None and (approved_training_source is not None or approved_training_protocol is not None):
            raise ValueError("DPO approval and accepted-SFT source/protocol approval are mutually exclusive")
        training_receipt = approved_training_receipt(source_receipt,
            approved_training_source, approved_training_protocol)
        if not isinstance(base, AsyncVLLMPolicy):
            raise TypeError("An existing native AsyncVLLMPolicy is required")
        if base.lora_request is not None and not self.budget.legacy and training_receipt is None and approved_dpo is None:
            raise ValueError("v3 native LoRA requires explicit approval of its actual training source/protocol")
        adapter_binding = None
        if base.lora_request is not None:
            from .native_lora import validate_native_adapter
            actual = base.provenance.get("adapter")
            if not isinstance(actual, dict) or not actual.get("path"):
                raise ValueError("Native LoRA requires the actual loader adapter metadata")
            metadata, adapter_binding = validate_native_adapter(actual["path"],
                source_receipt=validate_source_receipt(source_receipt),
                approved_derivative_id=approved_derivative_id,
                model_config=base.model.vllm_config.model_config.hf_config.to_dict(),
                approved_training_source_receipt=training_receipt,
                approved_training_protocol=approved_training_protocol,
                evaluation_protocol=self.budget.protocol,
                **({"approved_dpo": approved_dpo} if approved_dpo is not None else {}))
            if metadata != actual or Path(base.lora_request.lora_path).absolute() != Path(metadata["path"]):
                raise ValueError("Native LoRA request differs from the actual checked adapter")
        elif approved_derivative_id is not None or training_receipt is not None or approved_dpo is not None:
            raise ValueError("Native LoRA derivative flag requires an actual adapter")
        if base.evaluation_sampling != EvaluationSampling():
            raise ValueError("This new protocol requires greedy/neutral EvaluationSampling()")
        if not base.web_structured_final or not base.web_structured_actions:
            raise ValueError("The original structured-output path must be enabled")
        if base.max_prompt_tokens + self.budget.total_tokens > base.max_model_len:
            raise ValueError("Reserve max_prompt_tokens plus the complete native thought+JSON budget")
        config = base.model.vllm_config.structured_outputs_config
        if (config.backend != "xgrammar" or not config.disable_fallback
                or config.reasoning_parser or config.reasoning_parser_plugin
                or config.enable_in_reasoning):
            raise ValueError("Use xgrammar without fallback, reasoning parser, or in-reasoning grammar")
        tokenizer = base.tokenizer
        if (tokenizer.encode("<think>", add_special_tokens=False) != [OPEN_THINK]
                or tokenizer.encode("</think>", add_special_tokens=False) != [END_THINK]):
            raise ValueError("Native Thinking2507 opening/closing IDs do not match")
        self.suffix_ids = list(tokenizer.encode(NATIVE_SUFFIX, add_special_tokens=False))
        if not self.suffix_ids or END_THINK in base.eos_token_ids or not base.eos_token_ids:
            raise ValueError("Native suffix and non-thinking EOS IDs are required")
        if (type(tokenizer.eos_token_id) is not int
                or tokenizer.eos_token_id not in base.eos_token_ids):
            raise ValueError("The tokenizer primary EOS must be one configured integer EOS")
        self.source_receipt = validate_source_receipt(source_receipt)
        if trace_sink is None:
            raise ValueError("Native evaluation requires a durable trace sink")
        self.base, self.trace_sink = base, trace_sink
        self.traces = []
        self.trace_index = {}
        self.provenance = copy.deepcopy(base.provenance)
        if adapter_binding is not None:
            self.provenance["native_lora_binding"] = adapter_binding
        self.provenance.update(
            source_commit=self.source_receipt["source_commit"],
            inference_protocol=self.budget.protocol,
            prompt_encoding="HF encode_messages; native Thinking2507 template; effective thinking=True",
            prompt_encoding_version=PROMPT_ENCODING_VERSION,
            native_thinking=native_configuration(self.source_receipt, budget=self.budget),
        )
        if self.prompt_history_config.enabled:
            self.provenance["prompt_history"] = self.prompt_history_config.to_dict()
        if timing_configuration is not None:
            self.provenance["native_phase_timing"] = timing_configuration

        if not self.budget.legacy:
            self.provenance["sampling"] = "Native thought profile=" + self.budget.thought_sampling_profile + "; JSON=greedy-neutral; see native_thinking"

    async def __call__(self, messages, *, max_new_tokens, temperature):
        raise ValueError("Native evaluation requires generate_structured with an explicit original schema")

    def _completion(self, trace, *, text="", finish_reason=None):
        tokens = [token for phase in trace["phases"] for token in phase.get("token_ids", [])]
        if finish_reason is None:
            finish_reason = (trace["phases"][-1].get("finish_reason") if trace["phases"] else None)
        return PolicyCompletion(text, list(trace["original_prompt_ids"]), tokens,
                                finish_reason=finish_reason or "unknown")

    def _fail(self, trace, reason):
        trace.update(status="failed", failure_reason=reason)
        raise NativeThinkingFailure(reason, completion=self._completion(trace), trace=trace)

    async def _phase(self, trace, name, prompt, params):
        from vllm.sampling_params import RequestOutputKind

        request_id = uuid.uuid4().hex
        phase = {
            "name": name, "request_id": request_id,
            "prompt_ids": list(prompt), "max_tokens": params.max_tokens,
            "n": params.n, "seed": params.seed,
            "stop_token_ids": list(params.stop_token_ids),
            "ignore_eos": params.ignore_eos,
            "include_stop_str_in_output": params.include_stop_str_in_output,
            "skip_special_tokens": params.skip_special_tokens,
            "output_kind": RequestOutputKind(params.output_kind).name,
            "structured_outputs": params.structured_outputs is not None,
            "sampling": (self.budget.thought_sampling if name == "thought" else self.base.evaluation_sampling).to_dict(),
        }
        if not self.budget.legacy:
            phase["effective_sampling"] = {key: getattr(params, key)
                for key in EvaluationSampling().to_dict()}
        trace["phases"].append(phase)
        self.base._active_requests.add(request_id)
        completed, response = False, None
        timing = NativePhaseTimer(phase) if self.phase_timing else None
        try:
            async for item in self.base.model.generate(
                    {"prompt_token_ids": list(prompt)}, params, request_id,
                    lora_request=self.base.lora_request):
                response = item
                if timing is not None:
                    timing.observe(item)
                # Retain every actually observed response snapshot even when
                # a later engine/cancellation error prevents normal return.
                phase["response_prompt_ids"] = list(item.prompt_token_ids)
                phase["response_finished"] = item.finished
                phase["raw_outputs"] = [{
                    "token_ids": list(output.token_ids),
                    "engine_output_text": output.text,
                    "finish_reason": output.finish_reason,
                    "stop_reason": output.stop_reason,
                } for output in item.outputs]
                if len(item.outputs) == 1:
                    phase.update(phase["raw_outputs"][0])
            if response is None:
                self._fail(trace, name + "_no_response")
            if len(response.outputs) != 1:
                self._fail(trace, name + "_output_count_mismatch")
            tokens = phase["token_ids"]
            if phase["response_prompt_ids"] != prompt:
                self._fail(trace, name + "_prompt_id_mismatch")
            if not response.finished:
                self._fail(trace, name + "_nonterminal_response")
            if (not tokens or len(tokens) > params.max_tokens
                    or any(type(token) is not int or token < 0 for token in tokens)):
                self._fail(trace, name + "_invalid_token_ids")
            phase["decoded_text_raw"] = self.base.tokenizer.decode(tokens, skip_special_tokens=False)
            if phase["finish_reason"] not in {"stop", "length"}:
                self._fail(trace, name + "_unknown_finish_reason")
            completed = True
            if timing is not None:
                timing.mark("completion")
            return phase
        except BaseException as error:
            if timing is not None:
                timing.mark("exception")
            phase["exception_type"] = type(error).__name__
            raise
        finally:
            try:
                if not completed:
                    if timing is not None:
                        timing.mark("abort_start")
                    try:
                        await self.base.model.abort(request_id)
                    except BaseException as error:
                        if timing is not None:
                            timing.mark("abort_exception")
                            timing.record["abort_exception_type"] = type(error).__name__
                        raise
                    else:
                        if timing is not None:
                            timing.mark("abort_completion")
                    finally:
                        if timing is not None:
                            timing.mark("abort_end")
                    phase["aborted"] = True
            finally:
                if timing is not None:
                    timing.end()
                self.base._active_requests.discard(request_id)

    async def generate_structured(self, messages, *, json_schema, max_new_tokens, temperature, trace_metadata):
        from jsonschema import Draft202012Validator
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        if self.base._closed:
            raise RuntimeError("The native AsyncLLM has been closed")
        if type(max_new_tokens) is not int or max_new_tokens != self.budget.total_tokens:
            raise ValueError("Declare max_new_tokens equal to the actual thought budget plus 4096 JSON")
        options = self.base._sampling_options(temperature)
        # A Web review flag may be enabled in a mixed-benchmark run. CUAD has
        # its own original 40-word final contract; never route it to the Web
        # 240-character review recognizer or Web-only whitespace grammar.
        structured_options = (cuad_structured_options(json_schema) if is_cuad_structured_schema(json_schema)
                              else self.base._structured_options(json_schema))
        if not structured_options or json_schema is None:
            raise ValueError("An explicit original JSON schema is required")
        loop = asyncio.get_running_loop()
        if self.base._owner_loop is None:
            self.base._owner_loop = loop
        if self.base._owner_loop is not loop:
            raise RuntimeError("Native requests must run on the AsyncLLM owning event loop")
        if set(trace_metadata) != {"trace_id", "task_id", "rank", "turn_index"}:
            raise ValueError("Native trace requires exact task/turn ownership metadata")
        history_plan = None
        if self.prompt_history_config.enabled:
            try:
                history_plan = fit_prompt_history(
                    self.base.tokenizer, messages, max_prompt_tokens=self.base.max_prompt_tokens,
                    max_new_tokens=self.budget.total_tokens, max_model_len=self.base.max_model_len,
                    config=self.prompt_history_config)
            except PromptHistoryBudgetExceeded as error:
                # This is a pre-generation failure, not model output. Retain
                # the full actual caller request and measured IDs without
                # inventing a thought, a submitted request, or a completion.
                trace = {"protocol": self.budget.protocol,
                    "source_commit": self.source_receipt["source_commit"],
                    **copy.deepcopy(trace_metadata), "status": "preflight_failure",
                    "failure_reason": error.reason, "exception_type": type(error).__name__,
                    "generation_submitted": False, "phases": [],
                    "original_prompt_ids": list(error.original_prompt_ids),
                    "original_prompt_ids_scope": "measured full caller request; never submitted",
                    "json_schema": copy.deepcopy(json_schema),
                    "declared_max_new_tokens": self.budget.total_tokens,
                    "thought_max_tokens": self.budget.thought_tokens, "json_max_tokens": JSON_TOKENS,
                    "native_thinking": native_configuration(self.source_receipt, budget=self.budget),
                    "prompt_history": copy.deepcopy(error.diagnostics),
                    "prompt_history_original_messages": copy.deepcopy(messages),
                    "completion_ids": [], "observed_completion_tokens": 0,
                    "public_text_contains_thought": False}
                self.traces.append(trace)
                self.trace_index[trace["trace_id"]] = trace
                if not self.budget.legacy and "native_lora_binding" in self.provenance:
                    trace["native_lora_binding"] = copy.deepcopy(self.provenance["native_lora_binding"])
                self.trace_sink(copy.deepcopy(trace))
                raise
            prompt = history_plan.prompt_ids
        else:
            prompt = encode_messages(self.base.tokenizer, messages, self.base.max_prompt_tokens,
                                     allow_truncation=self.base.allow_truncation)
        if prompt[-len(self.suffix_ids):] != self.suffix_ids:
            raise ValueError("The original encoded prompt must end in the native open-thinking suffix")
        if len(prompt) + self.budget.total_tokens > self.base.max_model_len:
            raise ValueError("The real prompt plus complete native protocol exceeds max_model_len")
        trace = {
            "protocol": self.budget.protocol, "source_commit": self.source_receipt["source_commit"],
            "original_prompt_ids": list(prompt), "json_schema": copy.deepcopy(json_schema),
            "declared_max_new_tokens": self.budget.total_tokens,
            "thought_max_tokens": self.budget.thought_tokens, "json_max_tokens": JSON_TOKENS,
            "raw_ids_scope": "All terminal response IDs; on interruption only actually observed responses are known",
            "phases": [], "status": "running", "failure_reason": None,
        }
        trace.update(copy.deepcopy(trace_metadata))
        if history_plan is not None:
            trace["prompt_history"] = copy.deepcopy(history_plan.report)
            trace["prompt_history_original_messages"] = copy.deepcopy(messages)
        trace["native_thinking"] = native_configuration(self.source_receipt, budget=self.budget)
        if not self.budget.legacy and "native_lora_binding" in self.provenance:
            trace["native_lora_binding"] = copy.deepcopy(self.provenance["native_lora_binding"])
        self.traces.append(trace)
        self.trace_index[trace["trace_id"]] = trace
        try:
            thought_params = SamplingParams(
                n=1, **self.budget.thought_sampling.to_dict(), max_tokens=self.budget.thought_tokens, seed=self.base.seed,
                ignore_eos=False, stop_token_ids=sorted(set(self.base.eos_token_ids + [END_THINK])),
                skip_special_tokens=False, include_stop_str_in_output=True,
                output_kind=RequestOutputKind.FINAL_ONLY,
            )
            thought = await self._phase(trace, "thought", prompt, thought_params)
            ids = thought["token_ids"]
            if (thought["finish_reason"] != "stop"
                    or type(thought["stop_reason"]) is not int
                    or thought["stop_reason"] != END_THINK
                    or ids[-1] != END_THINK or ids.count(END_THINK) != 1
                    or any(token in self.base.eos_token_ids for token in ids)):
                self._fail(trace, "thought_did_not_stop_on_real_close")
            # This is the only continuation input: no decode/encode/template pass.
            json_prompt = prompt + ids
            json_params = SamplingParams(
                n=1, **options, max_tokens=JSON_TOKENS, seed=self.base.seed,
                ignore_eos=False, stop_token_ids=list(self.base.eos_token_ids),
                skip_special_tokens=False, include_stop_str_in_output=True,
                output_kind=RequestOutputKind.FINAL_ONLY, **structured_options,
            )
            final = await self._phase(trace, "json", json_prompt, json_params)
            if not _stage2_eos_valid(final, self.base.eos_token_ids, self.base.tokenizer.eos_token_id):
                self._fail(trace, "json_did_not_finish_with_real_eos")
            # Exclude only the already verified terminal EOS token. Generic
            # skip_special_tokens=True could erase a legitimate generated
            # special-token spelling inside a JSON string.
            text = self.base.tokenizer.decode(final["token_ids"][:-1], skip_special_tokens=False)
            try:
                value = _strict_json(text)
                Draft202012Validator(json_schema).validate(value)
            except (ValueError, TypeError, RecursionError) as error:
                final["json_validation_error_type"] = type(error).__name__
                self._fail(trace, "json_not_complete_or_schema_valid")
            except Exception as error:
                # jsonschema.ValidationError is not a ValueError. Preserve its
                # class only; never repair/extract/retry a rejected final value.
                final["json_validation_error_type"] = type(error).__name__
                self._fail(trace, "json_not_complete_or_schema_valid")
            trace.update(status="completed", final_text=text,
                         json_complete=True, json_schema_valid=True)
            return self._completion(trace, text=text, finish_reason=final["finish_reason"])
        except BaseException as error:
            if trace["status"] == "running":
                trace.update(status="interrupted" if isinstance(error, asyncio.CancelledError)
                             else "runtime_error", exception_type=type(error).__name__)
            raise
        finally:
            trace["completion_ids"] = self._completion(trace).completion_ids
            trace["observed_completion_tokens"] = len(trace["completion_ids"])
            trace["public_text_contains_thought"] = False
            if self.trace_sink is not None:
                self.trace_sink(copy.deepcopy(trace))

    def bind_task(self, task_id, rank):
        return NativeEpisodePolicy(self, task_id, rank)

    async def close(self):
        try:
            await self.base.close()
        finally:
            close = getattr(self.trace_sink, "close", None)
            if close is not None:
                close()


def native_configuration(source_receipt, *, budget=None):
    budget = NativeBudget() if budget is None else budget
    result = {'enabled': True, 'encode_messages_requested_enable_thinking': False, 'native_template_ignores_enable_thinking': True, 'effective_enable_thinking': True, 'native_suffix': NATIVE_SUFFIX, 'open_think_id': OPEN_THINK, 'end_think_id': END_THINK, 'thought_max_tokens': THOUGHT_TOKENS, 'json_max_tokens': JSON_TOKENS, 'total_max_new_tokens': TOTAL_TOKENS, 'stage1_structured_outputs': False, 'reasoning_parser': '', 'stage2_original_schema': False, 'stage2_original_public_schema_postvalidation': True, 'stage2_backend_schema_policy': {'cuad': 'structural-schema-without-string-patterns', 'widesearch': 'existing-backend-policy'}, 'stage2_allow_eos': True, 'stage2_requires_terminal_eos': True, 'stage2_stop_reason_matches_terminal_eos': True, 'stage2_none_stop_reason_only_for_primary_eos': True, 'reject_duplicate_json_keys': True, 'stage2_validate_complete_json_and_schema': True, 'public_json_decode': 'Remove terminal EOS ID only; preserve all other generated text', 'same_real_prompt_and_completion_ids': True, 'phase2_retemplate': False, 'forced_close': False, 'thought_json_extraction': False, 'retry': False, 'inference_only': True, 'comparable_to_old_4096_protocol': False}
    result.update(protocol=PROTOCOL, source_receipt=validate_source_receipt(source_receipt),
                  benchmarks=["widesearch", "cuad"],
                  cuad_structured=cuad_structured_configuration(True), sidecar="native_thinking_traces_rank{rank}.jsonl",
                  fresh_generation_only=True, sampling_profile="greedy-neutral")
    if not budget.legacy:
        result.update(protocol=budget.protocol, thought_max_tokens=budget.thought_tokens,
            total_max_new_tokens=budget.total_tokens,
            thought_sampling_profile=budget.thought_sampling_profile,
            thought_sampling=budget.thought_sampling.to_dict(), json_sampling=EvaluationSampling().to_dict(),
            thought_effective_sampling=effective_sampling(budget.thought_sampling),
            json_effective_sampling=effective_sampling(EvaluationSampling()),
            sampling_profile="phase-specific-thinking" if budget.thought_sampling_profile == "thinking" else "greedy-neutral",
            phase_sampling_seed="same explicit seed per physical phase request; no shared RNG history")
    return result


def validate_source_receipt(value):
    if (not isinstance(value, dict)
            or set(value) != {"version", "source_commit", "base_source_commit", "protocol", "frozen"}
            or value["version"] != "dcc-native-inference-source-v1"
            or not isinstance(value["source_commit"], str)
            or re.fullmatch(r"[0-9a-f]{40}", value["source_commit"]) is None
            or value["source_commit"] == BASE_SOURCE_COMMIT
            or value["base_source_commit"] != BASE_SOURCE_COMMIT
            or value["protocol"] != PROTOCOL or value["frozen"] is not True):
        raise ValueError("A new frozen native source receipt is required; the combined code is not 5bed")
    return copy.deepcopy(value)


def native_options(args, tasks, sampling, source_root):
    enabled = getattr(args, "native_thinking", False)
    budget = NativeBudget(getattr(args, "native_thought_tokens", 4096),
                          getattr(args, "native_thought_sampling", "greedy"))
    training_source = getattr(args, "native_lora_training_source", None)
    training_protocol = getattr(args, "native_lora_training_protocol", None)
    dpo_path = getattr(args, "native_lora_dpo_approval", None)
    if dpo_path is not None and (training_source is not None or training_protocol is not None):
        raise ValueError("DPO approval and accepted-SFT source/protocol approval are mutually exclusive")
    receipt_path = getattr(args, "native_source_receipt", None)
    if type(enabled) is not bool:
        raise ValueError("native_thinking must be a boolean")
    if not enabled:
        if receipt_path is not None:
            raise ValueError("--native-source-receipt requires --native-thinking")
        if (getattr(args, "native_lora_derivative_id", None) is not None or not budget.legacy
                or training_source is not None or training_protocol is not None or dpo_path is not None):
            raise ValueError("Native LoRA flags require --native-thinking")
        return None
    if (args.inference_backend != "vllm" or not args.vllm_async
            or not args.web_structured_final or not args.web_structured_actions
            or args.max_new_tokens != budget.total_tokens
            or sampling != EvaluationSampling()):
        raise ValueError("Native evaluation requires async structured vLLM, full thought+4096 JSON budget, neutral greedy JSON")
    if not tasks or any(task.benchmark not in {"widesearch", "cuad"} for task in tasks):
        raise ValueError("Native evaluation supports CUAD and WideSearch tasks only")
    if any(task.benchmark == "cuad" for task in tasks):
        preflight_cuad_structured()
    if getattr(args, "retry_infrastructure", False):
        raise ValueError("Initial native evaluation is fresh-only and permits no retries")
    if receipt_path is None:
        raise ValueError("--native-thinking requires --native-source-receipt")
    path = Path(receipt_path).resolve()
    if path != (Path(source_root) / ".dcc-native-source.json").resolve():
        raise ValueError("The source receipt must belong to the actual frozen source root")
    receipt = validate_source_receipt(_strict_json(path.read_text()))
    approved = getattr(args, "native_lora_derivative_id", None)
    training_receipt = approved_training_receipt(receipt, training_source, training_protocol)
    from .native_dpo_adapter import approval_from_arguments
    approved_dpo = approval_from_arguments(args)
    if args.adapter is not None:
        if not budget.legacy and training_receipt is None and approved_dpo is None:
            raise ValueError("v3 native LoRA requires explicit approval of its actual training source/protocol")
        from transformers import AutoConfig
        from .native_lora import validate_native_adapter
        validate_native_adapter(args.adapter, source_receipt=receipt, approved_derivative_id=approved,
            model_config=AutoConfig.from_pretrained(args.model, local_files_only=True, trust_remote_code=False).to_dict(),
            approved_training_source_receipt=training_receipt,
            approved_training_protocol=training_protocol, evaluation_protocol=budget.protocol,
            **({"approved_dpo": approved_dpo} if approved_dpo is not None else {}))
    elif approved is not None or training_receipt is not None or approved_dpo is not None:
        raise ValueError("Native LoRA derivative flag requires an actual adapter")
    return native_configuration(receipt, budget=budget)


class NativeTraceJournal:
    """One fresh durable output journal per rank, never a home/tmp cache."""

    def __init__(self, path):
        self.stream = Path(path).open("x", encoding="utf-8")

    def __call__(self, trace):
        self.stream.write(json.dumps(trace, ensure_ascii=False, allow_nan=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self):
        if not self.stream.closed:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()


class NativeEpisodePolicy:
    """Task-local ownership, sharing only the native engine with other tasks."""

    def __init__(self, parent, task_id, rank):
        self.parent, self.task_id, self.rank = parent, task_id, rank
        self.references = []
        self.turn_index = 0

    async def __call__(self, *args, **kwargs):
        raise ValueError("Native evaluation requires an explicit CUAD/Web structured-action path")

    async def generate_structured(self, messages, **kwargs):
        trace_id = uuid.uuid4().hex
        metadata = {"trace_id": trace_id, "task_id": self.task_id,
                    "rank": self.rank, "turn_index": self.turn_index}
        self.turn_index += 1
        try:
            return await self.parent.generate_structured(messages, trace_metadata=metadata, **kwargs)
        finally:
            trace = self.parent.trace_index.get(trace_id)
            if trace is not None:
                self.references.append({**metadata, "status": trace["status"],
                    "failure_reason": trace["failure_reason"],
                    "physical_prompt_tokens": sum(len(phase["prompt_ids"]) for phase in trace["phases"]),
                    "completion_tokens_by_stage": {phase["name"]: len(phase.get("token_ids", []))
                                                   for phase in trace["phases"]}})


def validate_native_sidecars(output, saved, manifest, runtime, owners, tasks, *, tokenizer=None):
    """Validate native trajectory integrity; failed model attempts keep coverage.

    The original score/guards still decide answer quality. No failure is dropped
    or repaired here. This guard is entered only for the new explicit protocol.
    """
    from jsonschema import Draft202012Validator
    from .episode import _WebEvidence, _answer_columns
    from .web_structured import web_action_schema, web_final_schema
    from .web_review import web_review_schema

    config = manifest.get("native_thinking")
    history_configuration = validate_prompt_history_records((manifest, runtime or {}))
    runtime_config = (runtime or {}).get("native_thinking")
    if config is None and runtime_config is None:
        return None
    if not isinstance(config, dict) or config != runtime_config:
        raise ValueError("Native manifest/runtime configuration differs")
    budget = budget_from_configuration(config)
    if config != native_configuration(config.get("source_receipt"), budget=budget):
        raise ValueError("Native protocol is not the complete supported declaration")
    binding = (runtime or {}).get("native_lora_binding")
    approval = manifest.get("native_lora_training_approval")
    training_receipt = (approved_training_receipt(config["source_receipt"],
        approval.get("source_commit"), approval.get("protocol")) if isinstance(approval, dict) else None)
    if approval is not None and (not isinstance(approval, dict) or set(approval) != {"source_commit", "protocol"} or training_receipt is None):
        raise ValueError("Invalid caller-approved actual training source/protocol")
    dpo_approval = manifest.get("native_lora_dpo_approval")
    if dpo_approval is not None:
        if approval is not None or manifest.get("adapter") is None or (runtime or {}).get("adapter") is None:
            raise ValueError("DPO sidecar approval requires an actual adapter and excludes accepted-SFT approval")
        from .native_dpo_adapter import validate_dpo_sidecar_binding
        validate_dpo_sidecar_binding(binding, dpo_approval,
            derivative_id=manifest.get("native_lora_derivative_id"),
            source_receipt=config["source_receipt"], evaluation_protocol=budget.protocol)
    elif manifest.get("adapter") is not None:
        if (not isinstance(binding, dict) or not manifest.get("native_lora_derivative_id")
                or binding.get("derivative_id") != manifest["native_lora_derivative_id"]
                or binding.get("training_source_receipt") != (training_receipt or config["source_receipt"])
                or binding.get("protocol") != (approval["protocol"] if approval else PROTOCOL)
                or binding.get("model_revision") != MODEL_REVISION
                or binding.get("checked_actual_attention_shapes") is not True
                or (runtime or {}).get("adapter") is None):
            raise ValueError("Native LoRA sidecars lack the approved actual adapter binding")
        if (not budget.legacy and training_receipt is None) or (training_receipt is not None and (
                binding.get("evaluation_source_receipt") != config["source_receipt"]
                or binding.get("evaluation_protocol") != budget.protocol
                or binding.get("caller_approved_training") != approval)):
            raise ValueError("Native LoRA sidecars lack separate actual training/evaluation source binding")
    elif binding is not None or manifest.get("native_lora_derivative_id") is not None or approval is not None:
        raise ValueError("Base native sidecars cannot claim a LoRA adapter binding")
    if (manifest.get("max_new_tokens") != budget.total_tokens
            or manifest.get("evaluation_sampling") != EvaluationSampling().to_dict()
            or not manifest.get("inference_parameters", {}).get("async_engine")
            or (runtime or {}).get("source_commit") != config["source_receipt"]["source_commit"]):
        raise ValueError("Native source/budget/sampling declaration differs")
    if not budget.legacy:
        context = manifest.get("inference_parameters", {}).get("max_model_len")
        prompt_budget = manifest.get("max_prompt_tokens")
        if (type(context) is not int or type(prompt_budget) is not int or prompt_budget < 1
                or context < prompt_budget + budget.total_tokens
                or (runtime or {}).get("max_model_len") != context
                or (runtime or {}).get("max_prompt_tokens") != prompt_budget):
            raise ValueError("Native v3 runtime/manifest context cannot hold the full prompt+completion")
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(manifest["model"], local_files_only=True)
    task_map = {task.id: task for task in tasks}
    if any(task.benchmark not in {"widesearch", "cuad"} for task in tasks):
        raise ValueError("Native traces support CUAD and WideSearch tasks only")
    eos = (runtime or {}).get("eos_token_ids", [])
    if (not isinstance(eos, list) or not eos or any(type(token) is not int or token < 0 for token in eos)
            or eos != sorted(set(eos)) or END_THINK in eos or tokenizer.eos_token_id not in eos):
        raise ValueError("Native runtime EOS declaration differs from the tokenizer")
    traces = {}
    for rank in range(manifest["world_size"]):
        path = Path(output) / f"native_thinking_traces_rank{rank}.jsonl"
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            trace = _strict_json(line)
            trace_id = trace.get("trace_id")
            if (not isinstance(trace_id, str) or trace_id in traces
                    or trace.get("rank") != rank
                    or owners.get(trace.get("task_id")) != rank
                    or trace.get("native_thinking") != config
                    or trace.get("source_commit") != config["source_receipt"]["source_commit"]
                    or trace.get("protocol") != budget.protocol
                    or (not budget.legacy and trace.get("native_lora_binding") != binding)):
                raise ValueError("Native trace ownership/source/protocol mismatch")
            traces[trace_id] = trace
    used = set()
    for task_id, record in saved.items():
        refs = record.get("native_thinking_traces")
        steps = record.get("steps", [])
        if not isinstance(refs, list) or not isinstance(steps, list):
            raise ValueError("Native predictions require trace references and steps")
        # A cancelled FINAL_ONLY request has no PolicyCompletion. Its one extra
        # terminal trace still belongs to this failed episode and denominator.
        trailing_interruption = len(refs) == len(steps) + 1 and record.get("status") == "infrastructure_error"
        trailing_preflight = (history_configuration.enabled and len(refs) == len(steps) + 1
                              and record.get("status") == "context_budget_failure")
        if len(refs) != len(steps) and not trailing_interruption and not trailing_preflight:
            raise ValueError("Every native model call must have one durable trace reference")
        evidence = _WebEvidence()
        applied = set()
        for index, ref in enumerate(refs):
            trace_id = ref.get("trace_id")
            if trace_id not in traces or trace_id in used:
                raise ValueError("Native trace missing or reused")
            used.add(trace_id)
            trace = traces[trace_id]
            if (trace.get("task_id") != task_id or trace.get("turn_index") != index
                    or trace.get("rank") != owners[task_id]):
                raise ValueError("Native task/turn trace ownership differs")
            for event_index, event in enumerate(record.get("tool_events", [])):
                if event_index not in applied and event.get("turn", -1) < index:
                    evidence.observe(event.get("tool"), event.get("observation", {}))
                    applied.add(event_index)
            if task_map[task_id].benchmark == "cuad":
                schema = cuad_final_schema() if index == manifest["max_steps"] - 1 else cuad_action_schema()
            else:
                if index == manifest["max_steps"] - 1:
                    schema = web_final_schema(_answer_columns(task_map[task_id]), evidence.fetched_sources)
                else:
                    schema = web_action_schema(_answer_columns(task_map[task_id]), evidence.fetched_sources,
                        successful_search=bool(evidence.searches), cached_rows=manifest.get("web_cached_rows", {}).get("enabled", False))
                if manifest.get("web_explicit_review", {}).get("enabled", False):
                    schema = web_review_schema(schema)
            if trace.get("json_schema") != schema:
                raise ValueError("Native schema differs from original Web task/evidence state")
            history_error = None
            if history_configuration.enabled:
                # Reconstruct each actual caller request from the unabridged
                # episode, including request-only guidance. Do not accept a
                # self-consistent invented history supplied only by a trace.
                episode_messages = record.get("messages")
                if not isinstance(episode_messages, list):
                    raise ValueError("Whole-turn native audit requires the complete episode history")
                assistant_positions = [j for j, message in enumerate(episode_messages)
                                       if message.get("role") == "assistant"]
                cutoff = assistant_positions[index] if index < len(assistant_positions) else len(episode_messages)
                original_messages = copy.deepcopy(episode_messages[:cutoff])
                final_turn = index == manifest["max_steps"] - 1
                if manifest.get("web_navigation_ledger", {}).get("enabled", False) and task_map[task_id].benchmark == "widesearch":
                    from .web_navigation import WebNavigationLedger
                    ledger = WebNavigationLedger()
                    ledger.observe_events([event for event in record.get("tool_events", [])
                                           if event.get("turn", -1) < index])
                    original_messages.insert(len(original_messages) - final_turn,
                                             {"role": "user", "content": ledger.render()})
                if manifest.get("web_research_checkpoints", {}).get("enabled", False) and task_map[task_id].benchmark == "widesearch":
                    from .web_research import web_research_checkpoint
                    checkpoint = web_research_checkpoint(_answer_columns(task_map[task_id]),
                                                         turn=index, max_steps=manifest["max_steps"])
                    original_messages.insert(len(original_messages) - final_turn,
                                             {"role": "user", "content": checkpoint})
                if trace.get("prompt_history_original_messages") != original_messages:
                    raise ValueError("Whole-turn native trace differs from the full episode request")
                try:
                    plan = fit_prompt_history(tokenizer, original_messages,
                        max_prompt_tokens=manifest["max_prompt_tokens"], max_new_tokens=budget.total_tokens,
                        max_model_len=manifest["inference_parameters"]["max_model_len"],
                        config=history_configuration)
                except PromptHistoryBudgetExceeded as error:
                    history_error = error
                    if (trace.get("status") != "preflight_failure"
                            or not prompt_history_json_equal(trace.get("prompt_history"), error.diagnostics)
                            or trace.get("original_prompt_ids") != error.original_prompt_ids):
                        raise ValueError("Native preflight diagnostics do not replay from the complete request")
                else:
                    if (not prompt_history_json_equal(trace.get("prompt_history"), plan.report)
                            or trace.get("original_prompt_ids") != plan.prompt_ids):
                        raise ValueError("Native whole-turn effective prompt or audit report differs")
            elif "prompt_history" in trace or "prompt_history_original_messages" in trace:
                raise ValueError("Native history packing lacks explicit manifest/runtime opt-in")
            if trace.get("status") == "preflight_failure":
                if (history_error is None or not trailing_preflight or index != len(refs) - 1
                        or trace.get("failure_reason") != history_error.reason
                        or trace.get("generation_submitted") is not False
                        or trace.get("original_prompt_ids_scope") != "measured full caller request; never submitted"
                        or trace.get("phases") != [] or trace.get("completion_ids") != []
                        or trace.get("observed_completion_tokens") != 0
                        or trace.get("public_text_contains_thought") is not False):
                    raise ValueError("Native context failure must retain a real zero-generation preflight")
                expected_ref = {"trace_id": trace_id, "task_id": task_id, "rank": owners[task_id],
                    "turn_index": index, "status": "preflight_failure", "failure_reason": history_error.reason,
                    "physical_prompt_tokens": 0, "completion_tokens_by_stage": {}}
                expected_failure = {"reason": history_error.reason, "diagnostics": history_error.diagnostics,
                                    "request_messages": original_messages, "generation_submitted": False}
                if (not prompt_history_json_equal(ref, expected_ref)
                        or not prompt_history_json_equal(record.get("metrics", {}).get("prompt_history_failure"), expected_failure)
                        or trace.get("declared_max_new_tokens") != budget.total_tokens
                        or trace.get("thought_max_tokens") != budget.thought_tokens
                        or trace.get("json_max_tokens") != JSON_TOKENS):
                    raise ValueError("Native preflight reference/summary/budgets differ")
                continue
            prompt = trace.get("original_prompt_ids")
            phases = trace.get("phases", [])
            if (not isinstance(prompt, list) or not prompt
                    or any(type(token) is not int or token < 0 for token in prompt)
                    or not isinstance(phases, list) or not 1 <= len(phases) <= 2
                    or (not budget.legacy and (len(prompt) > manifest["max_prompt_tokens"]
                        or len(prompt) + budget.total_tokens > manifest["inference_parameters"]["max_model_len"]))
                    or trace.get("declared_max_new_tokens") != budget.total_tokens
                    or trace.get("thought_max_tokens") != budget.thought_tokens
                    or trace.get("json_max_tokens") != JSON_TOKENS):
                raise ValueError("Native original IDs/phase count/budgets differ")
            suffix = tokenizer.encode(NATIVE_SUFFIX, add_special_tokens=False)
            if prompt[-len(suffix):] != suffix:
                raise ValueError("Native original prompt lacks the native thinking suffix")
            combined = [token for phase in phases for token in phase.get("token_ids", [])]
            if (trace.get("completion_ids") != combined
                    or trace.get("observed_completion_tokens") != len(combined)
                    or trace.get("public_text_contains_thought") is not False):
                raise ValueError("Native combined real IDs differ")
            expected_reference = {
                "trace_id": trace_id, "task_id": task_id, "rank": owners[task_id], "turn_index": index,
                "status": trace.get("status"), "failure_reason": trace.get("failure_reason"),
                "physical_prompt_tokens": sum(len(phase["prompt_ids"]) for phase in phases),
                "completion_tokens_by_stage": {phase["name"]: len(phase.get("token_ids", [])) for phase in phases},
            }
            if ref != expected_reference:
                raise ValueError("Native prediction trace reference differs from durable trace")
            interrupted = trace.get("status") in {"interrupted", "runtime_error"}
            observed_failure = None
            for phase_index, phase in enumerate(phases):
                name = "thought" if phase_index == 0 else "json"
                expected_prompt = prompt if phase_index == 0 else prompt + phases[0].get("token_ids", [])
                if (phase.get("name") != name or phase.get("prompt_ids") != expected_prompt
                        or phase.get("max_tokens") != (budget.thought_tokens if phase_index == 0 else JSON_TOKENS) or phase.get("n") != 1
                        or phase.get("stop_token_ids") != (sorted(set(eos + [END_THINK])) if phase_index == 0 else eos)
                        or phase.get("ignore_eos") is not False
                        or phase.get("include_stop_str_in_output") is not True
                        or phase.get("skip_special_tokens") is not False
                        or phase.get("output_kind") != "FINAL_ONLY"
                        or phase.get("structured_outputs") is not (phase_index == 1)
                        or phase.get("sampling") != (budget.thought_sampling if phase_index == 0 else EvaluationSampling()).to_dict()
                        or ("seed" in manifest and phase.get("seed") != manifest["seed"])):
                    raise ValueError("Native phase prompt/decoding configuration differs")
                if not budget.legacy and phase.get("effective_sampling") != effective_sampling(
                        budget.thought_sampling if phase_index == 0 else EvaluationSampling()):
                    raise ValueError("Native phase actual effective sampling differs")
                raw = phase.get("raw_outputs")
                reason = None
                if raw is None:
                    reason = name + "_no_response"
                    if any(key in phase for key in ("token_ids", "response_prompt_ids", "decoded_text_raw")):
                        raise ValueError("Native missing response contains invented IDs/text")
                elif not isinstance(raw, list) or len(raw) != 1:
                    reason = name + "_output_count_mismatch"
                    if "token_ids" in phase:
                        raise ValueError("Native multiple outputs have no unique completion IDs")
                else:
                    output = raw[0]
                    if any(phase.get(key) != output.get(key) for key in
                           ("token_ids", "engine_output_text", "finish_reason", "stop_reason")):
                        raise ValueError("Native raw engine output differs from copied fields")
                    tokens = output["token_ids"]
                    if phase.get("response_prompt_ids") != expected_prompt:
                        reason = name + "_prompt_id_mismatch"
                    elif phase.get("response_finished") is not True:
                        reason = name + "_nonterminal_response"
                    elif (not tokens or len(tokens) > (budget.thought_tokens if phase_index == 0 else JSON_TOKENS)
                          or any(type(token) is not int or token < 0 for token in tokens)):
                        reason = name + "_invalid_token_ids"
                    elif phase.get("finish_reason") not in {"stop", "length"}:
                        reason = name + "_unknown_finish_reason"
                    if tokens and all(type(token) is int and token >= 0 for token in tokens):
                        decoded = tokenizer.decode(tokens, skip_special_tokens=False)
                        if "decoded_text_raw" in phase and phase["decoded_text_raw"] != decoded:
                            raise ValueError("Native raw text does not decode from its actual IDs")
                        if not interrupted and reason is None and "decoded_text_raw" not in phase:
                            raise ValueError("Native terminal response lacks its raw decoding")
                if reason is None and phase_index == 0:
                    ids = phase["token_ids"]
                    if (phase["finish_reason"] != "stop" or type(phase["stop_reason"]) is not int
                            or phase["stop_reason"] != END_THINK or ids[-1] != END_THINK
                            or ids.count(END_THINK) != 1 or any(token in eos for token in ids)):
                        reason = "thought_did_not_stop_on_real_close"
                if reason is not None:
                    observed_failure = reason
                    if phase_index != len(phases) - 1:
                        raise ValueError("Native generation continued after a failed phase")
            expected_text = ""
            if not interrupted and observed_failure is None:
                if len(phases) != 2:
                    raise ValueError("Native terminal result lacks its JSON phase")
                if not _stage2_eos_valid(phases[1], eos, tokenizer.eos_token_id):
                    observed_failure = "json_did_not_finish_with_real_eos"
                else:
                    candidate = tokenizer.decode(phases[1]["token_ids"][:-1], skip_special_tokens=False)
                    try:
                        Draft202012Validator(schema).validate(_strict_json(candidate))
                    except Exception:
                        observed_failure = "json_not_complete_or_schema_valid"
                    else:
                        expected_text = candidate
            if trace.get("status") == "completed":
                if observed_failure is not None or interrupted or trace.get("failure_reason") is not None:
                    raise ValueError("Native completed status hides a failed trajectory")
                if (trace.get("final_text") != expected_text or trace.get("json_complete") is not True
                        or trace.get("json_schema_valid") is not True):
                    raise ValueError("Native final text/JSON validation differs")
            elif trace.get("status") == "failed":
                expected_text = ""
                if (index != len(refs) - 1 or record.get("status") != "model_protocol_failure"
                        or trace.get("failure_reason") != observed_failure or observed_failure is None):
                    raise ValueError("A real native failure must remain a terminal model_protocol_failure")
                failure = record.get("metrics", {}).get("model_protocol_failure")
                if failure != {"protocol": budget.protocol, "trace_id": trace_id, "reason": observed_failure}:
                    raise ValueError("Native failure summary differs from durable trace")
            elif interrupted:
                if (not trailing_interruption or index != len(refs) - 1
                        or record.get("status") != "infrastructure_error" or not trace.get("exception_type")):
                    raise ValueError("Native interruption must remain a terminal infrastructure failure")
                continue  # No fabricated PolicyCompletion for an unfinished FINAL_ONLY response.
            else:
                raise ValueError("Unknown native trace status")
            step = steps[index]
            if (step.get("text") != expected_text
                    or ("prompt_ids" in step and step["prompt_ids"] != prompt)
                    or ("completion_ids" in step and step["completion_ids"] != combined)):
                raise ValueError("Prediction tokens/text differ from native trace")
        if record.get("status") == "model_protocol_failure" and (not refs or traces[refs[-1]["trace_id"]]["status"] != "failed"):
            raise ValueError("Model protocol failure lacks its failed trace")
        if record.get("status") == "context_budget_failure" and (
                not history_configuration.enabled or not refs
                or traces[refs[-1]["trace_id"]]["status"] != "preflight_failure"):
            raise ValueError("Context budget failure lacks its declared terminal preflight")
    if used != traces.keys():
        raise ValueError("Native sidecars contain unreferenced attempts; no implicit retry/overwrite")
    result = {"traces": len(traces), "model_protocol_failures": sum(record.get("status") == "model_protocol_failure" for record in saved.values()),
              "infrastructure_errors": sum(record.get("status") == "infrastructure_error" for record in saved.values())}
    if history_configuration.enabled:
        result["context_budget_failures"] = sum(record.get("status") == "context_budget_failure" for record in saved.values())
    return result
