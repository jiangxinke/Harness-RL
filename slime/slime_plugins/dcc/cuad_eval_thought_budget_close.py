"""Independent, opt-in CUAD EVAL thought closing and actual-request proof.

This policy has no behavior capture or trainable segments. Closing selects the
real token in the last existing thought slot; it never appends a token.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import asdict
import json

FIELD = "cuad_thought_budget_close"
POSITIONS = "forced_thought_positions"
PROTOCOL = "hf-native-CUAD-thought-budget-close-evaluation-v1"
VERSION = "dcc-CUAD-budget-close-native-ownership-v1"
END_THINK = 151668
JSON_SAMPLING = "greedy-neutral-xgrammar-structural; public-schema-postvalidation"
FORBIDDEN_FIELDS = ("web_thought_budget_close", "web_train_thought_budget_close",
    "cuad_train_thought_budget_close", "cuad_forced_thought_positions",
    "native_web_prompt_history", "native_web_train_prompt_history",
    "web_thought_logprobs_source", "web_protocol")


def configuration(thought_tokens):
    if type(thought_tokens) is not int or thought_tokens < 1:
        raise ValueError("Positive integer CUAD EVAL thought budget required")
    return dict(version="cuad-thought-budget-close-evaluation-v1", enabled=True,
        evaluation_only=True, thought_tokens=thought_tokens, forced_eos_token_id=END_THINK,
        forced_position=thought_tokens - 1, position_indexing="zero-based-generated-thought",
        extra_tokens=0, json_sampling=JSON_SAMPLING, behavior_capture=False)


def validate_configuration(profile, *, thought_tokens):
    expected = configuration(thought_tokens)
    if (type(profile) is not dict or profile.keys() != expected.keys()
            or any(type(profile[key]) is not type(value) or profile[key] != value
                   for key, value in expected.items())):
        raise ValueError("Exact CUAD EVAL thought budget close configuration required")
    return deepcopy(expected)


def _tokens(value):
    return type(value) is list and all(type(token) is int and token >= 0 for token in value)


def validate_fields(profile, positions, thought, *, thought_tokens):
    validate_configuration(profile, thought_tokens=thought_tokens)
    if (not _tokens(thought)
            or not 0 < len(thought) <= thought_tokens
            or type(positions) is not list or any(type(p) is not int for p in positions)
            or positions != ([thought_tokens - 1] if len(thought) == thought_tokens else [])
            or positions and (thought[-1] != END_THINK or thought.count(END_THINK) != 1)):
        raise ValueError("Actual CUAD EVAL thought budget closing policy/position differs")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


class OwnedCuadEvaluationPolicy:
    """Drain and own cancelled native calls, without turning them into steps."""
    def __init__(self, bound, *, task_id, rank, sink):
        self.bound, self.task_id, self.rank, self.sink = bound, task_id, rank, sink
        self.turn = 0

    async def __call__(self, *args, **kwargs):
        raise ValueError("CUAD budget-close evaluation requires structured native calls")

    async def generate_structured(self, messages, *, json_schema, **kwargs):
        from .episode import ModelProtocolFailure
        from .hf_native_structured import native_structured_schema_kind
        if native_structured_schema_kind(json_schema) != "cuad":
            raise ValueError("Exact CUAD evaluation schema required")
        row = dict(task_id=self.task_id, shard_index=self.rank, turn=self.turn,
                   messages=deepcopy(messages), json_schema=deepcopy(json_schema))
        self.turn += 1
        control = self.bound.new_request_control()
        pending = asyncio.create_task(self.bound.generate_structured(messages, json_schema=json_schema,
            _request_control=control, **kwargs))
        cancelled = False
        try:
            while True:
                try:
                    completion = await asyncio.shield(pending)
                    break
                except asyncio.CancelledError:
                    cancelled = True
                    control.expire()
                    if pending.done():
                        completion = pending.result()
                        break
        except ModelProtocolFailure as error:
            if error.trace_id != getattr(error.completion, "trace_id", None):
                raise ValueError("CUAD protocol failure and completion trace IDs differ") from error
            self._record(row, error.completion, cancelled=cancelled)
            if cancelled:
                raise asyncio.CancelledError() from error
            raise
        except BaseException as error:
            row.update(request_status="error", delivery="cancelled" if cancelled else "error",
                       error_type=type(error).__name__, error=str(error))
            preflight = getattr(error, "preflight", None)
            if preflight is not None:
                row["native_hf_preflight"] = deepcopy(dict(preflight))
            self.sink(row)
            if cancelled:
                raise asyncio.CancelledError() from error
            raise
        self._record(row, completion, cancelled=cancelled)
        if cancelled:
            raise asyncio.CancelledError()
        return completion

    def _record(self, row, completion, *, cancelled):
        from .hf_native_policy import CANCELLED_BEFORE_JSON
        if completion.native_status == CANCELLED_BEFORE_JSON and not cancelled:
            raise ValueError("Cancelled CUAD prefix requires cancelled delivery")
        trace_id = getattr(completion, "trace_id", None)
        if type(trace_id) is not str or not trace_id:
            raise ValueError("Actual CUAD native trace ID required")
        row.update(request_status="native", delivery="cancelled" if cancelled else "delivered",
                   trace_id=trace_id, completion=asdict(completion))
        self.sink(row)


def verify_native_ownership(tasks, predictions, owners, traces, *, rank, profile,
                            thought_tokens, json_tokens, max_prompt_tokens, eos_ids, tokenizer=None):
    """Verify closed actual traces/owners/steps, and public requests when generating.

    Score-only repeats cross-link checks without loading model weights. A
    preflight is a failed, unsubmitted request, never a completion or TRAIN
    proof. Cancelled completions stay owned and receive no benchmark credit.
    """
    from jsonschema import Draft202012Validator
    from .episode import parse_action
    from .hf_native_policy import CANCELLED_BEFORE_JSON, NativeHFRequestExpiredBeforeSubmission, _strict_json
    from .hf_native_structured import native_structured_schema_kind
    from .hf_policy import _chat_tokens
    from .native_hf_context import NativeHFPromptBudgetExceeded, PREFLIGHT_PROTOCOL, REASON

    expected = [task.id for task in tasks]
    if (profile not in {"raw", "thinking"} or not _tokens(eos_ids) or not eos_ids
            or len(set(eos_ids)) != len(eos_ids) or END_THINK in eos_ids
            or any(type(n) is not int or n < 1 for n in (thought_tokens, json_tokens, max_prompt_tokens))
            or len(expected) != len(set(expected))
            or any(task.benchmark != "cuad" for task in tasks)
            or [row.get("task_id") for row in predictions] != expected):
        raise ValueError("Exact CUAD EVAL configuration and assigned task order required")
    index = {}
    for trace in traces:
        key = trace.get("trace_id")
        if type(key) is not str or not key or key in index:
            raise ValueError("Missing or duplicate CUAD native trace ID")
        index[key] = trace
    if any(owner.get("task_id") not in expected for owner in owners):
        raise ValueError("Foreign CUAD native request owner")
    used, preflight_ids = set(), set()
    statuses = {"completed", "thought_did_not_stop_on_real_close", CANCELLED_BEFORE_JSON,
                "json_did_not_finish_with_real_eos", "json_not_complete_or_schema_valid"}
    forbidden = FORBIDDEN_FIELDS
    behavior = ("logprobs", "thought_logprobs", "thought_logprobs_source", "json_logprobs", "json_logprobs_source")
    for task, result in zip(tasks, predictions, strict=True):
        if result.get("benchmark") != "cuad" or result.get("split") != task.split:
            raise ValueError("Original CUAD task ownership differs")
        rows = [owner for owner in owners if owner.get("task_id") == task.id]
        if ([owner.get("turn") for owner in rows] != list(range(len(rows)))
                or any(type(owner.get("turn")) is not int or type(owner.get("shard_index")) is not int
                       or owner["shard_index"] != rank for owner in rows)):
            raise ValueError("Exact sequential CUAD request ownership required")
        if any(owner.get("request_status") == "error" or owner.get("delivery") == "cancelled"
               for owner in rows[:-1]):
            raise ValueError("Exact sequential CUAD request ownership required")
        delivered = [owner for owner in rows if owner.get("request_status") == "native"
                     and owner.get("delivery") == "delivered"]
        if canonical([owner["completion"] for owner in delivered]) != canonical(result.get("steps")):
            raise ValueError("Delivered CUAD requests differ from retained episode steps")
        for owner in rows:
            if native_structured_schema_kind(owner.get("json_schema")) != "cuad":
                raise ValueError("Exact public CUAD schema required")
            if owner.get("request_status") == "error":
                if (result.get("status") not in {"infrastructure_error", "context_budget_failure"}
                        or owner.get("delivery") not in {"error", "cancelled"}
                        or owner.get("delivery") == "cancelled" and result["status"] != "infrastructure_error"
                        or any(key in owner for key in ("completion", "trace_id", FIELD, POSITIONS))):
                    raise ValueError("Unsubmitted CUAD error cannot carry a native completion")
                record = owner.get("native_hf_preflight")
                if owner.get("error_type") == NativeHFRequestExpiredBeforeSubmission.__name__:
                    if record is not None or owner["delivery"] != "cancelled":
                        raise ValueError("Queued expiry cannot fabricate a CUAD preflight")
                    continue
                if owner.get("error_type") == NativeHFPromptBudgetExceeded.__name__:
                    if (not isinstance(record, dict) or record.get("protocol") != PREFLIGHT_PROTOCOL
                            or record.get("status") != REASON or record.get("task_id") != task.id
                            or type(record.get("trace_id")) is not str or not record["trace_id"]
                            or record["trace_id"] in index or record["trace_id"] in preflight_ids
                            or record.get("generation_submitted") is not False
                            or record.get("thought_sampling_profile") != profile
                            or record.get("thought_logprobs_source") is not None
                            or record.get("thought_tokens") != thought_tokens or record.get("json_tokens") != json_tokens
                            or record.get("effective_prompt_limit") != max_prompt_tokens
                            or not _tokens(record.get("prompt_ids")) or len(record["prompt_ids"]) <= max_prompt_tokens
                            or any(key in record for key in (*forbidden, FIELD, POSITIONS))):
                        raise ValueError("CUAD EVAL preflight must remain real, unsubmitted and nontrainable")
                    preflight_ids.add(record["trace_id"])
                    if tokenizer is not None and _chat_tokens(tokenizer, owner["messages"]) != record["prompt_ids"]:
                        raise ValueError("CUAD preflight differs from its actual full request")
                    if result["status"] == "context_budget_failure":
                        failure = result.get("metrics", {}).get("prompt_history_failure", {})
                        if (failure.get("native_hf_preflight") != record
                                or failure.get("generation_submitted") is not False
                                or failure.get("request_messages") != owner["messages"]):
                            raise ValueError("CUAD context outcome differs from the actual preflight")
                elif record is not None or result["status"] != "infrastructure_error":
                    raise ValueError("Untraced CUAD error lacks its retained unreliable outcome")
                continue
            if (owner.get("request_status") != "native" or owner.get("delivery") not in {"delivered", "cancelled"}
                    or owner["delivery"] == "cancelled" and result.get("status") != "infrastructure_error"):
                raise ValueError("Unknown or inconsistent CUAD native delivery")
            key, completion = owner.get("trace_id"), owner.get("completion", {})
            if key not in index or key in used or completion.get("trace_id") != key:
                raise ValueError("Missing, reused or foreign CUAD trace binding")
            trace = index[key]
            count, ids, prompt = completion.get("thought_token_count"), completion.get("completion_ids"), completion.get("prompt_ids")
            if (type(count) is not int or not _tokens(ids) or not 0 < count <= len(ids)
                    or not _tokens(prompt) or not 0 < len(prompt) <= max_prompt_tokens
                    or trace.get("protocol") != PROTOCOL or trace.get("task_id") != task.id
                    or trace.get("thought_sampling_profile") != profile
                    or trace.get("json_sampling_profile") != JSON_SAMPLING
                    or trace.get("prompt_ids") != prompt or trace.get("thought_ids") != ids[:count]
                    or trace.get("json_ids") != ids[count:] or trace.get("thought_token_count") != count
                    or trace.get("status") not in statuses or trace.get("status") != completion.get("native_status")
                    or trace.get("final_text") != completion.get("text")
                    or trace.get("json_public_schema") != owner["json_schema"]
                    or trace.get("json_stop_ids") != eos_ids or completion.get("json_stop_ids") != eos_ids
                    or completion.get("phase_lengths") != [count, len(ids) - count]
                    or len(ids) - count > json_tokens
                    or any(value.get(k) is not None for value in (trace, completion) for k in behavior)
                    or any(k in value for value in (trace, completion) for k in forbidden)):
                raise ValueError("CUAD native prompt/schema/tokens/status/provenance binding differs")
            for value in (trace, completion):
                validate_fields(value.get(FIELD), value.get(POSITIONS), ids[:count], thought_tokens=thought_tokens)
            thought, final = ids[:count], ids[count:]
            closed = thought[-1] == END_THINK and thought.count(END_THINK) == 1 and not set(thought).intersection(eos_ids)
            status = trace["status"]
            if status == "thought_did_not_stop_on_real_close":
                if closed or final or completion["text"]:
                    raise ValueError("Failed CUAD thought must retain only its actual failed prefix")
            elif not closed:
                raise ValueError("CUAD JSON or cancellation lacks a real unique thought close")
            if status == CANCELLED_BEFORE_JSON and (owner["delivery"] != "cancelled" or final or completion["text"]):
                raise ValueError("Cancelled CUAD prefix cannot fabricate JSON or delivered steps")
            if status == "completed":
                if not final or final[-1] not in eos_ids or set(final[:-1]).intersection(eos_ids):
                    raise ValueError("CUAD native success lacks a real final EOS")
                Draft202012Validator(owner["json_schema"]).validate(_strict_json(completion["text"]))
                if tokenizer is not None and tokenizer.decode(final[:-1], skip_special_tokens=False) != completion["text"]:
                    raise ValueError("CUAD JSON text differs from the actual native token IDs")
            if tokenizer is not None and _chat_tokens(tokenizer, owner["messages"]) != prompt:
                raise ValueError("CUAD public request differs from its actual native prompt IDs")
            used.add(key)
        if result.get("status") == "completed":
            if not delivered or any(owner["completion"]["native_status"] != "completed" for owner in delivered):
                raise ValueError("Completed CUAD episode requires actual successful delivered calls")
            action = parse_action(delivered[-1]["completion"]["text"])
            if action is None or action.get("answer") != result.get("answer"):
                raise ValueError("CUAD scored answer differs from the actual native final JSON")
    if used != set(index):
        raise ValueError("Unowned CUAD native trace remains in the closed shard")
    return dict(version=VERSION, requests=len(owners), native_traces=len(traces),
        delivered_steps=sum(len(row["steps"]) for row in predictions),
        cancelled_native_completions=sum(owner.get("request_status") == "native" and owner.get("delivery") == "cancelled" for owner in owners),
        all_native_traces_owned=True, evaluation_only=True, behavior_capture=False)
