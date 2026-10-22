"""Native HF TRAIN context failures, with replayable request provenance.

Plain outcome strings are not proof of an HF preflight. Live proofs originate
in the HF exception; serialized proofs must be rebound to actual native journal
records and the encoded failed request before they can provide gradients.
This module stays lightweight for the shared, non-HF EpisodeRunner.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
import uuid

from .prompt_history import PromptHistoryBudgetExceeded
from . import web_train_thought_budget_close as train_close
from . import cuad_train_thought_budget_close as cuad_close

CONTEXT_HANDLING = "hf-native-context-sampled-prefix-v2"
PREFLIGHT_PROTOCOL = "hf-native-request-preflight-v1"
HF_PROTOCOL = "hf-native-two-stage-thought-only-policy-v1"
REASON = "full_context_exceeds_native_HF_limit"


class _VerifiedPreflight(dict):
    """JSON-compatible proof; ordinary deserialization does not verify it."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verified = False

    @classmethod
    def issue(cls, record):
        value = cls(deepcopy(record))
        value.verified = True
        return value


class NativeHFPromptBudgetExceeded(PromptHistoryBudgetExceeded):
    """Raised by the actual full-request HF preflight, before generation."""
    def __init__(self, prompt_ids, *, limit, task_id, thought_tokens, json_tokens,
                 sampling_profile, behavior_source, web_train_thought_budget_close=None,
                 cuad_train_thought_budget_close=None):
        super().__init__(REASON, diagnostics={"original_prompt_tokens": len(prompt_ids),
            "minimum_prompt_tokens": len(prompt_ids), "effective_prompt_limit": limit},
            original_prompt_ids=prompt_ids)
        self.preflight = _VerifiedPreflight.issue(dict(protocol=PREFLIGHT_PROTOCOL,
            trace_id=uuid.uuid4().hex, task_id=task_id, status=REASON,
            generation_submitted=False, prompt_ids=list(prompt_ids),
            effective_prompt_limit=limit, thought_tokens=thought_tokens,
            json_tokens=json_tokens, thought_sampling_profile=sampling_profile,
            thought_logprobs_source=behavior_source,
            **({train_close.FIELD: deepcopy(web_train_thought_budget_close)}
               if web_train_thought_budget_close is not None else {}),
            **({cuad_close.FIELD: deepcopy(cuad_train_thought_budget_close)}
               if cuad_train_thought_budget_close is not None else {})))


def context_handling_configuration():
    return dict(version=CONTEXT_HANDLING, initial_prompt="discard-whole-group",
        verified_post_sampling_overflow="retain-actual-thought-prefix-with-zero-reward",
        unverified_context="discard-whole-group", infrastructure="discard-whole-group",
        preflight_protocol=PREFLIGHT_PROTOCOL, serialized_proof="verify-native-journals-and-request-tokens")


def thought_policy_fields(step, *, thought_tokens):
    """Dispatch independent newly captured policies; old raw steps stay raw."""
    if (getattr(step, train_close.FIELD, None) is not None
            and getattr(step, cuad_close.FIELD, None) is not None):
        raise ValueError("A completion cannot claim both Web and CUAD singleton policies")
    web_fields = train_close.step_fields(step, thought_tokens=thought_tokens)
    cuad_fields = cuad_close.step_fields(step, thought_tokens=thought_tokens)
    return web_fields or cuad_fields


def validated_thought_step(step, *, thought_tokens):
    """The same actual-token/behavior checks for both context and normal groups."""
    from .native_behavior_logprobs import BEHAVIOR_LOGPROB_SOURCE
    count = getattr(step, "thought_token_count", None)
    if (type(count) is not int or not 0 <= count <= thought_tokens
            or count > len(step.completion_ids) or not step.prompt_ids):
        raise ValueError("Native completion must retain actual thought token boundaries")
    thought = list(step.completion_ids[:count])
    if any(type(token) is not int or token < 0 for token in thought + step.prompt_ids):
        raise ValueError("Actual nonnegative integer token IDs required")
    behavior = getattr(step, "thought_logprobs", None)
    fields = thought_policy_fields(step, thought_tokens=thought_tokens)
    expected_source = (cuad_close.SOURCE if cuad_close.FIELD in fields else train_close.SOURCE
                       if train_close.FIELD in fields else BEHAVIOR_LOGPROB_SOURCE)
    if (not isinstance(behavior, list) or len(behavior) != count
            or getattr(step, "thought_logprobs_source", None) != expected_source
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   or value > 1e-6 for value in behavior)):
        raise ValueError("Finite aligned actual sampled thought behavior logprobs required")
    return list(step.prompt_ids), thought, list(behavior)


def _preflight_behavior_source(record):
    from .native_behavior_logprobs import BEHAVIOR_LOGPROB_SOURCE
    if any(key in record for key in (train_close.POSITIONS, cuad_close.POSITIONS)):
        raise ValueError("Unsubmitted preflight cannot carry sampled singleton positions")
    if train_close.FIELD in record and cuad_close.FIELD in record:
        raise ValueError("A preflight cannot claim both Web and CUAD singleton policies")
    if cuad_close.FIELD in record:
        cuad_close.validate_configuration(record[cuad_close.FIELD], thought_tokens=record.get("thought_tokens"))
        return cuad_close.SOURCE
    profile = record.get(train_close.FIELD)
    if profile is not None:
        train_close.validate_configuration(profile, thought_tokens=record.get("thought_tokens"))
        return train_close.SOURCE
    if train_close.FIELD in record:
        raise ValueError("Legacy preflight cannot carry an empty singleton TRAIN identity")
    return BEHAVIOR_LOGPROB_SOURCE


def _validate_failure(result, record):
    from .native_behavior_logprobs import BEHAVIOR_LOGPROB_SOURCE
    failure = result.metrics.get("prompt_history_failure", {})
    ids, limit = record.get("prompt_ids"), record.get("effective_prompt_limit")
    if (record.get("protocol") != PREFLIGHT_PROTOCOL or record.get("status") != REASON
            or record.get("task_id") != result.task_id
            or type(record.get("trace_id")) is not str or not record["trace_id"]
            or record.get("generation_submitted") is not False
            or record.get("thought_sampling_profile") != "raw"
            or record.get("thought_logprobs_source") != _preflight_behavior_source(record)
            or any(type(record.get(key)) is not int or record[key] < 1
                   for key in ("effective_prompt_limit", "thought_tokens", "json_tokens"))
            or type(ids) is not list or len(ids) <= limit
            or any(type(token) is not int or token < 0 for token in ids)
            or failure.get("reason") != REASON or failure.get("generation_submitted") is not False
            or failure.get("diagnostics") != dict(original_prompt_tokens=len(ids),
                minimum_prompt_tokens=len(ids), effective_prompt_limit=limit)
            or failure.get("operation") != "policy"
            or type(failure.get("turn_index")) is not int or failure["turn_index"] != len(result.steps)
            or type(failure.get("prior_model_calls")) is not int
            or failure["prior_model_calls"] != len(result.steps)
            or failure.get("stage") != ("after_model_call" if result.steps else "initial_prompt")
            or not isinstance(failure.get("request_messages"), list)
            or result.reward != 0):
        raise ValueError("Native HF context failure provenance differs from actual request")


def retains_sampled_context_prefix(result):
    """Only a verified raw-HF post-sampling failure is part of the objective."""
    if result.status != "context_budget_failure":
        return False
    record = result.metrics.get("prompt_history_failure", {}).get("native_hf_preflight")
    if not isinstance(record, _VerifiedPreflight) or not record.verified:
        return False
    # Evaluation/capture-disabled requests have no behavior-policy provenance.
    from .native_behavior_logprobs import BEHAVIOR_LOGPROB_SOURCE
    if (record.get("thought_sampling_profile") != "raw"
            or record.get("thought_logprobs_source") != _preflight_behavior_source(record)):
        return False
    _validate_failure(result, record)
    if not result.steps:
        return False
    from .hf_native_policy import END_THINK, NativeHFCompletion
    used = {record["trace_id"]}
    for step in result.steps:
        prompt, thought, _ = validated_thought_step(step, thought_tokens=record["thought_tokens"])
        fields = thought_policy_fields(step, thought_tokens=record["thought_tokens"])
        if any(fields.get(key) != record.get(key) for key in (train_close.FIELD, cuad_close.FIELD)):
            raise ValueError("Sampled context prefix and failed request thought policies differ")
        trace_id = getattr(step, "trace_id", None)
        phases = getattr(step, "phase_lengths", None)
        if (not isinstance(step, NativeHFCompletion) or type(trace_id) is not str
                or not trace_id or trace_id in used or not thought or thought[-1] != END_THINK
                or thought.count(END_THINK) != 1 or step.native_status != "completed"
                or type(phases) not in (tuple, list) or len(phases) != 2
                or any(type(value) is not int or value < 1 for value in phases)
                or tuple(phases) != (len(thought), len(step.completion_ids) - len(thought))
                or phases[1] > record["json_tokens"]
                or any(type(token) is not int or token < 0 for token in step.completion_ids)
                or len(prompt) > record["effective_prompt_limit"]):
            raise ValueError("Verified native HF sampled prefix and completion ownership required")
        used.add(trace_id)
    return True


def unreliable_context_or_environment_group(results):
    """Shared by TRAIN reward assignment and GRPO; never change group members."""
    return any(result.status == "infrastructure_error" or
        (result.status == "context_budget_failure" and not retains_sampled_context_prefix(result))
        for result in results)


def restore_verified_native_context(result, *, traces, tokenizer):
    """Rebind a deserialized failure using actual saved native/error records.

    The caller supplies its closed native journal, not a diagnostic synthesized
    from the outcome. All retained steps must match their own raw native traces;
    the failed full prompt is re-encoded with the same tokenizer. No token or
    missing behavior score is reconstructed or filled in.
    """
    from .hf_policy import _chat_tokens
    if result.status != "context_budget_failure":
        raise ValueError("Native context failure required for provenance restoration")
    failure = result.metrics.get("prompt_history_failure", {})
    record = failure.get("native_hf_preflight")
    if not isinstance(record, dict):
        raise ValueError("Persisted native HF preflight record required")
    index = {row["trace_id"]: row for row in traces}
    if len(index) != len(traces) or index.get(record.get("trace_id")) != record:
        raise ValueError("Actual native HF preflight error journal record required")
    _validate_failure(result, record)
    if _chat_tokens(tokenizer, failure["request_messages"]) != record["prompt_ids"]:
        raise ValueError("Failed request does not encode to actual HF preflight prompt IDs")
    for step in result.steps:
        count = getattr(step, "thought_token_count", None)
        row = index.get(getattr(step, "trace_id", None), {})
        from .hf_native_policy import JOINT_PROTOCOL, JointNativeHFCompletion
        joint = isinstance(step, JointNativeHFCompletion)
        fields = thought_policy_fields(step, thought_tokens=record["thought_tokens"])
        owner_policy = cuad_close if cuad_close.FIELD in fields else train_close
        expected_protocol = owner_policy.PROTOCOL if fields else JOINT_PROTOCOL if joint else HF_PROTOCOL
        if fields:
            trace_fields = owner_policy.validate_fields(row.get(owner_policy.FIELD),
                row.get(owner_policy.POSITIONS), row.get("thought_ids"), row.get("thought_logprobs"),
                thought_tokens=record["thought_tokens"])
            if trace_fields != fields:
                raise ValueError("Retained singleton policy differs from its actual native journal")
        if (row.get("protocol") != expected_protocol or row.get("task_id") != result.task_id
                or row.get("thought_sampling_profile") != "raw"
                or row.get("prompt_ids") != step.prompt_ids
                or type(count) is not int or row.get("thought_token_count") != count
                or row.get("thought_ids") != step.completion_ids[:count]
                or row.get("json_ids") != step.completion_ids[count:]
                or row.get("thought_logprobs") != step.thought_logprobs
                or row.get("thought_logprobs_source") != step.thought_logprobs_source
                or row.get("status") != step.native_status or row.get("final_text") != step.text):
            raise ValueError("Retained sampled step differs from actual native generation journal")
        if ({key: row.get(key) for key in fields} != fields
                or any(key in row and key not in fields for key in
                    (train_close.FIELD, train_close.POSITIONS, cuad_close.FIELD, cuad_close.POSITIONS))):
            raise ValueError("Retained singleton policy differs from its actual native journal")
        if joint and (row.get("json_logprobs") != step.json_logprobs
                or row.get("json_logprobs_source") != step.json_logprobs_source
                or row.get("json_public_schema") != step.json_public_schema
                or row.get("json_stop_ids") != step.json_stop_ids):
            raise ValueError("Retained sampled JSON differs from actual native generation journal")
    metrics = deepcopy(result.metrics)
    metrics["prompt_history_failure"]["native_hf_preflight"] = _VerifiedPreflight.issue(record)
    restored = replace(result, metrics=metrics)
    retains_sampled_context_prefix(restored)  # Validate actual boundaries/logprobs, including zero-prefix cases.
    return restored
