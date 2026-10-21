"""CPU context-failure oracles; actual HF preflight, synthetic neural decoder."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json

import pytest
import torch

from slime_plugins.dcc import EpisodeRunner
from slime_plugins.dcc.data import Task
from slime_plugins.dcc.episode import EpisodeResult
from slime_plugins.dcc.hf_native_policy import END_THINK, NativeHFCompletion, NativeHFPolicy, PROTOCOL
from slime_plugins.dcc.hf_policy import _chat_tokens
from slime_plugins.dcc.native_behavior_logprobs import BEHAVIOR_LOGPROB_SOURCE
from slime_plugins.dcc.native_hf_context import PREFLIGHT_PROTOCOL, restore_verified_native_context
from slime_plugins.dcc.online_grpo import group_signal


class Tokenizer:
    def encode(self, text):
        return [1000 + ord(char) for char in text]
    def apply_chat_template(self, messages, **kwargs):
        return self.encode(json.dumps(messages)) + [151667]


def cuad_context(*, initial=False, capture=True, profile="raw"):
    """Real queue, runner, fixed request budget and preflight; no model load."""
    tokenizer, traces, decodes, requests = Tokenizer(), [], [], []
    policy = object.__new__(NativeHFPolicy)
    policy.tokenizer, policy.suffix_ids = tokenizer, [151667]
    policy.thought_tokens, policy.json_tokens, policy.batch_size = 4, 256, 1
    policy.sampling_profile, policy.capture_thought_logprobs = profile, capture
    policy.allow_truncation, policy.mixed_schema_batching = False, False
    policy.trace_sink, policy.queue, policy.worker = traces.append, asyncio.Queue(), None
    def decode(prompts, schema, task_ids):
        decodes.append(prompts)
        text = json.dumps({"tool": "read_document", "arguments": {"start": 0, "end": 5000}})
        final = tokenizer.encode(text) + [2]
        step = NativeHFCompletion(text, list(prompts[0]), [7, END_THINK] + final,
            thought_token_count=2, phase_lengths=(2, len(final)), trace_id="synthetic-native-1",
            thought_logprobs=[-1.25, -.75] if capture else None,
            thought_logprobs_source=BEHAVIOR_LOGPROB_SOURCE if capture else None)
        traces.append(dict(protocol=PROTOCOL, trace_id=step.trace_id, task_id=task_ids[0],
            status="completed", thought_sampling_profile=profile, prompt_ids=step.prompt_ids,
            thought_ids=[7, END_THINK], json_ids=final, thought_token_count=2,
            final_text=text, json_public_schema=schema,
            thought_logprobs=step.thought_logprobs, thought_logprobs_source=step.thought_logprobs_source))
        return [step]
    policy._generate_encoded_batch = decode
    class Bound:
        async def generate_structured(self, messages, **kwargs):
            requests.append(deepcopy(messages))
            if len(requests) == 1:
                policy.max_prompt_tokens = len(_chat_tokens(tokenizer, messages)) + (-1 if initial else 16)
            return await policy.bind_task("synthetic-cuad").generate_structured(messages, **kwargs)
    runner = EpisodeRunner(max_steps=2, max_new_tokens=260, initial_chunks=0,
        evaluation_temperature=1., cuad_structured_actions=True)
    task = Task("synthetic-cuad", "cuad", "train", "Find clause.", "source clause " * 500, ["source clause"])
    async def run():
        try:
            return await runner.run(task, Bound(), evaluation=True)
        finally:
            await policy.close()
    result = asyncio.run(run())
    assert not torch.cuda.is_initialized()
    return result, traces, tokenizer, decodes, requests


def signal(success, failed):
    return group_signal([success, failed], group_size=2, max_steps=2, thought_tokens=4)


def successful_sibling(result):
    return EpisodeResult(result.task_id, result.benchmark, result.split, [], "completed", 1., {},
        list(result.steps), [], [])


def deserialize(row):
    row = json.loads(json.dumps(row))
    row["steps"] = [NativeHFCompletion(**step) for step in row["steps"]]
    return EpisodeResult(**row)


def test_actual_post_action_preflight_retains_exact_sampled_prefix_with_zero_reward():
    failed, traces, _, decodes, requests = cuad_context()
    assert failed.status == "context_budget_failure" and failed.reward == 0
    assert len(requests) == 2 and len(decodes) == 1 and len(failed.steps) == 1
    assert failed.trainable_steps == []  # Preserve the legacy evaluation property.
    failure = failed.metrics["prompt_history_failure"]
    assert failure["generation_submitted"] is False and failure["turn_index"] == 1
    assert failure["stage"] == "after_model_call" and failure["prior_model_calls"] == 1
    assert [trace["protocol"] for trace in traces] == [PROTOCOL, PREFLIGHT_PROTOCOL]
    group = signal(successful_sibling(failed), failed)
    assert group.discarded_reason is None and group.rewards == [1., 0.]
    assert group.advantages[0] == pytest.approx(1 / 1.0002)
    assert group.advantages[1] == -group.advantages[0]
    assert group.segments[1][0].completion_ids == [7, END_THINK]
    assert group.segments[1][0].thought_logprobs == [-1.25, -.75]
    assert len(group.segments[1]) == 1  # No invented completion for the failed call.


def test_initial_overflow_has_no_sample_and_discards_group():
    failed, traces, _, decodes, requests = cuad_context(initial=True)
    assert failed.steps == [] and decodes == [] and len(requests) == 1
    assert failed.metrics["prompt_history_failure"]["stage"] == "initial_prompt"
    assert len(traces) == 1 and traces[0]["protocol"] == PREFLIGHT_PROTOCOL
    group = signal(successful_sibling(failed), failed)
    assert group.discarded_reason and group.advantages == [0., 0.]


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("profile", ["raw", "thinking", "greedy"])
def test_capture_disabled_evaluation_keeps_completion_only_sink(initial, profile):
    failed, traces, _, decodes, _ = cuad_context(initial=initial, capture=False, profile=profile)
    assert failed.status == "context_budget_failure" and failed.reward == 0
    assert len(traces) == len(decodes) == (0 if initial else 1)
    assert all(row["protocol"] == PROTOCOL for row in traces)
    assert signal(successful_sibling(failed), failed).discarded_reason


def test_serialized_outcome_needs_real_error_and_generation_journals_for_same_signal(tmp_path):
    failed, traces, tokenizer, _, _ = cuad_context()
    success = successful_sibling(failed)
    before = signal(success, failed)
    # Real serialization APIs followed by actual JSONL bytes on disk.
    outcomes = tmp_path / "outcomes.jsonl"
    native = tmp_path / "native.jsonl"
    outcomes.write_text("".join(json.dumps(row.to_dict()) + "\n" for row in [success, failed]))
    native.write_text("".join(json.dumps(row) + "\n" for row in traces))
    restored_success, unverified = [deserialize(json.loads(line)) for line in outcomes.read_text().splitlines()]
    assert signal(restored_success, unverified).discarded_reason
    restored = restore_verified_native_context(unverified,
        traces=[json.loads(line) for line in native.read_text().splitlines()], tokenizer=tokenizer)
    assert signal(restored_success, restored) == before


@pytest.mark.parametrize("corruption", ["no_error", "prompt", "messages", "logprobs", "source", "tokens", "ownership"])
def test_persisted_context_proof_requires_exact_request_tokens_and_native_trace(corruption):
    failed, traces, tokenizer, _, _ = cuad_context()
    failed = deserialize(failed.to_dict())
    traces = deepcopy(traces)
    if corruption == "no_error":
        traces = traces[:-1]
    elif corruption == "prompt":
        traces[-1]["prompt_ids"][0] += 1
        failed.metrics["prompt_history_failure"]["native_hf_preflight"]["prompt_ids"][0] += 1
    elif corruption == "messages":
        failed.metrics["prompt_history_failure"]["request_messages"][0]["content"] += "forged"
    elif corruption == "logprobs":
        failed.steps[0].thought_logprobs[0] = -9.
    elif corruption == "source":
        failed.steps[0].thought_logprobs_source = "teacher-forced-replay"
    elif corruption == "tokens":
        failed.steps[0].completion_ids[0] += 1
    else:
        failed.steps[0].trace_id = "foreign"
    with pytest.raises(ValueError):
        restore_verified_native_context(failed, traces=traces, tokenizer=tokenizer)


@pytest.mark.parametrize("scores", [None, [], [float("nan"), -.75], [-1., .1]])
def test_verified_context_prefix_rejects_missing_or_invalid_actual_behavior(scores):
    failed, _, _, _, _ = cuad_context()
    failed.steps[0].thought_logprobs = scores
    with pytest.raises(ValueError, match="actual sampled thought behavior"):
        signal(successful_sibling(failed), failed)


@pytest.mark.parametrize("corruption", ["trace_id", "phase_lengths", "no_thought", "json_token"])
def test_context_prefix_requires_original_nonempty_native_token_boundaries_and_ownership(corruption):
    failed, _, _, _, _ = cuad_context()
    step = failed.steps[0]
    if corruption == "trace_id":
        step.trace_id = None
    elif corruption == "phase_lengths":
        step.phase_lengths = (True, len(step.completion_ids) - 2)
    elif corruption == "no_thought":
        step.thought_token_count, step.thought_logprobs = 0, []
    else:
        step.completion_ids[-1] = -1
    with pytest.raises(ValueError):
        signal(successful_sibling(failed), failed)


def test_status_reason_and_forged_plain_diagnostics_do_not_rescue_context_or_infrastructure():
    failed, _, _, _, _ = cuad_context()
    success = successful_sibling(failed)
    failure = failed.metrics["prompt_history_failure"]
    failure["native_hf_preflight"] = dict(failure["native_hf_preflight"])
    assert signal(success, failed).discarded_reason
    failed.status = "infrastructure_error"
    assert signal(success, failed).discarded_reason
