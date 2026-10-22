"""Synthetic CPU controls for the explicit Web TRAIN singleton-close policy.

These tests exercise actual sampling, owned journals and replay mathematics;
they are neither benchmark results nor evidence of model capability.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import asdict, replace
import json
from types import SimpleNamespace
from threading import Event

import pytest
import torch
from transformers import GenerationConfig, LogitsProcessorList

from slime_plugins.dcc import online_web as web
from slime_plugins.dcc import web_train_thought_budget_close as close
from slime_plugins.dcc.cuad_structured import cuad_final_schema
from slime_plugins.dcc.episode import EpisodeResult, ModelProtocolFailure
from slime_plugins.dcc.hf_native_policy import (BudgetClosedNativeHFCompletion,
    NativeHFCompletion, NativeHFPolicy, PROTOCOL, _explicit_generation_kwargs)
from slime_plugins.dcc.hf_policy import _chat_tokens
from slime_plugins.dcc.native_behavior_logprobs import (BEHAVIOR_LOGPROB_SOURCE,
    RowSelectiveThoughtBudgetClose, StreamingBudgetClosedThoughtLogprobs)
from slime_plugins.dcc.native_hf_context import (NativeHFPromptBudgetExceeded,
    restore_verified_native_context)
from slime_plugins.dcc.online_grpo import (GroupSignal, ThoughtSegment,
    apply_policy_update, group_signal, thought_policy_loss)
from slime_plugins.dcc.prompt_history import PromptHistoryConfig
from slime_plugins.dcc.web_structured import web_final_schema
from slime_plugins.dcc.web_train_prompt_history import (web_train_history_budgets,
    web_train_history_configuration)
from test_dcc_hf_native_policy import (ScriptedModel, _call_generation_config,
    final_ids, messages, tokenizer)
from test_dcc_native_behavior_logprobs import tiny_model
from test_dcc_online_grpo import load_trainer
from test_dcc_online_web import COMMIT, public_row, runner
from test_dcc_web_hf_prompt_history import CONFIG, URL, budgets, history_messages, normalized


class SampledModel(ScriptedModel):
    """Invented logits, actual processor chain and actual multinomial selection."""
    def generate(self, **kwargs):
        config = _call_generation_config(kwargs)
        if close.END_THINK not in config.eos_token_id:
            return super().generate(**kwargs)
        self.calls.append({**kwargs, "generation_config": config})
        ids = kwargs["input_ids"]
        done = torch.zeros(len(ids), dtype=torch.bool)
        self.observed = []
        for position in range(config.max_new_tokens):
            raw = torch.zeros((len(ids), self.config.vocab_size))
            for row, plan in enumerate(self.thoughts):
                raw[row, plan[min(position, len(plan) - 1)]] = 20.
            scores = raw
            for processor in kwargs["logits_processor"]:
                scores = processor(ids, scores)
            sampled = torch.multinomial(scores.softmax(-1), 1)
            sampled[done] = config.pad_token_id
            self.observed.append((raw.clone(), scores.clone(), sampled.clone()))
            ids = torch.cat((ids, sampled), dim=1)
            done |= torch.isin(sampled[:, 0], torch.tensor(config.eos_token_id))
            if done.all():
                break
        return ids


def native(tokenizer, *, thoughts=None, finals=None, sink=None, budget=4,
           enabled=True, history=None, limits=None):
    thoughts = thoughts or [[10065, 10066, 10067, 10068]]
    text = json.dumps({"answer": [{"id": "a", "value": "8", "citations": [URL]}]})
    model = SampledModel(thoughts, finals or [final_ids(tokenizer, text=text)])
    model.config = SimpleNamespace(vocab_size=151670, max_position_embeddings=10000)
    options = {} if enabled is None else {close.FIELD: enabled}
    if history is not None:
        options["web_train_prompt_history_config"] = history
    torch.manual_seed(37)
    policy = NativeHFPolicy(model, tokenizer,
        **(limits or dict(thought_tokens=budget, json_tokens=256, max_prompt_tokens=4096)),
        capture_thought_logprobs=True, mixed_schema_batching=True,
        trace_sink=sink, **options)
    return policy


def test_real_tiny_hf_controlled_sampling_tracks_processed_scores_and_preserves_rng():
    model = tiny_model()
    ids = torch.tensor([[2, 3, 4], [5, 6, 7], [2, 2, 8]])
    attention = torch.tensor([[0, 1, 1], [1, 1, 1], [0, 0, 1]])
    config = GenerationConfig(do_sample=True, temperature=1., top_p=1., top_k=0,
        pad_token_id=2, eos_token_id=[31], max_new_tokens=7, num_beams=1,
        num_return_sequences=1, min_length=0, early_stopping=False,
        repetition_penalty=1., renormalize_logits=False,
        return_dict_in_generate=True, output_scores=True, output_logits=True, use_cache=True)
    def run(tracked):
        rule = RowSelectiveThoughtBudgetClose([True, False, True],
            prompt_width=3, thought_tokens=7, close_id=31)
        tracker = StreamingBudgetClosedThoughtLogprobs(rule)
        torch.manual_seed(37)
        with torch.inference_mode():
            output = model.generate(input_ids=ids, attention_mask=attention,
                logits_processor=LogitsProcessorList([rule, tracker] if tracked else [rule]),
                **_explicit_generation_kwargs(config))
        rng = torch.get_rng_state().clone()
        rows = output.sequences[:, 3:].tolist()
        rows = [row[:next((i + 1 for i, token in enumerate(row) if token == 31), len(row))] for row in rows]
        return output, rows, rng, tracker
    baseline, _, rng, _ = run(False)
    output, rows, tracked_rng, tracker = run(True)
    assert torch.equal(output.sequences, baseline.sequences)
    assert torch.equal(rng, tracked_rng)
    actual = tracker.finish(rows)
    for row, tokens in enumerate(rows):
        oracle = [output.scores[t][row].log_softmax(-1)[token].item()
                  for t, token in enumerate(tokens)]
        assert actual[row] == pytest.approx(oracle, abs=1e-6)
        forced = row != 1 and len(tokens) == 7
        assert tracker.forced_positions[row] == ([6] if forced else [])
        if forced:
            assert tokens[-1] == 31 and actual[row][-1] == 0
            assert output.logits[-1][row].log_softmax(-1)[31] < 0
    assert rows[0][-1] == rows[2][-1] == 31
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("budget,thoughts", [(1, [[10065]]),
    (4, [[10065, 10066, 10067, 10068]]), (4, [[10065, close.END_THINK]]),
    (4, [[10065, 151645]])])
def test_actual_native_web_close_natural_close_early_eos_and_budget_one(tokenizer, budget, thoughts):
    traces = []
    policy = native(tokenizer, thoughts=thoughts, budget=budget, sink=traces.append)
    result = policy._generate_batch([messages()], web_final_schema(["id", "value"], [URL]), ["web"])[0]
    completion = result.completion if isinstance(result, ModelProtocolFailure) else result
    assert isinstance(completion, BudgetClosedNativeHFCompletion)
    assert completion.web_train_thought_budget_close == close.configuration(budget)
    count = completion.thought_token_count
    assert completion.forced_thought_positions == ([budget - 1] if count == budget else [])
    assert traces[0]["protocol"] == close.PROTOCOL
    assert completion.thought_logprobs_source == close.SOURCE
    if count == budget:
        assert completion.completion_ids[count - 1] == close.END_THINK
        assert completion.thought_logprobs[-1] == 0
    elif thoughts[0][-1] == close.END_THINK:
        assert completion.thought_logprobs[-1] < 0
    else:
        assert isinstance(result, ModelProtocolFailure)
        assert len(policy.model.calls) == 1 and completion.completion_ids == thoughts[0]
    for call in policy.model.calls:
        assert call["generation_config"].forced_eos_token_id is None
    if len(policy.model.calls) == 2:
        thought, final = policy.model.calls
        assert len(thought["logits_processor"]) == 2
        assert not final["generation_config"].do_sample
        assert len(final["logits_processor"]) == 1
        assert final["input_ids"][0].tolist() == completion.prompt_ids + completion.completion_ids[:count]


def test_mixed_web_cuad_ragged_batch_has_row_selective_policy_and_identities(tokenizer):
    traces = []
    web_text = json.dumps({"answer": [{"id": "a", "value": "8", "citations": [URL]}]})
    policy = native(tokenizer, thoughts=[[10065] * 4, [10066] * 3 + [close.END_THINK],
        [10067, close.END_THINK]], finals=[final_ids(tokenizer, text=web_text),
            final_ids(tokenizer), final_ids(tokenizer, text=web_text)], sink=traces.append)
    web_schema = web_final_schema(["id", "value"], [URL])
    results = policy._generate_batch([messages("A"), messages("longer CUAD"), messages("B")],
        [web_schema, cuad_final_schema(), web_schema], ["A", "cuad", "B"])
    assert all(isinstance(result, NativeHFCompletion) for result in results)
    assert [type(result) for result in results] == [BudgetClosedNativeHFCompletion,
        NativeHFCompletion, BudgetClosedNativeHFCompletion]
    assert [trace["protocol"] for trace in traces] == [close.PROTOCOL, PROTOCOL, close.PROTOCOL]
    assert results[0].forced_thought_positions == [3] and results[2].forced_thought_positions == []
    assert results[1].thought_logprobs_source == BEHAVIOR_LOGPROB_SOURCE
    assert results[1].thought_logprobs[-1] < 0
    assert close.FIELD not in asdict(results[1]) and close.POSITIONS not in traces[1]
    assert results[2].thought_token_count == 2
    raw, processed, _ = policy.model.observed[-1]
    torch.testing.assert_close(processed[1], raw[1], rtol=0, atol=0)
    assert torch.isneginf(processed[0]).sum() == processed.shape[-1] - 1


def test_default_missing_and_false_preserve_serialization_scores_ids_rng_and_configuration(tokenizer):
    records = []
    for enabled in (None, False):
        traces = []
        policy = native(tokenizer, enabled=enabled, thoughts=[[10065, close.END_THINK]], sink=traces.append)
        result = policy._generate_batch([messages()], web_final_schema(["id", "value"], [URL]), ["web"])[0]
        assert type(result) is NativeHFCompletion
        assert close.FIELD not in asdict(result) and close.FIELD not in traces[0]
        assert close.FIELD not in policy.provenance
        records.append((normalized(asdict(result)), normalized(traces), policy.provenance,
            torch.get_rng_state().tolist(), [call["generation_config"].to_dict() for call in policy.model.calls]))
    assert records[0] == records[1]
    train = load_trainer()
    args = SimpleNamespace(model="synthetic", source_commit=COMMIT, group_size=2,
        max_steps=3, thought_tokens=4, enable_online_web=True, cuad_draws_per_cycle=1, web_draws_per_cycle=1)
    baseline = train.configuration(args, [], [], 1)
    args.web_train_thought_budget_close = False
    assert train.configuration(args, [], [], 1) == baseline
    args.web_train_thought_budget_close = True
    enabled = train.configuration(args, [], [], 1)
    assert enabled[close.FIELD] == close.configuration(4)
    assert enabled["objective_denominator"] == baseline["objective_denominator"] == 24
    assert close.VARIANT in enabled["algorithm"]


@pytest.mark.parametrize("extra", [[], ["--enable-online-web", "--joint-json-policy"],
    ["--enable-online-web", "--policy-surrogate", "episode-gspo"]])
def test_unsupported_modes_rejected_before_source_data_model_or_cuda(monkeypatch, extra):
    train = load_trainer()
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid mode must fail before source, data, model or CUDA")
    for name in ("load_tasks", "validate_source", "load_policy_model"):
        monkeypatch.setattr(train, name, forbidden)
    monkeypatch.setattr(torch.cuda, "set_device", forbidden)
    with pytest.raises(SystemExit) as error:
        train.main(["--model", "synthetic", "--train-data", "absent", "--output-dir", "unused",
            "--source-commit", COMMIT, "--web-train-thought-budget-close", *extra])
    assert error.value.code == 2
    with pytest.raises(ValueError, match="requires online Web"):
        asyncio.run(train.train(SimpleNamespace(web_train_thought_budget_close=True), [], [], 1))


@pytest.mark.parametrize("options", [dict(capture_thought_logprobs=False),
    dict(capture_thought_logprobs=True, sampling_profile="greedy"),
    dict(capture_thought_logprobs=True, joint_json_policy=True),
    dict(capture_thought_logprobs=True, web_thought_budget_close=True),
    dict(capture_thought_logprobs=True, allow_truncation=True)])
def test_native_control_restrictions_before_model_access(tokenizer, options):
    with pytest.raises(ValueError):
        NativeHFPolicy(None, tokenizer, web_train_thought_budget_close=True, **options)


def test_analytical_forced_mask_keeps_fixed_denominator_and_natural_close_gradient():
    raw = torch.tensor([-.2, -.3, -7.], requires_grad=True)
    segment = ThoughtSegment([1], [4, 5, close.END_THINK], [-.2, -.3, 0.],
        close.configuration(3), (2,))
    scorer = close.scorer_for(segment, lambda *args: raw)
    current = scorer(None, [1], [4, 5, close.END_THINK])
    assert current.tolist() == pytest.approx([-.2, -.3, 0.])
    old = torch.tensor(segment.thought_logprobs)
    loss = thought_policy_loss(current, old, 1., group_size=2, max_steps=4,
        thought_tokens=3, policy_mask=close.policy_mask(segment, current))
    assert loss.item() == pytest.approx(-2 / 24)
    loss.backward()
    torch.testing.assert_close(raw.grad, torch.tensor([-1 / 24, -1 / 24, 0.]))
    natural = replace(segment, completion_ids=[4, close.END_THINK],
        thought_logprobs=[-.2, -.3], forced_thought_positions=())
    original = lambda *args: raw[:2]
    assert close.scorer_for(natural, original) is original
    with pytest.raises(ValueError, match="exact captured"):
        scorer(None, [2], segment.completion_ids)


def test_actual_replay_diagnostic_reference_and_adam_skip_use_singleton_policy(monkeypatch):
    train = load_trainer()
    parameter = torch.nn.Parameter(torch.tensor([-.4, -.5, -8.]))
    model = torch.nn.Linear(1, 1)
    segment = ThoughtSegment([1], [4, 5, close.END_THINK], [-.4, -.5, 0.], close.configuration(3), (2,))
    signal = GroupSignal([[segment], []], [1., -1.], [1., 0.], .5)
    calls = []
    def raw_scorer(model, prompt, completion):
        calls.append((model.training, prompt, completion))
        return parameter
    monkeypatch.setattr(train, "validate_base_reference_model", lambda model: None)
    def reference(model, scorer, prompt, completion):
        result = scorer(model, prompt, completion).detach()
        assert result[-1] == 0
        return result
    monkeypatch.setattr(train, "frozen_base_logprobs", reference)
    report = train.replay_group(model, signal, raw_scorer, group_size=2, max_steps=2,
        thought_tokens=3, clip=.2, reference_kl_beta=.03,
        reference_base_model="synthetic", reference_source_commit=COMMIT)
    assert len(calls) == 3 and all(call[2] == segment.completion_ids for call in calls)
    assert report["forced_close_scored_tokens"] == 1
    assert report["stochastic_scored_tokens"] == 2
    assert report["reference_kl_unweighted_loss"] == 0
    assert report["importance_ratio_min"] == report["importance_ratio_max"] == 1
    torch.testing.assert_close(parameter.grad, torch.tensor([-1 / 12, -1 / 12, 0.]))
    optimizer = torch.optim.AdamW([parameter], lr=.1)
    _, applied = apply_policy_update([parameter], optimizer)
    assert applied
    optimizer.zero_grad(set_to_none=True)
    deterministic = ThoughtSegment([1], [close.END_THINK], [0.], close.configuration(1), (0,))
    signal = GroupSignal([[deterministic], []], [1., -1.], [1., 0.], .5)
    before, step = parameter.detach().clone(), optimizer.state[parameter]["step"].item()
    report = train.replay_group(model, signal, lambda *args: parameter[-1:],
        group_size=2, max_steps=2, thought_tokens=1, clip=.2)
    assert report["policy_loss"] == 0 and report["stochastic_scored_tokens"] == 0
    assert report["importance_ratio_min"] is None
    norm, applied = apply_policy_update([parameter], optimizer)
    assert norm == 0 and not applied
    assert torch.equal(parameter, before) and optimizer.state[parameter]["step"] == step


def closed_group(tmp_path, tokenizer, *, history=None):
    task = web.public_web_task(public_row())
    journals = web.PublicGroupJournals(tmp_path / "group", task=task, group_size=2,
        iteration=1, rank=0, source_commit=COMMIT)
    request = history_messages() if history else messages()
    limits = budgets(tokenizer, request) if history else None
    text = json.dumps({"answer": [{"id": "a", "value": "8", "citations": [URL]}]})
    policy = native(tokenizer, thoughts=[[10065] * 4] * 2,
        finals=[final_ids(tokenizer, text=text)] * 2, history=history, limits=limits,
        sink=lambda row: journals.enqueue("native_traces", row))
    owners = [web.OwnedEpisodePolicy(policy.bind_task(task.id), task_id=task.id,
        iteration=1, rank=0, sample=sample, sink=lambda row: journals.enqueue("request_owners", row))
        for sample in range(2)]
    async def run():
        try:
            return await asyncio.gather(*(owner.generate_structured(request,
                json_schema=web_final_schema(["id", "value"], [URL]),
                max_new_tokens=policy.thought_tokens + policy.json_tokens, temperature=1.) for owner in owners))
        finally:
            await policy.close()
    completions = asyncio.run(run())
    results = [EpisodeResult(task.id, "widesearch", "train", [], "completed", 0., {}, [step], [], [])
        for step in completions]
    group = journals.close_generation(results, policy=policy)
    owners = web._read_jsonl(journals.directory / "request_owners.jsonl")
    traces = web._read_jsonl(journals.directory / "native_traces.jsonl")
    return group, policy, owners, traces


@pytest.mark.parametrize("history", [None, CONFIG])
def test_actual_closed_group_ownership_history_and_signal(tmp_path, tokenizer, history):
    group, policy, owners, traces = closed_group(tmp_path, tokenizer, history=history)
    kwargs = dict(eos_ids=policy.eos_ids, tokenizer=tokenizer, iteration=1, rank=0,
        web_train_thought_budget_close=close.configuration(policy.thought_tokens))
    if history:
        kwargs.update(web_train_prompt_history_config=history, history_budgets=web_train_history_budgets(policy))
        for trace in traces:
            assert trace["native_web_train_prompt_history"] == web_train_history_configuration(
                history, thought_policy=close.configuration(policy.thought_tokens))
    assert web.verify_owned_native_group(group.task, group.results, owners, traces, **kwargs)
    training_results = [replace(result, reward=float(i)) for i, result in enumerate(group.results)]
    signal = group_signal(training_results, group_size=2, max_steps=3, thought_tokens=policy.thought_tokens)
    assert signal.discarded_reason is None
    assert all(segments[0].forced_thought_positions == (policy.thought_tokens - 1,) for segments in signal.segments)
    with pytest.raises(ValueError):
        web.verify_owned_native_group(group.task, group.results, owners, traces,
            **dict(kwargs, web_train_thought_budget_close=None))
    for field in (close.FIELD, close.POSITIONS, "thought_logprobs_source", "protocol"):
        bad = deepcopy(traces)
        bad[0].pop(field)
        with pytest.raises(ValueError):
            web.verify_owned_native_group(group.task, group.results, owners, bad, **kwargs)
    for collection in ("owners", "traces"):
        bad_owners, bad_traces = deepcopy(owners), deepcopy(traces)
        rows = bad_owners if collection == "owners" else bad_traces
        rows[0][close.FIELD]["enabled"] = 1
        with pytest.raises(ValueError):
            web.verify_owned_native_group(group.task, group.results, bad_owners, bad_traces, **kwargs)
    bad_traces = deepcopy(traces)
    bad_traces[0][close.POSITIONS] = [float(policy.thought_tokens - 1)]
    with pytest.raises(ValueError):
        web.verify_owned_native_group(group.task, group.results, owners, bad_traces, **kwargs)


@pytest.mark.parametrize("corruption", ["source", "profile", "positions", "token", "logprob"])
def test_old_or_corrupted_traces_cannot_be_relabelled_as_controlled_train(tokenizer, corruption):
    policy = native(tokenizer)
    step = policy._generate_batch([messages()], web_final_schema(["id", "value"], [URL]))[0]
    if corruption == "source":
        step.thought_logprobs_source = BEHAVIOR_LOGPROB_SOURCE
    elif corruption == "profile":
        step.web_train_thought_budget_close["forced_position"] -= 1
    elif corruption == "positions":
        step.forced_thought_positions = []
    elif corruption == "token":
        step.completion_ids[3] -= 1
    else:
        step.thought_logprobs[3] = -.5
    result = EpisodeResult("synthetic", "widesearch", "train", [], "completed", 1., {}, [step], [], [])
    with pytest.raises(ValueError):
        group_signal([result, replace(result, reward=0.)], group_size=2, max_steps=2, thought_tokens=4)


@pytest.mark.parametrize("field,value", [("enabled", 1), ("thought_tokens", 4.),
    ("forced_position", 3.), ("forced_logprob", False), ("forced_policy_gradient", 0)])
def test_policy_identity_requires_exact_scalar_types(field, value):
    profile = close.configuration(4)
    profile[field] = value
    with pytest.raises(ValueError):
        close.validate_configuration(profile, thought_tokens=4)
    with pytest.raises(ValueError):
        web_train_history_configuration(CONFIG, thought_policy=profile)


def test_new_completion_subclass_cannot_lose_identity_or_downgrade_to_legacy():
    step = BudgetClosedNativeHFCompletion("", [1], [2, close.END_THINK],
        thought_token_count=2, thought_logprobs=[-.5, -.5],
        thought_logprobs_source=BEHAVIOR_LOGPROB_SOURCE)
    with pytest.raises(ValueError, match="captured policy identity"):
        close.step_fields(step, thought_tokens=4)
    old = NativeHFCompletion("", [1], [close.END_THINK], thought_token_count=1,
        thought_logprobs=[0.], thought_logprobs_source=close.SOURCE)
    with pytest.raises(ValueError, match="captured policy identity"):
        close.step_fields(old, thought_tokens=1)


def test_real_preflight_metadata_binds_controlled_policy_and_rejects_old_source(tokenizer):
    traces = []
    policy = native(tokenizer, sink=traces.append)
    policy.max_prompt_tokens = 1
    result = policy._generate_batch([messages()], web_final_schema(["id", "value"], [URL]), ["web"])[0]
    assert isinstance(result, NativeHFPromptBudgetExceeded)
    assert result.preflight[close.FIELD] == close.configuration(4)
    assert result.preflight["thought_logprobs_source"] == close.SOURCE
    assert result.preflight["generation_submitted"] is False
    assert policy.model.calls == []
    cuad = policy._generate_batch([messages()], cuad_final_schema(), ["cuad"])[0]
    assert isinstance(cuad, NativeHFPromptBudgetExceeded)
    assert close.FIELD not in cuad.preflight
    assert cuad.preflight["thought_logprobs_source"] == BEHAVIOR_LOGPROB_SOURCE


def context_episode(tokenizer, *, initial=False):
    traces = []
    action = json.dumps({"plan": "Read source", "checks": "Check evidence",
        "tool": "web_search", "arguments": {"query": "invented source", "count": 8}})
    policy = native(tokenizer, finals=[final_ids(tokenizer, text=action)], sink=traces.append)
    task = web.public_web_task(public_row())
    calls = []
    class Bound:
        async def generate_structured(self, request, **kwargs):
            calls.append(deepcopy(request))
            if len(calls) == 1:
                policy.max_prompt_tokens = len(_chat_tokens(tokenizer, request)) + (-1 if initial else 16)
            return await policy.bind_task(task.id).generate_structured(request, **kwargs)
    async def run():
        try:
            episode_runner = runner()
            episode_runner.max_new_tokens = policy.thought_tokens + policy.json_tokens
            episode_runner.evaluation_temperature = 1.
            return await episode_runner.run(task, Bound(), evaluation=True)
        finally:
            await policy.close()
    return asyncio.run(run()), traces, calls


@pytest.mark.parametrize("initial", [False, True])
def test_actual_context_failure_restores_only_real_controlled_prefix(tokenizer, initial):
    failed, traces, calls = context_episode(tokenizer, initial=initial)
    assert failed.status == "context_budget_failure" and failed.reward == 0
    preflight = failed.metrics["prompt_history_failure"]["native_hf_preflight"]
    assert preflight[close.FIELD] == close.configuration(4)
    assert preflight["thought_logprobs_source"] == close.SOURCE
    assert len(calls) == (1 if initial else 2)
    sibling = replace(failed, status="completed", reward=1., metrics={})
    signal = group_signal([sibling, failed], group_size=2, max_steps=3, thought_tokens=4)
    if initial:
        assert signal.discarded_reason and failed.steps == [] and len(traces) == 1
        return
    assert signal.discarded_reason is None and len(failed.steps) == 1
    assert signal.segments[1][0].forced_thought_positions == (3,)
    serialized = json.loads(json.dumps(failed.to_dict()))
    serialized["steps"] = [BudgetClosedNativeHFCompletion(**step) for step in serialized["steps"]]
    restored = EpisodeResult(**serialized)
    assert group_signal([sibling, restored], group_size=2, max_steps=3, thought_tokens=4).discarded_reason
    restored = restore_verified_native_context(restored, traces=traces, tokenizer=tokenizer)
    assert group_signal([sibling, restored], group_size=2, max_steps=3, thought_tokens=4) == signal
    for field, value in ((close.POSITIONS, []), (close.POSITIONS, [3.]),
                         (close.FIELD, dict(close.configuration(4), enabled=1))):
        corrupted = deepcopy(traces)
        corrupted[0][field] = value
        with pytest.raises(ValueError):
            restore_verified_native_context(restored, traces=corrupted, tokenizer=tokenizer)


def test_actual_tiny_lora_checkpoint_replay_matches_direct_projected_reference_gradients():
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    from slime_plugins.dcc.hf_chunked_logprobs import completion_logprobs_chunked
    from slime_plugins.dcc.reference_kl import frozen_base_logprobs, thought_reference_kl_loss
    torch.manual_seed(337)
    model = get_peft_model(Qwen3MoeForCausalLM(Qwen3MoeConfig(
        vocab_size=151670, hidden_size=8, intermediate_size=12, moe_intermediate_size=4,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        num_experts=4, num_experts_per_tok=2, attention_dropout=0., max_position_embeddings=64)),
        LoraConfig(r=2, lora_alpha=4, lora_dropout=0.,
            target_modules=["q_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    with torch.no_grad():
        for name, value in model.named_parameters():
            if "lora_B" in name:
                value.normal_(0, .04)
    direct = deepcopy(model)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    prompt, thought = [2, 3, 4], [5, 6, close.END_THINK]
    def raw(value, prompt_ids, completion_ids):
        return completion_logprobs_chunked(value, prompt_ids, completion_ids, head_chunk_tokens=2)
    with torch.no_grad():
        old = raw(model, prompt, thought).tolist()
    old[-1] = 0.  # Synthetic analytical denominator; never inserted in a live trace.
    segment = ThoughtSegment(prompt, thought, old, close.configuration(3), (2,))
    controlled = close.scorer_for(segment, raw)
    reference = frozen_base_logprobs(model, controlled, prompt, thought)
    assert reference[-1] == 0 and not reference.requires_grad
    assert model.get_model_status().enabled is True
    report = load_trainer().replay_group(model,
        GroupSignal([[segment], []], [-.7, .7], [0., 1.], .5), raw,
        group_size=2, max_steps=2, thought_tokens=3, clip=.2,
        replay_diagnostic="full", reference_kl_beta=.1,
        reference_base_model="synthetic-random-tiny-Qwen3Moe")
    with direct.disable_adapter(), torch.no_grad():
        expected_reference = controlled(direct, prompt, thought)
    direct.train()
    current = controlled(direct, prompt, thought)
    current.retain_grad()
    mask = close.policy_mask(segment, current)
    expected = thought_policy_loss(current, torch.tensor(old), -.7,
        group_size=2, max_steps=2, thought_tokens=3, clip=.2, policy_mask=mask)
    expected = expected + .1 * thought_reference_kl_loss(current, expected_reference,
        group_size=2, max_steps=2, thought_tokens=3)
    expected.backward()
    assert current[-1] == expected_reference[-1] == 0
    assert current.grad[-1] == 0
    assert report["objective_loss"] == pytest.approx(expected.item(), abs=1e-7)
    direct_parameters = dict(direct.named_parameters())
    nonzero = 0
    for name, value in model.named_parameters():
        if "lora_" in name:
            assert value.grad is not None
            torch.testing.assert_close(value.grad, direct_parameters[name].grad, rtol=2e-5, atol=1e-7)
            nonzero += int(bool(value.grad.abs().sum() > 0))
        else:
            assert value.grad is None and not value.requires_grad
    assert nonzero > 0 and model.get_model_status().enabled is True
    assert not torch.cuda.is_initialized()


def test_cancelled_late_native_completion_preserves_actual_controlled_ids_and_owner(tokenizer):
    entered, release = Event(), Event()
    traces, owners = [], []
    policy = native(tokenizer, sink=traces.append)
    original = policy.model.generate
    def delayed(**kwargs):
        if close.END_THINK in _call_generation_config(kwargs).eos_token_id:
            entered.set()
            assert release.wait(5)
        return original(**kwargs)
    policy.model.generate = delayed
    owner = web.OwnedEpisodePolicy(policy.bind_task("synthetic-cancel"), task_id="synthetic-cancel",
        iteration=1, rank=0, sample=0, sink=owners.append)
    async def run():
        pending = asyncio.create_task(owner.generate_structured(messages(),
            json_schema=web_final_schema(["id", "value"], [URL]), max_new_tokens=260, temperature=1.))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            pending.cancel()
            await asyncio.sleep(0)
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await pending
        finally:
            release.set()
            await policy.close()
    asyncio.run(run())
    assert len(owners) == len(traces) == 1
    row, trace = owners[0], traces[0]
    assert row["delivery"] == "cancelled" and row["request_status"] == "native"
    assert row[close.FIELD] == trace[close.FIELD] == close.configuration(4)
    assert row[close.POSITIONS] == trace[close.POSITIONS] == [3]
    assert row["completion_ids"][:4] == trace["thought_ids"] == [10065, 10066, 10067, close.END_THINK]
    assert row["thought_logprobs"][-1] == trace["thought_logprobs"][-1] == 0
    assert row["trace_id"] == trace["trace_id"] and policy.worker is None
