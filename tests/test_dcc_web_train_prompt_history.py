"""Invented CPU requests through native sampling, ownership and TRAIN replay.

These controls do not represent a model experiment or benchmark evaluation.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("xgrammar")

from slime_plugins.dcc import online_web as web
from slime_plugins.dcc.cuad_structured import cuad_final_schema
from slime_plugins.dcc.episode import EpisodeResult
from slime_plugins.dcc.hf_native_policy import END_THINK, NativeHFCompletion, NativeHFPolicy
from slime_plugins.dcc.hf_policy import _chat_tokens
from slime_plugins.dcc.native_hf_context import (NativeHFPromptBudgetExceeded, PREFLIGHT_PROTOCOL,
    context_handling_configuration, restore_verified_native_context)
from slime_plugins.dcc.online_grpo import group_signal
from slime_plugins.dcc.prompt_history import PromptHistoryConfig
from slime_plugins.dcc.web_structured import web_final_schema
from slime_plugins.dcc.web_train_prompt_history import (FAILURE_FIELD, FIELDS,
    web_train_history_budgets, web_train_history_configuration, web_train_history_plan)
from test_dcc_hf_native_policy import (ScriptedModel, _call_generation_config, final_ids,
    messages, tokenizer)
from test_dcc_online_grpo import load_trainer
from test_dcc_online_web import COMMIT, FixtureTools, public_row, runner
from test_dcc_web_hf_prompt_history import (CONFIG, URL, budgets, history_messages,
    normalized, tiny_fast_tokenizer)


class SampledCPUModel(ScriptedModel):
    """Sample actual IDs from invented raw logits via the native score tracker."""
    def generate(self, **kwargs):
        config = _call_generation_config(kwargs)
        if END_THINK not in config.eos_token_id:
            return super().generate(**kwargs)
        self.calls.append({**kwargs, "generation_config": config})
        ids = kwargs["input_ids"]
        for token in (10077, END_THINK):
            scores = torch.full((len(ids), self.config.vocab_size), -20.)
            scores[:, token] = 0.
            for processor in kwargs["logits_processor"]:
                scores = processor(ids, scores)
            # There is no forced-close generation option. This is the fixture's
            # actual model distribution, observed before multinomial sampling.
            sampled = torch.multinomial(scores.softmax(-1), 1,
                generator=torch.Generator().manual_seed(token))
            assert sampled[:, 0].tolist() == [token] * len(ids)
            ids = torch.cat((ids, sampled), dim=1)
        return ids


_MISSING = object()


def native(tokenizer, *, config=CONFIG, sink=None, finals=None, **kwargs):
    model = SampledCPUModel([], finals or [final_ids(tokenizer, text=json.dumps({
        "answer": [{"id": "a", "value": "8", "citations": [URL]}]}))])
    model.config = SimpleNamespace(vocab_size=151670, max_position_embeddings=10000)
    limits = budgets(tokenizer, history_messages())
    options = {} if config is _MISSING else {"web_train_prompt_history_config": config}
    value = NativeHFPolicy(model, tokenizer, **limits, capture_thought_logprobs=True,
        trace_sink=sink, **options, **kwargs)
    return value, limits


def verification_kwargs(policy):
    return dict(eos_ids=policy.eos_ids, tokenizer=policy.tokenizer, iteration=1, rank=0,
        web_train_prompt_history_config=policy.web_train_prompt_history_config,
        history_budgets=web_train_history_budgets(policy))


def test_missing_none_and_disabled_preserve_native_records_and_configuration(tokenizer):
    records = []
    for config in (_MISSING, None, PromptHistoryConfig()):
        traces = []
        policy, _ = native(tokenizer, config=config, sink=traces.append)
        completion = policy._generate_batch([history_messages(count=1)],
            web_final_schema(["id", "value"], [URL]), ["default"])[0]
        assert isinstance(completion, NativeHFCompletion)
        assert all(key not in policy.provenance for key in FIELDS)
        assert all(key not in traces[0] for key in FIELDS)
        assert all(key not in asdict(completion) for key in FIELDS)
        records.append((normalized(asdict(completion)), normalized(traces), policy.provenance,
            [call["generation_config"].to_dict() for call in policy.model.calls]))
    assert records[0] == records[1] == records[2]
    train = load_trainer()
    args = SimpleNamespace(model="invented", source_commit=COMMIT, group_size=2,
        max_steps=3, thought_tokens=4, enable_online_web=True,
        cuad_draws_per_cycle=1, web_draws_per_cycle=1)
    baseline = train.configuration(args, [], [], 1)
    for off in (None, False):
        args.web_train_prompt_history_whole_turns = off
        args.web_train_prompt_history_keep_recent_turns = None
        assert train.configuration(args, [], [], 1) == baseline
    args.web_train_prompt_history_whole_turns = True
    args.web_train_prompt_history_keep_recent_turns = 2
    enabled = train.configuration(args, [], [], 1)
    assert enabled["native_web_train_prompt_history"] == web_train_history_configuration(CONFIG)
    assert enabled["objective_denominator"] == baseline["objective_denominator"] == 24
    assert enabled["context_failure_handling"] == baseline["context_failure_handling"]


@pytest.mark.parametrize("extra,match", [
    (["--web-train-prompt-history-whole-turns"], "requires --enable-online-web"),
    (["--web-train-prompt-history-keep-recent-turns", "2"], "requires --web-train-prompt-history-whole-turns"),
    (["--web-train-prompt-history-whole-turns", "--web-train-prompt-history-keep-recent-turns", "0"], "positive integer"),
])
def test_cli_rejects_invalid_train_history_before_source_data_or_cuda(monkeypatch, capsys, extra, match):
    train = load_trainer()
    def forbidden(*args, **kwargs):
        raise AssertionError("No source, data, model or CUDA work is allowed")
    for attribute in ("load_tasks", "validate_source", "load_policy_model"):
        monkeypatch.setattr(train, attribute, forbidden)
    monkeypatch.setattr(torch.cuda, "set_device", forbidden)
    with pytest.raises(SystemExit) as error:
        train.main(["--model", "invented", "--train-data", "absent", "--output-dir", "unused",
            "--source-commit", COMMIT, *extra])
    assert error.value.code == 2 and match in capsys.readouterr().err
    with pytest.raises(ValueError, match="requires --enable-online-web"):
        asyncio.run(train.train(SimpleNamespace(web_train_prompt_history_whole_turns=True), [], [], 1))


@pytest.mark.parametrize("options", [dict(allow_truncation=True), dict(joint_json_policy=True),
    dict(web_thought_budget_close=True), dict(prompt_history_config=CONFIG)])
def test_train_history_cannot_relax_original_decoding_restrictions(tokenizer, options):
    with pytest.raises(ValueError, match="Web TRAIN history|Web evaluation-only"):
        native(tokenizer, **options)


def test_train_history_requires_actual_raw_capture_before_model_access(tokenizer):
    for options in (dict(capture_thought_logprobs=False),
                    dict(capture_thought_logprobs=True, sampling_profile="greedy")):
        with pytest.raises(ValueError, match="requires raw behavior capture"):
            NativeHFPolicy(None, tokenizer, web_train_prompt_history_config=CONFIG, **options)


def test_real_local_tokenizer_recomputes_complete_whole_turns(tiny_fast_tokenizer):
    t = tiny_fast_tokenizer
    original = history_messages()
    saved = deepcopy(original)
    plan = web_train_history_plan(t, original, config=CONFIG, **budgets(t, original))
    assert original == saved and plan.report["compacted"]
    assert plan.messages[:2] == original[:2] and plan.messages[-5:] == original[-5:]
    assert plan.prompt_ids == _chat_tokens(t, plan.messages)
    assert plan.report["omitted_observed_urls"]
    assert not torch.cuda.is_initialized()


def test_mixed_web_failure_cuad_web_retains_actual_sampling_and_selected_alignment(tokenizer):
    good_a, bad, good_b = history_messages(), history_messages(), history_messages()
    good_b[1]["content"] += " Distinct request B."
    bad[-2]["content"] = 'Tool observation (source material):\n' + json.dumps({"url": URL, "text": "X" * 15000})
    original = deepcopy([good_a, bad, messages(), good_b])
    web_schema = web_final_schema(["id", "value"], [URL])
    web_text = json.dumps({"answer": [{"id": "a", "value": "8", "citations": [URL]}]})
    finals = [final_ids(tokenizer, text=web_text), final_ids(tokenizer), final_ids(tokenizer, text=web_text)]
    traces = []
    policy, limits = native(tokenizer, sink=traces.append, finals=finals, mixed_schema_batching=True)
    completions = policy._generate_batch(original, [web_schema, web_schema, cuad_final_schema(), web_schema],
        ["A", "bad", "CUAD", "B"])
    failure = completions[1]
    assert isinstance(failure, NativeHFPromptBudgetExceeded)
    assert failure.reason == "full_context_exceeds_native_HF_limit"
    assert failure.original_prompt_ids == _chat_tokens(tokenizer, bad)
    assert failure.diagnostics == dict(original_prompt_tokens=len(failure.original_prompt_ids),
        minimum_prompt_tokens=len(failure.original_prompt_ids), effective_prompt_limit=limits["max_prompt_tokens"])
    assert traces[0] == failure.preflight and traces[0]["protocol"] == PREFLIGHT_PROTOCOL
    assert not any(key in traces[0] for key in FIELDS)
    proof = getattr(failure, FAILURE_FIELD)
    assert proof["prompt_history"]["minimum_prompt_tokens"] < len(failure.original_prompt_ids)
    assert [trace["task_id"] for trace in traces] == ["bad", "A", "CUAD", "B"]
    for request, completion, trace in zip([good_a, good_b], [completions[0], completions[3]], [traces[1], traces[3]], strict=True):
        plan = web_train_history_plan(tokenizer, request, config=CONFIG, **limits)
        assert plan.report["compacted"]
        assert completion.prompt_ids == trace["prompt_ids"] == plan.prompt_ids
        assert trace["prompt_history_original_messages"] == request
        assert trace["prompt_history"] == plan.report
        assert completion.completion_ids[:2] == [10077, END_THINK]
        assert len(completion.thought_logprobs) == completion.thought_token_count == 2
        expected = []
        for token in (10077, END_THINK):
            # Retain the real vocabulary position: FP32 reductions can differ
            # slightly when the singleton high logit moves between positions.
            logits = torch.full((3, 151670), -20.)
            logits[:, token] = 0.
            expected.append(-logits.logsumexp(-1)[0].item())
        assert completion.thought_logprobs == pytest.approx(expected, abs=1e-6)
        assert trace["thought_logprobs"] == completion.thought_logprobs
        assert completion.thought_logprobs_source == web.BEHAVIOR_SCORES
    assert all(key not in traces[2] for key in FIELDS)
    assert completions[2].prompt_ids == _chat_tokens(tokenizer, original[2])
    baseline, _ = native(tokenizer, config=None, finals=[final_ids(tokenizer)])
    cuad = baseline._generate_batch([original[2]], cuad_final_schema(), ["CUAD"])[0]
    assert normalized(asdict(completions[2])) == normalized(asdict(cuad))
    assert original == [good_a, bad, messages(), good_b]
    assert policy.model.calls[0]["generation_config"].forced_eos_token_id is None


def compacted_group(tmp_path, tokenizer):
    task = web.public_web_task(public_row())
    journals = web.PublicGroupJournals(tmp_path / "group", task=task, group_size=2,
        iteration=1, rank=0, source_commit=COMMIT)
    text = json.dumps({"answer": [{"id": "a", "value": "8", "citations": [URL]}]})
    policy, _ = native(tokenizer, sink=lambda row: journals.enqueue("native_traces", row),
        finals=[final_ids(tokenizer, text=text)] * 2)
    owners = [web.OwnedEpisodePolicy(policy.bind_task(task.id), task_id=task.id,
        iteration=1, rank=0, sample=sample,
        sink=lambda row: journals.enqueue("request_owners", row)) for sample in range(2)]
    async def run():
        try:
            return await asyncio.gather(*(owner.generate_structured(history_messages(),
                json_schema=web_final_schema(["id", "value"], [URL]),
                max_new_tokens=260, temperature=1.) for owner in owners))
        finally:
            await policy.close()
    completions = asyncio.run(run())
    results = [EpisodeResult(task.id, "widesearch", "train", [], "completed", 0., {}, [step], [], [])
               for step in completions]
    group = journals.close_generation(results, policy=policy)
    owners = web._read_jsonl(journals.directory / "request_owners.jsonl")
    traces = web._read_jsonl(journals.directory / "native_traces.jsonl")
    return group, policy, owners, traces


def test_closed_compacted_owners_replay_actual_prompts_and_reject_tampering(tmp_path, tokenizer):
    group, policy, owners, traces = compacted_group(tmp_path, tokenizer)
    assert group.closure["native_ownership_verified"]
    assert all(trace["prompt_history"]["compacted"] for trace in traces)
    # Exercise the optimizer replay entrypoint. JSON is never given to scorer;
    # actual raw behavior remains old even when teacher forcing differs.
    scored = [replace(group.results[0], reward=1.), group.results[1]]
    signal = group_signal(scored, group_size=2, max_steps=3, thought_tokens=4)
    seen = []
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(-.1)
    def scorer(model, prompt, completion):
        seen.append((list(prompt), list(completion)))
        return (model.weight * len(seen)).reshape(1).expand(len(completion))
    report = load_trainer().replay_group(model, signal, scorer,
        group_size=2, max_steps=3, thought_tokens=4, clip=.2, replay_diagnostic="off")
    assert seen == [(step.prompt_ids, step.completion_ids[:step.thought_token_count])
                    for result in scored for step in result.steps]
    assert report["scored_tokens"] == 4 and report["old_logprob_source"] == "actual-sampled-thought-behavior"
    old = torch.tensor(scored[0].steps[0].thought_logprobs)
    expected_gradient = -sum(((torch.tensor(-.1 * (i + 1)) - old).exp()
        * signal.advantages[i] * (i + 1)).sum() for i in range(2)) / 24
    assert model.weight.grad.item() == pytest.approx(expected_gradient.item(), abs=1e-7)
    assert signal.segments[0][0].prompt_ids == list(group.final_prompt_ids[0])
    # A report cannot grant ownership by declaring that it was enabled.
    for damage in ("original", "omitted", "report_type", "declaration", "prompt", "behavior", "close", "eos", "flag_off", "budget"):
        changed_owners, changed_traces = deepcopy(owners), deepcopy(traces)
        kwargs = verification_kwargs(policy)
        trace = changed_traces[0]
        if damage == "original":
            trace["prompt_history_original_messages"][1]["content"] += " forged"
        elif damage == "omitted":
            trace["prompt_history"]["omitted_message_indices"] = []
        elif damage == "report_type":
            trace["prompt_history"]["compacted"] = 1
        elif damage == "declaration":
            trace["native_web_train_prompt_history"]["training_only"] = False
        elif damage == "prompt":
            trace["prompt_ids"][0] += 1
        elif damage == "behavior":
            trace["thought_logprobs"][0] = -9.
        elif damage == "close":
            trace["thought_ids"][-1] = 10065
        elif damage == "eos":
            trace["json_ids"][-1] = 10065
        elif damage == "flag_off":
            kwargs.pop("history_budgets")
            kwargs.pop("web_train_prompt_history_config")
        else:
            kwargs["history_budgets"]["max_prompt_tokens"] += 1
        with pytest.raises(ValueError):
            web.verify_owned_native_group(group.task, group.results, changed_owners, changed_traces, **kwargs)


class RunnerOverflowTools(FixtureTools):
    async def web_search(self, query, count=8):
        result = await super().web_search(query, count=count)
        if "oversized" in query:
            result["results"][0]["snippet"] = "invented unread search result " * 1000
        return result


class RunnerNativeFixture:
    """Shared runner + actual native preflight, invented successful decodes."""
    def __init__(self, tokenizer, *, initial=False):
        self.tokenizer, self.eos_ids, self.worker = tokenizer, [tokenizer.eos_token_id], None
        self.trace_sink, self.serial, self.bound = None, 0, 0
        self.web_train_prompt_history_config = CONFIG
        self.thought_tokens, self.json_tokens = 4, 256
        self.max_prompt_tokens, self.max_model_len = None, 100000
        self.initial = initial
        self.request_bounds = []

    def bind_task(self, task_id, rank=0):
        sample, parent = self.bound, self
        self.bound += 1
        class Bound:
            turn = 0
            async def generate_structured(self, messages, *, json_schema, **kwargs):
                original_prompt_tokens = len(_chat_tokens(parent.tokenizer, messages))
                if parent.max_prompt_tokens is None:
                    parent.max_prompt_tokens = original_prompt_tokens + (-1 if parent.initial else 3000)
                # Independent fixture admission conditions run before the
                # compactor/native request. The healthy sibling never needs
                # compaction; only the invented huge observation exceeds it.
                assert parent.max_prompt_tokens + parent.thought_tokens + parent.json_tokens <= parent.max_model_len
                if parent.initial or (sample == 1 and self.turn > 0):
                    assert original_prompt_tokens > parent.max_prompt_tokens
                    if not parent.initial:
                        assert any("invented unread search result " in row["content"] for row in messages)
                else:
                    assert original_prompt_tokens <= parent.max_prompt_tokens
                parent.request_bounds.append(dict(sample=sample, turn=self.turn,
                    original_prompt_tokens=original_prompt_tokens, limit=parent.max_prompt_tokens))
                action = ({"tool": "web_search", "arguments": {"query": "invented oversized source" if sample else "invented source", "count": 8}}
                          if self.turn == 0 else {"tool": "fetch_url", "arguments": {"url": "https://example.com/invented", "start": 0}}
                          if self.turn == 1 else {"answer": [{"id": "a", "value": "8", "citations": ["https://example.com/invented"]}]})
                self.turn += 1
                action = {"plan": "Read source", "checks": "Exact public columns", **action}
                native = object.__new__(NativeHFPolicy)
                for name in ("tokenizer", "thought_tokens", "json_tokens", "max_prompt_tokens", "max_model_len",
                             "web_train_prompt_history_config", "trace_sink", "eos_ids"):
                    setattr(native, name, getattr(parent, name))
                native.suffix_ids = NativeHFPolicy.validate_tokenizer(parent.tokenizer)["native_suffix_ids"]
                native.allow_truncation, native.mixed_schema_batching = False, False
                native.sampling_profile, native.capture_thought_logprobs = "raw", True
                def decode(prompts, schema, task_ids, *, history_metadata):
                    parent.serial += 1
                    text = json.dumps(action)
                    thought, final = [10077, END_THINK], final_ids(parent.tokenizer, text=text)
                    assert len(final) <= parent.json_tokens
                    trace_id = f"synthetic-training-native-{parent.serial}"
                    completion = NativeHFCompletion(text, prompts[0], thought + final,
                        thought_token_count=2, phase_lengths=(2, len(final)), trace_id=trace_id,
                        thought_logprobs=[-.5, -.25], thought_logprobs_source=web.BEHAVIOR_SCORES)
                    parent.trace_sink(dict(protocol=web.HF_PROTOCOL, trace_id=trace_id, task_id=task_id,
                        status="completed", prompt_ids=prompts[0], thought_ids=thought, json_ids=final,
                        thought_token_count=2, final_text=text, thought_sampling_profile="raw",
                        thought_logprobs=completion.thought_logprobs, thought_logprobs_source=web.BEHAVIOR_SCORES,
                        json_public_schema=deepcopy(schema), **history_metadata[0]))
                    return [completion]
                native._generate_encoded_batch = decode
                result = native._generate_batch([messages], json_schema, [task_id])[0]
                if isinstance(result, BaseException):
                    raise result
                return result
        return Bound()

    async def close(self):
        self.worker = None


@pytest.mark.parametrize("initial", [False, True])
def test_no_fit_preserves_full_preflight_context_prefix_and_serialized_ownership(tmp_path, tokenizer, initial):
    policy = RunnerNativeFixture(tokenizer, initial=initial)
    task = web.public_web_task(public_row())
    real_runner = runner()
    real_runner.max_new_tokens, real_runner.evaluation_temperature = 260, 1.
    real_runner.web_tools_factory = RunnerOverflowTools
    group = asyncio.run(web.collect_public_group(real_runner, task, policy, group_size=2,
        rank=0, iteration=1, directory=tmp_path / "group", source_commit=COMMIT))
    print(json.dumps(dict(fixture_original_request_bounds=policy.request_bounds,
        thought_tokens=policy.thought_tokens, json_tokens=policy.json_tokens,
        max_model_len=policy.max_model_len)))
    owners = web._read_jsonl(tmp_path / "group/request_owners.jsonl")
    traces = web._read_jsonl(tmp_path / "group/native_traces.jsonl")
    errors = [row for row in owners if row["request_status"] == "error"]
    assert len(errors) == (2 if initial else 1)
    for row in errors:
        old, proof = row["native_hf_preflight"], row[FAILURE_FIELD]
        assert not any(key in old for key in FIELDS)
        assert old in traces and old["prompt_ids"] == _chat_tokens(tokenizer, row["messages"])
        assert proof["generation_submitted"] is False
        failure = group.results[row["sample"]].metrics["prompt_history_failure"]
        assert failure["reason"] == "full_context_exceeds_native_HF_limit"
        assert failure["diagnostics"] == dict(original_prompt_tokens=len(old["prompt_ids"]),
            minimum_prompt_tokens=len(old["prompt_ids"]), effective_prompt_limit=policy.max_prompt_tokens)
    assert context_handling_configuration()["version"] == "hf-native-context-sampled-prefix-v2"
    scored = ([group.results[0], group.results[1]] if initial
              else [replace(group.results[0], reward=1.), group.results[1]])
    signal = group_signal(scored, group_size=2, max_steps=3, thought_tokens=4)
    assert bool(signal.discarded_reason) is initial
    if not initial:
        assert scored[0].status == "completed" and len(scored[0].steps) == 3
        assert scored[1].status == "context_budget_failure" and len(scored[1].steps) == 1
        assert signal.segments[1][0].completion_ids == [10077, END_THINK]
        assert signal.segments[1][0].thought_logprobs == [-.5, -.25]
    restored = []
    for row in web._read_jsonl(tmp_path / "group/public_outcomes.jsonl"):
        row.pop("sample")
        row["steps"] = [NativeHFCompletion(**step) for step in row["steps"]]
        result = EpisodeResult(**row)
        if result.status == "context_budget_failure":
            result = restore_verified_native_context(result, traces=traces, tokenizer=tokenizer)
        restored.append(result)
    web.verify_owned_native_group(task, restored, owners, traces, **verification_kwargs(policy))
    if not initial:
        restored[0] = replace(restored[0], reward=1.)
    assert group_signal(restored, group_size=2, max_steps=3, thought_tokens=4) == signal
    for damage in ("missing", "reason", "report", "original", "flag_off", "budget"):
        changed = deepcopy(owners)
        row = next(row for row in changed if row["request_status"] == "error")
        kwargs = verification_kwargs(policy)
        if damage == "missing":
            row.pop(FAILURE_FIELD)
        elif damage == "reason":
            row[FAILURE_FIELD]["reason"] = "invented"
        elif damage == "report":
            row[FAILURE_FIELD]["prompt_history"]["original_prompt_tokens"] += 1
        elif damage == "original":
            row[FAILURE_FIELD]["prompt_history_original_messages"][1]["content"] += "forged"
        elif damage == "flag_off":
            kwargs.pop("web_train_prompt_history_config")
            kwargs.pop("history_budgets")
        else:
            kwargs["history_budgets"]["thought_tokens"] += 1
        with pytest.raises(ValueError):
            web.verify_owned_native_group(task, group.results, changed, traces, **kwargs)
