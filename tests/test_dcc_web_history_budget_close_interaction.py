"""CPU tiny-HF interaction checks; invented inputs, no training/quality claim."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import asdict
import json

import pytest
import torch
from transformers import GenerationConfig, Qwen3MoeConfig, Qwen3MoeForCausalLM

pytest.importorskip("xgrammar")

from slime_plugins.dcc.data import Task
from slime_plugins.dcc.hf_compute import precision_observation
from slime_plugins.dcc.hf_native_policy import (END_THINK, NATIVE_SUFFIX, NativeHFCompletion,
    NativeHFPolicy, WEB_THOUGHT_BUDGET_CLOSE_PROTOCOL, web_thought_budget_close_configuration)
from slime_plugins.dcc.hf_policy import _chat_tokens
from slime_plugins.dcc.prompt_history import PromptHistoryConfig
from slime_plugins.dcc.web_hf_evaluation import OwnedEvaluationPolicy, verify_native_ownership
from slime_plugins.dcc.web_hf_prompt_history import FIELDS, native_web_history_plan
from slime_plugins.dcc.web_structured import web_final_schema
from test_dcc_hf_native_policy import _call_generation_config, web_dump
from test_dcc_online_web_evaluate import arguments, evaluator
from test_dcc_web_hf_prompt_history import (URL, budgets, history_messages, normalized,
    tiny_fast_tokenizer)

CONFIG = PromptHistoryConfig(True, 2)
ANSWER = {"answer": [{"name": "Alpha", "citations": [URL]}]}


def tiny_policy(tokenizer, *, thought, limit, traces, **options):
    """Control LM-head preferences while retaining actual HF generate/grammar."""
    model = Qwen3MoeForCausalLM(Qwen3MoeConfig(vocab_size=len(tokenizer), hidden_size=8,
        intermediate_size=12, moe_intermediate_size=4, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4, num_experts=4,
        num_experts_per_tok=2, attention_dropout=0., max_position_embeddings=10000)).eval()
    model.generation_config = GenerationConfig(eos_token_id=[151645, 151643])
    final = tokenizer.encode(web_dump(ANSWER), add_special_tokens=False) + [tokenizer.eos_token_id]
    state, calls = {}, []

    def fixture_logits(module, args, output):
        target = state["tokens"][min(state["cursor"], len(state["tokens"]) - 1)]
        state["cursor"] += 1
        logits = torch.full_like(output, -10000.)
        logits[..., target] = 10000.
        return logits

    class ObserveLastThought:
        def __init__(self, width):
            self.width = width

        def __call__(self, ids, scores):
            if ids.shape[1] == self.width + 3:
                state["last_support"] = (torch.isfinite(scores).sum().item(),
                    (scores[0, END_THINK] - scores[0].logsumexp(-1)).item())
            return scores

    model.lm_head.register_forward_hook(fixture_logits)
    generate = model.generate

    def actual_generate(**kwargs):
        config = _call_generation_config(kwargs)
        is_thought = END_THINK in config.eos_token_id
        state.update(tokens=thought if is_thought else final, cursor=0)
        prompt = kwargs["input_ids"].tolist()
        if is_thought:
            kwargs["logits_processor"].append(ObserveLastThought(len(prompt[0])))
        result = generate(**kwargs)
        calls.append(dict(phase="thought" if is_thought else "json", prompt=prompt,
            ids=result[:, kwargs["input_ids"].shape[1]:].tolist(),
            forced_eos=config.forced_eos_token_id))
        return result

    model.generate = actual_generate
    policy = NativeHFPolicy(model, tokenizer, thought_tokens=4, json_tokens=256,
        max_prompt_tokens=limit, trace_sink=traces.append, **options)
    return policy, calls, state


def test_real_compact_prompt_forced_close_and_closed_score_only_replay(
        tiny_fast_tokenizer, evaluator, tmp_path, monkeypatch):
    tokenizer, ev = tiny_fast_tokenizer, evaluator
    monkeypatch.setattr(ev, "NativeHFPolicy", NativeHFPolicy)
    messages = history_messages()
    original = deepcopy(messages)
    limits = budgets(tokenizer, messages)
    traces, owners = [], []
    thought = tokenizer.encode("ABCD", add_special_tokens=False)
    assert len(thought) == 4
    policy, calls, state = tiny_policy(tokenizer, thought=thought,
        limit=limits["max_prompt_tokens"], traces=traces,
        prompt_history_config=CONFIG, web_thought_budget_close=True, sampling_profile="thinking")
    schema = web_final_schema(["name"], [URL])
    task = Task("cpu-TRAIN-curriculum-combination", "widesearch", "validation",
        "Invented CPU interaction input.", metadata={"evaluation": {"required": ["name"]}})

    async def run():
        owner = OwnedEvaluationPolicy(policy.bind_task(task.id, 0), task_id=task.id,
            rank=0, sink=owners.append)
        try:
            return await owner.generate_structured(messages, json_schema=schema,
                max_new_tokens=260, temperature=1.)
        finally:
            await policy.close()

    completion = asyncio.run(run())
    assert isinstance(completion, NativeHFCompletion)
    plan = native_web_history_plan(tokenizer, messages, config=CONFIG, **limits)
    suffix = tokenizer.encode(NATIVE_SUFFIX, add_special_tokens=False)
    assert plan.report["compacted"] and plan.report["omitted_message_indices"]
    assert messages == original and plan.messages[:2] == messages[:2]
    assert plan.messages[-5:] == messages[-5:]
    assert completion.prompt_ids == plan.prompt_ids == calls[0]["prompt"][0]
    assert completion.prompt_ids != _chat_tokens(tokenizer, messages)
    assert completion.prompt_ids[-len(suffix):] == suffix and END_THINK not in suffix
    actual_thought = thought[:3] + [END_THINK]
    assert calls[0]["ids"] == [actual_thought] and calls[0]["forced_eos"] == END_THINK
    assert state["last_support"] == (1, 0.)
    assert calls[1]["prompt"] == [plan.prompt_ids + actual_thought]
    assert calls[1]["forced_eos"] is None and calls[1]["ids"][0][-1] == tokenizer.eos_token_id
    assert completion.completion_ids == actual_thought + calls[1]["ids"][0]
    assert json.loads(completion.text) == ANSWER
    assert traces[0]["protocol"] == WEB_THOUGHT_BUDGET_CLOSE_PROTOCOL
    assert traces[0]["forced_thought_positions"] == [3]
    assert traces[0]["prompt_history_original_messages"] == original
    assert completion.thought_logprobs is completion.thought_logprobs_source is None

    # Save only local tokenizer/config: score-only must never ask for weights.
    model_dir = tmp_path / "tiny-tokenizer-and-config-only"
    tokenizer.save_pretrained(model_dir)
    policy.model.config.save_pretrained(model_dir)
    assert not list(model_dir.glob("*.safetensors")) and not list(model_dir.glob("*.bin"))
    args = arguments(ev, tmp_path / "closed-combination", model=str(model_dir),
        max_prompt_tokens=limits["max_prompt_tokens"], prompt_history_whole_turns=True,
        prompt_history_keep_recent_turns=2, web_thought_budget_close=True)
    args.output_dir.mkdir()
    configuration = ev.manifest(args, [task])
    runtime = ev.runtime_identity(policy.model, tokenizer, policy)
    stored = json.loads(json.dumps(asdict(completion)))
    prediction = dict(task_id=task.id, benchmark=task.benchmark, split=task.split,
        status="completed", rollout_labels_removed=True, rollout_scored=False,
        steps=[stored], metrics={}, answer=ANSWER["answer"], task_seed=ev.task_seed(args.seed, task.id))
    rows = dict(predictions=[prediction], request_owners=owners, native_traces=traces)
    writers = {}
    for name, records in rows.items():
        writer = ev.OrderedDurableJSONLWriter(args.output_dir / f"{name}_rank0.jsonl")
        writers[name] = writer
        for record in records:
            writer.write_line(json.dumps(record) + "\n")
    ev.close_durable_journals(writers.values())
    declaration = web_thought_budget_close_configuration(4)
    ownership = verify_native_ownership([task], rows["predictions"],
        json.loads(json.dumps(owners)), traces, rank=0, profile=args.profile,
        eos_ids=policy.eos_ids, tokenizer=tokenizer, prompt_history_config=CONFIG,
        history_budgets=limits, web_thought_budget_close=declaration)
    ev.write_json(args.output_dir / "config_rank0.json", configuration)
    ev.write_json(args.output_dir / "rank0.done.json", dict(version=ev.VERSION,
        source_commit=args.source_commit, shard_index=0, tasks=1,
        generation_complete=True, policy_drained=True, journals_closed=True,
        native_token_encoding_verified=True, native_ownership=ownership,
        closed_journals={name: len(records) for name, records in rows.items()},
        rollout_labels_removed=True, rollout_scored=False, runtime=runtime,
        precision=precision_observation(policy.model)))

    def no_model_load(*args, **kwargs):
        pytest.fail("score-only replay attempted to load model weights")
    monkeypatch.setattr(ev, "load_policy_model", no_model_load)
    expected, saved, receipts, *_ = ev.load_closed_run(args, [task])
    assert expected == configuration and saved == [prediction]
    assert receipts[0]["native_ownership"]["all_native_traces_owned"] is True
    assert len(calls) == 2 and not ev.metric_calls
    # Both declarations remain binding on score-only; no regeneration/scoring.
    args.prompt_history_keep_recent_turns = 1
    with pytest.raises(ValueError, match="configuration, precision, runtime or closure differs"):
        ev.load_closed_run(args, [task])
    args.prompt_history_keep_recent_turns = 2
    args.web_thought_budget_close = False
    with pytest.raises(ValueError, match="configuration, precision, runtime or closure differs"):
        ev.load_closed_run(args, [task])
    assert len(calls) == 2 and not ev.metric_calls and not torch.cuda.is_initialized()


def test_combination_preserves_real_early_close_without_extra_tokens(tiny_fast_tokenizer):
    tokenizer = tiny_fast_tokenizer
    messages = history_messages()
    traces = []
    thought = tokenizer.encode("A", add_special_tokens=False) + [END_THINK]
    policy, calls, _ = tiny_policy(tokenizer, thought=thought,
        limit=budgets(tokenizer, messages)["max_prompt_tokens"], traces=traces,
        prompt_history_config=CONFIG, web_thought_budget_close=True)
    completion = policy._generate_batch([messages], web_final_schema(["name"], [URL]))[0]
    assert isinstance(completion, NativeHFCompletion)
    assert calls[0]["ids"] == [thought] and completion.thought_token_count == len(thought)
    assert traces[0]["prompt_history"]["compacted"]
    assert traces[0]["forced_thought_positions"] == []
    assert calls[1]["ids"][0][-1] == tokenizer.eos_token_id
    assert not torch.cuda.is_initialized()


def test_both_switches_default_off_keep_actual_outputs_configs_and_rng(tiny_fast_tokenizer):
    tokenizer, outputs = tiny_fast_tokenizer, []
    messages = history_messages(count=1)
    thought = tokenizer.encode("A", add_special_tokens=False) + [END_THINK]
    for options in ({}, dict(prompt_history_config=PromptHistoryConfig(), web_thought_budget_close=False)):
        traces = []
        policy, calls, _ = tiny_policy(tokenizer, thought=thought, limit=6000,
            traces=traces, **options)
        torch.manual_seed(83)
        completion = policy._generate_batch([messages], web_final_schema(["name"], [URL]))[0]
        assert isinstance(completion, NativeHFCompletion)
        assert not any(key in policy.provenance or key in traces[0] for key in FIELDS)
        assert "web_thought_budget_close" not in policy.provenance
        assert "web_thought_budget_close" not in traces[0] and "forced_thought_positions" not in traces[0]
        assert completion.prompt_ids == _chat_tokens(tokenizer, messages)
        outputs.append((normalized(asdict(completion)), normalized(traces), calls,
            policy.provenance, torch.get_rng_state()))
    assert outputs[0][:-1] == outputs[1][:-1]
    assert torch.equal(outputs[0][-1], outputs[1][-1])
    assert not torch.cuda.is_initialized()
