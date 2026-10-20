"""Synthetic CPU protocol + real tiny Qwen MoE/PEFT gradient controls.

Fixtures are invented only for unit tests; they are never training completions.
Run with CUDA_VISIBLE_DEVICES='' before importing torch/PEFT.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "scripts"),
                str(Path(__file__).resolve().parents[1] / "slime")]

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
from peft import LoraConfig, get_peft_model
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

import dcc_native_sft as SFT
from dcc_native_dpo_data import (
    PREFERENCE_CONTRACT, NativeFinalCandidate, NativeSequence, PreferencePair,
    PublicFinalPrefix, TrainRewardReceipt, best_worst_pair, validate_final_candidate,
)
from dcc_native_dpo import (
    CachedReference, OfflineNativeDPOTrainer, ReferenceIdentity, _sequence_sum,
    backward_pair_low_memory, cache_reference_scores, dpo_loss,
    validate_deterministic_scoring,
)
from slime_plugins.dcc.eval_sampling import EvaluationSampling
from slime_plugins.dcc.native_budget import effective_sampling
from slime_plugins.dcc.native_thinking_eval import native_configuration
from slime_plugins.dcc.hf_policy import completion_logprobs


class Tokenizer:
    eos_token_id = 151645
    chat_template = "synthetic CPU template"
    tags = {"<|im_start|>": 151644, "<|im_end|>": 151645, "<|endoftext|>": 151643,
            "<think>": 151667, "</think>": 151668}
    def __len__(self):
        return 160000
    def encode(self, text, *, add_special_tokens=False):
        return [token for part in re.split('(' + '|'.join(map(re.escape, self.tags)) + ')', text)
                for token in ([self.tags[part]] if part in self.tags else [10000 + ord(char) for char in part])]
    def decode(self, ids, *, skip_special_tokens=False):
        inverse = {value: key for key, value in self.tags.items()}
        return ''.join(inverse[token] if token in inverse else chr(token - 10000)
                       for token in ids if not (skip_special_tokens and token in inverse))
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        text = ''.join('<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'
                       for message in messages)
        text += SFT.NATIVE_SUFFIX if add_generation_prompt else ''
        return self.encode(text) if tokenize else text


def _physical_fixture(*, positive_task=True, empty_answer=True, seed=42):
    """Full synthetic physical trace; positive_task never changes policy input."""
    tokenizer = Tokenizer()
    identity = SFT.NativeIdentity("synthetic-test-derivative", (151643, 151645), "f" * 40,
        1024, 1024 + 8192, seed)
    schema = {"type": "object", "properties": {"answer": {"type": "array", "items": {
        "type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"],
        "additionalProperties": False}}}, "required": ["answer"], "additionalProperties": False}
    messages = [{"role": "system", "content": "Use only this public contract."},
                {"role": "user", "content": "Alpha agrees. Extract qualifying clauses."}]
    public = json.dumps({"answer": [] if empty_answer else [{"text": "Alpha agrees."}]})
    prompt = SFT.encode_messages(tokenizer, messages, identity.max_prompt_tokens, allow_truncation=False)
    thought = tokenizer.encode("Consider visible evidence.</think>")
    final = tokenizer.encode(public) + [151645]
    source_receipt = {"version": "dcc-native-inference-source-v1", "source_commit": "f" * 40,
        "base_source_commit": SFT.BASE_SOURCE_COMMIT, "protocol": SFT.PROTOCOL, "frozen": True}
    phases = []
    for index, (name, ids) in enumerate((("thought", thought), ("json", final))):
        sampling = identity.budget.thought_sampling if index == 0 else EvaluationSampling()
        raw = {"token_ids": ids, "engine_output_text": "synthetic test engine text",
               "finish_reason": "stop", "stop_reason": ids[-1]}
        phase = {"name": name, **raw, "raw_outputs": [deepcopy(raw)],
            "prompt_ids": prompt if index == 0 else prompt + thought,
            "response_prompt_ids": prompt if index == 0 else prompt + thought,
            "response_finished": True, "max_tokens": 4096, "n": 1, "seed": seed,
            "ignore_eos": False, "skip_special_tokens": False, "include_stop_str_in_output": True,
            "structured_outputs": index == 1, "output_kind": "FINAL_ONLY", "sampling": sampling.to_dict(),
            "stop_token_ids": sorted([151643, 151645, 151668]) if index == 0 else [151643, 151645],
            "request_id": f"synthetic-{seed}-{name}", "decoded_text_raw": tokenizer.decode(ids)}
        phases.append(phase)
    trace_id = f"synthetic-trace-{seed}"
    record = {"source": "cuad_train", "benchmark": "cuad", "split": "train", "task_id": "synthetic-train",
        "stage": "final", "messages": messages, "completion": public,
        "grounding_fields": [] if empty_answer else ["Alpha agrees."],
        "metadata": {"category": "Synthetic category", "positive": positive_task}}
    trace = {"protocol": identity.budget.protocol, "source_commit": identity.source_commit,
        "task_id": record["task_id"], "turn_index": 0, "rank": 0, "trace_id": trace_id,
        "native_thinking": native_configuration(source_receipt, budget=identity.budget),
        "status": "completed", "failure_reason": None, "json_complete": True, "json_schema_valid": True,
        "public_text_contains_thought": False, "declared_max_new_tokens": 8192,
        "thought_max_tokens": 4096, "json_max_tokens": 4096, "json_schema": schema,
        "original_prompt_ids": prompt, "phases": phases, "completion_ids": thought + final,
        "observed_completion_tokens": len(thought) + len(final), "final_text": public}
    bundle = {"contract": SFT.CONTRACT, "record": record, "identity": identity.manifest(tokenizer),
        "collector": {"split": "train", "task_input": "public-train-task-without-gold",
            "gold_visible_to_policy": False, "recorded_messages": True, "synthetic_thought": False,
            "inserted_close": False, "inserted_eos": False, "episode_id": f"synthetic-episode-{seed}",
            "turn": 0, "rank": 0, "trace_id": trace_id}, "native_trace": trace}
    frozen_identity = identity.manifest(tokenizer)
    frozen_identity.pop("generation_seed")
    prefix = PublicFinalPrefix("synthetic-prefix", record["task_id"], "cuad", "cuad_train",
        json.dumps(messages), json.dumps(schema), "synthetic-frozen-harness", json.dumps(frozen_identity))
    receipt = TrainRewardReceipt(record["task_id"], trace_id, prefix.prefix_id, prefix.harness_id,
                                 "synthetic-TRAIN-reward-v1", 0.0, True)
    envelope = {"contract": PREFERENCE_CONTRACT, "prefix_id": prefix.prefix_id, "native_bundle": bundle}
    return tokenizer, identity, prefix, receipt, envelope


@pytest.mark.parametrize("positive_task,empty_answer", [(True, True), (False, False)])
def test_wrong_complete_cuad_answers_retain_exact_physical_tokens_without_relaxing_sft(positive_task, empty_answer):
    tokenizer, identity, prefix, reward, envelope = _physical_fixture(
        positive_task=positive_task, empty_answer=empty_answer)
    before = deepcopy(envelope)
    candidate = validate_final_candidate(envelope, tokenizer, prefix=prefix,
                                          expected_identity=identity, reward=reward)
    trace = envelope["native_bundle"]["native_trace"]
    assert candidate.sequence.completion == tuple(trace["completion_ids"])
    assert candidate.sequence.prompt == tuple(trace["original_prompt_ids"])
    assert candidate.sequence.completion[candidate.sequence.phase_lengths[0] - 1] == 151668
    assert candidate.sequence.completion[-1] == 151645
    assert envelope == before
    with pytest.raises(ValueError, match="positive/negative provenance"):
        SFT.validate_native_record(envelope["native_bundle"], tokenizer,
            expected_identity=identity, original_schema=json.loads(prefix.schema_json))


@pytest.mark.parametrize("damage", ["close", "eos", "continuation", "schema", "prompt", "test", "reward_owner", "budget"])
def test_preference_contract_rejects_physical_or_public_state_tampering(damage):
    tokenizer, identity, prefix, reward, envelope = _physical_fixture()
    trace = envelope["native_bundle"]["native_trace"]
    if damage == "close":
        trace["phases"][0]["token_ids"][-1] = 151645
    elif damage == "eos":
        trace["phases"][1]["token_ids"].pop()
    elif damage == "continuation":
        trace["phases"][1]["prompt_ids"].pop()
    elif damage == "schema":
        trace["json_schema"] = {"type": "object"}
    elif damage == "prompt":
        envelope["native_bundle"]["record"]["messages"][1]["content"] = "Different public state"
    elif damage == "test":
        envelope["native_bundle"]["record"]["split"] = "test"
    elif damage == "reward_owner":
        reward = replace(reward, trace_id="unrelated")
    else:
        trace["thought_max_tokens"] = 16384
    with pytest.raises(ValueError):
        validate_final_candidate(envelope, tokenizer, prefix=prefix, expected_identity=identity, reward=reward)


def _candidate(*, seed=1, reward=1.0, prompt=(3, 4, 151667), completion=None, prefix=None):
    prefix = prefix or PublicFinalPrefix("math-prefix", "math-task", "cuad", "cuad_train",
        '[{"role":"user","content":"synthetic public input"}]',
        '{"type":"object","properties":{"answer":{"type":"array"}},"required":["answer"],"additionalProperties":false}',
        "fixed-harness", '{"derivative":"synthetic","budget":"fixed"}')
    completion = completion or ((6, 151668, 7, 8, 151645) if seed == 1 else (9, 10, 151668, 11, 12, 151645))
    close = completion.index(151668) + 1
    return NativeFinalCandidate(prefix, NativeSequence(tuple(prompt), tuple(completion),
        (close, len(completion) - close)), f"trace-{seed}", f"episode-{seed}", seed,
        '{"derivative":"synthetic","budget":"fixed"}', reward, "fixed-TRAIN-reward")


def _pair():
    return PreferencePair(_candidate(), _candidate(seed=2, reward=0.0))


@pytest.mark.parametrize("change", ["task", "schema", "harness", "reward", "prompt", "seed", "identity"])
def test_pairing_rejects_unmatched_context_or_duplicate_sampling(change):
    chosen, rejected = _candidate(), _candidate(seed=2, reward=0.0)
    if change in {"task", "schema", "harness"}:
        field = {"task": "task_id", "schema": "schema_json", "harness": "harness_id", "reward": "reward_version"}[change]
        value = '{"properties":{"answer":{}},"type":"object","required":["answer"],"additionalProperties":false}' if change == "schema" else "different"
        rejected = replace(rejected, prefix=replace(rejected.prefix, **{field: value}))
    elif change == "reward":
        rejected = replace(rejected, reward_version="different")
    elif change == "prompt":
        rejected = replace(rejected, sequence=replace(rejected.sequence, prompt=(5, 4, 151667)))
    elif change == "seed":
        rejected = replace(rejected, generation_seed=1)
    else:
        rejected = replace(rejected, frozen_identity_json="different")
    with pytest.raises(ValueError):
        PreferencePair(chosen, rejected)


def test_ties_are_omitted_and_one_best_worst_pair_is_chosen():
    assert best_worst_pair([_candidate(reward=0), _candidate(seed=2, reward=0)]) is None
    pair = best_worst_pair([_candidate(reward=1), _candidate(seed=2, reward=.5), _candidate(seed=3, reward=0)])
    assert pair.chosen.reward == 1 and pair.rejected.reward == 0
    with pytest.raises(ValueError, match="closure"):
        TrainRewardReceipt("x", "x", "x", "x", "x", 1, False)
    with pytest.raises(ValueError, match="Official WideSearch"):
        replace(pair.chosen.prefix, benchmark="widesearch", source="official")


def _tiny_model(dropout=0):
    torch.manual_seed(27)
    config = Qwen3MoeConfig(vocab_size=151670, hidden_size=8, intermediate_size=12,
        moe_intermediate_size=4, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=4, num_experts=4, num_experts_per_tok=2,
        attention_dropout=0.0)
    model = get_peft_model(Qwen3MoeForCausalLM(config), LoraConfig(r=2, lora_alpha=4,
        lora_dropout=dropout, target_modules=["q_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    with torch.no_grad():
        for name, value in model.named_parameters():
            if "lora_B" in name:
                value.normal_(0, .2)
    assert all(p.device.type == "cpu" for p in model.parameters())
    return model


def _identity(mode="initial_adapter"):
    return ReferenceIdentity("synthetic-derivative", "synthetic-tokenizer", "CPU-FP32-test-only",
                              mode, "synthetic-SFT-snapshot" if mode == "initial_adapter" else None)


def test_sum_objective_matches_standard_dpo_not_token_means():
    pair = _pair()
    reference = CachedReference(pair.key, _identity(), -70., -75.)
    chosen, rejected = torch.tensor([-3., -2., -4., -5., -6.]), torch.tensor([-2., -3., -4., -5., -6., -7.])
    actual = dpo_loss(chosen.sum(), rejected.sum(), reference, beta=.1)
    expected = -torch.nn.functional.logsigmoid(.1 * ((chosen.sum()+70) - (rejected.sum()+75)))
    torch.testing.assert_close(actual, expected)
    assert not torch.isclose(actual, dpo_loss(chosen.mean(), rejected.mean(), reference, beta=.1))


def test_low_memory_gradient_matches_direct_real_qwen_moe_lora():
    direct = _tiny_model()
    replay = deepcopy(direct)
    pair = _pair()
    reference = cache_reference_scores(direct, [pair], _identity(), logprob_fn=completion_logprobs)[pair.key]
    # Nonzero current/reference margin, with no dropout or updates during replay.
    for model in (direct, replay):
        with torch.no_grad():
            for name, value in model.named_parameters():
                if "lora_B" in name:
                    value.add_(.03)
    direct.train()
    loss = dpo_loss(_sequence_sum(direct, pair.chosen.sequence, completion_logprobs),
                    _sequence_sum(direct, pair.rejected.sequence, completion_logprobs), reference, beta=.1)
    loss.backward()
    observation = backward_pair_low_memory(replay, pair, reference, beta=.1, logprob_fn=completion_logprobs)
    torch.testing.assert_close(observation[0], loss.detach(), rtol=2e-5, atol=2e-6)
    magnitude = 0.
    for (name, expected), (_, actual) in zip(direct.named_parameters(), replay.named_parameters()):
        if not expected.requires_grad:
            continue
        assert expected.grad is not None and actual.grad is not None
        torch.testing.assert_close(actual.grad, expected.grad, rtol=4e-5, atol=2e-6)
        magnitude += actual.grad.abs().sum().item()
    assert magnitude > 0
    assert not torch.cuda.is_initialized()


def test_base_reference_is_distinct_from_enabled_warm_adapter_reference():
    model, pair = _tiny_model(), _pair()
    warm = cache_reference_scores(model, [pair], _identity(), logprob_fn=completion_logprobs)[pair.key]
    base = cache_reference_scores(model, [pair], _identity("base"), logprob_fn=completion_logprobs)[pair.key]
    assert abs(warm.chosen_sum-base.chosen_sum) + abs(warm.rejected_sum-base.rejected_sum) > 1e-5
    with model.disable_adapter():
        with pytest.raises(ValueError, match="disabled adapter"):
            cache_reference_scores(model, [pair], _identity(), logprob_fn=completion_logprobs)
    with pytest.raises(ValueError, match="snapshot identity"):
        ReferenceIdentity("b", "t", "q", "initial_adapter")


def test_low_memory_optimizer_update_matches_direct_and_empty_step_preserves_momentum():
    low_model = _tiny_model()
    direct_model = deepcopy(low_model)
    pair = _pair()
    trainer = OfflineNativeDPOTrainer(low_model, [pair], _identity(), logprob_fn=completion_logprobs,
        optimizer_factory=lambda p: torch.optim.AdamW(p, lr=1e-3, weight_decay=0.0))
    direct_opt = torch.optim.AdamW([p for p in direct_model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.)
    reference = trainer.references[pair.key]
    direct_model.train()
    direct_loss = dpo_loss(_sequence_sum(direct_model, pair.chosen.sequence, completion_logprobs),
        _sequence_sum(direct_model, pair.rejected.sequence, completion_logprobs), reference, beta=.1)
    direct_loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in direct_model.parameters() if p.requires_grad], 1.)
    direct_opt.step()
    report = trainer.step([pair])
    assert report["algorithm"] == "offline_native_final_DPO; not RL"
    assert report["measured_KL"] is False
    for expected, actual in zip(direct_model.parameters(), low_model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-7)
    before = [p.detach().clone() for p in trainer.parameters]
    assert trainer.step([])["optimizer_step_skipped"] is True
    assert trainer.optimizer_steps == 1
    for expected, actual in zip(before, trainer.parameters):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.cuda.is_initialized()


def test_dropout_and_wrong_sequence_cache_are_rejected():
    model, pair = _tiny_model(dropout=.2), _pair()
    with pytest.raises(ValueError, match="zero dropout"):
        validate_deterministic_scoring(model)
    model = _tiny_model()
    reference = CachedReference(pair.key, _identity(), -70, -75)
    changed = PreferencePair(replace(pair.chosen, sequence=replace(pair.chosen.sequence,
        prompt=(4, 3, 151667))), replace(pair.rejected, sequence=replace(pair.rejected.sequence,
        prompt=(4, 3, 151667))))
    with pytest.raises(ValueError, match="different exact IDs"):
        backward_pair_low_memory(model, changed, reference, beta=.1, logprob_fn=completion_logprobs)


class TinyScores(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scores = torch.nn.Parameter(torch.tensor([.1, -.2, .3]))
        self.peft_config = {"default": "synthetic unit-test-only adapter"}


def _tiny_score_fn(model, prompt, completion):
    return model.scores.log_softmax(0)[torch.tensor([token % 3 for token in completion])]


def test_unequal_rank_pair_counts_have_global_equal_pair_gradient():
    pairs = [_pair(), PreferencePair(_candidate(seed=3, reward=1), _candidate(seed=4, reward=0,
        completion=(7, 151668, 6, 6, 151645)))]
    reference_model = TinyScores()
    refs = cache_reference_scores(reference_model, pairs, _identity(), logprob_fn=_tiny_score_fn)
    direct = TinyScores()
    direct_loss = sum(dpo_loss(_sequence_sum(direct, p.chosen.sequence, _tiny_score_fn),
        _sequence_sum(direct, p.rejected.sequence, _tiny_score_fn), refs[p.key], beta=.1) for p in pairs) / 2
    direct_loss.backward()
    rank0, rank1 = TinyScores(), TinyScores()
    for pair in pairs:
        backward_pair_low_memory(rank0, pair, refs[pair.key], beta=.1,
                                 logprob_fn=_tiny_score_fn, scale=2/2)
    # Existing production gradient sync materializes missing local grads as0.
    rank1.scores.grad = torch.zeros_like(rank1.scores)
    averaged = (rank0.scores.grad + rank1.scores.grad) / 2
    torch.testing.assert_close(averaged, direct.scores.grad)
    called = []
    def synchronize(parameters, world):
        called.append(world)
        for p in parameters:
            p.grad = direct.scores.grad.clone()
    trainer = OfflineNativeDPOTrainer(TinyScores(), pairs, _identity(), logprob_fn=_tiny_score_fn,
        optimizer_factory=lambda p: torch.optim.SGD(p, lr=.01), world_size=2, gradient_sync=synchronize)
    report = trainer.step([], global_pair_count=2)
    assert called == [2] and report["local_pairs"] == 0 and report["optimizer_steps"] == 1
    with pytest.raises(ValueError, match="verified global"):
        trainer.step([])


def test_immutable_sequence_rejects_fabricated_phase_boundary():
    pair = _pair()
    with pytest.raises(ValueError, match="phase boundaries"):
        replace(pair.chosen.sequence, phase_lengths=(1, 1))
    with pytest.raises(ValueError, match="closing-token"):
        replace(pair.chosen.sequence, phase_lengths=(3, 2))
    with pytest.raises(ValueError, match="bounded offline TRAIN reward"):
        replace(pair.chosen, reward=float("nan"))


@pytest.mark.parametrize("damage", ["anyOf", "tool", "missing_answer", "additional"])
def test_public_prefix_rejects_early_action_or_weakened_final_schema(damage):
    _, _, prefix, _, _ = _physical_fixture()
    schema = json.loads(prefix.schema_json)
    if damage == "anyOf":
        schema["anyOf"] = [{"required": ["tool"]}, {"required": ["answer"]}]
    elif damage == "tool":
        schema["properties"]["tool"] = {"type": "string"}
    elif damage == "missing_answer":
        schema["required"] = []
    else:
        schema["additionalProperties"] = True
    with pytest.raises(ValueError, match="final-only"):
        replace(prefix, schema_json=json.dumps(schema))


def test_public_prefix_canonical_contract_roundtrips_without_labels_or_seed():
    _, _, prefix, _, _ = _physical_fixture()
    public = prefix.to_dict()
    actual = PublicFinalPrefix.from_dict(public)
    assert actual.to_dict() == public
    assert "generation_seed" not in public["native_identity"]
    assert not {"reward", "reward_version", "gold", "polarity", "category", "completion"}.intersection(public)
