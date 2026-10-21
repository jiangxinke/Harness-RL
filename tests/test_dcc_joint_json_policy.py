"""CPU mechanics tests; synthetic tiny models do not measure benchmark gains."""
import asyncio
from dataclasses import asdict, replace
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3MoeConfig, Qwen3MoeForCausalLM

import dcc_online_grpo as trainer
from slime_plugins.dcc import EpisodeRunner
from slime_plugins.dcc.cuad_structured import cuad_final_schema
from slime_plugins.dcc.data import Task
from slime_plugins.dcc.episode import ModelProtocolFailure
from slime_plugins.dcc import hf_native_policy as native
from slime_plugins.dcc.native_joint_grpo import (
    JSON_BEHAVIOR_SOURCE, JointSegment, grammar_token_logprobs,
    joint_group_signal, json_prefix_bitmasks, segment_scorer,
)
from slime_plugins.dcc.online_grpo import group_signal, thought_policy_loss
from slime_plugins.dcc.reference_kl import thought_reference_kl_loss


@pytest.fixture
def tiny(monkeypatch):
    # Small IDs make the real HF/xgrammar CPU path cheap to exercise.
    monkeypatch.setattr(native, "OPEN_THINK", 3)
    monkeypatch.setattr(native, "END_THINK", 2)
    vocab = ["<pad>", "<eos>", "</think>", "<think>", "<unk>"] + [chr(i) for i in range(32, 127)]
    raw = Tokenizer(models.WordLevel({t: i for i, t in enumerate(vocab)}, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    raw.decoder = decoders.Fuse()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>", eos_token="<eos>",
        unk_token="<unk>", additional_special_tokens=["<think>", "</think>"])
    tok.chat_template = ("{% for message in messages %}a{% endfor %}"
                         "{% if add_generation_prompt %}" + native.NATIVE_SUFFIX + "{% endif %}")
    torch.manual_seed(107)
    cfg = Qwen3MoeConfig(vocab_size=len(vocab), hidden_size=16, intermediate_size=24,
        moe_intermediate_size=8, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, head_dim=8, num_experts=2, num_experts_per_tok=1,
        max_position_embeddings=512, eos_token_id=1, pad_token_id=0, attention_dropout=0.0)
    model = Qwen3MoeForCausalLM(cfg).eval()
    head = torch.nn.Linear(16, len(vocab), bias=True)
    with torch.no_grad():
        head.weight.copy_(model.lm_head.weight)
        head.bias.zero_()
        head.bias[2] = 6  # Real sampled thought close; still finite, stochastic.
        head.bias[1] = -6  # JSON grammar permits real EOS only after completion.
        head.bias[vocab.index('"')] = 3
        head.bias[vocab.index(']')] = 2
        head.bias[vocab.index('}')] = 2
        head.bias[vocab.index(' ')] = -6
    model.lm_head = head
    return model, tok


def make_policy(tiny, *, joint=True, json_tokens=192, **kwargs):
    model, tok = tiny
    return native.NativeHFPolicy(model, tok, thought_tokens=4, json_tokens=json_tokens,
        max_prompt_tokens=128, batch_size=8, capture_thought_logprobs=True,
        joint_json_policy=joint, **kwargs)


def segment_from_completion(completion):
    return JointSegment(completion.prompt_ids, completion.completion_ids,
        completion.thought_logprobs, completion.json_logprobs, completion.thought_token_count,
        completion.json_public_schema, completion.json_stop_ids)


@pytest.mark.parametrize("json_tokens", [2, 192])
def test_weighted_replay_retains_real_complete_or_truncated_json_boundary(tiny, json_tokens):
    """Actual HF/xgrammar behavior through replay; phase losses use the old oracle."""
    model, _ = tiny
    policy = make_policy(tiny, json_tokens=json_tokens)
    torch.manual_seed(41)
    output = policy._generate_encoded_batch([[5]], cuad_final_schema(), ["weighted-TRAIN-fixture"])[0]
    completion = output.completion if isinstance(output, ModelProtocolFailure) else output
    segment = segment_from_completion(completion)
    count = segment.thought_count
    final = segment.completion_ids[count:]
    assert count > 0 and final
    if json_tokens == 2:
        assert len(final) == 2 and final[-1] not in policy.eos_ids
    else:
        assert final[-1] in policy.eos_ids
    actual_old = list(segment.behavior_logprobs)
    actual_ids = list(segment.completion_ids)
    current = segment_scorer(segment, policy.compiler, backend="original")(
        model, segment.prompt_ids, segment.completion_ids)
    old = torch.tensor(actual_old)
    budget = 4 + json_tokens
    expected = thought_policy_loss(current[:count], old[:count], -.4,
        group_size=2, max_steps=2, thought_tokens=budget)
    expected += 4 * thought_policy_loss(current[count:], old[count:], -.4,
        group_size=2, max_steps=2, thought_tokens=budget)
    expected.backward()
    gradients = [p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    signal = SimpleNamespace(segments=[[segment], []], advantages=[-.4, 0.])
    report = trainer.replay_group(model, signal, None, group_size=2, max_steps=2,
        thought_tokens=4, json_tokens=json_tokens, clip=.2, joint_json_policy=True,
        grammar_compiler=policy.compiler, logprob_backend="original", replay_diagnostic="off",
        json_surrogate_weight=4)
    assert report["policy_loss"] == pytest.approx(expected.item(), rel=1e-6, abs=1e-8)
    for parameter, gradient in zip(model.parameters(), gradients, strict=True):
        assert (parameter.grad is None) == (gradient is None)
        if gradient is not None:
            torch.testing.assert_close(parameter.grad, gradient, rtol=2e-5, atol=2e-7)
    assert segment.behavior_logprobs == actual_old and segment.completion_ids == actual_ids


@pytest.mark.parametrize("allowed", [[1], [0, 2]])
def test_masked_probability_and_gradient(allowed):
    logits = torch.tensor([[0.3, -0.1, 0.7]], requires_grad=True)
    bitmask = torch.tensor([[sum(1 << index for index in allowed)]], dtype=torch.int32)
    target = torch.tensor([allowed[0]])
    actual = grammar_token_logprobs(logits, target, bitmask)
    expected = logits[0, allowed[0]] - logits[0, allowed].logsumexp(0)
    torch.testing.assert_close(actual[0], expected)
    actual.sum().backward()
    if len(allowed) == 1:
        assert actual.item() == 0
        assert torch.equal(logits.grad, torch.zeros_like(logits))
    else:
        assert logits.grad[0, 1].item() == 0
        assert logits.grad[0, allowed[0]] > 0


def test_exact_schema_prefix_support_and_eos(tiny):
    policy = make_policy(tiny)
    text = '{"plan":"review","checks":"observed","answer":[]}'
    ids = tiny[1].encode(text, add_special_tokens=False) + [1]
    masks = json_prefix_bitmasks(policy.compiler, cuad_final_schema(), ids, eos_ids=[1])
    assert masks.shape == (len(ids), (tiny[0].config.vocab_size + 31) // 32)
    assert json_prefix_bitmasks(policy.compiler, cuad_final_schema(), ids[:-3], eos_ids=[1]).shape[0] == len(ids) - 3
    for bad in ([tiny[1].convert_tokens_to_ids('}')], ids + [1]):
        with pytest.raises(ValueError):
            json_prefix_bitmasks(policy.compiler, cuad_final_schema(), bad, eos_ids=[1])
    with pytest.raises(ValueError, match="stop IDs"):
        json_prefix_bitmasks(policy.compiler, cuad_final_schema(), ids, eos_ids=[0])


@pytest.mark.parametrize("eos_ids", [[1, 0], [0, 1]])
def test_multiple_eos_compiler_order_preserves_exact_support(tiny, eos_ids):
    model, tok = tiny
    model.generation_config.eos_token_id = list(eos_ids)
    policy = make_policy(tiny)
    text = '{"plan":"review","checks":"observed","answer":[]}'
    prefix = tok.encode(text, add_special_tokens=False)
    for stop in eos_ids:
        ids = prefix + [stop]
        masks = json_prefix_bitmasks(policy.compiler, cuad_final_schema(), ids, eos_ids=eos_ids)
        reordered = json_prefix_bitmasks(policy.compiler, cuad_final_schema(), ids,
                                         eos_ids=list(reversed(eos_ids)))
        assert torch.equal(masks, reordered)
        # Both real stops become legal only after the complete JSON value.
        for actual_stop in eos_ids:
            assert not any((int(row[actual_stop // 32]) >> (actual_stop % 32)) & 1
                           for row in masks[:-1])
            assert (int(masks[-1, actual_stop // 32]) >> (actual_stop % 32)) & 1
        for illegal in ([stop], prefix[:-1] + [stop], ids + [stop]):
            with pytest.raises(ValueError):
                json_prefix_bitmasks(policy.compiler, cuad_final_schema(), illegal, eos_ids=eos_ids)
    for mismatched in ([1], [0], [1, 0, 4], [1, 0, 1]):
        with pytest.raises(ValueError, match="stop IDs"):
            json_prefix_bitmasks(policy.compiler, cuad_final_schema(), prefix + [1], eos_ids=mismatched)
    assert model.generation_config.eos_token_id == eos_ids
    assert policy.eos_ids == eos_ids


@pytest.mark.parametrize("eos_ids", [[1], [1, 0]])
def test_actual_sampling_current_scores_padding_and_chunked_gradients(tiny, eos_ids):
    model, tok = tiny
    model.generation_config.eos_token_id = list(eos_ids)
    policy = make_policy(tiny)
    torch.manual_seed(41)
    outputs = policy._generate_encoded_batch([[5], [5, 6, 7]], cuad_final_schema(), ["x", "y"])
    completions = [r.completion if isinstance(r, ModelProtocolFailure) else r for r in outputs]
    assert all(c.completion_ids[c.thought_token_count - 1] == 2 for c in completions)
    assert all(c.json_logprobs_source == JSON_BEHAVIOR_SOURCE for c in completions)
    assert any(c.completion_ids[-1] in eos_ids for c in completions)
    assert all(c.json_stop_ids == eos_ids for c in completions)
    assert model.generation_config.eos_token_id == policy.eos_ids == eos_ids
    for completion in completions:
        segment = segment_from_completion(completion)
        actual_old = torch.tensor(segment.behavior_logprobs)
        original = segment_scorer(segment, policy.compiler, backend="original")
        chunked = segment_scorer(segment, policy.compiler, backend="qwen3-moe-chunked", head_chunk_tokens=7)
        with torch.no_grad():
            reference = original(model, segment.prompt_ids, segment.completion_ids)
            observed = chunked(model, segment.prompt_ids, segment.completion_ids)
        torch.testing.assert_close(reference, actual_old, rtol=0, atol=1e-5)
        torch.testing.assert_close(observed, actual_old, rtol=0, atol=1e-5)
    # Check checkpointed decoder/head replay, with a head chunk crossing phases.
    segment = segment_from_completion(completions[0])
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    gradients = []
    for backend in ("original", "qwen3-moe-chunked"):
        model.zero_grad(set_to_none=True)
        score = segment_scorer(segment, policy.compiler, backend=backend, head_chunk_tokens=7)
        values = score(model, segment.prompt_ids, segment.completion_ids)
        old = torch.tensor(segment.behavior_logprobs)
        thought_policy_loss(values, old, 0.7, group_size=2, max_steps=2,
                            thought_tokens=196).backward()
        gradients.append([p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()])
    for original, chunked in zip(*gradients, strict=True):
        assert (original is None) == (chunked is None)
        if original is not None:
            torch.testing.assert_close(original, chunked, rtol=1e-4, atol=1e-6)
    assert not torch.cuda.is_initialized()


def collect(tiny, *, json_tokens=192):
    traces = []
    policy = make_policy(tiny, json_tokens=json_tokens, trace_sink=traces.append)
    runner = EpisodeRunner(max_steps=1, max_new_tokens=4 + json_tokens,
        temperature=1, evaluation_temperature=1, cuad_structured_actions=True, initial_chunks=1)
    task = Task("toy-cuad", "cuad", "train", "Find a qualifying clause.",
                document="An unrelated contract sentence.", gold=[])
    torch.manual_seed(61)
    results = asyncio.run(trainer.collect_group(runner, task, policy, group_size=8, rank=0))
    return policy, results, traces


def test_collection_signal_replay_uses_sampled_behavior_and_kl(tiny):
    from peft import LoraConfig, get_peft_model
    model, tok = tiny
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
        target_modules=["q_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    # Nonzero adapter makes the frozen-base K3 test exercise a real penalty.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.04)
    policy, results, traces = collect((model, tok))
    signal = joint_group_signal(results, group_size=8, max_steps=1, thought_tokens=4,
        json_tokens=192, eos_ids=[1], advantage_normalization="centered")
    assert signal.discarded_reason is None
    assert max(signal.rewards) > min(signal.rewards), [r.status for r in results]
    assert all(isinstance(s, JointSegment) for steps in signal.segments for s in steps)
    assert all(row["protocol"] == native.JOINT_PROTOCOL for row in traces)
    persisted = [r.to_dict(include_tokens=True) for r in results]
    assert all("json_logprobs" in row["steps"][0] and "json_public_schema" in row["steps"][0] for row in persisted)
    assert all("json_logprobs" not in r.to_dict(include_tokens=False)["steps"][0] for r in results)
    report = trainer.replay_group(model, signal, None, group_size=8, max_steps=1,
        thought_tokens=4, json_tokens=192, clip=0.2, replay_diagnostic="full",
        joint_json_policy=True, grammar_compiler=policy.compiler,
        logprob_backend="qwen3-moe-chunked", head_chunk_tokens=7,
        reference_kl_beta=0.02, reference_base_model="toy", reference_source_commit="0" * 40)
    assert report["json_scored_tokens"] > 0
    assert report["maximum_behavior_replay_logprob_difference"] < 1e-5
    assert report["maximum_behavior_current_logprob_difference"] < 1e-5
    assert report["importance_ratio_outside_clip_fraction"] == 0
    assert report["reference_kl_unweighted_loss"] > 0
    assert report["reference_kl"]["json_tokens"] is True
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.parameters() if p.requires_grad)


def test_truncated_json_failures_retain_sampled_tokens_and_zero_reward(tiny):
    policy, results, traces = collect(tiny, json_tokens=2)
    assert all(r.status == "model_protocol_failure" and r.reward == 0 for r in results)
    signal = joint_group_signal(results, group_size=8, max_steps=1, thought_tokens=4,
                                json_tokens=2, eos_ids=[1])
    assert signal.advantages == [0] * 8
    assert all(len(s.json_logprobs) == 2 for segments in signal.segments for s in segments)
    assert all(len(s.completion_ids) == s.thought_count + 2 for segments in signal.segments for s in segments)
    assert all(trace["json_ids"][-1] != 1 for trace in traces)


def test_failed_thoughts_retain_raw_behavior_without_json(tiny):
    model, _ = tiny
    with torch.no_grad():
        model.lm_head.bias[1:3] = -100
    policy, results, traces = collect(tiny, json_tokens=2)
    assert all(r.status == "model_protocol_failure" and r.reward == 0 for r in results)
    signal = joint_group_signal(results, group_size=8, max_steps=1,
        thought_tokens=4, json_tokens=2, eos_ids=[1])
    assert signal.advantages == [0] * 8
    assert all(trace["json_ids"] == [] and trace["json_logprobs_source"] is None for trace in traces)
    for segments in signal.segments:
        segment = segments[0]
        assert len(segment.completion_ids) == segment.thought_count == 4
        assert segment.json_logprobs == []
        for backend in ("original", "qwen3-moe-chunked"):
            with torch.no_grad():
                replay = segment_scorer(segment, policy.compiler, backend=backend)(
                    model, segment.prompt_ids, segment.completion_ids)
            torch.testing.assert_close(replay, torch.tensor(segment.behavior_logprobs), rtol=0, atol=1e-5)


def test_joint_cli_cpu_preflight_checks_grammar_without_model_weights(tiny, tmp_path, monkeypatch, capsys):
    model, tokenizer = tiny
    metadata = tmp_path / "model"
    metadata.mkdir()
    model.config.save_pretrained(metadata)
    model.generation_config.save_pretrained(metadata)
    tokenizer.save_pretrained(metadata)
    data = tmp_path / "train.jsonl"
    task = Task("toy-cuad", "cuad", "train", "Find a clause.", document="No qualifying clause.", gold=[])
    data.write_text(json.dumps(task.to_dict()) + "\n")
    monkeypatch.setattr(trainer, "validate_source", lambda *args, **kwargs: None)
    result = trainer.main(["--model", str(metadata), "--train-data", str(data),
        "--output-dir", str(tmp_path / "unused-output"), "--source-commit", "0" * 40,
        "--steps", "1", "--max-steps", "1", "--thought-tokens", "4", "--json-tokens", "192",
        "--max-prompt-tokens", "8192", "--joint-json-policy", "--cpu-preflight",
        "--cuad-task-sampling", "uniform"])
    assert result == 0
    report = json.loads(capsys.readouterr().out)
    support = report["native_grammar_preflight"]["joint_json_support_preflight"]
    assert report["cpu_preflight_only"] and not report["CUDA_initialized"]
    assert support["legal_complete_JSON_and_EOS"] and not support["actual_model_sampling"]
    assert support["behavior_source"] == JSON_BEHAVIOR_SOURCE
    assert report["training_config"]["json_policy_loss"] is True
    assert not (tmp_path / "unused-output").exists()


def test_greedy_traces_and_web_are_rejected_at_runtime(tiny, capsys):
    policy, results, _ = collect(tiny, json_tokens=2)
    legacy = native.NativeHFCompletion(**{k: v for k, v in asdict(results[0].steps[0]).items()
        if k not in {"json_logprobs", "json_logprobs_source", "json_public_schema", "json_stop_ids"}})
    altered = [replace(results[0], steps=[legacy])] + results[1:]
    with pytest.raises(ValueError, match="Fresh stochastic"):
        joint_group_signal(altered, group_size=8, max_steps=1, thought_tokens=4, json_tokens=2, eos_ids=[1])
    from slime_plugins.dcc.web_structured import web_final_schema
    schema = web_final_schema(["name"], [])
    with pytest.raises(ValueError, match="CUAD only"):
        asyncio.run(policy.generate_structured([], json_schema=schema, max_new_tokens=6, temperature=1))
    with pytest.raises(ValueError, match="CUAD only"):
        policy._generate_encoded_batch([[5]], schema, ["web"])
    args = SimpleNamespace(joint_json_policy=True, enable_online_web=True)
    with pytest.raises(ValueError, match="CUAD only"):
        trainer.configuration(args, [], [], 1)
    with pytest.raises(ValueError, match="CUAD only"):
        asyncio.run(trainer.train(args, [], [], 1, web_inputs=object()))
    args.enable_online_web = False
    web_tasks = [SimpleNamespace(benchmark="widesearch")]
    with pytest.raises(ValueError, match="CUAD only"):
        trainer.configuration(args, [], web_tasks, 1)
    with pytest.raises(ValueError, match="CUAD only"):
        asyncio.run(trainer.train(args, web_tasks, [], 1))
    with pytest.raises(SystemExit) as rejected:
        trainer.main(["--model", "unused", "--train-data", "unused", "--output-dir", "unused",
            "--source-commit", "0" * 40, "--joint-json-policy", "--enable-online-web"])
    assert rejected.value.code == 2
    assert "supports CUAD only" in capsys.readouterr().err


@pytest.mark.parametrize("normalization", ["std", "centered"])
def test_advantage_clip_fixed_budget_and_k3_match_original_math(tiny, normalization):
    policy, results, _ = collect(tiny, json_tokens=2)
    # Synthetic positive/negative outcomes isolate group arithmetic from model ability.
    changed = [replace(results[0], status="completed", reward=0.8),
               replace(results[1], status="completed", reward=0.2)]
    old_signal = group_signal(changed, group_size=2, max_steps=1, thought_tokens=4,
                              advantage_normalization=normalization)
    joint = joint_group_signal(changed, group_size=2, max_steps=1, thought_tokens=4,
                               json_tokens=2, eos_ids=[1], advantage_normalization=normalization)
    assert joint.advantages == old_signal.advantages
    current = torch.tensor([-0.1, -0.5, -2.0], requires_grad=True)
    old = torch.tensor([-0.5, -0.1, -1.0])
    advantage = joint.advantages[0]
    ratio = (current - old).exp()
    expected = -torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage).sum() / 12
    loss = thought_policy_loss(current, old, advantage, group_size=2, max_steps=1, thought_tokens=6)
    torch.testing.assert_close(loss, expected)
    reference = old - 0.1
    k3 = thought_reference_kl_loss(current, reference, group_size=2, max_steps=1, thought_tokens=6)
    d = reference - current
    torch.testing.assert_close(k3, (torch.expm1(d) - d).sum() / 12)


def test_default_completion_fields_remain_original(tiny):
    policy = make_policy(tiny, joint=False, json_tokens=2)
    torch.manual_seed(41)
    result = policy._generate_encoded_batch([[5]], cuad_final_schema(), ["x"])[0]
    completion = result.completion if isinstance(result, ModelProtocolFailure) else result
    assert type(completion) is native.NativeHFCompletion
    assert set(asdict(completion)) == {"text", "prompt_ids", "completion_ids", "logprobs", "finish_reason",
        "thought_token_count", "phase_lengths", "native_status", "thought_logprobs",
        "thought_logprobs_source", "trace_id"}
    assert policy.provenance["protocol"] == native.PROTOCOL
    assert "json_logprobs_source" not in policy.provenance


@pytest.mark.parametrize("beta", [0, 0.02])
def test_default_configuration_matches_frozen_original_and_joint_scope(beta):
    reference = os.environ.get("DCC_THOUGHT_ONLY_REFERENCE_SOURCE")
    if reference is None:
        pytest.skip("Set a read-only original source export to compare default configuration")
    spec = importlib.util.spec_from_file_location("_dcc_online_grpo_reference",
        Path(reference) / "scripts/dcc_online_grpo.py")
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    arguments = dict(model="toy", source_commit="0" * 40, group_size=2,
        max_steps=1, thought_tokens=4, json_tokens=192, reference_kl_beta=beta)
    old = baseline.configuration(SimpleNamespace(**arguments), [], [], 1)
    args = SimpleNamespace(**arguments, joint_json_policy=False)
    current = trainer.configuration(args, [], [], 1)
    assert json.dumps(current) == json.dumps(old)
    assert current.get("algorithm_variant") == (trainer.REFERENCE_KL_VARIANT if beta else None)
    args.joint_json_policy = True
    joint = trainer.configuration(args, [], [], 1)
    assert joint["native_policy_protocol"] == native.JOINT_PROTOCOL
    assert joint["json_policy_loss"] is True
    assert joint["policy_loss_tokens"] == "actual sampled thought and grammar-normalized JSON, including real stops"
    assert joint["behavior_policy"]["old_logprob_source"]["json"] == JSON_BEHAVIOR_SOURCE
    assert joint["objective_denominator"] == 2 * (4 + 192)
    assert joint["reference_kl"]["json_tokens"] is True
    if beta:
        assert joint["algorithm_variant"] == "frozen-base-reference-k3-nonzero-advantage-joint-v1"
        assert joint["reference_kl"]["active_scope"] == "nonzero-advantage-sampled-thought-and-JSON-segments"


def test_default_outputs_and_scores_match_frozen_original(tiny, monkeypatch):
    reference = os.environ.get("DCC_THOUGHT_ONLY_REFERENCE_SOURCE")
    if reference is None:
        pytest.skip("Set a read-only original source export to compare exact default outputs")
    spec = importlib.util.spec_from_file_location("slime_plugins.dcc._thought_only_reference",
        Path(reference) / "slime/slime_plugins/dcc/hf_native_policy.py")
    baseline = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    monkeypatch.setattr(baseline, "OPEN_THINK", 3)
    monkeypatch.setattr(baseline, "END_THINK", 2)
    for module in (baseline, native):
        monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: 0.0))
        monkeypatch.setattr(module, "uuid", SimpleNamespace(uuid4=lambda: SimpleNamespace(hex="fixed")))
    model, tok = tiny
    records = []
    for cls in (baseline.NativeHFPolicy, native.NativeHFPolicy):
        traces = []
        policy = cls(model, tok, thought_tokens=4, json_tokens=32, max_prompt_tokens=128,
                     capture_thought_logprobs=True, trace_sink=traces.append)
        torch.manual_seed(92)
        outputs = policy._generate_encoded_batch([[5], [5, 6, 7]], cuad_final_schema(), ["x", "y"])
        completions = [r.completion if isinstance(r, ModelProtocolFailure) else r for r in outputs]
        records.append(json.dumps(dict(completions=[asdict(c) for c in completions],
                                       traces=traces, provenance=policy.provenance)))
    assert records[0] == records[1]
