"""Actual tiny CPU BF16 compute tests; no actual GPU/NF4 claim."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'scripts'),
                str(Path(__file__).resolve().parents[1] / 'slime')]

import pytest
import torch
from transformers import GenerationConfig, Qwen3MoeConfig, Qwen3MoeForCausalLM

from dcc_online_grpo import replay_group
from slime_plugins.dcc import hf_policy, hf_quantization
from slime_plugins.dcc.hf_chunked_logprobs import completion_logprobs_chunked, resolve_logprob_backend
from slime_plugins.dcc.hf_compute import (BF16, LEGACY, adapter_compute_profile,
    compute_context, compute_profile_configuration, configure_compute_profile,
    precision_observation, validate_compute_profile)
from slime_plugins.dcc.hf_policy import completion_logprobs


def tiny_model(seed=59):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        config = Qwen3MoeConfig(vocab_size=97, hidden_size=32, intermediate_size=48,
            moe_intermediate_size=16, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8, num_experts=4, num_experts_per_tok=2,
            max_position_embeddings=256, attention_dropout=0.)
        return Qwen3MoeForCausalLM(config).to(torch.bfloat16)


@pytest.fixture
def actual_prepared_loader(monkeypatch):
    """Acquisition/NF4 inspection are fixtures; PEFT prep/LoRA are actual."""
    import transformers
    class Tokenizer:
        pad_token_id = 0
    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained',
                        staticmethod(lambda *a, **k: tiny_model()))
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained',
                        staticmethod(lambda *a, **k: Tokenizer()))
    monkeypatch.setattr(torch.cuda, 'set_device', lambda rank: None)
    monkeypatch.setattr(hf_quantization, 'nf4_load_kwargs', lambda: {})
    monkeypatch.setattr(hf_quantization, 'quantization_report',
                        lambda model: {'mode': 'CPU-nonquantized-fixture'})
    def load(*, profile=LEGACY, adapter=None, trainable=False):
        return hf_policy.load_policy_model('invented-CPU-model', quantization='nf4',
            trainable=trainable, lora_rank=2, adapter=adapter, compute_profile=profile)[0]
    return load


def saved_config(model, profile):
    return {'compute_profile': compute_profile_configuration(profile),
            'arguments': {'compute_profile': profile}, 'precision': precision_observation(model)}


@pytest.mark.parametrize('profile', [LEGACY, BF16])
def test_actual_shared_loader_preserves_storage_and_adapter_roundtrip(actual_prepared_loader, tmp_path, profile):
    trained = actual_prepared_loader(profile=profile, trainable=True)
    with torch.no_grad():
        for name, parameter in trained.named_parameters():
            if 'lora_B' in name:
                parameter.fill_(.04)
    trained.eval()
    adapter = tmp_path / 'actual-tiny-CPU-adapter'
    trained.save_pretrained(adapter)
    (adapter / 'training_config.json').write_text(json.dumps(saved_config(trained, profile)))
    base = actual_prepared_loader(profile=profile).eval()
    restored = actual_prepared_loader(profile=profile, adapter=str(adapter)).eval()
    observations = [precision_observation(model) for model in (trained, base, restored)]
    assert all(set(value['base_parameter_dtypes']) == {'torch.float32'} for value in observations)
    assert observations[0]['adapter_parameter_dtypes'] == observations[2]['adapter_parameter_dtypes']
    assert set(observations[0]['adapter_parameter_dtypes']) == {'torch.float32'}
    assert observations[0]['compute_profile'] == observations[1]['compute_profile'] == observations[2]['compute_profile']
    assert observations[0]['compute_context'] == observations[1]['compute_context'] == observations[2]['compute_context']
    assert observations[0]['compute_context']['autocast_enabled'] is (profile == BF16)
    ids = torch.tensor([[3, 4, 5]])
    logits = []
    for model in (trained, base, restored):
        with torch.no_grad(), compute_context(model):
            logits.append(model(ids, use_cache=False).logits)
    assert all(value.dtype == (torch.bfloat16 if profile == BF16 else torch.float32) for value in logits)
    torch.testing.assert_close(logits[0], logits[2], rtol=0, atol=0)
    assert not torch.equal(logits[0], logits[1])
    assert adapter_compute_profile(adapter) == compute_profile_configuration(profile)
    assert not torch.cuda.is_initialized()


def test_legacy_profile_does_not_change_model_outputs(actual_prepared_loader):
    model = actual_prepared_loader().eval()
    ids = torch.tensor([[3, 4, 5]])
    with torch.no_grad():
        direct = model(ids, use_cache=False).logits
        with compute_context(model):
            selected = model(ids, use_cache=False).logits
    torch.testing.assert_close(direct, selected, atol=0, rtol=0)
    assert selected.dtype == torch.float32


@pytest.mark.parametrize('backend', ['original', 'qwen3-moe-chunked'])
def test_actual_bf16_old_current_replay_and_finite_fp32_lora_gradients(actual_prepared_loader, backend):
    model = actual_prepared_loader(profile=BF16, trainable=True)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'lora_B' in name:
                parameter.normal_(0, .02)
    scorer, _ = resolve_logprob_backend(backend, 3 if backend != 'original' else None)
    segment = SimpleNamespace(prompt_ids=[2, 7, 11], completion_ids=[31, 9, 71, 4, 18])
    with torch.no_grad():
        segment.thought_logprobs = scorer(model, segment.prompt_ids, segment.completion_ids).tolist()
    signal = SimpleNamespace(segments=[[segment]], advantages=[1.])
    report = replay_group(model, signal, scorer, group_size=2, max_steps=1, thought_tokens=8, clip=.2)
    assert report['maximum_replay_current_logprob_difference'] == 0
    assert report['scored_tokens'] == 5
    gradients = [parameter.grad for name, parameter in model.named_parameters() if 'lora_' in name]
    assert gradients and all(gradient is not None and gradient.dtype == torch.float32
                             and torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum().item() for gradient in gradients) > 0
    assert precision_observation(model)['compute_context']['autocast_dtype'] == 'torch.bfloat16'
    assert not torch.cuda.is_initialized()


def test_actual_bf16_hf_sampling_original_and_chunked_replay_share_context(actual_prepared_loader, tmp_path):
    model = actual_prepared_loader(profile=BF16, trainable=True).eval()
    prompt = [3, 4, 5]
    head_dtypes = []
    model.get_output_embeddings().register_forward_hook(
        lambda module, args, output: head_dtypes.append(output.dtype))
    cfg = GenerationConfig(max_new_tokens=5, do_sample=True, temperature=1., top_p=1., top_k=0,
        eos_token_id=2, pad_token_id=0, num_beams=1, num_return_sequences=1, use_cache=True,
        return_dict_in_generate=True, output_logits=True)
    with torch.no_grad(), compute_context(model):
        torch.manual_seed(37)
        from slime_plugins.dcc.hf_native_policy import _explicit_generation_kwargs
        generated = model.generate(input_ids=torch.tensor([prompt]), **_explicit_generation_kwargs(cfg))
    response = generated.sequences[0, len(prompt):].tolist()
    rows = torch.stack([logit[0].float() for logit in generated.logits])
    target = torch.tensor(response)
    sampled = rows.gather(-1, target[:, None]).squeeze(-1) - rows.logsumexp(-1)
    with torch.no_grad():
        original = completion_logprobs(model, prompt, response)
        chunked = completion_logprobs_chunked(model, prompt, response, head_chunk_tokens=3)
    assert head_dtypes and set(head_dtypes) == {torch.bfloat16}
    # Transformers promotes the returned per-step logits to FP32 for sampling.
    assert all(logit.dtype == torch.float32 for logit in generated.logits)
    assert sampled.dtype == original.dtype == chunked.dtype == torch.float32
    # BF16 batched decoder/head and KV decoding need not round identically.
    torch.testing.assert_close(sampled, original, atol=.04, rtol=.003)
    torch.testing.assert_close(original, chunked, atol=.04, rtol=.003)
    (tmp_path / 'sample_replay_bf16.actual.json').write_text(json.dumps({
        'sample_original_max_abs_difference': (sampled - original).abs().max().item(),
        'original_chunked_max_abs_difference': (original - chunked).abs().max().item(),
        'head_dtypes': sorted(set(map(str, head_dtypes))),
        'generation_returned_logits_dtypes': sorted(set(str(logit.dtype) for logit in generated.logits)),
        'precision': precision_observation(model), 'CPU_only_no_actual_NF4': True}, indent=2))
    assert not torch.cuda.is_initialized()


def test_actual_native_hf_phase_uses_bf16_in_model_and_real_xgrammar(monkeypatch):
    from test_dcc_hf_native_policy import Tokenizer
    from slime_plugins.dcc.hf_native_policy import END_THINK, NativeHFPolicy
    import xgrammar
    tokenizer = Tokenizer()
    vocab = [''] * 151670
    for char in map(chr, range(128)):
        vocab[10000 + ord(char)] = char
    for text, token in tokenizer.tags.items():
        vocab[token] = text
    monkeypatch.setattr(xgrammar.TokenizerInfo, 'from_huggingface', staticmethod(
        lambda *a, **k: xgrammar.TokenizerInfo(vocab, xgrammar.VocabType.RAW,
            vocab_size=len(vocab), stop_token_ids=[151645, 151643])))
    config = Qwen3MoeConfig(vocab_size=151670, hidden_size=8, intermediate_size=12,
        moe_intermediate_size=4, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=4, num_experts=4, num_experts_per_tok=2,
        max_position_embeddings=4096, attention_dropout=0.)
    model = Qwen3MoeForCausalLM(config).eval()
    model.generation_config = GenerationConfig(eos_token_id=[151645, 151643])
    with torch.no_grad():
        model.lm_head.weight.zero_()
    model.lm_head.bias = torch.nn.Parameter(torch.full((config.vocab_size,), -10000.), requires_grad=False)
    configure_compute_profile(model, BF16)
    value = NativeHFPolicy(model, tokenizer, thought_tokens=4, json_tokens=8, max_prompt_tokens=2048)
    dtypes = []
    model.lm_head.register_forward_hook(lambda module, args, output: dtypes.append(output.dtype))
    with torch.no_grad():
        model.lm_head.bias[END_THINK] = 10000.
        thought = value._phase([[10065]], phase='thought')
        model.lm_head.bias[END_THINK] = -10000.
        model.lm_head.bias[tokenizer.eos_token_id] = 10000.
        final = value._phase([[10065, END_THINK]], phase='json',
            grammar=value.compiler.compile_json_schema({'const': 'X'}))
    assert thought == [[END_THINK]]
    assert tokenizer.decode(final[0][:-1]) == '"X"' and final[0][-1] == tokenizer.eos_token_id
    assert dtypes and set(dtypes) == {torch.bfloat16}
    assert value.provenance['compute_profile'] == compute_profile_configuration(BF16)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('profile', [None, True, 'bf16', {}, 1])
def test_unknown_compute_profiles_fail(profile):
    with pytest.raises(ValueError, match='Unknown HF compute profile'):
        validate_compute_profile(profile)


def test_checkpoint_profile_mismatch_is_rejected_before_gpu_or_model_access(monkeypatch, tmp_path):
    checkpoint = tmp_path / 'adapter'
    checkpoint.mkdir()
    (checkpoint / 'training_config.json').write_text(json.dumps({'compute_profile': compute_profile_configuration(BF16),
        'precision': {'compute_profile': compute_profile_configuration(BF16)}}))
    monkeypatch.setattr(torch.cuda, 'set_device', lambda *a: pytest.fail('must reject before device access'))
    with pytest.raises(ValueError, match='differs from adapter training profile'):
        hf_policy.load_policy_model('unused', adapter=str(checkpoint))
    assert validate_compute_profile(BF16, adapter=checkpoint) == compute_profile_configuration(BF16)
    (checkpoint / 'training_config.json').write_text(json.dumps({'compute_profile': compute_profile_configuration(BF16),
        'arguments': {'compute_profile': LEGACY}}))
    with pytest.raises(ValueError, match='arguments and compute profile disagree'):
        validate_compute_profile(BF16, adapter=checkpoint)


def test_legacy_adapter_and_external_autocast_fail_closed(tmp_path):
    checkpoint = tmp_path / 'legacy'
    checkpoint.mkdir()
    assert adapter_compute_profile(checkpoint) == compute_profile_configuration(LEGACY)
    with pytest.raises(ValueError, match='differs from adapter training profile'):
        validate_compute_profile(BF16, adapter=checkpoint)
    model = tiny_model().float()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        with pytest.raises(ValueError, match='Active autocast context differs'):
            with compute_context(model):
                pass
    assert not torch.cuda.is_initialized()


def test_matched_evaluator_records_and_rejects_training_or_base_profile_mismatch(tmp_path, monkeypatch):
    from test_dcc_online_evaluate import load_evaluator, arguments, tasks
    evaluator = load_evaluator()
    base = arguments(evaluator, tmp_path / 'base', compute_profile=BF16)
    record = evaluator.manifest(base, tasks(evaluator))
    assert record['comparison']['precision']['compute_profile'] == compute_profile_configuration(BF16)
    checkpoint = tmp_path / 'adapter'
    checkpoint.mkdir()
    (checkpoint / 'training_config.json').write_text(json.dumps({'compute_profile': compute_profile_configuration(BF16),
        'precision': {'compute_profile': compute_profile_configuration(BF16)}}))
    matched = arguments(evaluator, tmp_path / 'adapter-eval', adapter=str(checkpoint), compute_profile=BF16)
    assert evaluator.manifest(matched, tasks(evaluator))['condition']['training']['compute_profile'] == compute_profile_configuration(BF16)
    matched.compute_profile = LEGACY
    with pytest.raises(ValueError, match='differs from adapter training profile'):
        evaluator.manifest(matched, tasks(evaluator))
    assert evaluator.parser().get_default('compute_profile') == LEGACY


def test_actual_loader_rejects_changed_recorded_storage_precision(actual_prepared_loader, tmp_path):
    model = actual_prepared_loader(profile=BF16, trainable=True)
    adapter = tmp_path / 'adapter'
    model.save_pretrained(adapter)
    config = saved_config(model, BF16)
    config['precision']['base_parameter_dtypes'] = {'torch.bfloat16': 123}
    (adapter / 'training_config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='Reloaded adapter precision differs'):
        actual_prepared_loader(profile=BF16, adapter=str(adapter))


# Reuse the existing synthetic evaluator fixture, with real CPU autocast
# configuration. These are file-contract checks, not model-quality evidence.
from test_dcc_online_evaluate import evaluator


def test_matched_bf16_evaluator_checks_runtime_and_checkpoint_precision(evaluator, monkeypatch, tmp_path):
    import asyncio
    from test_dcc_online_evaluate import arguments, tasks
    original_loader = evaluator.load_policy_model
    def load(*args, **kwargs):
        model, tokenizer = original_loader(*args, **kwargs)
        configure_compute_profile(model, kwargs['compute_profile'])
        return model, tokenizer
    monkeypatch.setattr(evaluator, 'load_policy_model', load)
    all_tasks = tasks(evaluator)[:1]
    base = arguments(evaluator, tmp_path / 'base', compute_profile=BF16)
    asyncio.run(evaluator.generate(base, all_tasks))
    evaluator.score(base, all_tasks)
    checkpoint = tmp_path / 'adapter'
    checkpoint.mkdir()
    trained, _ = load('fixture', adapter='fixture', compute_profile=BF16)
    config = saved_config(trained, BF16)
    (checkpoint / 'training_config.json').write_text(json.dumps(config))
    adapted = arguments(evaluator, tmp_path / 'adapted', adapter=str(checkpoint),
        base_output_dir=base.output_dir, compute_profile=BF16)
    asyncio.run(evaluator.generate(adapted, all_tasks))
    report = evaluator.score(adapted, all_tasks)
    assert report['matched_base']['failure_zero_macro_f1_delta'] == 0
    assert report['precision'] == report['condition']['training']['precision']
    (adapted.output_dir / 'report.json').unlink()
    base_report = base.output_dir / 'report.json'
    before = json.loads(base_report.read_text())
    damaged = deepcopy(before)
    damaged['precision']['compute_context']['autocast_dtype'] = 'torch.float32'
    base_report.write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match='Base and adapter differ'):
        evaluator.score(adapted, all_tasks)
    base_report.write_text(json.dumps(before))
    checkpoint_config = checkpoint / 'training_config.json'
    config['precision']['adapter_parameter_dtypes'] = {'torch.bfloat16': 2}
    checkpoint_config.write_text(json.dumps(config))
    # Metadata was read at generation time: changing the checkpoint afterward
    # also invalidates the closed shard's declared adapter condition.
    with pytest.raises(ValueError, match='configuration, ownership or completed coverage'):
        evaluator.score(adapted, all_tasks)


def test_bf16_checkpoint_requires_its_actual_precision_receipt(tmp_path):
    checkpoint = tmp_path / 'missing-precision'
    checkpoint.mkdir()
    (checkpoint / 'training_config.json').write_text(json.dumps({
        'compute_profile': compute_profile_configuration(BF16)}))
    with pytest.raises(ValueError, match='missing its recorded training precision'):
        validate_compute_profile(BF16, adapter=checkpoint)
