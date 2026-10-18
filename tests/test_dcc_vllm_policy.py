"""CPU contracts for the optional vLLM evaluation backend, without an engine."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from types import SimpleNamespace

import pytest

from slime_plugins.dcc.hf_policy import encode_messages
from slime_plugins.dcc.vllm_policy import (
    VLLMPolicy,
    _adapter_metadata,
    _engine_environment,
    _validate_launch_environment,
    load_vllm_policy,
)


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize and not enable_thinking
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return [ord(c) + 3 for c in text + ("<assistant>" if add_generation_prompt else "")]

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens
        return "".join(chr(value - 3) for value in ids if value >= 3)


class Engine:
    def __init__(self, **kwargs):
        self.kwargs, self.calls = kwargs, []

    def generate(self, prompts, sampling, *, lora_request, use_tqdm):
        assert not use_tqdm
        self.calls.append((prompts, sampling, lora_request))
        return [SimpleNamespace(
            prompt_token_ids=item["prompt_token_ids"],
            outputs=[SimpleNamespace(token_ids=[ord(c) + 3 for c in '{"answer": []}'] + [1],
                                     finish_reason="stop")],
        ) for item in prompts]


@pytest.fixture
def runtime(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    fake = SimpleNamespace(__version__="0.11.2", LLM=Engine,
                           SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setitem(sys.modules, "vllm", fake)
    monkeypatch.setitem(sys.modules, "vllm.lora", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vllm.lora.request", SimpleNamespace(
        LoRARequest=lambda *args: args))
    return fake


def messages():
    return [{"role": "system", "content": "Use source evidence."},
            {"role": "user", "content": "What is the contract term?"}]


def test_generation_preserves_exact_hf_prompt_ids_and_lora_request(runtime):
    tokenizer, engine, request = Tokenizer(), Engine(), object()
    policy = VLLMPolicy(engine, tokenizer, lora_request=request, eos_token_ids=[1, 2], seed=17)
    result = policy.generate([messages()], 32, 0)[0]
    assert result.prompt_ids == encode_messages(tokenizer, messages(), 16384)
    assert result.completion_ids[-1] == 1
    assert result.text == '{"answer": []}' and result.logprobs is None
    assert result.finish_reason == "stop"
    prompts, sampling, actual_request = engine.calls[0]
    assert prompts == [{"prompt_token_ids": result.prompt_ids}]
    assert actual_request is request
    assert vars(sampling) == dict(
        n=1, temperature=0.0, top_p=1.0, top_k=-1,
        presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0,
        max_tokens=32, seed=17, ignore_eos=False, stop_token_ids=[1, 2],
        skip_special_tokens=True,
    )


def test_long_prompt_uses_the_same_system_question_and_recent_evidence_budget(runtime):
    tokenizer, engine = Tokenizer(), Engine()
    conversation = messages() + [
        {"role": "user", "content": "old evidence " * 100},
        {"role": "assistant", "content": "search"},
        {"role": "user", "content": "NEW EVIDENCE"},
    ]
    policy = VLLMPolicy(engine, tokenizer, max_prompt_tokens=300, max_model_len=400)
    result = policy.generate([conversation], 100, 0)[0]
    assert len(result.prompt_ids) == 300
    assert result.prompt_ids == encode_messages(tokenizer, conversation, 300)
    assert "What is the contract term?" in tokenizer.decode(result.prompt_ids, skip_special_tokens=True)
    assert "NEW EVIDENCE" in tokenizer.decode(result.prompt_ids, skip_special_tokens=True)
    with pytest.raises(ValueError, match="refusing to shorten"):
        policy.generate([conversation], 101, 0)
    assert len(engine.calls) == 1


def test_evaluation_cannot_accidentally_use_training_sampling(runtime):
    policy = VLLMPolicy(Engine(), Tokenizer())
    with pytest.raises(ValueError, match="greedy evaluation only"):
        policy.generate([messages()], 10, 1.0)
    with pytest.raises(ValueError, match="positive integer"):
        policy.generate([messages()], 0, 0)


@pytest.mark.parametrize("damage", ["count", "prompt", "choices", "aborted"])
def test_bad_engine_results_are_rejected(runtime, damage):
    engine = Engine()
    original = engine.generate

    def damaged(*args, **kwargs):
        rows = original(*args, **kwargs)
        if damage == "count":
            return []
        if damage == "prompt":
            rows[0].prompt_token_ids = [999]
        if damage == "choices":
            rows[0].outputs = []
        if damage == "aborted":
            rows[0].outputs[0].finish_reason = "abort"
        return rows

    engine.generate = damaged
    with pytest.raises(RuntimeError):
        VLLMPolicy(engine, Tokenizer()).generate([messages()], 10, 0)


def test_async_requests_batch_and_close_without_overlapping_engine_access(runtime):
    async def run():
        engine = Engine()
        policy = VLLMPolicy(engine, Tokenizer(), batch_size=4)
        results = await asyncio.gather(*(policy(messages(), max_new_tokens=20, temperature=0) for _ in range(4)))
        await policy.close()
        assert policy.worker is None
        assert len(engine.calls) == 1 and len(results) == 4

    asyncio.run(run())


def test_close_shuts_down_engine_core_once_after_draining(runtime):
    async def run():
        engine, calls = Engine(), []
        engine.llm_engine = SimpleNamespace(engine_core=SimpleNamespace(shutdown=lambda: calls.append(True)))
        policy = VLLMPolicy(engine, Tokenizer())
        await policy(messages(), max_new_tokens=20, temperature=0)
        await policy.close()
        await policy.close()
        assert calls == [True]
        with pytest.raises(RuntimeError, match="closed"):
            policy.generate([messages()], 10, 0)

    asyncio.run(run())


@pytest.mark.parametrize("name,value", [("WORLD_SIZE", "8"), ("RANK", "1"), ("LOCAL_RANK", "1")])
def test_multirank_torchrun_is_rejected_before_engine_creation(runtime, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="one launcher process"):
        _validate_launch_environment(1)


def test_gpu_visibility_and_existing_process_group_are_checked(runtime, monkeypatch):
    import torch

    with pytest.raises(ValueError, match="exactly 2 visible GPUs"):
        _validate_launch_environment(2)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    with pytest.raises(ValueError, match="fresh process"):
        _validate_launch_environment(1)


def test_engine_environment_is_restored_even_on_failure(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("MASTER_PORT", "12345")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    with pytest.raises(RuntimeError):
        with _engine_environment():
            assert "RANK" not in os.environ and "MASTER_PORT" not in os.environ
            assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
            raise RuntimeError("engine loading failed")
    assert os.environ["RANK"] == "0" and os.environ["MASTER_PORT"] == "12345"
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "fork"


def adapter_checkpoint(path, **overrides):
    config = dict(peft_type="LORA", target_modules=["q_proj", "v_proj", "o_proj"],
                  r=8, lora_alpha=16, bias="none", base_model_name_or_path="base-model")
    config.update(overrides)
    (path / "adapter_config.json").write_text(json.dumps(config))
    (path / "adapter_model.safetensors").write_bytes(b"test checkpoint placeholder")
    return path


@pytest.mark.parametrize("change", [{"target_modules": ["gate_proj"]}, {"r": 0},
                                     {"r": True}, {"bias": "all"},
                                     {"modules_to_save": ["lm_head"]}, {"use_dora": True},
                                     {"use_rslora": True}, {"rank_pattern": {"q_proj": 16}},
                                     {"alpha_pattern": {"q_proj": 32}}])
def test_unsupported_adapter_is_rejected_before_loading_an_engine(tmp_path, change):
    with pytest.raises(ValueError):
        _adapter_metadata(adapter_checkpoint(tmp_path, **change))


def test_loader_preserves_versions_revision_adapter_and_memory_provenance(runtime, monkeypatch, tmp_path):
    import transformers

    tokenizer = Tokenizer()
    config = SimpleNamespace(model_type="qwen3_moe", _commit_hash="resolved-model-revision")
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **kw: config)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.setattr(transformers.GenerationConfig, "from_pretrained", lambda *a, **kw: SimpleNamespace(eos_token_id=[1, 2]))
    adapter_checkpoint(tmp_path)
    policy = load_vllm_policy("model-id", adapter=str(tmp_path), revision="model-revision",
                             tokenizer_revision="tokenizer-revision", seed=11, batch_size=4,
                             max_prompt_tokens=8192, max_model_len=12288, enforce_eager=True)
    args = policy.model.kwargs
    assert args["dtype"] == "bfloat16" and args["quantization"] is None
    assert args["enable_lora"] and args["max_lora_rank"] == 8
    assert args["lora_dtype"] == "auto"
    assert args["max_num_seqs"] == 4 and args["max_model_len"] == 12288
    assert args["generation_config"] == "vllm" and args["kv_cache_dtype"] == "auto"
    assert args["revision"] == "model-revision" and args["tokenizer_revision"] == "tokenizer-revision"
    assert policy.provenance["adapter"]["config"]["r"] == 8
    assert policy.provenance["adapter"]["weights_bytes"] > 0
    assert policy.provenance["resolved_config_revision"] == "resolved-model-revision"
    assert policy.provenance["seed"] == 11 and policy.eos_token_ids == [1, 2]
    assert policy.provenance["lora_dtype"] == "auto (bfloat16)"
    assert policy.lora_request == ("dcc", 1, str(tmp_path.resolve()))
    json.dumps(policy.provenance, allow_nan=False)


def test_loader_requires_pinned_environment(runtime):
    runtime.__version__ = "0.10.2"
    with pytest.raises(ValueError, match="0.11.2"):
        load_vllm_policy("model-id")
