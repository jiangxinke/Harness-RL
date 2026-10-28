"""Exercise the real one-step CPU trainer with replayable/unreplayable groups."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")
from peft import LoraConfig, get_peft_model
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

# Override only for isolated pre-merge verification against source/candidate.
_SOURCE = Path(os.environ.get("DCC_GRPO_SOURCE", Path(__file__).resolve().parents[1] / "scripts/dcc_train.py"))
_SPEC = importlib.util.spec_from_file_location("dcc_grpo_eligibility_tests", _SOURCE)
_TRAIN = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TRAIN)
from slime_plugins.dcc.episode import EpisodeResult, PolicyCompletion


def _episode(status, reward, *, tokens=True, alternative=False):
    completion = [9, 10, 11] if alternative else [6, 7, 8]
    text = " ".join(map(str, completion))
    messages = [{"role":"system","content":"fixture"},
                {"role":"user","content":"Find the clause"},
                {"role":"user","content":'Initial contract retrieval (source material):\n{"text":"contract"}'},
                {"role":"assistant","content":text}]
    return EpisodeResult(
        task_id="fixture-train", benchmark="cuad", split="train", answer=[],
        status=status, reward=reward, metrics={}, messages=messages if tokens else [], tool_events=[],
        steps=[PolicyCompletion(text, [3, 4, 5], completion)] if tokens else [],
    )


def _tiny_adapter():
    torch.manual_seed(27)
    config = Qwen3MoeConfig(
        vocab_size=48, hidden_size=16, intermediate_size=24,
        moe_intermediate_size=8, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, num_experts=4,
        num_experts_per_tok=2, attention_dropout=0.0,
    )
    model = get_peft_model(Qwen3MoeForCausalLM(config), LoraConfig(
        r=2, lora_alpha=4, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM",
    ))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(mean=0.0, std=0.2)
    return model


def _run_one_step(tmp_path, monkeypatch, results):
    cuda_was_initialized = torch.cuda.is_initialized()
    model = _tiny_adapter()
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    before = {name: value.detach().clone() for name, value in model.named_parameters() if value.requires_grad}
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.setenv(name, "1" if name == "WORLD_SIZE" else "0")
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: torch.get_rng_state())
    task = SimpleNamespace(id="fixture-train", split="train", benchmark="cuad", gold=["fixture"])
    monkeypatch.setattr(_TRAIN, "load_tasks", lambda _: [task])
    tokenizer = SimpleNamespace(save_pretrained=lambda _: None,
        apply_chat_template=lambda messages, **kwargs: [3, 4, 5],
        decode=lambda tokens, **kwargs: " ".join(map(str, tokens)))
    monkeypatch.setattr(_TRAIN, "load_policy_model", lambda *args, **kwargs: (model, tokenizer))
    monkeypatch.setattr(_TRAIN, "load_retrieval_profile", lambda _: None)
    monkeypatch.setattr(_TRAIN, "supervised_example", lambda *args, **kwargs: None)

    class Runner:
        def __init__(self, **kwargs):
            self.calls = 0
        async def run(self, task, policy):
            result = results[self.calls]
            self.calls += 1
            return result

    class Policy:
        def __init__(self, *args, **kwargs):
            pass
        async def close(self):
            pass

    monkeypatch.setattr(_TRAIN, "EpisodeRunner", Runner)
    monkeypatch.setattr(_TRAIN, "HFPolicy", Policy)
    args = SimpleNamespace(
        seed=42, train_data="fixture.jsonl", output_dir=str(tmp_path / "output"),
        model="fixture-base", adapter=None, lora_rank=2, quantization="none",
        sft_steps=0, rl_steps=1, learning_rate=1e-3, max_steps=1,
        max_new_tokens=8, initial_chunks=1, retrieval_profile=None,
        group_size=len(results), max_prompt_tokens=32, sft_coefficient=0.0,
        save_every=0, kl_coefficient=0.0,
    )
    asyncio.run(_TRAIN.train(args))
    report = next(json.loads(line) for line in (tmp_path / "output/train_rank0.jsonl").read_text().splitlines()
                  if json.loads(line).get("stage") == "grpo")
    movement = sum((value.detach() - before[name]).abs().sum().item()
                   for name, value in model.named_parameters() if value.requires_grad)
    assert torch.cuda.is_initialized() == cuda_was_initialized
    return report, movement


@pytest.mark.parametrize("status,tokens", [
    ("model_protocol_failure", True),
    ("context_budget_failure", True),
    ("infrastructure_error", True),
    ("completed", False),
])
def test_unreplayable_members_do_not_leave_unmatched_positive_update(tmp_path, monkeypatch, status, tokens):
    report, movement = _run_one_step(tmp_path, monkeypatch, [
        _episode("completed", 1.0), _episode(status, 0.0, tokens=tokens),
    ])
    assert report["reward_std"] == 0.0
    assert report["policy_segments"] == 0
    assert report["gradient_norm"] == 0.0
    assert movement == 0.0


@pytest.mark.parametrize("failure_status", ["completed", "invalid_answer", "model_evidence_failure"])
def test_replayable_success_failure_pair_has_real_lora_update(tmp_path, monkeypatch, failure_status):
    report, movement = _run_one_step(tmp_path, monkeypatch, [
        _episode("completed", 1.0), _episode(failure_status, 0.0, alternative=True),
    ])
    assert report["reward_std"] == 0.5
    assert report["policy_segments"] == 2
    assert report["gradient_norm"] > 0.0
    assert movement > 0.0


def test_all_unreplayable_group_has_no_policy_update(tmp_path, monkeypatch):
    report, movement = _run_one_step(tmp_path, monkeypatch, [
        _episode("model_protocol_failure", 0.0), _episode("context_budget_failure", 0.0),
    ])
    assert report["reward_std"] == 0.0
    assert report["policy_segments"] == 0
    assert report["gradient_norm"] == 0.0
    assert movement == 0.0
