"""CPU regressions for dropout compatibility of imported on-policy adapters."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

_SPEC = importlib.util.spec_from_file_location(
    "dcc_grpo_dropout_tests", Path(__file__).resolve().parents[1] / "scripts/dcc_train.py"
)
_TRAIN = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TRAIN)


@pytest.fixture
def restored_adapter(tmp_path):
    def restore(dropout: float, *, attention_dropout: float = 0.0):
        torch.manual_seed(27)
        config = Qwen3MoeConfig(
            vocab_size=48, hidden_size=16, intermediate_size=24,
            moe_intermediate_size=8, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            num_experts=4, num_experts_per_tok=2,
            attention_dropout=attention_dropout,
        )
        base = Qwen3MoeForCausalLM(config)
        untouched_base = deepcopy(base)
        adapter = get_peft_model(
            base,
            LoraConfig(
                r=2, lora_alpha=4, lora_dropout=dropout,
                target_modules=["q_proj", "v_proj", "o_proj"],
                task_type="CAUSAL_LM",
            ),
        )
        # Real trained adapters have nonzero B. Initial zero B hides dropout.
        with torch.no_grad():
            for name, parameter in adapter.named_parameters():
                if "lora_B" in name:
                    parameter.normal_(mean=0.0, std=0.2)
        saved = tmp_path / "imported-adapter"
        adapter.save_pretrained(saved)
        return PeftModel.from_pretrained(untouched_base, saved, is_trainable=True)

    return restore


def _old_and_current(model):
    model.eval()
    with torch.no_grad():
        old = _TRAIN.completion_logprobs(model, [3, 4, 5], [6, 7, 8]).detach()
    model.train()
    torch.manual_seed(41)
    current = _TRAIN.completion_logprobs(model, [3, 4, 5], [6, 7, 8])
    return old, current


def test_imported_dropout_breaks_preupdate_ratio_and_is_rejected(restored_adapter):
    model = restored_adapter(0.35)
    assert model.peft_config["default"].lora_dropout == 0.35
    old, current = _old_and_current(model)
    assert not torch.allclose((current - old).exp(), torch.ones_like(old), atol=1e-5)
    with pytest.raises(ValueError, match=r"On-policy GRPO.*lora_dropout.*--rl-steps 0"):
        _TRAIN.validate_grpo_dropout(model, rl_steps=1)
    assert model.peft_config["default"].lora_dropout == 0.35
    assert any(
        isinstance(module, torch.nn.Dropout) and module.p == 0.35
        for module in model.modules()
    )


def test_zero_dropout_import_preserves_onpolicy_ratio_and_gradient(restored_adapter):
    model = restored_adapter(0.0)
    _TRAIN.validate_grpo_dropout(model, rl_steps=1)
    old, current = _old_and_current(model)
    torch.testing.assert_close((current - old).exp(), torch.ones_like(old))
    _TRAIN.grpo_loss(current, old, 1.0).backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum().item() for gradient in gradients) > 0


def test_sft_only_import_keeps_dropout_and_supports_backward(restored_adapter):
    model = restored_adapter(0.35)
    _TRAIN.validate_grpo_dropout(model, rl_steps=0)
    model.train()
    loss = -_TRAIN.completion_logprobs(model, [3, 4, 5], [6, 7, 8]).mean()
    loss.backward()
    assert model.peft_config["default"].lora_dropout == 0.35
    assert any(
        isinstance(module, torch.nn.Dropout) and module.p == 0.35 and module.training
        for module in model.modules()
    )
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum().item() for gradient in gradients) > 0


def test_functional_qwen_attention_dropout_is_also_rejected(restored_adapter):
    model = restored_adapter(0.0, attention_dropout=0.25)
    old, current = _old_and_current(model)
    assert not torch.allclose((current - old).exp(), torch.ones_like(old), atol=1e-5)
    with pytest.raises(ValueError, match=r"attention_dropout=0\.25"):
        _TRAIN.validate_grpo_dropout(model, rl_steps=1)
    _TRAIN.validate_grpo_dropout(model, rl_steps=0)


def test_incompatible_import_fails_before_optimizer_creation(restored_adapter, tmp_path, monkeypatch):
    model = restored_adapter(0.35)
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.setenv(name, "1" if name == "WORLD_SIZE" else "0")
    monkeypatch.setattr(torch.cuda, "set_device", lambda rank: None)
    monkeypatch.setattr(_TRAIN, "load_tasks", lambda path: [SimpleNamespace(split="train", benchmark="cuad")])
    monkeypatch.setattr(_TRAIN, "load_policy_model", lambda *args, **kwargs: (model, object()))

    def forbidden_optimizer(*args, **kwargs):
        pytest.fail("incompatible GRPO model reached optimizer construction")

    monkeypatch.setattr(torch.optim, "AdamW", forbidden_optimizer)
    args = SimpleNamespace(
        seed=42, train_data="fixture.jsonl", output_dir=str(tmp_path / "output"),
        model="fixture-base", adapter="fixture-adapter", lora_rank=2,
        quantization="none", sft_steps=1, rl_steps=1,
    )
    with pytest.raises(ValueError, match="On-policy GRPO requires zero dropout"):
        asyncio.run(_TRAIN.train(args))
