"""Optional real CPU NF4 expert coverage, gradients and adapter round-trip.

Run with configs/dcc/requirements-quantized.txt in a separate environment.
No model download or CUDA initialization is needed.
"""

from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
bnb = pytest.importorskip("bitsandbytes")
pytest.importorskip("peft")

if transformers.__version__ != "4.57.6":
    pytest.skip("Full-expert NF4 uses its separate pinned environment", allow_module_level=True)

_POLICY = importlib.import_module("slime_plugins.dcc.hf_policy")
completion_logprobs = _POLICY.completion_logprobs
load_policy_model = _POLICY.load_policy_model
_QUANTIZATION = importlib.import_module("slime_plugins.dcc.hf_quantization")
nf4_load_kwargs = _QUANTIZATION.nf4_load_kwargs
quantization_report = _QUANTIZATION.quantization_report


def test_nf4_refuses_fused_transformers_before_model_load(monkeypatch):
    monkeypatch.setattr(importlib.import_module("transformers"), "__version__", "5.20.1")
    with pytest.raises(ValueError, match="fused MoE"):
        nf4_load_kwargs()


def test_full_expert_nf4_gradients_and_adapter_roundtrip(tmp_path, monkeypatch):
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(17)
    torch.set_num_threads(2)
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=2,
        num_experts_per_tok=1,
    )
    original = Qwen3MoeForCausalLM(config)
    original_parameters = sum(parameter.numel() for parameter in original.parameters())
    source = tmp_path / "tiny-base"
    original.save_pretrained(source)
    del original
    original_loader = AutoModelForCausalLM.from_pretrained
    calls = []

    def load_on_cpu(name, **kwargs):
        assert "experts_implementation" not in kwargs
        assert kwargs["quantization_config"].bnb_4bit_use_double_quant
        assert kwargs["quantization_config"].bnb_4bit_compute_dtype == torch.bfloat16
        calls.append(name)
        kwargs["device_map"] = {"": "cpu"}
        return original_loader(name, **kwargs)

    class Tokenizer:
        padding_side = "right"
        pad_token_id = 0

    monkeypatch.setattr(torch.cuda, "set_device", lambda rank: None)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda name: Tokenizer())
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", load_on_cpu)
    model, tokenizer = load_policy_model(str(source), trainable=True, lora_rank=2, quantization="nf4")
    assert tokenizer.padding_side == "left"
    report = model.dcc_quantization_report
    assert report["logical_base_parameters"] == original_parameters
    assert report["expert_4bit_coverage"] == 1.0
    assert report["expert_4bit_matrices"] == 12
    assert report["expert_parameters"] == 2 * 2 * 3 * 32 * 32
    assert report["parameter_storage_bytes"] > 0
    assert report["quantization_metadata_bytes"] > 0
    assert 0 < report["fp32_fraction_of_logical_parameters"] < 1
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in name for name, _ in trainable)
    assert model.get_base_model().model.layers[0].mlp.gate.weight.dtype == torch.float32
    expert = model.get_base_model().model.layers[0].mlp.experts[0]
    assert all(isinstance(getattr(expert, name), bnb.nn.Linear4bit) for name in ("gate_proj", "up_proj", "down_proj"))
    frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    before = {name: p.detach().clone() for name, p in trainable}
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=1e-3)
    model.train()
    loss = -completion_logprobs(model, [1, 2, 3], [4, 5]).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in trainable)
    assert sum(p.grad.abs().sum().item() for _, p in trainable) > 0
    optimizer.step()
    assert any(not torch.equal(before[name], p) for name, p in trainable)
    assert all(torch.equal(frozen[name], p) for name, p in model.named_parameters() if name in frozen)
    model.eval()
    expected = completion_logprobs(model, [1, 2, 3], [4, 5]).detach()
    adapter = tmp_path / "adapter"
    model.save_pretrained(adapter)
    restored, _ = load_policy_model(str(source), adapter=str(adapter), quantization="nf4")
    restored.eval()
    actual = completion_logprobs(restored, [1, 2, 3], [4, 5]).detach()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert len(calls) == 2
    assert not any(parameter.requires_grad for parameter in restored.parameters())
    assert restored.dcc_quantization_report["expert_4bit_coverage"] == 1.0

    # A supposedly quantized model must fail rather than silently retaining a
    # large BF16/FP32 expert matrix, including after a future loader change.
    base = restored.get_base_model()
    base.model.layers[0].mlp.experts[0].gate_proj = torch.nn.Linear(32, 32, bias=False)
    with pytest.raises(ValueError, match="escaped NF4"):
        quantization_report(base)


def test_default_loader_keeps_grouped_mm_without_importing_nf4(monkeypatch):
    from types import SimpleNamespace
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    class Tokenizer:
        pad_token_id = 0

    sentinel = object()
    captured = {}

    def load(name, **kwargs):
        captured.update(kwargs)
        return sentinel

    quantization = importlib.import_module("slime_plugins.dcc.hf_quantization")
    monkeypatch.setattr(quantization, "nf4_load_kwargs", lambda: pytest.fail("default imported NF4 loader"))
    monkeypatch.setattr(torch.cuda, "set_device", lambda rank: None)
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda name: SimpleNamespace(num_experts=4))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda name: Tokenizer())
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", load)
    model, _ = load_policy_model("unused")
    assert model is sentinel
    assert captured["experts_implementation"] == "grouped_mm"
    assert "quantization_config" not in captured
