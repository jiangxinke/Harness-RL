"""Architecture-specific loader regressions without downloads or a GPU."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from slime_plugins.dcc.hf_policy import load_policy_model


def test_real_dense_qwen3_loads_and_generates_without_moe_keyword(tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(17)
    config = Qwen3Config(
        vocab_size=16, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=64,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    source = tmp_path / "tiny-dense-qwen3"
    Qwen3ForCausalLM(config).save_pretrained(source)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}, unk_token="[UNK]")),
        pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]",
    )
    tokenizer.save_pretrained(source)
    original_loader = AutoModelForCausalLM.from_pretrained
    calls = []

    def load_on_cpu(name, **kwargs):
        # Only redirect device placement. In particular, never discard an
        # experts keyword: the actual dense model must reject a regression.
        assert "experts_implementation" not in kwargs
        assert kwargs["dtype"] == torch.bfloat16
        assert kwargs["attn_implementation"] == "sdpa"
        kwargs["device_map"] = {"": "cpu"}
        calls.append(name)
        return original_loader(name, **kwargs)

    monkeypatch.setattr(torch.cuda, "set_device", lambda rank: None)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", load_on_cpu)
    model, restored_tokenizer = load_policy_model(str(source))
    assert isinstance(model, Qwen3ForCausalLM)
    assert model.dtype == torch.bfloat16
    assert restored_tokenizer.padding_side == "left"
    model.eval()
    prompt = torch.tensor([[1, 3, 4]])
    with torch.inference_mode():
        generated = model.generate(
            input_ids=prompt, attention_mask=torch.ones_like(prompt),
            max_new_tokens=2, do_sample=False, pad_token_id=0,
        )
    assert generated.shape[0] == 1 and 3 < generated.shape[1] <= 5
    torch.testing.assert_close(generated[:, :3], prompt)
    assert calls == [str(source)]


def test_real_moe_config_preserves_grouped_mm_dispatch(tmp_path, monkeypatch):
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3MoeConfig
    from slime_plugins.dcc import hf_quantization

    source = tmp_path / "tiny-moe-config"
    Qwen3MoeConfig(num_experts=4).save_pretrained(source)
    captured = {}
    sentinel = object()

    def capture(name, **kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(torch.cuda, "set_device", lambda rank: None)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda name: SimpleNamespace(pad_token_id=0))
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", capture)
    monkeypatch.setattr(hf_quantization, "nf4_load_kwargs", lambda: pytest.fail("BF16 must not enter NF4"))
    model, _ = load_policy_model(str(source))
    assert model is sentinel
    assert captured["experts_implementation"] == "grouped_mm"
    assert captured["dtype"] == torch.bfloat16
    assert captured["attn_implementation"] == "sdpa"
    assert "quantization_config" not in captured


def test_nf4_dispatch_does_not_enter_bf16_architecture_selection(monkeypatch):
    import peft
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from slime_plugins.dcc import hf_quantization

    model = SimpleNamespace()
    quantization_config = object()
    report = {"mode": "nf4", "expert_4bit_coverage": 1.0}
    captured, reports, preparations = {}, [], []

    def capture(name, **kwargs):
        captured.update(kwargs)
        return model

    def prepare(value, **kwargs):
        preparations.append((value, kwargs))
        return value

    def inspect(value):
        reports.append(value)
        return report

    monkeypatch.setattr(torch.cuda, "set_device", lambda rank: None)
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda name: pytest.fail("NF4 must keep its separate loader path"))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda name: SimpleNamespace(pad_token_id=0))
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", capture)
    monkeypatch.setattr(hf_quantization, "nf4_load_kwargs", lambda: {"quantization_config": quantization_config})
    monkeypatch.setattr(hf_quantization, "quantization_report", inspect)
    monkeypatch.setattr(peft, "prepare_model_for_kbit_training", prepare)
    loaded, _ = load_policy_model("unused", quantization="nf4")
    assert loaded is model and model.dcc_quantization_report == report
    assert captured["quantization_config"] is quantization_config
    assert "experts_implementation" not in captured
    assert reports == [model, model]
    assert preparations == [(model, {"use_gradient_checkpointing": False, "gradient_checkpointing_kwargs": {"use_reentrant": False}})]
