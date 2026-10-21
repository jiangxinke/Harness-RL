"""Native CPU admission must exercise the actual tokenizer/grammar backend."""
from __future__ import annotations

import builtins
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def load_trainer():
    spec = importlib.util.spec_from_file_location("dcc_online_grpo_preflight_tests",
        Path(__file__).resolve().parents[1] / "scripts/dcc_online_grpo.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    return trainer


def forbid_runtime(monkeypatch, trainer):
    from transformers import AutoModelForCausalLM
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU preflight must not load weights or initialize CUDA")
    monkeypatch.setattr(trainer, "load_policy_model", forbidden)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", forbidden)
    monkeypatch.setattr(torch.cuda, "set_device", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("error", [ModuleNotFoundError("No module named 'xgrammar'"),
    ImportError("xgrammar binary import failed"), OSError("undefined binary symbol")])
def test_cpu_preflight_rejects_missing_or_broken_native_dependency(monkeypatch, error):
    from transformers import AutoTokenizer
    from slime_plugins.dcc.hf_native_policy import NativeHFPolicy
    trainer = load_trainer()
    forbid_runtime(monkeypatch, trainer)
    tokenizer = object()
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    monkeypatch.setattr(NativeHFPolicy, "validate_tokenizer", lambda value: {})
    def runner_forbidden(*args, **kwargs):
        raise AssertionError("Dependency failure must precede prompt-plan checks")
    monkeypatch.setattr(trainer, "make_runner", runner_forbidden)
    original_import = builtins.__import__
    def import_with_failed_backend(name, *args, **kwargs):
        if name == "xgrammar":
            raise error
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", import_with_failed_backend)
    with pytest.raises(type(error), match=str(error)):
        trainer.cpu_preflight(SimpleNamespace(model="metadata-only"), [], [], 1)
    assert not torch.cuda.is_initialized()


@pytest.fixture
def tiny_tokenizer_metadata(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    from transformers import PreTrainedTokenizerFast, Qwen3Config
    # A real byte-level HF tokenizer; no model or synthetic model facade.
    vocab = {token: index for index, token in enumerate(sorted(
        tokenizers.pre_tokenizers.ByteLevel.alphabet()))}
    for token in ("<|im_end|>", "<pad>", "<|endoftext|>", "<unk>"):
        vocab[token] = len(vocab)
    backend = tokenizers.Tokenizer(tokenizers.models.BPE(vocab, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = tokenizers.decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<|im_end|>",
        pad_token="<pad>", unk_token="<unk>", additional_special_tokens=["<|endoftext|>"])
    tokenizer.save_pretrained(tmp_path)
    Qwen3Config(vocab_size=len(tokenizer), eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id).save_pretrained(tmp_path)
    return tmp_path, tokenizer, vocab["<|endoftext|>"]


@pytest.mark.parametrize("separate_generation_config", [False, True])
def test_real_cpu_compiler_uses_checkpoint_vocab_and_generation_eos(
        tiny_tokenizer_metadata, monkeypatch, separate_generation_config):
    xgr = pytest.importorskip("xgrammar")
    from transformers import AutoTokenizer, GenerationConfig
    from slime_plugins.dcc.cuad_structured import cuad_final_schema
    from slime_plugins.dcc import hf_native_structured
    trainer = load_trainer()
    forbid_runtime(monkeypatch, trainer)
    path, tokenizer, extra_eos = tiny_tokenizer_metadata
    eos_ids = [tokenizer.eos_token_id]
    if separate_generation_config:
        eos_ids.append(extra_eos)
        GenerationConfig(eos_token_id=eos_ids, pad_token_id=tokenizer.pad_token_id).save_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    original_info = xgr.TokenizerInfo.from_huggingface
    original_compile = hf_native_structured.compile_native_structured_schema
    observed = {}
    def info(actual, *, vocab_size, stop_token_ids):
        assert actual is tokenizer and vocab_size == len(tokenizer)
        assert stop_token_ids == eos_ids
        observed["tokenizer_info"] = original_info(actual, vocab_size=vocab_size,
                                                  stop_token_ids=stop_token_ids)
        return observed["tokenizer_info"]
    def compile_schema(compiler, schema, **kwargs):
        assert schema == cuad_final_schema()
        observed["grammar"] = original_compile(compiler, schema, **kwargs)
        return observed["grammar"]
    monkeypatch.setattr(xgr.TokenizerInfo, "from_huggingface", staticmethod(info))
    monkeypatch.setattr(hf_native_structured, "compile_native_structured_schema", compile_schema)
    receipt = trainer.native_grammar_cpu_preflight(path, tokenizer)
    assert receipt["schema_compiled"] and receipt["vocab_size"] == len(tokenizer)
    assert receipt["stop_token_ids"] == eos_ids
    assert receipt["generation_config_source"] == (
        "generation_config.json" if separate_generation_config else "model_config")
    matcher = xgr.GrammarMatcher(observed["grammar"])
    final = json.dumps({"plan": "Read source", "checks": "Checked", "answer": []})
    for token in tokenizer.encode(final, add_special_tokens=False):
        assert matcher.accept_token(token)
    assert matcher.accept_token(tokenizer.eos_token_id) and matcher.is_terminated()
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("contents", [b"{", b"\xff"])
def test_cpu_preflight_rejects_present_corrupt_generation_config(
        tiny_tokenizer_metadata, monkeypatch, contents):
    from transformers import GenerationConfig
    trainer = load_trainer()
    forbid_runtime(monkeypatch, trainer)
    path, tokenizer, _ = tiny_tokenizer_metadata
    (path / "generation_config.json").write_bytes(contents)
    def fallback_forbidden(*args, **kwargs):
        raise AssertionError("A present corrupt generation config must not fall back")
    monkeypatch.setattr(GenerationConfig, "from_model_config", fallback_forbidden)
    with pytest.raises(OSError, match="not a valid JSON file"):
        trainer.native_grammar_cpu_preflight(path, tokenizer)
    assert not torch.cuda.is_initialized()


def test_cpu_preflight_fallback_preserves_legacy_generation_metadata(
        tiny_tokenizer_metadata, monkeypatch):
    from packaging.version import Version
    from transformers import AutoConfig, GenerationConfig, __version__
    trainer = load_trainer()
    forbid_runtime(monkeypatch, trainer)
    path, tokenizer, _ = tiny_tokenizer_metadata
    config_path = path / "config.json"
    metadata = json.loads(config_path.read_text())
    metadata["max_length"] = 73
    config_path.write_text(json.dumps(metadata))
    assert not (path / "generation_config.json").exists()
    original_from_model_config = GenerationConfig.from_model_config
    observed = {}
    def from_model_config(config):
        observed["input"] = config
        observed["generation"] = original_from_model_config(config)
        return observed["generation"]
    monkeypatch.setattr(GenerationConfig, "from_model_config", from_model_config)
    receipt = trainer.native_grammar_cpu_preflight(path, tokenizer)
    assert receipt["generation_config_source"] == "model_config"
    assert observed["generation"].max_length == 73
    if Version(__version__).major >= 5:
        assert isinstance(observed["input"], dict)
        assert observed["input"]["max_length"] == 73
        assert "max_length" not in AutoConfig.from_pretrained(path, local_files_only=True).to_dict()
    else:
        assert isinstance(observed["input"], type(AutoConfig.from_pretrained(
            path, local_files_only=True)))
        assert observed["input"].max_length == 73
    assert not torch.cuda.is_initialized()


def test_cpu_preflight_supplies_missing_pad_from_primary_eos(
        tiny_tokenizer_metadata, monkeypatch):
    trainer = load_trainer()
    forbid_runtime(monkeypatch, trainer)
    path, tokenizer, _ = tiny_tokenizer_metadata
    tokenizer.pad_token = None
    assert tokenizer.pad_token_id is None
    receipt = trainer.native_grammar_cpu_preflight(path, tokenizer)
    assert receipt["schema_compiled"]
    assert tokenizer.pad_token_id == tokenizer.eos_token_id
    assert not torch.cuda.is_initialized()


def test_cpu_preflight_rejects_padding_still_missing_after_fallback(
        tiny_tokenizer_metadata, monkeypatch):
    trainer = load_trainer()
    forbid_runtime(monkeypatch, trainer)
    path, tokenizer, _ = tiny_tokenizer_metadata
    broken = SimpleNamespace(eos_token_id=tokenizer.eos_token_id,
        eos_token=tokenizer.eos_token, pad_token_id=None, pad_token=None)
    with pytest.raises(ValueError, match="An explicit padding token is required"):
        trainer.native_grammar_cpu_preflight(path, broken)
    assert broken.pad_token == tokenizer.eos_token
    assert not torch.cuda.is_initialized()


def test_cpu_preflight_keeps_native_compiler_receipt(monkeypatch):
    from transformers import AutoTokenizer
    from slime_plugins.dcc.hf_native_policy import NativeHFPolicy
    trainer = load_trainer()
    forbid_runtime(monkeypatch, trainer)
    tokenizer = object()
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    monkeypatch.setattr(NativeHFPolicy, "validate_tokenizer", lambda value: {})
    receipt = {"schema_compiled": True, "public_schema": "cuad_final"}
    def grammar(model_name, actual_tokenizer):
        assert model_name == "metadata-only" and actual_tokenizer is tokenizer
        return receipt
    monkeypatch.setattr(trainer, "native_grammar_cpu_preflight", grammar)
    monkeypatch.setattr(trainer, "make_runner", lambda args: SimpleNamespace(
        initial_messages=lambda task: []))
    monkeypatch.setattr(trainer, "encode_messages", lambda *a, **k: [1, 2, 3])
    monkeypatch.setattr(trainer, "configuration", lambda *a: {"unchanged": True})
    task = SimpleNamespace(benchmark="cuad")
    result = trainer.cpu_preflight(SimpleNamespace(model="metadata-only", max_prompt_tokens=32),
                                   [task], [[0]], 1)
    assert result["native_grammar_preflight"] == receipt
    assert result["unique_planned_tasks"] == 1 and result["max_initial_final_prompt"] == 3
    assert result["training_config"] == {"unchanged": True}
    assert result["CUDA_initialized"] is False and not torch.cuda.is_initialized()
