"""Real tiny-model pruning, exact shape accounting and LoRA round-trip."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

_SPEC = importlib.util.spec_from_file_location("dcc_prune_experts", Path(__file__).resolve().parents[1] / "scripts" / "dcc_prune_experts.py")
prune = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prune)


def tiny_model(*, tied=False):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(19)
    torch.set_num_threads(2)
    config = Qwen3MoeConfig(
        vocab_size=48, hidden_size=16, intermediate_size=24,
        moe_intermediate_size=8, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        num_experts=4, num_experts_per_tok=2, tie_word_embeddings=tied,
    )
    return Qwen3MoeForCausalLM(config).eval()


def calibration(model):
    # Deliberately choose a different per-layer permutation. The original IDs
    # are not assumed to match across layers, so this catches global remaps.
    config = prune.normalize_config(model.config.to_dict())
    value = {
        "schema_version": 1, "kind": "qwen3_moe_routing_calibration", "split": "train",
        "source_model_id": "test/Qwen3MoE", "source_revision": "test-revision",
        "config": {k: config[k] for k in prune._CONFIG_FIELDS}, "task_ids": ["train-0"],
        "layers": {"0": {"counts": [1, 4, 1, 2], "total_tokens": 4}, "1": {"counts": [4, 1, 2, 1], "total_tokens": 4}},
    }
    return value


@pytest.mark.parametrize("tied", [False, True])
def test_real_export_forward_remaps_and_lora_roundtrip(tmp_path, tied):
    pytest.importorskip("peft")
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM
    from safetensors import safe_open

    source = tmp_path / "source"
    original = tiny_model(tied=tied)
    original.save_pretrained(source, max_shard_size="6KB")
    original_count = sum(p.numel() for p in original.parameters())
    config, metadata = prune.checkpoint_metadata(source)
    assert prune.parameter_count(metadata, config) == original_count
    counts = calibration(original)
    plan = prune.pruning_plan(config, metadata, counts, 3)
    assert plan["selected_experts"] == {"0": [0, 1, 3], "1": [0, 1, 2]}
    assert plan["removed_parameters"] == 2 * (3 * 8 * 16 + 16)
    destination = tmp_path / "smaller"
    result = prune.export_checkpoint(source, destination, counts, 3, model_id="test/Qwen3MoE-3E", workers=2)
    assert result["logical_parameters"] == original_count - 800
    assert result["physically_removed_experts"]
    assert not list(tmp_path.glob("smaller.partial-*"))
    exported_config, exported_meta = prune.checkpoint_metadata(destination)
    assert prune.parameter_count(exported_meta, exported_config) == result["logical_parameters"]
    assert exported_config["num_experts"] == 3
    assert exported_config["dcc_expert_pruning"]["source_model_id"] == "test/Qwen3MoE"
    # Compare every exported value to its expected original tensor. All three
    # projections and router rows are checked, including cross-shard remaps.
    for old_name, new_name in plan["name_map"].items():
        with safe_open(source / metadata[old_name]["file"], framework="pt") as stream:
            expected = stream.get_tensor(old_name)
        match = prune._GATE.fullmatch(old_name) or prune._FUSED.fullmatch(old_name)
        if match:
            expected = expected[plan["selected_experts"][str(int(match[1]))]]
        with safe_open(destination / exported_meta[new_name]["file"], framework="pt") as stream:
            actual = stream.get_tensor(new_name)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    restored = AutoModelForCausalLM.from_pretrained(destination).eval()
    assert sum(p.numel() for p in restored.parameters()) == result["logical_parameters"]
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.inference_mode():
        assert torch.isfinite(restored(ids).logits).all()
    restored = get_peft_model(restored, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules=["q_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    parameters = [p for p in restored.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in parameters]
    frozen = {name: p.detach().clone() for name, p in restored.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW(parameters, lr=1e-2)
    restored.train()
    loss = restored(ids, labels=ids).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    assert sum(p.grad.abs().sum().item() for p in parameters) > 0
    optimizer.step()
    assert any(not torch.equal(old, p) for old, p in zip(before, parameters, strict=True))
    assert all(torch.equal(frozen[name], p) for name, p in restored.named_parameters() if name in frozen)
    restored.eval()
    with torch.inference_mode():
        expected_logits = restored(ids).logits.clone()
    adapter = tmp_path / "adapter"
    restored.save_pretrained(adapter)
    reloaded = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(destination), adapter).eval()
    with torch.inference_mode():
        torch.testing.assert_close(reloaded(ids).logits, expected_logits, rtol=1e-5, atol=1e-5)


def test_router_counter_counts_real_masked_training_tokens_and_removes_hooks():
    model = tiny_model()
    counter = prune.RouterCounter(model)
    ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    with torch.inference_mode(), counter.valid_tokens(mask):
        model(ids, attention_mask=mask)
    report = counter.report(source_model_id="test/Qwen3MoE", source_revision="rev", task_ids=["train-a", "train-b"])
    assert all(layer["total_tokens"] == 5 and sum(layer["counts"]) == 10 for layer in report["layers"].values())
    before = {k: v.clone() for k, v in counter.counts.items()}
    counter.close()
    with torch.inference_mode():
        model(ids)
    assert all(torch.equal(before[k], v) for k, v in counter.counts.items())


def test_merge_rejects_test_data_overlaps_and_wrong_dimensions():
    model = tiny_model()
    config = prune.normalize_config(model.config.to_dict())
    a, b = calibration(model), calibration(model)
    b["task_ids"] = ["train-1"]
    merged = prune.merge_calibrations([a, b], config)
    assert merged["layers"]["0"]["counts"] == [2, 8, 2, 4]
    assert merged["layers"]["0"]["total_tokens"] == 8
    with pytest.raises(ValueError, match="overlapping"):
        prune.merge_calibrations([a, a], config)
    b["split"] = "test"
    with pytest.raises(ValueError, match="training-only"):
        prune.merge_calibrations([a, b], config)
    b["split"] = "train"
    b["config"]["num_experts"] = 3
    with pytest.raises(ValueError, match="dimensions"):
        prune.validate_calibration(b, config)
    a["layers"]["0"]["counts"][0] += 1
    with pytest.raises(ValueError, match="sum"):
        prune.validate_calibration(a, config)


def test_plan_rejects_missing_expert_wrong_revision_limit_and_topk(tmp_path):
    model = tiny_model()
    source = tmp_path / "source"
    model.save_pretrained(source)
    config, metadata = prune.checkpoint_metadata(source)
    counts = calibration(model)
    with pytest.raises(ValueError, match="revision mismatch"):
        prune.pruning_plan(config, metadata, counts, 3, source_revision="different-rev")
    with pytest.raises(ValueError, match="exceeding limit"):
        prune.pruning_plan(config, metadata, counts, 3, max_parameters=100)
    with pytest.raises(ValueError, match="top-k"):
        prune.pruning_plan(config, metadata, counts, 1)
    metadata.pop(next(k for k in metadata if ".experts." in k))
    with pytest.raises(ValueError, match="expert matrix"):
        prune.pruning_plan(config, metadata, counts, 3)


def test_export_failure_cleans_staging_and_never_overwrites(tmp_path, monkeypatch):
    import safetensors.torch

    model = tiny_model()
    source, destination = tmp_path / "source", tmp_path / "output"
    model.save_pretrained(source)
    counts = calibration(model)
    def fail(*args, **kwargs):
        raise OSError("simulated storage failure")
    monkeypatch.setattr(safetensors.torch, "save_file", fail)
    with pytest.raises(OSError, match="storage failure"):
        prune.export_checkpoint(source, destination, counts, 3)
    assert not destination.exists()
    assert not list(tmp_path.glob("output.partial-*"))
    destination.mkdir()
    (destination / "user-content").write_text("preserve")
    with pytest.raises(FileExistsError):
        prune.export_checkpoint(source, destination, counts, 3)
    assert (destination / "user-content").read_text() == "preserve"
