"""Real tiny-model whole loop and actual eight-rank CPU Gloo collectives."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "scripts"),
                str(Path(__file__).resolve().parents[1] / "slime")]

import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("peft")
from peft import LoraConfig, get_peft_model
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
import torch.distributed as dist
import torch.multiprocessing as mp

from dcc_native_dpo import OfflineNativeDPOTrainer, ReferenceIdentity, _sequence_sum, cache_reference_scores, dpo_loss
from dcc_native_dpo_data import PreferencePair, PublicFinalPrefix
from dcc_native_dpo_train import (
    GlobalPairDeck, QUANTIZATION_ID, collective_dpo_step, parser, run_dpo_loop,
    synchronize_pair_gradients, validate_fresh_nf4_lora,
)
import dcc_native_dpo_train as TRAIN
from slime_plugins.dcc.hf_policy import completion_logprobs

_SPEC = importlib.util.spec_from_file_location("native_dpo_math_fixture",
    Path(__file__).with_name("test_dcc_native_dpo.py"))
_FIXTURE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FIXTURE)


def model():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    torch.manual_seed(27)
    config = Qwen3MoeConfig(vocab_size=151670, hidden_size=8, intermediate_size=12,
        moe_intermediate_size=4, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=4, num_experts=4, num_experts_per_tok=2,
        attention_dropout=0.0)
    result = get_peft_model(Qwen3MoeForCausalLM(config), LoraConfig(r=16, lora_alpha=32,
        lora_dropout=0, target_modules=["q_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    result.enable_input_require_grads()
    result.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    assert all(parameter.device.type == "cpu" for parameter in result.parameters())
    return result


def pairs(count=5):
    rows = []
    for index in range(count):
        prefix = replace(_FIXTURE._candidate().prefix,
            prefix_id=f"synthetic-prefix-{index}", task_id=f"synthetic-task-{index}")
        chosen = _FIXTURE._candidate(seed=2 * index + 1, reward=1, prefix=prefix,
            prompt=(3 + index, 4, 151667), completion=(6 + index, 151668, 7, 8, 151645))
        rejected = _FIXTURE._candidate(seed=2 * index + 2, reward=0, prefix=prefix,
            prompt=chosen.sequence.prompt, completion=(9, 10 + index, 151668, 11, 12, 151645))
        rows.append(PreferencePair(chosen, rejected))
    return rows


def identity():
    return ReferenceIdentity("synthetic-base", "synthetic-tokenizer", "CPU-FP32-test-only", "base")


class SavedTokenizer:
    def save_pretrained(self, destination):
        (destination / "tokenizer_config.json").write_text('{"synthetic_cpu_test":true}\n')


def parameters(value):
    return {name: tensor.detach().clone() for name, tensor in value.named_parameters() if tensor.requires_grad}


def direct_update(value, batch, references, optimizer):
    optimizer.zero_grad(set_to_none=True)
    loss = sum(dpo_loss(_sequence_sum(value, pair.chosen.sequence, completion_logprobs),
        _sequence_sum(value, pair.rejected.sequence, completion_logprobs), references[pair.key], beta=.1)
        for pair in batch) / len(batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in value.parameters() if p.requires_grad], 1.)
    optimizer.step()
    return float(loss.detach())


def test_real_tiny_qwen_moe_whole_loop_matches_direct_and_publishes_complete_artifacts(tmp_path):
    actual, rows = model(), pairs(3)
    expected = deepcopy(actual)
    before = parameters(actual)
    references = cache_reference_scores(expected, rows, identity(), logprob_fn=completion_logprobs)
    optimizer = torch.optim.AdamW([p for p in expected.parameters() if p.requires_grad], lr=1e-6, weight_decay=0.)
    deck = GlobalPairDeck(len(rows), seed=42)
    for _ in range(2):
        direct_update(expected, [rows[i] for i in deck.next(2)], references, optimizer)
    output = tmp_path / "fresh"
    trainer = run_dpo_loop(actual, SavedTokenizer(), rows, identity(), output=output,
        steps=2, pairs_per_step=2, save_initial=True, save_every=1,
        config={"training_source_commit": "e" * 40, "collection_source_commit": "f" * 40})
    assert trainer.optimizer_steps == 2
    assert any(not torch.equal(value, before[name]) for name, value in parameters(actual).items())
    for name, value in parameters(actual).items():
        torch.testing.assert_close(value, parameters(expected)[name], atol=2e-9, rtol=2e-5)
    logs = [json.loads(line) for line in (output / "train_rank0.jsonl").read_text().splitlines()]
    updates = [row for row in logs if row["event"] == "step"]
    assert [row["step"] for row in updates] == [1, 2]
    assert [row["global_pairs"] for row in updates] == [2, 1]
    assert [row["local_pairs"] for row in updates] == [2, 1]
    assert all(row["measured_KL"] is False for row in updates)
    terminal = json.loads((output / "training_complete.json").read_text())
    assert terminal["complete"] is True and terminal["optimizer_steps"] == 2
    for name, step in (("initial", 0), ("step_0001", 1), ("step_0002", 2), ("final", 2)):
        checkpoint = json.loads((output / name / "checkpoint_complete.json").read_text())
        assert checkpoint["complete"] is True and checkpoint["optimizer_steps"] == step
        assert (output / name / "adapter_model.safetensors").is_file()
    state = torch.load(output / "final" / "training_state.pt", weights_only=False, map_location="cpu")
    assert state["optimizer_steps"] == 2 and "cuda_rng" not in state and state["resume_supported"] is False
    config = json.loads((output / "final" / "training_config.json").read_text())
    assert config["reference"]["mode"] == "base" and config["reward_contract"].endswith("receipt-v1")
    assert config["training_method"].startswith("offline native final-decision DPO")
    assert str(tmp_path) not in json.dumps(config)
    assert not torch.cuda.is_initialized()


def test_whole_loop_refuses_existing_output_before_adapter_update(tmp_path):
    value = model()
    before = parameters(value)
    (tmp_path / "existing.txt").write_text("keep")
    with pytest.raises(RuntimeError, match="must be empty"):
        run_dpo_loop(value, SavedTokenizer(), pairs(1), identity(), output=tmp_path, steps=1)
    assert all(torch.equal(tensor, parameters(value)[name]) for name, tensor in before.items())


def test_cli_has_no_warm_resume_quantization_or_scorer_bypass():
    flags = {flag for action in parser()._actions for flag in action.option_strings}
    assert not flags.intersection({"--adapter", "--resume", "--quantization", "--chunked-logprobs"})
    assert parser().get_default("beta") == .1 and parser().get_default("learning_rate") == 1e-6


def test_production_gate_rejects_warm_or_unquantized_model():
    value = model()
    with pytest.raises(ValueError, match="NF4"):
        validate_fresh_nf4_lora(value)
    # This is a synthetic report solely to exercise configuration gates; it
    # does NOT claim this CPU model is quantized or validate real NF4 loading.
    value.dcc_quantization_report = {"mode": "nf4", "transformers_version": "4.57.6",
        "expert_4bit_coverage": 1., "expert_parameters": 96, "expert_4bit_parameters": 96,
        "expert_4bit_matrices": 12}
    validate_fresh_nf4_lora(value)
    with torch.no_grad():
        next(p for name, p in value.named_parameters() if "lora_B" in name).fill_(1.)
    with pytest.raises(ValueError, match="zero-initialized"):
        validate_fresh_nf4_lora(value)


def _eight_rank_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized()
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=8,
                            timeout=timedelta(seconds=90))
    try:
        value, rows = model(), pairs(5)
        # Genuine unequal counts, including four ranks without local pairs.
        owners = [0, 0, 1, 3, 6]
        local = [pair for pair, owner in zip(rows, owners) if owner == rank]
        trainer = OfflineNativeDPOTrainer(value, local, identity(), logprob_fn=completion_logprobs,
            optimizer_factory=lambda p: torch.optim.AdamW(p, lr=1e-6, weight_decay=0.), world_size=8,
            gradient_sync=lambda p, world: synchronize_pair_gradients(p, world, distributed=dist))
        report = collective_dpo_step(trainer, local, distributed=dist)
        after = parameters(value)
        state_steps = [int(s["step"]) for s in trainer.optimizer.state.values()]
        skipped = collective_dpo_step(trainer, [], distributed=dist)
        assert skipped["optimizer_step_skipped"] is True and trainer.optimizer_steps == 1
        assert all(torch.equal(tensor, parameters(value)[name]) for name, tensor in after.items())
        assert [int(s["step"]) for s in trainer.optimizer.state.values()] == state_steps
        assert not torch.cuda.is_initialized()
        torch.save({"parameters": after, "report": report, "skipped": skipped},
                   Path(directory) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_actual_eight_gloo_ranks_unequal_pairs_empty_ranks_and_zero_global_skip(tmp_path):
    require_cuda_hidden = os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert require_cuda_hidden and not torch.cuda.is_initialized()
    mp.start_processes(_eight_rank_worker, args=((tmp_path / "gloo-rendezvous").as_uri(), str(tmp_path)),
                       nprocs=8, join=True, start_method="fork")
    expected, rows = model(), pairs(5)
    references = cache_reference_scores(expected, rows, identity(), logprob_fn=completion_logprobs)
    optimizer = torch.optim.AdamW([p for p in expected.parameters() if p.requires_grad], lr=1e-6, weight_decay=0.)
    loss = direct_update(expected, rows, references, optimizer)
    expected_parameters = parameters(expected)
    reports = []
    for rank in range(8):
        result = torch.load(tmp_path / f"rank{rank}.pt", weights_only=False, map_location="cpu")
        reports.append(result["report"])
        assert result["report"]["global_pairs"] == 5 and result["report"]["optimizer_steps"] == 1
        assert result["skipped"]["global_pairs"] == 0 and result["skipped"]["optimizer_step_skipped"] is True
        assert result["report"]["global_mean_dpo_loss"] == pytest.approx(loss, abs=1e-7)
        for name, tensor in result["parameters"].items():
            torch.testing.assert_close(tensor, expected_parameters[name], atol=2e-9, rtol=2e-5)
    assert [r["local_pairs"] for r in reports] == [2, 1, 0, 1, 0, 0, 1, 0]
    assert not torch.cuda.is_initialized()


def test_public_prefix_schema_roundtrip_preserves_decoder_property_order():
    value = _FIXTURE._candidate().prefix.to_dict()
    value["json_schema"] = {"type": "object", "properties": {
        "plan": {"type": "array"}, "checks": {"type": "array"}, "answer": {"type": "array"}},
        "required": ["plan", "checks", "answer"], "additionalProperties": False}
    prefix = PublicFinalPrefix.from_dict(value)
    assert list(json.loads(prefix.schema_json)["properties"]) == ["plan", "checks", "answer"]
    assert list(prefix.to_dict()["json_schema"]["properties"]) == ["plan", "checks", "answer"]


def test_no_reward_file_access_before_all_generation_closures(monkeypatch):
    tokenizer, native, prefix, _, _ = _FIXTURE._physical_fixture()
    opened = []
    def not_closed(*args, **kwargs):
        raise ValueError("missing rank closure")
    monkeypatch.setitem(sys.modules, "dcc_collect_native_preferences", SimpleNamespace(
        load_frozen=lambda *args: ({}, [prefix], native), read_closed_g4=not_closed))
    monkeypatch.setattr(TRAIN, "read_train_rewards", lambda *args, **kwargs: opened.append(True))
    args = SimpleNamespace(approved_freeze="synthetic-freeze", generation_dir="synthetic-generation",
        generation_world_size=8, expected_source="f" * 40, train_rewards="never-open",
        reward_version="synthetic-v1", base_derivative_id=native.derivative_id)
    with pytest.raises(ValueError, match="missing rank closure"):
        TRAIN.load_dpo_pairs(args, tokenizer)
    assert opened == []


def test_training_source_requires_actual_clean_committed_runtime(monkeypatch):
    calls = []
    def git(argv, **kwargs):
        calls.append(argv)
        output = "e" * 40 if "rev-parse" in argv else ""
        if "status" in argv:
            output = "?? scripts/dcc_native_dpo_train.py"
        return SimpleNamespace(returncode=0, stdout=output)
    monkeypatch.setattr(TRAIN.subprocess, "run", git)
    with pytest.raises(ValueError, match="modified or untracked"):
        TRAIN.training_source_identity("e" * 40)
    with pytest.raises(ValueError, match="runtime commit differs"):
        TRAIN.training_source_identity("f" * 40)
    assert any("ls-files" in argv for argv in calls)
