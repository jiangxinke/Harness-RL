"""CPU-only checks for fresh output ownership and truthful terminal receipts."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
from threading import Barrier
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import dcc_training_io as io


def claim(path):
    io.prepare_training_output(path, rank=0, world_size=1)


def checkpoint(path, step=2, *, sync=False):
    def write(destination):
        (destination / "adapter_config.json").write_text('{"peft_type":"LORA"}')
        (destination / "adapter_model.safetensors").write_bytes(b"test fixture only")
        (destination / "training_config.json").write_text('{"steps":2}')
        (destination / "training_state.pt").write_bytes(b"test fixture only")
    io.publish_checkpoint(path, "final", step, write, sync=sync)


def journal(path, rank=0, *, sync=False):
    log = io.TrainingJournal(path, rank, sync=sync)
    log.record(event="start", world_size=1)
    log.record(event="step", step=1, loss=1.5)
    log.record(event="step", step=2, loss=1.0)
    log.record(event="finished", optimizer_steps=2)
    return log


def test_reject_nonempty_output_without_changing_it(tmp_path):
    existing = tmp_path / "existing.txt"
    existing.write_text("preserve this run")
    with pytest.raises(RuntimeError, match="must be empty"):
        claim(tmp_path)
    assert list(tmp_path.iterdir()) == [existing]
    assert existing.read_text() == "preserve this run"


def test_competing_claims_accept_exactly_one_owner(tmp_path):
    gate = Barrier(2)
    def attempt():
        gate.wait()
        try:
            claim(tmp_path)
            return "accepted"
        except RuntimeError:
            return "rejected"
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: attempt(), range(2)))
    assert sorted(results) == ["accepted", "rejected"]
    with pytest.raises(RuntimeError, match="must be empty"):
        claim(tmp_path)


def test_rank_log_exclusivity_and_exact_closed_roundtrip(tmp_path, monkeypatch):
    claim(tmp_path)
    def unexpected_sync(*args):
        raise AssertionError("Default mode must not fsync")
    monkeypatch.setattr(io.os, "fsync", unexpected_sync)
    checkpoint(tmp_path)
    with journal(tmp_path) as log:
        with pytest.raises(FileExistsError):
            io.TrainingJournal(tmp_path, 0)
        io.finalize_training(log, 2, world_size=1)
        assert log.stream.closed
    rows = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert rows == [{"event": "start", "world_size": 1},
                    {"event": "step", "step": 1, "loss": 1.5},
                    {"event": "step", "step": 2, "loss": 1.0},
                    {"event": "finished", "optimizer_steps": 2}]
    receipt = json.loads((tmp_path / io.TRAINING_COMPLETE).read_text())
    assert receipt["logs"] == [{"rank": 0, "file": "train_rank0.jsonl", "records": 4,
                                "optimizer_steps": 2, "closed_log_validated": True}]
    assert receipt["sync_requested"] is False
    original_files = {"adapter_config.json", "adapter_model.safetensors", "training_config.json", "training_state.pt"}
    marker = json.loads((tmp_path / "final" / io.CHECKPOINT_COMPLETE).read_text())
    assert {entry["file"] for entry in marker["files"]} == original_files
    assert {p.name for p in (tmp_path / "final").iterdir()} == original_files | {io.CHECKPOINT_COMPLETE}


def test_sync_is_only_requested_at_checkpoint_or_terminal_boundaries(tmp_path, monkeypatch):
    claim(tmp_path)
    calls = []
    monkeypatch.setattr(io.os, "fsync", lambda descriptor: calls.append(descriptor))
    log = journal(tmp_path, sync=True)
    assert calls == []  # Four writes, including both optimizer steps.
    log.checkpoint_boundary()
    assert len(calls) == 1
    checkpoint(tmp_path)
    io.finalize_training(log, 2, world_size=1)
    assert log.stream.closed and len(calls) == 4  # log, terminal log, receipt file, directory.


@pytest.mark.parametrize("corruption", ["missing_prefix", "duplicate", "reordered", "truncated", "wrong_finish", "failed_event"])
def test_corrupt_closed_log_cannot_publish_training_success(tmp_path, corruption):
    claim(tmp_path)
    checkpoint(tmp_path)
    log = journal(tmp_path)
    log.stream.close()
    rows = log.path.read_text().splitlines(keepends=True)
    if corruption == "missing_prefix":
        rows.pop(1)
    elif corruption == "duplicate":
        rows.insert(2, rows[1])
    elif corruption == "reordered":
        rows[1], rows[2] = rows[2], rows[1]
    elif corruption == "truncated":
        rows[-1] = rows[-1].rstrip("\n")
    elif corruption == "wrong_finish":
        rows[-1] = '{"event":"finished","optimizer_steps":1}\n'
    else:
        rows[1] = '{"event":"failed","error":"injected failure"}\n'
    log.path.write_text("".join(rows))
    with pytest.raises(RuntimeError, match="terminal validation"):
        io.finalize_training(log, 2, world_size=1)
    assert not (tmp_path / io.TRAINING_COMPLETE).exists()


class FaultyStream:
    def __init__(self, stream, *, write=False, close=False):
        self.wrapped, self.fail_write, self.fail_close = stream, write, close
    def __getattr__(self, name):
        return getattr(self.wrapped, name)
    def write(self, value):
        if self.fail_write:
            raise OSError("injected write failure")
        return self.wrapped.write(value)
    def close(self):
        self.wrapped.close()
        if self.fail_close:
            raise OSError("injected close failure")


def test_write_failure_does_not_publish_success(tmp_path):
    claim(tmp_path)
    log = io.TrainingJournal(tmp_path, 0)
    log.stream = FaultyStream(log.stream, write=True)
    with pytest.raises(OSError, match="write failure"):
        with log:
            log.record(event="start")
    assert log.records == 0 and log.stream.closed
    assert not (tmp_path / io.TRAINING_COMPLETE).exists()


@pytest.mark.parametrize("failure", ["close", "fsync"])
def test_terminal_close_or_fsync_failure_blocks_success(tmp_path, monkeypatch, failure):
    claim(tmp_path)
    checkpoint(tmp_path)
    log = journal(tmp_path, sync=failure == "fsync")
    if failure == "close":
        log.stream = FaultyStream(log.stream, close=True)
    else:
        def fail(*args):
            raise OSError("injected fsync failure")
        monkeypatch.setattr(io.os, "fsync", fail)
    with pytest.raises(RuntimeError, match="terminal validation"):
        io.finalize_training(log, 2, world_size=1)
    assert log.stream.closed
    assert not (tmp_path / io.TRAINING_COMPLETE).exists()


def test_receipt_directory_sync_failure_removes_success_marker(tmp_path, monkeypatch):
    claim(tmp_path)
    checkpoint(tmp_path)
    log = journal(tmp_path, sync=True)
    def fail(*args):
        raise OSError("injected receipt directory sync failure")
    monkeypatch.setattr(io, "_sync_directory", fail)
    with pytest.raises(OSError, match="receipt directory sync"):
        io.finalize_training(log, 2, world_size=1)
    assert not (tmp_path / io.TRAINING_COMPLETE).exists()


def test_interrupted_checkpoint_never_publishes_final_directory(tmp_path):
    claim(tmp_path)
    def interrupted(destination):
        (destination / "training_state.pt").write_bytes(b"incomplete")
        raise OSError("injected writer interruption")
    with pytest.raises(OSError, match="writer interruption"):
        io.publish_checkpoint(tmp_path, "final", 2, interrupted)
    assert not (tmp_path / "final").exists()
    incomplete = list(tmp_path.glob(".final.incomplete-*"))
    assert len(incomplete) == 1
    assert (incomplete[0] / "training_state.pt").read_bytes() == b"incomplete"
    assert not (incomplete[0] / io.CHECKPOINT_COMPLETE).exists()


@pytest.mark.parametrize("failure", ["file_sync", "post_rename_directory_sync"])
def test_checkpoint_sync_failure_has_no_ready_marker(tmp_path, monkeypatch, failure):
    claim(tmp_path)
    def fail(*args):
        raise OSError("injected checkpoint sync failure")
    if failure == "file_sync":
        monkeypatch.setattr(io.os, "fsync", fail)
    else:
        original = io._sync_directory
        def fail_parent(path):
            if path == tmp_path:
                fail()
            original(path)
        monkeypatch.setattr(io, "_sync_directory", fail_parent)
    with pytest.raises(OSError, match="checkpoint sync failure"):
        checkpoint(tmp_path, sync=True)
    assert not (tmp_path / "final" / io.CHECKPOINT_COMPLETE).exists()
    assert not (tmp_path / io.TRAINING_COMPLETE).exists()


def test_existing_checkpoint_is_never_replaced(tmp_path):
    claim(tmp_path)
    checkpoint(tmp_path)
    original = {p.name: p.read_bytes() for p in (tmp_path / "final").iterdir()}
    with pytest.raises(FileExistsError):
        checkpoint(tmp_path, step=99)
    assert {p.name: p.read_bytes() for p in (tmp_path / "final").iterdir()} == original


def _distributed_worker(rank, rendezvous, output, corrupt_rank):
    import torch.distributed as dist
    from datetime import timedelta
    output = Path(output)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        io.prepare_training_output(output, rank=rank, world_size=2, distributed=dist)
        if rank == 0:
            checkpoint(output)
        dist.barrier()
        with journal(output, rank=rank) as log:
            if rank == corrupt_rank:
                log.stream.close()
                rows = log.path.read_text().splitlines(keepends=True)
                log.path.write_text("".join([rows[0], *rows[2:]]))
            if corrupt_rank is None:
                io.finalize_training(log, 2, world_size=2, distributed=dist)
            else:
                with pytest.raises(RuntimeError, match="terminal validation"):
                    io.finalize_training(log, 2, world_size=2, distributed=dist)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("corrupt_rank", [None, 1])
def test_two_cpu_ranks_require_both_closed_complete_logs(tmp_path, corrupt_rank):
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("CPU Gloo unavailable")
    torch.multiprocessing.spawn(_distributed_worker,
        args=((tmp_path / "rendezvous").as_uri(), str(tmp_path / "run"), corrupt_rank), nprocs=2, join=True)
    receipt = tmp_path / "run" / io.TRAINING_COMPLETE
    assert receipt.exists() is (corrupt_rank is None)
    if receipt.exists():
        assert {item["rank"] for item in json.loads(receipt.read_text())["logs"]} == {0, 1}


def _load_trainer(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_cpu_trainer_keeps_exact_optimizer_updates_and_checkpoint_layout(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    candidate = _load_trainer(SCRIPTS / "dcc_tool_sft.py", "dcc_io_candidate_trainer")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state", torch.get_rng_state)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.Parameter(torch.tensor([1.0, -0.5]))
        def save_pretrained(self, destination):
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "adapter_config.json").write_text('{"r":1}')
            safetensors.save_file(self.state_dict(), str(destination / "adapter_model.safetensors"))
    class Tokenizer:
        def save_pretrained(self, destination):
            (destination / "tokenizer_config.json").write_text('{}')
    class Sampler:
        def __init__(self, *args, **kwargs):
            self.available, self.cuad, self.cuad_tools = [], {}, {}
            self.cuad_missing_stages, self.skips = [], {}
            self.web_fraction, self.rng = 1.0, random.Random(42)
    def backward(model, examples):
        loss = model.lora_A.square().mean()
        loss.backward()
        return loss.item()
    reference = Model()
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01, weight_decay=0.0)
    expected = {"initial": (deepcopy(reference.state_dict()), deepcopy(reference_optimizer.state_dict()), 0)}
    for step in (1, 2):
        reference_optimizer.zero_grad(set_to_none=True)
        reference.lora_A.square().mean().backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
        reference_optimizer.step()
        expected[f"step_{step:04d}"] = (deepcopy(reference.state_dict()), deepcopy(reference_optimizer.state_dict()), step)
    expected["final"] = expected["step_0002"]
    models = []
    for module in (candidate,):
        model = Model()
        models.append(model)
        monkeypatch.setattr(module, "load_policy_model", lambda *args, _model=model, **kwargs: (_model, Tokenizer()))
        monkeypatch.setattr(module, "load_records", lambda _: [])
        monkeypatch.setattr(module, "load_retrieval_profile", lambda _: None)
        monkeypatch.setattr(module, "EpisodeRunner", lambda **kwargs: None)
        monkeypatch.setattr(module, "MixedSampler", Sampler)
        monkeypatch.setattr(module, "sample_batch", lambda *args, _module=module: [_module.Example("test", "final", "test", [1], [2])])
        monkeypatch.setattr(module, "backward_batch", backward)
        args = SimpleNamespace(native_thinking=False, seed=42, records="unused", cuad_tool_records=None,
            cuad_train_data=None, initial_chunks=1, retrieval_profile=None, output_dir=str(tmp_path / module.__name__),
            adapter=None, model="unused", lora_rank=1, quantization="none", web_fraction=1.0,
            negative_fraction=0.0, max_prompt_tokens=4, max_new_tokens=4, batch_tokens=8,
            cuad_tool_fraction=0.0, learning_rate=0.01, save_initial=True, steps=2,
            samples_per_step=1, max_grad_norm=1.0, save_every=1, sync_artifacts=False)
        module.train(args)
    torch.testing.assert_close(models[0].lora_A, reference.lora_A, rtol=0, atol=0)
    for name, (expected_model, expected_optimizer, step) in expected.items():
        changed = Path(args.output_dir) / name
        assert {p.name for p in changed.iterdir()} == {
            "adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json",
            "training_config.json", "training_state.pt", io.CHECKPOINT_COMPLETE}
        saved_model = safetensors.load_file(str(changed / "adapter_model.safetensors"))
        torch.testing.assert_close(saved_model["lora_A"], expected_model["lora_A"], rtol=0, atol=0)
        saved = torch.load(changed / "training_state.pt", weights_only=False)
        assert saved["optimizer_steps"] == step
        assert saved["sampler_rng"] == random.Random(42).getstate()
        assert saved["optimizer"]["param_groups"] == expected_optimizer["param_groups"]
        for index, values in expected_optimizer["state"].items():
            for key, value in values.items():
                torch.testing.assert_close(value, saved["optimizer"]["state"][index][key], rtol=0, atol=0)
    assert (Path(args.output_dir) / io.TRAINING_COMPLETE).exists()
    with pytest.raises(RuntimeError, match="must be empty"):
        candidate.train(args)
