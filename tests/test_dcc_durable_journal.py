"""Real CPU filesystem and episode tests; no weights, GPU, network or labels."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from dcc_collect_native_train import Journal, RecordingPolicy, _finish_collection
from slime_plugins.dcc.data import Task
from slime_plugins.dcc.durable_journal import OrderedDurableJSONLWriter
from slime_plugins.dcc.episode import EpisodeRunner, PolicyCompletion
from slime_plugins.dcc.native_thinking_eval import NativeTraceJournal


class FastNative:
    def __init__(self):
        self.trace_index = {}
        self.completed = False
        self.budget = SimpleNamespace(protocol="synthetic-cpu-only")

    async def generate_structured(self, messages, **kwargs):
        await asyncio.sleep(.002)
        self.completed = True
        return PolicyCompletion('{"answer": []}', [1], [2], finish_reason="stop")


def test_slow_real_fsync_does_not_enter_recording_policy_deadline(tmp_path, monkeypatch):
    real_fsync = os.fsync
    writes = []

    def slow_fsync(descriptor):
        writes.append(threading.get_ident())
        time.sleep(.08)
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", slow_fsync)
    journal = Journal(tmp_path / "requests.jsonl")
    native = FastNative()
    policy = RecordingPolicy(native, journal, task_id="cpu-only", rank=0, episode_id="cpu-only")
    runner = EpisodeRunner(max_steps=1, initial_chunks=0, timeout_seconds=.05,
                           cuad_structured_actions=True)
    task = Task("cpu-only", "cuad", "train", "Find a clause.", document="A contract.")
    try:
        result = asyncio.run(runner.run(task, policy, evaluation=True, score_train=False))
    finally:
        journal.close()
    assert result.status == "completed"
    assert native.completed
    assert journal.count == 2
    assert len((tmp_path / "requests.jsonl").read_text().splitlines()) == 2
    assert journal.stream.closed
    assert len(writes) == 3  # Each record and final close really reached fsync.
    assert set(writes) == {journal._worker.ident}
    assert journal._worker.ident != threading.get_ident()


def test_native_trace_slow_real_fsync_allows_event_loop_deadline(tmp_path, monkeypatch):
    real_fsync = os.fsync
    entered, release = threading.Event(), threading.Event()

    def held_fsync(descriptor):
        entered.set()
        assert release.wait(5)
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", held_fsync)
    journal = NativeTraceJournal(tmp_path / "traces.jsonl")

    async def run():
        journal({"trace_id": "synthetic"})
        # Writer storage is held, yet the event loop must still execute its timer.
        async with asyncio.timeout(.02):
            await asyncio.sleep(1)

    try:
        with pytest.raises(TimeoutError):
            asyncio.run(run())
        assert entered.wait(2)
        assert journal.count == 0
    finally:
        release.set()
        journal.close()
    assert journal.count == 1


def test_order_is_linearized_across_concurrent_producers_and_bytes_are_exact(tmp_path):
    journal = OrderedDurableJSONLWriter(tmp_path / "records.jsonl")
    expected, lock = {}, threading.Lock()

    def producer(index):
        result = {}
        for ordinal in range(20):
            line = json.dumps({"producer": index, "ordinal": ordinal, "text": "合同𝛼"},
                              ensure_ascii=False) + "\n"
            sequence = journal.write_line(line)
            result[sequence] = line
        with lock:
            expected.update(result)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(producer, range(8)))
    journal.close()
    assert (tmp_path / "records.jsonl").read_bytes() == "".join(
        expected[index] for index in range(160)).encode("utf-8")
    assert journal.count == 160
    journal.close()
    with pytest.raises(ValueError, match="closed"):
        journal.write_line("{}\n")


def test_serializers_preserve_their_original_json_bytes_and_snapshot(tmp_path):
    value = {"z": "合同", "a": {"second": 2, "first": 1}}
    expected_collector = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
    expected_trace = json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n"
    compact = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
    collector, trace = Journal(tmp_path / "collector.jsonl"), NativeTraceJournal(tmp_path / "trace.jsonl")
    collector.write(value)
    collector.write_line(compact)  # preferences public schema/property order.
    trace(value)
    value["a"]["first"] = 999
    with pytest.raises(ValueError):
        collector.write({"x": float("nan")})
    with pytest.raises(ValueError):
        trace({"x": float("inf")})
    collector.close()
    trace.close()
    assert (tmp_path / "collector.jsonl").read_bytes() == (expected_collector + compact).encode()
    assert (tmp_path / "trace.jsonl").read_bytes() == expected_trace.encode()
    assert (collector.count, trace.count) == (2, 1)


def test_exclusive_creation_does_not_modify_existing_bytes(tmp_path):
    path = tmp_path / "existing.jsonl"
    path.write_bytes(b"existing immutable evidence\n")
    with pytest.raises(FileExistsError):
        Journal(path)
    assert path.read_bytes() == b"existing immutable evidence\n"


def test_close_waits_for_real_fsync_and_count_is_only_durable_records(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    real_fsync = os.fsync

    def held_fsync(descriptor):
        entered.set()
        assert release.wait(5)
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", held_fsync)
    journal = Journal(tmp_path / "count.jsonl")
    journal.write({"id": 1})
    assert entered.wait(2)
    assert journal.count == 0
    with ThreadPoolExecutor(max_workers=1) as executor:
        finished = executor.submit(journal.close)
        try:
            time.sleep(.02)
            assert not finished.done()
            assert journal.count == 0
        finally:
            release.set()
        finished.result(timeout=5)
    assert journal.count == 1
    assert journal.stream.closed


class FaultyStream:
    def __init__(self, real, failure):
        self.real, self.failure = real, failure
        self.calls = []

    @property
    def closed(self):
        return self.real.closed

    def write(self, value):
        self.calls.append(("write", threading.get_ident()))
        if self.failure == "write":
            raise OSError("synthetic write failure")
        if self.failure == "short_write":
            return self.real.write(value[:-1])
        return self.real.write(value)

    def flush(self):
        self.calls.append(("flush", threading.get_ident()))
        if self.failure == "flush":
            raise OSError("synthetic flush failure")
        return self.real.flush()

    def fileno(self):
        return self.real.fileno()

    def close(self):
        self.calls.append(("close", threading.get_ident()))
        self.real.close()
        if self.failure == "close":
            raise OSError("synthetic close failure")


@pytest.mark.parametrize("failure", ["write", "short_write", "flush", "fsync", "close", "close_fsync"])
def test_write_flush_fsync_and_close_fail_closed_including_repeated_close(tmp_path, monkeypatch, failure):
    original_open = Path.open
    real_fsync = os.fsync
    streams, fsync_calls = [], []

    def fake_open(path, *args, **kwargs):
        stream = FaultyStream(original_open(path, *args, **kwargs), failure)
        streams.append(stream)
        return stream

    def failed_fsync(descriptor):
        fsync_calls.append(threading.get_ident())
        if failure == "fsync" or failure == "close_fsync" and len(fsync_calls) == 2:
            raise OSError("synthetic fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(Path, "open", fake_open)
    monkeypatch.setattr(os, "fsync", failed_fsync)
    journal = Journal(tmp_path / "failed.jsonl")
    journal.write({"id": 1})
    marker = tmp_path / "closed.json"
    for _ in range(2):
        with pytest.raises(OSError):
            journal.close()
            marker.write_text("completion")
    assert not marker.exists()
    assert journal.count == (1 if failure in {"close", "close_fsync"} else 0)
    assert streams[0].closed
    assert set(thread_id for _, thread_id in streams[0].calls) == {journal._worker.ident}
    assert not journal._worker.is_alive()
    with pytest.raises(OSError):
        journal.write({"after": "background failure"})


@pytest.mark.parametrize("primary_failure", [False, True])
def test_collection_finishes_workers_before_any_close_and_attempts_every_sink(primary_failure):
    events = []
    failure = RuntimeError("primary worker error") if primary_failure else None

    class Native:
        async def close(self):
            assert "worker stopped" in events
            events.append("engine closed")
            raise OSError("native close failure")

    class Sink:
        def __init__(self, name):
            self.name = name
        def close(self):
            assert "worker stopped" in events
            events.append(self.name)
            raise OSError(self.name + " failure")

    async def run():
        started = asyncio.Event()
        async def one():
            try:
                started.set()
                await asyncio.Event().wait()
            finally:
                # A cancellation handler may itself need another event-loop turn.
                await asyncio.sleep(0)
                events.append("worker stopped")
        workers = [asyncio.create_task(one())]
        await started.wait()
        await _finish_collection(Native(), [Sink("sink1"), Sink("sink2")], workers, failure)

    if failure is None:
        with pytest.raises(OSError, match="native close failure") as raised:
            asyncio.run(run())
        assert len(raised.value.__notes__) == 2
    else:
        asyncio.run(run())
        assert len(failure.__notes__) == 3
    assert events == ["worker stopped", "engine closed", "sink1", "sink2"]


def test_native_close_preserves_engine_error_and_also_closes_trace_sink():
    from slime_plugins.dcc.native_thinking_eval import NativeThinkingPolicy
    events = []
    primary = RuntimeError("engine release failure")

    class Base:
        async def close(self):
            events.append("engine")
            raise primary

    class Trace:
        def close(self):
            events.append("trace")
            raise OSError("trace close failure")

    # Bypass only construction; execute the real policy cleanup implementation.
    policy = object.__new__(NativeThinkingPolicy)
    policy.base, policy.trace_sink = Base(), Trace()
    with pytest.raises(RuntimeError) as raised:
        asyncio.run(policy.close())
    assert raised.value is primary
    assert events == ["engine", "trace"]
    assert any("trace close failure" in note for note in primary.__notes__)


def test_real_collector_writer_failure_never_closes_generation_and_releases_engine(monkeypatch):
    pytest.importorskip("vllm")
    pytest.importorskip("xgrammar")
    import test_dcc_native_train_data_fixes as fixture
    original_open = Path.open
    original_engine_init = fixture.Engine.__init__
    original_journal_init = OrderedDurableJSONLWriter.__init__
    engines, journals = [], []

    def failing_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if path.name == "requests_rank0.jsonl" and args and args[0] == "x":
            return FaultyStream(stream, "write")
        return stream

    def track_engine(self, *args, **kwargs):
        original_engine_init(self, *args, **kwargs)
        engines.append(self)

    def track_journal(self, *args, **kwargs):
        original_journal_init(self, *args, **kwargs)
        journals.append(self)

    monkeypatch.setattr(Path, "open", failing_open)
    monkeypatch.setattr(fixture.Engine, "__init__", track_engine)
    monkeypatch.setattr(OrderedDurableJSONLWriter, "__init__", track_journal)
    example = fixture.NativeCollectorIntegrationTests()
    example.setUp()
    try:
        with pytest.raises(OSError, match="synthetic write failure"):
            example.generate()
        assert len(engines) == 1 and engines[0].closed
        assert len(journals) == 3
        assert all(journal.stream.closed and not journal._worker.is_alive() for journal in journals)
        assert not (example.root / "generation/generation_closed_rank0.json").exists()
    finally:
        example.tearDown()
