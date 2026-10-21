"""CPU-only phase timing regressions with synthetic engine/tokenizer fixtures."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "slime")]

from slime_plugins.dcc import EpisodeRunner
from slime_plugins.dcc import native_phase_timing as timing
from slime_plugins.dcc.cuad_structured import cuad_final_schema
from slime_plugins.dcc.native_thinking_eval import NativeThinkingPolicy, validate_native_sidecars
from test_dcc_native_thinking import RECEIPT, cuad_task, declarations, make_policy


def policy(tmp_path, *, enabled, scripts=None):
    tok, engine, original = make_policy(tmp_path, values=[{"plan": "Review", "checks": "Read", "answer": []}],
                                        scripts=scripts)
    native = NativeThinkingPolicy(original.base, source_receipt=RECEIPT,
        trace_sink=original.trace_sink, budget=original.budget, phase_timing=enabled)
    return tok, engine, native


async def episode(tmp_path, native, *, timeout=10):
    bound = native.bind_task("contract", 0)
    result = await EpisodeRunner(max_steps=1, max_new_tokens=native.budget.total_tokens,
        initial_chunks=1, cuad_structured_actions=True, timeout_seconds=timeout).run(
            cuad_task(), bound, evaluation=True)
    record = result.to_dict()
    record["native_thinking_traces"] = bound.references
    manifest, runtime = declarations(1, budget=native.budget)
    if native.phase_timing:
        manifest["native_phase_timing"] = runtime["native_phase_timing"] = timing.configuration(True)
    await native.close()
    audit = validate_native_sidecars(tmp_path, {"contract": record}, manifest, runtime,
                                    {"contract": 0}, [cuad_task()], tokenizer=native.base.tokenizer)
    return result, record, audit


def semantic(value):
    if isinstance(value, dict):
        return {k: semantic(v) for k, v in value.items() if k not in {"timing", "request_id", "trace_id"}}
    if isinstance(value, (list, tuple)):
        return [semantic(v) for v in value]
    return value


def test_default_off_never_reads_clock_and_preserves_full_native_trajectory(tmp_path, monkeypatch):
    def unavailable_clock():
        raise AssertionError("Disabled timing must not read the clock")
    monkeypatch.setattr(timing, "monotonic", unavailable_clock)
    old = tmp_path / "old"; old.mkdir()
    _, old_engine, baseline = policy(old, enabled=False)
    _, old_record, _ = asyncio.run(episode(old, baseline))
    assert all("timing" not in p for p in baseline.traces[0]["phases"])
    assert "native_phase_timing" not in baseline.provenance
    monkeypatch.undo()
    new = tmp_path / "new"; new.mkdir()
    _, engine, native = policy(new, enabled=True)
    result, record, audit = asyncio.run(episode(new, native))
    assert result.status == "completed" and audit["traces"] == 1
    assert semantic(old_record) == semantic(record)
    assert semantic(baseline.traces) == semantic(native.traces)
    assert len(engine.calls) == len(old_engine.calls) == 2
    assert not engine.aborts and not old_engine.aborts
    for phase in native.traces[0]["phases"]:
        t = phase["timing"]
        assert (t["dispatch_monotonic_seconds"] <= t["first_response_monotonic_seconds"]
                <= t["last_response_monotonic_seconds"] <= t["completion_monotonic_seconds"]
                <= t["end_monotonic_seconds"])
        assert t["elapsed_seconds"] == t["end_monotonic_seconds"] - t["dispatch_monotonic_seconds"]
        assert "abort_start_monotonic_seconds" not in t
        assert "engine_metrics" not in t
    assert json.loads((new / "native_thinking_traces_rank0.jsonl").read_text())["phases"][0]["timing"]


@pytest.mark.parametrize("blocked_phase", ["thought", "json"])
def test_episode_deadline_retains_cancelled_phase_abort_and_original_failure(tmp_path, blocked_phase):
    from test_dcc_native_thinking import FakeTokenizer
    scripts = [{"block": True}] if blocked_phase == "thought" else [
        {"ids": FakeTokenizer().encode("Reason.</think>")}, {"block": True}]
    results = []
    for enabled in [False, True]:
        directory = tmp_path / str(enabled); directory.mkdir()
        _, engine, native = policy(directory, enabled=enabled, scripts=scripts)
        result, record, audit = asyncio.run(episode(directory, native, timeout=0.02))
        assert result.status == "infrastructure_error"
        assert result.detail == "Episode deadline exceeded after 0.02 seconds (operation=policy, turn_index=0)"
        assert audit["infrastructure_errors"] == 1 and len(engine.aborts) == 1
        assert not native.base._active_requests
        trace = native.traces[0]; phase = trace["phases"][-1]
        assert trace["exception_type"] == phase["exception_type"] == "CancelledError"
        assert phase["name"] == blocked_phase and phase["aborted"] is True
        if enabled:
            t = phase["timing"]
            assert (t["dispatch_monotonic_seconds"] <= t["exception_monotonic_seconds"]
                    <= t["abort_start_monotonic_seconds"] <= t["abort_completion_monotonic_seconds"]
                    <= t["abort_end_monotonic_seconds"] <= t["end_monotonic_seconds"])
            assert t["abort_elapsed_seconds"] >= 0
            assert "completion_monotonic_seconds" not in t and "first_response_monotonic_seconds" not in t
            assert "engine_metrics" not in t
        results.append((record, trace))
    assert semantic(results[0]) == semantic(results[1])


@pytest.mark.parametrize("abort_fails", [False, True])
def test_engine_or_abort_error_identity_and_cleanup_are_unchanged(tmp_path, abort_fails):
    _, engine, native = policy(tmp_path, enabled=True)
    generation_error = RuntimeError("synthetic generation failure")
    abort_error = ConnectionError("synthetic abort failure")
    async def failed_generate(*args, **kwargs):
        raise generation_error
        yield  # Make an async iterator without any output or engine work.
    async def failed_abort(request_id):
        engine.aborts.append(request_id)
        raise abort_error
    engine.generate = failed_generate
    if abort_fails:
        engine.abort = failed_abort
    async def run():
        try:
            with pytest.raises(type(abort_error if abort_fails else generation_error)) as caught:
                await native.generate_structured([{"role": "user", "content": "Synthetic request"}],
                    json_schema=cuad_final_schema(), max_new_tokens=native.budget.total_tokens,
                    temperature=0.0, trace_metadata={"trace_id": "synthetic", "task_id": "contract", "rank": 0, "turn_index": 0})
            assert caught.value is (abort_error if abort_fails else generation_error)
        finally:
            await native.close()
    asyncio.run(run())
    assert len(engine.aborts) == 1 and not native.base._active_requests
    phase = native.traces[0]["phases"][0]; t = phase["timing"]
    assert phase["exception_type"] == "RuntimeError"
    assert ("aborted" in phase) is not abort_fails
    assert "abort_end_monotonic_seconds" in t and "end_monotonic_seconds" in t
    if abort_fails:
        assert t["abort_exception_type"] == "ConnectionError" and "abort_completion_monotonic_seconds" not in t


def test_request_metrics_are_verbatim_finite_observations_and_clock_is_local(monkeypatch):
    clock = iter([10.0, 10.25, 10.5, 11.0, 11.25])
    monkeypatch.setattr(timing, "monotonic", lambda: next(clock))
    phase = {}; timer = timing.NativePhaseTimer(phase)
    timer.observe(SimpleNamespace(metrics=SimpleNamespace(time_in_queue=0.125, first_token_time=700.0,
        decode_time=1.75, model_forward_time=12.0, last_token_time=float("inf"),
        scheduler_time=float("nan"), finished_time=None, arrival_time=True)))
    timer.mark("completion"); timer.end()
    t = phase["timing"]
    assert t["engine_metrics"] == {"time_in_queue": 0.125, "first_token_time": 700.0,
                                   "decode_time": 1.75, "model_forward_time": 12.0}
    assert t["elapsed_seconds"] == 1.25
    assert t["first_response_elapsed_seconds"] == 0.25
    assert "time_in_decode" not in t["engine_metrics"]
    assert t["engine_metrics_source"] == "vllm.RequestOutput.metrics"
    json.dumps(phase, allow_nan=False)


def test_optional_metrics_accessor_failure_does_not_replace_response():
    class Response:
        @property
        def metrics(self):
            raise LookupError("synthetic unsupported metrics accessor")
    phase = {}; timer = timing.NativePhaseTimer(phase)
    timer.observe(Response()); timer.end()
    assert phase["timing"]["engine_metrics_read_error_type"] == "LookupError"
    assert "engine_metrics" not in phase["timing"]


def test_phase_timing_configuration_and_cli_default_are_explicit():
    parser = argparse.ArgumentParser(); timing.add_arguments(parser)
    assert parser.parse_args([]).native_phase_timing is False
    assert parser.parse_args(["--native-phase-timing"]).native_phase_timing is True
    assert timing.configuration(False) is None
    assert timing.configuration(True)["extra_engine_calls"] is False
    with pytest.raises(ValueError): timing.configuration(1)
