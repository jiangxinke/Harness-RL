"""Scoring precision follows saved generation and rejects mixed judge shards."""

import asyncio
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest

from slime_plugins.dcc import PolicyCompletion, Task


def cli():
    return runpy.run_path(str(Path(__file__).parents[1] / "scripts/dcc_model_evaluate.py"))


def test_prediction_precision_comes_from_manifests_and_rejects_mixed_shards(tmp_path):
    read = cli()["_prediction_quantization"]
    assert read(tmp_path) == "unrecorded"
    (tmp_path / "manifest_rank0.json").write_text("{}")
    assert read(tmp_path) == "none"
    (tmp_path / "manifest_rank0.json").write_text('{"quantization":"nf4"}')
    assert read(tmp_path) == "nf4"
    (tmp_path / "manifest_rank1.json").write_text("{}")
    with pytest.raises(ValueError, match="mixed policy quantization"):
        read(tmp_path)


def test_cpu_score_retains_quantized_policy_provenance(monkeypatch, tmp_path):
    score = cli()["score"]
    task = Task("contract", "cuad", "test", "Find clause", document="exact clause", gold=["exact clause"])
    monkeypatch.setitem(score.__globals__, "load_tasks", lambda path: [task])
    (tmp_path / "manifest_rank0.json").write_text('{"quantization":"nf4"}')
    (tmp_path / "predictions_rank0.jsonl").write_text(json.dumps({
        "task_id": task.id, "status": "completed", "answer": [{"text": "exact clause", "confidence": 1}],
    }) + "\n")
    args = SimpleNamespace(data=["tasks.jsonl"], limit=None, output_dir=str(tmp_path),
                           merge_judge=False, local_judge=False, model_name="example/model",
                           max_steps=3, max_new_tokens=1024, max_prompt_tokens=8192,
                           initial_chunks=4, seed=42, quantization="none")
    score(args)
    report = json.loads((tmp_path / "scores_strict.json").read_text())
    assert report["policy_quantization"] == "nf4"
    assert report["policy_inference_backend"] == "hf"
    assert "judge_quantization" not in report


def test_merge_rejects_judge_precision_change_before_publishing(monkeypatch, tmp_path):
    score = cli()["score"]
    task = Task("web", "widesearch", "test", "Find entries", gold=[])
    monkeypatch.setitem(score.__globals__, "load_tasks", lambda path: [task])
    (tmp_path / "manifest_rank0.json").write_text("{}")
    (tmp_path / "predictions_rank0.jsonl").write_text(json.dumps({
        "task_id": task.id, "status": "evidence_failure", "answer": [],
    }) + "\n")
    (tmp_path / "scores_strict.json").write_text('{"complete":true}')
    (tmp_path / "scores_judge_rank0.json").write_text(json.dumps({
        "judge_quantization": "nf4", "policy_quantization": "none",
        "widesearch": {"per_task": []},
    }))
    args = SimpleNamespace(data=["tasks.jsonl"], limit=None, output_dir=str(tmp_path),
                           merge_judge=True, quantization="none")
    with pytest.raises(ValueError, match="Judge quantization differs"):
        score(args)
    assert not (tmp_path / "scores_local_judge.json").exists()


def test_vllm_evaluation_uses_optional_policy_and_keeps_resume_separate(monkeypatch, tmp_path):
    evaluate = cli()["evaluate"]
    task = Task("contract", "cuad", "validation", "Find clause", document="exact clause", gold=["exact clause"])
    monkeypatch.setitem(evaluate.__globals__, "load_tasks", lambda path: [task])
    for name in ("RANK", "LOCAL_RANK"):
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    calls = []

    class Policy:
        provenance = {"backend": "vllm", "runtime": "test double"}

        async def __call__(self, messages, **kwargs):
            return PolicyCompletion('{"answer":[{"text":"exact clause","confidence":1}]}', [1], [2])

        async def close(self):
            calls.append("closed")

    def load(model, **kwargs):
        calls.append(kwargs)
        return Policy()

    monkeypatch.setitem(sys.modules, "slime_plugins.dcc.vllm_policy", SimpleNamespace(load_vllm_policy=load))
    args = SimpleNamespace(model="example/model", adapter="adapter", data=["tasks.jsonl"],
                           max_steps=3, max_new_tokens=1024, max_prompt_tokens=8192,
                           initial_chunks=4, seed=42, limit=0, output_dir=str(tmp_path),
                           retry_infrastructure=False, inference_backend="vllm", quantization="none",
                           vllm_tensor_parallel_size=1, vllm_memory_utilization=.88,
                           vllm_batched_tokens=8192, vllm_enforce_eager=True,
                           batch_size=1, timeout=30, save_tokens=False)
    asyncio.run(evaluate(args))
    assert calls[0]["max_model_len"] == 9216
    assert calls[-1] == "closed"
    manifest = json.loads((tmp_path / "manifest_rank0.json").read_text())
    assert manifest["inference_backend"] == "vllm"
    assert json.loads((tmp_path / "inference_runtime.json").read_text())["backend"] == "vllm"
    prediction = json.loads((tmp_path / "predictions_rank0.jsonl").read_text())
    assert prediction["status"] == "completed"
    args.inference_backend = "hf"
    with pytest.raises(ValueError, match="configuration changed"):
        asyncio.run(evaluate(args))


@pytest.mark.parametrize("world,precision,match", [(8, "none", "one evaluation process"), (1, "nf4", "BF16")])
def test_vllm_rejects_unsupported_launches_before_loading(monkeypatch, world, precision, match):
    evaluate = cli()["evaluate"]
    monkeypatch.setenv("WORLD_SIZE", str(world))
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")
    with pytest.raises(ValueError, match=match):
        asyncio.run(evaluate(SimpleNamespace(inference_backend="vllm", quantization=precision)))


def stub_evaluation(monkeypatch, tmp_path, policy, tasks):
    evaluate = cli()["evaluate"]
    monkeypatch.setitem(evaluate.__globals__, "load_tasks", lambda path: tasks)
    for name in ("RANK", "LOCAL_RANK"):
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setitem(sys.modules, "slime_plugins.dcc.vllm_policy", SimpleNamespace(
        load_vllm_policy=lambda *args, **kwargs: policy,
    ))
    args = SimpleNamespace(model="example/model", adapter=None, data=["tasks.jsonl"],
                           max_steps=3, max_new_tokens=1024, max_prompt_tokens=8192,
                           initial_chunks=4, seed=42, limit=0, output_dir=str(tmp_path),
                           retry_infrastructure=False, inference_backend="vllm", quantization="none",
                           vllm_tensor_parallel_size=1, vllm_memory_utilization=.88,
                           vllm_batched_tokens=8192, vllm_enforce_eager=True,
                           batch_size=1, timeout=30, save_tokens=False)
    return evaluate, args


@pytest.mark.parametrize("close_fails", [False, True])
@pytest.mark.parametrize("failure_type", [RuntimeError, asyncio.CancelledError])
def test_episode_failure_waits_for_siblings_before_closing_and_preserves_error(monkeypatch, tmp_path, close_fails, failure_type):
    events, streams = [], []
    failure = failure_type("original episode failure")
    started = asyncio.Event()
    tasks = [Task(name, "cuad", "validation", "Find clause", document="A contract") for name in ("fail", "sibling")]

    class Policy:
        provenance = {}

        async def close(self):
            assert not streams[0].closed
            events.append("policy_closed")
            if close_fails:
                raise OSError("policy cleanup failure")

    class Runner:
        def __init__(self, **kwargs):
            pass

        async def run(self, task, policy, **kwargs):
            if task.id == "fail":
                await started.wait()
                raise failure
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                events.append("sibling_cancelled")
                await asyncio.sleep(0)
                # An episode can finish while handling cancellation. Its
                # prediction must be persisted before the stream is closed.
                events.append("sibling_finished")
                return SimpleNamespace(status="completed", steps=[], to_dict=lambda **kwargs: {
                    "task_id": task.id, "status": "completed", "answer": [],
                })

    original_open = Path.open

    def track_open(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        if path.name != "predictions_rank0.jsonl" or mode != "a":
            return stream
        streams.append(stream)

        def close():
            stream.close()
            events.append("stream_closed")

        return SimpleNamespace(write=stream.write, close=close)

    evaluate, args = stub_evaluation(monkeypatch, tmp_path, Policy(), tasks)
    monkeypatch.setitem(evaluate.__globals__, "EpisodeRunner", Runner)
    monkeypatch.setattr(Path, "open", track_open)
    with pytest.raises(failure_type) as caught:
        asyncio.run(evaluate(args))
    assert caught.value is failure
    assert events == ["sibling_cancelled", "sibling_finished", "policy_closed", "stream_closed"]
    assert streams[0].closed
    assert json.loads((tmp_path / "predictions_rank0.jsonl").read_text())["task_id"] == "sibling"
    assert not (tmp_path / "rank0.done").exists()
    if close_fails:
        assert any("policy cleanup failure" in note for note in failure.__notes__)


def test_runtime_metadata_failure_still_closes_loaded_policy(monkeypatch, tmp_path):
    events = []
    failure = OSError("metadata write failed")

    class Policy:
        provenance = {}

        async def close(self):
            events.append("closed")

    evaluate, args = stub_evaluation(monkeypatch, tmp_path, Policy(), [
        Task("contract", "cuad", "validation", "Find clause", document="A contract"),
    ])
    original_write = Path.write_text

    def fail_metadata(path, *args, **kwargs):
        if path.name == "inference_runtime.json":
            raise failure
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_metadata)
    with pytest.raises(OSError) as caught:
        asyncio.run(evaluate(args))
    assert caught.value is failure
    assert events == ["closed"]
    assert not (tmp_path / "predictions_rank0.jsonl").exists()
    assert not (tmp_path / "rank0.done").exists()


def test_cleanup_failure_after_success_still_closes_stream_and_prevents_done(monkeypatch, tmp_path):
    failure = OSError("policy cleanup failure")
    streams = []

    class Policy:
        provenance = {}

        async def __call__(self, messages, **kwargs):
            return PolicyCompletion('{"answer":[]}', [1], [2])

        async def close(self):
            raise failure

    original_open = Path.open

    def track_open(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        if path.name == "predictions_rank0.jsonl" and mode == "a":
            streams.append(stream)
        return stream

    evaluate, args = stub_evaluation(monkeypatch, tmp_path, Policy(), [
        Task("contract", "cuad", "validation", "Find clause", document="A contract"),
    ])
    monkeypatch.setattr(Path, "open", track_open)
    with pytest.raises(OSError) as caught:
        asyncio.run(evaluate(args))
    assert caught.value is failure
    assert streams[0].closed
    assert json.loads((tmp_path / "predictions_rank0.jsonl").read_text())["status"] == "completed"
    assert not (tmp_path / "rank0.done").exists()
