"""Scoring precision follows saved generation and rejects mixed judge shards."""

import asyncio
import json
import os
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
    (tmp_path / "manifest_rank0.json").write_text(json.dumps({
        "quantization": "nf4", "max_steps": 7, "max_new_tokens": 2048,
        "max_prompt_tokens": 16384, "initial_chunks": 2, "seed": 17,
    }))
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
    assert report["generation"] == {
        "max_steps": 7, "max_new_tokens": 2048, "max_prompt_tokens": 16384,
        "initial_chunks": 2, "seed": 17,
    }


def logical_predictions(tmp_path):
    tasks = [Task(f"contract{i}", "cuad", "test", "Find clause", document="A contract", gold=[]) for i in range(2)]
    manifest = {"eval_num_shards": 2, "world_size": 2, "model": "example/model",
                "inference_backend": "vllm", "adapter_identity": None}
    runtime = {"backend": "vllm", "model": "example/model", "adapter": None}
    for rank, task in enumerate(tasks):
        (tmp_path / f"manifest_rank{rank}.json").write_text(json.dumps(manifest))
        (tmp_path / f"predictions_rank{rank}.jsonl").write_text(json.dumps({
            "task_id": task.id, "status": "completed", "answer": [],
        }) + "\n")
        (tmp_path / f"rank{rank}.done").write_text(json.dumps({"tasks": 1, "world_size": 2}))
        (tmp_path / f"inference_runtime_rank{rank}.json").write_text(json.dumps(runtime))
    return tasks


@pytest.mark.parametrize("corruption,match", [
    ("overlap", "multiple shards"),
    ("missing", "Coverage mismatch"),
    ("done", "completion marker"),
    ("done_count", "completion marker"),
    ("runtime_missing", "no inference runtime"),
    ("runtime_mixed", "runtime or adapter differs"),
    ("adapter", "adapter differs from"),
    ("ownership", "task ownership"),
    ("status", "unknown episode status"),
    ("manifest", "mixed generation manifests"),
])
def test_logical_score_rejects_incomplete_or_mixed_runs(tmp_path, corruption, match):
    tasks = logical_predictions(tmp_path)
    if corruption == "overlap":
        with (tmp_path / "predictions_rank1.jsonl").open("a") as stream:
            stream.write((tmp_path / "predictions_rank0.jsonl").read_text())
    elif corruption == "missing":
        (tmp_path / "predictions_rank1.jsonl").write_text("")
    elif corruption == "done":
        (tmp_path / "rank1.done").unlink()
    elif corruption == "done_count":
        (tmp_path / "rank1.done").write_text('{"tasks":2,"world_size":2}')
    elif corruption == "runtime_missing":
        (tmp_path / "inference_runtime_rank1.json").unlink()
    elif corruption == "runtime_mixed":
        (tmp_path / "inference_runtime_rank1.json").write_text('{"adapter":{"path":"changed"}}')
    elif corruption == "adapter":
        for path in tmp_path.glob("manifest_rank*.json"):
            path.write_text(json.dumps(json.loads(path.read_text()) | {"adapter_identity": {"path": "changed"}}))
    elif corruption == "ownership":
        first, second = (tmp_path / "predictions_rank0.jsonl"), (tmp_path / "predictions_rank1.jsonl")
        first_text = first.read_text()
        first.write_text(second.read_text())
        second.write_text(first_text)
    elif corruption == "status":
        path = tmp_path / "predictions_rank0.jsonl"
        path.write_text(json.dumps(json.loads(path.read_text()) | {"status": "running"}) + "\n")
    elif corruption == "manifest":
        path = tmp_path / "manifest_rank1.json"
        path.write_text(json.dumps(json.loads(path.read_text()) | {"max_steps": 8}))
    with pytest.raises(ValueError, match=match):
        cli()["_score_predictions"](tmp_path, tasks)


def test_logical_score_keeps_same_shard_retry_and_legacy_manifests(tmp_path):
    tasks = logical_predictions(tmp_path)
    retry = {"task_id": tasks[0].id, "status": "evidence_failure", "answer": []}
    with (tmp_path / "predictions_rank0.jsonl").open("a") as stream:
        stream.write(json.dumps(retry) + "\n")
    saved, manifest, runtime = cli()["_score_predictions"](tmp_path, tasks)
    assert len(saved) == len(tasks) and saved[tasks[0].id] == retry
    assert manifest["eval_num_shards"] == 2 and runtime["backend"] == "vllm"
    for path in tmp_path.glob("manifest_rank*.json"):
        old = json.loads(path.read_text())
        del old["adapter_identity"]
        path.write_text(json.dumps(old))
    assert cli()["_score_predictions"](tmp_path, tasks)[0] == saved


def test_adapter_identity_detects_replaced_symlink_target_without_hashing(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    for path in (first, second):
        path.mkdir()
        (path / "adapter_config.json").write_text('{"r":8}')
        (path / "adapter_model.safetensors").write_bytes(b"tiny")
    timestamp = (first / "adapter_model.safetensors").stat().st_mtime_ns
    os.utime(second / "adapter_model.safetensors", ns=(timestamp, timestamp))
    adapter = tmp_path / "current"
    adapter.symlink_to(first, target_is_directory=True)
    namespace = cli()
    identity = namespace["_adapter_identity"](adapter)
    assert identity["path"] == str(adapter)
    namespace["_check_saved_adapter"](adapter, {"adapter": identity})
    adapter.unlink()
    adapter.symlink_to(second, target_is_directory=True)
    assert identity["weights_bytes"] == namespace["_adapter_identity"](adapter)["weights_bytes"]
    assert identity["weights_mtime_ns"] == namespace["_adapter_identity"](adapter)["weights_mtime_ns"]
    with pytest.raises(ValueError, match="checkpoint identity changed"):
        namespace["_check_saved_adapter"](adapter, {"adapter": identity})


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


@pytest.mark.parametrize("explicit_shard", [False, True])
def test_vllm_evaluation_uses_optional_policy_and_keeps_resume_separate(monkeypatch, tmp_path, explicit_shard):
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
    args = SimpleNamespace(model="example/model", adapter=None, data=["tasks.jsonl"],
                           max_steps=3, max_new_tokens=1024, max_prompt_tokens=8192,
                           initial_chunks=4, seed=42, limit=0, output_dir=str(tmp_path),
                           retry_infrastructure=False, inference_backend="vllm", quantization="none",
                           vllm_tensor_parallel_size=1, vllm_memory_utilization=.88,
                           vllm_batched_tokens=8192, vllm_enforce_eager=True,
                           batch_size=1, timeout=30, save_tokens=False)
    if explicit_shard:
        args.eval_shard_index, args.eval_num_shards = 0, 1
    asyncio.run(evaluate(args))
    assert calls[0]["max_model_len"] == 9216
    assert calls[-1] == "closed"
    manifest = json.loads((tmp_path / "manifest_rank0.json").read_text())
    assert manifest["inference_backend"] == "vllm"
    assert json.loads((tmp_path / "inference_runtime.json").read_text())["backend"] == "vllm"
    prediction = json.loads((tmp_path / "predictions_rank0.jsonl").read_text())
    assert prediction["status"] == "completed"
    args.inference_backend = "hf"
    with pytest.raises(ValueError, match="configuration changed|require vLLM generation"):
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


@pytest.mark.parametrize("index,count,backend,flag,match", [
    (None, 2, "vllm", None, "Provide both"),
    (0, None, "vllm", None, "Provide both"),
    (-1, 2, "vllm", None, "0 <= shard index"),
    (2, 2, "vllm", None, "0 <= shard index"),
    (0, 0, "vllm", None, "positive shard count"),
    (0, 2, "hf", None, "require vLLM generation"),
    (0, 2, "vllm", "score_only", "without scoring or judge"),
    (0, 2, "vllm", "local_judge", "without scoring or judge"),
    (0, 2, "vllm", "merge_judge", "without scoring or judge"),
])
def test_explicit_shards_reject_invalid_configuration_before_loading(monkeypatch, index, count, backend, flag, match):
    evaluate = cli()["evaluate"]
    for name in ("RANK", "LOCAL_RANK"):
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    args = SimpleNamespace(eval_shard_index=index, eval_num_shards=count, inference_backend=backend)
    if flag:
        setattr(args, flag, True)
    with pytest.raises(ValueError, match=match):
        asyncio.run(evaluate(args))


@pytest.mark.parametrize("name,value", [("RANK", "1"), ("LOCAL_RANK", "1"), ("WORLD_SIZE", "2")])
def test_explicit_data_shards_do_not_enable_multirank_vllm_launchers(monkeypatch, name, value):
    evaluate = cli()["evaluate"]
    for key, default in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1")):
        monkeypatch.setenv(key, default)
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="one evaluation process"):
        asyncio.run(evaluate(SimpleNamespace(inference_backend="vllm", eval_shard_index=0, eval_num_shards=2)))


def test_scoring_rejects_explicit_generation_shards():
    with pytest.raises(ValueError, match="require vLLM generation"):
        cli()["score"](SimpleNamespace(inference_backend="vllm", eval_shard_index=0, eval_num_shards=2))


def test_independent_vllm_shards_cover_once_resume_and_score_shared_predictions(monkeypatch, tmp_path):
    import torch

    policies, seeds = [], []

    class Policy:
        def __init__(self):
            self.provenance = {"backend": "vllm", "model": "example/model", "adapter": None}
            self.closed = False

        async def __call__(self, messages, **kwargs):
            return PolicyCompletion('{"answer":[]}', [1], [2])

        async def close(self):
            self.closed = True

    tasks = [Task(f"contract{i}", "cuad", "validation", "Find clause", document="A contract", gold=[]) for i in range(5)]
    evaluate, args = stub_evaluation(monkeypatch, tmp_path, None, tasks)

    def load(*unused, **kwargs):
        assert [os.environ[key] for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE")] == ["0", "0", "1"]
        policy = Policy()
        policies.append(policy)
        return policy

    monkeypatch.setitem(sys.modules, "slime_plugins.dcc.vllm_policy", SimpleNamespace(load_vllm_policy=load))
    monkeypatch.setattr(torch, "manual_seed", seeds.append)
    first = SimpleNamespace(**vars(args), eval_shard_index=0, eval_num_shards=2)
    second = SimpleNamespace(**vars(args), eval_shard_index=1, eval_num_shards=2)

    async def run():
        await asyncio.gather(evaluate(first), evaluate(second))

    asyncio.run(run())
    assert seeds == [42, 43]
    assert len(policies) == 2 and all(policy.closed for policy in policies)
    ids = []
    for rank in range(2):
        records = [json.loads(line) for line in (tmp_path / f"predictions_rank{rank}.jsonl").read_text().splitlines()]
        ids.append({record["task_id"] for record in records})
        manifest = json.loads((tmp_path / f"manifest_rank{rank}.json").read_text())
        assert manifest["world_size"] == manifest["eval_num_shards"] == 2
        assert json.loads((tmp_path / f"rank{rank}.done").read_text())["tasks"] == len(records)
        assert json.loads((tmp_path / f"inference_runtime_rank{rank}.json").read_text()) == policies[rank].provenance
    assert ids[0] == {task.id for task in tasks[::2]}
    assert ids[1] == {task.id for task in tasks[1::2]}
    assert not ids[0] & ids[1] and ids[0] | ids[1] == {task.id for task in tasks}
    assert not (tmp_path / "inference_runtime.json").exists()
    before = [path.read_text() for path in sorted(tmp_path.glob("predictions_rank*.jsonl"))]
    asyncio.run(run())
    assert len(policies) == 2
    assert before == [path.read_text() for path in sorted(tmp_path.glob("predictions_rank*.jsonl"))]

    for change in ({"eval_num_shards": 3}, {"max_new_tokens": 1025}, {"eval_shard_index": 2, "eval_num_shards": 3},
                   {"inference_backend": "hf", "eval_shard_index": None, "eval_num_shards": None}):
        changed = SimpleNamespace(**(vars(first) | change))
        with pytest.raises(ValueError, match="configuration changed"):
            asyncio.run(evaluate(changed))
    assert len(policies) == 2
    scorer = cli()["score"]
    monkeypatch.setitem(scorer.__globals__, "load_tasks", lambda path: tasks)
    score_args = SimpleNamespace(**vars(args), merge_judge=False, local_judge=False, model_name="example/model")
    scorer(score_args)
    report = json.loads((tmp_path / "scores_strict.json").read_text())
    assert report["policy_inference_backend"] == "vllm"
    assert report["complete"]


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
    (tmp_path / "rank0.done").write_text('{"tasks":1,"world_size":1}')
    original_write = Path.write_text

    def fail_metadata(path, *args, **kwargs):
        if "inference_runtime" in path.name:
            raise failure
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_metadata)
    with pytest.raises(OSError) as caught:
        asyncio.run(evaluate(args))
    assert caught.value is failure
    assert events == ["closed"]
    assert not (tmp_path / "predictions_rank0.jsonl").exists()
    assert not (tmp_path / "rank0.done").exists()


def test_completed_vllm_resume_rechecks_adapter_before_reusing_predictions(monkeypatch, tmp_path):
    class Policy:
        provenance = {"backend": "vllm", "model": "example/model"}

        async def __call__(self, messages, **kwargs):
            return PolicyCompletion('{"answer":[]}', [1], [2])

        async def close(self):
            pass

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r":8}')
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"first")
    evaluate, args = stub_evaluation(monkeypatch, tmp_path, Policy(), [
        Task("contract", "cuad", "validation", "Find clause", document="A contract"),
    ])
    args.adapter = str(adapter)
    asyncio.run(evaluate(args))
    before = (tmp_path / "predictions_rank0.jsonl").read_text()
    weights.write_bytes(b"second")
    with pytest.raises(ValueError, match="checkpoint identity changed"):
        asyncio.run(evaluate(args))
    assert (tmp_path / "predictions_rank0.jsonl").read_text() == before


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
