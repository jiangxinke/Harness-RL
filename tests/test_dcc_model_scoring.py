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
