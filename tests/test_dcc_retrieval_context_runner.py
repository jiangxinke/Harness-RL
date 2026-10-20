"""Exercise bounded contract context through both episode retrieval routes."""

import asyncio
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from slime_plugins.dcc import EpisodeRunner, PolicyCompletion, Task
from slime_plugins.dcc.retrieval import search_document
from slime_plugins.dcc.tools import DocumentTools


def contract_task():
    return Task(
        id="heldout__Governing Law", benchmark="cuad", split="validation",
        prompt="Which law governs this contract?",
        document="Background obligations. " * 120 + "Delaware law governs this contract. "
        + "Further obligations. " * 120,
        gold=["private label absent from document"],
    )


def test_context_is_bounded_and_identical_for_initial_and_followup_search():
    task = contract_task()
    runner = EpisodeRunner(max_steps=2, initial_chunks=2, retrieval_context_chars=400,
                           retrieval_max_context_chars=6500)
    initial = json.loads(runner.initial_messages(task)[-1]["content"].split("\n", 1)[1])

    async def policy(messages, **kwargs):
        if len(messages) == 3:
            action = {"tool": "search_document", "arguments": {"query": task.prompt, "top_k": 2}}
        else:
            action = {"answer": []}
        return PolicyCompletion(json.dumps(action))

    result = asyncio.run(runner.run(task, policy, evaluation=True))
    observed = result.tool_events[0]["observation"]
    assert observed == initial
    assert observed["hits"]
    assert sum(len(hit["text"]) for hit in observed["hits"]) <= 6500
    for hit in observed["hits"]:
        assert hit["text"] == task.document[hit["start"]:hit["end"]]
    assert "private label" not in json.dumps(result.messages)


def test_disabled_context_preserves_original_initial_observation():
    task = contract_task()
    observed = json.loads(EpisodeRunner(initial_chunks=2).initial_messages(task)[-1]["content"].split("\n", 1)[1])
    assert observed == search_document(DocumentTools(task.document), task.prompt, top_k=2)


@pytest.mark.parametrize("options", [
    {"retrieval_context_chars": -1}, {"retrieval_context_chars": True},
    {"retrieval_context_chars": 1.5}, {"retrieval_max_context_chars": 0},
    {"retrieval_max_context_chars": False},
])
def test_invalid_context_budget_rejected(options):
    with pytest.raises(ValueError, match="retrieval_.*context_chars"):
        EpisodeRunner(**options)


@pytest.mark.parametrize("changed", [
    {"retrieval_context_chars": 800},
    {"retrieval_max_context_chars": 12000},
    {"timeout": 300},
])
def test_resume_rejects_changed_context_or_timeout(monkeypatch, tmp_path, changed):
    evaluate = runpy.run_path(str(Path(__file__).parents[1] / "scripts/dcc_model_evaluate.py"))["evaluate"]
    task = contract_task()
    monkeypatch.setitem(evaluate.__globals__, "load_tasks", lambda _: [task])
    monkeypatch.setitem(evaluate.__globals__, "load_policy_model", lambda *a, **k: pytest.fail("completed run must not load a model"))
    for key, value in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1")):
        monkeypatch.setenv(key, value)
    args = SimpleNamespace(model="frozen-base", adapter=None, data=["heldout.jsonl"],
        max_steps=3, max_new_tokens=1024, max_prompt_tokens=8192, initial_chunks=4,
        seed=42, limit=None, output_dir=str(tmp_path), retrieval_profile=None,
        retrieval_context_chars=400, retrieval_max_context_chars=16000,
        timeout=900, retry_infrastructure=False)
    (tmp_path / "predictions_rank0.jsonl").write_text(json.dumps({"task_id": task.id, "status": "completed"}) + "\n")
    asyncio.run(evaluate(args))
    manifest = json.loads((tmp_path / "manifest_rank0.json").read_text())
    assert manifest["retrieval_context"] == {"context_chars": 400, "max_context_chars": 16000}
    assert manifest["timeout_seconds"] == 900
    args.__dict__.update(changed)
    with pytest.raises(ValueError, match="configuration changed"):
        asyncio.run(evaluate(args))
