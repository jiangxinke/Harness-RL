"""Judge balancing retains every episode and its original generation provenance."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
from slime_plugins.dcc import Task


def cli():
    return runpy.run_path(str(Path(__file__).parents[1] / "scripts/dcc_model_evaluate.py"))


class IdentityOnlyTask:
    def __init__(self, task_id):
        self.id = task_id

    def __getattr__(self, name):
        raise AssertionError(f"Scheduling must not inspect task {name}")


def predictions(tasks):
    return {task.id: {"task_id": task.id, "answer": []} for task in tasks}


def test_balanced_200_keeps_25_per_rank_and_spreads_heavy_answers():
    schedule = cli()["_judge_task_schedule"]
    tasks = [IdentityOnlyTask(f"task{index}") for index in range(200)]
    saved = predictions(tasks)
    for index, rows in ((0, 100), (8, 90), (16, 80), (24, 70)):
        saved[tasks[index].id]["answer"] = [{"name": f"entry{row}", "value": row} for row in range(rows)]
    assigned, plan = schedule(tasks, saved, 8, "balanced")
    assert [len(shard) for shard in assigned] == [25] * 8
    assert len({task.id for shard in assigned for task in shard}) == 200
    assert {task.id for shard in assigned for task in shard} == set(saved)
    assert len({rank for rank, shard in enumerate(assigned) if any(task.id in {"task0", "task8", "task16", "task24"} for task in shard)}) == 4
    assert plan["prediction_only"] and not plan["reference_or_scores_used"]
    assert schedule(tasks, saved, 8, "balanced")[1] == plan
    assert plan["estimated_cost_units_by_rank"][0] < sum(128 * (1 + 2 * rows) for rows in (100, 90, 80, 70))


@pytest.mark.parametrize("count,world", [(0, 3), (2, 5), (11, 3), (200, 8)])
def test_capacity_ties_and_default_round_robin_are_deterministic(count, world):
    schedule = cli()["_judge_task_schedule"]
    tasks = [IdentityOnlyTask(str(index)) for index in range(count)]
    saved = predictions(tasks)
    old, old_plan = schedule(tasks, saved, world, "round_robin")
    assigned, plan = schedule(tasks, saved, world, "balanced")
    assert old_plan is None and old == [tasks[rank::world] for rank in range(world)]
    assert assigned == old
    assert plan["capacities"] == [len(shard) for shard in old]


@pytest.mark.parametrize("world", [0, -1, True, 1.0])
def test_invalid_world_is_rejected(world):
    with pytest.raises(ValueError, match="positive shard count"):
        cli()["_judge_task_schedule"]([], {}, world, "balanced")


def test_duplicate_or_missing_prediction_ids_are_rejected():
    schedule = cli()["_judge_task_schedule"]
    with pytest.raises(ValueError, match="unique tasks"):
        schedule([IdentityOnlyTask("a"), IdentityOnlyTask("a")], {"a": {}}, 1, "balanced")
    with pytest.raises(ValueError, match="saved predictions"):
        schedule([IdentityOnlyTask("missing")], {}, 1, "balanced")


def test_markdown_and_invalid_answers_remain_in_schedule():
    tasks = [IdentityOnlyTask(value) for value in ("markdown", "bad", "empty")]
    saved = {"markdown": {"answer": "| a | b |\n| --- | --- |\n| one | two |"}, "bad": {"answer": "not a table"}, "empty": {"answer": []}}
    assigned, plan = cli()["_judge_task_schedule"](tasks, saved, 3, "balanced")
    assert {task.id for shard in assigned for task in shard} == set(saved)
    assert sorted(plan["estimated_cost_units_by_rank"])[:2] == [128, 128]


def balanced_shards():
    namespace = cli()
    tasks = [IdentityOnlyTask(f"task{index}") for index in range(200)]
    saved = predictions(tasks)
    saved["task0"]["answer"] = [{"name": "large answer"}] * 20
    assigned, plan = namespace["_judge_task_schedule"](tasks, saved, 8, "balanced")
    shards = [(rank, {"judge_shard_index": rank, "judge_task_schedule": json.loads(json.dumps(plan)),
                      "widesearch": {"per_task": [{"task_id": task.id} for task in shard]}})
              for rank, shard in enumerate(assigned)]
    return namespace["_check_judge_schedule"], tasks, saved, shards, plan


def test_saved_schedule_can_be_recomputed_from_answers():
    check, tasks, saved, shards, plan = balanced_shards()
    assert check(shards, tasks, saved, "balanced") == plan


@pytest.mark.parametrize("corruption", ["mixed", "merge_mode", "cost", "ownership", "rank", "bool_rank", "missing", "changed_answer", "nondict"])
def test_merge_rejects_mixed_or_corrupted_scoring_assignments(corruption):
    check, tasks, saved, shards, _ = balanced_shards()
    mode = "balanced"
    if corruption == "mixed":
        del shards[0][1]["judge_task_schedule"]
    elif corruption == "merge_mode":
        mode = "round_robin"
    elif corruption == "cost":
        shards[0][1]["judge_task_schedule"]["estimated_cost_units_by_rank"][0] += 1
    elif corruption == "ownership":
        first, second = shards[0][1]["widesearch"]["per_task"], shards[1][1]["widesearch"]["per_task"]
        first[0], second[0] = second[0], first[0]
    elif corruption == "rank":
        shards[0][1]["judge_shard_index"] = 1
    elif corruption == "bool_rank":
        shards[0][1]["judge_shard_index"] = False
    elif corruption == "missing":
        shards.pop()
    elif corruption == "changed_answer":
        saved["task1"]["answer"] = [{"name": "new larger answer"}] * 100
    elif corruption == "nondict":
        shards[0][1]["judge_task_schedule"] = []
    with pytest.raises(ValueError, match="scheduling|scoring schedule"):
        check(shards, tasks, saved, mode)


def test_balancing_checks_original_generation_ownership_before_loading_judge(monkeypatch, tmp_path):
    score = cli()["score"]
    tasks = [Task(str(index), "widesearch", "test", "Synthetic question", gold=[]) for index in range(2)]
    monkeypatch.setitem(score.__globals__, "load_tasks", lambda path: tasks)

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid generation ownership must fail before model loading")

    monkeypatch.setitem(score.__globals__, "load_policy_model", forbidden)
    manifest = {"world_size": 2, "model": "example/policy"}
    for rank in range(2):
        (tmp_path / f"manifest_rank{rank}.json").write_text(json.dumps(manifest))
        (tmp_path / f"rank{rank}.done").write_text(json.dumps({"tasks": 1, "world_size": 2}))
        (tmp_path / f"predictions_rank{rank}.jsonl").write_text(json.dumps({"task_id": tasks[1-rank].id, "status": "completed", "answer": []}) + "\n")
    original_files = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    args = SimpleNamespace(data=["synthetic"], limit=None, output_dir=str(tmp_path),
                           local_judge=True, merge_judge=False, judge_scheduling="balanced")
    with pytest.raises(ValueError, match="task ownership"):
        score(args)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == original_files


@pytest.mark.parametrize("manifest,error", [
    ({"model": "example/policy"}, "manifests do not cover"),
    ({"world_size": 1}, "model provenance"),
])
def test_balanced_score_rejects_unrecorded_generation_ownership(tmp_path, manifest, error):
    task = IdentityOnlyTask("a")
    (tmp_path / "manifest_rank0.json").write_text(json.dumps(manifest))
    (tmp_path / "predictions_rank0.jsonl").write_text('{"task_id":"a","answer":[]}\n')
    with pytest.raises(ValueError, match=error):
        cli()["_score_predictions"](tmp_path, [task], require_ownership=True)
