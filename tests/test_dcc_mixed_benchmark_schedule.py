"""Synthetic CPU controls for task scheduling; no rollout or model is run."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from functools import partial
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slime_plugins.dcc import online_web as web
from slime_plugins.dcc.data import Task


@pytest.fixture(scope="module")
def trainer():
    spec = importlib.util.spec_from_file_location("mixed_schedule_test_trainer",
        Path(__file__).resolve().parents[1] / "scripts/dcc_online_grpo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tasks():
    cuad = [Task("c0", "cuad", "train", "Q", "C", ["a"]),
            Task("c1", "cuad", "train", "Q", "C", []),
            Task("c2", "cuad", "train", "Q", "C", ["b"]),
            Task("c3", "cuad", "train", "Q", "C", ["c"])]
    public = [Task(f"w{index}", "widesearch", "train", "Q", gold=None) for index in range(5)]
    return cuad, public


def test_default_matches_original_golden_and_explicit_default(trainer, tasks):
    # Captured from the unmodified planner: repeats cross both independent decks.
    expected = [[1, 8, 2, 7], [3, 5, 0, 4], [1, 6, 0, 7], [3, 5, 2, 4]]
    options = dict(steps=4, world_size=4, seed=83, cuad_plan=trainer.task_plan)
    assert web.mixed_task_plan(*tasks, **options) == expected
    assert web.mixed_task_plan(*tasks, **options, schedule="slot-interleaved") == expected
    # The default still accepts an incomplete iteration cycle.
    options["steps"] = 1
    assert web.mixed_task_plan(*tasks, **options) == expected[:1]


@pytest.mark.parametrize("world_size", [1, 2, 3, 8])
@pytest.mark.parametrize("draws", [(1, 1), (3, 1), (2, 3)])
@pytest.mark.parametrize("sampling", ["stratified-3to1", "uniform", "stratified-3to1-span-cap5"])
def test_blocked_preserves_decks_counts_and_rank_rotation(trainer, tasks, world_size, draws, sampling):
    cuad, public = tasks
    c, w = draws
    steps = 2 * (c + w)
    options = dict(steps=steps, world_size=world_size, seed=83,
        cuad_draws_per_cycle=c, web_draws_per_cycle=w,
        cuad_plan=partial(trainer.task_plan, sampling=sampling))
    original = web.mixed_task_plan(cuad, public, **options)
    blocked = web.mixed_task_plan(cuad, public, **options, schedule="iteration-blocked")
    flat_original = [index for row in original for index in row]
    flat_blocked = [index for row in blocked for index in row]
    assert Counter(flat_blocked) == Counter(flat_original)
    assert len(flat_blocked) == steps * world_size
    assert sum(index < len(cuad) for index in flat_blocked) == 2 * c * world_size
    # A stronger guarantee than the multiset: each deck is consumed in exactly
    # its original order, including repeated draws and deck-cycle boundaries.
    for is_cuad in (True, False):
        select = lambda values: [index for index in values if (index < len(cuad)) == is_cuad]
        assert select(flat_blocked) == select(flat_original)
    expected_kinds = (["cuad"] * c + ["widesearch"] * w) * 2
    all_tasks = cuad + public
    for row, kind in zip(blocked, expected_kinds, strict=True):
        assert len(row) == world_size
        assert {all_tasks[index].benchmark for index in row} == {kind}
    for rank in range(world_size):
        kinds = [all_tasks[row[rank]].benchmark for row in blocked]
        assert kinds == expected_kinds
        assert Counter(kinds) == {"cuad": 2 * c, "widesearch": 2 * w}
    assert blocked == web.mixed_task_plan(cuad, public, **options, schedule="iteration-blocked")


@pytest.mark.parametrize("world_size", [1, 2, 8])
def test_blocked_rejects_incomplete_iteration_cycle_before_drawing(trainer, tasks, monkeypatch, world_size):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid schedules must not draw either deck")
    monkeypatch.setattr(web, "public_web_deck", forbidden)
    with pytest.raises(ValueError, match="complete benchmark cycle"):
        web.mixed_task_plan(*tasks, steps=3, world_size=world_size, seed=83,
            cuad_plan=forbidden, schedule="iteration-blocked")


def test_unknown_schedule_is_rejected(trainer, tasks):
    with pytest.raises(ValueError, match="Unknown mixed benchmark schedule"):
        web.mixed_task_plan(*tasks, steps=4, world_size=2, seed=83,
            cuad_plan=trainer.task_plan, schedule="invented")


def configuration_args():
    return SimpleNamespace(model="invented", source_commit="a" * 40, group_size=4,
        max_steps=12, thought_tokens=4096, enable_online_web=True,
        cuad_draws_per_cycle=1, web_draws_per_cycle=1, steps=4)


def test_default_configuration_omits_option_and_enabled_discloses_order(trainer):
    args = configuration_args()
    original = trainer.configuration(args, [], [], 8)
    args.mixed_benchmark_schedule = "slot-interleaved"
    assert trainer.configuration(args, [], [], 8) == original
    assert "mixed_benchmark_schedule" not in original["arguments"]
    assert "benchmark_schedule" not in original["online_web"]
    args.mixed_benchmark_schedule = "iteration-blocked"
    enabled = trainer.configuration(args, [], [], 8)
    assert enabled["arguments"]["mixed_benchmark_schedule"] == "iteration-blocked"
    assert enabled["online_web"]["benchmark_schedule"] == dict(mode="iteration-blocked",
        scope="global-iteration", optimization_order_changed=True)
    restored = deepcopy(enabled)
    del restored["arguments"]["mixed_benchmark_schedule"]
    del restored["online_web"]["benchmark_schedule"]
    assert restored == original


@pytest.mark.parametrize("extra,match", [
    (["--mixed-benchmark-schedule", "iteration-blocked"], "requires --enable-online-web"),
    (["--enable-online-web", "--mixed-benchmark-schedule", "iteration-blocked", "--steps", "3"],
        "complete benchmark cycle"),
    (["--enable-online-web", "--mixed-benchmark-schedule", "iteration-blocked",
      "--cuad-draws-per-cycle", "2"], "complete benchmark cycle"),
    (["--enable-online-web", "--mixed-benchmark-schedule", "iteration-blocked",
      "--web-draws-per-cycle", "0"], "positive integer"),
    (["--mixed-benchmark-schedule", "invented"], "invalid choice"),
])
def test_cli_rejects_invalid_mode_before_source_data_or_model(trainer, monkeypatch, capsys, extra, match):
    def forbidden(*args, **kwargs):
        raise AssertionError("No source/data/model/runtime work is allowed")
    for name in ("load_tasks", "validate_source", "load_policy_model", "train", "cpu_preflight"):
        monkeypatch.setattr(trainer, name, forbidden)
    with pytest.raises(SystemExit) as error:
        trainer.main(["--model", "invented", "--train-data", "absent", "--output-dir", "unused",
            "--source-commit", "a" * 40, *extra])
    assert error.value.code == 2
    assert match in capsys.readouterr().err


def test_cli_wires_mode_to_plan_and_keeps_default_configuration(trainer, tasks, monkeypatch, capsys):
    cuad, public = tasks
    monkeypatch.setattr(trainer, "load_tasks", lambda path: cuad)
    monkeypatch.setattr(trainer, "validate_source", lambda *args, **kwargs: None)
    monkeypatch.setattr(web, "WebTrainingInputs", lambda *args, **kwargs: SimpleNamespace(tasks=public))
    monkeypatch.setattr(trainer, "cpu_preflight",
        lambda args, selected_tasks, plan, world: trainer.configuration(args, plan, selected_tasks, world))
    command = ["--model", "invented", "--train-data", "absent", "--output-dir", "unused",
        "--source-commit", "a" * 40, "--enable-online-web", "--web-train-data", "absent-public",
        "--web-train-labels", "absent-labels", "--web-binding", "absent-binding",
        "--cpu-preflight", "--plan-world-size", "4", "--seed", "83"]
    assert trainer.main(command) == 0
    original = json.loads(capsys.readouterr().out)
    assert trainer.main(command + ["--mixed-benchmark-schedule", "slot-interleaved"]) == 0
    assert json.loads(capsys.readouterr().out) == original
    assert "mixed_benchmark_schedule" not in original["arguments"]
    assert "benchmark_schedule" not in original["online_web"]
    assert trainer.main(command + ["--mixed-benchmark-schedule", "iteration-blocked"]) == 0
    blocked = json.loads(capsys.readouterr().out)
    task_kind = {task.id: task.benchmark for task in cuad + public}
    assert [{task_kind[task_id] for task_id in row} for row in blocked["planned_task_ids"]] == [
        {"cuad"}, {"widesearch"}, {"cuad"}, {"widesearch"}]
    assert Counter(task_id for row in blocked["planned_task_ids"] for task_id in row) == Counter(
        task_id for row in original["planned_task_ids"] for task_id in row)
