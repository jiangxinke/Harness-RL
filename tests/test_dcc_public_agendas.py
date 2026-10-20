"""Actual-tool and actual-runner checks for label-independent TRAIN agendas."""
from dataclasses import replace
import asyncio
import importlib.util
import json
from pathlib import Path
import sys

import pytest

path = Path(__file__).resolve().parents[1] / "scripts/dcc_prepare_cuad_public_agendas.py"
sys.path.insert(0, str(path.parent))
spec = importlib.util.spec_from_file_location("dcc_public_agenda_tests", path)
agenda = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = agenda
spec.loader.exec_module(agenda)


def task_for_variant(variant):
    for index in range(100):
        task_id = f"independent-train-{index}"
        if agenda.agenda_choice(task_id, 47, 0)[0] == variant:
            return agenda.Task(task_id, "cuad", "train", 'Category: Renewal. Details: Find renewal clauses.',
                               document='Renewal occurs annually.\n' + 'Ordinary source contract language.\n' * 1000
                                        + 'The secret tail clause is unavailable to this trajectory.',
                               gold=['Renewal occurs annually.'])
    raise AssertionError("Missing seeded agenda variant")


@pytest.mark.parametrize("variant", [0, 1, 2])
def test_labels_cannot_change_public_actions_or_observations(variant):
    task = task_for_variant(variant)
    public = agenda.project_public(task)
    changed = replace(task, gold=[], metadata={"private_label": "different"})
    assert not hasattr(public, "gold") and not hasattr(public, "metadata")
    assert agenda.project_public(changed) == public
    before = agenda.plan_public_agenda(public, None)
    after = agenda.plan_public_agenda(agenda.project_public(changed), None)
    assert before == after
    assert "gold" not in before and "positive" not in before
    with pytest.raises(TypeError, match="label-free"):
        agenda.plan_public_agenda(task, None)


@pytest.mark.parametrize("variant", [0, 1, 2])
def test_real_runner_replays_every_actual_teacher_prefix(variant):
    task = task_for_variant(variant)
    public = agenda.project_public(task)
    plan = agenda.plan_public_agenda(public, None)
    records = agenda.attach_targets(task, plan)
    assert records[-1]["stage"] == "final"
    assert json.loads(records[-1]["completion"])["answer"][0]["text"] == task.gold[0]
    assert len(plan["actions"]) == (2 if variant == 0 else 3)
    assert asyncio.run(agenda.replay_agenda(public, plan, None, records)) == len(records)
    if variant == 0:
        assert records[-1]["messages"][-1]["content"].endswith("Remaining turns: 2.")
    else:
        assert "Final answer now; tools are unavailable." in records[-1]["messages"][-1]["content"]


@pytest.mark.parametrize("variant", [0, 1, 2])
def test_unsupported_positive_keeps_tools_and_never_becomes_gold_negative(variant):
    task = task_for_variant(variant)
    plan = agenda.plan_public_agenda(agenda.project_public(task), None)
    unavailable = replace(task, gold=['The secret tail clause is unavailable to this trajectory.'])
    records = agenda.attach_targets(unavailable, plan)
    assert records and all(row["stage"] != "final" for row in records)
    assert all(row["metadata"]["positive"] is True and row["metadata"]["unsupported_positive"] is True for row in records)
    assert all('"tool"' in row["completion"] and '"answer"' not in row["completion"] for row in records)
    assert asyncio.run(agenda.replay_agenda(agenda.project_public(task), plan, None, records)) == len(records) + 1


def test_true_negative_has_real_source_history_and_empty_final():
    task = replace(task_for_variant(1), gold=[])
    plan = agenda.plan_public_agenda(agenda.project_public(task), None)
    records = agenda.attach_targets(task, plan)
    assert records[-1]["stage"] == "final"
    assert json.loads(records[-1]["completion"])["answer"] == []
    assert all(not observation.get("error") for observation in plan["observations"])
    assert any("text" in observation for observation in plan["observations"])


def test_single_hit_and_single_query_do_not_fabricate_repeated_tool_calls():
    task = agenda.PublicCuadTask("tiny", "Renewal", "Renewal occurs annually.")
    for pass_index in range(6):
        plan = agenda.plan_public_agenda(task, None, pass_index=pass_index)
        encoded = [json.dumps(action, sort_keys=True) for action in plan["actions"]]
        assert len(encoded) == len(set(encoded)) == 2
        assert asyncio.run(agenda.replay_agenda(task, plan, None, [])) == 3


def test_distinct_requested_windows_do_not_repeat_same_truncated_observation():
    original = task_for_variant(1)
    public = agenda.PublicCuadTask(original.id, original.prompt, "Renewal applies. " + "source text. " * 250)
    plan = agenda.plan_public_agenda(public, None)
    ranges = [(o["start"], o["end"]) for o in plan["observations"] if "text" in o]
    assert len(ranges) == len(set(ranges))
    assert asyncio.run(agenda.replay_agenda(public, plan, None, [])) == len(plan["actions"]) + 1


def test_official_splits_never_enter_public_projection():
    with pytest.raises(ValueError, match="TRAIN"):
        agenda.project_public(replace(task_for_variant(0), split="test"))


def test_public_subset_is_fixed_before_labels_and_independent_of_input_order():
    tasks = [replace(task_for_variant(0), id=f"subset-{index}") for index in range(30)]
    before = agenda.select_public_tasks([agenda.project_public(task) for task in tasks], max_tasks=12)
    changed = [agenda.project_public(replace(task, gold=[], metadata={"changed": True}))
               for task in reversed(tasks)]
    assert agenda.select_public_tasks(changed, max_tasks=12) == before
    assert len(before) == 12
    with pytest.raises(ValueError, match="public tasks"):
        agenda.select_public_tasks(tasks, max_tasks=12)


@pytest.mark.parametrize("lengths, expected", [([2, 1], 1.0), ([1, 1], 0.5), ([1, 2], 0.0)])
def test_train_feature_auc_handles_direction_and_ties(lengths, expected):
    from dcc_preflight_public_agendas import length_auc

    rows = [{"positive": True, "length": lengths[0]}, {"positive": False, "length": lengths[1]}]
    assert length_auc(rows, "length") == expected
