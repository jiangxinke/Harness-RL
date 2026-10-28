"""Check DCC G2PO against the pinned author source and independent arithmetic."""
from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from slime_plugins.dcc.credit_adapter import CreditVisit
from slime_plugins.dcc.g2po import compute_advantages


ROOT = Path(__file__).resolve().parents[1]
PINNED_SHA256 = "4f8c2e1730545a8139de71bfc30ad75391e1ed748bba2e8dc2c023a13ea348ff"


def trajectory(episode, observations, reward, *, task="q", invalid=()):
    return [CreditVisit(
        episode=episode, step=i, task_id=task, observation=observation,
        next_observation=observations[i + 1] if i + 1 < len(observations) else "terminal",
        terminal=i == len(observations) - 1, success=reward == 10, task_reward=float(reward),
        is_action_valid=i not in invalid, completion=None,
    ) for i, observation in enumerate(observations)]


@pytest.fixture(scope="module")
def official():
    metadata = json.loads((ROOT / "third_party/g2po-source.json").read_text(encoding="utf-8"))
    source = metadata["original_source"]
    assert metadata["commit"] == "b8ffaf56ad241ce73507908e6ed30209cafd324d"
    assert hashlib.sha256(source.encode()).hexdigest() == PINNED_SHA256
    tree = ast.parse(source)
    # DataProto is an annotation; every numeric statement remains untouched.
    tree.body = [statement for statement in tree.body
                 if not isinstance(statement, ast.ImportFrom) or statement.module != "verl"]
    namespace = {"DataProto": SimpleNamespace}
    exec(compile(tree, "pinned_g2po/core_g2po.py", "exec"), namespace)
    return SimpleNamespace(**namespace)


def original_oracle(official, visits, *, mode="mean_std_norm", weight=1.0, penalty=0.1):
    order = sorted(range(len(visits)), key=lambda i: (visits[i].task_id, visits[i].episode, visits[i].step))
    rows = [visits[i] for i in order]
    uid = np.array([row.task_id for row in rows])
    traj = np.array([json.dumps([row.task_id, row.episode]) for row in rows])
    rewards = np.array([row.task_reward for row in rows], dtype=float)
    valid = np.array([row.is_action_valid for row in rows], dtype=int)
    mask = torch.ones((len(rows), 1))
    batch = SimpleNamespace(batch={"responses": torch.zeros_like(mask)}, non_tensor_batch={
        "uid": uid, "traj_uid": traj, "episode_rewards": rewards,
        "anchor_obs": np.array([row.observation for row in rows]),
    })
    values, groups, next_groups = official.compute_group_aggregation_values(batch)
    reward_tensor = torch.tensor((rewards - penalty * (1 - valid))[:, None], dtype=torch.float32)
    if penalty == 0.1:
        actual, _ = official.compute_g2po_outcome_advantage(
            reward_tensor, values, mask, uid, traj, groups, next_groups, valid,
            mode=mode, step_advantage_w=weight,
        )
    else:
        episode = official.episode_norm_reward(reward_tensor, mask, uid, traj,
                                               remove_std=mode == "mean_norm")
        step = official.compute_step_level_advantage(values, mask, valid, uid, groups, next_groups,
                                                     mode=mode, invalid_action_penalty=penalty)
        actual = episode + weight * step
    expected = [0.0] * len(rows)
    for i, original in enumerate(order):
        expected[original] = float(actual[i, 0])
    return expected


def assert_author_and_constant_guard(official, visits, actual, diagnostic, **kwargs):
    expected = original_oracle(official, visits, **kwargs)
    rows = diagnostic["rows"]
    assert [r["upstream_advantage"] for r in rows] == pytest.approx(expected, abs=1e-6)
    fields = {"episode": "episode_normalization_score", "node": "node_raw_score",
              "edge": "edge_normalization_score"}
    for component, raw_field in fields.items():
        corrected = set()
        for fix in diagnostic["normalization_corrections"][component]:
            indices = fix["indices"]
            assert len(indices) > 1
            assert all(rows[i][raw_field] == fix["constant_score"] for i in indices)
            assert fix["upstream_advantages"] == [rows[i][f"upstream_{component}_advantage"] for i in indices]
            assert len({rows[i]["task_id"] for i in indices}) == 1
            if component == "node":
                assert len({rows[i]["node"] for i in indices}) == 1
            corrected.update(indices)
        for i, row in enumerate(rows):
            assert row[f"{component}_advantage"] == (0. if i in corrected else row[f"upstream_{component}_advantage"])
    assert actual == pytest.approx([
        r["episode_advantage"] + kwargs.get("weight", 1.) * (r["node_advantage"] + r["edge_advantage"])
        for r in rows
    ], abs=2e-6)


def independent_terms(visits, *, mode="mean_std_norm", weight=1.0, penalty=0.1):
    """A scalar reference derived from the three definitions, without Torch."""
    def norm(numbers):
        if len(numbers) < 2:
            return [0.0] * len(numbers)
        mean = math.fsum(numbers) / len(numbers)
        scale = math.sqrt(math.fsum((v - mean) ** 2 for v in numbers) / (len(numbers) - 1)) + 1e-6
        return [(v - mean) / scale if mode == "mean_std_norm" else v - mean for v in numbers]

    trajectories, nodes, tasks = defaultdict(list), defaultdict(list), defaultdict(list)
    for i, visit in enumerate(visits):
        trajectories[(visit.task_id, visit.episode)].append(i)
        nodes[(visit.task_id, visit.observation)].append(i)
        tasks[visit.task_id].append(i)
    visit_returns = [v.task_reward * .95 ** (len(trajectories[(v.task_id, v.episode)]) - v.step)
                     for v in visits]
    value = {key: math.fsum(visit_returns[i] for i in indices) / len(indices)
             for key, indices in nodes.items()}
    current = [value[(v.task_id, v.observation)] for v in visits]
    target = [v.task_reward if v.terminal else value[(v.task_id, v.next_observation)] for v in visits]
    episode, node, edge = [0.0] * len(visits), [0.0] * len(visits), [0.0] * len(visits)
    for indices in tasks.values():
        ep = norm([visits[i].task_reward - penalty * (not visits[i].is_action_valid) for i in indices])
        ec = norm([target[i] - current[i] for i in indices])
        for i, a, b in zip(indices, ep, ec):
            episode[i], edge[i] = a, b
    for indices in nodes.values():
        nc = norm([target[i] - penalty * (not visits[i].is_action_valid) for i in indices])
        for i, a in zip(indices, nc):
            node[i] = a
    return dict(episode=episode, node=node, edge=edge, values=current,
                combined=[a + weight * (b + c) for a, b, c in zip(episode, node, edge)])


def test_vendor_algorithm_body_is_identical_to_pinned_source():
    metadata = json.loads((ROOT / "third_party/g2po-source.json").read_text(encoding="utf-8"))
    vendor = (ROOT / "slime/slime_plugins/dcc/credit_upstream/core_g2po.py").read_text(encoding="utf-8")
    vendor = vendor[vendor.index("import numpy as np\n"):]
    vendor = vendor.replace("from typing import Any as DataProto", "from verl import DataProto")
    assert vendor == metadata["original_source"]
    assert hashlib.sha256(vendor.encode()).hexdigest() == PINNED_SHA256
    assert "Apache License" in metadata["license_text"]
    assert "Nanyang Technological University" in metadata["notice_text"]


@pytest.mark.parametrize("mode", ["mean_norm", "mean_std_norm"])
def test_shared_root_and_both_terminal_values_match_hand_calculation(official, mode):
    visits = trajectory(0, ["A", "B"], 10) + trajectory(1, ["A", "C"], 0)
    actual, diagnostic = compute_advantages(visits, mode=mode)
    assert [row["node_value"] for row in diagnostic["rows"]] == pytest.approx([4.5125, 9.5, 4.5125, 0])
    assert [row["edge_raw_gain"] for row in diagnostic["rows"]] == pytest.approx([4.9875, .5, -4.5125, 0])
    assert [row["next_node"] for row in diagnostic["rows"]][1::2] == [-1, -2]
    assert_author_and_constant_guard(official, visits, actual, diagnostic, mode=mode)
    independent = independent_terms(visits, mode=mode)
    for component in ("episode", "node", "edge"):
        assert [row[f"{component}_advantage"] for row in diagnostic["rows"]] == pytest.approx(independent[component], abs=2e-6)


def test_repeated_visits_and_self_loop_are_not_deduplicated(official):
    visits = trajectory(0, ["A", "B", "B"], 10) + trajectory(1, ["A", "B"], 0)
    actual, diagnostic = compute_advantages(visits)
    assert [row["node_value"] for row in diagnostic["rows"]] == pytest.approx([4.286875, 6.175, 6.175, 4.286875, 6.175])
    assert diagnostic["rows"][1]["edge_raw_gain"] == 0
    assert diagnostic["rows"][1]["edge_advantage"] < 0
    assert diagnostic["tasks"][0]["self_loop_visits"] == 1
    assert sorted(n["visits"] for n in diagnostic["nodes"]) == [2, 3]
    assert_author_and_constant_guard(official, visits, actual, diagnostic)
    expected_macro = [4 / (math.sqrt(30) + 1e-6)] * 3 + [-6 / (math.sqrt(30) + 1e-6)] * 2
    assert [row["episode_advantage"] for row in diagnostic["rows"]] == pytest.approx(expected_macro, abs=1e-6)


def test_all_zero_returns_only_produce_credit_if_invalid_action_penalty_supplies_it(official):
    legal = trajectory(0, ["A", "B"], 0) + trajectory(1, ["A", "C"], 0)
    actual, diagnostic = compute_advantages(legal)
    assert actual == [0] * 4
    assert all(row["local_advantage"] == 0 for row in diagnostic["rows"])
    invalid = [replace(legal[0], is_action_valid=False), *legal[1:]]
    actual, diagnostic = compute_advantages(invalid)
    assert_author_and_constant_guard(official, invalid, actual, diagnostic)
    assert diagnostic["rows"][0]["episode_advantage"] < 0
    assert diagnostic["rows"][0]["node_advantage"] < 0
    assert actual[0] < 0 < actual[2]
    assert all(row["edge_advantage"] == 0 for row in diagnostic["rows"])
    assert compute_advantages(invalid, invalid_penalty=0)[0] == [0] * 4


def test_same_terminal_success_can_still_have_nonzero_edge_credit():
    visits = trajectory(0, ["A", "B", "C"], 10) + trajectory(1, ["A", "C"], 10)
    _, diagnostic = compute_advantages(visits)
    assert all(row["episode_advantage"] == 0 for row in diagnostic["rows"])
    assert any(abs(row["edge_advantage"]) > .01 for row in diagnostic["rows"])


def test_phase_markers_and_tasks_remain_separate_even_with_reused_episode_ids(official):
    visits = (trajectory(0, ["active:A", "active:B", "final:B"], 10, task="q1")
              + trajectory(1, ["active:A", "final:B"], 0, task="q1")
              + trajectory(0, ["active:A", "active:B", "final:B"], 0, task="q2")
              + trajectory(1, ["active:A", "final:B"], 10, task="q2"))
    actual, diagnostic = compute_advantages(visits)
    assert_author_and_constant_guard(official, visits, actual, diagnostic)
    assert [task["node_count"] for task in diagnostic["tasks"]] == [3, 3]
    assert diagnostic["rows"][1]["node"] != diagnostic["rows"][2]["node"]
    assert diagnostic["rows"][0]["node_value"] != diagnostic["rows"][5]["node_value"]


@pytest.mark.parametrize("mode", ["mean_norm", "mean_std_norm"])
@pytest.mark.parametrize("penalty", [0.0, 0.1, 0.3])
def test_random_graphs_match_author_code_and_independent_scalar_reference(official, mode, penalty):
    rng = random.Random(265)
    for _ in range(20):
        visits = []
        for task in ("q1", "q2"):
            for episode in range(rng.randint(2, 5)):
                length = rng.randint(1, 7)
                observations = ["root"] + [rng.choice(["A", "B", "C", "D"]) for _ in range(length - 1)]
                visits += trajectory(episode, observations, rng.choice([0, 10]), task=task,
                                     invalid=[i for i in range(length) if rng.random() < .2])
        rng.shuffle(visits)
        weight = rng.choice([0.0, 0.8, 1.0, 1.2])
        actual, diagnostic = compute_advantages(visits, mode=mode, weight=weight, invalid_penalty=penalty)
        assert_author_and_constant_guard(official, visits, actual, diagnostic,
                                         mode=mode, weight=weight, penalty=penalty)
        independent = independent_terms(visits, mode=mode, weight=weight, penalty=penalty)
        assert actual == pytest.approx(independent["combined"], abs=3e-5, rel=3e-6)
        for component in ("episode", "node", "edge"):
            assert [row[f"{component}_advantage"] for row in diagnostic["rows"]] == pytest.approx(independent[component], abs=3e-5, rel=3e-6)
        assert [row["step"] for row in diagnostic["rows"]] == [v.step for v in visits]
        assert [row["episode"] for row in diagnostic["rows"]] == [v.episode for v in visits]


@pytest.mark.parametrize("change, message", [
    ({"task_reward": 5.0}, "binary terminal rewards"),
    ({"task_reward": float("nan")}, "binary terminal rewards"),
    ({"success": True}, "binary terminal rewards"),
    ({"step": 2}, "consecutive steps"),
    ({"terminal": True}, "exactly one terminal"),
    ({"next_observation": "wrong"}, "next_observation"),
])
def test_incomplete_or_inconsistent_records_fail_before_credit(change, message):
    visits = trajectory(0, ["A", "B"], 0) + trajectory(1, ["A", "C"], 0)
    visits[0] = replace(visits[0], **change)
    with pytest.raises(ValueError, match=message):
        compute_advantages(visits)


def test_duplicate_steps_and_single_visit_task_fail():
    visits = trajectory(0, ["A", "B"], 0)
    with pytest.raises(ValueError, match="consecutive steps"):
        compute_advantages(visits + [visits[0]])
    with pytest.raises(ValueError, match="at least two"):
        compute_advantages(trajectory(0, ["A"], 10))


@pytest.mark.parametrize("options", [{"weight": -1}, {"weight": float("nan")},
                                     {"invalid_penalty": -1}, {"invalid_penalty": float("inf")},
                                     {"mode": "unknown"}])
def test_invalid_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        compute_advantages([], **options)


def test_empty_input_and_zero_weight():
    assert compute_advantages([])[0] == []
    visits = trajectory(0, ["A", "B"], 10) + trajectory(1, ["A", "C"], 0)
    actual, diagnostic = compute_advantages(visits, weight=0)
    assert actual == [row["episode_advantage"] for row in diagnostic["rows"]]


def test_equal_invalid_scores_zero_all_components_and_keep_original_input_indices(official):
    visits = ([v for episode in range(8)
               for v in trajectory(episode, ["same"], 10, task="equal", invalid=(0,))]
              + trajectory(0, ["A", "B"], 10, task="mixed")
              + trajectory(1, ["A", "C"], 0, task="mixed"))
    random.Random(2910).shuffle(visits)
    actual, diagnostic = compute_advantages(visits)
    assert_author_and_constant_guard(official, visits, actual, diagnostic)
    equal_rows = [row for row in diagnostic["rows"] if row["task_id"] == "equal"]
    assert len(equal_rows) == 8
    assert all(row["episode_raw_score"] == 9.9 and row["node_raw_score"] == 9.9
               and row["edge_raw_gain"] == .5 for row in equal_rows)
    assert all(row["episode_advantage"] == row["node_advantage"] == row["edge_advantage"] == 0.
               for row in equal_rows)
    assert [row["episode"] for row in diagnostic["rows"]] == [visit.episode for visit in visits]
