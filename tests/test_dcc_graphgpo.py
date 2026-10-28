"""GraphGPO arithmetic and real-observation checks against pinned author code."""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
from dataclasses import replace
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

import numpy as np
import pytest
import torch

from slime_plugins.dcc.credit_adapter import CreditVisit
from slime_plugins.dcc.credit_upstream import core_graphgpo
from slime_plugins.dcc.graphgpo import compute_advantages


PROVENANCE = Path(__file__).resolve().parents[1] / "third_party/graphgpo-source.json"


class ReferenceBatch:
    def __init__(self, tensor, other):
        self.batch = tensor
        self.non_tensor_batch = other

    def __len__(self):
        return len(self.non_tensor_batch["uid"])

    def __getitem__(self, index):
        return SimpleNamespace(
            batch={k: v[index] for k, v in self.batch.items()},
            non_tensor_batch={k: v[index] for k, v in self.non_tensor_batch.items()},
        )


def author_code():
    """Execute unchanged definitions from the original, pinned source text."""
    source_record = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    namespace = dict(
        np=np, torch=torch, defaultdict=defaultdict, Counter=Counter,
        SequenceMatcher=SequenceMatcher, DataProto=ReferenceBatch, uuid=uuid,
        List=list, Dict=dict, Any=object,
    )
    selections = {
        "gigpo/core_gigpo.py": {
            "to_hashable", "are_similar", "summarize_group_size", "episode_norm_reward",
            "build_step_group", "step_norm_reward",
        },
        "recipe/GraphGPO/core_graph.py": {
            "to_hashable", "StateCanonicalizer", "nested_defaultdict", "ComplexGraph",
            "compute_graph_path_returns", "compute_graphgpo_outcome_advantage",
        },
        "recipe/GraphGPO/graphgpo_ray_trainer.py": {"apply_invalid_action_penalty"},
    }
    by_path = {entry["path"]: entry for entry in source_record["sources"]}
    for path, names in selections.items():
        entry = by_path[path]
        assert hashlib.sha256(entry["text"].encode()).hexdigest() == entry["sha256"]
        tree = ast.parse(entry["text"])
        definitions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        assert {n.name for n in definitions} == names
        exec(compile(ast.Module(body=definitions, type_ignores=[]), path, "exec"), namespace)
    return namespace


def reference(visits, *, omega=.2, weight=1., mode="mean_std_norm", invalid_penalty=.1):
    code = author_code()
    size = len(visits)
    anchors = np.array(["observation:" + v.observation for v in visits], dtype=object)
    successors = np.array([
        "success" if v.terminal and v.success else "terminal:failure" if v.terminal
        else "observation:" + v.next_observation for v in visits
    ], dtype=object)
    indices = np.array([v.task_id for v in visits], dtype=object)
    trajectories = np.array([str(v.episode) for v in visits], dtype=object)
    batch = ReferenceBatch(
        {"input_ids": torch.zeros((size, 1), dtype=torch.long),
         "prompts": torch.zeros((size, 1), dtype=torch.long),
         "attention_mask": torch.ones((size, 2), dtype=torch.long),
         "token_level_scores": torch.tensor([[v.task_reward] for v in visits], dtype=torch.float32)},
        dict(uid=indices, traj_uid=trajectories, anchor_obs=anchors, next_obs=successors,
             rewards=np.array([v.task_reward if v.terminal else 0. for v in visits], dtype=np.float32),
             active_masks=np.ones(size, dtype=bool),
             is_action_valid=np.array([v.is_action_valid for v in visits], dtype=bool),
             episode_rewards=np.array([v.task_reward for v in visits], dtype=np.float32),
             text_actions=np.array([v.completion.text for v in visits], dtype=object)),
    )
    graphs = code["ComplexGraph"].from_data(batch)
    code["ComplexGraph"].Calculate_all_shortest_path(graphs)
    graph_returns = code["compute_graph_path_returns"](batch, graphs, omega, False)
    batch.batch["step_rewards"] = graph_returns.clone()
    code["apply_invalid_action_penalty"](batch, invalid_penalty)
    response_mask = torch.ones((size, 1), dtype=torch.float32)
    combined, _ = code["compute_graphgpo_outcome_advantage"](
        batch.batch["token_level_scores"], batch.batch["step_rewards"], response_mask,
        anchors, indices, trajectories, step_advantage_w=weight, mode=mode,
    )
    episode = code["episode_norm_reward"](
        batch.batch["token_level_scores"], response_mask, indices, trajectories,
        remove_std=mode == "mean_norm",
    )
    groups = code["build_step_group"](anchors, indices)
    graph_adv = code["step_norm_reward"](
        batch.batch["step_rewards"], response_mask, groups, remove_std=mode == "mean_norm",
    )
    return dict(scores=combined[:, 0].tolist(), episode=episode[:, 0].tolist(),
                local=graph_adv[:, 0].tolist(), graph_returns=graph_returns.tolist(),
                adjusted=batch.batch["step_rewards"].tolist())


def trajectory(episode, states, success, *, invalid=()):
    return [CreditVisit(
        episode=episode, step=step, task_id="contract", observation=state,
        next_observation=states[step + 1] if step + 1 < len(states) else "end observation",
        terminal=step + 1 == len(states), success=success, task_reward=10. * success,
        is_action_valid=step not in invalid,
        completion=SimpleNamespace(text=f"action-{episode}-{step}"),
    ) for step, state in enumerate(states)]


def assert_matches_author(visits, **kwargs):
    scores, diagnostic = compute_advantages(visits, **kwargs)
    expected = reference(visits, **kwargs)
    assert scores == pytest.approx(expected["scores"], abs=2e-6)
    for key, field in [("episode", "episode_advantage"), ("local", "local_advantage"),
                       ("graph_returns", "graph_return"), ("adjusted", "adjusted_graph_return")]:
        assert [r[field] for r in diagnostic["rows"]] == pytest.approx(expected[key], abs=2e-6)
    return scores, diagnostic


def test_vendored_definitions_are_identical_to_pinned_author_source():
    record = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    assert record["commit"] == "20bd331bdbc9026a5668e11362178e10ab7400c8"
    by_path = {entry["path"]: entry for entry in record["sources"]}
    assert by_path["recipe/GraphGPO/core_graph.py"]["sha256"] == "4dfe370d878d7c797fad521fca5c87f6a26ba0bbe93533762673ee4645fdfd3d"
    assert by_path["gigpo/core_gigpo.py"]["sha256"] == "fa709b91baf2331c4857fee32a249c4ca8d8d69e864d1e71f08730cc8c5cc456"
    for entry in by_path.values():
        assert hashlib.sha256(entry["text"].encode()).hexdigest() == entry["sha256"]
    original = ast.parse(by_path["recipe/GraphGPO/core_graph.py"]["text"])
    original_functions = {n.name: n for n in original.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    vendor = ast.parse(Path(core_graphgpo.__file__).read_text(encoding="utf-8"))
    functions = {n.name: n for n in vendor.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert set(functions) == {item["name"] for item in record["vendored_definitions"]}
    for item in record["vendored_definitions"]:
        # AST fields vary between Python versions; parse both pinned source and
        # vendor with this interpreter while retaining the fixed byte hashes.
        assert ast.dump(functions[item["name"]], include_attributes=False) == ast.dump(
            original_functions[item["name"]], include_attributes=False,
        )


def test_shared_state_shortcut_loop_and_failed_branch_have_correct_credit():
    visits = trajectory(0, ["A", "B", "B"], True) + trajectory(1, ["A", "C", "B"], False)
    scores, diagnostic = assert_matches_author(visits)
    assert [r["graph_return"] for r in diagnostic["rows"]] == pytest.approx([2., 2., 10., .4, 2., .08])
    assert [diagnostic["rows"][i]["local_advantage"] for i in [1, 2, 5]] == pytest.approx(
        [-.3851966167, 1.1353163440, -.7501197273], abs=2e-6)
    assert diagnostic["rows"][1]["local_advantage"] < 0 < scores[1]
    assert len(diagnostic["graphs"][0]["edges"]) == len(visits)
    assert diagnostic["rows"][5]["target_reachable"] is False


@pytest.mark.parametrize("mode", ["mean_std_norm", "mean_norm"])
@pytest.mark.parametrize("weight", [0., .7, 1.])
def test_invalid_penalty_is_applied_to_both_components_after_graph_return(mode, weight):
    visits = trajectory(0, ["A", "B", "B"], True, invalid=(1,)) + trajectory(1, ["A", "C", "B"], False)
    _, diagnostic = assert_matches_author(visits, mode=mode, weight=weight)
    rows = diagnostic["rows"]
    assert rows[1]["graph_return"] == pytest.approx(2.)
    assert rows[1]["adjusted_graph_return"] == pytest.approx(1.9)
    assert rows[1]["adjusted_episode_score"] == pytest.approx(9.9)
    assert rows[0]["adjusted_graph_return"] == pytest.approx(rows[0]["graph_return"])
    assert rows[0]["adjusted_episode_score"] == pytest.approx(10.)


def test_all_failed_without_invalid_actions_has_zero_signal_and_no_fake_path():
    visits = trajectory(0, ["A", "B", "A"], False) + trajectory(1, ["A", "C"], False)
    scores, diagnostic = assert_matches_author(visits)
    assert scores == [0.] * len(visits)
    assert diagnostic["nonzero_local_steps"] == 0
    assert all(not row["target_reachable"] for row in diagnostic["rows"])
    assert all(row["distance_target"] == 1 for row in diagnostic["rows"])


def test_all_failed_with_invalid_action_has_explicit_penalty_signal():
    visits = trajectory(0, ["A", "B"], False, invalid=(0,)) + trajectory(1, ["A", "C"], False)
    scores, diagnostic = assert_matches_author(visits)
    assert scores[0] < 0 < scores[2]
    assert diagnostic["rows"][0]["invalid_penalty"] == pytest.approx(.1)


def test_singleton_group_and_one_step_episode_follow_author_behavior():
    scores, diagnostic = assert_matches_author(trajectory(0, ["only"], True))
    assert diagnostic["rows"][0]["local_advantage"] == 0
    assert scores[0] == pytest.approx(10. / (1. + 1e-6))


def test_all_successful_can_still_have_local_path_length_signal():
    visits = trajectory(0, ["A"], True) + trajectory(1, ["A", "B"], True)
    scores, diagnostic = assert_matches_author(visits)
    assert all(row["episode_advantage"] == 0 for row in diagnostic["rows"])
    assert scores[0] > 0 > scores[1]


def test_ordinary_success_text_is_not_a_terminal_success_state():
    visits = trajectory(0, ["success", "terminal:failure"], False)
    _, diagnostic = assert_matches_author(visits)
    nodes = diagnostic["graphs"][0]["nodes"]
    assert len([n for n in nodes if n["kind"] == "success"]) == 1
    assert any(n["kind"] == "observation" and n["observation"] == "success" for n in nodes)
    assert all(not r["target_reachable"] for r in diagnostic["rows"])


def test_real_archived_control_change_rollouts_keep_contract_observations():
    fixture = json.loads(PROVENANCE.read_text(encoding="utf-8"))["legacy_fixture"]
    visits = []
    for raw in fixture["visits"]:
        row = dict(raw)
        row["observation"] = fixture["observations"][row.pop("observation_id")]
        next_id = row.pop("next_observation_id")
        row["next_observation"] = fixture["observations"][next_id] if next_id is not None else "terminal"
        row["completion"] = SimpleNamespace(text=row.pop("completion_text"))
        visits.append(CreditVisit(**row))
    _, diagnostic = assert_matches_author(visits)
    assert len(visits) == 12
    final_rows = [row for row in diagnostic["rows"] if row["terminal"]]
    assert len({row["source_node"] for row in final_rows}) == 3
    assert not any("Final turn: tools are unavailable" in v.observation for v in visits)
    assert all("Change of Control" in v.observation for v in visits)
    assert diagnostic["success_episodes"] == 2


@pytest.mark.parametrize("change", ["task", "missing_step", "disconnected", "early_terminal", "reward", "outcome"])
def test_incomplete_or_inconsistent_visits_are_rejected(change):
    visits = trajectory(0, ["A", "B"], True) + trajectory(1, ["A", "C"], False)
    edits = {
        "task": {"task_id": "another contract"}, "missing_step": {"step": 4},
        "disconnected": {"observation": "unconnected"}, "early_terminal": {"terminal": False},
        "reward": {"task_reward": 5.}, "outcome": {"success": True},
    }
    visits[-1] = replace(visits[-1], **edits[change])
    with pytest.raises(ValueError):
        compute_advantages(visits)


@pytest.mark.parametrize("kwargs", [dict(omega=0), dict(omega=1), dict(weight=-1),
                                    dict(invalid_penalty=-.1), dict(mode="invented")])
def test_invalid_algorithm_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        compute_advantages(trajectory(0, ["A"], True), **kwargs)


def test_empty_visits_return_empty_diagnostics():
    scores, diagnostic = compute_advantages([])
    assert scores == []
    assert diagnostic["rows"] == []
    assert diagnostic["graphs"] == []
