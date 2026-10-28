"""Author-source oracle and independent arithmetic checks for GiGPO."""
from __future__ import annotations

import ast
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from slime_plugins.dcc.credit_adapter import CreditVisit
from slime_plugins.dcc.gigpo import compute_advantages
from slime_plugins.dcc.credit_upstream import core_gigpo


def visit(ep, step, obs, nxt, *, terminal=False, success=False, valid=True, task="task"):
    return CreditVisit(ep, step, task, obs, nxt, terminal, success,
                       10.0 * success, valid, None)


def original_author_module():
    source = json.loads((Path(__file__).parents[1] / "third_party/gigpo-source.json").read_text())["original_source"]
    namespace = {}
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module == "verl")]
    namespace["DataProto"] = SimpleNamespace
    exec(compile(tree, "<fixed original GiGPO source>", "exec"), namespace)
    return SimpleNamespace(**namespace)


def oracle(visits, *, gamma=.95, mode="mean_std_norm", penalty=.1):
    author = original_author_module()
    n = len(visits)
    task = np.array([v.task_id for v in visits], dtype=object)
    traj = np.array([f"{v.task_id}:{v.episode}" for v in visits], dtype=object)
    mask = torch.ones(n, 1)
    data = SimpleNamespace(batch={"input_ids":mask}, non_tensor_batch={
        "rewards":np.array([v.task_reward if v.terminal else 0 for v in visits], dtype=np.float32),
        "traj_uid":traj, "active_masks":np.ones(n),
    })
    returns = author.compute_step_discounted_returns(data, gamma)
    invalid = torch.tensor([not v.is_action_valid for v in visits], dtype=torch.float32)
    scores = torch.tensor([v.task_reward for v in visits])[:, None] - penalty * invalid[:, None]
    # These fixtures have a single action phase, so the unmodified complete
    # author entrypoint is an independent oracle for the adapter.
    with contextlib.redirect_stdout(io.StringIO()):
        values, _ = author.compute_gigpo_outcome_advantage(scores, returns - penalty * invalid,
            mask, np.array([v.observation for v in visits], dtype=object), task, traj,
            mode=mode, enable_similarity=False)
    return values[:, 0].tolist()


def test_vendored_functions_are_unchanged_from_author_source():
    source = json.loads((Path(__file__).parents[1] / "third_party/gigpo-source.json").read_text())["original_source"]
    original = {n.name:ast.dump(n) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    saved = Path(core_gigpo.__file__).read_text()
    actual = {n.name:ast.dump(n) for n in ast.parse(saved).body if isinstance(n, ast.FunctionDef)}
    assert actual == original


@pytest.mark.parametrize("mode", ["mean_norm", "mean_std_norm"])
@pytest.mark.parametrize("penalty", [0, .1])
def test_random_complete_trajectories_match_original_entrypoint(mode, penalty):
    rng = np.random.default_rng(47)
    for _ in range(20):
        visits = []
        for ep in range(4):
            length = int(rng.integers(1, 6))
            success = bool(rng.integers(0, 2))
            states = ["act\n" + str(int(rng.integers(0, 4))) for j in range(length)]
            for step, state in enumerate(states):
                visits.append(visit(ep, step, state, states[step+1] if step+1 < length else "terminal",
                    terminal=step == length-1, success=success, valid=bool(rng.integers(0, 2))))
        got, _ = compute_advantages(visits, mode=mode, invalid_penalty=penalty, similarity_threshold=1)
        assert got == pytest.approx(oracle(visits, mode=mode, penalty=penalty), abs=1e-6)


def test_discount_and_invalid_penalty_follow_author_trainer_order():
    visits = [visit(0,0,"act\nA","act\nB", success=True, valid=False),
              visit(0,1,"act\nB","terminal", terminal=True, success=True),
              visit(1,0,"act\nA","act\nC"),
              visit(1,1,"act\nC","terminal",terminal=True)]
    _, d = compute_advantages(visits, similarity_threshold=1)
    assert [r["discounted_return"] for r in d["rows"]] == pytest.approx([9.5,10,0,0])
    assert d["rows"][0]["episode_score"] == pytest.approx(9.9)
    assert d["rows"][0]["local_score"] == pytest.approx(9.4)
    # The penalty belongs to this visit, not a fabricated discounted outcome.
    assert d["rows"][1]["local_score"] == 10


def test_repeated_success_state_prefers_shorter_continuation():
    visits = [visit(0,0,"act\nA","act\nA",success=True),
              visit(0,1,"act\nA","terminal",terminal=True,success=True)]
    _, d = compute_advantages(visits, similarity_threshold=1, invalid_penalty=0)
    assert d["rows"][0]["episode_advantage"] == 0
    assert d["rows"][0]["local_advantage"] < 0 < d["rows"][1]["local_advantage"]


def test_zero_outcomes_with_invalid_action_are_not_all_zero():
    visits = [visit(0,0,"act\nA","terminal",terminal=True,valid=False),
              visit(1,0,"act\nA","terminal",terminal=True)]
    scores, d = compute_advantages(visits, similarity_threshold=1)
    assert scores[0] < 0 < scores[1]
    assert d["rows"][0]["local_advantage"] < 0


def test_phase_prevents_false_similarity_merge_and_singleton_local_is_zero():
    body = "same evidence " * 200
    visits = [visit(0,0,"act\n"+body,"terminal",terminal=True,success=True),
              visit(1,0,"final\n"+body,"terminal",terminal=True)]
    _, d = compute_advantages(visits, similarity_threshold=.9)
    assert len(d["groups"]) == 2
    assert [r["local_advantage"] for r in d["rows"]] == [0,0]


def test_all_legal_zero_outcomes_have_zero_credit():
    visits = [visit(0,0,"act\nA","terminal",terminal=True),
              visit(1,0,"act\nA","terminal",terminal=True)]
    scores, _ = compute_advantages(visits)
    assert scores == [0,0]
