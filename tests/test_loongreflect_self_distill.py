from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from slime.utils.tensor_backper import TensorBackuper

from slime_plugins.loongreflect.self_distill import (
    build_update_objectives,
    dense_tree_advantages,
    ema_tau,
    low_variance_reverse_kl,
    materialize_update_objective,
    schedule_rollout_loss,
)


def test_dense_tree_advantages_are_position_aligned():
    rollout_data = {
        "position_rewards": [torch.tensor([0.2, 0.7]), torch.tensor([-0.3])],
        "kl": [torch.tensor([0.1, 0.2]), torch.tensor([0.4])],
    }
    dense_tree_advantages(SimpleNamespace(kl_coef=0.5), rollout_data)
    assert torch.allclose(rollout_data["advantages"][0], torch.tensor([0.15, 0.6]))
    assert torch.allclose(rollout_data["advantages"][1], torch.tensor([-0.5]))
    assert all(
        torch.equal(advantage, returns)
        for advantage, returns in zip(
            rollout_data["advantages"], rollout_data["returns"], strict=True
        )
    )


def test_low_variance_reverse_kl_is_nonnegative_and_has_gradient():
    student = torch.tensor([-1.0, -2.0, -9.0], requires_grad=True)
    teacher = torch.tensor([-1.0, -1.5, -1.0])
    loss = low_variance_reverse_kl(student, teacher, clip=10.0).mean()
    assert loss.item() >= 0.0
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_joint_and_fast_slow_weights_are_materialized_per_sample():
    rollout_data = {
        "response_lengths": [2, 3],
        "loss_masks": [torch.ones(2), torch.ones(3)],
    }
    args = SimpleNamespace(reflect_train_mode="joint", reflect_distill_weight=0.5)
    schedule_rollout_loss(args, 0, rollout_data)
    assert [value.item() for value in rollout_data["reflect_policy_weights"]] == [
        1.0,
        1.0,
    ]
    assert [value.item() for value in rollout_data["reflect_distill_weights"]] == [
        0.5,
        0.5,
    ]

    args = SimpleNamespace(
        reflect_train_mode="fast_slow",
        reflect_fast_updates_per_step=2,
        reflect_fast_distill_weight=1.0,
    )
    schedule_rollout_loss(args, 0, rollout_data)
    assert rollout_data["reflect_policy_weights"][0].item() == 0.0
    assert rollout_data["reflect_distill_weights"][0].item() == 1.0
    objectives = build_update_objectives(args)
    materialize_update_objective(rollout_data, objectives[-1])
    assert rollout_data["reflect_policy_weights"][0].item() == 1.0
    assert rollout_data["reflect_distill_weights"][0].item() == 0.0


def test_fast_slow_is_intra_step_reflect_then_policy():
    args = SimpleNamespace(
        reflect_train_mode="fast_slow",
        reflect_fast_updates_per_step=3,
        reflect_fast_distill_weight=0.75,
    )
    objectives = build_update_objectives(args)
    assert [objective.name for objective in objectives] == [
        "reflect_1",
        "reflect_2",
        "reflect_3",
        "verifier",
    ]
    assert [
        (objective.policy_weight, objective.distill_weight) for objective in objectives
    ] == [
        (0.0, 0.75),
        (0.0, 0.75),
        (0.0, 0.75),
        (1.0, 0.0),
    ]
    assert ema_tau(0, 1, 0.996, 1.0) == 0.996


def test_ablation_objectives_remove_exactly_one_training_signal():
    distill_only = build_update_objectives(
        SimpleNamespace(reflect_train_mode="distill_only", reflect_distill_weight=0.5)
    )
    assert [objective.name for objective in distill_only] == ["distill_only"]
    assert (distill_only[0].policy_weight, distill_only[0].distill_weight) == (0.0, 0.5)

    policy_only = build_update_objectives(
        SimpleNamespace(reflect_train_mode="policy_only")
    )
    assert [objective.name for objective in policy_only] == ["policy_only"]
    assert (policy_only[0].policy_weight, policy_only[0].distill_weight) == (
        1.0,
        0.0,
    )


def test_fast_slow_update_count_and_weight_are_validated():
    with pytest.raises(ValueError, match="must be at least 1"):
        build_update_objectives(
            SimpleNamespace(
                reflect_train_mode="fast_slow",
                reflect_fast_updates_per_step=0,
                reflect_fast_distill_weight=1.0,
            )
        )
    with pytest.raises(ValueError, match="must be positive"):
        build_update_objectives(
            SimpleNamespace(
                reflect_train_mode="fast_slow",
                reflect_fast_updates_per_step=2,
                reflect_fast_distill_weight=0.0,
            )
        )
    with pytest.raises(ValueError, match="must be exactly 1"):
        build_update_objectives(
            SimpleNamespace(
                reflect_train_mode="fast_slow",
                reflect_fast_updates_per_step=3,
                reflect_fast_distill_weight=1.0,
                reflect_verifier_updates_per_step=2,
            )
        )


def test_tensor_backuper_ema_update(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    original_empty_like = torch.empty_like
    monkeypatch.setattr(
        torch, "empty_like", lambda tensor, **kwargs: original_empty_like(tensor)
    )
    parameter = torch.tensor([2.0, 4.0])
    backuper = TensorBackuper.create(lambda: [("weight", parameter)], single_tag=None)
    backuper.backup("actor")
    parameter.copy_(torch.tensor([0.0, 0.0]))
    backuper.backup("ema_teacher")
    backuper.ema_update(src_tag="actor", dst_tag="ema_teacher", tau=0.75)
    assert torch.allclose(
        backuper.get("ema_teacher")["weight"], torch.tensor([0.5, 1.0])
    )
