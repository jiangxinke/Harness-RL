import json
from types import SimpleNamespace

import torch

from slime_plugins.loongreflect.metrics import (
    append_step_metrics,
    compute_local_step_totals,
    finalize_step_metrics,
    summarize_tree_for_training,
)


def test_step_metrics_upsert_replaces_retried_steps_and_sorts(tmp_path):
    path = tmp_path / "metrics.jsonl"
    append_step_metrics(path, {"step": 1, "value": "old"})
    append_step_metrics(path, {"step": 0, "value": "first"})
    append_step_metrics(path, {"step": 1, "value": "retried"})

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"step": 0, "value": "first"},
        {"step": 1, "value": "retried"},
    ]


def test_step_metrics_does_not_chmod_temporary_path(monkeypatch, tmp_path):
    """BlobFuse can lose a just-closed path between close and chmod."""

    def reject_path_chmod(*_args, **_kwargs):
        raise AssertionError("append_step_metrics must set mode while opening")

    monkeypatch.setattr(
        "slime_plugins.loongreflect.metrics.os.chmod", reject_path_chmod
    )
    path = tmp_path / "metrics.jsonl"
    append_step_metrics(path, {"step": 0, "value": "ok"})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "step": 0,
        "value": "ok",
    }


def test_tree_summary_counts_unique_model_outputs():
    tree = {
        "status": "terminated",
        "termination_reason": "max_output_tokens_reached",
        "model_turns": 2,
        "output_tokens": 123,
        "nodes": [
            {
                "kind": "think",
                "metadata": {
                    "model_turn_id": 0,
                    "raw_output": "<think>x</think><search>q</search>",
                },
            },
            # Protocol salvage can duplicate one turn; it must count once.
            {
                "kind": "think",
                "metadata": {
                    "model_turn_id": 0,
                    "raw_output": "<think>x</think><search>q</search>",
                },
            },
            {
                "kind": "think",
                "metadata": {
                    "model_turn_id": 1,
                    "raw_output": "<think>y</think><reflect>r</reflect><backtrack>b</backtrack>",
                },
            },
            {"kind": "reflect", "metadata": {"source": "harness"}},
        ],
    }
    summary = summarize_tree_for_training(tree)
    assert summary["think_count"] == 2
    assert summary["search_count"] == 1
    assert summary["reflect_count"] == 1
    assert summary["backtrack_count"] == 1
    assert summary["output_tokens"] == 123
    assert summary["truncated"] == 1


def test_step_metrics_are_rollout_means_and_distill_reward_is_negative_kl(monkeypatch):
    monkeypatch.setenv("REFLECT_TRAIN_MODE", "joint")
    args = SimpleNamespace(
        reflect_train_mode="joint",
        reflect_distill_weight=0.5,
        reflect_distill_kl_clip=10.0,
    )
    rollout_data = {
        "rollout_ids": [10, 10, 11],
        "metadata": [
            {
                "reflect_tree_metrics": {
                    "think_count": 2,
                    "search_count": 1,
                    "reflect_count": 1,
                    "backtrack_count": 0,
                    "output_tokens": 100,
                    "model_turns": 2,
                    "completed": 1,
                    "truncated": 0,
                    "answer_leak_count": 0,
                }
            },
            {"reflect_tree_metrics": None},
            {
                "reflect_tree_metrics": {
                    "think_count": 1,
                    "search_count": 0,
                    "reflect_count": 0,
                    "backtrack_count": 0,
                    "output_tokens": 40,
                    "model_turns": 1,
                    "completed": 0,
                    "truncated": 1,
                    "answer_leak_count": 1,
                }
            },
        ],
        "loss_masks": [torch.tensor([1, 1]), torch.tensor([1]), torch.tensor([1, 1])],
        "position_rewards": [
            torch.tensor([1.0, 1.0]),
            torch.tensor([3.0]),
            torch.tensor([-1.0, -1.0]),
        ],
        "reflect_masks": [
            torch.tensor([1, 0]),
            torch.tensor([0]),
            torch.tensor([0, 0]),
        ],
        "log_probs": [
            torch.tensor([-1.0, -2.0]),
            torch.tensor([-3.0]),
            torch.tensor([-1.0, -1.0]),
        ],
        "teacher_log_probs": [
            torch.tensor([-0.5, -2.0]),
            torch.tensor([-3.0]),
            torch.tensor([-1.0, -1.0]),
        ],
    }
    totals = compute_local_step_totals(args, rollout_data)
    metrics = finalize_step_metrics(7, args, totals)
    assert metrics["num_rollout_trees"] == 2
    assert metrics["think_count_mean"] == 1.5
    assert metrics["output_tokens_mean"] == 70.0
    # Tree 10 mean=(1+1+3)/3; tree 11 mean=-1; average over trees.
    assert abs(metrics["other_reward_mean"] - (1 / 3)) < 1e-6
    assert metrics["distill_reward_mean"] < 0
    assert metrics["rollout_reward_mean"] < metrics["other_reward_mean"]
    assert metrics["answer_leak_count"] == 1


def test_ablation_rollout_reward_reports_only_the_optimized_signal(monkeypatch):
    rollout_data = {
        "rollout_ids": [1],
        "metadata": [
            {
                "reflect_tree_metrics": {
                    "think_count": 1,
                    "search_count": 0,
                    "reflect_count": 1,
                    "backtrack_count": 0,
                    "output_tokens": 1,
                    "model_turns": 1,
                    "completed": 1,
                    "truncated": 0,
                    "answer_leak_count": 0,
                }
            }
        ],
        "loss_masks": [torch.tensor([1])],
        "position_rewards": [torch.tensor([2.0])],
        "reflect_masks": [torch.tensor([1])],
        "log_probs": [torch.tensor([-1.0])],
        "teacher_log_probs": [torch.tensor([-0.5])],
    }
    args = SimpleNamespace(reflect_distill_weight=0.5, reflect_distill_kl_clip=10.0)

    monkeypatch.setenv("REFLECT_TRAIN_MODE", "distill_only")
    distill = finalize_step_metrics(
        0, args, compute_local_step_totals(args, rollout_data)
    )
    assert (
        abs(distill["rollout_reward_mean"] - 0.5 * distill["distill_reward_mean"])
        < 1e-6
    )

    monkeypatch.setenv("REFLECT_TRAIN_MODE", "policy_only")
    policy = finalize_step_metrics(
        0, args, compute_local_step_totals(args, rollout_data)
    )
    assert policy["rollout_reward_mean"] == policy["other_reward_mean"]


def test_dp_fragments_reconstruct_one_tree_reward_and_fraction(monkeypatch):
    monkeypatch.setenv("REFLECT_TRAIN_MODE", "joint")
    args = SimpleNamespace(
        reflect_train_mode="joint",
        reflect_distill_weight=0.5,
        reflect_distill_kl_clip=10.0,
    )
    tree_metrics = {
        "think_count": 2,
        "search_count": 1,
        "reflect_count": 1,
        "backtrack_count": 0,
        "output_tokens": 4,
        "model_turns": 2,
        "completed": 1,
        "truncated": 0,
        "answer_leak_count": 0,
    }

    # One four-token tree split 2+2 over two DP ranks. Only the first fragment
    # owns the one-per-tree metadata, while both carry global denominators.
    fragments = []
    for metadata, rewards, student, teacher, reflect_mask in (
        (tree_metrics, [2.0, 2.0], [-1.0, -1.0], [-0.5, -1.0], [1, 0]),
        (None, [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [0, 0]),
    ):
        fragments.append(
            compute_local_step_totals(
                args,
                {
                    "rollout_ids": [17],
                    "metadata": [{"reflect_tree_metrics": metadata}],
                    "loss_masks": [torch.tensor([1, 1])],
                    "rollout_mask_sums": torch.tensor([4.0]),
                    "position_rewards": [torch.tensor(rewards)],
                    "reflect_masks": [torch.tensor(reflect_mask)],
                    "rollout_reflect_mask_sums": torch.tensor([1.0]),
                    "log_probs": [torch.tensor(student)],
                    "teacher_log_probs": [torch.tensor(teacher)],
                },
            )
        )
    totals = {key: sum(fragment[key] for fragment in fragments) for key in fragments[0]}
    metrics = finalize_step_metrics(3, args, totals)

    # Whole-tree other reward is (2+2-1-1)/4 = 0.5. The tree and its reflect
    # coverage are counted once even though two DP ranks contributed tokens.
    assert metrics["num_rollout_trees"] == 1
    assert metrics["other_reward_mean"] == 0.5
    assert metrics["reflect_rollout_fraction"] == 1.0
    assert metrics["distill_reward_mean"] < 0
    assert metrics["rollout_reward_mean"] < 0.5


def test_global_denominator_rejects_inconsistent_fragment_metadata():
    args = SimpleNamespace(
        reflect_train_mode="joint",
        reflect_distill_weight=0.5,
        reflect_distill_kl_clip=10.0,
    )
    rollout_data = {
        "rollout_ids": [1, 1],
        "metadata": [{"reflect_tree_metrics": None}] * 2,
        "loss_masks": [torch.tensor([1]), torch.tensor([1])],
        "rollout_mask_sums": torch.tensor([2.0, 3.0]),
        "position_rewards": [torch.tensor([1.0]), torch.tensor([1.0])],
        "reflect_masks": [torch.tensor([0]), torch.tensor([0])],
        "rollout_reflect_mask_sums": torch.tensor([0.0, 0.0]),
        "log_probs": [torch.tensor([-1.0]), torch.tensor([-1.0])],
        "teacher_log_probs": [torch.tensor([-1.0]), torch.tensor([-1.0])],
    }

    import pytest

    with pytest.raises(ValueError, match="inconsistent rollout_mask_sums"):
        compute_local_step_totals(args, rollout_data)


def test_tree_owner_with_zero_local_tokens_still_owns_global_denominators():
    args = SimpleNamespace(
        reflect_train_mode="joint",
        reflect_distill_weight=0.5,
        reflect_distill_kl_clip=10.0,
    )
    totals = compute_local_step_totals(
        args,
        {
            "rollout_ids": [9],
            "metadata": [
                {
                    "reflect_tree_metrics": {
                        "think_count": 1,
                        "search_count": 0,
                        "reflect_count": 1,
                        "backtrack_count": 0,
                        "output_tokens": 2,
                        "model_turns": 1,
                        "completed": 0,
                        "truncated": 0,
                        "answer_leak_count": 0,
                    }
                }
            ],
            # A suffix-only owner fragment can have no local trainable token,
            # while another DP rank owns the trainable/reflect tokens.
            "loss_masks": [torch.tensor([0])],
            "rollout_mask_sums": torch.tensor([2.0]),
            "position_rewards": [torch.tensor([0.0])],
            "reflect_masks": [torch.tensor([0])],
            "rollout_reflect_mask_sums": torch.tensor([1.0]),
            "log_probs": [torch.tensor([-1.0])],
            "teacher_log_probs": [torch.tensor([-1.0])],
        },
    )

    assert totals["trees"] == 1
    assert totals["reward_trees"] == 1
    assert totals["distill_trees"] == 1
    assert totals["other_reward_tree_means"] == 0
    assert totals["distill_reward_tree_means"] == 0
