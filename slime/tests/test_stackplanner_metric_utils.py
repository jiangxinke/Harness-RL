from __future__ import annotations

from types import SimpleNamespace

from slime.utils.stackplanner_metric_utils import compute_stackplanner_sample_metrics


def test_role_capability_and_infrastructure_counts_follow_actor_mask():
    samples = [
        SimpleNamespace(
            metadata={"role": "central"},
            train_metadata={
                "capability_labels": ["action_choice", "user_output", None]
            },
            loss_mask=[1, 0, 0],
            response_length=3,
            remove_sample=False,
        ),
        SimpleNamespace(
            metadata={
                "role": "researcher",
                "infrastructure_failure": {"kind": "timeout"},
            },
            train_metadata=None,
            loss_mask=[0, 0],
            response_length=2,
            remove_sample=True,
        ),
    ]

    assert compute_stackplanner_sample_metrics(samples) == {
        "role_tokens/central": 1,
        "role_tokens/researcher": 0,
        "capability_tokens/action_choice": 1,
        "infrastructure_failure_count": 1,
        "removed_sample_count": 1,
    }
