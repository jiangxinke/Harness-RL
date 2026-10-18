from __future__ import annotations

import pytest

from slime_plugins.dcc import Task
from slime_plugins.dcc.evaluation import evaluate_results


def wide(task_id):
    return Task(
        task_id,
        "widesearch",
        "test",
        "Find entries",
        gold=[{"name": "alpha"}],
        metadata={
            "evaluation": {
                "required": ["name"],
                "unique_columns": ["name"],
                "eval_pipeline": {
                    "name": {"metric": ["exact_match"], "preprocess": ["norm_str"]}
                },
            }
        },
    )


def test_missing_predictions_stay_in_benchmark_denominator():
    report = evaluate_results(
        [wide("a"), wide("b")],
        [{"task_id": "a", "answer": [{"name": "alpha"}], "status": "completed"}],
    )
    assert report["prediction_coverage"] == 0.5
    assert report["widesearch"]["metric_denominator"] == 2
    assert report["widesearch"]["score"] == 0.5
    assert not report["complete"]


def test_judge_failures_make_report_incomplete_and_have_no_fabricated_score():
    def judge(prompt):
        raise OSError("offline")

    report = evaluate_results(
        [wide("a")],
        [{"task_id": "a", "answer": [{"name": "alpha"}], "status": "completed"}],
        judge=judge,
    )
    assert report["widesearch"]["judge_failures"] == 1
    assert report["widesearch"]["score"] is None
    assert report["widesearch"]["metric_denominator"] == 0
    assert not report["complete"]


def test_duplicate_or_foreign_prediction_ids_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_results([wide("a")], [{"task_id": "a"}, {"task_id": "a"}])
    with pytest.raises(ValueError, match="outside"):
        evaluate_results([wide("a")], [{"task_id": "other"}])


def test_infrastructure_episode_does_not_count_as_complete_evaluation():
    report = evaluate_results(
        [wide("a")], [{"task_id": "a", "answer": [], "status": "infrastructure_error"}]
    )
    assert not report["complete"]
    assert report["widesearch"]["score"] == 0
