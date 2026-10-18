from __future__ import annotations

import json

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
    with pytest.raises(ValueError, match="at least one"):
        evaluate_results([], [])


def test_infrastructure_episode_does_not_count_as_complete_evaluation():
    report = evaluate_results(
        [wide("a")], [{"task_id": "a", "answer": [], "status": "infrastructure_error"}]
    )
    assert not report["complete"]
    assert report["widesearch"]["score"] == 0


def test_search_coverage_counts_actual_scopes_and_exports_no_operator_details():
    report = evaluate_results(
        [wide("a"), wide("b"), wide("c")],
        [
            {
                "task_id": "a",
                "answer": [],
                "tool_events": [
                    {
                        "tool": "web_search",
                        "observation": {
                            "configured_backend": "duckduckgo_wikipedia",
                            "backend": "wikipedia",
                            "search_scope": "encyclopedia",
                            "fallback_reason": "DuckDuckGo challenged this request",
                        },
                    },
                    {
                        "tool": "web_search",
                        "observation": {
                            "configured_backend": "serper",
                            "backend": "serper",
                            "search_scope": "web",
                            "fallback_reason": None,
                        },
                    },
                    {"tool": "fetch_url", "observation": {"backend": "wikipedia"}},
                ],
            },
            {
                "task_id": "b",
                "answer": [],
                "tool_events": [
                    {
                        "tool": "web_search",
                        "observation": {
                            "configured_backend": "http://private.invalid/server/path",
                            "backend": "http://private.invalid",
                            "search_scope": "private/path",
                            "fallback_reason": "HTTP retrieval failed http://private.invalid/server/path",
                        },
                    },
                ],
            },
        ],
    )
    coverage = report["widesearch"]["search_coverage"]
    assert coverage["search_calls"] == 3
    assert coverage["tasks_with_search"] == 2
    assert coverage["tasks_without_search"] == 1
    assert coverage["tasks_with_web_search"] == 1
    assert coverage["tasks_with_encyclopedia_search"] == 1
    assert coverage["encyclopedia_search_fraction"] == pytest.approx(1 / 3)
    assert coverage["tasks_with_encyclopedia_search_fraction"] == pytest.approx(1 / 3)
    assert coverage["backend_counts"] == {"wikipedia": 1, "serper": 1, "unknown": 1}
    assert coverage["fallback_calls"] == 2
    assert coverage["fallback_reason_counts"] == {"challenge": 1, "provider_error": 1}
    assert "private.invalid" not in json.dumps(coverage)
    assert "server/path" not in json.dumps(coverage)


def test_search_coverage_handles_legacy_missing_and_malformed_observations():
    report = evaluate_results(
        [wide("a"), wide("b")],
        [
            {
                "task_id": "a",
                "tool_events": ["bad", {"tool": "web_search", "observation": None}],
            },
            {"task_id": "b", "tool_events": None},
        ],
    )
    coverage = report["widesearch"]["search_coverage"]
    assert coverage["backend_counts"] == {"unknown": 1}
    assert coverage["scope_counts"] == {"unknown": 1}
    assert coverage["encyclopedia_search_fraction"] == 0
    no_search = evaluate_results([wide("a")], [])
    assert (
        no_search["widesearch"]["search_coverage"]["encyclopedia_search_fraction"]
        is None
    )


def test_search_coverage_recognizes_public_web_indexes():
    report = evaluate_results(
        [wide("a")],
        [{"task_id": "a", "answer": [], "tool_events": [
            {"tool": "web_search", "observation": {
                "configured_backend": "public_web", "backend": backend,
                "search_scope": "web", "fallback_reason": "duckduckgo_lite: challenge",
            }} for backend in ("seznam", "baidu_mobile")
        ]}],
    )
    coverage = report["widesearch"]["search_coverage"]
    assert coverage["backend_counts"] == {"seznam": 1, "baidu_mobile": 1}
    assert coverage["configured_backend_counts"] == {"public_web": 2}
    assert coverage["encyclopedia_search_fraction"] == 0
