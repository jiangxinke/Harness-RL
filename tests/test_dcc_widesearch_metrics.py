import pytest

from slime_plugins.dcc.widesearch_metrics import JudgeError, evaluate_widesearch


def configuration():
    return {
        "required": ["name", "year", "description"],
        "unique_columns": ["name"],
        "eval_pipeline": {
            "name": {"preprocess": ["norm_str"], "metric": ["exact_match"]},
            "year": {
                "preprocess": ["extract_number"],
                "metric": ["number_near"],
                "criterion": 0.01,
            },
            "description": {
                "preprocess": ["norm_str"],
                "metric": ["llm_judge"],
                "criterion": "same meaning",
            },
        },
    }


def test_row_and_item_metrics_have_different_denominators():
    gold = [
        {"name": "alpha", "year": "2000", "description": "first"},
        {"name": "beta", "year": "2001", "description": "second"},
    ]
    prediction = [
        {"name": "alpha", "year": "2000", "description": "wrong"},
        {"name": "extra", "year": "2000", "description": "other"},
    ]
    result = evaluate_widesearch(prediction, gold, configuration())
    assert result["f1_by_row"] == 0
    assert result["f1_by_item"] == pytest.approx(1 / 3)
    assert result["score"] == 0
    assert result["evaluation_protocol"] == "strict_exact_proxy"


def test_markdown_and_json_rows_score_equally():
    gold = [{"name": "alpha", "year": "2000", "description": "first"}]
    text = "```markdown\n| Name | Year | Description |\n| --- | --- | --- |\n| ALPHA | 2000 | first |\n```"
    result = evaluate_widesearch(text, gold, configuration())
    assert result["score"] == 1
    assert result["f1_by_item"] == 1


def test_custom_judge_aligns_entities_and_scores_cells_separately():
    calls = []

    def judge(prompt):
        calls.append(prompt)
        if "align two vocabularies" in prompt:
            return {"A": "alpha"}
        return {"idx_0": 1}

    gold = [{"name": "alpha", "year": "2000", "description": "first"}]
    result = evaluate_widesearch(
        [{"name": "A", "year": "2000", "description": "initial"}],
        gold,
        configuration(),
        judge=judge,
    )
    assert result["score"] == 1
    assert result["evaluation_protocol"] == "official_protocol_custom_judge"
    assert len(calls) == 2
    assert "same meaning" in calls[1]


def test_judge_failures_do_not_become_zero_rewards():
    def judge(prompt):
        return "bad json"

    with pytest.raises(JudgeError, match="Malformed"):
        evaluate_widesearch(
            [{"name": "A", "year": "2000", "description": "x"}],
            [{"name": "a", "year": "2000", "description": "x"}],
            configuration(),
            judge=judge,
        )


def test_duplicate_primary_keys_removed_before_counting():
    row = {"name": "alpha", "year": "2000", "description": "first"}
    result = evaluate_widesearch([row, row], [row], configuration())
    assert result["num_pred_rows"] == 1
    assert result["score"] == 1


def test_missing_columns_are_not_silently_ignored():
    result = evaluate_widesearch(
        [{"name": "a"}],
        [{"name": "a", "year": "1", "description": "x"}],
        configuration(),
    )
    assert result["score"] == 0
    assert result["msg"] == "Response columns do not match required columns"


def test_empty_matching_keys_have_finite_zero_scores():
    result = evaluate_widesearch(
        [{"name": "b", "year": "2000", "description": "x"}],
        [{"name": "a", "year": "2000", "description": "x"}],
        configuration(),
    )
    assert result["f1_by_row"] == 0
    assert result["f1_by_item"] == 0


def test_native_citation_metadata_is_separate_from_answer_columns():
    row = {"name": "alpha", "year": "2000", "description": "first"}
    with_citations = {**row, "citations": ["https://example.com/alpha"]}
    assert evaluate_widesearch([with_citations], [row], configuration())["score"] == 1
    assert (
        evaluate_widesearch([{**row, "extra": "bad"}], [row], configuration())["score"]
        == 0
    )


def test_judge_may_leave_unmatched_entities_unchanged():
    def judge(prompt):
        return {"unmatched": "unmatched"}

    row = {"name": "alpha", "year": "2000", "description": "first"}
    result = evaluate_widesearch(
        [{**row, "name": "unmatched"}], [row], configuration(), judge=judge
    )
    assert result["score"] == 0
    assert result["matched_rows"] == 0
