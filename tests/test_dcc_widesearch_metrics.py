import ast

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


def test_judge_batches_keep_full_reference_and_global_cell_indices():
    rows = [
        {"name": f"entity{index}", "year": "2000", "description": "text"}
        for index in range(70)
    ]
    alignment_sizes, cell_indices = [], []

    def judge(prompt):
        if "align two vocabularies" in prompt:
            values = ast.literal_eval(
                prompt.split("The vocabulary to be aligned is as follows:\n", 1)[
                    1
                ].split("\n\nThe reference vocabulary", 1)[0]
            )
            reference = ast.literal_eval(
                prompt.split("The reference vocabulary is as follows:\n", 1)[1].split(
                    "\n\nThe alignment rules", 1
                )[0]
            )
            assert len(reference) == 70
            alignment_sizes.append(len(values))
            return {value: value for value in values}
        pairs = ast.literal_eval(
            prompt.split("====== response-start ======\n", 1)[1].split(
                "\n====== response-end", 1
            )[0]
        )
        cell_indices.append(list(pairs))
        return {key: 1 for key in pairs}

    result = evaluate_widesearch(rows, rows, configuration(), judge=judge)
    assert alignment_sizes == [32, 32, 6]
    assert [len(batch) for batch in cell_indices] == [32, 32, 6]
    assert [key for batch in cell_indices for key in batch] == [
        f"idx_{index}" for index in range(70)
    ]
    assert result["f1_by_item"] == result["score"] == 1
    assert result["num_gold_rows"] == 70
    assert result["judge_batch_size"] == 32


def test_cell_character_budget_never_drops_or_truncates_large_items():
    from slime_plugins.dcc.widesearch_metrics import _judge_cells

    seen = []

    def judge(prompt):
        pairs = ast.literal_eval(
            prompt.split("====== response-start ======\n", 1)[1].split(
                "\n====== response-end", 1
            )[0]
        )
        seen.extend(pairs.values())
        return {key: 1 for key in pairs}

    predicted = ["a" * 40, "b" * 400, "c" * 40]
    values = _judge_cells(
        predicted, predicted, "same", judge, batch_size=32, max_chars=150
    )
    assert values == [1, 1, 1]
    assert [item["response"] for item in seen] == predicted


def test_missing_cell_in_later_batch_fails_instead_of_shrinking_denominator():
    from slime_plugins.dcc.widesearch_metrics import _judge_cells

    def judge(prompt):
        return {"idx_0": 1, "idx_1": 1}

    with pytest.raises(JudgeError, match="omitted a cell"):
        _judge_cells(["x"] * 3, ["x"] * 3, "same", judge, batch_size=2, max_chars=1000)
