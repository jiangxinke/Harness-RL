"""Pre-registration arithmetic boundaries with synthetic data only.

These tests do not install/adopt a reward version or add a live scorer. Existing
scorers and evidence guards are imported unchanged; no private labels are read.
"""
from copy import deepcopy
import math

import pytest

from dcc_collect_native_train import web_train_evaluation
from slime_plugins.dcc.data import Task
from slime_plugins.dcc.episode import _WebEvidence
from slime_plugins.dcc.metrics import cuad_predictions, cuad_task_metrics
from slime_plugins.dcc.widesearch_metrics import evaluate_widesearch

URL = "https://example.org/public-list"
COLUMNS = ["name", "city", "year"]
GOLD = [{"name": "Alpha", "city": "East", "year": "2000"},
        {"name": "Beta", "city": "West", "year": "2001"}]


def public_task():
    return Task("synthetic-independent-train", "widesearch", "train", "List the public entities.",
        metadata={"evaluation": {"required": COLUMNS}})


def evaluation(*, compound):
    task = public_task()
    label = {"gold": deepcopy(GOLD), "evaluation": {"required": COLUMNS}}
    if not compound:
        label["evaluation"].update(unique_columns=["name"], eval_pipeline={
            column: {"preprocess": [], "metric": ["exact_match"]} for column in COLUMNS})
    return web_train_evaluation(label, task.to_dict())


def cited(rows):
    return [dict(row, citations=[URL]) for row in rows]


def evidence():
    value = _WebEvidence()
    value.observe("web_search", {"results": [{"url": URL}]})
    value.observe("fetch_url", {"url": URL, "text": "Actual public source text."})
    return value


def proposed_web_arithmetic(rows, profile, *, guard=None):
    """Local reference arithmetic only, intentionally absent from production."""
    guard = evidence() if guard is None else guard
    if guard.reject_reasons(public_task(), rows):
        return 0.0, None
    score = evaluate_widesearch(rows, GOLD, profile)
    assert score["evaluation_protocol"] == "strict_exact_proxy"
    return 0.75 * score["f1_by_row"] + 0.25 * score["f1_by_item"], score


@pytest.mark.parametrize("rows", [[], GOLD, [GOLD[0]],
    [dict(GOLD[0], year="wrong"), GOLD[1]], [GOLD[0], GOLD[0], GOLD[1]]])
def test_required_only_compound_key_makes_row_and_item_f1_identical(rows):
    profile = evaluation(compound=True)
    assert profile["unique_columns"] == COLUMNS
    score = evaluate_widesearch(cited(rows), GOLD, profile)
    assert score["f1_by_row"] == score["f1_by_item"]
    assert 0.75 * score["f1_by_row"] + 0.25 * score["f1_by_item"] == pytest.approx(score["f1_by_row"])


def test_original_entity_key_separates_complete_rows_from_partial_cells():
    # A: one fully correct row. B: both entities but every row has one wrong cell.
    # B preserves item F1 while destroying complete-row F1. C then raises item
    # F1 but still reduces complete-row F1; the proposed mixture favors A.
    # Nothing here changes the original entity key or scorer.
    profile = evaluation(compound=False)
    a, a_score = proposed_web_arithmetic(cited([GOLD[0]]), profile)
    b, b_score = proposed_web_arithmetic(cited([dict(row, year="wrong") for row in GOLD]), profile)
    assert a_score["f1_by_row"] == pytest.approx(2 / 3)
    assert b_score["f1_by_row"] == 0
    assert b_score["f1_by_item"] == pytest.approx(2 / 3)
    assert b_score["f1_by_item"] == a_score["f1_by_item"]
    assert a > b
    c_rows = [dict(row, year="wrong") for row in GOLD]
    c_rows[0] = deepcopy(GOLD[0])
    c, c_score = proposed_web_arithmetic(cited(c_rows), profile)
    assert c_score["f1_by_item"] > a_score["f1_by_item"]
    assert c_score["f1_by_row"] < a_score["f1_by_row"]
    assert a > c


@pytest.mark.parametrize("case", ["empty", "unread", "nosearch", "nofetch", "missingcolumn"])
def test_evidence_guard_rejection_proposes_zero_without_repairing_candidate(case):
    rows, guard = cited(GOLD), evidence()
    if case == "empty": rows = []
    elif case == "unread": rows[0]["citations"] = ["https://unread.example.org"]
    elif case == "nosearch": guard.searches = 0
    elif case == "nofetch": guard.fetches = 0
    else: del rows[0]["year"]
    original = deepcopy(rows)
    reward, score = proposed_web_arithmetic(rows, evaluation(compound=False), guard=guard)
    assert reward == 0 and score is None and rows == original


def test_proposal_is_bounded_and_duplicate_rows_keep_original_denominators():
    profile = evaluation(compound=False)
    reward, score = proposed_web_arithmetic(cited(GOLD + [GOLD[0]]), profile)
    assert reward == 1 and score["num_pred_rows"] == 2 and score["num_gold_rows"] == 2
    for rows in [GOLD, [GOLD[0]], [dict(row, year="wrong") for row in GOLD]]:
        reward, _ = proposed_web_arithmetic(cited(rows), profile)
        assert math.isfinite(reward) and 0 <= reward <= 1


def original_cuad_reward(gold, answer, document):
    # Exactly the current EpisodeRunner reward arithmetic, no proposed change.
    f1 = cuad_task_metrics("synthetic-cuad", gold, answer)["f1"]
    spans = cuad_predictions(answer)
    normalized = " ".join(document.split())
    grounded = [" ".join(item["text"].split()) in normalized for item in spans]
    return f1 * (sum(grounded) / len(grounded) if grounded else 1.0)


@pytest.mark.parametrize("gold,answer,expected", [
    ([], [], 1.0), (["Alpha keeps secrets."], [], 0.0),
    ([], [{"text": "Alpha keeps secrets.", "confidence": 1.0}], 0.0),
    (["Alpha keeps secrets."], [{"text": "Alpha keeps secrets.", "confidence": 0.5}], 1.0),
    (["Alpha keeps secrets."], [{"text": "Alpha keeps secret.", "confidence": 0.5}], 0.0),
])
def test_cuad_original_grounding_and_negative_empty_boundaries(gold, answer, expected):
    assert original_cuad_reward(gold, answer, "Alpha keeps secrets.") == expected
