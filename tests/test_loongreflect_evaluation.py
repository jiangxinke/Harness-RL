from __future__ import annotations

import json
from pathlib import Path

from slime_plugins.loongreflect.core import RolloutTree
from slime_plugins.loongreflect.evaluation import (
    DATASET_FILES,
    FORCED_ANSWER_PREFIX,
    force_final_answer,
    normalize_answer,
    original_dataset_weights,
    prediction_from_tree,
    score_prediction,
    select_eval_rows,
    summarize_records,
)
from slime_plugins.loongreflect.slime_adapter import GeneratedTurn


class ForcedModel:
    base_prompt = "base"

    def __init__(self) -> None:
        self.prompt = None

    def _fit_generation_prompt(self, continuation, forced_prefix, max_tokens):
        assert forced_prefix == FORCED_ANSWER_PREFIX
        assert max_tokens == 256
        return "base" + continuation + forced_prefix, continuation

    def _generate(self, prompt, *, max_tokens, stop, temperature):
        self.prompt = prompt
        assert stop == ["</answer>"]
        assert temperature == 0.0
        return GeneratedTurn("Paris</answer>", [1, 2], [-0.1, -0.2], "stop")


def _max_step_tree() -> RolloutTree:
    tree = RolloutTree("capital?")
    active = tree.root_id
    for index in range(32):
        active = tree.append(active, "think", f"turn {index}")
        active = tree.append(active, "reflect", f"continue {index}")
    terminal_reflect = tree.append(
        active,
        "reflect",
        "terminated",
        {"source": "harness", "termination_reason": "max_steps_reached"},
    )
    tree.append(
        terminal_reflect,
        "backtrack",
        "max_steps_reached",
        {"termination": True, "termination_reason": "max_steps_reached"},
    )
    tree.model_turns = 32
    tree.status = "terminated"
    tree.termination_reason = "max_steps_reached"
    return tree


def test_normalized_multi_alias_recall_and_lexical_accuracy():
    assert normalize_answer("The, Eiffel Tower!") == "eiffel tower"
    f1, accuracy = score_prediction(
        "Her Luck in London", ["wrong", "Her Luck In London"]
    )
    assert f1 == 1.0
    assert accuracy == 1.0
    partial_f1, partial_accuracy = score_prediction("New York", ["New York City"])
    assert partial_f1 == 2 / 3
    assert partial_accuracy == 0.0


def test_forced_turn_after_any_answerless_termination():
    tree = _max_step_tree()
    model = ForcedModel()

    assert force_final_answer(tree, model)
    assert prediction_from_tree(tree) == "Paris"
    assert tree.model_turns == 33
    assert tree.termination_reason == "forced_answer_after_max_steps"
    assert model.prompt.endswith(FORCED_ANSWER_PREFIX)
    assert not force_final_answer(tree, model)

    early = _max_step_tree()
    early.model_turns = 31
    early.termination_reason = "invalid_action_retry_limit"
    assert force_final_answer(early, ForcedModel())
    assert early.model_turns == 32
    assert early.termination_reason == "forced_answer_after_termination"

    other_terminal = _max_step_tree()
    other_terminal.termination_reason = "invalid_action_retry_limit"
    assert force_final_answer(other_terminal, ForcedModel())


def test_forced_turn_recovers_unmarked_controller_failure_prefix():
    tree = RolloutTree("capital?")
    think = tree.append(tree.root_id, "think", "Paris is likely.")
    reflect = tree.append(think, "reflect", "The model produced an invalid action.")
    tree.append(reflect, "backtrack", "invalid action", {"source": "harness"})
    tree.model_turns = 2
    tree.status = "terminated"
    tree.termination_reason = "invalid_action_retry_limit"

    model = ForcedModel()
    assert force_final_answer(tree, model)
    assert "<think>Paris is likely.</think>" in model.prompt
    assert tree.termination_reason == "forced_answer_after_termination"


def test_balanced_630_to_315_selection_is_deterministic():
    rows = [
        {
            "source": source,
            "split": "dev",
            "query": f"{source}-{index}",
            "answer": [str(index)],
        }
        for source in DATASET_FILES
        for index in range(90)
    ]
    selected = select_eval_rows(rows, 315)
    selected_again = select_eval_rows(list(reversed(rows)), 315)
    assert len(selected) == 315
    assert [row["query"] for row in selected] == [
        row["query"] for row in selected_again
    ]
    counts = {
        source: sum(row["source"] == source for row in selected)
        for source in DATASET_FILES
    }
    assert set(counts.values()) == {45}


def test_original_sizes_and_weighted_metrics(tmp_path: Path):
    expected = {
        "2wikimultihopqa": 2,
        "bamboogle": 3,
        "frames": 5,
        "hotpotqa": 7,
        "musique": 11,
        "nq": 17,
        "triviaqa": 23,
    }
    for source, paths in DATASET_FILES.items():
        remaining = expected[source]
        for index, relative in enumerate(paths):
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            count = remaining if index == len(paths) - 1 else remaining // 2
            remaining -= count
            path.write_text(
                "".join(json.dumps({"id": item}) + "\n" for item in range(count))
            )

    assert original_dataset_weights(tmp_path) == expected
    records = [
        {"source": source, "f1": float(index % 2), "accuracy": float(index % 3 == 0)}
        for index, source in enumerate(expected)
    ]
    summary = summarize_records(records, expected)
    wanted_f1 = sum(
        expected[source] * record["f1"] for source, record in zip(expected, records)
    ) / sum(expected.values())
    wanted_acc = sum(
        expected[source] * record["accuracy"]
        for source, record in zip(expected, records)
    ) / sum(expected.values())
    assert summary["original_size_weighted_f1"] == wanted_f1
    assert summary["original_size_weighted_accuracy"] == wanted_acc


def test_terminated_predictions_are_excluded_from_metrics():
    weights = {source: 1 for source in DATASET_FILES}
    records = [
        {
            "source": source,
            "prediction": "ok",
            "status": "completed",
            "f1": 1.0,
            "accuracy": 1.0,
        }
        for source in DATASET_FILES
    ]
    records.append(
        {
            "source": "frames",
            "prediction": "",
            "status": "terminated",
            "f1": 0.0,
            "accuracy": None,
        }
    )

    summary = summarize_records(records, weights)

    assert summary["total_count"] == 8
    assert summary["included_count"] == 7
    assert summary["excluded_terminated_count"] == 1
    assert summary["per_dataset"]["frames"]["included_count"] == 1
    assert summary["original_size_weighted_f1"] == 1.0
