"""CPU-only online reward controls; synthetic data, no model or network."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from slime_plugins.dcc import online_web as web
from slime_plugins.dcc.episode import EpisodeResult
from test_dcc_online_web import ToyTokenizer, URL, COMMIT, public_row, collect, inputs
from test_dcc_online_grpo import load_trainer


class ClosedLabels:
    def __init__(self, label):
        self.label, self.reads = label, 0

    def label_after_close(self, group):
        assert group.closure["journals_flushed"]
        self.reads += 1
        return self.label


def fixture_group(prediction=None, *, gold=None, fetched=None, visible=None, statuses=None):
    task, tokenizer = web.public_web_task(public_row()), ToyTokenizer()
    gold = deepcopy(gold if gold is not None else [
        {"id": "r-alpha", "value": "1.2"}, {"id": "r-beta", "value": "8"},
        {"id": "r-gamma", "value": "3.4"}, {"id": "r-delta", "value": "5.6"}])
    rows = deepcopy(prediction if prediction is not None else gold)
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                row.setdefault("citations", [URL])
    text = fetched if fetched is not None else "\n".join(
        str(row["id"]) + "," + str(row["value"]) for row in gold)
    events = [{"tool": "web_search", "observation": {"results": [{"url": URL}]}},
              {"tool": "fetch_url", "observation": {"url": URL, "text": text}}]
    results = tuple(EpisodeResult(task.id, "widesearch", "train", deepcopy(rows), status,
        0.0, {}, [], [], deepcopy(events)) for status in (statuses or ["completed", "completed"]))
    prompt = tokenizer.encode(visible if visible is not None else text)
    group = web.ClosedPublicGroup(task, results, {"journals_flushed": True}, COMMIT,
                                  tuple(prompt for _ in results))
    label = dict(task_id=task.id, benchmark="widesearch", split="train", gold=gold,
        evaluation=task.metadata["evaluation"], label_protocol=web.LABEL_PROTOCOL,
        train_reward=web.configuration())
    return group, ClosedLabels(label), tokenizer


def score(group, labels, tokenizer, variant=web.ROW_GROUNDED_REWARD):
    return web.score_closed_group(group, inputs=labels, tokenizer=tokenizer, reward_variant=variant)


def test_default_keeps_original_table_gate_and_receipt_shape():
    group, labels, tokenizer = fixture_group()
    rows = deepcopy(group.results[0].answer)
    rows[1]["value"] = "8.0"
    group = replace(group, results=tuple(replace(r, answer=rows) for r in group.results))
    implicit = web.score_closed_group(group, inputs=labels, tokenizer=tokenizer)
    explicit = score(group, labels, tokenizer, web.TABLE_GROUNDED_REWARD)
    assert implicit == explicit and [r.reward for r in implicit[0]] == [0, 0]
    receipt = implicit[1]["receipts"][0]
    assert receipt["wrapper"] == web.WRAPPER and receipt["reward_protocol"] == web.PROTOCOL
    assert receipt["train_metric"] == dict(name="train_exact_row_f1",
        scope="widesearch-train-only", value=0.0, evaluated=False)
    assert receipt["native_cell_grounding_rejections"] == ["cell_absent_from_actual_fetched_observation"]
    assert set(receipt) == {"wrapper", "sample", "task_id", "status", "generation_closed_before_scoring",
        "reward_protocol", "label_protocol", "train_reward", "train_metric",
        "native_cell_grounding_rejections", "benchmark_score", "gradient_group_discarded"}


def test_three_of_four_rows_survive_without_reducing_either_denominator():
    group, labels, tokenizer = fixture_group()
    rows = deepcopy(group.results[0].answer)
    rows[1]["value"] = "8.0"
    group = replace(group, results=(replace(group.results[0], answer=rows),
                                   replace(group.results[1], status="model_protocol_failure", answer=[])))
    results, report = score(group, labels, tokenizer)
    assert [r.reward for r in results] == [.75, 0]
    assert [r.status for r in results] == ["completed", "model_protocol_failure"]
    assert report["all_outcomes_retained"] and not report["gradient_group_discarded"]
    receipt = report["receipts"][0]
    assert receipt["reward_protocol"] == web.ROW_REWARD_PROTOCOL != web.PROTOCOL
    assert receipt["train_metric"]["name"] == web.ROW_REWARD_METRIC != "train_exact_row_f1"
    assert receipt["raw_literal_diagnostic"]["train_exact_row_f1"] == .75
    grounding = receipt["per_row_grounding"]
    assert (grounding["true_positive_rows"], grounding["prediction_rows"], grounding["gold_rows"]) == (3, 4, 4)
    assert [r["eligible"] for r in grounding["rows"]] == [True, False, True, True]
    assert not receipt["benchmark_score"]
    assert report["receipts"][1]["raw_literal_diagnostic"] is None
    assert all(r.reward == 0 for r in group.results)


@pytest.mark.parametrize("bad", [
    [{"id": "r-beta", "value": 8}], [{"id": "r-beta"}],
    [{"id": "r-beta", "value": "8", "extra": "forbidden"}],
    [{"id": "r-beta", "value": "8", "citations": []}],
    [{"id": "r-beta", "value": "8", "citations": [URL, URL]}],
    [{"id": None, "value": None}], [{"id": "", "value": ""}],
    "[{invalid-json]", '[{"id":"r-alpha","id":"r-beta","value":"8"}]',
])
def test_any_malformed_prediction_makes_the_whole_table_zero(bad):
    prediction = [{"id": "r-alpha", "value": "1.2"}, *bad] if isinstance(bad, list) else bad
    group, labels, tokenizer = fixture_group(prediction)
    results, report = score(group, labels, tokenizer)
    assert [r.reward for r in results] == [0, 0]
    for receipt in report["receipts"]:
        assert not receipt["raw_literal_diagnostic"]["prediction_valid"]
        assert receipt["per_row_grounding"]["table_rejections"] == ["invalid_literal_prediction"]
        assert not receipt["per_row_grounding"]["rows"]


def test_duplicate_predictions_remain_false_positives():
    gold = [{"id": "r-beta", "value": "8"}]
    group, labels, tokenizer = fixture_group(gold * 2 + [{"id": "r-beta", "value": "8.0"}], gold=gold)
    results, report = score(group, labels, tokenizer)
    assert results[0].reward == .5  # 2*1/(3 predictions + 1 gold), not 2*1/(1+1).
    grounding = report["receipts"][0]["per_row_grounding"]
    assert grounding["true_positive_rows"] == 1 and grounding["eligible_rows"] == 2
    assert grounding["prediction_rows"] == 3 and grounding["gold_rows"] == 1


@pytest.mark.parametrize("fetched,visible,reason", [
    ("unrelated source", None, "cell_absent_from_actual_fetched_observation"),
    (None, "unrelated final prompt", "cell_absent_from_actual_encoded_final_prompt"),
])
def test_gold_matching_rows_need_both_actual_fetch_and_encoded_final_prompt(fetched, visible, reason):
    group, labels, tokenizer = fixture_group(fetched=fetched, visible=visible)
    results, report = score(group, labels, tokenizer)
    assert results[0].reward == 0
    receipt = report["receipts"][0]
    assert receipt["raw_literal_diagnostic"]["train_exact_row_f1"] == 1
    assert all(row["rejections"] == [reason] for row in receipt["per_row_grounding"]["rows"])


@pytest.mark.parametrize("value", [None, "", "   "])
def test_null_or_empty_raw_literal_gold_does_not_bypass_native_cell_gate(value):
    group, labels, tokenizer = fixture_group(gold=[{"id": "r-beta", "value": value}])
    results, report = score(group, labels, tokenizer)
    assert results[0].reward == 0
    receipt = report["receipts"][0]
    assert receipt["raw_literal_diagnostic"]["train_exact_row_f1"] == 1
    assert receipt["per_row_grounding"]["rows"][0]["rejections"] == ["native_existing_nonempty_string_cell_gate"]


def test_empty_table_and_missing_fetch_preserve_public_native_gate():
    group, labels, tokenizer = fixture_group(prediction=[], gold=[])
    results, report = score(group, labels, tokenizer)
    assert results[0].reward == 0
    assert "empty_answer" in report["receipts"][0]["per_row_grounding"]["table_rejections"]
    group, labels, tokenizer = fixture_group()
    group = replace(group, results=tuple(replace(r, tool_events=r.tool_events[:1]) for r in group.results))
    results, report = score(group, labels, tokenizer)
    assert results[0].reward == 0
    assert "missing_successful_fetch" in report["receipts"][0]["per_row_grounding"]["table_rejections"]


def test_unreliable_group_keeps_all_outcomes_and_skips_scoring():
    group, labels, tokenizer = fixture_group(statuses=["completed", "infrastructure_error"])
    results, report = score(group, labels, tokenizer)
    assert len(results) == 2 and [r.reward for r in results] == [0, 0]
    assert report["gradient_group_discarded"]
    assert all(r["raw_literal_diagnostic"] is None for r in report["receipts"])


def test_row_variant_reads_labels_only_after_real_group_durable_close(tmp_path, monkeypatch):
    data, _ = inputs(tmp_path)
    original, reads = web._read_jsonl, []
    def observed(path, **kwargs):
        if Path(path) == data.labels_path:
            assert (tmp_path / "group/generation_closed.json").is_file()
            reads.append("after-close")
        return original(path, **kwargs)
    monkeypatch.setattr(web, "_read_jsonl", observed)
    group, policy = collect(tmp_path, ["ok", "model-failure"])
    assert not reads and data.label_reads == 0
    with pytest.raises(ValueError, match="before label access"):
        score(replace(group, closure={}), data, policy.tokenizer)
    assert not reads and data.label_reads == 0
    results, _ = score(group, data, policy.tokenizer)
    assert reads == ["after-close"] and [r.reward for r in results] == [1, 0]


@pytest.mark.parametrize("extra,expected", [([], web.TABLE_GROUNDED_REWARD),
    (["--web-train-reward-variant", web.ROW_GROUNDED_REWARD], web.ROW_GROUNDED_REWARD)])
def test_cli_default_and_opt_in_record_separate_reward_metadata(monkeypatch, extra, expected):
    trainer = load_trainer()
    group, _, _ = fixture_group()
    cuad = [web.Task("positive", "cuad", "train", "Q", "contract", ["span"])]
    monkeypatch.setattr(trainer, "load_tasks", lambda _: cuad)
    monkeypatch.setattr(trainer, "validate_source", lambda *a, **k: None)
    monkeypatch.setattr(web, "WebTrainingInputs", lambda *a, **k: SimpleNamespace(tasks=[group.task]))
    monkeypatch.setenv("WORLD_SIZE", "1")
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU CLI tests may not load a model or initialize CUDA")
    monkeypatch.setattr(trainer, "load_policy_model", forbidden)
    monkeypatch.setattr(torch.cuda, "set_device", forbidden)
    captured = []
    async def capture(args, tasks, plan, world_size, *, web_inputs):
        config = trainer.configuration(args, plan, tasks, world_size)
        captured.append(config["online_web"]["online_train_reward"])
        assert args.web_train_reward_variant == expected
    monkeypatch.setattr(trainer, "train", capture)
    assert trainer.main(["--model", "not-loaded", "--train-data", "unused",
        "--output-dir", "unused", "--source-commit", COMMIT, "--enable-online-web",
        "--web-train-data", "unused", "--web-train-labels", "unused", "--web-binding", "unused", *extra]) == 0
    assert captured == [web.online_reward_configuration(expected)]
    if expected == web.ROW_GROUNDED_REWARD:
        assert captured[0]["raw_literal_train_reward"] == web.configuration()
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("mode", [web.ROW_GROUNDED_REWARD, "unknown"])
def test_cli_rejects_unenabled_or_unknown_variant_before_input_access(monkeypatch, mode):
    trainer = load_trainer()
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid CLI may not read inputs")
    monkeypatch.setattr(trainer, "load_tasks", forbidden)
    with pytest.raises(SystemExit) as caught:
        trainer.main(["--model", "not-loaded", "--train-data", "unused", "--output-dir", "unused",
            "--source-commit", COMMIT, "--web-train-reward-variant", mode])
    assert caught.value.code == 2
