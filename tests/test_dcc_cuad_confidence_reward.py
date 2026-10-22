"""Reward/advantage oracles; these CPU tests do not measure model improvement."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

import dcc_online_grpo as trainer
from slime_plugins.dcc.cuad_confidence_reward import (
    VARIANT, calibrate_cuad_train_group, validate_confidence_reward_weight)
from slime_plugins.dcc.data import Task
from slime_plugins.dcc.episode import EpisodeResult
from slime_plugins.dcc.metrics import cuad_task_metrics, evaluate_cuad
from slime_plugins.dcc.native_behavior_logprobs import BEHAVIOR_LOGPROB_SOURCE
from slime_plugins.dcc.online_grpo import group_signal


def task(**kwargs):
    return Task(**dict(id="contract__Clause", benchmark="cuad", split="train",
        prompt="Find the clause.", document="alpha beta gamma", gold=["alpha beta gamma"], **kwargs))


def episode(t, answer, *, status="completed"):
    step = SimpleNamespace(prompt_ids=[1, 2], completion_ids=[3, 4], thought_token_count=1,
        thought_logprobs=[-1.0], thought_logprobs_source=BEHAVIOR_LOGPROB_SOURCE)
    metrics = cuad_task_metrics(t.id, t.gold, answer) if status == "completed" else {}
    return EpisodeResult(t.id, t.benchmark, t.split, answer, status,
        metrics.get("f1", 0), metrics, [step], [], [])


def test_disabled_path_is_identity_and_does_not_read_labels_or_episodes():
    class Unreadable:
        def __getattribute__(self, name):
            raise AssertionError("disabled calibration must not read inputs")
    results = Unreadable()
    assert calibrate_cuad_train_group(Unreadable(), results) is results


def test_analytic_reward_preserves_official_scores_and_actor_trace():
    t = task()
    r = episode(t, [{"text": "alpha beta gamma", "confidence": .8}])
    before = evaluate_cuad({t.id: t.gold}, {t.id: r.answer})
    shaped = calibrate_cuad_train_group(t, [r], weight=.05)[0]
    assert shaped.reward == pytest.approx(1 * (1 - .05 * .04))
    assert shaped.metrics["cuad_confidence_reward"]["clipped_gold_normalized_brier"] == pytest.approx(.04)
    assert "cuad_confidence_reward" not in r.metrics and r.reward == 1
    assert shaped.answer is r.answer and shaped.steps is r.steps and shaped.messages is r.messages
    assert evaluate_cuad({t.id: t.gold}, {t.id: shaped.answer}) == before
    assert shaped.metrics["f1"] == r.metrics["f1"]


def test_redundant_matching_spans_cannot_dilute_confidence_error():
    t = task()
    answers = [[{"text": "alpha beta gamma", "confidence": .5}]]
    answers.append(answers[0] + [{"text": "alpha beta", "confidence": 1.}])
    r = [episode(t, a) for a in answers]
    assert r[0].metrics == r[1].metrics and r[0].reward == r[1].reward == 1
    shaped = calibrate_cuad_train_group(t, r, weight=.05)
    assert shaped[0].reward == shaped[1].reward == 1 - .05 * .25


def test_parties_substring_rule_uses_original_matching():
    t = replace(task(), id="contract__Parties", gold=["Alpha Inc"])
    answer = [{"text": "Alpha Inc agrees to license every listed item", "confidence": .8}]
    r = episode(t, answer)
    d = calibrate_cuad_train_group(t, [r], weight=.05)[0].metrics["cuad_confidence_reward"]
    assert d["sum_span_brier"] == pytest.approx(.04)
    other = replace(t, id="contract__Other clause")
    d = calibrate_cuad_train_group(other, [episode(other, answer)], weight=.05)[0].metrics["cuad_confidence_reward"]
    assert d["sum_span_brier"] == pytest.approx(.64)


def test_fixed_gold_denominator_and_clipping_bound_discount():
    t = replace(task(), gold=["alpha beta gamma", "other gold span"])
    r = episode(t, [{"text": "alpha beta gamma", "confidence": 0.}])
    shaped = calibrate_cuad_train_group(t, [r], weight=.05)[0]
    assert shaped.reward == pytest.approx(r.reward * (1 - .05 / 2))
    bad = episode(task(), [{"text": "alpha beta gamma", "confidence": 0.},
                          {"text": "wrong span", "confidence": 1.}])
    clipped = calibrate_cuad_train_group(task(), [bad], weight=.05)[0]
    assert clipped.reward == pytest.approx(.95 * bad.reward)
    assert clipped.metrics["cuad_confidence_reward"]["sum_span_brier"] == 2


@pytest.mark.parametrize("status", ["completed", "model_protocol_failure", "max_steps",
                                  "context_budget_failure", "infrastructure_error"])
def test_failures_and_incorrect_zero_reward_stay_zero(status):
    t = task()
    r = episode(t, [{"text": "wrong", "confidence": .5}], status=status)
    assert calibrate_cuad_train_group(t, [r], weight=1)[0].reward == 0


def test_empty_answer_success_keeps_original_reward():
    t = replace(task(), gold=[])
    r = episode(t, [])
    assert r.reward == calibrate_cuad_train_group(t, [r], weight=.05)[0].reward == 1


def test_centered_advantage_rewards_correct_span_confidence_without_changing_behavior():
    t = task()
    raw = [episode(t, [{"text": "alpha beta gamma", "confidence": p}]) for p in (.2, .9)]
    shaped = calibrate_cuad_train_group(t, raw, weight=.05)
    kwargs = dict(group_size=2, max_steps=2, thought_tokens=4, advantage_normalization="centered")
    original = group_signal(raw, **kwargs)
    signal = group_signal(shaped, **kwargs)
    assert original.advantages == [0, 0]
    assert signal.advantages == pytest.approx([-.01575, .01575])
    assert signal.segments == original.segments and signal.discarded_reason is None


@pytest.mark.parametrize("weight", [-.01, 1.01, float("nan"), float("inf"), True])
def test_invalid_weights_rejected(weight):
    with pytest.raises(ValueError, match="weight"):
        validate_confidence_reward_weight(weight, joint_json_policy=True)


@pytest.mark.parametrize("changes", [dict(split="test"), dict(split="validation"),
                                    dict(benchmark="widesearch"), dict(gold=None)])
def test_heldout_and_web_cannot_use_calibration_reward(changes):
    t = replace(task(), **changes)
    with pytest.raises(ValueError, match="CUAD TRAIN"):
        calibrate_cuad_train_group(t, [], weight=.05)


def test_duplicate_application_and_wrong_result_identity_are_rejected():
    t = task()
    r = episode(t, [])
    shaped = calibrate_cuad_train_group(t, [r], weight=.05)
    with pytest.raises(ValueError, match="already"):
        calibrate_cuad_train_group(t, shaped, weight=.05)
    with pytest.raises(ValueError, match="same CUAD TRAIN"):
        calibrate_cuad_train_group(t, [replace(r, task_id="other")], weight=.05)


def test_cli_rejects_nonjoint_before_reading_input_files(capsys):
    with pytest.raises(SystemExit) as e:
        trainer.main(["--model", "unused", "--train-data", "unused", "--output-dir", "unused",
            "--source-commit", "0" * 40, "--cuad-confidence-reward-weight", ".05"])
    assert e.value.code == 2 and "CUAD-only --joint-json-policy" in capsys.readouterr().err


def test_configuration_has_exact_default_identity_and_explicit_variant():
    args = SimpleNamespace(model="toy", source_commit="0" * 40, group_size=2,
        max_steps=1, thought_tokens=4, json_tokens=192, joint_json_policy=True)
    before = trainer.configuration(args, [], [], 1)
    args.cuad_confidence_reward_weight = 0
    assert trainer.configuration(args, [], [], 1) == before
    args.cuad_confidence_reward_weight = .05
    config = trainer.configuration(args, [], [], 1)
    assert config["cuad_train_confidence_reward"]["weight"] == .05
    assert VARIANT in config["algorithm"]
    assert config["objective_denominator"] == before["objective_denominator"]
