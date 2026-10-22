"""Opt-in CUAD TRAIN reward shaping; benchmark scoring is unchanged."""
from __future__ import annotations

from dataclasses import replace
import math

from .metrics import cuad_counts, cuad_predictions


VARIANT = "cuad-train-confidence-brier-discount-v1"


def validate_confidence_reward_weight(weight, *, joint_json_policy=False,
                                      online_web=False, benchmarks=()):
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("CUAD confidence reward weight must be finite and in [0, 1]")
    if weight and (not joint_json_policy or online_web or any(b != "cuad" for b in benchmarks)):
        raise ValueError("CUAD confidence reward requires CUAD-only --joint-json-policy")


def confidence_reward_configuration(weight):
    return dict(version=VARIANT, weight=weight,
        formula="original_reward * (1 - weight * min(1, sum_span_brier / max(1, gold_spans)))",
        target="any-gold-match-under-original-cuad_counts",
        scope="completed-CUAD-TRAIN-only", empty_prediction_brier=0,
        heldout_scoring_changed=False)


def calibrate_cuad_train_group(task, results, *, weight=0):
    """Return shaped copies after rollout, without changing actor inputs.

    A span's target is one iff the original scorer matches any reference.
    The multiplicative discount cannot turn an incorrect/failed zero reward
    positive. Empty predictions receive no discount. Weight zero is an exact
    no-op, including no access to TRAIN labels or episode contents.
    """
    validate_confidence_reward_weight(weight, joint_json_policy=True)
    if weight == 0:
        return results
    if task.benchmark != "cuad" or task.split != "train" or not isinstance(task.gold, list):
        raise ValueError("Confidence reward accepts labeled CUAD TRAIN groups only")
    shaped = []
    for result in results:
        if (result.task_id, result.benchmark, result.split) != (task.id, "cuad", "train"):
            raise ValueError("Confidence reward requires results from the same CUAD TRAIN task")
        if "cuad_confidence_reward" in result.metrics:
            raise ValueError("Confidence reward has already been applied")
        original = float(result.reward)
        if not math.isfinite(original) or not 0 <= original <= 1:
            raise ValueError("Finite original reward in [0, 1] required")
        if result.status != "completed" and original != 0:
            raise ValueError("Failed episodes must retain zero reward")
        predictions = cuad_predictions(result.answer) if result.status == "completed" else []
        errors = [(item["probability"] - float(cuad_counts(task.id, task.gold, [item["text"]])[1] == 0)) ** 2
                  for item in predictions]
        # A fixed TRAIN-label denominator prevents dilution by appending
        # redundant matching spans with perfect confidence and no extra TP.
        denominator = max(1, len(task.gold))
        brier = min(1.0, math.fsum(errors) / denominator)
        reward = original * (1 - weight * brier)
        diagnostics = dict(version=VARIANT, weight=weight, original_reward=original,
            clipped_gold_normalized_brier=brier, sum_span_brier=math.fsum(errors),
            gold_span_denominator=denominator, predicted_spans=len(predictions), shaped_reward=reward)
        shaped.append(replace(result, reward=reward,
            metrics=dict(result.metrics, cuad_confidence_reward=diagnostics)))
    return shaped
