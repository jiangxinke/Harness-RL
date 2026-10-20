"""Explicit private TRAIN-only literal text/null row reward candidate.

No benchmark evaluator imports, dataframe inference, numeric conversion, judge,
network or file access. Calling code must retain existing grounding/native checks.
This module is inert until a new label and explicit expected protocol agree.
"""
from __future__ import annotations

from collections import Counter
import json

PROTOCOL = "dcc-web-train-native-text-null-exact-v1"
LABEL_PROTOCOL = "dcc-web-train-native-text-null-label-v1"
TABLE_INPUT_PROTOCOL = "native-json-text-null-no-inference-v1"


class RewardContractError(ValueError):
    """The label, public task or caller approval is invalid; abort preparation."""


class _InvalidPrediction(ValueError):
    pass


def configuration():
    """Return a fresh complete declaration; callers cannot silently amend it."""
    return {
        "version": PROTOCOL,
        "scope": "widesearch-train-only",
        "label_protocol": LABEL_PROTOCOL,
        "table_input_protocol": TABLE_INPUT_PROTOCOL,
        "metric": "train_exact_row_f1",
        "cells": "json-string-or-null-only; exact-unicode-codepoints-and-type",
        "numeric_cells": "forbidden; exact-decimal-text-remains-json-string",
        "string_normalization": "none",
        "nulls": "null-only; distinct-from-empty-string-and-null-like-text",
        "row_key": "all-required-columns-as-ordered-exact-tuple",
        "row_order": "ignored",
        "row_nonemptiness": "at-least-one-nonempty-text-cell; original-native-guard-unchanged",
        "gold_duplicates": "reject-full-row-duplicates",
        "prediction_duplicates": "retain-all-occurrences; excess-is-false-positive",
        "columns": "exact-required-column-names; no-extra-or-missing-content-fields",
        "citations": "optional-1-to-4-distinct-nonempty-strings; excluded-from-content; existing-grounding-required",
        "empty_tables": "both-empty-f1=1; otherwise-2TP/(predicted+gold)",
        "invalid_prediction": "entire-table-reward-zero; never-drop-malformed-rows",
        "limits": {"rows": 4096, "columns": 32, "cell_chars": 65536,
                   "total_text_chars": 1000000, "prediction_json_chars": 2000000},
    }


def _same(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if type(left) is list:
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def _text(value, fail, *, allow_empty=True):
    if type(value) is not str or not allow_empty and not value:
        raise fail("Expected literal JSON text")
    try:
        value.encode("utf-8", "strict")
    except UnicodeError as error:
        raise fail("Unpaired surrogate is not valid UTF-8 text") from error
    return value


def _columns(public_task, label):
    public = public_task.get("metadata", {})
    if type(public) is not dict or set(public) != {"evaluation"}:
        raise RewardContractError("Public TRAIN metadata must contain only public answer columns")
    evaluation = public.get("evaluation")
    if type(evaluation) is not dict or set(evaluation) != {"required"}:
        raise RewardContractError("Public TRAIN evaluation must contain only required columns")
    required = evaluation["required"]
    limits = configuration()["limits"]
    if type(required) is not list or not 1 <= len(required) <= limits["columns"]:
        raise RewardContractError("Required columns exceed the explicit nonempty column bound")
    for column in required:
        _text(column, RewardContractError, allow_empty=False)
        if not column.strip() or column == "citations" or len(column) > limits["cell_chars"]:
            raise RewardContractError("Invalid exact required column name")
    if sum(map(len, required)) > limits["total_text_chars"]:
        raise RewardContractError("Required columns exceed the total text bound")
    if len(set(required)) != len(required):
        raise RewardContractError("Duplicate exact required columns")
    if not _same(label.get("evaluation"), {"required": required}):
        raise RewardContractError("New labels require exactly matching required-only evaluation; no legacy pipeline")
    return required


def _table(rows, required, *, gold):
    fail = RewardContractError if gold else _InvalidPrediction
    limits = configuration()["limits"]
    if type(rows) is not list or len(rows) > limits["rows"]:
        raise fail("Expected bounded native JSON row array")
    keys, used = [], sum(len(column) for column in required)
    for row in rows:
        if type(row) is not dict or set(row) not in (
                set(required), set(required) | ({"citations"} if not gold else set())):
            raise fail("Every row requires exactly the declared content columns; only prediction citations are metadata")
        values = []
        for column in required:
            value = row[column]
            if value is not None:
                _text(value, fail)
                if len(value) > limits["cell_chars"]:
                    raise fail("Literal cell exceeds the declared text bound")
                used += len(value)
            # Tuple tags make null distinct from every literal string.
            values.append(("null",) if value is None else ("string", value))
        if not any(value is not None and value != "" for value in (row[column] for column in required)):
            raise fail("Native rows need at least one nonempty text cell")
        if "citations" in row:
            citations = row["citations"]
            if type(citations) is not list or not 1 <= len(citations) <= 4:
                raise fail("Citation metadata requires one to four distinct nonempty strings")
            for citation in citations:
                _text(citation, fail, allow_empty=False)
                if len(citation) > limits["cell_chars"]:
                    raise fail("Citation metadata exceeds the declared text bound")
                used += len(citation)
            if len(set(citations)) != len(citations):
                raise fail("Duplicate citation metadata")
        if used > limits["total_text_chars"]:
            raise fail("Table exceeds the declared total text bound")
        keys.append(tuple(values))
    if gold and len(set(keys)) != len(keys):
        raise RewardContractError("Gold contains duplicate full required-column rows")
    return keys


def validate_label(label, public_task, *, expected_protocol):
    """Fail closed before a new selector uses a label under this protocol.

    Calling this on a legacy label does not adapt it. The production legacy
    route must remain a separate, unchanged branch in the selector.
    """
    if expected_protocol != PROTOCOL or type(expected_protocol) is not str:
        raise RewardContractError("Explicit caller approval for this TRAIN reward protocol is required")
    if type(label) is not dict or type(public_task) is not dict:
        raise RewardContractError("Label and public TRAIN task must be objects")
    for record in (label, public_task):
        if record.get("split") != "train" or record.get("benchmark") != "widesearch":
            raise RewardContractError("Literal reward is restricted to WideSearch TRAIN")
    if public_task.get("gold") is not None:
        raise RewardContractError("Public TRAIN task must not contain gold")
    task_id = _text(public_task.get("id"), RewardContractError, allow_empty=False)
    if len(task_id) > configuration()["limits"]["cell_chars"]:
        raise RewardContractError("Task identity exceeds the declared text bound")
    if label.get("task_id") != task_id:
        raise RewardContractError("Label/public task identity mismatch")
    if label.get("label_protocol") != LABEL_PROTOCOL:
        raise RewardContractError("A new explicit literal label protocol is required")
    if not _same(label.get("train_reward"), configuration()):
        raise RewardContractError("Label must declare the complete exact supported TRAIN reward configuration")
    required = _columns(public_task, label)
    gold_keys = _table(label.get("gold"), required, gold=True)
    return required, gold_keys


class _JSONNumericToken:
    """A numeric prediction is invalid without first rounding its lexeme."""
    __slots__ = ("lexeme",)

    def __init__(self, lexeme):
        self.lexeme = lexeme


def decode_prediction_json(text):
    """Strict JSON array only; no Markdown, duplicate keys or numeric coercion.

    Cells that are JSON numbers remain rejected tokens; they never become
    floating point values or strings that could accidentally match a label.
    """
    if type(text) is not str or len(text) > configuration()["limits"]["prediction_json_chars"]:
        raise _InvalidPrediction("Expected bounded native JSON text")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise _InvalidPrediction("Duplicate native JSON object key")
            result[key] = value
        return result
    def constant(_):
        raise _InvalidPrediction("Nonfinite JSON token")
    try:
        parsed = json.loads(text, object_pairs_hook=unique, parse_constant=constant,
                            parse_int=_JSONNumericToken, parse_float=_JSONNumericToken)
    except (ValueError, TypeError, RecursionError) as error:
        raise _InvalidPrediction("Invalid strict native JSON prediction") from error
    if type(parsed) is not list:
        raise _InvalidPrediction("Native JSON prediction must be a row array")
    return parsed


def evaluate_train_literal_rows(prediction, label, public_task, *, expected_protocol):
    """Return private TRAIN metrics only, after caller and label double approval."""
    required, gold_keys = validate_label(label, public_task, expected_protocol=expected_protocol)
    result = {"evaluation_protocol": PROTOCOL, "label_protocol": LABEL_PROTOCOL,
              "table_input_protocol": TABLE_INPUT_PROTOCOL,
              "train_reward": configuration(), "grounding_evaluated": False,
              "train_exact_row_precision": 0.0, "train_exact_row_recall": 0.0,
              "train_exact_row_f1": 0.0, "train_exact_table_match": 0,
              "train_num_gold_rows": len(gold_keys), "train_num_prediction_rows": None,
              "train_num_true_positive_rows": 0, "prediction_valid": False,
              "invalid_reason": None}
    try:
        rows = decode_prediction_json(prediction) if type(prediction) is str else prediction
        pred_keys = _table(rows, required, gold=False)
    except _InvalidPrediction as error:
        result["invalid_reason"] = str(error)
        return result
    n_pred, n_gold = len(pred_keys), len(gold_keys)
    true = sum((Counter(pred_keys) & Counter(gold_keys)).values())
    precision = true / n_pred if n_pred else float(n_gold == 0)
    recall = true / n_gold if n_gold else 1.0
    f1 = 2 * true / (n_pred + n_gold) if n_pred + n_gold else 1.0
    result.update(train_exact_row_precision=precision, train_exact_row_recall=recall,
                  train_exact_row_f1=f1, train_exact_table_match=int(true == n_pred == n_gold),
                  train_num_prediction_rows=n_pred, train_num_true_positive_rows=true,
                  prediction_valid=True)
    return result
