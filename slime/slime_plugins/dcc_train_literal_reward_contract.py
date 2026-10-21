"""Inert complete declaration and receipt validation for explicit TRAIN literal rewards.

No evaluator, network, tensor or file access. This is not a benchmark protocol.
Caller approvals assert provenance; they are not cryptographic producer evidence.
"""
from __future__ import annotations
import math

PROTOCOL = "dcc-web-train-native-text-null-exact-v1"
LABEL_PROTOCOL = "dcc-web-train-native-text-null-label-v1"
TABLE_INPUT_PROTOCOL = "native-json-text-null-no-inference-v1"


class RewardContractError(ValueError):
    """The label, public task or caller approval is invalid; abort preparation."""


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


LITERAL_RECEIPT = "native-final-train-literal-row-reward-receipt-v1"
LITERAL_REWARD_INPUTS = "native-final-train-literal-row-reward-inputs-v1"
LITERAL_DPO_COLLECTIONS = "native-final-dpo-literal-row-collections-v1"
PER_COLLECTION_RECEIPTS = "native-final-train-per-collection-reward-receipts-v1"
LITERAL_PROFILE_FIELDS = {"reward_contract", "train_reward", "train_metric"}
LEGACY_RECEIPT_KEYS = {"contract", "task_id", "trace_id", "prefix_id", "harness_id",
    "reward_version", "value", "generation_closed_before_scoring", "split"}
LITERAL_RECEIPT_KEYS = LEGACY_RECEIPT_KEYS | {
    "train_reward", "label_protocol", "train_metric", "evidence_guard"}


def validate_approval(train_reward, *, expected_protocol):
    if type(expected_protocol) is not str or expected_protocol != PROTOCOL:
        raise RewardContractError("Explicit new TRAIN reward protocol approval required")
    if not _same(train_reward, configuration()):
        raise RewardContractError("Complete exact supported TRAIN reward configuration required")
    return configuration()


def _bounded(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def validate_literal_receipt(row, *, reward_version, train_reward, evidence_guard_version):
    """Validate all metadata BEFORE a caller strips it to TrainRewardReceipt."""
    validate_approval(train_reward, expected_protocol=reward_version)
    if type(row) is not dict or set(row) != LITERAL_RECEIPT_KEYS:
        raise RewardContractError("Exact literal TRAIN reward receipt fields required")
    if (row['contract'] != LITERAL_RECEIPT or row['reward_version'] != PROTOCOL
            or row['label_protocol'] != LABEL_PROTOCOL or row['split'] != 'train'
            or row['generation_closed_before_scoring'] is not True
            or not _same(row['train_reward'], train_reward)):
        raise RewardContractError("Literal TRAIN reward receipt protocol/configuration differs")
    if not all(type(row[key]) is str and bool(row[key].strip())
            for key in ('task_id','trace_id','prefix_id','harness_id')) or not _bounded(row['value']):
        raise RewardContractError("Literal TRAIN reward ownership/value required")
    metric, guard = row['train_metric'], row['evidence_guard']
    if (type(metric) is not dict or set(metric) != {'name','scope','value','evaluated'}
            or metric['name'] != 'train_exact_row_f1' or metric['scope'] != 'widesearch-train-only'
            or not _bounded(metric['value']) or type(metric['evaluated']) is not bool
            or type(guard) is not dict or set(guard) != {'version','passed'}
            or guard['version'] != evidence_guard_version or type(guard['passed']) is not bool
            or metric['evaluated'] is not guard['passed']):
        raise RewardContractError("Explicit TRAIN-only metric and unchanged evidence guard required")
    if (guard['passed'] and row['value'] != metric['value']) or (
            not guard['passed'] and (row['value'] != 0 or metric['value'] != 0)):
        raise RewardContractError("Literal reward must equal guarded TRAIN exact-row F1")
    return {key: row[key] for key in LEGACY_RECEIPT_KEYS - {'contract'}}
