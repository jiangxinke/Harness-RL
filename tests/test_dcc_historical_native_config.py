"""CPU synthetic configuration regressions, independent of benchmark files."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SOURCE / "scripts"), str(SOURCE / "slime")]
import dcc_native_sft as N
from dcc_historical_native_config import HISTORICAL_TRAIN_SOURCE, historical_native_configuration
from slime_plugins.dcc.native_thinking_eval import native_configuration
from slime_plugins.dcc.prompt_history import PromptHistoryConfig

FIXTURE = runpy.run_path(str(SOURCE / "tests/test_dcc_native_training.py"))


def fixture(tmp_path, monkeypatch):
    tokenizer, identity, schema, bundle = FIXTURE["make_record"](tmp_path, monkeypatch)
    identity = replace(identity, source_commit=HISTORICAL_TRAIN_SOURCE)
    receipt = deepcopy(bundle["native_trace"]["native_thinking"]["source_receipt"])
    receipt["source_commit"] = HISTORICAL_TRAIN_SOURCE
    expected = historical_native_configuration(receipt, budget=identity.budget)
    # This is a synthetic protocol fixture, never a rewrite of real data.
    bundle["identity"] = identity.manifest(tokenizer)
    bundle["native_trace"].update(source_commit=HISTORICAL_TRAIN_SOURCE, native_thinking=expected)
    return tokenizer, identity, schema, bundle, receipt


def test_historical_profile_provenance_equals_approved_actual_trace_configuration(tmp_path, monkeypatch):
    tokenizer, identity, schema, bundle, receipt = fixture(tmp_path, monkeypatch)
    assert bundle["native_trace"]["native_thinking"] != native_configuration(receipt, budget=identity.budget)
    assert set(bundle["native_trace"]["native_thinking"]["cuad_structured"]) == {
        "enabled", "schema_version", "backend", "final_review_max_words"}
    root = tmp_path / "source"; root.mkdir()
    (root / ".dcc-native-source.json").write_text(json.dumps(receipt))
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(dict(version="dcc-native-train-profile-v1", identity=identity.manifest(tokenizer),
        web=dict(max_steps=8, explicit_review=False, cached_rows=False))))
    rows = tmp_path / "native.jsonl"; rows.write_text(json.dumps(bundle) + "\n")
    args = SimpleNamespace(adapter=None, cuad_train_data=None, native_expected_source=identity.source_commit,
        native_train_profile=str(profile), cuad_tool_records=None, records=str(rows), model="unused")
    data = N.prepare_native_training(args, source_root=root, tokenizer=tokenizer)
    assert data.configuration["rollout_native_thinking"] == bundle["native_trace"]["native_thinking"]


def test_historical_contract_rejects_inference_overlay_or_unapproved_source(tmp_path, monkeypatch):
    tokenizer, identity, schema, bundle, receipt = fixture(tmp_path, monkeypatch)
    bundle["native_trace"]["native_thinking"] = native_configuration(receipt, budget=identity.budget)
    with pytest.raises(ValueError, match="configuration"):
        N.validate_native_record(bundle, tokenizer, expected_identity=identity, original_schema=schema)
    with pytest.raises(ValueError, match="source"):
        historical_native_configuration({**receipt, "source_commit": "a" * 40}, budget=identity.budget)
    with pytest.raises(ValueError, match="Historical"):
        replace(identity, prompt_history=PromptHistoryConfig(True).to_dict()).manifest(tokenizer)
