"""Shared physical integrity stays separate from native accepted-SFT gates."""
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "scripts"),
                str(Path(__file__).resolve().parents[1] / "slime")]

import pytest
from jsonschema.exceptions import SchemaError

from dcc_native_physical import validate_native_physical
import dcc_native_sft as SFT
from slime_plugins.dcc.native_budget import NativeBudget

_SPEC = importlib.util.spec_from_file_location("native_physical_fixture",
    Path(__file__).with_name("test_dcc_native_training.py"))
_FIXTURE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FIXTURE)


@pytest.mark.parametrize("legacy", [True, False])
def test_shared_helper_preserves_full_sft_ids_and_phase_lengths(tmp_path, monkeypatch, legacy):
    tokenizer, identity, schema, bundle = _FIXTURE.make_record(tmp_path, monkeypatch,
        budget=NativeBudget() if legacy else NativeBudget(16384, "thinking"))
    physical = validate_native_physical(bundle, tokenizer,
        expected_identity=identity, original_schema=schema)
    sft = SFT.validate_native_record(bundle, tokenizer,
        expected_identity=identity, original_schema=schema)
    assert tuple(sft.prompt) == physical.prompt_ids
    assert tuple(sft.completion) == physical.completion_ids
    assert sft.native_phase_lengths == physical.phase_lengths
    assert physical.public_json == bundle["record"]["completion"]


@pytest.mark.parametrize("legacy", [True, False])
def test_sft_still_runs_original_acceptance_gate_after_shared_helper(tmp_path, monkeypatch, legacy):
    tokenizer, identity, schema, bundle = _FIXTURE.make_record(tmp_path, monkeypatch,
        budget=NativeBudget() if legacy else NativeBudget(16384, "thinking"))
    class RejectedSFT(ValueError):
        pass
    def rejected(*args, **kwargs):
        raise RejectedSFT("existing accepted-SFT gate")
    monkeypatch.setattr(SFT.legacy, "validate_record", rejected)
    physical = validate_native_physical(bundle, tokenizer,
        expected_identity=identity, original_schema=schema)
    assert physical.completion_ids
    with pytest.raises(RejectedSFT, match="accepted-SFT gate"):
        SFT.validate_native_record(bundle, tokenizer,
            expected_identity=identity, original_schema=schema)


def test_invalid_trusted_schema_keeps_jsonschema_error_category(tmp_path, monkeypatch):
    tokenizer, identity, _, bundle = _FIXTURE.make_record(tmp_path, monkeypatch)
    for validator in (validate_native_physical, SFT.validate_native_record):
        with pytest.raises(SchemaError):
            validator(bundle, tokenizer, expected_identity=identity,
                      original_schema={"type": "invalid-type"})
