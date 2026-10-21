"""A cache marker cannot silently fall through an old trusted resolver."""
from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'slime')]
import dcc_native_sft as N
import dcc_review_training as R
from slime_plugins.dcc.web_rows import web_cached_rows_configuration

F = runpy.run_path(str(ROOT / 'tests/test_dcc_native_train_data_fixes.py'))


@pytest.fixture
def legacy(tmp_path):
    case = F['NativeCollectorIntegrationTests'](); case.setUp()
    try:
        output, _ = case.generate()
        case.select(output, case.labels())
        web = case.root / 'selected/native_web_train.jsonl'
        yield case, web
    finally:
        case.tearDown()


def test_unmarked_legacy_lambda_and_bound_resolver_remain_equivalent_and_unmodified(legacy):
    case, web = legacy
    before = web.read_bytes()
    a = N.load_native_training_data(web, case.tokenizer, expected_identity=case.identity,
        source_receipt=case.receipt, schema_resolver=case.schemas.original_schema)
    b = N.load_native_training_data(web, case.tokenizer, expected_identity=case.identity,
        source_receipt=case.receipt, schema_resolver=lambda record, collector: case.schemas.original_schema(record, collector))
    assert len(a.records) == len(b.records) == 3 and a.records == b.records and a.examples == b.examples
    assert web.read_bytes() == before


@pytest.mark.parametrize('marker', [{'corrupt': True}, None, False, {}, {'version': 'old'}])
@pytest.mark.parametrize('wrapped', [False, True])
def test_marked_legacy_cannot_use_bound_old_profile_or_trusted_lambda(legacy, marker, wrapped):
    case, web = legacy
    rows = [json.loads(line) for line in web.read_text().splitlines()]
    for row in rows: row['collector']['web_cached_rows'] = deepcopy(marker)
    web.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    resolver = ((lambda record, collector: case.schemas.original_schema(record, collector))
                if wrapped else case.schemas.original_schema)
    with pytest.raises(ValueError, match='Cached TRAIN marker requires its explicit trusted Web profile'):
        N.load_native_training_data(web, case.tokenizer, expected_identity=case.identity,
                                   source_receipt=case.receipt, schema_resolver=resolver)


def test_explicit_new_profile_and_registry_still_validate_complete_marker(legacy):
    case, web = legacy
    bundle = json.loads(web.read_text().splitlines()[0])
    bundle['collector']['web_cached_rows'] = {'corrupt': True}
    new = N.WebSchemaProfile(20, True, True, True, R.WEB32K_CACHED_PROFILE,
                            cached_rows_configuration=web_cached_rows_configuration(True))
    schema = case.schemas.original_schema(bundle['record'], bundle['collector'])
    with pytest.raises(ValueError, match='Complete cached TRAIN request receipt required'):
        N.validate_native_record(bundle, case.tokenizer, expected_identity=case.identity,
            original_schema=schema, web_schema_profile=new, cached_sources={})


def test_real_cached_bundle_requires_profile_registry_and_preserves_native_targets(tmp_path, monkeypatch):
    h = runpy.run_path(str(ROOT / 'tests/test_dcc_train_cached_rows.py'))
    _, destination, tokenizer, identity, _, schemas, receipt, _ = h['select'](tmp_path, monkeypatch)
    registry = h['W'].load_sources(destination / 'native_web_cached_sources.jsonl', source_receipt=receipt,
                                   profile_version=R.WEB32K_CACHED_PROFILE)
    rows = list(h['C'].read_jsonl(destination / 'native_web_train.jsonl'))
    bundle = rows[-1]
    schema = schemas.web.original_schema(bundle['record'], bundle['collector'], cached_sources=registry)
    before = deepcopy(bundle)
    example = N.validate_native_record(bundle, tokenizer, expected_identity=identity,
        original_schema=schema, web_schema_profile=schemas.web, cached_sources=registry)
    assert example.completion == bundle['native_trace']['completion_ids'] and bundle == before
    for web_profile, sources in [(None, registry), (N.WebSchemaProfile(3), registry), (schemas.web, None)]:
        with pytest.raises(ValueError, match='Cached TRAIN marker requires its explicit trusted Web profile'):
            N.validate_native_record(bundle, tokenizer, expected_identity=identity,
                original_schema=schema, web_schema_profile=web_profile, cached_sources=sources)
    resolved = N.NativeSchemaProfile(schemas.web, schemas.cuad, registry)
    with pytest.raises(ValueError, match='Cached TRAIN marker requires its explicit trusted Web profile'):
        N.load_native_training_data(destination / 'native_web_train.jsonl', tokenizer,
            expected_identity=identity, source_receipt=receipt,
            schema_resolver=lambda record, collector: resolved.original_schema(record, collector))
