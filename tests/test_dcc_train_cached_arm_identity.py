"""Synthetic native records bind each cached TRAIN arm before physical replay."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import runpy
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'slime')]
import dcc_native_sft as N
import dcc_review_training as R
import dcc_native_physical as P
from slime_plugins.dcc.prompt_history import PromptHistoryConfig

H = runpy.run_path(str(ROOT / 'tests/test_dcc_train_cached_rows.py'))
H8 = runpy.run_path(str(ROOT / 'tests/test_dcc_train_cached_static8k.py'))


@pytest.fixture(scope='module', params=[8192, 16384])
def selected_arm(request, tmp_path_factory):
    root = tmp_path_factory.mktemp('synthetic-cached-arm-' + str(request.param))
    patch = pytest.MonkeyPatch()
    try:
        yield (H8['selected8'] if request.param == 8192 else H['select'])(root, patch)
    finally:
        patch.undo()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write(path, values):
    path.write_text(''.join(json.dumps(row) + '\n' for row in values))


def direct(selected_arm):
    _, destination, tokenizer, identity, approved, schemas, receipt, _ = selected_arm
    registry = H['W'].load_sources(destination / 'native_web_cached_sources.jsonl',
                                  source_receipt=receipt, profile_version=approved['version'])
    bundle = rows(destination / 'native_web_train.jsonl')[-1]
    schema = schemas.web.original_schema(bundle['record'], bundle['collector'], cached_sources=registry)
    return bundle, tokenizer, identity, schemas.web, registry, schema


def test_correct_direct_arm_preserves_all_original_ids(selected_arm):
    bundle, tokenizer, identity, web, registry, schema = direct(selected_arm)
    before = deepcopy(bundle)
    ex = N.validate_native_record(bundle, tokenizer, expected_identity=identity,
        original_schema=schema, web_schema_profile=web, cached_sources=registry)
    assert bundle == before and ex.prompt == bundle['native_trace']['original_prompt_ids']
    assert ex.completion == bundle['native_trace']['completion_ids']


@pytest.mark.parametrize('damage', ['other-arm', 'thought-type', 'greedy', 'prompt', 'prompt-type',
                                    'context', 'context-type', 'disabled-history', 'missing-history'])
def test_direct_guard_rejects_invalid_approved_identity_before_physical_gate(selected_arm, monkeypatch, damage):
    bundle, tokenizer, identity, web, registry, schema = direct(selected_arm)
    changes = {'other-arm': dict(thought_tokens=16384 if identity.thought_tokens == 8192 else 8192),
        'thought-type': dict(thought_tokens=str(identity.thought_tokens)),
        'greedy': dict(thought_sampling_profile='greedy'), 'prompt': dict(max_prompt_tokens=16384),
        'prompt-type': dict(max_prompt_tokens='32768'),
        'context': dict(max_model_len=32768 + identity.thought_tokens + 4095),
        'context-type': dict(max_model_len='53248'),
        'disabled-history': dict(prompt_history=PromptHistoryConfig(False).to_dict()),
        'missing-history': dict(prompt_history=None)}
    invalid = replace(identity, **changes[damage])
    monkeypatch.setattr(P, 'validate_native_physical', lambda *a, **k: (_ for _ in ()).throw(
        AssertionError('Invalid arm reached physical token gate')))
    with pytest.raises(ValueError):
        N.validate_native_record(bundle, tokenizer, expected_identity=invalid,
            original_schema=schema, web_schema_profile=web, cached_sources=registry)


@pytest.mark.parametrize('mode', ['empty-active-sidecar', 'full-relabelled-sidecar'])
def test_direct_loader_cannot_relabel_trusted_arm_even_when_sidecar_matches(selected_arm, tmp_path, mode):
    _, destination, tokenizer, identity, approved, schemas, receipt, _ = selected_arm
    other = R.WEB32K_CACHED_PROFILE if identity.thought_tokens == 8192 else R.WEB32K_CACHED_8K_PROFILE
    trusted = N.NativeSchemaProfile(replace(schemas.web, profile_version=other))
    bundles = rows(destination / 'native_web_train.jsonl')
    sources = rows(destination / 'native_web_cached_sources.jsonl')
    if mode == 'empty-active-sidecar':
        bundles, sources = bundles[:2], []
    else:
        for row in sources:
            row['profile_version'] = other
    web = tmp_path / 'native_web_train.jsonl'
    write(web, bundles); write(web.with_name('native_web_cached_sources.jsonl'), sources)
    with pytest.raises(ValueError, match='Cached TRAIN trusted profile arm differs'):
        N.load_native_training_data(web, tokenizer, expected_identity=identity,
            source_receipt=receipt, schema_resolver=trusted.original_schema)
