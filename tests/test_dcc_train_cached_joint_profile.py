"""Explicit joint TRAIN loader profile using synthetic independent collections."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import asyncio
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'slime')]
import dcc_collect_native_train as C
import dcc_collect_native_preferences as G
import dcc_native_sft as N
import dcc_review_training as R
from slime_plugins.dcc.native_thinking_eval import NativeThinkingPolicy

H = runpy.run_path(str(ROOT / 'tests/test_dcc_train_cached_rows.py'))
Q = runpy.run_path(str(ROOT / 'tests/test_dcc_cuad_retrieval_binding.py'))


def joint(web, cuad):
    out = deepcopy(web)
    out.update(version=R.JOINT_CACHED_CUAD_PROFILE, web_profile_version=R.WEB32K_CACHED_PROFILE,
        collection_overlay_id='synthetic-joint-training-only', cuad=deepcopy(cuad['cuad']),
        cuad_identity=deepcopy(cuad['identity']))
    out['cuad_identity_differences'] = sorted(k for k in set(out['identity']) | set(out['cuad_identity'])
        if out['identity'].get(k) != out['cuad_identity'].get(k))
    return out


def pure_profile():
    web = H['profile']()
    cuad = {'identity': deepcopy(web['identity']), 'cuad': {'max_steps': 3, 'structured_actions': True,
        'schema_version': Q['CUAD_STRUCTURED_SCHEMA_VERSION'], 'retrieval': Q['binding']()}}
    cuad['identity'].update(max_prompt_tokens=16384, max_model_len=36864)
    cuad['identity'].pop('prompt_history')
    return joint(web, cuad)


def overlay(profile):
    return dict(overlay_id=profile['collection_overlay_id'], explicit_training_profile=profile['version'],
                baseline_source_unchanged=False, web_cached_rows=H['W'].configuration())


@pytest.fixture(scope='module')
def selected(tmp_path_factory):
    root = tmp_path_factory.mktemp('synthetic-joint-cached-cuad')
    patch = pytest.MonkeyPatch()
    try:
        webroot = root / 'web'; webroot.mkdir()
        _, webdest, tokenizer, webidentity, webprofile, _, webreceipt, _ = H['select'](webroot, patch)
        cls = Q['F']['FIXTURE']['FakeTokenizer']
        patch.setattr(cls, '__len__', lambda self: 160000, raising=False)
        patch.setattr(cls, 'chat_template', 'synthetic-template', raising=False)
        raw = root / 'cuad-raw'
        values = [{'tool': 'search_document', 'arguments': {'query': Q['QUESTION'], 'top_k': 8}},
            {'tool': 'read_document', 'arguments': {'start': 0, 'end': len(Q['DOCUMENT'])}},
            {'plan': 'Use the public source.', 'checks': 'Check source text.',
             'answer': [{'text': Q['SPAN'], 'confidence': 0.9}]}]
        _, _, old = Q['F']['make_native'](raw, values=values)
        cuadreceipt = dict(old.source_receipt, source_commit='e' * 40)
        native = NativeThinkingPolicy(old.base, source_receipt=cuadreceipt, budget=old.budget,
                                      trace_sink=old.trace_sink)
        cuadprofile = Q['new_profile'](tokenizer)
        cuadprofile['identity'].update(source_commit='e' * 40, derivative_id=webidentity.derivative_id)
        cuadidentity = N.identity_from_manifest(cuadprofile['identity'], 'e' * 40)
        schemas = N.NativeSchemaProfile(R.approved_web_profile(cuadprofile, N.WebSchemaProfile),
                                       R.approved_cuad_profile(cuadprofile, N.CuadSchemaProfile))
        asyncio.run(C.generate([Q['task']()], native, raw, collection_id='synthetic-original-cuad',
                               profile=cuadprofile, schema_profile=schemas))
        labels = root / 'synthetic-labels.jsonl'
        labels.write_text(json.dumps({'task_id': Q['task']().id, 'benchmark': 'cuad', 'split': 'train',
            'gold': [Q['SPAN']], 'category': 'Synthetic confidentiality'}) + '\n')
        cuaddest = root / 'cuad-selected'
        C.select(raw, cuaddest, labels_path=labels, profile=cuadprofile, identity=cuadidentity,
                 schema_profile=schemas, receipt=cuadreceipt, tokenizer=tokenizer)
        profile = joint(webprofile, cuadprofile)
        source = root / 'source'; source.mkdir()
        (source / '.dcc-native-source.json').write_text(json.dumps(webreceipt))
        (source / '.dcc-training-code-overlay.json').write_text(json.dumps(overlay(profile)))
        profilepath = root / 'joint-profile.json'; profilepath.write_text(json.dumps(profile))
        args = SimpleNamespace(adapter=None, cuad_train_data=None, native_expected_source=webidentity.source_commit,
            native_train_profile=str(profilepath), records=str(webdest / 'native_web_train.jsonl'),
            cuad_tool_records=str(cuaddest / 'native_cuad_train.jsonl'), model='synthetic-no-weights')
        yield SimpleNamespace(root=root, source=source, profile=profile, args=args, tokenizer=tokenizer,
            webidentity=webidentity, cuadidentity=cuadidentity, webprofile=webprofile, cuadprofile=cuadprofile,
            webreceipt=webreceipt, cuadreceipt=cuadreceipt, webdest=webdest, cuaddest=cuaddest)
    finally:
        patch.undo()


def test_explicit_joint_profile_preserves_underlying_routes_and_complete_retrieval_bytes():
    profile = pure_profile(); before = deepcopy(profile)
    assert R.approved_web_profile(profile, N.WebSchemaProfile).profile_version == R.WEB32K_CACHED_PROFILE
    assert R.approved_cuad_profile(profile, N.CuadSchemaProfile).retrieval == profile['cuad']['retrieval']
    assert profile == before and profile['version'] not in R.CACHED_TRAIN_PROFILES


@pytest.mark.parametrize('damage', ['8k-route', 'review20-route', '8k-identity', 'no-cuad-identity',
    'no-differences', 'missing-retrieval', 'legacy-wrapper', 'model-difference', 'thought-difference', 'extra-key'])
def test_joint_profile_cannot_expand_arms_models_identity_allowlist_or_legacy_routes(damage):
    p = pure_profile()
    if damage == '8k-route': p['web_profile_version'] = R.WEB32K_CACHED_8K_PROFILE
    elif damage == 'review20-route': p['web_profile_version'] = R.REVIEW20_PROFILE
    elif damage == '8k-identity': p['identity']['thought_tokens'] = 8192
    elif damage == 'no-cuad-identity': p.pop('cuad_identity')
    elif damage == 'no-differences': p.pop('cuad_identity_differences')
    elif damage == 'missing-retrieval': p['cuad'].pop('retrieval')
    elif damage == 'legacy-wrapper': p['version'] = R.CUAD_RETRIEVAL_PROFILE
    elif damage in {'model-difference', 'thought-difference'}:
        k = 'derivative_id' if damage == 'model-difference' else 'thought_tokens'
        p['cuad_identity'][k] = 'other' if k == 'derivative_id' else 8192
        p['cuad_identity_differences'] = sorted(set(p['cuad_identity_differences']) | {k})
    else: p['some_extra_field'] = True
    with pytest.raises(ValueError): R.approved_web_profile(p, N.WebSchemaProfile)


@pytest.mark.parametrize('damage', ['old-profile', 'wrong-overlay', 'baseline', 'missing-cache-config', 'stale-cache-config'])
def test_joint_overlay_requires_its_own_version_and_complete_current_cache_config(tmp_path, damage):
    p = pure_profile(); o = overlay(p)
    if damage == 'old-profile': o['explicit_training_profile'] = R.WEB32K_CACHED_PROFILE
    elif damage == 'wrong-overlay': o['overlay_id'] = 'different'
    elif damage == 'baseline': o['baseline_source_unchanged'] = True
    elif damage == 'missing-cache-config': o.pop('web_cached_rows')
    else: o['web_cached_rows']['version'] = 'stale'
    (tmp_path / '.dcc-training-code-overlay.json').write_text(json.dumps(o))
    with pytest.raises(ValueError): R.bind_profile_source(p, tmp_path)


def test_joint_real_separate_native_collections_pass_full_loader_without_modifying_ids_or_sidecar(selected):
    paths = [selected.webdest / 'native_web_train.jsonl', selected.webdest / 'native_web_cached_sources.jsonl',
             selected.cuaddest / 'native_cuad_train.jsonl']
    before = {str(p): p.read_bytes() for p in paths}
    data = N.prepare_native_training(selected.args, source_root=selected.source, tokenizer=selected.tokenizer)
    assert (len(data.records), len(data.cuad_records)) == (4, 3)
    assert data.identity == selected.webidentity and data.cuad_identity == selected.cuadidentity
    assert data.strict_training_budgets and data.configuration['all_files_validated_before_model_load']
    assert data.configuration['training_profile_version'] == R.JOINT_CACHED_CUAD_PROFILE
    assert data.configuration['domain_identities'] == dict(independent_curriculum=selected.webprofile['identity'],
                                                          cuad_train=selected.cuadprofile['identity'])
    assert selected.profile['cuad_identity_differences'] == [
        'max_model_len', 'max_prompt_tokens', 'prompt_history', 'source_commit']
    for path in (paths[0], paths[2]):
        for bundle in C.read_jsonl(path):
            example = data.examples[N._segment_key(bundle['record'])]
            assert example.prompt == bundle['native_trace']['original_prompt_ids']
            assert example.completion == bundle['native_trace']['completion_ids']
    skips = Counter()
    data.encode_record(data.records[-1], selected.tokenizer, max_prompt=32768, max_response=20480,
                       batch_tokens=212992, skips=skips)
    assert skips == Counter() and before == {str(p): p.read_bytes() for p in paths}


@pytest.mark.parametrize('damage', ['sidecar-missing', 'sidecar-arm', 'cuad-retrieval', 'web-late-token', 'cuad-late-token',
                                    'empty-web', 'empty-cuad'])
def test_joint_full_loader_retains_sidecar_retrieval_all_file_and_nonempty_gates(selected, tmp_path, damage):
    web = tmp_path / 'native_web_train.jsonl'; cuad = tmp_path / 'native_cuad_train.jsonl'
    sidecar = tmp_path / 'native_web_cached_sources.jsonl'
    web.write_bytes((selected.webdest / web.name).read_bytes())
    cuad.write_bytes((selected.cuaddest / cuad.name).read_bytes())
    sidecar.write_bytes((selected.webdest / sidecar.name).read_bytes())
    if damage == 'sidecar-missing': sidecar.unlink()
    elif damage == 'sidecar-arm':
        row = json.loads(sidecar.read_text()); row['profile_version'] = R.WEB32K_CACHED_8K_PROFILE
        sidecar.write_text(json.dumps(row) + '\n')
    elif damage in {'cuad-retrieval', 'cuad-late-token', 'web-late-token'}:
        target = web if damage == 'web-late-token' else cuad
        bundles = list(C.read_jsonl(target))
        if damage == 'cuad-retrieval': bundles[-1]['collector']['cuad_retrieval']['retrieval_context_chars'] = 10
        else: bundles[-1]['native_trace']['completion_ids'][-1] = 1
        target.write_text(''.join(json.dumps(row) + '\n' for row in bundles))
    elif damage == 'empty-web': web.write_text(''); sidecar.write_text('')
    else: cuad.write_text('')
    args = SimpleNamespace(**vars(selected.args)); args.records = str(web); args.cuad_tool_records = str(cuad)
    with pytest.raises((ValueError, FileNotFoundError)):
        N.prepare_native_training(args, source_root=selected.source, tokenizer=selected.tokenizer)


def test_joint_profile_rejected_by_collection_before_tasks_runtime_output_or_label_access(selected, tmp_path, monkeypatch):
    monkeypatch.setattr(C, 'load_train_labels', lambda *a, **k: (_ for _ in ()).throw(AssertionError('labels opened')))
    with pytest.raises(ValueError, match='training-only'):
        C.load_approved(selected.args.native_train_profile, selected.source,
                        selected.webidentity.source_commit, selected.tokenizer)
    with pytest.raises(ValueError, match='training-only'):
        asyncio.run(C.generate([], None, tmp_path / 'never-created', collection_id='synthetic',
            profile=selected.profile, schema_profile=None))
    assert not (tmp_path / 'never-created').exists()
    with pytest.raises(ValueError, match='training-only'):
        C.read_closed(tmp_path / 'missing-journals', profile=selected.profile, receipt=selected.webreceipt, world_size=1)
    with pytest.raises(ValueError, match='training-only'):
        C.select(tmp_path / 'missing-journals', tmp_path / 'never-selected', labels_path=tmp_path / 'never-labels',
            profile=selected.profile, identity=selected.webidentity, schema_profile=None,
            receipt=selected.webreceipt, tokenizer=selected.tokenizer)
    assert not (tmp_path / 'never-selected').exists()


def test_joint_profile_remains_rejected_by_prospective_preferences_and_dpo_source_schema():
    with pytest.raises(ValueError, match='does not support cached-row'):
        G.source_schema_profile(pure_profile())
