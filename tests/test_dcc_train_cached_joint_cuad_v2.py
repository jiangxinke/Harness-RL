"""Synthetic CPU controls for explicit cached16k Web/CUAD v2 joint SFT only."""
from collections import Counter
from copy import deepcopy
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
import dcc_cuad_retrieval as Q
from slime_plugins.dcc.data import Task
from slime_plugins.dcc.native_thinking_eval import NativeThinkingPolicy

J = runpy.run_path(str(ROOT / 'tests/test_dcc_train_cached_joint_profile.py'))
H, F = J['H'], J['Q']['F']


def joint(web, cuad):
    result = J['joint'](web, cuad)
    result['version'] = R.JOINT_CACHED_CUAD_TASKBOUND_PROFILE
    return result


def pure_profile():
    result = J['pure_profile']()
    result['version'] = R.JOINT_CACHED_CUAD_TASKBOUND_PROFILE
    result['cuad']['retrieval'] = J['Q']['binding'](taskbound_contiguous=True)
    return result


@pytest.fixture(scope='module')
def selected(tmp_path_factory):
    root = tmp_path_factory.mktemp('synthetic-joint-cached-cuad-v2')
    monkeypatch = pytest.MonkeyPatch()
    try:
        webroot = root / 'web'; webroot.mkdir()
        _, webdest, tokenizer, webidentity, webprofile, _, webreceipt, _ = H['select'](webroot, monkeypatch)
        cls = F['FIXTURE']['FakeTokenizer']
        monkeypatch.setattr(cls, '__len__', lambda self: 160000, raising=False)
        monkeypatch.setattr(cls, 'chat_template', 'synthetic-template', raising=False)
        document = ' '.join('word%04d' % i for i in range(380))
        # No single original1600-character hit contains this complete answer.
        span = ' '.join(document[1300:1850].split())
        task = Task('synthetic-joint-cuad-v2', 'cuad', 'train', J['Q']['QUESTION'], document=document)
        raw = root / 'cuad-raw'
        values = [{'tool': 'search_document', 'arguments': {'query': 'contract clause', 'top_k': 4}},
                  {'answer': [{'text': span, 'confidence': .9}]}]
        _, engine, old = F['make_native'](raw, values=values)
        old.base.seed = 47
        cuadreceipt = dict(old.source_receipt, source_commit='e' * 40)
        native = NativeThinkingPolicy(old.base, source_receipt=cuadreceipt, budget=old.budget,
                                      trace_sink=old.trace_sink)
        cuadprofile = J['Q']['new_profile'](tokenizer)
        cuadprofile.update(version=R.CUAD_TASKBOUND_UNION_PROFILE)
        cuadprofile['cuad'].update(max_steps=4, retrieval=J['Q']['binding'](taskbound_contiguous=True))
        cuadprofile['identity'].update(source_commit='e' * 40, derivative_id=webidentity.derivative_id,
                                       generation_seed=47)
        cuadidentity = N.identity_from_manifest(cuadprofile['identity'], 'e' * 40)
        schemas = G.source_schema_profile(cuadprofile)
        asyncio.run(C.generate([task], native, raw, collection_id='synthetic-separate-cuad-v2',
                               profile=cuadprofile, schema_profile=schemas))
        assert len(engine.calls) == 4
        labels = root / 'synthetic-labels.jsonl'
        labels.write_text(json.dumps({'task_id': task.id, 'benchmark': 'cuad', 'split': 'train',
            'gold': [span], 'category': 'Synthetic confidentiality'}) + '\n')
        cuaddest = root / 'cuad-selected'
        C.select(raw, cuaddest, labels_path=labels, profile=cuadprofile, identity=cuadidentity,
                 schema_profile=schemas, receipt=cuadreceipt, tokenizer=tokenizer)
        assert len(list(C.read_jsonl(cuaddest / 'native_cuad_train.jsonl'))) == 2
        profile = joint(webprofile, cuadprofile)
        source = root / 'source'; source.mkdir()
        (source / '.dcc-native-source.json').write_text(json.dumps(webreceipt))
        (source / '.dcc-training-code-overlay.json').write_text(json.dumps(J['overlay'](profile)))
        profilepath = root / 'joint-profile.json'; profilepath.write_text(json.dumps(profile))
        args = SimpleNamespace(adapter=None, cuad_train_data=None, native_expected_source=webidentity.source_commit,
            native_train_profile=str(profilepath), records=str(webdest / 'native_web_train.jsonl'),
            cuad_tool_records=str(cuaddest / 'native_cuad_train.jsonl'), model='synthetic-no-weights')
        yield SimpleNamespace(root=root, source=source, profile=profile, args=args, tokenizer=tokenizer,
            webidentity=webidentity, cuadidentity=cuadidentity, webprofile=webprofile, cuadprofile=cuadprofile,
            webreceipt=webreceipt, cuadreceipt=cuadreceipt, webdest=webdest, cuaddest=cuaddest, schemas=schemas)
    finally:
        monkeypatch.undo()


def test_explicit_joint_v2_keeps_cached16k_web_and_exact_v2_cuad_binding():
    profile = pure_profile(); before = deepcopy(profile)
    web = R.approved_web_profile(profile, N.WebSchemaProfile)
    cuad = R.approved_cuad_profile(profile, N.CuadSchemaProfile)
    assert web.profile_version == R.WEB32K_CACHED_PROFILE
    assert cuad.retrieval == profile['cuad']['retrieval'] and cuad.contiguous_sources
    assert profile == before and profile['version'] not in R.CACHED_TRAIN_PROFILES
    assert R.JOINT_CACHED_CUAD_PROFILE != R.JOINT_CACHED_CUAD_TASKBOUND_PROFILE


@pytest.mark.parametrize('damage', ['old-joint-wrapper', 'old-binding', 'single-v2-wrapper', '8k-route',
    'review20-route', '8k-web-thought', 'cuad-thought', 'cuad-model', 'cuad-tokenizer',
    'no-cuad-identity', 'unlisted-source', 'extra-identity-difference', 'missing-retrieval', 'extra-key'])
def test_joint_v2_cannot_implicitly_upgrade_old_routes_or_expand_identity_allowlist(damage):
    profile = pure_profile()
    if damage == 'old-joint-wrapper': profile['version'] = R.JOINT_CACHED_CUAD_PROFILE
    elif damage == 'old-binding': profile['cuad']['retrieval']['version'] = Q.VERSION
    elif damage == 'single-v2-wrapper': profile['version'] = R.CUAD_TASKBOUND_UNION_PROFILE
    elif damage == '8k-route': profile['web_profile_version'] = R.WEB32K_CACHED_8K_PROFILE
    elif damage == 'review20-route': profile['web_profile_version'] = R.REVIEW20_PROFILE
    elif damage == '8k-web-thought': profile['identity']['thought_tokens'] = 8192
    elif damage in {'cuad-thought', 'cuad-model', 'cuad-tokenizer'}:
        key = {'cuad-thought': 'thought_tokens', 'cuad-model': 'derivative_id', 'cuad-tokenizer': 'tokenizer_revision'}[damage]
        profile['cuad_identity'][key] = 8192 if key == 'thought_tokens' else 'foreign'
        profile['cuad_identity_differences'] = sorted(set(profile['cuad_identity_differences']) | {key})
    elif damage == 'no-cuad-identity': profile.pop('cuad_identity')
    elif damage == 'unlisted-source': profile['cuad_identity']['source_commit'] = 'e' * 40
    elif damage == 'extra-identity-difference': profile['cuad_identity_differences'].append('not-real')
    elif damage == 'missing-retrieval': profile['cuad'].pop('retrieval')
    else: profile['unknown'] = True
    with pytest.raises(ValueError): R.approved_web_profile(profile, N.WebSchemaProfile)


@pytest.mark.parametrize('damage', ['old-joint-overlay', 'single-v2-overlay', 'missing-cache', 'stale-cache'])
def test_joint_v2_requires_matching_explicit_training_overlay(tmp_path, damage):
    profile = pure_profile(); overlay = J['overlay'](profile)
    if damage == 'old-joint-overlay': overlay['explicit_training_profile'] = R.JOINT_CACHED_CUAD_PROFILE
    elif damage == 'single-v2-overlay': overlay['explicit_training_profile'] = R.CUAD_TASKBOUND_UNION_PROFILE
    elif damage == 'missing-cache': overlay.pop('web_cached_rows')
    else: overlay['web_cached_rows']['version'] = 'old'
    (tmp_path / '.dcc-training-code-overlay.json').write_text(json.dumps(overlay))
    with pytest.raises(ValueError): R.bind_profile_source(profile, tmp_path)


def test_full_loader_validates_separate_sources_seeds_and_contiguous_native_grounding(selected):
    paths = [selected.webdest / 'native_web_train.jsonl', selected.webdest / 'native_web_cached_sources.jsonl',
             selected.cuaddest / 'native_cuad_train.jsonl']
    before = {str(path): path.read_bytes() for path in paths}
    data = N.prepare_native_training(selected.args, source_root=selected.source, tokenizer=selected.tokenizer)
    assert (len(data.records), len(data.cuad_records)) == (4, 2)
    assert data.identity == selected.webidentity and data.cuad_identity == selected.cuadidentity
    assert data.identity.source_commit == 'f' * 40 and data.cuad_identity.source_commit == 'e' * 40
    assert data.identity.generation_seed == 42 and data.cuad_identity.generation_seed == 47
    assert data.strict_training_budgets and data.configuration['all_files_validated_before_model_load']
    assert data.configuration['training_profile_version'] == R.JOINT_CACHED_CUAD_TASKBOUND_PROFILE
    assert data.configuration['domain_identities'] == dict(independent_curriculum=selected.webprofile['identity'],
                                                          cuad_train=selected.cuadprofile['identity'])
    assert selected.profile['cuad_identity_differences'] == [
        'generation_seed', 'max_model_len', 'max_prompt_tokens', 'prompt_history', 'source_commit']
    for path, receipt in ((paths[0], selected.webreceipt), (paths[2], selected.cuadreceipt)):
        for bundle in C.read_jsonl(path):
            example = data.examples[N._segment_key(bundle['record'])]
            assert example.prompt == bundle['native_trace']['original_prompt_ids']
            assert example.completion == bundle['native_trace']['completion_ids']
            assert bundle['native_trace']['native_thinking']['source_receipt'] == receipt
    final = list(C.read_jsonl(paths[2]))[-1]
    with pytest.raises(ValueError, match='absent'): N.legacy.validate_cuad_record(final['record'])
    # The v2 trusted profile is still mandatory even with a matching row marker.
    schema = selected.schemas.original_schema(final['record'], final['collector'])
    with pytest.raises(ValueError, match='explicit trusted profile'):
        N.validate_native_record(final, selected.tokenizer, expected_identity=selected.cuadidentity,
                                 original_schema=schema)
    skips = Counter()
    for record in data.records + data.cuad_records:
        data.encode_record(record, selected.tokenizer, max_prompt=32768, max_response=20480,
                           batch_tokens=212992, skips=skips)
    assert skips == Counter() and before == {str(path): path.read_bytes() for path in paths}


@pytest.mark.parametrize('damage', ['web-source', 'cuad-source', 'cuad-receipt', 'web-late-token', 'cuad-late-token',
    'cuad-prompt-id', 'cuad-binding-v1', 'cuad-binding-missing', 'cuad-offset', 'cuad-source-header',
    'sidecar-arm', 'empty-web', 'empty-cuad'])
def test_joint_v2_full_loader_rejects_source_token_offset_and_late_file_damage(selected, tmp_path, damage):
    web = tmp_path / 'native_web_train.jsonl'; cuad = tmp_path / 'native_cuad_train.jsonl'
    sidecar = tmp_path / 'native_web_cached_sources.jsonl'
    web.write_bytes((selected.webdest / web.name).read_bytes())
    cuad.write_bytes((selected.cuaddest / cuad.name).read_bytes())
    sidecar.write_bytes((selected.webdest / sidecar.name).read_bytes())
    if damage == 'empty-web': web.write_text(''); sidecar.write_text('')
    elif damage == 'empty-cuad': cuad.write_text('')
    elif damage == 'sidecar-arm':
        row = json.loads(sidecar.read_text()); row['profile_version'] = R.WEB32K_CACHED_8K_PROFILE
        sidecar.write_text(json.dumps(row) + '\n')
    else:
        target = web if damage.startswith('web-') else cuad
        bundles = list(C.read_jsonl(target)); last = bundles[-1]
        if damage in {'web-source', 'cuad-source'}: last['native_trace']['source_commit'] = 'a' * 40
        elif damage == 'cuad-receipt': last['native_trace']['native_thinking']['source_receipt'] = selected.webreceipt
        elif damage.endswith('late-token'): last['native_trace']['completion_ids'][-1] = 1
        elif damage == 'cuad-prompt-id': last['native_trace']['original_prompt_ids'][0] = 1
        elif damage == 'cuad-binding-v1': last['collector']['cuad_retrieval']['version'] = Q.VERSION
        elif damage == 'cuad-binding-missing': last['collector'].pop('cuad_retrieval')
        elif damage == 'cuad-offset':
            message = next(m for m in last['record']['messages'] if m['content'].startswith('Initial contract retrieval'))
            prefix, payload = message['content'].split('\n', 1); payload = json.loads(payload)
            payload['hits'][0]['start'] += 1
            message['content'] = prefix + '\n' + json.dumps(payload)
        else:
            message = next(m for m in last['record']['messages'] if m['content'].startswith('Initial contract retrieval'))
            message['content'] = message['content'].split('\n', 1)[1]
        target.write_text(''.join(json.dumps(row) + '\n' for row in bundles))
    args = SimpleNamespace(**vars(selected.args)); args.records = str(web); args.cuad_tool_records = str(cuad)
    with pytest.raises((ValueError, FileNotFoundError)):
        N.prepare_native_training(args, source_root=selected.source, tokenizer=selected.tokenizer)


def test_joint_v2_rejected_by_collection_selector_preferences_and_dpo_before_inputs(selected, tmp_path, monkeypatch):
    monkeypatch.setattr(C, 'load_train_labels', lambda *a, **k: (_ for _ in ()).throw(AssertionError('labels opened')))
    with pytest.raises(ValueError, match='training-only'):
        C.load_approved(selected.args.native_train_profile, selected.source,
                        selected.webidentity.source_commit, selected.tokenizer)
    with pytest.raises(ValueError, match='training-only'):
        asyncio.run(C.generate([], None, tmp_path / 'never-created', collection_id='synthetic',
                               profile=selected.profile, schema_profile=None))
    with pytest.raises(ValueError, match='training-only'):
        C.read_closed(tmp_path / 'missing-journals', profile=selected.profile, receipt=selected.webreceipt, world_size=1)
    with pytest.raises(ValueError, match='training-only'):
        C.select(tmp_path / 'missing-journals', tmp_path / 'never-selected', labels_path=tmp_path / 'never-labels',
            profile=selected.profile, identity=selected.webidentity, schema_profile=None,
            receipt=selected.webreceipt, tokenizer=selected.tokenizer)
    with pytest.raises(ValueError, match='does not support cached-row'):
        G.source_schema_profile(selected.profile)
    assert not (tmp_path / 'never-created').exists() and not (tmp_path / 'never-selected').exists()
