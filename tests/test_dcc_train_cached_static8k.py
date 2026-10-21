"""Separate static 8k/16k native TRAIN identities; synthetic CPU tests only."""
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
import dcc_native_sft as N
import dcc_review_training as R
from slime_plugins.dcc.data import Task
from slime_plugins.dcc.native_budget import NativeBudget
from slime_plugins.dcc.native_thinking_eval import NativeThinkingPolicy
from slime_plugins.dcc.prompt_history import PromptHistoryConfig

H = runpy.run_path(str(ROOT / 'tests/test_dcc_train_cached_rows.py'))
F = H['FIXTURE']


def profile8(identity=None):
    result = H['profile'](identity)
    result['version'] = R.WEB32K_CACHED_8K_PROFILE
    if identity is None:
        result['identity'].update(thought_tokens=8192, max_model_len=45056)
    return result


@pytest.mark.parametrize('version,thought,context', [
    (R.WEB32K_CACHED_PROFILE, 16384, 53248),
    (R.WEB32K_CACHED_8K_PROFILE, 8192, 45056)])
def test_each_static_profile_requires_its_exact_complete_native_budget(version, thought, context):
    p = H['profile'](); p['version'] = version
    p['identity'].update(thought_tokens=thought, max_model_len=context)
    before = deepcopy(p)
    assert R.approved_web_profile(p, N.WebSchemaProfile).profile_version == version
    assert p == before
    for other in [4096, 8192, 16384, True, '8192', None]:
        if type(other) is int and other == thought:
            continue
        bad = deepcopy(p); bad['identity']['thought_tokens'] = other
        with pytest.raises(ValueError): R.approved_web_profile(bad, N.WebSchemaProfile)
    for other in [context - 1, True, str(context), None]:
        bad = deepcopy(p); bad['identity']['max_model_len'] = other
        with pytest.raises(ValueError): R.approved_web_profile(bad, N.WebSchemaProfile)
    bad = deepcopy(p); bad['identity']['thought_sampling_profile'] = 'greedy'
    with pytest.raises(ValueError): R.approved_web_profile(bad, N.WebSchemaProfile)


def test_historical_web32k_profile_cannot_silently_shrink_to_8k():
    p = H['profile'](); p['version'] = R.WEB32K_PROFILE
    p['web'].pop('cached_rows_configuration'); p['web']['cached_rows'] = False
    p['identity'].update(thought_tokens=8192, max_model_len=45056)
    with pytest.raises(ValueError): R.approved_web_profile(p, N.WebSchemaProfile)


@pytest.mark.parametrize('damage', ['version', 'config', 'old-overlay', 'disabled-history'])
def test_8k_requires_independent_overlay_and_complete_config(tmp_path, damage):
    p = profile8()
    overlay = {'overlay_id': p['collection_overlay_id'], 'explicit_training_profile': p['version'],
               'baseline_source_unchanged': False, 'web_cached_rows': deepcopy(p['web']['cached_rows_configuration'])}
    if damage == 'version': overlay['explicit_training_profile'] = R.WEB32K_CACHED_PROFILE
    elif damage == 'config': overlay['web_cached_rows']['version'] = 'cached-literal-record-windows-v2'
    elif damage == 'old-overlay': overlay['baseline_source_unchanged'] = True
    else:
        p['identity']['prompt_history'] = PromptHistoryConfig(False).to_dict()
        with pytest.raises(ValueError): R.approved_web_profile(p, N.WebSchemaProfile)
        return
    (tmp_path / '.dcc-training-code-overlay.json').write_text(json.dumps(overlay))
    with pytest.raises(ValueError): R.bind_profile_source(p, tmp_path)


def generate8(tmp_path, monkeypatch, *, actual_budget=8192, approved_budget=8192):
    cls = F['FakeTokenizer']
    monkeypatch.setattr(cls, '__len__', lambda self: 160000, raising=False)
    monkeypatch.setattr(cls, 'chat_template', 'synthetic-template', raising=False)
    output = tmp_path / 'raw'; output.mkdir()
    budget = NativeBudget(actual_budget, 'thinking')
    tokenizer, engine, old = F['make_policy'](output, values=H['values'](), budget=budget)
    base = old.base; base.max_prompt_tokens = 32768; base.max_model_len = 53248
    base.seed = 42; base.allow_truncation = False
    base.provenance['web_cached_rows'] = H['W'].configuration()
    native = NativeThinkingPolicy(base, source_receipt=old.source_receipt, budget=budget,
        trace_sink=old.trace_sink, prompt_history_config=PromptHistoryConfig(True), phase_timing=True)
    identity = N.NativeIdentity('synthetic-static-budget-only', (151643, 151645), 'f' * 40,
        32768, 53248, 42, approved_budget, 'thinking', PromptHistoryConfig(True).to_dict())
    approved = profile8(identity.manifest(tokenizer))
    schemas = N.NativeSchemaProfile(R.approved_web_profile(approved, N.WebSchemaProfile))
    task = Task('synthetic-static8k-public', 'widesearch', 'train', 'List the public names and years.',
                metadata={'evaluation': {'required': ['Name', 'Year']}})
    try:
        asyncio.run(C.generate([task], native, output, collection_id='synthetic-static8k',
            profile=approved, schema_profile=schemas, timeout_seconds=1800, web_tools_factory=H['Tools']))
    except BaseException:
        asyncio.run(native.close()); raise
    return output, tokenizer, identity, approved, schemas, old.source_receipt, engine


def selected8(tmp_path, monkeypatch):
    output, tokenizer, identity, approved, schemas, receipt, engine = generate8(tmp_path, monkeypatch)
    from dcc_train_web_literal_reward import configuration, LABEL_PROTOCOL
    labels = tmp_path / 'synthetic-labels.jsonl'
    labels.write_text(json.dumps({'task_id': 'synthetic-static8k-public', 'benchmark': 'widesearch', 'split': 'train',
        'gold': [{'Name': 'Alpha', 'Year': '2001'}, {'Name': 'Beta', 'Year': '2002'}],
        'evaluation': {'required': ['Name', 'Year']}, 'label_protocol': LABEL_PROTOCOL,
        'train_reward': configuration()}) + '\n')
    destination = tmp_path / 'selected'
    summary = C.select(output, destination, labels_path=labels, profile=approved, identity=identity,
        schema_profile=schemas, receipt=receipt, tokenizer=tokenizer, web_train_reward_protocol=C.LITERAL_WEB_TRAIN_REWARD)
    assert (summary['accepted_episodes'], summary['rejected_episodes'], summary['web_segments']) == (1, 0, 4)
    return output, destination, tokenizer, identity, approved, schemas, receipt, engine


def test_static8k_real_native_collector_selector_preparation_loader_preserves_tokens_and_closing(tmp_path, monkeypatch):
    output, destination, tokenizer, identity, approved, schemas, receipt, engine = selected8(tmp_path, monkeypatch)
    bundles = list(C.read_jsonl(destination / 'native_web_train.jsonl'))
    assert len(bundles) == 4 and len(engine.calls) == 8
    for index, bundle in enumerate(bundles):
        trace = bundle['native_trace']
        assert bundle['identity']['thought_tokens'] == trace['thought_max_tokens'] == 8192
        assert trace['declared_max_new_tokens'] == 12288 and trace['json_max_tokens'] == 4096
        assert trace['native_thinking']['thought_max_tokens'] == 8192
        assert trace['phases'][0]['max_tokens'] == 8192
        assert trace['phases'][1]['max_tokens'] == 4096
        assert trace['completion_ids'][-1] == 151645 and 151668 in trace['completion_ids']
        assert len(trace['phases']) == 2 and all('timing' in p for p in trace['phases'])
    sidecar = list(C.read_jsonl(destination / 'native_web_cached_sources.jsonl'))
    assert sidecar[0]['profile_version'] == R.WEB32K_CACHED_8K_PROFILE
    loaded = N.load_native_training_data(destination / 'native_web_train.jsonl', tokenizer,
        expected_identity=identity, source_receipt=receipt, schema_resolver=schemas.original_schema)
    for record, bundle in zip(loaded.records, bundles, strict=True):
        example = loaded.examples[N._segment_key(record)]
        assert example.completion == bundle['native_trace']['completion_ids']
        assert example.prompt == bundle['native_trace']['original_prompt_ids']
    source = tmp_path / 'source'; source.mkdir()
    (source / '.dcc-native-source.json').write_text(json.dumps(receipt))
    (source / '.dcc-training-code-overlay.json').write_text(json.dumps({
        'overlay_id': approved['collection_overlay_id'], 'explicit_training_profile': approved['version'],
        'baseline_source_unchanged': False, 'web_cached_rows': approved['web']['cached_rows_configuration']}))
    approved_path = tmp_path / 'approved.json'; approved_path.write_text(json.dumps(approved))
    args = SimpleNamespace(adapter=None, cuad_train_data=None, native_expected_source=identity.source_commit,
        native_train_profile=str(approved_path), cuad_tool_records=None, records=str(destination / 'native_web_train.jsonl'),
        model='synthetic-no-weights')
    prepared = N.prepare_native_training(args, source_root=source, tokenizer=tokenizer)
    assert prepared.strict_training_budgets
    assert prepared.configuration['training_profile_version'] == R.WEB32K_CACHED_8K_PROFILE
    assert prepared.configuration['rollout_native_thinking']['thought_max_tokens'] == 8192
    closed = json.loads((output / 'generation_closed_rank0.json').read_text())
    assert closed['profile']['version'] == R.WEB32K_CACHED_8K_PROFILE and 'native_phase_timing' in closed


def test_static8k_runtime_rejects_16k_actual_budget_before_any_request(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='Actual native phase budgets'):
        generate8(tmp_path, monkeypatch, actual_budget=16384)
    assert not (tmp_path / 'raw/generation_closed_rank0.json').exists()
    assert not (tmp_path / 'raw/requests_rank0.jsonl').exists()


@pytest.mark.parametrize('damage', ['identity-arm', 'sidecar-arm', 'closing', 'budget', 'phase-budget'])
def test_static8k_loader_rejects_arm_substitution_budgets_and_real_close_tampering(tmp_path, monkeypatch, damage):
    _, destination, tokenizer, identity, approved, schemas, receipt, _ = selected8(tmp_path, monkeypatch)
    web = destination / 'native_web_train.jsonl'; bundles = list(C.read_jsonl(web))
    if damage == 'identity-arm':
        identity = replace(identity, thought_tokens=16384)
    elif damage == 'sidecar-arm':
        source = destination / 'native_web_cached_sources.jsonl'; rows = list(C.read_jsonl(source))
        rows[0]['profile_version'] = R.WEB32K_CACHED_PROFILE
        source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    elif damage == 'closing':
        trace = bundles[-1]['native_trace']; trace['completion_ids'].remove(151668)
    elif damage == 'budget': bundles[-1]['native_trace']['thought_max_tokens'] = 16384
    else: bundles[-1]['native_trace']['phases'][0]['max_tokens'] = 16384
    web.write_text(''.join(json.dumps(bundle) + '\n' for bundle in bundles))
    with pytest.raises(ValueError):
        N.load_native_training_data(web, tokenizer, expected_identity=identity, source_receipt=receipt,
                                   schema_resolver=schemas.original_schema)
