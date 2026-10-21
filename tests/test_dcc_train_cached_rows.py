"""Synthetic CPU source/collector/selector/physical-native-loader integration."""
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
import dcc_train_cached_rows as W
from slime_plugins.dcc.data import Task
from slime_plugins.dcc.native_budget import NativeBudget
from slime_plugins.dcc.native_thinking_eval import NativeThinkingPolicy
from slime_plugins.dcc.prompt_history import PromptHistoryConfig
from slime_plugins.dcc.tools import WebTools

FIXTURE = runpy.run_path(str(ROOT / 'tests/test_dcc_native_thinking.py'))
URL = 'https://example.org/synthetic.csv'
TEXT = 'Name,Year,,\nAlpha,2001,,\n' + 'Unselected public filler,0,,\n' * 700 + 'Beta,2002,,\n'


class Tools(WebTools):
    async def web_search(self, query, **kwargs):
        return {'results': [{'url': URL, 'title': 'Synthetic public table'}]}

    async def fetch_url(self, url, start=0, select_rows=None):
        if select_rows is None:
            self.pages[W.canonical(url)] = TEXT
        return await super().fetch_url(url, start, select_rows)


def action(tool=None, **arguments):
    result = {'plan': 'Read the original public source.', 'checks': 'Copy exact supported source cells.'}
    result.update({'tool': tool, 'arguments': arguments} if tool else
                  {'answer': [{'Name': name, 'Year': year, 'citations': [URL]}
                              for name, year in [('Alpha', '2001'), ('Beta', '2002')]]})
    return result


def values():
    return [action('web_search', query='synthetic public table'), action('fetch_url', url=URL + '#rows'),
            action('fetch_url', url=URL, select_rows={
                'format': 'csv', 'columns': ['Name', 'Year'],
                'where': [{'column': 'Year', 'op': 'num_ge', 'value': '2001'}],
                'csv_trailing_empty_columns': 'verified_empty_padding'}), action()]


def profile(identity=None):
    return {'version': R.WEB32K_CACHED_PROFILE, 'collection_overlay_id': 'synthetic-v3-overlay',
            'identity': deepcopy(identity) if identity else {
                'max_prompt_tokens': 32768, 'max_model_len': 53248,
                'thought_tokens': 16384, 'thought_sampling_profile': 'thinking',
                'prompt_history': PromptHistoryConfig(True).to_dict()},
            'web': {'max_steps': 20, 'explicit_review': True, 'cached_rows': True,
                    'navigation_ledger': True, 'cached_rows_configuration': W.configuration()}}


def generate(tmp_path, monkeypatch, *, timing=True):
    cls = FIXTURE['FakeTokenizer']
    monkeypatch.setattr(cls, '__len__', lambda self: 160000, raising=False)
    monkeypatch.setattr(cls, 'chat_template', 'synthetic-template', raising=False)
    output = tmp_path / 'raw'; output.mkdir()
    budget = NativeBudget(16384, 'thinking')
    tokenizer, engine, old = FIXTURE['make_policy'](output, values=values(), budget=budget)
    base = old.base
    base.max_prompt_tokens = 32768
    base.max_model_len = 53248
    base.seed = 42
    base.allow_truncation = False
    base.provenance['web_cached_rows'] = W.configuration()
    native = NativeThinkingPolicy(base, source_receipt=old.source_receipt, budget=budget,
        trace_sink=old.trace_sink, prompt_history_config=PromptHistoryConfig(True), phase_timing=timing)
    identity = N.NativeIdentity('synthetic-v3-only', (151643, 151645), 'f' * 40, 32768, 53248, 42,
                               16384, 'thinking', PromptHistoryConfig(True).to_dict())
    approved = profile(identity.manifest(tokenizer))
    schemas = N.NativeSchemaProfile(R.approved_web_profile(approved, N.WebSchemaProfile))
    task = Task('synthetic-v3-public-only', 'widesearch', 'train', 'List the public names and years.',
                metadata={'evaluation': {'required': ['Name', 'Year']}})
    asyncio.run(C.generate([task], native, output, collection_id='synthetic-cache-v3',
        profile=approved, schema_profile=schemas, timeout_seconds=1800, web_tools_factory=Tools))
    return output, tokenizer, identity, approved, schemas, old.source_receipt, engine


def select(tmp_path, monkeypatch):
    output, tokenizer, identity, approved, schemas, receipt, engine = generate(tmp_path, monkeypatch)
    # Synthetic TRAIN labels are created only after the actual journals close.
    from dcc_train_web_literal_reward import configuration as reward_configuration, LABEL_PROTOCOL
    labels = tmp_path / 'synthetic-labels.jsonl'
    labels.write_text(json.dumps({'task_id': 'synthetic-v3-public-only', 'benchmark': 'widesearch',
        'split': 'train', 'gold': [{'Name': 'Alpha', 'Year': '2001'}, {'Name': 'Beta', 'Year': '2002'}],
        'evaluation': {'required': ['Name', 'Year']}, 'train_reward': reward_configuration(),
        'label_protocol': LABEL_PROTOCOL}) + '\n')
    destination = tmp_path / 'selected'
    summary = C.select(output, destination, labels_path=labels, profile=approved,
        identity=identity, schema_profile=schemas, receipt=receipt, tokenizer=tokenizer,
        web_train_reward_protocol=C.LITERAL_WEB_TRAIN_REWARD)
    assert (summary['accepted_episodes'], summary['rejected_episodes'], summary['web_segments']) == (1, 0, 4), (summary, (destination / 'audit.jsonl').read_text())
    return output, destination, tokenizer, identity, approved, schemas, receipt, engine


def test_real_collector_selector_loader_keeps_actual_native_ids_and_single_source_publication(tmp_path, monkeypatch):
    output, destination, tokenizer, identity, approved, schemas, receipt, engine = select(tmp_path, monkeypatch)
    sources = list(C.read_jsonl(output / 'web_cached_sources_rank0.jsonl'))
    assert len(sources) == 1 and sources[0]['source']['text'] == TEXT
    assert (destination / 'native_web_cached_sources.jsonl').read_bytes() == (output / 'web_cached_sources_rank0.jsonl').read_bytes()
    requests = [r for r in C.read_jsonl(output / 'requests_rank0.jsonl') if r['event'] == 'request']
    assert [len(r['web_cached_rows']['source_ids']) for r in requests] == [0, 0, 1, 1]
    assert all('source' not in r['web_cached_rows'] for r in requests)
    closed = json.loads((output / 'generation_closed_rank0.json').read_text())
    assert closed['web_cached_source_records'] == 1 and 'native_phase_timing' in closed
    loaded = N.load_native_training_data(destination / 'native_web_train.jsonl', tokenizer,
        expected_identity=identity, source_receipt=receipt, schema_resolver=schemas.original_schema)
    bundles = list(C.read_jsonl(destination / 'native_web_train.jsonl'))
    assert len(loaded.records) == 4 and len(engine.calls) == 8
    for bundle, record in zip(bundles, loaded.records, strict=True):
        example = loaded.examples[N._segment_key(record)]
        assert example.prompt == bundle['native_trace']['original_prompt_ids']
        assert example.completion == bundle['native_trace']['completion_ids']
    final = bundles[-1]
    assert 'Beta' not in json.loads(final['record']['messages'][5]['content'].split('\n', 1)[1].rsplit('\nRemaining turns:', 1)[0])['text']
    assert 'Beta' in tokenizer.decode(final['native_trace']['original_prompt_ids'], skip_special_tokens=False)
    registry = W.load_sources(destination / 'native_web_cached_sources.jsonl', source_receipt=receipt, profile_version=approved['version'])
    assert schemas.web.original_schema(final['record'], final['collector'], cached_sources=registry) == final['native_trace']['json_schema']
    source_root = tmp_path / 'source'; source_root.mkdir()
    (source_root / '.dcc-native-source.json').write_text(json.dumps(receipt))
    overlay = {'overlay_id': approved['collection_overlay_id'], 'explicit_training_profile': approved['version'],
               'baseline_source_unchanged': False, 'web_cached_rows': W.configuration()}
    (source_root / '.dcc-training-code-overlay.json').write_text(json.dumps(overlay))
    approved_path = tmp_path / 'profile.json'; approved_path.write_text(json.dumps(approved))
    args = SimpleNamespace(adapter=None, cuad_train_data=None, native_expected_source=identity.source_commit,
        native_train_profile=str(approved_path), cuad_tool_records=None,
        records=str(destination / 'native_web_train.jsonl'), model='synthetic-no-weights')
    prepared = N.prepare_native_training(args, source_root=source_root, tokenizer=tokenizer)
    assert prepared.strict_training_budgets and prepared.configuration['web']['cached_rows_configuration'] == W.configuration()


@pytest.mark.parametrize('damage', ['version', 'missing', 'extra', 'disabled', 'boolean', 'steps', 'history', 'source-config'])
def test_profile_and_source_never_accept_partial_stale_or_implicitly_enabled_config(tmp_path, damage):
    approved = profile()
    if damage == 'version': approved['web']['cached_rows_configuration']['version'] = 'cached-literal-record-windows-v2'
    elif damage == 'missing': del approved['web']['cached_rows_configuration']['limits']
    elif damage == 'extra': approved['web']['cached_rows_configuration']['new'] = True
    elif damage == 'disabled': approved['web']['cached_rows'] = False
    elif damage == 'boolean': approved['web']['cached_rows_configuration']['enabled'] = 1
    elif damage == 'steps': approved['web']['max_steps'] = 19
    elif damage == 'history': approved['identity']['prompt_history'] = PromptHistoryConfig(False).to_dict()
    else:
        (tmp_path / '.dcc-training-code-overlay.json').write_text(json.dumps({
            'overlay_id': approved['collection_overlay_id'], 'explicit_training_profile': approved['version'],
            'baseline_source_unchanged': False, 'web_cached_rows': {'enabled': True}}))
        with pytest.raises(ValueError): R.bind_profile_source(approved, tmp_path)
        return
    with pytest.raises(ValueError): R.approved_web_profile(approved, N.WebSchemaProfile)


@pytest.mark.parametrize('version', [R.LEGACY_PROFILE, R.REVIEW20_PROFILE, R.WEB32K_PROFILE])
def test_old_profiles_cannot_be_widened_to_new_cached_rows_config(version):
    approved = profile(); approved['version'] = version
    with pytest.raises(ValueError): R.approved_web_profile(approved, N.WebSchemaProfile)


@pytest.mark.parametrize('damage', ['missing-sidecar', 'content', 'offset', 'read-progress', 'source-owner', 'task-owner',
    'source-commit', 'source-version', 'duplicate', 'orphan', 'early', 'unknown-ref', 'duplicate-ref', 'cleared-ref',
    'future-ref', 'url', 'source-index'])
def test_native_loader_rejects_tampered_source_receipts_provenance_and_real_observations(tmp_path, monkeypatch, damage):
    _, destination, tokenizer, identity, approved, schemas, receipt, _ = select(tmp_path, monkeypatch)
    source_path = destination / 'native_web_cached_sources.jsonl'
    rows = list(C.read_jsonl(source_path)); bundles = list(C.read_jsonl(destination / 'native_web_train.jsonl'))
    if damage == 'missing-sidecar': source_path.unlink()
    elif damage == 'content': rows[0]['source']['text'] = rows[0]['source']['text'].replace('Beta', 'Fake')
    elif damage in {'offset', 'read-progress'}:
        observation_message = bundles[-1]['record']['messages'][-2]
        prefix = 'Tool observation (source material):\n'; text = observation_message['content'][len(prefix):]
        value = json.loads(text.rsplit('\nRemaining turns:', 1)[0])
        if damage == 'offset': value['row_view']['records'][1]['start'] += 1
        else: value['row_view']['records'][1]['read_progress']['new_text'] = False
        observation_message['content'] = prefix + json.dumps(value, ensure_ascii=False) + '\nRemaining turns: 17.'
    elif damage == 'source-owner': rows[0]['episode_id'] = 'foreign-episode'
    elif damage == 'task-owner': rows[0]['task_id'] = 'foreign-task'
    elif damage == 'source-commit': rows[0]['source_receipt']['source_commit'] = 'e' * 40
    elif damage == 'source-version': rows[0]['configuration']['version'] = 'cached-literal-record-windows-v2'
    elif damage == 'duplicate': rows.append(deepcopy(rows[0]))
    elif damage == 'orphan':
        row = deepcopy(rows[0]); row['episode_id'] = 'orphan'; row['source_id'] = W.source_id('orphan', 0); rows.append(row)
    elif damage == 'early': rows[0]['available_turn'] = 1
    elif damage == 'unknown-ref': bundles[-1]['collector']['web_cached_rows']['source_ids'] = ['unknown']
    elif damage == 'duplicate-ref': bundles[-1]['collector']['web_cached_rows']['source_ids'] *= 2
    elif damage == 'cleared-ref': bundles[-1]['collector']['web_cached_rows']['source_ids'] = []
    elif damage == 'future-ref': bundles[0]['collector']['web_cached_rows']['source_ids'] = [rows[0]['source_id']]
    elif damage == 'url': rows[0]['source']['url'] = 'https://other.example.org/'
    else: rows[0]['source_index'] = 1
    if damage != 'missing-sidecar': source_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    native_path = destination / 'native_web_train.jsonl'
    native_path.write_text(''.join(json.dumps(row) + '\n' for row in bundles))
    with pytest.raises((ValueError, FileNotFoundError)):
        N.load_native_training_data(native_path, tokenizer, expected_identity=identity,
                                   source_receipt=receipt, schema_resolver=schemas.original_schema)


@pytest.mark.parametrize('mutation', ['overwrite', 'clear', 'alias', 'preseed'])
def test_publisher_fails_on_cache_replacement_eviction_alias_and_preseed(tmp_path, mutation):
    class Journal:
        def __init__(self): self.rows = []
        def write(self, row): self.rows.append(deepcopy(row))
    publisher = W.CachePublisher(Journal(), task_id='task', episode_id='ep', source_receipt={},
                                 profile_version=R.WEB32K_CACHED_PROFILE)
    tools = WebTools()
    if mutation == 'preseed':
        tools.pages[URL] = TEXT
        with pytest.raises(ValueError): publisher.attach(tools)
        return
    publisher.attach(tools)
    tools.pages[URL] = TEXT
    publisher.snapshot(2)
    if mutation == 'overwrite': tools.pages[URL] += 'changed'
    elif mutation == 'clear': tools.pages.clear()
    else: tools.pages[URL + '#alias'] = TEXT
    with pytest.raises(ValueError): publisher.snapshot(3)


@pytest.mark.parametrize('damage', ['source-missing', 'source-duplicate', 'source-orphan', 'closed-config', 'request-config', 'missing-active-ref', 'early-ref'])
def test_closed_cache_validation_precedes_any_label_access(tmp_path, monkeypatch, damage):
    output, tokenizer, identity, approved, schemas, receipt, _ = generate(tmp_path, monkeypatch)
    source_path = output / 'web_cached_sources_rank0.jsonl'
    rows = list(C.read_jsonl(source_path))
    if damage == 'source-missing': source_path.unlink()
    elif damage in {'source-duplicate', 'source-orphan'}:
        extra = deepcopy(rows[0])
        if damage == 'source-orphan':
            extra['episode_id'] = 'orphan'; extra['source_id'] = W.source_id('orphan', 0)
        rows.append(extra); source_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        closed_path = output / 'generation_closed_rank0.json'; closed = json.loads(closed_path.read_text())
        closed['web_cached_source_records'] = 2; closed_path.write_text(json.dumps(closed))
    elif damage == 'closed-config':
        closed_path = output / 'generation_closed_rank0.json'; closed = json.loads(closed_path.read_text())
        closed['web_cached_rows']['version'] = 'old'; closed_path.write_text(json.dumps(closed))
    else:
        journal = output / 'requests_rank0.jsonl'; events = list(C.read_jsonl(journal))
        request = next(event for event in events if event['event'] == 'request' and event['turn_index'] == (0 if damage == 'early-ref' else 3))
        if damage == 'request-config': request['web_cached_rows']['configuration']['enabled'] = 1
        elif damage == 'missing-active-ref': request['web_cached_rows']['source_ids'] = []
        else: request['web_cached_rows']['source_ids'] = [rows[0]['source_id']]
        journal.write_text(''.join(json.dumps(event) + '\n' for event in events))
    monkeypatch.setattr(C, 'load_train_labels', lambda *a, **k: (_ for _ in ()).throw(AssertionError('labels opened before cache validation')))
    with pytest.raises((ValueError, FileNotFoundError)):
        C.select(output, tmp_path / 'must-not-exist', labels_path=tmp_path / 'never-open', profile=approved,
                 identity=identity, schema_profile=schemas, receipt=receipt, tokenizer=tokenizer)
    assert not (tmp_path / 'must-not-exist').exists()


def test_multiple_source_publications_reject_reordering_and_reference_reordering(tmp_path):
    class Journal:
        def __init__(self): self.rows = []
        def write(self, row): self.rows.append(deepcopy(row))
    journal = Journal(); publisher = W.CachePublisher(journal, task_id='task', episode_id='ep', source_receipt={},
                                                     profile_version=R.WEB32K_CACHED_PROFILE)
    tools = publisher.attach(WebTools())
    tools.pages[URL] = 'Name\nAlpha\n'; publisher.snapshot(2)
    tools.pages[URL + '?second'] = 'Name\nBeta\n'; actual = publisher.snapshot(3)
    source_path = tmp_path / 'sources.jsonl'
    source_path.write_text(''.join(json.dumps(row) + '\n' for row in journal.rows))
    registry = W.load_sources(source_path, source_receipt={}, profile_version=R.WEB32K_CACHED_PROFILE)
    owner = {'episode_id': 'ep', 'turn': 3, 'web_cached_rows': actual}
    assert len(W.receipt_sources(owner, registry, task_id='task')) == 2
    owner['web_cached_rows']['source_ids'].reverse()
    with pytest.raises(ValueError): W.receipt_sources(owner, registry, task_id='task')
    source_path.write_text(''.join(json.dumps(row) + '\n' for row in reversed(journal.rows)))
    with pytest.raises(ValueError): W.load_sources(source_path, source_receipt={}, profile_version=R.WEB32K_CACHED_PROFILE)


def test_cached_rows_cli_mismatch_fails_before_public_data_and_engine_load(tmp_path, monkeypatch):
    import transformers
    approved = profile(); schemas = N.NativeSchemaProfile(R.approved_web_profile(approved, N.WebSchemaProfile))
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **k: object())
    monkeypatch.setattr(C, 'load_approved', lambda *a, **k: (approved, object(), schemas, {}))
    monkeypatch.setattr(C, 'load_public_train', lambda *a, **k: (_ for _ in ()).throw(AssertionError('public data opened')))
    monkeypatch.setattr(sys, 'argv', ['collector', 'generate', '--source-root', 'synthetic', '--expected-source', 'f' * 40,
        '--profile', 'synthetic', '--model', 'synthetic', '--output-dir', str(tmp_path / 'not-created'),
        '--public-train-data', 'not-opened', '--collection-id', 'synthetic', '--web-whole-turn-context'])
    # Keep the test independent of history option spelling; profile check for
    # cached flag still occurs before public task/engine access.
    monkeypatch.setattr(C, 'prompt_history_from_arguments', lambda _: PromptHistoryConfig(True))
    sys.argv.remove('--web-whole-turn-context')
    with pytest.raises(ValueError, match='Cached rows flag'):
        C.main()
    assert not (tmp_path / 'not-created').exists()



def test_direct_legacy_runner_does_not_gain_cached_rows_by_bypassing_approval():
    legacy = N.NativeSchemaProfile(N.WebSchemaProfile(3, cached_rows=True))
    runner = C.make_runner('widesearch', legacy, timeout_seconds=5)
    assert runner.web_cached_rows is False


def test_direct_legacy_generator_cannot_activate_new_cache_route(tmp_path, monkeypatch):
    cls = FIXTURE['FakeTokenizer']
    monkeypatch.setattr(cls, '__len__', lambda self: 160000, raising=False)
    monkeypatch.setattr(cls, 'chat_template', 'synthetic-template', raising=False)
    tokenizer, _, native = FIXTURE['make_policy'](tmp_path, values=[])
    identity = N.NativeIdentity('synthetic-legacy', (151643, 151645), 'f' * 40)
    approved = {'version': R.LEGACY_PROFILE, 'identity': identity.manifest(tokenizer),
                'web': {'max_steps': 3, 'explicit_review': False, 'cached_rows': True}}
    with pytest.raises(ValueError, match='separate explicit v3 profile'):
        asyncio.run(C.generate([], native, tmp_path / 'never-created', collection_id='synthetic',
            profile=approved, schema_profile=N.NativeSchemaProfile(N.WebSchemaProfile(3, cached_rows=True))))
    asyncio.run(native.close())
    assert not (tmp_path / 'never-created').exists()
