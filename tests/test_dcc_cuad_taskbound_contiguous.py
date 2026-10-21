"""Synthetic CPU controls for the explicit CUAD task-bound contiguous v2 route."""
from copy import deepcopy
import asyncio
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'slime')]
Q = runpy.run_path(str(ROOT / 'tests/test_dcc_cuad_retrieval_binding.py'))
C, G, N, R, F = (Q[name] for name in ('C', 'G', 'N', 'R', 'F'))
from dcc_review_training import CUAD_TASKBOUND_UNION_PROFILE, CUAD_RETRIEVAL_PROFILE
from slime_plugins.dcc.data import Task
from slime_plugins.dcc.episode import EpisodeRunner
from slime_plugins.dcc.tools import DocumentTools
from slime_plugins.dcc import retrieval


@pytest.fixture
def tokenizer(monkeypatch):
    cls = F['FIXTURE']['FakeTokenizer']
    monkeypatch.setattr(cls, '__len__', lambda self: 160000, raising=False)
    monkeypatch.setattr(cls, 'chat_template', 'synthetic-template', raising=False)
    return cls()


def profile(tokenizer):
    value = Q['new_profile'](tokenizer)
    value['version'] = CUAD_TASKBOUND_UNION_PROFILE
    value['cuad']['retrieval'] = Q['binding'](taskbound_contiguous=True)
    return value


def source_messages(task, ranges):
    return [{'role': 'system', 'content': 'Use the source.'}, {'role': 'user', 'content': task.prompt},
            {'role': 'user', 'content': 'Initial contract retrieval (source material):\n' + json.dumps({
                'document_length': len(task.document), 'hits': [
                    {'start': a, 'end': b, 'text': task.document[a:b], 'score': 1.0} for a,b in ranges]})}]


def record(task, messages, span):
    return {'source': 'cuad_train', 'benchmark': 'cuad', 'split': 'train', 'task_id': task.id,
            'stage': 'final', 'messages': messages,
            'completion': json.dumps({'answer': [{'text': span, 'confidence': .9}]}),
            'grounding_fields': [span], 'metadata': {'category': 'Confidentiality', 'positive': True}}


def test_current_task_routes_expansion_while_original_query_is_retained():
    value = Q['profile_value']()
    other = 'Extract termination clauses.'
    value['questions'][other] = {'category': other, 'terms': ['foreigncategory']}
    bound = R.make_retrieval_binding(json.dumps(value).encode(), taskbound_contiguous=True)
    runner = EpisodeRunner(**R.retrieval_runner_kwargs(bound))
    task = Q['task'](); tools = DocumentTools(task.document)
    with patch.object(retrieval, '_rank', wraps=retrieval._rank) as ranks:
        runner._search_document(tools, 'Termination', task=task)
    assert [call.args[1] for call in ranks.call_args_list] == ['Termination', 'confidential preserve']
    unknown = Task('unknown-question', 'cuad', 'train', 'Unfitted question.', document=task.document)
    with patch.object(retrieval, '_rank', wraps=retrieval._rank) as ranks:
        actual = runner._search_document(tools, other, task=unknown)
    ranks.assert_not_called()
    assert actual == tools.search_document(other, top_k=4)


def test_default_binding_and_unbound_short_query_preserve_v1():
    runner = EpisodeRunner(**R.retrieval_runner_kwargs(Q['binding']()))
    task = Q['task']();tools = DocumentTools(task.document)
    assert 'retrieval_task_bound_expansion' not in R.retrieval_runner_kwargs(Q['binding']())
    with patch.object(retrieval, '_rank', wraps=retrieval._rank) as ranks:
        assert runner._search_document(tools, 'Confidentiality') == tools.search_document('Confidentiality')
    ranks.assert_not_called()


@pytest.mark.parametrize('damage', ['no-task', 'wrong-document', 'wrong-domain'])
def test_task_anchor_cannot_be_omitted_or_cross_documents(damage):
    runner = EpisodeRunner(**R.retrieval_runner_kwargs(Q['binding'](taskbound_contiguous=True)))
    task = Q['task']();tools = DocumentTools(task.document)
    if damage == 'no-task': task = None
    elif damage == 'wrong-document': task = Task('other', 'cuad', 'train', Q['QUESTION'], document='other')
    else: task = Task('other', 'widesearch', 'train', Q['QUESTION'], document=task.document)
    with pytest.raises(ValueError): runner._search_document(tools, 'short', task=task)


@pytest.mark.parametrize('version,binding_version', [
    (CUAD_RETRIEVAL_PROFILE, R.TASKBOUND_UNION_VERSION), (CUAD_TASKBOUND_UNION_PROFILE, R.VERSION)])
def test_new_and_old_profile_markers_cannot_be_substituted(tokenizer, version, binding_version):
    value = profile(tokenizer);value['version'] = version
    value['cuad']['retrieval']['version'] = binding_version
    with pytest.raises(ValueError): G.source_schema_profile(value)


@pytest.mark.parametrize('right_start', [10,14])
def test_exact_overlap_or_adjacent_sources_union_under_existing_whitespace_rules(right_start):
    task = Task('invented-union', 'cuad', 'train', Q['QUESTION'], document='Alpha  preserves\nBeta rights!')
    messages = source_messages(task,[(0,14),(right_start,len(task.document))])
    span = 'Alpha preserves Beta rights!'
    row = record(task,messages,span)
    with pytest.raises(ValueError,match='absent'): N.legacy.validate_cuad_record(row)
    sources = R.contiguous_observed_sources(task,messages)
    assert sources == [task.document]
    assert N.legacy.validate_cuad_record(row,_native_cuad_sources=sources) is row
    altered = deepcopy(row);altered['grounding_fields'] = ['Alpha preserves beta rights!']
    altered['completion'] = json.dumps({'answer':[{'text':altered['grounding_fields'][0]}]})
    with pytest.raises(ValueError,match='absent'): N.legacy.validate_cuad_record(altered,_native_cuad_sources=sources)


@pytest.mark.parametrize('damage', ['one-character-gap','edited-text','offset','boolean-offset','document-length','foreign-document','invented-answer','encoded-hit-missing'])
def test_source_union_rejects_gap_changed_hits_other_document_and_hallucination(damage):
    task = Task('invented-union', 'cuad', 'train', Q['QUESTION'], document='Alpha preserves Beta rights!')
    ranges = [(0,14),(15,27)] if damage == 'one-character-gap' else [(0,16),(10,27)]
    messages = source_messages(task,ranges)
    payload = json.loads(messages[2]['content'].split('\n',1)[1])
    if damage == 'edited-text': payload['hits'][0]['text'] = 'edited source!!!'
    elif damage == 'offset': payload['hits'][0]['start'] += 1
    elif damage == 'boolean-offset': payload['hits'][0]['start'] = False
    elif damage == 'document-length': payload['document_length'] += 1
    elif damage == 'foreign-document': task = Task(task.id,'cuad','train',task.prompt,document='Other preserves Beta rights!')
    messages[2]['content'] = 'Initial contract retrieval (source material):\n' + json.dumps(payload)
    span = 'Invented liability rule.' if damage == 'invented-answer' else 'Alpha preserves Beta rights!'
    row = record(task,messages,span)
    with pytest.raises(ValueError):
        sources = R.contiguous_observed_sources(task,messages,
            encoded_prompt='source missing' if damage == 'encoded-hit-missing' else None)
        N.legacy.validate_cuad_record(row,_native_cuad_sources=sources)


def closed_union(tmp_path, tokenizer):
    # Unique words avoid accidental matching in another chunk. The answer crosses
    # the 1600-character chunk boundary and no one hit contains it.
    document = ' '.join('word%04d' % i for i in range(380))
    span = ' '.join(document[1300:1850].split())
    task = Task('invented-native-union','cuad','train',Q['QUESTION'],document=document)
    raw = tmp_path/'raw'
    _,engine,native = F['make_native'](raw,values=[{'answer':[{'text':span,'confidence':.9}]}])
    approved = profile(tokenizer);schemas = G.source_schema_profile(approved)
    asyncio.run(C.generate([task],native,raw,collection_id='invented-v2-union',profile=approved,schema_profile=schemas))
    assert len(engine.calls) == 2
    labels = tmp_path/'synthetic-labels.jsonl'
    labels.write_text(json.dumps({'task_id':task.id,'benchmark':'cuad','split':'train','gold':[span],'category':'Confidentiality'})+'\n')
    destination = tmp_path/'selected'
    C.select(raw,destination,labels_path=labels,profile=approved,identity=F['identity'](tokenizer),
             schema_profile=schemas,receipt=F['FIXTURE']['RECEIPT'],tokenizer=tokenizer)
    bundles = list(C.read_jsonl(destination/'native_cuad_train.jsonl'))
    assert len(bundles) == 1
    return destination/'native_cuad_train.jsonl', bundles[0], schemas


def test_full_native_collection_selector_and_loader_accept_union_without_changing_ids(tmp_path,tokenizer):
    path,bundle,schemas = closed_union(tmp_path,tokenizer)
    before = deepcopy(bundle)
    with pytest.raises(ValueError,match='absent'): N.legacy.validate_cuad_record(bundle['record'])
    schema = schemas.original_schema(bundle['record'],bundle['collector'])
    example = N.validate_native_record(bundle,tokenizer,expected_identity=F['identity'](tokenizer),
        original_schema=schema,cuad_schema_profile=schemas.cuad)
    assert example.prompt == bundle['native_trace']['original_prompt_ids']
    assert example.completion == bundle['native_trace']['completion_ids']
    assert example.native_phase_lengths == tuple(len(p['token_ids']) for p in bundle['native_trace']['phases'])
    loaded = list(N.iter_native_records(path,tokenizer,expected_identity=F['identity'](tokenizer),schema_resolver=schemas.original_schema))
    assert loaded == [bundle] and bundle == before
    with pytest.raises(ValueError,match='explicit trusted profile'):
        N.validate_native_record(bundle,tokenizer,expected_identity=F['identity'](tokenizer),original_schema=schema)
    broken = deepcopy(bundle);broken['collector'].pop('cuad_retrieval')
    with pytest.raises(ValueError):schemas.original_schema(broken['record'],broken['collector'])
    broken = deepcopy(bundle);broken['collector']['public_task']['id'] = 'foreign-task'
    with pytest.raises(ValueError):schemas.original_schema(broken['record'],broken['collector'])


@pytest.mark.parametrize('damage', ['text-only', 'unbound-offsets', 'changed-offset-in-observation', 'stripped-source-header'])
def test_native_visibility_binds_complete_observation_offsets_and_text(damage):
    task = Task('invented-encoded-binding','cuad','train',Q['QUESTION'],document='Alpha preserves Beta rights!')
    messages = source_messages(task,[(0,16),(10,len(task.document))])
    content = messages[2]['content']
    # Both quoted source strings remain present in every fabricated prompt.
    payload = json.loads(content.split('\n',1)[1])
    texts = ' '.join(json.dumps(h['text'])[1:-1] for h in payload['hits'])
    if damage == 'text-only': encoded = texts
    elif damage == 'unbound-offsets': encoded = texts + ' start 0 end 16 start 10 end 27'
    elif damage == 'changed-offset-in-observation': encoded = content.replace('"start": 10','"start": 9')
    else: encoded = content.split('\n',1)[1]
    with pytest.raises(ValueError,match='Complete CUAD source observation with offsets'):
        R.contiguous_observed_sources(task,messages,encoded_prompt=encoded)
    assert R.contiguous_observed_sources(task,messages,encoded_prompt='system\n'+content+'\nassistant') == [task.document]


def test_repeated_identical_source_text_cannot_stand_in_for_wrong_offset_observation():
    text = 'Identical visible quotation.'
    task = Task('invented-repeated-offset','cuad','train',Q['QUESTION'],document=text+' gap '+text)
    offset = len(text)+5
    messages = source_messages(task,[(offset,offset+len(text))])
    wrong = source_messages(task,[(0,len(text))])[2]['content']
    assert text in wrong and text in messages[2]['content']
    with pytest.raises(ValueError,match='Complete CUAD source observation with offsets'):
        R.contiguous_observed_sources(task,messages,encoded_prompt=wrong)
    with pytest.raises(ValueError,match='Complete CUAD source observation with offsets'):
        R.contiguous_observed_sources(task,messages,encoded_prompt=messages[2]['content'].replace('"score": 1.0','"score":1.0'))
    assert R.contiguous_observed_sources(task,messages,encoded_prompt=messages[2]['content']) == [text]
