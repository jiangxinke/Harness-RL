"""Actual collection/physical APIs with exclusively invented CPU journals.

The source-approval loader alone is replaced with a fixture approval because
this /tmp candidate is intentionally not an adopted committed collection root.
Unchanged production read_closed, public replay, physical validation, TRAIN
label loader, scorer, original JSON schemas and prompt encoder all run.
"""
from __future__ import annotations
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'scripts')]
import dcc_grounded_final_builder as B
from test_grounded_final import SyntheticTokenizer
SOURCE=Path(os.environ.get('DCC_SYNTHETIC_RUNTIME_SOURCE',
    str(HERE.parent))).resolve()
API=B.import_parent(SOURCE)
Q=runpy.run_path(str(SOURCE/'tests/test_dcc_cuad_taskbound_contiguous.py'))
C,G,N,F=Q['C'],Q['G'],Q['N'],Q['F']
from slime_plugins.dcc.data import Task
from slime_plugins.dcc.episode import PolicyCompletion
from slime_plugins.dcc.eval_sampling import EvaluationSampling
from slime_plugins.dcc.native_budget import effective_sampling
from slime_plugins.dcc.native_thinking_eval import native_configuration


def fixture(root):
    tokenizer=SyntheticTokenizer();identity=F['identity'](tokenizer);receipt=F['FIXTURE']['RECEIPT']
    profile=Q['profile'](tokenizer);schemas=G.source_schema_profile(profile)
    document='Alpha agrees to preserve confidential information. Beta may terminate on notice.'
    gold=document.split('. ')[0]+'.'
    tasks=[Task('invented-positive','cuad','train',Q['Q']['QUESTION'],document=document),
           Task('invented-negative','cuad','train',Q['Q']['QUESTION'],document=document),
           Task('invented-unseen','cuad','train',Q['Q']['QUESTION'],document=document)]
    traces,requests,episodes=[],[],[]
    for task in tasks:
        values=[{'tool':'search_document','arguments':{'query':'confidential'}},
                {'tool':'read_document','arguments':{'start':0,'end':len(document)}},
                {'plan':'Explicitly synthetic review','checks':'Explicitly synthetic check','answer':[]}]
        trace_ids=[]
        class Policy:
            async def generate_structured(self,messages,**options):
                index=len(trace_ids);key=task.id+'-trace-'+str(index);trace_ids.append(key)
                metadata=dict(trace_id=key,task_id=task.id,rank=0,turn_index=index)
                prompt=API['hf'].encode_messages(tokenizer,messages,identity.max_prompt_tokens,allow_truncation=False)
                thought=tokenizer.encode('Explicitly synthetic possibly wrong thought.</think>')
                text=json.dumps(values[index]);final=tokenizer.encode(text)+[tokenizer.eos_token_id]
                phases=[]
                for phase_index,(name,ids) in enumerate((('thought',thought),('json',final))):
                    phase_prompt=prompt if phase_index==0 else prompt+thought
                    sampling=identity.budget.thought_sampling if phase_index==0 else EvaluationSampling()
                    raw=dict(token_ids=ids,engine_output_text='invented-no-engine-fixture',finish_reason='stop',stop_reason=ids[-1])
                    phases.append(dict(name=name,prompt_ids=phase_prompt,response_prompt_ids=phase_prompt,
                        response_finished=True,max_tokens=identity.budget.thought_tokens if phase_index==0 else 4096,
                        n=1,seed=42,ignore_eos=False,skip_special_tokens=False,include_stop_str_in_output=True,
                        structured_outputs=phase_index==1,output_kind='FINAL_ONLY',sampling=sampling.to_dict(),
                        effective_sampling=effective_sampling(sampling),stop_token_ids=[151643,151645,151668] if phase_index==0 else [151643,151645],
                        request_id=key+':'+name,raw_outputs=[deepcopy(raw)],decoded_text_raw=tokenizer.decode(ids,skip_special_tokens=False),**raw))
                trace=dict(**metadata,protocol=identity.budget.protocol,source_commit=identity.source_commit,
                    native_thinking=native_configuration(receipt,budget=identity.budget),status='completed',failure_reason=None,
                    json_complete=True,json_schema_valid=True,public_text_contains_thought=False,
                    declared_max_new_tokens=identity.budget.total_tokens,thought_max_tokens=identity.budget.thought_tokens,
                    json_max_tokens=4096,json_schema=options['json_schema'],original_prompt_ids=prompt,phases=phases,
                    completion_ids=thought+final,observed_completion_tokens=len(thought)+len(final),final_text=text)
                traces.append(trace);episode_id='invented-collection:'+task.id
                requests.append(dict(event='request',episode_id=episode_id,**metadata,messages=deepcopy(messages),options=deepcopy(options),
                                     cuad_retrieval=deepcopy(schemas.cuad.retrieval)))
                requests.append(dict(event='response',episode_id=episode_id,**metadata,native_trace_recorded=True,exception_type=None,
                    completion=dict(text=text,prompt_ids=prompt,completion_ids=thought+final,finish_reason='stop')))
                return PolicyCompletion(text,prompt,thought+final)
        runner=C.make_runner('cuad',schemas,timeout_seconds=900,total_tokens=identity.budget.total_tokens)
        result=asyncio.run(runner.run(task,Policy(),evaluation=True,score_train=False))
        assert result.status=='completed',result.detail
        episodes.append(dict(task_id=task.id,benchmark='cuad',split='train',episode_id='invented-collection:'+task.id,rank=0,
            trace_ids=trace_ids,public_task=task.to_dict(),status=result.status,answer=result.answer,exception_type=None,
            detail=result.detail,gold_visible_to_policy=False))
    raw=root/'closed';raw.mkdir()
    for name,rows in (('episodes_rank0.jsonl',episodes),('requests_rank0.jsonl',requests),('native_thinking_traces_rank0.jsonl',traces)):
        (raw/name).write_bytes(B.rows_bytes(rows))
    closed=dict(version=C.VERSION,collection_id='invented-collection',rank=0,world_size=1,profile=profile,source_receipt=receipt,
        task_ids=[t.id for t in tasks],request_events=len(requests),episodes=len(episodes),native_traces=len(traces),
        generation_journals_closed=True,labels_loaded=False,cuad_retrieval=deepcopy(schemas.cuad.retrieval))
    (raw/'generation_closed_rank0.json').write_text(B.canonical(closed))
    labels=root/'labels.jsonl';labels.write_bytes(B.rows_bytes([
        dict(task_id=t.id,benchmark='cuad',split='train',category='Invented confidentiality',
             gold=[] if t.id=='invented-negative' else ['Unobserved whole gold.'] if t.id=='invented-unseen' else [gold]) for t in tasks]))
    request=dict(protocol=B.PROTOCOL,parent_source_root=str(SOURCE),parent_source_commit='f'*40,profile='fixture-profile',
        collection_id='invented-collection',collection_world_size=1,closed_collection=str(raw),train_labels=str(labels),
        max_sequence_tokens=36864,minimum_positive=1,minimum_negative=1,allow_contiguous_union=False)
    approval=(profile,identity,schemas,receipt)
    return tokenizer,request,approval


def alter(path,fn):
    rows=[B.strict_json(line) for line in path.read_text().splitlines()];fn(rows);path.write_bytes(B.rows_bytes(rows))

class CollectionContracts(unittest.TestCase):
    def test_actual_closure_replay_physical_scorer_correct_earliest_tool_parent(self):
        with tempfile.TemporaryDirectory(prefix='dcc-synthetic-collection-') as temporary:
            tokenizer,request,approval=fixture(Path(temporary))
            with patch.object(C,'load_approved',return_value=approval):rows,audit,manifest=B.derive(request,tokenizer,API)
            self.assertEqual((len(rows),manifest['positive'],manifest['negative'],manifest['original_tasks']),(2,1,1,3))
            self.assertTrue(manifest['launchable']);self.assertEqual(len(audit),1)
            for row in rows:
                self.assertEqual(row['parent']['turn'],0);self.assertEqual(row['parent']['original_stage'],'search_document')
                self.assertFalse(row['parent']['original_response_was_final'])
                self.assertEqual(row['prompt_ids'],row['parent']['original_prompt_ids']+row['parent']['actual_thought_ids'])
                self.assertEqual(row['target_metric']['f1'],1.)
                self.assertFalse(row['target_is_actual_native_generation'])
            self.assertIn('does-not-prove-absence',next(row for row in rows if not row['positive'])['negative_supervision'])
    def test_missing_closure_and_public_replay_failure_precede_label_access(self):
        for damage in ('closure','replay'):
            with self.subTest(damage=damage),tempfile.TemporaryDirectory(prefix='dcc-synthetic-closure-') as temporary:
                tok,request,approval=fixture(Path(temporary));raw=Path(request['closed_collection'])
                if damage=='closure':(raw/'generation_closed_rank0.json').unlink()
                else:
                    def mutate(rows):rows[0]['messages'][2]['content']+=' forged public observation'
                    alter(raw/'requests_rank0.jsonl',mutate)
                with patch.object(C,'load_approved',return_value=approval),patch.object(C,'load_train_labels',side_effect=AssertionError('labels touched')) as labels:
                    with self.assertRaises((ValueError,FileNotFoundError)):B.derive(request,tok,API)
                    labels.assert_not_called()
    def test_physically_invalid_first_request_excluded_later_actual_parent_used(self):
        with tempfile.TemporaryDirectory(prefix='dcc-synthetic-physical-') as temporary:
            tok,request,approval=fixture(Path(temporary));raw=Path(request['closed_collection'])
            def mutate(rows):rows[0]['phases'][0]['stop_reason']=10001
            alter(raw/'native_thinking_traces_rank0.jsonl',mutate)
            with patch.object(C,'load_approved',return_value=approval):rows,audit,manifest=B.derive(request,tok,API)
            positive=next(row for row in rows if row['positive'])
            self.assertEqual(positive['parent']['turn'],1)
            self.assertTrue(any(row.get('disposition')=='original-parent-ineligible' for row in audit))
            self.assertEqual(manifest['original_tasks'],3)
    def test_partial_episode_status_kept_and_overlength_is_fatal_without_truncation(self):
        with tempfile.TemporaryDirectory(prefix='dcc-synthetic-overlength-') as temporary:
            tok,request,approval=fixture(Path(temporary));raw=Path(request['closed_collection'])
            def mutate(rows):rows[0]['status']='timeout'
            alter(raw/'episodes_rank0.jsonl',mutate)
            with patch.object(C,'load_approved',return_value=approval):rows,audit,manifest=B.derive(request,tok,API)
            self.assertEqual(next(row for row in rows if row['positive'])['parent']['episode_status'],'timeout')
            request['max_sequence_tokens']=1
            with patch.object(C,'load_approved',return_value=approval):rows,audit,manifest=B.derive(request,tok,API)
            self.assertEqual(rows,[]);self.assertFalse(manifest['launchable']);self.assertEqual(len(manifest['overlength']),2)
            self.assertEqual(manifest['deficit'],dict(positive=1,negative=1))

if __name__=='__main__':unittest.main(verbosity=2)
