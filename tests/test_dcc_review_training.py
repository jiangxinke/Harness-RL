"""CPU-only fake tools and model responses; no weights, network or held-out data."""
from copy import deepcopy
import ast,asyncio,json,random,runpy,sys
from collections import Counter,defaultdict
from pathlib import Path
from types import SimpleNamespace
import pytest
from jsonschema import ValidationError

SOURCE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(SOURCE/'scripts'),str(SOURCE/'slime')]
import dcc_collect_native_train as C
import dcc_native_sft as N
import dcc_tool_sft as T
from dcc_review_training import REVIEW20_PROFILE,WEB_STAGE_PROFILE,approved_web_profile
from slime_plugins.dcc.data import Task
from slime_plugins.dcc.episode import PolicyCompletion
URL='https://example.org/public';TEXT='Alpha is the public listed name.'

def profile():return N.WebSchemaProfile(20,True,False,True,REVIEW20_PROFILE)
def schema_profile():return N.NativeSchemaProfile(profile(),None)
def action(tool=None):
 result={'plan':'Inspect the original public task.','checks':'Use only read source text.'}
 result.update({'tool':tool,'arguments':{'query':'public name'}if tool=='web_search'else{'url':URL}}if tool else{'answer':[{'Name':'Alpha','citations':[URL]}]})
 return result
class Tools:
 async def web_search(self,query):return {'results':[{'url':URL,'title':'Public list'}]}
 async def fetch_url(self,url):return {'url':url,'text':TEXT,'start':0,'end':len(TEXT)}
 def citation_validity(self,answer):return float(all(URL in r.get("citations",[])for r in answer))
 async def close(self):pass
class Policy:
 def __init__(self,values):self.values=list(values);self.calls=[]
 async def generate_structured(self,messages,**options):
  value=self.values.pop(0);self.calls.append((deepcopy(messages),deepcopy(options),deepcopy(value)))
  return PolicyCompletion(json.dumps(value))

def recorded(length=3):
 task=Task('review-public-fixture','widesearch','train','Find the name in the public list.',metadata={'evaluation':{'required':['Name']}})
 values=[action('web_search'),action('fetch_url')]+[action('web_search')for _ in range(length-3)]+[action()]
 policy=Policy(values)
 runner=C.make_runner('widesearch',schema_profile(),timeout_seconds=5,web_tools_factory=Tools)
 result=asyncio.run(runner.run(task,policy,evaluation=True,score_train=False))
 assert result.status=='completed',result.detail
 records=[]
 for turn,(messages,options,value)in enumerate(policy.calls):
  stage='final'if'answer'in value else{'web_search':'search','fetch_url':'fetch'}[value['tool']]
  record=dict(source='independent_curriculum',split='train',task_id=task.id,stage=stage,messages=messages,completion=json.dumps(value))
  records.append((record,options,turn))
 return records

@pytest.mark.parametrize('length',[3,20])
def test_exact_real_runner_messages_and_review_schema_replay_without_target_edits(length):
 for record,options,turn in recorded(length):
  before=deepcopy(record)
  assert profile().original_schema(record,{'turn':turn})==options['json_schema']
  assert T.validate_record(record,_native_web_profile=profile())is record
  assert record==before

def test_default_v1_still_rejects_review_or_navigation_and_unknown_profile():
 for p in [N.WebSchemaProfile(20,True),N.WebSchemaProfile(20,False,False,True),N.WebSchemaProfile(20,True,False,True,'unknown')]:
  with pytest.raises(ValueError):p.validate()
 legacy={'version':'dcc-native-train-profile-v1','identity':{},'web':{'max_steps':8,'explicit_review':False,'cached_rows':False}}
 p=approved_web_profile(legacy,N.WebSchemaProfile);assert p.max_steps==8 and not p.explicit_review and not p.navigation_ledger
 legacy['web']['navigation_ledger']=True
 with pytest.raises(ValueError):approved_web_profile(legacy,N.WebSchemaProfile)

@pytest.mark.parametrize('damage',['steps','review','navigation','cached','boolsteps'])
def test_new_profile_requires_exact20_review_navigation_and_no_cached_rows(damage):
 p={'version':REVIEW20_PROFILE,'identity':{},'collection_overlay_id':'synthetic-explicit-overlay','web':dict(max_steps=20,explicit_review=True,cached_rows=False,navigation_ledger=True)}
 key,value={'steps':('max_steps',19),'review':('explicit_review',False),'navigation':('navigation_ledger',False),'cached':('cached_rows',True),'boolsteps':('max_steps',True)}[damage];p['web'][key]=value
 with pytest.raises(ValueError):approved_web_profile(p,N.WebSchemaProfile)

@pytest.mark.parametrize('damage',['ledger','system','remaining','historyreview','turn','extrauser','missingledger'])
def test_review_replay_rejects_tampered_public_history_and_control_bytes(damage):
 record,options,turn=deepcopy(recorded(3)[-1])
 if damage=='ledger':record['messages'][-1]['content']+=' forged'
 elif damage=='system':record['messages'][0]['content']+=' extra instructions'
 elif damage=='remaining':record['messages'][3]['content']=record['messages'][3]['content'].replace('Remaining turns: 19.','Remaining turns: 7.')
 elif damage=='historyreview':
  a=json.loads(record['messages'][2]['content']);a['plan']='x'*241;record['messages'][2]['content']=json.dumps(a)
 elif damage=='turn':turn+=1
 elif damage=='extrauser':record['messages'].insert(2,{'role':'user','content':'Evidence check: final answer rejected.'})
 else:record['messages'].pop()
 with pytest.raises((ValueError,ValidationError)):profile().original_schema(record,{'turn':turn})

def test_last_turn_review_control_is_exact_and_never_accepted_as_an_observation():
 record,_,turn=deepcopy(recorded(20)[-1]);record['messages'][-1]['content']+=' extra'
 with pytest.raises(ValueError):profile().original_schema(record,{'turn':turn})

def sampler(cls,rank=0,stage_profile=None):
 records=[dict(task_id='web-'+str(i),stage=s)for i in range(16)for s in T.STAGES]
 def encoder(r,*a,**k):return T.Example(r['task_id'],r['stage'],'public',[1],[2])
 kwargs=dict(rank=rank,world=8,seed=47,web_fraction=1.,negative_fraction=1/3,max_prompt=64,max_response=64,batch_tokens=256,record_encoder=encoder)
 if stage_profile is not None:kwargs['web_stage_profile']=stage_profile
 return cls(records,[],None,None,**kwargs)

def test_opt_in_stage_cycle_has_exact50_25_25_and_no_default_rng_or_sequence_change():
 parent=SOURCE/'tests/fixtures/dcc_original_mixed_sampler.snapshot.py'
 tree=ast.parse(parent.read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='MixedSampler')
 namespace=dict(T.__dict__);exec(compile(ast.Module(body=[node],type_ignores=[]),str(parent),'exec'),namespace);original=namespace['MixedSampler']
 for rank in range(8):
  a,b=sampler(original,rank),sampler(T.MixedSampler,rank)
  assert [(x.task_id,x.stage)for x in [a.sample()for _ in range(120)]]==[(x.task_id,x.stage)for x in [b.sample()for _ in range(120)]]
  weighted=sampler(T.MixedSampler,rank,WEB_STAGE_PROFILE)
  assert Counter(weighted.sample().stage for _ in range(120))=={'final':60,'search':30,'fetch':30}


def test_selector_rejects_whole_long_complete_episode_without_cropping_and_preserves_audit(tmp_path,monkeypatch):
 episode=dict(task_id='synthetic-cuad',benchmark='cuad',status='completed',episode_id='fixture-episode',trace_ids=['t'+str(i)for i in range(13)],answer=[])
 events=[]
 monkeypatch.setattr(C,'read_closed',lambda *a,**k:(events.append('all-journals-closed')or([episode],{},{})))
 monkeypatch.setattr(C,'load_train_labels',lambda *a,**k:(events.append('labels-after-closure')or{'synthetic-cuad':{}}))
 destination=tmp_path/'selected'
 result=C.select(tmp_path/'synthetic-closed',destination,labels_path=tmp_path/'not-opened-labels',profile={},identity=None,schema_profile=None,receipt={},tokenizer=None,maximum_episode_requests=12)
 assert events==['all-journals-closed','labels-after-closure']
 assert result['accepted_episodes']==0 and result['rejected_episodes']==1 and result['whole_trajectory_filter_only']is True
 assert 'never crop'in(destination/'audit.jsonl').read_text()
 assert(destination/'native_web_train.jsonl').read_bytes()==b'' and(destination/'native_cuad_train.jsonl').read_bytes()==b''
 assert len(episode['trace_ids'])==13


def test_selector_never_opens_labels_when_any_journal_is_unclosed(tmp_path,monkeypatch):
 def unclosed(*a,**k):raise ValueError('synthetic unclosed rank')
 monkeypatch.setattr(C,'read_closed',unclosed)
 monkeypatch.setattr(C,'load_train_labels',lambda *a,**k:(_ for _ in()).throw(AssertionError('labels opened')))
 with pytest.raises(ValueError,match='unclosed'):
  C.select(tmp_path,tmp_path/'selected',labels_path=tmp_path/'not-opened',profile={},identity=None,schema_profile=None,receipt={},tokenizer=None)
 assert not(tmp_path/'selected').exists()


def test_new_explicit_profile_preserves_real_native_ids_and_all_file_loader(tmp_path,monkeypatch):
 fixture=runpy.run_path(str(SOURCE/'tests/test_dcc_native_thinking.py'))
 cls=fixture['FakeTokenizer'];monkeypatch.setattr(cls,'__len__',lambda self:160000,raising=False);monkeypatch.setattr(cls,'chat_template','synthetic-template',raising=False)
 from slime_plugins.dcc.native_budget import NativeBudget
 budget=NativeBudget(16384,'thinking');record,options,turn=recorded(3)[0];value=json.loads(record['completion'])
 tokenizer,engine,native=fixture['make_policy'](tmp_path,values=[value],budget=budget)
 identity=N.NativeIdentity('synthetic-derivative',(151643,151645),'f'*40,16384,16384+budget.total_tokens,42,budget.thought_tokens,budget.thought_sampling_profile)
 async def produce():
  try:
   await native.generate_structured(record['messages'],max_new_tokens=budget.total_tokens,temperature=0.0,json_schema=options['json_schema'],trace_metadata=dict(trace_id='review-fixture-trace',task_id=record['task_id'],rank=0,turn_index=0))
  finally:await native.close()
 asyncio.run(produce());trace=native.traces[0]
 bundle=dict(contract=N.CONTRACT,record=deepcopy(record),identity=identity.manifest(tokenizer),collector=dict(split='train',task_input='public-train-task-without-gold',gold_visible_to_policy=False,recorded_messages=True,synthetic_thought=False,inserted_close=False,inserted_eos=False,episode_id='synthetic-reviewed-episode',turn=0,rank=0,trace_id='review-fixture-trace'),native_trace=deepcopy(trace))
 before=deepcopy(bundle)
 example=N.validate_native_record(bundle,tokenizer,expected_identity=identity,original_schema=options['json_schema'],web_schema_profile=profile())
 assert example.completion==trace['completion_ids'] and example.prompt==trace['original_prompt_ids'] and bundle==before
 with pytest.raises(ValueError):N.validate_native_record(bundle,tokenizer,expected_identity=identity,original_schema=options['json_schema'])
 root=tmp_path/'synthetic-source';root.mkdir();(root/'.dcc-native-source.json').write_text(json.dumps(fixture['RECEIPT']))
 overlay=dict(overlay_id='synthetic-explicit-overlay',explicit_training_profile=REVIEW20_PROFILE,baseline_source_unchanged=False)
 (root/'.dcc-training-code-overlay.json').write_text(json.dumps(overlay))
 p=dict(version=REVIEW20_PROFILE,identity=identity.manifest(tokenizer),collection_overlay_id=overlay['overlay_id'],web=dict(max_steps=20,explicit_review=True,cached_rows=False,navigation_ledger=True))
 approved=tmp_path/'profile.json';approved.write_text(json.dumps(p));records=tmp_path/'actual-fake-engine-records.jsonl';records.write_text(json.dumps(bundle)+'\n')
 args=SimpleNamespace(adapter=None,cuad_train_data=None,native_expected_source=identity.source_commit,native_train_profile=str(approved),cuad_tool_records=None,records=str(records),model='synthetic-no-weights')
 data=N.prepare_native_training(args,source_root=root,tokenizer=tokenizer)
 assert data.configuration['training_profile_version']==REVIEW20_PROFILE and data.configuration['validated_records']==1
 assert data.encode_record(data.records[0],tokenizer,max_prompt=16384,max_response=20480,batch_tokens=49152,skips=Counter()).completion==trace['completion_ids']
 assert bundle==before
