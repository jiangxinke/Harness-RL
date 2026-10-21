"""Synthetic CPU-only prospective literal TRAIN reward -> receipt -> DPO gates.

No real task, real reward, model weights, network, GPU or benchmark run is used.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'scripts'),
    str(Path(__file__).resolve().parents[1] / 'slime')]
import dcc_score_native_preferences as S
import dcc_native_dpo_train as TRAIN
from slime_plugins.dcc.data import Task
from slime_plugins.dcc.episode import _WebEvidence
from slime_plugins.dcc_train_literal_reward_contract import (
    PROTOCOL, LABEL_PROTOCOL, LITERAL_RECEIPT, LITERAL_REWARD_INPUTS, LITERAL_DPO_COLLECTIONS,
    PER_COLLECTION_RECEIPTS, configuration,
)

spec = importlib.util.spec_from_file_location('literal_reward_closed_fixture',
    Path(__file__).with_name('test_dcc_native_preference_rewards.py'))
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
closed_domains = F.closed_domains
URL = F.URL


def invented(gold=None):
    task = Task('invented-literal-task', 'widesearch', 'train', 'List the invented rows.',
        metadata={'evaluation': {'required': ['id','value']}})
    label = {'task_id':task.id,'benchmark':'widesearch','split':'train',
        'evaluation':{'required':['id','value']},
        'gold':gold if gold is not None else [{'id':'a','value':'8'},{'id':'b','value':'7.90'}],
        'label_protocol':LABEL_PROTOCOL,'train_reward':configuration()}
    guard=_WebEvidence(); guard.observe('web_search',{'results':[{'url':URL}]})
    guard.observe('fetch_url',{'url':URL,'text':'Invented public evidence containing the rows.'})
    answer=[dict(x,citations=[URL]) for x in label['gold']]
    return task,label,answer,guard


def new_score(task,label,answer,guard):
    return S.web_literal_reward(task,label,answer,guard,reward_version=PROTOCOL,train_reward=configuration())


def receipt():
    task,label,answer,guard=invented(); value,components=new_score(task,label,answer,guard)
    base={'contract':S.RECEIPT,'task_id':task.id,'trace_id':'synthetic-trace','prefix_id':'synthetic-prefix',
        'harness_id':'synthetic-harness','reward_version':PROTOCOL,'value':value,
        'generation_closed_before_scoring':True,'split':'train'}
    return S.literal_receipt(base,components,train_reward=configuration())


def write_row(path,row):
    path.write_text(json.dumps(row)+'\n');return path


def literal_manifest(closed_domains):
    tokenizer,manifest=closed_domains;manifest=deepcopy(manifest)
    manifest['version']=LITERAL_REWARD_INPUTS
    web=manifest['collections'][1];web.update(reward_version=PROTOCOL,train_reward=configuration())
    path=Path(web['train_labels']);labels=[json.loads(x) for x in path.read_text().splitlines()]
    labels[1].update(label_protocol=LABEL_PROTOCOL,train_reward=configuration())
    literal_path=path.with_name('separate-literal-Web-TRAIN.jsonl')
    F.write_jsonl(literal_path,[labels[1]])
    web['train_labels']=str(literal_path)
    return tokenizer,manifest


def dpo_manifest(manifest,profiles,output):
    fields=('collection_id','approved_freeze','generation_dir','generation_world_size','expected_source','reward_version')
    rows=[]
    for original,profile in zip(manifest['collections'],profiles):
        row={key:original[key] for key in fields};row['train_rewards']=str(output/profile['receipt_file'])
        if row['reward_version']==PROTOCOL:row['train_reward']=configuration()
        rows.append(row)
    return {'version':LITERAL_DPO_COLLECTIONS,'base_derivative_id':manifest['base_derivative_id'],
        'tokenizer_id':manifest['tokenizer_id'],'collections':rows}


def dpo_args(path):
    return SimpleNamespace(approved_collections=str(path),base_derivative_id='synthetic-derivative',
        tokenizer_id='synthetic-tokenizer',approved_freeze=None,generation_dir=None,train_rewards=None,
        reward_version=None,expected_source=None,train_reward=None)


def test_literal_exact_decimal_preserves_codepoints_while_legacy_reward_body_remains():
    task,label,answer,guard=invented();frozen=deepcopy((label,answer))
    assert new_score(task,label,answer,guard)[0]==1
    assert S.web_reward(task,label,answer,guard)[0]<1
    changed=deepcopy(answer);changed[1]['value']='7.9'
    value,components=new_score(task,label,changed,guard)
    assert value==.5 and components['train_exact_row_f1']==.5
    assert (label,answer)==frozen


@pytest.mark.parametrize('damage',['numeric','null-like','extra-column','missing-column','missing-citation','unread-citation'])
def test_literal_reward_cannot_infer_cells_or_bypass_existing_evidence_guard(damage):
    task,label,answer,guard=invented()
    if damage=='numeric':answer[0]['value']=8
    elif damage=='null-like':answer[0]['value']=None
    elif damage=='extra-column':answer[0]['extra']='text'
    elif damage=='missing-column':del answer[0]['value']
    elif damage=='missing-citation':del answer[0]['citations']
    else:answer[0]['citations']=['https://example.org/unread']
    value,components=new_score(task,label,answer,guard)
    assert value==0 or damage=='null-like' and value==.5
    assert 'f1_by_row' not in components and 'f1_by_item' not in components


def test_duplicate_predictions_count_false_positive_and_guard_rejection_skips_metric(monkeypatch):
    task,label,answer,guard=invented(gold=[{'id':'a','value':'8'}])
    assert new_score(task,label,answer+deepcopy(answer),guard)[0]==pytest.approx(2/3)
    def forbidden(*args,**kwargs):raise AssertionError('Do not call metric after evidence guard rejects')
    monkeypatch.setattr(S,'evaluate_train_literal_rows',forbidden)
    value,components=new_score(task,label,answer,_WebEvidence())
    assert value==0 and components['train_metric_evaluated'] is False
    label.pop('label_protocol')
    with pytest.raises(ValueError):new_score(task,label,answer,_WebEvidence())


def test_new_receipt_requires_complete_independent_approval_and_preserves_ownership(tmp_path):
    row=receipt();path=write_row(tmp_path/'receipt.jsonl',row)
    actual=TRAIN.read_train_rewards(path,reward_version=PROTOCOL,train_reward=configuration())
    assert actual[row['trace_id']].value==1 and actual[row['trace_id']].reward_version==PROTOCOL
    for kwargs in ({'reward_version':PROTOCOL}, {'reward_version':S.REWARD_VERSIONS['widesearch']},
                   {'reward_version':S.REWARD_VERSIONS['widesearch'],'train_reward':configuration()}):
        with pytest.raises(ValueError):TRAIN.read_train_rewards(path,**kwargs)
    with pytest.raises(ValueError):
        TRAIN.read_train_rewards(tmp_path/'unopened',reward_version=PROTOCOL,train_reward={})


DAMAGES=['extra-field','missing-field','old-contract','old-version','dev','closed-int','value-bool','value-nan',
    'config-omission','config-extra','config-bool','label-protocol','metric-name','metric-scope','metric-extra',
    'metric-value','metric-bool','metric-not-evaluated','guard-version','guard-bool','guard-extra','guard-failed',
    'ownership-empty']
@pytest.mark.parametrize('damage',DAMAGES)
def test_new_receipt_rejects_before_constructing_TrainRewardReceipt(tmp_path,monkeypatch,damage):
    row=receipt()
    if damage=='extra-field':row['extra']=True
    elif damage=='missing-field':del row['train_reward']
    elif damage=='old-contract':row['contract']=S.RECEIPT
    elif damage=='old-version':row['reward_version']=S.REWARD_VERSIONS['widesearch']
    elif damage=='dev':row['split']='dev'
    elif damage=='closed-int':row['generation_closed_before_scoring']=1
    elif damage=='value-bool':row['value']=True
    elif damage=='value-nan':row['value']=float('nan')
    elif damage=='config-omission':row['train_reward'].pop('row_key')
    elif damage=='config-extra':row['train_reward']['extra']=True
    elif damage=='config-bool':row['train_reward']['limits']['columns']=True
    elif damage=='label-protocol':row['label_protocol']='legacy'
    elif damage=='metric-name':row['train_metric']['name']='f1_by_item'
    elif damage=='metric-scope':row['train_metric']['scope']='official'
    elif damage=='metric-extra':row['train_metric']['extra']=1
    elif damage=='metric-value':row['train_metric']['value']=.5
    elif damage=='metric-bool':row['train_metric']['value']=True
    elif damage=='metric-not-evaluated':row['train_metric']['evaluated']=False
    elif damage=='guard-version':row['evidence_guard']['version']='relaxed'
    elif damage=='guard-bool':row['evidence_guard']['passed']=1
    elif damage=='guard-extra':row['evidence_guard']['extra']=1
    elif damage=='guard-failed':row['evidence_guard']['passed']=False;row['train_metric']['evaluated']=False
    else:row['task_id']=' '
    def forbidden(**kwargs):raise AssertionError('Receipt constructed before full contract validation')
    monkeypatch.setattr(TRAIN,'TrainRewardReceipt',forbidden)
    with pytest.raises(ValueError):TRAIN.read_train_rewards(write_row(tmp_path/'bad.jsonl',row),
        reward_version=PROTOCOL,train_reward=configuration())


def test_legacy_receipt_cannot_borrow_new_version_without_full_contract(tmp_path):
    row=receipt();row={k:v for k,v in row.items() if k in {
        'contract','task_id','trace_id','prefix_id','harness_id','reward_version','value',
        'generation_closed_before_scoring','split'}};row['contract']=S.RECEIPT
    with pytest.raises(ValueError):TRAIN.read_train_rewards(write_row(tmp_path/'borrowed.jsonl',row),
        reward_version=PROTOCOL,train_reward=configuration())


def test_actual_synthetic_original_g4_native_reward_receipts_dpo_roundtrip(closed_domains,tmp_path):
    tokenizer,manifest=literal_manifest(closed_domains)
    prepared=S.prepare_collections(manifest,tokenizer);result=S.score_prepared_collections(prepared)
    assert [[r['value'] for r in rows] for _,rows,_,_ in result]==[[1,0,0,1],[1,0,0,1]]
    assert set(result[0][1][0])=={'contract','task_id','trace_id','prefix_id','harness_id','reward_version',
        'value','generation_closed_before_scoring','split'}
    assert result[0][1][0]['contract']==S.RECEIPT
    assert result[1][1][0]['contract']==LITERAL_RECEIPT
    assert result[1][1][1]['evidence_guard']['passed'] is False
    assert result[1][1][1]['train_metric']=={'name':'train_exact_row_f1','scope':'widesearch-train-only',
        'value':0,'evaluated':False}
    profiles=S.publish_rewards(result,tmp_path/'rewards')
    path=tmp_path/'dpo.json';path.write_text(json.dumps(dpo_manifest(manifest,profiles,tmp_path/'rewards')))
    pairs,metadata=TRAIN.load_training_pairs(dpo_args(path),tokenizer)
    assert len(pairs)==2 and metadata['collection_merge_contract']==LITERAL_DPO_COLLECTIONS
    web=metadata['collections'][1]
    assert web['train_reward']==configuration() and web['train_metric']=='train_exact_row_f1'
    assert web['reward_contract']==LITERAL_RECEIPT
    assert 'train_reward' not in metadata['collections'][0]
    assert str(tmp_path) not in json.dumps(metadata)


@pytest.mark.parametrize('damage',['legacy-manifest','missing-config','changed-config','legacy-label','changed-label','g4closure','originalclosure','ownership'])
def test_new_mode_rejects_bad_manifest_or_prerequisites_without_labels(closed_domains,tmp_path,monkeypatch,damage):
    tokenizer,manifest=literal_manifest(closed_domains);row=manifest['collections'][1]
    if damage=='legacy-manifest':manifest['version']=S.VERSION
    elif damage=='missing-config':del row['train_reward']
    elif damage=='changed-config':row['train_reward']['limits']['columns']=True
    elif damage=='g4closure':(Path(row['generation_dir'])/'generation_closed_rank0.json').unlink()
    elif damage=='originalclosure':(Path(row['original_source_dir'])/'generation_closed_rank0.json').unlink()
    elif damage=='ownership':
        original=S.read_closed_g4;seen=[]
        def collision(*args,**kwargs):
            result=original(*args,**kwargs)
            if not seen:seen.extend(result['phase_request_ids'])
            else:result['phase_request_ids'][0]=seen[0]
            return result
        monkeypatch.setattr(S,'read_closed_g4',collision)
    else:
        path=Path(row['train_labels']);labels=[json.loads(x) for x in path.read_text().splitlines()]
        if damage=='legacy-label':labels[-1].pop('label_protocol')
        else:labels[-1]['train_reward']['row_key']='infer'
        F.write_jsonl(path,labels)
    if damage not in {'legacy-label','changed-label'}:
        def forbidden(*args,**kwargs):raise AssertionError('labels opened before complete prerequisites')
        monkeypatch.setattr(S,'load_train_labels',forbidden)
    with pytest.raises((ValueError,FileNotFoundError)):
        S.score_prepared_collections(S.prepare_collections(manifest,tokenizer))


@pytest.mark.parametrize('damage',['old-version','missing-config','changed-config','unexpected-config','new-without-literal'])
def test_new_dpo_manifest_requires_explicit_new_protocol_before_any_reward_access(closed_domains,tmp_path,monkeypatch,damage):
    tokenizer,manifest=literal_manifest(closed_domains)
    prepared=S.prepare_collections(manifest,tokenizer);profiles=S.publish_rewards(S.score_prepared_collections(prepared),tmp_path/'rewards')
    data=dpo_manifest(manifest,profiles,tmp_path/'rewards')
    if damage=='old-version':data['version']=TRAIN.COLLECTIONS_CONTRACT
    elif damage=='missing-config':del data['collections'][1]['train_reward']
    elif damage=='changed-config':data['collections'][1]['train_reward']['limits']['columns']=True
    elif damage=='unexpected-config':data['collections'][0]['train_reward']=configuration()
    else:data['collections']=data['collections'][:1]
    path=tmp_path/'bad-dpo.json';path.write_text(json.dumps(data))
    def forbidden(*args,**kwargs):raise AssertionError('reward receipt opened before explicit new manifest/config')
    monkeypatch.setattr(TRAIN,'read_train_rewards',forbidden)
    with pytest.raises(ValueError):TRAIN.load_training_pairs(dpo_args(path),tokenizer)


def synthetic_approval():
    """Configuration assertions only; synthetic NF4 claims are never run as NF4."""
    from dataclasses import replace,asdict
    from slime_plugins.dcc.native_dpo_adapter import DPO_APPROVAL,DPO_METHOD
    from dcc_native_dpo import ReferenceIdentity
    class Tokenizer(F.F.FIXTURE['FakeTokenizer']):
        chat_template='synthetic-template'
        def __len__(self):return 160000
    tokenizer=Tokenizer()
    identity=F.F.N.NativeIdentity('synthetic-base',(151643,151645),'d'*40,16384,36864,42,16384,'thinking')
    source={'version':'dcc-native-inference-source-v1','source_commit':'d'*40,
        'base_source_commit':F.F.N.BASE_SOURCE_COMMIT,'protocol':F.F.N.PROTOCOL,'frozen':True}
    native=identity.manifest(tokenizer);native.pop('generation_seed')
    profile={'collection_id':'synthetic-literal-Web','experiment_id':'synthetic-experiment','harness_id':'synthetic-harness',
        'schema_policy':'final-decision-only-v1','collection_source_commit':'d'*40,
        'collection_source_receipt':source,'collection_native_identity':native,
        'generation_world_size':1,'generation_closed_before_scoring':True,'attempt_count':4,'valid_candidates':4,
        'eligible_pairs':1,'omitted_prefixes':{},'training_sources':['independent_curriculum'],
        'pair_selection':'one-best-worst-per-frozen-prefix; reward-gap-strictly-greater-than-0.05',
        'frozen_prefix_ids':['synthetic-prefix'],'frozen_task_ids':['synthetic-task'],
        'attempt_trace_ids':[f'trace-{i}' for i in range(4)],'phase_request_ids':[f'phase-{i}' for i in range(8)],
        'selected_pairs':[{'prefix_id':'synthetic-prefix','chosen_trace_id':'trace-0','rejected_trace_id':'trace-1',
            'chosen_seed':0,'rejected_seed':1}],
        'reward_version':PROTOCOL,'reward_contract':LITERAL_RECEIPT,'train_reward':configuration(),
        'train_metric':'train_exact_row_f1'}
    profile['public_freeze']={'labels_loaded':False,'native_identity':native,'source_receipt':source,
        'prefix_ids':profile['frozen_prefix_ids'],'task_ids':profile['frozen_task_ids'],
        **{key:profile[key] for key in ('experiment_id','harness_id','schema_policy')}}
    cfg={'algorithm':TRAIN.ALGORITHM,'training_method':DPO_METHOD,'fresh_run':True,'warm_start':False,
        'resume_supported':False,'reference_models_loaded':0,'measured_KL':False,'gradient_weighting':'global-equal-pair',
        'sequence_score':'sum-all-actual-native-thought-close-JSON-EOS-token-logprobs',
        'candidate_contract':'native-final-preference-candidate-v1','reward_contract':PER_COLLECTION_RECEIPTS,
        'pair_contract':'native-final-best-worst-strict-gap-v1','minimum_reward_gap':.05,
        'pair_selection':profile['pair_selection'],'steps':1,'pairs_per_step':1,'lora_rank':16,'lora_alpha':32,
        'lora_target_modules':['q_proj','v_proj','o_proj'],'lora_dropout':0,'weight_decay':0,'max_grad_norm':1,
        'beta':.1,'learning_rate':1e-6,'training_source':{'version':'dcc-offline-native-dpo-training-source-v1',
            'source_commit':'e'*40,'tracked_runtime_clean':True,'training_entrypoint':'dcc_native_dpo_train.py'},
        'quantization':{'mode':'nf4','transformers_version':'4.57.6','expert_4bit_coverage':1.,'expert_parameters':96,
            'expert_4bit_parameters':96,'expert_4bit_matrices':12},
        'reference':asdict(ReferenceIdentity('synthetic-base','synthetic-tokenizer',TRAIN.QUANTIZATION_ID,'base')),
        'collection_merge_contract':LITERAL_DPO_COLLECTIONS,'collections':[profile],
        **{key:profile[key] for key in ('generation_closed_before_scoring','eligible_pairs','valid_candidates','attempt_count','training_sources')}}
    return {'version':DPO_APPROVAL,'training_configuration':cfg,'checkpoint_optimizer_steps':1}


def test_adoption_retains_complete_literal_profile_without_loosening_legacy_exact_fields():
    from slime_plugins.dcc.native_dpo_adapter import validate_dpo_approval
    approval=synthetic_approval();assert validate_dpo_approval(approval)==approval
    legacy=deepcopy(approval);cfg=legacy['training_configuration'];p=cfg['collections'][0]
    for field in ('reward_contract','train_reward','train_metric'):p.pop(field)
    p['reward_version']='synthetic-legacy-train-reward'
    cfg.update(reward_contract=TRAIN.REWARD_CONTRACT,collection_merge_contract=TRAIN.COLLECTIONS_CONTRACT)
    assert validate_dpo_approval(legacy)==legacy
    p['train_reward']=configuration()
    with pytest.raises(ValueError):validate_dpo_approval(legacy)


@pytest.mark.parametrize('damage',['old-mode','missing-config','changed-config','wrong-metric','wrong-receipt',
    'cuad-source','extra-field','legacy-version','new-mode-without-new-profile'])
def test_adoption_rejects_cross_protocol_and_missing_literal_config(damage):
    from slime_plugins.dcc.native_dpo_adapter import validate_dpo_approval
    value=synthetic_approval();cfg=value['training_configuration'];p=cfg['collections'][0]
    if damage=='old-mode':cfg.update(reward_contract=TRAIN.REWARD_CONTRACT,collection_merge_contract=TRAIN.COLLECTIONS_CONTRACT)
    elif damage=='missing-config':p.pop('train_reward')
    elif damage=='changed-config':p['train_reward']['limits']['rows']=True
    elif damage=='wrong-metric':p['train_metric']='f1_by_item'
    elif damage=='wrong-receipt':p['reward_contract']=TRAIN.REWARD_CONTRACT
    elif damage=='cuad-source':p['training_sources']=['cuad_train'];cfg['training_sources']=['cuad_train']
    elif damage=='extra-field':p['extra']=True
    elif damage=='legacy-version':p['reward_version']=S.REWARD_VERSIONS['widesearch']
    else:
        for field in ('reward_contract','train_reward','train_metric'):p.pop(field)
        p['reward_version']='synthetic-legacy-version'
    with pytest.raises(ValueError):validate_dpo_approval(value)


def test_literal_module_import_stays_inert_without_dcc_package_evaluators():
    import subprocess
    root=Path(__file__).resolve().parents[1]
    code=("import sys;sys.path[:0]="+repr([str(root/'scripts'),str(root/'slime')])+";"
        "import dcc_train_web_literal_reward;"
        "assert 'slime_plugins.dcc' not in sys.modules;"
        "assert not {'torch','pandas','requests','httpx'}.intersection(sys.modules)")
    result=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
