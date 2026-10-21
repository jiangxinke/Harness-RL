"""CPU-only controls for a real no-adapter arm sharing the BF16 runtime route."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
from unittest.mock import patch

import pytest
from slime_plugins.dcc import bf16_attention_guard as G
from test_dcc_bf16_attention import (CONFIG, LinearBase, FusedMoE, BaseLayerWithLoRA,
    UnquantizedLinearMethod, UnquantizedFusedMoEMethod, fake_model, fixture as adapter_fixture,
    loader, worker_case as adapter_worker_case)


def native_inventory(model):
    return G.validate_base_model(model,CONFIG,LinearBase=LinearBase,FusedMoE=FusedMoE,
        BaseLayerWithLoRA=BaseLayerWithLoRA,UnquantizedLinearMethod=UnquantizedLinearMethod,
        UnquantizedFusedMoEMethod=UnquantizedFusedMoEMethod)


def test_base_inventory_accepts_only_complete_native_model():
    actual=native_inventory(fake_model())
    assert len(actual['native_attention_modules'])==4 and len(actual['expert_module_names'])==2


@pytest.mark.parametrize('damage',['wrapper','manager','missing_expert','missing_attention',
    'wrong_expert_method','wrong_attention_method','wrong_expert_count','extra_expert','extra_collision'])
def test_base_inventory_rejects_lora_or_changed_native_model(damage):
    model=fake_model()
    if damage=='wrapper':model.rows.append(('model.layers.0.mlp.gate',BaseLayerWithLoRA(LinearBase())))
    elif damage=='manager':model.lora_manager=object()
    elif damage=='missing_expert':model.rows.pop(2)
    elif damage=='missing_attention':model.rows.pop(0)
    elif damage=='wrong_expert_method':model.rows[2][1].quant_method=object()
    elif damage=='wrong_attention_method':model.rows[0][1].quant_method=object()
    elif damage=='wrong_expert_count':model.rows[2][1].local_num_experts=124
    elif damage=='extra_expert':model.rows.append(('model.layers.99.mlp.experts',model.rows[2][1]))
    else:model.rows.append(('model.router.qkv_proj',model.rows[0][1]))
    with pytest.raises(ValueError):native_inventory(model)


def base_worker_case(tmp_path,monkeypatch,damage=None):
    monkeypatch.delenv(G.BINDING_KEY,raising=False)
    model=fake_model();calls=[];assignments=[]
    def forbidden_filter(unused):pytest.fail('Base must never invoke the LoRA module filter')
    class TrackingModule(types.ModuleType):
        def __setattr__(self,key,value):
            if key=='get_supported_lora_modules':assignments.append(value)
            super().__setattr__(key,value)
    models=TrackingModule('vllm.lora.models');models.get_supported_lora_modules=forbidden_filter
    utils=types.ModuleType('vllm.lora.utils');utils.get_supported_lora_modules=forbidden_filter
    assignments.clear()
    def forbidden_header(*args,**kwargs):pytest.fail('Base must never read an adapter header')
    def forbidden_factory(*args,**kwargs):pytest.fail('Base must never construct a module filter')
    monkeypatch.setattr(G,'validate_attention_header',forbidden_header)
    monkeypatch.setattr(G,'make_attention_filter',forbidden_factory)
    class Worker:
        def load_model(self):
            calls.append('parent_load')
            assert self.lora_config is None
            assert models.get_supported_lora_modules is forbidden_filter
            if damage=='load_failure':raise RuntimeError('synthetic base load failure')
            if damage=='wrapper':model.rows.append(('model.layers.0.self_attn.qkv_proj',BaseLayerWithLoRA(model.rows[0][1])))
            if damage=='manager':model.lora_manager=object()
            if damage=='missing_expert':model.rows.pop(2)
            if damage=='expert_method':model.rows[2][1].quant_method=object()
            self.model_runner=types.SimpleNamespace(model=model)
    names=['vllm','vllm.v1','vllm.v1.worker','vllm.v1.worker.gpu_worker','vllm.lora',
           'vllm.lora.layers','vllm.lora.layers.base','vllm.model_executor','vllm.model_executor.layers',
           'vllm.model_executor.layers.linear','vllm.model_executor.layers.fused_moe',
           'vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method']
    mods={name:types.ModuleType(name)for name in names}
    mods['vllm'].__version__='0.11.2';mods['vllm.v1.worker.gpu_worker'].Worker=Worker
    mods['vllm.lora'].models=models;mods['vllm.lora'].utils=utils
    mods['vllm.lora.models']=models;mods['vllm.lora.utils']=utils
    mods['vllm.lora.layers.base'].BaseLayerWithLoRA=BaseLayerWithLoRA
    mods['vllm.model_executor.layers.linear'].LinearBase=LinearBase
    mods['vllm.model_executor.layers.linear'].UnquantizedLinearMethod=UnquantizedLinearMethod
    mods['vllm.model_executor.layers.fused_moe'].FusedMoE=FusedMoE
    mods['vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method'].UnquantizedFusedMoEMethod=UnquantizedFusedMoEMethod
    torch=types.ModuleType('torch');torch.bfloat16='synthetic-bf16';mods['torch']=torch
    path=Path(G.__file__).with_name('bf16_attention_worker.py')
    spec=importlib.util.spec_from_file_location('slime_plugins.dcc._synthetic_matched_base_worker',path)
    module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,mods):
        spec.loader.exec_module(module);worker=module.BF16AttentionWorker()
        worker.model_config=types.SimpleNamespace(quantization=None,dtype=torch.bfloat16,hf_config=types.SimpleNamespace(to_dict=lambda:CONFIG))
        worker.parallel_config=types.SimpleNamespace(tensor_parallel_size=1,pipeline_parallel_size=1,enable_expert_parallel=False)
        worker.lora_config=object()if damage=='lora_config'else None
        audit=tmp_path/'audit'
        before=dict(os.environ)
        with G.worker_binding(adapter=None,model_config=CONFIG,audit_dir=audit):
            binding=json.loads(os.environ[G.BINDING_KEY]);assert binding['adapter']is None
            header=json.loads((audit/'header-validation.actual.json').read_text())
            assert header['header_validation']is None and header['lora_enabled']is False
            if damage=='config_changed':
                header['model_validation']['hidden_size']=16
                (audit/'header-validation.actual.json').write_text(json.dumps(header))
            if damage:
                with pytest.raises((ValueError,RuntimeError)):worker.load_model()
            else:worker.load_model()
        assert dict(os.environ)==before
        assert models.get_supported_lora_modules is forbidden_filter and not assignments
        if damage:
            assert not(audit/'worker-validation.actual.json').exists()
            if damage in {'lora_config','config_changed'}:assert not calls
            return None
        actual=G.validate_worker_receipts(audit,adapter=None)
        assert calls==['parent_load']
        assert actual['worker']['lora_enabled']is False
        assert actual['worker']['attention_filter_invocations']==0
        assert actual['worker']['actual_lora_manager_present']is False
        assert actual['worker']['actual_attention_lora_wrappers']==[]
        assert actual['worker']['actual_expert_blocks']==2
        return actual


def test_actual_worker_base_never_reads_adapter_or_installs_filter(tmp_path,monkeypatch):
    base_worker_case(tmp_path,monkeypatch)


@pytest.mark.parametrize('damage',['load_failure','wrapper','manager','missing_expert','expert_method','lora_config','config_changed'])
def test_base_worker_failure_never_creates_success_receipt_or_modifies_filter(tmp_path,monkeypatch,damage):
    base_worker_case(tmp_path,monkeypatch,damage)


def test_base_model_config_rejected_before_audit_creation(tmp_path,monkeypatch):
    monkeypatch.delenv(G.BINDING_KEY,raising=False)
    for bad in (dict(CONFIG,num_experts=128),dict(CONFIG,decoder_sparse_step=2),dict(CONFIG,quantization_config={'x':1})):
        with pytest.raises(ValueError):
            with G.worker_binding(adapter=None,model_config=bad,audit_dir=tmp_path/'audit'):pass
        assert not(tmp_path/'audit').exists()


def test_base_loader_same_identity_as_adapter_and_actual_lora_disabled(loader,tmp_path):
    load,calls=loader
    default=load('synthetic')
    base=load('synthetic',adapter=None,bf16_attention_only_lora=True,bf16_lora_validation_dir=tmp_path/'base-audit')
    adapter=adapter_fixture(tmp_path/'adapter')
    trained=load('synthetic',adapter=adapter,bf16_attention_only_lora=True,bf16_lora_validation_dir=tmp_path/'trained-audit')
    default_kwargs,default_binding=calls[0];base_kwargs,base_binding=calls[1];trained_kwargs,trained_binding=calls[2]
    assert default_binding is None and 'worker_cls'not in default_kwargs
    assert 'runtime_code_identity'not in default.provenance
    assert base_kwargs['worker_cls']==trained_kwargs['worker_cls']==G.WORKER_CLASS
    assert base_kwargs['enable_lora']is False and trained_kwargs['enable_lora']is True
    assert base.lora_request is None and trained.lora_request is not None
    assert json.loads(base_binding)['adapter']is None
    assert json.loads(trained_binding)['adapter']==str(adapter)
    assert base.provenance['adapter']is None
    assert base.provenance['runtime_code_identity']==trained.provenance['runtime_code_identity']==G.runtime_identity()
    assert G.runtime_identity()['schema']=='dcc-bf16-attention-runtime-v2'
    assert {key:value for key,value in base_kwargs.items()if key!='worker_cls'}==default_kwargs


def test_base_receipt_rejects_forged_adapter_filter_manager_or_wrapper_claims(tmp_path,monkeypatch):
    actual=base_worker_case(tmp_path,monkeypatch);path=tmp_path/'audit/worker-validation.actual.json'
    original=path.read_text()
    for key,value in [('adapter','pretend-zero-adapter'),('lora_enabled',True),('attention_filter_invocations',1),
                      ('actual_lora_manager_present',True),('actual_attention_lora_wrappers',['model.layers.0.self_attn.qkv_proj']),
                      ('header_validation',{}),('actual_expert_module_names',[])]:
        row=json.loads(original);row[key]=value;path.write_text(json.dumps(row))
        with pytest.raises(ValueError):G.validate_worker_receipts(tmp_path/'audit',adapter=None)
    path.write_text(original)


def test_real_base_receipt_and_strict_adapter_receipt_share_runtime_identity(tmp_path,monkeypatch):
    base_dir=tmp_path/'base';base_dir.mkdir()
    # The helper's temporary guard patches must end before the adapter control.
    with monkeypatch.context()as local:base=base_worker_case(base_dir,local)
    adapter_dir=tmp_path/'trained';adapter_dir.mkdir()
    adapter_worker_case(adapter_dir,monkeypatch)
    trained=json.loads((adapter_dir/'audit/worker-validation.actual.json').read_text())
    assert base['worker']['runtime_code_identity']==trained['runtime_code_identity']
    assert base['worker']['adapter']is None and trained['adapter']is not None
    assert base['worker']['header_validation']is None
    assert trained['header_validation']['target_modules']==['o_proj','q_proj','v_proj']


def test_async_base_constructor_has_no_adapter_or_lora_and_same_identity(loader,monkeypatch,tmp_path):
    load,calls=loader;seen=[]
    class AsyncEngineArgs:
        def __init__(self,**kwargs):self.kwargs=kwargs
    def construct(arguments):
        payload=json.loads(os.environ[G.BINDING_KEY])
        header=json.loads((Path(payload['audit_dir'])/'header-validation.actual.json').read_text())
        seen.append((arguments.kwargs,payload,header));return types.SimpleNamespace()
    monkeypatch.setitem(sys.modules,'vllm.engine',types.ModuleType('vllm.engine'))
    monkeypatch.setitem(sys.modules,'vllm.engine.arg_utils',types.SimpleNamespace(AsyncEngineArgs=AsyncEngineArgs))
    monkeypatch.setitem(sys.modules,'vllm.v1',types.ModuleType('vllm.v1'))
    monkeypatch.setitem(sys.modules,'vllm.v1.engine',types.ModuleType('vllm.v1.engine'))
    monkeypatch.setitem(sys.modules,'vllm.v1.engine.async_llm',types.SimpleNamespace(AsyncLLM=types.SimpleNamespace(from_engine_args=construct)))
    monkeypatch.setattr(G,'validate_attention_header',lambda *args,**kwargs:pytest.fail('Base cannot read adapter headers'))
    before=dict(os.environ)
    policy=load('synthetic',adapter=None,asynchronous=True,bf16_attention_only_lora=True,
                bf16_lora_validation_dir=tmp_path/'audit')
    assert dict(os.environ)==before and not calls and len(seen)==1
    kwargs,binding,header=seen[0]
    assert kwargs['worker_cls']==G.WORKER_CLASS and kwargs['enable_lora']is False
    assert binding['adapter']is None and header['header_validation']is None and header['lora_enabled']is False
    assert policy.lora_request is None and policy.provenance['adapter']is None
    assert policy.provenance['async_engine']is True
    assert policy.provenance['runtime_code_identity']==G.runtime_identity()
