"""CPU evaluator forwarding and failure controls; fixtures are synthetic only."""
import asyncio
import json
from pathlib import Path
import runpy
import sys
import types

import pytest
from slime_plugins.dcc import PolicyCompletion, Task
from slime_plugins.dcc import bf16_attention_guard as G

SOURCE=Path(__file__).resolve().parents[1]


def args_for(path,adapter=None,**updates):
    values=dict(model='synthetic/model',adapter=adapter,data=['synthetic.jsonl'],max_steps=3,
        max_new_tokens=1024,max_prompt_tokens=8192,initial_chunks=4,seed=42,limit=0,
        output_dir=str(path),retry_infrastructure=False,inference_backend='vllm',quantization='none',
        vllm_tensor_parallel_size=1,vllm_memory_utilization=.88,vllm_batched_tokens=8192,
        vllm_enforce_eager=True,batch_size=1,timeout=30,save_tokens=False)
    values.update(updates)
    return types.SimpleNamespace(**values)


@pytest.mark.parametrize('enabled,validation_fails,adapter_present',
    [(False,False,True),(True,False,True),(True,True,True),(True,False,False),(True,True,False)])
def test_evaluator_optin_before_generation_and_default_unchanged(monkeypatch,tmp_path,enabled,validation_fails,adapter_present):
    namespace=runpy.run_path(str(SOURCE/'scripts/dcc_model_evaluate.py'));evaluate=namespace['evaluate']
    task=Task('synthetic-contract','cuad','validation','Find clause',document='A contract',gold=[])
    monkeypatch.setitem(evaluate.__globals__,'load_tasks',lambda unused:[task])
    for key,value in [('WORLD_SIZE','1'),('LOCAL_RANK','0'),('RANK','0')]:monkeypatch.setenv(key,value)
    adapter=tmp_path/'adapter'
    if adapter_present:
        adapter.mkdir()
        (adapter/'adapter_config.json').write_text(json.dumps({'peft_type':'LORA','r':2,'target_modules':['q_proj','v_proj','o_proj']}))
        (adapter/'adapter_model.safetensors').write_bytes(b'synthetic-identity-only')
    adapter_value=str(adapter) if adapter_present else None
    calls=[]
    class Policy:
        provenance=dict(backend='vllm',quantization='none')
        if enabled:
            provenance.update(bf16_attention_only_lora=True,bf16_lora_validation=G.VALIDATION_LAYOUT,runtime_code_identity=G.runtime_identity())
        async def __call__(self,messages,**kwargs):
            calls.append('generate');return PolicyCompletion('{"answer":[]}',[1],[2])
        async def close(self):calls.append('close')
    def load(unused,**kwargs):
        assert 'quantization' not in kwargs
        if enabled:
            assert kwargs['bf16_attention_only_lora'] is True
            assert kwargs['bf16_lora_validation_dir']==str(tmp_path/'bf16_lora_validation_rank0')
        else:
            assert 'bf16_attention_only_lora' not in kwargs and 'bf16_lora_validation_dir' not in kwargs
        calls.append('load');return Policy()
    def validate(directory,*,adapter):
        assert directory==tmp_path/'bf16_lora_validation_rank0' and adapter==adapter_value
        calls.append('validate')
        if validation_fails:raise ValueError('synthetic incompatible BF16 receipts')
    monkeypatch.setitem(sys.modules,'slime_plugins.dcc.vllm_policy',types.SimpleNamespace(load_vllm_policy=load))
    monkeypatch.setattr(G,'validate_worker_receipts',validate)
    args=args_for(tmp_path,adapter=adapter_value,vllm_bf16_attention_only_lora=enabled)
    if validation_fails:
        with pytest.raises(ValueError,match='synthetic incompatible BF16 receipts'):asyncio.run(evaluate(args))
        assert calls==['load','validate','close']
        assert not (tmp_path/'rank0.done').exists() and not (tmp_path/'predictions_rank0.jsonl').exists()
    else:
        asyncio.run(evaluate(args))
        assert calls==(['load','validate','generate','close'] if enabled else ['load','generate','close'])
        manifest=json.loads((tmp_path/'manifest_rank0.json').read_text())
        runtime=json.loads((tmp_path/'inference_runtime.json').read_text())
        for record in (manifest,runtime):
            if enabled:
                assert record['bf16_attention_only_lora'] is True
                assert record['runtime_code_identity']==G.runtime_identity()
                assert record['bf16_lora_validation']==G.VALIDATION_LAYOUT
            else:
                assert 'bf16_attention_only_lora' not in record and 'runtime_code_identity' not in record


@pytest.mark.parametrize('updates',[
    dict(vllm_bf16_attention_only_lora=1),
    dict(vllm_bf16_attention_only_lora=True,quantization='nf4'),
    dict(vllm_bf16_attention_only_lora=True,adapter='synthetic',inference_backend='hf'),
    dict(vllm_bf16_attention_only_lora=True,adapter='synthetic',quantization='fp8'),
    dict(vllm_bf16_attention_only_lora=True,adapter='synthetic',vllm_tensor_parallel_size=2)])
def test_invalid_evaluator_optin_rejected_before_data_or_model(monkeypatch,tmp_path,updates):
    evaluate=runpy.run_path(str(SOURCE/'scripts/dcc_model_evaluate.py'))['evaluate']
    monkeypatch.setitem(evaluate.__globals__,'load_tasks',lambda unused:pytest.fail('must fail before reading data'))
    with pytest.raises((ValueError,TypeError),match='bf16|BF16'):
        asyncio.run(evaluate(args_for(tmp_path,**updates)))


def test_cli_exposes_explicit_disabled_by_default_flag(monkeypatch,capsys):
    namespace=runpy.run_path(str(SOURCE/'scripts/dcc_model_evaluate.py'))
    monkeypatch.setattr(sys,'argv',['dcc_model_evaluate.py','--help'])
    with pytest.raises(SystemExit) as stopped: namespace['main']()
    assert stopped.value.code==0
    assert '--vllm-bf16-attention-only-lora' in capsys.readouterr().out


# Reuse the CPU-only loader fixture; this checks the native async constructor.
from test_dcc_bf16_attention import loader, fixture as adapter_fixture


def test_async_loader_receives_same_optin_binding_and_restores_environment(loader,monkeypatch,tmp_path):
    import os
    load,calls=loader;adapter=adapter_fixture(tmp_path/'adapter');seen=[]
    class AsyncEngineArgs:
        def __init__(self,**kwargs): self.kwargs=kwargs
    def construct(arguments):
        seen.append((arguments.kwargs,json.loads(os.environ[G.BINDING_KEY])))
        return types.SimpleNamespace()
    monkeypatch.setitem(sys.modules,'vllm.engine',types.ModuleType('vllm.engine'))
    monkeypatch.setitem(sys.modules,'vllm.engine.arg_utils',types.SimpleNamespace(AsyncEngineArgs=AsyncEngineArgs))
    monkeypatch.setitem(sys.modules,'vllm.v1',types.ModuleType('vllm.v1'))
    monkeypatch.setitem(sys.modules,'vllm.v1.engine',types.ModuleType('vllm.v1.engine'))
    monkeypatch.setitem(sys.modules,'vllm.v1.engine.async_llm',types.SimpleNamespace(AsyncLLM=types.SimpleNamespace(from_engine_args=construct)))
    before=dict(os.environ)
    policy=load('synthetic',adapter=adapter,asynchronous=True,bf16_attention_only_lora=True,
                bf16_lora_validation_dir=tmp_path/'audit')
    assert not calls and len(seen)==1
    kwargs,binding=seen[0]
    assert kwargs['worker_cls']==G.WORKER_CLASS and kwargs['disable_log_stats'] is True
    assert binding['adapter']==str(adapter) and dict(os.environ)==before
    assert policy.provenance['async_engine'] is True
    assert policy.provenance['runtime_code_identity']==G.runtime_identity()
