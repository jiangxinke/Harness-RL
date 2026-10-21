"""Synthetic CPU contracts for explicit BF16 attention-only worker; no GPU claims."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import types
from unittest.mock import patch

import pytest
from slime_plugins.dcc import bf16_attention_guard as G

CONFIG = dict(model_type='qwen3_moe', num_experts=125, hidden_size=8, num_hidden_layers=2,
              num_attention_heads=2, num_key_value_heads=1, head_dim=4)


def fixture(path, targets=('q_proj', 'v_proj', 'o_proj')):
    path.mkdir()
    (path/'adapter_config.json').write_text(json.dumps(dict(peft_type='LORA', target_modules=list(targets), r=2)))
    header, offset = {}, 0
    shapes = dict(q_proj=(8,8), k_proj=(8,4), v_proj=(8,4), o_proj=(8,8))
    for layer in range(2):
        for target in targets:
            inputs, outputs = shapes.get(target, (8,8))
            for side, shape in [('A',[2,inputs]), ('B',[outputs,2])]:
                size = shape[0]*shape[1]*2
                name = f'base_model.model.model.layers.{layer}.self_attn.{target}.lora_{side}.weight'
                header[name] = dict(dtype='BF16', shape=shape, data_offsets=[offset,offset+size]); offset+=size
    raw=json.dumps(header).encode()
    (path/'adapter_model.safetensors').write_bytes(struct.pack('<Q',len(raw))+raw+b'\0'*offset)
    return path


class LinearBase: pass
class FusedMoE: pass
class UnquantizedLinearMethod: pass
class UnquantizedFusedMoEMethod: pass
class BaseLayerWithLoRA:
    def __init__(self, base_layer): self.base_layer=base_layer


def fake_model():
    rows=[]
    for layer in range(2):
        for leaf in ('qkv_proj','o_proj'):
            module=LinearBase(); module.quant_method=UnquantizedLinearMethod()
            rows.append((f'model.layers.{layer}.self_attn.{leaf}',module))
        expert=FusedMoE(); expert.quant_method=UnquantizedFusedMoEMethod()
        expert.global_num_experts=expert.local_num_experts=125
        rows.append((f'model.layers.{layer}.mlp.experts',expert))
    return types.SimpleNamespace(named_modules=lambda:iter(rows),parameters=lambda:iter(()),rows=rows)


def stock(model):
    return sorted({name.rsplit('.',1)[-1] for name,module in model.named_modules()})


def attention_filter(observations):
    return G.make_attention_filter(stock, model_config=CONFIG, targets=G.TARGETS,
        LinearBase=LinearBase, FusedMoE=FusedMoE, UnquantizedLinearMethod=UnquantizedLinearMethod,
        UnquantizedFusedMoEMethod=UnquantizedFusedMoEMethod, observations=observations)


def test_header_exact_qvo_complete_and_no_tensor_payload_reads(tmp_path):
    adapter=fixture(tmp_path/'adapter'); reads=[]; original=Path.open
    class Tracking:
        def __init__(self,stream): self.stream=stream
        def __enter__(self): return self
        def __exit__(self,*args): self.stream.close()
        def read(self,n=-1): reads.append(n); return self.stream.read(n)
    def opened(path,*args,**kwargs):
        stream=original(path,*args,**kwargs)
        return Tracking(stream) if path==adapter/'adapter_model.safetensors' and args and args[0]=='rb' else stream
    with patch.object(Path,'open',opened): header=G.validate_attention_header(adapter,CONFIG)
    assert header['target_modules']==['o_proj','q_proj','v_proj']
    assert header['tensor_pairs']==6 and header['tensor_payload_bytes_read']==0
    assert len(reads)==2 and reads[0]==8 and all(n>0 for n in reads)


@pytest.mark.parametrize('targets', [('q_proj','v_proj','o_proj','k_proj'), ('q_proj',), ('q_proj','v_proj','gate_proj')])
def test_reject_other_adapter_targets(tmp_path,targets):
    with pytest.raises(ValueError): G.validate_attention_header(fixture(tmp_path/'adapter',targets),CONFIG)


def test_filter_exact_inventory_and_expert_identity():
    model=fake_model(); obs=[]
    before=[(n,id(m),id(m.quant_method)) for n,m in model.rows if isinstance(m,FusedMoE)]
    assert attention_filter(obs)(model)==['o_proj','qkv_proj']
    assert obs[0]['expert_bindings']==before
    with pytest.raises(ValueError,match='exactly once'): attention_filter(obs)(model)


@pytest.mark.parametrize('damage', ['wrong_expert','wrong_attention','missing_attention','extra_collision','wrong_count','missing_expert'])
def test_filter_rejects_unsupported_runtime(damage):
    model=fake_model()
    if damage=='wrong_expert': model.rows[2][1].quant_method=object()
    elif damage=='wrong_attention': model.rows[0][1].quant_method=object()
    elif damage=='missing_attention': model.rows.pop(0)
    elif damage=='extra_collision': model.rows.append(('model.mlp.qkv_proj',model.rows[0][1]))
    elif damage=='wrong_count': model.rows[2][1].local_num_experts=124
    else: model.rows.pop(2)
    with pytest.raises(ValueError): attention_filter([])(model)


def test_binding_fresh_and_restores_on_error(tmp_path,monkeypatch):
    monkeypatch.delenv(G.BINDING_KEY,raising=False)
    adapter=fixture(tmp_path/'adapter'); audit=tmp_path/'audit'; before=dict(os.environ)
    with pytest.raises(RuntimeError,match='synthetic'):
        with G.worker_binding(adapter=adapter,model_config=CONFIG,audit_dir=audit):
            assert json.loads(os.environ[G.BINDING_KEY])['runtime_code_identity']==G.runtime_identity()
            raise RuntimeError('synthetic')
    assert dict(os.environ)==before
    with pytest.raises(FileExistsError):
        with G.worker_binding(adapter=adapter,model_config=CONFIG,audit_dir=audit): pass
    monkeypatch.setenv(G.BINDING_KEY,'prior')
    with pytest.raises(ValueError,match='prior'):
        with G.worker_binding(adapter=adapter,model_config=CONFIG,audit_dir=tmp_path/'new'): pass
    assert not (tmp_path/'new').exists()


def worker_case(tmp_path,monkeypatch,damage=None):
    monkeypatch.delenv(G.BINDING_KEY,raising=False)
    adapter=fixture(tmp_path/'adapter'); model=fake_model()
    models=types.ModuleType('vllm.lora.models'); models.get_supported_lora_modules=stock
    utils=types.ModuleType('vllm.lora.utils'); utils.get_supported_lora_modules=stock
    class Worker:
        def load_model(self):
            selected=models.get_supported_lora_modules(model)
            if damage=='load_failure': raise RuntimeError('synthetic model load failed')
            wrapped=[]; modules={}
            for name,module in model.rows:
                if name.rsplit('.',1)[-1] in selected:
                    wrapper=BaseLayerWithLoRA(module); modules[name]=wrapper
                    wrapped.extend(((name,wrapper),(name+'.base_layer',module)))
                else: wrapped.append((name,module))
            if damage=='extra_wrapper': wrapped.append(('model.layers.0.mlp.gate',BaseLayerWithLoRA(LinearBase())))
            if damage=='expert_mutated': next(m for n,m in wrapped if isinstance(m,FusedMoE)).quant_method=UnquantizedFusedMoEMethod()
            if damage=='manager_extra': modules['model.layers.0.mlp.gate']=object()
            model.rows[:]=wrapped
            model.lora_manager=types.SimpleNamespace(supported_lora_modules=selected,modules=modules)
            self.model_runner=types.SimpleNamespace(model=model)
    names=['vllm','vllm.v1','vllm.v1.worker','vllm.v1.worker.gpu_worker','vllm.lora',
           'vllm.lora.layers','vllm.lora.layers.base','vllm.model_executor','vllm.model_executor.layers',
           'vllm.model_executor.layers.linear','vllm.model_executor.layers.fused_moe',
           'vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method']
    mods={name:types.ModuleType(name) for name in names}
    mods['vllm'].__version__='0.11.2'; mods['vllm.v1.worker.gpu_worker'].Worker=Worker
    mods['vllm.lora'].models=models; mods['vllm.lora'].utils=utils
    mods['vllm.lora.models']=models; mods['vllm.lora.utils']=utils
    mods['vllm.lora.layers.base'].BaseLayerWithLoRA=BaseLayerWithLoRA
    mods['vllm.model_executor.layers.linear'].LinearBase=LinearBase
    mods['vllm.model_executor.layers.linear'].UnquantizedLinearMethod=UnquantizedLinearMethod
    mods['vllm.model_executor.layers.fused_moe'].FusedMoE=FusedMoE
    mods['vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method'].UnquantizedFusedMoEMethod=UnquantizedFusedMoEMethod
    torch=types.ModuleType('torch'); torch.bfloat16='synthetic-bf16'; mods['torch']=torch
    path=Path(G.__file__).with_name('bf16_attention_worker.py')
    spec=importlib.util.spec_from_file_location('slime_plugins.dcc._synthetic_bf16_worker',path)
    module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,mods):
        spec.loader.exec_module(module); worker=module.BF16AttentionWorker()
        worker.model_config=types.SimpleNamespace(quantization=None,dtype=torch.bfloat16,hf_config=types.SimpleNamespace(to_dict=lambda:CONFIG))
        worker.parallel_config=types.SimpleNamespace(tensor_parallel_size=1,pipeline_parallel_size=1,enable_expert_parallel=False)
        worker.lora_config=object()
        audit=tmp_path/'audit'
        with G.worker_binding(adapter=adapter,model_config=CONFIG,audit_dir=audit):
            if damage=='header_changed':
                config=json.loads((adapter/'adapter_config.json').read_text());config['r']=1
                (adapter/'adapter_config.json').write_text(json.dumps(config))
            if damage:
                with pytest.raises((ValueError,RuntimeError)): worker.load_model()
            else: worker.load_model()
        assert models.get_supported_lora_modules is stock
        if damage: assert not (audit/'worker-validation.actual.json').exists()
        else:
            actual=G.validate_worker_receipts(audit,adapter=adapter)
            assert actual['worker']['actual_expert_blocks']==2
            assert actual['worker']['expert_lora_wrappers']==0


def test_actual_worker_hook_lifecycle_with_synthetic_runtime(tmp_path,monkeypatch):
    worker_case(tmp_path,monkeypatch)


@pytest.mark.parametrize('damage',['load_failure','extra_wrapper','expert_mutated','manager_extra','header_changed'])
def test_worker_failure_restores_original_and_no_success_receipt(tmp_path,monkeypatch,damage):
    worker_case(tmp_path,monkeypatch,damage)


@pytest.fixture
def loader(monkeypatch):
    import torch
    import transformers
    from slime_plugins.dcc import vllm_policy
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    monkeypatch.setattr(torch.distributed,'is_initialized',lambda:False)
    for key in ('WORLD_SIZE','RANK','LOCAL_RANK',G.BINDING_KEY): monkeypatch.delenv(key,raising=False)
    monkeypatch.setattr(transformers.AutoConfig,'from_pretrained',lambda *a,**kw:types.SimpleNamespace(model_type='qwen3_moe',to_dict=lambda:deepcopy(CONFIG)))
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**kw:types.SimpleNamespace(pad_token_id=0,eos_token_id=1))
    monkeypatch.setattr(transformers.GenerationConfig,'from_pretrained',lambda *a,**kw:types.SimpleNamespace(eos_token_id=[1,2]))
    calls=[]
    def engine(**kwargs):
        calls.append((kwargs,os.environ.get(G.BINDING_KEY)))
        return types.SimpleNamespace(kwargs=kwargs)
    monkeypatch.setitem(sys.modules,'vllm',types.SimpleNamespace(__version__='0.11.2',LLM=engine))
    monkeypatch.setitem(sys.modules,'vllm.lora',types.SimpleNamespace())
    monkeypatch.setitem(sys.modules,'vllm.lora.request',types.SimpleNamespace(LoRARequest=lambda *args:args))
    return vllm_policy.load_vllm_policy,calls


def test_loader_default_and_false_are_identical_and_optin_is_bound(loader,tmp_path):
    load,calls=loader; adapter=fixture(tmp_path/'adapter')
    original=load('synthetic',adapter=adapter)
    explicit_false=load('synthetic',adapter=adapter,bf16_attention_only_lora=False)
    assert calls[0]==calls[1] and original.provenance==explicit_false.provenance
    assert 'worker_cls' not in calls[0][0] and calls[0][1] is None
    before=dict(os.environ)
    optimized=load('synthetic',adapter=adapter,bf16_attention_only_lora=True,bf16_lora_validation_dir=tmp_path/'audit')
    kwargs,binding=calls[2]
    assert kwargs['quantization'] is None and kwargs['worker_cls']==G.WORKER_CLASS
    assert json.loads(binding)['adapter']==str(adapter)
    assert optimized.provenance['runtime_code_identity']==G.runtime_identity()
    assert optimized.provenance['bf16_attention_only_lora'] is True and dict(os.environ)==before
    assert {k:v for k,v in kwargs.items() if k!='worker_cls'}==calls[0][0]


@pytest.mark.parametrize('bad',[dict(bf16_attention_only_lora=1),dict(bf16_attention_only_lora=True),
    dict(bf16_attention_only_lora=True,adapter='x',quantization='fp8'),
    dict(bf16_attention_only_lora=True,adapter='x',tensor_parallel_size=2),
    dict(bf16_lora_validation_dir='/synthetic')])
def test_loader_rejects_invalid_optin_before_engine(loader,bad):
    load,calls=loader
    with pytest.raises((ValueError,TypeError)): load('synthetic',**bad)
    assert not calls


def test_microbench_fixed_inputs_and_output_budget_contract():
    path=Path(__file__).resolve().parents[1]/'scripts/dcc_bf16_lora_latency.py'
    spec=importlib.util.spec_from_file_location('synthetic_latency_control',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    tokenizer=types.SimpleNamespace(encode=lambda text,**kw:[11,12,13])
    workload=module.build_workload(tokenizer,[4,7],2)
    assert workload[0]['prompts']==[{'prompt_token_ids':[11,12,13,11]}]*2
    assert workload[1]['prompts']==[{'prompt_token_ids':[11,12,13,11,12,13,11]}]*2
    prompt=workload[0]['prompts'][0]
    output=types.SimpleNamespace(prompt_token_ids=prompt['prompt_token_ids'],outputs=[types.SimpleNamespace(token_ids=[21,22],finish_reason='length')])
    assert module.checked_outputs([output],[prompt],2)[0]['token_ids']==[21,22]
    for damage in ('prompt','budget','finish'):
        bad=deepcopy(output)
        if damage=='prompt': bad.prompt_token_ids=[99]
        elif damage=='budget': bad.outputs[0].token_ids=[21]
        else: bad.outputs[0].finish_reason='stop'
        with pytest.raises(RuntimeError): module.checked_outputs([bad],[prompt],2)
    assert module.parser().parse_args(['--model','synthetic','--adapter','synthetic','--mode','stock','--output-dir','/synthetic']).enforce_eager is True
