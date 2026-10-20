"""CPU guards and synthetic engines; no actual model tensors or GPU launches."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).resolve().parents[1]
PKG=SOURCE/'slime/slime_plugins/dcc'
from slime_plugins.dcc import fp8_attention_guard as G
from slime_plugins.dcc.fp8_runtime import runtime_identity

CONFIG=dict(model_type='qwen3_moe',num_experts=125,hidden_size=8,num_hidden_layers=2,
            num_attention_heads=2,num_key_value_heads=1,head_dim=4)

def fixture(path):
    path.mkdir()
    config=dict(peft_type='LORA',target_modules=['q_proj','k_proj','v_proj','o_proj'],r=2,bias='none')
    (path/'adapter_config.json').write_text(json.dumps(config))
    shapes={'q_proj':(8,8),'k_proj':(8,4),'v_proj':(8,4),'o_proj':(8,8)}
    header={};offset=0
    for layer in range(2):
        for target,(inputs,outputs) in shapes.items():
            for side,shape in [('A',[2,inputs]),('B',[outputs,2])]:
                n=shape[0]*shape[1]*2
                key=f'base_model.model.model.layers.{layer}.self_attn.{target}.lora_{side}.weight'
                header[key]=dict(dtype='BF16',shape=shape,data_offsets=[offset,offset+n]);offset+=n
    write_header(path,header,offset)
    return header,offset

def write_header(path,header,size):
    raw=json.dumps(header).encode();(path/'adapter_model.safetensors').write_bytes(struct.pack('<Q',len(raw))+raw+b'\0'*size)

class LinearBase:pass
class FusedMoE:pass
Fp8LinearMethod=type('Fp8LinearMethod',(),{})
Fp8MoEMethod=type('Fp8MoEMethod',(),{})

def fake_model():
    rows=[]
    for layer in range(2):
        for leaf in ('qkv_proj','o_proj'):
            linear=LinearBase();linear.quant_method=Fp8LinearMethod();linear.quant_method.use_marlin=True
            rows.append((f'model.layers.{layer}.self_attn.{leaf}',linear))
        expert=FusedMoE();expert.global_num_experts=125;expert.local_num_experts=125;expert.quant_method=Fp8MoEMethod();expert.quant_method.use_marlin=True
        rows.append((f'model.layers.{layer}.mlp.experts',expert))
    for layer in range(2):
        router=LinearBase();router.quant_method=type('UnquantizedLinearMethod',(),{})()
        router.weight=types.SimpleNamespace(dtype='synthetic_bfloat16',shape=(125,8))
        rows.append((f'model.layers.{layer}.mlp.gate',router))
    return types.SimpleNamespace(named_modules=lambda:iter(rows),parameters=lambda:iter(()),rows=rows)

def stock(model):return list({name.rsplit('.',1)[-1]for name,module in model.named_modules()})

class Checks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='dcc-fp8-synthetic-cpu-')
        self.root=Path(self.tmp.name);self.adapter=self.root/'adapter';self.header,self.size=fixture(self.adapter)
    def tearDown(self):self.tmp.cleanup()
    def test_header_complete_and_reads_header_only(self):
        original=Path.open;reads=[];target=self.adapter/'adapter_model.safetensors'
        class Tracking:
            def __init__(self,stream):self.stream=stream
            def __enter__(self):return self
            def __exit__(self,*args):self.stream.close()
            def read(self,n=-1):reads.append(n);return self.stream.read(n)
        def opened(path,*args,**kwargs):
            stream=original(path,*args,**kwargs)
            return Tracking(stream)if path==target and args and args[0]=='rb'else stream
        with patch.object(Path,'open',opened):actual=G.validate_attention_header(self.adapter,CONFIG)
        self.assertEqual(reads,[8,len(json.dumps(self.header).encode())]);self.assertEqual(actual['tensor_pairs'],8)
        self.assertEqual(actual['selected_model_module_leaves'],['o_proj','qkv_proj']);self.assertEqual(actual['tensor_payload_bytes_read'],0)
    def test_header_missing_pair_rejected(self):
        del self.header[next(iter(self.header))];write_header(self.adapter,self.header,self.size)
        with self.assertRaisesRegex(ValueError,'incomplete'):G.validate_attention_header(self.adapter,CONFIG)
    def test_header_expert_rejected(self):
        self.header['base_model.model.model.layers.0.mlp.experts.0.gate_proj.lora_A.weight']=next(iter(self.header.values()))
        write_header(self.adapter,self.header,self.size)
        with self.assertRaisesRegex(ValueError,'expert'):G.validate_attention_header(self.adapter,CONFIG)
    def test_header_shape_dtype_offsets_rejected(self):
        key=next(iter(self.header))
        for field,value in [('shape',[2,99]),('dtype','I8'),('data_offsets',[0,99999])]:
            header=deepcopy(self.header);header[key][field]=value;write_header(self.adapter,header,self.size)
            with self.subTest(field=field),self.assertRaises(ValueError):G.validate_attention_header(self.adapter,CONFIG)
    def test_header_rejects_overlap_gaps_and_unclaimed_payload(self):
        for kind in ('overlap', 'gap', 'trailing'):
            header=deepcopy(self.header);size=self.size
            if kind=='overlap':
                first,second=list(header)[:2]
                header[second]['data_offsets']=header[first]['data_offsets'][:]
            elif kind=='gap':
                row=header[next(reversed(header))]
                row['data_offsets']=[value+2 for value in row['data_offsets']]
                size+=2
            else:
                size+=2
            write_header(self.adapter,header,size)
            with self.subTest(kind=kind),self.assertRaisesRegex(ValueError,'overlap|gap|trailing'):
                G.validate_attention_header(self.adapter,CONFIG)
    def test_header_rejects_noncanonical_layer_alias(self):
        key=next(iter(self.header));row=self.header.pop(key)
        self.header[key.replace('layers.0.','layers.00.')]=row
        write_header(self.adapter,self.header,self.size)
        with self.assertRaisesRegex(ValueError,'Unexpected tensor'):
            G.validate_attention_header(self.adapter,CONFIG)

    def test_metadata_and_duplicate_json_rejected(self):
        with self.assertRaises(ValueError):G.strict_json('{"r":2,"r":3}')
        with self.assertRaises(ValueError):G.strict_json('{"r":NaN}')
        for bad in [dict(CONFIG,num_experts=128),dict(CONFIG,hidden_size=9),dict(CONFIG,quantization_config={'x':1})]:
            with self.subTest(bad=bad),self.assertRaises(ValueError):G.validate_model_config(bad)
        config=json.loads((self.adapter/'adapter_config.json').read_text());config['target_modules']=['q_proj','gate_proj']
        (self.adapter/'adapter_config.json').write_text(json.dumps(config))
        with self.assertRaises(ValueError):G.validate_attention_header(self.adapter,CONFIG)
    def test_attention_filter_preserves_actual_experts(self):
        model=fake_model();obs=[];before=[(name,id(m),id(m.quant_method))for name,m in model.rows if isinstance(m,FusedMoE)]
        result=G.make_attention_filter(stock,targets=G.ATTENTION_TARGETS,LinearBase=LinearBase,FusedMoE=FusedMoE,observations=obs)(model)
        self.assertEqual(result,['o_proj','qkv_proj']);self.assertEqual(obs[0]['expert_bindings'],before)
        self.assertEqual(obs[0]['excluded_expert_blocks'],2)
    def test_attention_filter_rejects_wrong_kernel_and_collision(self):
        model=fake_model();next(m for name,m in model.rows if isinstance(m,FusedMoE)).quant_method.use_marlin=False
        filter=G.make_attention_filter(stock,targets=G.ATTENTION_TARGETS,LinearBase=LinearBase,FusedMoE=FusedMoE,observations=[])
        with self.assertRaises(ValueError):filter(model)
        model=fake_model();model.rows.append(('model.mlp.qkv_proj',model.rows[0][1]))
        with self.assertRaises(ValueError):filter(model)
    def test_binding_is_fresh_and_restores_env_on_error(self):
        audit=self.root/'validation';before=dict(os.environ)
        with self.assertRaisesRegex(RuntimeError,'synthetic'):
            with G.worker_binding(adapter=str(self.adapter),model_config=CONFIG,audit_dir=audit):
                payload=json.loads(os.environ['DCC_FP8_ATTENTION_WORKER_BINDING'])
                self.assertEqual(payload['quantization'],'fp8');self.assertEqual(payload['runtime_code_identity'],runtime_identity())
                self.assertTrue(os.environ['PYTHONPATH'].startswith(str(SOURCE/'slime')))
                raise RuntimeError('synthetic')
        self.assertEqual(dict(os.environ),before)
        with self.assertRaises(FileExistsError):
            with G.worker_binding(adapter=str(self.adapter),model_config=CONFIG,audit_dir=audit):pass
    def worker_case(self,adapter,fail=False):
        model=fake_model();models=types.ModuleType('vllm.lora.models');models.get_supported_lora_modules=stock
        utils=types.ModuleType('vllm.lora.utils');utils.get_supported_lora_modules=stock
        class Worker:
            def load_model(self):
                if self.lora_config is not None:
                    self.actual_supported=models.get_supported_lora_modules(model)
                if fail:raise RuntimeError('synthetic model load failed')
                self.model_runner=types.SimpleNamespace(model=model)
        names=['vllm','vllm.v1','vllm.v1.worker','vllm.v1.worker.gpu_worker',
               'vllm.lora','vllm.model_executor','vllm.model_executor.layers',
               'vllm.model_executor.layers.linear','vllm.model_executor.layers.fused_moe']
        mods={name:types.ModuleType(name)for name in names}
        mods['vllm'].__version__='0.11.2';mods['vllm.v1.worker.gpu_worker'].Worker=Worker
        mods['vllm.lora'].models=models;mods['vllm.lora'].utils=utils
        mods['vllm.lora.models']=models;mods['vllm.lora.utils']=utils
        mods['vllm.model_executor.layers.linear'].LinearBase=LinearBase
        mods['vllm.model_executor.layers.fused_moe'].FusedMoE=FusedMoE
        torch=types.ModuleType('torch');torch.bfloat16='synthetic_bfloat16'
        torch.cuda=types.SimpleNamespace(get_device_capability=lambda device:(8,0));mods['torch']=torch
        spec=importlib.util.spec_from_file_location('slime_plugins.dcc._synthetic_worker',PKG/'fp8_attention_worker.py')
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,mods):
            spec.loader.exec_module(module);worker=module.FP8AttentionWorker()
            worker.device=0;worker.model_config=types.SimpleNamespace(quantization='fp8',dtype=torch.bfloat16,hf_config=types.SimpleNamespace(to_dict=lambda:CONFIG))
            worker.parallel_config=types.SimpleNamespace(tensor_parallel_size=1,pipeline_parallel_size=1,enable_expert_parallel=False)
            worker.lora_config=None if adapter is None else object()
            quant=type('Fp8Config',(),{})();quant.is_checkpoint_fp8_serialized=False
            quant.activation_scheme='dynamic';quant.weight_block_size=None;quant.ignored_layers=[]
            worker.vllm_config=types.SimpleNamespace(quant_config=quant)
            worker.model_runner=types.SimpleNamespace(vllm_config=worker.vllm_config)
            audit=self.root/('base-validation'if adapter is None else 'lora-validation')
            with G.worker_binding(adapter=adapter,model_config=CONFIG,audit_dir=audit):
                if fail:
                    with self.assertRaisesRegex(RuntimeError,'synthetic model load failed'):worker.load_model()
                else:worker.load_model()
            self.assertIs(models.get_supported_lora_modules,stock)
            if fail:
                self.assertFalse((audit/'worker-validation.actual.json').exists());return
            record=G.validate_worker_receipts(audit,adapter=adapter)
            self.assertEqual(record['worker']['expert_lora_wrappers'],0)
            self.assertEqual(record['worker']['attention_filter_invocations'],int(adapter is not None))
    def test_spawned_worker_hook_adapter_lifecycle(self):self.worker_case(str(self.adapter))
    def test_spawned_worker_base_has_no_lora(self):self.worker_case(None)
    def test_spawned_worker_restores_filter_on_load_failure(self):self.worker_case(str(self.adapter),fail=True)
    def test_supported_router_exclusion_exact_config_and_actual_mixed_inventory(self):
        quant=type('Fp8Config',(),{})();quant.is_checkpoint_fp8_serialized=False
        quant.activation_scheme='dynamic';quant.weight_block_size=None;quant.ignored_layers=[]
        expected=['model.layers.0.mlp.gate','model.layers.1.mlp.gate']
        self.assertEqual(G.configure_fp8_router_exclusions(quant,CONFIG),expected)
        self.assertEqual(quant.ignored_layers,expected)
        actual=G.validate_mixed_model(fake_model(),CONFIG,LinearBase=LinearBase,FusedMoE=FusedMoE,bfloat16='synthetic_bfloat16')
        self.assertEqual(actual['router_modules'],expected);self.assertEqual(actual['actual_fp8_attention_modules'],4)
        with self.assertRaises(ValueError):G.configure_fp8_router_exclusions(quant,CONFIG)
    def test_mixed_inventory_rejects_quantized_router_wrong_dtype_or_missing_attention(self):
        for kind in ('quantized_router','wrong_dtype','missing_attention','wrong_experts'):
            model=fake_model()
            if kind=='quantized_router':model.rows[-1][1].quant_method=Fp8LinearMethod()
            elif kind=='wrong_dtype':model.rows[-1][1].weight.dtype='float16'
            elif kind=='missing_attention':model.rows.pop(0)
            else:next(m for n,m in model.rows if isinstance(m,FusedMoE)).global_num_experts=128
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                G.validate_mixed_model(model,CONFIG,LinearBase=LinearBase,FusedMoE=FusedMoE,bfloat16='synthetic_bfloat16')


# Loader checks use a synthetic engine; they do not establish GPU success.
import pytest


@pytest.fixture
def fp8_loader(monkeypatch):
    import torch
    import transformers
    from slime_plugins.dcc import vllm_policy
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: False)
    for key in ('WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'DCC_FP8_ATTENTION_WORKER_BINDING'):
        monkeypatch.delenv(key, raising=False)
    config=types.SimpleNamespace(model_type='qwen3_moe',to_dict=lambda:dict(CONFIG))
    monkeypatch.setattr(transformers.AutoConfig,'from_pretrained',lambda *args,**kwargs:config)
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *args,**kwargs:
                        types.SimpleNamespace(pad_token_id=0,eos_token_id=1))
    monkeypatch.setattr(transformers.GenerationConfig,'from_pretrained',lambda *args,**kwargs:
                        types.SimpleNamespace(eos_token_id=[1,2]))
    calls=[]
    def engine(**kwargs):
        binding=json.loads(os.environ['DCC_FP8_ATTENTION_WORKER_BINDING'])
        assert (Path(binding['audit_dir'])/'header-validation.actual.json').is_file()
        calls.append((kwargs,binding))
        return types.SimpleNamespace(kwargs=kwargs)
    monkeypatch.setitem(sys.modules,'vllm',types.SimpleNamespace(__version__='0.11.2',LLM=engine))
    monkeypatch.setitem(sys.modules,'vllm.lora',types.SimpleNamespace())
    monkeypatch.setitem(sys.modules,'vllm.lora.request',types.SimpleNamespace(LoRARequest=lambda *args:args))
    return vllm_policy.load_vllm_policy,calls


def test_explicit_fp8_loader_validates_header_and_binds_only_its_worker(fp8_loader,tmp_path):
    load,calls=fp8_loader
    adapter=tmp_path/'adapter';fixture(adapter)
    audit=tmp_path/'validation';before=dict(os.environ)
    policy=load('synthetic-qwen3',adapter=str(adapter),quantization='fp8',fp8_validation_dir=str(audit))
    kwargs,binding=calls[0]
    assert kwargs['quantization']=='fp8' and kwargs['worker_cls']==G.WORKER_CLASS
    assert kwargs['enable_lora'] and binding['adapter']==str(adapter)
    assert policy.provenance['fp8_contract']==G.QUANTIZATION
    assert policy.provenance['runtime_code_identity']==runtime_identity()
    assert policy.provenance['quantization']=='fp8'
    assert dict(os.environ)==before
    assert 'cuad-structural' not in json.dumps(policy.provenance['runtime_code_identity'])


def test_fp8_base_loader_has_no_adapter_and_requires_fresh_audit(fp8_loader,tmp_path):
    load,calls=fp8_loader
    audit=tmp_path/'validation'
    policy=load('synthetic-qwen3',quantization='fp8',fp8_validation_dir=str(audit))
    assert not calls[0][0]['enable_lora'] and calls[0][1]['adapter'] is None
    assert policy.provenance['adapter'] is None
    with pytest.raises(FileExistsError):load('synthetic-qwen3',quantization='fp8',fp8_validation_dir=str(audit))
    with pytest.raises(ValueError,match='validation directory'):load('synthetic-qwen3',quantization='fp8')
    assert len(calls)==1


def test_fp8_bad_header_or_parallelism_never_creates_an_engine(fp8_loader,tmp_path):
    load,calls=fp8_loader
    adapter=tmp_path/'adapter';header,size=fixture(adapter)
    del header[next(iter(header))];write_header(adapter,header,size)
    with pytest.raises(ValueError,match='incomplete'):
        load('synthetic-qwen3',adapter=str(adapter),quantization='fp8',fp8_validation_dir=str(tmp_path/'invalid'))
    with pytest.raises(ValueError,match='tensor_parallel_size=1'):
        load('synthetic-qwen3',quantization='fp8',tensor_parallel_size=2)
    with pytest.raises(ValueError,match='supports quantization'):
        load('synthetic-qwen3',quantization='nf4')
    assert not calls and not (tmp_path/'invalid').exists()

@pytest.mark.parametrize('validation_fails', [False, True])
def test_fp8_evaluator_checks_receipts_before_generation_and_always_closes(monkeypatch,tmp_path,validation_fails):
    import asyncio
    import runpy
    from slime_plugins.dcc import PolicyCompletion,Task
    namespace=runpy.run_path(str(SOURCE/'scripts/dcc_model_evaluate.py'))
    evaluate=namespace['evaluate']
    task=Task('synthetic-contract','cuad','validation','Find clause',document='A contract',gold=[])
    monkeypatch.setitem(evaluate.__globals__,'load_tasks',lambda unused:[task])
    for key,value in [('WORLD_SIZE','1'),('LOCAL_RANK','0'),('RANK','0')]:monkeypatch.setenv(key,value)
    calls=[]
    class Policy:
        provenance=dict(backend='vllm',quantization='fp8',fp8_contract=G.QUANTIZATION,
                        fp8_validation=G.VALIDATION_LAYOUT,runtime_code_identity=runtime_identity())
        async def __call__(self,messages,**kwargs):
            calls.append('generate')
            return PolicyCompletion('{"answer":[]}',[1],[2])
        async def close(self):calls.append('close')
    def load(unused,**kwargs):
        assert kwargs['quantization']=='fp8'
        assert kwargs['fp8_validation_dir']==str(tmp_path/'fp8_validation_rank0')
        calls.append('load');return Policy()
    def validate(directory,*,adapter):
        assert directory==tmp_path/'fp8_validation_rank0' and adapter is None
        calls.append('validate')
        if validation_fails:raise ValueError('synthetic incompatible worker receipts')
        return {'synthetic_only':True}
    monkeypatch.setitem(sys.modules,'slime_plugins.dcc.vllm_policy',types.SimpleNamespace(load_vllm_policy=load))
    monkeypatch.setattr(G,'validate_worker_receipts',validate)
    args=types.SimpleNamespace(model='synthetic/model',adapter=None,data=['synthetic.jsonl'],
        max_steps=3,max_new_tokens=1024,max_prompt_tokens=8192,initial_chunks=4,seed=42,limit=0,
        output_dir=str(tmp_path),retry_infrastructure=False,inference_backend='vllm',quantization='fp8',
        vllm_tensor_parallel_size=1,vllm_memory_utilization=.88,vllm_batched_tokens=8192,
        vllm_enforce_eager=True,batch_size=1,timeout=30,save_tokens=False)
    if validation_fails:
        with pytest.raises(ValueError,match='synthetic incompatible worker receipts'):asyncio.run(evaluate(args))
        assert calls==['load','validate','close']
        assert not (tmp_path/'rank0.done').exists()
        assert not (tmp_path/'predictions_rank0.jsonl').exists()
    else:
        asyncio.run(evaluate(args))
        assert calls==['load','validate','generate','close']
        manifest=json.loads((tmp_path/'manifest_rank0.json').read_text())
        runtime=json.loads((tmp_path/'inference_runtime.json').read_text())
        assert manifest['quantization']==runtime['quantization']=='fp8'
        assert manifest['runtime_code_identity']==runtime['runtime_code_identity']==runtime_identity()
        scorer=namespace['score']
        monkeypatch.setitem(scorer.__globals__,'load_tasks',lambda unused:[task])
        score_args=types.SimpleNamespace(**vars(args),merge_judge=False,local_judge=False,model_name='synthetic/model')
        scorer(score_args)
        # This is a synthetic one-task fixture, not an actual benchmark score.
        assert json.loads((tmp_path/'scores_strict.json').read_text())['policy_quantization']=='fp8'
