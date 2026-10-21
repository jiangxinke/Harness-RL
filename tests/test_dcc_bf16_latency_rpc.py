"""Named worker RPC transport/reporting controls; CPU only, no engine launches."""
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest
from slime_plugins.dcc import bf16_attention_guard as G
from test_dcc_bf16_attention import CONFIG, fixture as adapter_fixture, loader

SOURCE = Path(__file__).resolve().parents[1]


def latency_module():
    spec = importlib.util.spec_from_file_location('latency_named_rpc_control',SOURCE/'scripts/dcc_bf16_lora_latency.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_vllm_msgpack_named_rpc_and_result_roundtrip_reject_callback(monkeypatch):
    vllm = pytest.importorskip('vllm')
    if vllm.__version__ != '0.11.2':
        pytest.skip('Transport control targets the pinned production vLLM0.11.2')
    from vllm import envs
    from vllm.v1.serial_utils import MsgpackEncoder, MsgpackDecoder, UtilityResult
    from vllm.v1.engine import EngineCoreOutputs, UtilityOutput
    import torch
    monkeypatch.delenv('VLLM_ALLOW_INSECURE_SERIALIZATION',raising=False)
    assert envs.VLLM_ALLOW_INSECURE_SERIALIZATION is False
    encoder = MsgpackEncoder()
    # Exact SyncMPClient.call_utility envelope for LLM.collective_rpc(name).
    request = (0,71,'collective_rpc',(G.LATENCY_INVENTORY_RPC,None,(),None))
    encoded = encoder.encode(request)
    assert len(encoded) == 1
    assert MsgpackDecoder().decode(encoded) == [0,71,'collective_rpc',[G.LATENCY_INVENTORY_RPC,None,[],None]]
    inventory = [dict(lora_wrappers=[],expert_methods=[dict(name='model.layers.0.mlp.experts',method='UnquantizedFusedMoEMethod')],
        expert_lora_wrappers=0,expert_adapter_enabled=[],expert_lora_buffers=[],expert_lora_buffer_bytes=0,
        worker_cuda_allocated_bytes=0,worker_cuda_reserved_bytes=0,worker_cuda_peak_allocated_bytes=0,
        allocator_metrics_are_not_total_device_memory=True)]
    output = EngineCoreOutputs(utility_output=UtilityOutput(call_id=71,result=UtilityResult(inventory)))
    decoded = MsgpackDecoder(EngineCoreOutputs).decode(encoder.encode(output))
    assert decoded.utility_output.call_id == 71
    assert decoded.utility_output.result.result == inventory
    def callback(model):
        return None
    old_request = (0,71,'collective_rpc',('apply_model',None,(callback,),None))
    with pytest.raises(TypeError,match='not serializable'):
        encoder.encode(old_request)
    assert envs.VLLM_ALLOW_INSECURE_SERIALIZATION is False
    assert not torch.cuda.is_initialized()


def test_extension_inspects_real_cpu_tensor_buffers_without_callback_transport(monkeypatch):
    import torch
    class BaseLayerWithLoRA:
        pass
    class FusedMoE:
        pass
    class FusedMoEWithLoRA(BaseLayerWithLoRA,FusedMoE):
        pass
    expert = FusedMoEWithLoRA()
    expert.quant_method = type('UnquantizedFusedMoEMethod',(),{})()
    expert.adapter_enabled = torch.tensor([0,1],dtype=torch.int32)
    for name in ('w1_lora_a_stacked','w1_lora_b_stacked','w2_lora_a_stacked',
                 'w2_lora_b_stacked','w3_lora_a_stacked','w3_lora_b_stacked'):
        setattr(expert,name,torch.ones(2,3,dtype=torch.bfloat16))
    monkeypatch.setitem(sys.modules,'vllm.v1.worker.gpu_worker',types.SimpleNamespace(Worker=type('Worker',(),{})))
    monkeypatch.setitem(sys.modules,'vllm.lora.layers.base',types.SimpleNamespace(BaseLayerWithLoRA=BaseLayerWithLoRA))
    monkeypatch.setitem(sys.modules,'vllm.model_executor.layers.fused_moe',types.SimpleNamespace(FusedMoE=FusedMoE))
    spec = importlib.util.spec_from_file_location('slime_plugins.dcc._latency_extension_cpu',
        SOURCE/'slime/slime_plugins/dcc/bf16_attention_worker.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    worker = module.BF16LatencyWorkerExtension()
    worker.model_runner = types.SimpleNamespace(model=types.SimpleNamespace(named_modules=lambda:iter([('experts',expert)])))
    inventory = getattr(worker,G.LATENCY_INVENTORY_RPC)()
    assert inventory['expert_lora_wrappers'] == 1
    assert inventory['expert_adapter_enabled'] == [dict(name='experts',values=[0,1])]
    assert len(inventory['expert_lora_buffers']) == 6 and inventory['expert_lora_buffer_bytes'] == 72
    assert inventory['expert_methods'] == [dict(name='experts',method='UnquantizedFusedMoEMethod')]
    assert json.loads(json.dumps(inventory)) == inventory
    assert not torch.cuda.is_initialized()


def test_loader_inventory_is_explicit_and_same_extension_for_stock_filtered_base(loader,tmp_path):
    load,calls = loader
    adapter = adapter_fixture(tmp_path/'adapter')
    default = load('synthetic',adapter=adapter)
    explicit_false = load('synthetic',adapter=adapter,bf16_latency_inventory=False)
    assert calls[0] == calls[1] and default.provenance == explicit_false.provenance
    stock = load('synthetic',adapter=adapter,bf16_latency_inventory=True)
    filtered = load('synthetic',adapter=adapter,bf16_latency_inventory=True,
                    bf16_attention_only_lora=True,bf16_lora_validation_dir=tmp_path/'filtered-audit')
    base = load('synthetic',bf16_latency_inventory=True)
    assert 'worker_extension_cls' not in calls[0][0]
    assert {key:value for key,value in calls[2][0].items() if key!='worker_extension_cls'} == calls[0][0]
    for kwargs,binding in calls[2:]:
        assert kwargs['worker_extension_cls'] == G.LATENCY_WORKER_EXTENSION
    assert 'worker_cls' not in calls[2][0] and calls[3][0]['worker_cls'] == G.WORKER_CLASS
    assert calls[2][0]['enable_lora'] is True and calls[4][0]['enable_lora'] is False
    for policy in (stock,filtered,base):
        assert policy.provenance['bf16_latency_inventory'] == dict(worker_extension=G.LATENCY_WORKER_EXTENSION,
                                                                  rpc=G.LATENCY_INVENTORY_RPC)


@pytest.mark.parametrize('updates',[dict(bf16_latency_inventory=1),
    dict(bf16_latency_inventory=True,quantization='fp8'),dict(bf16_latency_inventory=True,tensor_parallel_size=2)])
def test_invalid_diagnostic_optin_rejected_before_engine(loader,updates):
    load,calls = loader
    with pytest.raises((TypeError,ValueError),match='BF16 latency|bf16_latency'):
        load('synthetic',**updates)
    assert not calls


@pytest.mark.parametrize('mode,count,worker_count,success',[('stock',1,1,True),('filtered',0,1,True),('base',0,1,True),
    ('stock',0,1,False),('filtered',1,1,False),('stock',1,0,False),('filtered',0,2,False)])
def test_actual_report_named_rpc_keeps_mode_inventory_guards(monkeypatch,tmp_path,mode,count,worker_count,success):
    import transformers
    from slime_plugins.dcc import vllm_policy
    module = latency_module()
    adapter = adapter_fixture(tmp_path/'adapter')
    monkeypatch.setattr(transformers.AutoConfig,'from_pretrained',lambda *a,**kw:types.SimpleNamespace(to_dict=lambda:CONFIG))
    events = []
    class Engine:
        def reset_prefix_cache(self):
            events.append('reset')
        def generate(self,prompts,sampling,**kw):
            events.append('generate')
            return [types.SimpleNamespace(prompt_token_ids=prompt['prompt_token_ids'],
                outputs=[types.SimpleNamespace(token_ids=[21,22],finish_reason='length')]) for prompt in prompts]
        def collective_rpc(self,method):
            assert type(method) is str and method == G.LATENCY_INVENTORY_RPC
            events.append('named-rpc')
            return [dict(expert_lora_wrappers=count)]*worker_count
        def apply_model(self,*args,**kwargs):
            pytest.fail('No callable may cross the engine RPC boundary')
    class Policy:
        model = Engine()
        tokenizer = types.SimpleNamespace(encode=lambda *a,**kw:[11,12,13])
        lora_request = None
        provenance = {'synthetic':'CPU control only'}
        async def close(self):
            events.append('close')
    def load(*args,**kwargs):
        assert kwargs['bf16_latency_inventory'] is True
        assert kwargs.get('bf16_attention_only_lora',False) == (mode=='filtered')
        assert kwargs['adapter'] == (None if mode=='base' else str(adapter))
        return Policy()
    monkeypatch.setattr(vllm_policy,'load_vllm_policy',load)
    monkeypatch.setattr(G,'validate_worker_receipts',lambda *a,**kw:events.append('validate-filtered'))
    monkeypatch.setitem(sys.modules,'vllm',types.SimpleNamespace(SamplingParams=lambda **kwargs:types.SimpleNamespace(**kwargs)))
    output = tmp_path/'run'
    argv = ['--model','synthetic','--adapter',str(adapter),'--mode',mode,'--output-dir',str(output),
            '--prompt-lengths','4','--completion-tokens','2','--warmups','1','--repetitions','1',
            '--max-model-len','32','--max-batched-tokens','8']
    if success:
        module.main(argv)
        summary = json.loads((output/'summary.actual.json').read_text())
        assert summary['closed'] is True and len(summary['measured']) == 1
        assert summary['schema'] == 'dcc-bf16-lora-fixed-token-latency-v2'
        assert summary['worker_inventory_rpc'] == G.LATENCY_INVENTORY_RPC
    else:
        with pytest.raises(RuntimeError,match='worker|wrappers'):
            module.main(argv)
        assert not (output/'summary.actual.json').exists()
    assert events[-1] == 'close' and events.count('named-rpc') == 1
    assert events.count('generate') == events.count('reset') == 2
