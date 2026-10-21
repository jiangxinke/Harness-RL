"""Opt-in BF16 attention LoRA guards; importing this module never loads a GPU."""
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import time

from .fp8_attention_guard import require, strict_json, validate_model_config
from .fp8_attention_guard import validate_attention_header as _validate_header

TARGETS = frozenset(('q_proj', 'v_proj', 'o_proj'))
BINDING_KEY = 'DCC_BF16_ATTENTION_WORKER_BINDING'
WORKER_CLASS = 'slime_plugins.dcc.bf16_attention_worker.BF16AttentionWorker'
VALIDATION_LAYOUT = {
    'header_validation_receipt': 'header-validation.actual.json',
    'worker_validation_receipt': 'worker-validation.actual.json',
}
IDENTITY = {
    'schema': 'dcc-bf16-attention-runtime-v1',
    'implementation': 'qwen3-moe-bf16-qvo-lora-native-experts-v1',
    'quantization': 'none', 'dtype': 'bfloat16',
    'target_modules': sorted(TARGETS), 'expert_lora': False,
    'worker_extension': WORKER_CLASS, 'vllm_version': '0.11.2',
    'tensor_parallel_size': 1,
}


def runtime_identity():
    return deepcopy(IDENTITY)


def validate_attention_header(adapter, model_config):
    require(adapter is not None, 'Explicit BF16 attention-only LoRA requires an adapter')
    header = _validate_header(adapter, model_config)
    require(set(header['target_modules']) == TARGETS,
            'BF16 attention-only worker requires exactly q_proj/v_proj/o_proj targets')
    require(model_config.get('decoder_sparse_step', 1) == 1
            and not model_config.get('mlp_only_layers'),
            'One unquantized MoE block per decoder layer is required')
    return header


def make_attention_filter(original, *, model_config, targets, LinearBase, FusedMoE,
                          UnquantizedLinearMethod, UnquantizedFusedMoEMethod, observations):
    """Filter before vLLM constructs wrappers; leave the native expert method intact."""
    _, layers, _, _, _ = validate_model_config(model_config)
    require(set(targets) == TARGETS, 'Only q_proj/v_proj/o_proj targets are supported')
    leaves = {'qkv_proj', 'o_proj'}
    expected = {f'model.layers.{i}.self_attn.{leaf}' for i in range(layers) for leaf in leaves}
    expected_experts = {f'model.layers.{i}.mlp.experts' for i in range(layers)}

    def attention_only(model):
        require(not observations, 'Attention filter must run exactly once per fresh worker')
        require(leaves <= set(original(model)), 'Actual model lacks requested attention projections')
        selected, experts = [], []
        for name, module in model.named_modules():
            if isinstance(module, FusedMoE):
                require(name in expected_experts
                        and type(module.quant_method) is UnquantizedFusedMoEMethod,
                        'Every expert block must retain its native unquantized method')
                require(module.global_num_experts == module.local_num_experts == 125,
                        'Actual expert count changed')
                experts.append((name, id(module), id(module.quant_method)))
            if name.rsplit('.', 1)[-1] in leaves:
                require(name in expected and isinstance(module, LinearBase)
                        and type(module.quant_method) is UnquantizedLinearMethod,
                        'Selected module must be an unquantized attention linear')
                selected.append((name, id(module), id(module.quant_method)))
        require({row[0] for row in selected} == expected and len(selected) == len(expected)
                and {row[0] for row in experts} == expected_experts
                and len(experts) == len(expected_experts),
                'Actual attention/expert layer inventory differs from the model configuration')
        observations.append(dict(selected_bindings=selected, expert_bindings=experts))
        return sorted(leaves)

    return attention_only


@contextmanager
def worker_binding(*, adapter, model_config, audit_dir):
    require(os.environ.get(BINDING_KEY) is None, 'A prior BF16 worker binding is forbidden')
    require(audit_dir is not None, 'An explicit fresh BF16 validation directory is required')
    path = Path(audit_dir)
    require(path.is_absolute() and '..' not in path.parts, 'Absolute BF16 validation directory required')
    header = validate_attention_header(adapter, model_config)
    path.mkdir(parents=False, exist_ok=False)
    adapter = str(Path(adapter).absolute())
    record = dict(schema='dcc-bf16-attention-header-actual-validation-v1',
        runtime_code_identity=runtime_identity(), adapter=adapter, header_validation=header,
        validated_before_engine_creation=True, tensor_payload_bytes_read=0,
        diagnostic_only=True, benchmark_gain_claimed=False, created_unix=time.time())
    with (path / VALIDATION_LAYOUT['header_validation_receipt']).open('x') as stream:
        json.dump(record, stream, indent=2); stream.write('\n')
    payload = dict(schema='dcc-bf16-attention-worker-binding-v1', adapter=adapter,
                   audit_dir=str(path), runtime_code_identity=runtime_identity())
    source_slime = str(Path(__file__).resolve().parents[2])
    saved = {key: os.environ.get(key) for key in (BINDING_KEY, 'PYTHONPATH')}
    try:
        os.environ[BINDING_KEY] = json.dumps(payload)
        os.environ['PYTHONPATH'] = source_slime + (os.pathsep + saved['PYTHONPATH'] if saved['PYTHONPATH'] else '')
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def validate_worker_receipts(audit_dir, *, adapter):
    path = Path(audit_dir)
    header = strict_json((path / VALIDATION_LAYOUT['header_validation_receipt']).read_text())
    worker = strict_json((path / VALIDATION_LAYOUT['worker_validation_receipt']).read_text())
    for row, schema in ((header, 'dcc-bf16-attention-header-actual-validation-v1'),
                        (worker, 'dcc-bf16-attention-worker-actual-validation-v1')):
        require(row.get('schema') == schema and row.get('runtime_code_identity') == runtime_identity()
                and row.get('adapter') == str(Path(adapter).absolute())
                and row.get('diagnostic_only') is True and row.get('benchmark_gain_claimed') is False,
                'Actual BF16 worker identity differs')
    require(header['validated_before_engine_creation'] is True
            and header['tensor_payload_bytes_read'] == 0
            and worker['tensor_payload_bytes_read_by_header_guard'] == 0
            and header['header_validation'] == worker['header_validation']
            and worker['header_validated_before_model_load'] is True
            and worker['attention_filter_invocations'] == 1
            and worker['original_filter_restored'] is True
            and worker['expert_lora_wrappers'] == 0
            and worker['all_expert_methods'] == 'UnquantizedFusedMoEMethod'
            and worker['actual_global_num_experts'] == worker['actual_local_num_experts'] == 125
            and worker['actual_expert_blocks'] > 0
            and len(worker['selected_attention_modules']) == 2 * worker['actual_expert_blocks']
            and worker['selected_attention_modules'] == worker['actual_attention_lora_wrappers'],
            'Actual BF16 attention-only wrapper validation missing')
    return {'header': header, 'worker': worker}
