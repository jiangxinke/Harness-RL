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
    'schema': 'dcc-bf16-attention-runtime-v2',
    'implementation': 'qwen3-moe-bf16-native-experts-base-or-qvo-lora-v2',
    'adapter_modes': ['base without LoRA', 'standard q/v/o LoRA'],
    'quantization': 'none', 'dtype': 'bfloat16',
    'target_modules': sorted(TARGETS), 'expert_lora': False,
    'worker_extension': WORKER_CLASS, 'vllm_version': '0.11.2',
    'tensor_parallel_size': 1,
}


def runtime_identity():
    return deepcopy(IDENTITY)


HEADER_SCHEMA = 'dcc-bf16-attention-header-actual-validation-v2'
BINDING_SCHEMA = 'dcc-bf16-attention-worker-binding-v2'
WORKER_SCHEMA = 'dcc-bf16-attention-worker-actual-validation-v2'


def model_validation(model_config):
    hidden, layers, heads, kv_heads, dimension = validate_model_config(model_config)
    require(model_config.get('decoder_sparse_step', 1) == 1
            and not model_config.get('mlp_only_layers'),
            'One unquantized MoE block per decoder layer is required')
    return dict(model_type='qwen3_moe', num_experts=125, hidden_size=hidden,
                num_hidden_layers=layers, num_attention_heads=heads,
                num_key_value_heads=kv_heads, head_dim=dimension, quantization='none')


def validate_base_model(model, model_config, *, LinearBase, FusedMoE, BaseLayerWithLoRA,
                        UnquantizedLinearMethod, UnquantizedFusedMoEMethod):
    """Inspect the real base inventory without installing a LoRA manager/filter."""
    layers = model_validation(model_config)['num_hidden_layers']
    expected_attention = {f'model.layers.{i}.self_attn.{leaf}'
                          for i in range(layers) for leaf in ('qkv_proj', 'o_proj')}
    expected_experts = {f'model.layers.{i}.mlp.experts' for i in range(layers)}
    attention, experts = [], []
    require(getattr(model, 'lora_manager', None) is None, 'Base worker must not have a LoRA manager')
    for name, module in model.named_modules():
        require(not isinstance(module, BaseLayerWithLoRA)
                and type(module).__name__ != 'FusedMoEWithLoRA',
                'Base worker must not contain any LoRA wrapper')
        if isinstance(module, FusedMoE):
            require(name in expected_experts
                    and type(module.quant_method) is UnquantizedFusedMoEMethod
                    and module.global_num_experts == module.local_num_experts == 125,
                    'Base experts must use the original unquantized method and expert count')
            experts.append(name)
        if name.rsplit('.', 1)[-1] in {'qkv_proj', 'o_proj'}:
            require(name in expected_attention and isinstance(module, LinearBase)
                    and type(module.quant_method) is UnquantizedLinearMethod,
                    'Base attention must use native unquantized linears')
            attention.append(name)
    require(set(experts) == expected_experts and len(experts) == len(expected_experts)
            and set(attention) == expected_attention and len(attention) == len(expected_attention),
            'Base attention/expert inventory differs from actual model configuration')
    return dict(native_attention_modules=sorted(attention), expert_module_names=sorted(experts))


def validate_attention_header(adapter, model_config):
    model_validation(model_config)
    require(adapter is not None, 'Adapter header validation requires a nonempty adapter')
    header = _validate_header(adapter, model_config)
    require(set(header['target_modules']) == TARGETS,
            'BF16 attention-only worker requires exactly q_proj/v_proj/o_proj targets')
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
    model = model_validation(model_config)
    header = None if adapter is None else validate_attention_header(adapter, model_config)
    path.mkdir(parents=False, exist_ok=False)
    adapter = None if adapter is None else str(Path(adapter).absolute())
    record = dict(schema=HEADER_SCHEMA,
        runtime_code_identity=runtime_identity(), adapter=adapter, header_validation=header,
        model_validation=model, lora_enabled=adapter is not None,
        validated_before_engine_creation=True, tensor_payload_bytes_read=0,
        diagnostic_only=True, benchmark_gain_claimed=False, created_unix=time.time())
    with (path / VALIDATION_LAYOUT['header_validation_receipt']).open('x') as stream:
        json.dump(record, stream, indent=2); stream.write('\n')
    payload = dict(schema=BINDING_SCHEMA, adapter=adapter,
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
    expected_adapter = None if adapter is None else str(Path(adapter).absolute())
    enabled = adapter is not None
    for row, schema in ((header, HEADER_SCHEMA), (worker, WORKER_SCHEMA)):
        require(row.get('schema') == schema and row.get('runtime_code_identity') == runtime_identity()
                and row.get('adapter') == expected_adapter and row.get('lora_enabled') is enabled
                and row.get('diagnostic_only') is True and row.get('benchmark_gain_claimed') is False,
                'Actual BF16 worker identity differs')
    require(header['model_validation'] == worker['model_validation']
            and worker['model_validated_before_model_load'] is True
            and header['validated_before_engine_creation'] is True
            and header['tensor_payload_bytes_read'] == 0
            and worker['tensor_payload_bytes_read_by_header_guard'] == 0
            and header['header_validation'] == worker['header_validation']
            and worker['header_validated_before_model_load'] is enabled
            and worker['attention_filter_invocations'] == int(enabled)
            and worker['original_filter_restored'] is True
            and worker['expert_lora_wrappers'] == 0
            and worker['all_expert_methods'] == 'UnquantizedFusedMoEMethod'
            and worker['actual_global_num_experts'] == worker['actual_local_num_experts'] == 125,
            'Actual BF16 attention-only wrapper validation missing')
    layers = header['model_validation']['num_hidden_layers']
    require(type(layers) is int and layers > 0, 'Invalid actual model layer count')
    expected_attention = sorted(f'model.layers.{i}.self_attn.{leaf}'
                                for i in range(layers) for leaf in ('qkv_proj', 'o_proj'))
    expected_experts = sorted(f'model.layers.{i}.mlp.experts' for i in range(layers))
    require(worker['actual_expert_blocks'] == layers
            and worker['actual_expert_module_names'] == expected_experts
            and worker['actual_native_attention_modules'] == expected_attention,
            'Actual native BF16 module inventory is incomplete')
    if enabled:
        actual = worker['header_validation']
        require(actual is not None and set(actual['target_modules']) == TARGETS
                and actual['actual_header_validation'] is True and actual['expert_lora'] is False
                and actual['tensor_payload_bytes_read'] == 0
                and worker['selected_attention_modules'] == expected_attention
                and worker['actual_attention_lora_wrappers'] == expected_attention
                and worker['actual_lora_manager_present'] is True,
                'Actual q/v/o adapter validation is incomplete')
    else:
        require(worker['header_validation'] is None
                and worker['selected_attention_modules'] == []
                and worker['actual_attention_lora_wrappers'] == []
                and worker['actual_lora_manager_present'] is False,
                'Base worker must have no adapter, module filter, wrappers or LoRA manager')
    return {'header': header, 'worker': worker}
