"""Header-only FP8 attention-LoRA guards; import has no model or GPU dependency."""
from contextlib import contextmanager
import json
import os
import time
from pathlib import Path
import re
import struct

ATTENTION_TARGETS = frozenset(('q_proj', 'k_proj', 'v_proj', 'o_proj'))
QUANTIZATION = {
    'quantization': 'fp8', 'weight_format': 'FP8 E4M3 per-tensor online from BF16',
    'kernel': 'Marlin W8A16', 'activation_dtype': 'bfloat16',
    'kv_cache_dtype': 'auto (bfloat16)', 'checkpoint_was_fp8': False,
    'attention_only_lora':True,'expert_lora':False,
    'router_dtype':'bfloat16','router_quantization':'none',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


def strict_json(raw):
    return json.loads(raw, object_pairs_hook=strict_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def validate_model_config(config):
    require(config.get('model_type') == 'qwen3_moe'
            and config.get('num_experts') == 125
            and not config.get('quantization_config'),
            'Only the original unquantized 125-expert Qwen3 MoE input checkpoint is supported')
    values = [config.get(key) for key in
              ('hidden_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads')]
    require(all(type(value) is int and value > 0 for value in values), 'Invalid attention dimensions')
    hidden, layers, heads, kv_heads = values
    require(hidden % heads == 0 and heads % kv_heads == 0, 'Inconsistent attention dimensions')
    dimension = config.get('head_dim', hidden // heads)
    require(type(dimension) is int and dimension > 0, 'Invalid attention head dimension')
    return hidden, layers, heads, kv_heads, dimension


def validate_attention_header(adapter, model_config):
    """Read config plus safetensors header only, never tensor payload bytes."""
    hidden, layers, heads, kv_heads, dimension = validate_model_config(model_config)
    path = Path(adapter)
    config = strict_json((path / 'adapter_config.json').read_text())
    targets, rank = config.get('target_modules'), config.get('r')
    require(config.get('peft_type') == 'LORA' and isinstance(targets, list)
            and targets and all(isinstance(target, str) for target in targets)
            and len(targets) == len(set(targets))
            and set(targets) <= ATTENTION_TARGETS, 'Actual adapter must target attention projections only')
    require(type(rank) is int and 1 <= rank <= 64, 'Unsupported attention LoRA rank')
    require(config.get('bias', 'none') == 'none' and not config.get('modules_to_save'),
            'LoRA bias/modules_to_save are unsupported')
    unsupported = ('use_dora', 'use_rslora', 'rank_pattern', 'alpha_pattern', 'fan_in_fan_out',
                   'lora_bias', 'target_parameters', 'layer_replication', 'alora_invocation_tokens',
                   'use_qalora')
    require(not any(config.get(key) for key in unsupported), 'Only standard attention LoRA is supported')
    weights = path / 'adapter_model.safetensors'
    before = weights.stat()
    with weights.open('rb') as stream:
        raw_length = stream.read(8)
        require(len(raw_length) == 8, 'Truncated safetensors header length')
        length = struct.unpack('<Q', raw_length)[0]
        require(2 <= length <= 16 * 1024 ** 2 and 8 + length <= before.st_size,
                'Invalid safetensors header boundary')
        raw_header = stream.read(length)
        require(len(raw_header) == length, 'Truncated safetensors header')
    header = strict_json(raw_header)
    shapes = {'q_proj': (hidden, heads * dimension), 'k_proj': (hidden, kv_heads * dimension),
              'v_proj': (hidden, kv_heads * dimension), 'o_proj': (heads * dimension, hidden)}
    expected = {(layer, target, side) for layer in range(layers)
                for target in targets for side in ('A', 'B')}
    found, extents = set(), []
    for key, row in header.items():
        if key == '__metadata__':
            continue
        match = re.fullmatch(r'base_model\.model\.model\.layers\.(0|[1-9]\d*)\.self_attn\.([qkvo]_proj)\.lora_([AB])\.weight', key)
        require(match is not None, 'Unexpected tensor: expert or nonattention LoRA is forbidden')
        layer, target, side = int(match[1]), match[2], match[3]
        require((layer, target, side) in expected, 'Unexpected adapter target/layer')
        require(isinstance(row, dict) and row.get('dtype') in {'F16', 'BF16', 'F32'},
                'Unsupported adapter tensor dtype')
        inputs, outputs = shapes[target]
        shape = [rank, inputs] if side == 'A' else [outputs, rank]
        require(row.get('shape') == shape, 'Adapter header shape differs from actual model')
        offsets = row.get('data_offsets')
        require(isinstance(offsets, list) and len(offsets) == 2
                and all(type(value) is int for value in offsets)
                and 0 <= offsets[0] <= offsets[1] <= before.st_size - 8 - length,
                'Invalid tensor payload offsets')
        elements = shape[0] * shape[1]
        require(offsets[1] - offsets[0] == elements * (4 if row['dtype'] == 'F32' else 2),
                'Tensor payload byte extent differs from header shape')
        extents.append(tuple(offsets))
        found.add((layer, target, side))
    require(found == expected, 'Empty or incomplete attention LoRA header')
    cursor = 0
    for start, end in sorted(extents):
        require(start == cursor, 'Tensor payload offsets overlap or contain a gap')
        cursor = end
    require(cursor == before.st_size - 8 - length, 'Unclaimed trailing tensor payload bytes')
    after = weights.stat()
    require(before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns,
            'Actual adapter changed during header-only validation')
    leaves = sorted(({'qkv_proj'} if set(targets) & {'q_proj', 'k_proj', 'v_proj'} else set())
                    | ({'o_proj'} if 'o_proj' in targets else set()))
    return dict(target_modules=sorted(targets), selected_model_module_leaves=leaves,
                rank=rank, tensor_pairs=len(found) // 2, tensor_payload_bytes_read=0,
                actual_header_validation=True, expert_lora=False,
                weights_bytes=before.st_size, weights_mtime_ns=before.st_mtime_ns)


def make_attention_filter(original, *, targets, LinearBase, FusedMoE, observations):
    """Restrict the stock manager before it constructs any expert LoRA wrapper."""
    leaves = ({'qkv_proj'} if set(targets) & {'q_proj', 'k_proj', 'v_proj'} else set())
    if 'o_proj' in targets:
        leaves.add('o_proj')
    require(leaves and set(targets) <= ATTENTION_TARGETS, 'No valid attention-only targets')

    def attention_only(model):
        supported = set(original(model))
        require(leaves <= supported, 'Actual model lacks requested attention projections')
        selected, experts = [], []
        for name, module in model.named_modules():
            if isinstance(module, FusedMoE):
                method = module.quant_method
                require(type(method).__name__ == 'Fp8MoEMethod' and method.use_marlin is True,
                        'All expert blocks must use actual FP8-Marlin')
                require(module.global_num_experts == 125 and module.local_num_experts == 125, 'Actual expert count changed')
                experts.append((name, id(module), id(method)))
            if name.split('.')[-1] in leaves:
                require('.self_attn.' in name and isinstance(module, LinearBase)
                        and not isinstance(module, FusedMoE), 'Selected LoRA module is not an attention linear')
                require(type(module.quant_method).__name__ == 'Fp8LinearMethod'
                        and module.quant_method.use_marlin is True,
                        'Selected attention projection must use actual FP8-Marlin')
                selected.append(name)
        require(experts and selected and {name.split('.')[-1] for name in selected} == leaves,
                'Actual quantized model must expose attention and expert blocks')
        observations.append(dict(selected_module_names=selected, expert_bindings=experts,
                                 excluded_expert_blocks=len(experts), selected_module_leaves=sorted(leaves)))
        return sorted(leaves)

    return attention_only


VALIDATION_LAYOUT = {
    'directory_pattern': 'fp8_validation_rank{rank}',
    'header_validation_receipt': 'header-validation.actual.json',
    'worker_validation_receipt': 'worker-validation.actual.json',
}
WORKER_CLASS = 'slime_plugins.dcc.fp8_attention_worker.FP8AttentionWorker'


@contextmanager
def worker_binding(*, adapter, model_config, audit_dir):
    """Validate before engine creation and bind the fresh spawned worker."""
    from .fp8_runtime import runtime_identity
    validate_model_config(model_config)
    require(audit_dir is not None, 'An explicit fresh FP8 validation directory is required')
    path = Path(audit_dir)
    require(path.is_absolute() and '..' not in path.parts, 'Absolute FP8 validation directory required')
    header = validate_attention_header(adapter, model_config) if adapter is not None else None
    path.mkdir(parents=False, exist_ok=False)
    record = dict(schema='dcc-fp8-attention-header-actual-validation-v1',
        quantization='fp8', fp8_contract=QUANTIZATION, runtime_code_identity=runtime_identity(),
        adapter=None if adapter is None else str(Path(adapter).absolute()),
        header_validation=header, validated_before_engine_creation=True,
        tensor_payload_bytes_read=0, actual_model_config_validation=True,
        actual_num_experts=125, created_unix=time.time(), diagnostic_only=True,
        benchmark_gain_claimed=False)
    with (path / VALIDATION_LAYOUT['header_validation_receipt']).open('x') as stream:
        json.dump(record, stream, indent=2); stream.write('\n')
    payload = dict(schema='dcc-fp8-attention-worker-binding-v1',
        adapter=record['adapter'], audit_dir=str(path), quantization='fp8',
        runtime_code_identity=runtime_identity())
    source_slime = str(Path(__file__).resolve().parents[2])
    keys = ('DCC_FP8_ATTENTION_WORKER_BINDING', 'PYTHONPATH')
    saved = {key: os.environ.get(key) for key in keys}
    try:
        require(saved[keys[0]] is None, 'A prior FP8 worker binding is forbidden')
        os.environ[keys[0]] = json.dumps(payload)
        os.environ['PYTHONPATH'] = source_slime + (os.pathsep + saved['PYTHONPATH'] if saved['PYTHONPATH'] else '')
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def validate_worker_receipts(audit_dir, *, adapter):
    """Read fresh actual receipts after load; caller retains engine cleanup ownership."""
    from .fp8_runtime import runtime_identity
    path = Path(audit_dir)
    header = strict_json((path / VALIDATION_LAYOUT['header_validation_receipt']).read_text())
    worker = strict_json((path / VALIDATION_LAYOUT['worker_validation_receipt']).read_text())
    for row, schema in ((header, 'dcc-fp8-attention-header-actual-validation-v1'),
                        (worker, 'dcc-fp8-attention-worker-actual-validation-v1')):
        require(row.get('schema') == schema and row.get('quantization') == 'fp8'
                and row.get('runtime_code_identity') == runtime_identity()
                and row.get('fp8_contract') == QUANTIZATION, 'Actual FP8 validation identity differs')
        require(row.get('diagnostic_only') is True and row.get('benchmark_gain_claimed') is False,
                'Runtime validation must not claim a benchmark gain')
    expected_adapter = None if adapter is None else str(Path(adapter).absolute())
    require(header['adapter'] == expected_adapter and worker['adapter'] == expected_adapter
            and header['header_validation'] == worker['header_validation']
            and header['validated_before_engine_creation'] is True
            and header['tensor_payload_bytes_read'] == 0
            and worker['tensor_payload_bytes_read_by_header_guard'] == 0,
            'Actual pre-model-load header guards differ')
    require(worker['actual_num_experts'] == 125 and worker['actual_expert_blocks'] > 0
            and worker['all_expert_methods'] == 'Fp8MoEMethod(use_marlin=True)'
            and worker['expert_lora_wrappers'] == 0 and worker['original_filter_restored'] is True
            and worker['header_validated_before_model_load'] is True,
            'Actual worker did not preserve FP8-Marlin experts without expert LoRA')
    if adapter is None:
        require(worker['header_validation'] is None and worker['attention_filter_invocations'] == 0
                and worker['selected_attention_modules'] == [], 'Unexpected base LoRA state')
    else:
        actual = worker['header_validation']
        require(actual['actual_header_validation'] is True and actual['expert_lora'] is False
                and actual['tensor_payload_bytes_read'] == 0
                and worker['attention_filter_invocations'] == 1
                and worker['selected_attention_modules'], 'Actual attention-only LoRA validation missing')
    require(worker.get('router_dtype')=='bfloat16'and worker.get('router_quant_method')=='UnquantizedLinearMethod'
            and worker.get('router_blocks')==worker.get('actual_expert_blocks')
            and worker.get('router_blocks',0)>0
            and len(worker.get('router_modules',[]))==worker['router_blocks']
            and worker.get('actual_fp8_attention_modules')==2*worker['router_blocks']
            and worker.get('all_attention_methods')=='Fp8LinearMethod(use_marlin=True)'
            and worker.get('router_ignore_layers')==worker.get('router_modules')
            and worker.get('actual_global_num_experts')==worker.get('actual_local_num_experts')==125,
            'Actual BF16-router and FP8-attention/expert compatibility validation missing')
    return {'header':header,'worker':worker}


def router_prefixes(model_config):
    _,layers,_,_,_=validate_model_config(model_config)
    require(model_config.get('decoder_sparse_step',1)==1 and not model_config.get('mlp_only_layers'),
            'This runtime requires one 125-expert MoE block/router per decoder layer')
    return [f'model.layers.{layer}.mlp.gate'for layer in range(layers)]


def configure_fp8_router_exclusions(quant_config,model_config):
    """Use the installed FP8 config's supported exact-prefix ignore mechanism."""
    require(type(quant_config).__name__=='Fp8Config'
            and quant_config.is_checkpoint_fp8_serialized is False
            and quant_config.activation_scheme=='dynamic'
            and quant_config.weight_block_size is None
            and quant_config.ignored_layers==[],
            'Fresh actual online per-tensor Fp8Config with no prior exclusions required')
    names=router_prefixes(model_config)
    quant_config.ignored_layers=list(names)
    return names


def validate_mixed_model(model,model_config,*,LinearBase,FusedMoE,bfloat16):
    """Inspect actual methods/dtypes/shapes only; do not read any tensor data."""
    hidden,layers,_,_,_=validate_model_config(model_config)
    expected_routers=router_prefixes(model_config)
    expected_attention={f'model.layers.{layer}.self_attn.{leaf}'for layer in range(layers)
                        for leaf in ('qkv_proj','o_proj')}
    routers,attention,experts=[],[],[]
    for name,module in model.named_modules():
        if name.endswith('.mlp.gate'):
            require(name in expected_routers and isinstance(module,LinearBase)
                    and type(module.quant_method).__name__=='UnquantizedLinearMethod'
                    and module.weight.dtype==bfloat16
                    and list(module.weight.shape)==[125,hidden],
                    'Every actual 125-output router must remain original BF16/unquantized')
            routers.append(name)
        if isinstance(module,FusedMoE):
            require(type(module.quant_method).__name__=='Fp8MoEMethod'
                    and module.quant_method.use_marlin is True
                    and module.global_num_experts==module.local_num_experts==125,
                    'Actual experts must remain FP8-Marlin with 125 experts')
            experts.append(name)
        if isinstance(module,LinearBase):
            base_name=name.removesuffix('.base_layer')
            if base_name in expected_attention:
                require(type(module.quant_method).__name__=='Fp8LinearMethod'
                        and module.quant_method.use_marlin is True,
                        'Actual attention must remain FP8-Marlin beneath optional LoRA')
                attention.append(base_name)
    require(sorted(routers)==sorted(expected_routers)and set(attention)==expected_attention
            and len(attention)==len(expected_attention)and len(experts)==layers,
            'Complete actual BF16-router/FP8-expert/FP8-attention inventory required')
    return dict(router_modules=routers,router_blocks=len(routers),router_dtype='bfloat16',
        router_quant_method='UnquantizedLinearMethod',router_weight_shape=[125,hidden],
        actual_fp8_attention_modules=len(attention),all_attention_methods='Fp8LinearMethod(use_marlin=True)',
        actual_expert_blocks=len(experts))
