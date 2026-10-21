"""Explicit vLLM 0.11.2 BF16 base/q-v-o worker; default workers are untouched."""
import json
import os
from pathlib import Path
import time

from vllm.v1.worker.gpu_worker import Worker

from .bf16_attention_guard import (BINDING_KEY, BINDING_SCHEMA, HEADER_SCHEMA, WORKER_SCHEMA,
    VALIDATION_LAYOUT, make_attention_filter, model_validation, require, runtime_identity,
    strict_json, validate_attention_header, validate_base_model)


class BF16AttentionWorker(Worker):
    def load_model(self):
        import torch
        import vllm
        from vllm.lora import models as M, utils as U
        from vllm.lora.layers.base import BaseLayerWithLoRA
        from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
        from vllm.model_executor.layers.fused_moe import FusedMoE
        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod

        require(vllm.__version__ == '0.11.2', 'Exact reviewed vLLM version required')
        require(self.model_config.quantization is None and self.model_config.dtype == torch.bfloat16,
                'Unquantized BF16 weights/activations required')
        require(self.parallel_config.tensor_parallel_size == 1
                and self.parallel_config.pipeline_parallel_size == 1
                and not self.parallel_config.enable_expert_parallel,
                'Only a single-GPU BF16 attention-only worker is supported')
        config = self.model_config.hf_config.to_dict()
        validated_model = model_validation(config)
        payload = strict_json(os.environ[BINDING_KEY])
        require(payload['schema'] == BINDING_SCHEMA
                and payload['runtime_code_identity'] == runtime_identity(), 'Missing explicit BF16 worker binding')
        audit_dir = Path(payload['audit_dir'])
        require(audit_dir.is_dir() and audit_dir.is_absolute(), 'Fresh validation directory required')
        adapter = payload['adapter']
        enabled = adapter is not None
        require(enabled == (self.lora_config is not None), 'Actual adapter/LoRA configuration mismatch')
        header = validate_attention_header(adapter, config) if enabled else None
        before = strict_json((audit_dir / VALIDATION_LAYOUT['header_validation_receipt']).read_text())
        require(before['schema'] == HEADER_SCHEMA
                and before['runtime_code_identity'] == runtime_identity()
                and before['adapter'] == adapter and before['header_validation'] == header
                and before['model_validation'] == validated_model and before['lora_enabled'] is enabled
                and before['validated_before_engine_creation'] is True,
                'Header/config changed between engine creation and worker load')
        require(M.get_supported_lora_modules is U.get_supported_lora_modules,
                'A prior LoRA module filter is forbidden in this fresh worker')
        original = M.get_supported_lora_modules
        observations = []
        if enabled:
            M.get_supported_lora_modules = make_attention_filter(original, model_config=config,
                targets=header['target_modules'], LinearBase=LinearBase, FusedMoE=FusedMoE,
                UnquantizedLinearMethod=UnquantizedLinearMethod,
                UnquantizedFusedMoEMethod=UnquantizedFusedMoEMethod, observations=observations)
        started = time.time()
        try:
            super().load_model()
        finally:
            if enabled:
                M.get_supported_lora_modules = original
        require(len(observations) == int(enabled), 'Unexpected number of attention filter invocations')
        model = self.model_runner.model
        expected = {row[0]: row for row in observations[0]['selected_bindings']} if enabled else {}
        actual_experts, wrappers = [], []
        for name, module in model.named_modules():
            require(type(module).__name__ != 'FusedMoEWithLoRA', 'Unexpected expert LoRA wrapper')
            if isinstance(module, FusedMoE):
                require(type(module.quant_method) is UnquantizedFusedMoEMethod
                        and module.global_num_experts == module.local_num_experts == 125,
                        'Post-load expert method differs from native BF16')
                actual_experts.append((name, id(module), id(module.quant_method)))
            if isinstance(module, BaseLayerWithLoRA):
                require(name in expected and isinstance(module.base_layer, LinearBase)
                        and (name, id(module.base_layer), id(module.base_layer.quant_method)) == expected[name],
                        'Unexpected nonattention wrapper or replaced attention base layer')
                wrappers.append(name)
        manager = getattr(model, 'lora_manager', None)
        if enabled:
            require(actual_experts == observations[0]['expert_bindings'],
                    'Expert modules or native methods changed during attention LoRA installation')
            require(set(wrappers) == set(expected) and len(wrappers) == len(expected),
                    'Actual attention LoRA wrappers differ from selected modules')
            require(manager is not None and set(manager.supported_lora_modules) == {'qkv_proj', 'o_proj'}
                    and set(manager.modules) == set(expected), 'Actual manager includes nonattention modules')
            native_attention = sorted(expected)
        else:
            base = validate_base_model(model, config, LinearBase=LinearBase, FusedMoE=FusedMoE,
                BaseLayerWithLoRA=BaseLayerWithLoRA, UnquantizedLinearMethod=UnquantizedLinearMethod,
                UnquantizedFusedMoEMethod=UnquantizedFusedMoEMethod)
            native_attention = base['native_attention_modules']
        record = dict(schema=WORKER_SCHEMA,
            runtime_code_identity=runtime_identity(), adapter=adapter, header_validation=header,
            model_validation=validated_model, lora_enabled=enabled,
            pid=os.getpid(), started_unix=started, ended_unix=time.time(),
            model_validated_before_model_load=True, header_validated_before_model_load=enabled,
            actual_expert_blocks=len(actual_experts),
            actual_expert_module_names=sorted(row[0] for row in actual_experts),
            actual_native_attention_modules=native_attention, actual_lora_manager_present=manager is not None,
            actual_global_num_experts=125, actual_local_num_experts=125,
            all_expert_methods='UnquantizedFusedMoEMethod', expert_lora_wrappers=0,
            attention_filter_invocations=len(observations), selected_attention_modules=sorted(expected),
            actual_attention_lora_wrappers=sorted(wrappers),
            original_filter_restored=M.get_supported_lora_modules is original,
            tensor_payload_bytes_read_by_header_guard=0,
            model_parameter_storage_bytes_before_adapter_activation=sum(
                parameter.numel() * parameter.element_size() for parameter in model.parameters()),
            storage_metric='parameter storage only, not allocator peak or KV cache capacity',
            diagnostic_only=True, benchmark_gain_claimed=False)
        with (audit_dir / VALIDATION_LAYOUT['worker_validation_receipt']).open('x') as stream:
            json.dump(record, stream, indent=2); stream.write('\n')
