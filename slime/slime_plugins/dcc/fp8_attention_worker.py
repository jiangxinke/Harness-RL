"""Dedicated vLLM 0.11.2 worker; patch applies only around this fresh model load."""
import json
import os
from pathlib import Path
import time

from vllm.v1.worker.gpu_worker import Worker

from .fp8_attention_guard import (QUANTIZATION, make_attention_filter, require,
                                  strict_json,validate_attention_header,validate_model_config,
                                  configure_fp8_router_exclusions,validate_mixed_model)


class FP8AttentionWorker(Worker):
    def load_model(self):
        import torch
        import vllm
        from vllm.lora import models as M, utils as U
        from vllm.model_executor.layers.linear import LinearBase
        from vllm.model_executor.layers.fused_moe import FusedMoE

        require(vllm.__version__ == '0.11.2', 'Exact reviewed vLLM version required')
        require(self.model_config.quantization == 'fp8' and self.model_config.dtype == torch.bfloat16,
                'Explicit FP8 weights and BF16 activations required')
        require(self.parallel_config.tensor_parallel_size == 1
                and self.parallel_config.pipeline_parallel_size == 1
                and not self.parallel_config.enable_expert_parallel,
                'Only a single-GPU FP8-Marlin worker is supported')
        require(torch.cuda.get_device_capability(self.device) == (8, 0), 'FP8-Marlin attention LoRA requires SM80')
        from .fp8_runtime import runtime_identity
        config = self.model_config.hf_config.to_dict()
        validate_model_config(config)
        payload = strict_json(os.environ['DCC_FP8_ATTENTION_WORKER_BINDING'])
        require(payload['schema'] == 'dcc-fp8-attention-worker-binding-v1'
                and payload['quantization'] == 'fp8'
                and payload['runtime_code_identity'] == runtime_identity(), 'Missing explicit worker binding')
        audit_dir = Path(payload['audit_dir'])
        require(audit_dir.is_dir() and audit_dir.is_absolute(), 'Fresh validation directory required')
        adapter = payload['adapter']
        header = validate_attention_header(adapter, config) if adapter is not None else None
        require((adapter is not None) == (self.lora_config is not None), 'Adapter/LoRA configuration mismatch')
        require(M.get_supported_lora_modules is U.get_supported_lora_modules,
                'A prior LoRA module filter is forbidden in this fresh worker')
        router_ignored=configure_fp8_router_exclusions(self.vllm_config.quant_config,config)
        require(self.model_runner.vllm_config is self.vllm_config,
                'Actual model runner must share the configured FP8 config before model construction')
        original = M.get_supported_lora_modules
        observations = []
        if header is not None:
            M.get_supported_lora_modules = make_attention_filter(original,
                targets=header['target_modules'], LinearBase=LinearBase,
                FusedMoE=FusedMoE, observations=observations)
        started=time.time()
        try:
            super().load_model()
        finally:
            M.get_supported_lora_modules = original
        model=self.model_runner.model
        mixed=validate_mixed_model(model,config,LinearBase=LinearBase,FusedMoE=FusedMoE,bfloat16=torch.bfloat16)
        require(self.vllm_config.quant_config.ignored_layers==router_ignored,
                'Exact router-only exclusions must remain active for this loaded model')
        actual_experts = []
        for name, module in model.named_modules():
            require(type(module).__name__ != 'FusedMoEWithLoRA', 'An expert LoRA wrapper was unexpectedly created')
            if isinstance(module, FusedMoE):
                require(type(module.quant_method).__name__ == 'Fp8MoEMethod'
                        and module.quant_method.use_marlin is True and module.global_num_experts == 125 and module.local_num_experts == 125,
                        'Post-load expert quantization differs from FP8-Marlin')
                actual_experts.append((name, id(module), id(module.quant_method)))
        require(actual_experts, 'No actual FP8 expert blocks were loaded')
        if header is not None:
            require(len(observations) == 1 and observations[0]['expert_bindings'] == actual_experts,
                    'Expert modules or quantization methods changed during attention LoRA installation')
        else:
            require(not observations, 'A base-only run must not invoke the LoRA manager')
        record = dict(schema='dcc-fp8-attention-worker-actual-validation-v1',
            pid=os.getpid(), started_unix=started, ended_unix=time.time(),
            quantization='fp8', fp8_contract=QUANTIZATION, runtime_code_identity=runtime_identity(),
            adapter=adapter, header_validation=header, header_validated_before_model_load=True,
            actual_expert_blocks=len(actual_experts), actual_num_experts=125,
            actual_global_num_experts=125, actual_local_num_experts=125,
            all_expert_methods='Fp8MoEMethod(use_marlin=True)', expert_lora_wrappers=0,
            attention_filter_invocations=len(observations),
            selected_attention_modules=observations[0]['selected_module_names'] if observations else [],
            original_filter_restored=M.get_supported_lora_modules is original,
            tensor_payload_bytes_read_by_header_guard=0,
            model_parameter_storage_bytes_before_adapter_activation=sum(
                parameter.numel() * parameter.element_size() for parameter in model.parameters()),
            storage_metric='parameter storage only, not allocator peak or KV cache capacity',
            diagnostic_only=True, benchmark_gain_claimed=False)
        record.update(mixed,router_ignore_layers=router_ignored,
            router_exclusion_applied_before_model_construction=True)
        with (audit_dir/'worker-validation.actual.json').open('x')as stream:
            json.dump(record, stream, indent=2); stream.write('\n')
