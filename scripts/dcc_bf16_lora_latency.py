#!/usr/bin/env python3
"""One fresh engine per invocation; caller owns GPU selection and run isolation.

Run stock/filtered in separate processes on the same GPU, then reverse their
order. This fixed-token diagnostic is not a benchmark quality evaluation.
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'slime'))


def parse_lengths(value):
    try:
        result = [int(item) for item in value.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Comma-separated positive token lengths required') from exc
    if not result or any(item < 1 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError('Distinct positive token lengths required')
    return result


def build_workload(tokenizer, lengths, batch_size):
    # No chat template, task files, benchmark labels or model output influences this input.
    unit = tokenizer.encode('Runtime latency control. Read the following fixed text. ', add_special_tokens=False)
    if not unit or batch_size < 1:
        raise ValueError('A nonempty fixed-token unit and positive batch size are required')
    return [{'prompt_tokens': length,
             'prompts': [{'prompt_token_ids': (unit * ((length + len(unit) - 1) // len(unit)))[:length]}
                         for _ in range(batch_size)]} for length in lengths]


def inspect_model(model):
    """Runs in the actual worker via the pinned vLLM public apply_model API."""
    import torch
    from vllm.lora.layers.base import BaseLayerWithLoRA
    from vllm.model_executor.layers.fused_moe import FusedMoE
    wrappers, experts, buffers, enabled = [], [], [], []
    for name, module in model.named_modules():
        if isinstance(module, BaseLayerWithLoRA):
            wrappers.append({'name': name, 'type': type(module).__name__})
        if isinstance(module, FusedMoE):
            experts.append({'name': name, 'method': type(module.quant_method).__name__})
        if type(module).__name__ == 'FusedMoEWithLoRA':
            enabled.append({'name': name, 'values': module.adapter_enabled.detach().cpu().tolist()})
            for weight in ('w1_lora_a_stacked', 'w1_lora_b_stacked', 'w2_lora_a_stacked',
                           'w2_lora_b_stacked', 'w3_lora_a_stacked', 'w3_lora_b_stacked'):
                tensor = getattr(module, weight)
                buffers.append({'name': name + '.' + weight, 'shape': list(tensor.shape),
                                'bytes': tensor.numel() * tensor.element_size()})
    return dict(lora_wrappers=wrappers, expert_methods=experts,
        expert_lora_wrappers=len(enabled), expert_adapter_enabled=enabled,
        expert_lora_buffers=buffers, expert_lora_buffer_bytes=sum(row['bytes'] for row in buffers),
        worker_cuda_allocated_bytes=torch.cuda.memory_allocated(),
        worker_cuda_reserved_bytes=torch.cuda.memory_reserved(),
        worker_cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        allocator_metrics_are_not_total_device_memory=True)


def checked_outputs(outputs, prompts, completion_tokens):
    if len(outputs) != len(prompts):
        raise RuntimeError('Engine output count differs from fixed input count')
    result = []
    for row, prompt in zip(outputs, prompts):
        if list(row.prompt_token_ids) != prompt['prompt_token_ids'] or len(row.outputs) != 1:
            raise RuntimeError('Actual engine prompt IDs or choice count differ')
        choice = row.outputs[0]
        ids = list(choice.token_ids)
        if len(ids) != completion_tokens or choice.finish_reason != 'length':
            raise RuntimeError('Engine did not complete the fixed decode token budget')
        result.append(dict(token_ids=ids, finish_reason=choice.finish_reason))
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--model', required=True)
    result.add_argument('--adapter', required=True, help='Same q/v/o adapter for both stock and filtered runs')
    result.add_argument('--mode', required=True, choices=('stock', 'filtered', 'base'))
    result.add_argument('--output-dir', required=True, help='Fresh absolute directory; parent must already exist')
    result.add_argument('--prompt-lengths', type=parse_lengths, default=[512, 4096, 16384])
    result.add_argument('--completion-tokens', type=int, default=256)
    result.add_argument('--repetitions', type=int, default=3)
    result.add_argument('--warmups', type=int, default=1)
    result.add_argument('--batch-size', type=int, default=1)
    result.add_argument('--max-model-len', type=int, default=53248)
    result.add_argument('--max-batched-tokens', type=int, default=8192)
    result.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    result.add_argument('--seed', type=int, default=42)
    result.add_argument('--enforce-eager', action=argparse.BooleanOptionalAction, default=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if min(args.completion_tokens, args.repetitions, args.warmups, args.batch_size,
           args.max_batched_tokens) < 1 or max(args.prompt_lengths) + args.completion_tokens > args.max_model_len:
        raise ValueError('Positive fixed budgets and sufficient model context are required')
    output = Path(args.output_dir)
    if not output.is_absolute() or '..' in output.parts:
        raise ValueError('Fresh absolute output directory required')
    # Validate the same adapter for all modes before any GPU engine construction.
    from transformers import AutoConfig
    from slime_plugins.dcc.bf16_attention_guard import validate_attention_header, validate_worker_receipts
    header = validate_attention_header(args.adapter, AutoConfig.from_pretrained(args.model).to_dict())
    output.mkdir(parents=False, exist_ok=False)
    manifest = dict(schema='dcc-bf16-lora-fixed-token-latency-v1', configuration=vars(args),
                    adapter_header=header, diagnostic_only=True, benchmark_gain_claimed=False,
                    prefix_cache_reset_before_every_call=True, source_tasks_or_labels_read=False)
    (output/'manifest.actual.json').write_text(json.dumps(manifest, indent=2) + '\n')
    from slime_plugins.dcc.vllm_policy import load_vllm_policy
    from vllm import SamplingParams
    options = dict(bf16_attention_only_lora=True,
                   bf16_lora_validation_dir=str(output/'bf16-validation')) if args.mode == 'filtered' else {}
    started = time.perf_counter()
    policy = load_vllm_policy(args.model, adapter=None if args.mode == 'base' else args.adapter,
        batch_size=args.batch_size, max_prompt_tokens=max(args.prompt_lengths),
        max_model_len=args.max_model_len, tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.max_batched_tokens, seed=args.seed,
        enforce_eager=args.enforce_eager, **options)
    load_seconds = time.perf_counter() - started
    try:
        if args.mode == 'filtered':
            validate_worker_receipts(output/'bf16-validation', adapter=args.adapter)
        engine = policy.model
        workload = build_workload(policy.tokenizer, args.prompt_lengths, args.batch_size)
        (output/'workload.actual.json').write_text(json.dumps(workload) + '\n')
        (output/'policy-provenance.actual.json').write_text(json.dumps(policy.provenance, indent=2) + '\n')
        sampling = SamplingParams(n=1, temperature=0.0, top_p=1.0, top_k=-1, min_p=0.0,
            presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0,
            max_tokens=args.completion_tokens, seed=args.seed, ignore_eos=True,
            stop_token_ids=[], skip_special_tokens=False)
        measured = []
        with (output/'calls.actual.jsonl').open('x', buffering=1) as stream:
            for case in workload:
                for iteration in range(args.warmups + args.repetitions):
                    engine.reset_prefix_cache()
                    before = time.perf_counter()
                    outputs = engine.generate(case['prompts'], sampling,
                                              lora_request=policy.lora_request, use_tqdm=False)
                    elapsed = time.perf_counter() - before
                    choices = checked_outputs(outputs, case['prompts'], args.completion_tokens)
                    row = dict(prompt_tokens=case['prompt_tokens'], iteration=iteration,
                        warmup=iteration < args.warmups, elapsed_seconds=elapsed,
                        decoded_tokens=args.batch_size * args.completion_tokens,
                        generated_tokens_per_second_including_prefill=args.batch_size * args.completion_tokens / elapsed,
                        outputs=choices)
                    stream.write(json.dumps(row) + '\n')
                    if not row['warmup']:
                        measured.append({key: value for key, value in row.items() if key != 'outputs'})
        inventory = engine.apply_model(inspect_model)
        if len(inventory) != 1:
            raise RuntimeError('Expected exactly one actual model worker')
        if args.mode == 'filtered' and inventory[0]['expert_lora_wrappers'] != 0:
            raise RuntimeError('Filtered worker unexpectedly has expert wrappers')
        if args.mode == 'stock' and inventory[0]['expert_lora_wrappers'] < 1:
            raise RuntimeError('Stock worker lacks the expert wrappers being investigated')
        summary = dict(manifest, closed=True, load_seconds=load_seconds,
                       measured=measured, actual_worker_inventory=inventory)
        (output/'summary.actual.json').write_text(json.dumps(summary, indent=2) + '\n')
    finally:
        asyncio.run(policy.close())


if __name__ == '__main__':
    main()
