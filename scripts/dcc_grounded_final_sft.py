#!/usr/bin/env python3
"""Fresh LoRA SFT on independently reviewed grounded TRAIN gold JSON targets.

Use torchrun for data parallel training. This entry never starts generation or
changes native-only loaders. A separate native evaluation binding is required.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib
import math
import os
from pathlib import Path
import random
import re
import sys
import subprocess
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dcc_grounded_final_builder import (
    PROTOCOL, MASK_VERSION, canonical, digest, encode_target, require,
    strict_json, verify_conditioning,
)

LINEAGE = dict(version='dcc-grounded-final-target-lineage-v1',
    target_origin='annotation-supervised-grounded-final', target_is_actual_native_generation=False,
    conditioning='actual-original-P-plus-complete-actually-generated-naturally-closed-H',
    loss_on_prompt=False, loss_on_thought=False, review_values_supervised=False,
    confidence_values_supervised=False, synthetic_eos_is_training_token=True,
    target_tokenization='one-exact-full-JSON-with-offsets-then-primary-EOS',
    source_mapping='whitespace-only-locate-output-exact-observed-document-substring',
    negative_authority='explicit-TRAIN-empty-annotation-not-source-absence-proof',
    original_native_selector_modified=False, mask_version=MASK_VERSION)
APPROVAL_FIELDS = ('training_target_contract', 'training_source_receipt',
                   'parent_collection_source_receipt', 'training_target_lineage',
                   'corrected_final_binding')


def training_binding(manifest, *, training_source_commit, derivative_id):
    require(type(training_source_commit) is str
            and re.fullmatch('[0-9a-f]{40}', training_source_commit) is not None,
            'Actual new frozen training implementation commit required')
    require(training_source_commit != manifest['parent_source_receipt']['source_commit'],
            'New implementation must not impersonate original collection source')
    identity = manifest['parent_profile']['identity']
    require(derivative_id == identity['derivative_id'], 'Actual approved derivative identity differs')
    return dict(training_target_contract=PROTOCOL,
        training_source_receipt=dict(version='dcc-grounded-final-training-source-v1',
            source_commit=training_source_commit, frozen=True),
        parent_collection_source_receipt=manifest['parent_source_receipt'],
        training_target_lineage=dict(LINEAGE),
        corrected_final_binding=dict(version='dcc-grounded-final-adapter-binding-v1',
            derivative_id=derivative_id, model_revision=identity['model_revision'],
            generation_protocol=identity['protocol'], validated_synthetic_final_records=manifest['rows'],
            mask_version=MASK_VERSION, derived_manifest=manifest, native_only_training=False))



def training_source_identity(source_root, expected_source):
    """Same actual clean committed-runtime guard used by native DPO entry."""
    root = Path(source_root).resolve(strict=True)
    require(type(expected_source) is str and re.fullmatch('[0-9a-f]{40}', expected_source),
            'Explicit approved 40-hex actual training source required')
    require(Path(__file__).resolve() == root / 'scripts/dcc_grounded_final_sft.py'
            and Path(sys.modules[encode_target.__module__].__file__).resolve()
                == root / 'scripts/dcc_grounded_final_builder.py',
            'Actual trainer and builder must run inside the supplied committed runtime')
    def git(*arguments):
        result = subprocess.run(['git', '-C', str(root), *arguments], capture_output=True, text=True)
        require(result.returncode == 0, 'Actual committed training source checkout required')
        return result.stdout.strip()
    require(git('rev-parse', 'HEAD') == expected_source,
            'Actual runtime commit differs from caller-approved training implementation')
    git('ls-files', '--error-unmatch', 'scripts/dcc_grounded_final_sft.py',
        'scripts/dcc_grounded_final_builder.py', 'scripts/dcc_tool_sft.py',
        'scripts/dcc_training_io.py', 'slime/slime_plugins/dcc/hf_policy.py')
    require(not git('status', '--porcelain', '--untracked-files=normal', '--',
                    'scripts', 'slime/slime_plugins/dcc'),
            'Training runtime has modified or untracked code; commit actual implementation first')
    return dict(version='dcc-grounded-final-training-source-v1', source_commit=expected_source, frozen=True)


def validate_base_metadata(model, identity):
    """Check existing small pruned-artifact metadata; no weight hash or load."""
    root = Path(model).resolve(strict=True)
    config = strict_json((root / 'config.json').read_text())
    pruning = config.get('dcc_expert_pruning')
    require(type(pruning) is dict and config.get('model_type') == 'qwen3_moe'
            and pruning.get('derivative_model_id') == identity['derivative_id']
            and pruning.get('source_revision') == identity['model_revision']
            and pruning.get('source_model_id') == identity['model_repository'],
            'Actual model metadata differs from collection derivative/source revision')
    require(type(pruning.get('logical_parameters')) is int
            and 0 < pruning['logical_parameters'] < 30_000_000_000
            and type(config.get('num_experts')) is int and config['num_experts'] > 0
            and config['num_experts'] == pruning.get('experts_per_layer')
            and pruning.get('calibration_split') == 'train',
            'Explicit under-30B TRAIN-derived Qwen3 MoE base metadata required')
    artifact = strict_json((root / 'expert-pruning.json').read_text())
    require(all(artifact.get(key) == value for key, value in pruning.items())
            and artifact.get('physically_removed_experts') is True,
            'Actual derivative artifact receipt differs from local model configuration')
    return dict(model_repository=identity['model_repository'], model_revision=identity['model_revision'],
                derivative_id=identity['derivative_id'], pruning=pruning,
                architecture={key:config[key] for key in ('model_type','num_experts','hidden_size',
                    'num_hidden_layers','num_attention_heads','num_key_value_heads')})


def validate_loaded_base(model, metadata):
    config = model.config.to_dict()
    require(config.get('dcc_expert_pruning') == metadata['pruning']
            and all(config.get(key) == value for key,value in metadata['architecture'].items()),
            'Loaded base configuration differs from preflight-approved actual derivative')


def import_runtime(source_root):
    root = Path(source_root).resolve(strict=True)
    sys.path[:0] = [str(root / 'scripts'), str(root / 'slime')]
    names = ('dcc_tool_sft', 'dcc_training_io', 'slime_plugins.dcc.hf_policy')
    result = [importlib.import_module(name) for name in names]
    for name, module in zip(names, result):
        expected = root / ('slime' if name.startswith('slime_plugins.') else 'scripts')
        expected = expected / (name.replace('.', '/') + '.py' if name.startswith('slime_plugins.') else name + '.py')
        require(Path(module.__file__).resolve() == expected,
                'Actual trainer APIs differ from explicitly supplied frozen runtime')
    return result


def load_reviewed_rows(dataset, approval, tokenizer, *, training_source_commit, derivative_id):
    dataset = Path(dataset)
    manifest = strict_json((dataset / 'manifest.json').read_text())
    require(manifest.get('protocol') == PROTOCOL and manifest.get('launchable') is True
            and manifest.get('native_only_training') is False and not manifest.get('overlength'),
            'Dataset has not passed its independent complete-target prerequisites')
    expected = training_binding(manifest, training_source_commit=training_source_commit,
                                derivative_id=derivative_id)
    require(set(approval) == set(APPROVAL_FIELDS) and canonical(approval) == canonical(expected),
            'Caller-frozen independent five-field approval differs from training data/config')
    payload = (dataset / 'rows.jsonl').read_bytes()
    require(len(payload) == manifest['rows_bytes'] and digest(payload) == manifest['rows_blake2b_256'],
            'Target bytes differ from the independently reviewed dataset manifest')
    identity = manifest['parent_profile']['identity']
    # HF __len__ can rebuild Rust added-token vocabulary metadata; call once.
    vocabulary = len(tokenizer)
    require(vocabulary == identity['tokenizer_vocabulary_size']
            and tokenizer.chat_template == identity['chat_template']
            and tokenizer.eos_token_id == identity['primary_eos_token_id'],
            'Actual local tokenizer differs from original public collection identity')
    rows = [strict_json(line) for line in payload.decode('utf-8').splitlines()]
    require(len(rows) == manifest['rows'] and len({row['task_id'] for row in rows}) == len(rows),
            'One exact target per original task required')
    for row in rows:
        require(row.get('protocol') == PROTOCOL and row.get('benchmark') == 'cuad'
                and row.get('split') == 'train' and type(row.get('positive')) is bool
                and row['positive'] == bool(row.get('gold'))
                and row.get('target_is_actual_native_generation') is False
                and row.get('original_prompt_and_thought_receive_loss') is False
                and row.get('target_eos_is_synthetic_training_token') is True,
                'Explicit independent TRAIN target lineage required')
        parent = row['parent']
        verify_conditioning(parent['original_prompt_ids'], parent['actual_thought_ids'],
                            row['prompt_ids'], eos_ids=identity['eos_token_ids'])
        all_ids = row['prompt_ids'] + row['target_ids']
        require(all(type(token) is int and 0 <= token < vocabulary for token in all_ids),
                'Dataset token IDs exceed actual tokenizer vocabulary')
        spans = [value['text'] for value in strict_json(row['target_text'])['answer']]
        require(set(spans) == {value['target_text'] for value in row['span_mappings']},
                'Complete JSON spans differ from reviewed exact document mappings')
        target = encode_target(tokenizer, spans, final_review='plan' in strict_json(row['target_text']),
                               eos_ids=identity['eos_token_ids'])
        require(all(canonical(row[key]) == canonical(target[key]) for key in target),
                'Actual target/token mask differs from single full-JSON tokenization')
        require(len(row['prompt_ids']) + len(row['target_ids']) == row['total_tokens']
                and row['total_tokens'] <= manifest['max_sequence_tokens']
                and row['total_tokens'] <= identity['max_model_len'],
                'Complete sequence exceeds reviewed budget; never crop or skip')
    require(Counter(row['positive'] for row in rows) ==
            Counter({True: manifest['positive'], False: manifest['negative']}),
            'Actual TRAIN polarity counts differ from independent approval')
    return rows, manifest, expected


def sample_plan(rows, *, seed, world, steps):
    """Identical global 2:1 quota stream, task decks before logical rank selection."""
    require(type(world) is int and world > 0 and type(steps) is int and steps > 0,
            'Positive typed world and optimizer step count required')
    pools = {positive: [index for index, row in enumerate(rows) if row['positive'] is positive]
             for positive in (True, False)}
    require(all(pools.values()), 'Both annotation polarities must have eligible tasks')
    # Stable task ordering makes all process plans independent of JSON row order.
    for pool in pools.values():
        pool.sort(key=lambda index: rows[index]['task_id'])
    rngs = {positive: random.Random(seed + (1 if positive else 2) * 1_000_003)
            for positive in (True, False)}
    decks, used, plan = {True: [], False: []}, Counter(), []
    for step in range(steps):
        batch = []
        for _ in range(world):
            number = sum(used.values()) + 1
            positive = max((True, False), key=lambda value:
                (2 if value else 1) * number - 3 * used[value])
            if not decks[positive]:
                decks[positive] = pools[positive].copy()
                rngs[positive].shuffle(decks[positive])
            batch.append(decks[positive].pop())
            used[positive] += 1
        plan.append(batch)
    return plan


def masked_nll(logprobs, loss_mask):
    import torch
    require(logprobs.ndim == 1 and logprobs.numel() == len(loss_mask)
            and all(type(value) is bool for value in loss_mask) and any(loss_mask),
            'Exact nonempty whole-token target loss mask required')
    selection = torch.tensor(loss_mask, dtype=torch.bool, device=logprobs.device)
    return -logprobs[selection].mean()


def backward_row(model, row, logprob_fn):
    import torch
    values = logprob_fn(model, row['prompt_ids'], row['target_ids'])
    loss = masked_nll(values, row['loss_mask'])
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite corrected-final loss; refusing update')
    loss.backward()
    return float(loss.detach())


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--runtime-source-root', type=Path, required=True)
    result.add_argument('--training-source-commit', required=True)
    result.add_argument('--derivative-id', required=True)
    result.add_argument('--model', required=True)
    result.add_argument('--dataset', type=Path, required=True)
    result.add_argument('--approval', type=Path, required=True,
        help='Independently root-reviewed five-field approval, never auto-generated by training')
    result.add_argument('--output-dir', type=Path, required=True)
    result.add_argument('--steps', type=int, default=32)
    result.add_argument('--save-every', type=int, default=8)
    result.add_argument('--learning-rate', type=float, default=1e-5)
    result.add_argument('--max-grad-norm', type=float, default=1.)
    result.add_argument('--lora-rank', type=int, default=16)
    result.add_argument('--seed', type=int, default=47)
    result.add_argument('--quantization', choices=('nf4', 'none'), default='nf4')
    result.add_argument('--cpu-preflight', action='store_true')
    result.add_argument('--plan-world-size', type=int, default=8)
    result.add_argument('--sync-artifacts', action='store_true')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    require(args.steps > 0 and args.save_every > 0 and args.lora_rank > 0
            and args.seed >= 0 and math.isfinite(args.learning_rate) and args.learning_rate > 0
            and math.isfinite(args.max_grad_norm) and args.max_grad_norm > 0,
            'Finite positive training configuration required')
    actual_source = training_source_identity(args.runtime_source_root, args.training_source_commit)
    trainer, io, hf = import_runtime(args.runtime_source_root)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    rows, manifest, binding = load_reviewed_rows(args.dataset, strict_json(args.approval.read_text()),
        tokenizer, training_source_commit=args.training_source_commit, derivative_id=args.derivative_id)
    require(binding['training_source_receipt'] == actual_source, 'Actual training source binding differs')
    base_metadata = validate_base_metadata(args.model, manifest['parent_profile']['identity'])
    world = args.plan_world_size if args.cpu_preflight else int(os.environ.get('WORLD_SIZE', '1'))
    plan = sample_plan(rows, seed=args.seed, world=world, steps=args.steps)
    exposures = Counter(rows[index]['positive'] for batch in plan for index in batch)
    config = dict(training_method='annotation-supervised grounded CUAD gold-final SFT; not RL',
        **binding, actual_base_identity=base_metadata, native_thinking=dict(enabled=False, protocol=PROTOCOL,
            origin='annotation-supervised-grounded-final', requires_new_explicit_evaluation_binding=True),
        quantization=args.quantization, lora_rank=args.lora_rank, lora_alpha=2*args.lora_rank,
        lora_dropout=0., target_modules=['q_proj', 'v_proj', 'o_proj'],
        learning_rate=args.learning_rate, max_grad_norm=args.max_grad_norm,
        optimizer='AdamW', weight_decay=0., steps=args.steps, save_every=args.save_every,
        seed=args.seed, world_size=world, samples_per_rank=1, warm_start=False,
        loss_configuration=dict(version=MASK_VERSION, example_weighting='global-equal-example',
            token_weighting='per-example-mean-over-unmasked-JSON-and-EOS-tokens'),
        sampler_configuration=dict(version='global-annotation-polarity-task-decks-v1',
            positive_weight='2/3', negative_weight='1/3', task_weight='uniform-without-replacement-per-deck',
            samples_per_rank=1, full_sequence_never_truncated=True, fresh_only=True))
    if args.cpu_preflight:
        print(canonical(dict(protocol=PROTOCOL, cpu_preflight_only=True, CUDA_initialized=False,
            optimizer_steps=args.steps, world_size=world, actual_examples=world*args.steps,
            positive_examples=exposures[True], negative_examples=exposures[False],
            unique_positive=len({rows[index]['task_id'] for batch in plan for index in batch if rows[index]['positive']}),
            unique_negative=len({rows[index]['task_id'] for batch in plan for index in batch if not rows[index]['positive']}),
            max_consumed_sequence=max(rows[index]['total_tokens'] for batch in plan for index in batch),
            training_config=config, plan=[[rows[index]['task_id'] for index in batch] for batch in plan])))
        return 0
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    rank, local = int(os.environ.get('RANK', '0')), int(os.environ.get('LOCAL_RANK', '0'))
    require(0 <= rank < world, 'Valid actual distributed rank required')
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(hours=2))
    io.prepare_training_output(args.output_dir, rank=rank, world_size=world, distributed=dist)
    torch.manual_seed(args.seed)
    started = time.monotonic()
    with io.TrainingJournal(args.output_dir, rank, sync=args.sync_artifacts) as journal:
        def record(**value):
            line = journal.record(**value)
            if rank == 0:
                print(line, flush=True)
        record(event='start', **config)
        model, loaded_tokenizer = hf.load_policy_model(args.model, trainable=True, rank=local,
            lora_rank=args.lora_rank, quantization=args.quantization)
        validate_loaded_base(model, base_metadata)
        require(loaded_tokenizer.chat_template == tokenizer.chat_template
                and len(loaded_tokenizer) == len(tokenizer), 'Model loader tokenizer changed')
        parameters = [parameter for name, parameter in model.named_parameters() if parameter.requires_grad]
        require(parameters and all('lora_' in name for name, parameter in model.named_parameters()
            if parameter.requires_grad), 'Only attention LoRA parameters may be trained')
        require(all(module.p == 0 for name, module in model.named_modules()
            if 'lora_dropout' in name and isinstance(module, torch.nn.Dropout)), 'LoRA dropout must be zero')
        if world > 1:
            for parameter in parameters:
                dist.broadcast(parameter.data, src=0)
        if hasattr(model, 'dcc_quantization_report'):
            record(event='quantization', **model.dcc_quantization_report)
        optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.)
        def save(name, step):
            journal.checkpoint_boundary()
            if world > 1:
                dist.barrier()
            if rank == 0:
                def writer(destination):
                    model.save_pretrained(destination)
                    loaded_tokenizer.save_pretrained(destination)
                    torch.save(dict(optimizer=optimizer.state_dict(), optimizer_steps=step,
                        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                        sampler='fresh-only-full-plan', sampler_step=step), destination / 'training_state.pt')
                    (destination / 'training_config.json').write_text(canonical(config) + '\n')
                io.publish_checkpoint(args.output_dir, name, step, writer, sync=journal.sync)
            if world > 1:
                dist.barrier()
        save('initial', 0)
        for step, indices in enumerate(plan, 1):
            row = rows[indices[rank]]
            begin = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = backward_row(model, row, hf.completion_logprobs)
            global_examples = trainer.average_example_gradients(parameters, 1, world)
            require(global_examples == world, 'Every rank must contribute exactly one complete example')
            norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            if not torch.isfinite(norm) or norm <= 0:
                raise FloatingPointError('Expected finite nonzero adapter gradient; refusing update')
            optimizer.step()
            record(event='step', step=step, loss=loss, gradient_norm=norm.item(),
                seconds=time.monotonic()-begin, global_examples=global_examples,
                samples=[dict(task_id=row['task_id'], stage='corrected_final', category=row['category'],
                    positive=row['positive'], parent_trace_id=row['parent']['trace_id'],
                    original_stage=row['parent']['original_stage'],
                    prompt_tokens=len(row['prompt_ids']), completion_tokens=len(row['target_ids']),
                    supervised_tokens=sum(row['loss_mask']), target_is_actual_native_generation=False)])
            if step % args.save_every == 0:
                save(f'step_{step:04d}', step)
        save('final', args.steps)
        record(event='finished', optimizer_steps=args.steps, elapsed=time.monotonic()-started,
               actual_global_examples=world*args.steps)
        io.finalize_training(journal, args.steps, world_size=world, distributed=dist)
    if world > 1:
        dist.destroy_process_group()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
