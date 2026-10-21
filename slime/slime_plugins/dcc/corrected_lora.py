"""Explicit grounded TRAIN-reference LoRA binding, separate from native SFT.

Only saved declarations and standard attention safetensors headers are checked.
An independent caller declaration is required; never infer it from the adapter.
This is metadata binding, not proof of optimizer updates or data replay.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re

PROTOCOL = 'cuad-observed-gold-json-sft-v1'
ORIGIN = 'annotation-supervised-grounded-final'
MASK_VERSION = 'answer-json-and-eos-without-review-or-confidence-values-v1'
APPROVAL_FIELDS = frozenset({'training_target_contract', 'training_source_receipt',
    'parent_collection_source_receipt', 'training_target_lineage', 'corrected_final_binding'})


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':'))


def target_lineage():
    return dict(version='dcc-grounded-final-target-lineage-v1', target_origin=ORIGIN,
        target_is_actual_native_generation=False,
        conditioning='actual-original-P-plus-complete-actually-generated-naturally-closed-H',
        loss_on_prompt=False, loss_on_thought=False, review_values_supervised=False,
        confidence_values_supervised=False, synthetic_eos_is_training_token=True,
        target_tokenization='one-exact-full-JSON-with-offsets-then-primary-EOS',
        source_mapping='whitespace-only-locate-output-exact-observed-document-substring',
        negative_authority='explicit-TRAIN-empty-annotation-not-source-absence-proof',
        original_native_selector_modified=False, mask_version=MASK_VERSION)


def validate_corrected_approval(value):
    from .native_budget import BUDGET_PROTOCOL, LEGACY_PROTOCOL
    from .native_thinking_eval import MODEL_REVISION, validate_source_receipt
    from .hf_policy import PROMPT_ENCODING_VERSION

    require(isinstance(value, dict) and set(value) == APPROVAL_FIELDS,
        'An independent complete five-field corrected-training declaration is required')
    require(value['training_target_contract'] == PROTOCOL,
        'Explicit grounded TRAIN-reference target contract is required')
    parent = validate_source_receipt(value['parent_collection_source_receipt'])
    source = value['training_source_receipt']
    require(isinstance(source, dict) and set(source) == {'version', 'source_commit', 'frozen'}
        and source.get('version') == 'dcc-grounded-final-training-source-v1'
        and isinstance(source.get('source_commit'), str)
        and re.fullmatch(r'[0-9a-f]{40}', source['source_commit']) is not None
        and source['source_commit'] != parent['source_commit'] and source['frozen'] is True,
        'Separate actual frozen grounded-final training-code source required')
    require(canonical(value['training_target_lineage']) == canonical(target_lineage()),
        'Corrected target lineage, P/H masks or synthetic-target identity differs')
    binding = value['corrected_final_binding']
    fields = {'version', 'derivative_id', 'model_revision', 'generation_protocol',
        'validated_synthetic_final_records', 'mask_version', 'derived_manifest', 'native_only_training'}
    require(isinstance(binding, dict) and set(binding) == fields
        and binding.get('version') == 'dcc-grounded-final-adapter-binding-v1'
        and isinstance(binding.get('derivative_id'), str) and bool(binding['derivative_id'].strip())
        and binding.get('model_revision') == MODEL_REVISION
        and binding.get('generation_protocol') in {LEGACY_PROTOCOL, BUDGET_PROTOCOL}
        and binding.get('mask_version') == MASK_VERSION
        and binding.get('native_only_training') is False
        and type(binding.get('validated_synthetic_final_records')) is int
        and binding['validated_synthetic_final_records'] > 0,
        'Independent corrected-final adapter declaration is invalid')
    manifest = binding['derived_manifest']
    require(isinstance(manifest, dict) and manifest.get('protocol') == PROTOCOL
        and manifest.get('target_contract') == 'synthetic-TRAIN-gold-JSON-with-real-masked-PH'
        and manifest.get('native_only_training') is False
        and manifest.get('training_method') == 'supervised fine-tuning; not RL'
        and manifest.get('mask_version') == MASK_VERSION
        and manifest.get('original_journals_closed_before_labels') is True
        and manifest.get('original_native_selector_modified') is False
        and canonical(manifest.get('parent_source_receipt')) == canonical(parent)
        and manifest.get('confidence_values_supervised') is False
        and manifest.get('held_out_reads') is False and manifest.get('launchable') is True
        and type(manifest.get('rows')) is int
        and manifest['rows'] == binding['validated_synthetic_final_records']
        and all(type(manifest.get(k)) is int and manifest[k] > 0 for k in ('positive', 'negative'))
        and manifest['positive'] + manifest['negative'] == manifest['rows'],
        'Corrected TRAIN manifest closure, counts or held-out boundary differs')
    identity = manifest.get('parent_profile', {}).get('identity')
    require(isinstance(identity, dict) and identity.get('source_commit') == parent['source_commit']
        and identity.get('derivative_id') == binding['derivative_id']
        and identity.get('model_revision') == MODEL_REVISION
        and identity.get('protocol') == binding['generation_protocol']
        and identity.get('prompt_encoding_version') == PROMPT_ENCODING_VERSION,
        'Corrected parent base/tokenizer/encoding identity differs')
    return copy.deepcopy(value)


def approval_from_arguments(args):
    path = getattr(args, 'corrected_lora_approval', None)
    if path is None:
        return None
    from .native_thinking_eval import _strict_json
    return validate_corrected_approval(_strict_json(Path(path).read_text(encoding='utf-8')))


def corrected_binding(approval, *, derivative_id, source_receipt, evaluation_protocol,
                      tensor_pairs, checkpoint_optimizer_steps, configured_optimizer_steps):
    from .native_thinking_eval import MODEL_REVISION, validate_source_receipt
    from .native_budget import BUDGET_PROTOCOL, LEGACY_PROTOCOL
    approval = validate_corrected_approval(approval)
    actual = approval['corrected_final_binding']
    require(derivative_id == actual['derivative_id'], 'Corrected adapter differs from approved base derivative')
    require(evaluation_protocol in {LEGACY_PROTOCOL, BUDGET_PROTOCOL},
        'Corrected adapter requires the unchanged explicit native evaluation protocol')
    require(type(tensor_pairs) is int and tensor_pairs > 0, 'Actual corrected attention header pairs required')
    require(type(configured_optimizer_steps) is int and configured_optimizer_steps > 0
        and type(checkpoint_optimizer_steps) is int
        and 0 <= checkpoint_optimizer_steps <= configured_optimizer_steps,
        'Actual complete corrected checkpoint optimizer step required')
    return dict(version='dcc-grounded-final-evaluation-binding-v1', training_target_contract=PROTOCOL,
        training_source_receipt=copy.deepcopy(approval['training_source_receipt']),
        parent_collection_source_receipt=copy.deepcopy(approval['parent_collection_source_receipt']),
        evaluation_source_receipt=validate_source_receipt(source_receipt),
        evaluation_protocol=evaluation_protocol, caller_approved_corrected_training=approval,
        derivative_id=derivative_id, model_revision=MODEL_REVISION, lora_rank=16,
        tensor_pairs=tensor_pairs, checked_actual_attention_shapes=True,
        checkpoint_optimizer_steps=checkpoint_optimizer_steps,
        configured_optimizer_steps=configured_optimizer_steps,
        native_only_training=False, metadata_binding_only=True)


def validate_corrected_sidecar_binding(binding, approval, *, derivative_id,
                                       source_receipt, evaluation_protocol):
    require(isinstance(binding, dict), 'Actual distinct corrected_lora_binding required')
    expected = corrected_binding(approval, derivative_id=derivative_id,
        source_receipt=source_receipt, evaluation_protocol=evaluation_protocol,
        tensor_pairs=binding.get('tensor_pairs'),
        checkpoint_optimizer_steps=binding.get('checkpoint_optimizer_steps'),
        configured_optimizer_steps=binding.get('configured_optimizer_steps'))
    require(canonical(binding) == canonical(expected),
        'Corrected sidecar binding differs from independent saved training declaration')
    return expected


def validate_corrected_adapter(adapter, *, source_receipt, approved_derivative_id,
                               model_config, evaluation_protocol, approved_corrected):
    from .native_lora import validate_native_attention_headers
    from .native_thinking_eval import _strict_json
    from .vllm_policy import _adapter_metadata

    approval = validate_corrected_approval(approved_corrected)
    metadata = _adapter_metadata(adapter)
    require(metadata is not None, 'An actual local corrected attention-LoRA checkpoint is required')
    path = Path(metadata['path'])
    config_text = (path / 'training_config.json').read_text()
    config = _strict_json(config_text)
    require(isinstance(config, dict) and APPROVAL_FIELDS <= config.keys()
        and canonical({key: config[key] for key in APPROVAL_FIELDS}) == canonical(approval),
        'Saved corrected-training declaration differs from independent caller declaration')
    require(canonical(config.get('native_thinking')) == canonical(dict(enabled=False,
        protocol=PROTOCOL, origin=ORIGIN, requires_new_explicit_evaluation_binding=True)),
        'Corrected training must stay outside native accepted-only training')
    require(type(config.get('steps')) is int and config['steps'] > 0
        and type(config.get('lora_rank')) is int and config['lora_rank'] == 16
        and config.get('lora_alpha') == 32
        and config.get('target_modules') == ['q_proj', 'v_proj', 'o_proj']
        and config.get('lora_dropout') == 0 and config.get('warm_start') is False
        and config.get('training_method') == 'annotation-supervised grounded CUAD gold-final SFT; not RL'
        and canonical(config.get('loss_configuration')) == canonical(dict(version=MASK_VERSION,
            example_weighting='global-equal-example',
            token_weighting='per-example-mean-over-unmasked-JSON-and-EOS-tokens')),
        'Actual corrected training steps/rank differ')
    marker_text = (path / 'checkpoint_complete.json').read_text()
    marker = _strict_json(marker_text)
    require(isinstance(marker, dict) and marker.get('complete') is True
        and type(marker.get('optimizer_steps')) is int
        and 0 <= marker['optimizer_steps'] <= config['steps'],
        'Actual corrected checkpoint is incomplete or has invalid optimizer step')
    actual = metadata['config']
    require(actual['r'] == 16 and actual.get('lora_alpha') == 32
        and set(actual['target_modules']) == {'q_proj', 'v_proj', 'o_proj'}
        and actual.get('lora_dropout', 0) == 0,
        'Corrected adapter must use standard fresh rank16/alpha32 q/v/o LoRA')
    rank, pairs = validate_native_attention_headers(path, metadata, model_config)
    require(rank == 16 and _adapter_metadata(adapter) == metadata
        and (path / 'training_config.json').read_text() == config_text
        and (path / 'checkpoint_complete.json').read_text() == marker_text,
        'Corrected adapter or configuration changed during header validation')
    return metadata, corrected_binding(approval, derivative_id=approved_derivative_id,
        source_receipt=source_receipt, evaluation_protocol=evaluation_protocol,
        tensor_pairs=pairs, checkpoint_optimizer_steps=marker['optimizer_steps'],
        configured_optimizer_steps=config['steps'])
