"""Preserved declaration for original approved budgeted native TRAIN exports.

The original source76 and CF exports share this exact declaration. Keeping it
separate prevents a later inference backend fix from relabelling saved TRAIN.
No source receipt, caller approval, budget or tensor checks are bypassed.
"""
from .native_budget import NativeBudget, effective_sampling
from .eval_sampling import EvaluationSampling
from .native_thinking_eval import (PROTOCOL, THOUGHT_TOKENS, JSON_TOKENS,
    TOTAL_TOKENS, NATIVE_SUFFIX, OPEN_THINK, END_THINK, validate_source_receipt)

CUAD_STRUCTURED_SCHEMA_VERSION = "cuad-public-document-actions-span-confidence-review-v1"

def cuad_structured_configuration(enabled=False):
    if type(enabled) is not bool:
        raise ValueError("cuad_structured_actions must be a boolean")
    return {"enabled": enabled,
            "schema_version": CUAD_STRUCTURED_SCHEMA_VERSION if enabled else None,
            "backend": "xgrammar" if enabled else None,
            "final_review_max_words": 40 if enabled else None}

def native_configuration(source_receipt, *, budget=None):
    budget = NativeBudget() if budget is None else budget
    result = {'enabled': True, 'encode_messages_requested_enable_thinking': False, 'native_template_ignores_enable_thinking': True, 'effective_enable_thinking': True, 'native_suffix': NATIVE_SUFFIX, 'open_think_id': OPEN_THINK, 'end_think_id': END_THINK, 'thought_max_tokens': THOUGHT_TOKENS, 'json_max_tokens': JSON_TOKENS, 'total_max_new_tokens': TOTAL_TOKENS, 'stage1_structured_outputs': False, 'reasoning_parser': '', 'stage2_original_schema': True, 'stage2_allow_eos': True, 'stage2_requires_terminal_eos': True, 'stage2_stop_reason_matches_terminal_eos': True, 'stage2_none_stop_reason_only_for_primary_eos': True, 'reject_duplicate_json_keys': True, 'stage2_validate_complete_json_and_schema': True, 'public_json_decode': 'Remove terminal EOS ID only; preserve all other generated text', 'same_real_prompt_and_completion_ids': True, 'phase2_retemplate': False, 'forced_close': False, 'thought_json_extraction': False, 'retry': False, 'inference_only': True, 'comparable_to_old_4096_protocol': False}
    result.update(protocol=PROTOCOL, source_receipt=validate_source_receipt(source_receipt),
                  benchmarks=["widesearch", "cuad"],
                  cuad_structured=cuad_structured_configuration(True), sidecar="native_thinking_traces_rank{rank}.jsonl",
                  fresh_generation_only=True, sampling_profile="greedy-neutral")
    if not budget.legacy:
        result.update(protocol=budget.protocol, thought_max_tokens=budget.thought_tokens,
            total_max_new_tokens=budget.total_tokens,
            thought_sampling_profile=budget.thought_sampling_profile,
            thought_sampling=budget.thought_sampling.to_dict(), json_sampling=EvaluationSampling().to_dict(),
            thought_effective_sampling=effective_sampling(budget.thought_sampling),
            json_effective_sampling=effective_sampling(EvaluationSampling()),
            sampling_profile="phase-specific-thinking" if budget.thought_sampling_profile == "thinking" else "greedy-neutral",
            phase_sampling_seed="same explicit seed per physical phase request; no shared RNG history")
    return result
