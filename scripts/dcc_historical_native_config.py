"""Exact frozen cf TRAIN native configuration, independent of inference overlays.

The builder body is copied from the explicitly approved historical TRAIN source;
no value is inferred from trace declarations. Only its fixed source may use it.
"""
from slime_plugins.dcc.native_budget import NativeBudget, effective_sampling
from slime_plugins.dcc.eval_sampling import EvaluationSampling

HISTORICAL_TRAIN_SOURCE = "cf89170a00eb7c25bf5de038558c479228a87587"
PROTOCOL = "native-thinking2507-real-close-two-stage-4096-4096-v2"
OPEN_THINK, END_THINK = 151667, 151668
THOUGHT_TOKENS, JSON_TOKENS, TOTAL_TOKENS = 4096, 4096, 8192
NATIVE_SUFFIX = "<|im_start|>assistant\n<think>\n"


def validate_source_receipt(value):
    from dcc_native_sft import validate_source_receipt as validate
    return validate(value, HISTORICAL_TRAIN_SOURCE)


def cuad_structured_configuration(enabled=False):
    # Exact original cf declaration; newer inference-only schema metadata must
    # not change this historical audit contract through an imported function.
    if type(enabled) is not bool:
        raise ValueError("CUAD structured flag must be a boolean")
    return {"enabled": enabled,
            "schema_version": "cuad-public-document-actions-span-confidence-review-v1" if enabled else None,
            "backend": "xgrammar" if enabled else None,
            "final_review_max_words": 40 if enabled else None}


def historical_native_configuration(source_receipt, *, budget=None):
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
