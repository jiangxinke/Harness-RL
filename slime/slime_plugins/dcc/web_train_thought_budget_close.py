"""Default-off Web TRAIN control policy, separate from evaluation decoding.

Before the last thought budget slot the policy is the unchanged raw softmax.
At that slot it is a parameter-independent singleton on the actual close ID.
Only newly captured/owned TRAIN completions may carry this identity.
"""
from __future__ import annotations

from copy import deepcopy
import math

END_THINK = 151668
FIELD = "web_train_thought_budget_close"
POSITIONS = "forced_thought_positions"
PROTOCOL = "hf-native-web-thought-budget-close-training-v1"
SOURCE = "streaming-generation-raw-prefix-singleton-close-v1"
VARIANT = "web-train-thought-budget-singleton-close-v1"


def configuration(thought_tokens):
    if type(thought_tokens) is not int or thought_tokens < 1:
        raise ValueError("Positive fixed thought budget required")
    return dict(version=VARIANT, enabled=True, training_only=True,
        supported_benchmark="widesearch", base_sampling="raw-T1-top_p1-top_k0",
        thought_tokens=thought_tokens, forced_position=thought_tokens - 1,
        position_indexing="zero-based-generated-thought", forced_token_id=END_THINK,
        forced_support="parameter-independent-singleton", forced_logprob=0,
        forced_policy_gradient=False, extra_tokens=0,
        json_sampling="unchanged-greedy-xgrammar", json_policy_loss=False,
        objective_denominator="group_size*max_steps*thought_tokens")


def from_arguments(args):
    enabled = getattr(args, "web_train_thought_budget_close", False)
    if type(enabled) is not bool:
        raise ValueError("Web TRAIN thought budget close must be an explicit boolean")
    if enabled and (not getattr(args, "enable_online_web", False)
            or getattr(args, "joint_json_policy", False)
            or getattr(args, "policy_surrogate", "token-grpo") != "token-grpo"):
        raise ValueError("Web TRAIN thought budget close requires online Web thought-only token-GRPO")
    return configuration(args.thought_tokens) if enabled else None


def validate_configuration(profile, *, thought_tokens=None):
    """Compare scalar types as well as values in the public policy identity."""
    if type(profile) is not dict:
        raise ValueError("Exact Web TRAIN thought budget close policy required")
    expected = configuration(profile.get("thought_tokens") if thought_tokens is None else thought_tokens)
    if (profile.keys() != expected.keys()
            or any(type(profile[key]) is not type(value) or profile[key] != value
                   for key, value in expected.items())):
        raise ValueError("Exact Web TRAIN thought budget close policy required")
    return deepcopy(expected)


def validate_fields(profile, positions, thought, behavior, *, thought_tokens):
    """Validate a newly captured singleton policy, never synthesize old scores."""
    validate_configuration(profile, thought_tokens=thought_tokens)
    if type(positions) is not list:
        raise ValueError("Exact Web TRAIN thought budget close policy/positions required")
    if (type(thought) is not list or type(behavior) is not list
            or any(type(token) is not int or token < 0 for token in thought)
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   or value > 1e-6 for value in behavior)):
        raise ValueError("Actual aligned token IDs and finite captured behavior required")
    expected = [thought_tokens - 1] if len(thought) == thought_tokens else []
    if (positions != expected or any(type(i) is not int for i in positions)
            or len(thought) > thought_tokens or len(behavior) != len(thought)
            or (positions and (thought[-1] != END_THINK or behavior[-1] != 0))):
        raise ValueError("Actual singleton close token, position and captured logprob differ")
    return {FIELD: deepcopy(profile), POSITIONS: list(positions)}


def step_fields(step, *, thought_tokens):
    """Old raw steps stay raw; a new source requires its own complete identity."""
    profile = getattr(step, FIELD, None)
    positions = getattr(step, POSITIONS, None)
    from .hf_native_policy import BudgetClosedNativeHFCompletion
    if profile is None and positions is None:
        if (isinstance(step, BudgetClosedNativeHFCompletion)
                or getattr(step, "thought_logprobs_source", None) == SOURCE):
            raise ValueError("Singleton behavior source requires its captured policy identity")
        return {}
    if getattr(step, "thought_logprobs_source", None) != SOURCE:
        raise ValueError("Old raw/evaluation traces cannot become singleton TRAIN behavior")
    if not isinstance(step, BudgetClosedNativeHFCompletion):
        raise ValueError("New singleton TRAIN completion type required")
    count = getattr(step, "thought_token_count", None)
    if type(count) is not int:
        raise ValueError("Actual thought boundary required")
    return validate_fields(profile, positions, step.completion_ids[:count],
        step.thought_logprobs, thought_tokens=thought_tokens)


def policy_mask(segment, values):
    """Keep the original fixed budget while excluding deterministic numerator slots."""
    import torch
    positions = getattr(segment, POSITIONS, ())
    mask = torch.ones_like(values, dtype=torch.bool)
    if positions:
        mask[list(positions)] = False
    return mask


def scorer_for(segment, raw_scorer):
    """Compute log q, not raw model likelihood at the controlled position.

    The actual tokens remain in the decoder input. The same wrapper is used
    for current, diagnostic replay and adapter-disabled frozen-base replay.
    """
    positions = getattr(segment, POSITIONS, ())
    if not positions:
        return raw_scorer

    def score(model, prompt_ids, completion_ids):
        import torch
        if prompt_ids != segment.prompt_ids or completion_ids != segment.completion_ids:
            raise ValueError("Singleton replay must use its exact captured segment")
        raw = raw_scorer(model, prompt_ids, completion_ids)
        return torch.where(policy_mask(segment, raw), raw, torch.zeros_like(raw))
    return score
