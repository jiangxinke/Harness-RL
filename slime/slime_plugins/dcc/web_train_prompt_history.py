"""Explicit, default-off whole-turn history for raw on-policy Web TRAIN.

Plans are recomputed from the owner's complete request, actual tokenizer and
runtime budgets. Neither stored reports nor reconstructed behavior scores are
inputs to training replay.
"""
from __future__ import annotations

from copy import deepcopy

from .prompt_history import PromptHistoryConfig, fit_prompt_history

VERSION = "hf-native-web-whole-turns-training-v1"
FIELDS = ("native_web_train_prompt_history", "prompt_history", "prompt_history_original_messages")
FAILURE_FIELD = "native_web_train_prompt_history_failure"
BUDGET_FIELDS = ("max_prompt_tokens", "max_model_len", "thought_tokens", "json_tokens")
NATIVE_SUFFIX = "<|im_start|>assistant\n<think>\n"


def web_train_history_from_arguments(args):
    enabled = getattr(args, "web_train_prompt_history_whole_turns", False)
    # Older programmatic configurations can explicitly leave optional flags None.
    enabled = False if enabled is None else enabled
    keep = getattr(args, "web_train_prompt_history_keep_recent_turns", None)
    config = PromptHistoryConfig(enabled, 1 if keep is None else keep)
    if not config.enabled:
        if keep is not None:
            raise ValueError("--web-train-prompt-history-keep-recent-turns requires --web-train-prompt-history-whole-turns")
        return None
    if not getattr(args, "enable_online_web", False):
        raise ValueError("Web TRAIN prompt history requires --enable-online-web")
    return config


def web_train_history_configuration(config, *, thought_policy=None):
    if not isinstance(config, PromptHistoryConfig) or not config.enabled:
        raise ValueError("An enabled Web TRAIN PromptHistoryConfig is required")
    result = dict(version=VERSION, training_only=True, supported_benchmark="widesearch",
        prompt_history=config.to_dict(), original_request="persist-full-and-recompute-at-group-close",
        prompt_encoding="unchanged-checkpoint-chat-template-with-native-open-think-suffix",
        behavior_logprobs="actual-streamed-raw-sampled-thought-scores",
        replay="saved-actual-prompt-and-completion-IDs", policy_gradient="actual-thought-only",
        objective_denominator="unchanged-fixed-group-step-thought-budget",
        no_fit="original-full-request-native-preflight-and-context-sampled-prefix-v2")
    if thought_policy is not None:
        from . import web_train_thought_budget_close as train_close
        train_close.validate_configuration(thought_policy)
        result.update(version=VERSION+"+"+train_close.VARIANT,
            behavior_logprobs="actual-streamed-raw-prefix-and-singleton-close-scores",
            **{train_close.FIELD: deepcopy(thought_policy)})
    return result


def web_train_history_plan(tokenizer, messages, *, config, max_prompt_tokens,
                           max_model_len, thought_tokens, json_tokens):
    web_train_history_configuration(config)
    if any(type(value) is not int or value < 1 for value in (thought_tokens, json_tokens)):
        raise ValueError("Both complete native generation phase budgets are required")
    plan = fit_prompt_history(tokenizer, messages, max_prompt_tokens=max_prompt_tokens,
        max_new_tokens=thought_tokens + json_tokens, max_model_len=max_model_len, config=config)
    suffix = list(tokenizer.encode(NATIVE_SUFFIX, add_special_tokens=False))
    if (not suffix or suffix.count(151667) != 1 or 151668 in suffix
            or plan.prompt_ids[-len(suffix):] != suffix):
        raise ValueError("Web TRAIN history must retain the real native open-thinking suffix")
    return plan


def web_train_history_fields(original_messages, report, config, *, thought_policy=None):
    return dict(native_web_train_prompt_history=web_train_history_configuration(config, thought_policy=thought_policy),
        prompt_history=deepcopy(report), prompt_history_original_messages=deepcopy(original_messages))


def web_train_history_failure_fields(original_messages, failure, config, *, thought_policy=None):
    return dict(**web_train_history_fields(original_messages, failure.diagnostics, config, thought_policy=thought_policy),
        reason=failure.reason, original_prompt_ids=list(failure.original_prompt_ids),
        generation_submitted=False)


def web_train_history_budgets(policy):
    return {key: getattr(policy, key) for key in BUDGET_FIELDS}
