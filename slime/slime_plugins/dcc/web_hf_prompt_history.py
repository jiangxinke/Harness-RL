"""Default-off native-HF Web evaluation history provenance.

The existing whole-turn algorithm and checkpoint chat template are reused
unchanged. Stored omission reports are diagnostics, never trusted replay input.
"""
from __future__ import annotations

from copy import deepcopy

from .prompt_history import PromptHistoryConfig, fit_prompt_history

VERSION = "hf-native-web-whole-turns-evaluation-v1"
FIELDS = ("native_web_prompt_history", "prompt_history", "prompt_history_original_messages")
NATIVE_SUFFIX = "<|im_start|>assistant\n<think>\n"


def native_web_history_configuration(config):
    if not isinstance(config, PromptHistoryConfig) or not config.enabled:
        raise ValueError("An enabled native HF Web evaluation history configuration is required")
    return dict(version=VERSION, evaluation_only=True, supported_benchmark="widesearch",
        prompt_history=config.to_dict(), original_request="persist-full-and-replay-with-same-tokenizer",
        prompt_encoding="unchanged-checkpoint-chat-template-with-native-open-think-suffix",
        policy_gradient="disabled-evaluation-only")


def native_web_history_plan(tokenizer, messages, *, config, max_prompt_tokens,
                            max_model_len, thought_tokens, json_tokens):
    native_web_history_configuration(config)
    if any(type(value) is not int or value < 1 for value in (thought_tokens, json_tokens)):
        raise ValueError("Both complete native generation phase budgets are required")
    plan = fit_prompt_history(tokenizer, messages, max_prompt_tokens=max_prompt_tokens,
        max_new_tokens=thought_tokens + json_tokens, max_model_len=max_model_len, config=config)
    suffix = list(tokenizer.encode(NATIVE_SUFFIX, add_special_tokens=False))
    if (not suffix or suffix.count(151667) != 1 or 151668 in suffix
            or plan.prompt_ids[-len(suffix):] != suffix):
        raise ValueError("Whole-turn history must retain the checkpoint's real native open-thinking suffix")
    return plan


def native_web_history_fields(original_messages, report, config):
    return dict(native_web_prompt_history=native_web_history_configuration(config),
        prompt_history=deepcopy(report), prompt_history_original_messages=deepcopy(original_messages))
