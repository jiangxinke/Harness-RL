"""Optional frozen-loaded-base K3 penalty for active online thought segments.

This is a separate algorithm variant, not a correction to the original GRPO.
The reference is the same frozen base/precision with PEFT adapters disabled;
it is neither the sampled behavior policy nor the initial adapter policy.
"""
from __future__ import annotations

import math

import torch


VARIANT = "frozen-base-reference-k3-nonzero-advantage-thought-v1"
SOURCE = "same-loaded-PEFT-model.disable_adapter-no-grad"
ACTIVE_SCOPE = "original-nonzero-advantage-scored-thought-segments-only"


def validate_reference_kl_beta(beta):
    if type(beta) not in (int, float) or not math.isfinite(beta) or beta < 0:
        raise ValueError("Reference KL beta must be finite and nonnegative")
    return beta


def reference_kl_configuration(beta=0, *, base_model=None, source_commit=None):
    validate_reference_kl_beta(beta)
    enabled = beta > 0
    return dict(enabled=enabled, beta=beta,
        base_model=base_model if enabled else None,
        source_commit=source_commit if enabled else None,
        source=SOURCE if enabled else "not-read",
        reference="frozen-loaded-base-with-all-adapters-disabled" if enabled else None,
        active_scope=ACTIVE_SCOPE if enabled else "not-scored",
        json_tokens=False, zero_advantage_segments=False,
        estimator="expm1(ref-current)-(ref-current)" if enabled else None,
        denominator="group_size*max_steps*thought_tokens",
        loss_report="policy_loss; reference_kl_loss=beta*unweighted_K3; objective_loss",
        behavior_denominator="actual-sampled-thought-behavior-unchanged")


def validate_base_reference_model(model):
    """Refuse ambiguous/mutable base references before any reference forward.

    This deliberately supports the trainer's single standard q/v/o LoRA path.
    In particular, disabling a merged adapter may unmerge base weights; mixed
    enabled states are not faithfully restored by all PEFT versions.
    """
    configs = getattr(model, "peft_config", None)
    if (not configs or len(configs) != 1
            or not callable(getattr(model, "disable_adapter", None))
            or not callable(getattr(model, "get_model_status", None))):
        raise ValueError("Base reference requires one standard unmerged PEFT q/v/o LoRA adapter")
    config = next(iter(configs.values()))
    if (str(config.peft_type).split(".")[-1] != "LORA"
            or config.modules_to_save or config.is_prompt_learning
            or getattr(config, "bias", "none") != "none"
            or getattr(config, "use_dora", False)
            or getattr(config, "alora_invocation_tokens", None) is not None
            or set(config.target_modules or ()) != {"q_proj", "v_proj", "o_proj"}
            or getattr(config, "target_parameters", None)
            or getattr(config, "trainable_token_indices", None)
            or getattr(config, "layer_replication", None)):
        raise ValueError("Base reference requires one standard unmerged PEFT q/v/o LoRA adapter")
    status = model.get_model_status()
    if status.enabled is not True or status.merged_adapters:
        raise ValueError("Base reference requires uniformly enabled, unmerged adapters")
    named = list(model.named_parameters())
    # PEFT enable_adapter_layers() can reenable partially frozen LoRA tensors.
    # Reject that state so a reference read cannot change trainability.
    if (not any("lora_" in name for name, _ in named)
            or any(value.requires_grad != ("lora_" in name) for name, value in named)):
        raise ValueError("Frozen base reference requires all and only LoRA parameters trainable")


def frozen_base_logprobs(model, scorer, prompt_ids, completion_ids):
    """Score before creating the current-policy graph; always restore state.

    Call validate_base_reference_model once before use. Eval + no_grad avoids
    reference checkpoint graphs. The adapter context is fully exited before
    any current forward/backward, including non-reentrant checkpoint replay.
    Do not run this helper concurrently with generation or another forward.
    """
    was_training = model.training
    try:
        model.eval()
        with model.disable_adapter(), torch.no_grad():
            reference = scorer(model, prompt_ids, completion_ids).detach().clone()
            if (reference.ndim != 1 or reference.numel() != len(completion_ids)
                    or not reference.numel() or not torch.isfinite(reference).all()):
                raise FloatingPointError("Aligned finite frozen-base thought scores required")
        return reference
    finally:
        model.train(was_training)


def thought_reference_kl_loss(current, reference, *, group_size, max_steps, thought_tokens):
    """Unweighted K3 SUM over this thought, divided by the fixed GRPO budget.

    expm1 avoids exp(x)-1 cancellation near zero. No clamp changes the K3
    objective: extreme drift/overflow is rejected before backward/update.
    This sampled-token regularizer is not an exact vocabulary-wide KL.
    """
    if (current.ndim != 1 or reference.shape != current.shape or not current.numel()
            or reference.device != current.device):
        raise ValueError("Aligned nonempty one-dimensional base/current thought scores required")
    if reference.requires_grad:
        raise ValueError("Frozen-base reference scores must be detached")
    if not torch.isfinite(current).all() or not torch.isfinite(reference).all():
        raise FloatingPointError("Finite frozen-base and current thought scores required")
    if any(type(value) is not int or value < 1 for value in (group_size, max_steps, thought_tokens)):
        raise ValueError("Positive fixed objective budgets required")
    if current.numel() > thought_tokens:
        raise ValueError("Thought scores exceed the declared sampled thought budget")
    difference = reference.float() - current.float()
    penalty = torch.expm1(difference) - difference
    loss = penalty.sum() / (group_size * max_steps * thought_tokens)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite frozen-base K3 penalty; refusing online update")
    return loss
