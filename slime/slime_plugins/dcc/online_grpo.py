"""Thought-only online policy gradient for the optional native HF runtime.

The thought is sampled from the raw model; the schema-constrained JSON is a
deterministic environment continuation. Its tokens must never receive a raw
policy loss. Model failures retain sampled thoughts and zero reward.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import math

import torch

from . import web_train_thought_budget_close as train_close
from . import cuad_train_thought_budget_close as cuad_close
from .native_hf_context import (unreliable_context_or_environment_group,
                                validated_thought_step)


ALGORITHM = "online-native-HF-thought-only-GRPO-v2"


@dataclass(frozen=True)
class ThoughtSegment:
    prompt_ids: list[int]
    completion_ids: list[int]
    thought_logprobs: list[float]
    web_train_thought_budget_close: dict | None = None
    forced_thought_positions: tuple[int, ...] = ()


@dataclass(frozen=True)
class CuadBudgetClosedThoughtSegment(ThoughtSegment):
    """Transient shared-group segment; joint_group_signal retains this identity."""
    cuad_train_thought_budget_close: dict | None = None
    cuad_forced_thought_positions: tuple[int, ...] = ()


@dataclass(frozen=True)
class GroupSignal:
    segments: list[list[ThoughtSegment]]
    advantages: list[float]
    rewards: list[float]
    reward_std: float
    discarded_reason: str | None = None
    advantage_normalization: str = "std"


def group_signal(results, *, group_size, max_steps, thought_tokens, advantage_normalization="std",
                 _joint_cuad_policy=False):
    """Keep model failures in the group, reject unreliable environment groups.

    Do not use EpisodeResult.trainable_steps: evaluation protocol failures are
    intentionally excluded there, but excluding them from this stochastic
    policy objective would condition the baseline on successful generations.
    ``std`` preserves (reward - mean) / (population_std + 1e-4); ``centered``
    uses reward - mean. The reported reward standard deviation is descriptive
    in both modes and is never used to renormalize a centered signal.
    """
    if advantage_normalization not in ("std", "centered"):
        raise ValueError("Advantage normalization must be std or centered")
    if len(results) != group_size or group_size < 2:
        raise ValueError("A complete same-task group of at least two episodes is required")
    if any(type(v) is not int or v < 1 for v in (max_steps, thought_tokens)):
        raise ValueError("Positive fixed thought/turn budgets required")
    if len({(r.task_id, r.benchmark, r.split) for r in results}) != 1:
        raise ValueError("All episodes in a GRPO group must use the same task")
    if any(r.split != "train" for r in results):
        raise ValueError("Only TRAIN episodes may provide policy gradients")
    rewards = [float(r.reward) for r in results]
    if any(not math.isfinite(r) or not 0 <= r <= 1 for r in rewards):
        raise ValueError("Finite rewards in [0, 1] required")
    if unreliable_context_or_environment_group(results):
        return GroupSignal([[] for _ in results], [0.0] * group_size, rewards, 0.0,
                           "unreliable_environment_or_context_group", advantage_normalization)
    allowed = {"completed", "max_steps", "invalid_answer", "evidence_failure",
               "model_protocol_failure", "context_budget_failure"}
    all_segments = []
    for result in results:
        if result.status not in allowed or len(result.steps) > max_steps:
            raise ValueError("Unknown episode status or excess model turns")
        if result.status != "completed" and result.reward != 0:
            raise ValueError("Unsuccessful model episodes must retain zero reward")
        segments = []
        for step in result.steps:
            prompt, thought, behavior = validated_thought_step(step, thought_tokens=thought_tokens)
            if thought:
                fields = train_close.step_fields(step, thought_tokens=thought_tokens)
                if fields and result.benchmark != "widesearch":
                    raise ValueError("Singleton thought control supports newly owned Web TRAIN only")
                cuad_fields = cuad_close.step_fields(step, thought_tokens=thought_tokens)
                if cuad_fields:
                    if not _joint_cuad_policy or result.benchmark != "cuad":
                        raise ValueError("CUAD singleton thought control requires joint JSON group construction")
                    segments.append(CuadBudgetClosedThoughtSegment(prompt, thought, behavior,
                        **{cuad_close.FIELD: cuad_fields[cuad_close.FIELD],
                           cuad_close.POSITIONS: tuple(cuad_fields[cuad_close.POSITIONS])}))
                    continue
                segments.append(ThoughtSegment(prompt, thought, behavior,
                    **({train_close.FIELD: fields[train_close.FIELD],
                        train_close.POSITIONS: tuple(fields[train_close.POSITIONS])} if fields else {})))
        if not segments:
            return GroupSignal([[] for _ in results], [0.0] * group_size, rewards, 0.0,
                               "missing_sampled_thought_group", advantage_normalization)
        all_segments.append(segments)
    # Equal represented rewards must be exactly zero even at float32 values
    # such as 0.1, rather than a subtraction/epsilon amplification artifact.
    if max(rewards) == min(rewards):
        return GroupSignal(all_segments, [0.0] * group_size, rewards, 0.0,
                           advantage_normalization=advantage_normalization)
    mean = math.fsum(rewards) / group_size
    std = math.sqrt(math.fsum((r - mean) ** 2 for r in rewards) / group_size)
    advantages = ([(r - mean) / (std + 1e-4) for r in rewards]
                  if advantage_normalization == "std" else [r - mean for r in rewards])
    return GroupSignal(all_segments, advantages, rewards, std,
                       advantage_normalization=advantage_normalization)


def thought_policy_loss(current, old, advantage, *, group_size, max_steps, thought_tokens, clip=0.2,
                        policy_mask=None):
    """Token-clipped GRPO with a fixed budget denominator.

    No division by observed response length or by observed successful turns:
    short/failed thoughts and longer successful thoughts share the same weight.
    """
    if current.ndim != 1 or old.shape != current.shape or not current.numel():
        raise ValueError("Aligned nonempty one-dimensional thought logprobs required")
    if not torch.isfinite(current).all() or not torch.isfinite(old).all():
        raise FloatingPointError("Finite actual behavior and current thought scores required")
    if old.requires_grad or not math.isfinite(advantage) or not 0 < clip < 1:
        raise ValueError("Detached old scores, finite advantage and clip in (0,1) required")
    if any(type(v) is not int or v < 1 for v in (group_size, max_steps, thought_tokens)):
        raise ValueError("Positive fixed objective budgets required")
    if current.numel() > thought_tokens:
        raise ValueError("Thought scores exceed the declared sampled thought budget")
    ratio = (current - old).exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    if policy_mask is not None:
        if policy_mask.shape != current.shape or policy_mask.dtype != torch.bool or policy_mask.device != current.device:
            raise ValueError("Aligned boolean deterministic-slot policy mask required")
        surrogate = surrogate.masked_fill(~policy_mask, 0)
    return -surrogate.sum() / (group_size * max_steps * thought_tokens)


def wait_for_update_ranks(*, world_size=1, distributed=None, control_group=None,
                          synchronize=None, timeout_seconds=7200):
    """Wait on the CPU before submitting any gradient collective to CUDA.

    Rollout and replay lengths differ across ranks, especially for mixed Web
    groups. An early rank must not leave a GPU collective waiting throughout
    another rank's generation. This rendezvous changes no gradient arithmetic
    and retains a finite timeout and NCCL's existing watchdog monitoring.
    """
    if (type(world_size) is not int or world_size < 1
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("Positive world size and finite readiness timeout required")
    if world_size == 1:
        return
    if (distributed is None or control_group is None or not callable(synchronize)
            or distributed.get_backend(control_group) != "gloo"
            or distributed.get_world_size(control_group) != world_size):
        raise ValueError("A full-world Gloo readiness group and device synchronization are required")
    synchronize()
    distributed.monitored_barrier(group=control_group,
        timeout=timedelta(seconds=timeout_seconds), wait_all_ranks=True)


def apply_policy_update(parameters, optimizer, *, world_size=1, distributed=None, max_grad_norm=1.0):
    """Average replicas and skip Adam completely when the global gradient is zero.

    Adam momentum can change weights on a zero-signal batch if step() is called;
    those changes must not be recorded as new policy-gradient learning.
    """
    if not parameters or world_size < 1 or not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("Trainable parameters, valid world size and gradient bound required")
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        if world_size > 1:
            distributed.all_reduce(parameter.grad, op=distributed.ReduceOp.SUM)
            parameter.grad.div_(world_size)
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    if not torch.isfinite(norm):
        raise FloatingPointError("Non-finite online gradient; checkpoint refused")
    applied = bool(norm.item() > 0)
    if applied:
        optimizer.step()
    return norm.item(), applied
