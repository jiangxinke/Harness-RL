"""Dense tree credit and answer-aware reflect self-distillation helpers."""

from __future__ import annotations

import math
import os
import re
from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

TEACHER_SYSTEM_PROMPT = """
You are the private EMA teacher for a reflective search policy. The reference
answer is provided only so you can judge whether the student's progress
assessment is directionally sound.

PRIVATE REFERENCE ANSWER:
{answer}

Score the student's existing continuation; do not rewrite it. The reference is
privileged and must never be exposed to the student. In particular, inside any
<reflect>...</reflect> block you must NEVER quote, name, paraphrase, hint at, or
claim to know the reference answer. A good reflection discusses evidence, gaps,
contradictions, and the next verification step without revealing the answer.
Continue to obey the original XML action protocol.
""".strip()


def format_reference_answer(label: Any) -> str:
    if isinstance(label, (list, tuple)):
        return " | ".join(str(value) for value in label)
    return str(label)


def build_teacher_prompt(tokenizer: Any, query: str, label: Any) -> str:
    """Build the private prompt used only by the EMA teacher forward pass."""
    return tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": TEACHER_SYSTEM_PROMPT.format(answer=format_reference_answer(label)),
            },
            {"role": "user", "content": query},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def normalized_answer(value: Any) -> str:
    text = re.sub(r"[^\w\s]", " ", str(value).lower())
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def reflection_leaks_answer(reflection: str, label: Any) -> bool:
    """Conservative audit flag; it never changes the private teacher prompt."""
    reflection_norm = normalized_answer(reflection)
    references = label if isinstance(label, (list, tuple)) else [label]
    return any(reference_norm and reference_norm in reflection_norm for reference in references if reference is not None for reference_norm in [normalized_answer(reference)])


def dense_tree_advantages(args: Namespace, rollout_data: dict[str, Any]) -> None:
    """Use response-aligned process rewards directly as token advantages.

    ``position_rewards`` already carries tree credit for every sampled model
    token. Reference-policy KL reward shaping remains supported and is applied
    position-wise when ``--kl-coef`` is non-zero.
    """
    position_rewards = rollout_data.get("position_rewards")
    if position_rewards is None:
        raise ValueError("dense tree training requires position_rewards")
    kl = rollout_data["kl"]
    advantages = []
    for reward, token_kl in zip(position_rewards, kl, strict=True):
        reward = reward.to(device=token_kl.device, dtype=torch.float32)
        advantages.append(reward - float(args.kl_coef) * token_kl)
    rollout_data["advantages"] = advantages
    rollout_data["returns"] = [advantage.clone() for advantage in advantages]


def low_variance_reverse_kl(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    clip: float = 10.0,
) -> torch.Tensor:
    """Non-negative k3 estimator for reverse KL on student samples."""
    if not math.isfinite(clip) or clip <= 0:
        raise ValueError("reverse-KL clip must be finite and positive")
    # Clipping after exp can leave inf in autograd and produce 0 * inf = NaN.
    # Both bounds are already beyond the clipped region of exp(-x)-1+x,
    # so bounding x first preserves the objective and its saturated gradient.
    ratio = student_log_probs.float() - teacher_log_probs.float()
    bounded = ratio.clamp(min=-1.0 - math.log1p(clip), max=clip + 1.0)
    return (torch.expm1(-bounded) + bounded).clamp(min=0.0, max=clip)


def ema_tau(step: int, total_steps: int, tau_base: float, tau_final: float) -> float:
    """Cosine schedule from ``tau_base`` to the slower ``tau_final``."""
    if total_steps <= 1:
        # A one-step smoke run still needs to exercise a real EMA update.  It
        # is also the natural value of the cosine schedule at step zero.
        return tau_base
    progress = min(max(step, 0) / (total_steps - 1), 1.0)
    return tau_final - (tau_final - tau_base) * 0.5 * (1.0 + math.cos(math.pi * progress))


@dataclass(frozen=True)
class ReflectUpdateObjective:
    """One real optimizer update within a Reflect outer step."""

    name: str
    policy_weight: float
    distill_weight: float
    phase: float


def build_update_objectives(args: Namespace) -> list[ReflectUpdateObjective]:
    """Build the ordered optimizer-update plan for one rollout step.

    Joint performs one fused policy/distillation update. Fast-Slow reuses the
    same rollout batch for several *real* distillation-only optimizer updates,
    followed by exactly one verifier (policy/reward) optimizer update. This is deliberately
    an intra-step schedule; it never assigns different objectives to different
    rollout ids.
    """
    mode = os.environ.get("REFLECT_TRAIN_MODE", getattr(args, "reflect_train_mode", "joint"))
    if mode == "joint":
        return [
            ReflectUpdateObjective(
                name="joint",
                policy_weight=1.0,
                distill_weight=float(getattr(args, "reflect_distill_weight", 0.5)),
                phase=0.0,
            )
        ]
    if mode == "distill_only":
        weight = float(getattr(args, "reflect_distill_weight", 0.5))
        if weight <= 0:
            raise ValueError(f"reflect_distill_weight must be positive for distill_only, got {weight}")
        return [
            ReflectUpdateObjective(
                name="distill_only",
                policy_weight=0.0,
                distill_weight=weight,
                phase=3.0,
            )
        ]
    if mode == "policy_only":
        return [
            ReflectUpdateObjective(
                name="policy_only",
                policy_weight=1.0,
                distill_weight=0.0,
                phase=4.0,
            )
        ]
    if mode != "fast_slow":
        raise ValueError(f"unknown reflect_train_mode: {mode!r}")

    reflect_updates = int(getattr(args, "reflect_fast_updates_per_step", 3))
    if reflect_updates < 1:
        raise ValueError(f"reflect_fast_updates_per_step must be at least 1, got {reflect_updates}")
    reflect_weight = float(getattr(args, "reflect_fast_distill_weight", 1.0))
    if reflect_weight <= 0:
        raise ValueError(f"reflect_fast_distill_weight must be positive, got {reflect_weight}")
    verifier_updates = int(getattr(args, "reflect_verifier_updates_per_step", 1))
    if verifier_updates != 1:
        raise ValueError(f"reflect_verifier_updates_per_step must be exactly 1 for Fast-Slow, got {verifier_updates}")

    objectives = [
        ReflectUpdateObjective(
            name=f"reflect_{index + 1}",
            policy_weight=0.0,
            distill_weight=reflect_weight,
            phase=1.0,
        )
        for index in range(reflect_updates)
    ]
    objectives.append(
        ReflectUpdateObjective(
            name="verifier",
            policy_weight=1.0,
            distill_weight=0.0,
            phase=2.0,
        )
    )
    return objectives


def materialize_update_objective(rollout_data: dict[str, Any], objective: ReflectUpdateObjective) -> None:
    """Write one objective's scalar weights into every compact sample."""
    count = len(rollout_data["response_lengths"])
    device = rollout_data["loss_masks"][0].device
    rollout_data["reflect_policy_weights"] = [torch.tensor(objective.policy_weight, dtype=torch.float32, device=device) for _ in range(count)]
    rollout_data["reflect_distill_weights"] = [torch.tensor(objective.distill_weight, dtype=torch.float32, device=device) for _ in range(count)]
    rollout_data["reflect_train_phases"] = [torch.tensor(objective.phase, dtype=torch.float32, device=device) for _ in range(count)]


def schedule_rollout_loss(args: Namespace, rollout_id: int, rollout_data: dict[str, Any]) -> None:
    """Materialize the first objective before logging and actor training.

    ``MegatronTrainRayActor`` advances through the complete ordered plan and
    calls :func:`materialize_update_objective` before every optimizer update.
    The ``rollout_id`` argument remains part of slime's plugin contract but no
    longer controls Fast-Slow scheduling.
    """
    del rollout_id
    materialize_update_objective(rollout_data, build_update_objectives(args)[0])


def _reflect_distill_loss(
    args: Namespace,
    batch: dict[str, Any],
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean
    from slime.backends.megatron_utils.loss import get_log_probs_and_entropy

    _, output = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        with_entropy=False,
    )
    student = torch.cat(output["log_probs"], dim=0)
    teacher = torch.cat(batch["teacher_log_probs"], dim=0).to(student.device)
    masks = batch["reflect_masks"]
    mask = torch.cat(masks, dim=0).to(student.device, dtype=torch.float32)
    if student.shape != teacher.shape or student.shape != mask.shape:
        raise ValueError(f"reflect distillation alignment mismatch: student={student.shape}, teacher={teacher.shape}, mask={mask.shape}")
    token_kl = low_variance_reverse_kl(
        student,
        teacher,
        clip=float(getattr(args, "reflect_distill_kl_clip", 10.0)),
    )
    reducer = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        masks,
        batch.get("rollout_reflect_mask_sums"),
        args.calculate_per_token_loss,
    )
    loss = reducer(token_kl)
    mean_kl = reducer(token_kl.detach())
    if mask.numel() == 0 or mask.sum() == 0:
        loss = loss + 0.0 * logits.sum()
    return loss, mean_kl


def reflect_joint_loss(
    args: Namespace,
    batch: dict[str, Any],
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Shared implementation for both joint and fast-slow schedules."""
    from slime.backends.megatron_utils.loss import policy_loss_function

    policy_loss, reported = policy_loss_function(args, batch, logits, sum_of_sample_mean)
    distill_loss, reflect_kl = _reflect_distill_loss(args, batch, logits)
    policy_weight = torch.stack(batch["reflect_policy_weights"]).mean().to(logits.device)
    configured_distill_weight = torch.stack(batch["reflect_distill_weights"]).mean().to(logits.device)
    # A micro-batch can legitimately contain only search/answer nodes. Do not
    # report a non-zero effective distillation weight when no reflect token can
    # contribute on that micro-batch.
    has_reflect = torch.stack([mask.sum().to(logits.device) for mask in batch["reflect_masks"]]).sum() > 0
    distill_weight = configured_distill_weight * has_reflect.to(torch.float32)
    phase = torch.stack(batch["reflect_train_phases"]).mean().to(logits.device)
    loss = policy_weight * policy_loss + distill_weight * distill_loss

    # Constant schedule values must be reported as per-rollout contributions.
    # A tree can emit several compact samples in different micro-batches;
    # logging the same constant once per compact sample would misleadingly turn
    # weight 1.0 into N. These masses sum to one over every rollout.
    policy_metric_mass = sum(mask.sum().to(logits.device, dtype=torch.float32) / torch.clamp_min(denom.to(logits.device, dtype=torch.float32), 1.0) for mask, denom in zip(batch["loss_masks"], batch["rollout_mask_sums"], strict=True))
    reflect_metric_mass = sum(
        mask.sum().to(logits.device, dtype=torch.float32) / torch.clamp_min(denom.to(logits.device, dtype=torch.float32), 1.0)
        for mask, denom in zip(
            batch["reflect_masks"],
            batch["rollout_reflect_mask_sums"],
            strict=True,
        )
    )
    reported = dict(reported)
    reported.update(
        loss=loss.detach().clone(),
        reflect_distill_loss=distill_loss.detach().clone(),
        reflect_reverse_kl=reflect_kl.detach().clone(),
        reflect_policy_weight=(policy_weight * policy_metric_mass).detach().clone(),
        reflect_distill_weight=(distill_weight * reflect_metric_mass).detach().clone(),
        reflect_train_phase=(phase * policy_metric_mass).detach().clone(),
    )
    return loss, reported
