"""Opt-in episode-level GSPO surrogate with the original fixed token budget.

The geometric-mean importance ratio is an intentional surrogate, not an
unbiased full-trajectory likelihood ratio. Old scores are always the actual
sampled behavior scores, including grammar normalization for joint JSON.
"""
from __future__ import annotations

import math

import torch


VARIANT = "episode-GSPO-fixed-budget-v1"
POLICY_SURROGATES = ("token-grpo", "episode-gspo")


def validate_policy_surrogate(mode, reference_kl_beta=0):
    if mode not in POLICY_SURROGATES:
        raise ValueError("Policy surrogate must be token-grpo or episode-gspo")
    if mode == "episode-gspo" and reference_kl_beta != 0:
        raise ValueError("episode-gspo currently requires reference_kl_beta=0")


def gspo_configuration():
    return dict(mode="episode-gspo", version=VARIANT,
        ratio="exp(sum(current_logprob-actual_behavior_logprob)/actual_episode_scored_tokens)",
        clip_unit="whole-episode-across-all-sampled-turns",
        episode_weight="actual_episode_scored_tokens/fixed_objective_denominator",
        token_count="all-actual-scored-positions-including-singleton-grammar-and-real-stops",
        gradient_accumulation="single-replay-per-segment-episode-adapter-buffer",
        extra_adapter_gradient_buffers=1, teacher_forced_replay_is_behavior=False,
        unbiased_full_trajectory_importance_sampling=False, reference_kl_supported=False)


def episode_policy_loss_from_sum(logratio_sum, num_tokens, advantage, *, group_size,
                                 max_steps, objective_tokens, clip=0.2):
    """Length-weighted GSPO: -(N/D) min(s*A, clip(s)*A), s=exp(sum(delta)/N)."""
    if (not isinstance(logratio_sum, torch.Tensor) or logratio_sum.ndim != 0
            or not logratio_sum.is_floating_point()):
        raise ValueError("A scalar floating episode logratio sum is required")
    if (any(type(v) is not int or v < 1 for v in (group_size, max_steps, objective_tokens))
            or type(num_tokens) is not int or not 0 < num_tokens <= max_steps * objective_tokens):
        raise ValueError("Positive fixed budgets and a nonempty bounded episode are required")
    if not math.isfinite(advantage) or not 0 < clip < 1:
        raise ValueError("Finite advantage and clip in (0,1) required")
    if not torch.isfinite(logratio_sum):
        raise FloatingPointError("Finite actual behavior/current episode logratio required")
    ratio = (logratio_sum / num_tokens).exp()
    if not torch.isfinite(ratio):
        raise FloatingPointError("Finite episode importance ratio required")
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    loss = -(num_tokens / (group_size * max_steps * objective_tokens)) * surrogate
    if not torch.isfinite(loss):
        raise FloatingPointError("Finite episode policy loss required")
    return loss


def episode_policy_loss(current, old, advantage, *, group_size, max_steps, objective_tokens, clip=0.2):
    """Tensor oracle over all actual scored tokens of one complete episode."""
    if (current.ndim != 1 or old.shape != current.shape or not current.numel()
            or not current.is_floating_point() or not old.is_floating_point()):
        raise ValueError("Aligned nonempty one-dimensional episode logprobs required")
    if old.requires_grad:
        raise ValueError("Actual sampled old scores must be detached")
    if not torch.isfinite(current).all() or not torch.isfinite(old).all():
        raise FloatingPointError("Finite actual behavior and current episode scores required")
    return episode_policy_loss_from_sum((current.double() - old.double()).sum(), current.numel(),
        advantage, group_size=group_size, max_steps=max_steps,
        objective_tokens=objective_tokens, clip=clip)


def replay_episode_gspo_group(model, signal, *, scorer_for, old_values, group_size,
                              max_steps, objective_tokens, clip, replay_diagnostic,
                              joint_json_policy=False):
    """Replay once per segment; retain one episode's adapter gradients, not graphs.

    Backpropagate sum(logp)/D segment by segment into parameter.grad. Once the
    episode ratio is known, multiply those gradients by D*d(loss)/d(sum(logp))
    and add them to a separate accumulator. Parameters stay fixed throughout.
    No distributed collective, clipping or optimizer step occurs here; the
    existing caller performs those once after all local episodes.
    """
    if (any(type(v) is not int or v < 1 for v in (group_size, max_steps, objective_tokens))
            or not 0 < clip < 1 or replay_diagnostic not in {"full", "first", "off"}):
        raise ValueError("Valid fixed GSPO budgets, clip and diagnostic mode required")
    if any(not math.isfinite(a) for a in signal.advantages):
        raise ValueError("Finite group advantages required")
    if len(signal.segments) > group_size:
        raise ValueError("GSPO episode summaries cannot exceed the original group size")
    episodes = [(episode_index, segments, advantage) for episode_index, (segments, advantage) in
                enumerate(zip(signal.segments, signal.advantages, strict=True)) if advantage != 0]
    if any(not segments or len(segments) > max_steps for _, segments, _ in episodes):
        raise ValueError("Every nonzero-advantage episode needs its original bounded turns")
    scored = [(segment, advantage) for _, segments, advantage in episodes for segment in segments]
    diagnostics = {}
    if replay_diagnostic != "off":
        model.eval()
        with torch.no_grad():
            selected = scored if replay_diagnostic == "full" else scored[:1]
            for index, (segment, _) in enumerate(selected):
                replay = scorer_for(segment)(model, segment.prompt_ids, segment.completion_ids).detach()
                old = torch.as_tensor(old_values(segment), device=replay.device, dtype=replay.dtype)
                _validate_segment(replay, old, segment, objective_tokens)
                diagnostics[index] = replay
    model.train()
    denominator = group_size * max_steps * objective_tokens
    parameters = [p for p in model.parameters() if p.requires_grad]
    if episodes and not parameters:
        raise ValueError("Trainable parameters are required for GSPO replay")
    # These are the caller's existing gradients or the previous episode's
    # detached gradients. At most one extra adapter-gradient buffer is needed.
    accumulated = [p.grad for p in parameters]
    for parameter in parameters:
        parameter.grad = None
    tokens, diagnostic_tokens, index = 0, 0, 0
    loss_total, behavior_current = 0.0, 0.0
    behavior_replay, replay_current = None, None
    token_min, token_max, token_outside = None, None, 0
    ratios, episode_lengths, episode_summaries, clipped_episodes = [], [], [], 0
    try:
        for episode_index, segments, advantage in episodes:
            delta_sums, count = [], 0
            for segment in segments:
                current = scorer_for(segment)(model, segment.prompt_ids, segment.completion_ids)
                old = torch.as_tensor(old_values(segment), device=current.device, dtype=current.dtype)
                _validate_segment(current, old, segment, objective_tokens)
                observed = current.detach()
                difference = (observed - old).abs().max().item()
                behavior_current = max(behavior_current, difference)
                if index in diagnostics:
                    replay = diagnostics[index]
                    difference = (replay - old).abs().max().item()
                    behavior_replay = max(behavior_replay or 0.0, difference)
                    difference = (observed - replay).abs().max().item()
                    replay_current = max(replay_current or 0.0, difference)
                    diagnostic_tokens += current.numel()
                token_ratios = (observed - old).exp()
                if not torch.isfinite(token_ratios).all():
                    raise FloatingPointError("Non-finite actual behavior token importance ratio")
                lo, hi = token_ratios.min().item(), token_ratios.max().item()
                token_min = min(token_min if token_min is not None else lo, lo)
                token_max = max(token_max if token_max is not None else hi, hi)
                token_outside += ((token_ratios < 1 - clip) | (token_ratios > 1 + clip)).sum().item()
                delta_sums.append((observed.double() - old.double()).sum().item())
                count += current.numel()
                # Same forward supplies the score and its gradient. There is
                # no no-grad/grad two-pass numerical parity assumption.
                (current.double().sum() / denominator).backward()
                index += 1
            total = torch.tensor(math.fsum(delta_sums), dtype=torch.float64, requires_grad=True)
            loss = episode_policy_loss_from_sum(total, count, advantage, group_size=group_size,
                max_steps=max_steps, objective_tokens=objective_tokens, clip=clip)
            coefficient = torch.autograd.grad(loss, total)[0].item() * denominator
            # Report the same tensor operation used by the loss, including
            # rounding at exact clipping boundaries.
            ratio = (total.detach() / count).exp().item()
            gradient_clipped = ((advantage > 0 and ratio > 1 + clip)
                                or (advantage < 0 and ratio < 1 - clip))
            ratios.append(ratio)
            episode_lengths.append(count)
            clipped_episodes += int(gradient_clipped)
            loss_total += loss.item()
            tokens += count
            episode_summaries.append(dict(episode_index=episode_index, scored_tokens=count,
                logratio_sum=total.detach().item(), episode_ratio=ratio, advantage=advantage,
                policy_loss=loss.item(), gradient_clipped=gradient_clipped))
            for i, parameter in enumerate(parameters):
                gradient = parameter.grad
                if gradient is not None:
                    # A clipped episode has exactly zero contribution, even
                    # when some unscaled values would otherwise be large.
                    if not torch.isfinite(gradient).all():
                        raise FloatingPointError("Non-finite episode gradient; update refused")
                    gradient.mul_(coefficient)
                    if not torch.isfinite(gradient).all():
                        raise FloatingPointError("Non-finite weighted episode gradient; update refused")
                    if accumulated[i] is None:
                        accumulated[i] = gradient
                    else:
                        accumulated[i].add_(gradient)
                    if not torch.isfinite(accumulated[i]).all():
                        raise FloatingPointError("Non-finite accumulated episode gradient; update refused")
                parameter.grad = None
    except BaseException:
        # Fail closed: no partial local gradient is left for an accidental step.
        for parameter in parameters:
            parameter.grad = None
        raise
    for parameter, gradient in zip(parameters, accumulated, strict=True):
        parameter.grad = gradient
    report = dict(policy_loss=loss_total, scored_segments=len(scored), scored_tokens=tokens,
        algorithm_variant=VARIANT, policy_surrogate=gspo_configuration(),
        objective_denominator=denominator, episode_summaries=episode_summaries,
        scored_episodes=len(episodes), episode_scored_token_counts=episode_lengths,
        episode_gradient_clipped_count=clipped_episodes,
        importance_ratio_unit="whole-episode-geometric-mean",
        importance_ratio_min=min(ratios) if ratios else None,
        importance_ratio_max=max(ratios) if ratios else None,
        importance_ratio_outside_clip_fraction=(sum(r < 1 - clip or r > 1 + clip for r in ratios)
                                                 / len(ratios) if ratios else None),
        token_importance_ratio_min=token_min, token_importance_ratio_max=token_max,
        token_importance_ratio_outside_clip_fraction=token_outside / tokens if tokens else None,
        replay_diagnostic_mode=replay_diagnostic,
        replay_diagnostic_scope={"full": "all-nonzero-advantage-segments",
            "first": "first-nonzero-advantage-segment-per-group", "off": "not-collected"}[replay_diagnostic],
        diagnostic_scored_segments=len(diagnostics), diagnostic_scored_tokens=diagnostic_tokens,
        old_logprob_source="actual-sampled-thought-behavior",
        maximum_behavior_replay_logprob_difference=behavior_replay,
        maximum_behavior_current_logprob_difference=behavior_current if tokens else None,
        maximum_replay_current_logprob_difference=replay_current)
    if joint_json_policy:
        report.update(old_logprob_source="actual-sampled-raw-thought-and-post-xgrammar-JSON-behavior",
            thought_scored_tokens=sum(s.thought_count for s, _ in scored),
            json_scored_tokens=sum(len(s.json_logprobs) for s, _ in scored))
    return report


def _validate_segment(current, old, segment, objective_tokens):
    if (current.ndim != 1 or old.shape != current.shape
            or not current.is_floating_point() or not old.is_floating_point()
            or not 0 < current.numel() <= objective_tokens
            or current.numel() != len(segment.completion_ids) or old.requires_grad):
        raise ValueError("Aligned nonempty bounded actual behavior/current segment scores required")
    if not torch.isfinite(current).all() or not torch.isfinite(old).all():
        raise FloatingPointError("Finite actual behavior and current segment scores required")
