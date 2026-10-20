"""CPU regressions for finite clipped objectives with non-finite intermediates."""

import math

import pytest
import torch

from slime.utils.ppo_utils import (
    calculate_log_probs_and_entropy,
    compute_approx_kl,
    compute_policy_loss,
)


NUM_GPUS = 0


def _old_kl(student, reference, kind, weight=None):
    ratio = student.float() - reference.float()
    if kind == "k1":
        result = ratio
    elif kind == "k2":
        result = ratio.square() / 2.0
    else:
        result = (-ratio).exp() - 1.0 + ratio
    if weight is not None:
        result = weight * result
    return result.clamp(-10.0, 10.0) if kind == "low_var_kl" else result


def _old_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high, eps_clip_c=None):
    ratio = (-ppo_kl).exp()
    first = -ratio * advantages
    second = -ratio.clamp(1.0 - eps_clip, 1.0 + eps_clip_high) * advantages
    loss = torch.maximum(first, second)
    clipfrac = (second > first).float()
    if eps_clip_c is not None:
        loss = torch.where(advantages < 0, torch.minimum(-eps_clip_c * advantages, loss), loss)
    return loss, clipfrac


@pytest.mark.parametrize("kind", ["k1", "k2", "k3", "low_var_kl"])
@pytest.mark.parametrize("weighted", [False, True])
def test_kl_ordinary_values_and_all_input_gradients_match_original(kind, weighted):
    student = torch.tensor([-6.0, -2.0, -0.5, 0.0, 0.2, 1.0, 6.0], requires_grad=True)
    reference = torch.zeros_like(student, requires_grad=True)
    # Zero weights have a nonzero weight derivative even though the value and
    # student derivative vanish. Include both saturated and unsaturated terms.
    weight = torch.tensor([0.0, 0.1, 1.0, 5.0, 1e-6, 100.0, 0.0], requires_grad=True) if weighted else None
    old_student = student.detach().clone().requires_grad_()
    old_reference = reference.detach().clone().requires_grad_()
    old_weight = weight.detach().clone().requires_grad_() if weighted else None
    result = compute_approx_kl(student, reference, kind, weight)
    expected = _old_kl(old_student, old_reference, kind, old_weight)
    torch.testing.assert_close(result, expected, rtol=2e-6, atol=1e-7)
    upstream = torch.tensor([0.2, -0.4, 0.7, 1.0, 1.3, 0.5, -0.3])
    (result * upstream).sum().backward()
    (expected * upstream).sum().backward()
    torch.testing.assert_close(student.grad, old_student.grad, rtol=2e-6, atol=1e-7)
    torch.testing.assert_close(reference.grad, old_reference.grad, rtol=2e-6, atol=1e-7)
    if weighted:
        torch.testing.assert_close(weight.grad, old_weight.grad, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_clipped_kl_exp100_has_finite_policy_gradient(masked, weighted):
    student = torch.tensor([-100.0, 100.0], requires_grad=True)
    weight = torch.ones(2, requires_grad=True) if weighted else None
    kl = compute_approx_kl(student, torch.zeros(2), "low_var_kl", weight)
    torch.testing.assert_close(kl, torch.full((2,), 10.0))
    (kl * (0.0 if masked else 1.0)).sum().backward()
    torch.testing.assert_close(student.grad, torch.zeros(2))
    if weighted:
        torch.testing.assert_close(weight.grad, torch.zeros(2))


def test_weighted_clipped_kl_keeps_weight_before_clamp_at_subnormal_fp32_weights():
    student = torch.tensor([-100.0, -100.0, 100.0], requires_grad=True)
    weight = torch.tensor([0.0, 1e-44, 1e-2])
    result = compute_approx_kl(student, torch.zeros(3), "low_var_kl", weight)
    # Build the high precision reference before clipping, rather than reusing
    # _old_kl's deliberate FP32 conversion (whose exp(100) overflows).
    raw = torch.expm1(-student.detach().double()) + student.detach().double()
    expected = (weight.double() * raw).clamp(-10.0, 10.0).float()
    torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-7)
    result.sum().backward()
    expected_grad = (weight.double() * (1.0 - (-student.detach().double()).exp())).float()
    torch.testing.assert_close(student.grad, expected_grad, rtol=1e-6, atol=1e-7)
    assert torch.isfinite(student.grad).all()


def test_differentiable_production_importance_ratio_matches_original_actor_gradient():
    student = torch.tensor([-10.0, -3.0, 0.0, 2.0], requires_grad=True)
    old_student = student.detach().clone().requires_grad_()
    result = compute_approx_kl(student, torch.zeros(4), "low_var_kl", student.exp())
    expected = _old_kl(old_student, torch.zeros(4), "low_var_kl", old_student.exp())
    torch.testing.assert_close(result, expected, rtol=2e-6, atol=2e-7)
    result.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(student.grad, old_student.grad, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("masked", [False, True])
def test_unrepresentable_active_importance_gradient_fails_before_actor_chain_rule(masked):
    student = torch.tensor([-1000.0], requires_grad=True)
    # This is the production, differentiable IS-ratio expression. Its FP32
    # underflow must not turn an unrepresentable weight derivative into NaN*0.
    weight = student.exp()
    result = compute_approx_kl(student, torch.zeros(1), "low_var_kl", weight)
    assert torch.isfinite(result).all()
    if masked:
        (result * 0.0).sum().backward()
        torch.testing.assert_close(student.grad, torch.zeros(1))
    else:
        with pytest.raises(RuntimeError, match="importance gradient is non-finite"):
            result.sum().backward()


def test_unbounded_kl_overflow_fails_instead_of_clipping():
    with pytest.raises(RuntimeError, match="non-finite unbounded KL"):
        compute_approx_kl(torch.tensor([-100.0]), torch.zeros(1), "k3")


@pytest.mark.parametrize("which", ["student", "reference", "weight"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_kl_nonfinite_inputs_fail_fast(which, invalid):
    inputs = {"student": torch.zeros(1), "reference": torch.zeros(1), "weight": torch.ones(1)}
    inputs[which] = torch.tensor([invalid])
    with pytest.raises(RuntimeError, match="non-finite"):
        compute_approx_kl(inputs["student"], inputs["reference"], "low_var_kl", inputs["weight"])


def test_negative_importance_ratio_fails_fast():
    with pytest.raises(RuntimeError, match="nonnegative"):
        compute_approx_kl(torch.zeros(1), torch.zeros(1), "low_var_kl", torch.tensor([-1.0]))


@pytest.mark.parametrize("dual_clip", [None, 3.0])
@pytest.mark.parametrize("eps_clip,eps_high", [(0.2, 0.28), (1.0, 4.0)])
def test_policy_loss_ordinary_values_boundaries_and_gradients_match_original(dual_clip, eps_clip, eps_high):
    ratios = [0.1, 0.8, 1.0, 1.28, 3.0, 6.0, 12.0]
    if eps_clip < 1.0:
        ratios.extend([1.0 - eps_clip, 1.0 + eps_high])
    ppo_kl = -torch.tensor([math.log(r) for r in ratios] * 3, requires_grad=True)
    ppo_kl.retain_grad()
    advantages = torch.tensor([1.0] * len(ratios) + [-1.0] * len(ratios) + [0.0] * len(ratios), requires_grad=True)
    old_kl = ppo_kl.detach().clone().requires_grad_()
    old_advantages = advantages.detach().clone().requires_grad_()
    result, clipfrac = compute_policy_loss(ppo_kl, advantages, eps_clip, eps_high, dual_clip)
    expected, expected_clipfrac = _old_policy_loss(old_kl, old_advantages, eps_clip, eps_high, dual_clip)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(clipfrac, expected_clipfrac, rtol=0, atol=0)
    result.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(ppo_kl.grad, old_kl.grad, rtol=0, atol=0)
    torch.testing.assert_close(advantages.grad, old_advantages.grad, rtol=0, atol=0)


@pytest.mark.parametrize("advantage,dual_clip,expected", [(1.0, None, -1.28), (0.0, None, 0.0), (-1.0, 3.0, 3.0)])
def test_policy_exp100_is_finite_when_original_objective_is_bounded(advantage, dual_clip, expected):
    log_ratio = torch.tensor([100.0], requires_grad=True)
    result, clipfrac = compute_policy_loss(-log_ratio, torch.tensor([advantage]), 0.2, 0.28, dual_clip)
    torch.testing.assert_close(result, torch.tensor([expected]))
    torch.testing.assert_close(clipfrac, torch.tensor([float(advantage > 0)]))
    result.sum().backward()
    torch.testing.assert_close(log_ratio.grad, torch.zeros(1))


def test_unbounded_negative_advantage_ppo_overflow_fails_fast():
    with pytest.raises(RuntimeError, match="non-finite active PPO loss"):
        compute_policy_loss(torch.tensor([-100.0]), torch.tensor([-1.0]), 0.2, 0.28)


@pytest.mark.parametrize("which", ["ppo_kl", "advantages"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_policy_nonfinite_inputs_fail_fast(which, invalid):
    inputs = {"ppo_kl": torch.zeros(1), "advantages": torch.ones(1)}
    inputs[which] = torch.tensor([invalid])
    with pytest.raises(RuntimeError, match="non-finite"):
        compute_policy_loss(inputs["ppo_kl"], inputs["advantages"], 0.2, 0.28)


@pytest.mark.parametrize("mode", ["logprob", "entropy", "both"])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("chunk_size", [-1, 1])
def test_unused_invalid_vocab_row_has_zero_gradient(mode, masked, chunk_size):
    logits = torch.tensor([[0.0, 1.0], [float("-inf"), float("-inf")]], requires_grad=True)
    keep = torch.ones_like(logits, dtype=torch.bool) if masked else None
    log_probs, entropy = calculate_log_probs_and_entropy(
        logits, torch.tensor([1, 0]), tp_group=None, with_entropy=True,
        with_entropy_grad=True, chunk_size=chunk_size, log_prob_keep_mask=keep,
    )
    loss = (log_probs[0].sum() if mode != "entropy" else 0.0) + (entropy[0] if mode != "logprob" else 0.0)
    assert torch.isfinite(loss)
    loss.backward()
    reference = logits.detach()[:1].clone().requires_grad_()
    log_softmax = reference.log_softmax(-1)
    expected_loss = (log_softmax[0, 1] if mode != "entropy" else 0.0)
    if mode != "logprob":
        expected_loss = expected_loss - (log_softmax.exp() * log_softmax).sum()
    expected_loss.backward()
    torch.testing.assert_close(logits.grad[:1], reference.grad, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(logits.grad[1], torch.zeros(2))


@pytest.mark.parametrize("mode", ["logprob", "entropy", "both"])
def test_active_invalid_vocab_row_still_fails_fast(mode):
    logits = torch.tensor([[float("-inf"), float("-inf")]], requires_grad=True)
    log_probs, entropy = calculate_log_probs_and_entropy(
        logits, torch.tensor([0]), tp_group=None, with_entropy=True, with_entropy_grad=True,
    )
    loss = (log_probs.sum() if mode != "entropy" else 0.0) + (entropy.sum() if mode != "logprob" else 0.0)
    with pytest.raises(RuntimeError, match="non-finite active gradient"):
        loss.backward()


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_vocab_upstream_gradient_still_fails_fast(invalid):
    logits = torch.tensor([[0.0, 1.0]], requires_grad=True)
    log_probs, _ = calculate_log_probs_and_entropy(logits, torch.tensor([1]), tp_group=None, with_entropy=False)
    with pytest.raises(RuntimeError, match="non-finite active gradient"):
        log_probs.backward(torch.full_like(log_probs, invalid))
