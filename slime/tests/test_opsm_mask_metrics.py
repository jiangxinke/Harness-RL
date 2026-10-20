"""CPU OPSM statistics: exclude non-actions without changing rejection or weights."""
from types import SimpleNamespace

import _cp_dist_helpers
import pytest
import torch
from megatron.core import mpu
from slime.backends.megatron_utils import loss as LOSS
from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean, slice_log_prob_with_cp
from slime.utils.ppo_utils import compute_opsm_mask

NUM_GPUS = 0


def _set_cp(monkeypatch, size, rank):
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: size)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: rank)


def _batch():
    # Includes partial/zero masks, an empty response, mixed advantage signs,
    # unequal lengths, and a response owned by only one CP rank.
    totals = [7, 7, 7, 3, 12, 35]
    masks = [torch.ones(4), torch.tensor([1., 0., 0., 0.]), torch.zeros(4),
             torch.empty(0), torch.tensor([1., 0., 1., 0., 0., 0., 1.]), torch.tensor([0., 1., 0.])]
    advantages = [torch.full_like(m, -1.) for m in masks]
    advantages[4] = torch.tensor([-1., 1., -1., -1., 1., -1., 1.])
    return totals, masks, advantages


@pytest.mark.parametrize("mask,expected", [([1, 1, 1, 1], 1.), ([1, 0, 0, 0], 1.), ([0, 0, 0, 0], 0.)])
def test_opsm_metric_excludes_non_actor_tokens(monkeypatch, mask, expected):
    _set_cp(monkeypatch, 1, 0)
    kept, metric = compute_opsm_mask(SimpleNamespace(opsm_delta=.5),
        [torch.full((4,), -2.)], [torch.full((4,), -1.)], [torch.full((4,), -1.)], [torch.tensor(mask)])
    torch.testing.assert_close(kept, torch.full((4,), 1. if expected == 0 else 0.))
    assert float(metric) == expected


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_opsm_metric_and_actor_gradient_are_conserved_across_cp(monkeypatch, cp_size):
    totals, masks, advantages = _batch()
    lengths = list(map(len, masks))
    full_log_probs = [torch.full((n,), -2.) for n in lengths]
    full_old_log_probs = [torch.full((n,), -1.) for n in lengths]
    theta = torch.tensor(1., requires_grad=True)
    metric_sum, losses, empty_ranks = 0., [], 0
    for rank in range(cp_size):
        _set_cp(monkeypatch, cp_size, rank)
        local = lambda tensors: [slice_log_prob_with_cp(x, t, n)
            for x, t, n in zip(tensors, totals, lengths, strict=True)]
        local_masks, local_advantages = local(masks), local(advantages)
        kept, metric = compute_opsm_mask(SimpleNamespace(opsm_delta=.5),
            full_log_probs, full_old_log_probs, local_advantages, masks, local_loss_masks=local_masks)
        # Every nonempty actor sequence has KL=1; the zero-mask sequence has
        # KL=0. The existing rejection rule therefore drops negative signs.
        rejected = [(a < 0) if m.any() else torch.zeros_like(a, dtype=torch.bool)
                    for a, m in zip(advantages, masks, strict=True)]
        assert torch.equal(kept, 1 - torch.cat(local(rejected)).float())
        metric_sum += float(metric)
        empty_ranks += not any(len(a) for a in local_advantages)
        token_values = torch.cat(local([torch.arange(n).float() + 1 for n in lengths]))
        reducer = get_sum_of_sample_mean(totals, lengths, masks)
        losses.append(reducer(theta * token_values * kept))
    # Fractions per sample: 1 + 1 + 0 + 0 + 2/3 + 1. No sequence
    # reweighting or extra normalization by the number of samples is added.
    assert metric_sum == pytest.approx(11 / 3)
    sum(losses).backward()
    assert float(theta.grad) == pytest.approx(7 / 3)
    assert torch.isfinite(theta.grad)
    if cp_size == 4:
        assert empty_ranks > 0


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_policy_caller_passes_response_aligned_cp_masks(monkeypatch, cp_size):
    totals, masks, advantages = _batch()
    lengths = list(map(len, masks))
    for rank in range(cp_size):
        _set_cp(monkeypatch, cp_size, rank)
        local = lambda tensors: [slice_log_prob_with_cp(x, t, n)
            for x, t, n in zip(tensors, totals, lengths, strict=True)]
        full_current = [torch.full((n,), -2.) for n in lengths]
        full_old = [torch.full((n,), -1.) for n in lengths]
        current, old = local(full_current), local(full_old)
        full_by_id = {id(x): y for x, y in zip(current + old, full_current + full_old, strict=True)}
        monkeypatch.setattr(LOSS, "get_rollout_top_p_logprob_kwargs", lambda *args: {})
        monkeypatch.setattr(LOSS, "get_log_probs_and_entropy", lambda *args, **kw: (None, {"log_probs": current}))
        monkeypatch.setattr(LOSS, "all_gather_with_cp", lambda x, *args: full_by_id[id(x)])

        class ReachedOPSM(Exception):
            pass

        def capture(**kwargs):
            assert "local_loss_masks" in kwargs
            assert len(kwargs["local_loss_masks"]) == len(masks)
            assert all(torch.equal(a, b) for a, b in zip(kwargs["local_loss_masks"], local(masks), strict=True))
            raise ReachedOPSM

        # Stop the real caller at OPSM; no PPO/KL numerical function is run.
        monkeypatch.setattr(LOSS, "compute_opsm_mask", capture)
        args = SimpleNamespace(use_rollout_logprobs=False, use_opsm=True, advantage_estimator="grpo")
        batch = dict(advantages=local(advantages), log_probs=old, total_lengths=totals,
            response_lengths=lengths, unconcat_tokens=[], loss_masks=masks)
        with pytest.raises(ReachedOPSM):
            LOSS.policy_loss_function(args, batch, torch.empty(0), lambda x: x.sum())
