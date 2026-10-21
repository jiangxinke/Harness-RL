"""CPU regressions for real GRPO reward statistics and response propagation."""
from types import SimpleNamespace

import _cp_dist_helpers  # Install CPU topology metadata before backend imports.
import pytest
import torch
from megatron.core import mpu

from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp
from slime.backends.megatron_utils.loss import compute_advantages_and_returns
from slime.rollout.reward_normalization import normalize_group_rewards_by_rollout
from slime.utils.ppo_utils import compute_policy_loss

NUM_GPUS = 0


def sample(group, episode, **changes):
    fields = dict(group_index=group, rollout_id=episode, response_length=8,
                  loss_mask=[1] * 8, metadata={}, remove_sample=False)
    fields.update(changes)
    return SimpleNamespace(**fields)


def normalize(samples, rewards, *, std=True):
    return normalize_group_rewards_by_rollout(
        samples, rewards, n_samples_per_prompt=8, normalize_std=std)


@pytest.mark.parametrize("reward", [.7, .8, .1])
@pytest.mark.parametrize("count", [1, 8, 16, 32])
@pytest.mark.parametrize("std", [False, True])
def test_identical_nonbinary_rewards_have_exact_zero_advantage(reward, count, std):
    items = [sample(0, index) for index in range(count)]
    assert normalize(items, [reward] * count, std=std) == [0.] * count


@pytest.mark.parametrize("std", [False, True])
def test_close_distinct_rewards_keep_real_signal_and_unbiased_std(std):
    # All differences survive FP32 input conversion, but are much smaller than
    # the common offset. Never replace genuine low variance by an epsilon gate.
    rewards = torch.tensor([.8] * 4 + [.8000001192092896] * 4, dtype=torch.float32)
    items = [sample(0, index) for index in range(len(rewards))]
    # Independent FP64 reference over the actual FP32 observations.
    expected = rewards.double() - rewards.double().mean()
    if std:
        expected /= expected.std(unbiased=True) + 1e-6
    actual = torch.tensor(normalize(items, rewards.tolist(), std=std))
    torch.testing.assert_close(actual, expected.float(), rtol=1e-6, atol=1e-9)
    assert torch.all(actual[:4] < 0) and torch.all(actual[4:] > 0)
    assert actual.sum().item() == 0.


def test_groups_fanout_and_ineligible_masks_do_not_change_centering():
    items = ([sample(10, "a")] * 3 + [sample(10, "b")]
             + [sample(10, "masked", loss_mask=[0] * 8)]
             + [sample(10, "empty", response_length=0, loss_mask=[])]
             + [sample(10, "removed", remove_sample=True)]
             + [sample(10, "outage", metadata={"infrastructure_failure": {"kind": "offline"}})]
             + [sample(11, index) for index in range(8)])
    rewards = [2., 2., 2., 4., float("nan"), 999., -999., 999.] + [.8] * 8
    assert normalize(items, rewards, std=False) == [-1.] * 3 + [1.] + [0.] * 12
    actual = normalize(items, rewards)
    expected = (torch.tensor([-1., 1.]) / (torch.tensor([2., 4.]).std() + 1e-6)).tolist()
    assert actual == [expected[0]] * 3 + [expected[1]] + [0.] * 12


def test_all_masked_groups_ignore_nonfinite_rewards():
    items = [sample(0, 0, loss_mask=[0] * 8),
             sample(0, 1, response_length=0, loss_mask=[])]
    assert normalize(items, [float("nan"), float("inf")]) == [0., 0.]


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_constant_group_grpo_propagates_zero_through_actual_cp_slices(monkeypatch, cp_size):
    monkeypatch.setattr(mpu, "is_pipeline_last_stage", lambda: True, raising=False)
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: cp_size)
    items = [sample(0, index) for index in range(8)]
    rewards = normalize(items, [.8] * 8)
    args = SimpleNamespace(use_rollout_logprobs=False, kl_coef=0.,
        custom_advantage_function_path=None, advantage_estimator="grpo",
        use_opd=False, normalize_advantages=False)
    response_lengths = [8] * 8
    total_lengths = [32] * 8
    local_lengths = []
    for rank in range(cp_size):
        monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: rank)
        logprobs = [slice_log_prob_with_cp(torch.zeros(response), total, response)
                    for total, response in zip(total_lengths, response_lengths, strict=True)]
        data = dict(rollout_log_probs=logprobs, rewards=rewards,
                    response_lengths=response_lengths, total_lengths=total_lengths,
                    loss_masks=[torch.ones(response) for response in response_lengths])
        compute_advantages_and_returns(args, data)
        for old, advantage, returns in zip(logprobs, data["advantages"], data["returns"], strict=True):
            assert advantage.shape == old.shape and returns.shape == old.shape
            assert torch.equal(advantage, torch.zeros_like(old))
        local = torch.cat(data["advantages"])
        theta = torch.tensor(0., requires_grad=True)
        losses, _ = compute_policy_loss(-theta.expand(local.numel()), local, .2, .2)
        losses.sum().backward()
        assert theta.grad.item() == 0. and torch.isfinite(theta.grad)
        local_lengths.append(sum(value.numel() for value in logprobs))
    assert sum(local_lengths) == sum(response_lengths)
    if cp_size == 4:
        assert 0 in local_lengths  # Empty CP shards still have a finite zero gradient.
    assert not torch.cuda.is_initialized()
