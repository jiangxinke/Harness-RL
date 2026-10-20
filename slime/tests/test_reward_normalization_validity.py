"""Reward statistics must reflect eligible policy episodes, before loss masking."""
from types import SimpleNamespace

import pytest
import torch

from slime.rollout.reward_normalization import normalize_group_rewards_by_rollout

NUM_GPUS = 0


def sample(group=0, episode=None, **changes):
    fields = dict(group_index=group, rollout_id=episode, remove_sample=False,
                  response_length=2, loss_mask=[1, 1], metadata={})
    fields.update(changes)
    return SimpleNamespace(**fields)


def normalize(samples, rewards, *, std=True, size=4):
    return normalize_group_rewards_by_rollout(samples, rewards,
        n_samples_per_prompt=size, normalize_std=std)


@pytest.mark.parametrize("invalid", [
    {"remove_sample": True}, {"loss_mask": [0, 0]},
    {"loss_mask": [], "response_length": 0},
    {"loss_mask": None, "response_length": 0},
    {"metadata": {"infrastructure_failure": {"kind": "search outage"}}},
])
def test_invalid_episode_cannot_create_variance_between_equal_successes(invalid):
    items = [sample(episode=0), sample(episode=1), sample(episode=2, **invalid)]
    assert normalize(items, [1.0, 1.0, 0.0]) == [0.0, 0.0, 0.0]


def test_normal_model_failure_is_retained_and_scale_and_unbiased_std_unchanged():
    items = [sample(episode=0), sample(episode=1), sample(episode=2, remove_sample=True)]
    # A model-generated wrong answer is an eligible zero-reward observation.
    centered = normalize(items, [4.0, 0.0, 999.0], std=False)
    assert centered == [2.0, -2.0, 0.0]
    expected = torch.tensor([2.0, -2.0]) / (torch.tensor([4.0, 0.0]).std() + 1e-6)
    assert normalize(items, [4.0, 0.0, 999.0]) == expected.tolist() + [0.0]


def test_all_valid_fanout_groups_keep_original_exact_values():
    items = [sample(10, "a"), sample(10, "a"), sample(10, "b"),
             sample(11, "c"), sample(11, "d"), sample(11, "d")]
    rewards = [2.0, 2.0, 4.0, 10.0, 14.0, 14.0]
    assert normalize(items, rewards, std=False) == [-1.0, -1.0, 1.0, -2.0, 2.0, 2.0]
    expected = torch.tensor([-1.0, 1.0]) / (torch.tensor([2.0, 4.0]).std() + 1e-6)
    actual = normalize(items, rewards)
    assert actual[:3] == [expected[0].item(), expected[0].item(), expected[1].item()]


def test_fanout_counts_only_eligible_siblings_once_and_keeps_invalid_zero():
    items = [sample(3, "a"), sample(3, "a"), sample(3, "a", loss_mask=[0, 0]),
             sample(3, "b"), sample(3, "outage", remove_sample=True)]
    assert normalize(items, [1.0, 1.0, 1000.0, 3.0, -1000.0], std=False) == [-1., -1., 0., 1., 0.]


def test_eligibility_does_not_move_legacy_chunk_boundaries():
    items = [sample(None, None), sample(None, None, remove_sample=True),
             sample(None, None), sample(None, None)]
    assert normalize(items, [1.0, 0.0, 10.0, 14.0], std=False, size=2) == [0., 0., -2., 2.]


def test_eligibility_does_not_merge_prompt_groups():
    items = [sample(0, "a"), sample(0, "b", remove_sample=True),
             sample(1, "c"), sample(1, "d")]
    assert normalize(items, [1.0, 0.0, 10.0, 14.0], std=False) == [0., 0., -2., 2.]


@pytest.mark.parametrize("valid_count", [0, 1])
def test_zero_or_single_valid_episode_stays_finite_zero(valid_count):
    items = [sample(episode=0, remove_sample=not valid_count), sample(episode=1, loss_mask=[0, 0])]
    values = normalize(items, [123.0, float("nan")])
    assert values == [0.0, 0.0]
    assert torch.isfinite(torch.tensor(values)).all()


def test_normal_model_failure_and_unmasked_tokens_are_not_relabelled():
    items = [sample(episode=0, loss_mask=None), sample(episode=1, loss_mask=[0, 1])]
    assert normalize(items, [0., 2.], std=False) == [-1., 1.]
    assert items[0].loss_mask is None and items[1].loss_mask == [0, 1]
