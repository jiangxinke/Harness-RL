from types import SimpleNamespace

import pytest

from slime.rollout.reward_normalization import normalize_group_rewards_by_rollout


NUM_GPUS = 0


def sample(group_index, rollout_id):
    return SimpleNamespace(group_index=group_index, rollout_id=rollout_id)


@pytest.mark.unit
def test_fanout_samples_count_each_episode_once_and_broadcast_advantage():
    samples = [
        sample(10, "episode-a"),
        sample(10, "episode-a"),
        sample(10, "episode-a"),
        sample(10, "episode-b"),
    ]

    normalized = normalize_group_rewards_by_rollout(
        samples,
        [1.0, 1.0, 1.0, 3.0],
        n_samples_per_prompt=2,
        normalize_std=False,
    )

    # Unique episode rewards [1, 3] center to [-1, 1].  Episode A's three
    # role/trajectory segments do not move the mean toward 1.
    assert normalized == [-1.0, -1.0, -1.0, 1.0]


@pytest.mark.unit
def test_prompt_groups_are_normalized_independently_with_uneven_fanout():
    samples = [
        sample(20, 1),
        sample(20, 1),
        sample(20, 2),
        sample(21, 3),
        sample(21, 4),
        sample(21, 4),
    ]

    normalized = normalize_group_rewards_by_rollout(
        samples,
        [2.0, 2.0, 4.0, 10.0, 14.0, 14.0],
        n_samples_per_prompt=2,
        normalize_std=False,
    )

    assert normalized == [-1.0, -1.0, 1.0, -2.0, 2.0, 2.0]


@pytest.mark.unit
def test_missing_ids_keep_fixed_size_legacy_grouping():
    samples = [sample(None, None) for _ in range(4)]
    normalized = normalize_group_rewards_by_rollout(
        samples,
        [1.0, 3.0, 10.0, 14.0],
        n_samples_per_prompt=2,
        normalize_std=False,
    )

    assert normalized == [-1.0, 1.0, -2.0, 2.0]


@pytest.mark.unit
def test_inconsistent_team_reward_for_one_episode_is_rejected():
    samples = [sample(10, "episode-a"), sample(10, "episode-a")]
    with pytest.raises(ValueError, match="inconsistent episode rewards"):
        normalize_group_rewards_by_rollout(
            samples,
            [1.0, 2.0],
            n_samples_per_prompt=2,
            normalize_std=True,
        )


@pytest.mark.unit
def test_single_unique_episode_has_zero_advantage_without_nan():
    samples = [sample(10, "episode-a"), sample(10, "episode-a")]
    normalized = normalize_group_rewards_by_rollout(
        samples,
        [1.0, 1.0],
        n_samples_per_prompt=2,
        normalize_std=True,
    )
    assert normalized == [0.0, 0.0]
