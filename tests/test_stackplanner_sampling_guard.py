"""The built-in adapter cannot replay exact nucleus-token membership yet."""
import pytest

from slime_plugins.stackplanner.config import TrainingConfig


@pytest.mark.parametrize('top_p', [.9, .95, .999])
@pytest.mark.parametrize('rollout_only', [False, True])
def test_builtin_rollout_rejects_unsupported_nucleus_before_generation(top_p, rollout_only):
    with pytest.raises(ValueError, match='built-in StackPlanner rollout currently requires top_p=1.0'):
        TrainingConfig.from_mapping({'run_id': 'sampling-guard', 'top_p': top_p,
                                     'rollout_only': rollout_only})


def test_builtin_default_sampling_keeps_existing_configuration():
    config = TrainingConfig.from_mapping({'run_id': 'sampling-default'})
    assert config.top_p == 1.0
    assert config.custom_generate_function == 'slime_plugins.stackplanner.generate.generate'


def test_custom_generator_keeps_its_existing_top_p_support():
    config = TrainingConfig.from_mapping({'run_id': 'custom-sampling', 'top_p': .9,
        'custom_generate_function': 'custom_rollout.generate'})
    assert config.top_p == .9


def test_train_only_replay_does_not_invoke_builtin_generator():
    config = TrainingConfig.from_mapping({'run_id': 'sampling-replay', 'top_p': .9,
        'train_only': True, 'replay_path': 'rollouts/{rollout_id}.pt'})
    assert config.top_p == .9
