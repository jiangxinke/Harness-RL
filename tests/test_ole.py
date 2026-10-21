from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from slime.utils.types import Sample
from slime_plugins.stackplanner.capability.runtime_taxonomy import active_capabilities
from slime_plugins.stackplanner.capability.sample_transform import (
    fan_out_capability_samples,
)
from slime_plugins.stackplanner.cli import build_slime_command, resolve_config
from slime_plugins.stackplanner.config import TrainingConfig
from slime_plugins.stackplanner.ole import (
    apply_selection_to_advantages,
    generate_rollout,
    select_prompt_groups,
)
from slime_plugins.stackplanner.rewards import normalize_by_episode

ROOT = Path(__file__).resolve().parent.parent


def _args(**overrides):
    return SimpleNamespace(
        **{
            "reward_key": None,
            "advantage_estimator": "grpo",
            "rewards_normalization": True,
            "grpo_std_normalization": True,
            "n_samples_per_prompt": 2,
            **overrides,
        }
    )


def _config(**overrides):
    return TrainingConfig.from_mapping(
        {"run_id": "ole-test", "ole_enabled": True, **overrides}
    )


def _sample(group, episode, reward):
    return Sample(
        group_index=group,
        rollout_id=episode,
        index=episode,
        tokens=[1, 2, 3, 4],
        response_length=3,
        loss_mask=[1, 0, 1],
        rollout_log_probs=[-0.2, 0.0, -0.3],
        reward=reward,
        train_metadata={"existing": "preserved"},
    )


def _batch():
    return [
        _sample(0, 0, 0.0),
        _sample(0, 1, 0.2),
        _sample(1, 2, 0.4),
        _sample(1, 3, 1.0),
    ]


def test_default_command_is_unchanged_and_cli_can_override_yaml(tmp_path):
    config, forwarded, _ = resolve_config(
        ["--config", str(ROOT / "configs/experiment_a_central.yaml")]
    )
    baseline = build_slime_command(config)
    assert config.ole_enabled is False
    assert not any("stackplanner.ole." in arg for arg in baseline)
    enabled = copy.deepcopy(config)
    enabled.ole_enabled = True
    command = build_slime_command(enabled)
    # Removing the two optional hook options recovers the original command exactly.
    hooks = command.index("--rollout-function-path")
    assert command[:hooks] + command[hooks + 4 :] == baseline
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"run_id": "override", "ole_enabled": True}))
    disabled, _, _ = resolve_config(["--config", str(path), "--no-ole-enabled"])
    assert disabled.ole_enabled is False
    enabled, forwarded, _ = resolve_config(
        ["--run-id", "cli", "--ole-enabled", "--ole-k", "0.5"]
    )
    assert enabled.ole_enabled and enabled.ole_k == 0.5 and forwarded == []


@pytest.mark.parametrize(
    "key,value",
    [
        ("ole_enabled", "false"),
        ("ole_threshold_percent", 0),
        ("ole_threshold_percent", 1.1),
        ("ole_k", -1),
        ("ole_k", float("nan")),
        ("ole_init_threshold", float("inf")),
        ("ole_threshold_percent", True),
    ],
)
def test_invalid_ole_config_fails_early(key, value):
    with pytest.raises(ValueError, match=key):
        _config(**{key: value})


@pytest.mark.parametrize(
    "flag",
    [
        "--rollout-function-path=x",
        "--rollout-data-postprocess-path",
        "--data-source-path",
        "--disable-rollout-global-dataset",
    ],
)
def test_ole_hooks_cannot_be_silently_overridden(flag):
    with pytest.raises(ValueError, match="OLE owns"):
        _config(extra_slime_args=[flag])


def test_disabled_selection_is_an_exact_noop():
    samples = _batch()
    before = copy.deepcopy(samples)
    state = {"threshold": 0.9}
    assert (
        select_prompt_groups(
            _args(), _config(ole_enabled=False), samples, state, rollout_id=0
        )
        == {}
    )
    assert samples == before
    assert state == {"threshold": 0.9}


def test_previous_threshold_population_std_and_episode_deduplication():
    samples = _batch()
    # Unequal segment counts must not bias the group score.
    samples.extend(copy.deepcopy(samples[0]) for _ in range(4))
    before = copy.deepcopy(samples)
    state = {}
    metrics = select_prompt_groups(
        _args(), _config(ole_threshold_percent=0.5), samples, state, rollout_id=0
    )
    assert metrics["ole/threshold"] == 0.0
    assert metrics["ole/next_threshold"] == pytest.approx(0.6)
    assert metrics["ole/episodes"] == 4
    assert (
        metrics["ole/discarded_total"] == 0
    )  # Current quantile is for the next batch.
    assert samples[0].train_metadata["ole"]["score"] == pytest.approx(0.2)
    assert samples[2].train_metadata["ole"]["score"] == pytest.approx(1.0)
    metrics = select_prompt_groups(_args(), _config(), samples, state, rollout_id=1)
    assert metrics["ole/discarded_groups"] == 1
    assert metrics["ole/discarded_episodes"] == 2
    assert not samples[0].train_metadata["ole"]["selected"]
    assert samples[2].train_metadata["ole"]["selected"]
    for old, new in zip(before, samples, strict=True):
        metadata = new.train_metadata
        assert metadata["existing"] == "preserved"
        assert {
            key: value
            for key, value in new.to_dict().items()
            if key != "train_metadata"
        } == {
            key: value
            for key, value in old.to_dict().items()
            if key != "train_metadata"
        }
    # Reward normalization sees exactly the original rewards and sample grouping.
    assert normalize_by_episode(_args(), before) == normalize_by_episode(
        _args(), samples
    )


def test_threshold_ties_singletons_and_empty_batch():
    state = {"threshold": 0.25}
    sample = _sample(0, 0, 0.25)
    select_prompt_groups(_args(), _config(), [sample], state, rollout_id=0)
    assert sample.train_metadata["ole"]["selected"]  # Strictly less, never <=.
    assert state["threshold"] == 0.25
    metrics = select_prompt_groups(_args(), _config(), [], state, rollout_id=1)
    assert metrics["ole/groups"] == 0 and state["threshold"] == 0.25


@pytest.mark.parametrize(
    "group_rewards,selected",
    [
        ([[0.0, 0.8], [0.0, 0.8]], [True, True]),
        ([[0.8]], [True]),
        ([[0.0, 0.2], [0.0, 0.8], [0.0, 1.0]], [False, True, True]),
        ([[0.001, 0.14, 0.93]], [True]),
    ],
)
def test_repeated_batch_keeps_groups_equal_to_previous_quantile(group_rewards, selected):
    state = {}
    config = _config(ole_threshold_percent=0.5)
    for rollout_id in range(3):
        samples = [
            _sample(group, episode, reward)
            for group, rewards in enumerate(group_rewards)
            for episode, reward in enumerate(rewards)
        ]
        metrics = select_prompt_groups(_args(), config, samples, state, rollout_id=rollout_id)
        if rollout_id:
            assert [sample.train_metadata["ole"]["selected"] for sample in samples] == [
                selected[group]
                for group, rewards in enumerate(group_rewards)
                for _ in rewards
            ]
            assert metrics["ole/discarded_groups"] == selected.count(False)


def test_repeated_decimal_reward_batch_retains_nonzero_policy_gradient():
    from slime.utils.ppo_utils import compute_policy_loss

    samples = [_sample(group, group * 2 + episode, reward)
               for group in range(2) for episode, reward in enumerate((0.0, 0.8))]
    state = {}
    for rollout_id in range(2):
        select_prompt_groups(_args(), _config(), samples, state, rollout_id=rollout_id)
    _, normalized = normalize_by_episode(_args(), samples)
    original = [torch.tensor([value]) for value in normalized]
    data = {"advantages": original, "metadata": [sample.train_metadata for sample in samples]}
    apply_selection_to_advantages(None, 1, data)
    assert torch.equal(torch.stack(data["advantages"]), torch.stack(original))
    parameter = torch.nn.Parameter(torch.zeros(4, 1))
    loss, _ = compute_policy_loss(-parameter, torch.stack(data["advantages"]), 0.2, 0.2)
    loss.mean().backward()
    assert torch.isfinite(parameter.grad).all()
    assert torch.count_nonzero(parameter.grad) == 4


def test_infrastructure_and_empty_actor_episodes_do_not_bias_scores():
    samples = _batch()
    removed = _sample(0, 4, -100)
    removed.remove_sample = True
    failed = _sample(0, 5, -100)
    failed.metadata = {"infrastructure_failure": {"kind": "adapter"}}
    frozen = _sample(0, 6, -100)
    frozen.loss_mask = [0, 0, 0]
    metrics = select_prompt_groups(
        _args(), _config(), samples + [removed, failed, frozen], {}, rollout_id=0
    )
    assert metrics["ole/episodes"] == 4
    assert samples[0].train_metadata["ole"]["score"] == pytest.approx(0.2)


def test_malformed_episode_metadata_or_rewards_fail():
    samples = _batch()
    samples[0].group_index = None
    with pytest.raises(ValueError, match="group_index"):
        select_prompt_groups(_args(), _config(), samples, {}, rollout_id=0)
    for rewards, message in [([0.1, 0.2], "inconsistent"), ([float("nan")], "finite")]:
        with pytest.raises(ValueError, match=message):
            select_prompt_groups(
                _args(),
                _config(),
                [_sample(0, 0, reward) for reward in rewards],
                {},
                rollout_id=0,
            )


@pytest.mark.parametrize("scope", ["central-capability", "all-agent-action-output"])
def test_capo_fanout_preserves_selection_and_loss_masks(monkeypatch, scope):
    monkeypatch.setenv("SP_DECOUPLING_SCOPE", scope)
    samples = _batch()
    capabilities = active_capabilities()
    for sample in samples:
        sample.train_metadata["capability_labels"] = [
            capabilities[0],
            None,
            capabilities[-1],
        ]
    select_prompt_groups(
        _args(), _config(ole_init_threshold=0.5), samples, {}, rollout_id=0
    )
    children = fan_out_capability_samples(_args(), samples)
    assert len(children) == len(samples) * len(capabilities)
    for index, sample in enumerate(samples):
        siblings = children[index * len(capabilities) : (index + 1) * len(capabilities)]
        assert all(
            child.train_metadata["ole"] == sample.train_metadata["ole"]
            for child in siblings
        )
        assert [
            sum(child.loss_mask[token] for child in siblings) for token in range(3)
        ] == sample.loss_mask


def test_checkpoint_restores_threshold_and_discard_count(tmp_path):
    from slime.rollout.data_source import RolloutDataSourceWithBuffer

    args = _args(
        rollout_global_dataset=True,
        prompt_data=None,
        buffer_filter_path=None,
        save=str(tmp_path),
        load=str(tmp_path),
        rollout_shuffle=False,
    )
    source = RolloutDataSourceWithBuffer(args)
    state = source.metadata.setdefault("stackplanner_ole", {})
    select_prompt_groups(args, _config(), _batch(), state, rollout_id=0)
    source.save(0)
    resumed = RolloutDataSourceWithBuffer(args)
    resumed.load(0)
    restored = resumed.metadata["stackplanner_ole"]
    assert restored == state
    original_samples, resumed_samples = _batch(), _batch()
    assert select_prompt_groups(
        args, _config(), original_samples, state, rollout_id=1
    ) == select_prompt_groups(args, _config(), resumed_samples, restored, rollout_id=1)
    assert original_samples == resumed_samples


def test_rollout_wrapper_bypasses_eval_and_records_training_metrics(
    monkeypatch, tmp_path
):
    samples = _batch()
    result = SimpleNamespace(samples=[samples], metrics={"existing": 1})
    eval_result = object()
    module = ModuleType("slime.rollout.sglang_rollout")
    module.generate_rollout = (
        lambda *args, evaluation=False: eval_result if evaluation else result
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    path = tmp_path / "resolved.json"
    path.write_text(json.dumps(_config().as_dict()))
    monkeypatch.setenv("SP_TRAINING_RESOLVED_CONFIG", str(path))
    source = SimpleNamespace(metadata={})
    assert generate_rollout(_args(), 0, source, evaluation=True) is eval_result
    assert source.metadata == {}
    assert generate_rollout(_args(), 0, source) is result
    assert result.metrics["existing"] == 1 and result.metrics["ole/groups"] == 2
    assert source.metadata["stackplanner_ole"]["rollout_id"] == 0
    # Debug serialization/replay retains all OLE decisions.
    assert (
        Sample.from_dict(samples[0].to_dict()).train_metadata
        == samples[0].train_metadata
    )


def test_advantage_hook_handles_pipeline_stages_and_rejects_unannotated_replay():
    apply_selection_to_advantages(None, 0, {})
    with pytest.raises(ValueError, match="aligned"):
        apply_selection_to_advantages(None, 0, {"advantages": [torch.ones(1)]})
    with pytest.raises(ValueError, match="replay"):
        apply_selection_to_advantages(
            None, 0, {"advantages": [torch.ones(1)], "metadata": [{}]}
        )


@pytest.mark.parametrize("threshold", [0.0, 0.5, 2.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_policy_gradients_and_optimizer_updates(threshold, dtype):
    """Exercise actual slime PPO math on CPU or OLE_TEST_DEVICE, including all rejected."""
    from slime.utils.ppo_utils import compute_policy_loss

    device = os.environ.get("OLE_TEST_DEVICE", "cpu")
    seed = int(os.environ.get("OLE_TEST_SEED", "42"))
    torch.manual_seed(seed)
    samples = _batch()
    select_prompt_groups(
        _args(), _config(ole_init_threshold=threshold), samples, {}, rollout_id=0
    )
    _, normalized = normalize_by_episode(_args(), samples)
    returns = [
        torch.full((3,), reward, dtype=dtype, device=device) for reward in normalized
    ]
    original_returns = [value.clone() for value in returns]
    data = {
        "advantages": list(returns),
        "returns": returns,
        "metadata": [sample.train_metadata for sample in samples],
    }
    apply_selection_to_advantages(None, 0, data)
    parameter = torch.nn.Parameter(torch.randn(4, 3, device=device, dtype=dtype))
    before = parameter.detach().clone()
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    advantages = torch.stack(data["advantages"])
    loss, _ = compute_policy_loss(-parameter.float(), advantages.float(), 0.2, 0.2)
    loss.mean().backward()
    selected = torch.tensor(
        [sample.train_metadata["ole"]["selected"] for sample in samples], device=device
    )
    assert torch.isfinite(parameter.grad).all()
    assert torch.count_nonzero(parameter.grad[~selected]) == 0
    if selected.any():
        assert torch.count_nonzero(parameter.grad[selected]) > 0
    optimizer.step()
    assert torch.equal(parameter.detach()[~selected], before[~selected])
    for actual, original in zip(returns, original_returns, strict=True):
        assert torch.equal(actual, original)
