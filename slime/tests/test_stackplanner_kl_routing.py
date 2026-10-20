"""Configured reference KL must affect the selected production objective."""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import _cp_dist_helpers  # Install the repository's CPU parallel-metadata adapter first.
import pytest
import torch
from megatron.core import mpu
from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean
from slime.backends.megatron_utils.loss import (
    compute_advantages_and_returns,
    get_log_probs_and_entropy,
    policy_loss_function,
)
from slime_plugins.stackplanner.cli import build_slime_command
from slime_plugins.stackplanner.config import DecouplingScope, Experiment, TrainingConfig, TrainingMode

NUM_GPUS = 0
ROOT = Path(__file__).resolve().parents[2]


def _kl_options(config):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--kl-coef', type=float, default=0.)
    parser.add_argument('--kl-loss-coef', type=float, default=0.)
    parser.add_argument('--use-kl-loss', action='store_true')
    parser.add_argument('--advantage-estimator')
    parsed, _ = parser.parse_known_args(build_slime_command(config)[2:])
    return parsed


@pytest.mark.parametrize('preset', ['experiment_b_all_agents.yaml', 'experiment_c_capability_decoupled.yaml'])
def test_nonzero_grpo_presets_enable_reference_kl_loss(preset):
    raw = json.loads((ROOT / 'configs' / preset).read_text())
    raw['mode'] = TrainingMode(raw['mode'])
    raw['experiment'] = Experiment(raw['experiment'])
    raw['decoupling_scope'] = DecouplingScope(raw['decoupling_scope'])
    # Exercise real preset command rendering. Capability-evidence preflight is
    # separate and would require external artifacts unrelated to KL routing.
    config = TrainingConfig(**raw)
    options = _kl_options(config)
    assert config.kl_coefficient == .001
    assert options.use_kl_loss
    assert options.kl_loss_coef == .001
    assert options.kl_coef == 0.


@pytest.mark.parametrize('algorithm', ['grpo', 'gspo', 'cispo', 'ppo'])
@pytest.mark.parametrize('coefficient', [0., .001])
def test_kl_route_preserves_zero_default_and_ppo_shaping(algorithm, coefficient):
    config = TrainingConfig(run_id='kl-routing', algorithm=algorithm, kl_coefficient=coefficient)
    command = build_slime_command(config)
    options = _kl_options(config)
    assert options.advantage_estimator == algorithm
    if algorithm == 'ppo':
        assert options.kl_coef == coefficient
        assert not options.use_kl_loss
        assert '--kl-loss-coef' not in command
    elif coefficient:
        assert options.kl_coef == 0.
        assert options.use_kl_loss
        assert options.kl_loss_coef == coefficient
    else:
        assert options.kl_coef == 0.
        assert not options.use_kl_loss
        assert '--kl-loss-coef' not in command


@pytest.mark.parametrize('algorithm', ['grpo', 'gspo', 'cispo'])
def test_emitted_group_kl_flags_change_real_loss_and_gradient(monkeypatch, algorithm):
    monkeypatch.setattr(mpu, 'is_pipeline_last_stage', lambda: True, raising=False)
    monkeypatch.setattr(mpu, 'get_tensor_model_parallel_group', lambda: None, raising=False)
    monkeypatch.setattr(mpu, 'get_context_parallel_group', lambda: None, raising=False)
    monkeypatch.setattr(mpu, 'get_context_parallel_world_size', lambda: 1)
    base = dict(use_rollout_logprobs=False, kl_loss_type='k3', custom_advantage_function_path=None,
        use_opd=False, normalize_advantages=False, rollout_top_p=1., rollout_temperature=1.,
        log_probs_chunk_size=-1, allgather_cp=False, entropy_coef=0., use_opsm=False,
        eps_clip=.2, eps_clip_high=.2, get_mismatch_metrics=False, use_tis=False,
        custom_pg_loss_reducer_function_path=None, use_unbiased_kl=False)
    tokens = [torch.tensor([0, 1, 0])]
    initial = torch.tensor([[[.3, -.2], [.1, -.4], [0., 0.]]])
    losses, gradients = [], []
    for coefficient in (0., .001):
        config = TrainingConfig(run_id='kl-backward', algorithm=algorithm, kl_coefficient=coefficient)
        args = SimpleNamespace(**base, **vars(_kl_options(config)))
        _, old = get_log_probs_and_entropy(initial, args=args, unconcat_tokens=tokens,
            total_lengths=[3], response_lengths=[2])
        old = old['log_probs'][0].detach()
        data = dict(log_probs=[old], ref_log_probs=[old - 1.], rewards=[1.],
            total_lengths=[3], response_lengths=[2], loss_masks=[torch.ones(2)],
            unconcat_tokens=tokens, rollout_mask_sums=torch.tensor([2.]))
        compute_advantages_and_returns(args, data)
        reducer = get_sum_of_sample_mean([3], [2], data['loss_masks'], data['rollout_mask_sums'])
        logits = initial.clone().requires_grad_()
        loss, metrics = policy_loss_function(args, data, logits, reducer)
        loss.backward()
        assert ('kl_loss' in metrics) == bool(coefficient)
        losses.append(loss.item())
        gradients.append(logits.grad)
    # k3 KL for log-policy minus log-reference = 1 is exp(-1)-1+1.
    assert losses[1] - losses[0] == pytest.approx(.001 * math.exp(-1), abs=1e-7)
    assert not torch.equal(gradients[0], gradients[1])
    assert not torch.cuda.is_initialized()
