"""PPO terminal rewards must follow the response-token CP layout.

Only Megatron's parallel-group metadata uses the existing CPU adapter. The
production advantage/CP/GAE functions and torch.distributed gloo collectives
are real; neither model forwards nor CUDA are needed for these arithmetic
contracts.
"""
from datetime import timedelta
from types import SimpleNamespace
import json
from pathlib import Path

import _cp_dist_helpers
import pytest
import torch
import torch.distributed as dist
import torch.distributed.nn
import torch.multiprocessing as mp
from megatron.core import mpu
from slime.backends.megatron_utils.cp_utils import all_gather_with_cp, slice_log_prob_with_cp
from slime.backends.megatron_utils.loss import compute_advantages_and_returns

NUM_GPUS = 0


def _serial_gae(values, rewards, gamma, lambd):
    """Direct scalar recurrence, independent of production's batched GAE."""
    advantage = torch.zeros_like(values)
    next_value = values.new_zeros(())
    carry = values.new_zeros(())
    for index in reversed(range(len(values))):
        delta = rewards[index] + gamma * next_value - values[index]
        carry = delta + gamma * lambd * carry
        advantage[index] = carry
        next_value = values[index]
    return advantage, advantage + values


def _worker(rank, cp_size, rendezvous, result_path, with_kl, include_empty_local):
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
        rank=rank, world_size=cp_size, timeout=timedelta(seconds=90))
    try:
        _cp_dist_helpers.stub_megatron_in_worker(cp_size, rank)
        mpu.is_pipeline_last_stage = lambda: True
        mpu.get_context_parallel_group = lambda: dist.group.WORLD
        args = SimpleNamespace(use_rollout_logprobs=False, kl_coef=.2 if with_kl else 0.,
            kl_loss_type='k1', custom_advantage_function_path=None,
            advantage_estimator='ppo', gamma=.9, lambd=.8, use_opd=False,
            normalize_advantages=False)
        # Keep the padded-token numerical failure separate from empty local
        # responses, so the former has a direct before/after assertion failure.
        total_lengths = [12, 9, 17, 8]
        response_lengths = [6, 6, 8, 3]
        rewards = [1., 2., .75, 0.]
        if include_empty_local:
            total_lengths += [10, 7, 5]
            response_lengths += [1, 2, 0]
            rewards += [-.5, 1., 99.]
        full_values = [torch.linspace(-.2, .4, length) for length in response_lengths]
        full_logps = [torch.linspace(-1.5, -.5, length) for length in response_lengths]
        full_ref_logps = [torch.full((length,), -.9) for length in response_lengths]

        def local(tensors):
            return [slice_log_prob_with_cp(t, total, response)
                for t, total, response in zip(tensors, total_lengths, response_lengths, strict=True)]

        data = dict(log_probs=local(full_logps), ref_log_probs=local(full_ref_logps),
            values=local(full_values), rewards=rewards, total_lengths=total_lengths,
            response_lengths=response_lengths,
            loss_masks=[torch.ones(length) for length in response_lengths])
        compute_advantages_and_returns(args, data)
        actual = []
        for index, (total, response, reward) in enumerate(zip(total_lengths, response_lengths, rewards, strict=True)):
            actual_adv = all_gather_with_cp(data['advantages'][index], total, response)
            actual_return = all_gather_with_cp(data['returns'][index], total, response)
            token_rewards = -args.kl_coef * (full_logps[index] - full_ref_logps[index])
            if response > 0:
                token_rewards[-1] += reward
            expected_adv, expected_return = _serial_gae(full_values[index], token_rewards, args.gamma, args.lambd)
            torch.testing.assert_close(actual_adv, expected_adv, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(actual_return, expected_return, atol=1e-6, rtol=1e-5)
            actual.append(actual_return.tolist())
        assert not torch.cuda.is_initialized()
        if rank == 0:
            Path(result_path).write_text(json.dumps(actual))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('with_kl', [False, True])
@pytest.mark.parametrize('include_empty_local', [False, True])
def test_ppo_terminal_reward_matches_serial_with_and_without_cp(tmp_path, with_kl, include_empty_local):
    by_cp = []
    for cp_size in (1, 2):
        result = tmp_path / f'result-{cp_size}.json'
        mp.spawn(_worker, args=(cp_size, str(tmp_path / f'rendezvous-{cp_size}'),
            str(result), with_kl, include_empty_local), nprocs=cp_size, join=True)
        by_cp.append(json.loads(result.read_text()))
    for cp1, cp2 in zip(*by_cp, strict=True):
        torch.testing.assert_close(torch.tensor(cp1), torch.tensor(cp2), atol=1e-6, rtol=1e-5)
