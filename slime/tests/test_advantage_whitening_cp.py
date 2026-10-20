"""Global masked advantage whitening must not depend on the CP partition.

Real CPU gloo collectives exercise the production GAE, CP slicing and
whitening. Only Megatron parallel-group metadata is adapted for CPU.
"""
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace

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


def _inputs(dp_rank, short):
    if short:
        totals, responses = [10], [1]
        rewards = [1. + dp_rank]
        masks = [torch.ones(1, dtype=torch.int)]
    else:
        totals, responses = [12, 9], [8, 4]
        rewards = [1. + dp_rank * .3, -.4 + dp_rank * .5]
        masks = [torch.tensor([1, 0, 1, 1, 0, 1, 1, 1]), torch.tensor([0, 1, 1, 1])]
    values = [torch.linspace(-.2, .4, length) + dp_rank * .1 for length in responses]
    logps = [torch.linspace(-1.5, -.5, length) for length in responses]
    ref_logps = [torch.full((length,), -.9) for length in responses]
    return totals, responses, rewards, masks, values, logps, ref_logps


def _serial_advantages(dp_rank, short):
    totals, responses, rewards, masks, values, logps, refs = _inputs(dp_rank, short)
    advantages = []
    for reward, value, logp, ref, mask in zip(rewards, values, logps, refs, masks, strict=True):
        # Only actor tokens contribute KL rewards before the serial GAE.
        token_rewards = -.2 * (logp - ref) * mask
        token_rewards[-1] += reward
        result = torch.zeros_like(value)
        next_value, carry = value.new_zeros(()), value.new_zeros(())
        for token in reversed(range(len(value))):
            delta = token_rewards[token] + .9 * next_value - value[token]
            carry = delta + .9 * .8 * carry
            result[token], next_value = carry, value[token]
        advantages.append(result)
    return advantages


def _worker(rank, cp_size, dp_size, short, rendezvous, result_path):
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
        rank=rank, world_size=cp_size * dp_size, timeout=timedelta(seconds=45))
    try:
        cp_rank, dp_rank = rank % cp_size, rank // cp_size
        _cp_dist_helpers.stub_megatron_in_worker(cp_size, cp_rank)
        cp_groups = [dist.new_group(list(range(dp * cp_size, (dp + 1) * cp_size)))
                     for dp in range(dp_size)]
        dp_groups = [dist.new_group([dp * cp_size + cp for dp in range(dp_size)])
                     for cp in range(cp_size)]
        mpu.is_pipeline_last_stage = lambda: True
        mpu.get_context_parallel_group = lambda: cp_groups[dp_rank]
        requested_groups = []
        def dp_group(with_context_parallel=False):
            requested_groups.append(with_context_parallel)
            return dist.group.WORLD if with_context_parallel else dp_groups[cp_rank]
        mpu.get_data_parallel_group = dp_group
        args = SimpleNamespace(use_rollout_logprobs=False, kl_coef=.2, kl_loss_type='k1',
            custom_advantage_function_path=None, advantage_estimator='ppo', gamma=.9,
            lambd=.8, use_opd=False, normalize_advantages=False)
        totals, responses, rewards, masks, values, logps, refs = _inputs(dp_rank, short)
        def local(tensors):
            return [slice_log_prob_with_cp(t, total, response) if cp_size > 1 else t
                    for t, total, response in zip(tensors, totals, responses, strict=True)]
        data = dict(log_probs=local(logps), ref_log_probs=local(refs), values=local(values),
            rewards=rewards, total_lengths=totals, response_lengths=responses, loss_masks=masks)
        raw_expected = _serial_advantages(dp_rank, short)
        compute_advantages_and_returns(args, data)
        for actual, expected in zip(data['advantages'], local(raw_expected), strict=True):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        assert requested_groups == []  # Default-disabled normalization is unchanged.
        raw_returns = [t.clone() for t in data['returns']]
        args.normalize_advantages = True
        compute_advantages_and_returns(args, data)
        assert requested_groups == [True]  # Include CP exactly once, including empty slices.
        for actual, expected in zip(data['returns'], raw_returns, strict=True):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

        # Independent masked global statistics: each actor token across DP is
        # counted once, independent of its CP owner or integer mask dtype.
        valid = []
        for dp in range(dp_size):
            all_masks = _inputs(dp, short)[3]
            valid.extend(a[m.bool()] for a, m in zip(_serial_advantages(dp, short), all_masks, strict=True))
        eligible = torch.cat(valid)
        mean = eligible.mean()
        variance = eligible.var(unbiased=True) if eligible.numel() > 1 else eligible.new_zeros(())
        expected = [(a - mean) * torch.rsqrt(variance + 1e-8) for a in raw_expected]
        actual_full = []
        for actual, full_expected, total, response in zip(data['advantages'], expected, totals, responses, strict=True):
            full = all_gather_with_cp(actual, total, response) if cp_size > 1 else actual
            torch.testing.assert_close(full, full_expected, atol=1e-6, rtol=1e-5)
            actual_full.append(full.tolist())
        if short and cp_size > 1:
            # At least one CP rank has no local advantage or mask elements.
            empty = torch.tensor(int(sum(t.numel() for t in data['advantages']) == 0))
            dist.all_reduce(empty, group=cp_groups[dp_rank])
            assert empty.item() > 0
        results = [None] * (cp_size * dp_size) if rank == 0 else None
        dist.gather_object({'dp_rank': dp_rank, 'cp_rank': cp_rank, 'advantages': actual_full},
                           results, dst=0)
        if rank == 0:
            Path(result_path).write_text(json.dumps([r['advantages'] for r in results if r['cp_rank'] == 0]))
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('dp_size', [1, 2])
@pytest.mark.parametrize('short', [False, True], ids=['masked-variable-responses', 'empty-local-cp-slices'])
def test_whitened_advantages_match_serial_and_cp1_with_global_masked_stats(tmp_path, dp_size, short):
    results = []
    for cp_size in (1, 2):
        output = tmp_path / f'cp{cp_size}.json'
        mp.spawn(_worker, args=(cp_size, dp_size, short, str(tmp_path / f'cp{cp_size}.rendezvous'),
                              str(output)), nprocs=cp_size * dp_size, join=True)
        results.append(json.loads(output.read_text()))
    for dp1, dp2 in zip(*results, strict=True):
        for cp1, cp2 in zip(dp1, dp2, strict=True):
            torch.testing.assert_close(torch.tensor(cp1), torch.tensor(cp2), atol=1e-6, rtol=1e-5)
