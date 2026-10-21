"""Removed and empty REINFORCE++ rows must preserve valid returns under CP."""
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import _cp_dist_helpers
import pytest
import torch
import torch.distributed as dist
import torch.distributed.nn
import torch.multiprocessing as mp
from megatron.core import mpu

from slime.backends.megatron_utils import cp_utils
from slime.backends.megatron_utils.loss import compute_advantages_and_returns

NUM_GPUS = 0


def _discounted_oracle(kl, mask, reward, kl_coef, gamma):
    """Direct discounted sums, independent of the production recurrence."""
    actor_positions = [index for index, keep in enumerate(mask.tolist()) if keep]
    if not actor_positions:
        return torch.zeros_like(kl)
    token_rewards = [-kl_coef * value if keep else 0.
                     for value, keep in zip(kl.tolist(), mask.tolist(), strict=True)]
    token_rewards[actor_positions[-1]] += reward
    return torch.tensor([
        sum(gamma ** (future - index) * token_rewards[future]
            for future in range(index, len(token_rewards)))
        for index in range(len(token_rewards))
    ], dtype=kl.dtype)


def _worker(rank, cp_size, rendezvous, kl_coef, gamma):
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
        rank=rank, world_size=cp_size, timeout=timedelta(seconds=90))
    try:
        _cp_dist_helpers.stub_megatron_in_worker(cp_size, rank)
        mpu.is_pipeline_last_stage = lambda: True
        mpu.get_context_parallel_group = lambda: dist.group.WORLD
        mpu.get_data_parallel_group = lambda with_context_parallel: dist.group.WORLD
        args = SimpleNamespace(use_rollout_logprobs=True, kl_coef=kl_coef,
            kl_loss_type='k1', custom_advantage_function_path=None,
            advantage_estimator='reinforce_plus_plus', gamma=gamma,
            use_opd=False, normalize_advantages=True)

        # The one-token valid row has an empty CP2 slice. The first valid
        # row ends in a tool token, so reward belongs on its last actor token.
        totals = [12, 10, 9, 2, 7, 5]
        responses = [6, 1, 4, 1, 3, 0]
        masks = [torch.tensor(values, dtype=torch.float32) for values in
                 ([1, 0, 1, 0, 1, 0], [1], [0, 1, 0, 1], [0], [0, 0, 0], [])]
        rewards = [1.2, -.4, .7, 99., -99., 123.]
        raw_kl = [torch.linspace(-.35, .55, length) for length in responses]
        old = [values - .9 for values in raw_kl]
        refs = [torch.full_like(values, -.9) for values in raw_kl]
        actual_kl = [before - reference for before, reference in zip(old, refs, strict=True)]

        def local(values, count):
            return [cp_utils.slice_log_prob_with_cp(value, total, response)
                    for value, total, response in zip(
                        values[:count], totals[:count], responses[:count], strict=True)]

        gather = cp_utils.all_gather_with_cp
        gathered_shapes = []

        def record_gather(tensor, total, response):
            gathered_shapes.append((total, response))
            return gather(tensor, total, response)

        runs = []
        for count in (3, 6):
            data = dict(rollout_log_probs=local(old, count), ref_log_probs=local(refs, count),
                rewards=rewards[:count], total_lengths=totals[:count],
                response_lengths=responses[:count], loss_masks=masks[:count])
            # Record calls while keeping the actual Gloo CP collectives. Even
            # zero-mask and empty-response rows must follow the gather path.
            before = len(gathered_shapes)
            with patch.object(cp_utils, 'all_gather_with_cp', record_gather):
                compute_advantages_and_returns(args, data)
            expected_gathers = list(zip(totals[:count], responses[:count], strict=True)) if cp_size > 1 else []
            assert gathered_shapes[before:] == expected_gathers
            runs.append(data)
            for index in range(count):
                actual = gather(data['returns'][index], totals[index], responses[index])
                expected = _discounted_oracle(actual_kl[index], masks[index], rewards[index], kl_coef, gamma)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                if index >= 3:
                    assert torch.equal(data['returns'][index], torch.zeros_like(data['returns'][index]))

        # Zero-mask rows cannot change the valid-token whitening denominator,
        # mean or variance, even with large irrelevant scalar rewards.
        for field in ('returns', 'advantages'):
            for baseline, mixed in zip(runs[0][field], runs[1][field][:3], strict=True):
                torch.testing.assert_close(baseline, mixed, atol=1e-6, rtol=1e-5)
        if cp_size == 2:
            empty_valid_slice = int(runs[1]['returns'][1].numel() == 0)
            empty_count = torch.tensor(empty_valid_slice)
            dist.all_reduce(empty_count)
            assert empty_count.item() == 1
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('cp_size', [pytest.param(1, id='cp1'), pytest.param(2, id='cp2')])
@pytest.mark.parametrize('kl_coef,gamma', [
    pytest.param(.2, .9, id='discounted_kl'),
    pytest.param(0., 1., id='undiscounted_no_kl'),
])
def test_reinforce_returns_ignore_removed_and_empty_rows(tmp_path, cp_size, kl_coef, gamma):
    mp.spawn(_worker, args=(cp_size, str(tmp_path / 'rendezvous'), kl_coef, gamma),
             nprocs=cp_size, join=True)
