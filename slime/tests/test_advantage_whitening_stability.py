"""CPU gloo regressions for stable global masked advantage statistics.

Run with CUDA_VISIBLE_DEVICES=''. Only Megatron topology metadata is adapted;
all whitening, GAE, PPO loss, backward and distributed collectives are real.
"""
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import _cp_dist_helpers
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from megatron.core import mpu
from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean
from slime.backends.megatron_utils.loss import compute_advantages_and_returns
from slime.utils.distributed_utils import distributed_masked_whiten
from slime.utils.ppo_utils import compute_policy_loss

NUM_GPUS = 0


def _oracle(values, mask, shift_mean):
    # Independent reference: concatenate actual global observations on CPU,
    # then compute weighted centered variance in float64 without raw moments.
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, (values.tolist(), mask.tolist()))
    full_values = torch.tensor([v for vs, _ in gathered for v in vs], dtype=torch.float64)
    full_mask = torch.tensor([w for _, ws in gathered for w in ws], dtype=torch.float64)
    count = full_mask.sum()
    mean = (full_values * full_mask).sum() / count
    denominator = count - 1 if count >= 2 else count
    variance = ((full_values - mean).square() * full_mask).sum() / denominator
    expected = (values.double() - mean) / torch.sqrt(variance + 1e-8)
    return (expected if shift_mean else expected + mean).to(values.dtype), variance


def _worker(rank, world_size, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous,
        rank=rank, world_size=world_size, timeout=timedelta(seconds=60))
    try:
        _cp_dist_helpers.stub_megatron_in_worker(1, 0)
        mpu.is_pipeline_last_stage = lambda: True
        mpu.get_data_parallel_group = lambda with_context_parallel=False: dist.group.WORLD
        ordinary = torch.tensor([-.5, -.25, 0., .25, .5, .75, 1., 1.25])
        weighted_mask = torch.tensor([.5, 0., 1., 2., 0., .25, 1., 0.])
        cases = [
            ("constant", torch.full((8,), .8), torch.ones(8), True),
            ("high-offset-small-variance", 1024. + torch.tensor([-4., -3., -2., -1., 1., 2., 3., 4.]) / 128,
             torch.ones(8), True),
            ("weighted", ordinary, weighted_mask, True),
            ("shift-mean-false", ordinary, weighted_mask, False),
            ("fractional-count-below-two", torch.tensor([.2, .8]) if rank == 0 else torch.empty(0),
             torch.tensor([.25, .5]) if rank == 0 else torch.empty(0), True),
            ("singleton", torch.tensor([.8]) if rank == 0 else torch.empty(0),
             torch.ones(1) if rank == 0 else torch.empty(0), True),
        ]
        if world_size == 2:
            cases.extend([
                ("empty-local-mask", torch.full((8,), .8), torch.ones(8) if rank == 0 else torch.zeros(8), True),
                ("empty-local-tensor", torch.full((8,), .8) if rank == 0 else torch.empty(0),
                 torch.ones(8) if rank == 0 else torch.empty(0), True),
            ])
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            # Half precision raw square/sum statistics overflow or round away
            # variance; input observations themselves remain finite/exact.
            cases.append((str(dtype), torch.arange(128, 136).to(dtype), torch.ones(8), True))
        for name, values, mask, shift_mean in cases:
            expected, variance = _oracle(values, mask, shift_mean)
            actual = distributed_masked_whiten(values, mask, dist.group.WORLD, shift_mean=shift_mean)
            assert actual.dtype == values.dtype, name
            assert torch.isfinite(actual).all(), name
            tolerance = {torch.float16: 1e-3, torch.bfloat16: 8e-3}.get(values.dtype, 2e-6)
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance, msg=name)
            if name == "constant":
                assert torch.equal(actual, torch.zeros_like(actual))
            if name == "high-offset-small-variance":
                assert 0 < variance < .01  # This is not a zero-variance control.
        with pytest.raises(ValueError, match=r"^The global mask sum across all participating GPUs is zero\.$"):
            distributed_masked_whiten(torch.tensor([.8]), torch.zeros(1), dist.group.WORLD)

        # Exercise the complete original PPO path, including its actual GAE.
        # Under gamma=lambda=1, zero critic/KL and terminal reward .8, all raw
        # advantages are .8; normalization must yield zero, not NaN or an offset.
        masks = [torch.ones(8) if rank == 0 else torch.zeros(8)]
        args = SimpleNamespace(use_rollout_logprobs=False, kl_coef=0., kl_loss_type="k1",
            custom_advantage_function_path=None, advantage_estimator="ppo", gamma=1.,
            lambd=1., use_opd=False, normalize_advantages=False)
        def batch():
            return dict(log_probs=[torch.full((8,), -1.)], values=[torch.zeros(8)], rewards=[.8],
                total_lengths=[12], response_lengths=[8], loss_masks=masks)
        raw = batch()
        compute_advantages_and_returns(args, raw)
        torch.testing.assert_close(raw["advantages"][0], torch.full((8,), .8), rtol=0, atol=0)
        args.normalize_advantages = True
        normalized = batch()
        compute_advantages_and_returns(args, normalized)
        assert torch.equal(normalized["advantages"][0], torch.zeros(8))
        torch.testing.assert_close(normalized["returns"][0], raw["returns"][0], rtol=0, atol=0)
        theta = torch.tensor(0., requires_grad=True)
        token_loss, _ = compute_policy_loss(-theta.expand(8), normalized["advantages"][0], .2, .2)
        loss = get_sum_of_sample_mean([12], [8], masks)(token_loss)
        loss.backward()
        assert torch.isfinite(loss) and loss.item() == 0.
        assert torch.isfinite(theta.grad) and theta.grad.item() == 0.
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2])
def test_global_masked_whitening_is_stable_and_ppo_backward_finite(tmp_path, world_size):
    # One spawn per topology; all numerical/masking/dtype cases share workers.
    mp.spawn(_worker, args=(world_size, str(Path(tmp_path) / "gloo")), nprocs=world_size, join=True)
