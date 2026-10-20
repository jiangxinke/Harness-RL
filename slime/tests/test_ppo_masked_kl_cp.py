"""PPO KL rewards must exclude tool tokens before GAE, including under CP."""
from datetime import timedelta
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


def serial_gae(values, rewards, gamma, lambd):
    advantages = torch.zeros_like(values)
    carry = values.new_zeros(())
    for i in reversed(range(len(values))):
        next_value = values[i + 1] if i + 1 < len(values) else 0.
        carry = rewards[i] + gamma * next_value - values[i] + gamma * lambd * carry
        advantages[i] = carry
    return advantages, advantages + values


def worker(rank, cp_size, rendezvous, use_rollout, kl_coef):
    dist.init_process_group("gloo", init_method="file://" + rendezvous,
        rank=rank, world_size=cp_size, timeout=timedelta(seconds=90))
    try:
        _cp_dist_helpers.stub_megatron_in_worker(cp_size, rank)
        mpu.is_pipeline_last_stage = lambda: True
        mpu.get_context_parallel_group = lambda: dist.group.WORLD
        args = SimpleNamespace(use_rollout_logprobs=use_rollout, kl_coef=kl_coef,
            kl_loss_type="k1", custom_advantage_function_path=None,
            advantage_estimator="ppo", gamma=.9, lambd=.8, use_opd=False,
            normalize_advantages=False)
        # Middle tool spans must not influence earlier trainable advantages.
        # Also include all-one masks, a fully masked sample, an empty response,
        # and a one-token response with an empty local CP slice.
        totals = [5, 12, 9, 5, 10, 7]
        responses = [3, 6, 6, 0, 1, 2]
        scalar_rewards = [0., 1.25, -.4, 99., .8, .3]
        masks = [torch.tensor(v) for v in ([1.,0.,1.], [1.,0.,0.,1.,0.,1.],
            [1.,1.,1.,1.,1.,1.], [], [1.], [0.,0.])]
        values = [torch.linspace(-.2, .4, length) for length in responses]
        old_logps = [torch.where(mask.bool(), torch.full_like(mask, -1.2), torch.zeros_like(mask))
            for mask in masks]
        def local(ts):
            return [slice_log_prob_with_cp(t, total, response)
                for t, total, response in zip(ts, totals, responses, strict=True)]
        actual_runs = []
        for masked_ref in (-10., 0.):
            refs = [torch.where(mask.bool(), torch.full_like(mask, -.9), torch.full_like(mask, masked_ref))
                for mask in masks]
            data = dict(ref_log_probs=local(refs), values=local(values), rewards=scalar_rewards,
                total_lengths=totals, response_lengths=responses, loss_masks=masks)
            data["rollout_log_probs" if use_rollout else "log_probs"] = local(old_logps)
            compute_advantages_and_returns(args, data)
            run = []
            for i, (total, response) in enumerate(zip(totals, responses, strict=True)):
                actual_adv = all_gather_with_cp(data["advantages"][i], total, response)
                actual_return = all_gather_with_cp(data["returns"][i], total, response)
                token_rewards = -kl_coef * (old_logps[i] - refs[i]) * masks[i]
                if response:
                    token_rewards[-1] += scalar_rewards[i]
                expected_adv, expected_return = serial_gae(values[i], token_rewards, args.gamma, args.lambd)
                torch.testing.assert_close(actual_adv, expected_adv, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(actual_return, expected_return, rtol=1e-5, atol=1e-6)
                # KL diagnostics must remain raw KL, rather than aliased shaped rewards.
                torch.testing.assert_close(data["kl"][i], local([(old_logps[j]-refs[j]) if kl_coef else torch.zeros_like(old_logps[j])
                    for j in range(len(totals))])[i], rtol=0, atol=0)
                run.append(actual_adv.tolist())
            actual_runs.append(run)
        for before, after in zip(*actual_runs, strict=True):
            torch.testing.assert_close(torch.tensor(before), torch.tensor(after), rtol=0, atol=0)
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("cp_size,use_rollout,kl_coef", [(1, False, .2), (2, True, .2), (2, False, 0.)])
def test_masked_tool_kl_never_changes_trainable_gae(tmp_path, cp_size, use_rollout, kl_coef):
    mp.spawn(worker, args=(cp_size, str(tmp_path / "rendezvous"),
        use_rollout, kl_coef), nprocs=cp_size, join=True)
