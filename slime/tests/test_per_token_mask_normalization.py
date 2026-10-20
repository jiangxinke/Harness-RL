"""Synthetic CPU regression: masked segments cannot change token weighting."""
from types import SimpleNamespace
import _cp_dist_helpers
import pytest
import torch
from megatron.core import mpu
from slime.backends.megatron_utils import cp_utils as CP
from slime.backends.megatron_utils import loss as LOSS

NUM_GPUS = 0

def options():
    return SimpleNamespace(loss_type='policy_loss',calculate_per_token_loss=True,
        recompute_loss_function=False,rollout_top_p=1.,rollout_temperature=1.,
        log_probs_chunk_size=-1,allgather_cp=False,entropy_coef=0.,use_opsm=False,
        advantage_estimator='ppo',use_rollout_logprobs=False,eps_clip=.2,eps_clip_high=.2,
        get_mismatch_metrics=False,use_tis=False,custom_pg_loss_reducer_function_path=None,
        use_kl_loss=False)

def training_batch(count):
    return dict(total_lengths=[3]*count,response_lengths=[2]*count,
        loss_masks=[torch.tensor([1,1])]+[torch.tensor([0,0])]*(count-1),
        rollout_mask_sums=torch.tensor([2.]+[0.]*(count-1)),
        unconcat_tokens=[torch.tensor([0,1,0])]*count,
        advantages=[torch.ones(2)]*count)

def single_batch(removed):
    batch=training_batch(1)
    if removed:
        batch['loss_masks']=[torch.zeros(2,dtype=torch.int)]
        batch['rollout_mask_sums']=torch.tensor([0.])
    return batch

@pytest.fixture(autouse=True)
def topology(monkeypatch):
    monkeypatch.setattr(mpu,'get_tensor_model_parallel_group',lambda:None,raising=False)
    monkeypatch.setattr(mpu,'get_context_parallel_world_size',lambda:1)
    monkeypatch.setattr(mpu,'get_data_parallel_world_size',lambda **kwargs:1,raising=False)
    monkeypatch.setattr(CP.dist,'all_reduce',lambda *args,**kwargs:None)

@pytest.mark.parametrize('pack_together',[True,False],ids=['same-microbatch','separate-microbatches'])
def test_removed_segment_preserves_effective_token_loss_gradient_and_report(pack_together):
    def run(include_removed):
        theta=torch.tensor(.3,requires_grad=True)
        batches=([training_batch(2)] if pack_together else [single_batch(False),single_batch(True)]) if include_removed else [single_batch(False)]
        losses,counts,logs=[],[],[]
        for batch in batches:
            # Real production PPO logits/log-probs/loss/backward on invented tokens.
            n=sum(batch['total_lengths'])
            logits=torch.stack([theta.expand(n),(-theta).expand(n)],dim=-1).unsqueeze(0)
            loss,count,log=LOSS.loss_function(options(),batch,len(batches),1,logits)
            losses.append(loss); counts.append(count); logs.append(log)
        total=sum(losses)/sum(counts)
        total.backward()
        report=CP.reduce_train_step_metrics(logs,calculate_per_token_loss=True,
            step_global_batch_size=1,cp_size=1,dp_with_cp_group=None)
        return float(total.detach()),float(theta.grad),report,int(sum(counts))
    baseline,added=run(False),run(True)
    assert added[3] == baseline[3] == 2, (baseline,added)
    assert added[0] == pytest.approx(baseline[0])
    assert added[1] == pytest.approx(baseline[1])
    assert added[2] == pytest.approx(baseline[2])

def test_global_zero_actor_tokens_report_finite_zero():
    theta=torch.tensor(.3,requires_grad=True)
    logits=torch.stack([theta.expand(3),(-theta).expand(3)],dim=-1).unsqueeze(0)
    loss,count,log=LOSS.loss_function(options(),single_batch(True),1,1,logits)
    assert int(count)==0
    loss.backward()
    assert float(theta.grad)==0
    report=CP.reduce_train_step_metrics([log],calculate_per_token_loss=True,
        step_global_batch_size=1,cp_size=1,dp_with_cp_group=None)
    assert all(value==0 for value in report.values())

def test_rollout_mean_scaling_is_unchanged_with_removed_segment():
    args=options()
    args.calculate_per_token_loss=False
    theta=torch.tensor(.3,requires_grad=True)
    batch=training_batch(2)
    logits=torch.stack([theta.expand(6),(-theta).expand(6)],dim=-1).unsqueeze(0)
    reducer=CP.get_sum_of_sample_mean(batch['total_lengths'],batch['response_lengths'],
        batch['loss_masks'],batch['rollout_mask_sums'])
    unscaled,_=LOSS.policy_loss_function(args,batch,logits,reducer)
    scaled,count,log=LOSS.loss_function(args,batch,3,2,logits)
    torch.testing.assert_close(scaled,unscaled*3/2)
    assert int(count)==1
    assert float(log['values'][0])==0
