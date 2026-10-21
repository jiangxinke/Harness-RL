"""Policy-gradient oracles, failure coverage, and real LoRA replay checks."""
from __future__ import annotations

import importlib.util
import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from slime_plugins.dcc.episode import EpisodeResult
from slime_plugins.dcc.native_behavior_logprobs import BEHAVIOR_LOGPROB_SOURCE
from slime_plugins.dcc.online_grpo import (apply_policy_update, group_signal,
                                          thought_policy_loss)


def episode(reward, *, status="completed", thought=(6, 7), json_ids=(8, 9)):
    step = SimpleNamespace(prompt_ids=[3, 4, 5],
        completion_ids=list(thought) + list(json_ids), thought_token_count=len(thought),
        thought_logprobs=[-3.5] * len(thought), thought_logprobs_source=BEHAVIOR_LOGPROB_SOURCE)
    return EpisodeResult("task", "cuad", "train", [], status, reward, {},
                         [step], [], [])


def signal(results):
    return group_signal(results, group_size=len(results), max_steps=2, thought_tokens=8)


def test_failed_model_thoughts_remain_negative_training_examples():
    success = episode(1)
    failed = episode(0, status="model_protocol_failure", thought=(10,), json_ids=())
    assert failed.trainable_steps == []  # Legacy evaluation property is unchanged.
    group = signal([success, failed])
    assert group.discarded_reason is None
    assert group.advantages[0] > 0 > group.advantages[1]
    assert group.segments[0][0].completion_ids == [6, 7]
    assert group.segments[1][0].completion_ids == [10]
    assert all(8 not in s.completion_ids and 9 not in s.completion_ids
               for segments in group.segments for s in segments)


@pytest.mark.parametrize("reward", [0.0, 0.1, float(torch.tensor(0.1)), 1.0])
def test_equal_rewards_have_exactly_zero_advantages(reward):
    group = signal([episode(reward) for _ in range(7)])
    assert group.advantages == [0.0] * 7
    assert group.reward_std == 0


@pytest.mark.parametrize("status", ["infrastructure_error", "context_budget_failure"])
def test_infrastructure_discards_whole_group_without_dropping_only_failures(status):
    group = signal([episode(1), episode(0, status=status)])
    assert group.discarded_reason
    assert group.advantages == [0, 0] and group.segments == [[], []]


def test_group_rejects_heldout_or_untyped_token_boundaries():
    heldout = episode(1)
    heldout.split = "validation"
    with pytest.raises(ValueError, match="same task"):
        signal([heldout, episode(0)])
    other = episode(0)
    other.steps[0].thought_token_count = True
    with pytest.raises(ValueError, match="boundaries"):
        signal([episode(1), other])


def test_fixed_budget_loss_matches_analytic_categorical_policy_gradient():
    logits = torch.tensor([0.2, -0.4, 0.1], dtype=torch.float64, requires_grad=True)
    logprobs = logits.log_softmax(-1)
    loss = thought_policy_loss(logprobs[[0, 1]], logprobs[[0, 1]].detach(), 0.7,
        group_size=4, max_steps=2, thought_tokens=8)
    actual = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    oracle = -0.7 * (logprobs[0] + logprobs[1]) / 64
    expected = torch.autograd.grad(oracle, logits)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_response_length_does_not_change_each_tokens_weight():
    gradients = []
    for length in (1, 7):
        current = torch.zeros(length, requires_grad=True)
        thought_policy_loss(current, current.detach(), -1.0,
            group_size=2, max_steps=2, thought_tokens=8).backward()
        gradients.append(current.grad)
    assert gradients[0][0] == gradients[1][0] == 1 / 32


def test_zero_signal_batch_does_not_apply_existing_adam_momentum():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    parameter.grad = torch.tensor([1.0])
    norm, applied = apply_policy_update([parameter], optimizer)
    assert norm > 0 and applied
    before = parameter.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    norm, applied = apply_policy_update([parameter], optimizer)
    assert norm == 0 and not applied
    assert torch.equal(parameter.detach(), before)
    assert optimizer.state[parameter]["step"] == 1


def test_nonfinite_global_gradient_never_steps_optimizer():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    parameter.grad = torch.tensor([float("nan")])
    with pytest.raises(FloatingPointError):
        apply_policy_update([parameter], optimizer)
    assert parameter.item() == 1.0 and not optimizer.state


def load_trainer():
    spec = importlib.util.spec_from_file_location("dcc_online_grpo_tests",
        Path(__file__).resolve().parents[1] / "scripts/dcc_online_grpo.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    return trainer


def test_real_qwen3_moe_lora_replay_improves_rewarded_thought_odds():
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    from slime_plugins.dcc.hf_policy import completion_logprobs
    trainer = load_trainer()
    torch.manual_seed(13)
    config = Qwen3MoeConfig(vocab_size=32, hidden_size=16, intermediate_size=24,
        moe_intermediate_size=8, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, num_experts=4,
        num_experts_per_tok=2, attention_dropout=0.0)
    model = get_peft_model(Qwen3MoeForCausalLM(config), LoraConfig(
        r=2, lora_alpha=4, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=0.03, weight_decay=0)
    model.eval()
    def odds():
        with torch.no_grad():
            return (completion_logprobs(model, [3, 4, 5], [6])
                    - completion_logprobs(model, [3, 4, 5], [7])).item()
    before = odds()
    results = [episode(1, thought=(6,)),
               episode(0, status="model_protocol_failure", thought=(7,), json_ids=())]
    # This optimizer oracle uses a recorded synthetic behavior distribution;
    # native cached-generation capture is tested separately against output_logits.
    for result in results:
        step = result.steps[0]
        with torch.no_grad():
            step.thought_logprobs = completion_logprobs(model, step.prompt_ids,
                step.completion_ids[:step.thought_token_count]).tolist()
    group = signal(results)
    report = trainer.replay_group(model, group, completion_logprobs,
        group_size=2, max_steps=2, thought_tokens=8, clip=0.2)
    norm, applied = apply_policy_update(parameters, optimizer)
    model.eval()
    assert report["scored_tokens"] == 2 and report["scored_segments"] == 2
    assert report["maximum_replay_current_logprob_difference"] < 1e-5
    assert applied and norm > 0
    assert odds() > before
    assert not torch.cuda.is_initialized()


def test_unlabeled_web_data_is_rejected_before_any_training():
    from slime_plugins.dcc.data import Task
    trainer = load_trainer()
    task = Task("public-web", "widesearch", "train", "Find entities", gold=None,
                metadata={"source": "independent_curriculum"})
    with pytest.raises(ValueError, match="reward labels"):
        trainer.validate_train_tasks([task])
    labeled = Task("labeled-web", "widesearch", "train", "Find entities", gold=[{"name": "entity"}],
                   metadata={"source": "independent_curriculum"})
    with pytest.raises(ValueError, match="CUAD only"):
        trainer.validate_train_tasks([labeled])


def test_group_exception_drains_sibling_before_policy_close():
    trainer = load_trainer()
    events = []
    class Runner:
        counter = 0
        async def run(self, task, policy, **kwargs):
            self.counter += 1
            if self.counter == 1:
                await asyncio.sleep(0.01)
                raise RuntimeError("original rollout error")
            try:
                await asyncio.sleep(60)
            finally:
                events.append("sibling_cancelled")
    class Policy:
        def bind_task(self, *args):
            return self
        async def close(self):
            assert events == ["sibling_cancelled"]
            events.append("policy_drained")
    with pytest.raises(RuntimeError, match="original rollout error"):
        asyncio.run(trainer.collect_group(Runner(), SimpleNamespace(id="task"), Policy(),
                                           group_size=2, rank=0))
    assert events == ["sibling_cancelled", "policy_drained"]


def _distributed_update_worker(rank, rendezvous, output):
    import torch.distributed as distributed
    from datetime import timedelta
    distributed.init_process_group("gloo", init_method="file://" + rendezvous,
        rank=rank, world_size=2, timeout=timedelta(seconds=60))
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1, weight_decay=0)
    # One rank has no local policy signal; both must apply the global update.
    parameter.grad = torch.tensor([2.0]) if rank == 0 else None
    norm, applied = apply_policy_update([parameter], optimizer,
        world_size=2, distributed=distributed)
    assert norm == 1.0 and applied
    torch.testing.assert_close(parameter.detach(), torch.tensor([0.9]))
    before = parameter.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    norm, applied = apply_policy_update([parameter], optimizer,
        world_size=2, distributed=distributed)
    assert norm == 0 and not applied and torch.equal(parameter.detach(), before)
    # Opposing local gradients also cancel before the decision to step Adam.
    parameter.grad = torch.tensor([1.0 if rank == 0 else -1.0])
    norm, applied = apply_policy_update([parameter], optimizer,
        world_size=2, distributed=distributed)
    assert norm == 0 and not applied and torch.equal(parameter.detach(), before)
    Path(output + str(rank)).write_text("passed")
    distributed.destroy_process_group()


def test_real_gloo_global_update_and_zero_signal_momentum(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_distributed_update_worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path / "result-")),
        nprocs=2, join=True)
    assert [(tmp_path / f"result-{rank}").read_text() for rank in range(2)] == ["passed", "passed"]


def test_real_trainer_closes_checkpoints_and_distinguishes_zero_update_iteration(tmp_path, monkeypatch):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    import slime_plugins.dcc.hf_native_policy as native
    trainer = load_trainer()
    torch.manual_seed(19)
    model = get_peft_model(Qwen3MoeForCausalLM(Qwen3MoeConfig(
        vocab_size=32, hidden_size=16, intermediate_size=24, moe_intermediate_size=8,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, num_experts=4, num_experts_per_tok=2, attention_dropout=0.0)),
        LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0,
                   target_modules=["q_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    class Tokenizer:
        def save_pretrained(self, path):
            (path / "tokenizer_config.json").write_text("{}")
    class Policy:
        def __init__(self, *args, **kwargs):
            pass
        def bind_task(self, *args):
            return self
        async def close(self):
            pass
    class Runner:
        count = 0
        async def run(self, task, policy, **kwargs):
            self.count += 1
            result = episode(1 if self.count == 1 else 0,
                status="completed" if self.count == 1 else "model_protocol_failure",
                thought=(6 if self.count % 2 else 7,), json_ids=(8, 9))
            # Production uses the NativeHFCompletion dataclass; its fields are
            # serializable by EpisodeResult.to_dict and preserve JSON IDs too.
            step = result.steps[0]
            result.steps = [native.NativeHFCompletion(
                text="", prompt_ids=step.prompt_ids, completion_ids=step.completion_ids,
                thought_token_count=1, phase_lengths=(1, 2),
                thought_logprobs=step.thought_logprobs,
                thought_logprobs_source=step.thought_logprobs_source)]
            return result
    for name, value in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1")):
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: torch.get_rng_state())
    monkeypatch.setattr(native, "NativeHFPolicy", Policy)
    monkeypatch.setattr(trainer, "load_policy_model", lambda *a, **k: (model, Tokenizer()))
    monkeypatch.setattr(trainer, "make_runner", lambda args: Runner())
    args = SimpleNamespace(model="CPU-fixture-only", adapter=None, source_commit="0" * 40,
        output_dir=tmp_path / "training", steps=2, seed=47, lora_rank=2,
        quantization="none", learning_rate=0.01, logprob_backend="original",
        head_chunk_tokens=None, thought_tokens=8, json_tokens=8,
        max_prompt_tokens=32, max_steps=2, group_size=2, clip=0.2,
        max_grad_norm=1.0, sync_artifacts=False, save_every=1, replay_diagnostic="off")
    task = SimpleNamespace(id="task")
    asyncio.run(trainer.train(args, [task], [[0], [0]], 1))
    complete = json.loads((args.output_dir / "training_complete.json").read_text())
    assert complete["complete"] and complete["has_policy_gradient_updates"]
    assert complete["rollout_iterations"] == 2 and complete["optimizer_updates"] == 1
    records = [json.loads(line) for line in (args.output_dir / "train_rank0.jsonl").read_text().splitlines()]
    for iteration in (1, 2):
        phases = [r["event"] for r in records if r.get("iteration") == iteration]
        ordered = ["collect_begin", "collect_end", "signal", "replay_begin", "replay_end", "apply_begin", "apply_end", "step"]
        assert [event for event in phases if event in ordered] == ordered
    saves = [r for r in records if r["event"] in {"save_begin", "save_end"}]
    assert [r["event"] for r in saves] == ["save_begin", "save_end"] * 4
    assert all(r.get("seconds", 0) >= 0 for r in records)
    steps = [r for r in records if r["event"] == "step"]
    assert all(r["replay_diagnostic_mode"] == "off" and r["diagnostic_scored_segments"] == 0
               and r["maximum_behavior_replay_logprob_difference"] is None for r in steps)
    assert [r["optimizer_step_applied"] for r in steps] == [True, False]
    assert [r["optimizer_updates"] for r in steps] == [1, 1]
    from safetensors.torch import load_file
    first = load_file(args.output_dir / "iteration_0001/adapter_model.safetensors")
    final = load_file(args.output_dir / "final/adapter_model.safetensors")
    assert all(torch.equal(first[k], final[k]) for k in first)
    state = torch.load(args.output_dir / "final/training_state.pt", weights_only=False)
    assert state["optimizer_steps"] == 1 and state["rollout_iterations"] == 2
    saved_config = json.loads((args.output_dir / "final/training_config.json").read_text())
    assert saved_config["replay_diagnostic_mode"] == saved_config["arguments"]["replay_diagnostic"] == "off"
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("invalid", [None, (), [-2.0], [-2.0, float("nan")],
    [-2.0, float("inf")], [-2.0, True], [-2.0, "-3"], [-2.0, .1]])
def test_group_refuses_missing_malformed_or_nonfinite_actual_behavior_scores(invalid):
    failed = episode(0, status="model_protocol_failure")
    failed.steps[0].thought_logprobs = invalid
    with pytest.raises(ValueError, match="actual sampled thought behavior"):
        signal([episode(1), failed])


def test_group_rejects_reconstructed_score_source_even_for_constant_rewards():
    one = episode(0)
    one.steps[0].thought_logprobs_source = "teacher-forced-old"
    with pytest.raises(ValueError, match="actual sampled thought behavior"):
        signal([one, episode(0)])


def test_replay_uses_actual_behavior_denominator_and_reports_replay_discrepancy():
    trainer = load_trainer()
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(-1.0)
    segment = SimpleNamespace(prompt_ids=[1], completion_ids=[2], thought_logprobs=[-1.4])
    group = SimpleNamespace(segments=[[segment]], advantages=[-1.0])
    def scorer(model, prompt, completion):
        return model.weight.reshape(1)
    report = trainer.replay_group(model, group, scorer,
        group_size=2, max_steps=1, thought_tokens=1, clip=.2)
    # Negative advantage selects the unclipped exp(.4), not a reconstructed
    # ratio of one. Its loss and derivative are both exp(.4)/2.
    expected = torch.exp(torch.tensor(.4)) / 2
    assert report["policy_loss"] == pytest.approx(expected.item())
    assert model.weight.grad.item() == pytest.approx(expected.item())
    assert report["old_logprob_source"] == "actual-sampled-thought-behavior"
    assert report["maximum_behavior_replay_logprob_difference"] == pytest.approx(.4)
    assert report["maximum_behavior_current_logprob_difference"] == pytest.approx(.4)
    assert report["maximum_replay_current_logprob_difference"] == 0
    assert report["importance_ratio_min"] == pytest.approx(2 * expected.item())
    assert report["importance_ratio_max"] == pytest.approx(2 * expected.item())
    assert report["importance_ratio_outside_clip_fraction"] == 1


@pytest.mark.parametrize("mode", ["full", "first", "off"])
def test_constant_group_retains_behavior_segments_without_inventing_replay_evidence(mode):
    trainer = load_trainer()
    group = signal([episode(0), episode(0, status="model_protocol_failure")])
    assert len(group.segments) == 2 and all(segment for segment in group.segments)
    model = torch.nn.Linear(1, 1)
    def forbidden(*args):
        raise AssertionError("An exact zero group must not replay")
    report = trainer.replay_group(model, group, forbidden,
        group_size=2, max_steps=2, thought_tokens=8, clip=.2, replay_diagnostic=mode)
    assert report["scored_tokens"] == 0
    assert report["maximum_behavior_replay_logprob_difference"] is None
    assert report["importance_ratio_min"] is None
    assert report["replay_diagnostic_mode"] == mode
    assert report["diagnostic_scored_segments"] == report["diagnostic_scored_tokens"] == 0


def test_replay_diagnostic_modes_preserve_clipped_categorical_loss_and_gradients():
    """Actual-old numerical drift clips both signs; only diagnostics get skipped."""
    trainer = load_trainer()
    logits = torch.tensor([-.3, .1, -.2, .4])
    current = logits.log_softmax(0)
    ratios = [1.4, .7, .7, 1.4]
    old = [float(current[i]) - math.log(ratio) for i, ratio in enumerate(ratios)]
    segments = [SimpleNamespace(prompt_ids=[10 + i], completion_ids=[i],
                                thought_logprobs=[old[i]]) for i in range(4)]
    group = SimpleNamespace(segments=[segments[:2], segments[2:]], advantages=[1., -1.])
    # Pos ratio1.4 and neg ratio.7 are clipped constants. The remaining two
    # derivatives are -.7*dlogp[1]/4 and +1.4*dlogp[3]/4.
    expected_gradient = -.7 * logits.softmax(0)
    expected_gradient[1] -= .7
    expected_gradient[3] += 1.4
    expected_gradient /= 4
    reports, gradients = {}, {}
    for mode, diagnostic_count in [("full", 4), ("first", 1), ("off", 0)]:
        model = torch.nn.Linear(1, 4, bias=False)
        with torch.no_grad():
            model.weight[:, 0].copy_(logits)
        calls = []
        def scorer(value, prompt, completion):
            index = completion[0]
            calls.append((value.training, torch.is_grad_enabled(), index))
            # Deliberately distinct diagnostic numerics. They must never be
            # substituted for the actual generation behavior denominator.
            delta = .04 * (index + 1) if not value.training else 0.
            return value.weight[:, 0].log_softmax(0)[[index]] + delta
        report = trainer.replay_group(model, group, scorer, group_size=2,
            max_steps=2, thought_tokens=1, clip=.2, replay_diagnostic=mode)
        assert len(calls) == 4 + diagnostic_count
        assert sum(not training and not grad for training, grad, _ in calls) == diagnostic_count
        assert [(grad, index) for training, grad, index in calls if training] == [(True, i) for i in range(4)]
        assert report["scored_segments"] == report["scored_tokens"] == 4
        assert report["diagnostic_scored_segments"] == report["diagnostic_scored_tokens"] == diagnostic_count
        assert report["replay_diagnostic_mode"] == mode
        assert report["policy_loss"] == pytest.approx(.075, abs=1e-7)
        assert report["importance_ratio_min"] == pytest.approx(.7)
        assert report["importance_ratio_max"] == pytest.approx(1.4)
        assert report["importance_ratio_outside_clip_fraction"] == 1.
        assert report["maximum_behavior_current_logprob_difference"] == pytest.approx(-math.log(.7))
        if mode == "off":
            assert report["maximum_behavior_replay_logprob_difference"] is None
            assert report["maximum_replay_current_logprob_difference"] is None
            assert report["replay_diagnostic_scope"] == "not-collected"
        else:
            assert report["maximum_replay_current_logprob_difference"] == pytest.approx(.16 if mode == "full" else .04)
            assert report["replay_diagnostic_scope"] == ("all-nonzero-advantage-segments" if mode == "full"
                else "first-nonzero-advantage-segment-per-group")
        gradients[mode] = model.weight.grad[:, 0]
        torch.testing.assert_close(gradients[mode], expected_gradient, rtol=1e-6, atol=1e-7)
        reports[mode] = report
    assert torch.equal(gradients["full"], gradients["first"]) and torch.equal(gradients["full"], gradients["off"])
    assert len({report["policy_loss"] for report in reports.values()}) == 1
    assert not torch.cuda.is_initialized()


def test_first_diagnostic_uses_first_nonzero_advantage_segment_only():
    trainer = load_trainer()
    ignored = SimpleNamespace(prompt_ids=[99], completion_ids=[99], thought_logprobs=[-1.])
    active = SimpleNamespace(prompt_ids=[1], completion_ids=[2], thought_logprobs=[-1.])
    group = SimpleNamespace(segments=[[ignored], [active]], advantages=[0., 1.])
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(-1.)
    calls = []
    def scorer(value, prompt, completion):
        assert prompt == [1] and completion == [2]
        calls.append((value.training, torch.is_grad_enabled()))
        return value.weight.reshape(1)
    report = trainer.replay_group(model, group, scorer, group_size=2,
        max_steps=1, thought_tokens=1, clip=.2, replay_diagnostic="first")
    assert calls == [(False, False), (True, True)]
    assert report["diagnostic_scored_segments"] == report["scored_segments"] == 1


@pytest.mark.parametrize("mode", ["full", "first", "off"])
@pytest.mark.parametrize("failure", ["behavior_nan", "behavior_shape", "current_nan", "ratio_inf"])
def test_every_diagnostic_mode_retains_actual_behavior_current_and_ratio_checks(mode, failure):
    trainer = load_trainer()
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(-1.)
    behavior = {"behavior_nan": [float("nan")], "behavior_shape": [], "current_nan": [-1.], "ratio_inf": [-1000.]}[failure]
    segment = SimpleNamespace(prompt_ids=[1], completion_ids=[2], thought_logprobs=behavior)
    group = SimpleNamespace(segments=[[segment]], advantages=[1.])
    def scorer(value, prompt, completion):
        result = value.weight.reshape(1)
        return result * float("nan") if failure == "current_nan" and value.training else result
    with pytest.raises(FloatingPointError):
        trainer.replay_group(model, group, scorer, group_size=2,
            max_steps=1, thought_tokens=1, clip=.2, replay_diagnostic=mode)


def test_invalid_replay_diagnostic_fails_before_any_forward():
    trainer = load_trainer()
    model = torch.nn.Linear(1, 1)
    def forbidden(*args):
        raise AssertionError("Invalid mode may not run a decoder")
    with pytest.raises(ValueError, match="full, first or off"):
        trainer.replay_group(model, signal([episode(1), episode(0)]), forbidden,
            group_size=2, max_steps=2, thought_tokens=8, clip=.2, replay_diagnostic="unknown")
