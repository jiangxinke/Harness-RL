"""Numerical oracles for fixed-budget episode GSPO; CPU only, no model downloads."""
from __future__ import annotations

import importlib.util
import json
import math
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from slime_plugins.dcc.episode_gspo import (VARIANT, episode_policy_loss,
    episode_policy_loss_from_sum)
from slime_plugins.dcc.native_joint_grpo import grammar_token_logprobs
from slime_plugins.dcc.online_grpo import apply_policy_update, thought_policy_loss


@pytest.fixture
def trainer():
    spec = importlib.util.spec_from_file_location("_dcc_gspo_trainer",
        Path(__file__).resolve().parents[1] / "scripts/dcc_online_grpo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


KW = dict(group_size=2, max_steps=2, objective_tokens=4, clip=.2)


@pytest.mark.parametrize("advantage", [1., -.3])
def test_ratio_one_matches_original_loss_and_categorical_gradient_across_turns(advantage):
    logits = torch.tensor([.2, -.4, .1], dtype=torch.float64, requires_grad=True)
    turns = [logits.log_softmax(0)[[0, 1, 2]], logits.log_softmax(0)[[2, 0]]]
    old = [turn.detach().clone() for turn in turns]
    original = sum(thought_policy_loss(turn, behavior, advantage,
        group_size=2, max_steps=2, thought_tokens=4, clip=.2)
        for turn, behavior in zip(turns, old, strict=True))
    actual = episode_policy_loss(torch.cat(turns), torch.cat(old), advantage, **KW)
    torch.testing.assert_close(actual, original, rtol=1e-14, atol=1e-14)
    expected_gradient = torch.autograd.grad(original, logits, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(actual, logits)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("advantage,ratio,active", [
    (1., 1.4, False), (1., .7, True), (-1., .7, False), (-1., 1.4, True)])
def test_clipping_both_signs_matches_scalar_value_and_token_derivative(advantage, ratio, active):
    current = torch.tensor([-1., -2., -3.], dtype=torch.float64, requires_grad=True)
    old = current.detach() - math.log(ratio)
    loss = episode_policy_loss(current, old, advantage, **KW)
    clipped = max(.8, min(1.2, ratio))
    assert loss.item() == pytest.approx(-3 / 16 * min(ratio * advantage, clipped * advantage))
    gradient = torch.autograd.grad(loss, current)[0]
    expected = -advantage * ratio / 16 if active else 0.
    torch.testing.assert_close(gradient, torch.full_like(current, expected), rtol=1e-13, atol=1e-14)


def test_episode_ratio_combines_turns_before_clipping_and_keeps_fixed_length_weight():
    current = torch.tensor([-1., -2.], dtype=torch.float64, requires_grad=True)
    old = current.detach() - torch.tensor([math.log(2), math.log(.5)], dtype=torch.float64)
    loss = episode_policy_loss(current, old, 1., **KW)
    gradient = torch.autograd.grad(loss, current)[0]
    torch.testing.assert_close(gradient, torch.full_like(current, -1 / 16))
    # A short episode does not get a larger per-token coefficient.
    short = torch.tensor([-1.], dtype=torch.float64, requires_grad=True)
    short_loss = episode_policy_loss(short, short.detach(), 1., **KW)
    assert torch.autograd.grad(short_loss, short)[0].item() == pytest.approx(-1 / 16)
    assert loss.item() == pytest.approx(2 * short_loss.item())


def test_actual_grammar_support_has_zero_illegal_and_singleton_gradients():
    logits = torch.tensor([[.2, -.1, .5, -.2], [1., 2., 3., 4.], [.1, .3, -.2, -.4]],
                          dtype=torch.float64, requires_grad=True)
    masks = torch.tensor([[0b0011], [0b0100], [0b1010]], dtype=torch.int32)
    scores = grammar_token_logprobs(logits, torch.tensor([1, 2, 3]), masks)
    assert scores[1].item() == 0
    loss = episode_policy_loss(scores, scores.detach(), .5, **KW)
    gradient = torch.autograd.grad(loss, logits)[0]
    assert torch.equal(gradient[0, 2:], torch.zeros(2, dtype=torch.float64))
    assert torch.equal(gradient[1], torch.zeros(4, dtype=torch.float64))
    assert torch.equal(gradient[2, [0, 2]], torch.zeros(2, dtype=torch.float64))
    assert gradient[0, 0] > 0 > gradient[0, 1]
    illegal = grammar_token_logprobs(logits, torch.tensor([2, 2, 3]), masks)
    with pytest.raises(FloatingPointError):
        episode_policy_loss(illegal, scores.detach(), .5, **KW)


@pytest.mark.parametrize("failure", ["empty", "shape", "old_grad", "current_nan", "old_inf"])
def test_tensor_loss_rejects_unusable_behavior_or_current(failure):
    current = torch.tensor([-1., -2.], requires_grad=True)
    old = current.detach().clone()
    if failure == "empty":
        current, old = current[:0], old[:0]
    elif failure == "shape":
        old = old[:1]
    elif failure == "old_grad":
        old.requires_grad_()
    elif failure == "current_nan":
        current = current * float("nan")
    elif failure == "old_inf":
        old[0] = -float("inf")
    with pytest.raises((ValueError, FloatingPointError)):
        episode_policy_loss(current, old, 1., **KW)


@pytest.mark.parametrize("count,total", [(0, 0.), (9, 0.), (1, float("nan")), (1, 1000.)])
def test_scalar_loss_rejects_zero_excess_or_nonfinite_episode(count, total):
    with pytest.raises((ValueError, FloatingPointError)):
        episode_policy_loss_from_sum(torch.tensor(total, dtype=torch.float64), count, 1., **KW)


@pytest.mark.parametrize("mode,diagnostic_count", [("full", 3), ("first", 1), ("off", 0)])
def test_single_replay_buffer_matches_full_graph_oracle_and_preserves_saved_behavior(
        trainer, mode, diagnostic_count):
    logits = torch.tensor([-.3, .1, -.2, .4], dtype=torch.float64)
    model = torch.nn.Linear(1, 4, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight[:, 0].copy_(logits)
    logp = logits.log_softmax(0)
    def segment(ids, ratios):
        return SimpleNamespace(prompt_ids=[10], completion_ids=ids,
            thought_logprobs=[logp[i].item() - math.log(r) for i, r in zip(ids, ratios, strict=True)])
    segments = [[segment([0], [1.])], [segment([0, 1], [1.4, .7]), segment([2], [1.1])],
                [segment([3], [.7])]]
    group = SimpleNamespace(segments=segments, advantages=[0., .5, -.5])
    before_old = json.dumps([[s.thought_logprobs for s in ep] for ep in segments])
    oracle_logits = logits.clone().requires_grad_()
    oracle = sum(episode_policy_loss(
        oracle_logits.log_softmax(0)[[i for s in ep for i in s.completion_ids]],
        torch.tensor([p for s in ep for p in s.thought_logprobs], dtype=torch.float64), advantage,
        group_size=3, max_steps=2, objective_tokens=4, clip=.2)
        for ep, advantage in zip(segments, group.advantages, strict=True) if advantage)
    expected = torch.autograd.grad(oracle, oracle_logits)[0]
    # The helper adds to, rather than destroys, caller-owned accumulated grads.
    model.weight.grad = torch.full_like(model.weight, .25)
    calls, backward_seen = [], []
    def scorer(value, prompt, completion):
        assert prompt == [10]
        calls.append((value.training, torch.is_grad_enabled(), completion))
        result = value.weight[:, 0].log_softmax(0)[completion]
        if value.training:
            # The previous segment's graph must already have been consumed.
            assert len(backward_seen) == sum(training for training, _, _ in calls) - 1
            result.register_hook(lambda gradient: backward_seen.append(True))
        else:
            result = result + .04  # Diagnostic numerics are deliberately different.
        return result
    report = trainer.replay_group(model, group, scorer, group_size=3, max_steps=2,
        thought_tokens=4, clip=.2, policy_surrogate="episode-gspo", replay_diagnostic=mode)
    torch.testing.assert_close(model.weight.grad[:, 0], expected + .25, rtol=1e-12, atol=1e-13)
    assert report["policy_loss"] == pytest.approx(oracle.item())
    assert len(calls) == 3 + diagnostic_count and len(backward_seen) == 3
    assert report["scored_tokens"] == 4 and report["scored_segments"] == 3
    assert report["scored_episodes"] == 2 and report["episode_scored_token_counts"] == [3, 1]
    assert report["episode_gradient_clipped_count"] == 1
    assert report["importance_ratio_min"] == pytest.approx(.7)
    assert report["importance_ratio_max"] == pytest.approx((1.4 * .7 * 1.1) ** (1 / 3))
    assert report["importance_ratio_outside_clip_fraction"] == .5
    assert report["token_importance_ratio_outside_clip_fraction"] == .75
    assert report["diagnostic_scored_segments"] == diagnostic_count
    assert report["maximum_behavior_current_logprob_difference"] == pytest.approx(-math.log(.7))
    assert report["objective_denominator"] == 24
    assert [row["episode_index"] for row in report["episode_summaries"]] == [1, 2]
    assert [row["scored_tokens"] for row in report["episode_summaries"]] == [3, 1]
    assert [row["gradient_clipped"] for row in report["episode_summaries"]] == [False, True]
    assert sum(row["policy_loss"] for row in report["episode_summaries"]) == report["policy_loss"]
    for row in report["episode_summaries"]:
        assert row["advantage"] == group.advantages[row["episode_index"]]
        assert row["episode_ratio"] == pytest.approx(math.exp(row["logratio_sum"] / row["scored_tokens"]))
        r, a = row["episode_ratio"], row["advantage"]
        rebuilt = -row["scored_tokens"] / 24 * min(r * a, max(.8, min(1.2, r)) * a)
        assert row["policy_loss"] == pytest.approx(rebuilt)
    assert json.dumps([[s.thought_logprobs for s in ep] for ep in segments]) == before_old
    assert not torch.cuda.is_initialized()


def test_joint_replay_uses_same_segment_support_and_actual_json_old_scores(trainer, monkeypatch):
    from slime_plugins.dcc import native_joint_grpo
    model = torch.nn.Linear(1, 4, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight[:, 0].copy_(torch.tensor([.2, -.1, .5, -.2]))
    masks = torch.tensor([[0b0011], [0b0100]], dtype=torch.int32)
    def scores(value):
        logits = value.weight[:, 0]
        raw = logits.log_softmax(0)[[0]]
        masked = grammar_token_logprobs(logits.expand(2, -1), torch.tensor([1, 2]), masks)
        return torch.cat((raw, masked))
    with torch.no_grad():
        actual_old = scores(model).tolist()
    segment = native_joint_grpo.JointSegment([1], [0, 1, 2], actual_old[:1], actual_old[1:],
        1, {"actual_public_schema": True}, [2])
    compiler = object()
    def scorer_for(s, c, **kwargs):
        assert s is segment and c is compiler
        return lambda value, prompt, completion: scores(value)
    monkeypatch.setattr(native_joint_grpo, "segment_scorer", scorer_for)
    oracle = episode_policy_loss(scores(model), torch.tensor(actual_old, dtype=torch.float64),
        1., group_size=2, max_steps=1, objective_tokens=4)
    expected = torch.autograd.grad(oracle, model.weight)[0]
    group = SimpleNamespace(segments=[[segment], []], advantages=[1., 0.])
    report = trainer.replay_group(model, group, None, group_size=2, max_steps=1,
        thought_tokens=1, json_tokens=3, joint_json_policy=True, grammar_compiler=compiler,
        policy_surrogate="episode-gspo", clip=.2, replay_diagnostic="off")
    torch.testing.assert_close(model.weight.grad, expected, rtol=1e-7, atol=1e-9)
    assert report["thought_scored_tokens"] == 1 and report["json_scored_tokens"] == 2
    assert report["importance_ratio_min"] == pytest.approx(1.)
    assert report["old_logprob_source"] == "actual-sampled-raw-thought-and-post-xgrammar-JSON-behavior"
    assert segment.json_logprobs == actual_old[1:]


def test_constant_group_does_not_replay_or_step_adam_with_momentum(trainer):
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
    model.weight.sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    before = model.weight.detach().clone()
    def forbidden(*args):
        raise AssertionError("No replay for a constant group")
    report = trainer.replay_group(model, SimpleNamespace(segments=[[], []], advantages=[0., 0.]),
        forbidden, group_size=2, max_steps=2, thought_tokens=4, clip=.2,
        policy_surrogate="episode-gspo")
    norm, applied = apply_policy_update(list(model.parameters()), optimizer)
    assert report["scored_tokens"] == 0 and report["importance_ratio_min"] is None
    assert report["episode_summaries"] == []
    assert norm == 0 and not applied
    assert torch.equal(before, model.weight.detach())
    assert optimizer.state[model.weight]["step"] == 1


def test_all_clipped_episodes_also_skip_adam_momentum(trainer):
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
    model.weight.sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    before = model.weight.detach().clone()
    segments = [[SimpleNamespace(prompt_ids=[1], completion_ids=[2],
        thought_logprobs=[before.item() - math.log(ratio)])] for ratio in (1.4, .7)]
    group = SimpleNamespace(segments=segments, advantages=[1., -1.])
    report = trainer.replay_group(model, group, lambda value, *_: value.weight.reshape(1),
        group_size=2, max_steps=1, thought_tokens=1, clip=.2,
        policy_surrogate="episode-gspo", replay_diagnostic="off")
    norm, applied = apply_policy_update(list(model.parameters()), optimizer)
    assert report["episode_gradient_clipped_count"] == 2
    assert norm == 0 and not applied and torch.equal(before, model.weight.detach())
    assert optimizer.state[model.weight]["step"] == 1


@pytest.mark.parametrize("advantage", [1., -1.])
@pytest.mark.parametrize("target", [.8, 1.2, math.nextafter(.8, 0.), math.nextafter(1.2, 2.)])
def test_reported_clip_boundary_and_gradient_use_same_tensor_ratio(trainer, advantage, target):
    model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.fill_(-1.)
    old = [-1. - math.log(target)]
    segment = SimpleNamespace(prompt_ids=[1], completion_ids=[2], thought_logprobs=old)
    oracle = episode_policy_loss(model.weight.reshape(1), torch.tensor(old, dtype=torch.float64),
        advantage, group_size=2, max_steps=1, objective_tokens=1, clip=.2)
    expected_gradient = torch.autograd.grad(oracle, model.weight)[0]
    expected_ratio = (model.weight.detach().reshape(1) - torch.tensor(old, dtype=torch.float64)).mean().exp().item()
    report = trainer.replay_group(model, SimpleNamespace(segments=[[segment], []], advantages=[advantage, 0.]),
        lambda value, *_: value.weight.reshape(1), group_size=2, max_steps=1,
        thought_tokens=1, clip=.2, policy_surrogate="episode-gspo", replay_diagnostic="off")
    summary = report["episode_summaries"][0]
    assert summary["episode_ratio"] == expected_ratio
    assert summary["gradient_clipped"] == ((advantage > 0 and expected_ratio > 1.2)
                                           or (advantage < 0 and expected_ratio < .8))
    torch.testing.assert_close(model.weight.grad, expected_gradient, rtol=1e-14, atol=1e-14)


def test_actual_group_signal_retains_failed_episodes_and_their_negative_gradient(trainer):
    from slime_plugins.dcc.episode import EpisodeResult
    from slime_plugins.dcc.native_behavior_logprobs import BEHAVIOR_LOGPROB_SOURCE
    from slime_plugins.dcc.online_grpo import group_signal
    model = torch.nn.Linear(1, 3, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight[:, 0].copy_(torch.tensor([.1, -.3, .4]))
    results = []
    for token, reward, status in [(0, 1., "completed"), (1, 0., "model_protocol_failure")]:
        step = SimpleNamespace(prompt_ids=[2], completion_ids=[token], thought_token_count=1,
            thought_logprobs=[model.weight[:, 0].log_softmax(0)[token].detach().item()],
            thought_logprobs_source=BEHAVIOR_LOGPROB_SOURCE)
        results.append(EpisodeResult("TRAIN", "cuad", "train", [], status, reward, {}, [step], [], []))
    group = group_signal(results, group_size=2, max_steps=1, thought_tokens=4,
                         advantage_normalization="centered")
    assert group.advantages == [.5, -.5] and group.rewards == [1., 0.]
    assert results[1].trainable_steps == [] and group.segments[1]
    logits = model.weight[:, 0]
    oracle = -.5 * (logits.log_softmax(0)[0] - logits.log_softmax(0)[1]) / 8
    expected = torch.autograd.grad(oracle, model.weight)[0]
    report = trainer.replay_group(model, group,
        lambda value, prompt, completion: value.weight[:, 0].log_softmax(0)[completion],
        group_size=2, max_steps=1, thought_tokens=4, clip=.2,
        policy_surrogate="episode-gspo", replay_diagnostic="off")
    torch.testing.assert_close(model.weight.grad, expected, rtol=1e-12, atol=1e-13)
    assert report["scored_episodes"] == 2 and report["scored_tokens"] == 2


def _gspo_gloo_worker(rank, rendezvous, output):
    import torch.distributed as distributed
    from slime_plugins.dcc.episode_gspo import replay_episode_gspo_group
    from slime_plugins.dcc.online_grpo import wait_for_update_ranks
    distributed.init_process_group("gloo", init_method="file://" + rendezvous,
        rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(-1.)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.1, weight_decay=0)
        def segment(count):
            return SimpleNamespace(prompt_ids=[1], completion_ids=[2] * count,
                                   thought_logprobs=[-1.] * count)
        group = SimpleNamespace(segments=[[segment(2)], [segment(1)]],
            advantages=[.5, -.5] if rank else [0., 0.])
        replay_episode_gspo_group(model, group,
            scorer_for=lambda segment: lambda value, prompt, completion:
                value.weight.reshape(1).expand(len(completion)),
            old_values=lambda segment: segment.thought_logprobs,
            group_size=2, max_steps=1, objective_tokens=2, clip=.2, replay_diagnostic="off")
        wait_for_update_ranks(world_size=2, distributed=distributed,
                              control_group=distributed.group.WORLD, synchronize=lambda: None)
        norm, applied = apply_policy_update(list(model.parameters()), optimizer,
            world_size=2, distributed=distributed)
        before = model.weight.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        zero = SimpleNamespace(segments=[[], []], advantages=[0., 0.])
        replay_episode_gspo_group(model, zero, scorer_for=lambda _: None, old_values=lambda _: None,
            group_size=2, max_steps=1, objective_tokens=2, clip=.2, replay_diagnostic="off")
        zero_norm, zero_applied = apply_policy_update(list(model.parameters()), optimizer,
            world_size=2, distributed=distributed)
        Path(output, f"gspo_rank{rank}.json").write_text(json.dumps(dict(norm=norm, applied=applied,
            weight=model.weight.item(), zero_norm=zero_norm, zero_applied=zero_applied,
            zero_unchanged=torch.equal(before, model.weight.detach()),
            adam_steps=optimizer.state[model.weight]["step"].item(),
            cuda_initialized=torch.cuda.is_initialized())))
    finally:
        distributed.destroy_process_group()


def test_real_gloo_gspo_buffer_averages_active_and_constant_ranks_and_skips_zero_adam(tmp_path):
    torch.multiprocessing.spawn(_gspo_gloo_worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True)
    rows = [json.loads((tmp_path / f"gspo_rank{rank}.json").read_text()) for rank in range(2)]
    assert rows[0] == rows[1]
    assert rows[0]["norm"] == pytest.approx(.0625) and rows[0]["applied"]
    assert rows[0]["zero_norm"] == 0 and rows[0]["zero_applied"] is False
    assert rows[0]["zero_unchanged"] and rows[0]["adam_steps"] == 1
    assert not rows[0]["cuda_initialized"]


@pytest.mark.parametrize("failure", ["empty", "old_nan", "current_nan", "ratio_inf"])
def test_replay_rejects_bad_segment_without_leaving_partial_gradients(trainer, failure):
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(-1.)
    good = SimpleNamespace(prompt_ids=[1], completion_ids=[2], thought_logprobs=[-1.])
    bad = SimpleNamespace(prompt_ids=[3], completion_ids=[4], thought_logprobs=[-1.])
    if failure == "empty":
        bad.completion_ids, bad.thought_logprobs = [], []
    elif failure == "old_nan":
        bad.thought_logprobs = [float("nan")]
    elif failure == "ratio_inf":
        bad.thought_logprobs = [-1001.]
    def scorer(value, prompt, completion):
        result = value.weight.reshape(1)[:len(completion)]
        return result * float("nan") if failure == "current_nan" and prompt == [3] else result
    with pytest.raises((ValueError, FloatingPointError)):
        trainer.replay_group(model, SimpleNamespace(segments=[[good, bad], []], advantages=[1., 0.]),
            scorer, group_size=2, max_steps=2, thought_tokens=1, clip=.2,
            policy_surrogate="episode-gspo", replay_diagnostic="off")
    assert model.weight.grad is None and model.weight.item() == -1.


@pytest.mark.parametrize("joint", [False, True])
def test_configuration_keeps_default_identical_and_gspo_has_explicit_identity(trainer, joint):
    args = SimpleNamespace(model="CPU-fixture", source_commit="a" * 40,
        group_size=4, max_steps=2, thought_tokens=4, json_tokens=8, joint_json_policy=joint)
    original = trainer.configuration(args, [], [], 1)
    args.policy_surrogate = "token-grpo"
    assert trainer.configuration(args, [], [], 1) == original
    args.policy_surrogate = "episode-gspo"
    config = trainer.configuration(args, [], [], 1)
    assert config["algorithm"] == original["algorithm"] + "+" + VARIANT
    assert config["algorithm_variant"] == VARIANT
    assert config["objective_denominator"] == original["objective_denominator"]
    assert config["behavior_policy"] == original["behavior_policy"]
    assert config["policy_surrogate"]["unbiased_full_trajectory_importance_sampling"] is False


@pytest.mark.parametrize("extra", [["--policy-surrogate", "unknown"],
    ["--policy-surrogate", "episode-gspo", "--reference-kl-beta", ".01"],
    ["--policy-surrogate", "episode-gspo", "--enable-online-web"]])
def test_invalid_cli_combinations_rejected_before_inputs_or_cuda(trainer, monkeypatch, extra):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid options must not read inputs, load a model or initialize CUDA")
    monkeypatch.setattr(trainer, "load_tasks", forbidden)
    monkeypatch.setattr(torch.cuda, "set_device", forbidden)
    with pytest.raises(SystemExit) as error:
        trainer.main(["--model", "CPU-fixture", "--train-data", "unused.jsonl",
            "--output-dir", "unused-output", "--source-commit", "0" * 40, *extra])
    assert error.value.code == 2 and not torch.cuda.is_initialized()


def test_gspo_web_rejected_by_configuration_and_train_before_cuda(trainer, monkeypatch):
    import asyncio
    args = SimpleNamespace(policy_surrogate="episode-gspo", enable_online_web=True)
    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported Web GSPO must not initialize CUDA")
    monkeypatch.setattr(torch.cuda, "set_device", forbidden)
    with pytest.raises(ValueError, match="CUAD only"):
        trainer.configuration(args, [], [], 1)
    with pytest.raises(ValueError, match="CUAD only"):
        asyncio.run(trainer.train(args, [], [], 1))
    args.enable_online_web = False
    with pytest.raises(ValueError, match="CUAD only"):
        asyncio.run(trainer.train(args, [], [], 1, web_inputs=object()))


def test_gspo_cli_records_explicit_mode_and_original_budgets_without_cuda(trainer, monkeypatch):
    from slime_plugins.dcc.data import Task
    task = Task("TRAIN-fixture", "cuad", "train", "Find clause", document="clause", gold=["clause"])
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(trainer, "load_tasks", lambda _: [task])
    monkeypatch.setattr(trainer, "validate_source", lambda _: None)
    def forbidden(*args, **kwargs):
        raise AssertionError("CLI metadata check must not load weights or initialize CUDA")
    monkeypatch.setattr(torch.cuda, "set_device", forbidden)
    monkeypatch.setattr(trainer, "load_policy_model", forbidden)
    observed = []
    async def capture(args, tasks, plan, world_size, *, web_inputs):
        config = trainer.configuration(args, plan, tasks, world_size)
        observed.append(config)
        assert config["arguments"]["policy_surrogate"] == "episode-gspo"
        assert config["objective_denominator"] == 4 * 2 * 8
        assert config["advantage_normalization"] == "std"
    monkeypatch.setattr(trainer, "train", capture)
    assert trainer.main(["--model", "CPU-fixture", "--train-data", "unused.jsonl",
        "--output-dir", "unused-output", "--source-commit", "0" * 40,
        "--group-size", "4", "--thought-tokens", "8", "--json-tokens", "8",
        "--policy-surrogate", "episode-gspo"]) == 0
    assert len(observed) == 1 and not torch.cuda.is_initialized()


def test_real_tiny_qwen3_moe_lora_buffer_matches_full_episode_graph(trainer):
    """Random CPU weights and synthetic recorded old distribution; no benchmark claim."""
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    from slime_plugins.dcc.hf_policy import completion_logprobs
    torch.manual_seed(19)
    config = Qwen3MoeConfig(vocab_size=24, hidden_size=16, intermediate_size=24,
        moe_intermediate_size=8, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, num_experts=4, num_experts_per_tok=2,
        attention_dropout=0.)
    model = get_peft_model(Qwen3MoeForCausalLM(config), LoraConfig(r=2, lora_alpha=4,
        lora_dropout=0., target_modules=["q_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    parameters = [p for p in model.parameters() if p.requires_grad]
    segments = []
    for episode_ids in [([5, 6], [7]), ([8],)]:
        turns = []
        for ids in episode_ids:
            prompt = [2, 3, 4]
            with torch.no_grad():
                old = completion_logprobs(model, prompt, list(ids)).tolist()
            turns.append(SimpleNamespace(prompt_ids=prompt, completion_ids=list(ids), thought_logprobs=old))
        segments.append(turns)
    group = SimpleNamespace(segments=segments, advantages=[.5, -.5])
    model.train()
    oracle = sum(episode_policy_loss(torch.cat([completion_logprobs(model, s.prompt_ids, s.completion_ids)
                                               for s in turns]),
        torch.tensor([v for s in turns for v in s.thought_logprobs]), advantage,
        group_size=2, max_steps=2, objective_tokens=4)
        for turns, advantage in zip(segments, group.advantages, strict=True))
    oracle.backward()
    expected = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
    model.zero_grad(set_to_none=True)
    report = trainer.replay_group(model, group, completion_logprobs,
        group_size=2, max_steps=2, thought_tokens=4, clip=.2,
        policy_surrogate="episode-gspo", replay_diagnostic="off")
    assert report["policy_loss"] == pytest.approx(oracle.item(), abs=1e-8)
    for parameter, gradient in zip(parameters, expected, strict=True):
        assert (parameter.grad is None) == (gradient is None)
        if gradient is not None:
            torch.testing.assert_close(parameter.grad, gradient, rtol=2e-5, atol=1e-8)
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=0)
    norm, applied = apply_policy_update(parameters, optimizer)
    assert norm > 0 and applied and not torch.cuda.is_initialized()


@pytest.mark.parametrize("joint", [False, True])
def test_matched_evaluator_recognizes_gspo_update_receipts(trainer, tmp_path, joint):
    spec = importlib.util.spec_from_file_location("_dcc_gspo_evaluator",
        Path(__file__).resolve().parents[1] / "scripts/dcc_online_evaluate.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    args = SimpleNamespace(model="CPU-fixture", source_commit="a" * 40,
        group_size=2, max_steps=2, thought_tokens=4, json_tokens=8,
        joint_json_policy=joint, policy_surrogate="episode-gspo")
    config = trainer.configuration(args, [], [], 1)
    config.update(rollout_iterations=2, optimizer_updates=1)
    (tmp_path / "training_config.json").write_text(json.dumps(config))
    (tmp_path / "checkpoint_complete.json").write_text(json.dumps(dict(complete=True, optimizer_steps=1)))
    assert evaluator.adapter_condition(str(tmp_path))["online_policy_gradient_updates_recorded"] is True
