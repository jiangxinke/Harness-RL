"""CPU analytic oracles for an optional phase coefficient; no quality claims."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import random
from types import SimpleNamespace

import pytest
import torch

from slime_plugins.dcc import native_joint_grpo as joint
from slime_plugins.dcc.online_grpo import apply_policy_update, thought_policy_loss


@pytest.fixture
def trainer():
    spec = importlib.util.spec_from_file_location("_dcc_weight_trainer",
        Path(__file__).resolve().parents[1] / "scripts/dcc_online_grpo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


KW = dict(group_size=2, max_steps=3, thought_tokens=4, json_tokens=5, clip=.2)


def shared_parameter_case(monkeypatch):
    """One shared categorical parameter vector, two raw and two masked positions.

    The final singleton represents a real constrained stop. Its gradient is
    zero even at weight four; the preceding JSON position has two legal tokens.
    """
    model = torch.nn.Module()
    model.theta = torch.nn.Parameter(torch.tensor([.2, -.3, .1]))
    features = torch.tensor([[1., .5, 1.5], [.5, 1., -.3], [1.2, -.2, .7], [.3, 1., .8]])
    bias = torch.tensor([[.1, -.1, .2], [0., .2, -.1], [-.2, .1, .3], [.2, -.3, .1]])
    labels = [0, 1, 2, 1]
    supports = [[0, 1, 2], [0, 1, 2], [0, 2], [1]]
    masks = torch.tensor([[5], [2]], dtype=torch.int32)
    compiler = object()
    def values(model):
        logits = features * model.theta + bias
        raw = logits[:2].log_softmax(-1)[torch.arange(2), torch.tensor(labels[:2])]
        constrained = joint.grammar_token_logprobs(logits[2:], torch.tensor(labels[2:]), masks)
        return torch.cat((raw, constrained))
    def factory(segment, actual_compiler, **kwargs):
        assert actual_compiler is compiler
        assert segment.thought_count == 2 and segment.completion_ids == labels
        assert segment.json_public_schema == {"synthetic-support": "fixed"}
        def score(model, prompt_ids, completion_ids):
            assert prompt_ids == [9] and completion_ids == labels
            return values(model)
        return score
    monkeypatch.setattr(joint, "segment_scorer", factory)
    current = values(model).detach()
    ratios = torch.tensor([.7, 1.4, 1.1, 1.])
    old = current - ratios.log()
    segment = joint.JointSegment([9], labels, old[:2].tolist(), old[2:].tolist(), 2,
        {"synthetic-support": "fixed"}, [1])
    return model, segment, compiler, values, features, bias, labels, supports


def analytic_oracle(model, segment, values, features, bias, labels, supports, advantage, weight):
    """Categorical derivatives in Python arithmetic, independent of autograd."""
    logits = (features * model.theta.detach() + bias).tolist()
    current = values(model).detach().tolist()
    denominator = KW['group_size'] * KW['max_steps'] * (KW['thought_tokens'] + KW['json_tokens'])
    gradient = [0.] * len(model.theta)
    loss = 0.
    for index, (label, allowed) in enumerate(zip(labels, supports, strict=True)):
        ratio = math.exp(current[index] - segment.behavior_logprobs[index])
        coefficient = 1 if index < segment.thought_count else weight
        loss -= coefficient * min(ratio * advantage, min(1.2, max(.8, ratio)) * advantage) / denominator
        active = ratio <= 1.2 if advantage > 0 else ratio >= .8
        if not active:
            continue
        normalizer = math.fsum(math.exp(logits[index][token]) for token in allowed)
        for token in allowed:
            probability = math.exp(logits[index][token]) / normalizer
            derivative = float(features[index, token]) * ((token == label) - probability)
            gradient[token] -= coefficient * advantage * ratio * derivative / denominator
    return loss, torch.tensor(gradient)


@pytest.mark.parametrize('advantage', [1., -.4])
def test_weight_four_actual_replay_matches_shared_parameter_categorical_oracle(trainer, monkeypatch, advantage):
    case = shared_parameter_case(monkeypatch)
    model, segment, compiler, values, features, bias, labels, supports = case
    old = deepcopy(segment.behavior_logprobs)
    expected_loss, expected_gradient = analytic_oracle(model, segment, values, features, bias,
        labels, supports, advantage, 4)
    signal = SimpleNamespace(segments=[[segment], []], advantages=[advantage, 0.])
    report = trainer.replay_group(model, signal, None, **KW, joint_json_policy=True,
        grammar_compiler=compiler, replay_diagnostic='off', json_surrogate_weight=4)
    assert report['policy_loss'] == pytest.approx(expected_loss, abs=2e-8)
    torch.testing.assert_close(model.theta.grad, expected_gradient, rtol=2e-6, atol=2e-8)
    assert report['thought_scored_tokens'] == report['json_scored_tokens'] == 2
    assert report['json_surrogate']['json_weight'] == 4
    assert report['json_surrogate']['denominator'] == 'group_size*max_steps*(thought_tokens+json_tokens)'
    assert segment.behavior_logprobs == old and segment.completion_ids == labels
    # The masked singleton stop has no parameter gradient before any weighting.
    stop = values(model)[-1]
    assert stop.item() == 0
    torch.testing.assert_close(torch.autograd.grad(stop, model.theta)[0], torch.zeros_like(model.theta))


@pytest.mark.parametrize('advantage', [1., -.5])
@pytest.mark.parametrize('thought_count', [0, 1, 2, 4])
def test_clip_boundaries_and_phase_split_keep_fixed_budget(advantage, thought_count):
    ratios = torch.tensor([.8, 1.2, .799, 1.201], dtype=torch.float64)
    current = ratios.log().requires_grad_()
    old = torch.zeros_like(current)
    loss = joint.weighted_joint_policy_loss(current, old, advantage,
        thought_count=thought_count, json_surrogate_weight=4, **KW)
    gradient = torch.autograd.grad(loss, current)[0]
    # At an exact min/clamp tie, preserve the installed Torch/old-loss
    # subgradient rather than prescribing a different boundary convention.
    legacy = thought_policy_loss(current, old, advantage,
        group_size=2, max_steps=3, thought_tokens=9, clip=.2)
    legacy_gradient = torch.autograd.grad(legacy, current)[0].tolist()
    denominator = 2 * 3 * (4 + 5)
    expected_loss, expected_gradient = 0., []
    for index, ratio in enumerate(current.detach().exp().tolist()):
        weight = 1 if index < thought_count else 4
        expected_loss -= weight * min(ratio * advantage, min(1.2, max(.8, ratio)) * advantage) / denominator
        active = ratio <= 1.2 if advantage > 0 else ratio >= .8
        if min(abs(ratio-.8), abs(ratio-1.2)) < 1e-14:
            expected_gradient.append(weight * legacy_gradient[index])
        else:
            expected_gradient.append(-weight * advantage * ratio / denominator if active else 0.)
    assert loss.item() == pytest.approx(expected_loss, abs=1e-14)
    torch.testing.assert_close(gradient, torch.tensor(expected_gradient, dtype=torch.float64), rtol=1e-13, atol=1e-14)


def test_default_replay_uses_old_loss_with_exact_optimizer_and_rng_parity(trainer, monkeypatch):
    model, segment, compiler, values, *_ = shared_parameter_case(monkeypatch)
    signal = SimpleNamespace(segments=[[segment], []], advantages=[.7, 0.])
    initial = model.theta.detach().clone()
    def forbidden(*args, **kwargs):
        raise AssertionError('Default must never enter the weighted loss helper')
    monkeypatch.setattr(trainer, 'weighted_joint_policy_loss', forbidden)
    results = []
    rng_before = torch.get_rng_state().clone(), random.getstate()
    for option in ({}, {'json_surrogate_weight': 1}):
        with torch.no_grad():
            model.theta.copy_(initial)
        model.zero_grad(set_to_none=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=0)
        report = trainer.replay_group(model, signal, None, **KW, joint_json_policy=True,
            grammar_compiler=compiler, replay_diagnostic='off', **option)
        gradient = model.theta.grad.clone()
        apply_policy_update(list(model.parameters()), optimizer, max_grad_norm=10)
        results.append((report, gradient, model.theta.detach().clone(), deepcopy(optimizer.state_dict())))
    with torch.no_grad():
        model.theta.copy_(initial)
    model.zero_grad(set_to_none=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=0)
    old_loss = thought_policy_loss(values(model), torch.tensor(segment.behavior_logprobs), .7,
        group_size=2, max_steps=3, thought_tokens=9, clip=.2)
    old_loss.backward()
    legacy_gradient = model.theta.grad.clone()
    apply_policy_update(list(model.parameters()), optimizer, max_grad_norm=10)
    for report, gradient, parameters, state in results:
        assert 'json_surrogate' not in report and 'algorithm_variant' not in report
        assert report['policy_loss'] == old_loss.item()
        assert torch.equal(gradient, legacy_gradient) and torch.equal(parameters, model.theta)
        for key, value in optimizer.state_dict()['state'][0].items():
            assert torch.equal(state['state'][0][key], value)
    assert results[0][0] == results[1][0]
    assert torch.equal(torch.get_rng_state(), rng_before[0]) and random.getstate() == rng_before[1]


@pytest.mark.parametrize('joint_enabled', [False, True])
def test_default_configuration_identity_omits_option_and_nondefault_is_explicit(trainer, joint_enabled):
    args = SimpleNamespace(model='CPU-fixture', source_commit='a' * 40,
        group_size=2, max_steps=3, thought_tokens=4, json_tokens=5, joint_json_policy=joint_enabled)
    original = trainer.configuration(args, [], [], 1)
    args.json_surrogate_weight = 1.
    assert trainer.configuration(args, [], [], 1) == original
    assert 'json_surrogate_weight' not in original['arguments'] and 'json_surrogate' not in original
    if joint_enabled:
        args.json_surrogate_weight = 4.
        weighted = trainer.configuration(args, [], [], 1)
        assert weighted['objective_denominator'] == original['objective_denominator'] == 54
        assert weighted['behavior_policy'] == original['behavior_policy']
        assert weighted['reference_kl'] == original['reference_kl']
        assert weighted['algorithm'] == original['algorithm'] + '+' + joint.JSON_SURROGATE_VARIANT
        assert weighted['arguments']['json_surrogate_weight'] == 4.


def test_reference_k3_remains_unweighted_in_actual_replay(trainer, monkeypatch):
    model, segment, compiler, values, *_ = shared_parameter_case(monkeypatch)
    signal = SimpleNamespace(segments=[[segment], []], advantages=[.7, 0.])
    reference = values(model).detach() + torch.tensor([-.1, .1, -.2, 0.])
    monkeypatch.setattr(trainer, 'validate_base_reference_model', lambda _: None)
    monkeypatch.setattr(trainer, 'frozen_base_logprobs', lambda *args: reference)
    reports, gradients = [], []
    for weight, beta in ((1, 0), (4, 0), (1, .2), (4, .2)):
        model.zero_grad(set_to_none=True)
        reports.append(trainer.replay_group(model, signal, None, **KW,
            joint_json_policy=True, grammar_compiler=compiler, replay_diagnostic='off',
            json_surrogate_weight=weight, reference_kl_beta=beta))
        gradients.append(model.theta.grad.clone())
    assert reports[2]['reference_kl_unweighted_loss'] == reports[3]['reference_kl_unweighted_loss']
    assert reports[2]['reference_kl_loss'] == reports[3]['reference_kl_loss']
    torch.testing.assert_close(gradients[3] - gradients[1], gradients[2] - gradients[0], rtol=2e-5, atol=2e-8)


@pytest.mark.parametrize('extra', [
    ['--json-surrogate-weight', '0'], ['--json-surrogate-weight', '-1'],
    ['--json-surrogate-weight', 'nan'], ['--json-surrogate-weight', 'inf'],
    ['--json-surrogate-weight', '4'],
    ['--json-surrogate-weight', '4', '--joint-json-policy', '--policy-surrogate', 'episode-gspo'],
    ['--json-surrogate-weight', '4', '--joint-json-policy', '--enable-online-web']])
def test_invalid_cli_combinations_rejected_before_inputs_or_cuda(trainer, monkeypatch, extra):
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid option must fail before reading inputs or initializing CUDA')
    monkeypatch.setattr(trainer, 'load_tasks', forbidden)
    monkeypatch.setattr(torch.cuda, 'set_device', forbidden)
    with pytest.raises(SystemExit) as error:
        trainer.main(['--model','CPU-fixture','--train-data','unused.jsonl',
            '--output-dir','unused-output','--source-commit','0' * 40,*extra])
    assert error.value.code == 2


@pytest.mark.parametrize('kind', ['web_flag', 'web_task', 'web_inputs', 'thought_only', 'gspo'])
def test_nondefault_runtime_rejected_before_cuda(trainer, monkeypatch, kind):
    args = SimpleNamespace(json_surrogate_weight=4, joint_json_policy=kind != 'thought_only',
        policy_surrogate='episode-gspo' if kind == 'gspo' else 'token-grpo',
        enable_online_web=kind == 'web_flag')
    tasks = [SimpleNamespace(benchmark='widesearch' if kind == 'web_task' else 'cuad')]
    def forbidden(*args, **kwargs):
        raise AssertionError('Unsupported weighted mode must not initialize CUDA')
    monkeypatch.setattr(torch.cuda, 'set_device', forbidden)
    with pytest.raises(ValueError, match='CUAD joint JSON token-GRPO'):
        asyncio.run(trainer.train(args, tasks, [], 1,
            web_inputs=object() if kind == 'web_inputs' else None))
    if kind != 'web_inputs':
        with pytest.raises(ValueError, match='CUAD joint JSON token-GRPO'):
            trainer.configuration(args, [], tasks, 1)


def test_weighted_zero_group_does_not_replay_or_apply_existing_adam_momentum(trainer):
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    model.weight.grad = torch.ones_like(model.weight)
    apply_policy_update(list(model.parameters()), optimizer)
    optimizer.zero_grad(set_to_none=True)
    before, state = model.weight.detach().clone(), deepcopy(optimizer.state_dict())
    def forbidden(*args, **kwargs):
        raise AssertionError('Zero advantages must not replay')
    report = trainer.replay_group(model, SimpleNamespace(segments=[[],[]], advantages=[0.,0.]),
        forbidden, **KW, joint_json_policy=True, grammar_compiler=object(),
        replay_diagnostic='off', json_surrogate_weight=4)
    norm, applied = apply_policy_update(list(model.parameters()), optimizer)
    assert report['policy_loss'] == 0 and report['scored_tokens'] == 0
    assert norm == 0 and not applied and torch.equal(model.weight, before)
    for key,value in state['state'][0].items():
        assert torch.equal(value, optimizer.state_dict()['state'][0][key])
