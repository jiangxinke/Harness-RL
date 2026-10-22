"""CPU mechanics only: newly sampled CUAD joint control, ownership and math.

All examples/models are synthetic. These tests contain no benchmark outcomes.
"""
from __future__ import annotations
import asyncio
from copy import deepcopy
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest
import torch

import dcc_online_grpo as trainer
from slime_plugins.dcc import cuad_train_thought_budget_close as close
from slime_plugins.dcc import web_train_thought_budget_close as web_close
from slime_plugins.dcc import hf_native_policy as native
from slime_plugins.dcc import online_web as web
from slime_plugins.dcc.cuad_structured import cuad_final_schema
from slime_plugins.dcc.data import Task
from slime_plugins.dcc.episode import EpisodeResult, ModelProtocolFailure
from slime_plugins.dcc.hf_policy import _chat_tokens
from slime_plugins.dcc.native_behavior_logprobs import BEHAVIOR_LOGPROB_SOURCE
from slime_plugins.dcc.native_hf_context import (NativeHFPromptBudgetExceeded, REASON,
    _preflight_behavior_source, restore_verified_native_context, validated_thought_step)
from slime_plugins.dcc.native_joint_grpo import (CuadBudgetClosedJointSegment,
    JSON_BEHAVIOR_SOURCE, joint_group_signal, segment_scorer, weighted_joint_policy_loss)
from slime_plugins.dcc.online_grpo import group_signal, thought_policy_loss
from slime_plugins.dcc.reference_kl import frozen_base_logprobs
from slime_plugins.dcc.web_hf_evaluation import prediction_row, verify_native_ownership
from slime_plugins.dcc.web_structured import web_final_schema
from test_dcc_joint_json_policy import tiny as tiny_joint


@pytest.fixture
def tiny(tiny_joint, monkeypatch):
    monkeypatch.setattr(close, 'END_THINK', native.END_THINK)
    return tiny_joint


def policy_for(tiny, *, enabled=True, budget=4, json_tokens=2, traces=None, **kwargs):
    options = {} if enabled is None else {close.FIELD: enabled}
    return native.NativeHFPolicy(*tiny, thought_tokens=budget, json_tokens=json_tokens,
        max_prompt_tokens=128, capture_thought_logprobs=True, joint_json_policy=True,
        trace_sink=None if traces is None else traces.append, **options, **kwargs)


def sample(tiny, *, force=True, **kwargs):
    model, _ = tiny
    if force:
        with torch.no_grad():
            model.lm_head.bias[native.END_THINK] = -40
    traces = []
    policy = policy_for(tiny, traces=traces, **kwargs)
    torch.manual_seed(41)
    result = policy._generate_encoded_batch([[5]], cuad_final_schema(), ['synthetic-cuad'])[0]
    completion = result.completion if isinstance(result, ModelProtocolFailure) else result
    return policy, completion, traces


def segment(completion):
    count = completion.thought_token_count
    return CuadBudgetClosedJointSegment(completion.prompt_ids, completion.completion_ids,
        completion.thought_logprobs, completion.json_logprobs, count,
        completion.json_public_schema, completion.json_stop_ids,
        **{close.FIELD: getattr(completion, close.FIELD),
           close.POSITIONS: tuple(getattr(completion, close.POSITIONS))})


def normalized(value):
    if isinstance(value, dict):
        return {key: normalized(item) for key, item in value.items()
                if key not in {'trace_id', 'batch_phase_elapsed_seconds'}}
    if isinstance(value, (list, tuple)):
        return [normalized(item) for item in value]
    return value


def test_default_missing_and_false_keep_real_ids_scores_rng_serialization_and_config(tiny):
    records = []
    for enabled in (None, False):
        traces = []
        policy = policy_for(tiny, enabled=enabled, traces=traces)
        torch.manual_seed(41)
        result = policy._generate_encoded_batch([[5]], cuad_final_schema(), ['synthetic-cuad'])[0]
        result = result.completion if isinstance(result, ModelProtocolFailure) else result
        assert type(result) is native.JointNativeHFCompletion
        assert close.FIELD not in asdict(result) and close.FIELD not in traces[0]
        assert close.FIELD not in policy.provenance
        records.append((normalized(asdict(result)), normalized(traces), policy.provenance,
                        torch.get_rng_state().clone()))
    assert records[0][:-1] == records[1][:-1]
    assert torch.equal(records[0][-1], records[1][-1])
    args = SimpleNamespace(model='synthetic', source_commit='0' * 40, group_size=2,
        max_steps=2, thought_tokens=4, json_tokens=2, joint_json_policy=True)
    baseline = trainer.configuration(args, [], [], 1)
    setattr(args, close.FIELD, False)
    assert trainer.configuration(args, [], [], 1) == baseline
    setattr(args, close.FIELD, True)
    enabled = trainer.configuration(args, [], [], 1)
    assert enabled[close.FIELD] == close.configuration(4)
    assert enabled['objective_denominator'] == baseline['objective_denominator'] == 24
    assert enabled['native_policy_protocol'] == close.PROTOCOL
    assert enabled['behavior_policy']['old_logprob_source']['json'] == JSON_BEHAVIOR_SOURCE
    assert web_close.FIELD not in enabled


@pytest.mark.parametrize('extra', [[], ['--joint-json-policy', '--policy-surrogate', 'episode-gspo'],
    ['--joint-json-policy', '--enable-online-web'], ['--web-train-thought-budget-close', '--enable-online-web']])
def test_cli_rejects_unsupported_modes_before_source_data_model_cuda(monkeypatch, extra):
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid mode accessed source/data/model/CUDA')
    for name in ('load_tasks', 'validate_source', 'load_policy_model'):
        monkeypatch.setattr(trainer, name, forbidden)
    monkeypatch.setattr(torch.cuda, 'set_device', forbidden)
    with pytest.raises(SystemExit) as result:
        trainer.main(['--model', 'synthetic', '--train-data', 'absent', '--output-dir', 'unused',
            '--source-commit', '0' * 40, '--cuad-train-thought-budget-close', *extra])
    assert result.value.code == 2


@pytest.mark.parametrize('options', [dict(), dict(joint_json_policy=True),
    dict(joint_json_policy=True, capture_thought_logprobs=True, sampling_profile='greedy'),
    dict(joint_json_policy=True, capture_thought_logprobs=True, allow_truncation=True),
    dict(joint_json_policy=True, capture_thought_logprobs=True, web_train_thought_budget_close=True)])
def test_native_mode_restrictions_precede_model_access(options):
    with pytest.raises(ValueError):
        native.NativeHFPolicy(None, None, cuad_train_thought_budget_close=True, **options)


@pytest.mark.parametrize('budget', [1, 4])
def test_actual_processed_singleton_logp_zero_and_json_stays_masked_stochastic(tiny, budget):
    policy, completion, traces = sample(tiny, budget=budget)
    assert type(completion) is native.CuadBudgetClosedJointNativeHFCompletion
    count = completion.thought_token_count
    assert count == budget and completion.completion_ids[count - 1] == close.END_THINK
    assert completion.thought_logprobs[-1] == 0
    assert getattr(completion, close.POSITIONS) == [budget - 1]
    assert completion.thought_logprobs_source == close.SOURCE
    assert completion.json_logprobs_source == JSON_BEHAVIOR_SOURCE
    assert len(completion.json_logprobs) == len(completion.completion_ids) - count == 2
    assert traces[0]['protocol'] == close.PROTOCOL
    assert traces[0]['json_sampling_profile'] == 'multinomial-T1-top_p1-top_k0-xgrammar'
    assert web_close.FIELD not in asdict(completion) and web_close.POSITIONS not in traces[0]
    score = segment_scorer(segment(completion), policy.compiler, backend='original')
    current = score(tiny[0], completion.prompt_ids, completion.completion_ids)
    assert current[count - 1] == 0
    torch.testing.assert_close(current.detach(), torch.tensor(segment(completion).behavior_logprobs), atol=1e-5, rtol=1e-5)


def test_natural_early_close_keeps_raw_score_and_has_no_forced_mask(tiny):
    policy, completion, _ = sample(tiny, force=False)
    assert 0 < completion.thought_token_count < 4
    assert completion.completion_ids[completion.thought_token_count - 1] == close.END_THINK
    assert completion.thought_logprobs[-1] < 0
    assert getattr(completion, close.POSITIONS) == []
    values = segment_scorer(segment(completion), policy.compiler, backend='original')(
        tiny[0], completion.prompt_ids, completion.completion_ids)
    assert values[completion.thought_token_count - 1] < 0


@pytest.mark.parametrize('weight', [1, 4])
def test_analytical_weighted_numerator_mask_keeps_BplusJ_denominator_and_gradients(weight):
    current = torch.tensor([-.3, 0., -.7, -.2], requires_grad=True)
    old = current.detach().clone()
    mask = torch.tensor([True, False, True, True])
    if weight == 1:
        loss = thought_policy_loss(current, old, -.5, group_size=2, max_steps=3,
            thought_tokens=4, policy_mask=mask)
    else:
        loss = weighted_joint_policy_loss(current, old, -.5, thought_count=2,
            group_size=2, max_steps=3, thought_tokens=2, json_tokens=2,
            json_surrogate_weight=weight, policy_mask=mask)
    assert loss.item() == pytest.approx(.5 * (1 + 2 * weight) / 24)
    loss.backward()
    torch.testing.assert_close(current.grad, torch.tensor([.5, 0., .5 * weight, .5 * weight]) / 24)


@pytest.mark.parametrize('backend,weight', [('original', 1), ('qwen3-moe-chunked', 4)])
def test_real_lora_q_current_diagnostic_reference_and_weighted_loss_share_policy(tiny, backend, weight):
    from peft import LoraConfig, get_peft_model
    model, tok = tiny
    with torch.no_grad():
        model.lm_head.bias[native.END_THINK] = -40
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
        target_modules=['q_proj', 'v_proj', 'o_proj'], task_type='CAUSAL_LM'))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'lora_B' in name:
                parameter.normal_(std=.04)
    policy, completion, _ = sample((model, tok), force=False)
    captured = segment(completion)
    score = segment_scorer(captured, policy.compiler, backend=backend, head_chunk_tokens=3)
    model.eval()
    reference = frozen_base_logprobs(model, score, captured.prompt_ids, captured.completion_ids)
    current = score(model, captured.prompt_ids, captured.completion_ids)
    old = torch.tensor(captured.behavior_logprobs)
    assert current[3] == reference[3] == old[3] == 0
    mask = torch.tensor([True, True, True, False, True, True])
    ratio = (current - old).exp()
    surrogate = torch.minimum(ratio * .4, ratio.clamp(.8, 1.2) * .4)
    weights = torch.tensor([1, 1, 1, 1, weight, weight])
    oracle_policy = -(surrogate[mask] * weights[mask]).sum() / (2 * 2 * 6)
    difference = reference - current
    oracle_kl = (torch.expm1(difference) - difference).sum() / (2 * 2 * 6)
    expected = oracle_policy + .03 * oracle_kl
    expected.backward()
    gradients = [parameter.grad.detach().clone() if parameter.grad is not None else None
                 for parameter in model.parameters()]
    model.zero_grad(set_to_none=True)
    report = trainer.replay_group(model, SimpleNamespace(segments=[[captured], []], advantages=[.4, 0.]), None,
        group_size=2, max_steps=2, thought_tokens=4, json_tokens=2, clip=.2,
        joint_json_policy=True, grammar_compiler=policy.compiler,
        logprob_backend=backend, head_chunk_tokens=3, replay_diagnostic='full',
        reference_kl_beta=.03, reference_base_model='synthetic', json_surrogate_weight=weight)
    assert report['policy_loss'] == pytest.approx(oracle_policy.item(), abs=1e-7)
    assert report['reference_kl_unweighted_loss'] == pytest.approx(oracle_kl.item(), abs=1e-7)
    assert report['forced_close_scored_tokens'] == 1 and report['stochastic_scored_tokens'] == 5
    assert report['importance_ratio_outside_clip_fraction'] == 0
    assert report['maximum_behavior_replay_logprob_difference'] < 1e-5
    assert report['maximum_replay_current_logprob_difference'] < 1e-5
    for parameter, gradient in zip(model.parameters(), gradients, strict=True):
        assert (parameter.grad is None) == (gradient is None)
        if gradient is not None:
            torch.testing.assert_close(parameter.grad, gradient, rtol=2e-4, atol=1e-6)
    assert not torch.cuda.is_initialized()


def test_new_joint_identity_survives_groups_and_cannot_relabel_raw_or_Web(tiny):
    policy, completion, _ = sample(tiny)
    result = EpisodeResult('synthetic-cuad', 'cuad', 'train', [], 'model_protocol_failure', 0., {}, [completion], [], [])
    signal = joint_group_signal([result, deepcopy(result)], group_size=2, max_steps=2,
        thought_tokens=4, json_tokens=2, eos_ids=policy.eos_ids)
    assert signal.discarded_reason is None and signal.advantages == [0, 0]
    assert type(signal.segments[0][0]) is CuadBudgetClosedJointSegment
    assert signal.segments[0][0].cuad_forced_thought_positions == (3,)
    with pytest.raises(ValueError, match='joint JSON group'):
        group_signal([result, deepcopy(result)], group_size=2, max_steps=2, thought_tokens=4)
    bad_cases = [replace(completion, thought_logprobs_source=BEHAVIOR_LOGPROB_SOURCE),
                 replace(completion, cuad_train_thought_budget_close=None),
                 replace(completion, cuad_forced_thought_positions=[]),
                 replace(completion, thought_logprobs=[-.1, -.2, -.3, -.4]),
                 replace(completion, thought_token_count=0, completion_ids=[], thought_logprobs=[])]
    for bad in bad_cases:
        with pytest.raises(ValueError):
            validated_thought_step(bad, thought_tokens=4)
    old = asdict(completion)
    old.pop(close.FIELD)
    old.pop(close.POSITIONS)
    old = native.JointNativeHFCompletion(**old)
    with pytest.raises(ValueError):
        validated_thought_step(old, thought_tokens=4)
    forged = SimpleNamespace(**asdict(completion), **{web_close.FIELD: web_close.configuration(4)})
    with pytest.raises(ValueError, match='both Web and CUAD'):
        validated_thought_step(forged, thought_tokens=4)
    with pytest.raises(ValueError):
        trainer.replay_group(tiny[0], signal, None, group_size=2, max_steps=2,
                             thought_tokens=4, clip=.2, joint_json_policy=False)


def test_cuad_mode_rejects_mixed_Web_rows_before_any_phase(tiny, monkeypatch):
    policy = policy_for(tiny, mixed_schema_batching=True)
    def forbidden(*args, **kwargs):
        raise AssertionError('Foreign schema submitted a model phase')
    monkeypatch.setattr(policy, '_phase', forbidden)
    with pytest.raises(ValueError, match='CUAD only'):
        policy._generate_encoded_batch([[5], [5, 6]],
            [cuad_final_schema(), web_final_schema(['name'], [])], ['cuad', 'web'])


def test_ragged_CUAD_cancellation_preserves_row_identity_and_actual_stochastic_JSON(tiny, monkeypatch):
    model, _ = tiny
    with torch.no_grad():
        model.lm_head.bias[native.END_THINK] = -40
    traces = []
    policy = policy_for(tiny, traces=traces, mixed_schema_batching=True)
    controls = [native.NativeHFRequestControl(), native.NativeHFRequestControl()]
    original, calls = policy._phase, []
    def phase(prompts, **kwargs):
        calls.append((kwargs['phase'], len(prompts)))
        rows = original(prompts, **kwargs)
        if kwargs['phase'] == 'thought':
            controls[0].expire()
        return rows
    monkeypatch.setattr(policy, '_phase', phase)
    torch.manual_seed(41)
    results = policy._generate_encoded_batch([[5], [5, 6, 7]],
        [cuad_final_schema(), cuad_final_schema()], ['cancelled', 'sibling'], request_controls=controls)
    outputs = [r.completion if isinstance(r, ModelProtocolFailure) else r for r in results]
    assert calls == [('thought', 2), ('json', 1)]
    first, second = outputs
    assert first.native_status == native.CANCELLED_BEFORE_JSON
    assert first.json_logprobs == [] and first.json_logprobs_source is None
    assert first.completion_ids == traces[0]['thought_ids'] and traces[0]['json_ids'] == []
    assert getattr(first, close.POSITIONS) == getattr(second, close.POSITIONS) == [3]
    assert first.thought_logprobs[-1] == second.thought_logprobs[-1] == 0
    assert second.json_logprobs_source == JSON_BEHAVIOR_SOURCE and second.json_logprobs
    assert [row['task_id'] for row in traces] == ['cancelled', 'sibling']
    assert [item.prompt_ids for item in outputs] == [[5], [5, 6, 7]]
    unreliable = EpisodeResult('synthetic', 'cuad', 'train', [], 'infrastructure_error', 0., {}, [first], [], [])
    assert joint_group_signal([unreliable, deepcopy(unreliable)], group_size=2, max_steps=2,
        thought_tokens=4, json_tokens=2, eos_ids=policy.eos_ids).discarded_reason


def test_expired_CUAD_queue_emits_no_completion_trace_or_sampled_policy_fields(tiny, monkeypatch):
    traces = []
    policy = policy_for(tiny, traces=traces)
    def forbidden(*args, **kwargs):
        raise AssertionError('Expired request submitted a model phase')
    monkeypatch.setattr(policy, '_generate_batch', forbidden)
    control = native.NativeHFRequestControl()
    control.expire()
    async def run():
        try:
            with pytest.raises(native.NativeHFRequestExpiredBeforeSubmission) as error:
                await policy.generate_structured([{'role': 'user', 'content': 'synthetic'}],
                    json_schema=cuad_final_schema(), max_new_tokens=6, temperature=1.,
                    _task_id='synthetic-expired', _request_control=control)
            return error.value
        finally:
            await policy.close()
    error = asyncio.run(run())
    assert not control.started and traces == []
    assert close.FIELD not in vars(error) and close.POSITIONS not in vars(error)


def test_real_CUAD_preflight_has_planned_identity_without_sampled_positions(tiny):
    policy = policy_for(tiny)
    policy.max_prompt_tokens = 1
    failure = policy._generate_batch([[{'role': 'user', 'content': 'synthetic'}]],
        cuad_final_schema(), ['synthetic-cuad'])[0]
    assert isinstance(failure, NativeHFPromptBudgetExceeded)
    assert failure.preflight[close.FIELD] == close.configuration(4)
    assert failure.preflight['thought_logprobs_source'] == close.SOURCE
    assert close.POSITIONS not in failure.preflight
    assert _preflight_behavior_source(failure.preflight) == close.SOURCE
    for corrupt in (dict(failure.preflight, **{close.POSITIONS: []}),
                    dict(failure.preflight, **{web_close.FIELD: web_close.configuration(4)}),
                    dict(failure.preflight, **{close.FIELD: None})):
        with pytest.raises(ValueError):
            _preflight_behavior_source(corrupt)


def test_actual_CUAD_joint_context_prefix_restores_only_with_its_native_policy_journal(tiny):
    policy, completion, traces = sample(tiny, json_tokens=192)
    assert completion.native_status == 'completed'
    messages = [{'role': 'user', 'content': 'invented request'} for _ in range(150)]
    failure = policy._generate_batch([messages], cuad_final_schema(), ['synthetic-cuad'])[0]
    assert isinstance(failure, NativeHFPromptBudgetExceeded)
    failed = EpisodeResult('synthetic-cuad', 'cuad', 'train', [], 'context_budget_failure', 0.,
        {'prompt_history_failure': dict(reason=REASON, generation_submitted=False,
            diagnostics=failure.diagnostics, operation='policy', turn_index=1,
            prior_model_calls=1, stage='after_model_call', request_messages=messages,
            native_hf_preflight=failure.preflight)}, [completion], [], [])
    sibling = replace(failed, status='completed', reward=1., metrics={})
    def signal(result):
        return joint_group_signal([sibling, result], group_size=2, max_steps=2,
            thought_tokens=4, json_tokens=192, eos_ids=policy.eos_ids)
    before = signal(failed)
    assert before.discarded_reason is None
    assert before.segments[1][0].cuad_forced_thought_positions == (3,)
    serialized = json.loads(json.dumps(failed.to_dict()))
    serialized['steps'] = [native.CuadBudgetClosedJointNativeHFCompletion(**row)
                           for row in serialized['steps']]
    unverified = EpisodeResult(**serialized)
    assert signal(unverified).discarded_reason
    restored = restore_verified_native_context(unverified, traces=traces, tokenizer=tiny[1])
    assert signal(restored) == before
    corruptions = ((close.POSITIONS, []), (close.FIELD, None),
                   (web_close.FIELD, web_close.configuration(4)))
    for key, value in corruptions:
        corrupted = deepcopy(traces)
        corrupted[0][key] = value
        with pytest.raises(ValueError):
            restore_verified_native_context(unverified, traces=corrupted, tokenizer=tiny[1])
    with pytest.raises(ValueError, match='error journal'):
        restore_verified_native_context(unverified, traces=traces[:-1], tokenizer=tiny[1])


@pytest.mark.parametrize('enabled', [False, True])
def test_real_CPU_trainer_closes_native_journals_checkpoints_and_independent_terminal_identity(
        tiny, tmp_path, monkeypatch, enabled):
    from peft import LoraConfig, get_peft_model
    model, tokenizer = tiny
    with torch.no_grad():
        model.lm_head.bias[native.END_THINK] = -40
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
        target_modules=['q_proj', 'v_proj', 'o_proj'], task_type='CAUSAL_LM'))
    class Runner:
        async def run(self, task, policy, **kwargs):
            # Both arms actually sample their own native thought policy. The
            # deliberately tiny JSON budget makes these synthetic failures.
            with pytest.raises(ModelProtocolFailure) as failure:
                await policy.generate_structured([{'role': 'user', 'content': 'invented'}],
                    json_schema=cuad_final_schema(), max_new_tokens=6, temperature=1.)
            return EpisodeResult(task.id, 'cuad', 'train', [], 'model_protocol_failure',
                                 0., {}, [failure.value.completion], [], [])
    for key in ('RANK', 'LOCAL_RANK'):
        monkeypatch.setenv(key, '0')
    monkeypatch.setenv('WORLD_SIZE', '1')
    monkeypatch.setattr(torch.cuda, 'set_device', lambda _: None)
    monkeypatch.setattr(torch.cuda, 'get_rng_state', torch.get_rng_state)
    def no_cuda(*args, **kwargs):
        raise AssertionError('Synthetic trainer must stay on CPU')
    monkeypatch.setattr(torch.cuda, '_lazy_init', no_cuda)
    monkeypatch.setattr(trainer, 'load_policy_model', lambda *args, **kwargs: (model, tokenizer))
    monkeypatch.setattr(trainer, 'make_runner', lambda args: Runner())
    args = SimpleNamespace(model='synthetic-CPU-only', adapter=None, source_commit='0' * 40,
        output_dir=tmp_path / 'training', steps=1, seed=47, lora_rank=2,
        quantization='none', learning_rate=.01, logprob_backend='original',
        head_chunk_tokens=None, thought_tokens=4, json_tokens=2,
        max_prompt_tokens=32, max_steps=2, group_size=2, clip=.2,
        max_grad_norm=1., sync_artifacts=False, save_every=0, replay_diagnostic='off',
        native_batch_size=2, joint_json_policy=True, **{close.FIELD: enabled})
    task = Task('synthetic-cuad', 'cuad', 'train', 'Invented request', 'Invented clause', [])
    asyncio.run(trainer.train(args, [task], [[0]], 1))
    complete = json.loads((args.output_dir / 'training_complete.json').read_text())
    config = json.loads((args.output_dir / 'final' / 'training_config.json').read_text())
    rows = [json.loads(line) for line in
            (args.output_dir / 'native_traces_rank0.jsonl').read_text().splitlines()]
    assert complete['complete'] and complete['optimizer_updates'] == 0
    assert complete['has_policy_gradient_updates'] is False
    assert complete['algorithm'] == config['algorithm']
    assert complete['ranks'][0]['closed_journals']['traces'] == len(rows) == 2
    assert complete['ranks'][0]['closed_journals']['rollouts'] == 2
    assert (close.FIELD in complete) is enabled
    assert (close.FIELD in config) is enabled
    if enabled:
        assert complete[close.FIELD] == config[close.FIELD] == close.configuration(4)
        assert complete['algorithm_variant'] == config['algorithm_variant'] == close.VARIANT
        assert all(row['protocol'] == close.PROTOCOL and row[close.POSITIONS] == [3] for row in rows)
    else:
        assert 'algorithm_variant' not in complete
        assert all(close.FIELD not in row and close.POSITIONS not in row for row in rows)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('kind', ['train', 'evaluation'])
@pytest.mark.parametrize('error_type', [native.NativeHFRequestExpiredBeforeSubmission.__name__, 'OSError'])
def test_Web_error_owners_reject_foreign_CUAD_fields_including_empty_values(tiny, kind, error_type):
    _, tokenizer = tiny
    task = Task('synthetic-owner', 'widesearch', 'train' if kind == 'train' else 'validation',
                'Invented request', metadata={'evaluation': {'required': ['name']}})
    result = EpisodeResult(task.id, task.benchmark, task.split, [], 'infrastructure_error', 0., {}, [], [], [])
    owner = dict(task_id=task.id, sample=0, iteration=1, rank=0, shard_index=0, turn=0,
        request_status='error', delivery='cancelled', error_type=error_type,
        messages=[{'role': 'user', 'content': 'synthetic'}], json_schema=web_final_schema(['name'], []))
    def verify(row):
        if kind == 'train':
            return web.verify_owned_native_group(task, [result, deepcopy(result)], [row], [],
                eos_ids=[1], tokenizer=tokenizer, iteration=1, rank=0)
        return verify_native_ownership([task], [prediction_row(result, seed=1)], [row], [],
            rank=0, profile='raw', eos_ids=[1], tokenizer=tokenizer)
    verify(owner)
    if error_type == 'OSError':
        verify(dict(owner, native_hf_preflight=None, completion=None))
    for key, value in ((close.FIELD, close.configuration(4)), (close.FIELD, None), (close.POSITIONS, [])):
        with pytest.raises(ValueError, match='CUAD singleton'):
            verify(dict(owner, **{key: value}))
