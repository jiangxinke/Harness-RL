"""Synthetic CPU HF/xgrammar mechanics and closed CUAD EVAL evidence only."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from slime_plugins.dcc import cuad_eval_thought_budget_close as close
from slime_plugins.dcc import hf_native_policy as native
from slime_plugins.dcc import EpisodeRunner
from slime_plugins.dcc.cuad_structured import cuad_final_schema
from slime_plugins.dcc.data import Task
from slime_plugins.dcc.episode import ModelProtocolFailure
from slime_plugins.dcc.hf_policy import _chat_tokens
from slime_plugins.dcc.native_hf_context import NativeHFPromptBudgetExceeded
from slime_plugins.dcc.web_structured import web_final_schema
from test_dcc_joint_json_policy import tiny as tiny_model


@pytest.fixture
def tiny(tiny_model, monkeypatch):
    monkeypatch.setattr(close, 'END_THINK', native.END_THINK)
    return tiny_model


def policy(tiny, *, enabled=True, profile='raw', budget=4, json_tokens=96,
           thoughts=None, traces=None, batch_size=4):
    model, tokenizer = tiny
    if hasattr(model, '_eval_fixture_hook'):
        model._eval_fixture_hook.remove()
    if not hasattr(model, '_eval_original_generate'):
        model._eval_original_generate = model.generate
    original = model._eval_original_generate
    thought_values = thoughts if thoughts is not None else [tokenizer.encode('a')[0]] * budget
    final = tokenizer.encode('{"plan":"Read","checks":"Ok","answer":[]}') + [tokenizer.eos_token_id]
    state, calls = {}, []
    def logits(module, args, output):
        token = state['tokens'][min(state['cursor'], len(state['tokens']) - 1)]
        state['cursor'] += 1
        value = torch.full_like(output, -10000.)
        value[..., token] = 10000.
        return value
    model._eval_fixture_hook = model.lm_head.register_forward_hook(logits)
    def generate(**kwargs):
        config = kwargs.get('generation_config')
        eos_ids = config.eos_token_id if config is not None else kwargs['eos_token_id']
        phase = 'thought' if native.END_THINK in eos_ids else 'json'
        state.update(tokens=thought_values if phase == 'thought' else final, cursor=0)
        result = original(**kwargs)
        calls.append(dict(phase=phase, ids=result[:, kwargs['input_ids'].shape[1]:].tolist(), config=config, kwargs=kwargs))
        return result
    model.generate = generate
    options = {} if enabled is None else {close.FIELD: enabled}
    value = native.NativeHFPolicy(model, tokenizer, thought_tokens=budget, json_tokens=json_tokens,
        max_prompt_tokens=128, batch_size=batch_size, sampling_profile=profile,
        trace_sink=None if traces is None else traces.append, **options)
    return value, calls


def task():
    return Task('synthetic-CUAD', 'cuad', 'train', 'Extract a clause.', 'Invented contract.', [])


def public_messages():
    return [{'role': 'user', 'content': 'Invented public request.'}]


def run_owned(value, *, timeout=10, rank=0):
    owners = []
    async def run():
        bound = close.OwnedCuadEvaluationPolicy(value.bind_task(task().id, rank),
            task_id=task().id, rank=rank, sink=owners.append)
        result = await EpisodeRunner(max_steps=1, max_new_tokens=value.thought_tokens + value.json_tokens,
            evaluation_temperature=1.0, timeout_seconds=timeout,
            cuad_structured_actions=True).run(task(), bound, evaluation=True, score_train=False)
        await value.close()
        row = result.to_dict(include_tokens=True)
        row.pop('reward')
        row.update(rollout_labels_removed=True, rollout_scored=False)
        return row
    # Match JSON persistence, where tuples become lists.
    return json.loads(json.dumps(asyncio.run(run()))), json.loads(json.dumps(owners))


def verify(value, predictions, owners, traces, *, tokenizer=True):
    return close.verify_native_ownership([task()], predictions, owners, json.loads(json.dumps(traces)),
        rank=0, profile=value.sampling_profile, thought_tokens=value.thought_tokens,
        json_tokens=value.json_tokens, max_prompt_tokens=value.max_prompt_tokens,
        eos_ids=value.eos_ids, tokenizer=value.tokenizer if tokenizer else None)


@pytest.mark.parametrize('profile,budget', [('raw', 4), ('thinking', 4), ('raw', 1)])
def test_real_hf_last_existing_slot_singleton_greedy_json_and_complete_proof(tiny, profile, budget):
    traces = []
    value, calls = policy(tiny, profile=profile, budget=budget, traces=traces)
    observed = []
    generate = value.model.generate
    class Observer:
        def __call__(self, ids, scores):
            if ids.shape[1] == self.width + budget - 1:
                observed.append((int(torch.isfinite(scores).sum()), float(scores[0, native.END_THINK] - scores[0].logsumexp(-1))))
            return scores
    def inspect(**kwargs):
        cfg = kwargs.get('generation_config')
        eos = cfg.eos_token_id if cfg else kwargs['eos_token_id']
        if native.END_THINK in eos:
            observer = Observer();observer.width = kwargs['input_ids'].shape[1]
            kwargs['logits_processor'].append(observer)
        return generate(**kwargs)
    value.model.generate = inspect
    row, owners = run_owned(value)
    assert row['status'] == 'completed', row['detail']
    assert observed == [(1, 0.)]
    assert calls[0]['ids'][0][-1] == native.END_THINK and len(calls[0]['ids'][0]) == budget
    cfg = calls[1]['config']
    assert (cfg.do_sample if cfg else calls[1]['kwargs']['do_sample']) is False
    assert calls[1]['ids'][0][-1] in value.eos_ids
    completion = row['steps'][0]
    assert completion[close.POSITIONS] == [budget - 1]
    assert completion['thought_logprobs'] is completion['thought_logprobs_source'] is None
    assert 'json_logprobs' not in completion
    assert traces[0]['protocol'] == value.protocol == close.PROTOCOL
    assert verify(value, [row], owners, traces)['all_native_traces_owned']
    assert verify(value, [row], owners, traces, tokenizer=False)['all_native_traces_owned']
    assert not torch.cuda.is_initialized()


def normalized(value):
    if isinstance(value, dict):
        return {key: normalized(item) for key, item in value.items()
                if key not in {'trace_id', 'batch_phase_elapsed_seconds'}}
    if isinstance(value, (list, tuple)):
        return [normalized(item) for item in value]
    return value


def test_omitted_false_default_ids_rng_and_serialization_are_identical(tiny):
    records = []
    for enabled in (None, False):
        traces = []
        value, _ = policy(tiny, enabled=enabled, thoughts=[native.END_THINK], traces=traces)
        torch.manual_seed(83)
        completion = value._generate_batch([public_messages()], cuad_final_schema(), [task().id])[0]
        assert type(completion) is native.NativeHFCompletion
        assert close.FIELD not in asdict(completion) and close.FIELD not in traces[0]
        assert close.FIELD not in value.provenance
        records.append((normalized(asdict(completion)), normalized(traces), torch.get_rng_state()))
    assert records[0][:-1] == records[1][:-1]
    assert torch.equal(records[0][-1], records[1][-1])


def test_early_natural_close_preserves_actual_ids_rng_and_has_no_forced_position(tiny):
    records = []
    for enabled in (False, True):
        value, calls = policy(tiny, enabled=enabled, thoughts=[native.END_THINK])
        torch.manual_seed(83)
        completion = value._generate_batch([public_messages()], cuad_final_schema(), [task().id])[0]
        records.append((completion.completion_ids, torch.get_rng_state()))
        if enabled:
            assert completion.forced_thought_positions == [] and completion.thought_token_count == 1
    assert records[0][0] == records[1][0] and torch.equal(records[0][1], records[1][1])


@pytest.mark.parametrize('options', [dict(capture_thought_logprobs=True), dict(joint_json_policy=True),
    dict(web_thought_budget_close=True), dict(web_train_thought_budget_close=True),
    dict(cuad_train_thought_budget_close=True), dict(allow_truncation=True),
    dict(sampling_profile='greedy'), dict(mixed_schema_batching=True), dict(web_bounded_whitespace=True),
    dict(cuad_thought_budget_close=1)])
def test_rejects_wrong_mode_before_model_access(options):
    with pytest.raises(ValueError):
        native.NativeHFPolicy(None, None, **{close.FIELD: True, **options})


def test_exact_cuad_schema_admission_precedes_generation(tiny):
    value, calls = policy(tiny)
    with pytest.raises(ValueError, match='exact CUAD evaluation schemas'):
        value._generate_batch([public_messages()], web_final_schema(['name'], []))
    assert not calls


@pytest.mark.parametrize('change', ['forced', 'protocol', 'capture', 'json_capture', 'tokens', 'task',
    'duplicate', 'unowned', 'schema', 'greedy', 'owner_steps', 'eos', 'train_identity'])
def test_actual_native_proof_rejects_tampered_records(tiny, change):
    traces = []
    value, _ = policy(tiny, traces=traces)
    row, owners = run_owned(value)
    if change == 'forced':traces[0][close.POSITIONS] = []
    elif change == 'protocol':traces[0]['protocol'] = native.PROTOCOL
    elif change == 'capture':traces[0]['thought_logprobs'] = [0.] * 4
    elif change == 'json_capture':owners[0]['completion']['json_logprobs_source'] = 'invented'
    elif change == 'tokens':traces[0]['thought_ids'][-1] = 5
    elif change == 'task':traces[0]['task_id'] = 'foreign'
    elif change == 'duplicate':traces.append(deepcopy(traces[0]))
    elif change == 'unowned':
        extra = deepcopy(traces[0]);extra['trace_id'] = 'unowned';traces.append(extra)
    elif change == 'schema':owners[0]['json_schema'] = web_final_schema(['name'], [])
    elif change == 'greedy':traces[0]['json_sampling_profile'] = 'multinomial'
    elif change == 'owner_steps':owners[0]['completion']['text'] = '{}'
    elif change == 'eos':traces[0]['json_stop_ids'] = [99]
    else:traces[0]['cuad_train_thought_budget_close'] = {}
    with pytest.raises(ValueError):verify(value, [row], owners, traces)


def test_real_preflight_remains_unsubmitted_without_native_completion_or_train_capture(tiny):
    traces = []
    value, calls = policy(tiny, traces=traces)
    value.max_prompt_tokens = 1
    row, owners = run_owned(value)
    assert row['status'] == 'context_budget_failure'
    assert not calls and not traces and row['steps'] == []
    assert owners[0]['error_type'] == NativeHFPromptBudgetExceeded.__name__
    proof = owners[0]['native_hf_preflight']
    assert proof['generation_submitted'] is False and proof['thought_logprobs_source'] is None
    assert close.POSITIONS not in proof and close.FIELD not in proof
    assert verify(value, [row], owners, traces)['native_traces'] == 0


def load_evaluator():
    spec = importlib.util.spec_from_file_location('CUAD_eval_budgetclose_subject', Path(__file__).resolve().parents[1] / 'scripts/dcc_online_evaluate.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def arguments(module, output, **extra):
    args = module.parser().parse_args(['--model', 'synthetic-CPU', '--data', 'unused', '--output-dir', str(output),
        '--source-commit', 'a' * 40, '--cuad-thought-budget-close', '--max-steps', '1',
        '--thought-tokens', '4', '--json-tokens', '96', '--max-prompt-tokens', '128', '--initial-chunks', '0'])
    for key, value in extra.items():setattr(args, key, value)
    return args


def test_actual_cpu_generate_score_matched_base_adapter_and_tamper(tiny, monkeypatch, tmp_path):
    module = load_evaluator()
    def load(*args, **kwargs):return tiny
    monkeypatch.setattr(module, 'load_policy_model', load)
    value, calls = policy(tiny)
    # Keep the actual independent NativeHFPolicy and installed tiny HF logits.
    items = [task()]
    base = arguments(module, tmp_path / 'base')
    asyncio.run(module.generate(base, items))
    baseline = module.score(base, items)
    assert baseline['native_ownership']['all_native_traces_owned'] is True
    assert baseline['comparison'][close.FIELD] == close.configuration(4)
    assert baseline['metric_task_denominator'] == 1
    trained = arguments(module, tmp_path / 'trained', adapter='synthetic-adapter', base_output_dir=base.output_dir)
    asyncio.run(module.generate(trained, items))
    report = module.score(trained, items)
    assert report['comparison'] == baseline['comparison'] and report['matched_base']['complete'] is True
    assert len(calls) == 4 and not torch.cuda.is_initialized()
    (trained.output_dir / 'report.json').unlink()
    receipt_path = trained.output_dir / 'rank0.done.json'
    receipt = json.loads(receipt_path.read_text())
    for field in ('cuad_train_thought_budget_close', 'cuad_forced_thought_positions', 'web_thought_budget_close'):
        altered = deepcopy(receipt);altered['native_policy'][field] = {}
        receipt_path.write_text(json.dumps(altered))
        with pytest.raises(ValueError, match='provenance'):module.score(trained, items)
    receipt_path.write_text(json.dumps(receipt))
    base_report = base.output_dir / 'report.json'
    original_base = base_report.read_text()
    altered_base = json.loads(original_base);altered_base['comparison'].pop(close.FIELD)
    base_report.write_text(json.dumps(altered_base))
    with pytest.raises(ValueError, match='Base and adapter'):module.score(trained, items)
    base_report.write_text(original_base)
    native_path = trained.output_dir / 'native_traces_rank0.jsonl'
    rows = module.read_jsonl(native_path);rows[0][close.POSITIONS] = []
    native_path.write_text(json.dumps(rows[0]) + '\n')
    with pytest.raises(ValueError, match='policy/position'):module.score(trained, items)
    assert not (trained.output_dir / 'report.json').exists()


@pytest.mark.parametrize('options', [['--profile', 'greedy'], ['--web-concise-reasoning'], ['--web-zip-member-fallback']])
def test_cli_refuses_non_evaluation_paths_before_source_data_or_model(monkeypatch, options):
    module = load_evaluator()
    def forbidden(*args, **kwargs):raise AssertionError('invalid option touched source/data/model')
    for name in ('validate_source', 'load_tasks', 'load_policy_model'):monkeypatch.setattr(module, name, forbidden)
    with pytest.raises(SystemExit) as error:
        module.main(['--model', 'synthetic', '--data', 'absent', '--output-dir', 'unused',
            '--source-commit', 'a' * 40, '--cuad-thought-budget-close', *options])
    assert error.value.code == 2


def test_cancelled_real_thought_prefix_retains_forced_position_and_no_json(tiny):
    import threading
    traces, owners, entered, release = [], [], threading.Event(), threading.Event()
    value, calls = policy(tiny, traces=traces)
    phase = value._phase
    def pause(*args, **kwargs):
        result = phase(*args, **kwargs)
        if kwargs['phase'] == 'thought':
            entered.set()
            if not release.wait(10):raise RuntimeError('synthetic cancellation barrier expired')
        return result
    value._phase = pause
    async def run():
        bound = close.OwnedCuadEvaluationPolicy(value.bind_task(task().id), task_id=task().id, rank=0, sink=owners.append)
        pending = asyncio.create_task(bound.generate_structured(public_messages(), json_schema=cuad_final_schema(),
            max_new_tokens=value.thought_tokens + value.json_tokens, temperature=1.0))
        try:
            assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 10), timeout=11)
            pending.cancel()
            await asyncio.sleep(0)
            release.set()
            with pytest.raises(asyncio.CancelledError):await pending
            await value.close()
        finally:release.set()
    asyncio.run(run())
    assert [call['phase'] for call in calls] == ['thought']
    assert owners[0]['delivery'] == 'cancelled'
    assert owners[0]['completion']['native_status'] == native.CANCELLED_BEFORE_JSON
    assert owners[0]['completion'][close.POSITIONS] == [3]
    assert traces[0]['json_ids'] == []
    row = dict(task_id=task().id, benchmark='cuad', split='train', status='infrastructure_error',
        answer=[], metrics={}, steps=[], rollout_labels_removed=True, rollout_scored=False)
    proof = verify(value, [row], json.loads(json.dumps(owners)), traces)
    assert proof['cancelled_native_completions'] == 1 and proof['delivered_steps'] == 0


def test_exact_training_close_adapter_metadata_admission(tiny, tmp_path):
    import dcc_online_grpo as trainer
    from slime_plugins.dcc import cuad_train_thought_budget_close as train_close
    module = load_evaluator()
    from types import SimpleNamespace
    args = SimpleNamespace(model='synthetic', source_commit='a' * 40, group_size=2,
        max_steps=1, thought_tokens=4, json_tokens=96, joint_json_policy=True,
        cuad_train_thought_budget_close=True)
    config = trainer.configuration(args, [], [], 1)
    config.update(rollout_iterations=1, optimizer_updates=1)
    output = tmp_path / 'adapter';output.mkdir()
    # Real publish_checkpoint markers contain completeness/steps and files;
    # the original training protocol belongs to training_config.json.
    (output / 'checkpoint_complete.json').write_text(json.dumps(dict(complete=True, optimizer_steps=1, files=[])))
    config_path = output / 'training_config.json'
    config_path.write_text(json.dumps(config))
    assert module.adapter_condition(str(output))['online_policy_gradient_updates_recorded'] is True
    for change in ('algorithm', 'protocol', 'field', 'source', 'joint'):
        altered = deepcopy(config)
        if change == 'algorithm':altered['algorithm'] += '+invented'
        elif change == 'protocol':altered['native_policy_protocol'] = close.PROTOCOL
        elif change == 'field':altered[train_close.FIELD]['extra_tokens'] = 1
        elif change == 'source':altered['behavior_policy']['old_logprob_source']['thought'] = 'invented'
        else:altered['arguments']['joint_json_policy'] = False
        config_path.write_text(json.dumps(altered))
        assert module.adapter_condition(str(output))['online_policy_gradient_updates_recorded'] is False


def test_eval_trace_rejects_orphan_train_positions_and_nonboolean_configuration(tiny):
    traces = []
    value, _ = policy(tiny, traces=traces)
    row, owners = run_owned(value)
    original = deepcopy(traces)
    traces[0]['cuad_forced_thought_positions'] = [3]
    with pytest.raises(ValueError):verify(value, [row], owners, traces)
    traces = deepcopy(original)
    traces[0][close.FIELD]['enabled'] = 1
    with pytest.raises(ValueError, match='configuration'):verify(value, [row], owners, traces)
