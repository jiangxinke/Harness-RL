"""Pure invented native metadata and static original sampler AST controls."""
import ast
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import random
import math
import sys

import pytest

HERE = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location('task_first_candidate', HERE / 'dcc_task_first_coverage.py')
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


@dataclass
class Example:
    task_id: str
    stage: str
    category: str
    prompt: tuple
    completion: tuple
    @property
    def tokens(self): return len(self.prompt) + len(self.completion)


def original():
    text = (HERE / 'dcc_tool_sft.py').read_text()
    definitions = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef | ast.ClassDef)
                   and n.name in {'MixedSampler', 'sample_batch'}]
    ns = dict(Counter=Counter, defaultdict=defaultdict, random=random,
              STAGES=('search', 'fetch', 'final'), CUAD_STAGES=('search', 'read', 'final'), encode_record=None)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), '<static-original-sampler>', 'exec'), ns)
    return ns['MixedSampler'], ns['sample_batch']


def inputs(web=24, positive=16, negative=8, variants=1):
    wr, cr = [], []
    for domain, count, polarity in [('web', web, None), ('cuad', positive, True), ('cuad', negative, False)]:
        target = wr if domain == 'web' else cr
        for i in range(count):
            task = f'{domain}-{polarity}-{i:03d}'
            category = 'rare' if i == 0 else 'common'
            for stage in ('search', 'fetch' if domain == 'web' else 'read', 'final'):
                for variant in range(variants if i == 0 else 1):
                    e = Example(task, stage if domain == 'web' else 'cuad_tool_' + stage,
                                category, (1,) * 7, (2,) * 3)
                    target.append(dict(task_id=task, stage=stage, metadata=dict(positive=polarity), example=e))
    return wr, cr


def make(wr, cr, *, rank=0, world=8, seed=47, batch_tokens=40, **kwargs):
    sampler, packing = original()
    defaults = dict(web_fraction=.5, negative_fraction=1/3, positive_final_first=False, require_full_batches=True)
    defaults.update(kwargs)
    return M.GlobalTaskFirstSampler(wr, [], None, None, sampler_class=sampler, batch_sampler=packing,
        rank=rank, world=world, seed=seed, max_prompt=100, max_response=100, batch_tokens=batch_tokens,
        cuad_records=cr, cuad_tool_fraction=1., record_encoder=lambda r, *a, **k: r['example'],
        recorded_cuad_stages=True, **defaults)


def test_exact_global_quotas_and_uniform_tasks():
    wr, cr = inputs(24, 16, 8, variants=20)
    s = make(wr, cr)
    for _ in range(6): s.next_batch(4)
    assert s.consumed == {('web', None): 96, ('cuad', True): 64, ('cuad', False): 32}
    assert set(s.consumed_tasks.values()) == {4}
    assert s.reserved == s.consumed
    assert s.coverage_summary()['pending_examples'] == 0


def test_task_first_removes_category_and_variant_amplification():
    wr, cr = inputs(24, 16, 8, variants=50)
    s = make(wr, cr, web_fraction=0)
    for _ in range(3): s.next_batch(3)
    positive = [n for (d, p, task), n in s.consumed_tasks.items() if p is True]
    negative = [n for (d, p, task), n in s.consumed_tasks.items() if p is False]
    assert positive == [3] * 16 and negative == [3] * 8
    assert ('web', None) not in s.consumed


def test_positive_final_first_consumption_guarantee():
    wr, cr = inputs(96, 64, 32)
    s = make(wr, cr, web_fraction=0, positive_final_first=True, batch_tokens=10)
    for _ in range(12): s.next_batch(1)
    positive = next(x for x in s.coverage_summary()['buckets'] if x['positive'] is True)
    assert positive['consumed'] == positive['consumed_unique_tasks'] == positive['consumed_unique_finals'] == 64
    assert all(s.consumed_stages['cuad', True, f'cuad-True-{i:03d}', 'final'] == 1 for i in range(64))


def test_independent_logical_processes_agree():
    wr, cr = inputs()
    processes = [make(wr, cr, rank=rank) for rank in range(8)]
    for _ in range(20):
        local = [s.next_batch(4) for s in processes]
        assert all(s.reserved == processes[0].reserved and s.consumed_tasks == processes[0].consumed_tasks for s in processes)
        assert len({id(e) for batch in local for e in batch}) > 1


def test_objects_reused_no_native_token_mutation():
    wr, cr = inputs()
    original_objects = {id(r['example']) for r in wr + cr}
    before = [(r['example'].prompt, r['example'].completion) for r in wr + cr]
    s = make(wr, cr)
    for _ in range(20): assert all(id(e) in original_objects for e in s.next_batch(4))
    assert [(r['example'].prompt, r['example'].completion) for r in wr + cr] == before


def test_pending_is_consumption_not_reserved_coverage():
    wr, cr = inputs()
    s = make(wr, cr, batch_tokens=15, require_full_batches=False)
    for _ in range(5): s.next_batch(4)
    assert sum(s.reserved.values()) - sum(s.consumed.values()) == 8
    assert s.coverage_summary()['pending_examples'] == 8
    assert sum(s.consumed.values()) == 40


def test_declared_full_batch_rejects_deferrals():
    wr, cr = inputs()
    s = make(wr, cr, batch_tokens=15)
    with pytest.raises(ValueError, match='packing deferral'): s.next_batch(4)


@pytest.mark.parametrize('kind', ['no-positive', 'no-final', 'polarity-conflict', 'category-conflict'])
def test_missing_or_inconsistent_training_tasks_fail(kind):
    wr, cr = inputs()
    if kind == 'no-positive': cr = [r for r in cr if r['metadata']['positive'] is False]
    elif kind == 'no-final': cr = [r for r in cr if not (r['metadata']['positive'] is True and r['stage'] == 'final')]
    elif kind == 'polarity-conflict':
        for r in cr:
            if r['task_id'] == 'cuad-False-000':
                r['task_id'] = 'cuad-True-000'; r['example'].task_id = 'cuad-True-000'
    else:
        next(r for r in cr if r['stage'] == 'final')['example'].category = 'inconsistent'
    with pytest.raises(ValueError): make(wr, cr)


@pytest.mark.parametrize('value', [-.1, 1.1, float('nan'), float('inf'), True])
def test_bad_fractions_rejected(value):
    wr, cr = inputs()
    with pytest.raises(ValueError): make(wr, cr, web_fraction=value)


def test_task_stages_maintain_longrun_final50():
    wr, cr = inputs(24, 16, 8)
    s = make(wr, cr, web_fraction=0)
    for _ in range(12): s.next_batch(4)
    stages = Counter()
    for (d, p, task, stage), n in s.consumed_stages.items(): stages[stage] += n
    assert stages == {'final': 192, 'search': 96, 'read': 96}


def test_stage_first_coverage_is_not_task_first_coverage():
    # Same category/polarity: task0 has three recorded stages, task1 final only.
    # Existing global per-stage decks therefore allocate 5/6 vs 1/6 task mass.
    wr, cr = inputs(1, 2, 0)
    cr = [r for r in cr if r['task_id'].endswith('000') or r['stage'] == 'final']
    for r in cr: r['example'].category = 'same-category'
    sampler, packing = original()
    spec = importlib.util.spec_from_file_location('original_static_global_stage', HERE / 'dcc_stage_task_coverage.py')
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    legacy_global = old.GlobalStageTaskCoverageSampler(wr, [], None, None,
        sampler_class=sampler, batch_sampler=packing, rank=0, world=1, seed=47,
        web_fraction=0, negative_fraction=0, max_prompt=100, max_response=100, batch_tokens=40,
        cuad_records=cr, cuad_tool_fraction=1., record_encoder=lambda r,*a,**k:r['example'], recorded_cuad_stages=True)
    current = make(wr, cr, world=1, web_fraction=0, negative_fraction=0)
    old_draws, new_draws = Counter(), Counter()
    for _ in range(60):
        old_draws.update(e.task_id for e in legacy_global.next_batch(1))
        new_draws.update(e.task_id for e in current.next_batch(1))
    assert old_draws == {'cuad-True-000': 50, 'cuad-True-001': 10}
    assert new_draws == {'cuad-True-000': 30, 'cuad-True-001': 30}


def test_oversized_native_example_never_skipped():
    wr, cr = inputs()
    cr[0]['example'].completion = (2,) * 100
    with pytest.raises(ValueError, match='never truncate or skip'): make(wr, cr)


@pytest.mark.parametrize('extra', [
    ['--native-task-first-coverage'],
    ['--native-positive-final-first', '--native-thinking'],
    ['--native-task-first-coverage', '--native-thinking', '--adapter', 'synthetic'],
    ['--native-task-first-coverage', '--native-thinking', '--native-global-stage-task-coverage'],
    ['--native-task-first-coverage', '--native-thinking', '--web-stage-profile', 'final50-search25-fetch25'],
])
def test_invalid_training_modes_rejected_before_loading_inputs(monkeypatch, extra):
    node = next(n for n in ast.parse((HERE/'dcc_tool_sft.py').read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'main')
    calls = []
    ns = dict(argparse=argparse, math=math, train=lambda args: calls.append(args))
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<production-cli>', 'exec'), ns)
    monkeypatch.setattr(sys, 'argv', ['synthetic', '--model', 'synthetic', '--records', 'synthetic',
                                     '--output-dir', 'synthetic', *extra])
    with pytest.raises(SystemExit) as error:
        ns['main']()
    assert error.value.code == 2 and not calls
