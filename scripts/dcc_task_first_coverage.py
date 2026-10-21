"""Opt-in, fresh-only task-first sampler for validated native SFT.

Full original native loading and replica/source checks remain caller duties.
This class reuses production MixedSampler once for validated encoding and the
original sample_batch packing function. It never rewrites IDs, tokens or labels.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import math
import random


TASK_FIRST_CONFIGURATION = dict(
    version='native-global-domain-polarity-task-first-v1',
    pool_scope='all originally validated native TRAIN examples',
    task_weight='uniform shuffled deck per domain/polarity, before stage or variant',
    category_weight='task-uniform; category-uniform weighting intentionally removed',
    domain_polarity='deterministic weighted-deficit quotas, no missing-polarity fallback',
    stage_weight='final50/search25/read-or-fetch25 within each available task',
    positive_final_first='explicit optional TRAIN curriculum; changes early stage weights',
    packing='original sample_batch, pending retained, consumption separate from reservation',
    fresh_only=True,
    saved_rng_is_not_full_resume_state=True,
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def fraction(value):
    require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
            'Finite explicit mixture fraction in [0,1] required')
    return Fraction(value).limit_denominator(1_000_000)


class _TaskPool:
    def __init__(self, tasks, seed, *, positive_final_first):
        self.rng = random.Random(seed)
        self.tasks = tasks
        self.ids = sorted(tasks)
        self.remaining = []
        self.stage_indices = {task: index for index, task in enumerate(self.ids)}
        self.variant_decks = {}
        self.first = set()
        self.positive_final_first = positive_final_first

    def draw(self):
        if not self.remaining:
            self.remaining = self.ids.copy()
            self.rng.shuffle(self.remaining)
        task = self.remaining.pop()
        stages = self.tasks[task]['stages']
        cycle = tuple(s for s in ('final', 'search', 'final', self.tasks[task]['read_stage']) if s in stages)
        if self.positive_final_first and task not in self.first:
            stage = 'final'
        else:
            stage = cycle[self.stage_indices[task] % len(cycle)]
            self.stage_indices[task] += 1
        self.first.add(task)
        key = task, stage
        deck = self.variant_decks.get(key)
        if not deck:
            deck = stages[stage].copy()
            self.rng.shuffle(deck)
            self.variant_decks[key] = deck
        return deck.pop()


class _LogicalRank:
    def __init__(self, owner, batch_tokens):
        self.owner = owner
        self.pending = None
        self.batch_tokens = batch_tokens

    def sample(self):
        if self.pending is not None:
            example, self.pending = self.pending, None
            return example
        return self.owner._draw()


class GlobalTaskFirstSampler:
    """Identical global CPU planning in each process; only local batch returned.

Use web_fraction=0 for a CUAD-only adapter, 1 for a Web-only adapter, or an
explicit joint fraction for one shared adapter. The original full native loader
still validates both input files before constructing this sampler. Input files,
record order, world, seed and budgets must match on every real process.
"""

    def __init__(self, records, cuad_tasks, tokenizer, runner, *, sampler_class,
                 batch_sampler, rank, world, seed, web_fraction, negative_fraction,
                 positive_final_first=False, require_full_batches=False, **options):
        require(type(world) is int and world > 0 and type(rank) is int and 0 <= rank < world,
                'Typed valid logical rank/world required')
        require(type(seed) is int and seed >= 0, 'Nonnegative typed seed required')
        require(type(positive_final_first) is bool and type(require_full_batches) is bool,
                'Explicit boolean curriculum/full-batch modes required')
        require(not cuad_tasks and options.get('record_encoder') is not None
                and options.get('recorded_cuad_stages') is True,
                'Validated native records only; no raw auxiliary or fallback encoding')
        require(options.get('web_stage_profile') is None, 'This candidate owns its explicit task-stage weighting')
        require(type(options.get('batch_tokens')) is int and options['batch_tokens'] > 0,
                'Positive typed batch token budget required')
        w, n = fraction(web_fraction), fraction(negative_fraction)
        self._weights = {('web', None): w, ('cuad', True): (1 - w) * (1 - n), ('cuad', False): (1 - w) * n}
        self._weights = {key: value for key, value in self._weights.items() if value > 0}
        self._reference = sampler_class(records, cuad_tasks, tokenizer, runner, rank=0, world=1, seed=seed,
            web_fraction=web_fraction, negative_fraction=negative_fraction, **options)
        require(not self._reference.skips, 'Never silently skip an approved native training example')
        tasks = {}
        self._example_bucket = {}
        self._example_stage = {}

        def add(domain, polarity, category, stage, example):
            require(type(example.task_id) is str and bool(example.task_id), 'Explicit native task identity required')
            require(type(example.tokens) is int and 0 < example.tokens <= self._reference.batch_tokens,
                    'Full native example must fit; never truncate or skip it')
            require(stage in (('search', 'fetch', 'final') if domain == 'web' else ('search', 'read', 'final')),
                    'Actual native stage required')
            bucket = domain, polarity
            taskmap = tasks.setdefault(bucket, {})
            task = taskmap.setdefault(example.task_id, dict(category=category, stages={},
                                        read_stage='fetch' if domain == 'web' else 'read'))
            require(task['category'] == category, 'Task has inconsistent TRAIN category')
            task['stages'].setdefault(stage, []).append(example)
            require(id(example) not in self._example_bucket, 'Native Example reused across sampler pools')
            self._example_bucket[id(example)] = bucket
            self._example_stage[id(example)] = stage

        for stage, examples in self._reference.web.items():
            for example in examples:
                add('web', None, example.category, stage, example)
        for polarity, categories in self._reference.cuad_tools.items():
            require(type(polarity) is bool, 'Original explicit CUAD TRAIN polarity required')
            for category, stages in categories.items():
                for stage, examples in stages.items():
                    for example in examples:
                        add('cuad', polarity, category, stage, example)
        owners = {}
        for (domain, polarity), taskmap in tasks.items():
            for task_id, task in taskmap.items():
                require((domain, task_id) not in owners, 'Task appears in both CUAD polarity pools')
                owners[domain, task_id] = polarity
                require('final' in task['stages'], 'Every accepted training task must retain its real native final')
        require(all(key in tasks and tasks[key] for key in self._weights),
                'Requested domain/polarity has zero accepted tasks; fix collection instead of fallback')
        self._pools = {key: _TaskPool(tasks[key], seed + 1_000_003 * (ordinal + 1),
            positive_final_first=positive_final_first and key == ('cuad', True))
            for ordinal, key in enumerate(self._weights)}
        self.rank, self.world = rank, world
        self._batch_sampler = batch_sampler
        self._ranks = [_LogicalRank(self, self._reference.batch_tokens) for _ in range(world)]
        self._require_full_batches = require_full_batches
        self.reserved = Counter()
        self.consumed = Counter()
        self.consumed_tasks = Counter()
        self.consumed_stages = Counter()
        self.steps = 0
        # Existing trainer saves only this RNG; fresh-only remains mandatory.
        self.rng = next(iter(self._pools.values())).rng

    def __getattr__(self, name):
        return getattr(self._reference, name)

    def _draw(self):
        count = sum(self.reserved.values())
        key = max(self._weights, key=lambda k: self._weights[k] * (count + 1) - self.reserved[k])
        example = self._pools[key].draw()
        self.reserved[key] += 1
        return example

    def next_batch(self, max_examples):
        require(type(max_examples) is int and max_examples > 0, 'Positive typed max_examples required')
        batches = [self._batch_sampler(rank, max_examples) for rank in self._ranks]
        if self._require_full_batches:
            require(all(len(batch) == max_examples and rank.pending is None for rank, batch in zip(self._ranks, batches)),
                    'Predeclared full-batch recipe encountered packing deferral')
        for batch in batches:
            require(batch, 'Empty native batch cannot train')
            for example in batch:
                key = self._example_bucket[id(example)]
                self.consumed[key] += 1
                self.consumed_tasks[(*key, example.task_id)] += 1
                self.consumed_stages[(*key, example.task_id, self._example_stage[id(example)])] += 1
        self.steps += 1
        return batches[self.rank]

    def bucket_summary(self):
        return [dict(domain=key[0], positive=key[1], weight=str(self._weights[key]),
                     tasks=len(pool.tasks), records=sum(len(v) for task in pool.tasks.values() for v in task['stages'].values()),
                     category_counts=dict(Counter(task['category'] for task in pool.tasks.values())))
                for key, pool in self._pools.items()]

    def coverage_summary(self):
        return dict(steps=self.steps,
            buckets=[dict(domain=key[0], positive=key[1], reserved=self.reserved[key], consumed=self.consumed[key],
                consumed_unique_tasks=sum((*key, task) in self.consumed_tasks for task in pool.tasks),
                consumed_unique_finals=sum((*key, task, 'final') in self.consumed_stages for task in pool.tasks),
                accepted_tasks=len(pool.tasks)) for key, pool in self._pools.items()],
            pending_examples=sum(rank.pending is not None for rank in self._ranks),
            trained_coverage_uses_consumed_examples=True, GPU_or_model_calls=0)
