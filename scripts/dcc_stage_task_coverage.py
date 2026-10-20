"""Opt-in global, category-reweighted task-first sampling for validated native SFT examples.

This is a bounded online planner, not a stored training schedule. It reuses the
trainer's bucket decisions and pending-example packing without copying tokens.
"""

from __future__ import annotations

import copy
import random


STAGE_TASK_COVERAGE_CONFIGURATION = {
    "version": "native-global-stage-task-coverage-v1",
    "pool_scope": "all eligible native TRAIN records on every logical rank",
    "decision_law": "original global domain/polarity/category-availability/actual-stage decisions",
    "selection_bucket": "domain/polarity/actual-stage; category-uniform weights intentionally removed",
    "task_selection": "shuffled task deck per bucket; cycle variants within task",
    "reservation_order": "optimizer step, then logical rank, including pending lookahead",
    "resume": "fresh training only; saved decision RNG is not complete planner state",
}


class _TaskDeck:
    def __init__(self, values, seed):
        self.rng = random.Random(seed)
        self.variants = {}
        for example in values:
            self.variants.setdefault(example.task_id, []).append(example)
        self.tasks = sorted(self.variants)
        self.remaining_tasks = []
        self.remaining_variants = {}

    def draw_replacement(self):
        task = self.rng.choice(self.tasks)
        return self.rng.choice(self.variants[task])

    def draw(self):
        if not self.remaining_tasks:
            self.remaining_tasks = self.tasks.copy()
            self.rng.shuffle(self.remaining_tasks)
        task = self.remaining_tasks.pop()
        remaining = self.remaining_variants.get(task)
        if not remaining:
            # These are references to existing Examples, never token copies.
            remaining = self.variants[task].copy()
            self.rng.shuffle(remaining)
            self.remaining_variants[task] = remaining
        return remaining.pop()


class _DecisionRNG:
    def __init__(self, seed, decks, coverage):
        self.base = random.Random(seed)
        self.decks = decks
        self.coverage = coverage

    def random(self):
        return self.base.random()

    def choice(self, values):
        # Consume exactly the original choice even when coverage replaces it.
        # Thus the global replacement shadow uses the same decision RNG stream.
        selected = self.base.choice(values)
        deck = self.decks.get(id(values))
        if deck is None:
            return selected
        return deck.draw() if self.coverage else deck.draw_replacement()

    def getstate(self):
        return self.base.getstate()


class GlobalStageTaskCoverageSampler:
    """Replay all logical ranks online; return only the caller's packed batch.

    Global pools are encoded once per real process and shared by shallow rank
    copies. State is O(records + tasks + world), independent of training steps.
    No distributed collectives, tokenization or tensor copies occur in planning.

    ``coverage=False`` is a CPU shadow control, not a separate training CLI mode.
    The original category choice still determines stage availability; the output
    category comes from the broad stage deck. Categories are therefore reweighted
    intentionally. Shadow replacement draws a uniform task then uniform variant
    from that same broad bucket with an independent selection RNG.
    All processes must use identical validated records, seed, world and budgets.
    """

    def __init__(self, records, cuad_tasks, tokenizer, runner, *, sampler_class,
                 batch_sampler, rank, world, seed, coverage=True, **options):
        if type(world) is not int or world < 1 or type(rank) is not int or not 0 <= rank < world:
            raise ValueError("Global task coverage requires 0 <= rank < world and world > 0")
        if cuad_tasks or options.get("record_encoder") is None or not options.get("recorded_cuad_stages"):
            raise ValueError("Global task coverage requires validated native records without legacy CUAD tasks")
        if type(coverage) is not bool:
            raise ValueError("coverage must be a boolean")
        reference = sampler_class(records, cuad_tasks, tokenizer, runner,
                                  rank=0, world=1, seed=seed, **options)
        pools, registrations = {}, []
        for stage, values in reference.web.items():
            key = ("web", stage)
            pools.setdefault(key, []).extend(values)
            registrations.append((key, key, values))
        for positive, categories in reference.cuad_tools.items():
            for category, stages in categories.items():
                for stage, values in stages.items():
                    key = ("cuad", positive, stage)
                    pools.setdefault(key, []).extend(values)
                    registrations.append((("cuad", positive, category, stage), key, values))
        self._decks = {key: _TaskDeck(values, seed + 1_000_003 * (ordinal + 1))
                       for ordinal, (key, values) in enumerate(sorted(pools.items()))}
        self._pool_decks = {id(values): self._decks[key] for _, key, values in registrations}
        self._bucket_keys = {id(values): key for _, key, values in registrations}
        self._decision_bucket_keys = {id(values): old_key for old_key, _, values in registrations}
        self._ranks = []
        for logical_rank in range(world):
            sampler = copy.copy(reference)
            sampler.rng = _DecisionRNG(seed + logical_rank, self._pool_decks, coverage)
            sampler.stage_index = sampler.cuad_stage_index = logical_rank
            sampler.pending = None
            self._ranks.append(sampler)
        self.rank = rank
        self._batch_sampler = batch_sampler

    def __getattr__(self, name):
        return getattr(self._ranks[self.rank], name)

    def next_batch(self, max_examples):
        if type(max_examples) is not int or max_examples < 1:
            raise ValueError("max_examples must be a positive integer")
        local_batch = None
        for rank, sampler in enumerate(self._ranks):
            batch = self._batch_sampler(sampler, max_examples)
            if rank == self.rank:
                local_batch = batch
        return local_batch

    def bucket_summary(self):
        return [dict(bucket=list(key), records=sum(map(len, deck.variants.values())),
                     tasks=len(deck.tasks))
                for key, deck in self._decks.items()]
