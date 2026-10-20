"""Stdlib CPU tests; no model, benchmark split, tokenizer or GPU imports."""

import argparse
import ast
from collections import Counter, defaultdict
from contextlib import redirect_stderr
from dataclasses import dataclass
import importlib.util
import io
import json
import math
from pathlib import Path
import random
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts/dcc_tool_sft.py"
DEFAULT_TRACE = json.loads((ROOT / "tests/fixtures/dcc_default_sampler_trace.json").read_text())


def load_sampler(path):
    source = path.read_text()
    nodes = [n for n in ast.parse(source).body
             if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in ("MixedSampler", "sample_batch")]
    ns = dict(Counter=Counter, defaultdict=defaultdict, random=random,
              STAGES=("search", "fetch", "final"), CUAD_STAGES=("search", "read", "final"), encode_record=None)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns["MixedSampler"], ns.get("sample_batch")


spec = importlib.util.spec_from_file_location("dcc_stage_task_coverage", ROOT / "scripts/dcc_stage_task_coverage.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Planner = module.GlobalStageTaskCoverageSampler
MixedSampler, sample_batch = load_sampler(TRAINER)


@dataclass(frozen=True)
class FakeExample:
    task_id: str
    stage: str
    category: str
    prompt: list
    completion: list
    native_phase_lengths: tuple = (1, 1)

    @property
    def tokens(self):
        return len(self.prompt) + len(self.completion)


def fixture(*, web=True, positives=13, negatives=19, variants=False, varied_lengths=False):
    records, cuad = [], []
    if web:
        for task in range(11):
            for stage in ("search", "fetch", "final"):
                records.append(dict(task_id=f"web-{task:03}", stage=stage, category="web"))
    for positive, count in ((True, positives), (False, negatives)):
        for task in range(count):
            for stage in ("search", "read", "final"):
                for variant in range(3 if variants and task == 0 else 1):
                    cuad.append(dict(task_id=f"cuad-{positive}-{task:03}", stage=stage,
                                     category=f"category-{task % 3}", variant=variant,
                                     metadata=dict(positive=positive)))
    for index, row in enumerate(records + cuad):
        row["example"] = FakeExample(row["task_id"], row["stage"] if row in records else "cuad_tool_" + row["stage"],
                                     row["category"], [index] * (1 + index % 8 if varied_lengths else 2),
                                     [1000 + index, 2000 + index])
    return records, cuad


def encoder(record, tokenizer, *, batch_tokens, skips, **kwargs):
    example = record["example"]
    if example.tokens > batch_tokens:
        skips["budget"] += 1
        return None
    return example


def options(cuad, **overrides):
    result = dict(seed=47, web_fraction=0.5, negative_fraction=1 / 3,
                  max_prompt=100, max_response=100, batch_tokens=100,
                  cuad_records=cuad, cuad_tool_fraction=1.0, recorded_cuad_stages=True,
                  record_encoder=encoder)
    result.update(overrides)
    return result


def planner(records, cuad, *, rank=0, world=4, **overrides):
    coverage = overrides.pop("coverage", True)
    return Planner(records, [], None, None, rank=rank, world=world,
                   sampler_class=MixedSampler, batch_sampler=sample_batch, coverage=coverage,
                   **options(cuad, **overrides))


def signature(examples):
    return [(e.task_id, e.stage, e.category, e.prompt[0], e.completion[0]) for e in examples]


class TraceRNG:
    def __init__(self, delegate, keys, trace):
        self.delegate, self.keys, self.trace = delegate, keys, trace

    def random(self):
        return self.delegate.random()

    def choice(self, values):
        result = self.delegate.choice(values)
        if id(values) in self.keys:
            self.trace.append((self.keys[id(values)], result))
        return result

    def getstate(self):
        return self.delegate.getstate()


def trace(planner):
    result = []
    for sampler in planner._ranks:
        local = []
        sampler.rng = TraceRNG(sampler.rng, planner._decision_bucket_keys, local)
        result.append(local)
    return result


class TaskCoverageTests(unittest.TestCase):
    def test_existing_original_sampler_default_behavior_guard(self):
        original, _ = load_sampler(ROOT / "tests/fixtures/dcc_original_mixed_sampler.snapshot.py")
        records, cuad = fixture(variants=True, varied_lengths=True)
        for rank in range(4):
            common = options(cuad, batch_tokens=17)
            old = original(records, [], None, None, rank=rank, world=4, **common)
            actual = MixedSampler(records, [], None, None, rank=rank, world=4, **common)
            for _ in range(100):
                self.assertEqual(signature(sample_batch(old, 4)), signature(sample_batch(actual, 4)))
                self.assertEqual(old.rng.getstate(), actual.rng.getstate())
                self.assertIs(old.pending, actual.pending)

    def test_default_rank_sequences_match_small_frozen_trace(self):
        records, cuad = fixture(variants=True, varied_lengths=True)
        def compact(examples):
            return [f"{e.task_id}|{e.stage}|{e.prompt[0]}" for e in examples]
        for expected in DEFAULT_TRACE["trace"]:
            rank = expected["rank"]
            common = options(cuad, batch_tokens=17)
            new = MixedSampler(records, [], None, None, rank=rank, world=4, **common)
            actual = [compact(sample_batch(new, 4)) for _ in range(DEFAULT_TRACE["steps"])]
            self.assertEqual(actual, expected["batches"])
            self.assertEqual(compact([new.pending]) if new.pending else [], expected["pending"])
            self.assertEqual(new.stage_index, expected["stage_index"])
            self.assertEqual(new.cuad_stage_index, expected["cuad_stage_index"])

    def test_default_config_exactly_matches_baseline(self):
        train = next(n for n in ast.parse(TRAINER.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "train")
        expression = next(n.value for n in train.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "config" for t in n.targets))
        args = types.SimpleNamespace(native_thinking=False, native_global_stage_task_coverage=False, seed=47,
                                     model="model", output_dir="output", native_phase_normalized_loss=False,
                                     native_completion_json_mixture_loss=False, learning_rate=1e-5)
        actual = eval(compile(ast.Expression(expression), str(TRAINER), "eval"), dict(args=args))
        self.assertEqual(actual, DEFAULT_TRACE["expected_default_config"])

    def test_cli_defaults_and_native_guard(self):
        tree = ast.parse(TRAINER.read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        received = []
        ns = dict(argparse=argparse, math=math, __doc__="test", train=received.append)
        exec(compile(ast.Module(body=[main], type_ignores=[]), "trainer-main", "exec"), ns)
        base = ["trainer", "--model", "model", "--records", "records", "--output-dir", "output"]
        with patch.object(sys, "argv", base):
            ns["main"]()
        self.assertFalse(received[-1].native_global_stage_task_coverage)
        with patch.object(sys, "argv", base + ["--native-global-stage-task-coverage"]), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                ns["main"]()
        self.assertEqual(error.exception.code, 2)
        with patch.object(sys, "argv", base + ["--native-thinking", "--native-global-stage-task-coverage"]):
            ns["main"]()
        self.assertTrue(received[-1].native_global_stage_task_coverage)

    def test_train_native_guard_precedes_gpu_and_preparation(self):
        train = next(n for n in ast.parse(TRAINER.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "train")
        ns = dict(MixedSampler=MixedSampler)
        exec(compile(ast.Module(body=[train], type_ignores=[]), "trainer-train", "exec"), ns)
        with self.assertRaisesRegex(ValueError, "requires --native-thinking"):
            ns["train"](types.SimpleNamespace(native_global_stage_task_coverage=True))

    def test_identical_global_shadow_decision_stream(self):
        records, cuad = fixture(variants=True)
        covered = planner(records, cuad)
        shadow = planner(records, cuad, coverage=False)
        actual, expected = trace(covered), trace(shadow)
        for _ in range(200):
            covered.next_batch(4)
            shadow.next_batch(4)
            for rank in range(4):
                self.assertEqual(covered._ranks[rank].rng.getstate(), shadow._ranks[rank].rng.getstate())
                self.assertEqual(covered._ranks[rank].stage_index, shadow._ranks[rank].stage_index)
                self.assertEqual(covered._ranks[rank].cuad_stage_index, shadow._ranks[rank].cuad_stage_index)
        self.assertEqual([[key for key, _ in rows] for rows in actual],
                         [[key for key, _ in rows] for rows in expected])
        self.assertNotEqual([[e.task_id for _, e in rows] for rows in actual],
                            [[e.task_id for _, e in rows] for rows in expected])
        keys = [key for rows in actual for key, _ in rows]
        self.assertEqual({key[0] for key in keys}, {"web", "cuad"})
        self.assertEqual({key[1] for key in keys if key[0] == "cuad"}, {True, False})
        self.assertEqual({key[2] for key in keys if key[0] == "cuad"}, {"category-0", "category-1", "category-2"})

    def test_final50_profile_preserves_stage_cycle_and_shadow_rng(self):
        records, cuad = fixture()
        stub = types.ModuleType("dcc_review_training")
        stub.WEB_STAGE_PROFILE = "final50-search25-fetch25"
        stub.WEB_STAGE_CYCLE = ("final", "search", "final", "fetch")
        with patch.dict(sys.modules, dcc_review_training=stub):
            covered = planner(records, cuad, web_fraction=1.0, web_stage_profile=stub.WEB_STAGE_PROFILE)
            shadow = planner(records, cuad, web_fraction=1.0, web_stage_profile=stub.WEB_STAGE_PROFILE, coverage=False)
        for rank, (left, right) in enumerate(zip(covered._ranks, shadow._ranks)):
            for index in range(20):
                self.assertEqual(left.sample().stage, stub.WEB_STAGE_CYCLE[(rank + index) % 4])
                right.sample()
                self.assertEqual(left.rng.getstate(), right.rng.getstate())

    def test_each_bucket_covers_tasks_before_repeating_and_cycles_variants(self):
        records, cuad = fixture(variants=True)
        value = planner(records, cuad)
        for deck in value._decks.values():
            for _ in range(4):
                draws = [deck.draw() for _ in deck.tasks]
                self.assertEqual(Counter(e.task_id for e in draws), Counter(deck.tasks))
        deck = module._TaskDeck([FakeExample("many", "final", "c", [], [i]) for i in range(5)] +
                                [FakeExample("one", "final", "c", [], [99])], 47)
        draws = [deck.draw() for _ in range(10)]
        self.assertEqual(Counter(e.task_id for e in draws), {"many": 5, "one": 5})
        self.assertEqual({e.completion[0] for e in draws if e.task_id == "many"}, set(range(5)))

    def test_global_pools_remove_rank_local_positive_restriction(self):
        records, cuad = fixture(web=False)
        value = planner(records, cuad, world=8, web_fraction=0, negative_fraction=0)
        all_positive = {r["task_id"] for r in cuad if r["metadata"]["positive"]}
        for rank, sampler in enumerate(value._ranks):
            observed = {sampler.sample().task_id for _ in range(1500)}
            self.assertEqual(observed, all_positive)
            old = MixedSampler(records, [], None, None, rank=rank, world=8,
                               **options(cuad, web_fraction=0, negative_fraction=0))
            old_ids = {e.task_id for categories in old.cuad_tools.values()
                       for stages in categories.values() for rows in stages.values() for e in rows}
            self.assertLess(len(old_ids & all_positive), len(all_positive))

    def test_independent_process_plans_return_correct_local_slice(self):
        records, cuad = fixture(variants=True, varied_lengths=True)
        copies = [planner(records, cuad, rank=rank, batch_tokens=17) for rank in range(4)]
        for _ in range(50):
            expected = [sample_batch(s, 4) for s in copies[0]._ranks]
            # Advance each independently constructed process through the same global plan.
            local = [expected[0]] + [value.next_batch(4) for value in copies[1:]]
            self.assertEqual([signature(rows) for rows in local], [signature(rows) for rows in expected])
            for other in copies[1:]:
                for left, right in zip(copies[0]._ranks, other._ranks):
                    self.assertEqual(left.rng.getstate(), right.rng.getstate())
                    self.assertIs(left.pending, right.pending)

    def test_encoding_once_per_record_no_tokens_or_examples_copied(self):
        records, cuad = fixture()
        calls = Counter()
        def counted(record, tokenizer, **kwargs):
            calls[id(record)] += 1
            return encoder(record, tokenizer, **kwargs)
        value = planner(records, cuad, world=8, record_encoder=counted)
        self.assertEqual(calls, {id(row): 1 for row in records + cuad})
        for other in value._ranks[1:]:
            self.assertIs(other.web, value._ranks[0].web)
            self.assertIs(other.cuad_tools, value._ranks[0].cuad_tools)
        originals = {id(row["example"]): row["example"] for row in records + cuad}
        for _ in range(80):
            for example in value.next_batch(4):
                self.assertIs(example, originals[id(example)])
                self.assertIs(example.prompt, originals[id(example)].prompt)
                self.assertIs(example.completion, originals[id(example)].completion)
        self.assertEqual(sum(calls.values()), len(records + cuad))

    def test_pending_keeps_reservation_and_consumes_no_new_rng(self):
        records, cuad = fixture(web=False, varied_lengths=True)
        value = planner(records, cuad, world=1, web_fraction=0, batch_tokens=10)
        traced = trace(value)[0]
        for _ in range(200):
            before = len(traced)
            pending = value.pending
            batch = value.next_batch(4)
            if pending is not None:
                self.assertIs(batch[0], pending)
            self.assertLessEqual(sum(e.tokens for e in batch), 10)
            newly_reserved = [e for _, e in traced[before:]]
            consumed_or_pending = batch + ([value.pending] if value.pending is not None else [])
            self.assertEqual([id(e) for e in consumed_or_pending],
                             [id(e) for e in ([pending] if pending is not None else []) + newly_reserved])
        self.assertGreater(len(traced), 0)
        self.assertIsNotNone(value.pending)
        state, count, pending = value.rng.getstate(), len(traced), value.pending
        self.assertIs(value._ranks[0].sample(), pending)
        self.assertEqual(state, value.rng.getstate())
        self.assertEqual(count, len(traced))

    def test_state_bound_independent_of_step_count(self):
        records, cuad = fixture(variants=True)
        value = planner(records, cuad, world=8)
        keys = set(vars(value))
        for _ in range(2000):
            value.next_batch(4)
        self.assertEqual(set(vars(value)), keys)
        for deck in value._decks.values():
            self.assertLessEqual(len(deck.remaining_tasks), len(deck.tasks))
            self.assertLessEqual(len(deck.remaining_variants), len(deck.tasks))
            self.assertLessEqual(sum(map(len, deck.remaining_variants.values())), sum(map(len, deck.variants.values())))

    def test_native_only_and_rank_world_guards(self):
        records, cuad = fixture()
        for changes in (dict(world=0), dict(rank=-1), dict(rank=4), dict(world=True),
                        dict(record_encoder=None), dict(recorded_cuad_stages=False), dict(coverage="yes")):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                planner(records, cuad, **changes)
        with self.assertRaises(ValueError):
            Planner(records, [object()], None, None, sampler_class=MixedSampler, batch_sampler=sample_batch,
                    rank=0, world=1, **options(cuad))
        value = planner(records, cuad)
        for invalid in (0, -1, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                value.next_batch(invalid)

    def test_global_bucket_summary_reports_global_availability(self):
        records, cuad = fixture()
        value = planner(records, cuad, world=8)
        rows = value.bucket_summary()
        self.assertEqual(sum(row["records"] for row in rows), len(records + cuad))
        self.assertEqual({row["tasks"] for row in rows if row["bucket"][0] == "web"}, {11})
        self.assertEqual(value.available, {stage: 11 for stage in ("search", "fetch", "final")})

    def test_categories_are_intentionally_task_reweighted(self):
        records, cuad = fixture(web=False, positives=21, negatives=0)
        cuad = [row for row in cuad if row["stage"] == "final"]
        for index, row in enumerate(cuad):
            category = "large" if index < 20 else "singleton"
            row["category"] = category
            example = row["example"]
            row["example"] = FakeExample(example.task_id, example.stage, category, example.prompt, example.completion)
        value = planner(records, cuad, world=1, web_fraction=0)
        rows = [value._ranks[0].sample() for _ in range(210)]
        self.assertEqual(Counter(e.task_id for e in rows), {row["task_id"]: 10 for row in cuad})
        self.assertEqual(Counter(e.category for e in rows), {"large": 200, "singleton": 10})

    def test_original_global_category_availability_stage_law_is_preserved(self):
        records, cuad = fixture(web=False, variants=True)
        cuad = [row for row in cuad if (row["category"] == "category-0" and row["stage"] == "final")
                or (row["category"] == "category-1" and row["stage"] == "search")
                or row["category"] == "category-2"]
        value = planner(records, cuad, world=1, web_fraction=0)
        events = trace(value)[0]
        reference = MixedSampler(records, [], None, None, rank=0, world=1, **options(cuad, web_fraction=0))
        changed_categories = 0
        for _ in range(1000):
            original = reference.sample()
            actual = value._ranks[0].sample()
            decision, _ = events[-1]
            self.assertEqual(decision, ("cuad", original.task_id.startswith("cuad-True"),
                                        original.category, original.stage.removeprefix("cuad_tool_")))
            self.assertEqual(actual.stage, original.stage)
            self.assertEqual(actual.task_id.startswith("cuad-True"), original.task_id.startswith("cuad-True"))
            self.assertEqual(value.rng.getstate(), reference.rng.getstate())
            changed_categories += actual.category != original.category
        self.assertGreater(changed_categories, 100)

    def test_shadow_replacement_is_task_uniform_despite_record_imbalance(self):
        examples = [FakeExample("many-records", "final", "c", [], [i]) for i in range(20)]
        examples += [FakeExample("one-record", "final", "c", [], [99])]
        deck = module._TaskDeck(examples, 47)
        counts = Counter(deck.draw_replacement().task_id for _ in range(4000))
        self.assertLess(abs(counts["many-records"] - counts["one-record"]), 150)

    def test_reservations_cover_every_task_across_logical_ranks(self):
        records, cuad = fixture(variants=True)
        value = planner(records, cuad, world=8)
        reservations = defaultdict(list)
        for key, deck in value._decks.items():
            draw = deck.draw
            def capture(key=key, draw=draw):
                example = draw()
                reservations[key].append(example.task_id)
                return example
            deck.draw = capture
        for _ in range(150):
            value.next_batch(4)
        for key, deck in value._decks.items():
            size = len(deck.tasks)
            rows = reservations[key]
            self.assertGreaterEqual(len(rows), size)
            for start in range(0, len(rows) - size + 1, size):
                self.assertEqual(set(rows[start:start + size]), set(deck.tasks))

    def test_web_only_and_one_polarity_fallback_match_shadow(self):
        for positives, negatives, web_fraction in ((0, 0, 1), (0, 19, 0), (13, 0, 0)):
            with self.subTest(positives=positives, negatives=negatives):
                records, cuad = fixture(web=web_fraction > 0, positives=positives, negatives=negatives)
                covered = planner(records, cuad, web_fraction=web_fraction)
                shadow = planner(records, cuad, web_fraction=web_fraction, coverage=False)
                actual, expected = trace(covered), trace(shadow)
                for _ in range(30):
                    covered.next_batch(4)
                    shadow.next_batch(4)
                self.assertEqual([[key for key, _ in rows] for rows in actual],
                                 [[key for key, _ in rows] for rows in expected])
                for left, right in zip(covered._ranks, shadow._ranks):
                    self.assertEqual(left.rng.getstate(), right.rng.getstate())

    def test_full_eight_rank_mixed_planning_matches_two_emulated_real_ranks(self):
        records, cuad = fixture(variants=True)
        reference = planner(records, cuad, world=8, rank=0)
        processes = [planner(records, cuad, world=8, rank=rank) for rank in (0, 7)]
        events = trace(reference)
        for _ in range(80):
            expected = [sample_batch(sampler, 4) for sampler in reference._ranks]
            for real_rank, process in zip((0, 7), processes):
                actual = process.next_batch(4)
                self.assertEqual(len(actual), 4)
                self.assertEqual(signature(actual), signature(expected[real_rank]))
                for left, right in zip(reference._ranks, process._ranks):
                    self.assertEqual(left.rng.getstate(), right.rng.getstate())
                    self.assertEqual(left.stage_index, right.stage_index)
                    self.assertEqual(left.cuad_stage_index, right.cuad_stage_index)
                    self.assertIsNone(right.pending)
        for rows in events:
            self.assertEqual({key[0] for key, _ in rows}, {"web", "cuad"})
            self.assertEqual({key[1] for key, _ in rows if key[0] == "cuad"}, {False, True})
            for key, example in rows:
                self.assertEqual(example.stage, key[-1] if key[0] == "web" else "cuad_tool_" + key[-1])
                if key[0] == "cuad":
                    self.assertEqual(example.task_id.startswith("cuad-True"), key[1])


if __name__ == "__main__":
    unittest.main()
