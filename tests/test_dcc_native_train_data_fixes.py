"""CPU regression tests using synthetic TRAIN examples and the real validators.

No weights, benchmark data files, web requests,
CUDA engines, real labels, SSH or staging are used by these tests.
"""
from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest
from unittest import mock

CANDIDATE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(CANDIDATE / 'scripts'), str(CANDIDATE / 'slime')]

import dcc_tool_sft as trainer
import dcc_collect_native_train as collector
from dcc_native_sft import NativeIdentity, NativeSchemaProfile, WebSchemaProfile, CuadSchemaProfile
from slime_plugins.dcc.data import Task
from slime_plugins.dcc.episode import EpisodeRunner, FINAL_TURN
from slime_plugins.dcc.cuad_structured import CUAD_STRUCTURED_SCHEMA_VERSION
from slime_plugins.dcc.widesearch_metrics import evaluate_widesearch

assert Path(trainer.__file__).resolve() == CANDIDATE / 'scripts/dcc_tool_sft.py'
assert Path(collector.__file__).resolve() == CANDIDATE / 'scripts/dcc_collect_native_train.py'

SPAN = 'Alpha agrees to preserve confidential information.'
DOCUMENT = SPAN + ' Beta may terminate on notice.'
URL = 'https://example.org/public'
COMMIT = 'c' * 40  # Synthetic fixture, never an assertion about actual source.


def public_web(columns=None):
    return {'metadata': {'evaluation': {'required': columns or ['name', 'year']}}}


def direct_cuad(*, positive=True):
    task = Task('cuad-fixture', 'cuad', 'train', 'Extract confidentiality clauses.', document=DOCUMENT)
    messages = EpisodeRunner(max_steps=4, initial_chunks=4).initial_messages(task)
    spans = [SPAN] if positive else []
    return {'source': 'cuad_train', 'benchmark': 'cuad', 'split': 'train',
            'task_id': task.id, 'stage': 'final', 'messages': messages,
            'completion': json.dumps({'plan': 'Inspect the public contract.',
                'checks': 'Use only the visible source.',
                'answer': [{'text': span, 'confidence': 0.8} for span in spans]}),
            'grounding_fields': spans,
            'metadata': {'category': 'Confidentiality', 'positive': positive, 'plan_id': 'cpu-fixture'}}


class WebEvaluationTests(unittest.TestCase):
    def test_required_only_gets_explicit_all_column_compound_key_and_exact_rules(self):
        label = {'evaluation': {'required': ['name', 'year']}}
        evaluation = collector.web_train_evaluation(label, public_web())
        self.assertEqual(evaluation, {'required': ['name', 'year'], 'unique_columns': ['name', 'year'],
            'eval_pipeline': {name: {'preprocess': [], 'metric': ['exact_match']} for name in ['name', 'year']}})
        evaluation['required'].append('extra')
        self.assertEqual(label, {'evaluation': {'required': ['name', 'year']}})

    def test_missing_train_evaluation_uses_the_same_public_only_strict_default(self):
        class DoNotReadGold:
            def __iter__(self):
                raise AssertionError('gold was used to choose a schema')
        task = public_web()
        before = copy.deepcopy(task)
        evaluation = collector.web_train_evaluation({'gold': DoNotReadGold()}, task)
        self.assertEqual(evaluation['unique_columns'], ['name', 'year'])
        self.assertEqual(task, before)

    def test_real_scorer_accepts_required_only_train_after_normalization(self):
        evaluation = collector.web_train_evaluation({'evaluation': {'required': ['name', 'year']}}, public_web())
        rows = [{'name': 'Alpha', 'year': '2001'}]
        result = evaluate_widesearch([{**rows[0], 'citations': [URL]}], rows, evaluation)
        self.assertEqual(result['f1_by_item'], 1.0)
        self.assertEqual(result['evaluation_protocol'], 'strict_exact_proxy')

    def test_compound_key_never_collapses_distinct_gold_rows_with_the_same_entity(self):
        evaluation = collector.web_train_evaluation({}, public_web())
        gold = [{'name': 'Alpha', 'year': '2001'}, {'name': 'Alpha', 'year': '2002'}]
        result = evaluate_widesearch(gold[:1], gold, evaluation)
        self.assertEqual((result['num_gold_rows'], result['num_pred_rows']), (2, 1))
        self.assertLess(result['f1_by_item'], 1.0)
        self.assertEqual(evaluate_widesearch(gold, gold, evaluation)['f1_by_item'], 1.0)

    def test_wrong_extra_missing_and_case_changed_cells_fail_perfect_train_threshold(self):
        evaluation = collector.web_train_evaluation({}, public_web())
        gold = [{'name': 'Alpha', 'year': '2001'}]
        for prediction in [[{'name': 'Alpha', 'year': '2002'}],
                           [*gold, {'name': 'Beta', 'year': '2001'}], [],
                           [{'name': 'alpha', 'year': '2001'}]]:
            with self.subTest(prediction=prediction):
                self.assertLess(evaluate_widesearch(prediction, gold, evaluation)['f1_by_item'], 1.0)

    def test_complete_explicit_train_profile_is_unchanged_and_not_mutated(self):
        full = {'required': ['name', 'year'], 'unique_columns': ['name'],
                'eval_pipeline': {'name': {'metric': ['exact_match']},
                                  'year': {'metric': ['number_near'], 'criterion': 0.01}}}
        before = copy.deepcopy(full)
        actual = collector.web_train_evaluation({'evaluation': full}, public_web())
        self.assertEqual(actual, before)
        actual['eval_pipeline']['year']['criterion'] = 0.0
        self.assertEqual(full, before)

    def test_mismatched_or_partially_specified_train_schema_is_not_silently_repaired(self):
        for evaluation in [None, {'required': ['year', 'name']}, {'required': ['name']},
                           {'required': ['name', 'year'], 'unique_columns': ['name']},
                           {'required': ['name', 'year'], 'eval_pipeline': {}}]:
            with self.subTest(evaluation=evaluation), self.assertRaises(ValueError):
                collector.web_train_evaluation({'evaluation': evaluation}, public_web())

    def test_heldout_scorer_still_requires_its_original_full_profile(self):
        with self.assertRaisesRegex(KeyError, 'unique_columns'):
            evaluate_widesearch([{'name': 'Alpha'}], [{'name': 'Alpha'}], {'required': ['name']})


class CuadInitialEvidenceTests(unittest.TestCase):
    def test_real_initial_retrieval_supports_a_direct_grounded_final_without_tools(self):
        record = direct_cuad()
        before = copy.deepcopy(record)
        self.assertEqual(len(record['messages']), 3)
        self.assertFalse(any(message['role'] == 'assistant' for message in record['messages']))
        self.assertIs(trainer.validate_cuad_record(record), record)
        self.assertEqual(record, before)

    def test_real_initial_retrieval_supports_a_direct_negative_without_fabricated_stages(self):
        record = direct_cuad(positive=False)
        self.assertIs(trainer.validate_cuad_record(record), record)
        self.assertEqual(record['grounding_fields'], [])
        self.assertEqual(json.loads(record['completion'])['answer'], [])
        self.assertFalse(any(message['role'] == 'assistant' for message in record['messages']))

    def test_initial_source_must_have_actual_nonempty_offset_consistent_text(self):
        bad = [None, {}, {'error': 'retrieval failed', 'hits': [{'start': 0, 'end': len(SPAN), 'text': SPAN}]},
               {'hits': []}, {'hits': [{'start': 0, 'end': len(SPAN) + 1, 'text': SPAN}]},
               {'hits': [{'start': False, 'end': len(SPAN), 'text': SPAN}]},
               {'hits': [{'start': -1, 'end': len(SPAN) - 1, 'text': SPAN}]},
               {'hits': [{'start': 0, 'end': 0, 'text': ''}]}]
        for observation in bad:
            with self.subTest(observation=observation), self.assertRaisesRegex(ValueError, 'successful historical source'):
                record = direct_cuad()
                record['messages'][2]['content'] = 'Initial contract retrieval (source material):\n' + json.dumps(observation)
                trainer.validate_cuad_record(record)

    def test_without_source_even_a_correctly_declared_or_empty_final_is_rejected(self):
        for positive in (True, False):
            record = direct_cuad(positive=positive)
            record['messages'] = record['messages'][:2]
            with self.subTest(positive=positive), self.assertRaisesRegex(ValueError, 'successful historical source'):
                trainer.validate_cuad_record(record)

    def test_text_not_in_actual_initial_source_remains_rejected(self):
        record = direct_cuad()
        span = 'Invented obligation absent from the public contract.'
        record['grounding_fields'] = [span]
        record['completion'] = json.dumps({'answer': [{'text': span, 'confidence': 0.8}]})
        with self.assertRaisesRegex(ValueError, 'absent from actual source'):
            trainer.validate_cuad_record(record)

    def test_two_initial_hits_cannot_be_joined_to_fabricate_one_span(self):
        record = direct_cuad()
        record['messages'][2]['content'] = 'Initial contract retrieval (source material):\n' + json.dumps(
            {'hits': [{'start': 0, 'end': 5, 'text': 'Alpha'}, {'start': 20, 'end': 26, 'text': 'agrees'}]})
        record['grounding_fields'] = ['Alpha agrees']
        record['completion'] = json.dumps({'answer': [{'text': 'Alpha agrees'}]})
        with self.assertRaisesRegex(ValueError, 'absent from actual source'):
            trainer.validate_cuad_record(record)

    def test_declared_grounding_cannot_hide_a_different_target(self):
        record = direct_cuad()
        record['grounding_fields'] = ['Invented source text']
        with self.assertRaisesRegex(ValueError, 'differs from declared grounding'):
            trainer.validate_cuad_record(record)

    def test_initial_retrieval_cannot_be_repeated_or_inserted_after_a_historical_tool(self):
        for kind in ['repeated', 'late']:
            record = direct_cuad()
            initial = copy.deepcopy(record['messages'][2])
            if kind == 'repeated':
                record['messages'].append(initial)
            else:
                record['messages'][2:] = [
                    {'role': 'assistant', 'content': json.dumps({'tool': 'search_document', 'arguments': {'query': 'public'}})},
                    {'role': 'user', 'content': 'Tool observation (source material):\n' + json.dumps({'hits': []}) + '\nRemaining turns: 2.'},
                    initial]
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'must precede all historical actions'):
                trainer.validate_cuad_record(record)

    def test_initial_hit_still_exposes_the_original_legal_read_windows(self):
        record = direct_cuad()
        observation = json.loads(record['messages'][2]['content'].split('\n', 1)[1])
        hit = observation['hits'][0]
        record['stage'] = 'read'
        record['completion'] = json.dumps({'tool': 'read_document', 'arguments': {'start': hit['start'], 'end': hit['end']}})
        self.assertIs(trainer.validate_cuad_record(record), record)
        record['completion'] = json.dumps({'tool': 'read_document', 'arguments': {'start': hit['start'] + 1, 'end': hit['end']}})
        with self.assertRaisesRegex(ValueError, 'CUAD read must use'):
            trainer.validate_cuad_record(record)

    def test_current_target_and_missing_historical_observation_still_rejected(self):
        record = direct_cuad()
        record['messages'].append({'role': 'assistant', 'content': json.dumps({'tool': 'search_document', 'arguments': {'query': 'public'}})})
        record['messages'].append({'role': 'user', 'content': FINAL_TURN})
        with self.assertRaisesRegex(ValueError, 'Invalid CUAD tool/observation history'):
            trainer.validate_cuad_record(record)


class Tokenizer:
    eos_token_id = 151645
    chat_template = 'CPU fake template; no official tokenizer or model weights'
    tags = {'<|im_start|>': 151644, '<|im_end|>': 151645, '<|endoftext|>': 151643,
            '<think>': 151667, '</think>': 151668}
    def __len__(self):
        return 151669
    def encode(self, text, *, add_special_tokens=False):
        parts = re.split('(' + '|'.join(re.escape(tag) for tag in self.tags) + ')', text)
        return [token for part in parts for token in
                ([self.tags[part]] if part in self.tags else [10000 + ord(char) for char in part])]
    def decode(self, ids, *, skip_special_tokens=False):
        reverse = {value: key for key, value in self.tags.items()}
        return ''.join(reverse[token] if token in reverse else chr(token - 10000)
                       for token in ids if not (skip_special_tokens and token in reverse))
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        from slime_plugins.dcc.native_thinking_eval import NATIVE_SUFFIX
        text = ''.join('<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'
                       for message in messages)
        text += NATIVE_SUFFIX if add_generation_prompt else ''
        return self.encode(text) if tokenize else text


class Engine:
    """Synthetic async token responses; only the vLLM parameter API is real."""
    def __init__(self, tokenizer, values, *, broken=False):
        from vllm.config import StructuredOutputsConfig
        self.vllm_config = types.SimpleNamespace(structured_outputs_config=StructuredOutputsConfig(
            backend='xgrammar', disable_fallback=True))
        self.scripts, self.calls, self.closed = [], [], False
        for value in values:
            self.scripts.extend([tokenizer.encode('Inspect the real public evidence.</think>'),
                                 tokenizer.encode(json.dumps(value)) + [151645]])
        if broken:
            self.scripts[0] = tokenizer.encode('No actual closing token')
    async def generate(self, prompt, params, request_id, *, lora_request):
        self.calls.append((copy.deepcopy(prompt), params, lora_request))
        ids = self.scripts.pop(0)
        output = types.SimpleNamespace(token_ids=ids, text='synthetic engine text',
                                       finish_reason='stop', stop_reason=ids[-1])
        yield types.SimpleNamespace(prompt_token_ids=prompt['prompt_token_ids'], finished=True, outputs=[output])
    async def abort(self, request_id):
        pass
    def shutdown(self):
        self.closed = True


class Web:
    async def web_search(self, **kwargs):
        return {'results': [{'url': URL}]}
    async def fetch_url(self, **kwargs):
        return {'url': URL, 'text': 'Acme is the supported public entity.'}
    def citation_validity(self, answer):
        return 1.0


class NativeCollectorIntegrationTests(unittest.TestCase):
    def setUp(self):
        from slime_plugins.dcc.native_thinking_eval import BASE_SOURCE_COMMIT, PROTOCOL
        self.temp = tempfile.TemporaryDirectory(prefix='dcc-data-fix-cpu-', dir=CANDIDATE.parent)
        self.root = Path(self.temp.name)
        self.tokenizer = Tokenizer()
        self.identity = NativeIdentity('CPU-fixture-derivative-only', (151643, 151645), COMMIT)
        self.receipt = {'version': 'dcc-native-inference-source-v1', 'source_commit': COMMIT,
                        'base_source_commit': BASE_SOURCE_COMMIT, 'protocol': PROTOCOL, 'frozen': True}
        self.profile = {'version': 'dcc-native-train-profile-v1', 'identity': self.identity.manifest(self.tokenizer),
            'web': {'max_steps': 3, 'explicit_review': False, 'cached_rows': False},
            'cuad': {'max_steps': 3, 'structured_actions': True, 'schema_version': CUAD_STRUCTURED_SCHEMA_VERSION}}
        self.schemas = NativeSchemaProfile(WebSchemaProfile(3), CuadSchemaProfile(3, True, CUAD_STRUCTURED_SCHEMA_VERSION))
        self.tasks = [Task('independent-web', 'widesearch', 'train', 'Find the public entity.',
                           metadata={'evaluation': {'required': ['name']}}),
                      Task('cuad-positive', 'cuad', 'train', 'Extract confidentiality clauses.', document=DOCUMENT),
                      Task('cuad-negative', 'cuad', 'train', 'Extract assignment clauses.', document=DOCUMENT)]
    def tearDown(self):
        self.temp.cleanup()
    def labels(self, *, missing_evaluation=False, wrong_web=False):
        values = [{'task_id': 'independent-web', 'benchmark': 'widesearch', 'split': 'train',
                   'gold': [{'name': 'Wrong' if wrong_web else 'Acme'}], 'evaluation': {'required': ['name']}},
                  {'task_id': 'cuad-positive', 'benchmark': 'cuad', 'split': 'train',
                   'gold': [SPAN], 'category': 'Confidentiality'},
                  {'task_id': 'cuad-negative', 'benchmark': 'cuad', 'split': 'train',
                   'gold': [], 'category': 'Assignment'}]
        if missing_evaluation:
            del values[0]['evaluation']
        path = self.root / 'synthetic-train-labels.jsonl'
        path.write_text(''.join(json.dumps(value) + '\n' for value in values))
        return path
    def generate(self, *, broken=False):
        from slime_plugins.dcc.eval_sampling import EvaluationSampling
        from slime_plugins.dcc.native_thinking_eval import NativeThinkingPolicy, NativeTraceJournal
        from slime_plugins.dcc.vllm_policy import AsyncVLLMPolicy
        values = [{'tool': 'web_search', 'arguments': {'query': 'public entity'}},
                  {'tool': 'fetch_url', 'arguments': {'url': URL}},
                  {'answer': [{'name': 'Acme', 'citations': [URL]}]},
                  # Turn-zero schema permits only answer. The plan/checks
                  # fields belong to the actual last-turn schema and must
                  # not be inserted into an early native completion.
                  {'answer': [{'text': SPAN, 'confidence': 0.8}]},
                  {'answer': []}]
        engine = Engine(self.tokenizer, values, broken=broken)
        output = self.root / 'generation'
        output.mkdir()
        base = AsyncVLLMPolicy(engine, self.tokenizer, max_prompt_tokens=self.identity.max_prompt_tokens, max_model_len=self.identity.max_model_len,
            eos_token_ids=[151643, 151645], evaluation_sampling=EvaluationSampling(), allow_truncation=False,
            web_structured_final=True, web_structured_actions=True, provenance={})
        native = NativeThinkingPolicy(base, source_receipt=self.receipt, budget=self.identity.budget,
            trace_sink=NativeTraceJournal(output / 'native_thinking_traces_rank0.jsonl'))
        with mock.patch.object(collector, 'load_train_labels', side_effect=AssertionError('labels reached generation')):
            asyncio.run(collector.generate(self.tasks, native, output, collection_id='cpu-fixture-only',
                profile=self.profile, schema_profile=self.schemas, web_tools_factory=Web))
        self.assertTrue(engine.closed)
        self.assertTrue(native.trace_sink.stream.closed)
        return output, engine
    def select(self, output, labels):
        return collector.select(output, self.root / 'selected', labels_path=labels,
            profile=self.profile, identity=self.identity, schema_profile=self.schemas,
            receipt=self.receipt, tokenizer=self.tokenizer)

    def test_real_native_policy_closed_journals_export_direct_positive_and_negative(self):
        output, engine = self.generate()
        journal_before = {path.name: path.read_bytes() for path in output.iterdir()}
        summary = self.select(output, self.labels())
        self.assertEqual((summary['accepted_episodes'], summary['rejected_episodes'],
                          summary['web_segments'], summary['cuad_segments']), (3, 0, 3, 2))
        self.assertTrue(summary['nonempty_mixed_training_data'])
        self.assertTrue(summary['exported_files_fully_native_validated'])
        self.assertEqual(summary['web_required_only_default'], 'all-required-compound-key-exact-match-v1')
        self.assertEqual(len(engine.calls), 10)
        self.assertEqual(journal_before, {path.name: path.read_bytes() for path in output.iterdir()})
        decisions = list(collector.read_jsonl(self.root / 'selected/decisions.jsonl'))
        self.assertEqual(decisions[0]['train_evaluation']['unique_columns'], ['name'])
        rows = list(collector.read_jsonl(self.root / 'selected/native_cuad_train.jsonl'))
        self.assertEqual([row['record']['stage'] for row in rows], ['final', 'final'])
        self.assertEqual([row['collector']['turn'] for row in rows], [0, 0])
        self.assertFalse(any(message['role'] == 'assistant' for row in rows for message in row['record']['messages']))
        self.assertTrue(all(not row['collector'][key] for row in rows
                            for key in ['synthetic_thought', 'inserted_close', 'inserted_eos']))

    def test_real_selector_missing_evaluation_uses_public_required_without_gold_schema_inference(self):
        output, _ = self.generate()
        summary = self.select(output, self.labels(missing_evaluation=True))
        self.assertEqual((summary['accepted_episodes'], summary['web_segments']), (3, 3))

    def test_original_closed_journal_gate_still_precedes_all_labels_access(self):
        output, _ = self.generate()
        (output / 'generation_closed_rank0.json').unlink()
        with mock.patch.object(collector, 'load_train_labels') as labels:
            with self.assertRaises(FileNotFoundError):
                self.select(output, self.root / 'must-not-open')
            labels.assert_not_called()

    def test_correct_schema_does_not_make_a_wrong_web_answer_eligible(self):
        output, _ = self.generate()
        summary = self.select(output, self.labels(wrong_web=True))
        self.assertEqual((summary['accepted_episodes'], summary['rejected_episodes'], summary['web_segments']), (2, 1, 0))
        decision = next(collector.read_jsonl(self.root / 'selected/decisions.jsonl'))
        self.assertIn('TRAIN correctness threshold not met', decision['reason'])
        self.assertFalse(summary['nonempty_mixed_training_data'])

    def test_tampered_native_prompt_ids_are_rejected_despite_valid_initial_grounding(self):
        output, _ = self.generate()
        trace_path = output / 'native_thinking_traces_rank0.jsonl'
        traces = list(collector.read_jsonl(trace_path))
        positive = next(trace for trace in traces if trace['task_id'] == 'cuad-positive')
        positive['original_prompt_ids'][0] += 1
        # Keep response in agreement so rejection must come from exact public
        # message encoding, rather than only the response/trace join.
        request_path = output / 'requests_rank0.jsonl'
        events = list(collector.read_jsonl(request_path))
        response = next(event for event in events if event['event'] == 'response' and event['task_id'] == 'cuad-positive')
        response['completion']['prompt_ids'] = list(positive['original_prompt_ids'])
        for path, values in [(trace_path, traces), (request_path, events)]:
            path.write_text(''.join(json.dumps(value) + '\n' for value in values))
        summary = self.select(output, self.labels())
        self.assertEqual((summary['accepted_episodes'], summary['rejected_episodes']), (2, 1))
        audit = next(collector.read_jsonl(self.root / 'selected/audit.jsonl'))
        self.assertIn('Recorded prompt differs from exact original message encoding', audit['reason'])


    def literal_labels(self, **kwargs):
        from dcc_train_web_literal_reward import LABEL_PROTOCOL, configuration
        path = self.labels(**kwargs)
        rows = list(collector.read_jsonl(path))
        rows[0].update(label_protocol=LABEL_PROTOCOL, train_reward=configuration())
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return path

    def select_literal(self, output, labels):
        return collector.select(output, self.root / 'selected', labels_path=labels,
            profile=self.profile, identity=self.identity, schema_profile=self.schemas,
            receipt=self.receipt, tokenizer=self.tokenizer,
            web_train_reward_protocol=collector.LITERAL_WEB_TRAIN_REWARD)

    def test_literal_reward_exports_real_native_records_without_benchmark_scorer(self):
        from dcc_train_web_literal_reward import configuration
        output, _ = self.generate()
        before = {path.name: path.read_bytes() for path in output.iterdir()}
        with mock.patch('slime_plugins.dcc.widesearch_metrics.evaluate_widesearch',
                        side_effect=AssertionError('benchmark scorer reached literal TRAIN')):
            summary = self.select_literal(output, self.literal_labels())
        self.assertEqual((summary['accepted_episodes'], summary['web_segments'], summary['cuad_segments']),
                         (3, 3, 2))
        self.assertTrue(summary['exported_files_fully_native_validated'])
        self.assertEqual(summary['web_train_reward'], configuration())
        self.assertEqual(summary['web_selection_score'], collector.LITERAL_WEB_TRAIN_REWARD)
        self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir()})
        decision = next(collector.read_jsonl(self.root / 'selected/decisions.jsonl'))
        self.assertEqual(decision['score_protocol'], collector.LITERAL_WEB_TRAIN_REWARD)
        self.assertEqual(decision['train_reward'], configuration())
        self.assertEqual(decision['score'], 1.0)

    def test_literal_selector_requires_both_explicit_protocol_and_complete_labels(self):
        output, _ = self.generate()
        with self.assertRaises(ValueError):
            self.select(output, self.literal_labels())
        self.assertFalse((self.root / 'selected').exists())
        with self.assertRaises(ValueError):
            self.select_literal(output, self.labels())
        self.assertFalse((self.root / 'selected').exists())
        labels = self.literal_labels()
        rows = list(collector.read_jsonl(labels))
        rows[0]['train_reward'].pop('row_key')
        labels.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        with self.assertRaises(ValueError):
            self.select_literal(output, labels)
        self.assertFalse((self.root / 'selected').exists())

    def test_literal_reward_rejects_wrong_content_after_real_native_generation(self):
        output, _ = self.generate()
        summary = self.select_literal(output, self.literal_labels(wrong_web=True))
        self.assertEqual((summary['accepted_episodes'], summary['rejected_episodes'],
                          summary['web_segments']), (2, 1, 0))
        self.assertFalse(summary['nonempty_mixed_training_data'])


class BudgetedCollectorIntegrationTests(NativeCollectorIntegrationTests):
    def setUp(self):
        super().setUp()
        self.identity = NativeIdentity('CPU-fixture-derivative-only', (151643, 151645), COMMIT,
                                       16384, 36864, 42, 16384, 'thinking')
        self.profile['identity'] = self.identity.manifest(self.tokenizer)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
