"""CPU native-HF boundaries; invented completions and real xgrammar/HF APIs."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
import re
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'slime'))

import pytest
import torch
from transformers import GenerationConfig

xgr = pytest.importorskip('xgrammar')

from slime_plugins.dcc import EpisodeRunner
from slime_plugins.dcc.cuad_structured import cuad_backend_schema, cuad_final_schema
from slime_plugins.dcc.data import Task
from slime_plugins.dcc.episode import ModelProtocolFailure
from slime_plugins.dcc.hf_native_policy import END_THINK, NativeHFCompletion, NativeHFPolicy
from slime_plugins.dcc.hf_policy import encode_messages
from slime_plugins.dcc.prompt_history import PromptHistoryBudgetExceeded


class Tokenizer:
    eos_token_id, pad_token_id = 151645, 0
    tags = {'<|im_start|>': 151644, '<|im_end|>': 151645, '<|endoftext|>': 151643,
            '<think>': 151667, '</think>': 151668}

    def encode(self, text, *, add_special_tokens=False):
        return [token for part in re.split('(' + '|'.join(map(re.escape, self.tags)) + ')', text)
                for token in ([self.tags[part]] if part in self.tags else [10000 + ord(char) for char in part])]

    def decode(self, ids, *, skip_special_tokens=False):
        inverse = {value: key for key, value in self.tags.items()}
        return ''.join(inverse[token] if token in inverse else chr(token - 10000)
                       for token in ids if not (skip_special_tokens and token in inverse))

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        text = ''.join('<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'
                       for message in messages)
        text += '<|im_start|>assistant\n<think>\n' if add_generation_prompt else ''
        return self.encode(text) if tokenize else text


class ScriptedModel:
    """Only the control-flow fixture is scripted; grammar APIs remain real."""
    device = torch.device('cpu')
    config = SimpleNamespace(vocab_size=151670, max_position_embeddings=4096)

    def __init__(self, thoughts, finals):
        self.thoughts, self.finals = thoughts, finals
        self.calls = []
        self.generation_config = GenerationConfig(eos_token_id=[151645, 151643], repetition_penalty=3.0,
            forced_eos_token_id=151645, min_new_tokens=900, top_p=0.4, top_k=2)

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        thought = END_THINK in kwargs['generation_config'].eos_token_id
        rows = self.thoughts if thought else self.finals
        assert len(rows) == len(kwargs['input_ids'])
        width = max(map(len, rows))
        return torch.cat((kwargs['input_ids'], torch.tensor([row + [0] * (width - len(row)) for row in rows])), dim=1)


@pytest.fixture
def tokenizer(monkeypatch):
    tokenizer = Tokenizer()
    # Explicit raw character metadata for this invented tokenizer only. The
    # separate receipt exercises from_huggingface on the actual local tokenizer.
    vocab = [''] * 151670
    for char in list(map(chr, range(128))) + list("中文名称指标值😀"):
        vocab[10000 + ord(char)] = char
    for text, index in tokenizer.tags.items():
        vocab[index] = text
    def info(value, *, vocab_size, stop_token_ids):
        assert value is tokenizer and vocab_size == len(vocab)
        return xgr.TokenizerInfo(vocab, xgr.VocabType.RAW, vocab_size=vocab_size,
                                 stop_token_ids=stop_token_ids)
    monkeypatch.setattr(xgr.TokenizerInfo, 'from_huggingface', staticmethod(info))
    return tokenizer


def messages(content='invented CPU question'):
    return [{'role': 'system', 'content': 'Use the supplied source.'}, {'role': 'user', 'content': content}]


def final_ids(tokenizer, *, text=None, eos=True):
    text = text or '{"plan":"Read source","checks":"Complete","answer":[]}'
    return tokenizer.encode(text) + ([tokenizer.eos_token_id] if eos else [])


def policy(tokenizer, thoughts=None, finals=None, **kwargs):
    model = ScriptedModel(thoughts or [[10065, END_THINK]], finals or [final_ids(tokenizer)])
    return NativeHFPolicy(model, tokenizer, thought_tokens=4, json_tokens=256,
                          max_prompt_tokens=2048, **kwargs)


def test_raw_two_phase_ids_and_neutral_generation_options(tokenizer):
    value = policy(tokenizer)
    original = encode_messages(tokenizer, messages(), 2048, allow_truncation=False)
    result = value._generate_batch([messages()], cuad_final_schema())[0]
    assert isinstance(result, NativeHFCompletion)
    assert result.prompt_ids == original
    assert result.thought_token_count == 2
    assert result.completion_ids == [10065, END_THINK] + final_ids(tokenizer)
    assert result.phase_lengths == (2, len(final_ids(tokenizer)))
    assert json.loads(result.text)['answer'] == []
    thought, final = value.model.calls
    assert thought['input_ids'][0].tolist() == original
    assert final['input_ids'][0].tolist() == original + [10065, END_THINK]
    for call in value.model.calls:
        cfg = call['generation_config']
        assert call['use_model_defaults'] is False
        assert cfg.repetition_penalty == 1 and cfg.forced_eos_token_id is None
        assert cfg.min_new_tokens is None and cfg.min_length == 0
    assert thought['generation_config'].do_sample is True
    assert thought['generation_config'].temperature == 1
    assert thought['generation_config'].top_p == 1 and thought['generation_config'].top_k == 0
    assert not thought['logits_processor']
    assert final['generation_config'].do_sample is False and len(final['logits_processor']) == 1
    assert END_THINK not in final['generation_config'].eos_token_id
    assert asdict(result)['thought_token_count'] == 2


@pytest.mark.parametrize('tokens', [[10065, 10066, 10067, 10068], [10065, 151645]])
def test_truncated_or_early_eos_thought_preserves_sampled_ids_and_zero_json(tokenizer, tokens):
    value = policy(tokenizer, thoughts=[tokens])
    failure = value._generate_batch([messages()], cuad_final_schema())[0]
    assert isinstance(failure, ModelProtocolFailure)
    assert failure.reason == 'thought_did_not_stop_on_real_close'
    assert failure.completion.completion_ids == tokens
    assert failure.completion.thought_token_count == len(tokens)
    assert failure.completion.finish_reason == ('stop' if tokens[-1] in value.eos_ids else 'length')
    assert len(value.model.calls) == 1


@pytest.mark.parametrize('text,eos,reason', [
    ('{"plan":"Read","checks":"Good","answer":[]}', False, 'json_did_not_finish_with_real_eos'),
    ('{"plan":"Read","plan":"Again","checks":"Good","answer":[]}', True, 'json_not_complete_or_schema_valid'),
    ('{"plan":" ","checks":"Good","answer":[]}', True, 'json_not_complete_or_schema_valid'),
])
def test_json_failures_keep_real_thought_and_full_observed_json(tokenizer, text, eos, reason):
    final = final_ids(tokenizer, text=text, eos=eos)
    value = policy(tokenizer, finals=[final])
    failure = value._generate_batch([messages()], cuad_final_schema())[0]
    assert isinstance(failure, ModelProtocolFailure) and failure.reason == reason
    assert failure.completion.text == '' and failure.completion.thought_token_count == 2
    assert failure.completion.completion_ids == [10065, END_THINK] + final


def test_mixed_batch_continues_only_closed_thoughts_and_keeps_ownership(tokenizer):
    traces = []
    value = policy(tokenizer, thoughts=[[10065, END_THINK], [10066, 10067]],
                   finals=[final_ids(tokenizer)], trace_sink=traces.append)
    results = value._generate_batch([messages('A'), messages('Longer B')], cuad_final_schema(), ['A', 'B'])
    assert isinstance(results[0], NativeHFCompletion) and isinstance(results[1], ModelProtocolFailure)
    assert value.model.calls[0]['input_ids'].shape[0] == 2
    assert value.model.calls[1]['input_ids'].shape[0] == 1
    assert [trace['task_id'] for trace in traces] == ['A', 'B']
    assert traces[1]['thought_ids'] == [10066, 10067] and traces[1]['json_ids'] == []


def test_real_grammar_masks_illegal_tokens_and_accepts_eos_after_complete_json(tokenizer):
    from xgrammar.contrib.hf import LogitsProcessor
    value = policy(tokenizer)
    compiled = value.compiler.compile_json_schema(cuad_backend_schema(cuad_final_schema()))
    processor = LogitsProcessor(compiled)
    prefix = torch.tensor([[10065]])
    tokens = final_ids(tokenizer)
    for token in tokens:
        scores = torch.zeros((1, 151670))
        masked = processor(prefix, scores)
        assert torch.isfinite(masked[0, token])
        if prefix.shape[1] == 1:
            assert torch.isneginf(masked[0, tokenizer.eos_token_id])
            assert torch.isneginf(masked[0, 10000 + ord('x')])
        prefix = torch.cat((prefix, torch.tensor([[token]])), dim=1)


@pytest.mark.parametrize('profile,do_sample,temperature,top_p,top_k', [
    ('greedy', False, 1.0, 1.0, 0), ('thinking', True, 0.6, 0.95, 20),
])
def test_eval_profiles_are_explicit_and_json_stays_greedy(tokenizer, profile, do_sample, temperature, top_p, top_k):
    value = policy(tokenizer, sampling_profile=profile)
    value._generate_batch([messages()], cuad_final_schema())
    thought, final = (call['generation_config'] for call in value.model.calls)
    assert (thought.do_sample, thought.temperature, thought.top_p, thought.top_k) == (do_sample, temperature, top_p, top_k)
    assert final.do_sample is False


def test_episode_runner_retains_failed_thought_and_policy_close_is_reusable(tokenizer):
    # The invented tokenizer uses one token per character; a real full harness
    # prompt therefore needs a larger fixture budget than our short unit prompts.
    value = NativeHFPolicy(ScriptedModel([[10065, 10066]], [final_ids(tokenizer)]), tokenizer,
                           thought_tokens=4, json_tokens=256, max_prompt_tokens=12000,
                           max_model_len=16384)
    task = Task('CPU-only', 'cuad', 'train', 'Extract a clause.', 'Invented source clause.', ['Invented source clause.'])
    runner = EpisodeRunner(max_steps=1, max_new_tokens=260, temperature=1.0,
                           evaluation_temperature=1.0, cuad_structured_actions=True)
    async def run_twice():
        first = await runner.run(task, value.bind_task(task.id), evaluation=True)
        await value.close()
        second = await runner.run(task, value.bind_task(task.id), evaluation=True)
        await value.close()
        return first, second
    for result in asyncio.run(run_twice()):
        assert result.status == 'model_protocol_failure' and result.reward == 0
        assert result.trainable_steps == []  # New trainer must deliberately use steps.
        assert result.steps[0].thought_token_count == 2
        assert result.steps[0].completion_ids == [10065, 10066]
    assert value.worker is None


def test_context_failure_occurs_before_sampling_and_keeps_other_requests(tokenizer):
    value = policy(tokenizer)
    results = value._generate_batch([messages('x' * 3000), messages()], cuad_final_schema())
    assert isinstance(results[0], PromptHistoryBudgetExceeded)
    assert isinstance(results[1], NativeHFCompletion)
    assert value.model.calls[0]['input_ids'].shape[0] == 1


def test_actual_transformers_generate_uses_xgrammar_processor_and_ignores_inherited_penalties(tokenizer):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    config = Qwen3MoeConfig(vocab_size=151670, hidden_size=8, intermediate_size=12,
        moe_intermediate_size=4, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=4, num_experts=4, num_experts_per_tok=2,
        attention_dropout=0.0, max_position_embeddings=4096)
    model = Qwen3MoeForCausalLM(config).eval()
    model.generation_config = GenerationConfig(eos_token_id=[151645, 151643],
        repetition_penalty=9.0, forced_eos_token_id=151645, min_new_tokens=999)
    value = NativeHFPolicy(model, tokenizer, thought_tokens=4, json_tokens=4, max_prompt_tokens=2048)
    compiled = value.compiler.compile_json_schema({'const': 'X'})
    prepared = []
    original = model._prepare_generation_config
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        prepared.append(result[0])
        return result
    model._prepare_generation_config = capture
    with torch.inference_mode():
        rows = value._phase([[10065, END_THINK]], phase='json', grammar=compiled)
    assert 1 <= len(rows[0]) <= 4
    assert prepared[0].forced_eos_token_id is None and prepared[0].min_new_tokens is None
    assert prepared[0].repetition_penalty == 1.0
    # Exercise actual raw HF sampling, rather than only inspecting forwarded
    # kwargs. Inherited forced EOS and minimum length must remain disabled.
    with torch.no_grad():
        model.lm_head.weight.zero_()
    torch.manual_seed(71)
    with torch.inference_mode():
        raw = value._phase([[10065]], phase='thought')
    assert len(raw[0]) == 4 and raw[0][-1] not in value.eos_ids
    assert prepared[-1].do_sample and prepared[-1].temperature == 1
    assert prepared[-1].top_p == 1 and prepared[-1].top_k == 0
    assert prepared[-1].forced_eos_token_id is None and prepared[-1].min_new_tokens is None
    # Concentrate the actual model distribution at a real stopping ID. No
    # generation logits processor injects or appends this closing token.
    model.lm_head.bias = torch.nn.Parameter(torch.full((config.vocab_size,), -10000.0), requires_grad=False)
    for stop in (END_THINK, tokenizer.eos_token_id):
        with torch.no_grad():
            model.lm_head.bias.fill_(-10000.0)
            model.lm_head.bias[stop] = 0
        with torch.inference_mode():
            stopped = value._phase([[10065]], phase='thought')
        assert stopped == [[stop]]
    assert not torch.cuda.is_initialized()


def test_non_cuad_schema_is_rejected_before_model_access(tokenizer):
    value = policy(tokenizer)
    with pytest.raises(ValueError, match='CUAD/Web schemas only'):
        value._generate_batch([messages()], {'type': 'object'})
    assert not value.model.calls

# Web backend coverage exercises the public generators and the actual pinned
# xgrammar compiler; no benchmark data, network or checkpoint is accessed.
from copy import deepcopy
from slime_plugins.dcc.hf_native_structured import (
    compile_native_structured_schema, native_structured_schema_kind,
)
from slime_plugins.dcc.web_review import web_review_schema
from slime_plugins.dcc.web_structured import web_action_schema, web_final_schema

WEB_URL = 'https://example.org/source?name=%22'
WEB_OTHER = 'https://example.org/other'


def web_dump(value):
    # Pinned xgrammar keeps property-key literals but escapes slash in enum
    # values. Preserve exact public keys while spelling URL values as it does.
    if isinstance(value, dict):
        return '{' + ','.join(json.dumps(key, ensure_ascii=False) + ':' + web_dump(item)
                              for key, item in value.items()) + '}'
    if isinstance(value, list):
        return '[' + ','.join(map(web_dump, value)) + ']'
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).replace('/', '\\/')


def web_wrap(schema, review):
    return web_review_schema(schema) if review else schema


def web_value(value, review):
    return {'plan': 'Inspect public source', 'checks': 'Use read evidence', **value} if review else value


@pytest.fixture(scope='module')
def byte_compiler():
    info = xgr.TokenizerInfo([bytes([i]) for i in range(256)] + [b'<eos>'], stop_token_ids=[256])
    return xgr.GrammarCompiler(info, max_threads=1)


def grammar_accepts(compiled, text):
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    return matcher.accept_string(text) and matcher.is_terminated()


@pytest.mark.parametrize('review', [False, True])
@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('multi', [False, True])
def test_web_admission_roundtrips_every_public_state(review, cached, multi):
    columns = ['中文', 'name/date']
    for urls in ([], [WEB_URL, WEB_OTHER]):
        schemas = [web_final_schema(columns, urls, multi_citation=multi)]
        schemas.extend(web_action_schema(columns, urls, successful_search=searched,
            cached_rows=cached, multi_citation=multi) for searched in (False, True))
        for bare in schemas:
            schema = web_wrap(bare, review)
            snapshot = deepcopy(schema)
            assert native_structured_schema_kind(schema) == ('web-review' if review else 'web')
            assert schema == snapshot


@pytest.mark.parametrize('review', [False, True])
@pytest.mark.parametrize('bounded', [False, True])
def test_actual_web_grammar_exact_escaped_columns_dynamic_read_urls_and_citations(byte_compiler, review, bounded):
    columns = ['中文', 'name/date']
    row = {columns[0]: '中😀', columns[1]: None, 'citations': [WEB_URL, WEB_OTHER]}
    value = web_value({'answer': [row]}, review)
    schema = web_wrap(web_final_schema(columns, [WEB_URL, WEB_OTHER], multi_citation=True), review)
    compiled = compile_native_structured_schema(byte_compiler, schema, web_bounded_whitespace=bounded)
    assert grammar_accepts(compiled, web_dump(value))
    # The same source row cannot cite a URL that this episode has not read.
    other_schema = web_wrap(web_final_schema(columns, [WEB_URL], multi_citation=True), review)
    assert not grammar_accepts(compile_native_structured_schema(byte_compiler, other_schema,
        web_bounded_whitespace=bounded), web_dump(value))
    for damage in ('empty-cell', 'unread', 'five-citations', 'missing-column'):
        bad = deepcopy(value)
        row = bad['answer'][0]
        if damage == 'empty-cell': row[columns[0]] = ''
        elif damage == 'unread': row['citations'] = ['https://example.org/unread']
        elif damage == 'five-citations': row['citations'] = [WEB_URL] * 5
        else: del row[columns[1]]
        assert not grammar_accepts(compiled, web_dump(bad))
    no_read = compile_native_structured_schema(byte_compiler,
        web_wrap(web_final_schema(columns, []), review), web_bounded_whitespace=bounded)
    assert grammar_accepts(no_read, web_dump(web_value({'answer': []}, review)))
    assert not grammar_accepts(no_read, web_dump(value))


@pytest.mark.parametrize('review', [False, True])
@pytest.mark.parametrize('bounded', [False, True])
def test_actual_web_grammar_search_first_cached_rows_and_legacy_citation_cap(byte_compiler, review, bounded):
    search = web_value({'tool': 'web_search', 'arguments': {'query': 'public source', 'count': 10}}, review)
    fetch = web_value({'tool': 'fetch_url', 'arguments': {'url': WEB_URL, 'select_rows': {
        'format': 'csv', 'columns': ['中文'], 'where': [{'column': '中文', 'op': 'eq', 'value': '中'}],
        'numeric_source_policy': 'exclude_uncomparable',
        'csv_trailing_empty_columns': 'verified_empty_padding', 'limit': 64}}}, review)
    first = compile_native_structured_schema(byte_compiler, web_wrap(web_action_schema(
        ['中文'], [], successful_search=False), review), web_bounded_whitespace=bounded)
    assert grammar_accepts(first, web_dump(search))
    assert not grammar_accepts(first, web_dump(fetch))
    assert not grammar_accepts(first, web_dump(web_value({'answer': []}, review)))
    ready = compile_native_structured_schema(byte_compiler, web_wrap(web_action_schema(
        ['中文'], [WEB_URL], successful_search=True, cached_rows=True), review), web_bounded_whitespace=bounded)
    assert grammar_accepts(ready, web_dump(search)) and grammar_accepts(ready, web_dump(fetch))
    uncached = compile_native_structured_schema(byte_compiler, web_wrap(web_action_schema(
        ['中文'], [WEB_URL], successful_search=True), review), web_bounded_whitespace=bounded)
    assert not grammar_accepts(uncached, web_dump(fetch))
    two = web_value({'answer': [{'中文': '中', 'citations': [WEB_URL, WEB_URL]}]}, review)
    assert not grammar_accepts(ready, web_dump(two))


@pytest.mark.parametrize('bounded', [False, True])
def test_actual_web_review_grammar_preserves_character_and_escape_limits(byte_compiler, bounded):
    schema = web_review_schema(web_final_schema(['name'], []))
    compiled = compile_native_structured_schema(byte_compiler, schema, web_bounded_whitespace=bounded)
    for length in (0, 1, 239, 240, 241):
        value = {'plan': '中' * length, 'checks': 'c', 'answer': []}
        assert grammar_accepts(compiled, web_dump(value)) is (1 <= length <= 240)
    for plan in ('a"b', 'a\\b', 'a\nb'):
        assert not grammar_accepts(compiled, web_dump({'plan': plan, 'checks': 'c', 'answer': []}))
    for count in (8, 9):
        text = '{"plan":"p","checks":"c",' + ' ' * count + '"answer":[]}'
        assert grammar_accepts(compiled, text) is (not bounded or count <= 8)


@pytest.mark.parametrize('damage', ['extra', 'review-bound', 'citation-cap', 'row', 'count', 'boolean-bound'])
def test_modified_web_schemas_are_rejected_before_model_generation(tokenizer, damage):
    schema = web_review_schema(web_action_schema(['name'], [WEB_URL], successful_search=True))
    if damage == 'extra': schema['extra'] = True
    elif damage == 'review-bound': schema['anyOf'][0]['properties']['plan']['maxLength'] = 241
    elif damage == 'citation-cap': schema['anyOf'][-1]['properties']['answer']['items']['anyOf'][0]['properties']['citations']['maxItems'] = 2
    elif damage == 'row': schema['anyOf'][-1]['properties']['answer']['items']['anyOf'][0]['additionalProperties'] = True
    elif damage == 'count': schema['anyOf'][0]['properties']['arguments']['properties']['count']['maximum'] = 11
    else: schema['anyOf'][0]['properties']['arguments']['properties']['count']['minimum'] = True
    value = policy(tokenizer)
    with pytest.raises(ValueError, match='exact public CUAD/Web'):
        value._generate_batch([messages()], schema)
    assert value.model.calls == []


@pytest.mark.parametrize('text', [
    '{"answer":[{"name":"Alpha","citations":["https://example.org/unread"]}]}',
    '{"answer":[{"name":"Alpha","citations":[]}]}',
    '{"answer":[],"answer":[]}',
])
def test_web_original_public_schema_postvalidation_keeps_failure_thoughts(tokenizer, text):
    value = policy(tokenizer, finals=[final_ids(tokenizer, text=text)])
    failure = value._generate_batch([messages()], web_final_schema(['name'], [WEB_URL]))[0]
    assert isinstance(failure, ModelProtocolFailure)
    assert failure.reason == 'json_not_complete_or_schema_valid'
    assert failure.completion.completion_ids == [10065, END_THINK] + final_ids(tokenizer, text=text)


def tiny_real_web_policy(tokenizer, values, *, thoughts=None, bounded=False):
    """Actual tiny HF model/generate and xgrammar, with fixture LM-head logits.

    This controls synthetic token preferences, not generation, EOS, grammar
    matching or EpisodeRunner. It tests runtime plumbing, not model quality.
    """
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    config = Qwen3MoeConfig(vocab_size=151670, hidden_size=8, intermediate_size=12,
        moe_intermediate_size=4, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=4, num_experts=4, num_experts_per_tok=2,
        attention_dropout=0.0, max_position_embeddings=32768)
    model = Qwen3MoeForCausalLM(config).eval()
    model.generation_config = GenerationConfig(eos_token_id=[151645, 151643])
    finals = iter(tokenizer.encode(web_dump(value)) + [tokenizer.eos_token_id] for value in values)
    thought_rows = iter(thoughts) if thoughts is not None else None
    state, calls = {}, []
    original_generate = model.generate
    def fixture_logits(module, args, output):
        cursor = state['cursor']
        target = state['tokens'][min(cursor, len(state['tokens']) - 1)]
        state['cursor'] += 1
        logits = torch.full_like(output, -10000.)
        logits[..., target] = 10000.
        return logits
    model.lm_head.register_forward_hook(fixture_logits)
    def generate(**kwargs):
        thought = END_THINK in kwargs['generation_config'].eos_token_id
        state['tokens'] = (next(thought_rows) if thought_rows is not None else [END_THINK]) if thought else next(finals)
        state['cursor'] = 0
        result = original_generate(**kwargs)
        calls.append({'phase': 'thought' if thought else 'json', 'ids': result[:, kwargs['input_ids'].shape[1]:].tolist()})
        return result
    model.generate = generate
    value = NativeHFPolicy(model, tokenizer, thought_tokens=4, json_tokens=1024,
        max_prompt_tokens=24000, web_bounded_whitespace=bounded)
    return value, calls


@pytest.mark.parametrize('review,bounded', [(False, False), (True, True)])
def test_actual_tiny_hf_web_generation_and_episode_runner_with_mock_tools(tokenizer, review, bounded):
    row = {'中文': '中', 'citations': [WEB_URL, WEB_OTHER]}
    actions = [
        {'tool': 'web_search', 'arguments': {'query': 'synthetic source'}},
        {'tool': 'fetch_url', 'arguments': {'url': WEB_URL}},
        {'tool': 'fetch_url', 'arguments': {'url': WEB_OTHER, 'select_rows': {
            'format': 'csv', 'columns': ['中文'], 'limit': 1}}},
        {'answer': [row]},
    ]
    class Web:
        async def web_search(self, query):
            return {'results': [{'url': WEB_URL, 'title': 'Synthetic public source'}, {'url': WEB_OTHER, 'title': 'Other'}]}
        async def fetch_url(self, url, **kwargs):
            return {'url': url, 'text': 'Synthetic public source says 中', 'start': 0, 'end': 30}
        def citation_validity(self, rows):
            return 1.
    value, calls = tiny_real_web_policy(tokenizer, [web_value(action, review) for action in actions], bounded=bounded)
    task = Task('invented-web', 'widesearch', 'validation', 'Find the observed fixture name.',
        metadata={'evaluation': {'required': ['中文']}})
    async def run():
        runner = EpisodeRunner(max_steps=4, max_new_tokens=1028, evaluation_temperature=1.,
            web_tools_factory=Web, web_structured_final=True, web_structured_actions=True,
            web_explicit_review=review, web_cached_rows=True, web_multi_citation=True)
        try:
            return await runner.run(task, value, evaluation=True)
        finally:
            await value.close()
    result = asyncio.run(run())
    assert result.status == 'completed', result.status
    assert result.answer == [row] and len(result.steps) == 4 and len(result.tool_events) == 3
    assert [call['phase'] for call in calls] == ['thought', 'json'] * 4
    assert all(call['ids'] == [[END_THINK]] for call in calls[::2])
    assert all(call['ids'][0][-1] == tokenizer.eos_token_id for call in calls[1::2])
    assert all(step.thought_token_count == 1 for step in result.steps)
    assert not torch.cuda.is_initialized()


def test_actual_tiny_hf_web_thought_failure_is_retained(tokenizer):
    observed = [10065, 10066, 10067, 10068]
    value, calls = tiny_real_web_policy(tokenizer, [], thoughts=[observed])
    failure = value._generate_batch([messages()], web_final_schema(['name'], []))[0]
    assert isinstance(failure, ModelProtocolFailure)
    assert failure.completion.completion_ids == observed
    assert failure.completion.thought_token_count == 4
    assert [call['phase'] for call in calls] == ['thought']
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('column', ['quoted"column', 'back\\slash'])
@pytest.mark.parametrize('review', [False, True])
def test_pinned_web_backend_rejects_unsupported_escaped_keys_before_sampling(tokenizer, column, review):
    schema = web_wrap(web_final_schema(['中文', column], [WEB_URL]), review)
    snapshot = deepcopy(schema)
    value = policy(tokenizer)
    with pytest.raises(ValueError, match='public column keys and source URLs must remain exact'):
        value._generate_batch([messages()], schema)
    assert schema == snapshot and value.model.calls == []
