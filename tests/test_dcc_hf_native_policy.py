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
    for char in map(chr, range(128)):
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
    with pytest.raises(ValueError, match='CUAD schemas only'):
        value._generate_batch([messages()], {'type': 'object'})
    assert not value.model.calls
