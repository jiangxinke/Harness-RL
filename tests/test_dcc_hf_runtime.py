"""CPU checks for the optional policy runtime and DCC training objectives."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

_POLICY = importlib.import_module("slime_plugins.dcc.hf_policy")
HFPolicy = _POLICY.HFPolicy
completion_logprobs = _POLICY.completion_logprobs
encode_messages = _POLICY.encode_messages

_SPEC = importlib.util.spec_from_file_location(
    "dcc_train_runtime_tests", Path(__file__).resolve().parents[1] / "scripts/dcc_train.py"
)
_TRAIN = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TRAIN)


class CharacterTokenizer:
    """A reversible tokenizer that exposes precisely what fits in context."""

    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, *, mapping_output=False):
        self.mapping_output = mapping_output

    def encode(self, text, *, add_special_tokens=False):
        return [ord(character) + 3 for character in text]

    def decode(self, ids, *, skip_special_tokens=False):
        return "".join(chr(value - 3) for value in ids if value >= 3)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize and not enable_thinking
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        tokens = self.encode(rendered)
        return {"input_ids": tokens} if self.mapping_output else tokens


class PositionLogitModel(torch.nn.Module):
    """Every causal position has independent logits, so shifts are observable."""

    def __init__(self):
        super().__init__()
        values = torch.arange(32 * 17, dtype=torch.float32).reshape(32, 17)
        self.scores = torch.nn.Parameter((values / 13).sin())
        self.last_logits_to_keep = None

    @property
    def device(self):
        return self.scores.device

    def forward(self, *, input_ids, use_cache, logits_to_keep):
        assert not use_cache
        self.last_logits_to_keep = logits_to_keep
        logits = self.scores[: input_ids.shape[1]][-logits_to_keep:]
        return SimpleNamespace(logits=logits.unsqueeze(0))


@pytest.mark.parametrize(
    ("prompt", "completion"),
    [([1], [3]), ([1, 2], [5, 6, 7]), ([2, 7, 8, 4], [3, 1])],
)
def test_completion_logprobs_use_exact_causal_response_positions(prompt, completion):
    model = PositionLogitModel()
    actual = completion_logprobs(model, prompt, completion)
    start = len(prompt) - 1
    stop = start + len(completion)
    all_logprobs = model.scores.detach().log_softmax(dim=-1)
    expected = all_logprobs[start:stop, completion]
    # Index each position with its own next token, rather than a cross product.
    expected = expected.diagonal()

    torch.testing.assert_close(actual, expected)
    assert actual.shape == (len(completion),)
    assert model.last_logits_to_keep == len(completion) + 1

    (-actual.mean()).backward()
    active_rows = model.scores.grad.abs().sum(dim=-1).nonzero().flatten().tolist()
    assert active_rows == list(range(start, stop))
    for position, token in zip(range(start, stop), completion, strict=True):
        assert model.scores.grad[position, token] < 0


@pytest.mark.parametrize("mapping_output", [False, True])
@pytest.mark.parametrize("later_turns", [False, True])
def test_prompt_truncation_preserves_original_question(mapping_output, later_turns):
    tokenizer = CharacterTokenizer(mapping_output=mapping_output)
    system = "Follow the evidence and return the required answer format."
    question = "Which agreement clause describes the original renewal date?"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
        {"role": "user", "content": "FIRST EVIDENCE " + "x" * 1200},
    ]
    if later_turns:
        messages.extend(
            [
                {"role": "assistant", "content": "Search for the renewal clause."},
                {"role": "tool", "content": "y" * 700 + " MOST RECENT EVIDENCE"},
            ]
        )
    encoded = encode_messages(tokenizer, messages, 400)
    visible = tokenizer.decode(encoded)

    assert len(encoded) <= 400
    assert system in visible
    assert question in visible
    assert "MOST RECENT EVIDENCE" in visible if later_turns else "FIRST EVIDENCE" in visible


def test_prompt_budget_refuses_to_drop_a_long_question():
    tokenizer = CharacterTokenizer()
    messages = [
        {"role": "system", "content": "Follow the evidence."},
        {"role": "user", "content": "question " * 60},
        {"role": "tool", "content": "evidence " * 100},
    ]
    with pytest.raises(ValueError, match="System prompt and question"):
        encode_messages(tokenizer, messages, 400)


def test_judge_prompt_never_silently_truncates_evidence():
    tokenizer = CharacterTokenizer()
    messages = [{"role": "user", "content": "evidence " * 100}]
    with pytest.raises(ValueError, match="refusing to truncate evidence"):
        encode_messages(tokenizer, messages, 300, allow_truncation=False)


def _supervised_example(gold, source, *, max_response=1000, benchmark="cuad"):
    tokenizer = CharacterTokenizer()
    messages = [
        {"role": "system", "content": "Answer using the supplied contract evidence."},
        {"role": "user", "content": "What is the contract term?"},
        {"role": "user", "content": source},
    ]
    runner = SimpleNamespace(initial_messages=lambda task: messages)
    task = SimpleNamespace(gold=gold, benchmark=benchmark)
    return _TRAIN.supervised_example(task, runner, tokenizer, 400, max_response), tokenizer


def test_supervised_example_requires_gold_in_actual_encoded_prompt():
    gold = "The term renews annually."
    hidden_source = "Visible introduction. " + "x" * 700 + gold + "y" * 700
    assert gold in hidden_source
    hidden, _ = _supervised_example([gold], hidden_source)
    assert hidden is None

    retained, tokenizer = _supervised_example([gold], gold + "x" * 1500)
    assert retained is not None
    prompt, completion = retained
    assert gold in tokenizer.decode(prompt)
    assert completion[-1] == tokenizer.eos_token_id
    assert json.loads(tokenizer.decode(completion)) == {
        "answer": [{"text": gold, "confidence": 0.9}]
    }


def test_supervised_example_rejects_partially_visible_multiple_gold_spans():
    source = "VISIBLE GOLD " + "x" * 700 + "HIDDEN GOLD" + "y" * 700
    example, _ = _supervised_example(["VISIBLE GOLD", "HIDDEN GOLD"], source)
    assert example is None


def test_supervised_example_accepts_visible_escaped_evidence():
    gold = 'The term is "annual".\nRenewal follows.'
    source = json.dumps({"excerpt": gold}, ensure_ascii=False)
    example, _ = _supervised_example([gold], source)
    assert example is not None


def test_supervised_example_supports_no_answer_and_enforces_response_budget():
    example, tokenizer = _supervised_example([], "No relevant clause exists.")
    assert example is not None
    assert json.loads(tokenizer.decode(example[1])) == {"answer": []}
    too_long, _ = _supervised_example([], "No relevant clause exists.", max_response=1)
    assert too_long is None


def test_supervised_example_skips_structured_web_curriculum_labels():
    gold = [{"name": "Example", "year": "2024"}]
    example, _ = _supervised_example(
        gold, "Example was founded in 2024.", benchmark="widesearch"
    )
    assert example is None


@pytest.mark.parametrize("advantage", [1.0, -1.0])
def test_grpo_advantage_moves_sample_probability_in_correct_direction(advantage):
    logit = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([logit], lr=0.2)
    before = logit.detach().sigmoid().item()
    logprob = torch.nn.functional.logsigmoid(logit).reshape(1)
    old_logprob = logprob.detach().clone()
    loss = _TRAIN.grpo_loss(logprob, old_logprob, advantage)
    loss.backward()

    assert logit.grad.item() * advantage < 0
    assert old_logprob.grad is None
    optimizer.step()
    assert (logit.detach().sigmoid().item() - before) * advantage > 0


@pytest.mark.parametrize(("ratio", "advantage"), [(1.5, 1.0), (0.5, -1.0)])
def test_grpo_clips_updates_past_the_improvement_boundary(ratio, advantage):
    logprob = torch.tensor([ratio], requires_grad=True).log()
    logprob.retain_grad()
    loss = _TRAIN.grpo_loss(logprob, torch.zeros(1), advantage)
    loss.backward()
    torch.testing.assert_close(logprob.grad, torch.zeros(1))


class RecordingGenerationModel:
    device = torch.device("cpu")

    def __init__(self):
        self.generation_config = SimpleNamespace(
            eos_token_id=1,
            repetition_penalty=1.5,
            min_p=0.2,
            top_h=0.5,
            typical_p=0.5,
            no_repeat_ngram_size=2,
            forced_eos_token_id=8,
            suppress_tokens=[9],
            sequence_bias={(10,): 3.0},
            num_beams=3,
        )
        self.options = None

    def eval(self):
        return self

    def generate(self, *, input_ids, **kwargs):
        self.options = kwargs
        completion = torch.tensor([[ord("x") + 3, 1]] * input_ids.shape[0])
        return torch.cat((input_ids, completion), dim=1)


def test_raw_sampling_overrides_inherited_filters_without_mutating_model():
    model = RecordingGenerationModel()
    original = vars(model.generation_config).copy()
    policy = HFPolicy(model, CharacterTokenizer(), raw_sampling=True)
    result = policy.generate([[{"role": "user", "content": "question"}]], 2, 1.0)
    options = model.options
    assert options["do_sample"]
    assert (options["temperature"], options["top_p"], options["top_k"]) == (1.0, 1.0, 0)
    assert options["repetition_penalty"] == 1.0
    assert options["no_repeat_ngram_size"] == 0
    assert options["typical_p"] == 1.0
    assert options["num_beams"] == 1
    assert all(options[key] is None for key in (
        "min_p", "top_h", "forced_eos_token_id", "suppress_tokens", "sequence_bias"
    ))
    assert vars(model.generation_config) == original
    assert result[0].completion_ids == [ord("x") + 3, 1]
    assert result[0].finish_reason == "stop"


@pytest.mark.parametrize("raw_sampling", [False, True])
def test_raw_sampling_preserves_existing_greedy_generation(raw_sampling):
    model = RecordingGenerationModel()
    policy = HFPolicy(model, CharacterTokenizer(), raw_sampling=raw_sampling)
    policy.generate([[{"role": "user", "content": "question"}]], 2, 0.0)
    assert model.options["do_sample"] is False
    assert all(model.options[key] is None for key in ("temperature", "top_p", "top_k"))
    assert "repetition_penalty" not in model.options


def test_default_sampling_keeps_existing_policy_configuration():
    model = RecordingGenerationModel()
    policy = HFPolicy(model, CharacterTokenizer())
    policy.generate([[{"role": "user", "content": "question"}]], 2, 0.8)
    assert (model.options["temperature"], model.options["top_p"], model.options["top_k"]) == (0.8, 0.95, 20)


def test_raw_sampling_rejects_temperature_inconsistent_with_logprobs():
    with pytest.raises(ValueError, match="temperature=1"):
        HFPolicy(None, None, raw_sampling=True).generate([], 2, 0.8)


def test_real_qwen_raw_generation_scores_match_unwarped_logprobs():
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3MoeConfig(
        vocab_size=256, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        intermediate_size=32, moe_intermediate_size=8, num_experts=4,
        num_experts_per_tok=2, bos_token_id=2, eos_token_id=1, pad_token_id=0,
        attention_dropout=0.0,
    )
    model = transformers.Qwen3MoeForCausalLM(config)
    inherited = {
        "temperature": 0.3, "top_p": 0.4, "top_k": 1,
        "typical_p": 0.5, "min_p": 0.2, "top_h": 0.5,
        "epsilon_cutoff": 0.2, "eta_cutoff": 0.2,
        "repetition_penalty": 1.5, "encoder_repetition_penalty": 1.5,
        "no_repeat_ngram_size": 1, "encoder_no_repeat_ngram_size": 1,
        "bad_words_ids": [[12]], "min_length": 100, "min_new_tokens": 10,
        "forced_bos_token_id": 13, "forced_eos_token_id": 14,
        "suppress_tokens": [15], "begin_suppress_tokens": [16],
        "sequence_bias": {(17,): 3.0}, "guidance_scale": 1.5,
        "exponential_decay_length_penalty": (0, 2.0),
        "renormalize_logits": True, "remove_invalid_values": True,
        "num_beams": 2,
    }
    for key, value in inherited.items():
        if hasattr(model.generation_config, key):
            setattr(model.generation_config, key, value)
    observed = []
    generate = model.generate

    def capture_scores(**kwargs):
        kwargs.update(return_dict_in_generate=True, output_scores=True)
        output = generate(**kwargs)
        observed.extend(output.scores)
        return output.sequences

    model.generate = capture_scores
    tokenizer = CharacterTokenizer()
    messages = [{"role": "user", "content": "a"}]
    policy = HFPolicy(model, tokenizer, raw_sampling=True)
    result = policy.generate([messages], 2, 1.0)[0]
    with torch.no_grad():
        prompt = torch.tensor([result.prompt_ids])
        raw_logits = model(input_ids=prompt, use_cache=False).logits[0, -1].float()
        torch.testing.assert_close(observed[0][0], raw_logits)
        logprobs = completion_logprobs(model, result.prompt_ids, result.completion_ids)
        sampled = raw_logits.log_softmax(-1)[result.completion_ids[0]]
        torch.testing.assert_close(logprobs[0], sampled)


def test_policy_close_waits_for_generation_after_request_timeout():
    async def scenario():
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        policy = HFPolicy(None, None, batch_size=1)

        def generate(messages, limit, temperature):
            started.set()
            if not release.wait(timeout=10):
                raise TimeoutError("Test generation was never released")
            finished.set()
            return [object() for _ in messages]

        policy.generate = generate
        request = asyncio.create_task(
            policy([], max_new_tokens=1, temperature=0.0)
        )
        closing = None
        try:
            assert await asyncio.to_thread(started.wait, 5)
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(request, timeout=0.01)
            assert request.cancelled()
            assert not finished.is_set()

            closing = asyncio.create_task(policy.close())
            await asyncio.sleep(0.02)
            assert not closing.done(), "close returned while model generation was active"
            release.set()
            await asyncio.wait_for(closing, timeout=5)
            assert finished.is_set()
            assert policy.worker is None
        finally:
            release.set()
            if closing is not None:
                await closing
            else:
                await policy.close()
            if not request.done():
                request.cancel()
            await asyncio.gather(request, return_exceptions=True)

    asyncio.run(scenario())
