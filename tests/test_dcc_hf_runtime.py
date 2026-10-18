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
