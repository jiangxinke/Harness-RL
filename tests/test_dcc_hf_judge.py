"""Threaded CPU coverage for the batched local benchmark judge."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from slime_plugins.dcc.hf_judge import HFBatchedJudge


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return {"input_ids": [0] * (len(messages[0]["content"]) + 2)}


class FakePolicy:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls = []
        self.generation_threads = []
        self.failure = None

    def generate(self, messages, max_new_tokens, temperature):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls.append((messages, max_new_tokens, temperature))
            self.generation_threads.append(threading.get_ident())
        try:
            time.sleep(0.01)
            if self.failure:
                raise RuntimeError(self.failure)
            return [
                SimpleNamespace(text="judged:" + group[0]["content"])
                for group in messages
            ]
        finally:
            with self.lock:
                self.active -= 1


def _parallel_calls(judge, prompts):
    barrier = threading.Barrier(len(prompts))

    def call(prompt):
        barrier.wait(timeout=5)
        return judge(prompt)

    with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        futures = [executor.submit(call, prompt) for prompt in prompts]
        return [future.result(timeout=10) for future in futures]


def test_concurrent_judge_calls_batch_without_model_reentry():
    policy = FakePolicy()
    tokenization_threads = []

    class ThreadCheckingTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            tokenization_threads.append(threading.get_ident())
            assert policy.active == 0, "Tokenization overlapped model generation"
            return super().apply_chat_template(messages, **kwargs)

    prompts = [f"prompt {index}" for index in range(8)]
    with HFBatchedJudge(
        policy, ThreadCheckingTokenizer(), model="public/judge", max_wait_seconds=0.05
    ) as judge:
        results = _parallel_calls(judge, prompts)
        assert judge.model == "public/judge"
    assert results == ["judged:" + prompt for prompt in prompts]
    sizes = [len(call[0]) for call in policy.calls]
    assert max(sizes) == 4
    assert sum(sizes) == len(prompts)
    assert policy.max_active == 1
    assert set(tokenization_threads) == set(policy.generation_threads)
    assert len(set(tokenization_threads)) == 1
    assert all(call[1:] == (4096, 0.0) for call in policy.calls)


def test_token_budget_accounts_for_padding_and_isolates_long_prompts():
    policy = FakePolicy()
    prompts = ["s" * 10, "l" * 200, "m" * 60, "n" * 60]
    with HFBatchedJudge(
        policy,
        FakeTokenizer(),
        model="public/judge",
        max_new_tokens=10,
        max_batch_tokens=150,
        max_wait_seconds=0.05,
    ) as judge:
        assert _parallel_calls(judge, prompts) == [
            "judged:" + prompt for prompt in prompts
        ]
    for messages, completion_tokens, _ in policy.calls:
        lengths = [len(group[0]["content"]) + 2 for group in messages]
        padded_tokens = len(messages) * (max(lengths) + completion_tokens)
        assert len(messages) == 1 or padded_tokens <= 150
        if any(length > 150 for length in lengths):
            assert len(messages) == 1
    assert policy.max_active == 1


def test_generation_exception_reaches_every_waiting_future_and_worker_recovers():
    policy = FakePolicy()
    policy.failure = "generation failed"
    with HFBatchedJudge(
        policy, FakeTokenizer(), model="public/judge", max_wait_seconds=0.05
    ) as judge:
        barrier = threading.Barrier(4)

        def failing_call(index):
            barrier.wait(timeout=5)
            with pytest.raises(RuntimeError, match="generation failed"):
                judge(str(index))

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(failing_call, index) for index in range(4)]
            for future in futures:
                future.result(timeout=10)
        policy.failure = None
        assert judge("recovery") == "judged:recovery"
    assert policy.max_active == 1


def test_close_drains_accepted_requests_and_waits_for_generation():
    started, release = threading.Event(), threading.Event()
    second_enqueued = threading.Event()
    close_enqueued = threading.Event()
    policy = FakePolicy()
    original_generate = policy.generate

    def blocked_generate(messages, max_new_tokens, temperature):
        started.set()
        if not release.wait(timeout=10):
            raise TimeoutError("Test generation was not released")
        return original_generate(messages, max_new_tokens, temperature)

    policy.generate = blocked_generate
    judge = HFBatchedJudge(
        policy, FakeTokenizer(), model="public/judge", max_batch=1
    )
    original_put = judge._queue.put

    def observed_put(request, *args, **kwargs):
        original_put(request, *args, **kwargs)
        if request is not None and request.messages[0]["content"] == "queued":
            second_enqueued.set()
        if request is None:
            close_enqueued.set()

    judge._queue.put = observed_put
    with ThreadPoolExecutor(max_workers=3) as executor:
        first = executor.submit(judge, "running")
        closing = None
        try:
            assert started.wait(timeout=5)
            second = executor.submit(judge, "queued")
            assert second_enqueued.wait(timeout=5)
            closing = executor.submit(judge.close)
            assert close_enqueued.wait(timeout=5)
            assert not closing.done()
            with pytest.raises(RuntimeError, match="closed"):
                judge("late request")
            release.set()
            assert first.result(timeout=5) == "judged:running"
            assert second.result(timeout=5) == "judged:queued"
            closing.result(timeout=5)
            assert not judge._thread.is_alive()
            judge.close()
        finally:
            release.set()
            if closing is None:
                judge.close()
            else:
                closing.result(timeout=5)


def test_wrong_output_batch_size_fails_all_callers():
    policy = FakePolicy()
    policy.generate = lambda messages, limit, temperature: []
    with (
        HFBatchedJudge(policy, FakeTokenizer(), model="public/judge") as judge,
        pytest.raises(RuntimeError, match="wrong batch size"),
    ):
        judge("prompt")


def test_tokenization_exception_reaches_caller_and_worker_recovers():
    class FailingTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            if messages[0]["content"] == "invalid":
                raise ValueError("invalid prompt")
            return super().apply_chat_template(messages, **kwargs)

    with HFBatchedJudge(FakePolicy(), FailingTokenizer(), model="public/judge") as judge:
        with pytest.raises(ValueError, match="invalid prompt"):
            judge("invalid")
        assert judge("valid") == "judged:valid"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_batch": 0},
        {"max_new_tokens": 0},
        {"max_batch_tokens": 0},
        {"max_wait_seconds": -1},
        {"max_wait_seconds": float("nan")},
    ],
)
def test_invalid_batch_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        HFBatchedJudge(FakePolicy(), FakeTokenizer(), model="public/judge", **kwargs)
