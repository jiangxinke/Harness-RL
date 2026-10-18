"""Synchronous, batched access to an exclusively owned local HF judge policy.

The evaluator may call this object from several threads. Only its dedicated
worker accesses ``policy.generate``; accepted requests finish before ``close``
returns. Pass a public model identifier for report provenance.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any


@dataclass
class _Request:
    messages: list[dict[str, str]]
    future: Future[str]
    prompt_tokens: int | None = None


class HFBatchedJudge:
    """Adapt ``HFPolicy.generate`` to the evaluator's ``judge(prompt)`` API.

    The token budget estimates padded prompt plus completion tokens across
    the batch. An individual request above that budget runs alone; the policy
    remains responsible for rejecting prompts beyond its context limit.
    Token estimation and generation share one worker to avoid concurrent
    access to mutable tokenizer state. Do not access the same policy model or
    tokenizer elsewhere while this judge is open.
    """

    def __init__(
        self,
        policy: Any,
        tokenizer: Any,
        *,
        model: str,
        max_new_tokens: int = 4096,
        max_batch: int = 4,
        max_wait_seconds: float = 0.02,
        max_batch_tokens: int = 32768,
    ):
        if max_new_tokens < 1 or max_batch < 1 or max_batch_tokens < 1:
            raise ValueError("Token limits and max_batch must be positive")
        if not math.isfinite(max_wait_seconds) or max_wait_seconds < 0:
            raise ValueError("max_wait_seconds must be finite and nonnegative")
        self.policy = policy
        self.tokenizer = tokenizer
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.max_batch = max_batch
        self.max_wait_seconds = max_wait_seconds
        self.max_batch_tokens = max_batch_tokens
        self._queue: queue.Queue[_Request | None] = queue.Queue()
        self._state_lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(
            target=self._work, name="dcc-hf-judge", daemon=True
        )
        self._thread.start()

    def __call__(self, prompt: str) -> str:
        if not isinstance(prompt, str):
            raise TypeError("Judge prompt must be a string")
        messages = [{"role": "user", "content": prompt}]
        future: Future[str] = Future()
        request = _Request(messages, future)
        # Enqueue while holding the close lock so every accepted request is
        # ordered before the shutdown sentinel.
        with self._state_lock:
            if self._closed:
                raise RuntimeError("Judge is closed")
            self._queue.put(request)
        return future.result()

    def close(self) -> None:
        if threading.current_thread() is self._thread:
            raise RuntimeError("Judge cannot close from its generation thread")
        with self._state_lock:
            if not self._closed:
                self._closed = True
                self._queue.put(None)
        self._thread.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _prompt_length(self, request: _Request) -> int:
        if request.prompt_tokens is None:
            tokens = self.tokenizer.apply_chat_template(
                request.messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if hasattr(tokens, "keys"):
                tokens = tokens["input_ids"]
            request.prompt_tokens = len(tokens)
        return request.prompt_tokens

    def _work(self) -> None:
        pending: _Request | None = None
        while True:
            first = pending if pending is not None else self._queue.get()
            pending = None
            if first is None:
                return
            try:
                longest = self._prompt_length(first)
            except Exception as exc:
                first.future.set_exception(exc)
                continue
            batch = [first]
            stop = False
            deadline = time.monotonic() + self.max_wait_seconds
            # Oversized requests run alone instead of consuming the budget
            # of unrelated evaluation calls.
            while (
                len(batch) < self.max_batch
                and longest + self.max_new_tokens <= self.max_batch_tokens
            ):
                try:
                    request = self._queue.get(
                        timeout=max(0.0, deadline - time.monotonic())
                    )
                except queue.Empty:
                    break
                if request is None:
                    stop = True
                    break
                try:
                    prompt_tokens = self._prompt_length(request)
                except Exception as exc:
                    request.future.set_exception(exc)
                    continue
                padded_tokens = (max(longest, prompt_tokens) + self.max_new_tokens) * (
                    len(batch) + 1
                )
                if padded_tokens > self.max_batch_tokens:
                    pending = request
                    break
                batch.append(request)
                longest = max(longest, prompt_tokens)
            try:
                values = self.policy.generate(
                    [request.messages for request in batch], self.max_new_tokens, 0.0
                )
                if len(values) != len(batch):
                    raise RuntimeError("Judge generation returned the wrong batch size")
                # Validate the complete batch before resolving any caller.
                texts = [value.text for value in values]
                if any(not isinstance(text, str) for text in texts):
                    raise TypeError("Judge generation must return text strings")
            except Exception as exc:
                for request in batch:
                    request.future.set_exception(exc)
            else:
                for request, text in zip(batch, texts, strict=True):
                    request.future.set_result(text)
            if stop:
                return
