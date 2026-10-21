"""Bounded-memory raw behavior scores from actual HF sampled tokens.

Only neutral raw sampling is supported. This processor observes each score
matrix without mutation; it stores one pending vocabulary matrix, then gathers
only the token selected by generate on its next invocation. Finished rows are
trimmed against actual returned tokens, including their first real stop token.
"""
from __future__ import annotations

import math

import torch
from transformers import LogitsProcessor

BEHAVIOR_LOGPROB_SOURCE = "streaming-generation-raw-logits-v1"


class StreamingRawThoughtLogprobs(LogitsProcessor):
    """Observe unwarped raw logits, without retaining a T x vocabulary history."""

    def __init__(self):
        self._pending = None
        self._tokens = []
        self._logprobs = []
        self._prompt_width = None
        self._finished = False

    def _selected(self, token_ids):
        # Clone: a view of input_ids would retain its entire growing storage.
        tokens = token_ids.detach().clone()
        logits = self._pending
        values = logits.gather(1, tokens[:, None]).squeeze(1) - logits.logsumexp(dim=-1)
        self._tokens.append(tokens)
        self._logprobs.append(values)

    def __call__(self, input_ids, scores):
        if self._finished:
            raise ValueError("A finished behavior tracker cannot be reused")
        if input_ids.ndim != 2 or scores.ndim != 2 or input_ids.shape[0] != scores.shape[0]:
            raise ValueError("Aligned batch token IDs and raw vocabulary scores required")
        if self._prompt_width is None:
            self._prompt_width = input_ids.shape[1]
        else:
            if (input_ids.shape[1] != self._prompt_width + len(self._tokens) + 1
                    or scores.shape != self._pending.shape or scores.device != self._pending.device):
                raise ValueError("Behavior tracking requires unchanged non-beam generation rows")
            self._selected(input_ids[:, -1])
        # Keep an independent FP32 copy: subsequent processors must not change
        # the recorded raw distribution. No scalar GPU reads or RNG calls here.
        self._pending = scores.detach().to(dtype=torch.float32, copy=True)
        return scores

    def finish(self, actual_token_rows):
        """Return finite logprobs aligned to first-stop-trimmed actual rows."""
        if self._finished or self._pending is None:
            raise ValueError("Exactly one finish after actual generation is required")
        self._finished = True
        batch, vocab = self._pending.shape
        steps = len(self._tokens) + 1
        if len(actual_token_rows) != batch:
            raise ValueError("Returned behavior batch differs from sampled batch")
        for row in actual_token_rows:
            if (not isinstance(row, list) or not 1 <= len(row) <= steps
                    or any(type(token) is not int or not 0 <= token < vocab for token in row)):
                raise ValueError("Actual sampled token rows and bounded lengths required")
        # Already-finished rows have no final selected token; this placeholder
        # is never retained. Their earlier real EOS/close remains in the row.
        last = torch.tensor([row[steps - 1] if len(row) == steps else 0
                             for row in actual_token_rows], device=self._pending.device)
        self._selected(last)
        ids = torch.stack(self._tokens, dim=1).cpu().tolist()
        values = torch.stack(self._logprobs, dim=1).cpu().tolist()
        self._pending = None
        self._tokens.clear()
        self._logprobs.clear()
        result = []
        for row, tracked, logprobs in zip(actual_token_rows, ids, values, strict=True):
            if tracked[:len(row)] != row:
                raise ValueError("Actual sampled tokens differ from behavior-score token IDs")
            selected = logprobs[:len(row)]
            if any(not math.isfinite(value) for value in selected):
                raise FloatingPointError("Non-finite actual sampled thought logprobs")
            result.append(selected)
        return result
