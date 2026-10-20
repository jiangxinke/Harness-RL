"""Explicit inference sampling; never used for trainable rollout probabilities."""

from __future__ import annotations

import dataclasses
import math


@dataclasses.dataclass(frozen=True, slots=True)
class EvaluationSampling:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0

    def __post_init__(self):
        for name in ("temperature", "top_p", "min_p", "presence_penalty",
                     "frequency_penalty", "repetition_penalty"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"Evaluation {name} must be a finite number")
        if not 0 <= self.temperature <= 2:
            raise ValueError("Evaluation temperature must be between 0 and 2")
        if not 0 < self.top_p <= 1 or not 0 <= self.min_p <= 1:
            raise ValueError("Evaluation top_p must be in (0,1] and min_p in [0,1]")
        if type(self.top_k) is not int or (self.top_k != -1 and self.top_k < 1):
            raise ValueError("Evaluation top_k must be -1 or a positive integer")
        if not -2 <= self.presence_penalty <= 2 or not -2 <= self.frequency_penalty <= 2:
            raise ValueError("Evaluation presence/frequency penalties must be between -2 and 2")
        if self.repetition_penalty <= 0:
            raise ValueError("Evaluation repetition_penalty must be positive")

    def to_dict(self):
        return dataclasses.asdict(self)
