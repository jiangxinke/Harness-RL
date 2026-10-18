"""Keep service outages out of dense policy credit, including caught failures."""

from __future__ import annotations

from functools import wraps
from typing import Any


def record_service_failure(kind: str):
    def decorate(function):
        @wraps(function)
        def wrapped(self, *args, **kwargs):
            try:
                return function(self, *args, **kwargs)
            except Exception as exc:
                failures = getattr(self, "infrastructure_failures", None)
                if failures is None:
                    failures = self.infrastructure_failures = []
                if len(failures) < 16:
                    failures.append({"kind": kind, "detail": repr(exc)[:300]})
                raise

        return wrapped

    return decorate


def invalidate_samples(samples: list[Any], failures: list[dict]) -> None:
    from slime.utils.types import Sample

    for sample in samples:
        sample.remove_sample = True
        sample.status = Sample.Status.ABORTED
        sample.reward = 0.0
        sample.loss_mask = [0] * sample.response_length
        sample.reflect_mask = [0] * sample.response_length
        sample.position_rewards = [0.0] * sample.response_length
        sample.metadata["infrastructure_failure"] = {
            "kind": "loongreflect_service_failure",
            "retryable": True,
            "failures": failures[:16],
        }
