"""Replaceable full-episode reward interface and deterministic fixture."""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

from .evaluation.metrics import normalize_answer


@dataclasses.dataclass(frozen=True, slots=True)
class EpisodeOutcome:
    episode_id: str
    prompt: str
    final_answer: str
    label: str | None = None
    status: str = "completed"
    artifacts: tuple[str, ...] = ()
    events: tuple[dict[str, Any], ...] = ()
    verifier: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True, slots=True)
class RewardResult:
    value: float
    components: dict[str, float]
    infrastructure_failure: bool = False
    retryable: bool = False
    detail: str = ""


class EpisodeReward(Protocol):
    def __call__(self, outcome: EpisodeOutcome, config: dict[str, Any]) -> RewardResult: ...


def _references(outcome: EpisodeOutcome) -> tuple[str, ...]:
    """Return ordered, de-duplicated answer aliases for one episode."""

    values: list[Any] = []
    for key in ("answers", "aliases"):
        candidate = outcome.verifier.get(key)
        if isinstance(candidate, str):
            values.append(candidate)
        elif isinstance(candidate, Iterable):
            values.extend(candidate)
    values.extend((outcome.label, outcome.verifier.get("expected")))
    return tuple(dict.fromkeys(str(value).strip() for value in values if value is not None and str(value).strip()))


def _token_scores(prediction: str, reference: str) -> tuple[float, float, float, float]:
    predicted_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    exact = float(bool(reference_tokens) and predicted_tokens == reference_tokens)
    if not predicted_tokens or not reference_tokens:
        return exact, 0.0, 0.0, 0.0
    predicted = Counter(predicted_tokens)
    expected = Counter(reference_tokens)
    overlap = sum(min(predicted[token], count) for token, count in expected.items())
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    f1 = 2 * precision * recall / (precision + recall) if overlap else 0.0
    return exact, precision, recall, f1


def deterministic_reward(outcome: EpisodeOutcome, config: dict[str, Any] | None = None) -> RewardResult:
    config = dict(config or {})
    if outcome.status in {"infrastructure-error", "adapter-error", "oom", "timeout-startup"}:
        return RewardResult(0.0, {"infrastructure": 0.0}, True, True, outcome.status)
    references = _references(outcome)
    scores = [_token_scores(outcome.final_answer, reference) for reference in references]
    exact = max((score[0] for score in scores), default=0.0)
    precision = max((score[1] for score in scores), default=0.0)
    recall = max((score[2] for score in scores), default=0.0)
    token_f1 = max((score[3] for score in scores), default=0.0)
    artifact_required = bool(outcome.verifier.get("artifact_required") or config.get("artifact_required"))
    artifact_ok = float(not artifact_required or bool(outcome.artifacts))
    metric = str(config.get("metric", "exact"))
    if metric not in {"exact", "token_f1", "token_recall", "exact_token_f1"}:
        raise ValueError(f"unsupported deterministic reward metric: {metric!r}")
    exact_weight = float(config.get("exact_weight", 0.5))
    if not 0.0 <= exact_weight <= 1.0:
        raise ValueError("exact_weight must be between 0 and 1")
    metric_value = {
        "exact": exact,
        "token_f1": token_f1,
        "token_recall": recall,
        "exact_token_f1": exact_weight * exact + (1.0 - exact_weight) * token_f1,
    }[metric]
    value = metric_value * artifact_ok
    return RewardResult(
        value,
        {
            "exact": exact,
            "token_precision": precision,
            "token_recall": recall,
            "token_f1": token_f1,
            "artifact": artifact_ok,
            "exact_weight": exact_weight,
        },
    )


def normalize_by_episode(args: Any, samples: list[Any]) -> tuple[list[float], list[float]]:
    """Use slime's GRPO normalizer while counting a fanned-out episode once."""

    from slime.rollout.reward_normalization import normalize_group_rewards_by_rollout

    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    if args.advantage_estimator in {"grpo", "gspo", "cispo", "reinforce_plus_plus_baseline"} and args.rewards_normalization:
        normalized = normalize_group_rewards_by_rollout(
            samples,
            raw_rewards,
            n_samples_per_prompt=args.n_samples_per_prompt,
            normalize_std=(args.advantage_estimator in {"grpo", "gspo", "cispo"} and args.grpo_std_normalization),
        )
        return raw_rewards, normalized
    return raw_rewards, raw_rewards


def load_jsonl_fixture(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or "prompt" not in row:
                raise ValueError(f"invalid task row at line {line_number}")
            rows.append(row)
    return rows
