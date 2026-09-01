"""Strict OpenAI-compatible Qwen2.5-3B answer judge client."""

from __future__ import annotations

import dataclasses
import json
import urllib.error
import urllib.request
from typing import Any

from .metrics import normalize_answer

_ANCHOR_NEGATIONS = frozenset({"not", "never", "no", "neither", "wrong", "incorrect", "false", "instead"})


@dataclasses.dataclass(frozen=True, slots=True)
class JudgeResult:
    correct: bool
    reason: str


def reference_anchored_match(prediction: str, references: list[str]) -> bool:
    """Accept deterministic exact aliases without asking a fallible small judge."""

    prediction_tokens = normalize_answer(prediction).split()
    for reference in references:
        reference_tokens = normalize_answer(reference).split()
        if not reference_tokens:
            continue
        if prediction_tokens == reference_tokens:
            return True
        if len(reference_tokens) < 2:
            continue
        width = len(reference_tokens)
        for index in range(len(prediction_tokens) - width + 1):
            if prediction_tokens[index : index + width] != reference_tokens:
                continue
            nearby = prediction_tokens[max(0, index - 3) : index]
            nearby.extend(prediction_tokens[index + width : index + width + 2])
            if not _ANCHOR_NEGATIONS.intersection(nearby):
                return True
    return False


def parse_judge_response(content: str) -> JudgeResult:
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("judge response is not strict JSON") from exc
    if not isinstance(raw, dict) or set(raw) != {"correct", "reason"}:
        raise ValueError("judge response must contain exactly 'correct' and 'reason'")
    if not isinstance(raw["correct"], bool) or not isinstance(raw["reason"], str):
        raise ValueError("judge response has invalid field types")
    return JudgeResult(raw["correct"], raw["reason"].strip())


def parse_judge_decision(content: str) -> JudgeResult:
    """Parse the minimal wire schema and derive consistent audit text."""
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("judge response is not strict JSON") from exc
    if not isinstance(raw, dict) or set(raw) != {"correct"}:
        raise ValueError("judge decision must contain exactly 'correct'")
    if not isinstance(raw["correct"], bool):
        raise ValueError("judge decision has an invalid field type")
    correct = raw["correct"]
    return JudgeResult(correct, "match" if correct else "mismatch")


class JudgeClient:
    PROTOCOL = "strict-binary-v5-deterministic-anchor"

    def __init__(self, base_url: str, *, model: str = "Qwen/Qwen2.5-3B-Instruct", api_key: str = "", timeout: float = 120) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def judge(self, *, question: str, references: list[str], prediction: str) -> JudgeResult:
        if reference_anchored_match(prediction, references):
            return JudgeResult(True, "deterministic_reference_match")
        system = """You are a strict reference-anchored answer evaluator.
Return correct=true only when the candidate's final answer contains the same
answer as at least one reference answer, allowing only harmless differences in
case, punctuation, articles, inflection, common abbreviations, or an obviously
equivalent alias. Return false when the candidate names a different person,
place, title, date, number, entity, or relation; merely discusses the reference
topic; says the answer is unknown; or contains both the reference and a
contradictory final answer. Never infer correctness from the question alone.
When uncertain, return false. Return only one JSON object with exactly this
field: {"correct": true}."""
        user = json.dumps(
            {"question": question, "reference_answers": references, "candidate_answer": prediction},
            ensure_ascii=False,
        )
        payload = {
            "model": self.model,
            "temperature": 0,
            # Bound structured decoding so a malformed answer cannot make the
            # local judge emit whitespace until the endpoint timeout.
            "max_tokens": 16,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "judge_result",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "correct": {"type": "boolean"},
                        },
                        "required": ["correct"],
                        "additionalProperties": False,
                    },
                },
            },
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw: Any = json.load(response)
            content = raw["choices"][0]["message"]["content"]
        except (urllib.error.URLError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"judge request failed: {exc}") from exc
        if not isinstance(content, str):
            raise RuntimeError("judge returned non-string content")
        return parse_judge_decision(content)
