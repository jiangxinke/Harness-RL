"""Public DCC observations and exact sampled decisions for credit estimators."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

from .episode import CUAD_FINAL_REVIEW, FINAL_TURN


@dataclass(frozen=True)
class CreditVisit:
    episode: int
    step: int
    task_id: str
    observation: str
    next_observation: str
    terminal: bool
    success: bool
    task_reward: float
    is_action_valid: bool
    completion: Any
    prompt_hash: str = ""
    evidence_hash: str = ""


def is_final_control(content: str) -> bool:
    return (content == CUAD_FINAL_REVIEW or content.startswith(FINAL_TURN)
            or content.startswith("[DCC_FINAL_CONTROL]"))


def source_payload(content: str):
    for prefix in ("Initial contract retrieval (source material):\n",
                   "Tool observation (source material):\n"):
        if content.startswith(prefix):
            value, end = json.JSONDecoder().raw_decode(content[len(prefix):])
            suffix = content[len(prefix) + end:]
            if suffix.strip() and not suffix.startswith("\nRemaining turns:"):
                raise ValueError("Unexpected text after a recorded source observation")
            return value
    return None


def public_observation(messages) -> str:
    """Exclude control text and retain the actual available action phase."""
    phase = "final" if messages and is_final_control(messages[-1]["content"]) else "act"
    for message in reversed(messages):
        if message["role"] != "user" or is_final_control(message["content"]):
            continue
        payload = source_payload(message["content"])
        if payload is not None:
            return phase + "\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    raise ValueError("A credit state requires a recorded public source observation")


def decision_prefixes(result):
    positions = [i for i, m in enumerate(result.messages) if m["role"] == "assistant"]
    if len(positions) != len(result.steps):
        raise ValueError("Decision messages and recorded token segments disagree")
    if any(result.messages[position]["content"] != step.text
           for position, step in zip(positions, result.steps, strict=True)):
        raise ValueError("Assistant text differs from its recorded completion")
    return [(result.messages[:i], result.messages[i + 1:positions[j + 1]
            if j + 1 < len(positions) else len(result.messages)])
            for j, i in enumerate(positions)]


def build_credit_visits(results, *, tokenizer=None, reward_scale=10.0):
    """Map complete replayable episodes to binary-success author reward units.

    Native task scores remain in EpisodeResult. Only this training boundary
    binarizes complete success, matching the authors' environment wrappers.
    """
    if not math.isfinite(reward_scale) or reward_scale <= 0:
        raise ValueError("Credit reward scale must be finite and positive")
    visits, excluded = [], []
    for episode, result in enumerate(results):
        if result.benchmark != "cuad" or result.split != "train":
            raise ValueError("This credit adapter accepts CUAD TRAIN episodes only")
        if not math.isfinite(result.reward) or not 0 <= result.reward <= 1:
            raise ValueError("Task reward must be finite and in [0, 1]")
        if not result.trainable_steps or len(result.trainable_steps) != len(result.steps):
            excluded.append(dict(episode=episode, reason="incomplete_replayable_episode"))
            continue
        prefixes = decision_prefixes(result)
        observations = [public_observation(before) for before, _ in prefixes]
        success = result.status == "completed" and result.reward >= 1 - 1e-12
        seen_sources = []
        for step, (completion, (before, after)) in enumerate(zip(result.steps, prefixes, strict=True)):
            if tokenizer is not None:
                from .hf_policy import _chat_tokens
                if _chat_tokens(tokenizer, before) != completion.prompt_ids:
                    raise ValueError("Recorded policy prompt differs from the credit-state prefix")
                if hasattr(tokenizer, "decode") and tokenizer.decode(
                        completion.completion_ids, skip_special_tokens=True) != completion.text:
                    raise ValueError("Recorded completion tokens differ from the assistant text")
            errors = []
            for message in after:
                if message["role"] == "user":
                    payload = source_payload(message["content"])
                    if isinstance(payload, dict) and payload.get("error"):
                        errors.append(payload["error"])
            terminal = step == len(result.steps) - 1
            valid = not errors and (not terminal or result.status == "completed")
            seen_sources = [source_payload(m["content"]) for m in before
                            if m["role"] == "user" and source_payload(m["content"]) is not None]
            visits.append(CreditVisit(
                episode, step, result.task_id, observations[step],
                observations[step + 1] if not terminal else ("terminal:success" if success else "terminal:failure"),
                terminal, success, reward_scale * float(success), valid, completion,
                hashlib.sha256(json.dumps(completion.prompt_ids).encode()).hexdigest(),
                hashlib.sha256(json.dumps(seen_sources, sort_keys=True).encode()).hexdigest(),
            ))
    return visits, {"state_version": "source-payload-and-action-phase-v2",
                    "reward_version": "complete-task-success-binary-v1",
                    "reward_scale": reward_scale, "excluded": excluded}


def unpack_scores(results, visits, scores):
    values = [[] for _ in results]
    for visit, score in zip(visits, scores, strict=True):
        if not math.isfinite(float(score)):
            raise ValueError("Non-finite policy advantage")
        values[visit.episode].append(float(score))
    return values
