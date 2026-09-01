"""Sample contract validation and StackPlanner episode metadata."""

from __future__ import annotations

from typing import Any


def validate_sample(sample: Any, *, require_trainable: bool = False) -> None:
    tokens = list(getattr(sample, "tokens", None) or [])
    response_length = int(getattr(sample, "response_length", 0) or 0)
    mask = list(getattr(sample, "loss_mask", None) or [])
    logprobs = list(getattr(sample, "rollout_log_probs", None) or [])
    if response_length < 0 or response_length > len(tokens):
        raise ValueError(f"invalid response_length {response_length} for {len(tokens)} tokens")
    if len(mask) != response_length:
        raise ValueError(f"loss mask length {len(mask)} != response length {response_length}")
    if len(logprobs) != response_length:
        raise ValueError(f"logprob length {len(logprobs)} != response length {response_length}")
    if any(value not in (0, 1, False, True) for value in mask):
        raise ValueError("loss mask must contain only 0/1 values")
    if require_trainable and not any(mask):
        raise ValueError("expected at least one trainable generated token")


def annotate_episode_samples(
    samples: list[Any],
    *,
    rollout_id: int,
    group_index: int | None,
    episode_id: str,
    role: str,
    session_id: str,
    mode: str,
    model_version: str,
    trainable: bool,
    reward: float,
    reward_components: dict[str, float] | None = None,
    action_id: str | None = None,
    task_id: str | None = None,
    parent_action_id: str | None = None,
    termination: str = "completed",
) -> list[Any]:
    for segment_index, sample in enumerate(samples):
        sample.rollout_id = rollout_id
        if group_index is not None:
            sample.group_index = group_index
        sample.reward = float(reward)
        metadata = dict(getattr(sample, "metadata", None) or {})
        metadata.update(
            {
                "episode_id": episode_id,
                "rollout_id": rollout_id,
                "role": role,
                "session_id": session_id,
                "training_mode": mode,
                "model_version": model_version,
                "action_id": action_id,
                "task_id": task_id,
                "parent_action_id": parent_action_id,
                "segment_index": segment_index,
                "termination": termination,
                "reward_components": dict(reward_components or {}),
            }
        )
        sample.metadata = metadata
        if not trainable:
            sample.loss_mask = [0] * int(getattr(sample, "response_length", 0) or 0)
        validate_sample(sample, require_trainable=trainable)
    return samples


def infrastructure_failure(sample: Any, *, kind: str, message: str, retryable: bool = True) -> Any:
    sample.remove_sample = True
    metadata = dict(getattr(sample, "metadata", None) or {})
    metadata.update(
        {
            "infrastructure_failure": {
                "kind": kind,
                "message": message,
                "retryable": bool(retryable),
            }
        }
    )
    sample.metadata = metadata
    sample.reward = 0.0
    if getattr(sample, "response_length", 0):
        sample.loss_mask = [0] * sample.response_length
    return sample
