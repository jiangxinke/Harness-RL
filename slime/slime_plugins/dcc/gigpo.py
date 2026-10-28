"""GiGPO advantages for ordinary CUAD policy completions.

Arithmetic follows langfengQ/verl-agent, commit 20bd331bdbc9026a5668e11362178e10ab7400c8.
Copyright 2025 Nanyang Technological University (NTU), Singapore, and the
verl-agent (GiGPO) team. Licensed under Apache-2.0:
https://www.apache.org/licenses/LICENSE-2.0
"""
from __future__ import annotations

from difflib import SequenceMatcher
import hashlib
import math

from .episode import CUAD_FINAL_REVIEW, FINAL_TURN
from .hf_policy import _chat_tokens


def _prefixes(result):
    prefixes, history = [], []
    for message in result.messages:
        if message["role"] == "assistant":
            prefixes.append(list(history))
        history.append(message)
    if result.status == "model_protocol_failure" and len(prefixes) + 1 == len(result.steps):
        prefixes.append(history)
    if len(prefixes) != len(result.steps):
        raise ValueError("CUAD messages and sampled decision count disagree")
    return prefixes


def _anchor(messages):
    observation = next((m["content"] for m in reversed(messages)
                        if m["role"] == "user" and m["content"] not in (CUAD_FINAL_REVIEW, FINAL_TURN)), None)
    if observation is None:
        raise ValueError("CUAD decision is missing its public observation")
    return messages[-1]["content"] in (CUAD_FINAL_REVIEW, FINAL_TURN), observation


def _normalize(values):
    if max(values) == min(values):
        return [0.0] * len(values)
    mean = math.fsum(values) / len(values)
    scale = math.sqrt(math.fsum((x - mean) ** 2 for x in values) / (len(values) - 1)) + 1e-6
    return [(x - mean) / scale for x in values]


def step_advantages(results, *, tokenizer, gamma=0.95, weight=1.0, similarity_threshold=0.9):
    """Use the same replayable episodes as dcc_train's terminal GRPO control."""
    if not 0 < gamma <= 1 or not math.isfinite(weight) or weight < 0 or not 0 < similarity_threshold <= 1:
        raise ValueError("Invalid GiGPO discount, step weight or similarity threshold")
    if not results or any(r.benchmark != "cuad" or r.split != "train" for r in results):
        raise ValueError("GiGPO requires CUAD TRAIN episodes")
    if len({r.task_id for r in results}) != 1:
        raise ValueError("GiGPO groups must contain one task")
    if any(not math.isfinite(r.reward) or not 0 <= r.reward <= 1 for r in results):
        raise ValueError("GiGPO requires finite terminal rewards in [0, 1]")
    rows, clusters = [], []
    combined = [[] for _ in results]
    for episode, result in enumerate(results):
        replayable = {id(step) for step in result.trainable_steps}
        if not replayable:
            continue
        for index, (step, prefix) in enumerate(zip(result.steps, _prefixes(result), strict=True)):
            if id(step) not in replayable:
                continue
            if _chat_tokens(tokenizer, prefix) != step.prompt_ids:
                raise ValueError("GiGPO pilot requires untruncated recorded prompts; increase --max-prompt-tokens")
            anchor = _anchor(prefix)
            group = next((i for i, c in enumerate(clusters) if c[0][0] == anchor[0] and
                          (c[0][1] == anchor[1] or similarity_threshold < 1 and
                           SequenceMatcher(None, c[0][1], anchor[1]).ratio() >= similarity_threshold)), None)
            if group is None:
                group = len(clusters)
                clusters.append((anchor, []))
            clusters[group][1].append(len(rows))
            rows.append(dict(episode=episode, step=index, group=group,
                observation_hash=hashlib.sha256(repr(anchor).encode()).hexdigest()[:16],
                discounted_return=result.reward * gamma ** (len(result.steps) - index - 1)))
    if not rows:
        return combined, dict(rows=[], group_sizes=[], nonzero_local_steps=0)
    # The upstream default counts trajectory rewards once per replayable visit.
    episode_values = _normalize([results[row["episode"]].reward for row in rows])
    local_values = [0.0] * len(rows)
    for _, members in clusters:
        values = _normalize([rows[i]["discounted_return"] for i in members])
        for i, value in zip(members, values, strict=True):
            local_values[i] = value
    for row, episode_value, local in zip(rows, episode_values, local_values, strict=True):
        value = episode_value + weight * local
        row.update(episode_advantage=episode_value, local_advantage=local, advantage=value)
        combined[row["episode"]].append(value)
    return combined, dict(rows=rows, group_sizes=[len(members) for _, members in clusters],
        groups_with_multiple_rollouts=sum(len({rows[i]["episode"] for i in members}) > 1 for _, members in clusters),
        nonzero_local_steps=sum(value != 0 for value in local_values),
        anchor="cuad-current-observation-and-final-phase-v1", episode_weighting="visits",
        std_ddof=1, epsilon=1e-6, upstream_commit="20bd331bdbc9026a5668e11362178e10ab7400c8")
