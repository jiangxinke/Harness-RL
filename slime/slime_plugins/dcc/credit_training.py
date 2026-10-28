"""One recorded environment boundary for the agent-credit comparisons."""
from __future__ import annotations

import numpy as np
import torch

from .credit_adapter import build_credit_visits, unpack_scores
from .credit_upstream import core_gigpo


def prepare_credit(results, *, tokenizer, method, mode="mean_std_norm",
                   invalid_penalty=.1, gamma=.95, weight=1.0,
                   similarity_threshold=.9, graph_omega=.2):
    visits, adaptation = build_credit_visits(results, tokenizer=tokenizer)
    if not visits:
        return [], [[] for _ in results], {**adaptation, "rows": []}
    if len({v.episode for v in visits}) < 2:
        return [], [[] for _ in results], {**adaptation, "rows": [],
            "skipped_group": "fewer_than_two_complete_episodes"}
    if method == "grpo":
        masks = torch.ones((len(visits), 1))
        raw = torch.tensor([v.task_reward - invalid_penalty * (not v.is_action_valid)
                            for v in visits], dtype=torch.float32)
        task = np.array([v.task_id for v in visits], dtype=object)
        trajectory = np.array([f"{v.task_id}:{v.episode}" for v in visits], dtype=object)
        scores = core_gigpo.episode_norm_reward(raw[:, None], masks, task, trajectory,
            remove_std=mode == "mean_norm").flatten().tolist()
        diagnostics = {"rows": [dict(episode=v.episode, step=v.step,
            is_action_valid=v.is_action_valid, task_reward=v.task_reward,
            episode_score=float(raw[i]), episode_advantage=scores[i], advantage=scores[i])
            for i, v in enumerate(visits)], "episode_weighting": "visits"}
    elif method == "gigpo":
        from .gigpo import compute_advantages
        scores, diagnostics = compute_advantages(visits, gamma=gamma, weight=weight,
            mode=mode, invalid_penalty=invalid_penalty, similarity_threshold=similarity_threshold)
    elif method == "graphgpo":
        from .graphgpo import compute_advantages
        scores, diagnostics = compute_advantages(visits, omega=graph_omega, weight=weight,
            mode=mode, invalid_penalty=invalid_penalty)
    elif method == "g2po":
        from .g2po import compute_advantages
        scores, diagnostics = compute_advantages(visits, weight=weight,
            mode=mode, invalid_penalty=invalid_penalty)
    else:
        raise ValueError("GraphPO requires its graph collector; unknown chain estimator")
    nested = unpack_scores(results, visits, scores)
    return list(zip((v.completion for v in visits), scores, strict=True)), nested, {
        **adaptation, **diagnostics, "method": method, "normalization": mode,
        "invalid_penalty": invalid_penalty,
    }
