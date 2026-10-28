"""GraphGPO credit on explicit DCC state transitions."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from types import SimpleNamespace

import numpy as np
import torch

from .credit_adapter import CreditVisit
from .credit_upstream import core_graphgpo as upstream


UPSTREAM_COMMIT = "20bd331bdbc9026a5668e11362178e10ab7400c8"
SUCCESS = "success"
FAILURE = "terminal:failure"


def _state(observation: str) -> str:
    # Environment text must not collide with the upstream success sentinel.
    return "observation:" + observation


def _validate(visits, omega, weight, mode, invalid_penalty):
    if not math.isfinite(omega) or not 0 < omega < 1:
        raise ValueError("GraphGPO omega must be finite and strictly between 0 and 1")
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("GraphGPO step weight must be finite and nonnegative")
    if not math.isfinite(invalid_penalty) or invalid_penalty < 0:
        raise ValueError("GraphGPO invalid-action penalty must be finite and nonnegative")
    if mode not in ("mean_std_norm", "mean_norm"):
        raise ValueError("GraphGPO mode must be mean_std_norm or mean_norm")
    if visits and len({v.task_id for v in visits}) != 1:
        raise ValueError("GraphGPO requires one task per rollout group")
    episodes = defaultdict(list)
    for visit in visits:
        if not isinstance(visit.observation, str) or not isinstance(visit.next_observation, str):
            raise ValueError("GraphGPO observations must be strings")
        if not math.isfinite(visit.task_reward) or visit.task_reward not in (0.0, 10.0):
            raise ValueError("GraphGPO requires binary task reward scaled to 0 or 10")
        if visit.task_reward != 10.0 * bool(visit.success):
            raise ValueError("GraphGPO task reward disagrees with task success")
        episodes[visit.episode].append(visit)
    for sequence in episodes.values():
        if [v.step for v in sequence] != list(range(len(sequence))):
            raise ValueError("GraphGPO requires complete, ordered episode steps starting at zero")
        if any(v.terminal for v in sequence[:-1]) or not sequence[-1].terminal:
            raise ValueError("GraphGPO requires exactly one terminal transition at the end")
        if len({(v.success, v.task_reward) for v in sequence}) != 1:
            raise ValueError("GraphGPO episode outcome must be identical across its visits")
        if any(a.next_observation != b.observation for a, b in zip(sequence, sequence[1:])):
            raise ValueError("GraphGPO transition does not match the next decision's observation")


def compute_advantages(
    visits: list[CreditVisit], *, omega=0.2, weight=1.0,
    mode="mean_std_norm", invalid_penalty=0.1,
):
    """Return visit-aligned advantages and both credit components with the graph."""
    visits = list(visits)
    _validate(visits, omega, weight, mode, invalid_penalty)
    metadata = dict(
        method="graphgpo", upstream_commit=UPSTREAM_COMMIT, omega=omega,
        graph_weight=weight, episode_weight=1.0, mode=mode,
        invalid_penalty=invalid_penalty, matching="exact",
        edge_cost=1.0, normalize_distance=False, std_ddof=1, epsilon=1e-6,
        episode_weighting="visits", graph_reward_formula="10 * omega ** distance(next_state)",
    )
    if not visits:
        return [], dict(metadata, rows=[], graphs=[], group_sizes=[], nonzero_local_steps=0)

    anchors = np.array([_state(v.observation) for v in visits], dtype=object)
    successors = np.array([
        SUCCESS if v.terminal and v.success else FAILURE if v.terminal
        else _state(v.next_observation) for v in visits
    ], dtype=object)
    tasks = np.array([v.task_id for v in visits], dtype=object)
    trajectories = np.array([str(v.episode) for v in visits], dtype=object)
    task_rewards = np.array([v.task_reward for v in visits], dtype=np.float32)
    validity = np.array([v.is_action_valid for v in visits], dtype=bool)
    batch = SimpleNamespace(
        batch={"input_ids": torch.zeros((len(visits), 1), dtype=torch.long)},
        non_tensor_batch=dict(
            uid=tasks, traj_uid=trajectories, anchor_obs=anchors, next_obs=successors,
            rewards=np.array([v.task_reward if v.terminal else 0.0 for v in visits], dtype=np.float32),
            active_masks=np.ones(len(visits), dtype=bool), is_action_valid=validity,
            episode_rewards=task_rewards,
            text_actions=np.array([getattr(v.completion, "text", str(v.completion)) for v in visits], dtype=object),
        ),
    )
    graphs = upstream.ComplexGraph.from_data(batch, enable_similarity=False)
    upstream.ComplexGraph.Calculate_all_shortest_path(graphs)
    graph_returns = upstream.compute_graph_path_returns(batch, graphs, omega, normalize_distance=False)
    penalties = torch.tensor((~validity).astype(np.float32)) * invalid_penalty
    # The upstream trainer applies this penalty after graph-return construction
    # and before both normalizations; it is not a discounted intermediate reward.
    adjusted_returns = graph_returns - penalties
    episode_scores = torch.tensor(task_rewards).unsqueeze(-1) - penalties.unsqueeze(-1)
    mask = torch.ones((len(visits), 1), dtype=torch.float32)
    combined, _ = upstream.compute_graphgpo_outcome_advantage(
        episode_scores, adjusted_returns, mask, anchors, tasks, trajectories,
        epsilon=1e-6, step_advantage_w=weight, episode_advantage_w=1.0, mode=mode,
    )
    remove_std = mode == "mean_norm"
    episode_advantages = upstream.episode_norm_reward(
        episode_scores, mask, tasks, trajectories, 1e-6, remove_std,
    )[:, 0]
    step_groups = upstream.build_step_group(anchors, tasks)
    graph_advantages = upstream.step_norm_reward(
        adjusted_returns, mask, step_groups, 1e-6, remove_std,
    )[:, 0]
    scores = combined[:, 0]
    if not torch.allclose(scores, episode_advantages + weight * graph_advantages, atol=1e-6, rtol=1e-6):
        raise RuntimeError("GraphGPO diagnostics do not reproduce the applied advantage")
    if not torch.isfinite(scores).all():
        raise ValueError("GraphGPO produced a non-finite advantage")

    graph = graphs[visits[0].task_id]
    distances, paths = graph.shortest_path_to_final
    node_ids = {state: f"n{index}" for index, state in enumerate(sorted(distances))}
    members = defaultdict(list)
    for index, state in enumerate(anchors):
        members[state].append(index)
    group_ids = {state: index for index, state in enumerate(members)}
    rows = []
    for index, visit in enumerate(visits):
        source, target = anchors[index], successors[index]
        rows.append(dict(
            episode=visit.episode, step=visit.step, task_id=visit.task_id,
            group=group_ids[source], source_node=node_ids[source], target_node=node_ids[target],
            observation_hash=hashlib.sha256(visit.observation.encode()).hexdigest(),
            group_size=len(members[source]), group_visit_indices=members[source],
            task_reward=visit.task_reward, terminal=bool(visit.terminal), success=bool(visit.success),
            is_action_valid=bool(visit.is_action_valid), invalid_penalty=float(penalties[index]),
            graph_return=float(graph_returns[index]), adjusted_graph_return=float(adjusted_returns[index]),
            adjusted_episode_score=float(episode_scores[index, 0]),
            distance_source=float(distances[source]), distance_target=float(distances[target]),
            target_reachable=bool(target == SUCCESS or paths.get(target)),
            episode_advantage=float(episode_advantages[index]),
            local_advantage=float(graph_advantages[index]),
            graph_advantage=float(graph_advantages[index]), advantage=float(scores[index]),
        ))
    nodes = [dict(
        id=node_ids[state], observation=state[len("observation:"):] if state.startswith("observation:") else None,
        kind="success" if state == SUCCESS else "failure" if state == FAILURE else "observation",
        distance=float(distances[state]), reachable=bool(state == SUCCESS or paths.get(state)),
        shortest_path=[node_ids[s] for s in paths.get(state, [])],
        outgoing_visits=members.get(state, []),
    ) for state in sorted(distances)]
    edges = [dict(source=node_ids[anchors[i]], target=node_ids[successors[i]],
                  episode=v.episode, step=v.step, visit_index=i, cost=1.0)
             for i, v in enumerate(visits)]
    return scores.tolist(), dict(
        metadata, rows=rows, graphs=[dict(task_id=visits[0].task_id, nodes=nodes, edges=edges)],
        group_sizes=[len(indices) for indices in members.values()],
        nonzero_local_steps=int(torch.count_nonzero(graph_advantages).item()),
        groups_with_multiple_rollouts=sum(len({visits[i].episode for i in indices}) > 1 for indices in members.values()),
        success_episodes=len({v.episode for v in visits if v.terminal and v.success}),
    )
