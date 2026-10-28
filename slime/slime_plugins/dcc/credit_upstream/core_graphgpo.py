# Copyright 2025 Nanyang Technological University (NTU), Singapore, and the verl-agent team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""Selected unchanged GraphGPO definitions from langfengQ/verl-agent.

Source commit: 20bd331bdbc9026a5668e11362178e10ab7400c8.
Source file: recipe/GraphGPO/core_graph.py.
Only imports are adapted; definition bodies are verified against the source
embedded in third_party/graphgpo-source.json.
"""
from collections import defaultdict, Counter
from difflib import SequenceMatcher
from typing import Any as DataProto

import numpy as np
import torch

from .core_gigpo import episode_norm_reward, build_step_group, step_norm_reward

def to_hashable(x):
    """Convert an object into a hashable type (used for clustering/grouping)."""
    if isinstance(x, (int, float, str, bool)):
        return x
    elif isinstance(x, (np.integer, np.floating)):
        return x.item()
    elif isinstance(x, np.ndarray):
        return tuple(x.flatten())
    elif isinstance(x, (list, tuple)):
        return tuple(to_hashable(e) for e in x)
    elif isinstance(x, dict):
        return tuple(sorted((k, to_hashable(v)) for k, v in x.items()))
    else:
        raise TypeError(f"Unsupported type: {type(x)}")


class StateCanonicalizer:
    """
    Maps similar states to a canonical representative.
    When enabled, states within `threshold` similarity are treated as the same node in the graph.
    """

    def __init__(self, enable: bool = False, threshold: float = 0.95):
        self.enable = enable
        self.threshold = threshold
        # List of (canonical_hashable, canonical_raw_str) pairs
        self._clusters: list = []
        # Cache: raw hashable -> canonical hashable
        self._cache: dict = {}

    def canonicalize(self, state):
        """
        Return the canonical hashable form of `state`.
        If similarity is disabled, falls back to exact `to_hashable`.
        """
        h = to_hashable(state)
        if not self.enable:
            return h

        # Check cache first
        if h in self._cache:
            return self._cache[h]

        raw_str = state if isinstance(state, str) else str(state)

        # Try to find a matching cluster
        for canonical_h, canonical_str in self._clusters:
            if SequenceMatcher(None, raw_str, canonical_str).ratio() >= self.threshold:
                self._cache[h] = canonical_h
                return canonical_h

        # No match — this state becomes a new canonical representative
        self._clusters.append((h, raw_str))
        self._cache[h] = h
        return h


def nested_defaultdict():
    return defaultdict(list)


class ComplexGraph:
    def __init__(self):
        # graph[u][v] = list of edge dicts (e.g. {"action":..., "traj_uid":...})
        self.graph = defaultdict(nested_defaultdict)
        self.nodes = set()
        self.uids = set()
        self.traj_to_uid = {}
        self.episode_rewards = None  # later for reward-based analysis
        self.final_state=None
        self.first_state=None
        self.shortest_path_to_final=None

    def add_edge(self, u, v, **attrs):
        self.graph[to_hashable(u)][to_hashable(v)].append(attrs)
        self.nodes.update([u, v])
        if "uid" in attrs:
            self.uids.add(attrs["uid"])
        if "traj_uid" in attrs and "uid" in attrs:
            self.traj_to_uid[attrs["traj_uid"]] = attrs["uid"]

    def neighbors(self, u):
        return list(self.graph[to_hashable(u)].keys())

    def edges(self, data=False):
        for u, targets in self.graph.items():
            for v, edges in targets.items():
                if data:
                    for attr in edges:
                        yield (u, v, attr)
                else:
                    yield (u, v)

    def has_edge(self, u, v):
        return v in self.graph[to_hashable(u)]

    def get_traj_path(self, traj_uid):
        traj_edges = []
        for u, targets in self.graph.items():
            for v, edges in targets.items():
                for attr in edges:
                    if attr.get("traj_uid") == traj_uid:
                        traj_edges.append((attr.get("data_id", 0), u, attr.get("action"), v))
        if not traj_edges:
            raise ValueError(f"Trajectory {traj_uid} not found in graph.")

        traj_edges.sort(key=lambda x: x[0])
        path = [(u, action, v) for _, u, action, v in traj_edges]
        return path

    def get_edges(self, u, v):
        return self.graph[to_hashable(u)][to_hashable(v)]

    def all_simple_paths(self, source, target, path=None, visited=None):
        source=to_hashable(source)
        target=to_hashable(target)
        if path is None:
            path = [source]
        if visited is None:
            visited = set([source])
        if source == target:
            yield list(path)
            return
        for neighbor in self.neighbors(source):
            if neighbor not in visited:
                yield from self.all_simple_paths(neighbor, target, path + [neighbor], visited | {neighbor})

    def __repr__(self):
        return f"ComplexGraph(num_nodes={len(self.nodes)}, num_edges={sum(len(v) for u in self.graph.values() for v in u.values())})"

    @classmethod
    def from_data(cls, data, enable_similarity=False, similarity_thresh=0.95):
        Gs = {}
        canonicalizers = {}
        for uid in set(data.non_tensor_batch['uid']):
            g = cls()
            canon = StateCanonicalizer(enable=enable_similarity, threshold=similarity_thresh)
            uid_indices = np.where(data.non_tensor_batch['uid'] == uid)[0]

            for i in uid_indices:
                weight = 1.0
                anchor_h = canon.canonicalize(data.non_tensor_batch['anchor_obs'][i])
                next_h   = canon.canonicalize(data.non_tensor_batch['next_obs'][i])
                g.add_edge(
                    anchor_h,
                    next_h,
                    action=data.non_tensor_batch['text_actions'][i],
                    traj_uid=data.non_tensor_batch['traj_uid'][i],
                    uid=data.non_tensor_batch['uid'][i],
                    data_id=i,
                    episode_rewards=data.non_tensor_batch['episode_rewards'][i],
                    weight=weight
                )

            # ── final_state: rollout_loop.py always writes 'success' as next_obs
            #    for the last step of a successful episode.
            g.final_state = canon.canonicalize('success')

            # ── first_state: per-trajectory, find the step with the smallest
            #    data_id (= true first step before shuffling), take its anchor_obs.
            #    Vote across trajectories to handle rare observation mismatches.
            traj_first_counts: Counter = Counter()
            for traj_uid in set(data.non_tensor_batch['traj_uid'][uid_indices]):
                traj_mask = uid_indices[
                    data.non_tensor_batch['traj_uid'][uid_indices] == traj_uid
                ]
                if len(traj_mask) == 0:
                    continue
                first_step = traj_mask[np.argmin(traj_mask)]   # smallest data_id
                state = canon.canonicalize(
                    data.non_tensor_batch['anchor_obs'][first_step]
                )
                traj_first_counts[state] += 1
            g.first_state = traj_first_counts.most_common(1)[0][0]

            g.canonicalizer = canon
            Gs[uid] = g
            canonicalizers[uid] = canon
        return Gs

    @staticmethod
    def find_success_states_per_task(data):
        success_states_per_task = defaultdict(list)
        success_idx = data.non_tensor_batch['episode_rewards'] > 0
        for traj_uid in set(data.non_tensor_batch['traj_uid'][success_idx]):
            traj_uid_ids = np.where(data.non_tensor_batch['traj_uid'] == traj_uid)[0]
            uid = data.non_tensor_batch['uid'][traj_uid_ids[0]]
            for anchor_obs in data.non_tensor_batch['anchor_obs'][traj_uid_ids]:
                success_states_per_task[(uid, traj_uid)].append(anchor_obs)
            success_states_per_task[(uid, traj_uid)].append(data.non_tensor_batch['next_obs'][traj_uid_ids[-1]])
        return success_states_per_task

    @staticmethod
    def Calculate_all_shortest_path(Gs:dict):
        for key in Gs.keys():
            Gs[key].calculate_path()

    def is_mandatory(self, s_from, s_mid, s_to):
        has_path = False
        for path in self.all_simple_paths(s_from, s_to):
            has_path = True
            if s_mid not in path:
                return False
        if not has_path:
            raise Exception("no path")
        return True

    @classmethod
    def get_golden_states(cls, golden_candidate, Gs):
        final_golden_states = defaultdict(list)
        for uid, candidates in golden_candidate.items():
            g = Gs[uid]
            remove_state = []
            if len(candidates) <= 2:
                final_golden_states[uid] = candidates
                continue
            for i in range(1, len(candidates) - 1):
                if not g.is_mandatory(candidates[0], candidates[i], candidates[-1]):
                    remove_state.append(i)
            final_golden_states[uid] = [candidates[i] for i in range(len(candidates)) if i not in remove_state]
        return final_golden_states

    def shortest_path(self, source, target):
        source=to_hashable(source)
        target=to_hashable(target)
        if source not in self.nodes or target not in self.nodes:
            raise ValueError("source or target not in graph")

        from collections import deque
        queue = deque([[source]])
        visited = set([source])

        while queue:
            path = queue.popleft()
            node = path[-1]
            if node == target:
                return path

            for neighbor in self.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])

        return None

    def shortest_path_weighted(self, source, target, weight_key="weight"):
        source=to_hashable(source)
        target=to_hashable(target)
        import heapq
        pq = [(0, source, [source])]
        visited = {}

        while pq:
            cost, node, path = heapq.heappop(pq)
            if node == target:
                return path, cost
            if node in visited and visited[node] <= cost:
                continue
            visited[node] = cost
            for neighbor in self.neighbors(node):
                edges = self.graph[node][neighbor]
                w = min(e.get(weight_key, 1) for e in edges)
                heapq.heappush(pq, (cost + w, neighbor, path + [neighbor]))
        return None, 51


    def all_to_target_shortest_paths(self, target, weight_key="weight"):
        if target is not None:
            target=to_hashable(target)

        import heapq

        reverse_graph = defaultdict(list)
        for u in self.graph:
            for v in self.graph[u]:
                for attr in self.graph[u][v]:
                    w = attr.get(weight_key, 1)
                    reverse_graph[v].append((u, w))

        # ---------- Dijkstra ----------
        distances = {n: float("inf") for n in self.nodes}
        paths = {n: [] for n in self.nodes}
        pq = [(0, target, [target])]
        distances[target] = 0

        while pq:
            cost, node, path = heapq.heappop(pq)
            if cost > distances[node]:
                continue

            for neighbor, w in reverse_graph[node]:
                new_cost = cost + w
                if new_cost < distances[neighbor]:
                    distances[neighbor] = new_cost
                    paths[neighbor] = [neighbor] + path
                    heapq.heappush(pq, (new_cost, neighbor, [neighbor] + path))
        
        far_distance=0
        for key,value in distances.items():
            if (value !=float("inf")) and (value>far_distance):
                far_distance=value
        
        for key,value in distances.items():
            if value==float("inf"):
                distances[key]=far_distance+1
        return distances, paths
    
    def calculate_path(self,weight_key="weight"):
        self.shortest_path_to_final=self.all_to_target_shortest_paths(self.final_state,weight_key="weight")

    def print_shortest_path(self, source, target, weighted: bool = False, weight_key: str = "weight"):
        source_hash = to_hashable(source)
        target_hash = to_hashable(target)
        if source_hash not in self.nodes:
            print(f"❌  {source}")
            return
        if target_hash not in self.nodes:
            print(f"❌ {target}")
            return
        
        if weighted:
            path, total_cost = self.shortest_path_weighted(source_hash, target_hash, weight_key)
            if path is None:
                print(f"❌")
                return
            print(f"✅")
        else:
            path = self.shortest_path(source_hash, target_hash)
            if path is None:
                print(f"❌")
                return
            print(f"✅")
        
        self._print_path_details(path, weighted, weight_key)

    def _print_path_details(self, path: list, weighted: bool, weight_key: str):
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            edges = self.get_edges(u, v)
            
            if weighted and edges:
                edge = min(edges, key=lambda x: x.get(weight_key, 1))
            else:
                edge = edges[0] if edges else {}
            
            action = edge.get("action", "")
            weight = edge.get(weight_key, 1.0) if weighted else "-"
            traj_uid = edge.get("traj_uid", "")
            print(f"  {i+1}: {u}")
            print(f"    ↳ : {action} | : {weight} | UID: {traj_uid}")
        print(f"  {len(path)}: {path[-1]}")

    def print_all_uid_shortest_paths(self, weighted: bool = False, weight_key: str = "weight"):

        if not self.first_state:
            print("❌")
            return
        if not self.final_state:
            print("❌")
            return
        
        self.print_shortest_path(
            source=self.first_state,
            target=self.final_state,
            weighted=weighted,
            weight_key=weight_key
        )


def compute_graph_path_returns(batch: DataProto, Gs: dict, gamma: float, normalize_distance: bool):
    """
    Compute discounted returns for each trajectory. (Eq. 5 in the paper)

    Args:
        batch (DataProto): Input batch.
        gamma (float): Discount factor.

    Returns:
        torch.Tensor: Discounted returns.
    """
    rewards = batch.non_tensor_batch['rewards'].astype(np.float32)
    traj_uids = batch.non_tensor_batch['traj_uid']
    uids=batch.non_tensor_batch['uid']
    active_masks = batch.non_tensor_batch['active_masks'].astype(np.float32)
    next_obs_=batch.non_tensor_batch['next_obs']
    old_obs_=batch.non_tensor_batch['anchor_obs']
    is_action_valid_=batch.non_tensor_batch['is_action_valid']
    returns_by_traj = {}
    unique_traj_uids = np.unique(traj_uids)
    for uid in unique_traj_uids:
        # Get indices for this trajectory
        traj_indices = np.where(traj_uids == uid)[0]
        uid_uid=uids[traj_indices[0]]
        # Extract rewards and masks for this trajectory
        traj_rewards = rewards[traj_indices]
        traj_active_masks = active_masks[traj_indices]
        new_obs=next_obs_[traj_indices]
        old_obs=old_obs_[traj_indices]
        is_action_valid=is_action_valid_[traj_indices]
        assert traj_active_masks.all(), "active_masks should be all 1s for the same trajectory"

        # Calculate returns
        traj_returns = np.zeros_like(traj_rewards)
        g=Gs[uid_uid]
        distances_dict,distances_path_dict=g.shortest_path_to_final

        # Use the graph's canonicalizer (if available) to map states consistently
        canon = getattr(g, 'canonicalizer', None)

        for t in range(len(traj_rewards)):
            weight=1.0
            if canon is not None:
                new_h = canon.canonicalize(new_obs[t])
                old_h = canon.canonicalize(old_obs[t])
            else:
                new_h = to_hashable(new_obs[t])
                old_h = to_hashable(old_obs[t])
            if normalize_distance:
                traj_returns[t] = 10*gamma**(distances_dict[new_h]-distances_dict[old_h]+weight)
            else:
                traj_returns[t] = 10*gamma**(distances_dict[new_h]+weight-1)

        # Store the results
        returns_by_traj[uid] = traj_returns

    # Recombine the returns into the original batch order
    all_returns = np.zeros_like(rewards)
    for i, uid in enumerate(traj_uids):
        traj_indices = np.where(traj_uids == uid)[0]
        idx_in_traj = np.where(traj_indices == i)[0][0]  # Find position of i in its trajectory
        all_returns[i] = returns_by_traj[uid][idx_in_traj]

    all_returns = torch.tensor(all_returns, dtype=torch.float32, device=batch.batch['input_ids'].device)
    return all_returns


def compute_graphgpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    anchor_obs: np.ndarray,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    step_advantage_w: float = 1.0,
    episode_advantage_w: float = 1.0,
    mode: str = "mean_std_norm",
):
    """
    Compute GraphGPO advantages.

    Joint advantage formula (Eq. 8 in GiGPO, adapted for GraphGPO):
        scores = step_advantage_w * step_adv + episode_advantage_w * episode_adv

    step_adv   -- normalized graph-path-returns from compute_graph_path_returns().
                  Steps whose next_obs is closer to the goal node receive higher returns
                  (10 * gamma^d), providing a dense step-level signal.
    episode_adv -- normalized episode-level outcome reward, same as GiGPO.
                   Set episode_advantage_w=0.0 to rely solely on the graph signal.

    Args:
        token_level_rewards: shape (bs, response_length) — per-token reward scores.
        step_rewards:        shape (bs,) — graph-path-return for each step, from
                             compute_graph_path_returns().
        response_mask:       shape (bs, response_length) — 1 for response tokens.
        anchor_obs:          shape (bs,) — observation at the start of each step,
                             used to group steps sharing the same state.
        index:               shape (bs,) — task uid, groups trajectories by prompt.
        traj_index:          shape (bs,) — trajectory uid within a task group.
        epsilon:             small constant to avoid division by zero in normalization.
        step_advantage_w:    weight on the graph step advantage.
        episode_advantage_w: weight on the episode-level advantage. Set to 0.0 to
                             disable episode signal and use only graph returns.
        mode:                "mean_std_norm" (subtract mean, divide by std) or
                             "mean_norm" (subtract mean only).
    """
    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Group steps that share the same anchor_obs within the same task (uid).
    # Steps from the same state compete against each other in normalization,
    # so the advantage reflects which action led to a better next state.
    step_group_uids = build_step_group(anchor_obs, index)
    step_advantages = step_norm_reward(step_rewards, response_mask, step_group_uids, epsilon, remove_std)

    if episode_advantage_w != 0.0:
        # Episode-level signal: measures how good this trajectory is relative to
        # other trajectories on the same task (same prompt uid).
        episode_advantages = episode_norm_reward(token_level_rewards, response_mask, index, traj_index, epsilon, remove_std)
        scores = step_advantage_w * step_advantages + episode_advantage_w * episode_advantages
    else:
        # Skip episode_norm_reward entirely when its weight is 0 to save compute.
        scores = step_advantage_w * step_advantages

    return scores, scores

