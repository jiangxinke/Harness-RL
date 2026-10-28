"""GraphPO equations 2--8 on a graph retaining original generation parents.

The paper has no public reference implementation. The estimator uses equation 8
(clip the combined advantage), population standard deviations, and additive
efficiency contributions when routes from several equivalence classes overlap.
"""

from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any


PAPER = "https://arxiv.org/html/2606.18954v1"
VERSION = "graphpo-paper-cuad-v1"
EXCLUDED_STATUSES = {"infrastructure_error", "model_protocol_failure", "context_budget_failure"}


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("GraphPO embeddings must have the same positive dimension")
    if any(not math.isfinite(x) for x in (*left, *right)):
        raise ValueError("GraphPO embeddings must contain finite numbers")
    nl = math.sqrt(math.fsum(x * x for x in left))
    nr = math.sqrt(math.fsum(x * x for x in right))
    if nl == 0 or nr == 0:
        raise ValueError("GraphPO embeddings must have nonzero norms")
    return max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(left, right)) / (nl * nr)))


def normalize(values: Sequence[float], *, epsilon: float = 1e-6) -> list[float]:
    if not values:
        return []
    if any(not math.isfinite(value) for value in values):
        raise ValueError("GraphPO normalization received a nonfinite value")
    if min(values) == max(values):
        return [0.0] * len(values)
    mean = math.fsum(values) / len(values)
    std = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / len(values))
    return [(value - mean) / (std + epsilon) for value in values]


@dataclasses.dataclass(frozen=True)
class GraphNode:
    id: int
    parent: int | None
    depth: int
    summary: str
    embedding: tuple[float, ...]
    terminal_success: float | None = None
    closed: bool = False
    status: str = "active"


class GraphPOGraph:
    def __init__(self, *, threshold: float = 0.92):
        if not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("GraphPO merging threshold must be in (0, 1]")
        self.threshold = threshold
        self.nodes: list[GraphNode] = []
        self.children: dict[int, list[int]] = defaultdict(list)
        self._classes: list[int] = []
        self.merge_log: list[dict[str, Any]] = []
        self._compared: set[tuple[int, int]] = set()

    def add_node(self, parent: int | None, summary: str, embedding: Sequence[float], *,
                 terminal_success: float | None = None, closed: bool = False,
                 status: str = "active") -> int:
        if not summary.strip():
            raise ValueError("GraphPO semantic summaries must not be empty")
        vector = tuple(float(value) for value in embedding)
        cosine(vector, vector)
        if self.nodes and len(vector) != len(self.nodes[0].embedding):
            raise ValueError("GraphPO embedding dimension changed within a graph")
        if terminal_success is not None and terminal_success not in (0.0, 1.0):
            raise ValueError("GraphPO node evidence uses binary success, not scaled rewards")
        if status in EXCLUDED_STATUSES and terminal_success is not None:
            raise ValueError("GraphPO infrastructure failures cannot provide terminal outcome evidence")
        if parent is None:
            if self.nodes or terminal_success is not None or closed:
                raise ValueError("GraphPO must start with exactly one nonterminal root")
            depth = 0
        else:
            if not 0 <= parent < len(self.nodes) or self.nodes[parent].closed:
                raise ValueError("GraphPO generated edges require an existing open parent")
            depth = self.nodes[parent].depth + 1
        index = len(self.nodes)
        self.nodes.append(GraphNode(index, parent, depth, summary, vector,
                                    terminal_success, closed or terminal_success is not None, status))
        self._classes.append(index)
        if parent is not None:
            self.children[parent].append(index)
        return index

    def find(self, node: int) -> int:
        parent = self._classes[node]
        if parent != node:
            self._classes[node] = self.find(parent)
        return self._classes[node]

    def classes(self) -> list[list[int]]:
        groups: dict[int, list[int]] = defaultdict(list)
        for node in range(len(self.nodes)):
            groups[self.find(node)].append(node)
        return sorted(groups.values(), key=lambda group: min(group))

    def path(self, node: int) -> list[int]:
        path = []
        while node is not None:
            path.append(node)
            node = self.nodes[node].parent
        return list(reversed(path))

    def _would_cycle(self, left: int, right: int) -> bool:
        a, b = self.find(left), self.find(right)
        representative = {i: (a if self.find(i) == b else self.find(i)) for i in range(len(self.nodes))}
        successors: dict[int, set[int]] = defaultdict(set)
        indegree = dict.fromkeys(representative.values(), 0)
        for node in self.nodes[1:]:
            source, destination = representative[node.parent], representative[node.id]
            if source == destination:
                return True
            if destination not in successors[source]:
                successors[source].add(destination)
                indegree[destination] += 1
        pending = [node for node, degree in indegree.items() if degree == 0]
        visited = 0
        while pending:
            node = pending.pop()
            visited += 1
            for child in successors[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    pending.append(child)
        return visited != len(indegree)

    def merge_pair(self, left: int, right: int) -> bool:
        if not (0 <= left < len(self.nodes) and 0 <= right < len(self.nodes)):
            raise ValueError("GraphPO merge refers to an unknown node")
        if left == right or self.find(left) == self.find(right):
            return False
        similarity = cosine(self.nodes[left].embedding, self.nodes[right].embedding)
        reason = None
        if self.nodes[left].status in EXCLUDED_STATUSES or self.nodes[right].status in EXCLUDED_STATUSES:
            reason = "excluded_failure"
        elif left in self.path(right) or right in self.path(left):
            reason = "ancestor_descendant"
        elif similarity < self.threshold:
            reason = "below_threshold"
        elif self._would_cycle(left, right):
            reason = "quotient_cycle"
        record = {"left": left, "right": right, "similarity": similarity,
                  "accepted": reason is None, "reason": reason}
        self.merge_log.append(record)
        if reason is not None:
            return False
        a, b = self.find(left), self.find(right)
        self._classes[max(a, b)] = min(a, b)
        return True

    def discover_equivalences(self) -> None:
        for right in range(1, len(self.nodes)):
            for left in range(right):
                pair = (left, right)
                if pair in self._compared:
                    continue
                self._compared.add(pair)
                self.merge_pair(left, right)

    def allocate(self, frontier: Iterable[int], base: int) -> tuple[dict[int, int], dict[str, Any]]:
        if type(base) is not int or base < 1:
            raise ValueError("GraphPO layer branching must be a positive integer")
        active = sorted(set(frontier))
        if any(not 0 <= node < len(self.nodes) or self.nodes[node].closed for node in active):
            raise ValueError("GraphPO frontier must contain only open graph nodes")
        representatives = {self.find(group[0]): min(group, key=lambda n: (self.nodes[n].depth, n))
                           for group in self.classes()}
        redundant = [node for node in active if node != representatives[self.find(node)]]
        recipients = [node for node in active if node not in redundant]
        allocation = {node: max(1, base // 2) if node in redundant else base for node in active}
        saved = len(active) * base - sum(allocation.values())
        if recipients:
            for slot in range(saved):
                allocation[recipients[slot % len(recipients)]] += 1
        return allocation, {
            "frontier": active, "base": base, "redundant": redundant,
            "recipients": recipients, "saved": saved,
            "reallocated": saved if recipients else 0,
            "unused": 0 if recipients else saved,
            "allocation": allocation,
        }


def graph_advantages(graph: GraphPOGraph, *, pooling_weight: float = 0.7,
                     efficiency_weight: float = 1.0, epsilon: float = 1e-6) -> tuple[dict[int, float], dict[str, Any]]:
    if (not math.isfinite(pooling_weight) or not 0 <= pooling_weight <= 1
            or not math.isfinite(efficiency_weight) or efficiency_weight < 0
            or not math.isfinite(epsilon) or epsilon <= 0):
        raise ValueError("GraphPO needs pooling in [0,1], nonnegative efficiency weight, and positive epsilon")
    if not graph.nodes:
        raise ValueError("GraphPO needs a sampled graph")
    count = [0] * len(graph.nodes)
    correct = [0.0] * len(graph.nodes)
    for node in reversed(graph.nodes):
        if node.terminal_success is not None:
            count[node.id], correct[node.id] = 1, node.terminal_success
        for child in graph.children[node.id]:
            count[node.id] += count[child]
            correct[node.id] += correct[child]
    scores: dict[int, float | None] = {}
    class_records = []
    for members in graph.classes():
        total_count = sum(count[n] for n in members)
        total_correct = math.fsum(correct[n] for n in members)
        for node in members:
            denominator = count[node] + pooling_weight * (total_count - count[node])
            scores[node] = ((correct[node] + pooling_weight * (total_correct - correct[node])) / denominator
                            if denominator else None)
        similarities = [cosine(graph.nodes[a].embedding, graph.nodes[b].embedding)
                        for j, a in enumerate(members) for b in members[j + 1:]]
        class_records.append({"members": members, "min_pair_similarity": min(similarities) if similarities else None})
    edge_rewards, edge_records = {}, {}
    for node in graph.nodes[1:]:
        parent = node.parent
        similarity = cosine(graph.nodes[parent].embedding, node.embedding)
        novelty = 1 - max(0.0, similarity)
        known = scores[parent] is not None and scores[node.id] is not None
        reward = (scores[node.id] - scores[parent]) * novelty if known else None
        if known:
            edge_rewards[node.id] = reward
        edge_records[node.id] = {"edge": node.id, "parent": parent, "child": node.id,
                                 "score_parent": scores[parent], "score_child": scores[node.id],
                                 "similarity": similarity, "novelty": novelty, "step_reward": reward,
                                 "correctness_advantage": 0.0 if known else None,
                                 "efficiency_advantage": 0.0,
                                 "correctness_estimated": known, "efficiency_supported": False,
                                 "estimated": known}
    correctness_groups = []
    for members in graph.classes():
        edges = [child for node in members for child in graph.children[node] if child in edge_rewards]
        if len(edges) != len(set(edges)):
            raise AssertionError("GraphPO generated edges must occur once per correctness group")
        values = normalize([edge_rewards[e] for e in edges], epsilon=epsilon)
        for edge, value in zip(edges, values):
            edge_records[edge]["correctness_advantage"] = value
        correctness_groups.append({"source_class": members, "edges": edges,
                                   "edge_rewards": [edge_rewards[e] for e in edges], "advantages": values})
    efficiency_groups = []
    for members in graph.classes():
        if len(members) < 2 or any(scores[node] is None for node in members):
            continue
        paths = [graph.path(node) for node in members]
        common_length = 0
        for entries in zip(*paths):
            if len(set(entries)) != 1:
                break
            common_length += 1
        ancestor = paths[0][common_length - 1]
        routes = [path[common_length:] for path in paths]
        lengths = [len(route) for route in routes]
        gate = math.fsum(scores[node] for node in members) / len(members)
        values = [gate * value for value in normalize([-length for length in lengths], epsilon=epsilon)]
        for route, value in zip(routes, values):
            for edge in route:
                edge_records[edge]["efficiency_advantage"] += value
                edge_records[edge]["efficiency_supported"] = True
        efficiency_groups.append({"members": members, "common_ancestor": ancestor,
                                  "routes": routes, "lengths": lengths, "gate": gate, "advantages": values})
    advantages = {}
    for edge, record in edge_records.items():
        correctness = record["correctness_advantage"]
        record["advantage"] = (correctness if correctness is not None else 0.0) + efficiency_weight * record["efficiency_advantage"]
        record["correctness_missing_reason"] = None if record["correctness_estimated"] else "one_or_both_endpoint_scores_unestimated"
        if record["correctness_estimated"] or record["efficiency_supported"]:
            advantages[edge] = record["advantage"]
    diagnostics = {
        "version": VERSION, "paper": PAPER, "implementation": "paper-based CUAD adaptation; no official code available",
        "pooling_weight": pooling_weight, "efficiency_weight": efficiency_weight,
        "threshold": graph.threshold, "normalization_ddof": 0, "epsilon": epsilon,
        "reward_definition": "binary terminal success pooled as [0,1]; no scaled 10-point reward or invalid-action penalty",
        "loss_definition": "equation 8: PPO clipping on correctness plus weighted efficiency advantage",
        "efficiency_route_allocation": "same coefficient per route edge; overlapping class contributions summed",
        "nodes": [{**dataclasses.asdict(node), "class": graph.find(node.id), "own_count": count[node.id],
                   "own_correct": correct[node.id], "score": scores[node.id]} for node in graph.nodes],
        "classes": class_records, "merges": graph.merge_log,
        "edges": list(edge_records.values()), "correctness_groups": correctness_groups,
        "efficiency_groups": efficiency_groups,
        "unestimated_edges": [edge for edge, value in edge_records.items() if not value["estimated"]],
        "unsupported_edges": [edge for edge, value in edge_records.items()
                              if not value["correctness_estimated"] and not value["efficiency_supported"]],
    }
    return advantages, diagnostics
