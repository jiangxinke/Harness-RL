"""Layerwise GraphPO sampling through replay of deterministic CUAD episodes."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import math
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from .data import Task
from .episode import EpisodeResult, EpisodeRunner, PolicyCompletion, parse_action
from .graphpo import GraphPOGraph, graph_advantages


def chat_tokens(tokenizer, messages) -> list[int]:
    values = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           enable_thinking=False)
    return list(values["input_ids"] if hasattr(values, "keys") else values)


class ReplayMismatch(ValueError):
    """A replayed branch no longer has its original policy context."""


class _NextDecision(Exception):
    def __init__(self, messages):
        self.messages = copy.deepcopy(messages)


@dataclasses.dataclass
class Expansion:
    completion: PolicyCompletion | None
    next_messages: list[dict[str, str]]
    result: EpisodeResult | None
    replayed_decisions: int
    generated_messages: list[dict[str, str]] | None
    seconds: float


async def expand_once(task: Task, runner: EpisodeRunner, policy, *, tokenizer,
                      prefix: Sequence[PolicyCompletion] = ()) -> Expansion:
    if task.benchmark != "cuad":
        raise ValueError("GraphPO prefix replay is restricted to deterministic CUAD tools")
    started = time.monotonic()
    position = 0
    generated = None
    generated_messages = None

    async def replay(messages, *, max_new_tokens, temperature):
        nonlocal position, generated, generated_messages
        tokens = chat_tokens(tokenizer, messages)
        if position < len(prefix):
            completion = prefix[position]
            if tokens != completion.prompt_ids:
                raise ReplayMismatch(f"GraphPO cached prefix mismatch at decision {position}")
            position += 1
            return completion
        if generated is not None:
            raise _NextDecision(messages)
        if position != len(prefix):
            raise ReplayMismatch("GraphPO replay did not reach the requested prefix")
        generated_messages = copy.deepcopy(messages)
        generated = await policy(messages, max_new_tokens=max_new_tokens, temperature=temperature)
        if not isinstance(generated, PolicyCompletion) or not generated.completion_ids:
            raise ValueError("GraphPO requires a generated PolicyCompletion with response tokens")
        if generated.prompt_ids != tokens:
            raise ReplayMismatch("GraphPO generation changed or truncated the original prompt")
        return generated

    try:
        result = await runner.run(task, replay)
    except _NextDecision as stopped:
        return Expansion(generated, stopped.messages, None, position, generated_messages,
                         time.monotonic() - started)
    if position != len(prefix):
        raise ReplayMismatch("GraphPO cached prefix terminated before its branch point")
    if generated is None and result.status not in {
        "infrastructure_error", "model_protocol_failure", "context_budget_failure"
    }:
        raise ReplayMismatch("GraphPO branch produced no new decision")
    return Expansion(generated, copy.deepcopy(result.messages), result, position,
                     generated_messages, time.monotonic() - started)


def terminal_success(result: EpisodeResult) -> float | None:
    if result.status in {"infrastructure_error", "model_protocol_failure", "context_budget_failure"}:
        return None
    if not math.isfinite(result.reward) or not 0 <= result.reward <= 1:
        raise ValueError("GraphPO CUAD task reward must be finite and in [0, 1]")
    # The runner owns the scoring protocol. Legacy copy diagnostics must not
    # override the shared audited success used by the other credit estimators.
    return float(result.status == "completed" and result.reward >= 1 - 1e-12)


def _usage(callback) -> dict[str, float]:
    value = getattr(callback, "usage", {})
    return {key: float(item) for key, item in value.items()
            if isinstance(item, (int, float)) and not isinstance(item, bool)} if isinstance(value, dict) else {}


def _usage_delta(before, after):
    return {key: value - before.get(key, 0.0) for key, value in after.items()}


def _latest_observation_error(messages):
    marker = "Tool observation (source material):\n"
    for message in reversed(messages):
        content = message.get("content", "")
        if message.get("role") != "user" or not content.startswith(marker):
            continue
        try:
            value, _ = json.JSONDecoder().raw_decode(content[len(marker):])
        except json.JSONDecodeError:
            return None
        return value.get("error") if isinstance(value, dict) else None
    return None


@dataclasses.dataclass
class GraphPOCollection:
    results: list[EpisodeResult]
    training_items: list[tuple[PolicyCompletion, float]]
    diagnostics: dict[str, Any]


async def collect_graphpo(
    task: Task, runner: EpisodeRunner, policy, *, tokenizer,
    summarize: Callable[[list[dict[str, str]]], Awaitable[str]],
    embed: Callable[[list[str]], Awaitable[list[list[float]]]],
    branching: Sequence[int] = (2, 2, 2, 1, 1, 1, 1, 1),
    threshold: float = 0.92, pooling_weight: float = 0.7,
    efficiency_weight: float = 1.0, epsilon: float = 1e-6,
    max_generated_edges: int | None = None,
    max_generated_tokens: int | None = None,
) -> GraphPOCollection:
    if task.benchmark != "cuad" or task.split != "train":
        raise ValueError("GraphPO collection requires a CUAD training task")
    if len(branching) != runner.max_steps or any(type(b) is not int or b < 1 for b in branching):
        raise ValueError("GraphPO needs one positive branching factor per runner decision")
    if runner.cuad_structured_actions:
        raise ValueError("GraphPO replay currently requires ordinary CUAD action generation")
    for name, value in (("max_generated_edges", max_generated_edges), ("max_generated_tokens", max_generated_tokens)):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{name} must be a positive integer when configured")
    started = time.monotonic()
    summary_usage_before, embedding_usage_before = _usage(summarize), _usage(embed)
    summary_calls = summary_source_tokens = summary_text_tokens = embedding_calls = 0
    summary_seconds = embedding_seconds = 0.0

    async def summarize_state(messages):
        nonlocal summary_calls, summary_source_tokens, summary_text_tokens, summary_seconds
        begin = time.monotonic()
        summary = await summarize(copy.deepcopy(messages))
        elapsed = time.monotonic() - begin
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("GraphPO summarizer must return nonempty semantic state text")
        summary_calls += 1
        summary_source_tokens += len(chat_tokens(tokenizer, messages))
        summary_text_tokens += len(tokenizer.encode(summary, add_special_tokens=False))
        summary_seconds += elapsed
        return summary

    async def embed_states(summaries):
        nonlocal embedding_calls, embedding_seconds
        begin = time.monotonic()
        vectors = await embed(summaries)
        embedding_seconds += time.monotonic() - begin
        embedding_calls += 1
        if len(vectors) != len(summaries):
            raise ValueError("GraphPO embedder returned a different number of vectors than summaries")
        return vectors

    graph = GraphPOGraph(threshold=threshold)
    initial = runner.initial_messages(task)
    root_summary = await summarize_state(initial)
    root_vector = (await embed_states([root_summary]))[0]
    graph.add_node(None, root_summary, root_vector)
    node_messages = {0: initial}
    completions: dict[int, PolicyCompletion] = {}
    results: list[EpisodeResult] = []
    terminal_paths = []
    budget_log = []
    expansion_log = []
    decision_records = []
    frontier = [0]
    generated_tokens = generated_prompt_tokens = replayed_decisions = generated_edges = 0
    excluded_expansions = []
    exhaustion = None
    for depth, base in enumerate(branching):
        if not frontier:
            break
        allocation, budget = graph.allocate(frontier, base)
        budget["layer"] = depth + 1
        budget_log.append(budget)
        jobs = [(node, branch) for node in frontier for branch in range(allocation[node])]
        new_frontier = []
        new_nodes = []
        for parent, branch in jobs:
            if max_generated_edges is not None and generated_edges >= max_generated_edges:
                exhaustion = "generated_edge_budget"
                break
            if max_generated_tokens is not None and generated_tokens >= max_generated_tokens:
                exhaustion = "generated_token_budget"
                break
            prefix_ids = graph.path(parent)[1:]
            expansion = await expand_once(task, runner, policy, tokenizer=tokenizer,
                                          prefix=[completions[edge] for edge in prefix_ids])
            replayed_decisions += expansion.replayed_decisions
            if expansion.completion is None:
                excluded_expansions.append({"parent": parent, "branch": branch,
                                            "status": expansion.result.status, "detail": expansion.result.detail})
                results.append(expansion.result)
                continue
            completion = expansion.completion
            generated_edges += 1
            generated_tokens += len(completion.completion_ids)
            generated_prompt_tokens += len(completion.prompt_ids)
            summary = await summarize_state(expansion.next_messages)
            vector = (await embed_states([summary]))[0]
            success = terminal_success(expansion.result) if expansion.result is not None else None
            node = graph.add_node(parent, summary, vector, terminal_success=success,
                                  closed=expansion.result is not None,
                                  status=expansion.result.status if expansion.result is not None else "active")
            completions[node] = completion
            node_messages[node] = expansion.next_messages
            new_nodes.append(node)
            if expansion.result is None:
                new_frontier.append(node)
            else:
                results.append(expansion.result)
                terminal_paths.append({"result_index": len(results) - 1, "node": node,
                                       "edges": prefix_ids + [node], "binary_success": success,
                                       "common_success_reward": None if success is None else 10.0 * success,
                                       "runner_reward": expansion.result.reward,
                                       "runner_metrics": copy.deepcopy(expansion.result.metrics)})
            action = parse_action(completion.text)
            observation_errors = []
            if expansion.result is not None:
                events = expansion.result.tool_events
                observation_errors = [event["observation"]["error"] for event in events
                                      if isinstance(event.get("observation"), dict) and event["observation"].get("error")]
            expansion_log.append({"node": node, "parent": parent, "layer": depth + 1,
                                  "branch": branch, "generated_tokens": len(completion.completion_ids),
                                  "prompt_tokens": len(completion.prompt_ids),
                                  "replayed_decisions": expansion.replayed_decisions,
                                  "seconds": expansion.seconds, "action_parsed": action is not None,
                                  "finish_reason": completion.finish_reason,
                                  "current_action_error": (_latest_observation_error(expansion.next_messages)
                                                           if action is None or "answer" not in action else None),
                                  "observed_errors_on_terminal_path": observation_errors})
            decision_records.append({"edge": node, "prefix_edges": prefix_ids,
                                     "request_messages": expansion.generated_messages,
                                     "completion": dataclasses.asdict(completion)})
        graph.discover_equivalences()
        budget["generated_nodes"] = new_nodes
        budget["actual_new_edges"] = len(new_nodes)
        frontier = new_frontier
        if exhaustion:
            break
    advantages, diagnostic = graph_advantages(graph, pooling_weight=pooling_weight,
                                              efficiency_weight=efficiency_weight, epsilon=epsilon)
    excluded_terminal_edges = {path["edges"][-1] for path in terminal_paths
                               if path["binary_success"] is None}
    trained_edge_ids = [edge for edge in sorted(completions)
                       if edge in advantages and edge not in excluded_terminal_edges]
    training_items = [(completions[edge], advantages[edge]) for edge in trained_edge_ids]
    diagnostic.update({
        "task_id": task.id, "branching": list(branching),
        "sampling": "layerwise independent continuations with equivalence-aware budget reallocation",
        "replay": "CUAD deterministic tool replay; every cached and new prompt checked against exact token IDs",
        "summary_inputs": "full original path messages including observations and final controls; no gold or verifier fields",
        "budget": budget_log, "terminal_paths": terminal_paths, "expansions": expansion_log,
        "decisions": decision_records, "node_messages": node_messages,
        "training_edge_ids": trained_edge_ids, "excluded_terminal_edges": sorted(excluded_terminal_edges),
        "excluded_expansions": excluded_expansions, "exhaustion": exhaustion,
        "open_unfinished_nodes": [node.id for node in graph.nodes if not node.closed and not graph.children[node.id]],
        "limits": {"generated_edges": max_generated_edges, "generated_tokens": max_generated_tokens,
                   "token_limit_semantics": "stop before next decision once reached; last completion may cross the limit"},
        "costs": {"generated_edges": generated_edges, "generated_response_tokens": generated_tokens,
                  "generated_prompt_tokens": generated_prompt_tokens, "replayed_decisions": replayed_decisions,
                  "terminal_rollouts": len(results), "summary_calls": summary_calls,
                  "summary_source_tokens": summary_source_tokens, "summary_text_tokens": summary_text_tokens,
                  "summary_token_count_note": "source/text counts; actual summarizer request usage is reported separately by callback",
                  "summary_call_seconds_sum": summary_seconds, "embedding_calls": embedding_calls,
                  "embedding_call_seconds_sum": embedding_seconds,
                  "summarizer_usage": _usage_delta(summary_usage_before, _usage(summarize)),
                  "embedder_usage": _usage_delta(embedding_usage_before, _usage(embed)),
                  "wall_seconds": time.monotonic() - started},
    })
    return GraphPOCollection(results, training_items, diagnostic)
