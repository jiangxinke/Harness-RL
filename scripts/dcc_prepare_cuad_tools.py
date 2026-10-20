"""Prepare grounded CUAD training-only search/read/final teacher records.

Labels select successful plans and final targets. Candidate queries depend only
on the question and a fitted training profile; read windows are anchored to
real search hits. No model generation or GPU allocation is needed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "slime"))

from slime_plugins.dcc import EpisodeRunner, Task, load_tasks
from slime_plugins.dcc.episode import FINAL_TURN, final_turn_message
from slime_plugins.dcc.retrieval import (
    METHOD,
    VERSION,
    RetrievalProfile,
    _category,
    document_read_windows,
    load_retrieval_profile,
    search_document,
)
from slime_plugins.dcc.tools import DocumentTools

INITIAL = "Initial contract retrieval (source material):\n"
OBSERVATION = "Tool observation (source material):\n"


def _normalize(value: str) -> str:
    return " ".join(value.split())


def candidate_queries(question: str, profile: RetrievalProfile | None) -> list[str]:
    """The complete query whitelist; this function cannot access task labels."""
    category = _category(question)
    terms = profile.questions.get(_normalize(question), ()) if profile else ()
    expanded = " ".join(terms)
    candidates = [
        category, f"{category} {expanded}", expanded,
        question.split("Details:", 1)[-1], question,
        f"{category} {' '.join(terms[:5])}",
        f"{category} {' '.join(terms[5:])}",
        " ".join(terms[::2]), " ".join(terms[1::2]),
    ]
    return list(dict.fromkeys(_normalize(query) for query in candidates if query.strip()))


def read_actions(hit: dict[str, Any]) -> list[dict[str, int]]:
    """Public hit-anchored windows, including the preceding clause boundary."""
    return document_read_windows(hit)


def _texts(observation: dict[str, Any]) -> list[str]:
    return [hit["text"] for hit in observation.get("hits", [])] + ([observation["text"]] if observation.get("text") else [])


def _visible(span: str, texts: list[str]) -> bool:
    # A target must occur inside an actual observation, rather than a synthetic
    # concatenation of disjoint excerpts or a private label-bearing prompt.
    return any(span in text for text in texts)


def _action(tool: str, arguments: dict[str, Any]) -> str:
    return json.dumps({"tool": tool, "arguments": arguments}, ensure_ascii=False)


def _observation(value: dict[str, Any], remaining: int) -> dict[str, str]:
    return {"role": "user", "content": OBSERVATION + json.dumps(value, ensure_ascii=False) + f"\nRemaining turns: {remaining}."}


def _fits(records: list[dict[str, Any]], tokenizer, max_prompt: int, max_response: int, batch_tokens: int) -> bool:
    for record in records:
        prompt = tokenizer.apply_chat_template(record["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if hasattr(prompt, "keys"):
            prompt = prompt["input_ids"]
        response = tokenizer.encode(record["completion"], add_special_tokens=False)
        if len(prompt) > max_prompt or len(response) + 1 > max_response or len(prompt) + len(response) + 1 > batch_tokens:
            return False
        if record["stage"] == "final":
            visible = _normalize(tokenizer.decode(prompt, skip_special_tokens=False))
            if any(_normalize(span) not in visible and _normalize(json.dumps(span, ensure_ascii=False)[1:-1]) not in visible for span in record["grounding_fields"]):
                return False
    return True


def _final_target(task: Task) -> str:
    """Use public question text for the plan, and train labels only in the target."""
    category = " ".join(_category(task.prompt).split()[:20])
    plan = f"Check {category} against the original question's conditions; extract qualifying verbatim spans."
    checks = (
        "Observed contract text supports the requested category and conditions; the answer contains the qualifying verbatim spans."
        if task.gold else
        "Observed contract text does not establish a clause meeting the requested category and conditions; return an empty answer."
    )
    return json.dumps({"plan": plan, "checks": checks,
                       "answer": [{"text": span, "confidence": 0.9} for span in task.gold]}, ensure_ascii=False)


def _records(task: Task, initial: dict, query: str, searched: dict, arguments: dict, read: dict, profile: RetrievalProfile | None,
             retrieval_context_chars: int, retrieval_max_context_chars: int, episode_max_steps: int = 3) -> list[dict[str, Any]]:
    messages = EpisodeRunner(max_steps=episode_max_steps, initial_chunks=0).initial_messages(task)
    messages.append({"role": "user", "content": INITIAL + json.dumps(initial, ensure_ascii=False)})
    initial_seen = [_visible(span, _texts(initial)) for span in task.gold]
    shared = {
        "source": "cuad_train", "benchmark": "cuad", "split": "train",
        "task_id": task.id, "grounding_fields": list(task.gold),
        "metadata": {
            "category": _category(task.prompt), "positive": bool(task.gold),
            "initial_visibility": ("partial" if any(initial_seen) else "none") if task.gold else "negative",
            "profile_terms": list(profile.questions.get(_normalize(task.prompt), ())) if profile else [],
            "query_policy": "question_category_train_profile_only",
            "teacher_version": 2,
            "retrieval_context_chars": retrieval_context_chars,
            "retrieval_max_context_chars": retrieval_max_context_chars,
        },
    }
    output = []
    for stage, completion, observation, remaining in [
        ("search", _action("search_document", {"query": query, "top_k": 4}), searched, episode_max_steps - 1),
        ("read", _action("read_document", arguments), read, episode_max_steps - 2),
        ("final", _final_target(task), None, 0),
    ]:
        # The third model call is forced-final only for a three-turn episode.
        # Longer episodes may finish normally after the same real search/read.
        if stage == "final" and episode_max_steps == 3:
            messages.append(final_turn_message(task))
        output.append({**deepcopy(shared), "stage": stage, "messages": deepcopy(messages), "completion": completion})
        if observation is not None:
            messages.append({"role": "assistant", "content": completion})
            messages.append(_observation(observation, remaining))
    return output


def prepare_records(
    tasks: list[Task], profile: RetrievalProfile | None, *, tokenizer,
    episodes: int = 96, negative_episodes: int = 0, seed: int = 42, initial_chunks: int = 4,
    retrieval_context_chars: int = 0, retrieval_max_context_chars: int = 16000,
    max_prompt_tokens: int = 8192, max_response_tokens: int = 1024,
    batch_tokens: int = 12288, episode_max_steps: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select hidden positives and optional negatives with real tool traces."""
    if type(episode_max_steps) is not int or episode_max_steps < 3:
        raise ValueError("episode_max_steps must be an integer of at least 3 for search, read and final")
    if episodes < 0 or negative_episodes < 0 or episodes + negative_episodes < 1:
        raise ValueError("episode counts must be nonnegative and request at least one episode")
    if min(initial_chunks, max_prompt_tokens, max_response_tokens, batch_tokens) < 1:
        raise ValueError("retrieval and token budgets must be positive")
    if type(retrieval_context_chars) is not int or retrieval_context_chars < 0 or type(retrieval_max_context_chars) is not int or retrieval_max_context_chars < 1:
        raise ValueError("retrieval context must be nonnegative with a positive character budget")
    ids: set[str] = set()
    for task in tasks:
        if task.benchmark != "cuad" or task.split != "train":
            raise ValueError("CUAD teachers accept CUAD train tasks only")
        if task.id in ids:
            raise ValueError("duplicate CUAD training task id")
        ids.add(task.id)
        if not isinstance(task.gold, list) or any(not isinstance(span, str) or not span.strip() for span in task.gold):
            raise ValueError("CUAD gold must be a list of nonempty source spans")
    rng = random.Random(seed)
    grouped: dict[str, list[Task]] = defaultdict(list)
    for task in sorted(tasks, key=lambda value: value.id):
        if task.gold:
            grouped[_category(task.prompt)].append(task)
    for values in grouped.values():
        rng.shuffle(values)
    cache: dict[str, DocumentTools] = {}
    counters: Counter = Counter()
    emitted: Counter = Counter()
    output = []
    while grouped and sum(emitted.values()) < episodes:
        for category in sorted(grouped):
            if sum(emitted.values()) >= episodes:
                break
            task = grouped[category].pop()
            if not grouped[category]:
                del grouped[category]
            counters["positive_tasks_considered"] += 1
            if task.document not in cache:
                cache[task.document] = DocumentTools(task.document)
            tools = cache[task.document]
            initial = search_document(tools, task.prompt, top_k=initial_chunks, profile=profile,
                                      context_chars=retrieval_context_chars, max_context_chars=retrieval_max_context_chars)
            missing = [span for span in task.gold if not _visible(span, _texts(initial))]
            if not missing:
                counters["already_fully_visible"] += 1
                continue
            counters["hidden_positive_tasks_considered"] += 1
            plans = []
            for query_index, query in enumerate(candidate_queries(task.prompt, profile)):
                searched = search_document(tools, query, top_k=4, profile=profile,
                                           context_chars=retrieval_context_chars, max_context_chars=retrieval_max_context_chars)
                for hit_index, hit in enumerate(searched["hits"]):
                    seen_reads = set()
                    for window_index, arguments in enumerate(read_actions(hit)):
                        read = tools.read_document(**arguments)
                        read_key = (read["start"], read["end"])
                        if read_key in seen_reads:
                            continue
                        seen_reads.add(read_key)
                        if not any(_visible(span, _texts(read)) for span in missing):
                            continue
                        observed = _texts(initial) + _texts(searched) + _texts(read)
                        if not all(_visible(span, observed) for span in task.gold):
                            continue
                        # Labels choose among public-query/fixed-window plans;
                        # they never create query terms or absolute offsets.
                        records = _records(task, initial, query, searched, arguments, read, profile,
                                           retrieval_context_chars, retrieval_max_context_chars, episode_max_steps)
                        cost = sum(len(message["content"]) for message in records[-1]["messages"])
                        plans.append((cost, query_index, hit_index, window_index, records))
            accepted = None
            for _, _, _, _, records in sorted(plans, key=lambda plan: plan[:4]):
                if _fits(records, tokenizer, max_prompt_tokens, max_response_tokens, batch_tokens):
                    accepted = records
                    break
            if accepted is None:
                counters["no_grounded_plan_fits_budgets"] += 1
                continue
            output.extend(accepted)
            emitted[category] += 1
    # Keep positive selection and its RNG independent of this optional stage.
    # Empty gold is a training label, never an inference shortcut: negatives
    # still execute search and a hit-anchored read before returning no spans.
    negative_grouped: dict[str, list[Task]] = defaultdict(list)
    negative_rng = random.Random(seed)
    for task in sorted(tasks, key=lambda value: value.id):
        if not task.gold:
            negative_grouped[_category(task.prompt)].append(task)
    for values in negative_grouped.values():
        negative_rng.shuffle(values)
    negative_emitted: Counter = Counter()
    while negative_grouped and sum(negative_emitted.values()) < negative_episodes:
        for category in sorted(negative_grouped):
            if sum(negative_emitted.values()) >= negative_episodes:
                break
            task = negative_grouped[category].pop()
            if not negative_grouped[category]:
                del negative_grouped[category]
            counters["negative_tasks_considered"] += 1
            if task.document not in cache:
                cache[task.document] = DocumentTools(task.document)
            tools = cache[task.document]
            initial = search_document(tools, task.prompt, top_k=initial_chunks, profile=profile,
                                      context_chars=retrieval_context_chars, max_context_chars=retrieval_max_context_chars)
            plans = []
            for query_index, query in enumerate(candidate_queries(task.prompt, profile)):
                searched = search_document(tools, query, top_k=4, profile=profile,
                                           context_chars=retrieval_context_chars, max_context_chars=retrieval_max_context_chars)
                for hit_index, hit in enumerate(searched["hits"]):
                    seen_reads = set()
                    for window_index, arguments in enumerate(read_actions(hit)):
                        read = tools.read_document(**arguments)
                        read_key = (read["start"], read["end"])
                        if read_key in seen_reads or not read.get("text", "").strip():
                            continue
                        seen_reads.add(read_key)
                        records = _records(task, initial, query, searched, arguments, read, profile,
                                           retrieval_context_chars, retrieval_max_context_chars, episode_max_steps)
                        cost = sum(len(message["content"]) for message in records[-1]["messages"])
                        plans.append((cost, query_index, hit_index, window_index, records))
            accepted = None
            for _, _, _, _, records in sorted(plans, key=lambda plan: plan[:4]):
                if _fits(records, tokenizer, max_prompt_tokens, max_response_tokens, batch_tokens):
                    accepted = records
                    break
            if accepted is None:
                counters["no_negative_tool_plan_fits_budgets"] += 1
                continue
            output.extend(accepted)
            negative_emitted[category] += 1
    categories = emitted + negative_emitted
    summary = {
        "source": "cuad_train", "benchmark": "cuad", "split": "train",
        "teacher_version": 2, "requested_episodes": episodes + negative_episodes,
        "requested_positive_episodes": episodes, "requested_negative_episodes": negative_episodes,
        "episodes": sum(categories.values()), "records": len(output),
        "positive_episodes": sum(emitted.values()), "negative_episodes": sum(negative_emitted.values()),
        "categories": dict(sorted(categories.items())), "counters": dict(counters),
        "positive_categories": dict(sorted(emitted.items())),
        "negative_categories": dict(sorted(negative_emitted.items())),
        "training_tasks": len(tasks), "seed": seed, "initial_chunks": initial_chunks,
        "episode_max_steps": episode_max_steps,
        "retrieval_context_chars": retrieval_context_chars,
        "retrieval_max_context_chars": retrieval_max_context_chars,
        "profile_method": METHOD if profile else None,
        "profile_version": VERSION if profile else None,
        "max_prompt_tokens": max_prompt_tokens, "max_response_tokens": max_response_tokens,
        "batch_tokens": batch_tokens, "encoding_verified": True,
        "query_policy": "question_category_train_profile_only",
        "read_policy": "search_hit_anchored_forward_and_symmetric_bounded_windows_v2",
    }
    return output, summary


def main(argv: list[str] | None = None, *, tokenizer=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--retrieval-profile", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=96, help="Hidden positive episodes to prepare")
    parser.add_argument("--negative-episodes", type=int, default=0,
                        help="Additional category-balanced train negatives; use --episodes 0 for negatives only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-chunks", type=int, default=4)
    parser.add_argument("--episode-max-steps", type=int, default=3,
                        help="Runtime turn budget, at least 3; larger budgets allow a normal third-turn final")
    parser.add_argument("--retrieval-context-chars", type=int, default=0)
    parser.add_argument("--retrieval-max-context-chars", type=int, default=16000)
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    parser.add_argument("--max-response-tokens", type=int, default=1024)
    parser.add_argument("--batch-tokens", type=int, default=12288)
    args = parser.parse_args(argv)
    manifest = args.output.with_suffix(args.output.suffix + ".manifest.json")
    if args.output.exists() or manifest.exists():
        raise FileExistsError("teacher output or manifest already exists")
    profile = load_retrieval_profile(args.retrieval_profile)
    tasks = [task for path in args.data for task in load_tasks(path)]
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    records, summary = prepare_records(
        tasks, profile, tokenizer=tokenizer, episodes=args.episodes,
        negative_episodes=args.negative_episodes, seed=args.seed, episode_max_steps=args.episode_max_steps,
        initial_chunks=args.initial_chunks, max_prompt_tokens=args.max_prompt_tokens,
        retrieval_context_chars=args.retrieval_context_chars,
        retrieval_max_context_chars=args.retrieval_max_context_chars,
        max_response_tokens=args.max_response_tokens, batch_tokens=args.batch_tokens,
    )
    if not records:
        raise RuntimeError("No requested CUAD train task produced a grounded tool trajectory")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
