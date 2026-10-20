"""Collect label-independent CUAD TRAIN agendas before attaching supervision.

The planner accepts an exact public projection, never a Task or label. Actions
are selected by a seeded task ID, public query order and actual ranked hits.
Labels may only attach supported final targets after the public journal closes.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "slime"))
from slime_plugins.dcc import EpisodeRunner, PolicyCompletion, Task, load_tasks
from slime_plugins.dcc.episode import final_turn_message
from slime_plugins.dcc.hf_policy import encode_messages
from slime_plugins.dcc.retrieval import _category, document_read_windows, load_retrieval_profile, search_document
from slime_plugins.dcc.tools import DocumentTools
from dcc_prepare_cuad_tools import candidate_queries, _final_target

VERSION = "public-agendas-v2"
INITIAL = "Initial contract retrieval (source material):\n"
OBSERVATION = "Tool observation (source material):\n"
VARIANTS = ("search_read", "search_read_read", "search_search_read")


@dataclass(frozen=True, slots=True)
class PublicCuadTask:
    id: str
    question: str
    document: str

    def task(self):
        return Task(self.id, "cuad", "train", self.question, document=self.document)


def project_public(task):
    if task.benchmark != "cuad" or task.split != "train":
        raise ValueError("Public agendas accept CUAD TRAIN tasks only")
    return PublicCuadTask(task.id, task.prompt, task.document)


def select_public_tasks(tasks, *, max_tasks=0, seed=47):
    """Freeze a task-ID-only subset before any target is attached."""
    if max_tasks < 0 or any(type(task) is not PublicCuadTask for task in tasks):
        raise ValueError("Selection requires public tasks and a nonnegative limit")
    if len({task.id for task in tasks}) != len(tasks):
        raise ValueError("Public task IDs must be unique")
    ordered = sorted(tasks, key=lambda task: (
        hashlib.blake2b(f"public-subset-v1:{seed}:{task.id}".encode(), digest_size=16).digest(),
        task.id))
    return sorted(ordered[:max_tasks] if max_tasks else ordered, key=lambda task: task.id)


def agenda_choice(task_id, seed, pass_index):
    # This tiny ID digest assigns a schedule; no corpus/weight hashing occurs.
    key = hashlib.blake2b(f"{seed}:{pass_index}:{task_id}".encode(), digest_size=8).digest()
    value = int.from_bytes(key, "little")
    return value % 3, (value // 3) % 2


def plan_public_agenda(task, profile, *, seed=47, pass_index=0, tools=None):
    if type(task) is not PublicCuadTask:
        raise TypeError("Planner requires the exact label-free PublicCuadTask projection")
    variant_index, geometry = agenda_choice(task.id, seed, pass_index)
    variant = VARIANTS[variant_index]
    queries = candidate_queries(task.question, profile)
    if not queries:
        raise ValueError("No public query for this task")
    first_query = queries[0]
    second_query = next((query for query in queries[1:] if query != first_query), task.question)
    tools = tools if tools is not None else DocumentTools(task.document)
    initial = search_document(tools, task.question, top_k=8, profile=profile,
                              context_chars=0, max_context_chars=16000)
    messages = EpisodeRunner(max_steps=4, initial_chunks=0).initial_messages(task.task())
    messages.append({"role": "user", "content": INITIAL + json.dumps(initial, ensure_ascii=False)})
    actions, observations = [], []

    def act(tool, arguments):
        action = {"tool": tool, "arguments": arguments}
        if tool == "search_document":
            observation = search_document(tools, profile=profile, context_chars=0,
                                          max_context_chars=16000, **arguments)
        else:
            observation = tools.read_document(**arguments)
        if observation.get("error"):
            raise ValueError("Public agenda produced an actual tool error")
        actions.append(action)
        observations.append(observation)
        return observation

    searched = act("search_document", {"query": first_query, "top_k": 4})
    if variant == "search_search_read" and second_query != first_query:
        searched = act("search_document", {"query": second_query, "top_k": 4})
    hits = searched.get("hits", [])
    if not hits:
        raise ValueError("No real ranked hit for the public reader")
    # Both geometries are pre-existing public hit-derived reader windows.
    window_index = 2 if geometry == 0 else 5
    first_window = document_read_windows(hits[0])[window_index]
    first_read = act("read_document", first_window)
    if variant == "search_read_read":
        windows = [document_read_windows(hit)[window_index] for hit in hits[1:]]
        first_range = (first_read["start"], first_read["end"])
        second_window = next((window for window in windows
                              if (lambda observed: (observed.get("start"), observed.get("end")) != first_range)(tools.read_document(**window))), None)
        if second_window is None and second_query != first_query:
            # A real alternate search replaces an impossible distinct read.
            act("search_document", {"query": second_query, "top_k": 4})
        elif second_window is not None:
            act("read_document", second_window)
    assert len(actions) in {2, 3}
    return {"schema": VERSION, "task_id": task.id, "category": _category(task.question),
            "seed": seed, "pass_index": pass_index, "variant": variant,
            "geometry": "forward12k" if geometry == 0 else "symmetric12k",
            "initial_messages": messages, "actions": actions, "observations": observations}


def agenda_prefixes(plan):
    messages = deepcopy(plan["initial_messages"])
    prefixes = []
    for turn, (action, observation) in enumerate(zip(plan["actions"], plan["observations"], strict=True)):
        completion = json.dumps(action, ensure_ascii=False)
        prefixes.append(("search" if action["tool"] == "search_document" else "read", deepcopy(messages), completion))
        messages.append({"role": "assistant", "content": completion})
        messages.append({"role": "user", "content": OBSERVATION + json.dumps(observation, ensure_ascii=False)
                         + f"\nRemaining turns: {3 - turn}."})
    if len(plan["actions"]) == 3:
        messages.append(final_turn_message(Task(plan["task_id"], "cuad", "train",
                                               plan["initial_messages"][1]["content"], document="public replay")))
    return prefixes, messages


def observed_texts(plan):
    initial = json.loads(plan["initial_messages"][-1]["content"][len(INITIAL):])
    texts = []
    for observation in [initial, *plan["observations"]]:
        texts.extend(hit["text"] for hit in observation.get("hits", []))
        if isinstance(observation.get("text"), str):
            texts.append(observation["text"])
    return texts


def attach_targets(task, plan):
    project_public(task)
    if task.id != plan["task_id"] or task.prompt != plan["initial_messages"][1]["content"]:
        raise ValueError("Frozen agenda and TRAIN target identity disagree")
    if not isinstance(task.gold, list) or any(not isinstance(x, str) or not x.strip() for x in task.gold):
        raise ValueError("TRAIN gold must be a list of nonempty spans")
    texts = observed_texts(plan)
    supported = all(any(span in text for text in texts) for span in task.gold)
    metadata = {"category": _category(task.prompt), "positive": bool(task.gold),
                "teacher_version": 3, "query_policy": VERSION, "plan_id": str(plan["pass_index"]),
                "public_variant": plan["variant"], "public_geometry": plan["geometry"],
                "tool_count": len(plan["actions"]), "final_supported": supported,
                "unsupported_positive": bool(task.gold) and not supported,
                "retrieval_context_chars": 0, "retrieval_max_context_chars": 16000}
    shared = {"source": "cuad_train", "benchmark": "cuad", "split": "train",
              "task_id": task.id, "grounding_fields": list(task.gold), "metadata": metadata}
    prefixes, final_messages = agenda_prefixes(plan)
    records = [{**deepcopy(shared), "stage": stage, "messages": messages, "completion": completion}
               for stage, messages, completion in prefixes]
    if supported:
        records.append({**deepcopy(shared), "stage": "final", "messages": final_messages,
                        "completion": _final_target(task)})
    return records


def fits_record(record, tokenizer, *, max_prompt=16384, max_response=4096, batch_tokens=32768):
    try:
        prompt = encode_messages(tokenizer, record["messages"], max_prompt, allow_truncation=False)
    except ValueError as error:
        if "exceeds its context budget" in str(error):
            return False, "prompt_budget"
        raise
    completion = tokenizer.encode(record["completion"], add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(completion) > max_response:
        return False, "target_budget"
    if len(prompt) + len(completion) > batch_tokens:
        return False, "batch_budget"
    if record["stage"] == "final":
        visible = " ".join(tokenizer.decode(prompt, skip_special_tokens=False).split()).replace('""', '"')
        if any(" ".join(span.split()) not in visible and
               " ".join(json.dumps(span, ensure_ascii=False)[1:-1].split()) not in visible
               for span in record["grounding_fields"]):
            return False, "encoded_evidence"
    return True, None


async def replay_agenda(public_task, plan, profile, records):
    prefixes, final_messages = agenda_prefixes(plan)
    final = next((row["completion"] for row in records if row["stage"] == "final"), '{"answer": []}')
    expected = [*prefixes, ("final", final_messages, final)]
    index = 0

    async def policy(messages, **kwargs):
        nonlocal index
        assert messages == expected[index][1], (public_task.id, index, "actual runner prompt drift")
        completion = expected[index][2]
        index += 1
        return PolicyCompletion(completion)

    runner = EpisodeRunner(max_steps=4, initial_chunks=8, retrieval_profile=profile,
                           retrieval_context_chars=0, retrieval_max_context_chars=16000)
    result = await runner.run(public_task.task(), policy, evaluation=True)
    assert result.status == "completed" and index == len(expected), (public_task.id, result.status)
    assert [event["observation"] for event in result.tool_events] == plan["observations"]
    return index


def distribution(values):
    return {"n": len(values), "min": min(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "max": max(values)} if values else {"n": 0}


def load_trainer():
    path = Path(__file__).with_name("dcc_tool_sft.py")
    spec = importlib.util.spec_from_file_location("dcc_public_agenda_trainer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--retrieval-profile", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--max-tasks", type=int, default=0,
                        help="Task-ID-only fixed subset; zero selects all TRAIN tasks")
    args = parser.parse_args(argv)
    if args.passes < 1:
        raise ValueError("Passes must be positive and frozen before collection")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    tasks = load_tasks(args.data)
    by_id = {task.id: task for task in tasks}
    public = select_public_tasks([project_public(task) for task in tasks],
                                 max_tasks=args.max_tasks, seed=args.seed)
    selection = {"schema": "public-subset-v1", "seed": args.seed,
                 "input_tasks": len(tasks), "max_tasks": args.max_tasks,
                 "selected_task_ids": [task.id for task in public],
                 "labels_used": False, "targets_constructed": False}
    (args.output_dir / "public-task-selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    tasks = [by_id[task.id] for task in public]
    profile = load_retrieval_profile(args.retrieval_profile)
    plans_path = args.output_dir / "public-agendas.jsonl"
    cache = {}
    collected = Counter()
    with plans_path.open("x") as stream:
        for task_index, task in enumerate(tasks):
            public = project_public(task)
            assert public == project_public(replace(task, gold=["label-change probe"], metadata={"private_label": True}))
            if public.document not in cache:
                cache[public.document] = DocumentTools(public.document)
            for pass_index in range(args.passes):
                plan = plan_public_agenda(public, profile, seed=args.seed, pass_index=pass_index, tools=cache[public.document])
                stream.write(json.dumps(plan, ensure_ascii=False, allow_nan=False) + "\n")
                collected[plan["variant"]] += 1
            if (task_index + 1) % 250 == 0:
                print(json.dumps({"event": "public_collection", "tasks": task_index + 1}), flush=True)
    # Close the entire label-free agenda journal before constructing targets.
    frozen = {"schema": VERSION, "agenda_bytes": plans_path.stat().st_size,
              "tasks": len(tasks), "passes": args.passes, "seed": args.seed,
              "variants": dict(collected), "targets_constructed": False,
              "planner_accepts_gold": False, "planner_label_projection_invariance_tasks": len(tasks)}
    (args.output_dir / "public-agenda-freeze.json").write_text(json.dumps(frozen, indent=2) + "\n")
    print(json.dumps({"event": "agendas_frozen_before_targets", **frozen}), flush=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    trainer = load_trainer()
    counts, drops, category_counts = Counter(), Counter(), defaultdict(Counter)
    lengths, initial_hits, read_counts, final_turns = defaultdict(list), defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    eligibility = defaultdict(lambda: defaultdict(Counter))
    sampled_diagnostics = []
    out_path = args.output_dir / "teacher_train.jsonl"
    with plans_path.open() as plans_stream, out_path.open("x") as out:
        for plan_index, line in enumerate(plans_stream):
            plan = json.loads(line)
            task = by_id[plan["task_id"]]
            label = "positive" if task.gold else "negative"
            records = attach_targets(task, plan)
            counts[f"{label}_agendas"] += 1
            supported = any(row["stage"] == "final" for row in records)
            counts[f"{label}_supported_agendas"] += supported
            counts[f"{label}_unsupported_agendas"] += not supported
            kept = []
            for row in records:
                fits, reason = fits_record(row, tokenizer)
                if not fits:
                    drops[f"{label}_{row['stage']}_{reason}"] += 1
                    continue
                trainer.validate_cuad_record(row)
                out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                counts[f"{label}_{row['stage']}_records"] += 1
                category_counts[(label, row["metadata"]["category"])][row["stage"]] += 1
                eligibility[task.id][label][row["stage"]] += 1
                kept.append(row)
            asyncio.run(replay_agenda(project_public(task), plan, profile, records))
            counts["actual_runner_replays"] += 1
            for observation in plan["observations"]:
                if "text" in observation:
                    lengths[label].append(len(observation["text"]))
            initial = json.loads(plan["initial_messages"][-1]["content"][len(INITIAL):])
            initial_hits[label][len(initial["hits"])] += 1
            read_counts[label][sum(action["tool"] == "read_document" for action in plan["actions"])] += 1
            final_turns[label][len(plan["actions"]) + 1] += 1
            if supported:
                sampled_diagnostics.append({"task_id": task.id, "positive": bool(task.gold),
                    "category": _category(task.prompt), "variant": plan["variant"],
                    "tool_count": len(plan["actions"]), "geometry": plan["geometry"],
                    "initial_hits": len(initial["hits"]),
                    "read_chars": sum(len(o.get("text", "")) for o in plan["observations"]),
                    "final_record_kept": any(row["stage"] == "final" for row in kept)})
            if (plan_index + 1) % 250 == 0:
                print(json.dumps({"event": "targets_and_real_runner_replay", "agendas": plan_index + 1, "counts": dict(counts)}), flush=True)
    rank_integrity = []
    ids = sorted(eligibility)
    for rank in range(8):
        local = defaultdict(Counter)
        for task_id in ids[rank::8]:
            label = "positive" if by_id[task_id].gold else "negative"
            for stage, count in eligibility[task_id][label].items():
                local[(label, _category(by_id[task_id].prompt))][stage] += count
        incomplete = [{"label": label, "category": category, "stages": dict(stages)}
                      for (label, category), stages in local.items() if set(stages) != {"search", "read", "final"}]
        rank_integrity.append({"rank": rank, "category_label_groups": len(local), "incomplete_groups": incomplete})
    report = {"schema": VERSION, "status": "constructed; pending actual eight-rank sampler preflight",
              "train_only": True, "official_evaluation_read": False, "gpu_used": False,
              "planner_journal_closed_before_targets": True, "public_freeze": frozen,
              "counts": dict(counts), "budget_drops": dict(drops),
              "unique_tasks_retained": len(eligibility),
              "read_text_chars": {label: distribution(v) for label, v in lengths.items()},
              "initial_hit_counts": {label: dict(v) for label, v in initial_hits.items()},
              "read_counts": {label: dict(v) for label, v in read_counts.items()},
              "scheduled_final_model_turns": {label: dict(v) for label, v in final_turns.items()},
              "category_counts": [{"label": k[0], "category": k[1], "stages": dict(v)} for k, v in sorted(category_counts.items())],
              "rank_stage_integrity": rank_integrity,
              "candidate": {"fresh_adapter": True, "rank": 16, "learning_rate": 1e-5, "steps": 128,
                            "world_size": 8, "samples_per_step": 4, "web_fraction": 0.5,
                            "negative_fraction": 1 / 3, "objective_unchanged": True,
                            "optimizer_unchanged": True, "trainer_source_unchanged": True,
                            "checkpoint_steps": [32, 64, 128], "gpu_launch_authorized_by_this_script": False}}
    (args.output_dir / "construction-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "final-state-feature-audit.json").write_text(json.dumps(sampled_diagnostics) + "\n")
    print(json.dumps({"event": "construction_complete", "counts": dict(counts), "budget_drops": dict(drops),
                      "incomplete_rank_groups": sum(len(x["incomplete_groups"]) for x in rank_integrity)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
