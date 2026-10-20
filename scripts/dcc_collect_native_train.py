#!/usr/bin/env python3
"""Fresh native TRAIN collection, followed by a separate closed-journal selector.

No labels argument exists for `generate`. Neither phase reads held-out data.
Runtime output paths and the actual approved source/derivative are caller inputs.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import os
from pathlib import Path
import sys
import uuid

sys.path[:0] = [str(Path(__file__).resolve().parent),
                str(Path(__file__).resolve().parents[1] / "slime")]

from dcc_native_sft import (
    CONTRACT, NativeIdentity, NativeSchemaProfile, WebSchemaProfile, CuadSchemaProfile,
    PROTOCOL, TOTAL_TOKENS, _strict_json, validate_native_record, validate_source_receipt,
    load_native_training_data, _segment_key, identity_from_manifest,
)
from jsonschema import ValidationError
from slime_plugins.dcc.data import Task
from slime_plugins.dcc.episode import EpisodeRunner, ModelProtocolFailure, PolicyCompletion
from slime_plugins.dcc.eval_sampling import EvaluationSampling
from slime_plugins.dcc.native_thinking_eval import NativeThinkingPolicy, NativeTraceJournal
from slime_plugins.dcc.prompt_history import (
    PromptHistoryConfig, add_prompt_history_arguments, prompt_history_from_arguments,
)

VERSION = "native-train-collection-closed-journals-v3"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def json_text(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


class Journal:
    def __init__(self, path):
        self.stream = Path(path).open("x", encoding="utf-8")
        self.count = 0

    def write(self, value):
        self.stream.write(json_text(value) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.count += 1

    def close(self):
        if not self.stream.closed:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()


def write_fresh_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json_text(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            require(bool(line.strip()) and line.endswith("\n"),
                    f"Blank or incomplete journal line {number}")
            yield _strict_json(line)


def load_approved(profile_path, source_root, expected_source, tokenizer):
    receipt = validate_source_receipt(
        _strict_json((Path(source_root) / ".dcc-native-source.json").read_text()), expected_source)
    profile = _strict_json(Path(profile_path).read_text())
    from dcc_review_training import approved_web_profile, bind_profile_source
    web = approved_web_profile(profile, WebSchemaProfile)
    bind_profile_source(profile, source_root)
    value = profile["identity"]
    identity = identity_from_manifest(value, expected_source)
    require(value == identity.manifest(tokenizer), "Approved identity differs from actual local tokenizer")
    require(not web.cached_rows, "Initial collection does not enable cached-row histories")
    cuad = None
    if "cuad" in profile:
        require(set(profile["cuad"]) == {"max_steps", "structured_actions", "schema_version"},
                "Invalid approved CUAD profile")
        cuad = CuadSchemaProfile(**profile["cuad"])
        cuad.validate()
    return profile, identity, NativeSchemaProfile(web, cuad), receipt


def load_public_train(path):
    """Accept a separately prepared public-only TRAIN file, never raw labeled data."""
    tasks, seen = [], set()
    for value in read_jsonl(path):
        require(isinstance(value, dict) and set(value) <= {
            "source", "id", "benchmark", "split", "prompt", "document", "gold", "metadata"},
            "Public TRAIN contains unsupported fields")
        require(value.get("split") == "train" and value.get("gold") is None,
                "Generation requires public TRAIN rows with gold absent/null")
        benchmark = value.get("benchmark")
        require(value.get("source") == ("independent_curriculum" if benchmark == "widesearch" else "cuad_train"),
                "Explicit independent Web TRAIN/CUAD TRAIN provenance required")
        metadata = value.get("metadata", {})
        if benchmark == "widesearch":
            require(isinstance(metadata, dict) and set(metadata) == {"evaluation"}
                    and isinstance(metadata["evaluation"], dict)
                    and set(metadata["evaluation"]) == {"required"},
                    "Web generation metadata may contain only public required columns")
            columns = metadata["evaluation"]["required"]
            require(isinstance(columns, list) and columns and len(columns) == len(set(columns))
                    and all(isinstance(column, str) and column.strip() and column != "citations" for column in columns),
                    "Invalid public answer columns")
        else:
            require(metadata == {}, "CUAD generation must have no label/category metadata")
        task = Task.from_dict({key: copy.deepcopy(item) for key, item in value.items() if key != "source"})
        require(task.id not in seen, "Public TRAIN task IDs must be globally unique")
        seen.add(task.id)
        tasks.append(task)
    require(tasks, "Public TRAIN input is empty")
    return tasks


def check_public_task(task):
    require(task.split == "train" and task.gold is None, "Task must remain public TRAIN without gold")
    if task.benchmark == "cuad":
        require(task.metadata == {}, "CUAD generation must have no label/category metadata")
    else:
        require(isinstance(task.metadata, dict) and set(task.metadata) == {"evaluation"}
                and isinstance(task.metadata["evaluation"], dict)
                and set(task.metadata["evaluation"]) == {"required"},
                "Web generation metadata may contain only public required columns")


def fixed_tasks(tasks, *, max_tasks_per_benchmark, rank, world_size):
    require(type(max_tasks_per_benchmark) is int and max_tasks_per_benchmark > 0
            and type(rank) is int and type(world_size) is int and world_size > 0 and 0 <= rank < world_size,
            "Invalid fixed selection/shard limits")
    selected = []
    for benchmark in ("widesearch", "cuad"):
        selected.extend(sorted((task for task in tasks if task.benchmark == benchmark),
                               key=lambda task: task.id)[:max_tasks_per_benchmark])
    return selected[rank::world_size]


class RecordingPolicy:
    """Record exact caller requests; never rewrite messages or original schemas."""
    def __init__(self, native, journal, *, task_id, rank, episode_id):
        self.native, self.journal = native, journal
        self.task_id, self.rank, self.episode_id = task_id, rank, episode_id
        self.trace_ids = []

    async def generate_structured(self, messages, **kwargs):
        trace_id = uuid.uuid4().hex
        metadata = {"trace_id": trace_id, "task_id": self.task_id,
                    "rank": self.rank, "turn_index": len(self.trace_ids)}
        self.trace_ids.append(trace_id)
        snapshot, options = copy.deepcopy(messages), copy.deepcopy(kwargs)
        self.journal.write({"event": "request", "episode_id": self.episode_id,
                            **metadata, "messages": snapshot, "options": options})
        error, completion = None, None
        try:
            if any(message["role"] == "user" and message["content"].startswith(
                    "Evidence check: final answer rejected.") for message in messages):
                # Runner has no feedback-off option. Stop before another native
                # request rather than remove feedback or train on another prompt.
                raise ModelProtocolFailure("training_unsupported_evidence_feedback",
                    completion=PolicyCompletion(""), protocol=self.native.budget.protocol, trace_id=trace_id)
            completion = await self.native.generate_structured(messages, trace_metadata=metadata, **kwargs)
            require(messages == snapshot and kwargs == options, "Generation mutated the recorded request")
            return completion
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            trace = self.native.trace_index.get(trace_id)
            self.journal.write({"event": "response", "episode_id": self.episode_id, **metadata,
                "native_trace_recorded": trace is not None, "exception_type": error,
                "completion": None if completion is None else {
                    "text": completion.text, "prompt_ids": list(completion.prompt_ids),
                    "completion_ids": list(completion.completion_ids), "finish_reason": completion.finish_reason}})


def make_runner(benchmark, schema_profile, *, timeout_seconds, web_tools_factory=None,
                total_tokens=TOTAL_TOKENS):
    profile = schema_profile.web if benchmark == "widesearch" else schema_profile.cuad
    require(profile is not None, "CUAD tasks require the approved CUAD profile")
    kwargs = {} if web_tools_factory is None else {"web_tools_factory": web_tools_factory}
    return EpisodeRunner(max_steps=profile.max_steps, max_new_tokens=total_tokens,
        temperature=0.0, evaluation_temperature=0.0, initial_chunks=4,
        timeout_seconds=timeout_seconds, auto_read_sources=0,
        web_structured_final=True, web_structured_actions=True, cuad_structured_actions=True,
        web_explicit_review=(benchmark == "widesearch" and schema_profile.web.explicit_review),
        web_navigation_ledger=(benchmark == "widesearch" and schema_profile.web.navigation_ledger),
        web_research_checkpoints=False, web_cached_rows=False, **kwargs)


async def generate(tasks, native, output, *, collection_id, profile, schema_profile,
                   rank=0, world_size=1, concurrency=1, timeout_seconds=900,
                   web_tools_factory=None):
    """Generate one fixed shard; no parameter or branch can load TRAIN labels."""
    require(isinstance(native, NativeThinkingPolicy), "Use the actual NativeThinkingPolicy")
    approved_identity = identity_from_manifest(profile["identity"], native.source_receipt["source_commit"])
    expected_history = (PromptHistoryConfig.from_dict(profile["identity"]["prompt_history"])
                        if "prompt_history" in profile["identity"] else PromptHistoryConfig())
    require(native.prompt_history_config == expected_history,
            "Actual prompt history differs from the explicit collection identity")
    require(native.budget == approved_identity.budget,
            "Actual native phase budgets/sampling differ from approved collection identity")
    require(type(concurrency) is int and concurrency > 0, "Positive request concurrency required")
    for task in tasks:
        check_public_task(task)
    require(len({task.id for task in tasks}) == len(tasks), "Generation task IDs must be unique")
    require(native.source_receipt["source_commit"] == profile["identity"]["source_commit"]
            and native.base.eos_token_ids == profile["identity"]["eos_token_ids"]
            and native.base.max_prompt_tokens == profile["identity"]["max_prompt_tokens"]
            and native.base.max_model_len == profile["identity"]["max_model_len"]
            and native.base.seed == profile["identity"]["generation_seed"]
            and native.base.allow_truncation is False,
            "Actual native runtime differs from approved collection identity")
    require(isinstance(collection_id, str) and collection_id.strip(), "Explicit collection ID required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    requests = Journal(output / f"requests_rank{rank}.jsonl")
    episodes = Journal(output / f"episodes_rank{rank}.jsonl")
    done = False
    try:
        semaphore = asyncio.Semaphore(concurrency)
        async def one(task):
            async with semaphore:
                episode_id = collection_id + ":" + task.id
                recorder = RecordingPolicy(native, requests, task_id=task.id, rank=rank, episode_id=episode_id)
                result, error = None, None
                try:
                    runner = make_runner(task.benchmark, schema_profile,
                        timeout_seconds=timeout_seconds, web_tools_factory=web_tools_factory,
                        total_tokens=native.budget.total_tokens)
                    result = await runner.run(task, recorder, evaluation=True, score_train=False)
                except Exception as exc:
                    error = type(exc).__name__
                episodes.write({"task_id": task.id, "benchmark": task.benchmark, "split": "train",
                    "episode_id": episode_id, "rank": rank, "trace_ids": recorder.trace_ids,
                    "public_task": task.to_dict(), "status": result.status if result else "collector_error",
                    "answer": result.answer if result else [], "exception_type": error,
                    "detail": result.detail if result else "", "gold_visible_to_policy": False})
        await asyncio.gather(*(one(task) for task in tasks))
        done = True
    finally:
        requests.close()
        episodes.close()
        await native.close()  # Also durably closes the actual native trace sink.
    if done:
        write_fresh_json(output / f"generation_closed_rank{rank}.json", {
            "version": VERSION, "collection_id": collection_id, "rank": rank, "world_size": world_size,
            "profile": copy.deepcopy(profile), "source_receipt": copy.deepcopy(native.source_receipt),
            "task_ids": [task.id for task in tasks], "request_events": requests.count,
            "episodes": episodes.count, "native_traces": len(native.traces),
            "generation_journals_closed": True, "labels_loaded": False})


def read_closed(output, *, profile, receipt, world_size):
    """Verify all rank closures/journal joins before selector can open labels."""
    output = Path(output)
    all_episodes, all_requests, all_traces, collection_id = [], {}, {}, None
    seen_tasks = set()
    for rank in range(world_size):
        closed = _strict_json((output / f"generation_closed_rank{rank}.json").read_text())
        require(closed.get("version") == VERSION and closed.get("rank") == rank
                and closed.get("world_size") == world_size and closed.get("profile") == profile
                and closed.get("source_receipt") == receipt
                and closed.get("generation_journals_closed") is True and closed.get("labels_loaded") is False,
                "Generation closure differs from independently approved source/profile")
        collection_id = closed["collection_id"] if collection_id is None else collection_id
        require(closed["collection_id"] == collection_id, "Collection ID differs across shards")
        episodes = list(read_jsonl(output / f"episodes_rank{rank}.jsonl"))
        events = list(read_jsonl(output / f"requests_rank{rank}.jsonl"))
        traces = list(read_jsonl(output / f"native_thinking_traces_rank{rank}.jsonl"))
        require(len(episodes) == closed["episodes"] == len(closed["task_ids"])
                and len(events) == closed["request_events"] and len(traces) == closed["native_traces"],
                "Closed journal counts differ")
        require({episode["task_id"] for episode in episodes} == set(closed["task_ids"]),
                "Closed journal task coverage differs")
        responses, requests = {}, {}
        for event in events:
            require(event["rank"] == rank and event["event"] in {"request", "response"}, "Invalid request owner/event")
            target = requests if event["event"] == "request" else responses
            require(event["trace_id"] not in target, "Repeated request/response event")
            target[event["trace_id"]] = event
        require(requests.keys() == responses.keys(), "Missing request/response closure")
        for trace in traces:
            key = trace["trace_id"]
            require(key not in all_traces and key in requests
                    and responses[key]["native_trace_recorded"] is True, "Unowned or duplicate native trace")
            require(all(trace[name] == requests[key][name] == responses[key][name]
                        for name in ("trace_id", "task_id", "rank", "turn_index")), "Native trace ownership differs")
            all_traces[key] = trace
        require({key for key, response in responses.items() if response["native_trace_recorded"]}
                == {trace["trace_id"] for trace in traces}, "Native trace coverage differs")
        for episode in episodes:
            require(episode["task_id"] not in seen_tasks and episode["rank"] == rank
                    and episode["split"] == "train" and episode["gold_visible_to_policy"] is False,
                    "Invalid or repeated generation episode")
            seen_tasks.add(episode["task_id"])
            public = Task.from_dict(episode["public_task"])
            check_public_task(public)
            require(public.id == episode["task_id"] and public.benchmark == episode["benchmark"],
                    "Recorded public task ownership differs")
            owned = [key for key, request in requests.items() if request["task_id"] == episode["task_id"]]
            require(owned == episode["trace_ids"], "Episode/request order or coverage differs")
            for index, key in enumerate(owned):
                require(requests[key]["episode_id"] == responses[key]["episode_id"] == episode["episode_id"]
                        and requests[key]["turn_index"] == index, "Episode/turn ownership differs")
            all_episodes.append(episode)
        require(not all_requests.keys() & requests.keys(), "Request ID reused across shards")
        all_requests.update({key: (request, responses[key]) for key, request in requests.items()})
    return all_episodes, all_requests, all_traces


def load_train_labels(path, selected):
    labels = {}
    for value in read_jsonl(path):
        require(isinstance(value, dict) and set(value) <= {
            "task_id", "benchmark", "split", "gold", "category", "evaluation"}
            and value.get("split") == "train", "Selector accepts TRAIN labels only")
        key = value["task_id"]
        require(key not in labels, "Duplicate TRAIN label task ID")
        if key in selected:
            require(value["benchmark"] == selected[key], "TRAIN label benchmark differs")
            require(isinstance(value.get("gold"), list), "TRAIN labels require an explicit gold list")
            if value["benchmark"] == "cuad":
                require(all(isinstance(span, str) and span.strip() for span in value["gold"])
                        and isinstance(value.get("category"), str) and value["category"].strip(),
                        "CUAD TRAIN needs source spans and an offline category")
            labels[key] = value
    require(labels.keys() == selected.keys(), "Selected fixed task IDs need exactly matching TRAIN labels")
    return labels


def web_train_evaluation(label, public_task):
    """Complete a required-only TRAIN schema without guessing an entity key.

    Public rollout input intentionally contains only required column names.
    A separate TRAIN label may supply the complete original scoring profile.
    With only those public names, require exact whole-row agreement: all
    required cells form the compound key, with no inferred primary key,
    preprocessing, aliases or judge. This is an explicitly strict TRAIN
    selection proxy, not a change to the held-out WideSearch evaluator.
    """
    public = public_task["metadata"]["evaluation"]
    columns = public["required"]
    require(isinstance(columns, list) and columns and len(columns) == len(set(columns))
            and all(isinstance(column, str) and column.strip() and column != "citations"
                    for column in columns), "Invalid public TRAIN answer columns")
    evaluation = label.get("evaluation", public)
    require(isinstance(evaluation, dict) and evaluation.get("required") == columns,
            "TRAIN scoring columns differ from the public answer schema")
    if set(evaluation) == {"required"}:
        return {"required": copy.deepcopy(columns), "unique_columns": copy.deepcopy(columns),
                "eval_pipeline": {column: {"preprocess": [], "metric": ["exact_match"]}
                                  for column in columns}}
    require("unique_columns" in evaluation and "eval_pipeline" in evaluation,
            "TRAIN scoring requires a complete evaluation profile or only required columns")
    return copy.deepcopy(evaluation)


def training_record(episode, request, trace, label):
    action = _strict_json(trace["final_text"])
    benchmark = episode["benchmark"]
    tools = ({"web_search": "search", "fetch_url": "fetch"} if benchmark == "widesearch" else
             {"search_document": "search", "read_document": "read"})
    stage = "final" if "answer" in action else tools.get(action.get("tool"))
    require(stage is not None, "Unsupported actual action for original training validator")
    record = {"source": "independent_curriculum" if benchmark == "widesearch" else "cuad_train",
              "split": "train", "task_id": episode["task_id"], "stage": stage,
              "messages": copy.deepcopy(request["messages"]), "completion": trace["final_text"]}
    if benchmark == "cuad":
        record.update(benchmark="cuad", grounding_fields=[row["text"] for row in episode["answer"]],
                      metadata={"category": label["category"], "positive": bool(label["gold"]),
                                "plan_id": episode["episode_id"]})
    return record


def select(output, destination, *, labels_path, profile, identity, schema_profile, receipt,
           tokenizer, world_size=1, minimum_score=1.0, maximum_episode_requests=None):
    require(type(world_size) is int and world_size > 0 and not isinstance(minimum_score, bool)
            and math.isfinite(minimum_score) and 0 <= minimum_score <= 1, "Invalid selector limits")
    require(maximum_episode_requests is None or type(maximum_episode_requests) is int
            and maximum_episode_requests > 0, "Positive complete-trajectory request limit required")
    episodes, requests, traces = read_closed(output, profile=profile, receipt=receipt, world_size=world_size)
    # This is intentionally the first labels access, after every rank is closed.
    labels = load_train_labels(labels_path, {episode["task_id"]: episode["benchmark"] for episode in episodes})
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    web, cuad = Journal(destination / "native_web_train.jsonl"), Journal(destination / "native_cuad_train.jsonl")
    audit, decisions = Journal(destination / "audit.jsonl"), Journal(destination / "decisions.jsonl")
    accepted = rejected = 0
    accepted_segments, accepted_phase_requests = set(), set()
    try:
        for episode in episodes:
            label, bundles, score = labels[episode["task_id"]], [], None
            try:
                require(episode["status"] == "completed", "Episode did not complete: " + episode["status"])
                require(maximum_episode_requests is None or len(episode["trace_ids"]) <= maximum_episode_requests,
                        "Completed trajectory exceeds predeclared request limit; never crop it")
                if episode["benchmark"] == "cuad":
                    from slime_plugins.dcc.metrics import cuad_task_metrics
                    score = cuad_task_metrics(episode["task_id"], label["gold"], episode["answer"])
                    measure, score_protocol = score["f1"], "cuad_train_f1"
                else:
                    from slime_plugins.dcc.widesearch_metrics import evaluate_widesearch
                    evaluation = web_train_evaluation(label, episode["public_task"])
                    score = evaluate_widesearch(episode["answer"], label["gold"], evaluation)
                    measure, score_protocol = score["f1_by_item"], "strict_exact_proxy"
                require(measure >= minimum_score, "TRAIN correctness threshold not met")
                for key in episode["trace_ids"]:
                    request, response = requests[key]
                    require(key in traces and response["exception_type"] is None
                            and response["completion"] is not None, "Request lacks a successful actual native response")
                    trace = traces[key]
                    completion = response["completion"]
                    require(completion["text"] == trace["final_text"]
                            and completion["prompt_ids"] == trace["original_prompt_ids"]
                            and completion["completion_ids"] == trace["completion_ids"], "Response/trace differs")
                    record = training_record(episode, request, trace, label)
                    schema = schema_profile.original_schema(record, {"turn": request["turn_index"]})
                    require(request["options"] == {"max_new_tokens": identity.budget.total_tokens, "temperature": 0.0,
                                                  "json_schema": schema}, "Recorded actual request schema/options differ")
                    bundle = {"contract": CONTRACT, "record": record, "identity": identity.manifest(tokenizer),
                        "collector": {"split": "train", "task_input": "public-train-task-without-gold",
                            "gold_visible_to_policy": False, "recorded_messages": True, "synthetic_thought": False,
                            "inserted_close": False, "inserted_eos": False, "episode_id": episode["episode_id"],
                            "turn": request["turn_index"], "rank": request["rank"], "trace_id": key},
                        "native_trace": copy.deepcopy(trace)}
                    validate_native_record(bundle, tokenizer, expected_identity=identity, original_schema=schema,
                                           web_schema_profile=schema_profile.web)
                    bundles.append(bundle)
                require(bundles, "Completed episode has no actual requests")
                segment_keys = [_segment_key(bundle["record"]) for bundle in bundles]
                phase_requests = [phase["request_id"] for bundle in bundles
                                  for phase in bundle["native_trace"]["phases"]]
                require(len(segment_keys) == len(set(segment_keys))
                        and not accepted_segments.intersection(segment_keys), "Repeated original TRAIN segment")
                require(len(phase_requests) == len(set(phase_requests))
                        and not accepted_phase_requests.intersection(phase_requests), "Reused actual phase request")
                for bundle in bundles:
                    (cuad if episode["benchmark"] == "cuad" else web).write(bundle)
                accepted_segments.update(segment_keys)
                accepted_phase_requests.update(phase_requests)
                accepted += 1
                decisions.write({"task_id": episode["task_id"], "accepted": True,
                                 "segments": len(bundles), "score": measure, "score_protocol": score_protocol,
                                 **({"train_evaluation": evaluation} if episode["benchmark"] == "widesearch" else {})})
            except (ValueError, TypeError, KeyError, ValidationError) as exc:
                rejected += 1
                audit.write({"task_id": episode["task_id"], "episode_id": episode["episode_id"],
                             "trace_ids": episode["trace_ids"], "reason": str(exc), "score": score,
                             "episode": episode})
                decisions.write({"task_id": episode["task_id"], "accepted": False, "reason": str(exc)})
    finally:
        for journal in (web, cuad, audit, decisions):
            journal.close()
    if web.count:
        load_native_training_data(destination / "native_web_train.jsonl", tokenizer,
            expected_identity=identity, source_receipt=receipt, schema_resolver=schema_profile.original_schema,
            cuad_path=destination / "native_cuad_train.jsonl" if cuad.count else None)
    summary = {"version": VERSION, "source_receipt": receipt, "accepted_episodes": accepted,
               "rejected_episodes": rejected, "web_segments": web.count, "cuad_segments": cuad.count,
               "minimum_train_score": minimum_score, "all_generation_journals_closed_before_labels": True,
               "exported_files_fully_native_validated": bool(web.count),
               "nonempty_mixed_training_data": bool(web.count and cuad.count),
               "held_out_labels_read": False, "web_selection_score": "strict_exact_proxy",
               "web_required_only_default": "all-required-compound-key-exact-match-v1"}
    if maximum_episode_requests is not None:
        summary["maximum_episode_requests"] = maximum_episode_requests
        summary["whole_trajectory_filter_only"] = True
    write_fresh_json(destination / "selection_summary.json", summary)
    write_fresh_json(destination / "native_train_profile.json", profile)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "select"):
        part = sub.add_parser(command)
        part.add_argument("--source-root", required=True)
        part.add_argument("--expected-source", required=True)
        part.add_argument("--profile", required=True)
        part.add_argument("--model", required=True, help="Approved local derivative/tokenizer")
        part.add_argument("--output-dir", required=True, help="Fresh durable generation directory")
        part.add_argument("--world-size", type=int, default=1)
        if command == "generate":
            add_prompt_history_arguments(part)
            part.add_argument("--public-train-data", required=True)
            part.add_argument("--collection-id", required=True)
            part.add_argument("--max-tasks-per-benchmark", type=int, default=2, help="Fixed pilot default, selected by task ID only")
            part.add_argument("--rank", type=int, default=0)
            part.add_argument("--concurrency", type=int, default=1)
            part.add_argument("--tensor-parallel-size", type=int, default=1)
            part.add_argument("--gpu-memory-utilization", type=float, default=0.85)
            part.add_argument("--vllm-enforce-eager", action="store_true", default=False,
                              help="Disable vLLM compilation/CUDA graphs for this fresh collection")
            part.add_argument("--timeout-seconds", type=float, default=900)
        else:
            part.add_argument("--train-labels", required=True)
            part.add_argument("--destination", required=True)
            part.add_argument("--minimum-train-score", type=float, default=1.0)
            part.add_argument("--maximum-episode-requests", type=int, default=None,
                              help="Reject whole longer trajectories; never truncate or synthesize targets")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    profile, identity, schemas, receipt = load_approved(
        args.profile, args.source_root, args.expected_source, tokenizer)
    if args.command == "select":
        print(json_text(select(args.output_dir, args.destination, labels_path=args.train_labels,
            profile=profile, identity=identity, schema_profile=schemas, receipt=receipt, tokenizer=tokenizer,
            world_size=args.world_size, minimum_score=args.minimum_train_score,
            maximum_episode_requests=args.maximum_episode_requests)))
        return
    history_configuration = prompt_history_from_arguments(args)
    declared_history = (PromptHistoryConfig.from_dict(profile["identity"]["prompt_history"])
                        if "prompt_history" in profile["identity"] else PromptHistoryConfig())
    require(history_configuration == declared_history,
            "Prompt history flags must exactly match the new approved collection profile")
    tasks = fixed_tasks(load_public_train(args.public_train_data), max_tasks_per_benchmark=args.max_tasks_per_benchmark,
                        rank=args.rank, world_size=args.world_size)
    if history_configuration.enabled:
        require(all(task.benchmark == "widesearch" for task in tasks),
                "Web32k fresh collection accepts Web TRAIN only; preserved CUAD requires its approved prior file")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    require(not list(output.glob(f"*rank{args.rank}.*")), "Generation rank output must be fresh")
    from slime_plugins.dcc.vllm_policy import load_vllm_policy
    base = load_vllm_policy(args.model, adapter=None, asynchronous=True,
        max_prompt_tokens=identity.max_prompt_tokens, max_model_len=identity.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization,
        seed=identity.generation_seed, evaluation_sampling=EvaluationSampling(),
        enforce_eager=args.vllm_enforce_eager,
        web_structured_final=True, web_structured_actions=True)
    require(base.eos_token_ids == list(identity.eos_token_ids)
            and identity.manifest(base.tokenizer) == profile["identity"], "Actual engine tokenizer/EOS differs")
    base.allow_truncation = False
    native = NativeThinkingPolicy(base, source_receipt=receipt, budget=identity.budget,
        prompt_history_config=history_configuration,
        trace_sink=NativeTraceJournal(output / f"native_thinking_traces_rank{args.rank}.jsonl"))
    asyncio.run(generate(tasks, native, output, collection_id=args.collection_id, profile=profile,
        schema_profile=schemas, rank=args.rank, world_size=args.world_size, concurrency=args.concurrency,
        timeout_seconds=args.timeout_seconds))


if __name__ == "__main__":
    main()
