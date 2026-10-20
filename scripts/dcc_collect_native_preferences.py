#!/usr/bin/env python3
"""Prospective TRAIN public-final prefix export and genuine native G4 collection.

Generation has no label/reward argument. Export accepts only independently
approved, already-closed raw collection journals, never selected SFT examples.
No model is loaded on import; CLI export/verify load only a local tokenizer.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
from dataclasses import replace
import json
from pathlib import Path
import random
import sys

sys.path[:0] = [str(Path(__file__).resolve().parent),
                str(Path(__file__).resolve().parents[1] / "slime")]

from jsonschema import Draft202012Validator, ValidationError
from dcc_collect_native_train import (
    Journal, check_public_task, json_text, read_closed, read_jsonl, require, write_fresh_json,
)
from dcc_native_dpo_data import (
    PREFERENCE_CONTRACT, PublicFinalPrefix, validate_final_only_schema,
)
from dcc_native_physical import validate_native_physical
from dcc_native_sft import (
    CONTRACT, CuadSchemaProfile, NativeIdentity, NativeSchemaProfile, WebSchemaProfile,
    _json_key, _strict_json, identity_from_manifest, validate_source_receipt,
)
from dcc_review_training import approved_web_profile
from slime_plugins.dcc.data import Task
from slime_plugins.dcc.episode import EpisodeRunner, _WebEvidence, final_turn_message
from slime_plugins.dcc.native_thinking_eval import NativeThinkingPolicy, NativeTraceJournal

VERSION = "native-final-g4-prospective-v1"
FINAL_INTERVENTION = "final-decision-only-v1"
ORIGINAL_ONLY = "original-final-only"
SELECTION = "task-id-order-seeded-shuffle-first-complete-final-per-task-v1"
BENCHMARKS = ("cuad", "widesearch")
OBSERVATION = "Tool observation (source material):\n"
FEEDBACK = "Evidence check: final answer rejected."


def checked_seeds(seeds):
    require(isinstance(seeds, (list, tuple)) and len(seeds) == 4
            and all(type(seed) is int and seed >= 0 for seed in seeds)
            and len(set(seeds)) == 4, "G4 requires exactly four fixed distinct integer seeds")
    return list(seeds)


def checked_benchmarks(benchmarks):
    require(isinstance(benchmarks, (list, tuple)) and bool(benchmarks)
            and all(isinstance(value, str) and value in BENCHMARKS for value in benchmarks)
            and len(set(benchmarks)) == len(benchmarks),
            "Explicit unique public benchmark scope required")
    return [value for value in BENCHMARKS if value in benchmarks]


def select_public_prefixes(eligible, *, benchmarks, maximum_per_benchmark, prefix_selection_seed):
    """Freeze a deterministic public-task sample without answer-quality access."""
    benchmarks = checked_benchmarks(benchmarks)
    require(type(maximum_per_benchmark) is int and maximum_per_benchmark > 0,
            "Positive predeclared per-benchmark prefix limit required")
    require(type(prefix_selection_seed) is int and prefix_selection_seed >= 0,
            "Nonnegative predeclared public prefix-selection seed required")
    selected = []
    for benchmark in benchmarks:
        population = sorted((item for item in eligible if item[0].benchmark == benchmark),
                            key=lambda item: item[0].task_id)
        require(len({item[0].task_id for item in population}) == len(population),
                "Exactly one eligible final prefix per public task required")
        random.Random(prefix_selection_seed).shuffle(population)
        selected.extend(population[:maximum_per_benchmark])
    return selected


def public_record(task, messages, completion):
    return {"source": "cuad_train" if task.benchmark == "cuad" else "independent_curriculum",
            "benchmark": task.benchmark, "split": "train", "task_id": task.id,
            "stage": "final", "messages": copy.deepcopy(messages), "completion": completion}


def collector_owner(request):
    return {"split": "train", "task_input": "public-train-task-without-gold",
            "gold_visible_to_policy": False, "recorded_messages": True,
            "synthetic_thought": False, "inserted_close": False, "inserted_eos": False,
            "episode_id": request["episode_id"], "turn": request["turn_index"],
            "rank": request["rank"], "trace_id": request["trace_id"]}


def public_request_schemas(task, messages, *, turn, schemas):
    """Replay exact public runner state, independent of any current completion.

    Tool observations are trusted actual closed-journal source material. They
    are never downloaded again or supplied by newly generated candidates.
    Earlier final attempts/feedback are forbidden, including rejected answers.
    """
    check_public_task(task)
    require(isinstance(messages, list) and bool(messages)
            and all(isinstance(m, dict) and set(m) == {"role", "content"}
                    and m["role"] in {"system", "user", "assistant"}
                    and isinstance(m["content"], str) and m["content"].strip()
                    and FEEDBACK not in m["content"] for m in messages),
            "Exact public messages without rejected-final feedback required")
    profile = schemas.cuad if task.benchmark == "cuad" else schemas.web
    require(profile is not None, "Approved public benchmark profile required")
    profile.validate()
    require(type(turn) is int and 0 <= turn < profile.max_steps,
            "Current public turn exceeds approved runner budget")
    anchors = EpisodeRunner(max_steps=profile.max_steps, initial_chunks=4).initial_messages(task)
    if task.benchmark == "widesearch" and profile.explicit_review:
        from dcc_review_training import review_request_state
        from slime_plugins.dcc.web_navigation import WEB_NAVIGATION_INSTRUCTIONS
        from slime_plugins.dcc.web_review import WEB_EXPLICIT_REVIEW_INSTRUCTIONS
        anchors[0]["content"] += WEB_EXPLICIT_REVIEW_INSTRUCTIONS + WEB_NAVIGATION_INSTRUCTIONS
        require(messages[:2] == anchors, "Closed public task differs from actual reviewed request anchors")
        original, _ = review_request_state(profile, {
            "source": "independent_curriculum", "split": "train", "task_id": task.id,
            "messages": messages}, {"turn": turn})
        anchor_count = 2
    else:
        require(messages[:len(anchors)] == anchors, "Exact original public task anchors required")
        anchor_count = len(anchors)
        original = None
    evidence, index, history_turn = _WebEvidence(), anchor_count, 0
    from slime_plugins.dcc.cuad_structured import cuad_action_schema, cuad_final_schema
    from slime_plugins.dcc.web_structured import web_action_schema, web_final_schema
    from slime_plugins.dcc.web_review import web_review_schema
    columns = task.metadata["evaluation"]["required"] if task.benchmark == "widesearch" else None
    while index < len(messages) and messages[index]["role"] == "assistant":
        require(history_turn < profile.max_steps - 1 and index + 1 < len(messages),
                "Historical action exceeds approved runner budget")
        action = _strict_json(messages[index]["content"])
        action_schema = (cuad_action_schema() if task.benchmark == "cuad" else
            web_action_schema(columns, evidence.fetched_sources,
                              successful_search=bool(evidence.searches), cached_rows=False))
        if task.benchmark == "widesearch" and profile.explicit_review:
            action_schema = web_review_schema(action_schema)
        Draft202012Validator(action_schema).validate(action)
        allowed_tools = ({"search_document", "read_document"} if task.benchmark == "cuad"
                         else {"web_search", "fetch_url"})
        require(isinstance(action, dict) and "answer" not in action
                and action.get("tool") in allowed_tools,
                "Earlier finals, rejected finals and unsupported historical tools are forbidden")
        observation_message = messages[index + 1]
        suffix = f"\nRemaining turns: {profile.max_steps - history_turn - 1}."
        require(observation_message["role"] == "user"
                and observation_message["content"].startswith(OBSERVATION)
                and observation_message["content"].endswith(suffix),
                "Exact paired actual tool observation/counter required")
        observation = _strict_json(observation_message["content"][len(OBSERVATION):-len(suffix)])
        require(observation_message["content"] == OBSERVATION
                + json.dumps(observation, ensure_ascii=False) + suffix,
                "Actual runner observation serialization differs")
        if task.benchmark == "widesearch":
            evidence.observe(action["tool"], observation)
        history_turn += 1
        index += 2
    require(history_turn == turn, "Actual turn differs from historical public tool count")
    final = (cuad_final_schema() if task.benchmark == "cuad"
             else web_final_schema(columns, evidence.fetched_sources))
    if task.benchmark == "widesearch" and profile.explicit_review:
        final = web_review_schema(final)
        # review_request_state already proves exact ledger/final controls.
    else:
        controls = ([final_turn_message(task, evidence if task.benchmark == "widesearch" else None)]
                    if turn == profile.max_steps - 1 else [])
        require(messages[index:] == controls, "Unexpected history, feedback, or final-turn control")
        original = final if controls else (
            cuad_action_schema() if task.benchmark == "cuad" else
            web_action_schema(columns, evidence.fetched_sources,
                              successful_search=bool(evidence.searches), cached_rows=False))
    validate_final_only_schema(final)
    return original, final, sorted(evidence.fetched_sources)


def source_schema_profile(profile):
    web = approved_web_profile(profile, WebSchemaProfile)
    require(not web.cached_rows, "Prospective collector does not support cached-row source histories")
    cuad = CuadSchemaProfile(**profile["cuad"]) if "cuad" in profile else None
    if cuad is not None:
        cuad.validate()
    return NativeSchemaProfile(web, cuad)


def export_prefixes(closed_source, destination, *, source_profile, source_receipt,
                    tokenizer, candidate_identity: NativeIdentity, candidate_source_receipt,
                    experiment_id, harness_id, seeds, schema_policy=ORIGINAL_ONLY,
                    maximum_per_benchmark=64, world_size=1, benchmarks=BENCHMARKS,
                    prefix_selection_seed=47):
    """Freeze IDs BEFORE prospective candidates; no SFT acceptance/reward filter.

    Only the original public completion's final-action shape and full physical
    integrity determine eligibility. Its answer, original thought, status,
    labels/category/polarity and episode reward never enter generation inputs.
    """
    seeds = checked_seeds(seeds)
    benchmarks = checked_benchmarks(benchmarks)
    require(schema_policy in {ORIGINAL_ONLY, FINAL_INTERVENTION}, "Explicit schema policy required")
    require(type(maximum_per_benchmark) is int and maximum_per_benchmark > 0,
            "Positive predeclared per-benchmark prefix limit required")
    require(type(prefix_selection_seed) is int and prefix_selection_seed >= 0,
            "Nonnegative predeclared public prefix-selection seed required")
    require(all(isinstance(v, str) and v.strip() for v in (experiment_id, harness_id)),
            "Explicit prospective experiment/harness identity required")
    validate_source_receipt(source_receipt, source_profile["identity"]["source_commit"])
    validate_source_receipt(candidate_source_receipt, candidate_identity.source_commit)
    require(candidate_identity.thought_sampling_profile == "thinking",
            "G4 requires an explicit stochastic native thought profile")
    identity_manifest = candidate_identity.manifest(tokenizer)
    identity_manifest.pop("generation_seed")
    schemas = source_schema_profile(source_profile)
    episodes, requests, traces = read_closed(closed_source, profile=source_profile,
        receipt=source_receipt, world_size=world_size)
    # Reject any prior/selected-SFT input: only complete raw joins enter here.
    eligible, audit, eligible_tasks = [], [], set()
    for episode in sorted(episodes, key=lambda row: (row["benchmark"], row["task_id"])):
        task = Task.from_dict(episode["public_task"])
        check_public_task(task)
        if task.benchmark not in benchmarks or task.id in eligible_tasks:
            continue
        for trace_id in episode["trace_ids"]:
            request, response = requests[trace_id]
            row = {"task_id": task.id, "trace_id": trace_id, "eligible": False}
            try:
                trace = traces.get(trace_id)
                require(trace is not None and trace.get("status") == "completed",
                        "Original request has no physically complete native trace")
                action = _strict_json(trace["final_text"])
                require(isinstance(action, dict) and "answer" in action and "tool" not in action,
                        "Original request was not a final decision")
                original, final, fetched = public_request_schemas(task, request["messages"],
                    turn=request["turn_index"], schemas=schemas)
                require(request["options"]["json_schema"] == original,
                        "Closed request schema differs from trusted public runner reconstruction")
                original_manifest = (source_profile["cuad_identity"]
                    if task.benchmark == "cuad" and "cuad_identity" in source_profile
                    else source_profile["identity"])
                original_identity = identity_from_manifest(original_manifest, original_manifest["source_commit"])
                require(original_identity.manifest(tokenizer) == original_manifest,
                        "Original identity differs from approved actual tokenizer")
                require(request["options"] == {"json_schema": original,
                    "max_new_tokens": original_identity.budget.total_tokens, "temperature": 0.0},
                    "Closed original request options differ from approved actual runner")
                original_record = public_record(task, request["messages"], trace["final_text"])
                bundle = {"contract": CONTRACT, "record": original_record,
                    "identity": original_manifest, "collector": collector_owner(request), "native_trace": trace}
                validate_native_physical(bundle, tokenizer, expected_identity=original_identity,
                                         original_schema=original)
                require(response["exception_type"] is None and response["completion"] is not None
                        and response["completion"]["text"] == trace["final_text"]
                        and response["completion"]["prompt_ids"] == trace["original_prompt_ids"]
                        and response["completion"]["completion_ids"] == trace["completion_ids"],
                        "Original actual response differs from its physically complete native trace")
                intervention = original != final
                require(not intervention or schema_policy == FINAL_INTERVENTION,
                        "Original action schema permits tools; explicit final-decision-only intervention required")
                prefix_id = f"{experiment_id}:{task.benchmark}:{task.id}:turn{request['turn_index']}"
                prefix = PublicFinalPrefix.from_dict({
                    "contract": "native-final-public-prefix-v1", "prefix_id": prefix_id,
                    "task_id": task.id, "benchmark": task.benchmark,
                    "source": "cuad_train" if task.benchmark == "cuad" else "independent_curriculum",
                    "split": "train", "harness_id": harness_id,
                    "messages": copy.deepcopy(request["messages"]), "json_schema": final,
                    "native_identity": copy.deepcopy(identity_manifest)})
                lineage = {"prefix_id": prefix_id, "task_id": task.id, "source_trace_id": trace_id,
                    "source_episode_id": episode["episode_id"], "turn": request["turn_index"],
                    "original_json_schema": copy.deepcopy(original), "generation_json_schema": copy.deepcopy(final),
                    "schema_policy": schema_policy, "intervention_applied": intervention,
                    "schema_source": "actual-public-runner-history-final-schema",
                    "fetched_sources": fetched, "source_journals_closed": True,
                    "original_completion_retained": False, "original_reward_used": False,
                    "sft_acceptance_used": False}
                eligible.append((prefix, lineage))
                eligible_tasks.add(task.id)
                row["eligible"] = True
                audit.append(row)
                break  # First eligible final per task, independent of answer quality.
            except (ValueError, KeyError, TypeError) as error:
                row["reason"] = type(error).__name__ + ": " + str(error).split("\n", 1)[0]
                audit.append(row)
            except ValidationError:
                row["reason"] = "Public schema/history validation failed"
                audit.append(row)
    selected = select_public_prefixes(eligible, benchmarks=benchmarks,
        maximum_per_benchmark=maximum_per_benchmark, prefix_selection_seed=prefix_selection_seed)
    require(bool(selected), "No eligible complete public final prefix; generation has not started")
    freeze = {"version": VERSION, "experiment_id": experiment_id, "harness_id": harness_id,
        "schema_policy": schema_policy, "native_identity": identity_manifest,
        "web_explicit_review": (schemas.web.explicit_review
                                and any(prefix.benchmark == "widesearch" for prefix, _ in selected)),
        "source_receipt": copy.deepcopy(candidate_source_receipt), "seeds": seeds,
        "prefix_ids": [prefix.prefix_id for prefix, _ in selected],
        "task_ids": [prefix.task_id for prefix, _ in selected], "selection_policy": SELECTION,
        "maximum_per_benchmark": maximum_per_benchmark, "benchmarks": benchmarks,
        "prefix_selection_seed": prefix_selection_seed, "labels_loaded": False}
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for filename, values in (("prefixes.jsonl", [p.to_dict() for p, _ in selected]),
                             ("lineage.jsonl", [lineage for _, lineage in selected]),
                             ("export_audit.jsonl", audit)):
        journal = Journal(destination / filename)
        try:
            for value in values:
                if filename == "prefixes.jsonl":
                    # The Web review decoder enforces official property order.
                    # Preserve the trusted schema order across the public wire.
                    journal.stream.write(json.dumps(value, ensure_ascii=False,
                        allow_nan=False, separators=(",", ":")) + "\n")
                    journal.count += 1
                else:
                    journal.write(value)
        finally:
            journal.close()
    write_fresh_json(destination / "export_source_approval.json", {
        "version": VERSION, "source_profile": copy.deepcopy(source_profile),
        "source_receipt": copy.deepcopy(source_receipt), "source_world_size": world_size,
        "source_journals_closed": True, "labels_loaded": False,
        "source_completion_or_reward_retained": False})
    # Last write establishes the pre-generation immutable ID/seed declaration.
    write_fresh_json(destination / "freeze.json", freeze)
    return freeze


def load_frozen(directory, tokenizer):
    directory = Path(directory)
    freeze = _strict_json((directory / "freeze.json").read_text())
    require(isinstance(freeze, dict) and set(freeze) == {"version", "experiment_id", "harness_id",
        "schema_policy", "native_identity", "source_receipt", "seeds", "prefix_ids", "task_ids",
        "selection_policy", "maximum_per_benchmark", "labels_loaded", "web_explicit_review",
        "benchmarks", "prefix_selection_seed"}
        and freeze["version"] == VERSION and freeze["labels_loaded"] is False
        and freeze["selection_policy"] == SELECTION
        and freeze["schema_policy"] in {ORIGINAL_ONLY, FINAL_INTERVENTION},
        "Explicit independently approved prospective freeze required")
    checked_seeds(freeze["seeds"])
    require(checked_benchmarks(freeze["benchmarks"]) == freeze["benchmarks"]
            and type(freeze["prefix_selection_seed"]) is int and freeze["prefix_selection_seed"] >= 0
            and type(freeze["maximum_per_benchmark"]) is int and freeze["maximum_per_benchmark"] > 0,
            "Frozen public domain/selection declarations differ")
    require(type(freeze["web_explicit_review"]) is bool, "Explicit Web review decoder declaration required")
    require("generation_seed" not in freeze["native_identity"], "Frozen seed belongs to candidate, not prefix")
    manifest = dict(freeze["native_identity"], generation_seed=freeze["seeds"][0])
    identity = identity_from_manifest(manifest, manifest["source_commit"])
    require(identity.manifest(tokenizer) == manifest and identity.thought_sampling_profile == "thinking",
            "Frozen stochastic identity differs from actual tokenizer")
    validate_source_receipt(freeze["source_receipt"], identity.source_commit)
    prefixes = [PublicFinalPrefix.from_dict(value) for value in read_jsonl(directory / "prefixes.jsonl")]
    require(bool(prefixes) and [p.prefix_id for p in prefixes] == freeze["prefix_ids"]
            and [p.task_id for p in prefixes] == freeze["task_ids"]
            and len(set(freeze["prefix_ids"])) == len(prefixes)
            and len(set(freeze["task_ids"])) == len(prefixes)
            and all(p.benchmark in freeze["benchmarks"] and p.harness_id == freeze["harness_id"]
                    and _strict_json(p.native_identity_json) == freeze["native_identity"] for p in prefixes),
            "Public prefixes differ from the frozen pre-generation identities/order")
    require(all(sum(prefix.benchmark == benchmark for prefix in prefixes)
                <= freeze["maximum_per_benchmark"] for benchmark in freeze["benchmarks"]),
            "Public prefixes exceed frozen per-domain cap")
    return freeze, prefixes, identity


def candidate_request(prefix, freeze, seed, rank):
    turn = sum(message["role"] == "assistant" for message in _strict_json(prefix.messages_json))
    trace_id = f"{freeze['experiment_id']}:{prefix.prefix_id}:seed{seed}"
    return {"trace_id": trace_id, "task_id": prefix.task_id, "rank": rank,
            "turn_index": turn, "episode_id": trace_id, "prefix_id": prefix.prefix_id,
            "generation_seed": seed, "messages": _strict_json(prefix.messages_json),
            "options": {"json_schema": _strict_json(prefix.schema_json)}}


async def generate_g4(prefixes, native, output, *, freeze, identity,
                      rank=0, world_size=1, concurrency=1):
    """Same-seed prefixes may overlap; seed changes await BOTH native phases.

    This deliberately avoids a base.seed race between thought and JSON phases.
    Failures remain genuine journaled attempts and are never retried/repaired.
    """
    require(isinstance(native, NativeThinkingPolicy), "Actual native two-phase policy required")
    require(type(world_size) is int and world_size > 0 and type(rank) is int and 0 <= rank < world_size
            and type(concurrency) is int and concurrency > 0, "Invalid shard/concurrency declaration")
    require(not native.traces and not native.trace_index, "Fresh native trace owner required")
    require(native.budget == identity.budget and native.source_receipt == freeze["source_receipt"]
            and native.base.allow_truncation is False
            and native.base.max_prompt_tokens == identity.max_prompt_tokens
            and native.base.max_model_len == identity.max_model_len
            and native.base.eos_token_ids == list(identity.eos_token_ids),
            "Actual native runtime differs from frozen candidate identity")
    from slime_plugins.dcc.prompt_history import PromptHistoryConfig
    expected_history = (PromptHistoryConfig.from_dict(identity.prompt_history)
                        if identity.prompt_history is not None else PromptHistoryConfig())
    require(native.prompt_history_config == expected_history, "Frozen prompt-history policy differs")
    require((native.base.provenance.get("web_explicit_review", {}).get("enabled") is True)
            is freeze["web_explicit_review"], "Actual Web review decoder differs from frozen public harness")
    manifest = identity.manifest(native.base.tokenizer)
    manifest.pop("generation_seed")
    require(manifest == freeze["native_identity"]
            and [p.prefix_id for p in prefixes] == freeze["prefix_ids"]
            and [p.task_id for p in prefixes] == freeze["task_ids"]
            and all(p.harness_id == freeze["harness_id"]
                    and _strict_json(p.native_identity_json) == manifest for p in prefixes),
            "Unfrozen generation inputs")
    checked_seeds(freeze["seeds"])
    owned = prefixes[rank::world_size]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    requests = Journal(output / f"requests_rank{rank}.jsonl")
    candidates = Journal(output / f"candidates_rank{rank}.jsonl")
    audit = Journal(output / f"audit_rank{rank}.jsonl")
    attempted, complete = [], False
    try:
        semaphore = asyncio.Semaphore(concurrency)
        for seed in freeze["seeds"]:
            native.base.seed = seed
            seeded_identity = replace(identity, generation_seed=seed)
            async def one(prefix):
                async with semaphore:
                    request = candidate_request(prefix, freeze, seed, rank)
                    attempted.append(request["trace_id"])
                    requests.write(dict(request, event="request"))
                    completion, error, valid = None, None, False
                    try:
                        completion = await native.generate_structured(copy.deepcopy(request["messages"]),
                            json_schema=copy.deepcopy(request["options"]["json_schema"]),
                            max_new_tokens=identity.budget.total_tokens, temperature=0.0,
                            trace_metadata={key: request[key] for key in ("trace_id", "task_id", "rank", "turn_index")})
                        trace = native.trace_index[request["trace_id"]]
                        record = {"source": prefix.source, "benchmark": prefix.benchmark,
                            "split": "train", "task_id": prefix.task_id, "stage": "final",
                            "messages": copy.deepcopy(request["messages"]), "completion": completion.text}
                        bundle = {"contract": CONTRACT, "record": record,
                            "identity": seeded_identity.manifest(native.base.tokenizer),
                            "collector": dict(collector_owner(request), generation_seed=seed,
                                prefix_id=prefix.prefix_id, schema_policy=freeze["schema_policy"]),
                            "native_trace": copy.deepcopy(trace)}
                        physical = validate_native_physical(bundle, native.base.tokenizer,
                            expected_identity=seeded_identity, original_schema=request["options"]["json_schema"])
                        require(list(physical.prompt_ids) == completion.prompt_ids
                                and list(physical.completion_ids) == completion.completion_ids,
                                "Actual response differs from physically validated native IDs")
                        candidates.write({"contract": PREFERENCE_CONTRACT,
                            "prefix_id": prefix.prefix_id, "native_bundle": bundle})
                        valid = True
                    except Exception as exc:
                        error = type(exc).__name__
                    finally:
                        trace = native.trace_index.get(request["trace_id"])
                        requests.write({"event": "response", **{key: request[key] for key in (
                            "trace_id", "task_id", "rank", "turn_index", "episode_id", "prefix_id", "generation_seed")},
                            "native_trace_recorded": trace is not None, "exception_type": error,
                            "completion": None if completion is None else {
                                "text": completion.text, "prompt_ids": list(completion.prompt_ids),
                                "completion_ids": list(completion.completion_ids),
                                "finish_reason": completion.finish_reason}})
                        audit.write({key: request[key] for key in (
                            "trace_id", "task_id", "rank", "turn_index", "episode_id", "prefix_id", "generation_seed")}
                            | {"candidate_valid": valid, "exception_type": error, "retried": False})
            await asyncio.gather(*(one(prefix) for prefix in owned))
            # Every call's thought AND JSON is now terminal before seed changes.
        complete = True
    finally:
        requests.close(); candidates.close(); audit.close()
        await native.close()  # Owns durable native trace journal closure.
    if complete:
        write_fresh_json(output / f"generation_closed_rank{rank}.json", {
            "version": VERSION, "freeze": copy.deepcopy(freeze), "rank": rank,
            "world_size": world_size, "prefix_ids": [p.prefix_id for p in owned],
            "attempted_trace_ids": attempted, "request_events": requests.count,
            "candidate_count": candidates.count, "audit_count": audit.count,
            "native_traces": len(native.traces), "generation_journals_closed": True,
            "labels_loaded": False})


def read_closed_g4(output, *, freeze, prefixes, tokenizer, world_size=1, expected_source=None):
    """Read ALL rank closures and exact ownership joins before reward access.

    Caller must supply its independently trusted freeze, not a candidate's
    self-claim. Return all valid envelopes and all actual attempt audits. No
    labels are loaded, no best/worst pair is selected, and no failure is hidden.
    """
    require(type(world_size) is int and world_size > 0, "Positive expected world size required")
    if expected_source is not None:
        require(freeze["native_identity"]["source_commit"] == expected_source,
                "Frozen prospective source differs from caller approval")
    require([p.prefix_id for p in prefixes] == freeze["prefix_ids"], "Frozen public prefix order differs")
    checked_seeds(freeze["seeds"])
    output, all_candidates, all_audit, all_ids = Path(output), [], [], set()
    all_phase_ids = set()
    for rank in range(world_size):
        owned = prefixes[rank::world_size]
        expected = {request["trace_id"]: request for seed in freeze["seeds"] for prefix in owned
                    for request in [candidate_request(prefix, freeze, seed, rank)]}
        closed = _strict_json((output / f"generation_closed_rank{rank}.json").read_text())
        require(closed.get("version") == VERSION and closed.get("freeze") == freeze
                and closed.get("rank") == rank and closed.get("world_size") == world_size
                and closed.get("prefix_ids") == [p.prefix_id for p in owned]
                and len(closed.get("attempted_trace_ids", [])) == len(expected)
                and set(closed["attempted_trace_ids"]) == set(expected)
                and closed.get("generation_journals_closed") is True and closed.get("labels_loaded") is False,
                "Missing/mismatched whole-generation rank closure")
        events = list(read_jsonl(output / f"requests_rank{rank}.jsonl"))
        traces = list(read_jsonl(output / f"native_thinking_traces_rank{rank}.jsonl"))
        candidates = list(read_jsonl(output / f"candidates_rank{rank}.jsonl"))
        audits = list(read_jsonl(output / f"audit_rank{rank}.jsonl"))
        require(len(events) == closed["request_events"] == 2 * len(expected)
                and len(traces) == closed["native_traces"]
                and len(candidates) == closed["candidate_count"]
                and len(audits) == closed["audit_count"] == len(expected), "Closed G4 journal counts differ")
        requests, responses, indexed_traces, indexed_audits, indexed_candidates = {}, {}, {}, {}, {}
        for event in events:
            require(event.get("event") in {"request", "response"}, "Unknown generation event")
            target = requests if event["event"] == "request" else responses
            key = event["trace_id"]
            require(key in expected and key not in target, "Duplicate/unowned candidate event")
            target[key] = event
        require(set(requests) == set(responses) == set(expected), "G4 request/response coverage differs")
        for trace in traces:
            key = trace["trace_id"]
            require(key in expected and key not in indexed_traces
                    and all(trace[field] == expected[key][field] for field in (
                        "trace_id", "task_id", "rank", "turn_index")), "Duplicate/unowned native trace")
            indexed_traces[key] = trace
            for phase in trace.get("phases", []):
                phase_id = phase.get("request_id")
                require(isinstance(phase_id, str) and phase_id.strip()
                        and phase_id not in all_phase_ids,
                        "Repeated actual native phase request across candidates/shards")
                all_phase_ids.add(phase_id)
        for row in audits:
            key = row["trace_id"]
            require(key in expected and key not in indexed_audits and row["retried"] is False
                    and all(row[field] == expected[key][field] for field in (
                        "task_id", "rank", "turn_index", "episode_id", "prefix_id", "generation_seed")),
                    "Duplicate/unowned candidate audit")
            indexed_audits[key] = row
        for envelope in candidates:
            bundle = envelope["native_bundle"]
            key = bundle["collector"]["trace_id"]
            require(envelope.get("contract") == PREFERENCE_CONTRACT
                    and set(envelope) == {"contract", "prefix_id", "native_bundle"}
                    and key in expected and key not in indexed_candidates,
                    "Duplicate/unowned candidate envelope")
            indexed_candidates[key] = envelope
        require(set(indexed_audits) == set(expected), "G4 audit coverage differs")
        for key, request in expected.items():
            response, audit = responses[key], indexed_audits[key]
            require(requests[key] == dict(request, event="request")
                    and all(response[field] == request[field] for field in (
                        "task_id", "rank", "turn_index", "episode_id", "prefix_id", "generation_seed"))
                    and response["native_trace_recorded"] is (key in indexed_traces)
                    and response["exception_type"] == audit["exception_type"], "Exact recorded request/response differs")
            require(type(audit["candidate_valid"]) is bool
                    and audit["candidate_valid"] is (key in indexed_candidates), "Candidate/audit coverage differs")
            if key in indexed_candidates:
                envelope = indexed_candidates[key]
                bundle = envelope["native_bundle"]
                prefix = next(p for p in owned if p.prefix_id == request["prefix_id"])
                manifest = dict(freeze["native_identity"], generation_seed=request["generation_seed"])
                identity = identity_from_manifest(manifest, manifest["source_commit"])
                require(envelope["prefix_id"] == prefix.prefix_id
                        and bundle["record"] == {"source": prefix.source, "benchmark": prefix.benchmark,
                            "split": "train", "task_id": prefix.task_id, "stage": "final",
                            "messages": request["messages"], "completion": bundle["native_trace"]["final_text"]}
                        and bundle["collector"] == dict(collector_owner(request),
                            generation_seed=request["generation_seed"], prefix_id=prefix.prefix_id,
                            schema_policy=freeze["schema_policy"])
                        and bundle["native_trace"] == indexed_traces[key]
                        and response["exception_type"] is None, "Candidate public ownership/physical trace differs")
                physical = validate_native_physical(bundle, tokenizer, expected_identity=identity,
                    original_schema=request["options"]["json_schema"])
                require(response["completion"] is not None
                        and response["completion"]["text"] == physical.public_json
                        and response["completion"]["prompt_ids"] == list(physical.prompt_ids)
                        and response["completion"]["completion_ids"] == list(physical.completion_ids),
                        "Closed response differs from actual native physical sequence")
        require(not all_ids.intersection(expected), "Trace ID reused across ranks")
        all_ids.update(expected)
        all_candidates.extend(candidates); all_audit.extend(audits)
    return {"freeze": copy.deepcopy(freeze), "generation_closed_before_scoring": True,
            "prefixes": prefixes, "candidates": all_candidates,
            "attempts": all_audit, "audits": all_audit,
            "phase_request_ids": sorted(all_phase_ids),
            "closure_receipt": {"version": VERSION, "experiment_id": freeze["experiment_id"],
                "world_size": world_size, "attempt_count": len(all_audit),
                "candidate_count": len(all_candidates), "generation_closed_before_scoring": True,
                "labels_loaded": False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--closed-source", required=True)
    export.add_argument("--source-profile", required=True)
    export.add_argument("--source-receipt", required=True)
    export.add_argument("--source-world-size", type=int, default=1)
    export.add_argument("--candidate-profile", required=True,
        help="Caller-approved {identity,source_receipt}; generation seed replaced by fixed G4 seeds")
    export.add_argument("--experiment-id", required=True)
    export.add_argument("--harness-id", required=True)
    export.add_argument("--schema-policy", choices=[ORIGINAL_ONLY, FINAL_INTERVENTION], default=ORIGINAL_ONLY)
    export.add_argument("--seeds", type=int, nargs=4, required=True)
    export.add_argument("--maximum-per-benchmark", type=int, default=64)
    export.add_argument("--benchmark", choices=BENCHMARKS, action="append",
        help="Repeat for explicit domain scope; default both. Freeze domains separately for different budgets.")
    export.add_argument("--prefix-selection-seed", type=int, default=47)
    export.add_argument("--destination", required=True)
    for command in (export, sub.add_parser("generate"), sub.add_parser("verify-closed")):
        command.add_argument("--model", required=True, help="Approved local tokenizer/derivative")
        if command is not export:
            command.add_argument("--frozen-prefix-dir", required=True)
            command.add_argument("--output-dir", required=True)
            command.add_argument("--world-size", type=int, default=1)
        if command.prog.endswith(" generate"):
            command.add_argument("--rank", type=int, default=0)
            command.add_argument("--concurrency", type=int, default=1)
            command.add_argument("--tensor-parallel-size", type=int, default=1)
            command.add_argument("--gpu-memory-utilization", type=float, default=0.85)
            command.add_argument("--vllm-enforce-eager", action="store_true")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    if args.command == "export":
        source_profile = _strict_json(Path(args.source_profile).read_text())
        candidate_profile = _strict_json(Path(args.candidate_profile).read_text())
        require(set(candidate_profile) == {"identity", "source_receipt"}, "Explicit candidate profile required")
        manifest = candidate_profile["identity"]
        identity = identity_from_manifest(manifest, manifest["source_commit"])
        require(identity.manifest(tokenizer) == manifest, "Candidate profile differs from actual tokenizer")
        freeze = export_prefixes(args.closed_source, args.destination, source_profile=source_profile,
            source_receipt=_strict_json(Path(args.source_receipt).read_text()), tokenizer=tokenizer,
            candidate_identity=identity, candidate_source_receipt=candidate_profile["source_receipt"],
            experiment_id=args.experiment_id, harness_id=args.harness_id, seeds=args.seeds,
            schema_policy=args.schema_policy, maximum_per_benchmark=args.maximum_per_benchmark,
            world_size=args.source_world_size, benchmarks=args.benchmark or BENCHMARKS,
            prefix_selection_seed=args.prefix_selection_seed)
        print(json_text({"version": VERSION, "prefix_count": len(freeze["prefix_ids"]), "labels_loaded": False}))
        return
    freeze, prefixes, identity = load_frozen(args.frozen_prefix_dir, tokenizer)
    if args.command == "verify-closed":
        closed = read_closed_g4(args.output_dir, freeze=freeze, prefixes=prefixes,
                                tokenizer=tokenizer, world_size=args.world_size)
        print(json_text({"version": VERSION, "attempts": len(closed["attempts"]),
            "valid_candidates": len(closed["candidates"]), "generation_closed_before_scoring": True}))
        return
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    require(not list(output.glob(f"*rank{args.rank}.*")), "Fresh candidate rank output required")
    from slime_plugins.dcc.vllm_policy import load_vllm_policy
    from slime_plugins.dcc.eval_sampling import EvaluationSampling
    from slime_plugins.dcc.prompt_history import PromptHistoryConfig
    base = load_vllm_policy(args.model, adapter=None, asynchronous=True,
        max_prompt_tokens=identity.max_prompt_tokens, max_model_len=identity.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization,
        seed=freeze["seeds"][0], evaluation_sampling=EvaluationSampling(),
        enforce_eager=args.vllm_enforce_eager, web_structured_final=True, web_structured_actions=True,
        web_explicit_review=freeze["web_explicit_review"])
    base.allow_truncation = False
    native = NativeThinkingPolicy(base, source_receipt=freeze["source_receipt"], budget=identity.budget,
        prompt_history_config=(PromptHistoryConfig.from_dict(identity.prompt_history)
                               if identity.prompt_history is not None else PromptHistoryConfig()),
        trace_sink=NativeTraceJournal(output / f"native_thinking_traces_rank{args.rank}.jsonl"))
    asyncio.run(generate_g4(prefixes, native, output, freeze=freeze, identity=identity,
        rank=args.rank, world_size=args.world_size, concurrency=args.concurrency))


if __name__ == "__main__":
    main()
