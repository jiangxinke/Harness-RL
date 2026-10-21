"""Public rejected-final replay with synthetic CPU tools and native responses.

The real runner, recording guard, cache publisher and closed-journal reader run.
Minimal fake traces assert ownership only, never native physical eligibility.
No models, tokenizers, network, real labels or benchmark files are accessed.
"""
from copy import deepcopy
import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from jsonschema import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "slime")]
import dcc_collect_native_train as C
import dcc_native_sft as N
import dcc_review_training as R
import dcc_train_cached_rows as W
from slime_plugins.dcc.data import Task
from slime_plugins.dcc.episode import PolicyCompletion
from slime_plugins.dcc.prompt_history import PromptHistoryConfig
from slime_plugins.dcc.tools import WebTools

URL = "https://example.org/public.csv"
TEXT = "Name\nAlpha\n"
RECEIPT = {"version": "synthetic-public-replay", "source_commit": "f" * 40}


def action(tool=None):
    result = {"plan": "Read the public list.", "checks": "Copy only observed names."}
    result.update({"tool": tool, "arguments": {"query": "public names"} if tool == "web_search"
                   else {"url": URL}} if tool else {"answer": []})
    return result


class Tools(WebTools):
    async def web_search(self, query, **kwargs):
        return {"results": [{"url": URL, "title": "Public names"}]}

    async def fetch_url(self, url, **kwargs):
        self.pages[URL] = TEXT
        return await super().fetch_url(url, **kwargs)


class SyntheticNative:
    """No engine or tokenizer: only an invented successful response ownership."""
    budget = SimpleNamespace(total_tokens=20480, protocol="synthetic-native-only")

    def __init__(self, values, journal):
        self.values, self.journal = list(values), journal
        self.trace_index = {}

    async def generate_structured(self, messages, *, trace_metadata, **options):
        value = self.values.pop(0)
        text = json.dumps(value)
        trace = {**trace_metadata, "final_text": text, "json_schema": options["json_schema"]}
        self.trace_index[trace_metadata["trace_id"]] = trace
        self.journal.write(trace)
        return PolicyCompletion(text, [10], [20])


def approved_profile(version):
    cached = version in R.CACHED_TRAIN_PROFILES
    web = {"max_steps": 20, "explicit_review": True, "cached_rows": cached,
           "navigation_ledger": True}
    if cached:
        web["cached_rows_configuration"] = W.configuration()
    thought = 8192 if version == R.WEB32K_CACHED_8K_PROFILE else 16384
    identity = {"max_prompt_tokens": 32768, "max_model_len": 32768 + thought + 4096,
                "thought_tokens": thought, "thought_sampling_profile": "thinking"}
    if version != R.REVIEW20_PROFILE:
        identity["prompt_history"] = PromptHistoryConfig(True).to_dict()
    return {"version": version, "collection_overlay_id": "synthetic-feedback-replay",
            "identity": identity, "web": web}


def collect_rank(output, approved, rank, *, world=1, early_turn=16):
    web = R.approved_web_profile(approved, N.WebSchemaProfile)
    task = Task(f"synthetic-public-{rank}", "widesearch", "train", "List the public names.",
                metadata={"evaluation": {"required": ["Name"]}})
    episode_id = "synthetic-collection:" + task.id
    requests = C.Journal(output / f"requests_rank{rank}.jsonl")
    traces = C.Journal(output / f"native_thinking_traces_rank{rank}.jsonl")
    sources = C.Journal(output / f"web_cached_sources_rank{rank}.jsonl") if web.cached_rows else None
    values = [action("web_search"), action("fetch_url")]
    values += [action("web_search") for _ in range(early_turn - 2)] + [action()]
    native = SyntheticNative(values, traces)
    recorder = C.RecordingPolicy(native, requests, task_id=task.id, rank=rank, episode_id=episode_id)
    tools = Tools()
    if sources is not None:
        publisher = W.CachePublisher(sources, task_id=task.id, episode_id=episode_id,
                                     source_receipt=RECEIPT, profile_version=approved["version"])
        publisher.attach(tools)
        recorder.web_cache_publisher = publisher
    runner = C.make_runner("widesearch", N.NativeSchemaProfile(web), timeout_seconds=30,
                           web_tools_factory=lambda: tools, total_tokens=native.budget.total_tokens)
    try:
        result = asyncio.run(runner.run(task, recorder, evaluation=True, score_train=False))
    finally:
        requests.close()
        traces.close()
        if sources is not None:
            sources.close()
    assert result.status == "model_protocol_failure"
    assert result.detail == "training_unsupported_evidence_feedback"
    episode = {"task_id": task.id, "benchmark": "widesearch", "split": "train",
               "episode_id": episode_id, "rank": rank, "trace_ids": recorder.trace_ids,
               "public_task": task.to_dict(), "status": result.status, "answer": result.answer,
               "exception_type": None, "detail": result.detail, "gold_visible_to_policy": False}
    with (output / f"episodes_rank{rank}.jsonl").open("w") as stream:
        stream.write(C.json_text(episode) + "\n")
    closed = {"version": C.VERSION, "collection_id": "synthetic-collection", "rank": rank,
              "world_size": world, "profile": approved, "source_receipt": RECEIPT,
              "task_ids": [task.id], "request_events": requests.count, "episodes": 1,
              "native_traces": traces.count, "generation_journals_closed": True, "labels_loaded": False}
    if sources is not None:
        closed.update(web_cached_rows=W.configuration(), web_cached_source_records=sources.count)
    C.write_fresh_json(output / f"generation_closed_rank{rank}.json", closed)
    rows = list(C.read_jsonl(output / f"requests_rank{rank}.jsonl"))
    before, after = [row for row in rows if row["event"] == "request"][-2:]
    response = rows[-1]
    assert after["turn_index"] == early_turn + 1
    assert response["exception_type"] == "ModelProtocolFailure"
    assert response["native_trace_recorded"] is False and response["completion"] is None
    assert response["trace_id"] not in native.trace_index and not native.values
    assert task.gold is None and task.metadata == {"evaluation": {"required": ["Name"]}}
    return web, before, after, response


def replay_request(web, request, registry=None):
    record = {"source": "independent_curriculum", "split": "train",
              "task_id": request["task_id"], "messages": request["messages"]}
    owner = {"episode_id": request["episode_id"], "turn": request["turn_index"]}
    if web.cached_rows:
        owner["web_cached_rows"] = request["web_cached_rows"]
    return R.review_request_state(web, record, owner, cached_sources=registry)


@pytest.mark.parametrize("version", [R.REVIEW20_PROFILE, R.WEB32K_PROFILE,
                                     R.WEB32K_CACHED_PROFILE, R.WEB32K_CACHED_8K_PROFILE])
@pytest.mark.parametrize("early_turn", [2, 18])
def test_real_guard_feedback_replays_exact_schema_turn_navigation_and_cache(tmp_path, version, early_turn):
    approved = approved_profile(version)
    web, before, after, response = collect_rank(tmp_path, approved, 0, early_turn=early_turn)
    registry = W.load_sources(tmp_path / "web_cached_sources_rank0.jsonl", source_receipt=RECEIPT,
                              profile_version=version) if web.cached_rows else None
    original = deepcopy(after)
    previous_schema, _ = replay_request(web, before, registry)
    schema, controls = replay_request(web, after, registry)
    assert previous_schema == before["options"]["json_schema"]
    assert schema == after["options"]["json_schema"] and after == original
    assert controls == set(range(2 + 2 * (early_turn + 1), len(after["messages"])))
    assert after["messages"][-(2 if early_turn == 18 else 1)] == before["messages"][-1]
    if web.cached_rows:
        assert before["web_cached_rows"] == after["web_cached_rows"] and len(registry) == 1
    # Public replay does not relax the existing accepted-SFT history gate.
    record = {"source": "independent_curriculum", "split": "train", "task_id": after["task_id"],
              "messages": after["messages"], "stage": "final", "completion": json.dumps(action())}
    owner = {"episode_id": after["episode_id"], "turn": after["turn_index"]}
    if web.cached_rows:
        owner["web_cached_rows"] = after["web_cached_rows"]
    with pytest.raises(ValueError, match="final target or unsupported tool"):
        web.original_schema(record, owner, cached_sources=registry)


@pytest.mark.parametrize("damage", ["feedback-bytes", "remaining", "feedback-role", "gold-property",
                                    "schema", "accepted-final", "turn", "navigation", "cache",
                                    "cache-on-feedback"])
def test_replay_rejects_tampered_feedback_schema_turn_navigation_and_cache(tmp_path, damage):
    approved = approved_profile(R.WEB32K_CACHED_PROFILE)
    web, _, request, _ = collect_rank(tmp_path, approved, 0, early_turn=16)
    registry = W.load_sources(tmp_path / "web_cached_sources_rank0.jsonl", source_receipt=RECEIPT,
                              profile_version=approved["version"])
    final_index, feedback_index = 34, 35
    if damage == "feedback-bytes":
        request["messages"][feedback_index]["content"] += " "
    elif damage == "remaining":
        message = request["messages"][feedback_index]
        message["content"] = message["content"].replace("Remaining turns: 3.", "Remaining turns: 4.")
    elif damage == "feedback-role":
        request["messages"][feedback_index]["role"] = "assistant"
    elif damage in {"gold-property", "schema", "accepted-final"}:
        value = json.loads(request["messages"][final_index]["content"])
        if damage == "gold-property":
            value["gold"] = "Invented forbidden field"
        elif damage == "schema":
            value["plan"] = "x" * 241
        else:
            value["answer"] = [{"Name": "Alpha", "citations": [URL]}]
        request["messages"][final_index]["content"] = json.dumps(value)
    elif damage == "turn":
        request["turn_index"] += 1
    elif damage == "navigation":
        request["messages"][-1]["content"] += " forged"
    elif damage == "cache":
        request["web_cached_rows"]["source_ids"] = []
    else:
        original = next(iter(registry.values()))
        extra = deepcopy(original)
        extra.update(source_id=original["source_id"] + "-unfetched", source_index=1, available_turn=17)
        registry[extra["source_id"]] = extra
        request["web_cached_rows"]["source_ids"].append(extra["source_id"])
        request["web_cached_rows"]["source_sha256"].append(extra["source_sha256"])
    with pytest.raises((ValueError, ValidationError)):
        replay_request(web, request, registry)


def test_closed_reader_preserves_all_eight_failed_tasks_and_missing_native_traces(tmp_path, monkeypatch):
    approved = approved_profile(R.WEB32K_CACHED_PROFILE)
    for rank in range(8):
        collect_rank(tmp_path, approved, rank, world=8, early_turn=16)
    monkeypatch.setattr(C, "load_train_labels", lambda *args, **kwargs:
                        (_ for _ in ()).throw(AssertionError("Public replay must not open labels")))
    registry = {}
    episodes, requests, traces = C.read_closed(tmp_path, profile=approved, receipt=RECEIPT,
                                               world_size=8, cached_source_registry=registry)
    assert len(episodes) == len({row["task_id"] for row in episodes}) == 8
    assert len(requests) == 8 * 18 and len(traces) == 8 * 17 and len(registry) == 8
    assert all(row["status"] == "model_protocol_failure" and row["gold_visible_to_policy"] is False
               and row["public_task"]["gold"] is None for row in episodes)
    failed = [(key, response) for key, (_, response) in requests.items() if not response["native_trace_recorded"]]
    assert len(failed) == 8
    assert all(key not in traces and response["completion"] is None
               and response["exception_type"] == "ModelProtocolFailure" for key, response in failed)
    # Claiming an absent trace exists still fails the unchanged ownership join.
    path = tmp_path / "requests_rank7.jsonl"
    rows = list(C.read_jsonl(path))
    rows[-1]["native_trace_recorded"] = True
    path.write_text("".join(C.json_text(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="Native trace coverage differs"):
        C.read_closed(tmp_path, profile=approved, receipt=RECEIPT, world_size=8,
                      cached_source_registry={})
