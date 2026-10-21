"""Real CPU collector/native validators with invented engine tokens and sources."""
from copy import deepcopy
from dataclasses import replace
import asyncio
import json
from pathlib import Path
import runpy
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "slime")]
import dcc_collect_native_train as C
import dcc_native_sft as N
import dcc_review_training as R
import dcc_train_cached_rows as W
from dcc_train_web_literal_reward import LABEL_PROTOCOL, configuration
from slime_plugins.dcc.data import Task
from slime_plugins.dcc.eval_sampling import EvaluationSampling
from slime_plugins.dcc.native_thinking_eval import BASE_SOURCE_COMMIT, PROTOCOL, NativeThinkingPolicy, NativeTraceJournal
from slime_plugins.dcc.prompt_history import PromptHistoryConfig
from slime_plugins.dcc.vllm_policy import AsyncVLLMPolicy

F = runpy.run_path(str(ROOT / "tests/test_dcc_native_train_data_fixes.py"))
PUBLIC = runpy.run_path(str(ROOT / "tests/test_dcc_review_feedback_replay.py"))


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def generated(tmp_path, *, enabled=True, cached=True, rejected_count=1):
    tokenizer = F["Tokenizer"]()
    version = R.WEB32K_CACHED_8K_PROFILE if cached else R.REVIEW20_PROFILE
    history = PromptHistoryConfig(cached)
    thought, prompt = (8192, 32768) if cached else (16384, 16384)
    identity = N.NativeIdentity("synthetic-feedback-derivative", (151643, 151645), "c" * 40,
        prompt, prompt + thought + 4096, 42, thought, "thinking",
        history.to_dict() if cached else None)
    receipt = {"version": "dcc-native-inference-source-v1", "source_commit": "c" * 40,
               "base_source_commit": BASE_SOURCE_COMMIT,
               "protocol": PROTOCOL, "frozen": True}
    profile = PUBLIC["approved_profile"](version)
    profile["identity"] = identity.manifest(tokenizer)
    if enabled is not None:
        profile["web"]["evidence_feedback"] = enabled
    web = R.approved_web_profile(profile, N.WebSchemaProfile)
    task = Task("synthetic-self-correction", "widesearch", "train", "List the public names.",
                metadata={"evaluation": {"required": ["Name"]}})
    action = PUBLIC["action"]
    final = action()
    final["answer"] = [{"Name": "Alpha", "citations": [PUBLIC["URL"]]}]
    values = [action("web_search"), action("fetch_url")]
    values += [action() for _ in range(rejected_count)] + [final]
    engine = F["Engine"](tokenizer, values)
    output = tmp_path / "generation"
    output.mkdir()
    base = AsyncVLLMPolicy(engine, tokenizer, max_prompt_tokens=identity.max_prompt_tokens,
        max_model_len=identity.max_model_len, eos_token_ids=list(identity.eos_token_ids),
        evaluation_sampling=EvaluationSampling(), allow_truncation=False,
        web_structured_final=True, web_structured_actions=True,
        provenance={"web_cached_rows": W.configuration()} if cached else {})
    native = NativeThinkingPolicy(base, source_receipt=receipt, budget=identity.budget,
        prompt_history_config=history,
        trace_sink=NativeTraceJournal(output / "native_thinking_traces_rank0.jsonl"))
    schemas = N.NativeSchemaProfile(web, None)
    asyncio.run(C.generate([task], native, output, collection_id="synthetic-feedback-collection",
        profile=profile, schema_profile=schemas, web_tools_factory=PUBLIC["Tools"]))
    label = {"task_id": task.id, "benchmark": "widesearch", "split": "train",
             "gold": [{"Name": "Alpha"}], "evaluation": {"required": ["Name"]},
             "label_protocol": LABEL_PROTOCOL, "train_reward": configuration()}
    labels = tmp_path / "TRAIN-labels.jsonl"
    write_jsonl(labels, [label])
    def select():
        return C.select(output, tmp_path / "selected", labels_path=labels, profile=profile,
            identity=identity, schema_profile=schemas, receipt=receipt, tokenizer=tokenizer,
            web_train_reward_protocol=C.LITERAL_WEB_TRAIN_REWARD)
    return output, select, schemas, identity, tokenizer, profile


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("rejected_count", [1, 2])
def test_native_recovery_keeps_all_traces_but_never_supervises_rejected_finals(tmp_path, cached, rejected_count):
    output, select, schemas, identity, tokenizer, _ = generated(
        tmp_path, cached=cached, rejected_count=rejected_count)
    original = {p.name: p.read_bytes() for p in output.iterdir()}
    summary = select()
    assert (summary["accepted_episodes"], summary["rejected_episodes"], summary["web_segments"]) == (1, 0, 3)
    assert summary["excluded_public_rejected_final_records"] == rejected_count
    assert summary["rejected_final_targets_receive_loss"] is False
    assert original == {p.name: p.read_bytes() for p in output.iterdir()}
    rows = list(C.read_jsonl(tmp_path / "selected/native_web_train.jsonl"))
    assert [row["collector"]["turn"] for row in rows] == [0, 1, 2 + rejected_count]
    excluded = list(C.read_jsonl(tmp_path / "selected/excluded_public_rejected_finals.jsonl"))
    assert [row["turn"] for row in excluded] == list(range(2, 2 + rejected_count))
    assert all(row["physical_native_validated"] and not row["positive_SFT_target"] for row in excluded)
    registry = None
    # The actual exported file loader revalidates physical IDs and all new
    # accepted-target grounding gates, independently of selector bookkeeping.
    if cached:
        registry = W.load_sources(tmp_path / "selected/native_web_cached_sources.jsonl",
            source_receipt=json.loads((output / "generation_closed_rank0.json").read_text())["source_receipt"],
            profile_version=schemas.web.profile_version)
        schemas = N.NativeSchemaProfile(schemas.web, None, registry)
    loaded = list(N.iter_native_records(tmp_path / "selected/native_web_train.jsonl", tokenizer,
        expected_identity=identity, schema_resolver=schemas.original_schema))
    assert len(loaded) == 3 and loaded[-1]["native_trace"] == rows[-1]["native_trace"]
    final_record, owner = rows[-1]["record"], rows[-1]["collector"]
    with pytest.raises(ValueError, match="final target or unsupported tool"):
        replace(schemas.web, evidence_feedback=False).original_schema(final_record, owner, cached_sources=registry)


@pytest.mark.parametrize("enabled", [None, False])
def test_absent_and_off_keep_original_failed_native_request(tmp_path, enabled):
    output, select, *_ = generated(tmp_path, enabled=enabled)
    episode = next(C.read_jsonl(output / "episodes_rank0.jsonl"))
    assert episode["status"] == "model_protocol_failure"
    assert episode["detail"] == "training_unsupported_evidence_feedback"
    assert len(list(C.read_jsonl(output / "native_thinking_traces_rank0.jsonl"))) == 3
    summary = select()
    assert summary["accepted_episodes"] == 0 and summary["web_segments"] == 0
    assert "web_evidence_feedback" not in summary
    assert not (tmp_path / "selected/excluded_public_rejected_finals.jsonl").exists()


@pytest.mark.parametrize("damage", ["prompt-ids", "phase-ownership", "history-action", "feedback", "missing-trace", "wrong-gold"])
def test_recovery_never_bypasses_physical_history_or_correctness_checks(tmp_path, damage):
    output, select, *_ = generated(tmp_path)
    trace_path, request_path = output / "native_thinking_traces_rank0.jsonl", output / "requests_rank0.jsonl"
    traces, events = list(C.read_jsonl(trace_path)), list(C.read_jsonl(request_path))
    bad = traces[2]
    if damage == "prompt-ids":
        bad["original_prompt_ids"][0] += 1
        next(row for row in events if row["event"] == "response" and row["turn_index"] == 2)["completion"]["prompt_ids"] = bad["original_prompt_ids"]
    elif damage == "phase-ownership":
        bad["phases"][0]["request_id"] = traces[0]["phases"][0]["request_id"]
    elif damage in {"history-action", "feedback"}:
        last = next(row for row in events if row["event"] == "request" and row["turn_index"] == 3)
        if damage == "history-action":
            value = json.loads(last["messages"][6]["content"])
            value["answer"] = [{"Name": "Alpha", "citations": [PUBLIC["URL"]]}]
            last["messages"][6]["content"] = json.dumps(value)
        else:
            last["messages"][7]["content"] += " forged"
    elif damage == "missing-trace":
        traces.pop(2)
        response = next(row for row in events if row["event"] == "response" and row["turn_index"] == 2)
        response["native_trace_recorded"] = False
        closed_path = output / "generation_closed_rank0.json"
        closed = json.loads(closed_path.read_text()); closed["native_traces"] -= 1
        closed_path.write_text(json.dumps(closed))
    else:
        label_path = tmp_path / "TRAIN-labels.jsonl"
        label = next(C.read_jsonl(label_path)); label["gold"] = [{"Name": "Wrong"}]
        write_jsonl(label_path, [label])
    write_jsonl(trace_path, traces); write_jsonl(request_path, events)
    try:
        summary = select()
    except ValueError:
        assert damage in {"history-action", "feedback"}
    else:
        assert summary["accepted_episodes"] == 0 and summary["web_segments"] == 0
        assert (tmp_path / "selected/native_web_train.jsonl").read_bytes() == b""
        assert (tmp_path / "selected/excluded_public_rejected_finals.jsonl").read_bytes() == b""


@pytest.mark.parametrize("bad", [0, 1, "true", None])
def test_evidence_feedback_requires_typed_explicit_review_profile(bad):
    profile = PUBLIC["approved_profile"](R.REVIEW20_PROFILE)
    profile["web"]["evidence_feedback"] = bad
    with pytest.raises(ValueError, match="evidence-feedback"):
        R.approved_web_profile(profile, N.WebSchemaProfile)
    with pytest.raises(ValueError, match="explicit public review"):
        N.WebSchemaProfile(20, evidence_feedback=True).validate()


@pytest.mark.parametrize("cached,rejections,late", [(False, 1, True), (True, 1, True),
                                                   (True, 1, False), (True, 2, False)])
def test_last_turn_recovery_and_new_fetch_keep_exact_cache_availability(tmp_path, monkeypatch, cached, rejections, late):
    second_url = "https://example.org/second-public.csv"
    original_tools = PUBLIC["Tools"]
    original_engine = F["Engine"]
    class TwoSources(original_tools):
        async def fetch_url(self, url, **kwargs):
            self.pages[url] = PUBLIC["TEXT"]
            return await super(original_tools, self).fetch_url(url, **kwargs)
    action = PUBLIC["action"]
    values = [action("web_search"), action("fetch_url")]
    if late:
        for turn in range(2, 18):
            search = action("web_search")
            search["arguments"]["query"] = f"Public names additional source {turn}"
            values.append(search)
    first_rejection = len(values)
    values.extend(action() for _ in range(rejections))
    if not late:
        fetch = action("fetch_url"); fetch["arguments"]["url"] = second_url
        values.append(fetch)
    final = action()
    final["answer"] = [{"Name": "Alpha", "citations": [PUBLIC["URL"] if late else second_url]}]
    values.append(final)
    monkeypatch.setitem(F, "Engine", lambda tokenizer, _: original_engine(tokenizer, deepcopy(values)))
    monkeypatch.setitem(PUBLIC, "Tools", TwoSources)
    output, select, *_ = generated(tmp_path, cached=cached)
    summary = select()
    assert summary["accepted_episodes"] == 1 and summary["rejected_episodes"] == 0
    rows = list(C.read_jsonl(tmp_path / "selected/native_web_train.jsonl"))
    excluded = list(C.read_jsonl(tmp_path / "selected/excluded_public_rejected_finals.jsonl"))
    assert [row["turn"] for row in excluded] == list(range(first_rejection, first_rejection + rejections))
    assert len(rows) == len(values) - rejections and rows[-1]["collector"]["turn"] == len(values) - 1
    assert json.loads(rows[-1]["record"]["completion"])["answer"] == final["answer"]
    if cached and not late:
        requests = [row for row in C.read_jsonl(output / "requests_rank0.jsonl") if row["event"] == "request"]
        assert [len(row["web_cached_rows"]["source_ids"]) for row in requests] == [0, 0] + [1] * (rejections + 1) + [2]
        sources = list(C.read_jsonl(tmp_path / "selected/native_web_cached_sources.jsonl"))
        assert sorted(row["available_turn"] for row in sources) == [2, len(values) - 1]
