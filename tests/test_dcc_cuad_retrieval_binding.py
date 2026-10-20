"""CPU-only portable CUAD binding tests with invented public source tasks.

No benchmark tasks, real labels, model weights, network or GPU are used.
The native integration uses the existing fake engine and tokenizer fixtures.
"""
from collections import Counter
from copy import deepcopy
import asyncio
import base64
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "slime")]
import dcc_collect_native_train as C
import dcc_collect_native_preferences as G
import dcc_cuad_retrieval as R
import dcc_native_sft as N
from dcc_review_training import (
    CUAD_RETRIEVAL_PROFILE, REVIEW20_PROFILE, WEB32K_PROFILE,
    approved_cuad_profile, approved_web_profile,
)
from slime_plugins.dcc.cuad_structured import CUAD_STRUCTURED_SCHEMA_VERSION
from slime_plugins.dcc.data import Task
from slime_plugins.dcc.episode import EpisodeRunner
from slime_plugins.dcc.prompt_history import PromptHistoryConfig
from slime_plugins.dcc.retrieval import load_retrieval_profile

F = runpy.run_path(str(ROOT / "tests/test_dcc_final_preferences.py"))
QUESTION = "Extract confidentiality clauses."
SPAN = "Alpha agrees to preserve confidential information."
DOCUMENT = SPAN + " Beta may terminate on notice."


def profile_value():
    return {"schema": "harness-rl.dcc.retrieval", "version": 1,
        "method": "train_gold_log_lift_rrf",
        "parameters": {"max_terms": 3, "min_examples": 1, "min_fraction": 0.0, "rrf_constant": 60.0},
        "training": {"benchmark": "cuad", "split": "train", "tasks": 2, "positive_tasks": 1,
            "contracts": 1, "unique_documents": 1, "task_ids_sha256": "1" * 64},
        "questions": {QUESTION: {"category": QUESTION, "terms": ["confidential", "preserve"]}}}


def binding(**kwargs):
    return R.make_retrieval_binding((json.dumps(profile_value(), indent=2) + "\n").encode(), **kwargs)


def new_profile(tokenizer):
    result = F["source_profile"](tokenizer)
    result.update(version=CUAD_RETRIEVAL_PROFILE, web_profile_version=REVIEW20_PROFILE,
        cuad={"max_steps": 3, "structured_actions": True,
              "schema_version": CUAD_STRUCTURED_SCHEMA_VERSION, "retrieval": binding()})
    return result


@pytest.fixture
def tokenizer(monkeypatch):
    cls = F["FIXTURE"]["FakeTokenizer"]
    monkeypatch.setattr(cls, "__len__", lambda self: 160000, raising=False)
    monkeypatch.setattr(cls, "chat_template", "synthetic-template", raising=False)
    return cls()


def task():
    return Task("invented-cuad", "cuad", "train", QUESTION, document=DOCUMENT)


def closed(tmp_path, tokenizer, *, early=False):
    raw = tmp_path / "raw"
    values = ([{"answer": [{"text": SPAN, "confidence": 0.9}]}] if early else [
        {"tool": "search_document", "arguments": {"query": QUESTION, "top_k": 8}},
        {"tool": "read_document", "arguments": {"start": 0, "end": len(DOCUMENT)}},
        {"plan": "Use the public source.", "checks": "Check the source text.",
         "answer": [{"text": SPAN, "confidence": 0.9}]}])
    _, engine, native = F["make_native"](raw, values=values)
    profile = new_profile(tokenizer)
    schemas = G.source_schema_profile(profile)
    with patch.object(C, "load_train_labels", side_effect=AssertionError("labels reached public generation")):
        asyncio.run(C.generate([task()], native, raw, collection_id="invented-collection",
                               profile=profile, schema_profile=schemas))
    assert len(engine.calls) == 2 * len(values)
    return raw, profile, schemas


def export(raw, destination, tokenizer, profile, *, early=False):
    args = F["export_args"](tokenizer, schema_policy=G.FINAL_INTERVENTION if early else G.ORIGINAL_ONLY)
    args["source_profile"] = profile
    return G.export_prefixes(raw, destination, **args)


def test_original_profile_utf8_bytes_and_actual_path_loader_are_identical(tmp_path):
    raw = (" \n" + json.dumps(profile_value(), ensure_ascii=False, indent=3) + "\n\n").encode()
    path = tmp_path / "profile.json"
    path.write_bytes(raw)
    bound = R.make_retrieval_binding(raw)
    assert base64.b64decode(bound["profile_bytes_base64"]) == raw
    assert R.retrieval_runner_kwargs(bound)["retrieval_profile"] == load_retrieval_profile(path)
    assert "path" not in json.dumps(bound) and "document" not in bound


@pytest.mark.parametrize("damage", ["nontrain", "schema", "method", "path", "parameter-path", "training-path",
                                    "question-path", "documents", "duplicate", "nonfinite", "utf8"])
def test_binding_rejects_nontrain_nonportable_or_malformed_original_bytes(damage):
    value = profile_value()
    if damage == "nontrain": value["training"]["split"] = "test"
    elif damage == "schema": value["schema"] = "different"
    elif damage == "method": value["method"] = "claimed-compatible"
    elif damage == "path": value["profile_path"] = "not-portable"
    elif damage == "parameter-path": value["parameters"]["path"] = "not-portable"
    elif damage == "training-path": value["training"]["path"] = "not-portable"
    elif damage == "question-path": value["questions"][QUESTION]["path"] = "not-portable"
    elif damage == "documents": value["training"]["unique_documents"] = True
    raw = json.dumps(value).encode()
    if damage == "duplicate": raw = raw.replace(b'"version": 1', b'"version": 1, "version": 1')
    elif damage == "nonfinite": raw = raw.replace(b'60.0', b'NaN')
    elif damage == "utf8": raw = b'\xff'
    with pytest.raises(ValueError): R.make_retrieval_binding(raw)


@pytest.mark.parametrize("key,value", [("initial_chunks", True), ("initial_chunks", 9),
    ("retrieval_context_chars", -1), ("retrieval_context_chars", True),
    ("retrieval_max_context_chars", 0), ("retrieval_max_context_chars", True)])
def test_binding_runtime_limits_reject_boolean_or_invalid_values(key, value):
    with pytest.raises(ValueError): binding(**{key: value})


def test_new_profile_requires_explicit_route_binding_and_source_overlay(tokenizer, tmp_path):
    profile = new_profile(tokenizer)
    before = deepcopy(profile)
    web = approved_web_profile(profile, N.WebSchemaProfile)
    assert web.profile_version == REVIEW20_PROFILE
    assert approved_cuad_profile(profile, N.CuadSchemaProfile).retrieval == profile["cuad"]["retrieval"]
    assert profile == before
    from dcc_review_training import bind_profile_source
    overlay = tmp_path / ".dcc-training-code-overlay.json"
    overlay.write_text(json.dumps({"overlay_id": profile["collection_overlay_id"],
        "explicit_training_profile": REVIEW20_PROFILE, "baseline_source_unchanged": False}))
    with pytest.raises(ValueError): bind_profile_source(profile, tmp_path)
    overlay.write_text(json.dumps({"overlay_id": profile["collection_overlay_id"],
        "explicit_training_profile": CUAD_RETRIEVAL_PROFILE, "baseline_source_unchanged": False}))
    bind_profile_source(profile, tmp_path)
    for damage in ("missing-binding", "missing-route", "legacy-binding"):
        broken = deepcopy(profile)
        if damage == "missing-binding": broken["cuad"].pop("retrieval")
        elif damage == "missing-route": broken.pop("web_profile_version")
        else:
            broken["version"] = REVIEW20_PROFILE
            broken.pop("web_profile_version")
        with pytest.raises(ValueError):
            G.source_schema_profile(broken)


def test_legacy_model_identity_schema_prompt_and_default_kwargs_do_not_gain_fields(tokenizer):
    identity = F["identity"](tokenizer)
    original = identity.manifest(tokenizer)
    assert N.identity_from_manifest(original, identity.source_commit).manifest(tokenizer) == original
    assert "retrieval" not in json.dumps(original)
    old = N.CuadSchemaProfile(3, True, CUAD_STRUCTURED_SCHEMA_VERSION)
    schemas = N.NativeSchemaProfile(N.WebSchemaProfile(3), old)
    with patch.object(C, "EpisodeRunner", wraps=EpisodeRunner) as factory:
        runner = C.make_runner("cuad", schemas, timeout_seconds=5)
    assert factory.call_args.kwargs["initial_chunks"] == 4
    assert not any(name.startswith("retrieval") for name in factory.call_args.kwargs)
    assert json.dumps(runner.initial_messages(task())) == json.dumps(EpisodeRunner(max_steps=3, initial_chunks=4).initial_messages(task()))
    assert original == F["identity"](tokenizer).manifest(tokenizer)


@pytest.mark.parametrize("context", [0, 800])
def test_explicit_runner_and_preference_replay_match_evaluation_retrieval(context, tokenizer):
    profile = new_profile(tokenizer)
    profile["cuad"]["retrieval"] = binding(retrieval_context_chars=context)
    schemas = G.source_schema_profile(profile)
    sample = Task("many-public-chunks", "cuad", "train", QUESTION,
        document=("Ordinary public contract sentence. " * 500) + DOCUMENT)
    runner = C.make_runner("cuad", schemas, timeout_seconds=5)
    expected = EpisodeRunner(max_steps=3, initial_chunks=8,
        retrieval_profile=R.retrieval_runner_kwargs(profile["cuad"]["retrieval"])["retrieval_profile"],
        retrieval_context_chars=context, retrieval_max_context_chars=16000)
    messages = runner.initial_messages(sample)
    assert messages == expected.initial_messages(sample)
    if context == 0:
        assert len(json.loads(messages[2]["content"].split("\n", 1)[1])["hits"]) == 8
    G.public_request_schemas(sample, messages, turn=0, schemas=schemas)
    with pytest.raises(ValueError, match="anchors"):
        G.public_request_schemas(sample, EpisodeRunner(max_steps=3, initial_chunks=4).initial_messages(sample),
                                 turn=0, schemas=schemas)


def test_web_route_is_identical_and_cuad_binding_never_enters_model_identity(tokenizer):
    old = F["source_profile"](tokenizer)
    profile = new_profile(tokenizer)
    assert old["identity"] == profile["identity"]
    schemas = G.source_schema_profile(profile)
    original = G.source_schema_profile(old)
    with patch.object(C, "EpisodeRunner", wraps=EpisodeRunner) as factory:
        actual = C.make_runner("widesearch", schemas, timeout_seconds=5)
    kwargs = factory.call_args.kwargs
    assert not any(name.startswith("retrieval") for name in kwargs)
    assert actual.initial_messages(F["web_task"]()) == C.make_runner("widesearch", original, timeout_seconds=5).initial_messages(F["web_task"]())


def test_bound_native_closed_collection_replays_every_request_without_label_access(tmp_path, tokenizer):
    raw, profile, schemas = closed(tmp_path, tokenizer)
    before = {p.name: p.read_bytes() for p in raw.iterdir()}
    episodes, requests, traces = C.read_closed(raw, profile=profile, receipt=F["FIXTURE"]["RECEIPT"], world_size=1)
    assert len(episodes) == 1 and len(requests) == len(traces) == 3
    for request, _ in requests.values():
        assert request["cuad_retrieval"] == profile["cuad"]["retrieval"]
        G.public_request_schemas(task(), request["messages"], turn=request["turn_index"], schemas=schemas)
    assert before == {p.name: p.read_bytes() for p in raw.iterdir()}


@pytest.mark.parametrize("damage", ["closure", "closure-bool", "request-binding", "request-bool",
                                    "initial", "search", "read", "control", "schema"])
def test_bound_journal_replay_fails_before_selector_can_open_labels(tmp_path, tokenizer, damage):
    raw, profile, schemas = closed(tmp_path, tokenizer)
    if damage in {"closure", "closure-bool"}:
        path = raw / "generation_closed_rank0.json"
        value = json.loads(path.read_text())
        if damage == "closure": value.pop("cuad_retrieval")
        else: value["cuad_retrieval"]["retrieval_context_chars"] = False
        path.write_text(json.dumps(value))
    else:
        path = raw / "requests_rank0.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        requests = [event for event in events if event["event"] == "request"]
        if damage == "request-binding": requests[0].pop("cuad_retrieval")
        elif damage == "request-bool": requests[0]["cuad_retrieval"]["retrieval_context_chars"] = False
        elif damage == "initial": requests[0]["messages"][2]["content"] += " changed"
        elif damage == "search": requests[1]["messages"][4]["content"] += " changed"
        elif damage == "read": requests[2]["messages"][6]["content"] += " changed"
        elif damage == "control": requests[2]["messages"][-1]["content"] += " changed"
        else: requests[0]["options"]["json_schema"] = {"type": "object"}
        path.write_text("".join(json.dumps(event) + "\n" for event in events))
    with patch.object(C, "load_train_labels", side_effect=AssertionError("labels opened")) as labels:
        with pytest.raises(ValueError):
            C.select(raw, tmp_path / "selected", labels_path=tmp_path / "must-not-open",
                profile=profile, identity=F["identity"](tokenizer), schema_profile=schemas,
                receipt=F["FIXTURE"]["RECEIPT"], tokenizer=tokenizer)
        labels.assert_not_called()


@pytest.mark.parametrize("early", [False, True])
def test_preference_export_retains_binding_and_replays_public_source_at_freeze_load(tmp_path, tokenizer, early):
    raw, profile, _ = closed(tmp_path, tokenizer, early=early)
    freeze = export(raw, tmp_path / "export", tokenizer, profile, early=early)
    actual, prefixes, identity = G.load_frozen(tmp_path / "export", tokenizer)
    assert actual == freeze and freeze["cuad_retrieval"] == profile["cuad"]["retrieval"]
    assert "retrieval" not in json.dumps(identity.manifest(tokenizer))
    approval = json.loads((tmp_path / "export/export_source_approval.json").read_text())
    assert approval["cuad_public_tasks"] == {task().id: task().to_dict()}
    assert approval["source_profile"] == profile
    assert len(prefixes) == 1


@pytest.mark.parametrize("damage", ["binding", "strip", "approval", "document", "profile", "prefix"])
def test_frozen_binding_or_public_anchor_tampering_is_rejected(tmp_path, tokenizer, damage):
    raw, profile, _ = closed(tmp_path, tokenizer)
    export(raw, tmp_path / "export", tokenizer, profile)
    if damage in {"binding", "strip"}:
        path = tmp_path / "export/freeze.json"
        value = json.loads(path.read_text())
        if damage == "binding": value["cuad_retrieval"]["retrieval_context_chars"] = 800
        else: value.pop("cuad_retrieval")
        path.write_text(json.dumps(value))
    elif damage == "approval": (tmp_path / "export/export_source_approval.json").unlink()
    elif damage in {"document", "profile"}:
        path = tmp_path / "export/export_source_approval.json"
        value = json.loads(path.read_text())
        if damage == "document": value["cuad_public_tasks"][task().id]["document"] += " altered"
        else: value["source_profile"]["cuad"]["retrieval"]["initial_chunks"] = 4
        path.write_text(json.dumps(value))
    else:
        path = tmp_path / "export/prefixes.jsonl"
        value = json.loads(path.read_text()); value["messages"][2]["content"] += " altered"
        path.write_text(json.dumps(value) + "\n")
    with pytest.raises(ValueError): G.load_frozen(tmp_path / "export", tokenizer)


def test_new_cuad_and_existing_web_share_native_training_without_identity_allowlist_changes(tmp_path, tokenizer):
    raw, profile, schemas = closed(tmp_path / "new-cuad", tokenizer)
    identity = F["identity"](tokenizer)
    receipt = F["FIXTURE"]["RECEIPT"]
    # The labels below are invented fixture expectations, never benchmark data.
    labels = tmp_path / "invented-cuad-labels.jsonl"
    labels.write_text(json.dumps({"task_id": task().id, "benchmark": "cuad", "split": "train",
        "gold": [SPAN], "category": "Confidentiality"}) + "\n")
    C.select(raw, tmp_path / "new-selected", labels_path=labels, profile=profile,
             identity=identity, schema_profile=schemas, receipt=receipt, tokenizer=tokenizer)
    old_raw = tmp_path / "old-web"
    old_profile = F["source_profile"](tokenizer)
    values = [F["REVIEW"]["action"]("web_search"), F["REVIEW"]["action"]("fetch_url"),
              F["REVIEW"]["action"]()]
    values[-1]["answer"][0]["name"] = values[-1]["answer"][0].pop("Name")
    _, _, native = F["make_native"](old_raw, values=values)
    web_task = Task("old-public-web", "widesearch", "train", "Find the public listed name.",
                    metadata={"evaluation": {"required": ["name"]}})
    asyncio.run(C.generate([web_task], native, old_raw, collection_id="existing-web-collection",
        profile=old_profile, schema_profile=G.source_schema_profile(old_profile),
        web_tools_factory=F["REVIEW"]["Tools"]))
    old_labels = tmp_path / "invented-web-labels.jsonl"
    old_labels.write_text(json.dumps({"task_id": web_task.id, "benchmark": "widesearch",
        "split": "train", "gold": [{"name": "Alpha"}], "evaluation": {"required": ["name"]}}) + "\n")
    C.select(old_raw, tmp_path / "old-selected", labels_path=old_labels, profile=old_profile,
             identity=identity, schema_profile=G.source_schema_profile(old_profile), receipt=receipt, tokenizer=tokenizer)
    root = tmp_path / "source"; root.mkdir()
    (root / ".dcc-native-source.json").write_text(json.dumps(receipt))
    (root / ".dcc-training-code-overlay.json").write_text(json.dumps({
        "overlay_id": profile["collection_overlay_id"], "explicit_training_profile": profile["version"],
        "baseline_source_unchanged": False}))
    profile_path = tmp_path / "approved-profile.json"; profile_path.write_text(json.dumps(profile))
    args = SimpleNamespace(adapter=None, cuad_train_data=None, native_expected_source=identity.source_commit,
        native_train_profile=str(profile_path), cuad_tool_records=str(tmp_path / "new-selected/native_cuad_train.jsonl"),
        records=str(tmp_path / "old-selected/native_web_train.jsonl"), model="unused")
    approved, actual_identity, _, _ = C.load_approved(profile_path, root, identity.source_commit, tokenizer)
    assert approved == profile and actual_identity == identity
    data = N.prepare_native_training(args, source_root=root, tokenizer=tokenizer)
    assert len(data.records) == len(data.cuad_records) == 3
    assert data.configuration["cuad"] == profile["cuad"]
    assert data.manifest == old_profile["identity"] == profile["identity"]
    rows = list(C.read_jsonl(args.cuad_tool_records))
    assert all(row["collector"]["cuad_retrieval"] == profile["cuad"]["retrieval"]
               and row["collector"]["public_task"] == task().to_dict() for row in rows)
    damaged = deepcopy(rows[0]); damaged["collector"].pop("cuad_retrieval")
    with pytest.raises(ValueError): schemas.original_schema(damaged["record"], damaged["collector"])
    legacy = N.CuadSchemaProfile(3, True, CUAD_STRUCTURED_SCHEMA_VERSION)
    with pytest.raises(ValueError, match="cannot strip"):
        legacy.original_schema(rows[0]["record"], rows[0]["collector"])


def test_new_profile_reuses_exact_existing_web32k_domain_identity_constraints(tokenizer):
    profile = new_profile(tokenizer)
    old_cuad = deepcopy(profile["identity"])
    profile["web_profile_version"] = WEB32K_PROFILE
    profile["identity"].update(max_prompt_tokens=32768, max_model_len=53248,
        prompt_history=PromptHistoryConfig(True).to_dict())
    profile["cuad_identity"] = old_cuad
    profile["cuad_identity_differences"] = sorted(k for k in set(profile["identity"]) | set(old_cuad)
        if profile["identity"].get(k) != old_cuad.get(k))
    assert approved_web_profile(profile, N.WebSchemaProfile).profile_version == WEB32K_PROFILE
    profile["cuad_identity"]["derivative_id"] = "different-model"
    profile["cuad_identity_differences"].append("derivative_id")
    profile["cuad_identity_differences"].sort()
    with pytest.raises(ValueError, match="domain identity differences"):
        approved_web_profile(profile, N.WebSchemaProfile)


def test_existing_web32k_full_native_bundle_and_bound_cuad_pass_joint_training_preflight(tmp_path, tokenizer, monkeypatch):
    raw, profile, schemas = closed(tmp_path / "new-cuad", tokenizer)
    cuad_identity = F["identity"](tokenizer)
    labels = tmp_path / "invented-labels.jsonl"
    labels.write_text(json.dumps({"task_id": task().id, "benchmark": "cuad", "split": "train",
        "gold": [SPAN], "category": "Confidentiality"}) + "\n")
    C.select(raw, tmp_path / "selected", labels_path=labels, profile=profile, identity=cuad_identity,
             schema_profile=schemas, receipt=F["FIXTURE"]["RECEIPT"], tokenizer=tokenizer)
    web_fixtures = runpy.run_path(str(ROOT / "tests/test_dcc_web32k_loader.py"))
    record, schema, turn = web_fixtures["public_web_record"]()
    _, web_identity, web_bundle, _ = web_fixtures["produce"](tmp_path / "existing-web32k", monkeypatch,
        record, schema, turn, managed=True, source="e" * 40)
    web_path = tmp_path / "web.jsonl"
    web_path.write_text(json.dumps(web_bundle) + "\n")
    original_web_bytes = web_path.read_bytes()
    profile.update(web_profile_version=WEB32K_PROFILE, identity=web_identity.manifest(tokenizer),
                   cuad_identity=cuad_identity.manifest(tokenizer))
    profile["cuad_identity_differences"] = sorted(k for k in set(profile["identity"]) | set(profile["cuad_identity"])
        if profile["identity"].get(k) != profile["cuad_identity"].get(k))
    root = tmp_path / "source"; root.mkdir()
    web_receipt = dict(F["FIXTURE"]["RECEIPT"], source_commit=web_identity.source_commit)
    (root / ".dcc-native-source.json").write_text(json.dumps(web_receipt))
    (root / ".dcc-training-code-overlay.json").write_text(json.dumps({
        "overlay_id": profile["collection_overlay_id"], "explicit_training_profile": profile["version"],
        "baseline_source_unchanged": False}))
    profile_path = tmp_path / "profile.json"; profile_path.write_text(json.dumps(profile))
    args = SimpleNamespace(adapter=None, cuad_train_data=None, native_expected_source=web_identity.source_commit,
        native_train_profile=str(profile_path), cuad_tool_records=str(tmp_path / "selected/native_cuad_train.jsonl"),
        records=str(web_path), model="unused")
    data = N.prepare_native_training(args, source_root=root, tokenizer=tokenizer)
    assert len(data.records) == 1 and len(data.cuad_records) == 3
    assert data.strict_training_budgets and data.identity == web_identity and data.cuad_identity == cuad_identity
    assert web_identity.source_commit == "e" * 40 and cuad_identity.source_commit == "f" * 40
    assert profile["cuad_identity_differences"] == [
        "max_model_len", "max_prompt_tokens", "prompt_history", "source_commit"]
    assert data.configuration["domain_identities"]["cuad_train"] == profile["cuad_identity"]
    skips = Counter()
    example = data.encode_record(data.records[0], tokenizer, max_prompt=32768, max_response=20480,
                                batch_tokens=212992, skips=skips)
    assert example.completion == web_bundle["native_trace"]["completion_ids"]
    assert skips == Counter() and web_path.read_bytes() == original_web_bytes


def test_bound_preference_g4_closes_and_can_share_existing_dpo_model_identity(tmp_path, tokenizer):
    raw, profile, _ = closed(tmp_path, tokenizer)
    export(raw, tmp_path / "export", tokenizer, profile)
    freeze, prefixes, identity = G.load_frozen(tmp_path / "export", tokenizer)
    final = {"plan": "Use the public source.", "checks": "Check the observed source.",
             "answer": [{"text": SPAN, "confidence": 0.9}]}
    _, engine, native = F["make_native"](tmp_path / "g4", values=[deepcopy(final) for _ in range(4)], review=False)
    asyncio.run(G.generate_g4(prefixes, native, tmp_path / "g4", freeze=freeze, identity=identity))
    actual = G.read_closed_g4(tmp_path / "g4", freeze=freeze, prefixes=prefixes,
                            tokenizer=tokenizer, expected_source=identity.source_commit)
    assert len(engine.calls) == 8 and len(actual["candidates"]) == 4
    assert actual["generation_closed_before_scoring"] is True
    from dcc_native_dpo_train import common_collection_identity, validate_public_collection_ownership
    old_web = F["identity"](tokenizer).manifest(tokenizer); old_web.pop("generation_seed")
    assert common_collection_identity(old_web) == common_collection_identity(freeze["native_identity"])
    def public_owner(name, native):
        return {"collection_id": name, "collection_native_identity": native,
            "frozen_prefix_ids": [name + "-prefix"], "frozen_task_ids": [name + "-task"],
            "attempt_trace_ids": [name + "-trace"], "phase_request_ids": [name + "-phase"]}
    validate_public_collection_ownership([public_owner("existing-web", old_web),
                                         public_owner("bound-cuad", freeze["native_identity"])])
