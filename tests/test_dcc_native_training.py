"""Budgeted native TRAIN integrity using synthetic engine responses, CPU only."""
from copy import deepcopy
import asyncio
import importlib
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
TRAIN = importlib.import_module("dcc_native_sft")
COLLECT = importlib.import_module("dcc_collect_native_train")
FIXTURE = runpy.run_path(str(Path(__file__).with_name("test_dcc_native_thinking.py")))


def make_record(tmp_path, monkeypatch, *, budget=None):
    from slime_plugins.dcc.native_budget import NativeBudget
    from slime_plugins.dcc.web_structured import web_action_schema
    cls = FIXTURE["FakeTokenizer"]
    monkeypatch.setattr(cls, "__len__", lambda self: 160000, raising=False)
    monkeypatch.setattr(cls, "chat_template", "synthetic-template", raising=False)
    budget = NativeBudget(16384, "thinking") if budget is None else budget
    value = {"tool": "web_search", "arguments": {"query": "public entity"}}
    tokenizer, engine, native = FIXTURE["make_policy"](tmp_path, values=[value], budget=budget)
    identity = TRAIN.NativeIdentity("synthetic-derivative", (151643, 151645), "f" * 40,
        16384, 16384 + budget.total_tokens, 42, budget.thought_tokens, budget.thought_sampling_profile)
    record = {"source": "independent_curriculum", "split": "train", "task_id": "fixture",
        "stage": "search", "messages": [
            {"role": "system", "content": 'Required answer columns: ["Name"]. Return action JSON.'},
            {"role": "user", "content": "Find the entity in a public source."}],
        "completion": json.dumps(value)}
    schema = web_action_schema(["Name"], [], successful_search=False)
    async def generate_and_close():
        try:
            await native.generate_structured(record["messages"], max_new_tokens=budget.total_tokens,
                temperature=0.0, json_schema=schema,
                trace_metadata={"trace_id": "fixture-trace", "task_id": "fixture", "rank": 0, "turn_index": 0})
        finally:
            await native.close()
    asyncio.run(generate_and_close())
    trace = native.traces[0]
    bundle = {"contract": TRAIN.CONTRACT, "record": record, "identity": identity.manifest(tokenizer),
        "collector": {"split": "train", "task_input": "public-train-task-without-gold",
            "gold_visible_to_policy": False, "recorded_messages": True, "synthetic_thought": False,
            "inserted_close": False, "inserted_eos": False, "episode_id": "synthetic-episode",
            "turn": 0, "rank": 0, "trace_id": "fixture-trace"}, "native_trace": deepcopy(trace)}
    return tokenizer, identity, schema, bundle


@pytest.mark.parametrize("legacy", [False, True])
def test_actual_phase_ids_are_training_targets_and_profile_roundtrips(tmp_path, monkeypatch, legacy):
    from slime_plugins.dcc.native_budget import NativeBudget
    tokenizer, identity, schema, bundle = make_record(tmp_path, monkeypatch,
        budget=NativeBudget() if legacy else NativeBudget(16384, "thinking"))
    actual = TRAIN.validate_native_record(bundle, tokenizer, expected_identity=identity, original_schema=schema)
    assert actual.completion == bundle["native_trace"]["completion_ids"]
    assert actual.prompt == bundle["native_trace"]["original_prompt_ids"]
    manifest = identity.manifest(tokenizer)
    assert TRAIN.identity_from_manifest(manifest, identity.source_commit) == identity
    assert ("thought_tokens" not in manifest) is legacy
    profile = {"version": "dcc-native-train-profile-v1", "identity": manifest,
               "web": {"max_steps": 8, "explicit_review": False, "cached_rows": False}}
    root = tmp_path / "source"; root.mkdir()
    (root / ".dcc-native-source.json").write_text(json.dumps(FIXTURE["RECEIPT"]))
    profile_path = tmp_path / "profile.json"; profile_path.write_text(json.dumps(profile))
    approved = COLLECT.load_approved(profile_path, root, identity.source_commit, tokenizer)
    assert approved[1] == identity
    rows = tmp_path / "train.jsonl"; rows.write_text(json.dumps(bundle) + "\n")
    args = SimpleNamespace(adapter=None, cuad_train_data=None, native_expected_source=identity.source_commit,
        native_train_profile=str(profile_path), cuad_tool_records=None, records=str(rows), model="unused")
    data = TRAIN.prepare_native_training(args, source_root=root, tokenizer=tokenizer)
    assert data.configuration["protocol"] == identity.budget.protocol
    assert data.configuration["validated_records"] == 1
    assert ("rollout_native_thinking" not in data.configuration) is legacy


@pytest.mark.parametrize("damage", ["budget", "sampling", "effective", "close", "identity", "config", "length"])
def test_budgeted_train_rejects_tampered_real_phase_contract(tmp_path, monkeypatch, damage):
    tokenizer, identity, schema, bundle = make_record(tmp_path, monkeypatch)
    trace = bundle["native_trace"]
    if damage == "budget": trace["thought_max_tokens"] = 4096
    elif damage == "sampling": trace["phases"][0]["sampling"]["temperature"] = 0.0
    elif damage == "effective": trace["phases"][1]["effective_sampling"]["top_k"] = -1
    elif damage == "close": trace["phases"][0]["stop_reason"] = 151645
    elif damage == "identity": bundle["identity"]["thought_sampling_profile"] = "greedy"
    elif damage == "config": trace["native_thinking"]["thought_sampling"]["top_p"] = 0.9
    else: trace["phases"][0]["max_tokens"] = 4096
    with pytest.raises(ValueError):
        TRAIN.validate_native_record(bundle, tokenizer, expected_identity=identity, original_schema=schema)


def test_budgeted_profile_reserves_actual_total_and_rejects_unknown_sampling(tmp_path, monkeypatch):
    tokenizer, identity, _, _ = make_record(tmp_path, monkeypatch)
    from dataclasses import replace
    with pytest.raises(ValueError, match="Reserve"):
        replace(identity, max_model_len=24576).manifest(tokenizer)
    with pytest.raises(ValueError, match="sampling"):
        replace(identity, thought_sampling_profile="unapproved").manifest(tokenizer)
