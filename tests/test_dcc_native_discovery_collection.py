"""CPU-only native collector caller and exact replay tests; synthetic tools only."""
from copy import deepcopy
from dataclasses import replace
import asyncio
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from jsonschema import Draft202012Validator, ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "slime")]
import dcc_collect_native_train as collector
import dcc_collect_native_preferences as preferences
import dcc_native_sft as native
from dcc_review_training import REVIEW20_PROFILE, WEB32K_PROFILE, approved_web_profile
from slime_plugins.dcc.data import Task
from slime_plugins.dcc.web_data_discovery import (
    WEB_DATA_DISCOVERY_GUIDANCE, WEB_DATA_DISCOVERY_PREFIX,
)
from slime_plugins.dcc.prompt_history import PromptHistoryConfig

helpers = runpy.run_path(str(ROOT / "tests/test_dcc_review_training.py"))


def recorded(enabled=None, length=3, version=WEB32K_PROFILE, *, multi_citation=None, citation_count=1):
    options = {} if enabled is None else {"data_discovery_guidance": enabled}
    if multi_citation is not None:
        options["multi_citation"] = multi_citation
    web = native.WebSchemaProfile(20, True, False, True, version, **options)
    task = Task("synthetic-discovery-train", "widesearch", "train",
                "Find the name in the public list.", metadata={"evaluation": {"required": ["Name"]}})
    action = helpers["action"]
    values = [action("web_search"), action("fetch_url")]
    citations = [helpers["URL"]]
    for index in range(1, citation_count):
        extra = action("fetch_url")
        extra["arguments"]["url"] = helpers["URL"] + str(index)
        citations.append(extra["arguments"]["url"])
        values.append(extra)
    final = action()
    final["answer"][0]["citations"] = citations
    policy = helpers["Policy"](values + [action("web_search") for _ in range(length - len(values) - 1)] + [final])
    runner = collector.make_runner("widesearch", native.NativeSchemaProfile(web, None),
        timeout_seconds=5, web_tools_factory=helpers["Tools"])
    result = asyncio.run(runner.run(task, policy, evaluation=True, score_train=False))
    assert result.status == "completed", result.detail
    records = []
    for turn, (messages, schema, action) in enumerate(policy.calls):
        stage = "final" if "answer" in action else {"web_search": "search", "fetch_url": "fetch"}[action["tool"]]
        records.append((dict(source="independent_curriculum", split="train", task_id=task.id,
                             stage=stage, messages=messages, completion=json.dumps(action)), schema, turn))
    return web, records


def test_absent_and_explicit_off_keep_identical_historical_message_and_schema_bytes():
    _, absent = recorded()
    _, off = recorded(False)
    assert json.dumps(absent) == json.dumps(off)
    assert all(WEB_DATA_DISCOVERY_PREFIX not in json.dumps(row) for row in off)
    _, both_off = recorded(False, multi_citation=False)
    assert json.dumps(absent) == json.dumps(both_off)


@pytest.mark.parametrize("version", [REVIEW20_PROFILE, WEB32K_PROFILE])
@pytest.mark.parametrize("length", [3, 20])
def test_guidance_is_once_and_exact_replay_preserves_all_actions_and_schema(version, length):
    _, off = recorded(False, length, version)
    web, on = recorded(True, length, version)
    for before, after in zip(off, on, strict=True):
        old_record, old_options, turn = before
        record, options, actual_turn = after
        original = deepcopy(record)
        expected = deepcopy(old_record)
        expected["messages"][0]["content"] += WEB_DATA_DISCOVERY_GUIDANCE
        assert record == expected and actual_turn == turn and options == old_options
        assert record["messages"][0]["content"].count(WEB_DATA_DISCOVERY_PREFIX) == 1
        assert web.original_schema(record, {"turn": turn}) == options["json_schema"]
        assert record == original


@pytest.mark.parametrize("damage", ["missing", "duplicate", "truncated", "wrong-profile"])
def test_on_profile_rejects_missing_duplicate_changed_or_off_guidance(damage):
    web, rows = recorded(True)
    record, _, turn = deepcopy(rows[-1])
    if damage == "missing":
        record["messages"][0]["content"] = record["messages"][0]["content"].removesuffix(WEB_DATA_DISCOVERY_GUIDANCE)
    elif damage == "duplicate":
        record["messages"][0]["content"] += WEB_DATA_DISCOVERY_GUIDANCE
    elif damage == "truncated":
        record["messages"][0]["content"] = record["messages"][0]["content"][:-1]
    else:
        web = recorded(False)[0]
    with pytest.raises(ValueError, match="Exact approved"):
        web.original_schema(record, {"turn": turn})


def approved_profile(version=WEB32K_PROFILE):
    return {
        "version": version, "collection_overlay_id": "synthetic-cpu-only",
        "identity": {"prompt_history": PromptHistoryConfig(True).to_dict(),
                     "max_prompt_tokens": 32768, "max_model_len": 53248,
                     "thought_tokens": 16384, "thought_sampling_profile": "thinking"},
        "web": {"max_steps": 20, "explicit_review": True, "cached_rows": False,
                "navigation_ledger": True},
    }


@pytest.mark.parametrize("enabled", [None, False, True])
def test_explicit_web32k_profile_keeps_optional_boolean_without_mutating_json(enabled):
    profile = approved_profile()
    if enabled is not None:
        profile["web"]["data_discovery_guidance"] = enabled
    before = deepcopy(profile)
    web = approved_web_profile(profile, native.WebSchemaProfile)
    assert web.data_discovery_guidance is (enabled is True) and profile == before


@pytest.mark.parametrize("bad", [1, 0, "true", None])
def test_profile_rejects_nonboolean_guidance(bad):
    profile = approved_profile()
    profile["web"]["data_discovery_guidance"] = bad
    with pytest.raises(ValueError, match="discovery guidance"):
        approved_web_profile(profile, native.WebSchemaProfile)


def test_legacy_profile_cannot_implicitly_enable_new_guidance():
    with pytest.raises(ValueError, match="explicit reviewed"):
        native.WebSchemaProfile(20, data_discovery_guidance=True).validate()


@pytest.mark.parametrize("enabled", [None, False, True])
def test_multi_citation_is_optional_boolean_and_keeps_approved_json_unchanged(enabled):
    profile = approved_profile()
    profile["web"]["data_discovery_guidance"] = True
    if enabled is not None:
        profile["web"]["multi_citation"] = enabled
    before = deepcopy(profile)
    web = approved_web_profile(profile, native.WebSchemaProfile)
    assert web.multi_citation is (enabled is True) and profile == before


@pytest.mark.parametrize("bad", [1, 0, "true", None])
def test_profile_rejects_nonboolean_multi_citation(bad):
    profile = approved_profile()
    profile["web"]["multi_citation"] = bad
    with pytest.raises(ValueError, match="multi-citation"):
        approved_web_profile(profile, native.WebSchemaProfile)


def test_legacy_profile_cannot_enable_multi_citation():
    with pytest.raises(ValueError, match="explicit reviewed"):
        native.WebSchemaProfile(20, multi_citation=True).validate()


@pytest.mark.parametrize("name", ["web_data_discovery_guidance", "web_multi_citation"])
@pytest.mark.parametrize("bad", [False, 0, 1, "true", None])
def test_prospective_optional_freeze_keys_require_explicit_true_without_bool_int_coercion(name, bad):
    with pytest.raises(ValueError, match="explicit true booleans"):
        preferences.optional_web_harness({name: bad})


@pytest.mark.parametrize("cap", [True, False, 0, 2, 3, 5, "1", None])
def test_prefix_citation_cap_requires_canonical_integer_in_every_row_alternative(cap):
    from slime_plugins.dcc.web_structured import web_final_schema
    schema = web_final_schema(["Name", "Other"], [helpers["URL"]])
    schema["properties"]["answer"]["items"]["anyOf"][-1]["properties"]["citations"]["maxItems"] = cap
    prefix = SimpleNamespace(benchmark="widesearch", messages_json=json.dumps([
        {"role": "system", "content": "Original synthetic system."}]), schema_json=json.dumps(schema))
    with pytest.raises(ValueError, match="citation caps"):
        preferences.validate_optional_web_prefixes({}, [prefix])


def test_discovery_marker_in_source_observation_does_not_enable_system_guidance():
    from slime_plugins.dcc.web_structured import web_final_schema
    prefix = SimpleNamespace(benchmark="widesearch", messages_json=json.dumps([
        {"role": "system", "content": "Original synthetic system."},
        {"role": "user", "content": "Quoted source: " + WEB_DATA_DISCOVERY_GUIDANCE}]),
        schema_json=json.dumps(web_final_schema(["Name"], [helpers["URL"]])))
    preferences.validate_optional_web_prefixes({}, [prefix])


@pytest.mark.parametrize("version", [REVIEW20_PROFILE, WEB32K_PROFILE])
@pytest.mark.parametrize("length", [4, 20])
@pytest.mark.parametrize("guidance", [False, True])
def test_multi_citation_replays_real_runner_two_source_final_and_all_request_schemas(version, length, guidance):
    web, rows = recorded(guidance, length, version, multi_citation=True, citation_count=2)
    for record, options, turn in rows:
        before = deepcopy(record)
        assert web.original_schema(record, {"turn": turn}) == options["json_schema"]
        assert record == before
    record, options, turn = rows[-1]
    final = json.loads(record["completion"])
    Draft202012Validator(options["json_schema"]).validate(final)
    off = native.WebSchemaProfile(20, True, False, True, version,
                                  data_discovery_guidance=guidance)
    with pytest.raises(ValidationError):
        Draft202012Validator(off.original_schema(record, {"turn": turn})).validate(final)
    final["answer"][0]["citations"].extend([helpers["URL"]] * 3)
    with pytest.raises(ValidationError):
        Draft202012Validator(options["json_schema"]).validate(final)
    final["answer"][0]["citations"] = ["https://unread.invalid/"]
    with pytest.raises(ValidationError):
        Draft202012Validator(options["json_schema"]).validate(final)


def test_optin_native_physical_targets_keep_real_ids_and_reject_off_schema(tmp_path, monkeypatch):
    fixture = runpy.run_path(str(ROOT / "tests/test_dcc_native_thinking.py"))
    cls = fixture["FakeTokenizer"]
    monkeypatch.setattr(cls, "__len__", lambda self: 160000, raising=False)
    monkeypatch.setattr(cls, "chat_template", "synthetic-template", raising=False)
    from slime_plugins.dcc.native_budget import NativeBudget

    budget = NativeBudget(16384, "thinking")
    web, rows = recorded(True, 4, REVIEW20_PROFILE, multi_citation=True, citation_count=2)
    record, options, turn = rows[-1]
    tokenizer, _, policy = fixture["make_policy"](
        tmp_path, values=[json.loads(record["completion"])], budget=budget)
    identity = native.NativeIdentity("synthetic-derivative", (151643, 151645), "f" * 40,
        16384, 16384 + budget.total_tokens, 42, budget.thought_tokens, budget.thought_sampling_profile)

    async def produce():
        try:
            await policy.generate_structured(record["messages"], max_new_tokens=budget.total_tokens,
                temperature=0.0, json_schema=options["json_schema"], trace_metadata={
                    "trace_id": "synthetic-optin-trace", "task_id": record["task_id"],
                    "rank": 0, "turn_index": turn})
        finally:
            await policy.close()

    asyncio.run(produce())
    trace = policy.traces[0]
    bundle = {"contract": native.CONTRACT, "record": deepcopy(record),
        "identity": identity.manifest(tokenizer), "native_trace": deepcopy(trace),
        "collector": {"split": "train", "task_input": "public-train-task-without-gold",
            "gold_visible_to_policy": False, "recorded_messages": True,
            "synthetic_thought": False, "inserted_close": False, "inserted_eos": False,
            "episode_id": "synthetic-optin-episode", "turn": turn, "rank": 0,
            "trace_id": "synthetic-optin-trace"}}
    before = deepcopy(bundle)
    example = native.validate_native_record(bundle, tokenizer, expected_identity=identity,
        original_schema=web.original_schema(record, {"turn": turn}), web_schema_profile=web)
    assert example.completion == trace["completion_ids"]
    assert example.prompt == trace["original_prompt_ids"] and bundle == before
    off = native.WebSchemaProfile(20, True, False, True, REVIEW20_PROFILE,
                                  data_discovery_guidance=True)
    with pytest.raises(ValueError, match="Recorded schema differs"):
        native.validate_native_record(bundle, tokenizer, expected_identity=identity,
            original_schema=off.original_schema(record, {"turn": turn}), web_schema_profile=off)


@pytest.mark.parametrize("cli_enabled,profile_enabled", [(True, True), (True, False), (False, True)])
def test_actual_cli_binds_profile_before_public_data_and_forwards_engine_optin(tmp_path, cli_enabled, profile_enabled):
    tokenizer = object()
    profile = {"identity": {"eos_token_ids": [11]}, "web": {"data_discovery_guidance": profile_enabled}}
    identity = SimpleNamespace(max_prompt_tokens=32768, max_model_len=53248,
        generation_seed=47, budget=object(), eos_token_ids=(11,),
        manifest=lambda actual: profile["identity"] if actual is tokenizer else None)
    base = SimpleNamespace(eos_token_ids=[11], tokenizer=tokenizer, allow_truncation=True)
    argv = ["dcc_collect_native_train.py", "generate", "--source-root", str(ROOT),
            "--expected-source", "f" * 40, "--profile", str(tmp_path / "synthetic.json"),
            "--model", "CPU_FAKE_MODEL", "--output-dir", str(tmp_path / "fresh"),
            "--public-train-data", str(tmp_path / "not-opened-public.jsonl"),
            "--collection-id", "synthetic-cpu-only"]
    if cli_enabled:
        argv.append("--web-data-discovery-guidance")
    with patch.object(sys, "argv", argv), \
         patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
         patch.object(collector, "load_approved", return_value=(profile, identity, object(), {})), \
         patch.object(collector, "load_public_train", return_value=[]) as public, \
         patch.object(collector, "fixed_tasks", return_value=[]), \
         patch("slime_plugins.dcc.vllm_policy.load_vllm_policy", return_value=base) as backend, \
         patch.object(collector, "NativeTraceJournal", return_value=object()), \
         patch.object(collector, "NativeThinkingPolicy", return_value=object()), \
         patch.object(collector, "generate", new=AsyncMock()):
        if cli_enabled != profile_enabled:
            with pytest.raises(ValueError, match="Discovery guidance flag"):
                collector.main()
            public.assert_not_called()
            backend.assert_not_called()
        else:
            collector.main()
            assert backend.call_args.kwargs["web_data_discovery_guidance"] is True


@pytest.mark.parametrize("cli_enabled,profile_enabled", [(False, False), (True, True), (True, False), (False, True)])
def test_actual_cli_binds_multi_citation_profile_before_public_data_and_engine(tmp_path, cli_enabled, profile_enabled):
    tokenizer = object()
    profile = {"identity": {"eos_token_ids": [11]}, "web": {"multi_citation": profile_enabled}}
    identity = SimpleNamespace(max_prompt_tokens=32768, max_model_len=53248,
        generation_seed=47, budget=object(), eos_token_ids=(11,),
        manifest=lambda actual: profile["identity"] if actual is tokenizer else None)
    base = SimpleNamespace(eos_token_ids=[11], tokenizer=tokenizer, allow_truncation=True)
    argv = ["dcc_collect_native_train.py", "generate", "--source-root", str(ROOT),
            "--expected-source", "f" * 40, "--profile", str(tmp_path / "synthetic.json"),
            "--model", "CPU_FAKE_MODEL", "--output-dir", str(tmp_path / "fresh"),
            "--public-train-data", str(tmp_path / "not-opened-public.jsonl"),
            "--collection-id", "synthetic-cpu-only"]
    if cli_enabled:
        argv.append("--web-multi-citation")
    with patch.object(sys, "argv", argv), \
         patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
         patch.object(collector, "load_approved", return_value=(profile, identity, object(), {})), \
         patch.object(collector, "load_public_train", return_value=[]) as public, \
         patch.object(collector, "fixed_tasks", return_value=[]), \
         patch("slime_plugins.dcc.vllm_policy.load_vllm_policy", return_value=base) as backend, \
         patch.object(collector, "NativeTraceJournal", return_value=object()), \
         patch.object(collector, "NativeThinkingPolicy", return_value=object()), \
         patch.object(collector, "generate", new=AsyncMock()):
        if cli_enabled != profile_enabled:
            with pytest.raises(ValueError, match="Multi-citation flag"):
                collector.main()
            public.assert_not_called()
            backend.assert_not_called()
        else:
            collector.main()
            assert backend.call_args.kwargs.get("web_multi_citation", False) is cli_enabled
            assert ("web_multi_citation" in backend.call_args.kwargs) is cli_enabled
            assert "web_data_discovery_guidance" not in backend.call_args.kwargs


@pytest.mark.parametrize("guidance", [False, True])
@pytest.mark.parametrize("multi_citation", [False, True])
@pytest.mark.parametrize("length", [4, 20])
def test_preference_public_replay_binds_both_original_and_final_schemas(guidance, multi_citation, length):
    web, rows = recorded(guidance, length, REVIEW20_PROFILE,
                         multi_citation=multi_citation, citation_count=2 if multi_citation else 1)
    record, options, turn = rows[-1]
    task = Task(record["task_id"], "widesearch", "train", record["messages"][1]["content"],
                metadata={"evaluation": {"required": ["Name"]}})
    original, final, fetched = preferences.public_request_schemas(task, record["messages"],
        turn=turn, schemas=native.NativeSchemaProfile(web))
    assert original == options["json_schema"]
    assert len(fetched) == (2 if multi_citation else 1)
    Draft202012Validator(final).validate(json.loads(record["completion"]))
    assert (original == final) is (length == 20)


@pytest.mark.parametrize("guidance", [False, True])
@pytest.mark.parametrize("multi_citation", [False, True])
def test_real_closed_collector_to_frozen_g4_binds_optional_harness_and_runtime(tmp_path, monkeypatch,
                                                                           guidance, multi_citation):
    from slime_plugins.dcc.web_data_discovery import web_data_discovery_configuration
    from slime_plugins.dcc.web_structured import web_multi_citation_configuration

    fixtures = runpy.run_path(str(ROOT / "tests/test_dcc_final_preferences.py"))
    cls = fixtures["FIXTURE"]["FakeTokenizer"]
    monkeypatch.setattr(cls, "__len__", lambda self: 160000, raising=False)
    monkeypatch.setattr(cls, "chat_template", "synthetic-template", raising=False)
    tokenizer = cls()
    profile = fixtures["source_profile"](tokenizer)
    configuration = {}
    if guidance:
        profile["web"]["data_discovery_guidance"] = True
        configuration["web_data_discovery_guidance"] = web_data_discovery_configuration(True)
    if multi_citation:
        profile["web"]["multi_citation"] = True
        configuration["web_multi_citation"] = web_multi_citation_configuration(True)
    final = helpers["action"]()
    values = [helpers["action"]("web_search"), helpers["action"]("fetch_url")]
    if multi_citation:
        extra = helpers["action"]("fetch_url")
        extra["arguments"]["url"] += "2"
        values.append(extra)
        final["answer"][0]["citations"].append(extra["arguments"]["url"])
    values.append(final)
    _, _, policy = fixtures["make_native"](tmp_path / "raw", values=values)
    policy.base.provenance.update(deepcopy(configuration))
    _, _, wrong_source = fixtures["make_native"](tmp_path / "wrong-source-engine", values=[])
    wrong_source.base.provenance.update(deepcopy(configuration))
    wrong_source.base.provenance["web_multi_citation"] = web_multi_citation_configuration(not multi_citation)
    with pytest.raises(ValueError, match="optional Web harness provenance"):
        asyncio.run(collector.generate([fixtures["web_task"]()], wrong_source,
            tmp_path / "source-not-created", collection_id="synthetic-optin-wrong",
            profile=profile, schema_profile=preferences.source_schema_profile(profile)))
    assert not (tmp_path / "source-not-created").exists()
    wrong_profile = deepcopy(profile)
    wrong_profile["web"]["multi_citation"] = not multi_citation
    with pytest.raises(ValueError, match="approved profile and actual runner"):
        asyncio.run(collector.generate([fixtures["web_task"]()], policy,
            tmp_path / "runner-not-created", collection_id="synthetic-optin-wrong-runner",
            profile=wrong_profile, schema_profile=preferences.source_schema_profile(profile)))
    assert not (tmp_path / "runner-not-created").exists()
    asyncio.run(collector.generate([fixtures["web_task"]()], policy, tmp_path / "raw",
        collection_id="synthetic-optin-original", profile=profile,
        schema_profile=preferences.source_schema_profile(profile), web_tools_factory=helpers["Tools"]))
    args = fixtures["export_args"](tokenizer, schema_policy=preferences.FINAL_INTERVENTION)
    args["source_profile"] = profile
    freeze = preferences.export_prefixes(tmp_path / "raw", tmp_path / "freeze", **args)
    assert ("web_data_discovery_guidance" in freeze) is guidance
    assert ("web_multi_citation" in freeze) is multi_citation
    loaded, prefixes, identity = preferences.load_frozen(tmp_path / "freeze", tokenizer)
    assert loaded == freeze
    approval_path = tmp_path / "freeze/export_source_approval.json"
    approval_bytes = approval_path.read_bytes()
    if not guidance and not multi_citation:
        approval_path.unlink()
        assert preferences.load_frozen(tmp_path / "freeze", tokenizer)[0] == freeze
        approval_path.write_bytes(approval_bytes)
    _, _, candidates = fixtures["make_native"](tmp_path / "g4", values=[deepcopy(final) for _ in range(4)])
    candidates.base.provenance.update(deepcopy(configuration))
    asyncio.run(preferences.generate_g4(prefixes, candidates, tmp_path / "g4",
        freeze=freeze, identity=identity))
    closed = preferences.read_closed_g4(tmp_path / "g4", freeze=freeze, prefixes=prefixes,
        tokenizer=tokenizer, world_size=1)
    assert len(closed["candidates"]) == 4 and closed["generation_closed_before_scoring"] is True
    _, _, wrong = fixtures["make_native"](tmp_path / "wrong-engine", values=[])
    wrong.base.provenance.update(deepcopy(configuration))
    wrong.base.provenance["web_multi_citation"] = web_multi_citation_configuration(not multi_citation)
    with pytest.raises(ValueError, match="optional Web harness provenance"):
        asyncio.run(preferences.generate_g4(prefixes, wrong, tmp_path / "not-created",
            freeze=freeze, identity=identity))
    assert not (tmp_path / "not-created").exists()
    for name, enabled, builder in (
            ("web_data_discovery_guidance", guidance, web_data_discovery_configuration),
            ("web_multi_citation", multi_citation, web_multi_citation_configuration)):
        for damage in ("boolean-int", "extra-field", "unsupported-version"):
            wrong.base.provenance.update({
                "web_data_discovery_guidance": web_data_discovery_configuration(guidance),
                "web_multi_citation": web_multi_citation_configuration(multi_citation)})
            malformed = builder(enabled)
            if damage == "boolean-int":
                malformed["enabled"] = int(enabled)
            elif damage == "extra-field":
                malformed["unapproved"] = True
            else:
                malformed["version"] = "unsupported-runtime-version"
            wrong.base.provenance[name] = malformed
            with pytest.raises(ValueError, match="optional Web harness provenance"):
                asyncio.run(preferences.generate_g4(prefixes, wrong, tmp_path / "malformed-not-created",
                    freeze=freeze, identity=identity))
            with pytest.raises(ValueError, match="optional Web harness provenance"):
                asyncio.run(collector.generate([fixtures["web_task"]()], wrong,
                    tmp_path / "malformed-source-not-created", collection_id="synthetic-malformed",
                    profile=profile, schema_profile=preferences.source_schema_profile(profile)))
            assert not (tmp_path / "malformed-not-created").exists()
            assert not (tmp_path / "malformed-source-not-created").exists()
    if guidance and multi_citation:
        original_prefix_bytes = (tmp_path / "freeze/prefixes.jsonl").read_bytes()
        for damage in ("removed-approval-and-flags", "missing-guidance", "changed-guidance",
                       "duplicate-guidance", "downgraded-caps"):
            altered_freeze, altered = deepcopy(freeze), prefixes[0]
            if damage == "removed-approval-and-flags":
                altered_freeze.pop("web_data_discovery_guidance")
                altered_freeze.pop("web_multi_citation")
                approval_path.unlink()
            elif damage == "downgraded-caps":
                schema = json.loads(altered.schema_json)
                for alternative in schema["properties"]["answer"]["items"]["anyOf"]:
                    alternative["properties"]["citations"]["maxItems"] = 1
                altered = replace(altered, schema_json=json.dumps(schema))
            else:
                messages = json.loads(altered.messages_json)
                if damage == "missing-guidance":
                    messages[0]["content"] = messages[0]["content"].removesuffix(WEB_DATA_DISCOVERY_GUIDANCE)
                elif damage == "changed-guidance":
                    messages[0]["content"] += "Changed suffix."
                else:
                    messages[0]["content"] += WEB_DATA_DISCOVERY_GUIDANCE
                altered = replace(altered, messages_json=json.dumps(messages))
            (tmp_path / "freeze/freeze.json").write_text(json.dumps(altered_freeze))
            (tmp_path / "freeze/prefixes.jsonl").write_text(json.dumps(altered.to_dict()) + "\n")
            with pytest.raises(ValueError, match="guidance|citation caps|data discovery"):
                preferences.load_frozen(tmp_path / "freeze", tokenizer)
            wrong.base.provenance.update({
                "web_data_discovery_guidance": web_data_discovery_configuration(
                    altered_freeze.get("web_data_discovery_guidance", False)),
                "web_multi_citation": web_multi_citation_configuration(
                    altered_freeze.get("web_multi_citation", False))})
            with pytest.raises(ValueError, match="guidance|citation caps|data discovery"):
                asyncio.run(preferences.generate_g4([altered], wrong,
                    tmp_path / "prefix-mismatch-not-created", freeze=altered_freeze, identity=identity))
            assert not (tmp_path / "prefix-mismatch-not-created").exists()
            if not approval_path.exists():
                approval_path.write_bytes(approval_bytes)
            (tmp_path / "freeze/prefixes.jsonl").write_bytes(original_prefix_bytes)
        (tmp_path / "freeze/freeze.json").write_text(json.dumps(freeze))
    damaged = deepcopy(freeze)
    if guidance:
        damaged.pop("web_data_discovery_guidance")
    else:
        damaged["web_data_discovery_guidance"] = True
    (tmp_path / "freeze/freeze.json").write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="approved original source profile|guidance|data discovery"):
        preferences.load_frozen(tmp_path / "freeze", tokenizer)
