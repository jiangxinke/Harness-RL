"""Synthetic CPU-only closed original/G4 -> TRAIN rewards -> DPO pair tests."""
from copy import deepcopy
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "scripts"),
                str(Path(__file__).resolve().parents[1] / "slime")]
import dcc_score_native_preferences as S
import dcc_native_dpo_train as TRAIN
from slime_plugins.dcc.data import Task
from slime_plugins.dcc.episode import _WebEvidence

_SPEC = importlib.util.spec_from_file_location("reward_g4_fixture", Path(__file__).with_name("test_dcc_final_preferences.py"))
F = importlib.util.module_from_spec(_SPEC); _SPEC.loader.exec_module(F)
URL = "https://example.org/public"


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def final(answer):
    return {"plan": "Read the original public task.", "checks": "Use actual observed evidence.", "answer": answer}


@pytest.fixture
def closed_domains(tmp_path, monkeypatch):
    cls = F.FIXTURE["FakeTokenizer"]
    monkeypatch.setattr(cls, "__len__", lambda self: 160000, raising=False)
    monkeypatch.setattr(cls, "chat_template", "synthetic-template", raising=False)
    tokenizer = cls()
    cuad = Task("synthetic-reward-cuad", "cuad", "train", "Extract confidentiality.",
                document=F.FIXTURE["DOCUMENT"])
    web = Task("synthetic-reward-web", "widesearch", "train", "Find the name in the public list.",
        metadata={"evaluation": {"required": ["name"]}})
    # Original Web final is rejected/empty, yet its actual public history remains
    # eligible. Original CUAD completion is wrong; neither original reward selects.
    _, _, web_native = F.make_native(tmp_path / "raw-web", values=[F.REVIEW["action"]("web_search"),
        F.REVIEW["action"]("fetch_url"), final([])])
    web_profile = F.source_profile(tokenizer)
    asyncio.run(F.C.generate([web], web_native, tmp_path / "raw-web", collection_id="original-web",
        profile=web_profile, schema_profile=F.G.source_schema_profile(web_profile), web_tools_factory=F.REVIEW["Tools"]))
    from slime_plugins.dcc.cuad_structured import CUAD_STRUCTURED_SCHEMA_VERSION
    profile = F.source_profile(tokenizer)
    profile["cuad"] = {"max_steps": 8, "structured_actions": True, "schema_version": CUAD_STRUCTURED_SCHEMA_VERSION}
    _, _, native = F.make_native(tmp_path / "raw-cuad", values=[{"answer": [{"text": "invented original", "confidence": .5}]}])
    asyncio.run(F.C.generate([cuad], native, tmp_path / "raw-cuad", collection_id="original-cuad",
        profile=profile, schema_profile=F.G.source_schema_profile(profile)))
    public_path = tmp_path / "public-train.jsonl"
    write_jsonl(public_path, [dict(task.to_dict(), source="cuad_train" if task.benchmark == "cuad" else "independent_curriculum")
        for task in (cuad, web)])
    labels_path = tmp_path / "synthetic-TRAIN-labels.jsonl"
    span = "Alpha agrees to preserve confidential information."
    write_jsonl(labels_path, [{"task_id": cuad.id, "benchmark": "cuad", "split": "train",
        "gold": [span], "category": "synthetic-confidence"}, {"task_id": web.id, "benchmark": "widesearch",
        "split": "train", "gold": [{"name": "Alpha"}], "evaluation": {"required": ["name"]}}])
    rows = []
    for benchmark, raw, values in [
        ("cuad", "raw-cuad", [final([{"text": span, "confidence": .5}]), final([]),
            final([{"text": "invented candidate", "confidence": .5}]), final([{"text": span, "confidence": 1.}])]),
        ("widesearch", "raw-web", [final([{"name": "Alpha", "citations": [URL]}]), final([]),
            final([{"name": "Wrong", "citations": [URL]}]), final([{"name": "Alpha", "citations": [URL]}])])]:
        export, generation = tmp_path / ("freeze-" + benchmark), tmp_path / ("g4-" + benchmark)
        kwargs = F.export_args(tokenizer, schema_policy=F.G.FINAL_INTERVENTION, benchmarks=[benchmark])
        kwargs["experiment_id"] += ":" + benchmark
        if benchmark == "cuad": kwargs["source_profile"] = profile
        F.G.export_prefixes(tmp_path / raw, export, **kwargs)
        freeze, prefixes, identity = F.G.load_frozen(export, tokenizer)
        _, _, native = F.make_native(generation, values=values, review=(benchmark == "widesearch"))
        asyncio.run(F.G.generate_g4(prefixes, native, generation, freeze=freeze, identity=identity))
        rows.append({"collection_id": benchmark + "-synthetic", "approved_freeze": str(export),
            "generation_dir": str(generation), "generation_world_size": 1, "expected_source": "f" * 40,
            "original_source_dir": str(tmp_path / raw), "public_train": str(public_path),
            "train_labels": str(labels_path), "reward_version": S.REWARD_VERSIONS[benchmark]})
    manifest = {"version": S.VERSION, "base_derivative_id": "synthetic-derivative",
        "tokenizer_id": "synthetic-tokenizer", "collections": rows}
    return tokenizer, manifest


def test_real_closed_original_and_g4_rewards_roundtrip_to_original_pair_loader(closed_domains, tmp_path):
    tokenizer, manifest = closed_domains
    prepared = S.prepare_collections(manifest, tokenizer)
    for item in prepared:
        lineage = list(S.read_jsonl(Path(item.specification["approved_freeze"]) / "lineage.jsonl"))[0]
        assert lineage["intervention_applied"] is True
        assert lineage["prompt_intervention"]["appended_message_count"] == 1
        messages = json.loads(item.prefixes[0].messages_json)
        assert len(messages) == lineage["prompt_intervention"]["source_message_count"] + 1
        assert messages[-1]["role"] == "user"
        if item.prefixes[0].benchmark == "cuad":
            assert "each at most 40 words" in messages[-1]["content"]
        else:
            assert URL in messages[-1]["content"] and "1 to 240 characters" in messages[-1]["content"]
    result = S.score_prepared_collections(prepared)
    assert [[row["value"] for row in rows] for _, rows, _, _ in result] == [[1., 0., 0., 1.], [1., 0., 0., 1.]]
    profiles = S.publish_rewards(result, tmp_path / "rewards")
    complete = json.loads((tmp_path / "rewards/reward_complete.json").read_text())
    assert complete["fullcounts"]["attempts"] == complete["fullcounts"]["rewarded_candidates"] == 8
    assert complete["fullcounts"]["positive_reward"] == complete["fullcounts"]["zero_reward"] == 4
    assert complete["fullcounts"]["row_item_equal_candidates"] == 3
    assert complete["fullcounts"]["guard_rejected_candidates"] == 1
    for row, profile in zip(manifest["collections"], profiles):
        args = SimpleNamespace(**{key: row[key] for key in (
            "approved_freeze", "generation_dir", "generation_world_size", "expected_source", "reward_version")},
            train_rewards=str(tmp_path / "rewards" / profile["receipt_file"]), base_derivative_id="synthetic-derivative")
        pairs, metadata = TRAIN.load_dpo_pairs(args, tokenizer)
        assert len(pairs) == metadata["eligible_pairs"] == 1
        assert pairs[0].chosen.reward == 1. and pairs[0].rejected.reward == 0.
        assert metadata["phase_request_ids"]
    assert str(tmp_path) not in json.dumps(complete)
    with pytest.raises(FileExistsError): S.publish_rewards(result, tmp_path / "rewards")


@pytest.mark.parametrize("benchmark", ["cuad", "widesearch"])
@pytest.mark.parametrize("damage", ["control_content", "control_role", "control_missing", "control_extra", "source_history"])
def test_original_binding_rejects_changed_control_or_original_public_history(closed_domains, benchmark, damage):
    tokenizer, manifest = closed_domains
    row = next(row for row in manifest["collections"] if row["collection_id"] == benchmark + "-synthetic")
    freeze, prefixes, _ = F.G.load_frozen(row["approved_freeze"], tokenizer)
    value = prefixes[0].to_dict()
    if damage == "control_content": value["messages"][-1]["content"] += " Altered final instruction."
    elif damage == "control_role": value["messages"][-1]["role"] = "system"
    elif damage == "control_missing": value["messages"].pop()
    elif damage == "control_extra": value["messages"].append(deepcopy(value["messages"][-1]))
    else: value["messages"][1]["content"] += " Altered original public history."
    prefixes[0] = type(prefixes[0]).from_dict(value)
    with pytest.raises(ValueError, match="actual original closed request and final control"):
        S._original_binding(row, freeze, prefixes, source_cache={}, public_cache={})


@pytest.mark.parametrize("damage", ["missing", "version", "applied", "preserved", "source_count",
                                    "appended_count", "appended_count_bool", "intervention", "extra_field"])
def test_original_binding_requires_exact_v2_prompt_intervention_lineage(closed_domains, damage):
    tokenizer, manifest = closed_domains
    row = manifest["collections"][1]
    freeze, prefixes, _ = F.G.load_frozen(row["approved_freeze"], tokenizer)
    path = Path(row["approved_freeze"]) / "lineage.jsonl"
    lineage = list(S.read_jsonl(path))
    prompt = lineage[0]["prompt_intervention"]
    if damage == "missing": del lineage[0]["prompt_intervention"]
    elif damage == "version": prompt["version"] = "unapproved"
    elif damage == "applied": prompt["applied"] = 1
    elif damage == "preserved": prompt["source_messages_preserved"] = 1
    elif damage == "source_count": prompt["source_message_count"] += 1
    elif damage == "appended_count": prompt["appended_message_count"] = 0
    elif damage == "appended_count_bool": prompt["appended_message_count"] = True
    elif damage == "intervention": lineage[0]["intervention_applied"] = False
    else: prompt["unapproved"] = True
    write_jsonl(path, lineage)
    with pytest.raises(ValueError, match="intervention lineage differs"):
        S._original_binding(row, freeze, prefixes, source_cache={}, public_cache={})


@pytest.mark.parametrize("damage", ["g4closure", "originalclosure", "publictask", "lineage", "promptlineage",
                                    "phase", "rewardversion", "v1approval", "v1freeze"])
def test_all_domains_and_original_public_binding_precede_any_label_access(closed_domains, monkeypatch, damage):
    tokenizer, manifest = closed_domains
    row = manifest["collections"][1]
    if damage == "g4closure": (Path(row["generation_dir"]) / "generation_closed_rank0.json").unlink()
    elif damage == "originalclosure": (Path(row["original_source_dir"]) / "generation_closed_rank0.json").unlink()
    elif damage == "publictask":
        path = Path(row["public_train"]); rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1]["prompt"] = "Different public task."
        write_jsonl(path, rows)
    elif damage == "lineage":
        path = Path(row["approved_freeze"]) / "lineage.jsonl"; rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["original_reward_used"] = True; write_jsonl(path, rows)
    elif damage == "promptlineage":
        path = Path(row["approved_freeze"]) / "lineage.jsonl"; rows = list(S.read_jsonl(path))
        rows[0]["prompt_intervention"]["appended_message_count"] = 0; write_jsonl(path, rows)
    elif damage in {"v1approval", "v1freeze"}:
        path = Path(row["approved_freeze"]) / ("export_source_approval.json" if damage == "v1approval" else "freeze.json")
        value = json.loads(path.read_text()); value["version"] = "native-final-g4-prospective-v1"
        path.write_text(json.dumps(value))
    elif damage == "rewardversion": row["reward_version"] = "unapproved"
    else:
        original = S.read_closed_g4
        ids = []
        def collision(*args, **kwargs):
            result = original(*args, **kwargs)
            if not ids: ids.extend(result["phase_request_ids"])
            else: result["phase_request_ids"][0] = ids[0]
            return result
        monkeypatch.setattr(S, "read_closed_g4", collision)
    label_access = []
    def forbidden(*args, **kwargs):
        label_access.append(True)
        raise AssertionError("TRAIN labels opened before all domain closure/public ownership")
    monkeypatch.setattr(S, "load_train_labels", forbidden)
    with pytest.raises((ValueError, FileNotFoundError)):
        prepared = S.prepare_collections(manifest, tokenizer)
        S.score_prepared_collections(prepared)
    assert label_access == []


@pytest.mark.parametrize("gold,answer,expected", [
    ([], [], 1.), (["Alpha keeps secrets."], [], 0.),
    ([], [{"text": "Alpha keeps secrets.", "confidence": 1.}], 0.),
    (["Alpha keeps secrets."], [{"text": "Alpha keeps secrets.", "confidence": .1}], 1.),
    (["Alpha keeps secrets."], [{"text": "Alpha keeps secret.", "confidence": .5}], 0.)])
def test_cuad_original_f1_grounding_and_negative_empty_rules(gold, answer, expected):
    task = Task("synthetic-cuad", "cuad", "train", "Extract a clause.", document="Alpha keeps secrets.")
    before = deepcopy(answer)
    reward, _ = S.cuad_reward(task, {"gold": gold}, answer)
    assert reward == expected and answer == before


def test_web_original_entity_profile_mixture_and_guard_do_not_rewrite_answer():
    columns = ["name", "city", "year"]
    task = Task("synthetic-web", "widesearch", "train", "List entities.", metadata={"evaluation": {"required": columns}})
    gold = [{"name": "Alpha", "city": "East", "year": "2000"}, {"name": "Beta", "city": "West", "year": "2001"}]
    label = {"gold": gold, "evaluation": {"required": columns, "unique_columns": ["name"],
        "eval_pipeline": {column: {"preprocess": [], "metric": ["exact_match"]} for column in columns}}}
    guard = _WebEvidence(); guard.observe("web_search", {"results": [{"url": URL}]})
    guard.observe("fetch_url", {"url": URL, "text": "Actual public source text."})
    rows = [dict(row, year="wrong", citations=[URL]) for row in gold]; before = deepcopy(rows)
    reward, components = S.web_reward(task, label, rows, guard)
    assert components["f1_by_row"] == 0 and components["f1_by_item"] == pytest.approx(2/3)
    assert reward == pytest.approx(1/6) and not components["required_only_compound_key"]
    assert rows == before and label["evaluation"]["unique_columns"] == ["name"]
    rows[0]["citations"] = ["https://unread.example.org"]
    reward, components = S.web_reward(task, label, rows, guard)
    assert reward == 0 and components["evidence_guard_rejections"]



def test_existing_header_profile_boundary_is_preserved_without_guessing_columns():
    task = Task("synthetic-uppercase-web", "widesearch", "train", "List names.",
        metadata={"evaluation": {"required": ["Name"]}})
    label = {"gold": [{"Name": "Alpha"}], "evaluation": {"required": ["Name"]}}
    answer = [{"Name": "Alpha", "citations": [URL]}]
    guard = _WebEvidence(); guard.observe("web_search", {"results": [{"url": URL}]})
    guard.observe("fetch_url", {"url": URL, "text": "Actual public evidence."})
    before = deepcopy(label)
    reward, components = S.web_reward(task, label, answer, guard)
    assert reward == 0 and components["scoring_message"] == "Response columns do not match required columns"
    assert label == before and answer[0]["Name"] == "Alpha"
