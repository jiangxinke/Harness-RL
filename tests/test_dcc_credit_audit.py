"""Independent numeric fixtures and corruption checks for the read-only auditor."""
from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import math
from pathlib import Path
import struct

import pytest

from slime_plugins.dcc.data import Task
from slime_plugins.dcc.episode import EpisodeRunner, PolicyCompletion
from slime_plugins.dcc.graphpo_collector import collect_graphpo


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("dcc_credit_audit", ROOT / "scripts/dcc_credit_audit.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class Tokenizer:
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, **kwargs):
        return self.encode(json.dumps(messages))


TOKENIZER = Tokenizer()


def make_rollout(success=True, *, task_id="fixture__Document Name", document="PROMOTION AGREEMENT."):
    task = Task(task_id, "cuad", "train", "Find the contract title.", document, [document.rstrip(".")])
    replies = iter([
        '{"tool":"search_document","arguments":{"query":"title"}}',
        json.dumps({"answer": [{"text": document.rstrip("."), "confidence": 1.}] if success else []}),
    ])

    async def policy(messages, **kwargs):
        text = next(replies)
        ids = TOKENIZER.encode(text)
        return PolicyCompletion(text, TOKENIZER.apply_chat_template(messages), ids, [-.2] * len(ids))

    result = asyncio.run(EpisodeRunner(max_steps=2, initial_chunks=1, cuad_protocol="cuad-audited-v2").run(task, policy))
    return task, result.to_dict()


def hand_fixture(method):
    task, good = make_rollout(True)
    _, bad = make_rollout(False)
    raw = [good, bad]
    evidence = audit.Evidence()
    records = [audit.episode_record(r, i + 1, "training", "cuad-audited-v2", evidence, Path("fixture.jsonl")) for i, r in enumerate(raw)]
    macro = 5. / (math.sqrt(100 / 3) + 1e-6)
    local = 5. / (math.sqrt(50) + 1e-6)
    root_local = 4.75 / (math.sqrt(45.125) + 1e-6)
    gains = [.2375, 5.25, .2375, -4.75]
    gain_mean = sum(gains) / 4
    gain_std = math.sqrt(sum((v - gain_mean) ** 2 for v in gains) / 3)
    rows, nested = [], [[], []]
    for i in range(4):
        episode, step = divmod(i, 2)
        sign = 1 if episode == 0 else -1
        row = dict(episode=episode, step=step, task_id=task.id, is_action_valid=True,
                   task_reward=10. if episode == 0 else 0., episode_advantage=sign * macro,
                   group=step, node=step)
        if method == "grpo":
            row["advantage"] = sign * macro
        elif method == "gigpo":
            row.update(discounted_return=(9.5 if step == 0 else 10.) if episode == 0 else 0.,
                       local_advantage=sign * (root_local if step == 0 else local))
            row["advantage"] = row["episode_advantage"] + row["local_advantage"]
        elif method == "graphgpo":
            distance = 1 if step == 0 else 0 if episode == 0 else 3
            glocal = 0. if step == 0 else sign * (4.96 / (math.sqrt(49.2032) + 1e-6))
            row.update(graph_return=10. * .2 ** distance, distance_target=distance,
                       target_reachable=not (episode == 1 and step == 1),
                       local_advantage=glocal, graph_advantage=glocal)
            row["advantage"] = row["episode_advantage"] + glocal
        else:
            nc = 0. if step == 0 else sign * local
            ec = (gains[i] - gain_mean) / (gain_std + 1e-6)
            row.update(discounted_return=(9.025 if step == 0 else 9.5) if episode == 0 else 0.,
                       node_value=4.5125 if step == 0 else 4.75,
                       next_value=4.75 if step == 0 else 10. if episode == 0 else 0.,
                       edge_raw_gain=gains[i], node_advantage=nc, edge_advantage=ec,
                       local_advantage=nc + ec)
            row["advantage"] = row["episode_advantage"] + nc + ec
        rows.append(row)
        nested[episode].append(row["advantage"])
    record = dict(iteration=1, stage=method, step=1, optimizer_step_applied=True, gradient_norm=1.,
                  rewards=[1., 0.], statuses=["completed", "completed"], task=task.id,
                  advantages=nested, unique_training_decisions=4, rollout_count=2,
                  credit=dict(rows=rows, gamma=.95, omega=.2, weight=1., similarity_threshold=1.,
                              invalid_penalty=.1, mode="mean_std_norm"))
    return task, raw, records, record, evidence


@pytest.mark.parametrize("method", ["grpo", "gigpo", "graphgpo", "g2po"])
def test_independent_hand_values_reconstruct_all_chain_methods(method):
    _, raw, records, record, evidence = hand_fixture(method)
    decisions, groups = audit.audit_chain(record, raw, records, method, {}, evidence, "hand")
    assert len(decisions) == 4
    assert all(c["status"] == "passed" for c in evidence.checks), evidence.checks
    if method != "grpo":
        assert [g["size"] for g in groups] == [2, 2]


def test_local_component_corruption_is_caught_even_if_total_is_unchanged():
    _, raw, records, record, evidence = hand_fixture("gigpo")
    record["credit"]["rows"][0]["local_advantage"] += .5
    record["credit"]["rows"][0]["episode_advantage"] -= .5
    audit.audit_chain(record, raw, records, "gigpo", {}, evidence, "corrupt")
    failed = {c["name"] for c in evidence.checks if c["status"] == "failed"}
    assert {"local_advantage", "episode_advantage"} <= failed
    assert "advantage" not in failed


def test_false_group_merging_is_caught_even_when_each_group_is_internally_consistent():
    _, raw, records, record, evidence = hand_fixture("gigpo")
    for row in record["credit"]["rows"]:
        row["group"] = 0
    audit.audit_chain(record, raw, records, "gigpo", {}, evidence, "group-corrupt")
    assert any(c["name"] == "complete_group_partition" and c["status"] == "failed" for c in evidence.checks)


def test_jsonl_keeps_unicode_line_separators_inside_strings(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps({"text": "a\u2028b\u2029c"}, ensure_ascii=False) + "\n", encoding="utf-8")
    evidence = audit.Evidence()
    assert evidence.load(path, lines=True) == [{"text": "a\u2028b\u2029c"}]


def test_graphpo_unique_edges_not_terminal_prefix_occurrences():
    task = Task("graph__Document Name", "cuad", "train", "Find title.", "PROMOTION AGREEMENT.", ["PROMOTION AGREEMENT"])

    async def policy(messages, **kwargs):
        count = sum(m["role"] == "assistant" for m in messages)
        text = ('{"answer":[{"text":"PROMOTION AGREEMENT","confidence":1}]}' if count == 2
                else '{"tool":"search_document","arguments":{"query":"promotion"}}')
        return PolicyCompletion(text, TOKENIZER.apply_chat_template(messages), TOKENIZER.encode(text))

    async def summary(messages):
        return "Observed promotion agreement"

    async def embed(texts):
        return [[1., 0.] for text in texts]

    summary.usage = dict(input_tokens=0, output_tokens=0, calls=0)
    embed.usage = dict(input_tokens=0, texts=0, calls=0)
    result = asyncio.run(collect_graphpo(task, EpisodeRunner(max_steps=3, cuad_protocol="cuad-audited-v2"),
                                        policy, tokenizer=TOKENIZER, summarize=summary, embed=embed,
                                        branching=(2, 2, 1)))
    raw = [r.to_dict() for r in result.results]
    evidence = audit.Evidence()
    records = [audit.episode_record(r, i + 1, "training", "cuad-audited-v2", evidence, Path("fixture")) for i, r in enumerate(raw)]
    decisions, groups = audit.audit_graphpo({"credit": json.loads(json.dumps(result.diagnostics))}, raw, records, evidence, "graph")
    assert len(decisions) == len(result.training_items) == 10
    assert sum(len(r["steps"]) for r in records) == 12
    assert len({d["edge_id"] for d in decisions}) == 10
    assert all(c["status"] == "passed" for c in evidence.checks), evidence.checks
    assert groups[0]["accepted_merge_pairs"]


def write_checkpoint(directory):
    final = directory / "train/final"
    final.mkdir(parents=True)
    config = dict(optimizer_steps=1, cuad_protocol="cuad-audited-v2", adv_estimator="gigpo")
    (final / "training_config.json").write_text(json.dumps(config), encoding="utf-8")
    (final / "adapter_config.json").write_text(json.dumps({"r": 1, "base_model_name_or_path": "fixture"}), encoding="utf-8")
    header = json.dumps({"layer.lora_A.weight": {"dtype": "F32", "shape": [1, 1], "data_offsets": [0, 4]},
                         "layer.lora_B.weight": {"dtype": "F32", "shape": [1, 1], "data_offsets": [4, 8]}}).encode()
    (final / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<ff", .25, .125))
    try:
        import torch
    except ImportError:
        return
    torch.save({"optimizer_steps": 1, "optimizer": {"state": {0: {"step": torch.tensor(1.)}}}}, final / "training_state.pt")


def write_run(directory):
    task, raw, _, record, _ = hand_fixture("gigpo")
    write_checkpoint(directory)
    train = directory / "train"
    evaluation = directory / "evaluation"
    evaluation.mkdir()
    eval_task, eval_record = make_rollout(True, task_id="evaluation__Document Name", document="OTHER AGREEMENT.")
    data = directory / "data"
    data.mkdir()
    train_data, eval_data = data / "pilot_train.jsonl", data / "pilot_validation.jsonl"
    train_data.write_text(json.dumps(task.to_dict()) + "\n", encoding="utf-8")
    eval_data.write_text(json.dumps(eval_task.to_dict()) + "\n", encoding="utf-8")
    manifest = dict(method="gigpo", status="not_started", source_commit="fixture",
                    commands={"train": ["--train-data", str(train_data)], "evaluate": ["--data", str(eval_data)]},
                    dataset_sha256={train_data.name: audit.sha(train_data), eval_data.name: audit.sha(eval_data)})
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (directory / "status.json").write_text(json.dumps(dict(state="finished", returncode=0)), encoding="utf-8")
    start = dict(event="start", algorithm="LoRA GIGPO", adv_estimator="gigpo", cuad_protocol="cuad-audited-v2")
    end = dict(event="finished", optimizer_steps=1, elapsed=1.)
    (train / "train_rank0.jsonl").write_text("\n".join(json.dumps(r) for r in [start, record, end]) + "\n", encoding="utf-8")
    (train / "rollouts_rank0.jsonl").write_text("\n".join(json.dumps(r) for r in raw) + "\n", encoding="utf-8")
    (evaluation / "rollouts.jsonl").write_text(json.dumps(eval_record) + "\n", encoding="utf-8")
    summary = dict(settings={"cuad_protocol": "cuad-audited-v2"}, training=False, rollouts=1,
                   tasks=[dict(task=eval_task.id, rewards=[1.], f1=[1.], statuses=["completed"], decisions=[2])])
    (evaluation / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return record


def test_full_readonly_audit_ignores_static_manifest_status_and_rescores_sources(tmp_path):
    write_run(tmp_path)
    before = {str(p): audit.sha(p) for p in tmp_path.rglob("*") if p.is_file()}
    report = audit.audit_run(tmp_path)
    after = {str(p): audit.sha(p) for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after
    assert not [c for c in report["checks"] if c["status"] == "failed"], report["checks"]
    assert report["source_rescoring"]["rescored_rollouts"] == 3
    assert report["training_statistics"]["rollout_count"] == 2
    assert report["checkpoint"]["optimizer_steps"] == 1
    if not (tmp_path / "train/final/training_state.pt").exists():
        assert report["verdict"] == "incomplete"
    else:
        assert report["verdict"] == "verified_record_consistency"


def test_corrupted_reward_and_update_counter_do_not_pass(tmp_path):
    record = write_run(tmp_path)
    path = tmp_path / "train/train_rank0.jsonl"
    lines = [json.loads(line) for line in path.open(encoding="utf-8")]
    lines[1]["rewards"][0] = 0.
    lines[1]["step"] = 2
    path.write_text("\n".join(json.dumps(r) for r in lines) + "\n", encoding="utf-8")
    report = audit.audit_run(tmp_path)
    assert report["verdict"] == "failed"
    failed = {c["name"] for c in report["checks"] if c["status"] == "failed"}
    assert {"reward_matches_rollout", "cumulative_optimizer_step"} <= failed


def test_repeated_error_feedback_is_not_reported_as_repeated_source_return():
    evidence = audit.Evidence()
    prefix = "Tool observation (source material):\n"
    messages = [{"role": "user", "content": "Initial contract retrieval (source material):\n{}"}]
    steps = []
    for _ in range(2):
        steps.append(dict(text="invalid", prompt_ids=[1], completion_ids=[2]))
        messages.extend([{"role": "assistant", "content": "invalid"},
                         {"role": "user", "content": prefix + '{"error":"invalid action"}'}])
    rows = audit.trace_rows(dict(messages=messages, steps=steps, status="max_steps", reward=0), "cuad-audited-v2", evidence, "errors")
    stats = audit.statistics([], rows)
    assert stats["repeated_error_feedback_decisions"] == 1
    assert stats["repeated_source_return_decisions"] == 0


def test_mathematical_constant_group_is_zero_independent_of_float32_artifacts():
    values = [6.883371829986572] * 8
    assert audit.normalize(values) == [0.] * 8
    torch = pytest.importorskip("torch")
    t = torch.tensor(values, dtype=torch.float32)
    replay = (t - t.mean()) / (t.std() + 1e-6)
    # The exact upstream result is reported separately; platform reductions may vary.
    assert torch.isfinite(replay).all()
    assert all(v == replay[0] for v in replay)


def test_guard_audit_accepts_only_exact_constant_groups_and_keeps_original_signal():
    rows = [dict(task_id="q", group=0, episode_score=10., episode_advantage=0.,
                 local_score=6.883371829986572, local_advantage=0.) for _ in range(8)]
    credit = dict(rows=rows, numeric_guard="exact_constant_groups_zero_v1", normalization_corrections={
        "episode": [], "local": [dict(indices=list(range(8)), constant_score=6.883371829986572,
                                       upstream_advantages=[-.3158363401889801] * 8)]})
    evidence = audit.Evidence()
    report = audit.audit_normalization_guard(credit, "gigpo", "mean_std_norm", evidence, "guard")
    assert report[1]["corrections"][0]["upstream_advantages"][0] == -.3158363401889801
    assert not [c for c in evidence.checks if c["status"] == "failed"], evidence.checks
    broken = copy.deepcopy(credit)
    broken["rows"][0]["local_score"] = 1.
    evidence = audit.Evidence()
    audit.audit_normalization_guard(broken, "gigpo", "mean_std_norm", evidence, "guard")
    assert any(c["name"] == "guard_only_exact_constant_groups" and c["status"] == "failed" for c in evidence.checks)


def test_guard_cannot_modify_nonconstant_groups():
    scale = math.sqrt(50) + 1e-6
    rows = [dict(task_id="q", group=0, adjusted_episode_score=reward,
                 episode_advantage=(reward - 5) / scale,
                 upstream_episode_advantage=(reward - 5) / scale,
                 adjusted_graph_return=value, local_advantage=changed,
                 upstream_local_advantage=original)
            for reward, value, changed, original in [(0., 2., -.5, -.6), (10., 10., .5, .6)]]
    evidence = audit.Evidence()
    audit.audit_normalization_guard(dict(rows=rows, numeric_guard="exact_constant_groups_zero_v1",
                                        normalization_corrections={"episode": [], "local": []}),
                                    "graphgpo", "mean_std_norm", evidence, "guard")
    assert any(c["name"] == "guard_nonconstant_values_unchanged" and c["status"] == "failed" for c in evidence.checks)


@pytest.mark.parametrize("text", [
    'Search: {"tool":"search_document","arguments":{"query":"title"}}',
    'Finish: ```json\n[{"text":"PROMOTION AGREEMENT","confidence":1}]\n```',
])
def test_v3_wrappers_are_parse_only_and_other_protocols_remain_strict(text):
    assert audit.parsed_action(text, "cuad-audited-v3") is not None
    assert audit.parsed_action(text, "cuad-audited-v2") is None
    assert audit.parsed_action(text, "cuad-audited-v1") is None
    task = Task("v3__Document Name", "cuad", "train", "Find the title.", "PROMOTION AGREEMENT.", ["PROMOTION AGREEMENT"])
    reply = ('Finish: [{"text":"PROMOTION AGREEMENT","confidence":1}]'
             if text.startswith("Search") else text)
    calls = iter([text, reply])

    async def policy(messages, **kwargs):
        raw = next(calls)
        return PolicyCompletion(raw, TOKENIZER.apply_chat_template(messages), TOKENIZER.encode(raw))

    result = asyncio.run(EpisodeRunner(max_steps=2, cuad_protocol="cuad-audited-v3").run(task, policy))
    assert result.reward == 1.
    evidence = audit.Evidence()
    rows = audit.trace_rows(result.to_dict(), "cuad-audited-v3", evidence, "v3")
    assert rows[0]["action_text"] == text
    assert result.steps[0].completion_ids == TOKENIZER.encode(text)
    assert rows[0]["token_signature"] == audit.digest([result.steps[0].prompt_ids, result.steps[0].completion_ids, text])
    assert rows[0]["action_aliases"]
    assert all(r["is_action_valid"] for r in rows)


@pytest.mark.parametrize("status, accepted", [
    ({"state": "running", "stage": "audit", "pid": 123}, True),
    ({"state": "running", "stage": "train", "pid": 123}, False),
    ({"state": "running", "stage": "evaluate", "pid": 123}, False),
    ({"state": "failed", "stage": "audit", "returncode": 1}, False),
    ({"state": "finished", "returncode": 1}, False),
])
def test_execution_state_accepts_audit_phase_only_after_artifacts_are_checked(tmp_path, status, accepted):
    write_run(tmp_path)
    (tmp_path / "status.json").write_text(json.dumps(status), encoding="utf-8")
    report = audit.audit_run(tmp_path)
    state_check = next(c for c in report["checks"] if c["name"] == "run_execution_state")
    assert state_check["status"] == ("passed" if accepted else "failed")
    assert report["audit_in_progress"] is accepted
    if not accepted:
        assert report["verdict"] == "failed"


@pytest.mark.parametrize("missing_artifact", ["train_finish", "evaluation"])
def test_audit_phase_does_not_bypass_training_finish_or_evaluation(tmp_path, missing_artifact):
    write_run(tmp_path)
    (tmp_path / "status.json").write_text(json.dumps({"state": "running", "stage": "audit"}), encoding="utf-8")
    if missing_artifact == "train_finish":
        path = tmp_path / "train/train_rank0.jsonl"
        records = [json.loads(line) for line in path.open(encoding="utf-8")]
        path.write_text("\n".join(json.dumps(r) for r in records if r.get("event") != "finished") + "\n", encoding="utf-8")
    else:
        (tmp_path / "evaluation/summary.json").unlink()
    report = audit.audit_run(tmp_path)
    assert next(c for c in report["checks"] if c["name"] == "run_execution_state")["status"] == "passed"
    assert report["verdict"] != "verified_record_consistency"
    assert any(c["required"] and c["status"] != "passed" for c in report["checks"])


def test_grpo_report_names_the_visit_weighted_episode_only_variant(tmp_path):
    write_run(tmp_path)
    _, _, _, record, _ = hand_fixture("grpo")
    record["credit"]["episode_weighting"] = "visits"
    path = tmp_path / "train/train_rank0.jsonl"
    start = dict(event="start", algorithm="LoRA GRPO", adv_estimator="grpo", cuad_protocol="cuad-audited-v2", invalid_action_penalty=.1)
    path.write_text("\n".join(json.dumps(r) for r in [start, record, {"event": "finished", "optimizer_steps": 1}]) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["method"] = "grpo"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config_path = tmp_path / "train/final/training_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["adv_estimator"] = "grpo"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    report = audit.audit_run(tmp_path)
    assert report["method"] == "grpo"
    assert report["method_variant"]["episode_weighting"] == "visits"
    assert report["method_variant"]["is_standard_episode_weighted_grpo"] is False
    assert report["method_variant"]["invalid_action_penalty"] == .1
