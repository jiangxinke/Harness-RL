"""Read-only audit of recorded DCC credit experiments; no model generation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from difflib import SequenceMatcher
import hashlib
import json
import math
from pathlib import Path
import shlex
import struct
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "slime"))
from slime_plugins.dcc.credit_adapter import is_final_control, source_payload
from slime_plugins.dcc.cuad_audit_protocol import parse_action as audited_action, action_aliases
from slime_plugins.dcc.episode import parse_action as legacy_action
from slime_plugins.dcc.cuad_audit_protocol import observed_contract_texts, source_grounded
from slime_plugins.dcc.metrics import cuad_predictions, cuad_task_metrics

FAILURES = {"infrastructure_error", "model_protocol_failure", "context_budget_failure"}
TOL = 5e-5


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(left, right):
    return finite(left) and finite(right) and math.isclose(left, right, abs_tol=TOL, rel_tol=1e-5)


class Evidence:
    def __init__(self):
        self.checks = []
        self.inputs = {}

    def check(self, name, passed, detail=None, *, scope="run", required=True):
        status = "missing" if passed is None else "passed" if passed else "failed"
        item = dict(name=name, status=status, scope=scope, required=required)
        if detail is not None:
            item["detail"] = detail
        self.checks.append(item)
        return passed

    def number(self, name, observed, expected, *, scope):
        return self.check(name, None if observed is None else close(observed, expected),
                          dict(observed=observed, reconstructed=expected), scope=scope)

    def load(self, path, *, lines=False, required=True):
        path = Path(path)
        if not path.is_file():
            self.check("input_exists", None, str(path), scope=str(path), required=required)
            return [] if lines else None
        self.inputs[str(path)] = dict(sha256=sha(path), bytes=path.stat().st_size)
        try:
            with path.open(encoding="utf-8-sig") as handle:
                # splitlines() also splits literal U+2028 inside JSON strings.
                result = [json.loads(line) for line in handle if line.strip()] if lines else json.load(handle)
            return result
        except (ValueError, OSError) as exc:
            self.check("input_parse", False, str(exc), scope=str(path))
            return [] if lines else None


def normalize(values, mode="mean_std_norm", *, ddof=1, singleton_zero=True, epsilon=1e-6):
    if not values:
        return []
    if len(values) == 1:
        return [0.0] if singleton_zero else [values[0] / (1 + epsilon) if mode == "mean_std_norm" else values[0]]
    mean = math.fsum(values) / len(values)
    std = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (len(values) - ddof))
    return [(value - mean) / (std + epsilon) if mode == "mean_std_norm" else value - mean for value in values]


def parsed_action(text, protocol):
    if protocol.startswith("cuad-audited"):
        return audited_action(text, allow_answer_array=protocol in {"cuad-audited-v2", "cuad-audited-v3"},
                              allow_wrappers=protocol == "cuad-audited-v3")
    return legacy_action(text)


def source_texts(payload):
    if not isinstance(payload, dict):
        return []
    result = [payload["text"]] if isinstance(payload.get("text"), str) else []
    result.extend(hit["text"] for hit in payload.get("hits", [])
                  if isinstance(hit, dict) and isinstance(hit.get("text"), str))
    return result


def trace_rows(rollout, protocol, evidence, scope):
    messages, steps = rollout.get("messages", []), rollout.get("steps", [])
    positions = [i for i, message in enumerate(messages) if message.get("role") == "assistant"]
    evidence.check("decision_message_count", len(positions) == len(steps),
                   dict(messages=len(positions), completions=len(steps)), scope=scope)
    rows, prior_returns, prior_arguments = [], {}, {}
    success = rollout.get("status") == "completed" and finite(rollout.get("reward")) and rollout["reward"] >= 1 - 1e-12
    for index, (position, completion) in enumerate(zip(positions, steps)):
        before = messages[:position]
        after = messages[position + 1:positions[index + 1] if index + 1 < len(positions) else len(messages)]
        payloads = [source_payload(message["content"]) for message in before
                    if message.get("role") == "user"]
        payloads = [payload for payload in payloads if payload is not None]
        errors, returned = [], []
        for message in after:
            if message.get("role") != "user":
                continue
            payload = source_payload(message["content"])
            if payload is not None:
                returned.append(payload)
                if isinstance(payload, dict) and payload.get("error"):
                    errors.append(payload["error"])
        phase = "final" if before and is_final_control(before[-1]["content"]) else "act"
        observation = phase + "\n" + json.dumps(payloads[-1], ensure_ascii=False, sort_keys=True) if payloads else None
        text = completion.get("text", "")
        evidence.check("completion_text_matches_message", text == messages[position].get("content"), scope=f"{scope}/step-{index}")
        ids, prompt = completion.get("completion_ids"), completion.get("prompt_ids")
        evidence.check("recorded_token_ids", bool(ids) and bool(prompt), scope=f"{scope}/step-{index}")
        logprobs = completion.get("logprobs")
        if logprobs is not None:
            evidence.check("old_logprob_length_and_finiteness", len(logprobs) == len(ids or []) and all(finite(x) for x in logprobs), scope=f"{scope}/step-{index}")
        action = parsed_action(text, protocol)
        terminal = index == len(steps) - 1
        valid = not errors and (not terminal or rollout.get("status") == "completed")
        token_signature = digest([prompt, ids, text])
        row = dict(step=index, action=action, action_text=text, phase=phase, terminal=terminal,
                   action_aliases=action_aliases(text) if action is not None and protocol == "cuad-audited-v3" else [],
                   success=success, task_reward=10.0 * success, is_action_valid=valid,
                   errors=errors, observation=observation,
                   observation_hash=digest(observation), evidence_hash=digest(payloads),
                   prompt_hash=hashlib.sha256(json.dumps(prompt or []).encode()).hexdigest(),
                   token_signature=token_signature, prompt_tokens=len(prompt or []),
                   response_tokens=len(ids or []), old_logprobs_recorded=logprobs is not None,
                   repeated_return_of_step=None, repeated_action_of_step=None,
                   new_source_texts=sum(text not in {t for p in payloads for t in source_texts(p)}
                                        for payload in returned for text in source_texts(payload)),
                   components={}, phenomena=[])
        if returned:
            return_hash = digest(returned)
            row["returned_payload_hash"] = return_hash
            row["return_kind"] = "error_feedback" if errors else "source_observation"
            if return_hash in prior_returns:
                row["repeated_return_of_step"] = prior_returns[return_hash]
                row["phenomena"].append("错误反馈与之前记录相同" if errors else
                    "工具返回与之前记录相同；不代表该动作的因果价值已确定")
            prior_returns.setdefault(return_hash, index)
        if isinstance(action, dict) and "tool" in action:
            action_hash = digest(action)
            if action_hash in prior_arguments:
                row["repeated_action_of_step"] = prior_arguments[action_hash]
            prior_arguments.setdefault(action_hash, index)
        if not valid:
            row["phenomena"].append("动作协议或执行失败")
        if terminal and rollout.get("status") == "completed" and not rollout.get("answer"):
            row["phenomena"].append("提交空答案；是否正确取决于该题是否确实无匹配条款")
        rows.append(row)
    return rows


def episode_record(rollout, index, phase, protocol, evidence, input_path):
    scope = f"{phase}/rollout-{index}"
    record = dict(index=index, phase=phase, input_file=str(input_path), jsonl_record=index,
                  task_id=rollout.get("task_id"), status=rollout.get("status"), reward=rollout.get("reward"),
                  answer=rollout.get("answer"), metrics=rollout.get("metrics", {}), steps=[])
    try:
        record["steps"] = trace_rows(rollout, protocol, evidence, scope)
    except (ValueError, TypeError, KeyError) as exc:
        evidence.check("trace_reconstruction", False, str(exc), scope=scope)
    metrics = record["metrics"]
    grounding_key = "observed_token_span_fraction" if protocol.startswith("cuad-audited") else "verbatim_span_fraction"
    if record["status"] == "completed":
        if finite(metrics.get("f1")) and finite(metrics.get(grounding_key)):
            evidence.number("terminal_reward_formula", record["reward"], metrics["f1"] * metrics[grounding_key], scope=scope)
        else:
            evidence.check("terminal_reward_formula", None, dict(needed=["f1", grounding_key]), scope=scope)
    return record


def attach_components(row, component, method):
    names = ("episode_advantage", "local_advantage", "graph_advantage", "node_advantage", "edge_advantage",
             "correctness_advantage", "efficiency_advantage", "advantage", "discounted_return",
             "node_value", "next_value", "edge_raw_gain", "graph_return", "distance_target", "target_reachable",
             "correctness_estimated", "efficiency_supported", "step_reward", "novelty", "group", "node")
    row["components"] = {name: component[name] for name in names if name in component}
    local_names = {"grpo": [], "gigpo": ["local_advantage"], "graphgpo": ["graph_advantage"],
                   "g2po": ["node_advantage", "edge_advantage"],
                   "graphpo": ["correctness_advantage", "efficiency_advantage"]}[method]
    for name in local_names:
        value = component.get(name)
        if finite(value) and abs(value) <= TOL:
            row["phenomena"].append(f"{name} 为零")
        if finite(value) and value > TOL and not row["is_action_valid"]:
            row["phenomena"].append(f"执行失败的动作仍有正 {name}；这是局部项现象，不是总优势推断")
        if finite(value) and value > TOL and row.get("terminal") and row.get("success") is False:
            row["phenomena"].append(f"未达到完整终局成功的最后动作有正 {name}；需结合格式、评分与组内比较解读")
        if finite(value) and value > TOL and row.get("repeated_return_of_step") is not None:
            kind = "重复错误反馈" if row.get("return_kind") == "error_feedback" else "重复工具返回"
            row["phenomena"].append(f"{kind}对应正 {name}")


def audit_normalization_guard(credit, method, mode, evidence, scope):
    if not credit.get("numeric_guard"):
        return None
    evidence.check("known_numeric_guard", credit["numeric_guard"] == "exact_constant_groups_zero_v1", scope=scope)
    rows = credit.get("rows", [])
    ep_field = {"graphgpo": "adjusted_episode_score", "g2po": "episode_normalization_score"}.get(method, "episode_score")
    plans = [("episode", ep_field, "episode_advantage", None, "float32")]
    if method == "gigpo":
        plans.append(("local", "local_score", "local_advantage", "group", "float32"))
    elif method == "graphgpo":
        plans.append(("local", "adjusted_graph_return", "local_advantage", "group", "float32"))
    elif method == "g2po":
        plans += [("node", "node_raw_score", "node_advantage", "node", "float64"),
                  ("edge", "edge_normalization_score", "edge_advantage", None, "float32")]
    corrections = credit.get("normalization_corrections")
    if not isinstance(corrections, dict):
        evidence.check("normalization_corrections_recorded", None, scope=scope)
        return None
    evidence.check("normalization_correction_components", set(corrections) == {p[0] for p in plans}, scope=scope)
    try:
        import torch
    except ImportError:
        torch = None
    reports = []
    for name, raw_field, advantage_field, grouping, dtype in plans:
        groups = defaultdict(list)
        for i, row in enumerate(rows):
            groups[(row.get("task_id", "same_task"), row.get(grouping) if grouping else None)].append(i)
        raw = [row.get(raw_field) for row in rows]
        if any(not finite(value) for value in raw):
            evidence.check("guard_raw_inputs", None, dict(component=name, needed=raw_field), scope=scope)
            continue
        changed = set()
        component_report = dict(component=name, raw_field=raw_field, dtype=dtype, corrections=[],
                                nonconstant_groups=0, nonconstant_replay="available" if torch is not None else "missing_torch")
        for fix in corrections.get(name, []):
            indices = fix.get("indices", [])
            valid_indices = (isinstance(indices, list) and len(indices) > 1 and len(set(indices)) == len(indices)
                             and all(type(i) is int and 0 <= i < len(rows) for i in indices))
            evidence.check("guard_correction_indices", valid_indices, dict(component=name, fix=fix), scope=scope)
            if not valid_indices:
                continue
            values = [raw[i] for i in indices]
            evidence.check("guard_only_exact_constant_groups", min(values) == max(values) == fix.get("constant_score"),
                           dict(component=name, inputs=values, fix=fix), scope=scope)
            evidence.check("guard_corrects_complete_group", set(indices) in [set(v) for v in groups.values()],
                           dict(component=name, indices=indices), scope=scope)
            evidence.check("guard_sets_exact_zero", all(rows[i].get(advantage_field) == 0.0 for i in indices),
                           dict(component=name, indices=indices), scope=scope)
            before = fix.get("upstream_advantages", [])
            evidence.check("guard_preserves_original_values", len(before) == len(indices) and all(finite(v) for v in before), scope=scope)
            changed.update(indices)
            component_report["corrections"].append(fix)
        for indices in groups.values():
            values = [raw[i] for i in indices]
            constant = len(indices) > 1 and min(values) == max(values)
            if constant:
                evidence.check("guard_constant_group_zero", all(rows[i].get(advantage_field) == 0.0 for i in indices),
                               dict(component=name, indices=indices), scope=scope)
            else:
                component_report["nonconstant_groups"] += 1
                evidence.check("guard_never_labels_nonconstant_group", not changed.intersection(indices), scope=scope)
                upstream_key = "upstream_" + advantage_field
                if all(upstream_key in rows[i] for i in indices):
                    evidence.check("guard_nonconstant_values_unchanged", all(rows[i][upstream_key] == rows[i].get(advantage_field) for i in indices),
                                   dict(component=name, indices=indices), scope=scope)
            if torch is None:
                continue
            tensor = torch.tensor(values, dtype=getattr(torch, dtype), device="cpu")
            if len(indices) == 1:
                before = tensor / (1 + 1e-6) if name == "episode" and mode == "mean_std_norm" else tensor if name == "episode" else torch.zeros_like(tensor)
            else:
                before = tensor - tensor.mean()
                if mode == "mean_std_norm":
                    before = before / (tensor.std() + 1e-6)
            before = before.float().tolist()
            if constant:
                matching_fix = next((f for f in corrections.get(name, []) if set(f.get("indices", [])) == set(indices)), None)
                if matching_fix:
                    recorded = dict(zip(matching_fix["indices"], matching_fix["upstream_advantages"]))
                    evidence.check("guard_upstream_replay", all(close(value, recorded[i]) for i, value in zip(indices, before)),
                                   dict(component=name, indices=indices, replay=before), scope=scope)
            else:
                evidence.check("guard_nonconstant_upstream_replay", all(close(value, rows[i].get(advantage_field)) for i, value in zip(indices, before)),
                               dict(component=name, indices=indices), scope=scope)
        if torch is None:
            evidence.check("guard_full_upstream_cpu_replay", None,
                           dict(component=name, reason="Torch未安装；已验证常数组/索引/独立数学值，未执行作者dtype数值复现。"),
                           scope=scope, required=False)
        reports.append(component_report)
    return reports


def audit_chain(round_record, rollouts, records, method, config, evidence, scope):
    credit = round_record.get("credit")
    if not isinstance(credit, dict):
        evidence.check("credit_diagnostics", None, scope=scope)
        return [], []
    recorded_rows = credit.get("rows", [])
    lookup = {(row.get("episode"), row.get("step")): row for row in recorded_rows}
    evidence.check("unique_credit_rows", len(lookup) == len(recorded_rows), scope=scope)
    visits = []
    for episode, (raw, record) in enumerate(zip(rollouts, records)):
        usable = raw.get("status") not in FAILURES and raw.get("steps") and all(s.get("completion_ids") for s in raw["steps"])
        if usable:
            for row in record["steps"]:
                visits.append(dict(row, episode=episode, task_id=raw["task_id"], source=row))
    if len({row["episode"] for row in visits}) < 2:
        evidence.check("small_group_skipped", not recorded_rows, scope=scope)
        return [], []
    evidence.check("all_replayable_decisions_have_credit", set(lookup) == {(v["episode"], v["step"]) for v in visits}, scope=scope)
    if not visits or any(v["observation"] is None for v in visits):
        evidence.check("public_observation", None, scope=scope)
        return [], []
    mode = credit.get("normalization", credit.get("mode", config.get("credit_norm", "mean_std_norm")))
    penalty = credit.get("invalid_penalty", config.get("invalid_action_penalty", .1))
    weight = credit.get("weight", credit.get("graph_weight", config.get("gigpo_step_weight", 1.0)))
    gamma = credit.get("gamma", config.get("gigpo_gamma", .95))
    task_groups = defaultdict(list)
    for i, v in enumerate(visits):
        task_groups[v["task_id"]].append(i)
    raw_episode = [v["task_reward"] - penalty * (not v["is_action_valid"]) for v in visits]
    episode = [0.] * len(visits)
    for members in task_groups.values():
        for i, value in zip(members, normalize([raw_episode[j] for j in members], mode, singleton_zero=False)):
            episode[i] = value
    local, node, edge = [0.] * len(visits), [0.] * len(visits), [0.] * len(visits)
    groups = []
    extra = [{} for _ in visits]
    if method == "gigpo":
        threshold = credit.get("similarity_threshold", config.get("gigpo_similarity_threshold", .9))
        for i, visit in enumerate(visits):
            group = next((g for g in groups if visits[g[0]]["task_id"] == visit["task_id"]
                          and visits[g[0]]["phase"] == visit["phase"] and
                          (visits[g[0]]["observation"] == visit["observation"] or threshold < 1 and
                           SequenceMatcher(None, visit["observation"], visits[g[0]]["observation"]).ratio() >= threshold)), None)
            if group is None:
                groups.append([i])
            else:
                group.append(i)
        raw = [v["task_reward"] * gamma ** (len(records[v["episode"]]["steps"]) - v["step"] - 1) for v in visits]
        for i, value in enumerate(raw):
            extra[i]["discounted_return"] = value
        for group in groups:
            for i, value in zip(group, normalize([raw[j] - penalty * (not visits[j]["is_action_valid"]) for j in group], mode)):
                local[i] = value
    elif method in {"graphgpo", "g2po"}:
        grouped = defaultdict(list)
        for i, v in enumerate(visits):
            grouped[(v["task_id"], v["observation"])].append(i)
        groups = list(grouped.values())
        if method == "graphgpo":
            omega = credit.get("omega", config.get("graphgpo_omega", .2))
            target = []
            for v in visits:
                if v["terminal"]:
                    target.append("success" if v["success"] else "terminal:failure")
                else:
                    target.append(records[v["episode"]]["steps"][v["step"] + 1]["observation"])
            raw = [0.] * len(visits)
            for members in task_groups.values():
                reverse, nodes = defaultdict(set), {"success"}
                for i in members:
                    reverse[target[i]].add(visits[i]["observation"])
                    nodes.update((target[i], visits[i]["observation"]))
                distance, queue = {"success": 0}, deque(["success"])
                while queue:
                    current = queue.popleft()
                    for parent in reverse[current]:
                        if parent not in distance:
                            distance[parent] = distance[current] + 1
                            queue.append(parent)
                fallback = max(distance.values()) + 1
                for i in members:
                    d = distance.get(target[i], fallback)
                    raw[i] = 10 * omega ** d
                    extra[i].update(graph_return=raw[i], distance_target=d, target_reachable=target[i] in distance)
            for group in groups:
                for i, value in zip(group, normalize([raw[j] - penalty * (not visits[j]["is_action_valid"]) for j in group], mode)):
                    local[i] = value
        else:
            raw = [v["task_reward"] * .95 ** (len(records[v["episode"]]["steps"]) - v["step"]) for v in visits]
            values = {key: math.fsum(raw[i] for i in members) / len(members) for key, members in grouped.items()}
            targets, gains = [], []
            for i, v in enumerate(visits):
                current = values[(v["task_id"], v["observation"])]
                destination = (v["task_reward"] if v["terminal"] else
                    values[(v["task_id"], records[v["episode"]]["steps"][v["step"] + 1]["observation"])])
                targets.append(destination)
                gains.append(destination - current)
                extra[i].update(discounted_return=raw[i], node_value=current, next_value=destination, edge_raw_gain=gains[i])
            for group in groups:
                for i, value in zip(group, normalize([targets[j] - penalty * (not visits[j]["is_action_valid"]) for j in group], mode)):
                    node[i] = value
            for members in task_groups.values():
                for i, value in zip(members, normalize([gains[j] for j in members], mode)):
                    edge[i] = value
            local = [a + b for a, b in zip(node, edge)]
    group_reports = []
    for number, members in enumerate(groups):
        histories = {visits[i]["evidence_hash"] for i in members}
        group_reports.append(dict(group=number, members=[dict(episode=visits[i]["episode"], step=visits[i]["step"]) for i in members],
                                  size=len(members), distinct_episodes=len({visits[i]["episode"] for i in members}),
                                  distinct_evidence_histories=len(histories), mixed_evidence_histories=len(histories) > 1))
        raw_field, advantage_field = {
            "gigpo": ("local_score", "local_advantage"),
            "graphgpo": ("adjusted_graph_return", "graph_advantage"),
            "g2po": ("node_raw_score", "node_advantage"),
        }.get(method, (None, None))
        raw_values = [lookup.get((visits[i]["episode"], visits[i]["step"]), {}).get(raw_field) for i in members]
        local_values = [lookup.get((visits[i]["episode"], visits[i]["step"]), {}).get(advantage_field) for i in members]
        if raw_values and all(finite(v) for v in raw_values):
            constant = min(raw_values) == max(raw_values)
            anomaly = constant and any(finite(v) and abs(v) > TOL for v in local_values)
            group_reports[-1].update(normalization_input_field=raw_field, normalization_inputs=raw_values,
                                     normalization_advantage_field=advantage_field,
                                     constant_normalization_inputs=constant,
                                     nonzero_advantage_on_constant_group=anomaly)
            if anomaly:
                evidence.check("mathematical_constant_group_zero", False,
                               dict(inputs=raw_values, recorded_advantages=local_values),
                               scope=f"{scope}/group-{number}")
                replay = dict(available=False, interpretation="未执行 float32 复现，不能仅凭异常推定具体数值路径。")
                try:
                    import torch
                    tensor = torch.tensor(raw_values, dtype=torch.float32, device="cpu")
                    mean, std = tensor.mean(), tensor.std()
                    reproduced = tensor - mean
                    if mode == "mean_std_norm":
                        reproduced = reproduced / (std + 1e-6)
                    replay = dict(available=True, dtype="float32", device="cpu", mean=float(mean),
                                  sample_std=float(std), advantages=reproduced.tolist(),
                                  matches_recording=all(close(a, b) for a, b in zip(reproduced.tolist(), local_values)),
                                  interpretation="复现上游有限精度计算，不代表它满足常数组应为零的数学性质。")
                    evidence.check("upstream_float32_replay_matches_recording", replay["matches_recording"],
                                   replay, scope=f"{scope}/group-{number}", required=False)
                except ImportError:
                    evidence.check("upstream_float32_replay", None, "Torch未安装；只验证数学常数组性质。",
                                   scope=f"{scope}/group-{number}", required=False)
                group_reports[-1]["upstream_float32_replay"] = replay
                for i in members:
                    visits[i]["source"]["phenomena"].append(
                        "归一化输入完全相同但局部优势非零：数值异常，不作为有效步骤信用解释")
        if method in {"gigpo", "graphgpo"}:
            actual_groups = {lookup.get((visits[i]["episode"], visits[i]["step"]), {}).get("group") for i in members}
            evidence.check("reconstructed_group_membership", len(actual_groups) == 1 and None not in actual_groups, scope=f"{scope}/group-{number}")
    if method in {"gigpo", "graphgpo", "g2po"}:
        field = "node" if method == "g2po" else "group"
        actual_partition = defaultdict(set)
        for v in visits:
            actual_partition[(v["task_id"], lookup.get((v["episode"], v["step"]), {}).get(field))].add((v["episode"], v["step"]))
        expected_partition = {frozenset((visits[i]["episode"], visits[i]["step"]) for i in group) for group in groups}
        evidence.check("complete_group_partition", {frozenset(g) for g in actual_partition.values()} == expected_partition, scope=scope)
    for i, visit in enumerate(visits):
        row_scope = f"{scope}/episode-{visit['episode']}/step-{visit['step']}"
        actual = lookup.get((visit["episode"], visit["step"]), {})
        expected = dict(episode_advantage=episode[i], advantage=episode[i] + weight * local[i], **extra[i])
        if method != "grpo":
            expected["local_advantage"] = local[i]
        if method == "graphgpo":
            expected["graph_advantage"] = local[i]
        if method == "g2po":
            expected.update(node_advantage=node[i], edge_advantage=edge[i])
        for name, value in expected.items():
            if isinstance(value, bool):
                evidence.check(name, actual.get(name) == value, scope=row_scope)
            else:
                evidence.number(name, actual.get(name), value, scope=row_scope)
        evidence.check("credit_action_validity", actual.get("is_action_valid") == visit["is_action_valid"], scope=row_scope)
        evidence.number("credit_task_reward", actual.get("task_reward"), visit["task_reward"], scope=row_scope)
        nested = round_record.get("advantages")
        if isinstance(nested, list) and visit["episode"] < len(nested) and visit["step"] < len(nested[visit["episode"]]):
            evidence.number("logged_training_advantage", nested[visit["episode"]][visit["step"]], expected["advantage"], scope=row_scope)
        else:
            evidence.check("logged_training_advantage", None, scope=row_scope)
        attach_components(visit["source"], actual, method)
    guard = audit_normalization_guard(credit, method, mode, evidence, scope)
    if guard is not None:
        group_reports.append(dict(kind="numeric_guard", components=guard))
    return [v["source"] for v in visits], group_reports


def checkpoint_audit(directory, update_count, evidence):
    final = directory / "train/final"
    config = evidence.load(final / "training_config.json") or {}
    evidence.number("checkpoint_optimizer_count", config.get("optimizer_steps"), update_count, scope="checkpoint")
    adapter_config = evidence.load(final / "adapter_config.json") or {}
    checkpoint = final / "adapter_model.safetensors"
    report = dict(training_config=config, adapter_config=adapter_config, optimizer_steps=update_count,
                  tensors=[], training_state=None)
    if not checkpoint.is_file():
        evidence.check("adapter_checkpoint", None, str(checkpoint), scope="checkpoint")
    else:
        evidence.inputs[str(checkpoint)] = dict(sha256=sha(checkpoint), bytes=checkpoint.stat().st_size)
        try:
            with checkpoint.open("rb") as handle:
                header_size = struct.unpack("<Q", handle.read(8))[0]
                if header_size > min(checkpoint.stat().st_size - 8, 100_000_000):
                    raise ValueError("Invalid safetensors header length")
                header = json.loads(handle.read(header_size))
                data_start = 8 + header_size
                for name, value in header.items():
                    if name == "__metadata__":
                        continue
                    start, end = value["data_offsets"]
                    dtype = value["dtype"]
                    formats = {"F64": ("d", 8), "F32": ("f", 4), "F16": ("e", 2), "BF16": ("H", 2)}
                    item = dict(name=name, shape=value["shape"], dtype=dtype, data_bytes=end - start)
                    evidence.check("adapter_tensor_bounds", 0 <= start <= end <= checkpoint.stat().st_size - data_start,
                                   item, scope=f"checkpoint/{name}")
                    if dtype not in formats:
                        item["numeric_inspection"] = "unsupported_dtype"
                        evidence.check("adapter_tensor_numeric", None, item, scope=f"checkpoint/{name}")
                    else:
                        fmt, width = formats[dtype]
                        expected_bytes = math.prod(value["shape"]) * width
                        evidence.check("adapter_tensor_shape", end - start == expected_bytes, scope=f"checkpoint/{name}")
                        handle.seek(data_start + start)
                        raw = handle.read(end - start)
                        nonzero, invalid = 0, 0
                        for (number,) in struct.iter_unpack("<" + fmt, raw):
                            if dtype == "BF16":
                                number = struct.unpack("<f", struct.pack("<I", number << 16))[0]
                            invalid += not math.isfinite(number)
                            nonzero += number != 0
                        item.update(nonzero_values=nonzero, nonfinite_values=invalid)
                        evidence.check("adapter_tensor_finite", invalid == 0, scope=f"checkpoint/{name}")
                    report["tensors"].append(item)
            b_tensors = [item for item in report["tensors"] if "lora_B" in item["name"]]
            evidence.check("lora_B_tensors_present", bool(b_tensors), scope="checkpoint")
            if update_count:
                evidence.check("adapter_has_nonzero_lora_B", any(item.get("nonzero_values", 0) for item in b_tensors),
                               "非零 LoRA B 与参数更新一致；没有初始检查点时不声称逐参数差分已经验证。", scope="checkpoint")
        except (ValueError, TypeError, KeyError, OSError, struct.error) as exc:
            evidence.check("adapter_checkpoint_parse", False, str(exc), scope="checkpoint")
    state_path = final / "training_state.pt"
    if not state_path.is_file():
        evidence.check("optimizer_state", None, str(state_path), scope="checkpoint")
    else:
        evidence.inputs[str(state_path)] = dict(sha256=sha(state_path), bytes=state_path.stat().st_size)
        try:
            import torch
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            states = state.get("optimizer", {}).get("state", {})
            steps = [float(v["step"]) for v in states.values() if "step" in v]
            report["training_state"] = dict(optimizer_steps=state.get("optimizer_steps"),
                                            parameter_states=len(states), step_min=min(steps) if steps else None,
                                            step_max=max(steps) if steps else None)
            evidence.number("saved_optimizer_update_count", state.get("optimizer_steps"), update_count, scope="checkpoint")
            evidence.check("saved_optimizer_parameter_states", bool(states) if update_count else True, scope="checkpoint")
            if steps:
                evidence.check("saved_optimizer_step_max", max(steps) == update_count, report["training_state"], scope="checkpoint")
        except ImportError:
            evidence.check("saved_optimizer_state_contents", None, "Torch 未安装；未反序列化 optimizer 状态。", scope="checkpoint")
        except Exception as exc:
            evidence.check("saved_optimizer_state_contents", None,
                           f"安全 CPU weights_only 加载失败，未降级为任意 pickle：{type(exc).__name__}: {exc}", scope="checkpoint")
    return report


def statistics(records, decisions):
    def average(values):
        values = [v for v in values if finite(v)]
        return math.fsum(values) / len(values) if values else None
    successes = [r for r in records if r["status"] == "completed" and finite(r["reward"]) and r["reward"] >= 1 - 1e-12]
    local_names = ("local_advantage", "graph_advantage", "node_advantage", "edge_advantage", "correctness_advantage", "efficiency_advantage")
    return dict(
        rollout_count=len(records), distinct_tasks=len({r["task_id"] for r in records}),
        completed_count=sum(r["status"] == "completed" for r in records), success_count=len(successes),
        success_rate=len(successes) / len(records) if records else None,
        mean_reward=average([r["reward"] for r in records]),
        mean_f1=average([r["metrics"].get("f1") for r in records]),
        mean_observed_grounding=average([r["metrics"].get("observed_token_span_fraction") for r in records]),
        statuses=dict(Counter(r["status"] for r in records)), unique_decisions=len(decisions),
        invalid_decisions=sum(not d["is_action_valid"] for d in decisions),
        repeated_return_decisions=sum(d.get("repeated_return_of_step") is not None for d in decisions),
        repeated_error_feedback_decisions=sum(d.get("repeated_return_of_step") is not None and d.get("return_kind") == "error_feedback" for d in decisions),
        repeated_source_return_decisions=sum(d.get("repeated_return_of_step") is not None and d.get("return_kind") == "source_observation" for d in decisions),
        repeated_action_decisions=sum(d.get("repeated_action_of_step") is not None for d in decisions),
        generated_response_tokens=sum(d.get("response_tokens", 0) for d in decisions),
        generated_prompt_tokens=sum(d.get("prompt_tokens", 0) for d in decisions),
        credit_components={name: dict(
            recorded=sum(finite(d.get("components", {}).get(name)) for d in decisions),
            zero=sum(finite(d.get("components", {}).get(name)) and abs(d["components"][name]) <= TOL for d in decisions),
            invalid_action_positive=sum(not d["is_action_valid"] and finite(d.get("components", {}).get(name)) and d["components"][name] > TOL for d in decisions),
            unsuccessful_terminal_positive=sum(d.get("terminal") and d.get("success") is False and finite(d.get("components", {}).get(name)) and d["components"][name] > TOL for d in decisions),
            repeated_return_positive=sum(d.get("repeated_return_of_step") is not None and finite(d.get("components", {}).get(name)) and d["components"][name] > TOL for d in decisions),
            repeated_source_return_positive=sum(d.get("repeated_return_of_step") is not None and d.get("return_kind") == "source_observation" and finite(d.get("components", {}).get(name)) and d["components"][name] > TOL for d in decisions),
        ) for name in local_names},
    )


def training_journal_audit(journal, rounds, evidence, raw_bindings):
    by_iteration = {r.get("iteration"): r for r in rounds}
    for record in journal:
        iteration = record.get("iteration")
        scope = f"training-decisions/iteration-{iteration}"
        items = record.get("items")
        if not isinstance(items, list):
            evidence.check("training_decision_items", None, scope=scope)
            continue
        reference_round = by_iteration.get(iteration, {})
        evidence.number("training_journal_count", len(items), reference_round.get("unique_training_decisions", -1), scope=scope)
        lookup = {(r.get("episode"), r.get("step")): r for r in reference_round.get("credit", {}).get("rows", [])}
        edges = {r["edge"]: r for r in reference_round.get("credit", {}).get("edges", [])}
        for i, item in enumerate(items):
            key = (item.get("episode"), item.get("step"))
            component = edges.get(item.get("edge_id"), {}) if "edge_id" in item else lookup.get(key, {})
            if component:
                evidence.number("journal_applied_advantage", item.get("advantage"), component.get("advantage"), scope=f"{scope}/{i}")
            else:
                evidence.check("journal_advantage_mapping", None, dict(index=i, episode_step=key, edge=item.get("edge_id")), scope=scope)
            ids = item.get("completion_ids", [])
            if "edge_id" in item:
                sources = {d["edge"]: d["completion"] for d in reference_round.get("credit", {}).get("decisions", [])}
                original = sources.get(item["edge_id"])
            else:
                original = raw_bindings.get((iteration, item.get("episode"), item.get("step")))
            if original is None:
                evidence.check("journal_original_token_mapping", None, dict(index=i, edge=item.get("edge_id"), episode_step=key), scope=scope)
            else:
                evidence.check("journal_original_token_mapping", ids == original.get("completion_ids") and item.get("prompt_ids") == original.get("prompt_ids"), scope=f"{scope}/{i}")
            logprobs = item.get("logprobs")
            evidence.check("journal_old_logprob_alignment", logprobs is not None and len(logprobs) == len(ids) and all(finite(v) for v in logprobs), scope=f"{scope}/{i}")
            reference = item.get("reference_logprobs")
            if reference is not None:
                evidence.check("journal_reference_logprob_alignment", len(reference) == len(ids) and all(finite(v) for v in reference), scope=f"{scope}/{i}")


def dataset_audit(run_dir, manifest, raw_train, raw_eval, protocol, evidence):
    targets = {}
    commands = manifest.get("commands", {})
    for name, expected_hash in manifest.get("dataset_sha256", {}).items():
        candidates = []
        for command in commands.values():
            words = command if isinstance(command, list) else shlex.split(command) if isinstance(command, str) else []
            candidates.extend(Path(word) for word in words if isinstance(word, str) and Path(word).name == name)
        candidates.extend(parent / "data" / name for parent in list(run_dir.parents)[:3])
        source = next((path for path in candidates if path.is_file()), None)
        if source is None:
            evidence.check("dataset_hash", None, dict(dataset=name, expected=expected_hash), required=False)
            continue
        actual_hash = sha(source)
        evidence.check("dataset_hash", actual_hash == expected_hash,
                       dict(path=str(source), expected=expected_hash, actual=actual_hash))
        if actual_hash != expected_hash:
            continue
        for task in evidence.load(source, lines=True):
            if isinstance(task, dict) and task.get("id"):
                targets[task["id"]] = task
    rescored, expected_count = 0, len(raw_train) + len(raw_eval)
    for phase, records in (("training", raw_train), ("evaluation", raw_eval)):
        for i, rollout in enumerate(records, 1):
            scope = f"{phase}/rollout-{i}/source-check"
            task = targets.get(rollout.get("task_id"))
            if not task or not isinstance(task.get("document"), str) or not isinstance(task.get("gold"), list):
                evidence.check("source_rescoring_inputs", None, rollout.get("task_id"), scope=scope, required=False)
                continue
            rescored += 1
            if rollout.get("status") != "completed":
                evidence.number("unfinished_reward_is_zero", rollout.get("reward"), 0., scope=scope)
                continue
            metrics = cuad_task_metrics(task["id"], task["gold"], rollout.get("answer"))
            for field in ("f1", "precision", "recall", "tp", "fp", "fn"):
                evidence.number("native_metric_recomputed_" + field, rollout.get("metrics", {}).get(field), metrics[field], scope=scope)
            spans = cuad_predictions(rollout.get("answer"))
            if protocol.startswith("cuad-audited"):
                exposed = observed_contract_texts(rollout.get("messages", []), task["document"])
                grounded = [any(source_grounded(span["text"], text) for text in exposed) for span in spans]
                fraction = sum(grounded) / len(grounded) if grounded else 1.
                evidence.number("observed_grounding_recomputed", rollout.get("metrics", {}).get("observed_token_span_fraction"), fraction, scope=scope)
            else:
                document = " ".join(task["document"].split())
                grounded = [" ".join(span["text"].split()) in document for span in spans]
                fraction = sum(grounded) / len(grounded) if grounded else 1.
            evidence.number("reward_recomputed_from_source", rollout.get("reward"), metrics["f1"] * fraction, scope=scope)
    evidence.check("gold_answer_rescoring_coverage", True if rescored == expected_count else None,
                   dict(rescored=rescored, expected=expected_count,
                        limitation="使用原生CUAD匹配规则重算，不等于对法律语义进行了人工判定。"), required=False)
    return dict(tasks_with_verified_source=len(targets), rescored_rollouts=rescored, expected_rollouts=expected_count)


def audit_run(run_dir):
    evidence = Evidence()
    manifest = evidence.load(run_dir / "manifest.json") or {}
    status = evidence.load(run_dir / "status.json") or {}
    evidence.check("run_terminal_state", None if not status else status.get("state") == "finished" and status.get("returncode") == 0,
                   status or "status.json 缺失", required=True)
    train_log = evidence.load(run_dir / "train/train_rank0.jsonl", lines=True)
    raw_train = evidence.load(run_dir / "train/rollouts_rank0.jsonl", lines=True)
    starts = [r for r in train_log if r.get("event") == "start"]
    ends = [r for r in train_log if r.get("event") == "finished"]
    rounds = [r for r in train_log if "iteration" in r]
    evidence.check("exactly_one_training_start", len(starts) == 1, dict(count=len(starts)))
    evidence.check("exactly_one_training_finish", len(ends) == 1 and bool(train_log) and train_log[-1].get("event") == "finished", dict(count=len(ends)))
    config = starts[0] if starts else {}
    method = str(manifest.get("method") or config.get("advantage_estimator") or (rounds[0].get("stage") if rounds else "unknown")).lower()
    protocol = config.get("cuad_protocol", "legacy")
    evidence.check("supported_method", method in {"grpo", "gigpo", "graphgpo", "g2po", "graphpo"}, method)
    training = [episode_record(r, i, "training", protocol, evidence, run_dir / "train/rollouts_rank0.jsonl") for i, r in enumerate(raw_train, 1)]
    update_count, offset, unique_decisions, round_reports = 0, 0, [], []
    raw_bindings = {}
    for position, round_record in enumerate(rounds, 1):
        scope = f"iteration-{round_record.get('iteration')}"
        evidence.number("iteration_sequence", round_record.get("iteration"), position, scope=scope)
        applied = round_record.get("optimizer_step_applied")
        evidence.check("explicit_optimizer_update_flag", isinstance(applied, bool), scope=scope)
        update_count += applied is True
        evidence.number("cumulative_optimizer_step", round_record.get("step"), update_count, scope=scope)
        grad = round_record.get("gradient_norm")
        evidence.check("optimizer_gradient_consistency", finite(grad) and (grad > 0) == bool(applied), dict(gradient_norm=grad, applied=applied), scope=scope)
        if round_record.get("stage") == "sft":
            continue
        rewards = round_record.get("rewards")
        if not isinstance(rewards, list):
            evidence.check("round_rewards", None, scope=scope)
            continue
        batch = raw_train[offset:offset + len(rewards)]
        batch_records = training[offset:offset + len(rewards)]
        evidence.check("round_rollout_count", len(batch) == len(rewards), scope=scope)
        for i, (reward, raw, record) in enumerate(zip(rewards, batch, batch_records)):
            evidence.number("reward_matches_rollout", reward, raw.get("reward"), scope=f"{scope}/episode-{i}")
            evidence.check("task_matches_rollout", raw.get("task_id") == round_record.get("task"), scope=f"{scope}/episode-{i}")
            if i < len(round_record.get("statuses", [])):
                evidence.check("status_matches_rollout", raw.get("status") == round_record["statuses"][i], scope=f"{scope}/episode-{i}")
            record.update(iteration=round_record.get("iteration"), episode_in_iteration=i)
            for j, completion in enumerate(raw.get("steps", [])):
                raw_bindings[(round_record.get("iteration"), i, j)] = completion
        try:
            if method == "graphpo":
                decisions, groups = audit_graphpo(round_record, batch, batch_records, evidence, scope, protocol=protocol)
            elif method in {"grpo", "gigpo", "graphgpo", "g2po"}:
                decisions, groups = audit_chain(round_record, batch, batch_records, method, config, evidence, scope)
            else:
                decisions, groups = [], []
            unique_decisions.extend(decisions)
            evidence.number("unique_training_decision_count", round_record.get("unique_training_decisions"), len(decisions), scope=scope)
        except Exception as exc:
            evidence.check("credit_reconstruction", False, f"{type(exc).__name__}: {exc}", scope=scope)
            groups = []
        round_reports.append(dict(iteration=round_record.get("iteration"), task=round_record.get("task"),
                                  rewards=rewards, optimizer_step_applied=applied, gradient_norm=grad,
                                  seconds=round_record.get("seconds"), groups=groups,
                                  costs=round_record.get("credit", {}).get("costs")))
        offset += len(rewards)
        if "rollout_count" in round_record:
            evidence.number("cumulative_rollout_count", round_record["rollout_count"], offset, scope=scope)
    evidence.check("all_training_rollouts_assigned", offset == len(raw_train), dict(assigned=offset, total=len(raw_train)))
    if ends:
        evidence.number("finished_optimizer_count", ends[0].get("optimizer_steps"), update_count, scope="training")
    checkpoint = checkpoint_audit(run_dir, update_count, evidence)
    for key, value in checkpoint["training_config"].items():
        if key != "optimizer_steps" and key in config:
            evidence.check("saved_configuration_matches_start", value == config[key],
                           dict(field=key, start=config[key], saved=value), scope="checkpoint")
    journal_path = run_dir / "train/training_decisions_rank0.jsonl"
    journal = evidence.load(journal_path, lines=True, required=method == "graphpo")
    if journal:
        evidence.check("training_journal_iteration_coverage", {r.get("iteration") for r in journal} ==
                       {r.get("iteration") for r in rounds if r.get("stage") != "sft"}, scope="training")
        training_journal_audit(journal, rounds, evidence, raw_bindings)
    else:
        evidence.check("training_logprobs_from_rollouts", all(d.get("old_logprobs_recorded") for d in unique_decisions),
                       "没有独立training journal；仅能核对原rollout中保存的old logprobs。", scope="training")
    if method == "graphpo":
        auxiliary = evidence.load(run_dir / "train/graphpo_auxiliary_rank0.jsonl", lines=True)
        by_iteration = {r.get("iteration"): r for r in rounds}
        evidence.check("graph_auxiliary_iteration_coverage", {r.get("iteration") for r in auxiliary} ==
                       {r.get("iteration") for r in rounds if r.get("stage") != "sft"}, scope="training")
        for entry in auxiliary:
            scope = f"iteration-{entry.get('iteration')}/auxiliary"
            costs = by_iteration.get(entry.get("iteration"), {}).get("credit", {}).get("costs", {})
            summaries, embeddings = entry.get("summaries", []), entry.get("embeddings", [])
            for key, value in (
                ("calls", len(summaries)),
                ("input_tokens", sum(len(s.get("prompt_ids", [])) for s in summaries)),
                ("output_tokens", sum(len(s.get("completion_ids", [])) for s in summaries)),
            ):
                evidence.number("summary_usage_" + key, costs.get("summarizer_usage", {}).get(key), value, scope=scope)
            for key, value in (
                ("calls", len(embeddings)),
                ("input_tokens", sum(sum(e.get("token_lengths", [])) for e in embeddings if not e.get("error"))),
                ("texts", sum(len(e.get("texts", [])) for e in embeddings if not e.get("error"))),
            ):
                evidence.number("embedding_usage_" + key, costs.get("embedder_usage", {}).get(key), value, scope=scope)
            evidence.check("embedding_provenance_recorded", bool(entry.get("embedding_provenance")), scope=scope)
    raw_eval = evidence.load(run_dir / "evaluation/rollouts.jsonl", lines=True)
    evaluation_summary = evidence.load(run_dir / "evaluation/summary.json") or {}
    evaluation = [episode_record(r, i, "evaluation", protocol, evidence, run_dir / "evaluation/rollouts.jsonl") for i, r in enumerate(raw_eval, 1)]
    evidence.number("evaluation_rollout_count", evaluation_summary.get("rollouts"), len(evaluation), scope="evaluation")
    per_task = defaultdict(list)
    for r in raw_eval:
        per_task[r.get("task_id")].append(r)
    for item in evaluation_summary.get("tasks", []):
        group = per_task.get(item.get("task"), [])
        scope = f"evaluation/{item.get('task')}"
        for field, values in (("rewards", [r.get("reward") for r in group]),
                              ("f1", [r.get("metrics", {}).get("f1") for r in group]),
                              ("statuses", [r.get("status") for r in group]),
                              ("decisions", [len(r.get("steps", [])) for r in group])):
            evidence.check("evaluation_summary_" + field, item.get(field) == values, dict(summary=item.get(field), actual=values), scope=scope)
    evidence.check("evaluation_task_coverage", {item.get("task") for item in evaluation_summary.get("tasks", [])} == set(per_task), scope="evaluation")
    evidence.check("evaluation_not_training", evaluation_summary.get("training") is False, scope="evaluation")
    for key in ("cuad_protocol", "max_steps", "max_new_tokens", "max_prompt_tokens", "initial_chunks"):
        if key in config:
            evidence.check("evaluation_matches_task_protocol", evaluation_summary.get("settings", {}).get(key) == config[key],
                           dict(field=key, training=config[key], evaluation=evaluation_summary.get("settings", {}).get(key)), scope="evaluation")
    evidence.check("train_eval_task_ids_disjoint", not ({r.get("task_id") for r in raw_train} & set(per_task)), scope="evaluation")
    evidence.check("tokenizer_roundtrip", None, "脚本不加载模型/tokenizer；验证记录的token形状与日志一致性，未重新解码token。", required=False)
    source_rescoring = dataset_audit(run_dir, manifest, raw_train, raw_eval, protocol, evidence)
    required = [c for c in evidence.checks if c["required"]]
    verdict = "failed" if any(c["status"] == "failed" for c in required) else "incomplete" if any(c["status"] == "missing" for c in required) else "verified_record_consistency"
    group_items = [g for r in round_reports for g in r["groups"] if "size" in g]
    group_statistics = dict(groups=len(group_items), singleton_groups=sum(g["size"] == 1 for g in group_items),
                            mixed_evidence_history_groups=sum(g.get("mixed_evidence_histories", False) for g in group_items),
                            constant_input_nonzero_local_groups=sum(g.get("nonzero_advantage_on_constant_group", False) for g in group_items))
    generation_rows = [s for r in training for s in r["steps"]] if method != "graphpo" else []
    costs = dict(train_elapsed_seconds=ends[0].get("elapsed") if ends else None,
                 train_peak_cuda_allocated_bytes=ends[0].get("peak_cuda_allocated_bytes") if ends else None,
                 evaluation_seconds=evaluation_summary.get("seconds"),
                 evaluation_peak_cuda_allocated_bytes=evaluation_summary.get("peak_cuda_allocated_bytes"),
                 generated_response_tokens=sum(s["response_tokens"] for s in generation_rows) if method != "graphpo" else
                    sum((r.get("costs") or {}).get("generated_response_tokens", 0) for r in round_reports),
                 generated_prompt_tokens=sum(s["prompt_tokens"] for s in generation_rows) if method != "graphpo" else
                    sum((r.get("costs") or {}).get("generated_prompt_tokens", 0) for r in round_reports),
                 graph_auxiliary_costs=[dict(iteration=r["iteration"], costs=r["costs"]) for r in round_reports if r["costs"]])
    return dict(version="dcc-credit-audit-v1", verdict=verdict, method=method, run_dir=str(run_dir),
                interpretation="所有异常描述均为本次记录中的现象，不是动作因果价值、训练收益或算法普遍缺陷的证明。",
                numeric_tolerance=dict(absolute=TOL, relative=1e-5),
                manifest=manifest, status=status, checkpoint=checkpoint,
                training_statistics=statistics(training, unique_decisions),
                evaluation_statistics=statistics(evaluation, [s for r in evaluation for s in r["steps"]]),
                group_statistics=group_statistics, costs=costs,
                source_rescoring=source_rescoring,
                rounds=round_reports, training_rollouts=training, evaluation_rollouts=evaluation,
                unique_training_decisions=unique_decisions, evaluation_summary=evaluation_summary,
                checks=evidence.checks, inputs=evidence.inputs,
                check_counts=dict(Counter(c["status"] for c in evidence.checks)))


def audit_graphpo(round_record, rollouts, records, evidence, scope, *, protocol="cuad-audited-v2"):
    credit = round_record.get("credit", {})
    nodes = {n["id"]: n for n in credit.get("nodes", [])}
    decisions = {d["edge"]: d for d in credit.get("decisions", [])}
    edges = {e["edge"]: e for e in credit.get("edges", [])}
    evidence.check("graph_nodes_present", bool(nodes), scope=scope)
    if not nodes:
        return [], []
    evidence.check("unique_generated_edges", len(decisions) == len(credit.get("decisions", [])) and set(decisions) == set(nodes) - {0}, scope=scope)
    evidence.check("generated_edge_diagnostics", set(edges) == set(nodes) - {0}, scope=scope)
    children, uf = defaultdict(list), {n: n for n in nodes}

    def find(n):
        while uf[n] != n:
            n = uf[n]
        return n

    def path(n):
        result = []
        while n is not None:
            if n in result or n not in nodes:
                raise ValueError("Generated parent links contain a cycle or missing node")
            result.append(n)
            n = nodes[n]["parent"]
        return list(reversed(result))

    def cosine(a, b):
        left, right = nodes[a]["embedding"], nodes[b]["embedding"]
        if not left or len(left) != len(right) or not all(finite(v) for v in left + right):
            raise ValueError("Invalid saved graph embedding")
        divisor = math.sqrt(math.fsum(v * v for v in left) * math.fsum(v * v for v in right))
        if not divisor:
            raise ValueError("Zero-norm saved graph embedding")
        return max(-1., min(1., math.fsum(x * y for x, y in zip(left, right)) / divisor))

    def acyclic():
        successors, degree = defaultdict(set), dict.fromkeys({find(n) for n in nodes}, 0)
        for n, item in nodes.items():
            if item["parent"] is None:
                continue
            a, b = find(item["parent"]), find(n)
            if a == b:
                return False
            if b not in successors[a]:
                successors[a].add(b)
                degree[b] += 1
        queue = deque(n for n, value in degree.items() if value == 0)
        count = 0
        while queue:
            count += 1
            for child in successors[queue.popleft()]:
                degree[child] -= 1
                if degree[child] == 0:
                    queue.append(child)
        return count == len(degree)

    for n, item in nodes.items():
        if n:
            children[item["parent"]].append(n)
            evidence.check("generation_parent_precedes_child", item["parent"] in nodes and item["parent"] < n, scope=f"{scope}/node-{n}")
            path(n)
    accepted = [m for m in credit.get("merges", []) if m.get("accepted")]
    for merge in accepted:
        a, b = merge["left"], merge["right"]
        evidence.check("merge_is_noncausal", a not in path(b) and b not in path(a), merge, scope=scope)
        evidence.check("merge_excludes_infrastructure_failure", nodes[a].get("status") not in FAILURES and nodes[b].get("status") not in FAILURES, merge, scope=scope)
        evidence.check("merge_threshold", cosine(a, b) + 1e-10 >= credit.get("threshold", .92), merge, scope=scope)
        uf[find(b)] = find(a)
        evidence.check("merged_quotient_is_DAG", acyclic(), merge, scope=scope)
    classes = defaultdict(list)
    for n in nodes:
        classes[find(n)].append(n)
    actual_partition = {frozenset(n for n in nodes if nodes[n].get("class") == item.get("class")) for item in nodes.values()}
    evidence.check("recorded_equivalence_classes", actual_partition == {frozenset(v) for v in classes.values()}, scope=scope)
    own = {}
    for n in sorted(nodes, reverse=True):
        own[n] = {n} if nodes[n].get("terminal_success") is not None else set()
        for child in children[n]:
            own[n].update(own[child])
        evidence.number("graph_own_terminal_count", nodes[n].get("own_count"), len(own[n]), scope=f"{scope}/node-{n}")
    weight = credit.get("pooling_weight", .7)
    scores = {}
    for members in classes.values():
        for n in members:
            denominator = len(own[n]) + weight * sum(len(own[q]) for q in members if q != n)
            numerator = sum(nodes[z]["terminal_success"] for z in own[n]) + weight * sum(
                nodes[z]["terminal_success"] for q in members if q != n for z in own[q])
            scores[n] = numerator / denominator if denominator else None
            observed = nodes[n].get("score")
            if scores[n] is None:
                evidence.check("unestimated_node_remains_unknown", observed is None, scope=f"{scope}/node-{n}")
            else:
                evidence.number("pooled_node_value", observed, scores[n], scope=f"{scope}/node-{n}")
    rewards, cor, eff, supported = {}, {}, defaultdict(float), set()
    epsilon = credit.get("epsilon", 1e-6)
    for n, item in nodes.items():
        if not n:
            continue
        parent = item["parent"]
        if scores[n] is not None and scores[parent] is not None:
            rewards[n] = (scores[n] - scores[parent]) * (1 - max(0., cosine(parent, n)))
            evidence.number("graph_edge_reward", edges.get(n, {}).get("step_reward"), rewards[n], scope=f"{scope}/edge-{n}")
        else:
            evidence.check("unknown_correctness_not_fabricated", edges.get(n, {}).get("step_reward") is None, scope=f"{scope}/edge-{n}")
    correctness_groups = []
    for members in classes.values():
        members = sorted(members)
        outgoing = [child for n in members for child in children[n] if child in rewards]
        values = normalize([rewards[e] for e in outgoing], ddof=0, epsilon=epsilon)
        cor.update(zip(outgoing, values))
        correctness_groups.append(dict(source_class=members, edges=outgoing, size=len(outgoing)))
        if len(members) > 1:
            history_hashes = {digest(credit.get("node_messages", {}).get(str(n), [])) for n in members}
            correctness_groups[-1]["distinct_path_histories"] = len(history_hashes)
        if len(members) < 2 or any(scores[n] is None for n in members):
            continue
        paths = [path(n) for n in members]
        length = 0
        for entries in zip(*paths):
            if len(set(entries)) != 1:
                break
            length += 1
        routes = [p[length:] for p in paths]
        gate = math.fsum(scores[n] for n in members) / len(members)
        for route, value in zip(routes, normalize([-len(p) for p in routes], ddof=0, epsilon=epsilon)):
            for e in route:
                eff[e] += gate * value
                supported.add(e)
    expected = {e: cor.get(e, 0) + credit.get("efficiency_weight", 1.) * eff.get(e, 0)
                for e in set(cor) | supported}
    for e, item in edges.items():
        escope = f"{scope}/edge-{e}"
        if e in cor:
            evidence.number("graph_correctness_advantage", item.get("correctness_advantage"), cor[e], scope=escope)
        else:
            evidence.check("missing_correctness_is_explicit", item.get("correctness_advantage") is None, scope=escope)
        evidence.number("graph_efficiency_advantage", item.get("efficiency_advantage"), eff.get(e, 0), scope=escope)
        if e in expected:
            evidence.number("graph_dual_advantage", item.get("advantage"), expected[e], scope=escope)
    excluded = set(credit.get("excluded_terminal_edges", []))
    training_ids = credit.get("training_edge_ids", [])
    evidence.check("each_generated_edge_trained_once", len(training_ids) == len(set(training_ids)) and set(training_ids) == set(expected) - excluded, scope=scope)
    by_edge, mapped_rollouts = {}, set()
    for terminal in credit.get("terminal_paths", []):
        index = terminal["result_index"]
        if not 0 <= index < len(records):
            evidence.check("terminal_rollout_reference", False, terminal, scope=scope)
            continue
        mapped_rollouts.add(index)
        raw, record = rollouts[index], records[index]
        success = None if raw.get("status") in FAILURES else float(raw.get("status") == "completed" and raw.get("reward", 0) >= 1 - 1e-12)
        evidence.check("graph_terminal_matches_real_rollout", terminal.get("binary_success") == success and nodes[terminal["node"]].get("terminal_success") == success, terminal, scope=scope)
        evidence.check("terminal_path_is_real_generation_path", terminal["edges"] == path(terminal["node"])[1:], scope=scope)
        evidence.check("terminal_path_step_count", len(terminal["edges"]) == len(record["steps"]), scope=scope)
        for e, step in zip(terminal["edges"], record["steps"]):
            decision = decisions.get(e, {})
            completion = decision.get("completion", {})
            evidence.check("shared_prefix_keeps_original_tokens", step["token_signature"] == digest([completion.get("prompt_ids"), completion.get("completion_ids"), completion.get("text")]), scope=f"{scope}/edge-{e}")
            step["edge_id"] = e
            attach_components(step, edges.get(e, {}), "graphpo")
            if e in by_edge:
                evidence.check("shared_edge_trace_identity", by_edge[e]["token_signature"] == step["token_signature"], scope=f"{scope}/edge-{e}")
            by_edge.setdefault(e, step)
    for e, decision in decisions.items():
        prefix = path(e)[1:-1]
        evidence.check("decision_prefix_edges", decision.get("prefix_edges") == prefix, scope=f"{scope}/edge-{e}")
        if e not in by_edge:
            completion = decision["completion"]
            history = credit.get("node_messages", {}).get(str(e), credit.get("node_messages", {}).get(e))
            if not isinstance(history, list):
                evidence.check("unfinished_edge_history", None, dict(edge=e), scope=scope)
                continue
            synthetic_view = dict(messages=history, steps=[decisions[i]["completion"] for i in prefix + [e]],
                                  status=nodes[e].get("status", "active"), reward=0.0)
            rows = trace_rows(synthetic_view, protocol, evidence, f"{scope}/original-edge-{e}")
            if not rows:
                continue
            step = rows[-1]
            step.update(edge_id=e, terminal=nodes[e].get("closed", False),
                        success=nodes[e].get("terminal_success"), task_reward=None,
                        is_action_valid=step["action"] is not None and not step["errors"])
            attach_components(step, edges[e], "graphpo")
            by_edge[e] = step
    evidence.check("all_generated_decisions_inspected", set(by_edge) == set(decisions), scope=scope)
    costs = credit.get("costs", {})
    for name, expected_value in (
        ("generated_edges", len(decisions)),
        ("generated_response_tokens", sum(len(d["completion"].get("completion_ids", [])) for d in decisions.values())),
        ("generated_prompt_tokens", sum(len(d["completion"].get("prompt_ids", [])) for d in decisions.values())),
    ):
        evidence.number("graph_cost_" + name, costs.get(name), expected_value, scope=scope)
    for callback in ("summarizer_usage", "embedder_usage"):
        evidence.check("graph_auxiliary_actual_usage", bool(costs.get(callback)), callback, scope=scope)
    unmatched = set(range(len(records))) - mapped_rollouts
    evidence.check("unmapped_rollouts_are_explicit_failures", all(rollouts[i].get("status") in FAILURES for i in unmatched), sorted(unmatched), scope=scope)
    return [by_edge[e] for e in training_ids if e in by_edge], [dict(
        kind="graphpo", accepted_merge_pairs=accepted, correctness_groups=correctness_groups,
        efficiency_groups=credit.get("efficiency_groups", []),
        untrained_decisions=[by_edge[e] for e in sorted(set(by_edge) - set(training_ids))],
        actual_generated_decisions=len(decisions), shared_prefix_occurrences=sum(len(r["steps"]) for r in records),
    )]


def main():
    parser = argparse.ArgumentParser(description="只读核对 DCC 训练、轨迹、逐步优势与检查点，输出 JSON。")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run_dir, output = args.run_dir.resolve(), args.output.resolve()
    if not run_dir.is_dir():
        parser.error("--run-dir 必须是已有方法结果目录")
    if output == run_dir or output.is_relative_to(run_dir / "train") or output.is_relative_to(run_dir / "evaluation") or output in {run_dir / "manifest.json", run_dir / "status.json"}:
        parser.error("输出不得覆盖训练日志、评估数据、manifest 或 status")
    report = audit_run(run_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(verdict=report["verdict"], output=str(output), checks=report["check_counts"]), ensure_ascii=False))
    return 1 if report["verdict"] == "failed" else 2 if report["verdict"] == "incomplete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
