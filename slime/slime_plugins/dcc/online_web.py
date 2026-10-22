"""Opt-in public-only online Web groups with durable close-before-label scoring.

TRAIN literal row reward is distinct from benchmark evaluation. The unchanged
public native schemas, citation guard and native SFT text gates remain in use.
External review bindings assert provenance; they are not self-authenticating.
"""
from __future__ import annotations

import asyncio
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random

from .data import Task
from .durable_journal import OrderedDurableJSONLWriter, close_durable_journals
from .episode import _WebEvidence, ModelProtocolFailure, WIDESEARCH_EVIDENCE_GUARD_VERSION
from .native_hf_context import (NativeHFPromptBudgetExceeded, PREFLIGHT_PROTOCOL, context_handling_configuration,
    restore_verified_native_context, unreliable_context_or_environment_group)
from .prompt_history import PromptHistoryConfig, PromptHistoryBudgetExceeded
from .web_train_prompt_history import (BUDGET_FIELDS, FAILURE_FIELD, FIELDS as TRAIN_HISTORY_FIELDS,
    web_train_history_budgets, web_train_history_configuration, web_train_history_failure_fields,
    web_train_history_fields, web_train_history_plan)
from slime_plugins.dcc_train_literal_reward_contract import PROTOCOL, LABEL_PROTOCOL, configuration, _same

WRAPPER = "dcc-hf-online-web-public-closed-literal-v2"
GROUP_CLOSE = "dcc-hf-online-web-group-close-v1"
BINDING = "dcc-hf-online-web-reviewed-binding-v1"
HF_PROTOCOL = "hf-native-two-stage-thought-only-policy-v1"
END_THINK = 151668
GROUNDING = "native-sft-fetched-text-and-actual-final-prompt-v1"
BEHAVIOR_SCORES = "streaming-generation-raw-logits-v1"
TABLE_GROUNDED_REWARD = "table-grounded-exact-v1"
ROW_GROUNDED_REWARD = "per-row-grounded-exact-v1"
ONLINE_REWARD_VARIANTS = (TABLE_GROUNDED_REWARD, ROW_GROUNDED_REWARD)
MIXED_BENCHMARK_SCHEDULES = ("slot-interleaved", "iteration-blocked")
ROW_REWARD_PROTOCOL = "dcc-web-online-train-per-row-grounded-exact-v1"
ROW_REWARD_WRAPPER = "dcc-hf-online-web-public-closed-per-row-grounded-v1"
ROW_REWARD_METRIC = "train_per_row_grounded_exact_f1"


def online_reward_configuration(variant=TABLE_GROUNDED_REWARD):
    """Online-only composition; the original literal label contract is unchanged."""
    if variant not in ONLINE_REWARD_VARIANTS:
        raise ValueError("Unknown online Web TRAIN reward variant")
    if variant == TABLE_GROUNDED_REWARD:
        return dict(variant=variant, metric="train_exact_row_f1",
                    grounding_scope="whole-table", train_reward=configuration())
    return dict(version=ROW_REWARD_PROTOCOL, variant=variant, scope="widesearch-train-only",
        metric=ROW_REWARD_METRIC, raw_literal_train_reward=configuration(),
        grounding_protocol=GROUNDING, grounding_scope="per-predicted-row",
        grounding_limitations="existing substring support across fetched-text pool; not semantic or citation-to-row entailment",
        table_prerequisites="strict-literal-validity-and-unchanged-public-citation-guard",
        true_positives="eligible-full-required-column-tuples; strict-multiset-gold-match",
        denominator="all-predicted-rows-plus-all-gold-rows",
        invalid_prediction="entire-table-zero; never-drop-malformed-rows",
        prediction_duplicates="retain-all-occurrences; excess-is-false-positive",
        empty_prediction="unchanged-native-nonempty-guard",
        failed_outcomes="retained-with-zero-reward", unreliable_group="unchanged-discard",
        label_access="after-own-durable-group-close", benchmark_score=False)


def public_web_task(row):
    """Reject labels/private metadata; do not silently strip a labeled input."""
    allowed = {"source", "id", "benchmark", "split", "prompt", "document", "gold", "metadata"}
    if (type(row) is not dict or set(row) - allowed
            or row.get("source") != "independent_curriculum"
            or row.get("benchmark") != "widesearch" or row.get("split") != "train"
            or row.get("gold") is not None or row.get("document", "") != ""):
        raise ValueError("Exact independent public-only Web TRAIN task required")
    meta = row.get("metadata")
    if (type(meta) is not dict or set(meta) != {"evaluation"}
            or type(meta["evaluation"]) is not dict or set(meta["evaluation"]) != {"required"}):
        raise ValueError("Only public required-column metadata is allowed")
    required = meta["evaluation"]["required"]
    if (type(required) is not list or not 1 <= len(required) <= 32
            or any(type(x) is not str or not x.strip() or x == "citations" for x in required)
            or len(set(required)) != len(required)):
        raise ValueError("Distinct public answer columns required")
    return Task.from_dict(row)


def public_web_deck(task_ids, *, draws, seed):
    """Only public strings enter the deck; no task.gold, source URL or label store."""
    if (type(task_ids) not in (list, tuple) or not task_ids
            or any(type(x) is not str or not x for x in task_ids)
            or len(set(task_ids)) != len(task_ids) or type(draws) is not int or draws < 1):
        raise ValueError("Nonempty unique public IDs and positive draw count required")
    rng, result = random.Random(seed), []
    while len(result) < draws:
        deck = list(task_ids)
        rng.shuffle(deck)
        result.extend(deck)
    return result[:draws]


def unchanged_native_cell_grounding(task, result, decoded_final_prompt):
    """The two existing SFT checks, separate from citation-only public guard.

    These retain native SFT's nonempty-string requirement. Literal reward can
    itself represent null/empty strings; those do not gain positive native
    grounding through this additional gate. Substring support remains a limited deterministic guard,
    not semantic/row-relation entailment.
    """
    guard, fetched = _WebEvidence(), []
    for event in result.tool_events:
        tool, observation = event.get("tool"), event.get("observation")
        guard.observe(tool, observation)
        if tool == "fetch_url" and type(observation) is dict and not observation.get("error"):
            text = observation.get("text")
            if type(text) is str and text.strip():
                fetched.append(text)
            fetched.extend(row["text"] for row in observation.get("row_view", {}).get("records", [])
                           if type(row) is dict and type(row.get("text")) is str)
    reasons = guard.reject_reasons(task, result.answer)
    if reasons:
        return reasons
    fields = [value for row in result.answer for key, value in row.items() if key != "citations"]
    if any(type(value) is not str or not value.strip() for value in fields):
        return ["native_existing_nonempty_string_cell_gate"]
    normalized = [" ".join(text.split()).replace('""', '"') for text in fetched]
    if any(not any(" ".join(value.split()) in text for text in normalized) for value in fields):
        return ["cell_absent_from_actual_fetched_observation"]
    import json
    visible = " ".join(decoded_final_prompt.split()).replace('""', '"')
    if any(" ".join(value.split()) not in visible
           and " ".join(json.dumps(value, ensure_ascii=False)[1:-1].split()) not in visible
           for value in fields):
        return ["cell_absent_from_actual_encoded_final_prompt"]
    return []


def _json_line(row):
    return json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"


def _read_jsonl(path, *, expected_identity=None):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate journal/data JSON key")
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError("Nonfinite journal/data JSON value: " + value)
    rows, digest, size = [], hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        for line in stream:
            digest.update(line)
            size += len(line)
            if line.strip():
                rows.append(json.loads(line.decode("utf-8"), object_pairs_hook=unique,
                                       parse_constant=nonfinite))
    if expected_identity is not None and not _same(
            {"sha256": digest.hexdigest(), "bytes": size}, expected_identity):
        raise ValueError("Private label file changed after reviewed preflight")
    return rows


def file_identity(path):
    """One streaming byte identity; never parse or expose private label values."""
    digest, size = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {"sha256": digest.hexdigest(), "bytes": size}


def binding_protocols():
    return dict(wrapper=WRAPPER, group_close=GROUP_CLOSE, native=HF_PROTOCOL,
                end_think_id=END_THINK, evidence_guard=WIDESEARCH_EVIDENCE_GUARD_VERSION,
                native_cell_grounding=GROUNDING, behavior_scores=BEHAVIOR_SCORES,
                context_failure_handling=context_handling_configuration(),
                reward=configuration())


class WebTrainingInputs:
    """Validate a separately reviewed binding before model work; load labels lazily.

    Preflight hashes the private file as opaque bytes. It does not deserialize
    labels. The first label deserialization occurs only after a genuine group
    durable closure. A new run must supply a binding for its actual source.
    """
    def __init__(self, public_path, labels_path, binding_path, *, source_commit):
        self.labels_path = Path(labels_path)
        self.binding = json.loads(Path(binding_path).read_text(encoding="utf-8"))
        value = self.binding
        if (type(value) is not dict or value.get("version") != BINDING
                or value.get("root_accepted") is not True
                or value.get("training_reward_admission_approved") is not True
                or type(value.get("review_record")) is not str or not value["review_record"].strip()
                or value.get("source_commit") != source_commit
                or not _same(value.get("protocols"), binding_protocols())):
            raise ValueError("Actual reviewed Web source/protocol binding required; pending records are not admission")
        if not _same(value.get("public_train"), file_identity(public_path)):
            raise ValueError("Reviewed public Web TRAIN byte identity differs")
        self.label_identity = file_identity(labels_path)
        if not _same(value.get("private_train_labels"), self.label_identity):
            raise ValueError("Reviewed private Web TRAIN label byte identity differs")
        evidence = value.get("source_evidence")
        if (type(evidence) is not dict or type(evidence.get("review_record")) is not str
                or not evidence["review_record"].strip()
                or evidence.get("reviewed") is not True
                or type(evidence.get("receipt_pairs")) is not int or evidence["receipt_pairs"] < 1
                or type(evidence.get("lineage_cells")) is not int or evidence["lineage_cells"] < 1
                or type(evidence.get("inventory")) is not dict
                or not evidence["inventory"]):
            raise ValueError("Separately reviewed source receipts and cell lineage binding required")
        self.tasks = [public_web_task(row) for row in _read_jsonl(public_path)]
        ids = [task.id for task in self.tasks]
        if not ids or len(set(ids)) != len(ids) or not _same(value.get("public_task_ids"), ids):
            raise ValueError("Complete unique reviewed public Web TRAIN IDs required")
        self.label_reads = 0

    def label_after_close(self, group):
        from dcc_train_web_literal_reward import validate_label
        if not isinstance(group, ClosedPublicGroup) or not group.closure.get("journals_flushed"):
            raise ValueError("Private labels require a genuinely closed public group")
        if group.source_commit != self.binding["source_commit"]:
            raise ValueError("Group/source binding differs")
        # Parse each group's labels only after its own closure; do not keep a
        # deserialized label corpus across the next public rollout. Identity and
        # parsing use the same bytes, avoiding a check/read replacement window.
        rows = _read_jsonl(self.labels_path, expected_identity=self.label_identity)
        self.label_reads += 1
        labels = {row.get("task_id"): row for row in rows}
        if len(labels) != len(rows) or set(labels) != {task.id for task in self.tasks}:
            raise ValueError("Private labels must cover exactly the reviewed public TRAIN IDs")
        for task in self.tasks:
            validate_label(labels[task.id], task.to_dict(), expected_protocol=PROTOCOL)
        return labels[group.task.id]


def validate_mixed_benchmark_schedule(schedule, *, steps, cuad_draws_per_cycle, web_draws_per_cycle):
    """The opt-in schedule must preserve draw counts across complete cycles."""
    if schedule not in MIXED_BENCHMARK_SCHEDULES:
        raise ValueError("Unknown mixed benchmark schedule")
    if schedule == "iteration-blocked":
        if any(type(value) is not int or value < 1 for value in
               (steps, cuad_draws_per_cycle, web_draws_per_cycle)):
            raise ValueError("iteration-blocked requires positive integer steps and per-benchmark draws")
        if steps % (cuad_draws_per_cycle + web_draws_per_cycle):
            raise ValueError("iteration-blocked requires --steps to be a complete benchmark cycle "
                             "(cuad_draws_per_cycle + web_draws_per_cycle)")


def mixed_task_plan(cuad_tasks, web_tasks, *, steps, world_size, seed,
                    cuad_draws_per_cycle=1, web_draws_per_cycle=1, cuad_plan,
                    schedule="slot-interleaved"):
    """Existing CUAD label strata; independent public-ID-only Web deck.

    This complete plan is not label-blind: the authorized CUAD planner retains
    its TRAIN gold positive/negative stratification. Web selection sees IDs only.
    Iteration-blocked scheduling changes optimizer order while retaining each
    benchmark's original draws and independent deck over complete cycles.
    """
    if any(type(v) is not int or v < 1 for v in
           (steps, world_size, cuad_draws_per_cycle, web_draws_per_cycle)):
        raise ValueError("Positive explicit mixture/plan budgets required")
    ids = [task.id for task in cuad_tasks + web_tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("CUAD and Web task identities must be globally unique")
    validate_mixed_benchmark_schedule(schedule, steps=steps,
        cuad_draws_per_cycle=cuad_draws_per_cycle, web_draws_per_cycle=web_draws_per_cycle)
    slots = (["cuad"] * cuad_draws_per_cycle + ["widesearch"] * web_draws_per_cycle)
    if schedule == "iteration-blocked":
        kinds = [slots[iteration % len(slots)] for iteration in range(steps) for _ in range(world_size)]
    else:
        kinds = [slots[index % len(slots)] for index in range(steps * world_size)]
    cuad_count = kinds.count("cuad")
    cuad_indices = iter(index for row in cuad_plan(cuad_tasks,
        steps=max(1, cuad_count), world_size=1, seed=seed) for index in row)
    deck = public_web_deck([task.id for task in web_tasks],
                          draws=max(1, kinds.count("widesearch")), seed=seed + 1)
    web_index = {task.id: len(cuad_tasks) + index for index, task in enumerate(web_tasks)}
    web_indices = iter(web_index[task_id] for task_id in deck)
    flat = [next(cuad_indices) if kind == "cuad" else next(web_indices) for kind in kinds]
    return [flat[index:index + world_size] for index in range(0, len(flat), world_size)]


class OwnedEpisodePolicy:
    """Own every actual native request, including a late timeout completion."""
    def __init__(self, bound_policy, *, task_id, iteration, rank, sample, sink):
        self.bound_policy = bound_policy
        self.identity = dict(task_id=task_id, iteration=iteration, rank=rank, sample=sample)
        self.requests, self.sink = [], sink

    async def __call__(self, *args, **kwargs):
        raise ValueError("The Web wrapper requires original structured native calls")

    async def generate_structured(self, messages, *, json_schema, **kwargs):
        row = {**self.identity, "turn": len(self.requests),
               "messages": deepcopy(messages), "json_schema": deepcopy(json_schema)}
        self.requests.append(row)
        factory = getattr(self.bound_policy, "new_request_control", None)
        control = factory() if callable(factory) else None
        pending = asyncio.create_task(self.bound_policy.generate_structured(
            messages, json_schema=json_schema, **kwargs,
            **({"_request_control": control} if control is not None else {})))
        cancelled = False
        try:
            while True:
                try:
                    completion = await asyncio.shield(pending)
                    break
                except asyncio.CancelledError:
                    # Expire queued work, but keep shielding any submitted
                    # phase through repeated sibling/deadline cancellations.
                    cancelled = True
                    if control is not None:
                        control.expire()
                    if pending.done():
                        completion = pending.result()
                        break
        except ModelProtocolFailure as error:
            if error.trace_id != getattr(error.completion, "trace_id", None):
                raise ValueError("Failed completion/exception trace ownership differs") from error
            self._record(row, error.completion, cancelled=cancelled)
            if cancelled:
                raise asyncio.CancelledError() from error
            raise
        except BaseException as error:
            row.update(request_status="error", delivery="cancelled" if cancelled else "error",
                       error_type=type(error).__name__, error=str(error))
            if isinstance(error, NativeHFPromptBudgetExceeded):
                row["native_hf_preflight"] = dict(error.preflight)
                if hasattr(error, FAILURE_FIELD):
                    row[FAILURE_FIELD] = deepcopy(getattr(error, FAILURE_FIELD))
            self.sink(row)
            if cancelled:
                raise asyncio.CancelledError() from error
            raise
        self._record(row, completion, cancelled=cancelled)
        if cancelled:
            raise asyncio.CancelledError()
        return completion

    def _record(self, row, completion, *, cancelled):
        from .hf_native_policy import CANCELLED_BEFORE_JSON
        if completion.native_status == CANCELLED_BEFORE_JSON and not cancelled:
            raise ValueError("A cancelled-before-JSON prefix requires cancelled delivery")
        trace_id = getattr(completion, "trace_id", None)
        if type(trace_id) is not str or not trace_id:
            raise ValueError("HF completion must carry its exact sink trace_id")
        probabilities = getattr(completion, "thought_logprobs", None)
        probability_source = getattr(completion, "thought_logprobs_source", None)
        if (type(probabilities) is not list or len(probabilities) != completion.thought_token_count
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in probabilities)
                or probability_source != BEHAVIOR_SCORES):
            raise ValueError("Web TRAIN requires actual streamed raw behavior thought logprobs")
        row.update(request_status="native", delivery="cancelled" if cancelled else "delivered",
            trace_id=trace_id, prompt_ids=list(completion.prompt_ids),
            completion_ids=list(completion.completion_ids),
            thought_token_count=completion.thought_token_count,
            native_status=completion.native_status, text=completion.text,
            thought_logprobs=list(probabilities), thought_logprobs_source=probability_source)
        self.sink(row)


@dataclass(frozen=True)
class ClosedPublicGroup:
    task: Task
    results: tuple
    closure: dict
    source_commit: str
    final_prompt_ids: tuple


def verify_owned_native_group(task, results, owner_rows, traces, *, eos_ids, tokenizer,
                              iteration, rank, web_train_prompt_history_config=None, history_budgets=None):
    """Replay exact closed journal bytes, original schemas and actual token IDs."""
    from jsonschema import Draft202012Validator
    from .hf_native_policy import _strict_json
    from .hf_native_policy import CANCELLED_BEFORE_JSON, NativeHFRequestExpiredBeforeSubmission
    from .hf_native_structured import native_structured_schema_kind
    from .hf_policy import _chat_tokens
    history_config = (PromptHistoryConfig() if web_train_prompt_history_config is None
                      else web_train_prompt_history_config)
    if not isinstance(history_config, PromptHistoryConfig):
        raise ValueError("An explicit Web TRAIN PromptHistoryConfig is required for ownership")
    if history_config.enabled:
        web_train_history_configuration(history_config)
        if (type(history_budgets) is not dict or set(history_budgets) != set(BUDGET_FIELDS)
                or any(type(history_budgets[key]) is not int or history_budgets[key] < 1
                       for key in BUDGET_FIELDS)
                or history_budgets["max_prompt_tokens"] + history_budgets["thought_tokens"]
                    + history_budgets["json_tokens"] > history_budgets["max_model_len"]):
            raise ValueError("Actual Web TRAIN policy budgets are required for history ownership")
    elif history_budgets is not None:
        raise ValueError("Web TRAIN history budgets require the explicit enabled configuration")
    if not eos_ids or END_THINK in eos_ids or len(results) < 2:
        raise ValueError("Exact native EOS IDs and complete group required")
    index = {row["trace_id"]: row for row in traces}
    if len(index) != len(traces):
        raise ValueError("Duplicate native trace IDs")
    used, prompts = set(), []
    allowed = {"completed", "thought_did_not_stop_on_real_close", CANCELLED_BEFORE_JSON,
               "json_did_not_finish_with_real_eos", "json_not_complete_or_schema_valid"}
    for sample, result in enumerate(results):
        rows = [row for row in owner_rows if row.get("sample") == sample]
        if (any((row.get("task_id"), row.get("iteration"), row.get("rank"))
                != (task.id, iteration, rank) for row in rows)
                or [row.get("turn") for row in rows] != list(range(len(rows)))):
            raise ValueError("Exact ordered sample/turn ownership required")
        delivered = [row for row in rows if row.get("request_status") == "native"
                     and row.get("delivery") == "delivered"]
        if len(delivered) != len(result.steps):
            raise ValueError("Every delivered native request must match a retained episode step")
        for step, row in zip(result.steps, delivered, strict=True):
            if (getattr(step, "trace_id", None) != row["trace_id"]
                    or step.prompt_ids != row["prompt_ids"]
                    or step.completion_ids != row["completion_ids"]
                    or step.text != row["text"]
                    or step.thought_token_count != row["thought_token_count"]
                    or step.native_status != row["native_status"]
                    or not _same(step.thought_logprobs, row["thought_logprobs"])
                    or step.thought_logprobs_source != row["thought_logprobs_source"]):
                raise ValueError("Delivered step/completion owner differs")
        for row in rows:
            if row.get("request_status") == "error":
                if (row.get("error_type") == NativeHFRequestExpiredBeforeSubmission.__name__
                        and (row.get("delivery") != "cancelled" or result.status != "infrastructure_error"
                             or any(key in row for key in ("trace_id", "completion", "completion_ids",
                                 "thought_logprobs", "native_hf_preflight", FAILURE_FIELD)))):
                    raise ValueError("Unsubmitted expired request requires cancelled infrastructure outcome")
                if result.status not in {"infrastructure_error", "context_budget_failure"}:
                    raise ValueError("Untraced native request error requires an unreliable episode")
                preflight = row.get("native_hf_preflight")
                if preflight is None and FAILURE_FIELD in row:
                    raise ValueError("Web TRAIN history failure requires its original native HF preflight")
                if preflight is not None:
                    failure = result.metrics.get("prompt_history_failure", {})
                    trace_id = preflight.get("trace_id")
                    if (trace_id in used or index.get(trace_id) != preflight
                            or row.get("error_type") != NativeHFPromptBudgetExceeded.__name__
                            or preflight.get("protocol") != PREFLIGHT_PROTOCOL
                            or preflight.get("task_id") != task.id
                            or preflight.get("generation_submitted") is not False
                            or _chat_tokens(tokenizer, row["messages"]) != preflight.get("prompt_ids")):
                        raise ValueError("Native HF preflight error request ownership differs")
                    if any(key in preflight for key in (*TRAIN_HISTORY_FIELDS, FAILURE_FIELD)):
                        raise ValueError("Original native HF preflight must retain its unchanged full-request proof")
                    if history_config.enabled:
                        if native_structured_schema_kind(row["json_schema"]) not in {"web", "web-review"}:
                            raise ValueError("Web TRAIN history failure requires an original Web schema")
                        try:
                            web_train_history_plan(tokenizer, row["messages"],
                                config=history_config, **history_budgets)
                        except PromptHistoryBudgetExceeded as history_error:
                            expected_failure = web_train_history_failure_fields(row["messages"], history_error, history_config)
                        else:
                            raise ValueError("Web TRAIN history preflight rejected a request that can fit")
                        if (not _same(row.get(FAILURE_FIELD), expected_failure)
                                or not _same(preflight.get("prompt_ids"), expected_failure["original_prompt_ids"])
                                or preflight.get("effective_prompt_limit") != history_budgets["max_prompt_tokens"]
                                or preflight.get("thought_tokens") != history_budgets["thought_tokens"]
                                or preflight.get("json_tokens") != history_budgets["json_tokens"]):
                            raise ValueError("Web TRAIN history failed-plan ownership differs from recomputed request")
                    elif FAILURE_FIELD in row:
                        raise ValueError("Unexpected Web TRAIN history failure without explicit configuration")
                    if result.status == "context_budget_failure":
                        if (row["messages"] != failure.get("request_messages")
                                or row["turn"] != failure.get("turn_index")
                                or preflight != failure.get("native_hf_preflight")):
                            raise ValueError("Native HF preflight error outcome ownership differs")
                        restore_verified_native_context(result, traces=traces, tokenizer=tokenizer)
                    elif row.get("delivery") != "cancelled":
                        raise ValueError("Infrastructure preflight error requires cancelled delivery")
                    used.add(trace_id)
                continue
            if row.get("request_status") != "native":
                raise ValueError("Unfinished or unknown native request outcome")
            if row["delivery"] == "cancelled" and result.status != "infrastructure_error":
                raise ValueError("Cancelled delivery requires retained infrastructure failure")
            trace_id, count = row["trace_id"], row["thought_token_count"]
            if (trace_id in used or trace_id not in index or type(count) is not int
                    or not 0 <= count <= len(row["completion_ids"])):
                raise ValueError("Missing/reused/foreign completion trace ownership")
            trace = index[trace_id]
            if FAILURE_FIELD in row or FAILURE_FIELD in trace:
                raise ValueError("A sampled native completion cannot carry a failed history plan")
            if history_config.enabled:
                if native_structured_schema_kind(row["json_schema"]) not in {"web", "web-review"}:
                    raise ValueError("Web TRAIN history requires an original Web schema")
                try:
                    plan = web_train_history_plan(tokenizer, row["messages"],
                        config=history_config, **history_budgets)
                except PromptHistoryBudgetExceeded as error:
                    raise ValueError("Saved Web TRAIN request has no fitting whole-turn plan") from error
                expected_history = web_train_history_fields(row["messages"], plan.report, history_config)
                if (not _same({key: trace.get(key) for key in TRAIN_HISTORY_FIELDS}, expected_history)
                        or "native_web_prompt_history" in trace):
                    raise ValueError("Web TRAIN history trace differs from recomputed original request")
                expected_prompt = plan.prompt_ids
            else:
                if any(key in trace for key in (*TRAIN_HISTORY_FIELDS, "native_web_prompt_history")):
                    raise ValueError("Unexpected Web TRAIN history without explicit configuration")
                expected_prompt = _chat_tokens(tokenizer, row["messages"])
            if (not _same(expected_prompt, row["prompt_ids"])
                    or trace.get("protocol") != HF_PROTOCOL or trace.get("task_id") != task.id
                    or trace.get("thought_sampling_profile") != "raw"
                    or trace.get("thought_logprobs_source") != BEHAVIOR_SCORES
                    or not _same(trace.get("thought_logprobs"), row["thought_logprobs"])
                    or row.get("thought_logprobs_source") != BEHAVIOR_SCORES
                    or trace.get("prompt_ids") != row["prompt_ids"]
                    or trace.get("thought_ids") != row["completion_ids"][:count]
                    or trace.get("json_ids") != row["completion_ids"][count:]
                    or trace.get("thought_token_count") != count
                    or trace.get("status") != row["native_status"]
                    or trace.get("json_public_schema") != row["json_schema"]
                    or trace.get("final_text") != row["text"] or trace.get("status") not in allowed):
                raise ValueError("Native messages/schema/token/status trace binding differs")
            native_structured_schema_kind(row["json_schema"])
            if trace["status"] == CANCELLED_BEFORE_JSON:
                thought = trace["thought_ids"]
                if (row["delivery"] != "cancelled" or result.status != "infrastructure_error"
                        or not thought or thought[-1] != END_THINK or thought.count(END_THINK) != 1
                        or set(thought).intersection(eos_ids) or trace["json_ids"] or row["text"]):
                    raise ValueError("Cancelled-before-JSON trace must retain only its actual closed thought")
            if result.status == "completed" and trace["status"] != "completed":
                raise ValueError("Completed episode contains failed native request")
            if trace["status"] == "completed":
                thought, final = trace["thought_ids"], trace["json_ids"]
                if (not thought or thought[-1] != END_THINK or thought.count(END_THINK) != 1
                        or set(thought).intersection(eos_ids) or not final
                        or final[-1] not in eos_ids or set(final[:-1]).intersection(eos_ids)):
                    raise ValueError("Native completion lacks real thinking close or EOS")
                if tokenizer.decode(final[:-1], skip_special_tokens=False) != row["text"]:
                    raise ValueError("Actual final token IDs do not decode to saved JSON")
                Draft202012Validator(row["json_schema"]).validate(_strict_json(row["text"]))
            used.add(trace_id)
        prompts.append(tuple(result.steps[-1].prompt_ids) if result.steps else ())
    if (used != set(index) or any(type(row.get("sample")) is not int
            or not 0 <= row["sample"] < len(results) for row in owner_rows)):
        raise ValueError("Unowned native trace/request in full group closure")
    return tuple(prompts)


class PublicGroupJournals:
    """Fresh per-group journals; no label reader or reward callback is held here."""
    def __init__(self, directory, *, task, group_size, iteration, rank, source_commit):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.task, self.group_size = task, group_size
        self.iteration, self.rank, self.source_commit = iteration, rank, source_commit
        self.writers = {}
        try:
            for name in ("native_traces", "request_owners", "public_outcomes"):
                self.writers[name] = OrderedDurableJSONLWriter(self.directory / f"{name}.jsonl")
        except BaseException as error:
            close_durable_journals(self.writers.values(), failure=error)
            raise
        self.closed = False

    def enqueue(self, name, row):
        self.writers[name].write_line(_json_line(row))

    def abort(self, error):
        close_durable_journals(self.writers.values(), failure=error)
        self.closed = True

    def close_generation(self, results, *, policy):
        """Called only after full sibling/model drain; return no marker on failure."""
        if self.closed or policy.worker is not None or len(results) != self.group_size:
            raise ValueError("Fresh full group and genuinely drained model worker required")
        for sample, result in enumerate(results):
            if ((result.task_id, result.benchmark, result.split)
                    != (self.task.id, "widesearch", "train") or result.reward != 0):
                raise ValueError("All same-task outcomes must remain unscored before close")
            self.enqueue("public_outcomes", dict(sample=sample, **result.to_dict(include_tokens=True)))
        close_durable_journals(self.writers.values())
        self.closed = True
        saved = {name: _read_jsonl(self.directory / f"{name}.jsonl") for name in self.writers}
        if any(len(saved[name]) != writer.count for name, writer in self.writers.items()):
            raise ValueError("Durable journal count differs from exact closed bytes")
        expected = [dict(sample=sample, **result.to_dict(include_tokens=True))
                    for sample, result in enumerate(results)]
        # JSON roundtrip converts dataclass tuples to the same serialized arrays.
        if not _same(saved["public_outcomes"], json.loads(json.dumps(expected))):
            raise ValueError("Closed public outcome journal does not retain every exact episode")
        prompts = verify_owned_native_group(self.task, results, saved["request_owners"],
            saved["native_traces"], eos_ids=policy.eos_ids, tokenizer=policy.tokenizer,
            iteration=self.iteration, rank=self.rank,
            **(dict(web_train_prompt_history_config=policy.web_train_prompt_history_config,
                    history_budgets=web_train_history_budgets(policy))
               if getattr(getattr(policy, "web_train_prompt_history_config", None), "enabled", False) else {}))
        closure = dict(version=GROUP_CLOSE, task_id=self.task.id, episodes=self.group_size,
            iteration=self.iteration, rank=self.rank, source_commit=self.source_commit,
            all_outcomes_persisted=True, policy_drained=True, native_ownership_verified=True,
            journals_flushed=True, rollout_labels_visible=False, rollout_scored=False,
            closed_journals={name: writer.count for name, writer in self.writers.items()})
        with (self.directory / "generation_closed.json").open("x", encoding="utf-8") as stream:
            stream.write(_json_line(closure))
            stream.flush()
            os.fsync(stream.fileno())
        return ClosedPublicGroup(self.task, tuple(results), closure, self.source_commit, prompts)


async def collect_public_group(runner, task, policy, *, group_size, rank, iteration,
                               directory, source_commit):
    """Public episodes, all sibling/native drains, closed journals, then return."""
    if (task.benchmark != "widesearch" or task.split != "train" or task.gold is not None
            or type(group_size) is not int or group_size < 2):
        raise ValueError("Public-only Web TRAIN group of at least two required")
    journals = PublicGroupJournals(directory, task=task, group_size=group_size,
                                  iteration=iteration, rank=rank, source_commit=source_commit)
    previous_sink = policy.trace_sink
    def trace_sink(row):
        journals.enqueue("native_traces", row)
        if previous_sink is not None:
            previous_sink(row)
    policy.trace_sink = trace_sink
    owners = [OwnedEpisodePolicy(policy.bind_task(task.id, rank), task_id=task.id,
        iteration=iteration, rank=rank, sample=sample,
        sink=lambda row: journals.enqueue("request_owners", row)) for sample in range(group_size)]
    pending = [asyncio.create_task(runner.run(task, owner, evaluation=True, score_train=False))
               for owner in owners]
    failure, results = None, None
    try:
        try:
            results = await asyncio.gather(*pending)
        except BaseException as error:
            failure = error
            raise
        finally:
            for worker in pending:
                if not worker.done():
                    worker.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            try:
                await policy.close()
            except BaseException as error:
                if failure is None:
                    raise
                failure.add_note(f"Policy drain: {type(error).__name__}: {error}")
        return journals.close_generation(results, policy=policy)
    except BaseException as error:
        failure = error
        journals.abort(error)
        raise
    finally:
        policy.trace_sink = previous_sink
        if not journals.closed:
            journals.abort(failure)


def score_closed_group(group, *, inputs, tokenizer, reward_variant=TABLE_GROUNDED_REWARD):
    """Original literal scorer and original native SFT gates, after actual close."""
    if reward_variant == ROW_GROUNDED_REWARD:
        return _score_row_grounded_closed_group(group, inputs=inputs, tokenizer=tokenizer)
    if reward_variant != TABLE_GROUNDED_REWARD:
        raise ValueError("Unknown online Web TRAIN reward variant")
    from dcc_train_web_literal_reward import evaluate_train_literal_rows
    if not isinstance(group, ClosedPublicGroup) or not group.closure.get("journals_flushed"):
        raise ValueError("Closed public Web TRAIN group required before label access")
    label = inputs.label_after_close(group)
    unreliable = unreliable_context_or_environment_group(group.results)
    allowed = {"completed", "max_steps", "invalid_answer", "evidence_failure",
               "model_protocol_failure", "infrastructure_error", "context_budget_failure"}
    scored, receipts = [], []
    for sample, result in enumerate(group.results):
        if result.status not in allowed or result.reward != 0:
            raise ValueError("All original unscored outcome statuses must remain in denominator")
        value, components, reasons = 0.0, None, []
        if result.status == "completed" and not unreliable:
            reasons = unchanged_native_cell_grounding(group.task, result,
                tokenizer.decode(group.final_prompt_ids[sample], skip_special_tokens=False))
            if not reasons:
                components = evaluate_train_literal_rows(result.answer, label, group.task.to_dict(),
                                                         expected_protocol=PROTOCOL)
                value = float(components["train_exact_row_f1"])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Finite bounded TRAIN literal reward required")
        receipt = dict(wrapper=WRAPPER, sample=sample, task_id=group.task.id,
            status=result.status, generation_closed_before_scoring=True,
            reward_protocol=PROTOCOL, label_protocol=LABEL_PROTOCOL, train_reward=configuration(),
            train_metric=dict(name="train_exact_row_f1", scope="widesearch-train-only",
                              value=value, evaluated=components is not None),
            native_cell_grounding_rejections=reasons, benchmark_score=False,
            gradient_group_discarded=unreliable)
        receipts.append(receipt)
        scored.append(replace(result, reward=value, metrics={**result.metrics, "train_literal": receipt}))
    return tuple(scored), dict(wrapper=WRAPPER, group_size=len(scored),
        status_counts=dict(Counter(result.status for result in scored)),
        all_outcomes_retained=True, model_failures_retained=True,
        gradient_group_discarded=unreliable, receipts=receipts)


def _row_grounded_components(task, result, label, decoded_final_prompt):
    """Keep every row in the denominator; grounding limits TP eligibility only.

    The existing guard checks substrings across the fetched-text pool, not
    semantic entailment or a row's relation to its particular cited source.
    Nonzero rewards can still be equal within a group and give zero advantage.
    """
    from dcc_train_web_literal_reward import evaluate_train_literal_rows, validate_label, _table
    raw = evaluate_train_literal_rows(result.answer, label, task.to_dict(), expected_protocol=PROTOCOL)
    grounding = dict(protocol=GROUNDING, scope="per-predicted-row", rows=[],
        table_rejections=[], eligible_rows=0, true_positive_rows=0,
        prediction_rows=raw["train_num_prediction_rows"], gold_rows=raw["train_num_gold_rows"])
    if not raw["prediction_valid"]:
        grounding["table_rejections"] = ["invalid_literal_prediction"]
        return 0.0, raw, grounding
    # Preserve the public schema/citation guard for the complete native answer.
    # Only the additional fetched-cell/final-prompt gates become row-scoped.
    guard = _WebEvidence()
    for event in result.tool_events:
        guard.observe(event.get("tool"), event.get("observation"))
    grounding["table_rejections"] = guard.reject_reasons(task, result.answer)
    if grounding["table_rejections"]:
        return 0.0, raw, grounding
    required, gold_keys = validate_label(label, task.to_dict(), expected_protocol=PROTOCOL)
    pred_keys = _table(result.answer, required, gold=False)
    eligible = Counter()
    for index, (row, key) in enumerate(zip(result.answer, pred_keys, strict=True)):
        reasons = unchanged_native_cell_grounding(task, replace(result, answer=[row]), decoded_final_prompt)
        grounding["rows"].append(dict(row_index=index, eligible=not reasons, rejections=reasons))
        if not reasons:
            eligible[key] += 1
    true = sum((eligible & Counter(gold_keys)).values())
    grounding.update(eligible_rows=sum(eligible.values()), true_positive_rows=true)
    # Do not score a filtered table: ineligible and duplicate rows still count.
    return 2 * true / (len(pred_keys) + len(gold_keys)), raw, grounding


def _score_row_grounded_closed_group(group, *, inputs, tokenizer):
    """Explicit online TRAIN variant; no benchmark/SFT/DPO scorer is changed."""
    if not isinstance(group, ClosedPublicGroup) or not group.closure.get("journals_flushed"):
        raise ValueError("Closed public Web TRAIN group required before label access")
    label = inputs.label_after_close(group)
    unreliable = unreliable_context_or_environment_group(group.results)
    allowed = {"completed", "max_steps", "invalid_answer", "evidence_failure",
               "model_protocol_failure", "infrastructure_error", "context_budget_failure"}
    profile = online_reward_configuration(ROW_GROUNDED_REWARD)
    scored, receipts = [], []
    for sample, result in enumerate(group.results):
        if result.status not in allowed or result.reward != 0:
            raise ValueError("All original unscored outcome statuses must remain in denominator")
        value, raw, grounding = 0.0, None, None
        if result.status == "completed" and not unreliable:
            value, raw, grounding = _row_grounded_components(group.task, result, label,
                tokenizer.decode(group.final_prompt_ids[sample], skip_special_tokens=False))
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Finite bounded TRAIN row-grounded reward required")
        receipt = dict(wrapper=ROW_REWARD_WRAPPER, sample=sample, task_id=group.task.id,
            status=result.status, generation_closed_before_scoring=True,
            reward_protocol=ROW_REWARD_PROTOCOL, reward_variant=ROW_GROUNDED_REWARD,
            label_protocol=LABEL_PROTOCOL, train_reward=profile,
            train_metric=dict(name=ROW_REWARD_METRIC, scope="widesearch-train-only",
                              value=value, evaluated=raw is not None),
            raw_literal_diagnostic=raw, per_row_grounding=grounding,
            benchmark_score=False, gradient_group_discarded=unreliable)
        receipts.append(receipt)
        scored.append(replace(result, reward=value, metrics={**result.metrics, "train_literal": receipt}))
    return tuple(scored), dict(wrapper=ROW_REWARD_WRAPPER, reward_variant=ROW_GROUNDED_REWARD,
        train_reward=profile, group_size=len(scored),
        status_counts=dict(Counter(result.status for result in scored)),
        all_outcomes_retained=True, model_failures_retained=True,
        gradient_group_discarded=unreliable, receipts=receipts)
