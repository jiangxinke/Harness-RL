"""Public Web evaluation inputs and complete native-request ownership.

This module neither opens reference labels nor computes reference metrics.
It supports independent Web tasks and prepared WideSearch tasks identically;
the backend field is not an assertion of official dataset provenance.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from .data import Task
from .episode import ModelProtocolFailure, parse_action

VERSION = "dcc-matched-HF-Web-native-ownership-v1"
PROVENANCE_FIELDS = ("source", "source_revision", "original_split", "curriculum_version",
                     "source_family", "language")
STATUSES = {"completed", "max_steps", "invalid_answer", "evidence_failure",
            "model_protocol_failure", "infrastructure_error", "context_budget_failure"}
SCORE_FIELDS = {"reward", "score", "precision_by_row", "recall_by_row", "f1_by_row",
                "precision_by_item", "recall_by_item", "f1_by_item", "judge_model",
                "evaluation_protocol", "alignment_protocol", "table_input_protocol"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":"))


def identity(value):
    """One compact identity of small task/config records, never model weights."""
    return hashlib.blake2b(canonical(value).encode("utf-8"), digest_size=16).hexdigest()


def read_jsonl(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key in evaluation data/journal")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError("Nonfinite evaluation JSON value: " + value)

    result = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n"):
                raise ValueError("Evaluation journal ends with an incomplete record")
            if line.strip():
                result.append(json.loads(line, object_pairs_hook=unique, parse_constant=nonfinite))
    return result


def public_task(task):
    """Only original public question and required output columns reach policy."""
    required = task.metadata.get("evaluation", {}).get("required")
    if (task.benchmark != "widesearch" or task.split not in {"validation", "test"}
            or not isinstance(required, list) or not required
            or any(type(column) is not str or not column.strip() or column == "citations" for column in required)
            or len(set(required)) != len(required)):
        raise ValueError("Held-out Web tasks with distinct public required columns are required")
    return replace(task, gold=None, metadata={"evaluation": {"required": list(required)}})


def load_public_tasks(path, *, split=None):
    """Read prepared JSONL, discarding references before constructing any Task.

    JSON decoding necessarily sees the prepared record. Gold, evaluation rules,
    teacher queries and source hints are not retained in the generation task.
    Only declared dataset provenance is retained for the manifest, then removed
    again by public_task before the runner sees the task.
    """
    tasks = []
    for row in read_jsonl(path):
        if split is not None and row.get("split") != split:
            continue
        meta = row.get("metadata", {})
        value = {key: row[key] for key in ("id", "benchmark", "split", "prompt")}
        value.update(document=row.get("document", ""), gold=None,
                     metadata={key: deepcopy(meta[key]) for key in PROVENANCE_FIELDS if key in meta})
        value["metadata"]["evaluation"] = {"required": deepcopy(meta.get("evaluation", {}).get("required"))}
        task = Task.from_dict(value)
        public_task(task)
        tasks.append(task)
    if not tasks or len({task.id for task in tasks}) != len(tasks):
        raise ValueError("Nonempty unique held-out Web task set required")
    return tasks


def task_identity(tasks):
    return identity([asdict(public_task(task)) for task in tasks])


def dataset_identity(tasks):
    return {"backend_field": "widesearch", "task_count": len(tasks),
            "identity_policy": "metadata declarations recorded verbatim; no automatic official benchmark claim",
            "declared_metadata": [{"task_id": task.id, "split": task.split,
                **{key: task.metadata[key] for key in PROVENANCE_FIELDS if key in task.metadata}}
                for task in tasks]}


def prediction_row(result, *, seed):
    if result.reward != 0 or SCORE_FIELDS.intersection(result.metrics):
        raise ValueError("Web generation must remain unscored before durable closure")
    row = result.to_dict(include_tokens=True)
    row.pop("reward")
    row.update(task_seed=seed, rollout_labels_removed=True, rollout_scored=False)
    return row


class OwnedEvaluationPolicy:
    """Own and drain every native request, including timeout-late completions."""
    def __init__(self, bound, *, task_id, rank, sink):
        self.bound, self.task_id, self.rank, self.sink = bound, task_id, rank, sink
        self.turn = 0

    async def __call__(self, *args, **kwargs):
        raise ValueError("Matched Web evaluation requires structured native calls")

    async def generate_structured(self, messages, *, json_schema, **kwargs):
        row = dict(task_id=self.task_id, shard_index=self.rank, turn=self.turn,
                   messages=deepcopy(messages), json_schema=deepcopy(json_schema))
        self.turn += 1
        pending = asyncio.create_task(self.bound.generate_structured(messages, json_schema=json_schema, **kwargs))
        cancelled = False
        try:
            try:
                completion = await asyncio.shield(pending)
            except asyncio.CancelledError:
                cancelled = True
                completion = await pending
        except ModelProtocolFailure as error:
            if error.trace_id != getattr(error.completion, "trace_id", None):
                raise ValueError("Protocol failure and native completion trace IDs differ") from error
            self._record(row, error.completion, cancelled=cancelled)
            if cancelled:
                raise asyncio.CancelledError() from error
            raise
        except BaseException as error:
            row.update(request_status="error", delivery="cancelled" if cancelled else "error",
                       error_type=type(error).__name__, error=str(error))
            self.sink(row)
            if cancelled:
                raise asyncio.CancelledError() from error
            raise
        self._record(row, completion, cancelled=cancelled)
        if cancelled:
            raise asyncio.CancelledError()
        return completion

    def _record(self, row, completion, *, cancelled):
        trace_id = getattr(completion, "trace_id", None)
        if type(trace_id) is not str or not trace_id:
            raise ValueError("Actual native completion trace ID required")
        row.update(request_status="native", delivery="cancelled" if cancelled else "delivered",
                   trace_id=trace_id, completion=asdict(completion))
        self.sink(row)


def verify_native_ownership(tasks, predictions, owners, traces, *, rank, profile, eos_ids, tokenizer=None):
    """Bind every persisted request/trace/step; optional token encoding replay.

    Generation supplies its real tokenizer before publishing its done marker.
    Score-only revalidates the complete persisted cross-links without loading
    a model/tokenizer. Cancelled native outputs remain owned, but earn no score.
    """
    from jsonschema import Draft202012Validator
    from .hf_native_policy import END_THINK, PROTOCOL, _strict_json
    from .hf_native_structured import _web_review_mode
    from .hf_policy import _chat_tokens

    if not eos_ids or END_THINK in eos_ids:
        raise ValueError("Exact native non-thinking EOS IDs required")
    expected = [task.id for task in tasks]
    if [row.get("task_id") for row in predictions] != expected:
        raise ValueError("Prediction ownership does not cover the exact assigned task order")
    index = {}
    for trace in traces:
        trace_id = trace.get("trace_id")
        if type(trace_id) is not str or not trace_id or trace_id in index:
            raise ValueError("Missing or duplicate native trace ID")
        index[trace_id] = trace
    if any(row.get("task_id") not in expected for row in owners):
        raise ValueError("Foreign native request owner")
    used = set()
    allowed = {"completed", "thought_did_not_stop_on_real_close", "json_did_not_finish_with_real_eos",
               "json_not_complete_or_schema_valid"}
    for task, result in zip(tasks, predictions, strict=True):
        public_task(task)
        if (result.get("status") not in STATUSES or result.get("benchmark") != "widesearch"
                or result.get("split") != task.split or result.get("rollout_labels_removed") is not True
                or result.get("rollout_scored") is not False or "reward" in result
                or SCORE_FIELDS.intersection(result.get("metrics", {}))):
            raise ValueError("Unscored original held-out Web outcome required")
        rows = [row for row in owners if row.get("task_id") == task.id]
        if ([row.get("turn") for row in rows] != list(range(len(rows)))
                or any(row.get("shard_index") != rank for row in rows)):
            raise ValueError("Exact sequential native request ownership required")
        delivered = [row for row in rows if row.get("request_status") == "native" and row.get("delivery") == "delivered"]
        if canonical([row["completion"] for row in delivered]) != canonical(result.get("steps")):
            raise ValueError("Delivered native requests differ from the retained episode steps")
        for owner in rows:
            if owner.get("request_status") == "error":
                if result["status"] not in {"infrastructure_error", "context_budget_failure"}:
                    raise ValueError("Untraced request error requires a retained unreliable outcome")
                continue
            if (owner.get("request_status") != "native"
                    or owner.get("delivery") not in {"delivered", "cancelled"}
                    or owner["delivery"] == "cancelled" and result["status"] != "infrastructure_error"):
                raise ValueError("Unknown or inconsistent native request outcome")
            trace_id, completion = owner.get("trace_id"), owner["completion"]
            if trace_id not in index or trace_id in used or completion.get("trace_id") != trace_id:
                raise ValueError("Missing/reused/foreign native trace binding")
            trace = index[trace_id]
            count, ids = completion.get("thought_token_count"), completion.get("completion_ids")
            if (type(count) is not int or not isinstance(ids, list) or not 0 < count <= len(ids)
                    or completion.get("native_status") not in allowed
                    or trace.get("protocol") != PROTOCOL or trace.get("task_id") != task.id
                    or trace.get("thought_sampling_profile") != profile
                    or trace.get("prompt_ids") != completion.get("prompt_ids")
                    or trace.get("thought_ids") != ids[:count] or trace.get("json_ids") != ids[count:]
                    or trace.get("thought_token_count") != count
                    or trace.get("status") != completion["native_status"]
                    or trace.get("json_public_schema") != owner["json_schema"]
                    or trace.get("final_text") != completion["text"]
                    or completion.get("phase_lengths") != [count, len(ids) - count]):
                raise ValueError("Native prompt/schema/token/status trace binding differs")
            if tokenizer is not None and _chat_tokens(tokenizer, owner["messages"]) != completion["prompt_ids"]:
                raise ValueError("Original public request does not encode to the actual native prompt IDs")
            # Round-trip the exact original public constructors. Do not compile
            # a grammar during score-only: actual generation already admitted
            # it through NativeHFPolicy before any model call.
            _web_review_mode(owner["json_schema"])
            if trace["status"] == "completed":
                thought, final = ids[:count], ids[count:]
                if (thought[-1] != END_THINK or thought.count(END_THINK) != 1 or set(thought).intersection(eos_ids)
                        or not final or final[-1] not in eos_ids or set(final[:-1]).intersection(eos_ids)):
                    raise ValueError("Native success lacks its real thinking close or final EOS")
                if tokenizer is not None and tokenizer.decode(final[:-1], skip_special_tokens=False) != completion["text"]:
                    raise ValueError("Native JSON text differs from the actually sampled token IDs")
                Draft202012Validator(owner["json_schema"]).validate(_strict_json(completion["text"]))
            used.add(trace_id)
        if result["status"] == "completed":
            if not delivered or any(row["completion"]["native_status"] != "completed" for row in delivered):
                raise ValueError("Completed Web episode requires delivered successful native calls")
            action = parse_action(delivered[-1]["completion"]["text"], widesearch_rows=True)
            if action is None or action.get("answer") != result.get("answer"):
                raise ValueError("Scored answer differs from the actual native final JSON")
    if used != set(index):
        raise ValueError("Unowned native trace remains in the closed evaluation shard")
    return {"version": VERSION, "requests": len(owners), "native_traces": len(traces),
            "delivered_steps": sum(len(row["steps"]) for row in predictions),
            "cancelled_native_completions": sum(row.get("request_status") == "native" and row.get("delivery") == "cancelled" for row in owners),
            "all_native_traces_owned": True}
