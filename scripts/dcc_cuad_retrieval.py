"""Explicit portable TRAIN-only CUAD harness binding and public replay.

The binding preserves the original profile bytes. It has no filesystem paths,
model identity fields, labels or contract text. Source documents used for replay
remain separate public-task provenance, never retrieval-profile parameters.
"""
from __future__ import annotations

import base64
import binascii
import json

from jsonschema import Draft202012Validator

VERSION = "cuad-train-retrieval-binding-v1"
OBSERVATION = "Tool observation (source material):\n"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def strict_json(text):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "Duplicate CUAD retrieval JSON key")
            value[key] = item
        return value
    return json.loads(text, object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite CUAD retrieval JSON")))


def retrieval_runner_kwargs(binding):
    """Validate portable bytes with the actual runtime loader's validator."""
    from slime_plugins.dcc.retrieval import parse_retrieval_profile

    require(isinstance(binding, dict) and set(binding) == {
        "version", "initial_chunks", "retrieval_context_chars", "retrieval_max_context_chars",
        "profile_bytes_base64"} and binding["version"] == VERSION,
        "Explicit portable CUAD retrieval binding required")
    require(type(binding["initial_chunks"]) is int and 1 <= binding["initial_chunks"] <= 8,
            "CUAD initial_chunks must be an integer between one and eight")
    require(type(binding["retrieval_context_chars"]) is int and binding["retrieval_context_chars"] >= 0,
            "CUAD retrieval context must be a nonnegative integer")
    require(type(binding["retrieval_max_context_chars"]) is int
            and binding["retrieval_max_context_chars"] > 0, "CUAD retrieval context cap must be positive")
    try:
        encoded = binding["profile_bytes_base64"]
        require(isinstance(encoded, str), "Original CUAD retrieval UTF-8 bytes required")
        raw = base64.b64decode(encoded, validate=True)
        require(base64.b64encode(raw).decode("ascii") == encoded, "Noncanonical CUAD profile byte encoding")
        value = strict_json(raw.decode("utf-8"))
        require(isinstance(value, dict) and set(value) == {
            "schema", "version", "method", "parameters", "training", "questions"},
            "Portable retrieval profile has unsupported fields")
        parameters, training, questions = value["parameters"], value["training"], value["questions"]
        require(isinstance(parameters, dict) and set(parameters) == {
            "max_terms", "min_examples", "min_fraction", "rrf_constant"},
            "Portable retrieval parameters have unsupported fields")
        require(isinstance(training, dict) and set(training) == {
            "benchmark", "split", "tasks", "positive_tasks", "contracts", "unique_documents",
            "task_ids_sha256"}, "Portable retrieval TRAIN provenance has unsupported fields")
        require(type(training["unique_documents"]) is int and type(training["tasks"]) is int
                and 1 <= training["unique_documents"] <= training["tasks"],
                "Invalid portable retrieval document summary")
        require(isinstance(questions, dict) and all(isinstance(entry, dict)
                and set(entry) == {"category", "terms"} for entry in questions.values()),
                "Portable retrieval question entries have unsupported fields")
        actual = parse_retrieval_profile(value)
    except (UnicodeError, binascii.Error, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Malformed portable CUAD retrieval profile bytes") from exc
    return {"initial_chunks": binding["initial_chunks"], "retrieval_profile": actual,
            "retrieval_context_chars": binding["retrieval_context_chars"],
            "retrieval_max_context_chars": binding["retrieval_max_context_chars"]}


def make_retrieval_binding(profile_bytes, *, initial_chunks=8, retrieval_context_chars=0,
                           retrieval_max_context_chars=16000):
    """Keep exact original UTF-8 bytes; no rewriting or path serialization."""
    require(isinstance(profile_bytes, bytes), "Original retrieval profile bytes required")
    binding = {"version": VERSION, "initial_chunks": initial_chunks,
        "retrieval_context_chars": retrieval_context_chars,
        "retrieval_max_context_chars": retrieval_max_context_chars,
        "profile_bytes_base64": base64.b64encode(profile_bytes).decode("ascii")}
    retrieval_runner_kwargs(binding)
    return binding


def public_cuad_task(value, *, task_id=None):
    from slime_plugins.dcc.data import Task

    require(isinstance(value, dict) and set(value) == {
        "id", "benchmark", "split", "prompt", "document", "gold", "metadata"}
        and value["benchmark"] == "cuad" and value["split"] == "train"
        and value["gold"] is None and value["metadata"] == {}
        and all(isinstance(value[name], str) for name in ("id", "prompt", "document")),
        "Bound CUAD replay requires the exact public TRAIN task without labels")
    task = Task.from_dict(value)
    require(task_id is None or task.id == task_id, "Bound CUAD public task ownership differs")
    return task


def public_request_schema(task, messages, *, turn, max_steps, binding):
    """Recompute all CUAD source observations before accessing any labels."""
    from slime_plugins.dcc.cuad_structured import cuad_action_schema, cuad_final_schema
    from slime_plugins.dcc.episode import EpisodeRunner, _document_tools, final_turn_message

    public_cuad_task(task.to_dict())
    require(type(turn) is int and 0 <= turn < max_steps, "Bound CUAD turn exceeds approved budget")
    require(isinstance(messages, list) and all(isinstance(m, dict) and set(m) == {"role", "content"}
            and m["role"] in {"system", "user", "assistant"} and isinstance(m["content"], str)
            for m in messages), "Exact bound CUAD public messages required")
    runner = EpisodeRunner(max_steps=max_steps, **retrieval_runner_kwargs(binding))
    anchors = runner.initial_messages(task)
    require(messages[:len(anchors)] == anchors, "Bound CUAD retrieval anchors differ from approved public replay")
    index, history_turn = len(anchors), 0
    tools = _document_tools(task.document)
    while index < len(messages) and messages[index]["role"] == "assistant":
        require(history_turn < max_steps - 1 and index + 1 < len(messages),
                "Bound CUAD historical tool exceeds approved budget")
        action = strict_json(messages[index]["content"])
        Draft202012Validator(cuad_action_schema()).validate(action)
        require(isinstance(action, dict) and "answer" not in action
                and action.get("tool") in {"search_document", "read_document"},
                "Bound CUAD historical action must be an original document tool")
        try:
            observation = (runner._search_document(tools, **action["arguments"])
                if action["tool"] == "search_document" else tools.read_document(**action["arguments"]))
        except (TypeError, ValueError, OverflowError) as exc:
            observation = {"error": f"invalid tool arguments: {exc}"}
        expected = {"role": "user", "content": OBSERVATION
            + json.dumps(observation, ensure_ascii=False)
            + f"\nRemaining turns: {max_steps - history_turn - 1}."}
        require(messages[index + 1] == expected, "Bound CUAD tool observation differs from actual document replay")
        index += 2
        history_turn += 1
    require(history_turn == turn, "Bound CUAD turn differs from replayed document tools")
    controls = [final_turn_message(task)] if turn == max_steps - 1 else []
    require(messages[index:] == controls, "Bound CUAD final control or public history differs")
    return cuad_final_schema() if controls else cuad_action_schema()
