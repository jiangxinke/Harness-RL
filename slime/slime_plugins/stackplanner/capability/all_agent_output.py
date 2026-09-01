"""Two-route exact-token taxonomy for actor-backed StackPlanner roles."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .annotator import (
    _exact_token_offsets,
    _locate_json_value,
    _tool_payload_spans,
)

ALL_AGENT_OUTPUT_CAPABILITIES = ("action_choice", "action_output")
ALL_AGENT_OUTPUT_TAXONOMY_VERSION = "stackplanner-all-agent-action-output-v1"


def annotate_turn(
    *,
    tokenizer: Any,
    output_ids: list[int],
    raw_output: str,
    parsed: Any,
    tools_schema: list[dict] | None,
    annotation_context: Mapping[str, Any] | None = None,
) -> dict[str, list[Any]]:
    """Label exact served tokens as action choice or concrete output.

    The semantic partition is identical to experiment C: the exact recognized
    StackPlanner action identifier is ``action_choice`` and every other served
    token is ``action_output``. Experiment B additionally records the explicit
    actor-backed policy role for every token so its probe must cover every role
    sharing the actor.
    """

    role = str((annotation_context or {}).get("role") or "").strip()
    if not role:
        raise ValueError("experiment B action/output annotation requires explicit session role context")
    del parsed, tools_schema
    offsets = _exact_token_offsets(tokenizer, output_ids, raw_output)
    action_choice_chars = [False for _ in raw_output]
    for payload_start, payload_end, payload in _tool_payload_spans(raw_output):
        name = payload.get("name")
        if not isinstance(name, str) or not name:
            continue
        name_span = _locate_json_value(raw_output, "name", payload_start, payload_end)
        if not name_span:
            continue
        choice_start, choice_end = name_span
        if choice_end - choice_start >= 2 and raw_output[choice_start] == '"' and raw_output[choice_end - 1] == '"':
            choice_start += 1
            choice_end -= 1
        for index in range(choice_start, choice_end):
            action_choice_chars[index] = True

    labels = ["action_choice" if any(action_choice_chars[start:end]) else "action_output" for start, end in offsets]
    if len(labels) != len(output_ids) or any(label not in ALL_AGENT_OUTPUT_CAPABILITIES for label in labels):
        raise AssertionError("invalid all-agent action/output annotation")
    return {
        "capability_labels": labels,
        "actor_roles": [role] * len(labels),
    }
