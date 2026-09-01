"""Build a safe, user-facing execution trace from persisted run events.

The trace is intentionally an *execution audit*, not a chain-of-thought dump.
It exposes observable model output, explicit SP action reasons, the exact
task-memory context captured for debug runs, subagent/tool activity, latency,
errors, and token usage. Provider-private reasoning fields are discarded and
credential-shaped values are redacted before anything reaches the browser.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

TRACE_SUMMARY_MAX_CHARS = 1200
TRACE_DETAIL_MAX_CHARS = 12000
TRACE_TOTAL_DETAIL_MAX_CHARS = 2_000_000

_HIDDEN_REASONING_KEYS = frozenset(
    {
        "chain_of_thought",
        "hidden_reasoning",
        "reasoning",
        "reasoning_content",
        "thinking",
        "thinking_content",
    }
)
_SENSITIVE_KEY_PATTERN = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|authorization|cookie|credential|password|secret|session|token)(?:$|[_-])",
    re.IGNORECASE,
)
_BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}")
_API_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")
_QUERY_SECRET_PATTERN = re.compile(r"(?i)([?&](?:api[_-]?key|access[_-]?token|token|secret|password)=)[^&#\s]+")
_TERMINAL_RUN_STATUSES = frozenset({"success", "error", "interrupted", "cancelled"})
_TERMINAL_SUBAGENT_STATUSES = frozenset({"completed", "failed", "cancelled", "timed_out"})


def _clip(value: Any, max_chars: int) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    suffix = "...<truncated>"
    return f"{text[: max_chars - len(suffix)]}{suffix}"


def _redact_text(value: str) -> str:
    redacted = _BEARER_PATTERN.sub("Bearer [REDACTED]", value)
    redacted = _API_KEY_PATTERN.sub("[REDACTED]", redacted)
    return _QUERY_SECRET_PATTERN.sub(r"\1[REDACTED]", redacted)


def sanitize_trace_value(value: Any, *, max_chars: int = TRACE_DETAIL_MAX_CHARS, _depth: int = 0) -> Any:
    """Return a bounded JSON-safe value with secrets and hidden reasoning removed."""

    if _depth > 10:
        return "<max-depth>"
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return _clip(_redact_text(value), max_chars)
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized_key = key.strip().lower()
            if normalized_key in _HIDDEN_REASONING_KEYS:
                continue
            if _SENSITIVE_KEY_PATTERN.search(normalized_key):
                sanitized[key] = "[REDACTED]"
                continue
            sanitized[key] = sanitize_trace_value(item, max_chars=max_chars, _depth=_depth + 1)
        return sanitized
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return [sanitize_trace_value(item, max_chars=max_chars, _depth=_depth + 1) for item in value]
    return _clip(_redact_text(repr(value)), max_chars)


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _content(event: Mapping[str, Any]) -> dict[str, Any]:
    value = event.get("content")
    return dict(value) if isinstance(value, Mapping) else {"text": value} if value is not None else {}


def _metadata(event: Mapping[str, Any]) -> dict[str, Any]:
    value = event.get("metadata")
    return dict(value) if isinstance(value, Mapping) else {}


def _payload(event: Mapping[str, Any]) -> dict[str, Any]:
    content = _content(event)
    value = content.get("payload")
    if isinstance(value, Mapping):
        return dict(value)
    metadata_payload = _metadata(event).get("payload")
    return dict(metadata_payload) if isinstance(metadata_payload, Mapping) else {}


def _event_time(event: Mapping[str, Any]) -> datetime | None:
    content = _content(event)
    metadata = _metadata(event)
    return _parse_datetime(content.get("occurred_at")) or _parse_datetime(metadata.get("occurred_at")) or _parse_datetime(event.get("created_at"))


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, bytes | bytearray | str):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping):
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    if isinstance(content, Mapping):
        nested = content.get("text") or content.get("content")
        return _message_text(nested)
    return "" if content is None else str(content)


def _usage(value: Any) -> dict[str, int] | None:
    if not isinstance(value, Mapping):
        return None

    def _int(*keys: str) -> int:
        for key in keys:
            try:
                if value.get(key) is not None:
                    return max(int(value[key]), 0)
            except (TypeError, ValueError):
                continue
        return 0

    input_tokens = _int("input_tokens", "prompt_tokens")
    output_tokens = _int("output_tokens", "completion_tokens")
    total_tokens = _int("total_tokens") or input_tokens + output_tokens
    if not any((input_tokens, output_tokens, total_tokens)):
        return None
    return {"input": input_tokens, "output": output_tokens, "total": total_tokens}


def _status_value(record: Any) -> str:
    status = getattr(record, "status", "unknown")
    return str(getattr(status, "value", status))


def _action_id(event: Mapping[str, Any]) -> str | None:
    value = _content(event).get("action_id") or _metadata(event).get("action_id")
    return str(value) if value else None


def _task_id(event: Mapping[str, Any]) -> str | None:
    value = _content(event).get("task_id") or _metadata(event).get("task_id") or _payload(event).get("task_id")
    return str(value) if value else None


def _sp_action_name(tool_name: Any) -> str:
    name = str(tool_name or "SP_ACTION")
    return name[3:].upper() if name.startswith("sp_") else name.upper()


def _tool_request_summary(tool_calls: list[dict[str, Any]]) -> str:
    summaries: list[str] = []
    for call in tool_calls:
        name = str(call.get("name") or "tool")
        args = call.get("args")
        focus = None
        if isinstance(args, Mapping):
            for key in ("query", "url", "path", "command", "task"):
                if args.get(key):
                    focus = args[key]
                    break
        summaries.append(f"{name}: {_clip(focus, 300)}" if focus is not None else name)
    return _clip("；".join(summaries), TRACE_SUMMARY_MAX_CHARS)


def _step(
    *,
    step_id: str,
    seq: int | None,
    kind: str,
    label: str,
    actor: str,
    status: str,
    started_at: datetime | None,
    ended_at: datetime | None = None,
    duration_ms: int | None = None,
    parent_id: str | None = None,
    summary: str | None = None,
    detail: Any = None,
    tokens: dict[str, int] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    if duration_ms is None and started_at is not None and ended_at is not None:
        duration_ms = max(int((ended_at - started_at).total_seconds() * 1000), 0)
    return {
        "id": step_id,
        "seq": seq,
        "parent_id": parent_id,
        "kind": kind,
        "label": label,
        "actor": actor,
        "status": status,
        "started_at": _iso(started_at),
        "ended_at": _iso(ended_at),
        "duration_ms": duration_ms,
        "offset_ms": None,
        "tokens": tokens,
        "summary": _clip(_redact_text(summary), TRACE_SUMMARY_MAX_CHARS) if summary else None,
        "detail": sanitize_trace_value(detail) if detail is not None else None,
        "error": _clip(_redact_text(error), TRACE_SUMMARY_MAX_CHARS) if error else None,
    }


def _llm_steps(events: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    response_events = [event for event in events if event.get("event_type") == "llm.ai.response"]
    last_lead_text_seq: int | None = None
    for event in response_events:
        metadata = _metadata(event)
        content = _content(event)
        text = _message_text(content.get("content")).strip()
        if metadata.get("caller", "lead_agent") == "lead_agent" and text and not content.get("tool_calls"):
            last_lead_text_seq = int(event.get("seq") or 0)

    steps: list[dict[str, Any]] = []
    for event in response_events:
        content = _content(event)
        metadata = _metadata(event)
        caller = str(metadata.get("caller") or "lead_agent")
        end = _event_time(event)
        try:
            latency_ms = max(int(metadata.get("latency_ms")), 0) if metadata.get("latency_ms") is not None else None
        except (TypeError, ValueError):
            latency_ms = None
        start = end - timedelta(milliseconds=latency_ms) if end is not None and latency_ms is not None else end
        seq = int(event.get("seq") or 0)
        text = _message_text(content.get("content")).strip()
        raw_tool_calls = content.get("tool_calls")
        tool_calls = [dict(item) for item in raw_tool_calls if isinstance(item, Mapping)] if isinstance(raw_tool_calls, list) else []
        token_usage = _usage(metadata.get("usage") or content.get("usage_metadata"))

        if caller == "lead_agent" and tool_calls:
            action_calls = [call for call in tool_calls if str(call.get("name") or "").startswith("sp_")]
            names = [_sp_action_name(call.get("name")) for call in (action_calls or tool_calls)]
            counts: dict[str, int] = defaultdict(int)
            for name in names:
                counts[name] += 1
            labels = [f"{name} ×{count}" if count > 1 else name for name, count in counts.items()]
            reasons: list[str] = []
            for call in action_calls:
                args = call.get("args")
                if isinstance(args, Mapping):
                    reason = args.get("reason") or args.get("task") or args.get("summary")
                    if reason:
                        reasons.append(str(reason))
            steps.append(
                _step(
                    step_id=f"llm-{seq}",
                    seq=seq,
                    kind="decision",
                    label=f"中枢决策 · {', '.join(labels)}",
                    actor="central",
                    status="completed",
                    started_at=start,
                    ended_at=end,
                    duration_ms=latency_ms,
                    summary="；".join(reasons) or text or None,
                    detail={"actions": action_calls or tool_calls},
                    tokens=token_usage,
                )
            )
            continue

        if caller == "lead_agent":
            is_final = bool(text) and seq == last_lead_text_seq
            steps.append(
                _step(
                    step_id=f"llm-{seq}",
                    seq=seq,
                    kind="final_answer" if is_final else "central_output",
                    label="最终回答" if is_final else "中枢显式输出",
                    actor="central",
                    status="completed",
                    started_at=start,
                    ended_at=end,
                    duration_ms=latency_ms,
                    summary=text or "模型返回了空文本",
                    detail={"tool_calls": tool_calls} if tool_calls else None,
                    tokens=token_usage,
                )
            )
            continue

        actor = caller.split(":", 1)[1] if ":" in caller else caller
        steps.append(
            _step(
                step_id=f"llm-{seq}",
                seq=seq,
                kind="model_call",
                label="中间件模型调用" if caller.startswith("middleware:") else "子智能体模型调用",
                actor=actor,
                status="completed",
                started_at=start,
                ended_at=end,
                duration_ms=latency_ms,
                summary=text or _tool_request_summary(tool_calls),
                detail={"tool_calls": tool_calls} if tool_calls else None,
                tokens=token_usage,
            )
        )
    return steps


def _action_steps(events: list[Mapping[str, Any]], run_status: str) -> list[dict[str, Any]]:
    created: dict[str, Mapping[str, Any]] = {}
    starts: dict[str, Mapping[str, Any]] = {}
    terminals: dict[str, Mapping[str, Any]] = {}
    for event in events:
        action_id = _action_id(event)
        if not action_id:
            continue
        event_type = str(event.get("event_type") or "")
        if event_type == "sp.action.created":
            created[action_id] = event
        elif event_type == "sp.handler.started":
            starts[action_id] = event
        elif event_type in {"sp.handler.completed", "sp.handler.failed"}:
            terminals[action_id] = event

    steps: list[dict[str, Any]] = []
    for action_id in dict.fromkeys([*created, *starts, *terminals]):
        created_event = created.get(action_id)
        start_event = starts.get(action_id) or created_event
        terminal_event = terminals.get(action_id)
        if start_event is None:
            continue
        created_payload = _payload(created_event or start_event)
        start_payload = _payload(start_event)
        terminal_payload = _payload(terminal_event) if terminal_event is not None else {}
        action_type = str(created_payload.get("action_type") or start_payload.get("action_type") or "UNKNOWN")
        start = _event_time(start_event)
        end = _event_time(terminal_event) if terminal_event is not None else None
        failed = terminal_event is not None and terminal_event.get("event_type") == "sp.handler.failed"
        status = "failed" if failed else "completed" if terminal_event is not None else "running" if run_status in {"pending", "running"} else "unknown"
        seq = int(start_event.get("seq") or (created_event.get("seq") if created_event else 0) or 0)
        detail = {**created_payload, "result": terminal_payload or None}
        steps.append(
            _step(
                step_id=f"action-{action_id}",
                seq=seq,
                kind="action",
                label=f"执行动作 · {action_type}",
                actor="central",
                status=status,
                started_at=start,
                ended_at=end,
                summary=str(created_payload.get("reason") or created_payload.get("task") or "") or None,
                detail=detail,
                error=str(terminal_payload.get("error") or "") or None,
            )
        )
    return steps


def _subagent_steps(events: list[Mapping[str, Any]], run_status: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        if str(event.get("event_type") or "").startswith("subagent."):
            task_id = _task_id(event)
            if task_id:
                grouped[task_id].append(event)

    output: list[dict[str, Any]] = []
    for task_id, task_events in grouped.items():
        task_events.sort(key=lambda event: (_event_time(event) or datetime.min.replace(tzinfo=UTC), int(event.get("seq") or 0)))
        start_event = next((event for event in task_events if event.get("event_type") == "subagent.start"), None)
        end_event = next((event for event in reversed(task_events) if event.get("event_type") == "subagent.end"), None)
        start_content = _content(start_event or {})
        end_content = _content(end_event or {})
        actor = str(start_content.get("subagent_type") or end_content.get("subagent_type") or "subagent")
        started_at = _event_time(start_event) if start_event is not None else _event_time(task_events[0])
        ended_at = _event_time(end_event) if end_event is not None else None
        status = str(end_content.get("status") or ("running" if run_status in {"pending", "running"} else "unknown"))
        parent_id = f"subagent-{task_id}"
        parent_seq = int((start_event or task_events[0]).get("seq") or 0)
        output.append(
            _step(
                step_id=parent_id,
                seq=parent_seq,
                kind="subagent",
                label=f"子智能体 · {actor}",
                actor=actor,
                status=status,
                started_at=started_at,
                ended_at=ended_at,
                summary=str(start_content.get("description") or start_content.get("prompt") or "") or None,
                detail={
                    "prompt": start_content.get("prompt"),
                    "result": end_content.get("result"),
                    "stop_reason": end_content.get("stop_reason"),
                },
                tokens=_usage(end_content.get("usage")),
                error=str(end_content.get("error") or "") or None,
            )
        )

        event_steps = [event for event in task_events if event.get("event_type") == "subagent.step"]
        for index, event in enumerate(event_steps):
            content = _content(event)
            at = _event_time(event)
            next_at = _event_time(event_steps[index + 1]) if index + 1 < len(event_steps) else ended_at
            if next_at is not None and at is not None and next_at < at:
                next_at = None
            seq = int(event.get("seq") or 0)
            message_index = content.get("message_index", seq)
            kind = str(content.get("kind") or "ai")
            text = _message_text(content.get("text")).strip()
            raw_tool_calls = content.get("tool_calls")
            tool_calls = [dict(item) for item in raw_tool_calls if isinstance(item, Mapping)] if isinstance(raw_tool_calls, list) else []
            if kind == "tool":
                tool_name = str(content.get("tool_name") or "tool")
                step_kind = "tool_result"
                label = f"工具结果 · {tool_name}"
                summary = text or "工具未返回文本"
                detail = {"tool_name": tool_name, "output": text, "truncated": content.get("truncated")}
            elif tool_calls:
                names = ", ".join(str(call.get("name") or "tool") for call in tool_calls)
                step_kind = "tool_request"
                label = f"请求工具 · {names}"
                summary = _tool_request_summary(tool_calls)
                detail = {"text": text or None, "tool_calls": tool_calls}
            else:
                step_kind = "subagent_output"
                label = "子智能体显式输出"
                summary = text or "模型返回了空文本"
                detail = {"text": text, "truncated": content.get("truncated")}
            output.append(
                _step(
                    # A subagent may emit multiple persisted observations for
                    # the same message index (for example the SP adapter and
                    # the native task event bridge).  The run-event sequence
                    # plus the stable local index keeps React keys unique while
                    # preserving deterministic trace exports.
                    step_id=f"{parent_id}-step-{seq}-{message_index}-{index}",
                    seq=seq,
                    parent_id=parent_id,
                    kind=step_kind,
                    label=label,
                    actor=actor,
                    status="completed" if status in _TERMINAL_SUBAGENT_STATUSES or next_at is not None else "running",
                    started_at=at,
                    ended_at=next_at,
                    summary=summary,
                    detail=detail,
                )
            )
    return output


def _auxiliary_steps(events: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for event in events:
        event_type = str(event.get("event_type") or "")
        seq = int(event.get("seq") or 0)
        at = _event_time(event)
        if event_type == "llm.human.input":
            content = _content(event)
            output.append(
                _step(
                    step_id=f"query-{seq}",
                    seq=seq,
                    kind="user_query",
                    label="用户请求",
                    actor="user",
                    status="completed",
                    started_at=at,
                    ended_at=at,
                    summary=_message_text(content.get("content") or content.get("text")),
                )
            )
        elif event_type in {"sp.central.context", "sp.central.decided"}:
            payload = _payload(event) or _content(event)
            if payload.get("prompt_context") or payload.get("memory_entries"):
                output.append(
                    _step(
                        step_id=f"decision-context-{seq}",
                        seq=seq,
                        kind="decision_context",
                        label="中枢决策上下文",
                        actor="central",
                        status="completed",
                        started_at=at,
                        ended_at=at,
                        summary=str(payload.get("reason") or "本轮中枢实际读取的任务上下文"),
                        detail=payload,
                    )
                )
        elif event_type.startswith("middleware:"):
            content = _content(event)
            output.append(
                _step(
                    step_id=f"middleware-{seq}",
                    seq=seq,
                    kind="middleware",
                    label=f"中间件 · {event_type.split(':', 1)[1]}",
                    actor="middleware",
                    status="completed",
                    started_at=at,
                    ended_at=at,
                    summary=str(content.get("action") or content.get("name") or event_type),
                    detail=content,
                )
            )
        elif event_type in {"llm.error", "run.error", "sp.central.failed", "sp.action.validation_failed", "sp.loop.max_iterations_exceeded"}:
            content = _content(event)
            payload = _payload(event)
            error = payload.get("error") or content.get("text") or content.get("error") or event_type
            output.append(
                _step(
                    step_id=f"error-{seq}",
                    seq=seq,
                    kind="error",
                    label=f"执行错误 · {event_type}",
                    actor="system",
                    status="failed",
                    started_at=at,
                    ended_at=at,
                    summary=str(error),
                    detail=payload or content,
                    error=str(error),
                )
            )
    return output


def _bound_total_detail_size(steps: list[dict[str, Any]]) -> None:
    """Keep a pathological long run from producing an unbounded JSON response."""

    remaining = TRACE_TOTAL_DETAIL_MAX_CHARS
    for step in steps:
        detail = step.get("detail")
        if detail is None:
            continue
        serialized = json.dumps(detail, ensure_ascii=False, default=str)
        if len(serialized) <= remaining:
            remaining -= len(serialized)
            continue
        preview_budget = max(min(remaining, TRACE_DETAIL_MAX_CHARS), 0)
        step["detail"] = {
            "_truncated": True,
            "_reason": "trace_detail_budget_exhausted",
            "_preview": _clip(serialized, preview_budget) if preview_budget else "",
        }
        remaining = 0


def build_debug_trace(record: Any, events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate persisted events into the execution-debug API response."""

    ordered_events = sorted(events, key=lambda event: int(event.get("seq") or 0))
    run_status = _status_value(record)
    started_at = _parse_datetime(getattr(record, "created_at", None))
    updated_at = _parse_datetime(getattr(record, "updated_at", None))
    event_times = [value for event in ordered_events if (value := _event_time(event)) is not None]
    effective_end = updated_at or (max(event_times) if event_times else None)
    ended_at = effective_end if run_status in _TERMINAL_RUN_STATUSES else None
    duration_ms = max(int((effective_end - started_at).total_seconds() * 1000), 0) if started_at and effective_end else None

    steps = [
        *_auxiliary_steps(ordered_events),
        *_llm_steps(ordered_events),
        *_action_steps(ordered_events, run_status),
        *_subagent_steps(ordered_events, run_status),
    ]
    for step in steps:
        step_start = _parse_datetime(step.get("started_at"))
        if started_at is not None and step_start is not None:
            step["offset_ms"] = max(int((step_start - started_at).total_seconds() * 1000), 0)
    steps.sort(
        key=lambda step: (
            _parse_datetime(step.get("started_at")) or datetime.max.replace(tzinfo=UTC),
            int(step.get("seq") or 0),
            str(step.get("id") or ""),
        )
    )
    _bound_total_detail_size(steps)

    metadata = getattr(record, "metadata", {})
    enabled = bool(metadata.get("debug_trace_enabled")) if isinstance(metadata, Mapping) else False
    return {
        "thread_id": str(getattr(record, "thread_id", "")),
        "run_id": str(getattr(record, "run_id", "")),
        "status": run_status,
        "enabled": enabled,
        "capture_level": "enhanced" if enabled else "baseline",
        "started_at": _iso(started_at),
        "ended_at": _iso(ended_at),
        "duration_ms": duration_ms,
        "event_count": len(ordered_events),
        "tokens": {
            "input": int(getattr(record, "total_input_tokens", 0) or 0),
            "output": int(getattr(record, "total_output_tokens", 0) or 0),
            "total": int(getattr(record, "total_tokens", 0) or 0),
            "llm_calls": int(getattr(record, "llm_call_count", 0) or 0),
            "lead_agent": int(getattr(record, "lead_agent_tokens", 0) or 0),
            "subagent": int(getattr(record, "subagent_tokens", 0) or 0),
            "middleware": int(getattr(record, "middleware_tokens", 0) or 0),
        },
        "steps": steps,
        "disclosure": {
            "hidden_chain_of_thought": False,
            "shows": [
                "observable_model_output",
                "explicit_action_reasons",
                "captured_task_context",
                "tool_and_subagent_io",
                "latency_and_token_usage",
            ],
        },
    }
