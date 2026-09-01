"""Read-only long-term memory recall normalization for SP."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from deerflow.sp.subagents import SPSubagentResult

DEFAULT_MAX_RECALL_ITEMS = 8
DEFAULT_MAX_ITEM_CHARS = 500
DEFAULT_MAX_SUMMARY_CHARS = 1200
DEFAULT_MAX_CONTENT_CHARS = 1800


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    return repr(value)


def _compact_text(value: Any, *, max_chars: int) -> str:
    text = value if isinstance(value, str) else json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 15]}...<truncated>"


def _parse_payload(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    stripped = raw.strip()
    if not stripped:
        return ""
    if stripped[0] not in "[{":
        return stripped
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return stripped


@dataclass(slots=True)
class MemoryRecallItem:
    """One normalized long-term memory hit returned by memory_recaller."""

    content: str
    source: str | None = None
    score: float | None = None
    scope: str | None = None
    memory_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Any, *, max_chars: int = DEFAULT_MAX_ITEM_CHARS) -> MemoryRecallItem:
        if not isinstance(payload, Mapping):
            return cls(content=_compact_text(payload, max_chars=max_chars))

        content = payload.get("content") or payload.get("summary") or payload.get("text") or payload.get("memory") or payload.get("value")
        if content is None:
            content = {key: value for key, value in payload.items() if key not in {"metadata"}}

        score = payload.get("score")
        try:
            score_value = float(score) if score is not None else None
        except (TypeError, ValueError):
            score_value = None

        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
        return cls(
            content=_compact_text(content, max_chars=max_chars),
            source=str(payload.get("source") or payload.get("source_ref") or "") or None,
            score=score_value,
            scope=str(payload.get("scope") or "") or None,
            memory_id=str(payload.get("memory_id") or payload.get("id") or "") or None,
            metadata=_json_safe(metadata) or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "source": self.source,
            "score": self.score,
            "scope": self.scope,
            "memory_id": self.memory_id,
            "metadata": _json_safe(self.metadata),
        }


@dataclass(slots=True)
class MemoryRecallResult:
    """Normalized read-only recall result stored back into TaskMemoryStack."""

    query: str
    summary: str
    items: list[MemoryRecallItem] = field(default_factory=list)
    source_task_id: str | None = None
    stop_reason: str | None = None
    dry_run_promotion: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_memory_content(self, *, max_chars: int = DEFAULT_MAX_CONTENT_CHARS) -> str:
        lines = [self.summary or f"Recalled {len(self.items)} long-term memory item(s)."]
        for item in self.items:
            prefix = "- "
            if item.source:
                prefix = f"- [{item.source}] "
            lines.append(f"{prefix}{item.content}")
        return _compact_text("\n".join(lines), max_chars=max_chars)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "summary": self.summary,
            "items": [item.to_dict() for item in self.items],
            "source_task_id": self.source_task_id,
            "stop_reason": self.stop_reason,
            "dry_run_promotion": self.dry_run_promotion,
            "metadata": _json_safe(self.metadata),
        }


def normalize_memory_recall_result(
    query: str,
    result: SPSubagentResult,
    *,
    max_items: int = DEFAULT_MAX_RECALL_ITEMS,
    max_item_chars: int = DEFAULT_MAX_ITEM_CHARS,
    max_summary_chars: int = DEFAULT_MAX_SUMMARY_CHARS,
) -> MemoryRecallResult:
    """Normalize memory_recaller output without writing long-term memory."""
    payload = _parse_payload(result.result)
    metadata: dict[str, Any] = {}
    summary = ""
    raw_items: list[Any] = []

    if isinstance(payload, Mapping):
        metadata = _json_safe(payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}) or {}
        for key in ("items", "memories", "results", "matches"):
            if isinstance(payload.get(key), list):
                raw_items = list(payload[key])
                break
        summary_value = payload.get("summary") or payload.get("answer")
        if isinstance(summary_value, str):
            summary = _compact_text(summary_value, max_chars=max_summary_chars)
        elif not raw_items and payload:
            raw_items = [payload]
    elif isinstance(payload, list):
        raw_items = payload
    elif payload:
        summary = _compact_text(payload, max_chars=max_summary_chars)

    items = [MemoryRecallItem.from_payload(item, max_chars=max_item_chars) for item in raw_items[:max_items]]
    if not summary:
        summary = f"Recalled {len(items)} relevant long-term memory item(s)." if items else "No relevant long-term memory found."

    if len(raw_items) > len(items):
        metadata = {**metadata, "truncated_item_count": len(raw_items) - len(items)}

    return MemoryRecallResult(
        query=query,
        summary=summary,
        items=items,
        source_task_id=result.task_id,
        stop_reason=result.stop_reason,
        dry_run_promotion=True,
        metadata=metadata,
    )
