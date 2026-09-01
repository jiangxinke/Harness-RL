"""SP artifact metadata carried as lightweight ThreadState refs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    return repr(value)


@dataclass(slots=True)
class SPArtifactMetadata:
    """Metadata for a SP mid-term artifact stored in DR2 outputs."""

    artifact_id: str = field(default_factory=lambda: f"spart_{uuid4().hex}")
    type: str = "generated_file"
    version: int = 1
    created_by: str = "system"
    created_at: str = field(default_factory=utc_now_iso)
    thread_id: str | None = None
    run_id: str | None = None
    stage: str | None = None
    source_entry_id: str | None = None
    parent_artifact_ids: list[str] = field(default_factory=list)
    is_current: bool = True
    summary: str = ""
    workspace_path: str = ""
    virtual_path: str = ""
    artifact_url: str | None = None
    content_hash: str = ""
    feedback_entry_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.artifact_id = str(self.artifact_id or f"spart_{uuid4().hex}")
        self.type = str(self.type or "generated_file")
        self.version = int(self.version or 1)
        self.created_by = str(self.created_by or "system")
        self.created_at = str(self.created_at or utc_now_iso())
        self.parent_artifact_ids = [str(item) for item in self.parent_artifact_ids]
        self.feedback_entry_ids = [str(item) for item in self.feedback_entry_ids]
        self.summary = str(self.summary or "")
        self.workspace_path = str(self.workspace_path or "")
        self.virtual_path = str(self.virtual_path or "")
        self.content_hash = str(self.content_hash or "")
        self.metadata = _json_safe(self.metadata) or {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SPArtifactMetadata:
        return cls(
            artifact_id=str(data.get("artifact_id") or data.get("id") or f"spart_{uuid4().hex}"),
            type=str(data.get("type") or "generated_file"),
            version=int(data.get("version") or 1),
            created_by=str(data.get("created_by") or "system"),
            created_at=str(data.get("created_at") or utc_now_iso()),
            thread_id=data.get("thread_id"),
            run_id=data.get("run_id"),
            stage=data.get("stage"),
            source_entry_id=data.get("source_entry_id"),
            parent_artifact_ids=list(data.get("parent_artifact_ids") or []),
            is_current=bool(data.get("is_current", True)),
            summary=str(data.get("summary") or ""),
            workspace_path=str(data.get("workspace_path") or ""),
            virtual_path=str(data.get("virtual_path") or ""),
            artifact_url=data.get("artifact_url"),
            content_hash=str(data.get("content_hash") or ""),
            feedback_entry_ids=list(data.get("feedback_entry_ids") or []),
            metadata=_json_safe(data.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "type": self.type,
            "version": self.version,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "thread_id": self.thread_id,
            "run_id": self.run_id,
            "stage": self.stage,
            "source_entry_id": self.source_entry_id,
            "parent_artifact_ids": list(self.parent_artifact_ids),
            "is_current": self.is_current,
            "summary": self.summary,
            "workspace_path": self.workspace_path,
            "virtual_path": self.virtual_path,
            "artifact_url": self.artifact_url,
            "content_hash": self.content_hash,
            "feedback_entry_ids": list(self.feedback_entry_ids),
            "metadata": _json_safe(self.metadata),
        }

    def to_ref(self) -> dict[str, Any]:
        """Return the lightweight ref intended for ThreadState."""
        return {
            "artifact_id": self.artifact_id,
            "type": self.type,
            "version": self.version,
            "thread_id": self.thread_id,
            "run_id": self.run_id,
            "stage": self.stage,
            "source_entry_id": self.source_entry_id,
            "parent_artifact_ids": list(self.parent_artifact_ids),
            "is_current": self.is_current,
            "summary": self.summary,
            "virtual_path": self.virtual_path,
            "artifact_url": self.artifact_url,
            "content_hash": self.content_hash,
            "feedback_entry_ids": list(self.feedback_entry_ids),
            "metadata": _json_safe(self.metadata),
        }
