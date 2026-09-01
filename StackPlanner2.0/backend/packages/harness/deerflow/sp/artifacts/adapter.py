"""SP artifact adapter backed by DeerFlow thread outputs."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from deerflow.agents.thread_state import ThreadState
from deerflow.config.paths import VIRTUAL_PATH_PREFIX, get_paths
from deerflow.sp.artifacts.metadata import SPArtifactMetadata

OUTPUTS_VIRTUAL_PREFIX = f"{VIRTUAL_PATH_PREFIX}/outputs"
DEFAULT_SUMMARY_MAX_CHARS = 320
DEFAULT_MATERIALIZED_TEXT_MAX_CHARS = 24_000
DEFAULT_FINALIZATION_PREVIEW_MAX_CHARS = 4_000
_FINALIZATION_PREVIEW_SUFFIXES = frozenset({".csv", ".json", ".log", ".tsv", ".txt"})
_BINARY_ARTIFACT_SUFFIXES = frozenset(
    {
        ".7z",
        ".avi",
        ".doc",
        ".docx",
        ".gif",
        ".gz",
        ".ico",
        ".jpeg",
        ".jpg",
        ".mov",
        ".mp3",
        ".mp4",
        ".pdf",
        ".png",
        ".ppt",
        ".pptx",
        ".tar",
        ".webm",
        ".webp",
        ".xls",
        ".xlsx",
        ".zip",
    }
)
LEGACY_FIELD_TYPES = {
    "report_outline": "outline",
    "observations": "research_observation",
    "data_collections": "data_collection",
    "original_report": "report",
    "final_report": "report_revision",
}


@dataclass(slots=True)
class SPArtifactWriteResult:
    """Result from writing SP mid-term memory into DR2 outputs."""

    metadata: SPArtifactMetadata
    state_update: dict[str, Any] = field(default_factory=dict)


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip().lower()).strip("-")
    return slug or "artifact"


def _summarize(content: str, *, max_chars: int = DEFAULT_SUMMARY_MAX_CHARS) -> str:
    text = " ".join(content.split())
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 15]}...<truncated>"


def _serialize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, indent=2, sort_keys=True)


def _extension_for_artifact(artifact_type: str, content: Any) -> str:
    if not isinstance(content, str):
        return ".json"
    if artifact_type in {
        "outline",
        "report",
        "report_revision",
        "evidence_bundle",
        "research_observation",
        "perception_observation",
        "data_collection",
    }:
        return ".md"
    return ".txt"


def _artifact_url(thread_id: str | None, virtual_path: str) -> str | None:
    if not thread_id:
        return None
    return f"/api/threads/{thread_id}/artifacts{quote(virtual_path, safe='/')}"


class SPArtifactAdapter:
    """Persist SP mid-term memory as DR2 artifact files and lightweight refs."""

    def write_text_artifact(
        self,
        content: Any,
        *,
        artifact_type: str,
        state: ThreadState | dict[str, Any],
        thread_id: str | None = None,
        run_id: str | None = None,
        created_by: str = "system",
        stage: str | None = None,
        source_entry_id: str | None = None,
        parent_artifact_ids: list[str] | None = None,
        feedback_entry_ids: list[str] | None = None,
        version: int | None = None,
        filename_hint: str | None = None,
        summary: str | None = None,
        metadata: dict[str, Any] | None = None,
        virtual_path_hint: str | None = None,
    ) -> SPArtifactWriteResult:
        """Write large SP content under DR2 outputs and return ThreadState refs."""
        payload = _serialize_content(content)
        outputs_path = self._outputs_path(state, thread_id=thread_id)
        outputs_path.mkdir(parents=True, exist_ok=True)

        effective_version = version or self._next_version(state, artifact_type)
        identity_payload = json.dumps(
            {
                "artifact_type": artifact_type,
                "content_hash": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                "parent_artifact_ids": parent_artifact_ids or [],
                "version": effective_version,
            },
            sort_keys=True,
        )
        artifact_id = f"spart_{hashlib.sha256(identity_payload.encode('utf-8')).hexdigest()[:16]}"
        extension = _extension_for_artifact(artifact_type, content)
        if virtual_path_hint is not None:
            normalized_hint = _normalize_output_virtual_path(virtual_path_hint)
            relative_path = Path(normalized_hint.removeprefix(f"{OUTPUTS_VIRTUAL_PREFIX}/"))
            name = relative_path.name
        else:
            name = filename_hint or f"{_safe_slug(artifact_type)}-v{effective_version}-{artifact_id}{extension}"
            safe_name = _safe_slug(name)
            relative_path = Path("sp") / _safe_slug(artifact_type) / safe_name
            candidate = outputs_path / relative_path
            if filename_hint and candidate.exists():
                suffix = Path(safe_name).suffix
                stem = Path(safe_name).stem
                unique_name = f"{stem}-v{effective_version}-{artifact_id[-8:]}{suffix}"
                relative_path = Path("sp") / _safe_slug(artifact_type) / unique_name
        file_path = outputs_path / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(payload, encoding="utf-8")

        virtual_path = f"{OUTPUTS_VIRTUAL_PREFIX}/{relative_path.as_posix()}"
        content_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        artifact_metadata = SPArtifactMetadata(
            artifact_id=artifact_id,
            type=artifact_type,
            version=effective_version,
            created_by=created_by,
            thread_id=thread_id,
            run_id=run_id,
            stage=stage,
            source_entry_id=source_entry_id,
            parent_artifact_ids=parent_artifact_ids or [],
            summary=_summarize(summary or payload),
            workspace_path=str(file_path),
            virtual_path=virtual_path,
            artifact_url=_artifact_url(thread_id, virtual_path),
            content_hash=content_hash,
            feedback_entry_ids=feedback_entry_ids or [],
            metadata=metadata or {},
        )
        refs = self._merge_ref(state.get("sp_current_artifact_refs"), artifact_metadata)
        return SPArtifactWriteResult(
            metadata=artifact_metadata,
            state_update={
                "artifacts": [virtual_path],
                "sp_current_artifact_refs": refs,
                **({"sp_current_report_version": str(effective_version)} if artifact_type in {"report", "report_revision"} else {}),
            },
        )

    def migrate_legacy_midterm_state(
        self,
        legacy_state: dict[str, Any],
        *,
        state: ThreadState | dict[str, Any],
        thread_id: str | None = None,
        run_id: str | None = None,
        created_by: str = "legacy_migration",
    ) -> dict[str, Any]:
        """Convert legacy SP mid-term fields into artifact refs.

        The returned update intentionally contains only refs and artifact paths,
        never the original report/research body fields.
        """
        combined_update: dict[str, Any] = {}
        for field_name, artifact_type in LEGACY_FIELD_TYPES.items():
            content = legacy_state.get(field_name)
            if content in (None, "", [], {}):
                continue
            result = self.write_text_artifact(
                content,
                artifact_type=artifact_type,
                state={**state, **combined_update},
                thread_id=thread_id,
                run_id=run_id,
                created_by=created_by,
                stage=_stage_for_legacy_field(field_name),
                metadata={"legacy_field": field_name},
            )
            combined_update = _merge_state_updates(combined_update, result.state_update)
        return combined_update

    def register_existing_artifact(
        self,
        virtual_path: str,
        *,
        artifact_type: str,
        state: ThreadState | dict[str, Any],
        thread_id: str | None = None,
        run_id: str | None = None,
        created_by: str = "system",
        stage: str | None = None,
        source_entry_id: str | None = None,
        parent_artifact_ids: list[str] | None = None,
        feedback_entry_ids: list[str] | None = None,
        summary: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SPArtifactWriteResult:
        """Register a subagent-created DR2 workspace/output file without copying its body."""
        normalized_path = _normalize_existing_virtual_path(virtual_path)
        self._require_existing_file(normalized_path, state=state, thread_id=thread_id)
        artifact_metadata_values = {"registered_existing_file": True, **(metadata or {})}
        if PurePosixPath(normalized_path).suffix.lower() in _FINALIZATION_PREVIEW_SUFFIXES:
            materialized = self.read_text_artifact(
                {"virtual_path": normalized_path},
                state=state,
                thread_id=thread_id,
                max_chars=DEFAULT_FINALIZATION_PREVIEW_MAX_CHARS,
            )
            if materialized is not None:
                preview, truncated = materialized
                artifact_metadata_values.update(
                    {
                        "finalization_preview": preview,
                        "finalization_preview_truncated": truncated,
                    }
                )
        effective_version = self._next_version(state, artifact_type)
        identity_payload = json.dumps(
            {
                "artifact_type": artifact_type,
                "virtual_path": normalized_path,
                "version": effective_version,
            },
            sort_keys=True,
        )
        artifact_metadata = SPArtifactMetadata(
            artifact_id=f"spart_{hashlib.sha256(identity_payload.encode('utf-8')).hexdigest()[:16]}",
            type=artifact_type,
            version=effective_version,
            created_by=created_by,
            thread_id=thread_id,
            run_id=run_id,
            stage=stage,
            source_entry_id=source_entry_id,
            parent_artifact_ids=parent_artifact_ids or [],
            summary=_summarize(summary or f"Generated artifact: {normalized_path}"),
            virtual_path=normalized_path,
            artifact_url=_artifact_url(thread_id, normalized_path) if normalized_path.startswith(f"{OUTPUTS_VIRTUAL_PREFIX}/") else None,
            content_hash="",
            feedback_entry_ids=feedback_entry_ids or [],
            metadata=artifact_metadata_values,
        )
        refs = self._merge_ref(state.get("sp_current_artifact_refs"), artifact_metadata)
        state_update: dict[str, Any] = {"sp_current_artifact_refs": refs}
        if normalized_path.startswith(f"{OUTPUTS_VIRTUAL_PREFIX}/"):
            state_update["artifacts"] = [normalized_path]
        if artifact_type in {"report", "report_revision", "final_report"}:
            state_update["sp_current_report_version"] = str(effective_version)
        return SPArtifactWriteResult(metadata=artifact_metadata, state_update=state_update)

    def read_text_artifact(
        self,
        ref: Mapping[str, Any],
        *,
        state: ThreadState | dict[str, Any],
        thread_id: str | None = None,
        max_chars: int = DEFAULT_MATERIALIZED_TEXT_MAX_CHARS,
    ) -> tuple[str, bool] | None:
        """Materialize one thread-scoped text artifact for a specialist.

        ThreadState intentionally stores only lightweight refs.  Synthesis
        specialists nevertheless need the referenced evidence body, so this
        method resolves only a ref's virtual path inside the active thread and
        returns bounded UTF-8 text.  For oversized files both the beginning and
        end are retained because source lists and limitations commonly live at
        the end of research artifacts.
        """
        virtual_path = ref.get("virtual_path")
        if not isinstance(virtual_path, str) or not virtual_path.startswith((f"{VIRTUAL_PATH_PREFIX}/outputs/", f"{VIRTUAL_PATH_PREFIX}/workspace/")):
            return None
        if PurePosixPath(virtual_path).suffix.lower() in _BINARY_ARTIFACT_SUFFIXES:
            return None
        try:
            file_path = self._require_existing_file(virtual_path, state=state, thread_id=thread_id)
        except (OSError, ValueError):
            return None

        limit = max(int(max_chars), 128)
        marker = "\n\n...<artifact-content-truncated; head and tail preserved>...\n\n"
        try:
            size = file_path.stat().st_size
            if size <= limit * 4:
                raw = file_path.read_bytes()
                if b"\x00" in raw:
                    return None
                text = raw.decode("utf-8")
                if len(text) <= limit:
                    return text, False
                return _bound_text_head_tail(text, max_chars=limit, marker=marker), True

            available = limit - len(marker)
            head_chars = max(int(available * 0.7), 1)
            tail_chars = max(available - head_chars, 1)
            with file_path.open("rb") as artifact_file:
                head_raw = artifact_file.read(head_chars * 4 + 4)
                artifact_file.seek(max(size - (tail_chars * 4 + 4), 0))
                tail_raw = artifact_file.read(tail_chars * 4 + 4)
            if b"\x00" in head_raw or b"\x00" in tail_raw:
                return None
            head = head_raw.decode("utf-8", errors="ignore")[:head_chars]
            tail = tail_raw.decode("utf-8", errors="ignore")[-tail_chars:]
            return f"{head}{marker}{tail}"[:limit], True
        except (OSError, UnicodeDecodeError):
            return None

    def _require_existing_file(
        self,
        virtual_path: str,
        *,
        state: ThreadState | dict[str, Any],
        thread_id: str | None,
    ) -> Path:
        """Resolve a registered path and reject stale artifact references."""
        thread_data = state.get("thread_data") or {}
        if virtual_path.startswith(f"{VIRTUAL_PATH_PREFIX}/outputs/"):
            root_value = thread_data.get("outputs_path")
            root = Path(str(root_value)).expanduser() if root_value else get_paths().sandbox_outputs_dir(thread_id or "")
            prefix = f"{VIRTUAL_PATH_PREFIX}/outputs/"
        else:
            root_value = thread_data.get("workspace_path")
            root = Path(str(root_value)).expanduser() if root_value else get_paths().sandbox_work_dir(thread_id or "")
            prefix = f"{VIRTUAL_PATH_PREFIX}/workspace/"

        root = root.resolve()
        relative = PurePosixPath(virtual_path[len(prefix) :])
        actual = (root / Path(*relative.parts)).resolve()
        try:
            actual.relative_to(root)
        except ValueError as exc:
            raise ValueError("SP existing artifact path escapes its thread directory") from exc
        if not actual.is_file():
            raise ValueError(f"SP existing artifact file does not exist: {virtual_path}")
        return actual

    def _outputs_path(self, state: ThreadState | dict[str, Any], *, thread_id: str | None) -> Path:
        thread_data = state.get("thread_data") or {}
        outputs_path = thread_data.get("outputs_path")
        if outputs_path:
            return Path(str(outputs_path)).expanduser().resolve()
        if not thread_id:
            raise ValueError("SPArtifactAdapter requires thread_data.outputs_path or thread_id")
        return get_paths().sandbox_outputs_dir(thread_id)

    def _next_version(self, state: ThreadState | dict[str, Any], artifact_type: str) -> int:
        refs = state.get("sp_current_artifact_refs") or {}
        version_types = _version_family(artifact_type)
        current_versions = [value.get("version") for key, value in refs.items() if key in version_types and isinstance(value, dict) and isinstance(value.get("version"), int)]
        history = refs.get("_history")
        history_versions: list[int] = []
        if isinstance(history, list):
            history_versions = [item["version"] for item in history if isinstance(item, dict) and item.get("type") in version_types and isinstance(item.get("version"), int)]
        versions = [*current_versions, *history_versions]
        return max(versions, default=0) + 1

    def _merge_ref(self, existing: Any, artifact_metadata: SPArtifactMetadata) -> dict[str, Any]:
        refs = dict(existing) if isinstance(existing, dict) else {}
        ref = artifact_metadata.to_ref()
        version_family = _version_family(artifact_metadata.type)
        for key, value in list(refs.items()):
            if key == "_history" or key not in version_family or not isinstance(value, dict):
                continue
            refs[key] = {**value, "is_current": False}
        refs[artifact_metadata.type] = ref
        history = [{**item, "is_current": False} if item.get("type") in version_family else item for item in refs.get("_history", []) if isinstance(item, dict)]
        matching_index = next(
            (index for index, item in enumerate(history) if item.get("artifact_id") == ref["artifact_id"] and item.get("version") == ref["version"]),
            None,
        )
        if matching_index is None:
            history.append(ref)
        else:
            history[matching_index] = ref
        refs["_history"] = history[-50:]
        return refs

    def bind_feedback(self, artifact_refs: dict[str, Any], feedback_entry_id: str, *, artifact_ids: list[str] | None = None) -> dict[str, Any]:
        """Bind a human feedback entry id to matching artifact refs."""
        target_ids = set(artifact_ids or [])
        refs = dict(artifact_refs)
        for key, value in list(refs.items()):
            if key == "_history" or not isinstance(value, dict):
                continue
            if target_ids and value.get("artifact_id") not in target_ids:
                continue
            refs[key] = _with_feedback_id(value, feedback_entry_id)
        history = refs.get("_history")
        if isinstance(history, list):
            refs["_history"] = [_with_feedback_id(item, feedback_entry_id) if isinstance(item, dict) and (not target_ids or item.get("artifact_id") in target_ids) else item for item in history]
        return refs


def _stage_for_legacy_field(field_name: str) -> str | None:
    return {
        "report_outline": "planning",
        "observations": "research",
        "data_collections": "research",
        "original_report": "reporting",
        "final_report": "revision",
    }.get(field_name)


def _version_family(artifact_type: str) -> set[str]:
    if artifact_type in {"report", "report_revision", "final_report"}:
        return {"report", "report_revision", "final_report"}
    return {artifact_type}


def _normalize_existing_virtual_path(value: str) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    allowed_prefixes = ("/mnt/user-data/outputs/", "/mnt/user-data/workspace/")
    if not raw.startswith(allowed_prefixes):
        raise ValueError("SP existing artifact path must be under /mnt/user-data/outputs or /mnt/user-data/workspace")
    path = PurePosixPath(raw)
    if ".." in path.parts:
        raise ValueError("SP existing artifact path cannot contain '..'")
    return path.as_posix()


def _normalize_output_virtual_path(value: str) -> str:
    normalized = _normalize_existing_virtual_path(value)
    if not normalized.startswith(f"{OUTPUTS_VIRTUAL_PREFIX}/"):
        raise ValueError("SP artifact content must be written under /mnt/user-data/outputs")
    return normalized


def _bound_text_head_tail(text: str, *, max_chars: int, marker: str) -> str:
    if len(text) <= max_chars:
        return text
    available = max(max_chars - len(marker), 2)
    head_chars = max(int(available * 0.7), 1)
    tail_chars = max(available - head_chars, 1)
    return f"{text[:head_chars]}{marker}{text[-tail_chars:]}"[:max_chars]


def _merge_state_updates(existing: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    if "artifacts" in new:
        merged["artifacts"] = list(dict.fromkeys([*merged.get("artifacts", []), *new["artifacts"]]))
    if "sp_current_artifact_refs" in new:
        merged["sp_current_artifact_refs"] = new["sp_current_artifact_refs"]
    if "sp_current_report_version" in new:
        merged["sp_current_report_version"] = new["sp_current_report_version"]
    return merged


def _with_feedback_id(ref: dict[str, Any], feedback_entry_id: str) -> dict[str, Any]:
    updated = dict(ref)
    feedback_ids = [str(item) for item in updated.get("feedback_entry_ids", [])]
    if feedback_entry_id not in feedback_ids:
        feedback_ids.append(feedback_entry_id)
    updated["feedback_entry_ids"] = feedback_ids
    return updated
