"""Human-in-the-loop state helpers for SP orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from deerflow.sp.actions.events import make_sp_event
from deerflow.sp.artifacts import SPArtifactAdapter
from deerflow.sp.memory import StackMemoryEntry, TaskMemoryStack
from deerflow.sp.memory.entry import utc_now_iso


@dataclass(slots=True)
class SPHumanInteraction:
    interaction_id: str = field(default_factory=lambda: f"sphitl_{uuid4().hex}")
    interaction_type: str = "clarification"
    question: str = ""
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    created_by_entry_id: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    resume_token: str | None = None
    status: str = "pending"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.interaction_id = str(self.interaction_id or f"sphitl_{uuid4().hex}")
        if self.resume_token is None:
            self.resume_token = f"resume_{self.interaction_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "interaction_id": self.interaction_id,
            "interaction_type": self.interaction_type,
            "question": self.question,
            "artifact_refs": self.artifact_refs,
            "created_by_entry_id": self.created_by_entry_id,
            "created_at": self.created_at,
            "resume_token": self.resume_token,
            "status": self.status,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class SPHumanFeedbackResult:
    state_update: dict[str, Any]
    memory_entry: StackMemoryEntry
    run_events: list[dict[str, Any]]


def record_human_feedback(
    state: dict[str, Any],
    feedback: str,
    *,
    thread_id: str | None = None,
    run_id: str | None = None,
    artifact_adapter: SPArtifactAdapter | None = None,
) -> SPHumanFeedbackResult:
    """Resolve pending HITL feedback into pinned critical task memory."""
    pending = state.get("sp_pending_human_interaction") if isinstance(state.get("sp_pending_human_interaction"), dict) else {}
    stack = TaskMemoryStack.from_dict(state.get("sp_task_memory"), thread_id=thread_id, run_id=run_id)
    entry = stack.append_feedback(
        feedback,
        thread_id=thread_id,
        run_id=run_id,
        stage=state.get("sp_current_stage"),
        metadata={
            "interaction_id": pending.get("interaction_id"),
            "interaction_type": pending.get("interaction_type"),
            "artifact_refs": pending.get("artifact_refs") or {},
        },
    )
    state_update: dict[str, Any] = {
        "sp_task_memory": stack.to_dict(),
        "sp_pending_human_interaction": None,
    }
    refs = state.get("sp_current_artifact_refs")
    if isinstance(refs, dict):
        adapter = artifact_adapter or SPArtifactAdapter()
        artifact_ids = _artifact_ids_from_pending(pending)
        if artifact_ids:
            state_update["sp_current_artifact_refs"] = adapter.bind_feedback(refs, entry.id, artifact_ids=artifact_ids)

    return SPHumanFeedbackResult(
        state_update=state_update,
        memory_entry=entry,
        run_events=[
            make_sp_event(
                "sp.human.feedback_received",
                run_id=run_id,
                interaction_id=pending.get("interaction_id"),
                feedback_entry_id=entry.id,
            )
        ],
    )


def _artifact_ids_from_pending(pending: dict[str, Any]) -> list[str]:
    artifact_refs = pending.get("artifact_refs")
    if not isinstance(artifact_refs, dict):
        return []
    ids: list[str] = []
    for value in artifact_refs.values():
        if isinstance(value, dict) and value.get("artifact_id"):
            ids.append(str(value["artifact_id"]))
    return ids
