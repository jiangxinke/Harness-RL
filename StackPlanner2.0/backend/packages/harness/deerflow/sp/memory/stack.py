"""Task memory stack for StackPlanner-on-DeerFlow."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from deerflow.sp.memory.entry import StackMemoryEntry, utc_now_iso

SP_TASK_MEMORY_VERSION = 1
DEFAULT_SUMMARIZE_TRIGGER_ENTRIES = 18
DEFAULT_SUMMARIZE_KEEP_RECENT = 6
DEFAULT_SUMMARIZE_SOURCE_LIMIT = 6
DEFAULT_EXPLICIT_SUMMARIZE_KEEP_RECENT = 4
DEFAULT_EXPLICIT_SUMMARIZE_SOURCE_LIMIT = 6
DEFAULT_SUMMARIZE_MIN_PROGRESS_ENTRIES = 4
DEFAULT_MIN_HISTORY_ENTRIES = 100
DEFAULT_HISTORY_SIZE_MULTIPLIER = 4
SUMMARIZATION_PROGRESS_ACTIONS = frozenset(
    {
        "delegate",
        "observe",
        "recall_memory",
        "reflect",
        "revise",
        "backtrack",
        "replan",
        "error",
    }
)


class TaskMemoryStack:
    """Explicit SP short-term memory stored as serializable thread state."""

    def __init__(
        self,
        entries: Iterable[StackMemoryEntry] | None = None,
        *,
        max_size: int = 50,
        max_history_size: int | None = None,
    ):
        self.max_size = max_size
        self.max_history_size = max(
            max_size,
            max_history_size if max_history_size is not None else max(DEFAULT_MIN_HISTORY_ENTRIES, max_size * DEFAULT_HISTORY_SIZE_MULTIPLIER),
        )
        self.entries: list[StackMemoryEntry] = list(entries or [])
        self._maintain_size()

    def append(self, entry: StackMemoryEntry) -> StackMemoryEntry:
        self.entries.append(entry)
        self._maintain_size()
        return entry

    def append_think(self, content: str, **kwargs: Any) -> StackMemoryEntry:
        return self.append(StackMemoryEntry(action="think", content=content, **kwargs))

    def append_delegate(self, content: str, *, actor: str = "central", **kwargs: Any) -> StackMemoryEntry:
        return self.append(StackMemoryEntry(action="delegate", actor=actor, content=content, **kwargs))

    def append_observe(self, content: str, *, actor: str, **kwargs: Any) -> StackMemoryEntry:
        return self.append(StackMemoryEntry(action="observe", actor=actor, content=content, **kwargs))

    def append_memory_recall(self, content: str, **kwargs: Any) -> StackMemoryEntry:
        return self.append(StackMemoryEntry(action="recall_memory", content=content, **kwargs))

    def append_reflect(self, content: str, **kwargs: Any) -> StackMemoryEntry:
        return self.append(StackMemoryEntry(action="reflect", content=content, **kwargs))

    def append_backtrack(self, content: str, **kwargs: Any) -> StackMemoryEntry:
        return self.append(StackMemoryEntry(action="backtrack", content=content, **kwargs))

    def append_replan(self, content: str, **kwargs: Any) -> StackMemoryEntry:
        return self.append(StackMemoryEntry(action="replan", content=content, **kwargs))

    def append_summary(self, content: str, *, parent_ids: list[str] | None = None, **kwargs: Any) -> StackMemoryEntry:
        return self.append(StackMemoryEntry(action="summarize", content=content, parent_ids=parent_ids or [], **kwargs))

    def append_feedback(self, content: str, *, actor: str = "human", **kwargs: Any) -> StackMemoryEntry:
        kwargs.pop("priority", None)
        kwargs.pop("status", None)
        return self.append(
            StackMemoryEntry(
                action="feedback",
                actor=actor,
                content=content,
                priority="critical",
                status="pinned",
                **kwargs,
            )
        )

    def append_finish(self, content: str, **kwargs: Any) -> StackMemoryEntry:
        kwargs.pop("stage", None)
        return self.append(StackMemoryEntry(action="finish", content=content, stage="finished", **kwargs))

    def get_active_entries(self) -> list[StackMemoryEntry]:
        return [entry for entry in self.entries if entry.status in {"active", "pinned"}]

    def get_pinned_entries(self) -> list[StackMemoryEntry]:
        return [entry for entry in self.entries if entry.status == "pinned" or entry.priority == "critical"]

    def get_recent_observations(self, count: int = 5) -> list[StackMemoryEntry]:
        observations = [entry for entry in self.get_active_entries() if entry.action == "observe"]
        return observations[-count:]

    def get_active_delegations(self) -> list[StackMemoryEntry]:
        return [entry for entry in self.get_active_entries() if entry.action == "delegate"]

    def select_summarization_source_ids(
        self,
        *,
        trigger_entries: int = DEFAULT_SUMMARIZE_TRIGGER_ENTRIES,
        keep_recent: int = DEFAULT_SUMMARIZE_KEEP_RECENT,
        source_limit: int = DEFAULT_SUMMARIZE_SOURCE_LIMIT,
    ) -> list[str]:
        """Select older non-critical active entries for automatic compaction.

        The selected window sits immediately before the newest entries so the
        CentralAgent has the source content in its bounded prompt context.
        """
        candidates = [entry for entry in self.entries if entry.status == "active" and entry.priority != "critical"]
        if len(candidates) < trigger_entries or source_limit <= 0:
            return []
        compactable = max(0, len(candidates) - max(keep_recent, 0))
        end = min(compactable, source_limit)
        return [entry.id for entry in candidates[:end]]

    def summarization_progress_since_last(self, *, current_run_id: str | None) -> int | None:
        """Count meaningful active entries added after this run's last summary.

        ``None`` means this run has not summarized yet. Free-form/explicit
        THINK, skipped actions, and summary entries do not count as progress,
        so a model cannot unlock another compaction through control churn.
        """
        if not current_run_id:
            return None
        latest_summary_index: int | None = None
        for index, entry in enumerate(self.entries):
            if entry.action == "summarize" and entry.run_id == current_run_id:
                latest_summary_index = index
        if latest_summary_index is None:
            return None
        return sum(1 for entry in self.entries[latest_summary_index + 1 :] if entry.status == "active" and entry.priority != "critical" and entry.action in SUMMARIZATION_PROGRESS_ACTIONS)

    def can_summarize(
        self,
        *,
        current_run_id: str | None,
        min_progress_entries: int = DEFAULT_SUMMARIZE_MIN_PROGRESS_ENTRIES,
    ) -> bool:
        """Allow repeated in-run compaction only after meaningful new progress."""
        progress = self.summarization_progress_since_last(current_run_id=current_run_id)
        return progress is None or progress >= max(1, min_progress_entries)

    def select_explicit_summarization_source_ids(
        self,
        *,
        current_run_id: str | None = None,
        keep_recent: int = DEFAULT_EXPLICIT_SUMMARIZE_KEEP_RECENT,
        source_limit: int = DEFAULT_EXPLICIT_SUMMARIZE_SOURCE_LIMIT,
    ) -> list[str]:
        """Select an older bounded window when SUMMARIZE omits source IDs.

        Explicit source IDs remain authoritative. This fallback never clears
        the whole working stack: it keeps the newest entries and removes at
        most source_limit older, non-critical entries.
        """
        # The stack is thread-scoped, not run-scoped.  Older entries from a
        # previous run are exactly what stage-boundary compaction must be able
        # to consume, so do not exclude them here.  ``current_run_id`` is kept
        # for API compatibility with callers from the first implementation.
        del current_run_id
        candidates = [entry.id for entry in self.entries if entry.status == "active" and entry.priority != "critical"]
        if source_limit <= 0 or len(candidates) <= max(keep_recent, 0):
            return []
        end = len(candidates) - max(keep_recent, 0)
        start = max(0, end - source_limit)
        return candidates[start:end]

    def get_checkpoint(self, entry_id_or_stage: str) -> StackMemoryEntry | None:
        for entry in reversed(self.entries):
            if entry.id == entry_id_or_stage or entry.stage == entry_id_or_stage:
                return entry
        return None

    def condense(self, source_entry_ids: list[str], summary: str, **kwargs: Any) -> StackMemoryEntry:
        """Pop source entries and append one summary entry.

        Summarization is a compaction operation, not just a status update. The
        selected entries leave the serialized stack so its size and the SP
        Memory view actually decrease. Pinned entries remain available, while
        the summary keeps the requested IDs as lineage metadata.
        """
        source_ids = list(dict.fromkeys(str(entry_id) for entry_id in source_entry_ids))
        by_id = {entry.id: entry for entry in self.entries}
        missing = [entry_id for entry_id in source_ids if entry_id not in by_id]
        if missing:
            raise ValueError(f"SUMMARIZE source memory entry not found: {', '.join(missing)}")
        inactive = [entry_id for entry_id in source_ids if by_id[entry_id].status not in {"active", "pinned"}]
        if inactive:
            raise ValueError(f"SUMMARIZE source memory is not active: {', '.join(inactive)}")

        # Preserve transitive lineage when an older summary is compacted again.
        # This keeps audit/tombstone semantics intact without retaining the old
        # summary body in active working memory.
        lineage_ids: list[str] = []
        for source_id in source_ids:
            source = by_id[source_id]
            lineage_ids.append(source_id)
            if source.action == "summarize":
                lineage_ids.extend(source.parent_ids)
        lineage_ids = list(dict.fromkeys(lineage_ids))
        source_id_set = set(source_ids)
        self.entries = [entry for entry in self.entries if entry.id not in source_id_set or entry.status == "pinned" or entry.priority == "critical"]
        return self.append_summary(summary, parent_ids=lineage_ids, **kwargs)

    def mark_backtracked(self, source_entry_ids: list[str], reason: str, **kwargs: Any) -> StackMemoryEntry:
        source_ids = set(source_entry_ids)
        for entry in self.entries:
            if entry.id in source_ids and entry.status != "pinned":
                entry.status = "pruned"
                entry.failure_note = reason
        return self.append_backtrack(reason, parent_ids=list(source_entry_ids), failure_note=reason, **kwargs)

    def revise(self, source_entry_ids: list[str], correction: str, reason: str, **kwargs: Any) -> StackMemoryEntry:
        """Logically pop incorrect entries and append their corrected replacement.

        The old entries remain in the checkpoint for auditability, but their
        ``superseded`` status keeps them out of the CentralAgent's active
        context. Critical or pinned human feedback is immutable.
        """
        source_ids = list(dict.fromkeys(str(entry_id) for entry_id in source_entry_ids))
        by_id = {entry.id: entry for entry in self.entries}
        missing = [entry_id for entry_id in source_ids if entry_id not in by_id]
        if missing:
            raise ValueError(f"REVISE target memory entry not found: {', '.join(missing)}")

        targets = [by_id[entry_id] for entry_id in source_ids]
        protected = [entry.id for entry in targets if entry.status == "pinned" or entry.priority == "critical"]
        if protected:
            raise ValueError(f"REVISE cannot invalidate pinned or critical memory: {', '.join(protected)}")
        human_authored = [entry.id for entry in targets if entry.actor == "human" or entry.action in {"user_request", "feedback"}]
        if human_authored:
            raise ValueError("REVISE cannot invalidate human-authored task memory: " + ", ".join(human_authored))
        inactive = [entry.id for entry in targets if entry.status != "active"]
        if inactive:
            raise ValueError(f"REVISE target memory is not active: {', '.join(inactive)}")

        for entry in targets:
            entry.status = "superseded"
            entry.failure_note = reason

        return self.append(
            StackMemoryEntry(
                action="revise",
                content=correction,
                parent_ids=source_ids,
                failure_note=reason,
                **kwargs,
            )
        )

    def prune(self, *, max_entries: int | None = None, max_chars: int | None = None) -> None:
        max_entries = max_entries if max_entries is not None else self.max_size

        def active_size() -> int:
            return len([entry for entry in self.entries if entry.status in {"active", "pinned"}])

        def active_chars() -> int:
            return sum(len(entry.content) for entry in self.entries if entry.status in {"active", "pinned"})

        for entry in self.entries:
            if entry.status == "pinned":
                continue
            if active_size() <= max_entries and (max_chars is None or active_chars() <= max_chars):
                break
            entry.status = "pruned"
            entry.failure_note = entry.failure_note or "Pruned from active SP working memory"

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": SP_TASK_MEMORY_VERSION,
            "max_size": self.max_size,
            "max_history_size": self.max_history_size,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, data: Any, *, thread_id: str | None = None, run_id: str | None = None, max_size: int = 50) -> TaskMemoryStack:
        if not data:
            return cls(max_size=max_size)
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return cls(max_size=max_size)
        if isinstance(data, list):
            return cls.from_legacy_memory_stack(data, thread_id=thread_id, run_id=run_id, max_size=max_size)
        if not isinstance(data, dict):
            return cls(max_size=max_size)

        entries_data = data.get("entries")
        if entries_data is None and "stack" in data:
            entries_data = data.get("stack")
        if entries_data is None and ("timestamp" in data or "agent_type" in data):
            entries_data = [data]
        if not isinstance(entries_data, list):
            return cls(
                max_size=int(data.get("max_size") or max_size),
                max_history_size=_optional_positive_int(data.get("max_history_size")),
            )

        entries = []
        for raw_entry in entries_data:
            if isinstance(raw_entry, dict):
                entry = StackMemoryEntry.from_dict(raw_entry)
                if thread_id and entry.thread_id is None:
                    entry.thread_id = thread_id
                if run_id and entry.run_id is None:
                    entry.run_id = run_id
                entries.append(entry)
        # Older SP2 builds marked summarized entries as ``condensed`` instead
        # of removing them. Normalize those checkpoints on restore so legacy
        # sessions receive the same compact-stack semantics as new sessions.
        entries = [entry for entry in entries if entry.status != "condensed"]
        return cls(
            entries,
            max_size=int(data.get("max_size") or max_size),
            max_history_size=_optional_positive_int(data.get("max_history_size")),
        )

    @classmethod
    def from_legacy_memory_stack(
        cls,
        data: Any,
        *,
        thread_id: str | None = None,
        run_id: str | None = None,
        max_size: int = 50,
    ) -> TaskMemoryStack:
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return cls(max_size=max_size)
        if not isinstance(data, list):
            return cls(max_size=max_size)

        entries = [StackMemoryEntry.from_legacy_dict(entry, thread_id=thread_id, run_id=run_id) for entry in data if isinstance(entry, dict)]
        return cls(entries, max_size=max_size)

    def size(self) -> int:
        return len(self.entries)

    def is_empty(self) -> bool:
        return not self.entries

    def _maintain_size(self) -> None:
        while len([entry for entry in self.entries if entry.status in {"active", "pinned"}]) > self.max_size:
            removable = next((entry for entry in self.entries if entry.status == "active"), None)
            if removable is None:
                break
            removable.status = "pruned"
            removable.failure_note = removable.failure_note or f"Pruned at {utc_now_iso()} because stack exceeded max_size"
        # Backtracks and revisions intentionally retain inactive entries for
        # auditability, but an unbounded audit tail eventually dominates every
        # checkpoint. Keep a generous bounded tail; lineage IDs remain on
        # summaries/revisions even when an old inactive body ages out.
        while len(self.entries) > self.max_history_size:
            removable_index = next(
                (index for index, entry in enumerate(self.entries) if entry.status not in {"active", "pinned"} and entry.priority != "critical"),
                None,
            )
            if removable_index is None:
                break
            self.entries.pop(removable_index)


def _optional_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
