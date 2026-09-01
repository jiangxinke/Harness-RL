"""Bounded prompt context for StackPlanner task memory.

The builder is intentionally pure: it reads already-restored SP task memory and
lightweight ThreadState refs, then renders a compact control context for the
CentralAgent. It never reads files, artifacts, tools, or long-term memory.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from deerflow.sp.memory import (
    StackMemoryEntry,
    TaskMemoryStack,
    select_correction_review_candidates,
)
from deerflow.sp.memory.stack import DEFAULT_SUMMARIZE_MIN_PROGRESS_ENTRIES

DEFAULT_CONTEXT_MAX_CHARS = 6000
DEFAULT_ENTRY_CONTENT_MAX_CHARS = 700
DEFAULT_RECENT_ENTRY_LIMIT = 12
FULL_HEADER_MIN_CHARS = 2600
MIN_SECTION_ENTRY_CHARS = 80
_AUTHORITATIVE_CONSTRAINT_PATTERN = re.compile(
    r"(?:"
    r"永久|不可牺牲|必须|不得|禁止|约束|要求|风险|作废|最晚|口径"
    r"|must\b|shall\b|required\b|never\b|constraint\b|supersed"
    r")",
    re.IGNORECASE,
)
_HARD_CONSTRAINT_PATTERN = re.compile(
    r"(?:"
    r"永久约束|不可牺牲|不是可牺牲|非可牺牲|硬性约束"
    r"|permanent\s+constraint|non[-\s]?sacrificable|hard\s+constraint"
    r")",
    re.IGNORECASE,
)


def _compact(value: Any, *, max_chars: int) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            text = repr(value)
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 15]}...<truncated>"


def _json_ref(value: Any, *, max_chars: int) -> str:
    if not value:
        return "{}"
    return _compact(value, max_chars=max_chars)


def _entry_priority_rank(entry: StackMemoryEntry) -> int:
    priority_rank = {"critical": 0, "high": 1, "normal": 2, "low": 3}.get(entry.priority, 2)
    return priority_rank


def _workflow_status(
    stack: TaskMemoryStack,
    artifact_refs: Mapping[str, Any] | None,
    *,
    current_run_id: str | None,
) -> dict[str, int]:
    """Expose compact stage evidence without asking Central to infer it from refs."""
    raw_refs: list[Mapping[str, Any]] = []
    if isinstance(artifact_refs, Mapping):
        history = artifact_refs.get("_history")
        if isinstance(history, list):
            raw_refs.extend(item for item in history if isinstance(item, Mapping))
        raw_refs.extend(item for key, item in artifact_refs.items() if key != "_history" and isinstance(item, Mapping))

    deduplicated: dict[str, Mapping[str, Any]] = {}
    for index, ref in enumerate(raw_refs):
        ref_run_id = str(ref.get("run_id")) if ref.get("run_id") else None
        if current_run_id and ref_run_id not in {None, current_run_id}:
            continue
        identity = str(ref.get("artifact_id") or ref.get("virtual_path") or f"{ref.get('type', 'unknown')}:{ref.get('version', index)}")
        deduplicated[identity] = ref

    counts = Counter(str(ref.get("type") or "unknown") for ref in deduplicated.values())
    status: dict[str, int] = {
        "pinned_feedback": sum(1 for entry in stack.entries if entry.action == "feedback" and entry.status in {"active", "pinned"}),
        "partial_or_blocked": sum(1 for entry in stack.get_active_entries() if str(entry.metadata.get("completion_status") or "") in {"partial", "blocked"}),
    }
    status.update({key: counts[key] for key in sorted(counts)})
    return status


@dataclass(slots=True)
class PromptContextBuilder:
    """Render SP short-term memory into a bounded CentralAgent context block."""

    max_chars: int = DEFAULT_CONTEXT_MAX_CHARS
    max_entry_chars: int = DEFAULT_ENTRY_CONTENT_MAX_CHARS
    recent_entry_limit: int = DEFAULT_RECENT_ENTRY_LIMIT

    def build(
        self,
        stack: TaskMemoryStack,
        *,
        current_stage: str | None = None,
        active_delegate_id: str | None = None,
        pending_human_interaction: Mapping[str, Any] | None = None,
        artifact_refs: Mapping[str, Any] | None = None,
        report_version: str | None = None,
        current_run_id: str | None = None,
    ) -> str:
        """Build the bounded SP context used by CentralAgent decisions."""
        pending_ref_chars = max(120, min(900, self.max_chars // 6))
        artifact_ref_chars = max(140, min(1200, self.max_chars // 5))
        compact_header = [
            "<sp-task-context>",
            "CentralAgent control context backed by short-term task memory. Pinned feedback is authoritative.",
            f"current_stage: {current_stage or 'unknown'}",
            f"current_run_id: {current_run_id or 'unknown'}",
            f"pending_human_interaction: {_json_ref(pending_human_interaction, max_chars=pending_ref_chars)}",
        ]
        full_header = [
            *compact_header,
            "",
            "priority_rules:",
            "- Critical or pinned human feedback outranks summaries, observations, and model plans.",
            "- Artifact refs point to Workspace/Artifact content; do not infer large artifact bodies from this block.",
            "- Memory order: critical_feedback and recent_task_memory first; long-term recall is only a fallback for historical reusable facts.",
            "- Do not call sp_recall_memory when this context already contains the answer to the current task.",
        ]
        header = full_header if self.max_chars >= FULL_HEADER_MIN_CHARS else compact_header

        pinned = self._select_pinned(stack)
        authoritative_constraints = self._select_authoritative_constraints(
            stack,
            exclude_ids={entry.id for entry in pinned},
        )
        authoritative_constraint_ids = {entry.id for entry in authoritative_constraints}
        # Automatic pressure is intentionally separate from the explicit
        # SUMMARIZE fallback. The latter may compact a small stack when Central
        # deliberately chooses a stage boundary, but it must not advertise
        # summarization on nearly every turn.
        pressure_ids = stack.select_summarization_source_ids()
        summary_progress = stack.summarization_progress_since_last(current_run_id=current_run_id)
        summarize_ready = stack.can_summarize(current_run_id=current_run_id)
        summarize_ids = pressure_ids if summarize_ready else []
        summarize_id_set = set(summarize_ids)
        summarize_entries = [entry for entry in stack.entries if entry.id in summarize_id_set]
        correction_candidates = select_correction_review_candidates(
            stack,
            current_run_id=current_run_id,
        )
        recent = self._select_recent(
            stack,
            exclude_ids={entry.id for entry in pinned} | authoritative_constraint_ids | summarize_id_set,
            current_run_id=current_run_id,
            correction_candidates=correction_candidates,
        )

        if summary_progress is None:
            cooldown = "ready (no summary committed in this run)"
        elif summarize_ready:
            cooldown = f"ready ({summary_progress} new entries since last summary)"
        else:
            cooldown = f"active ({summary_progress}/{DEFAULT_SUMMARIZE_MIN_PROGRESS_ENTRIES} new entries required)"
        footer = [""]
        if self.max_chars >= 1400:
            footer.append(f"workflow_status: {_json_ref(_workflow_status(stack, artifact_refs, current_run_id=current_run_id), max_chars=700)}")
        footer.extend(
            [
                "memory_control:",
                "memory_scope: short_term_task_memory",
                f"summarization_pressure: {str(bool(pressure_ids)).lower()}",
                f"summarization_needed: {str(bool(summarize_entries)).lower()}",
                f"summarization_cooldown: {cooldown}",
            ]
        )
        if correction_candidates:
            footer.append("correction_review_required: true")
            footer.append("correction_candidate_ids: " + ", ".join(f"id={entry.id}" for entry in correction_candidates))
        if summarize_ids:
            footer.append("summarization_candidate_ids: " + ", ".join(f"id={entry_id}" for entry_id in summarize_ids))
        # These execution refs are control-critical. Keep them in the reserved
        # footer so feedback or verbose observations can never clip them away.
        footer.extend(
            [
                f"active_delegate_id: {active_delegate_id or 'none'}",
                f"current_artifact_refs: {_json_ref(artifact_refs, max_chars=artifact_ref_chars)}",
                f"current_report_version: {report_version or 'none'}",
                "</sp-task-context>",
            ]
        )

        sections = [
            # _select_pinned already returns newest-first inside each priority
            # band; recent memory remains chronological and is reversed here.
            ("critical_feedback:", pinned, 3, False),
            (
                "authoritative_constraints:",
                authoritative_constraints,
                4,
                True,
            ),
            ("recent_task_memory:", recent, 4, True),
            (
                "summarization_candidates:",
                [entry for entry in summarize_entries if entry.id not in authoritative_constraint_ids],
                3,
                False,
            ),
        ]
        return self._compose_context(header, sections, footer)

    def _select_pinned(self, stack: TaskMemoryStack) -> list[StackMemoryEntry]:
        # Keep the newest item inside each priority band. This matters when a
        # long task accumulates more pinned feedback than the prompt window can
        # render: the latest correction must not be displaced by an old one.
        pinned = sorted(stack.get_pinned_entries(), key=lambda entry: entry.ts, reverse=True)
        pinned.sort(key=_entry_priority_rank)
        return pinned[: self.recent_entry_limit]

    def _select_authoritative_constraints(
        self,
        stack: TaskMemoryStack,
        *,
        exclude_ids: set[str],
    ) -> list[StackMemoryEntry]:
        """Keep explicit human constraints visible as a separate audit ledger."""
        eligible_actions = {"user_request", "feedback", "revise", "summarize"}
        constraints = [entry for entry in stack.get_active_entries() if entry.id not in exclude_ids and entry.action in eligible_actions and _AUTHORITATIVE_CONSTRAINT_PATTERN.search(entry.content)]
        # Constraint messages are generally short. A slightly wider window
        # than ordinary recent memory prevents stable requirements from being
        # displaced by a burst of corrections near the end of a long thread.
        return constraints[-(self.recent_entry_limit + 6) :]

    def _select_recent(
        self,
        stack: TaskMemoryStack,
        *,
        exclude_ids: set[str],
        current_run_id: str | None = None,
        correction_candidates: list[StackMemoryEntry] | None = None,
    ) -> list[StackMemoryEntry]:
        active = [entry for entry in stack.get_active_entries() if entry.id not in exclude_ids and entry.status != "pinned" and entry.priority != "critical"]
        if not current_run_id:
            return active[-self.recent_entry_limit :]

        current = [entry for entry in active if entry.run_id == current_run_id]
        # Carry compact summaries plus authoritative user updates across runs in
        # the same thread. Exclude prior model thoughts and execution noise, so
        # continuity improves without repopulating the Central stack with every
        # old observation.
        correction_ids = {entry.id for entry in correction_candidates or []}
        carryover = [entry for entry in active if entry.run_id != current_run_id and (entry.action in {"summarize", "user_request", "revise"} or entry.id in correction_ids)][-self.recent_entry_limit :]
        current_limit = max(self.recent_entry_limit - len(carryover), 0)
        selected_current = current[-current_limit:] if current_limit else []
        return [*carryover, *selected_current]

    def _format_entries(self, entries: Iterable[StackMemoryEntry], *, max_content_chars: int | None = None) -> list[str]:
        return [self._format_entry(entry, max_content_chars=max_content_chars) for entry in entries]

    def _format_entry(self, entry: StackMemoryEntry, *, max_content_chars: int | None = None) -> str:
        parts = [entry.action]
        if entry.actor:
            parts.append(f"actor={entry.actor}")
        if entry.stage:
            parts.append(f"stage={entry.stage}")
        if entry.priority:
            parts.append(f"priority={entry.priority}")
        if entry.action in {
            "user_request",
            "feedback",
            "revise",
        } and _HARD_CONSTRAINT_PATTERN.search(entry.content):
            parts.append("constraint_class=hard")
        if entry.result_ref:
            parts.append(f"result_ref={entry.result_ref}")
        stop_reason = entry.metadata.get("stop_reason")
        if stop_reason:
            parts.append(f"stop_reason={_compact(stop_reason, max_chars=80)}")
        completion_status = entry.metadata.get("completion_status")
        if completion_status:
            parts.append(f"completion_status={_compact(completion_status, max_chars=40)}")
        if entry.failure_note:
            parts.append(f"failure={_compact(entry.failure_note, max_chars=180)}")
        content = _compact(entry.content, max_chars=max_content_chars or self.max_entry_chars)
        return f"- id={entry.id} ({', '.join(parts)}): {content}"

    def _compose_context(
        self,
        header: list[str],
        sections: list[tuple[str, list[StackMemoryEntry], int, bool]],
        footer: list[str],
    ) -> str:
        """Fit memory sections while always preserving the control footer."""
        fixed_chars = len("\n".join([*header, *footer])) + 1
        available = max(self.max_chars - fixed_chars, 0)
        nonempty = [section for section in sections if section[1]]
        total_weight = sum(section[2] for section in nonempty) or 1
        rendered_sections: list[str] = []

        for index, (label, entries, weight, newest_first) in enumerate(nonempty):
            # Give the final section any rounding remainder.
            if index == len(nonempty) - 1:
                budget = available - len("\n".join(rendered_sections))
            else:
                budget = available * weight // total_weight
            rendered = self._render_section(
                label,
                entries,
                budget=max(budget, 0),
                newest_first=newest_first,
            )
            rendered_sections.extend(rendered)

        context = "\n".join([*header, *rendered_sections, *footer])
        if len(context) <= self.max_chars:
            return context

        # Defensive fallback for unusually tiny test/custom limits. Preserve
        # the opening and closing tags plus the footer by trimming only memory
        # section lines from the end.
        while rendered_sections and len(context) > self.max_chars:
            rendered_sections.pop()
            context = "\n".join([*header, *rendered_sections, *footer])
        if len(context) <= self.max_chars:
            return context

        minimal = [
            "<sp-task-context>",
            "Short-term task memory; pinned feedback is authoritative.",
            *footer[1:],
        ]
        return self._clip_lines(minimal)

    def _render_section(
        self,
        label: str,
        entries: list[StackMemoryEntry],
        *,
        budget: int,
        newest_first: bool,
    ) -> list[str]:
        if budget <= len(label) + 3:
            return []
        ordered = list(reversed(entries)) if newest_first else list(entries)
        heading = ["", label]
        remaining = budget - len("\n".join(heading)) - 1
        if remaining <= 0:
            return []

        rendered: list[str] = [*heading]
        per_entry = max(remaining // max(len(ordered), 1), MIN_SECTION_ENTRY_CHARS)
        max_content_chars = max(
            MIN_SECTION_ENTRY_CHARS,
            min(self.max_entry_chars, per_entry - 150),
        )
        for entry in ordered:
            line = self._format_entry(entry, max_content_chars=max_content_chars)
            additional = len(line) + 1
            if additional > remaining:
                # Try one compact final entry so the newest feedback/progress is
                # represented even when its normal metadata is relatively wide.
                compact_chars = max(MIN_SECTION_ENTRY_CHARS, remaining - 170)
                line = self._format_entry(entry, max_content_chars=compact_chars)
                additional = len(line) + 1
                if additional > remaining:
                    break
            rendered.append(line)
            remaining -= additional
        return rendered if len(rendered) > 2 else []

    def _clip_lines(self, lines: list[str]) -> str:
        full_context = "\n".join(lines)
        if len(full_context) <= self.max_chars:
            return full_context

        closing = "</sp-task-context>"
        marker = "...<sp-task-context-truncated>"
        body_lines = lines[:-1] if lines and lines[-1] == closing else lines
        reserved = len(marker) + len(closing) + 2
        output: list[str] = []
        total = 0
        for line in body_lines:
            additional = len(line) + 1
            if total + additional + reserved > self.max_chars:
                break
            output.append(line)
            total += additional
        output.extend([marker, closing])
        return "\n".join(output)[: self.max_chars]
