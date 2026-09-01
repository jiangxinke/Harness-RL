"""SUMMARIZE action handler."""

from __future__ import annotations

import re

from deerflow.agents.thread_state import MAX_CONSUMED_TOOL_OBSERVATION_IDS
from deerflow.sp.actions.events import make_sp_event
from deerflow.sp.actions.handlers.base import HandlerContext
from deerflow.sp.actions.schema import HandlerResult, SPAction

MAX_SUMMARY_CHARS = 1200
MIN_RATIO_GUARD_SOURCE_CHARS = 600
MAX_SUMMARY_TO_SOURCE_RATIO = 0.70
_AUTHORITATIVE_REQUIREMENT_PATTERN = re.compile(
    r"(?:"
    r"永久|不可牺牲|必须|不得|禁止|约束|质量要求|输出口径|预算口径"
    r"|must\b|shall\b|required\b|never\b|constraint\b"
    r")",
    re.IGNORECASE,
)
_REQUIREMENT_PREFIX_PATTERN = re.compile(
    r"^(?:"
    r"永久约束[\w-]*|补充偏好|补充风险|质量要求|输出口径|预算口径"
    r"|permanent\s+constraint|requirement|constraint"
    r")\s*[:：-]?\s*",
    re.IGNORECASE,
)
_ACKNOWLEDGEMENT_SUFFIX_PATTERN = re.compile(
    r"(?:[；;。,.，]\s*)?(?:只|仅|先)?(?:需|需要|要)?确认收到.*$",
    re.IGNORECASE,
)
_BOILERPLATE_REQUIREMENT_PATTERN = re.compile(
    r"(?:所有后续方案都)?必须满足|this\s+must\s+be\s+satisfied",
    re.IGNORECASE,
)
_ALNUM_OR_CJK_PATTERN = re.compile(r"[a-z0-9\u3400-\u9fff]+", re.IGNORECASE)


def _normalized_requirement_core(content: str) -> str:
    text = " ".join(content.split()).strip()
    text = _ACKNOWLEDGEMENT_SUFFIX_PATTERN.sub("", text)
    text = _REQUIREMENT_PREFIX_PATTERN.sub("", text)
    text = _BOILERPLATE_REQUIREMENT_PATTERN.sub("", text)
    return text.strip(" \t\r\n:：;；,.，。")


def _normalized_semantic_text(content: str) -> str:
    return "".join(_ALNUM_OR_CJK_PATTERN.findall(content.lower()))


def _semantic_bigrams(content: str) -> set[str]:
    normalized = _normalized_semantic_text(content)
    if len(normalized) < 2:
        return {normalized} if normalized else set()
    return {normalized[index : index + 2] for index in range(len(normalized) - 1)}


def _requirement_is_covered(requirement: str, summary: str) -> bool:
    requirement_normalized = _normalized_semantic_text(requirement)
    summary_normalized = _normalized_semantic_text(summary)
    if not requirement_normalized:
        return True
    if requirement_normalized in summary_normalized:
        return True
    requirement_bigrams = _semantic_bigrams(requirement)
    if not requirement_bigrams:
        return False
    overlap = len(requirement_bigrams & _semantic_bigrams(summary))
    return overlap / len(requirement_bigrams) >= 0.35


def _retain_uncovered_authoritative_requirements(
    summary: str,
    source_entries: list,
) -> tuple[str, list[str]]:
    """Prevent compaction from silently dropping explicit human constraints."""
    retained: list[tuple[str, str]] = []
    for entry in source_entries:
        if entry.action not in {"user_request", "feedback", "revise"} or not _AUTHORITATIVE_REQUIREMENT_PATTERN.search(entry.content):
            continue
        requirement = _normalized_requirement_core(entry.content)
        if requirement and not _requirement_is_covered(requirement, summary):
            retained.append((entry.id, requirement))
    if not retained:
        return summary, []
    requirement_lines = "\n".join(f"- {requirement}" for _, requirement in retained)
    return (
        f"{summary.rstrip()}\n\nAuthoritative requirements retained verbatim:\n{requirement_lines}",
        [entry_id for entry_id, _ in retained],
    )


class SummarizeHandler:
    def handle(self, action: SPAction, context: HandlerContext) -> HandlerResult:
        source_entry_ids = [str(entry_id) for entry_id in action.metadata.get("source_entry_ids", [])]
        if not source_entry_ids:
            source_entry_ids = context.stack.select_explicit_summarization_source_ids(current_run_id=context.run_id)
        summary = str(action.metadata.get("summary") or action.task or action.reason).strip()
        before_ids = {entry.id for entry in context.stack.entries}
        active_before = context.stack.get_active_entries()
        source_id_set = set(source_entry_ids)
        source_entries = [entry for entry in context.stack.entries if entry.id in source_id_set]
        removable_source_entries = [entry for entry in source_entries if entry.status == "active" and entry.priority != "critical"]
        summary, retained_requirement_ids = _retain_uncovered_authoritative_requirements(
            summary,
            removable_source_entries,
        )
        source_char_count = sum(len(entry.content) for entry in removable_source_entries)
        quality_error = _summary_quality_error(
            summary,
            source_char_count=source_char_count,
            requested_source_count=len(source_entry_ids),
            removable_source_count=len(removable_source_entries),
        )
        if quality_error:
            return HandlerResult(
                next_step="error_recoverable",
                idempotency_key=action.idempotency_key,
                error=quality_error,
                run_events=[
                    make_sp_event(
                        "sp.memory.summary_rejected",
                        action_id=action.action_id,
                        run_id=context.run_id,
                        memory_scope="short_term_task_memory",
                        reason="insufficient_compression",
                        requested_source_count=len(source_entry_ids),
                        removable_source_count=len(removable_source_entries),
                        source_char_count=source_char_count,
                        summary_char_count=len(summary),
                        error=quality_error,
                    )
                ],
            )
        source_tool_call_ids = [str(entry.metadata["tool_call_id"]) for entry in context.stack.entries if entry.id in source_entry_ids and entry.metadata.get("tool_call_id")]
        summary_metadata = {
            "action_id": action.action_id,
            **action.metadata,
            "memory_scope": "short_term_task_memory",
            "requested_source_entry_count": len(source_entry_ids),
            "source_entry_count": len(removable_source_entries),
            "source_char_count": source_char_count,
            **({"retained_authoritative_source_entry_ids": (retained_requirement_ids)} if retained_requirement_ids else {}),
        }
        if source_entry_ids:
            entry = context.stack.condense(
                source_entry_ids,
                summary,
                thread_id=context.thread_id,
                run_id=context.run_id,
                stage=action.stage,
                priority=action.priority,
                metadata=summary_metadata,
            )
        else:
            entry = context.stack.append_summary(
                summary,
                thread_id=context.thread_id,
                run_id=context.run_id,
                stage=action.stage,
                priority=action.priority,
                metadata=summary_metadata,
            )
        active_after = context.stack.get_active_entries()
        compression = {
            "active_entries_before": len(active_before),
            "active_entries_after": len(active_after),
            "source_entries": len(removable_source_entries),
            "source_chars": source_char_count,
            "summary_chars": len(summary),
        }
        if source_char_count:
            compression["char_ratio"] = round(len(summary) / source_char_count, 4)
        entry.metadata["compression"] = compression
        state_update = {
            "sp_last_run_summary": summary,
            # Retained as a compatibility/audit marker. Runtime loop prevention
            # now uses same-turn reservation plus progress-based cooldown.
            "sp_summarize_committed_run_id": context.run_id,
        }
        if source_tool_call_ids:
            previous_consumed = context.state.get("sp_consumed_tool_observation_ids")
            state_update["sp_consumed_tool_observation_ids"] = list(dict.fromkeys([*(previous_consumed or []), *source_tool_call_ids]))[-MAX_CONSUMED_TOOL_OBSERVATION_IDS:]
        if action.stage:
            state_update["sp_current_stage"] = action.stage
        events = []
        if source_entry_ids:
            popped_entry_ids = [entry_id for entry_id in source_entry_ids if entry_id in before_ids and all(existing.id != entry_id for existing in context.stack.entries)]
            if popped_entry_ids:
                events.append(
                    make_sp_event(
                        "sp.memory.popped",
                        action_id=action.action_id,
                        run_id=context.run_id,
                        source_entry_ids=popped_entry_ids,
                        entry_count=len(popped_entry_ids),
                    )
                )
            events.append(
                make_sp_event(
                    "sp.memory.condensed",
                    action_id=action.action_id,
                    run_id=context.run_id,
                    source_entry_ids=source_entry_ids,
                    popped_entry_ids=popped_entry_ids,
                    summary_entry_id=entry.id,
                    memory_scope="short_term_task_memory",
                    source_entry_count=len(source_entry_ids),
                    retained_entry_count=len(active_after),
                    source_char_count=source_char_count,
                    summary_char_count=len(summary),
                )
            )
        if retained_requirement_ids:
            events.append(
                make_sp_event(
                    "sp.memory.summary_requirements_retained",
                    action_id=action.action_id,
                    run_id=context.run_id,
                    source_entry_ids=retained_requirement_ids,
                    entry_count=len(retained_requirement_ids),
                    memory_scope="short_term_task_memory",
                )
            )
        events.append(
            make_sp_event(
                "sp.memory.summary_committed",
                action_id=action.action_id,
                run_id=context.run_id,
                summary_entry_id=entry.id,
                memory_scope="short_term_task_memory",
            )
        )
        return HandlerResult(
            next_step="continue",
            state_update=state_update,
            memory_entries=[entry],
            idempotency_key=action.idempotency_key,
            run_events=events,
        )


def _summary_quality_error(
    summary: str,
    *,
    source_char_count: int,
    requested_source_count: int,
    removable_source_count: int,
) -> str | None:
    if len(summary) > MAX_SUMMARY_CHARS:
        return f"SUMMARIZE output is too long: {len(summary)}/{MAX_SUMMARY_CHARS} characters. Keep only the goal, verified decisions/evidence refs, open issues, and next action."
    if requested_source_count and removable_source_count == 0:
        return "SUMMARIZE selected only protected or inactive entries; choose active non-critical source IDs."
    if source_char_count >= MIN_RATIO_GUARD_SOURCE_CHARS and len(summary) > int(source_char_count * MAX_SUMMARY_TO_SOURCE_RATIO):
        return f"SUMMARIZE did not compress the selected task memory enough: {len(summary)}/{source_char_count} characters. Produce a materially shorter decision record."
    return None
