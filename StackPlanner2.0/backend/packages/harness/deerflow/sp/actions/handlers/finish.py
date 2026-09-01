"""FINISH action handler."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import HumanMessage

from deerflow.sp.actions.events import make_sp_event
from deerflow.sp.actions.handlers.base import HandlerContext
from deerflow.sp.actions.schema import HandlerResult, SPAction
from deerflow.utils.messages import message_to_text

_REPORT_ARTIFACT_TYPES = frozenset({"report", "report_revision", "final_report"})
_INTERMEDIATE_ARTIFACT_TYPES = frozenset(
    {
        "outline",
        "research_observation",
        "data_collection",
        "evidence_bundle",
        "perception_observation",
    }
)
_PARALLEL_NONTERMINAL_SIBLING_KEY = "__sp_parallel_nonterminal_sibling"
_ARTIFACT_TYPE_FAMILIES = {
    "report": _REPORT_ARTIFACT_TYPES,
    "generated_file": frozenset({"generated_file"}),
}
_REPORT_DELIVERABLE_INTENT = re.compile(
    r"(?:报告|复盘|董事会材料|\breport\b|\bboard\s+report\b)",
    re.IGNORECASE,
)
_GENERATED_FILE_INTENT = re.compile(
    r"(?:代码|源码|网页|网站|程序|脚本|\bcode\b|\bwebsite\b|\bweb\s?page\b|\bscript\b|\bapp\b)",
    re.IGNORECASE,
)
_ARTIFACT_REVISION_REQUEST_PATTERN = re.compile(
    r"(?:"
    r"(?:修订|修改|更新|调整|改写).{0,48}(?:报告|文档|文件|代码|网页)"
    r"|(?:报告|文档|文件|代码|网页).{0,48}(?:修订|修改|更新|调整|改写)"
    r"|\b(?:revise|update|edit|rewrite|modify).{0,64}\b(?:report|document|file|code|website)\b"
    r"|\b(?:report|document|file|code|website)\b.{0,64}\b(?:revise|update|edit|rewrite|modify)\b"
    r")",
    re.IGNORECASE | re.DOTALL,
)


def _artifact_identifiers(ref: object) -> set[str]:
    if not isinstance(ref, dict):
        return set()
    return {str(value) for key in ("artifact_id", "artifact_url", "virtual_path") if (value := ref.get(key))}


def _artifact_is_complete(ref: object) -> bool:
    if not isinstance(ref, dict):
        return False
    metadata = ref.get("metadata")
    status = metadata.get("completion_status") if isinstance(metadata, dict) else None
    normalized = str(status or ref.get("completion_status") or "complete").strip().lower()
    return normalized == "complete"


def _iter_artifact_refs(refs: object):
    if not isinstance(refs, dict):
        return
    for key, ref in refs.items():
        if key == "_history" or not isinstance(ref, dict):
            continue
        yield ref
    history = refs.get("_history")
    if isinstance(history, list):
        for ref in history:
            if isinstance(ref, dict):
                yield ref


def _selected_artifact_ref(context: HandlerContext, action: SPAction) -> str | None:
    requested = {str(value) for value in action.input_refs}
    explicit = action.metadata.get("final_artifact_ref")
    if isinstance(explicit, str) and explicit.strip():
        requested.add(explicit.strip())
    if not requested:
        return None
    refs = context.state.get("sp_current_artifact_refs")
    for ref in _iter_artifact_refs(refs):
        if not _artifact_is_complete(ref):
            continue
        identifiers = _artifact_identifiers(ref)
        match = next((identifier for identifier in identifiers if identifier in requested), None)
        if match:
            return match
    return None


def _artifact_for_identifier(
    context: HandlerContext,
    identifier: str | None,
) -> Mapping[str, Any] | None:
    if not identifier:
        return None
    return next(
        (ref for ref in _iter_artifact_refs(context.state.get("sp_current_artifact_refs")) if identifier in _artifact_identifiers(ref)),
        None,
    )


def _grounded_final_summary(
    context: HandlerContext,
    action: SPAction,
    *,
    final_artifact_ref: str | None,
) -> tuple[str, bool]:
    selected = _artifact_for_identifier(context, final_artifact_ref)
    metadata = selected.get("metadata") if isinstance(selected, Mapping) else None
    preview = metadata.get("finalization_preview") if isinstance(metadata, Mapping) else None
    preview_truncated = metadata.get("finalization_preview_truncated") if isinstance(metadata, Mapping) else False
    if isinstance(preview, str) and preview.strip() and not preview_truncated:
        return preview.strip(), True
    return action.task or action.reason, False


def _required_artifact_type(action: SPAction) -> str | None:
    value = action.metadata.get("required_artifact_type")
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().lower().replace("-", "_")


def _matches_required_artifact_type(ref: object, required_type: str) -> bool:
    if not isinstance(ref, dict):
        return False
    artifact_type = str(ref.get("type") or "").strip().lower().replace("-", "_")
    family = _ARTIFACT_TYPE_FAMILIES.get(required_type, frozenset({required_type}))
    return artifact_type in family


def _artifact_family(ref: object) -> str | None:
    if not isinstance(ref, dict):
        return None
    artifact_type = str(ref.get("type") or "").strip().lower().replace("-", "_")
    if artifact_type in _REPORT_ARTIFACT_TYPES:
        return "report"
    if artifact_type == "generated_file":
        return "generated_file"
    return None


def _deliverable_intent(action: SPAction) -> str | None:
    text = "\n".join(
        value
        for value in (
            str(action.task or ""),
            str(action.reason or ""),
        )
        if value
    )
    report = bool(_REPORT_DELIVERABLE_INTENT.search(text))
    generated = bool(_GENERATED_FILE_INTENT.search(text))
    if report and not generated:
        return "report"
    if generated and not report:
        return "generated_file"
    return None


def _requires_current_run_revision_artifact(context: HandlerContext) -> bool:
    messages = context.state.get("messages")
    if not isinstance(messages, list):
        return False
    latest = next(
        (
            message
            for message in reversed(messages)
            if isinstance(message, HumanMessage) and getattr(message, "name", None) != "sp_task_memory_context" and not (isinstance(message.additional_kwargs, Mapping) and message.additional_kwargs.get("hide_from_ui") is True)
        ),
        None,
    )
    return bool(latest and _ARTIFACT_REVISION_REQUEST_PATTERN.search(message_to_text(latest)))


def _has_current_run_revision_artifact(
    context: HandlerContext,
    action: SPAction,
) -> bool:
    if not context.run_id:
        return False
    explicit = action.metadata.get("final_artifact_ref")
    requested = explicit.strip() if isinstance(explicit, str) and explicit.strip() else None
    required_type = _required_artifact_type(action)
    for ref in _iter_artifact_refs(context.state.get("sp_current_artifact_refs")):
        if str(ref.get("run_id") or "") != context.run_id:
            continue
        if not _artifact_is_complete(ref) or not _artifact_identifiers(ref):
            continue
        if requested and requested not in _artifact_identifiers(ref):
            continue
        if required_type and not _matches_required_artifact_type(ref, required_type):
            continue
        if str(ref.get("type") or "") in _REPORT_ARTIFACT_TYPES and not ref.get("is_current", True):
            continue
        return True
    return False


def _normalize_required_artifact_type(
    context: HandlerContext,
    action: SPAction,
) -> dict[str, str] | None:
    """Repair a type label when the exact ref and completion intent agree."""
    declared = _required_artifact_type(action)
    explicit = action.metadata.get("final_artifact_ref")
    if declared is None or not isinstance(explicit, str) or not explicit.strip():
        return None
    requested = explicit.strip()
    selected = next(
        (ref for ref in _iter_artifact_refs(context.state.get("sp_current_artifact_refs")) if requested in _artifact_identifiers(ref)),
        None,
    )
    actual = _artifact_family(selected)
    intent = _deliverable_intent(action)
    if actual is None or actual == declared or intent != actual:
        return None
    action.metadata["declared_required_artifact_type"] = declared
    action.metadata["required_artifact_type"] = actual
    return {
        "from_type": declared,
        "to_type": actual,
        "final_artifact_ref": requested,
    }


def _required_artifact_ref(context: HandlerContext, action: SPAction) -> str | None:
    required_type = _required_artifact_type(action)
    if required_type is None:
        return None
    refs = list(_iter_artifact_refs(context.state.get("sp_current_artifact_refs")))
    explicit = action.metadata.get("final_artifact_ref")
    if isinstance(explicit, str) and explicit.strip():
        requested = explicit.strip()
        for ref in refs:
            if requested in _artifact_identifiers(ref) and _matches_required_artifact_type(ref, required_type) and _artifact_is_complete(ref):
                return requested
        return None

    candidates = [ref for ref in refs if _matches_required_artifact_type(ref, required_type) and _artifact_is_complete(ref) and (str(ref.get("type") or "") not in _REPORT_ARTIFACT_TYPES or ref.get("is_current", True))]
    if not candidates:
        return None
    selected = max(candidates, key=lambda ref: int(ref.get("version") or 0))
    identifiers = _artifact_identifiers(selected)
    artifact_id = selected.get("artifact_id")
    return str(artifact_id) if artifact_id else (sorted(identifiers)[0] if identifiers else None)


def _has_current_human_feedback(context: HandlerContext) -> bool:
    raw_memory = context.state.get("sp_task_memory")
    entries = raw_memory.get("entries") if isinstance(raw_memory, dict) else None
    return bool(context.run_id and isinstance(entries, list) and any(isinstance(entry, dict) and entry.get("action") == "feedback" and entry.get("run_id") == context.run_id for entry in entries))


def _has_final_artifact(context: HandlerContext, action: SPAction) -> bool:
    if action.metadata.get("allow_without_artifact"):
        return True
    refs = context.state.get("sp_current_artifact_refs") or {}
    if not isinstance(refs, dict):
        return False
    explicit = _selected_artifact_ref(context, action)
    requested_explicit = action.metadata.get("final_artifact_ref")
    if isinstance(requested_explicit, str) and requested_explicit.strip() and explicit is None:
        return False
    run_id = context.run_id
    state_run_id = context.state.get("sp_loop_run_id")
    for ref in _iter_artifact_refs(refs):
        if not _artifact_is_complete(ref):
            continue
        artifact_type = str(ref.get("type") or "")
        if artifact_type in _INTERMEDIATE_ARTIFACT_TYPES:
            continue
        if artifact_type in _REPORT_ARTIFACT_TYPES and not ref.get("is_current", True):
            continue
        if not _artifact_identifiers(ref):
            continue
        if not run_id:
            return True
        if not ref.get("run_id") and state_run_id == run_id:
            return True
        if not state_run_id or _has_current_human_feedback(context):
            return True
        if run_id and ref.get("run_id") == run_id:
            return True
        if explicit and explicit in _artifact_identifiers(ref):
            return True
    return False


class FinishHandler:
    def handle(self, action: SPAction, context: HandlerContext) -> HandlerResult:
        if action.metadata.get(_PARALLEL_NONTERMINAL_SIBLING_KEY):
            return HandlerResult(
                next_step="error_recoverable",
                idempotency_key=action.idempotency_key,
                error="FINISH cannot run beside a non-terminal SP action; inspect the sibling result first",
                run_events=[
                    make_sp_event(
                        "sp.finish.rejected",
                        action_id=action.action_id,
                        run_id=context.run_id,
                        reason="parallel_nonterminal_sibling",
                    )
                ],
            )
        if context.state.get("sp_pending_human_interaction"):
            return HandlerResult(
                next_step="error_recoverable",
                idempotency_key=action.idempotency_key,
                error="FINISH is blocked while sp_pending_human_interaction exists",
                run_events=[
                    make_sp_event(
                        "sp.finish.rejected",
                        action_id=action.action_id,
                        run_id=context.run_id,
                        reason="pending_human_interaction",
                    )
                ],
            )
        type_normalization = _normalize_required_artifact_type(context, action)
        if _requires_current_run_revision_artifact(context) and not _has_current_run_revision_artifact(context, action):
            return HandlerResult(
                next_step="error_recoverable",
                idempotency_key=action.idempotency_key,
                error=("FINISH requires a newly completed artifact from the current revision run; an older artifact cannot satisfy the requested revision"),
                run_events=[
                    make_sp_event(
                        "sp.finish.rejected",
                        action_id=action.action_id,
                        run_id=context.run_id,
                        reason="stale_artifact_for_revision",
                    )
                ],
            )
        required_type = _required_artifact_type(action)
        required_ref = _required_artifact_ref(context, action)
        if required_type is not None and required_ref is None:
            return HandlerResult(
                next_step="error_recoverable",
                idempotency_key=action.idempotency_key,
                error=(f"FINISH requires required artifact type '{required_type}' and a matching final_artifact_ref"),
                run_events=[
                    make_sp_event(
                        "sp.finish.rejected",
                        action_id=action.action_id,
                        run_id=context.run_id,
                        reason="required_artifact_type_mismatch",
                        required_artifact_type=required_type,
                    )
                ],
            )
        if not _has_final_artifact(context, action):
            return HandlerResult(
                next_step="error_recoverable",
                idempotency_key=action.idempotency_key,
                error="FINISH requires a final artifact ref or allow_without_artifact=true",
                run_events=[
                    make_sp_event(
                        "sp.finish.rejected",
                        action_id=action.action_id,
                        run_id=context.run_id,
                        reason="missing_final_artifact",
                    )
                ],
            )

        final_artifact_ref = required_ref or _selected_artifact_ref(context, action)
        final_summary, summary_grounded = _grounded_final_summary(
            context,
            action,
            final_artifact_ref=final_artifact_ref,
        )
        entry = context.stack.append_finish(
            final_summary,
            thread_id=context.thread_id,
            run_id=context.run_id,
            priority=action.priority,
            metadata={"action_id": action.action_id, **action.metadata},
        )
        return HandlerResult(
            next_step="finish",
            state_update={
                "sp_current_stage": "finished",
                "sp_active_delegate_id": None,
                "sp_last_run_summary": final_summary,
                **({"sp_last_final_artifact_ref": final_artifact_ref} if final_artifact_ref else {}),
            },
            memory_entries=[entry],
            idempotency_key=action.idempotency_key,
            run_events=[
                *(
                    [
                        make_sp_event(
                            "sp.finish.artifact_type_normalized",
                            action_id=action.action_id,
                            run_id=context.run_id,
                            **type_normalization,
                        )
                    ]
                    if type_normalization
                    else []
                ),
                *(
                    [
                        make_sp_event(
                            "sp.finish.summary_grounded",
                            action_id=action.action_id,
                            run_id=context.run_id,
                            final_artifact_ref=final_artifact_ref,
                        )
                    ]
                    if summary_grounded
                    else []
                ),
                make_sp_event(
                    "sp.finish.accepted",
                    action_id=action.action_id,
                    run_id=context.run_id,
                ),
            ],
        )
