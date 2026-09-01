"""Router for validated SP CentralAgent actions."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from deerflow.sp.actions.events import make_sp_event
from deerflow.sp.actions.handlers import (
    AskHumanHandler,
    BacktrackHandler,
    DelegateHandler,
    FinishHandler,
    HandlerContext,
    MemoryRecallHandler,
    ReflectHandler,
    ReplanHandler,
    ReviseHandler,
    SummarizeHandler,
    ThinkHandler,
)
from deerflow.sp.actions.handlers.base import BaseActionHandler
from deerflow.sp.actions.schema import ActionType, ActionValidationError, HandlerResult, SPAction
from deerflow.sp.artifacts import SPArtifactAdapter
from deerflow.sp.memory import (
    StackMemoryEntry,
    TaskMemoryStack,
    select_correction_review_candidates,
)
from deerflow.sp.memory.stack import DEFAULT_SUMMARIZE_MIN_PROGRESS_ENTRIES
from deerflow.sp.subagents import SPSubagentExecutorProtocol

DEFAULT_MAX_LOOP_ITERATIONS = 20
MAX_IDEMPOTENCY_LEDGER_ENTRIES = 100
SP_SUMMARY_TURN_RESERVED_KEY = "__sp_summary_turn_reserved"
_LOCAL_COMBINATORIAL_WORK_PATTERN = re.compile(
    r"(?:"
    r"\b(?:pairings?|assignments?|permutations?|combinations?|configurations?|schedules?)\b"
    r"|\broom\s+assignment\b|分房|配对|排列|组合|排班|可行方案"
    r")",
    re.IGNORECASE,
)
_EXACT_WORK_PATTERN = re.compile(
    r"(?:"
    r"\b(?:all|every|exact(?:\s+count)?|enumerat\w*|generate|list|find|verify)\b"
    r"|所有|全部|每种|一共几种|穷举|列出|生成|找出|精确|验证"
    r")",
    re.IGNORECASE,
)
_LOCAL_CALCULATION_PATTERN = re.compile(
    r"(?:"
    r"\b(?:calculate|compute|recalculate|arithmetic|equation|price|cost|total)\b"
    r"|计算|重算|核算|实付|费用|价格|算式"
    r")",
    re.IGNORECASE,
)
_EXTERNAL_EVIDENCE_PATTERN = re.compile(
    r"(?:"
    r"\b(?:web|internet|online|search|browse|source|citation|official|latest|current|news)\b"
    r"|联网|上网|搜索|检索|资料|来源|引用|官网|最新|实时|新闻"
    r")",
    re.IGNORECASE,
)
_READ_ONLY_REMOTE_API_PATTERN = re.compile(
    r"(?:"
    r"(?:查询|获取|检索|读取|调用|请求).{0,48}(?:官方)?(?:API|接口|端点|网址|URL)"
    r"|(?:官方)?(?:API|接口|端点|网址|URL).{0,48}(?:查询|获取|检索|读取|调用|请求)"
    r"|\b(?:query|fetch|retrieve|get|read|call|request)\b.{0,64}\b(?:official\s+)?(?:api|endpoint|url)\b"
    r"|\b(?:official\s+)?(?:api|endpoint|url)\b.{0,64}\b(?:query|fetch|retrieve|get|read|call|request)\b"
    r")",
    re.IGNORECASE | re.DOTALL,
)
_REMOTE_API_IMPLEMENTATION_PATTERN = re.compile(
    r"(?:"
    r"实现|编写|修改|开发|构建|集成|客户端|程序|脚本|源码|代码"
    r"|\b(?:implement|write|modify|develop|build|integrate)\b"
    r"|\b(?:client|program|script|source\s+code|code)\b"
    r")",
    re.IGNORECASE,
)
_QUALIFIED_IDENTIFIER_PATTERN = re.compile(
    r"(?<![A-Z0-9_-])"
    r"[A-Z][A-Z0-9_-]*(?:\.[A-Z0-9_-]+){2,}"
    r"(?![A-Z0-9_-])"
)
_CLOSED_IDENTIFIER_REQUEST_PATTERN = re.compile(
    r"(?:"
    r"当前有效|只能|仅限|字段固定|不得出现|不要出现|严格"
    r"|\b(?:current|effective|only|strict|fixed)\b"
    r"|\bmust\s+not\b|\bdo\s+not\s+include\b"
    r")",
    re.IGNORECASE,
)


def _is_one_edit_memory_id(left: str, right: str) -> bool:
    """Return true only for one substitution, insertion, or deletion."""

    if left == right or not right.startswith("spmem_"):
        return False
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        return sum(first != second for first, second in zip(left, right, strict=True)) == 1
    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    short_index = long_index = differences = 0
    while short_index < len(shorter) and long_index < len(longer):
        if shorter[short_index] == longer[long_index]:
            short_index += 1
            long_index += 1
            continue
        differences += 1
        if differences > 1:
            return False
        long_index += 1
    return True


class ActionRouter:
    """Validate, route, and checkpoint SP actions without running business tools."""

    def __init__(self, handlers: Mapping[ActionType, BaseActionHandler] | None = None):
        self._handlers = dict(handlers or {})

    def execute(
        self,
        action_input: SPAction | dict[str, Any],
        *,
        state: Mapping[str, Any] | None = None,
        thread_id: str | None = None,
        run_id: str | None = None,
    ) -> HandlerResult:
        state = state or {}
        try:
            action = action_input if isinstance(action_input, SPAction) else SPAction.from_dict(action_input)
        except ActionValidationError as exc:
            return HandlerResult(
                next_step="error_recoverable",
                error=str(exc),
                run_events=[make_sp_event("sp.action.validation_failed", run_id=run_id, error=str(exc))],
            )

        self._normalize_delegate_target(action)
        self._normalize_memory_target_ids(action, state)

        identifier_guard = self._delegation_identifier_grounding_result(
            action,
            state,
            run_id=run_id,
        )
        if identifier_guard is not None:
            stack = TaskMemoryStack.from_dict(
                state.get("sp_task_memory"),
                thread_id=thread_id,
                run_id=run_id,
            )
            for entry in identifier_guard.memory_entries:
                stack.append(entry)
            identifier_guard.state_update = self._build_state_update(
                action,
                state,
                stack,
                identifier_guard,
                run_id=run_id,
            )
            return identifier_guard

        duplicate = self._duplicate_result(action, state, run_id=run_id)
        if duplicate is not None:
            stack = TaskMemoryStack.from_dict(state.get("sp_task_memory"), thread_id=thread_id, run_id=run_id)
            duplicate.state_update = self._build_state_update(
                action,
                state,
                stack,
                duplicate,
                run_id=run_id,
            )
            return duplicate

        limit_result = self._loop_limit_result(action, state, run_id=run_id)
        if limit_result is not None:
            return limit_result

        delegation_policy = self._delegation_policy_result(action, state, run_id=run_id)
        if delegation_policy is not None:
            stack = TaskMemoryStack.from_dict(state.get("sp_task_memory"), run_id=run_id)
            for entry in delegation_policy.memory_entries:
                stack.append(entry)
            delegation_policy.state_update = self._build_state_update(
                action,
                state,
                stack,
                delegation_policy,
                run_id=run_id,
            )
            # The rejected DELEGATE itself is now a visible policy checkpoint.
            # Recording it as another completed DELEGATE leaves the next model
            # turn in the exact same state and permanently blocks every retry
            # to that specialist. Expose an effective THINK checkpoint while
            # retaining the original action in the idempotency ledger.
            last_handler = dict(delegation_policy.state_update.get("sp_last_handler_result") or {})
            last_handler.update(
                {
                    "action_type": ActionType.THINK.value,
                    "target_agent": None,
                    "requested_action_type": ActionType.DELEGATE.value,
                    "requested_target_agent": action.target_agent,
                    "policy_checkpoint": ("same_target_requires_intermediate_control_action"),
                }
            )
            delegation_policy.state_update["sp_last_handler_result"] = last_handler
            return delegation_policy

        correction_reflection = self._forced_correction_reflection(
            action,
            state,
            thread_id=thread_id,
            run_id=run_id,
        )
        if correction_reflection is not None:
            return correction_reflection

        forced_reflection = self._forced_recovery_reflection(action, state, run_id=run_id)
        if forced_reflection is not None:
            return forced_reflection

        summary_policy = self._summary_policy_result(
            action,
            state,
            thread_id=thread_id,
            run_id=run_id,
        )
        if summary_policy is not None:
            return summary_policy

        handler = self._handlers.get(action.action_type)
        if handler is None:
            return self._unsupported_result(action, state, run_id=run_id)

        stack = TaskMemoryStack.from_dict(state.get("sp_task_memory"), thread_id=thread_id, run_id=run_id)
        context = HandlerContext(state=state, stack=stack, thread_id=thread_id, run_id=run_id)
        events = [
            make_sp_event(
                "sp.action.created",
                action_id=action.action_id,
                run_id=run_id,
                action_type=action.action_type.value,
                idempotency_key=action.idempotency_key,
                reason=action.reason,
                task=action.task,
                target_agent=action.target_agent,
                input_refs=list(action.input_refs),
                expected_output=action.expected_output,
                stage=action.stage,
                priority=action.priority,
                requires_human=action.requires_human,
                metadata=dict(action.metadata),
            ),
            *(
                [
                    make_sp_event(
                        "sp.delegate.target_normalized",
                        action_id=action.action_id,
                        run_id=run_id,
                        **action.metadata["target_agent_normalization"],
                    )
                ]
                if isinstance(
                    action.metadata.get("target_agent_normalization"),
                    Mapping,
                )
                else []
            ),
            *(
                [
                    make_sp_event(
                        "sp.action.target_normalized",
                        action_id=action.action_id,
                        run_id=run_id,
                        **action.metadata["memory_target_normalization"],
                    )
                ]
                if isinstance(
                    action.metadata.get("memory_target_normalization"),
                    Mapping,
                )
                else []
            ),
            make_sp_event("sp.handler.started", action_id=action.action_id, run_id=run_id, action_type=action.action_type.value),
        ]
        try:
            result = handler.handle(action, context)
        except Exception as exc:  # pragma: no cover - defensive runtime boundary
            result = HandlerResult(next_step="error_recoverable", idempotency_key=action.idempotency_key, error=str(exc))

        state_update = self._build_state_update(action, state, stack, result, run_id=run_id)
        result.state_update = {**result.state_update, **state_update}
        result.run_events = [
            *events,
            *result.run_events,
            make_sp_event(
                "sp.handler.failed" if result.next_step.startswith("error") else "sp.handler.completed",
                action_id=action.action_id,
                run_id=run_id,
                action_type=action.action_type.value,
                next_step=result.next_step,
                error=result.error,
            ),
        ]
        result.idempotency_key = action.idempotency_key
        return result

    @staticmethod
    def _normalize_delegate_target(action: SPAction) -> None:
        """Repair clear researcher/coder role inversions at the action boundary."""
        if action.action_type is not ActionType.DELEGATE:
            return
        text = "\n".join(
            value
            for value in (
                action.task,
                action.reason,
                action.expected_output,
            )
            if isinstance(value, str) and value.strip()
        )
        if action.target_agent == "coder" and _READ_ONLY_REMOTE_API_PATTERN.search(text) and not _REMOTE_API_IMPLEMENTATION_PATTERN.search(text):
            action.metadata.setdefault("declared_target_agent", "coder")
            action.metadata["target_agent_normalization"] = {
                "from_target": "coder",
                "to_target": "researcher",
                "reason": "read_only_remote_api_retrieval",
            }
            action.target_agent = "researcher"
            action.stage = "research"
            return
        if action.target_agent != "researcher":
            return
        if _EXTERNAL_EVIDENCE_PATTERN.search(text):
            return
        exact_combinatorial = bool(_LOCAL_COMBINATORIAL_WORK_PATTERN.search(text) and _EXACT_WORK_PATTERN.search(text))
        if not exact_combinatorial and not _LOCAL_CALCULATION_PATTERN.search(text):
            return

        action.metadata.setdefault("declared_target_agent", "researcher")
        action.metadata["target_agent_normalization"] = {
            "from_target": "researcher",
            "to_target": "coder",
            "reason": "self_contained_exact_work",
        }
        action.target_agent = "coder"
        if action.stage == "research":
            action.stage = "implementation"

    @staticmethod
    def _normalize_memory_target_ids(
        action: SPAction,
        state: Mapping[str, Any],
    ) -> None:
        """Repair a unique one-character typo in an opaque task-memory ID."""

        if action.action_type not in {
            ActionType.REFLECT,
            ActionType.REVISE,
            ActionType.BACKTRACK,
        }:
            return
        stack = TaskMemoryStack.from_dict(state.get("sp_task_memory"))
        known_ids = [entry.id for entry in stack.entries]
        if not known_ids:
            return

        replacements: list[dict[str, str]] = []

        def normalize(value: Any) -> str:
            target = str(value)
            if target in known_ids or not target.startswith("spmem_") or len(target) < 20:
                return target
            matches = [candidate for candidate in known_ids if _is_one_edit_memory_id(target, candidate)]
            if len(matches) != 1:
                return target
            replacements.append({"from": target, "to": matches[0]})
            return matches[0]

        raw_targets = action.metadata.get("target_entry_ids")
        if isinstance(raw_targets, list):
            action.metadata["target_entry_ids"] = [normalize(value) for value in raw_targets]
        source_ids = action.metadata.get("source_entry_ids")
        if isinstance(source_ids, list):
            action.metadata["source_entry_ids"] = [normalize(value) for value in source_ids]
        if action.action_type is ActionType.BACKTRACK and action.metadata.get("backtrack_target_type") == "entry" and action.metadata.get("backtrack_target_id"):
            action.metadata["backtrack_target_id"] = normalize(action.metadata["backtrack_target_id"])
        if replacements:
            action.metadata["memory_target_normalization"] = {
                "reason": "unique_one_edit_opaque_id",
                "replacements": replacements,
            }

    def _duplicate_result(self, action: SPAction, state: Mapping[str, Any], *, run_id: str | None) -> HandlerResult | None:
        ledger = state.get("sp_idempotency_ledger")
        previous = ledger.get(action.idempotency_key) if isinstance(ledger, Mapping) else None
        if not isinstance(previous, Mapping) and state.get("sp_last_idempotency_key") == action.idempotency_key:
            previous = state.get("sp_last_handler_result")
        if not isinstance(previous, Mapping):
            return None
        next_step = str(previous.get("next_step") or "continue")
        previous_action_type = previous.get("action_type")
        if previous_action_type and previous_action_type != action.action_type.value:
            error = f"idempotency_key {action.idempotency_key!r} was already used by {previous_action_type}, not {action.action_type.value}"
            return HandlerResult(
                next_step="error_recoverable",
                idempotency_key=action.idempotency_key,
                error=error,
                run_events=[
                    make_sp_event(
                        "sp.action.idempotency_collision",
                        action_id=action.action_id,
                        run_id=run_id,
                        idempotency_key=action.idempotency_key,
                        previous_action_type=previous_action_type,
                        action_type=action.action_type.value,
                        error=error,
                    )
                ],
            )

        # Recoverable failures are deliberately retryable with the same stable
        # key. Successful side effects remain protected by duplicate skipping.
        if next_step == "error_recoverable":
            return None

        # Once feedback has resolved an ASK_HUMAN action, replaying its stable
        # key must continue rather than recreating an already-answered request.
        if action.action_type == ActionType.ASK_HUMAN and not state.get("sp_pending_human_interaction"):
            next_step = "continue"
        if next_step not in {"continue", "interrupt", "finish", "error_recoverable", "error_fatal"}:
            next_step = "continue"
        return HandlerResult(
            next_step=next_step,  # type: ignore[arg-type]
            idempotency_key=action.idempotency_key,
            run_events=[make_sp_event("sp.action.duplicate_skipped", action_id=action.action_id, run_id=run_id, idempotency_key=action.idempotency_key)],
        )

    def _delegation_policy_result(self, action: SPAction, state: Mapping[str, Any], *, run_id: str | None) -> HandlerResult | None:
        """Require a new control decision before repeating the same delegation target."""
        if action.action_type != ActionType.DELEGATE:
            return None
        previous = state.get("sp_last_handler_result")
        if not isinstance(previous, Mapping):
            return None
        if previous.get("next_step") != "continue" or previous.get("action_type") != ActionType.DELEGATE.value:
            return None
        if action.target_agent == "reporter":
            # DelegateHandler has a stricter report-version guard that returns
            # FINISH when no explicit revision intent exists.
            return None
        if previous.get("target_agent") != action.target_agent:
            return None

        content = f"Blocked a consecutive delegation to {action.target_agent}. CentralAgent must inspect the previous result with THINK, REFLECT, REPLAN, or SUMMARIZE before delegating to the same specialist again."
        entry = StackMemoryEntry(
            thread_id=None,
            run_id=run_id,
            actor="policy",
            action="delegate_skipped",
            content=content,
            stage=action.stage,
            priority="high",
            metadata={"action_id": action.action_id, "target_agent": action.target_agent},
        )
        return HandlerResult(
            next_step="continue",
            memory_entries=[entry],
            idempotency_key=action.idempotency_key,
            run_events=[
                make_sp_event(
                    "sp.delegate.policy_blocked",
                    action_id=action.action_id,
                    run_id=run_id,
                    target_agent=action.target_agent,
                    reason="same_target_requires_intermediate_control_action",
                )
            ],
        )

    @staticmethod
    def _delegation_identifier_grounding_result(
        action: SPAction,
        state: Mapping[str, Any],
        *,
        run_id: str | None,
    ) -> HandlerResult | None:
        """Reject invented qualified identifiers in a closed user contract.

        This is intentionally narrow. It applies only when the user explicitly
        constrains the current/effective identifier set and the delegated task
        introduces a new token in a namespace the user already supplied. It
        does not prevent a researcher from discovering identifiers for an open
        exploratory request.
        """

        if action.action_type is not ActionType.DELEGATE or action.metadata.get("allow_derived_identifiers") is True:
            return None
        stack = TaskMemoryStack.from_dict(state.get("sp_task_memory"))
        human_entries = [entry for entry in stack.get_active_entries() if entry.actor == "human" and entry.action in {"user_request", "feedback"}]
        if not human_entries:
            return None
        latest_human = human_entries[-1].content
        if not _CLOSED_IDENTIFIER_REQUEST_PATTERN.search(latest_human):
            return None

        human_text = "\n".join(entry.content for entry in human_entries)
        user_identifiers = set(_QUALIFIED_IDENTIFIER_PATTERN.findall(human_text))
        if not user_identifiers:
            return None
        action_text = "\n".join(
            value
            for value in (
                action.task,
                action.reason,
                action.expected_output,
            )
            if isinstance(value, str) and value.strip()
        )
        action_identifiers = set(_QUALIFIED_IDENTIFIER_PATTERN.findall(action_text))
        namespaces = {identifier.split(".", 1)[0] for identifier in user_identifiers}
        ungrounded = sorted(identifier for identifier in action_identifiers - user_identifiers if identifier.split(".", 1)[0] in namespaces)
        if not ungrounded:
            return None

        allowed = sorted(user_identifiers)
        error = (
            "DELEGATE introduced identifier(s) absent from the authoritative "
            f"user task: {', '.join(ungrounded)}. Re-read the active user "
            "corrections and use only the current effective identifier from "
            f"this user-provided set: {', '.join(allowed)}. Do not guess a "
            "replacement identifier."
        )
        entry = StackMemoryEntry(
            thread_id=None,
            run_id=run_id,
            actor="policy",
            action="delegate_skipped",
            content=error,
            stage=action.stage,
            priority="high",
            metadata={
                "action_id": action.action_id,
                "target_agent": action.target_agent,
                "ungrounded_identifiers": ungrounded,
                "user_identifiers": allowed,
            },
        )
        return HandlerResult(
            next_step="error_recoverable",
            memory_entries=[entry],
            idempotency_key=action.idempotency_key,
            error=error,
            run_events=[
                make_sp_event(
                    "sp.delegate.ungrounded_identifier_rejected",
                    action_id=action.action_id,
                    run_id=run_id,
                    target_agent=action.target_agent,
                    ungrounded_identifiers=ungrounded,
                    user_identifiers=allowed,
                )
            ],
        )

    def _summary_policy_result(
        self,
        action: SPAction,
        state: Mapping[str, Any],
        *,
        thread_id: str | None,
        run_id: str | None,
    ) -> HandlerResult | None:
        """Block same-turn duplicates and no-progress summary loops.

        Multiple summaries are valid in a genuinely long run, but only after
        meaningful new work has entered short-term task memory. This replaces
        the previous one-summary-per-run hard stop, which forced later context
        to be pruned instead of compressed.
        """
        if action.action_type != ActionType.SUMMARIZE:
            return None
        stack = TaskMemoryStack.from_dict(
            state.get("sp_task_memory"),
            thread_id=thread_id,
            run_id=run_id,
        )
        same_turn_duplicate = bool(state.get(SP_SUMMARY_TURN_RESERVED_KEY))
        progress = stack.summarization_progress_since_last(current_run_id=run_id)
        cooldown_active = progress is not None and progress < DEFAULT_SUMMARIZE_MIN_PROGRESS_ENTRIES
        if not same_turn_duplicate and not cooldown_active:
            return None

        if same_turn_duplicate:
            error = "SUMMARIZE is already executing in this model turn; inspect the committed summary and continue."
            reason = "same_model_turn"
        else:
            error = f"SUMMARIZE cooldown is active; {progress}/{DEFAULT_SUMMARIZE_MIN_PROGRESS_ENTRIES} meaningful new task-memory entries exist since the last summary. Continue the task before compressing again."
            reason = "insufficient_new_progress"
        result = HandlerResult(
            next_step="continue",
            idempotency_key=action.idempotency_key,
            error=error,
            run_events=[
                make_sp_event(
                    "sp.memory.summary_repeat_blocked",
                    action_id=action.action_id,
                    run_id=run_id,
                    memory_scope="short_term_task_memory",
                    reason=reason,
                    progress_entries=progress,
                    required_progress_entries=DEFAULT_SUMMARIZE_MIN_PROGRESS_ENTRIES,
                    error=error,
                )
            ],
        )
        result.state_update = self._build_state_update(
            action,
            state,
            stack,
            result,
            run_id=run_id,
        )
        # A sibling summary starts from the same pre-action snapshot. Never
        # write that stale stack back over the summary committed by its sibling.
        result.state_update.pop("sp_task_memory", None)
        return result

    def _loop_limit_result(self, action: SPAction, state: Mapping[str, Any], *, run_id: str | None) -> HandlerResult | None:
        loop_iteration = self._loop_iteration_for_run(state, run_id)
        max_loop_iterations = int(state.get("sp_max_loop_iterations") or DEFAULT_MAX_LOOP_ITERATIONS)
        if loop_iteration < max_loop_iterations:
            return None
        error = f"SP action loop exceeded max iterations: {loop_iteration}/{max_loop_iterations}"
        result_summary = {"next_step": "error_fatal", "error": error, "action_id": action.action_id}
        return HandlerResult(
            next_step="error_fatal",
            state_update={"sp_last_handler_result": result_summary},
            idempotency_key=action.idempotency_key,
            error=error,
            run_events=[make_sp_event("sp.handler.failed", action_id=action.action_id, run_id=run_id, error=error)],
        )

    def _forced_recovery_reflection(
        self,
        action: SPAction,
        state: Mapping[str, Any],
        *,
        run_id: str | None,
    ) -> HandlerResult | None:
        """Force one REFLECT before a new action follows a recoverable failure."""
        previous = state.get("sp_last_handler_result")
        if not isinstance(previous, Mapping) or previous.get("next_step") != "error_recoverable":
            return None
        if not previous.get("action_type"):
            return None
        if action.action_type in {ActionType.REFLECT, ActionType.ASK_HUMAN}:
            return None
        if previous.get("action_type") == ActionType.FINISH.value and action.action_type is ActionType.DELEGATE and str(previous.get("error") or "").startswith("FINISH requires"):
            # FINISH validation already identifies the deterministic repair:
            # create or revise the missing deliverable. The availability
            # middleware narrows this model turn to DELEGATE, so replacing
            # that recovery with REFLECT loses the repair instruction and can
            # trap CentralAgent in a FINISH -> REFLECT loop.
            return None
        if previous.get("action_type") == action.action_type.value:
            # A corrected retry normally receives a fresh model/tool-call ID.
            # Treating that new ID as an unrelated action used to discard the
            # repair behind an automatic REFLECT (for example a reporter retry
            # after removing an invalid tool hint). The retry itself is the
            # recovery decision; normal loop/idempotency guards still apply.
            return None
        previous_key = previous.get("idempotency_key") or state.get("sp_last_idempotency_key")
        if action.idempotency_key == previous_key:
            # Retrying the same logical operation is explicitly allowed.
            return None

        reflect_handler = self._handlers.get(ActionType.REFLECT)
        if reflect_handler is None:
            return None
        failure = str(previous.get("error") or "The previous action failed and needs diagnosis.")
        reflect_action = SPAction.create(
            ActionType.REFLECT,
            action_id=f"sp-recovery-reflect-{action.action_id}",
            idempotency_key=f"sp-recovery-reflect-{previous_key or action.idempotency_key}",
            reason="Diagnose the previous recoverable action failure before continuing.",
            task=failure,
            stage=str(state.get("sp_current_stage") or action.stage or "verification"),
            priority="high",
            metadata={"failure_note": failure, "triggered_by_action_id": action.action_id},
        )
        result = self.execute(reflect_action, state=state, run_id=run_id)
        result.run_events.insert(
            0,
            make_sp_event(
                "sp.action.policy_reflection_forced",
                action_id=action.action_id,
                run_id=run_id,
                failed_action_id=previous.get("action_id"),
                requested_action_type=action.action_type.value,
            ),
        )
        return result

    def _forced_correction_reflection(
        self,
        action: SPAction,
        state: Mapping[str, Any],
        *,
        thread_id: str | None,
        run_id: str | None,
    ) -> HandlerResult | None:
        """Backtrack stale conclusions before executing work based on a correction."""
        if action.action_type not in {
            ActionType.DELEGATE,
            ActionType.REVISE,
            ActionType.REPLAN,
            ActionType.FINISH,
        }:
            return None
        stack = TaskMemoryStack.from_dict(
            state.get("sp_task_memory"),
            thread_id=thread_id,
            run_id=run_id,
        )
        candidates = select_correction_review_candidates(
            stack,
            current_run_id=run_id,
        )
        if not candidates:
            return None

        candidate_ids = [entry.id for entry in candidates]
        reflect_action = SPAction.create(
            ActionType.REFLECT,
            action_id=f"sp-correction-reflect-{action.action_id}",
            idempotency_key=f"sp-correction-reflect-{action.idempotency_key}",
            reason="The latest user correction invalidates prior active conclusions.",
            task=("Inspect and withdraw the stale prior conclusions before recalculating from the corrected user input."),
            stage="verification",
            priority="high",
            metadata={
                "target_entry_ids": candidate_ids,
                "triggered_by_action_id": action.action_id,
                "trigger": "user_correction",
            },
        )
        result = self.execute(
            reflect_action,
            state=state,
            thread_id=thread_id,
            run_id=run_id,
        )
        result.run_events.insert(
            0,
            make_sp_event(
                "sp.action.correction_reflection_forced",
                action_id=reflect_action.action_id,
                run_id=run_id,
                requested_action_id=action.action_id,
                requested_action_type=action.action_type.value,
                target_entry_ids=candidate_ids,
            ),
        )
        return result

    def _unsupported_result(self, action: SPAction, state: Mapping[str, Any], *, run_id: str | None) -> HandlerResult:
        error = f"No handler registered for {action.action_type.value}"
        result = HandlerResult(
            next_step="error_recoverable",
            idempotency_key=action.idempotency_key,
            error=error,
            run_events=[
                make_sp_event("sp.action.created", action_id=action.action_id, run_id=run_id, action_type=action.action_type.value),
                make_sp_event("sp.handler.failed", action_id=action.action_id, run_id=run_id, error=error),
            ],
        )
        result.state_update = self._build_state_update(
            action,
            state,
            TaskMemoryStack.from_dict(state.get("sp_task_memory")),
            result,
            run_id=run_id,
        )
        return result

    @staticmethod
    def _loop_iteration_for_run(state: Mapping[str, Any], run_id: str | None) -> int:
        state_run_id = state.get("sp_loop_run_id")
        if run_id and state_run_id and str(state_run_id) != run_id:
            return 0
        return int(state.get("sp_loop_iteration") or 0)

    def _build_state_update(
        self,
        action: SPAction,
        state: Mapping[str, Any],
        stack: TaskMemoryStack,
        result: HandlerResult,
        *,
        run_id: str | None,
    ) -> dict[str, Any]:
        loop_iteration = self._loop_iteration_for_run(state, run_id) + 1
        state_update: dict[str, Any] = {
            "sp_task_memory": stack.to_dict(),
            "sp_current_action_id": None,
            "sp_current_action": None,
            "sp_last_action_id": action.action_id,
            "sp_last_idempotency_key": action.idempotency_key,
            "sp_loop_iteration": loop_iteration,
            "sp_loop_run_id": run_id or state.get("sp_loop_run_id"),
            "sp_last_handler_result": {
                "next_step": result.next_step,
                "error": result.error,
                "action_id": action.action_id,
                "action_type": action.action_type.value,
                "idempotency_key": action.idempotency_key,
                "target_agent": action.target_agent,
                "loop_iteration": loop_iteration,
                "run_id": run_id,
            },
        }
        ledger = dict(state.get("sp_idempotency_ledger")) if isinstance(state.get("sp_idempotency_ledger"), Mapping) else {}
        ledger.pop(action.idempotency_key, None)
        ledger[action.idempotency_key] = dict(state_update["sp_last_handler_result"])
        state_update["sp_idempotency_ledger"] = dict(list(ledger.items())[-MAX_IDEMPOTENCY_LEDGER_ENTRIES:])
        if result.artifact_refs:
            existing_refs = state.get("sp_current_artifact_refs") if isinstance(state.get("sp_current_artifact_refs"), dict) else {}
            state_update["sp_current_artifact_refs"] = {**existing_refs, **result.artifact_refs}
        return state_update


def build_default_action_router(
    *,
    delegate_executor: SPSubagentExecutorProtocol | None = None,
    memory_recall_executor: SPSubagentExecutorProtocol | None = None,
    artifact_adapter: SPArtifactAdapter | None = None,
) -> ActionRouter:
    handlers: dict[ActionType, BaseActionHandler] = {
        ActionType.THINK: ThinkHandler(),
        ActionType.REFLECT: ReflectHandler(),
        ActionType.REVISE: ReviseHandler(),
        ActionType.BACKTRACK: BacktrackHandler(),
        ActionType.REPLAN: ReplanHandler(),
        ActionType.SUMMARIZE: SummarizeHandler(),
        ActionType.ASK_HUMAN: AskHumanHandler(),
        ActionType.RECALL_MEMORY: MemoryRecallHandler(executor=memory_recall_executor),
        ActionType.FINISH: FinishHandler(),
    }
    if delegate_executor is not None:
        handlers[ActionType.DELEGATE] = DelegateHandler(executor=delegate_executor, artifact_adapter=artifact_adapter)
    return ActionRouter(handlers)
