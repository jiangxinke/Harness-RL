"""SP control actions bound to StackPlanner's CentralAgent loop.

Only this SP vocabulary is exposed to the CentralAgent. Ordinary execution
tools are bound to delegated subagents by the SP runtime.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from threading import Lock
from typing import Any, Literal, Protocol

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ModelCallResult,
    ModelRequest,
    ModelResponse,
    hook_config,
)
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deerflow.agents.human_input import read_human_input_response
from deerflow.agents.thread_state import (
    MAX_CONSUMED_TOOL_OBSERVATION_IDS,
    SP_CONCURRENT_MERGE_MODE_KEY,
)
from deerflow.sp.actions import ActionType, SPAction, build_default_action_router
from deerflow.sp.actions.router import SP_SUMMARY_TURN_RESERVED_KEY
from deerflow.sp.hitl import record_human_feedback
from deerflow.sp.memory import StackMemoryEntry, TaskMemoryStack
from deerflow.sp.subagents import SPSubagentExecutorProtocol
from deerflow.subagents.status_contract import (
    SUBAGENT_STOP_REASON_VALUES,
    make_subagent_additional_kwargs,
)
from deerflow.utils.messages import message_to_text

SP_CONTROL_TOOL_NAMES = frozenset(
    {
        "sp_think",
        "sp_delegate",
        "sp_recall_memory",
        "sp_reflect",
        "sp_revise",
        "sp_backtrack",
        "sp_replan",
        "sp_summarize",
        "sp_ask_human",
        "sp_finish",
    }
)
SP_CONTROL_EXECUTION_TOOL_NAMES = SP_CONTROL_TOOL_NAMES | {"sp_thought"}
MAX_CONSECUTIVE_WEB_SEARCH_FAILURES = 2
MAX_IMPLICIT_THINK_CHARS = 900
SP_ARTIFACT_PATHS_KEY = "sp_artifact_paths"
MAX_RUN_LOCAL_GUARD_ENTRIES = 1024
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
_ACKNOWLEDGEMENT_ONLY_PATTERN = re.compile(
    r"(?:"
    r"(?:只|仅)(?:需|需要|要|回复|回答)?(?:确认)?(?:已)?收到"
    r"|先(?:只)?确认收到"
    r"|(?:just|only)\s+(?:acknowledge|confirm)(?:\s+(?:receipt|received))?"
    r")",
    re.IGNORECASE,
)
_ACKNOWLEDGEMENT_NEGATION_PATTERN = re.compile(
    r"(?:不要|不需|无需|不用)只?确认收到|do\s+not\s+(?:just\s+)?(?:acknowledge|confirm)",
    re.IGNORECASE,
)
_IMMEDIATE_DELIVERABLE_PATTERN = re.compile(
    r"(?:现在|立即|马上)(?:输出|给出|生成|执行)|输出最终|给出最终|final\s+(?:answer|output|result)",
    re.IGNORECASE,
)
_STRICT_JSON_REQUEST_PATTERN = re.compile(
    r"(?:严格\s*JSON|\bstrict\s+JSON\b)",
    re.IGNORECASE,
)
_JSON_FENCE_PATTERN = re.compile(
    r"\A\s*```(?:json)?\s*(.*?)\s*```\s*\Z",
    re.IGNORECASE | re.DOTALL,
)
_JSON_SORT_REQUEST_PATTERNS = (
    re.compile(
        r"按\s*`?(?P<field>[A-Za-z_][A-Za-z0-9_]*)`?\s*(?:字段)?\s*"
        r"(?P<direction>升序|降序)(?:排列|排序)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:sort(?:ed)?\s+)?by\s+`?(?P<field>[A-Za-z_][A-Za-z0-9_]*)`?"
        r"(?:\s+in)?\s+(?P<direction>ascending|descending)\s+order",
        re.IGNORECASE,
    ),
)
_PURE_REPORT_DELEGATION_PATTERN = re.compile(
    r"(?:"
    r"(?:生成|撰写|编写|起草|制作|修订|改写).{0,48}(?:报告|复盘|董事会材料|markdown文档)"
    r"|(?:报告|复盘|董事会材料|markdown文档).{0,48}(?:生成|撰写|编写|起草|制作|修订|改写)"
    r"|(?:generate|write|draft|produce|create|revise|rewrite).{0,64}\b(?:report|board report|markdown document)\b"
    r"|\b(?:report|board report|markdown document)\b.{0,64}(?:generate|write|draft|produce|create|revise|rewrite)"
    r")",
    re.IGNORECASE | re.DOTALL,
)
_CODE_OR_APP_DELIVERABLE_PATTERN = re.compile(
    r"(?:"
    r"代码|源码|程序|脚本|网页|网站|应用|前端|后端|接口|可执行|"
    r"\bcode\b|\bsource\s+code\b|\bscript\b|\bweb\s?page\b|\bwebsite\b|\bapp(?:lication)?\b|"
    r"\bfrontend\b|\bbackend\b|\bapi\b|\bexecutable\b"
    r")",
    re.IGNORECASE,
)
_CODER_READ_ONLY_PATTERN = re.compile(
    r"(?:"
    r"读取|阅读|检查|查看|提取|解析|梳理|汇总"
    r"|\bread\b|\binspect\b|\bexamine\b|\bextract\b|\bparse\b|\bsummarize\b"
    r")",
    re.IGNORECASE,
)
_CODER_MUTATION_PATTERN = re.compile(
    r"(?:"
    r"实现|修改|编辑|写入|创建|生成|运行|执行|测试|构建"
    r"|\bimplement\b|\bmodify\b|\bedit\b|\bwrite\b|\bcreate\b|\bgenerate\b|"
    r"\brun\b|\bexecute\b|\btest\b|\bbuild\b"
    r")",
    re.IGNORECASE,
)
_ARTIFACT_DELIVERABLE_REQUEST_PATTERN = re.compile(
    r"(?:"
    r"可下载|下载文件|文件交付|作为.{0,16}文件|生成.{0,32}(?:报告|文档|代码|网页|图表|文件)"
    r"|(?:报告|文档|代码|网页|图表).{0,32}(?:生成|交付|下载|保存为文件)"
    r"|\bdownloadable\b|\bdownload\s+(?:link|file)\b|\bdeliver.{0,24}\bfile\b"
    r"|\bgenerate.{0,32}\b(?:report|document|code|website|chart|file)\b"
    r"|\b(?:report|document|code|website|chart).{0,32}\b(?:file|download|deliver)\b"
    r")",
    re.IGNORECASE | re.DOTALL,
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
_SP_DIRECT_ANSWER_MESSAGE_NAME = "sp_direct_answer_required"
_SP_ARTIFACT_RECOVERY_MESSAGE_NAME = "sp_artifact_recovery_required"
_SP_CLARIFICATION_REQUIRED_MESSAGE_NAME = "sp_clarification_required"
_SP_REPORT_FINALIZATION_MESSAGE_NAME = "sp_report_finalization_required"
_SP_ARTIFACT_FINALIZATION_MESSAGE_NAME = "sp_artifact_finalization_required"
_SP_CONVERSATIONAL_FINALIZATION_MESSAGE_NAME = "sp_conversational_finalization_required"
_SP_REPORT_RECOVERY_EXHAUSTED_MESSAGE_NAME = "sp_report_recovery_exhausted"
_SP_DELEGATE_RECOVERY_EXHAUSTED_MESSAGE_NAME = "sp_delegate_recovery_exhausted"
_CJK_TEXT_PATTERN = re.compile(r"[\u3400-\u9fff]")
SP_REPORT_RECOVERY_MAX_ATTEMPTS = 3
SP_DELEGATE_RECOVERY_MAX_ATTEMPTS = 3
_SP_SPECIALIST_ROLES = frozenset({"researcher", "coder", "reporter", "outline", "perception"})


def _artifact_is_complete(ref: Mapping[str, Any]) -> bool:
    metadata = ref.get("metadata")
    status = metadata.get("completion_status") if isinstance(metadata, Mapping) else None
    return str(status or ref.get("completion_status") or "complete").strip().lower() == "complete"


@tool("sp_think", parse_docstring=True)
def sp_think(task: str, reason: str | None = None, stage: str | None = None) -> str:
    """Record an explicit StackPlanner thought or checkpoint before continuing.

    Args:
        task: The concise thought or checkpoint to record.
        reason: Why this checkpoint is useful.
        stage: Optional task stage.
    """
    return "StackPlanner THINK action recorded. Continue with the task."


@tool("sp_thought", parse_docstring=True)
def sp_thought(task: str, reason: str | None = None, stage: str | None = None) -> str:
    """Compatibility alias for a Qwen near-match of sp_think.

    Args:
        task: The concise thought or checkpoint to record.
        reason: Why this checkpoint is useful.
        stage: Optional task stage.
    """
    return sp_think.invoke({"task": task, "reason": reason, "stage": stage})


@tool("sp_delegate", parse_docstring=True)
def sp_delegate(
    target_agent: Literal["researcher", "coder", "reporter", "outline", "perception"],
    task: str,
    reason: str | None = None,
    stage: str | None = None,
    input_refs: list[str] | None = None,
    expected_output: str | None = None,
    revision_reason: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Delegate a separable task to a StackPlanner specialist.

    Args:
        target_agent: Specialist role to invoke. Use reporter for report,
            Markdown-document, or prose synthesis; use researcher for read-only
            external HTTP/API retrieval; use coder only for code, local
            shell/data processing, or executable/generated-file work.
        task: Concrete delegated task.
        reason: Why delegation is needed.
        stage: Current task stage.
        input_refs: Artifact or memory references for the specialist.
        expected_output: Expected result shape.
        revision_reason: Exact user feedback or evidence that requires revising an existing artifact.
        metadata: Optional structured delegation metadata. Use skill_names to
            load matching Skills and tool_names to narrow the specialist Tool
            allowlist; pass source or artifact references through input_refs.
    """
    return "StackPlanner DELEGATE action recorded. Inspect the specialist result before continuing."


@tool("sp_recall_memory", parse_docstring=True)
def sp_recall_memory(query: str, reason: str | None = None, stage: str | None = None) -> str:
    """Recall relevant long-term memory through the StackPlanner memory handler.

    Args:
        query: Historical memory query; inspect short-term task context first.
        reason: Why recall is needed.
        stage: Current task stage.
    """
    return "StackPlanner RECALL_MEMORY action recorded. Use the returned memory context."


@tool("sp_reflect", parse_docstring=True)
def sp_reflect(
    task: str,
    reason: str | None = None,
    stage: str | None = None,
    target_entry_ids: list[str] | None = None,
) -> str:
    """Diagnose progress, evidence, or a failed action.

    Args:
        task: What should be inspected or diagnosed.
        reason: Why reflection is needed.
        stage: Current task stage.
        target_entry_ids: Exact active task-memory entry IDs proven invalid by
            the diagnosis. When supplied, those entries are immediately
            backtracked after the reflection is recorded.
    """
    return "StackPlanner REFLECT action recorded. Continue after incorporating the diagnosis."


@tool("sp_revise", parse_docstring=True)
def sp_revise(
    target_entry_ids: list[str],
    correction: str,
    reason: str,
    stage: str | None = None,
) -> str:
    """Correct erroneous central task-memory entries and continue.

    Args:
        target_entry_ids: IDs of active SP memory entries proven to be wrong.
        correction: The corrected fact, decision, or plan to retain.
        reason: Evidence explaining why the target entries are wrong.
        stage: Current task stage.
    """
    return "StackPlanner REVISE action recorded. The invalid memory was superseded; continue from the correction."


@tool("sp_backtrack", parse_docstring=True)
def sp_backtrack(
    target_type: Literal["entry", "stage", "artifact_version", "delegation"],
    target_id: str,
    reason: str,
    rollback_scope: Literal["memory_only", "artifact_refs", "delegation", "stage", "full_working_state"] = "memory_only",
) -> str:
    """Backtrack StackPlanner working state without deleting artifact history.

    Args:
        target_type: Type of checkpoint to backtrack to.
        target_id: Entry, stage, artifact version, or delegation identifier.
        reason: Why the rollback is required.
        rollback_scope: Portion of working state to restore.
    """
    return "StackPlanner BACKTRACK action recorded. Replan from the restored state."


@tool("sp_replan", parse_docstring=True)
def sp_replan(task: str, reason: str | None = None, stage: str | None = None) -> str:
    """Create a revised StackPlanner plan after new evidence or failure.

    Args:
        task: Revised plan or next plan objective.
        reason: Why replanning is needed.
        stage: Current task stage.
    """
    return "StackPlanner REPLAN action recorded. Continue with the revised plan."


@tool("sp_summarize", parse_docstring=True)
def sp_summarize(
    summary: str,
    source_entry_ids: list[str] | None = None,
    reason: str | None = None,
    stage: str | None = None,
) -> str:
    """Condense repetitive task memory at a StackPlanner stage boundary.

    Args:
        summary: Concise compressed task-memory summary, not a full report.
            Prefer goal, decisions, evidence or artifact refs, open issues,
            and the next action.
        source_entry_ids: Active task-memory entry IDs to pop into this summary.
            The CentralAgent may choose the exact number; if omitted, the
            handler keeps the newest four eligible entries and pops at most six
            older entries as a safety fallback. A later summary in the same run
            requires at least four meaningful new task-memory entries.
        reason: Why summarization is useful.
        stage: Current task stage.
    """
    return "StackPlanner SUMMARIZE action recorded. Continue with the condensed context."


@tool("sp_ask_human", parse_docstring=True, return_direct=True)
def sp_ask_human(
    question: str,
    reason: str | None = None,
    interaction_type: str = "clarification",
    options: list[str] | None = None,
) -> str:
    """Interrupt the run and request human feedback.

    Args:
        question: Question to present to the user.
        reason: Why human input is required.
        interaction_type: Type of interaction.
        options: Optional choices.
    """
    return "StackPlanner ASK_HUMAN action recorded. Wait for the user's response."


@tool("sp_finish", parse_docstring=True)
def sp_finish(
    summary: str,
    allow_without_artifact: bool = False,
    final_artifact_ref: str | None = None,
    required_artifact_type: str | None = None,
) -> str:
    """Mark the StackPlanner task ready to finish.

    Args:
        summary: Concise user-facing completion summary.
        allow_without_artifact: Set true only when no artifact is expected.
        final_artifact_ref: Exact artifact ID or path selected for delivery.
        required_artifact_type: Required artifact family, usually report or generated_file.
    """
    return "StackPlanner FINISH action recorded. Provide the final response."


def build_sp_control_tools() -> list[Any]:
    """Return only SP control tools; ordinary DeerFlow tools stay unchanged."""
    return [
        sp_think,
        sp_delegate,
        sp_recall_memory,
        sp_reflect,
        sp_revise,
        sp_backtrack,
        sp_replan,
        sp_summarize,
        sp_ask_human,
        sp_finish,
    ]


def build_sp_execution_tools() -> list[Any]:
    """Return canonical SP tools plus hidden execution-only compatibility aliases."""

    return [*build_sp_control_tools(), sp_thought]


class SPAcknowledgementMiddleware(AgentMiddleware[AgentState]):
    """End explicit record-only turns without launching a planning workflow.

    The TaskMemory middleware has already persisted the authoritative human
    message before this hook runs.  A deterministic acknowledgement therefore
    avoids both information loss and an unnecessary Central/subagent loop.
    """

    state_schema = AgentState

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        del runtime
        messages = state.get("messages") if isinstance(state, Mapping) else None
        if not isinstance(messages, list):
            return None
        latest = next(
            (
                message
                for message in reversed(messages)
                if isinstance(message, HumanMessage) and getattr(message, "name", None) != "sp_task_memory_context" and not (isinstance(message.additional_kwargs, Mapping) and message.additional_kwargs.get("hide_from_ui") is True)
            ),
            None,
        )
        if latest is None:
            return None
        text = message_to_text(latest).strip()
        if not text or not _ACKNOWLEDGEMENT_ONLY_PATTERN.search(text) or _ACKNOWLEDGEMENT_NEGATION_PATTERN.search(text) or _IMMEDIATE_DELIVERABLE_PATTERN.search(text):
            return None
        acknowledgement = "收到。" if re.search(r"[\u3400-\u9fff]", text) else "Acknowledged."
        return {
            "jump_to": "end",
            "messages": [
                AIMessage(
                    content=acknowledgement,
                    additional_kwargs={
                        "stackplanner": {
                            "action_type": ActionType.THINK.value,
                            "status": "acknowledged",
                            "fast_path": True,
                        }
                    },
                )
            ],
        }

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        return self.before_model(state, runtime)


class SPTerminalActionMiddleware(AgentMiddleware[AgentState]):
    """Exit the native agent loop only after FINISH passed handler validation."""

    state_schema = AgentState

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        del runtime
        result = state.get("sp_last_handler_result") if isinstance(state, Mapping) else None
        if isinstance(result, Mapping) and result.get("next_step") == "finish":
            return {"jump_to": "end"}
        return None

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        return self.before_model(state, runtime)


class SPFinishAvailabilityMiddleware(AgentMiddleware[AgentState]):
    """Expose FINISH only after a user-deliverable artifact actually exists.

    Ordinary conversational answers end naturally when Central emits text, so
    advertising ``sp_finish`` before an artifact exists only invites invalid
    artifact references and retry loops.  ToolNode still owns the tool for
    replaying existing calls; this middleware only removes its schema from the
    next model binding.
    """

    state_schema = AgentState

    @staticmethod
    def _latest_visible_human_message(state: Mapping[str, Any]) -> HumanMessage | None:
        messages = state.get("messages")
        if not isinstance(messages, list):
            return None
        return next(
            (
                message
                for message in reversed(messages)
                if isinstance(message, HumanMessage) and getattr(message, "name", None) != "sp_task_memory_context" and not (isinstance(message.additional_kwargs, Mapping) and message.additional_kwargs.get("hide_from_ui") is True)
            ),
            None,
        )

    @classmethod
    def _requires_artifact_delivery(cls, state: Mapping[str, Any]) -> bool:
        latest = cls._latest_visible_human_message(state)
        return bool(latest and _ARTIFACT_DELIVERABLE_REQUEST_PATTERN.search(message_to_text(latest)))

    @classmethod
    def _requires_fresh_revision_artifact(cls, state: Mapping[str, Any]) -> bool:
        latest = cls._latest_visible_human_message(state)
        return bool(latest and _ARTIFACT_REVISION_REQUEST_PATTERN.search(message_to_text(latest)))

    @staticmethod
    def _artifact_candidates(state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        refs = state.get("sp_current_artifact_refs")
        if not isinstance(refs, Mapping):
            return []
        candidates: list[Mapping[str, Any]] = []
        candidates.extend(item for key, item in refs.items() if key != "_history" and isinstance(item, Mapping))
        history = refs.get("_history")
        if isinstance(history, list):
            candidates.extend(reversed([item for item in history if isinstance(item, Mapping)]))
        return candidates

    @classmethod
    def _has_finalizable_artifact(cls, state: Mapping[str, Any]) -> bool:
        candidates = cls._artifact_candidates(state)
        requires_fresh_artifact = cls._requires_fresh_revision_artifact(state)
        current_run_id = str(state.get("sp_loop_run_id") or "").strip()
        for ref in candidates:
            if not _artifact_is_complete(ref):
                continue
            if requires_fresh_artifact and (not current_run_id or str(ref.get("run_id") or "") != current_run_id):
                continue
            artifact_type = str(ref.get("type") or "")
            if artifact_type in _INTERMEDIATE_ARTIFACT_TYPES:
                continue
            if artifact_type in _REPORT_ARTIFACT_TYPES and not ref.get("is_current", True):
                continue
            if ref.get("artifact_id") or ref.get("virtual_path") or ref.get("artifact_url"):
                return True
        return False

    @staticmethod
    def _incomplete_current_report(
        state: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        refs = state.get("sp_current_artifact_refs")
        if not isinstance(refs, Mapping):
            return None
        current_run_id = str(state.get("sp_loop_run_id") or "").strip()
        for key, ref in refs.items():
            if key == "_history" or not isinstance(ref, Mapping) or str(ref.get("type") or "") not in _REPORT_ARTIFACT_TYPES or not ref.get("is_current", True) or _artifact_is_complete(ref):
                continue
            if current_run_id and ref.get("run_id") and str(ref.get("run_id")) != current_run_id:
                continue
            return ref
        return None

    @staticmethod
    def _reporter_attempt_count(state: Mapping[str, Any]) -> int:
        return SPFinishAvailabilityMiddleware._delegate_attempt_count(
            state,
            "reporter",
        )

    @staticmethod
    def _delegate_attempt_count(
        state: Mapping[str, Any],
        target_agent: str,
    ) -> int:
        stack = TaskMemoryStack.from_dict(state.get("sp_task_memory"))
        current_run_id = str(state.get("sp_loop_run_id") or "").strip()
        return sum(1 for entry in stack.entries if entry.action == "delegate" and entry.metadata.get("target_agent") == target_agent and (not current_run_id or str(entry.run_id or "") == current_run_id))

    @staticmethod
    def _latest_delegate_observation(
        state: Mapping[str, Any],
        *,
        target_agent: str | None = None,
    ) -> StackMemoryEntry | None:
        stack = TaskMemoryStack.from_dict(state.get("sp_task_memory"))
        current_run_id = str(state.get("sp_loop_run_id") or "").strip()
        for entry in reversed(stack.entries):
            if entry.action != "observe" or entry.actor not in _SP_SPECIALIST_ROLES or (target_agent is not None and entry.actor != target_agent):
                continue
            if current_run_id and str(entry.run_id or "") != current_run_id:
                continue
            return entry
        return None

    @classmethod
    def _latest_partial_report_observation(
        cls,
        state: Mapping[str, Any],
    ) -> StackMemoryEntry | None:
        entry = cls._latest_delegate_observation(
            state,
            target_agent="reporter",
        )
        if entry is None:
            return None
        status = str(entry.metadata.get("completion_status") or "").strip().lower()
        return entry if status in {"partial", "blocked"} else None

    @staticmethod
    def _pending_perception_questions(
        state: Mapping[str, Any],
    ) -> list[str]:
        previous = state.get("sp_last_handler_result")
        if not (isinstance(previous, Mapping) and previous.get("action_type") == ActionType.DELEGATE.value and previous.get("target_agent") == "perception"):
            return []
        refs = state.get("sp_current_artifact_refs")
        perception = refs.get("perception_observation") if isinstance(refs, Mapping) else None
        if not isinstance(perception, Mapping):
            return []
        current_run_id = str(state.get("sp_loop_run_id") or "").strip()
        if current_run_id and str(perception.get("run_id") or "") != current_run_id:
            return []
        metadata = perception.get("metadata")
        if not isinstance(metadata, Mapping):
            return []
        questions = metadata.get("clarification_questions")
        normalized = [str(value).strip() for value in questions if str(value).strip()] if isinstance(questions, list) else []
        if metadata.get("task_ready") is False and not normalized:
            normalized = ["Please provide the decision-critical missing information identified in the task brief."]
        return normalized[:5]

    @classmethod
    def _completed_report_from_previous_delegate(
        cls,
        state: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        previous = state.get("sp_last_handler_result")
        if not (isinstance(previous, Mapping) and previous.get("action_type") == ActionType.DELEGATE.value and previous.get("target_agent") == "reporter"):
            return None
        action_id = str(previous.get("action_id") or "")
        current_run_id = str(state.get("sp_loop_run_id") or "")
        for ref in cls._artifact_candidates(state):
            if str(ref.get("type") or "") not in _REPORT_ARTIFACT_TYPES or not ref.get("is_current", True) or not _artifact_is_complete(ref):
                continue
            metadata = ref.get("metadata")
            if not isinstance(metadata, Mapping):
                continue
            if action_id and metadata.get("delegate_action_id") != action_id:
                continue
            if current_run_id and str(ref.get("run_id") or "") != current_run_id:
                continue
            return ref
        return None

    @classmethod
    def _completed_deliverable_from_previous_delegate(
        cls,
        state: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        previous = state.get("sp_last_handler_result")
        if not (isinstance(previous, Mapping) and previous.get("action_type") == ActionType.DELEGATE.value and previous.get("target_agent") != "reporter"):
            return None
        action_id = str(previous.get("action_id") or "")
        current_run_id = str(state.get("sp_loop_run_id") or "")
        for ref in cls._artifact_candidates(state):
            artifact_type = str(ref.get("type") or "")
            if artifact_type in _INTERMEDIATE_ARTIFACT_TYPES or artifact_type in _REPORT_ARTIFACT_TYPES or not _artifact_is_complete(ref):
                continue
            metadata = ref.get("metadata")
            if not isinstance(metadata, Mapping):
                continue
            if action_id and metadata.get("delegate_action_id") != action_id:
                continue
            if current_run_id and str(ref.get("run_id") or "") != current_run_id:
                continue
            if ref.get("artifact_id") or ref.get("virtual_path") or ref.get("artifact_url"):
                return ref
        return None

    @classmethod
    def _artifact_recovery_instruction(cls, state: Mapping[str, Any]) -> str:
        partial_observation = cls._latest_partial_report_observation(state)
        incomplete_report = cls._incomplete_current_report(state)
        latest_report = next(
            (ref for ref in cls._artifact_candidates(state) if str(ref.get("type") or "") in _REPORT_ARTIFACT_TYPES and ref.get("is_current", True)),
            None,
        )
        if incomplete_report is not None or partial_observation is not None:
            recovery_report = incomplete_report or latest_report
            artifact_id = str(recovery_report.get("artifact_id") or "").strip() if recovery_report is not None else ""
            metadata = incomplete_report.get("metadata") if incomplete_report is not None else None
            raw_gaps = metadata.get("evidence_gaps") if isinstance(metadata, Mapping) else None
            gaps = [str(value).strip() for value in raw_gaps or [] if str(value).strip()]
            if not gaps and partial_observation is not None:
                observation_gaps = partial_observation.metadata.get("evidence_gaps")
                if isinstance(observation_gaps, list):
                    gaps = [str(value).strip() for value in observation_gaps if str(value).strip()]
                elif partial_observation.failure_note:
                    gaps = [partial_observation.failure_note]
            gap_text = "; ".join(gaps) if gaps else "the report quality gate is still incomplete"
            input_ref_text = f'input_refs=["{artifact_id}"], ' if artifact_id else ""
            return (
                "<sp-artifact-recovery-required>\n"
                "The previous report attempt cannot be finalized because the current report "
                f"is incomplete: {gap_text}. Call sp_delegate exactly once "
                'with target_agent="reporter", stage="revision", '
                f"{input_ref_text}and a non-empty revision_reason that resolves "
                "every listed evidence gap. Preserve all unaffected content "
                "and authoritative literals from the current report. The "
                "reporter must either successfully call write_file or return "
                "the complete revised Markdown in artifact_content. Do not "
                "call FINISH, summarize, or invent a download URL in this turn.\n"
                "</sp-artifact-recovery-required>"
            )

        latest_observation = cls._latest_delegate_observation(state)
        if latest_observation is not None:
            completion_status = str(latest_observation.metadata.get("completion_status") or "").strip().lower()
            target = str(latest_observation.metadata.get("target_agent") or latest_observation.actor or "")
            if completion_status in {"partial", "blocked"} and target in _SP_SPECIALIST_ROLES and target != "reporter":
                raw_gaps = latest_observation.metadata.get("evidence_gaps")
                gaps = [str(value).strip() for value in raw_gaps or [] if str(value).strip()]
                if not gaps and latest_observation.failure_note:
                    gaps = [latest_observation.failure_note]
                gap_text = "; ".join(gaps[:5]) or ("the delegated acceptance criteria remain incomplete")
                stage = {
                    "coder": "implementation",
                    "researcher": "research",
                    "outline": "planning",
                    "perception": "perception",
                }.get(target, "implementation")
                coder_instruction = (
                    " For coder recovery, include read_file, write_file, "
                    "str_replace, and bash in metadata.tool_names; inspect the "
                    "current file, edit source with a file tool, and run tests "
                    "in a separate bash call. Never redirect program stdout "
                    "into the source file or merely rerun the same failing "
                    "command."
                    if target == "coder"
                    else ""
                )
                return (
                    "<sp-artifact-recovery-required>\n"
                    f"The previous {target} result is {completion_status}: "
                    f"{gap_text}. Call sp_delegate exactly once with "
                    f'target_agent="{target}", stage="{stage}", and a '
                    "materially different recovery method that closes every "
                    f"listed gap.{coder_instruction} Do not call FINISH, "
                    "summarize, or claim success in this turn.\n"
                    "</sp-artifact-recovery-required>"
                )

        latest = cls._latest_visible_human_message(state)
        request_text = message_to_text(latest) if latest is not None else ""
        target = "coder" if _CODE_OR_APP_DELIVERABLE_PATTERN.search(request_text) else "reporter"
        stage = "implementation" if target == "coder" else "reporting"
        role = "executable/code/file" if target == "coder" else "report/Markdown"
        return (
            "<sp-artifact-recovery-required>\n"
            "The previous FINISH was rejected because no complete downloadable "
            f"artifact exists. Call sp_delegate exactly once with target_agent="
            f'"{target}" and stage="{stage}" to create the requested {role} '
            "deliverable from the authoritative task context. Do not call "
            "FINISH, summarize, or invent a download URL in this turn.\n"
            "</sp-artifact-recovery-required>"
        )

    def _filter_request(self, request: ModelRequest) -> ModelRequest:
        state = request.state if isinstance(request.state, Mapping) else {}
        previous = state.get("sp_last_handler_result")
        clarification_questions = self._pending_perception_questions(state)
        if clarification_questions:
            active_tools = [candidate for candidate in request.tools if getattr(candidate, "name", None) == "sp_ask_human"]
            messages = list(request.messages)
            if not any(getattr(message, "name", None) == _SP_CLARIFICATION_REQUIRED_MESSAGE_NAME for message in messages):
                questions = "\n".join(
                    f"{index}. {question}"
                    for index, question in enumerate(
                        clarification_questions,
                        start=1,
                    )
                )
                latest_visible = self._latest_visible_human_message(state)
                if latest_visible is None:
                    latest_visible = self._latest_visible_human_message({"messages": request.messages})
                language_instruction = " The latest user wrote in Chinese, so ask in Chinese." if latest_visible is not None and _CJK_TEXT_PATTERN.search(message_to_text(latest_visible)) else " Match the latest visible user's language."
                messages.append(
                    HumanMessage(
                        name=_SP_CLARIFICATION_REQUIRED_MESSAGE_NAME,
                        content=(
                            "<sp-clarification-required>\n"
                            "Perception determined that execution is not ready. "
                            "Call sp_ask_human exactly once now and combine the "
                            "following missing-information questions into one "
                            "concise prompt. Do not delegate, search, plan, or "
                            "invent project facts first."
                            f"{language_instruction}\n"
                            f"{questions}\n"
                            "</sp-clarification-required>"
                        ),
                        additional_kwargs={"hide_from_ui": True},
                    )
                )
            return request.override(tools=active_tools, messages=messages)
        completed_report = self._completed_report_from_previous_delegate(state)
        if completed_report is not None:
            active_tools = [candidate for candidate in request.tools if getattr(candidate, "name", None) == "sp_finish"]
            messages = list(request.messages)
            if not any(getattr(message, "name", None) == _SP_REPORT_FINALIZATION_MESSAGE_NAME for message in messages):
                artifact_id = str(completed_report.get("artifact_id") or completed_report.get("virtual_path") or "")
                messages.append(
                    HumanMessage(
                        name=_SP_REPORT_FINALIZATION_MESSAGE_NAME,
                        content=(
                            "<sp-report-finalization-required>\n"
                            "The reporter produced a complete current-run "
                            "report that passed acceptance checks. Call "
                            "sp_finish exactly once now with "
                            f'final_artifact_ref="{artifact_id}" and '
                            'required_artifact_type="report". Pass the concise '
                            "user-facing completion text in `summary` (not "
                            "`task`) and match the latest visible user's "
                            "language. Summarize the actual deliverable; do "
                            "not ask another question or delegate more work.\n"
                            "</sp-report-finalization-required>"
                        ),
                        additional_kwargs={"hide_from_ui": True},
                    )
                )
            return request.override(tools=active_tools, messages=messages)
        completed_deliverable = self._completed_deliverable_from_previous_delegate(state)
        if completed_deliverable is not None:
            active_tools = [candidate for candidate in request.tools if getattr(candidate, "name", None) == "sp_finish"]
            messages = list(request.messages)
            if not any(getattr(message, "name", None) == _SP_ARTIFACT_FINALIZATION_MESSAGE_NAME for message in messages):
                artifact_id = str(completed_deliverable.get("artifact_id") or completed_deliverable.get("virtual_path") or completed_deliverable.get("artifact_url") or "")
                artifact_type = str(completed_deliverable.get("type") or "generated_file")
                metadata = completed_deliverable.get("metadata")
                preview = metadata.get("finalization_preview") if isinstance(metadata, Mapping) and not metadata.get("finalization_preview_truncated") else None
                preview_instruction = (
                    f"\nThe following artifact preview is authoritative. Make the summary agree exactly with it and ignore stale prior conclusions:\n<artifact-finalization-preview>\n{preview}\n</artifact-finalization-preview>\n"
                    if isinstance(preview, str) and preview.strip()
                    else ""
                )
                messages.append(
                    HumanMessage(
                        name=_SP_ARTIFACT_FINALIZATION_MESSAGE_NAME,
                        content=(
                            "<sp-artifact-finalization-required>\n"
                            "The specialist produced a complete current-run "
                            "downloadable artifact. Call sp_finish exactly "
                            "once now with "
                            f'final_artifact_ref="{artifact_id}" and '
                            f'required_artifact_type="{artifact_type}". Pass '
                            "a concise user-facing completion text in "
                            "`summary`, match the latest visible user's "
                            "language, do not ask another question, and do "
                            f"not delegate more work.{preview_instruction}\n"
                            "</sp-artifact-finalization-required>"
                        ),
                        additional_kwargs={"hide_from_ui": True},
                    )
                )
            return request.override(tools=active_tools, messages=messages)
        latest_delegate_observation = self._latest_delegate_observation(state)
        latest_delegate_status = str(latest_delegate_observation.metadata.get("completion_status") or "").strip().lower() if latest_delegate_observation is not None else ""
        requires_artifact = self._requires_artifact_delivery(state)
        if latest_delegate_observation is not None and latest_delegate_observation.actor != "reporter" and latest_delegate_status == "complete" and not requires_artifact:
            active_tools = [candidate for candidate in request.tools if getattr(candidate, "name", None) == "sp_finish"]
            messages = list(request.messages)
            if not any(getattr(message, "name", None) == _SP_CONVERSATIONAL_FINALIZATION_MESSAGE_NAME for message in messages):
                messages.append(
                    HumanMessage(
                        name=_SP_CONVERSATIONAL_FINALIZATION_MESSAGE_NAME,
                        content=(
                            "<sp-conversational-finalization-required>\n"
                            "The specialist completed the requested non-file work. "
                            "Call sp_finish exactly once now with "
                            "allow_without_artifact=true and put a concise, "
                            "user-facing answer grounded in the latest specialist "
                            "observation in `summary`. Do not think, summarize, "
                            "delegate, or call another tool.\n"
                            "</sp-conversational-finalization-required>"
                        ),
                        additional_kwargs={"hide_from_ui": True},
                    )
                )
            return request.override(tools=active_tools, messages=messages)
        finish_rejected = isinstance(previous, Mapping) and previous.get("action_type") == ActionType.FINISH.value and previous.get("next_step") == "error_recoverable"
        has_finalizable_artifact = self._has_finalizable_artifact(state)
        incomplete_current_report = self._incomplete_current_report(state)
        delegate_incomplete = isinstance(previous, Mapping) and previous.get("action_type") == ActionType.DELEGATE.value and latest_delegate_status in {"partial", "blocked"}
        reporter_attempts = self._reporter_attempt_count(state)
        if incomplete_current_report is not None and reporter_attempts >= SP_REPORT_RECOVERY_MAX_ATTEMPTS:
            active_tools = [candidate for candidate in request.tools if getattr(candidate, "name", None) == "sp_ask_human"]
            messages = list(request.messages)
            if not any(getattr(message, "name", None) == _SP_REPORT_RECOVERY_EXHAUSTED_MESSAGE_NAME for message in messages):
                metadata = incomplete_current_report.get("metadata")
                raw_gaps = metadata.get("evidence_gaps") if isinstance(metadata, Mapping) else None
                gaps = [str(value).strip() for value in raw_gaps or [] if str(value).strip()]
                gap_text = "; ".join(gaps[:5]) or ("the report still fails its acceptance checks")
                messages.append(
                    HumanMessage(
                        name=_SP_REPORT_RECOVERY_EXHAUSTED_MESSAGE_NAME,
                        content=(
                            "<sp-report-recovery-exhausted>\n"
                            "The report reached the bounded recovery limit "
                            f"({SP_REPORT_RECOVERY_MAX_ATTEMPTS} reporter "
                            "attempts) "
                            "and must not be delegated again in this run. "
                            "Call sp_ask_human exactly once, briefly explain "
                            "that the latest partial artifact was retained, "
                            "state the unresolved quality gap, and ask for "
                            "the minimum decision or evidence needed to "
                            "continue. Match the latest visible user's "
                            f"language. Unresolved gap: {gap_text}\n"
                            "</sp-report-recovery-exhausted>"
                        ),
                        additional_kwargs={"hide_from_ui": True},
                    )
                )
            return request.override(tools=active_tools, messages=messages)
        if latest_delegate_observation is not None and latest_delegate_observation.actor != "reporter" and latest_delegate_status in {"partial", "blocked"}:
            target_agent = str(latest_delegate_observation.metadata.get("target_agent") or latest_delegate_observation.actor)
            delegate_attempts = self._delegate_attempt_count(
                state,
                target_agent,
            )
            if delegate_attempts >= SP_DELEGATE_RECOVERY_MAX_ATTEMPTS:
                active_tools = [candidate for candidate in request.tools if getattr(candidate, "name", None) not in SP_CONTROL_TOOL_NAMES]
                messages = list(request.messages)
                if not any(getattr(message, "name", None) == _SP_DELEGATE_RECOVERY_EXHAUSTED_MESSAGE_NAME for message in messages):
                    raw_gaps = latest_delegate_observation.metadata.get("evidence_gaps")
                    gaps = [str(value).strip() for value in raw_gaps or [] if str(value).strip()]
                    if not gaps and latest_delegate_observation.failure_note:
                        gaps = [latest_delegate_observation.failure_note]
                    gap_text = "; ".join(gaps[:5]) or ("the delegated acceptance criteria remain incomplete")
                    messages.append(
                        HumanMessage(
                            name=_SP_DELEGATE_RECOVERY_EXHAUSTED_MESSAGE_NAME,
                            content=(
                                "<sp-delegate-recovery-exhausted>\n"
                                f"The run reached its bounded recovery limit "
                                f"({delegate_attempts} {target_agent} attempts) "
                                "and must not delegate again in this run. "
                                "Answer the current user directly now. Briefly "
                                "state what remains unresolved and the smallest "
                                "useful next step; do not claim that a verified "
                                "file is ready or invent an artifact/download "
                                f"URL. Unresolved gap: {gap_text}\n"
                                "</sp-delegate-recovery-exhausted>"
                            ),
                            additional_kwargs={"hide_from_ui": True},
                        )
                    )
                return request.override(
                    tools=active_tools,
                    messages=messages,
                )
            active_tools = [candidate for candidate in request.tools if getattr(candidate, "name", None) == "sp_delegate"]
            messages = list(request.messages)
            if not any(getattr(message, "name", None) == _SP_ARTIFACT_RECOVERY_MESSAGE_NAME for message in messages):
                messages.append(
                    HumanMessage(
                        name=_SP_ARTIFACT_RECOVERY_MESSAGE_NAME,
                        content=self._artifact_recovery_instruction(state),
                        additional_kwargs={"hide_from_ui": True},
                    )
                )
            return request.override(tools=active_tools, messages=messages)
        direct_answer_recovery = finish_rejected and not has_finalizable_artifact and not requires_artifact
        if direct_answer_recovery:
            active_tools = [candidate for candidate in request.tools if getattr(candidate, "name", None) not in SP_CONTROL_TOOL_NAMES]
            messages = list(request.messages)
            if not any(getattr(message, "name", None) == _SP_DIRECT_ANSWER_MESSAGE_NAME for message in messages):
                messages.append(
                    HumanMessage(
                        name=_SP_DIRECT_ANSWER_MESSAGE_NAME,
                        content=(
                            "<sp-direct-answer-required>\n"
                            "The previous FINISH was rejected because no "
                            "user-deliverable artifact exists. Do not call any "
                            "action or tool. Answer the current user directly "
                            "now, reconciling every authoritative constraint "
                            "and the exact requested output shape.\n"
                            "</sp-direct-answer-required>"
                        ),
                        additional_kwargs={"hide_from_ui": True},
                    )
                )
            return request.override(tools=active_tools, messages=messages)
        artifact_recovery = incomplete_current_report is not None or ((finish_rejected or delegate_incomplete) and not has_finalizable_artifact and requires_artifact)
        if artifact_recovery:
            active_tools = [candidate for candidate in request.tools if getattr(candidate, "name", None) == "sp_delegate"]
            messages = list(request.messages)
            if not any(getattr(message, "name", None) == _SP_ARTIFACT_RECOVERY_MESSAGE_NAME for message in messages):
                messages.append(
                    HumanMessage(
                        name=_SP_ARTIFACT_RECOVERY_MESSAGE_NAME,
                        content=self._artifact_recovery_instruction(state),
                        additional_kwargs={"hide_from_ui": True},
                    )
                )
            return request.override(tools=active_tools, messages=messages)
        if has_finalizable_artifact:
            return request
        active_tools = [candidate for candidate in request.tools if getattr(candidate, "name", None) != "sp_finish"]
        if len(active_tools) == len(request.tools):
            return request
        return request.override(tools=active_tools)

    @staticmethod
    def _guarded_message_kwargs(
        message: AIMessage,
        *,
        status: str,
        target_agent: str,
    ) -> dict[str, Any]:
        additional_kwargs = {key: value for key, value in message.additional_kwargs.items() if key not in {"function_call", "tool_calls"}}
        additional_kwargs.update(
            {
                "stackplanner_generated": True,
                "stackplanner": {
                    "action_type": ActionType.DELEGATE.value,
                    "status": status,
                    "target_agent": target_agent,
                    "guard": "unverified_artifact_terminal_claim",
                },
            }
        )
        return additional_kwargs

    @classmethod
    def _after_model_update(
        cls,
        state: AgentState,
        runtime: Any,
    ) -> dict[str, Any] | None:
        del runtime
        if not isinstance(state, Mapping) or not cls._requires_artifact_delivery(state):
            return None
        if cls._has_finalizable_artifact(state):
            return None

        messages = state.get("messages")
        latest = messages[-1] if isinstance(messages, list) and messages else None
        if not isinstance(latest, AIMessage) or latest.tool_calls or not message_to_text(latest).strip():
            return None

        observation = cls._latest_delegate_observation(state)
        if observation is None or observation.actor == "reporter":
            return None
        completion_status = str(observation.metadata.get("completion_status") or "").strip().lower()
        if completion_status not in {"partial", "blocked"}:
            return None

        target_agent = str(observation.metadata.get("target_agent") or observation.actor).strip()
        if target_agent not in _SP_SPECIALIST_ROLES:
            return None
        attempts = cls._delegate_attempt_count(state, target_agent)
        raw_gaps = observation.metadata.get("evidence_gaps")
        gaps = [str(value).strip() for value in raw_gaps or [] if str(value).strip()]
        if not gaps and observation.failure_note:
            gaps = [observation.failure_note]
        gap_text = "; ".join(gaps[:5]) or "the delegated acceptance criteria remain incomplete"
        latest_human = cls._latest_visible_human_message(state)
        use_chinese = latest_human is not None and bool(_CJK_TEXT_PATTERN.search(message_to_text(latest_human)))
        message_id = str(latest.id or _implicit_think_message_id(latest))

        if attempts >= SP_DELEGATE_RECOVERY_MAX_ATTEMPTS:
            if use_chinese:
                content = f"当前产物尚未通过验证，本轮已达到 {attempts} 次恢复上限。未解决问题：{gap_text}。我不会把当前文件标记为可用；请稍后重试或缩小修复范围。"
            else:
                content = f"The current artifact is still unverified after {attempts} recovery attempts. Unresolved issue: {gap_text}. I will not present the current file as usable; please retry later or narrow the repair scope."
            guarded = latest.model_copy(
                update={
                    "id": message_id,
                    "content": content,
                    "tool_calls": [],
                    "invalid_tool_calls": [],
                    "additional_kwargs": cls._guarded_message_kwargs(
                        latest,
                        status="incomplete",
                        target_agent=target_agent,
                    ),
                }
            )
            return {"messages": [guarded]}

        stage = {
            "coder": "implementation",
            "researcher": "research",
            "outline": "planning",
            "perception": "perception",
        }.get(target_agent, "implementation")
        input_refs = [observation.result_ref] if observation.result_ref else []
        task = (
            f"Recover the incomplete {target_agent} deliverable using a materially "
            f"different method. Resolve every remaining evidence gap: {gap_text}. "
            "Inspect and preserve valid existing work. Do not claim success unless "
            "all requested acceptance checks actually pass."
        )
        metadata: dict[str, Any] = {
            "recovery_guard": "unverified_artifact_terminal_claim",
            "recovery_attempt": attempts + 1,
        }
        if target_agent == "coder":
            metadata["tool_names"] = [
                "read_file",
                "write_file",
                "str_replace",
                "bash",
            ]
            task += " For code, inspect the current source, edit it with a file tool, then run every old and new test in a separate bash call after the latest source change. Never redirect program output into source."
        call_seed = f"{message_id}:{target_agent}:{attempts + 1}"
        call_id = f"sp-recovery-{hashlib.sha256(call_seed.encode('utf-8')).hexdigest()[:16]}"
        guarded = latest.model_copy(
            update={
                "id": message_id,
                "content": "",
                "tool_calls": [
                    {
                        "name": "sp_delegate",
                        "args": {
                            "target_agent": target_agent,
                            "task": task,
                            "reason": ("The current artifact remains incomplete and the CentralAgent attempted to terminate without verification."),
                            "stage": stage,
                            "input_refs": input_refs,
                            "expected_output": ("A complete, verified, downloadable artifact with concrete execution or acceptance evidence."),
                            "revision_reason": gap_text,
                            "metadata": metadata,
                        },
                        "id": call_id,
                        "type": "tool_call",
                    }
                ],
                "invalid_tool_calls": [],
                "additional_kwargs": cls._guarded_message_kwargs(
                    latest,
                    status="recovery_required",
                    target_agent=target_agent,
                ),
            }
        )
        return {"messages": [guarded]}

    def after_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        return self._after_model_update(state, runtime)

    async def aafter_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        return self._after_model_update(state, runtime)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        return handler(self._filter_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        return await handler(self._filter_request(request))


class SPExecutorProvider(Protocol):
    def __call__(self, state: Mapping[str, Any], runtime: Any) -> SPSubagentExecutorProtocol | None:
        """Build an executor for the current DR2 agent turn."""


def _run_id(runtime: Any) -> str | None:
    context = getattr(runtime, "context", None)
    if isinstance(context, Mapping) and context.get("run_id"):
        return str(context["run_id"])
    return None


def _thread_id(runtime: Any) -> str | None:
    context = getattr(runtime, "context", None)
    if isinstance(context, Mapping) and context.get("thread_id"):
        return str(context["thread_id"])
    return None


def _has_parallel_nonterminal_sp_action(
    state: Mapping[str, Any],
    *,
    current_tool_call_id: str,
) -> bool:
    """Return whether the current FINISH has a sibling action still doing work."""
    messages = state.get("messages")
    if not isinstance(messages, list):
        return False
    latest_ai = next((message for message in reversed(messages) if isinstance(message, AIMessage)), None)
    if latest_ai is None:
        return False
    return any(str(call.get("id") or "") != current_tool_call_id and str(call.get("name") or "") in SP_CONTROL_TOOL_NAMES and str(call.get("name") or "") != "sp_finish" for call in latest_ai.tool_calls)


def _summary_model_turn_id(state: Mapping[str, Any]) -> str:
    """Return one stable key shared by sibling tool calls from a model turn."""
    messages = state.get("messages")
    if isinstance(messages, list):
        latest_ai = next((message for message in reversed(messages) if isinstance(message, AIMessage)), None)
        if latest_ai is not None:
            if latest_ai.id:
                return str(latest_ai.id)
            tool_call_ids = sorted(str(call.get("id") or "") for call in latest_ai.tool_calls if call.get("id"))
            if tool_call_ids:
                return hashlib.sha256("|".join(tool_call_ids).encode("utf-8")).hexdigest()[:20]
    return f"iteration:{int(state.get('sp_loop_iteration') or 0)}"


def _final_artifact_paths(
    state: Mapping[str, Any],
    state_update: Mapping[str, Any],
    *,
    run_id: str | None,
) -> list[str]:
    """Resolve every current-run user-downloadable artifact for SP FINISH."""
    refs = state_update.get("sp_current_artifact_refs") or state.get("sp_current_artifact_refs")
    if not isinstance(refs, Mapping):
        return []

    candidates: list[Mapping[str, Any]] = []
    history = refs.get("_history")
    if isinstance(history, list):
        candidates.extend(item for item in history if isinstance(item, Mapping))
    candidates.extend(item for key, item in refs.items() if key != "_history" and isinstance(item, Mapping))

    paths: list[str] = []
    seen_artifacts: set[str] = set()
    for ref in candidates:
        if not _artifact_is_complete(ref):
            continue
        artifact_type = str(ref.get("type") or "")
        if artifact_type in _INTERMEDIATE_ARTIFACT_TYPES:
            continue
        if artifact_type in _REPORT_ARTIFACT_TYPES and not ref.get("is_current", True):
            continue
        ref_run_id = ref.get("run_id")
        if run_id and ref_run_id and str(ref_run_id) != run_id:
            continue
        identity = str(ref.get("artifact_id") or ref.get("virtual_path") or "")
        if identity and identity in seen_artifacts:
            continue
        path = ref.get("virtual_path")
        if isinstance(path, str) and path.startswith("/mnt/user-data/outputs/"):
            paths.append(path)
            if identity:
                seen_artifacts.add(identity)
    return list(dict.fromkeys(paths))


def _delegate_artifact_paths(state_update: Mapping[str, Any]) -> list[str]:
    """Return files created by this delegation that the UI may download."""
    paths = state_update.get("artifacts")
    if not isinstance(paths, list):
        return []
    return list(dict.fromkeys(path for path in paths if isinstance(path, str) and path.startswith("/mnt/user-data/outputs/")))


def _normalize_delegate_target(payload: dict[str, Any]) -> tuple[str, str | None]:
    """Repair an unambiguous report/coder role mismatch at the action boundary."""
    target_agent = str(payload.get("target_agent") or "")
    if target_agent != "coder":
        return target_agent, None
    text = "\n".join(str(value) for key in ("task", "expected_output", "reason") if (value := payload.get(key)) not in (None, ""))
    if not _PURE_REPORT_DELEGATION_PATTERN.search(text):
        return target_agent, None
    if _CODE_OR_APP_DELIVERABLE_PATTERN.search(text):
        return target_agent, None
    payload["target_agent"] = "reporter"
    return "reporter", "pure_report_synthesis"


def _infer_coder_stage(payload: Mapping[str, Any]) -> str | None:
    """Recognize read-only file inspection when Central omitted its stage."""
    metadata = payload.get("metadata")
    tool_names = metadata.get("tool_names") if isinstance(metadata, Mapping) else None
    if isinstance(tool_names, list) and tool_names:
        normalized = {str(value).strip() for value in tool_names if isinstance(value, str) and value.strip()}
        if normalized and normalized <= {"read_file", "grep", "glob", "ls"}:
            return "perception"
    text = "\n".join(str(value) for key in ("task", "expected_output") if (value := payload.get(key)) not in (None, ""))
    if _CODER_READ_ONLY_PATTERN.search(text) and not _CODER_MUTATION_PATTERN.search(text):
        return "perception"
    return None


def _action_payload(tool_name: str, args: Mapping[str, Any], *, tool_call_id: str) -> dict[str, Any]:
    mapping = {
        "sp_think": ActionType.THINK,
        "sp_thought": ActionType.THINK,
        "sp_delegate": ActionType.DELEGATE,
        "sp_recall_memory": ActionType.RECALL_MEMORY,
        "sp_reflect": ActionType.REFLECT,
        "sp_revise": ActionType.REVISE,
        "sp_backtrack": ActionType.BACKTRACK,
        "sp_replan": ActionType.REPLAN,
        "sp_summarize": ActionType.SUMMARIZE,
        "sp_ask_human": ActionType.ASK_HUMAN,
        "sp_finish": ActionType.FINISH,
    }
    action_type = mapping[tool_name]
    payload = dict(args)
    payload["action_id"] = str(payload.get("action_id") or f"spact_{tool_call_id}")
    payload["action_type"] = action_type.value
    payload["idempotency_key"] = str(payload.get("idempotency_key") or f"spidem_{tool_call_id}")
    payload["reason"] = str(payload.get("reason") or f"CentralAgent selected {action_type.value}")
    if tool_name == "sp_delegate":
        revision_reason = payload.pop("revision_reason", None)
        target_agent, routing_reason = _normalize_delegate_target(payload)
        requested_stage = payload.get("stage")
        canonical_stage = {
            "perception": "perception",
            "researcher": "research",
        }.get(target_agent)
        if target_agent == "outline":
            canonical_stage = "revision" if revision_reason or requested_stage == "revision" else "planning"
        if target_agent == "reporter":
            canonical_stage = "revision" if revision_reason or requested_stage == "revision" else "reporting"
        if target_agent == "coder" and requested_stage is None:
            canonical_stage = _infer_coder_stage(payload)
        if canonical_stage is not None:
            payload["stage"] = canonical_stage
        if not revision_reason and target_agent in {"reporter", "outline"} and payload.get("stage") == "revision":
            # Models reliably express the revision rationale in `reason` even
            # when they omit the optional dedicated field. Keep the explicit
            # field in the public schema, with this deterministic fallback at
            # the transport boundary so report lineage cannot stall.
            revision_reason = payload["reason"]
        # Keep custom progress events aligned with the visible CentralAgent
        # tool call. The reserved key is overwritten here, so model-supplied
        # metadata cannot redirect another task card.
        payload["metadata"] = {
            **dict(payload.get("metadata") or {}),
            **({"revision_reason": revision_reason} if revision_reason else {}),
            **(
                {
                    "routed_from_agent": "coder",
                    "routing_reason": routing_reason,
                }
                if routing_reason
                else {}
            ),
            "__sp_parent_tool_call_id": tool_call_id,
        }
    elif tool_name == "sp_recall_memory":
        payload["metadata"] = {**dict(payload.get("metadata") or {}), "memory_query": payload.pop("query")}
    elif tool_name == "sp_reflect":
        target_entry_ids = payload.pop("target_entry_ids", None)
        if target_entry_ids is not None:
            payload["metadata"] = {
                **dict(payload.get("metadata") or {}),
                "target_entry_ids": target_entry_ids,
            }
    elif tool_name == "sp_revise":
        payload["task"] = payload.pop("correction")
        payload["metadata"] = {
            **dict(payload.get("metadata") or {}),
            "target_entry_ids": payload.pop("target_entry_ids"),
            "revision_reason": payload.get("reason"),
        }
    elif tool_name == "sp_backtrack":
        payload["metadata"] = {
            **dict(payload.get("metadata") or {}),
            "backtrack_target_type": payload.pop("target_type"),
            "backtrack_target_id": payload.pop("target_id"),
            "rollback_scope": payload.pop("rollback_scope", "memory_only"),
        }
    elif tool_name == "sp_ask_human":
        payload["metadata"] = {
            **dict(payload.get("metadata") or {}),
            "question": payload.pop("question"),
            "interaction_type": payload.pop("interaction_type", "clarification"),
            "options": payload.pop("options", None),
        }
    elif tool_name == "sp_finish":
        summary = payload.pop("summary", None) or payload.pop("task", None)
        payload["task"] = str(summary or "The requested deliverable is complete and ready for the user.")
        final_artifact_ref = payload.pop("final_artifact_ref", None)
        required_artifact_type = payload.pop("required_artifact_type", None)
        payload["metadata"] = {
            "allow_without_artifact": bool(payload.pop("allow_without_artifact", False)),
            **({"final_artifact_ref": final_artifact_ref} if final_artifact_ref else {}),
            **({"required_artifact_type": required_artifact_type} if required_artifact_type else {}),
        }
    elif tool_name == "sp_summarize":
        payload["task"] = payload.pop("summary")
        source_entry_ids = payload.pop("source_entry_ids", None)
        if source_entry_ids is not None:
            payload["metadata"] = {
                **dict(payload.get("metadata") or {}),
                "source_entry_ids": source_entry_ids,
            }
    return payload


class SPControlActionMiddleware(AgentMiddleware[AgentState]):
    """Execute SP control methods in-place inside the one DR2 agent loop."""

    state_schema = AgentState

    def __init__(self, *, executor_provider: SPExecutorProvider | None = None) -> None:
        super().__init__()
        self._executor_provider = executor_provider
        self._summary_run_lock = Lock()
        self._summary_committed_runs: OrderedDict[tuple[str, str, str], None] = OrderedDict()

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Any]) -> Any:
        tool_name = str(request.tool_call.get("name") or "")
        if tool_name not in SP_CONTROL_EXECUTION_TOOL_NAMES:
            return handler(request)

        return self._execute_control_tool(request)

    async def awrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[Any]]) -> Any:
        tool_name = str(request.tool_call.get("name") or "")
        if tool_name not in SP_CONTROL_EXECUTION_TOOL_NAMES:
            return await handler(request)

        # DELEGATE is intentionally synchronous at the handler boundary because
        # DR2's SubagentExecutor exposes a sync compatibility API. Running it
        # directly here blocks the Gateway event loop and also serializes
        # sibling tool calls that LangGraph intended to execute concurrently.
        # Offload every SP action for one consistent async boundary; lightweight
        # actions cost only one thread-pool hop, while delegated execution can
        # now overlap without freezing auth, SSE, or artifact endpoints.
        return await asyncio.to_thread(self._execute_control_tool, request)

    def _execute_control_tool(self, request: ToolCallRequest) -> Command[Any]:
        tool_name = str(request.tool_call.get("name") or "")
        tool_call_id = str(request.tool_call.get("id") or "sp-tool")

        payload = _action_payload(tool_name, request.tool_call.get("args") or {}, tool_call_id=tool_call_id)
        if tool_name == "sp_finish":
            payload["metadata"] = {
                **dict(payload.get("metadata") or {}),
                # A LangGraph ToolNode preserves every sibling Command.goto.
                # Marking this at the request boundary lets FinishHandler reject
                # a premature FINISH before it can emit goto=END.
                _PARALLEL_NONTERMINAL_SIBLING_KEY: _has_parallel_nonterminal_sp_action(
                    request.state,
                    current_tool_call_id=tool_call_id,
                ),
            }
        runtime = request.runtime
        run_id = _run_id(runtime)
        thread_id = _thread_id(runtime)
        effective_state = request.state
        reserved_summary = False
        if tool_name == "sp_summarize" and run_id:
            summary_key = (
                thread_id or "",
                run_id,
                _summary_model_turn_id(request.state),
            )
            with self._summary_run_lock:
                already_committed = summary_key in self._summary_committed_runs
                if not already_committed:
                    self._summary_committed_runs[summary_key] = None
                    self._summary_committed_runs.move_to_end(summary_key)
                    while len(self._summary_committed_runs) > MAX_RUN_LOCAL_GUARD_ENTRIES:
                        self._summary_committed_runs.popitem(last=False)
                    reserved_summary = True
            if already_committed:
                # Multiple tool calls emitted in one model turn can observe
                # the same pre-action state. Project the in-process commit
                # into this request so the router suppresses the sibling.
                effective_state = {
                    **request.state,
                    SP_SUMMARY_TURN_RESERVED_KEY: True,
                }
        executor = self._executor_provider(request.state, runtime) if self._executor_provider is not None else None
        router = build_default_action_router(delegate_executor=executor, memory_recall_executor=executor)
        result = router.execute(payload, state=effective_state, thread_id=thread_id, run_id=run_id)
        if reserved_summary and result.next_step.startswith("error"):
            with self._summary_run_lock:
                self._summary_committed_runs.pop(summary_key, None)
        action = SPAction.from_dict(payload)
        handler_summary = result.state_update.get("sp_last_handler_result")
        effective_action_type = action.action_type
        effective_action_id = action.action_id
        if isinstance(handler_summary, Mapping):
            try:
                effective_action_type = ActionType(str(handler_summary.get("action_type")))
            except ValueError:
                pass
            if handler_summary.get("action_id"):
                effective_action_id = str(handler_summary["action_id"])
        SPControlActionMiddleware._record_events(runtime, result.run_events)
        content_payload: dict[str, Any] = {
            "action_type": effective_action_type.value,
            "next_step": result.next_step,
            "error": result.error,
        }
        additional_kwargs: dict[str, Any] = {
            "stackplanner": {
                "action_type": effective_action_type.value,
                "action_id": effective_action_id,
                "action_label": f"SP {effective_action_type.value}",
            }
        }
        if effective_action_type is ActionType.DELEGATE:
            delegate_succeeded = result.error is None and not result.next_step.startswith("error")
            observation = next(
                (entry for entry in reversed(result.memory_entries) if isinstance(entry, StackMemoryEntry) and entry.action == "observe"),
                None,
            )
            result_brief = observation.content if observation is not None else None
            stop_reason_value = observation.metadata.get("stop_reason") if observation is not None else None
            stop_reason = str(stop_reason_value) if stop_reason_value in SUBAGENT_STOP_REASON_VALUES else None
            completion_status = observation.metadata.get("completion_status") if observation is not None else None
            content_payload.update(
                {
                    "result": result_brief,
                    "completion_status": completion_status,
                    "stop_reason": stop_reason,
                }
            )
            additional_kwargs.update(
                make_subagent_additional_kwargs(
                    "completed" if delegate_succeeded else "failed",
                    result=result_brief,
                    error=result.error,
                    stop_reason=stop_reason,
                )
            )
            artifact_paths = _delegate_artifact_paths(result.state_update)
            if artifact_paths:
                # Native DeerFlow's `task` tool already has a frontend result
                # contract. SP delegation runs through a control action, so it
                # must explicitly persist the files created by this action on
                # the matching ToolMessage. This survives history reloads and
                # lets the subtask card expose every output, not only reports.
                additional_kwargs[SP_ARTIFACT_PATHS_KEY] = artifact_paths

        content = json.dumps(content_payload, ensure_ascii=False)
        tool_message = ToolMessage(
            content=content,
            tool_call_id=str(request.tool_call.get("id") or "sp-tool"),
            name=tool_name,
            additional_kwargs=additional_kwargs,
        )
        messages: list[Any] = [tool_message]
        if result.next_step == "finish":
            final_paths = _final_artifact_paths(request.state, result.state_update, run_id=run_id)
            if final_paths:
                present_call_id = f"present-{tool_message.tool_call_id}"
                messages.extend(
                    [
                        AIMessage(
                            id=f"sp-present:{tool_message.tool_call_id}",
                            content="",
                            tool_calls=[
                                {
                                    "name": "present_files",
                                    "args": {"filepaths": final_paths},
                                    "id": present_call_id,
                                    "type": "tool_call",
                                }
                            ],
                            additional_kwargs={"stackplanner_generated": True},
                        ),
                        ToolMessage(
                            content="Successfully presented files",
                            tool_call_id=present_call_id,
                            name="present_files",
                        ),
                    ]
                )
            # ``sp_finish`` is return-direct so the native LangChain agent loop
            # actually stops after this ToolNode. Persist the user-facing
            # summary before exiting; otherwise a successful terminal action
            # would leave only hidden control/present_files messages and the UI
            # would appear blank.
            messages.append(
                AIMessage(
                    id=f"sp-final:{tool_call_id}",
                    content=(str(result.state_update.get("sp_last_run_summary") or "").strip() or action.task or "Task completed."),
                    additional_kwargs={
                        "stackplanner": {
                            "action_type": ActionType.FINISH.value,
                            "action_id": action.action_id,
                            "status": "finished",
                        }
                    },
                )
            )
        state_update = dict(result.state_update)
        task_memory = state_update.get("sp_task_memory")
        if isinstance(task_memory, Mapping):
            state_update["sp_task_memory"] = {
                **dict(task_memory),
                SP_CONCURRENT_MERGE_MODE_KEY: True,
            }
        artifact_refs = state_update.get("sp_current_artifact_refs")
        if tool_name == "sp_delegate" and isinstance(artifact_refs, Mapping):
            state_update["sp_current_artifact_refs"] = {
                **dict(artifact_refs),
                SP_CONCURRENT_MERGE_MODE_KEY: True,
            }
        update = {**state_update, "messages": messages}
        if result.next_step == "interrupt":
            pending = update.get("sp_pending_human_interaction") or {}
            human_input = _human_input_payload(pending, action=action, tool_call_id=tool_message.tool_call_id)
            tool_message.content = human_input["question"]
            tool_message.artifact = {"human_input": human_input}
            return Command(update=update, goto=END)
        if result.next_step in {"finish", "error_fatal"}:
            return Command(update=update, goto=END)
        return Command(update=update)

    @staticmethod
    def _record_events(runtime: Any, events: list[Mapping[str, Any]]) -> None:
        context = getattr(runtime, "context", None)
        journal = context.get("__run_journal") if isinstance(context, Mapping) else None
        record = getattr(journal, "record_custom_event", None)
        if not callable(record):
            return
        for event in events:
            occurred_at = event.get("ts")
            record(
                str(event.get("event_type") or "sp.event"),
                content={
                    "action_id": event.get("action_id"),
                    "payload": event.get("payload") or {},
                    "occurred_at": occurred_at,
                },
                metadata={
                    "source": "stackplanner",
                    "action_id": event.get("action_id"),
                    "run_id": event.get("run_id"),
                    "occurred_at": occurred_at,
                },
            )


class SPThinkLabelMiddleware(AgentMiddleware[AgentState]):
    """Record free-form CentralAgent output as implicit continuous-space THINK."""

    state_schema = AgentState

    def __init__(self) -> None:
        super().__init__()
        self._web_search_failures: OrderedDict[tuple[str, str], int] = OrderedDict()
        self._web_search_lock = Lock()

    def before_agent(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        """Resolve a HumanInputCard reply before the CentralAgent continues."""
        pending = state.get("sp_pending_human_interaction") if isinstance(state, Mapping) else None
        if not isinstance(pending, Mapping):
            return None
        feedback = _pending_human_feedback(state, pending)
        if not feedback:
            return None
        result = record_human_feedback(
            dict(state),
            feedback,
            thread_id=_thread_id(runtime),
            run_id=_run_id(runtime),
        )
        SPControlActionMiddleware._record_events(runtime, result.run_events)
        return result.state_update

    async def abefore_agent(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        return self.before_agent(state, runtime)

    def after_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        return self._after_model_update(state, runtime)

    async def aafter_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        return self._after_model_update(state, runtime)

    def _after_model_update(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        messages = state.get("messages") if isinstance(state, Mapping) else None
        latest = messages[-1] if isinstance(messages, list) and messages else None
        if not isinstance(latest, AIMessage):
            return None
        if any(str(call.get("name") or "") in SP_CONTROL_TOOL_NAMES for call in latest.tool_calls):
            return None
        # A tool call outside the SP vocabulary is a boundary violation, not a
        # free-form thought. Do not normalize it into THINK and hide the fault.
        if latest.tool_calls:
            return None
        message_id = str(latest.id or _implicit_think_message_id(latest))
        normalized_json = _normalize_strict_json_response(
            message_to_text(latest),
            latest_user_request=message_to_text(SPFinishAvailabilityMiddleware._latest_visible_human_message(state)),
        )
        update: dict[str, Any] = {
            "sp_last_handler_result": {
                "action_type": ActionType.THINK.value,
                "action_id": message_id,
                "next_step": "continue",
            }
        }
        if normalized_json is not None and normalized_json != message_to_text(latest):
            latest = latest.model_copy(
                update={
                    "content": normalized_json,
                    "additional_kwargs": {
                        **latest.additional_kwargs,
                        "stackplanner_strict_json_normalized": True,
                    },
                }
            )
            update["messages"] = [latest]
        memory_update, consumed_observation_ids = self._collect_tool_observations(state, runtime)
        if memory_update is not None:
            update["sp_task_memory"] = memory_update
        think_text = _implicit_think_text(latest)
        if think_text:
            stack = TaskMemoryStack.from_dict(
                update.get("sp_task_memory", state.get("sp_task_memory")),
                thread_id=_thread_id(runtime),
                run_id=_run_id(runtime),
            )
            already_recorded = any(entry.action == "think" and str(entry.metadata.get("model_message_id") or "") == message_id for entry in stack.entries)
            if not already_recorded:
                stack.append_think(
                    think_text,
                    actor="central",
                    thread_id=_thread_id(runtime),
                    run_id=_run_id(runtime),
                    metadata={
                        "action_type": ActionType.THINK.value,
                        "implicit": True,
                        "control_tag": "<think>",
                        "model_message_id": message_id,
                    },
                )
                update["sp_task_memory"] = stack.to_dict()
        if consumed_observation_ids is not None:
            update["sp_consumed_tool_observation_ids"] = consumed_observation_ids
        return update

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Any]) -> Any:
        # Tool calls may be executed concurrently. Returning a state update here
        # would make each tool write its own value to the single-value
        # ``sp_task_memory``/``sp_last_handler_result`` channels, which LangGraph
        # rejects when one model turn emits multiple tool calls. The next
        # ``after_model`` callback sees all ToolMessages and commits one merged
        # memory update for the whole turn.
        if self._web_search_budget_exhausted(request):
            return self._web_search_exhausted_message(request)
        result = handler(request)
        return self._guard_web_search_result(request, result)

    async def awrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[Any]]) -> Any:
        # The web gateway uses LangGraph's async stream path. Keep native tools
        # untouched there as well; observations are aggregated in aafter_model.
        if self._web_search_budget_exhausted(request):
            return self._web_search_exhausted_message(request)
        result = await handler(request)
        return self._guard_web_search_result(request, result)

    def _web_search_budget_exhausted(self, request: ToolCallRequest) -> bool:
        if str(request.tool_call.get("name") or "") != "web_search":
            return False
        runtime = getattr(request, "runtime", None)
        key = (_thread_id(runtime) or "", _run_id(runtime) or "")
        with self._web_search_lock:
            return self._web_search_failures.get(key, 0) >= MAX_CONSECUTIVE_WEB_SEARCH_FAILURES

    @staticmethod
    def _web_search_exhausted_message(request: ToolCallRequest) -> ToolMessage:
        return ToolMessage(
            content=json.dumps(
                {
                    "error": "WEB_SEARCH_ATTEMPTS_EXHAUSTED",
                    "message": "web_search failed repeatedly in this run; stop retrying and explain that live search is unavailable.",
                    "retryable": False,
                },
                ensure_ascii=False,
            ),
            tool_call_id=str(request.tool_call.get("id") or "web-search-guard"),
            name="web_search",
        )

    def _guard_web_search_result(self, request: ToolCallRequest, result: Any) -> Any:
        tool_name = str(request.tool_call.get("name") or "")
        if tool_name != "web_search":
            return result
        content = _tool_result_summary(result, max_chars=2000).lower()
        runtime = getattr(request, "runtime", None)
        key = (_thread_id(runtime) or "", _run_id(runtime) or "")
        failed = "no results found" in content or "web_search_unavailable" in content
        with self._web_search_lock:
            if not failed:
                self._web_search_failures.pop(key, None)
                return result
            failures = self._web_search_failures.get(key, 0) + 1
            self._web_search_failures[key] = failures
            self._web_search_failures.move_to_end(key)
            while len(self._web_search_failures) > MAX_RUN_LOCAL_GUARD_ENTRIES:
                self._web_search_failures.popitem(last=False)
        if failures <= MAX_CONSECUTIVE_WEB_SEARCH_FAILURES:
            return result
        return self._web_search_exhausted_message(request)

    def _collect_tool_observations(
        self,
        state: AgentState,
        runtime: Any,
    ) -> tuple[dict[str, Any] | None, list[str] | None]:
        messages = state.get("messages") if isinstance(state, Mapping) else None
        if not isinstance(messages, list):
            return None, None

        stack = TaskMemoryStack.from_dict(
            state.get("sp_task_memory"),
            thread_id=_thread_id(runtime),
            run_id=_run_id(runtime),
        )
        observed_tool_call_ids = {str(entry.metadata.get("tool_call_id")) for entry in stack.entries if entry.action == "observe" and entry.metadata.get("tool_call_id")}
        observed_keys = {str(entry.metadata.get("observation_key")) for entry in stack.entries if entry.metadata.get("observation_key")}
        consumed_ids = [str(tool_call_id) for tool_call_id in (state.get("sp_consumed_tool_observation_ids") or []) if tool_call_id]
        consumed_id_set = set(consumed_ids)
        initial_consumed_count = len(consumed_ids)
        changed = False
        # A summary permanently compacts everything before its control-tool
        # message.  Limit the legacy-history scan accordingly as an additional
        # guard for threads created before tombstones were introduced.
        scan_start = 0
        for index, message in enumerate(messages):
            if isinstance(message, ToolMessage) and str(message.name or "") == "sp_summarize":
                scan_start = index + 1
        for message in messages[scan_start:]:
            if not isinstance(message, ToolMessage):
                continue
            tool_name = str(message.name or "")
            if tool_name in SP_CONTROL_TOOL_NAMES or tool_name == "ask_clarification":
                continue
            tool_call_id = str(message.tool_call_id or message.id or "")
            if not tool_call_id or tool_call_id in observed_tool_call_ids or tool_call_id in consumed_id_set:
                continue
            # ToolMessage history is immutable. Tombstone every processed
            # message, including filtered noise, so condensing its memory
            # entry cannot make the historical message look new later.
            consumed_ids.append(tool_call_id)
            consumed_id_set.add(tool_call_id)
            summary = _tool_result_summary(message)
            if not summary or not _should_record_tool_observation(tool_name, summary):
                continue
            observation_key = _observation_key(tool_name, summary)
            if observation_key in observed_keys:
                continue
            is_error = _is_tool_error(summary)
            stack.append(
                StackMemoryEntry(
                    action="error" if is_error else "observe",
                    content=summary,
                    actor="deerflow",
                    thread_id=_thread_id(runtime),
                    run_id=_run_id(runtime),
                    priority="high" if is_error else "normal",
                    metadata={
                        "action_type": ActionType.THINK.value,
                        "tool_name": tool_name,
                        "tool_call_id": tool_call_id,
                        "observation_key": observation_key,
                        "result_chars": len(summary),
                    },
                )
            )
            observed_tool_call_ids.add(tool_call_id)
            observed_keys.add(observation_key)
            changed = True
        consumed_update = consumed_ids[-MAX_CONSUMED_TOOL_OBSERVATION_IDS:] if len(consumed_ids) != initial_consumed_count else None
        return (stack.to_dict() if changed else None), consumed_update


def _tool_result_summary(result: Any, *, max_chars: int = 700) -> str:
    message = result if isinstance(result, ToolMessage) else None
    if message is None:
        update = getattr(result, "update", None)
        messages = update.get("messages") if isinstance(update, Mapping) else None
        if isinstance(messages, list):
            message = next((item for item in reversed(messages) if isinstance(item, ToolMessage)), None)
    content = getattr(message, "content", "") if message is not None else ""
    if isinstance(content, list):
        content = " ".join(str(item) for item in content)
    text = " ".join(str(content or "").split())
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    suffix = "...<truncated>"
    return f"{text[: max_chars - len(suffix)]}{suffix}"


def _human_input_payload(
    pending: Mapping[str, Any],
    *,
    action: SPAction,
    tool_call_id: str,
) -> dict[str, Any]:
    metadata = pending.get("metadata") if isinstance(pending.get("metadata"), Mapping) else {}
    raw_options = metadata.get("options")
    option_values = [str(option) for option in raw_options if str(option).strip()] if isinstance(raw_options, list) else []
    question = str(pending.get("question") or action.metadata.get("question") or action.task or "Please provide feedback.")
    payload: dict[str, Any] = {
        "version": 1,
        "kind": "human_input_request",
        "source": "stackplanner",
        "request_id": str(pending.get("interaction_id") or action.action_id),
        "tool_call_id": str(tool_call_id),
        "clarification_type": str(pending.get("interaction_type") or "clarification"),
        "question": question,
        "input_mode": "choice_with_other" if option_values else "free_text",
    }
    if option_values:
        payload["options"] = [{"id": f"option-{index}", "label": option, "value": option} for index, option in enumerate(option_values, 1)]
    title = metadata.get("title")
    if isinstance(title, str) and title.strip():
        payload["title"] = title.strip()
    context = metadata.get("context")
    if isinstance(context, str) and context.strip():
        payload["context"] = context.strip()
    return payload


def _pending_human_feedback(state: Mapping[str, Any], pending: Mapping[str, Any]) -> str | None:
    messages = state.get("messages")
    if not isinstance(messages, list):
        return None
    request_id = str(pending.get("interaction_id") or "")
    request_index: int | None = None
    for index, message in enumerate(messages):
        if not isinstance(message, ToolMessage):
            continue
        artifact = getattr(message, "artifact", None)
        request = artifact.get("human_input") if isinstance(artifact, Mapping) else None
        if isinstance(request, Mapping) and str(request.get("request_id") or "") == request_id:
            request_index = index

    for message in reversed(messages[(request_index + 1) if request_index is not None else 0 :]):
        if not isinstance(message, HumanMessage):
            continue
        response = read_human_input_response(message.additional_kwargs)
        if response is not None and (not request_id or response["request_id"] == request_id):
            return response["value"].strip()
        if request_index is not None:
            fallback = message_to_text(message).strip()
            if fallback:
                return fallback
    return None


def _json_sort_requirement(user_request: str) -> tuple[str, bool] | None:
    for pattern in _JSON_SORT_REQUEST_PATTERNS:
        match = pattern.search(user_request)
        if match is None:
            continue
        direction = match.group("direction").lower()
        return match.group("field"), direction in {"降序", "descending"}
    return None


def _json_sort_key(value: Any) -> tuple[int, Any]:
    if isinstance(value, bool):
        return 0, int(value)
    if isinstance(value, int | float):
        return 1, value
    if isinstance(value, str):
        return 2, value.casefold()
    return 3, json.dumps(_json_safe_for_output(value), ensure_ascii=False, sort_keys=True)


def _json_safe_for_output(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe_for_output(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe_for_output(item) for item in value]
    return str(value)


def _normalize_strict_json_response(
    response_text: str,
    *,
    latest_user_request: str,
) -> str | None:
    """Enforce an explicit strict-JSON boundary without changing semantics."""

    if not _STRICT_JSON_REQUEST_PATTERN.search(latest_user_request):
        return None
    candidate = response_text.strip()
    fenced = _JSON_FENCE_PATTERN.fullmatch(candidate)
    if fenced is not None:
        candidate = fenced.group(1).strip()
    try:
        payload = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None

    sort_requirement = _json_sort_requirement(latest_user_request)
    if sort_requirement is not None and isinstance(payload, list) and payload and all(isinstance(item, Mapping) for item in payload):
        field, descending = sort_requirement
        if all(field in item for item in payload):
            present = [item for item in payload if item.get(field) is not None]
            missing = [item for item in payload if item.get(field) is None]
            payload = [
                *sorted(
                    present,
                    key=lambda item: _json_sort_key(item.get(field)),
                    reverse=descending,
                ),
                *missing,
            ]

    return json.dumps(
        _json_safe_for_output(payload),
        ensure_ascii=False,
        indent=2,
    )


def _implicit_think_text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, list):
        content = " ".join(str(item.get("text") or "") if isinstance(item, Mapping) else str(item) for item in content)
    text = " ".join(str(content or "").split())
    if len(text) <= MAX_IMPLICIT_THINK_CHARS:
        return text
    suffix = "...<truncated>"
    return f"{text[: MAX_IMPLICIT_THINK_CHARS - len(suffix)]}{suffix}"


def _implicit_think_message_id(message: AIMessage) -> str:
    digest = hashlib.sha256(_implicit_think_text(message).encode("utf-8")).hexdigest()[:16]
    return f"implicit-think:{digest}"


# These are execution details, not CentralAgent decisions. The model still
# receives the native tool message in the current turn; we simply do not make
# it durable task memory after the turn has completed.
_NON_MEMORY_TOOL_NAMES = frozenset(
    {
        "ls",
        "glob",
        "list_dir",
        "list_files",
        "read_file",
        "present_file",
        "present_files",
        "describe_skill",
    }
)
_NON_MEMORY_SUCCESS_MESSAGES = frozenset(
    {
        "ok",
        "success",
        "successfully presented files",
        "file presented",
    }
)
_URL_ONLY_RE = re.compile(r"^(?:https?://\S+\s*)+$", re.IGNORECASE)
_FILE_LISTING_RE = re.compile(r"(?:^|\s)[-dl][rwx-]{9}\s+\d+\s+\S+\s+\S+\s+\d+\s+", re.IGNORECASE)
_ERROR_MARKERS = (
    "error:",
    "traceback",
    "connection refused",
    "unsafe absolute path",
    "http 4",
    "http 5",
    "aborterror",
    "failed:",
)


def _is_tool_error(summary: str) -> bool:
    lowered = summary.strip().lower()
    return lowered.startswith(_ERROR_MARKERS) or any(marker in lowered for marker in _ERROR_MARKERS)


def _should_record_tool_observation(tool_name: str, summary: str) -> bool:
    normalized = " ".join(summary.split()).strip().lower()
    if tool_name.lower() in _NON_MEMORY_TOOL_NAMES:
        return False
    if normalized in _NON_MEMORY_SUCCESS_MESSAGES or _URL_ONLY_RE.fullmatch(normalized):
        return False
    if _FILE_LISTING_RE.search(normalized):
        return False
    if normalized.startswith("--- name:") or "this skill should be used when" in normalized or ("/mnt/skills/" in normalized and "skill" in normalized):
        return False
    return True


def _observation_key(tool_name: str, summary: str) -> str:
    normalized = " ".join(summary.split()).lower()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"{tool_name.lower()}:{digest}"
