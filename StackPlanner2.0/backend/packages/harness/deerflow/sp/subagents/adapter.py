"""Typed adapter boundary for SP delegation through DR2 subagents."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class SPSubagentStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


@dataclass(slots=True)
class SPSubagentTask:
    """Structured task sent from DelegateHandler to a subagent executor."""

    action_id: str
    subagent_type: str
    task: str
    description: str
    input_refs: list[str] = field(default_factory=list)
    expected_output: str | None = None
    context_refs: dict[str, Any] = field(default_factory=dict)
    thread_id: str | None = None
    run_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SPSubagentResult:
    """Normalized result returned by a SP subagent executor adapter."""

    status: SPSubagentStatus
    result: str | None = None
    error: str | None = None
    stop_reason: str | None = None
    task_id: str | None = None
    artifact_content: Any | None = None
    artifact_type: str | None = None
    artifact_metadata: dict[str, Any] = field(default_factory=dict)
    token_usage_records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        return self.status == SPSubagentStatus.COMPLETED


class SPSubagentExecutorProtocol(Protocol):
    """Executor dependency used by DelegateHandler.

    Runtime integration should wrap DR2's ``SubagentExecutor`` behind this
    protocol. Tests can provide a fake executor without touching tools.
    """

    def execute(self, task: SPSubagentTask) -> SPSubagentResult:
        """Execute a SP subagent task and return a normalized result."""
