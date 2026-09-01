"""Base types for SP action handlers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from deerflow.sp.actions.schema import HandlerResult, SPAction
from deerflow.sp.memory import TaskMemoryStack


@dataclass(slots=True)
class HandlerContext:
    """Runtime context shared by SP action handlers."""

    state: Mapping[str, Any]
    stack: TaskMemoryStack
    thread_id: str | None = None
    run_id: str | None = None


class BaseActionHandler(Protocol):
    """Protocol implemented by all concrete SP action handlers."""

    def handle(self, action: SPAction, context: HandlerContext) -> HandlerResult:
        """Execute a validated SP action."""
