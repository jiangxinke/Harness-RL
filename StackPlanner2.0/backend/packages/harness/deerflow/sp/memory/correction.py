"""Bounded detection of user corrections that invalidate prior task-memory conclusions."""

from __future__ import annotations

import re

from deerflow.sp.memory.entry import StackMemoryEntry
from deerflow.sp.memory.stack import TaskMemoryStack

CORRECTION_REVIEW_PATTERN = re.compile(
    r"(?:"
    r"(?:刚|前面|之前|上次).{0,32}(?:说错|算错|写错|有错|错了|错误|不对|有误|改了|更正|遗漏|漏掉)"
    r"|(?:检查|核对).{0,20}(?:上次|之前).{0,24}(?:结论|答案|安排|做法|结果)"
    r"|(?:刚|新).{0,64}(?:不能|改成|新增|取消).{0,64}(?:其他.{0,12}不变)"
    r"|(?:重新|再)(?:检查|核对|计算|算一遍)"
    r"|(?:纠正|更正|改正).{0,24}(?:结论|答案|安排|做法|结果)"
    r"|\b(?:previous|earlier|last).{0,48}\b(?:wrong|incorrect|mistake|conclusion|answer)\b"
    r"|\b(?:recheck|recalculate|correct the previous)\b"
    r")",
    re.IGNORECASE | re.DOTALL,
)


def select_correction_review_candidates(
    stack: TaskMemoryStack,
    *,
    current_run_id: str | None,
    limit: int = 3,
) -> list[StackMemoryEntry]:
    """Return recent active model conclusions targeted by an explicit correction."""
    if not current_run_id:
        return []
    if any(entry.run_id == current_run_id and entry.action == "reflect" and entry.metadata.get("trigger") == "user_correction" for entry in stack.entries):
        # One targeted reflection/backtrack is enough for a single user turn.
        # Without this reservation, a deep prior stack is peeled in repeated
        # batches (``limit`` entries at a time), consuming multiple model turns
        # before the corrected work can actually run.
        return []
    current_user_requests = [entry for entry in stack.get_active_entries() if entry.run_id == current_run_id and entry.actor == "human" and entry.action == "user_request"]
    if not current_user_requests or not CORRECTION_REVIEW_PATTERN.search(current_user_requests[-1].content):
        return []
    prior_model_entries = [entry for entry in stack.get_active_entries() if entry.run_id != current_run_id and entry.actor != "human" and entry.action in {"observe", "think", "finish"}]
    return prior_model_entries[-max(1, limit) :]
