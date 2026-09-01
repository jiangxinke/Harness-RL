"""SP subagent delegation adapters."""

from deerflow.sp.subagents.adapter import SPSubagentExecutorProtocol, SPSubagentResult, SPSubagentStatus, SPSubagentTask
from deerflow.sp.subagents.dr2_adapter import DR2SubagentExecutorAdapter, DR2SubagentExecutorLike, normalize_dr2_subagent_result, render_sp_subagent_prompt

__all__ = [
    "DR2SubagentExecutorAdapter",
    "DR2SubagentExecutorLike",
    "SPSubagentExecutorProtocol",
    "SPSubagentResult",
    "SPSubagentStatus",
    "SPSubagentTask",
    "normalize_dr2_subagent_result",
    "render_sp_subagent_prompt",
]
