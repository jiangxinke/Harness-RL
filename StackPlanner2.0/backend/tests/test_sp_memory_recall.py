"""Tests for SP read-only long-term memory recall normalization."""

import json

from deerflow.sp.memory import normalize_memory_recall_result
from deerflow.sp.subagents import DR2SubagentExecutorAdapter, SPSubagentResult, SPSubagentStatus, SPSubagentTask


def test_normalize_memory_recall_result_accepts_plain_text():
    result = normalize_memory_recall_result(
        "project preferences",
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Prefer phased migrations with tests after each unit.",
            task_id="mem-task-plain",
        ),
    )

    assert result.summary == "Prefer phased migrations with tests after each unit."
    assert result.items == []
    assert result.source_task_id == "mem-task-plain"
    assert result.dry_run_promotion is True


def test_normalize_memory_recall_result_bounds_items_and_content():
    payload = {
        "summary": "Relevant project memories",
        "items": [{"content": "x" * 80, "id": f"mem-{idx}"} for idx in range(5)],
    }

    result = normalize_memory_recall_result(
        "project preferences",
        SPSubagentResult(status=SPSubagentStatus.COMPLETED, result=json.dumps(payload), task_id="mem-task-json"),
        max_items=2,
        max_item_chars=20,
    )

    assert len(result.items) == 2
    assert result.items[0].content.endswith("...<truncated>")
    assert result.metadata["truncated_item_count"] == 3
    assert result.dry_run_promotion is True


def test_fenced_dr2_memory_recaller_result_keeps_items_end_to_end():
    class Executor:
        def execute(self, task: str):
            return type(
                "Result",
                (),
                {
                    "status": "completed",
                    "result": '```json\n{"summary":"Prior preference","items":[{"content":"Test every unit","memory_id":"mem-1"}]}\n```',
                    "task_id": "memory-task",
                    "error": None,
                    "stop_reason": None,
                },
            )()

    task = SPSubagentTask(
        action_id="act-memory",
        subagent_type="memory_recaller",
        task="Recall testing preferences",
        description="Need prior preferences",
    )
    adapted = DR2SubagentExecutorAdapter(lambda _: Executor()).execute(task)

    normalized = normalize_memory_recall_result("testing preferences", adapted)

    assert normalized.summary == "Prior preference"
    assert len(normalized.items) == 1
    assert normalized.items[0].content == "Test every unit"
    assert normalized.items[0].memory_id == "mem-1"
