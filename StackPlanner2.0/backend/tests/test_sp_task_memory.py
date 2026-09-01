"""Unit tests for StackPlanner task memory primitives."""

from __future__ import annotations

import json

from deerflow.sp.memory import StackMemoryEntry, TaskMemoryStack
from deerflow.sp.memory.entry import MAX_ENTRY_CONTENT_CHARS, MAX_METADATA_JSON_CHARS


def test_legacy_memory_stack_json_restores_feedback_as_pinned_critical():
    legacy = json.dumps(
        [
            {
                "timestamp": "2026-07-09T12:00:00",
                "action": "human_feedback",
                "agent_type": "human",
                "content": "Put the macro pipeline first",
                "result": {"priority": "HIGHEST"},
            }
        ],
    )

    stack = TaskMemoryStack.from_legacy_memory_stack(legacy, thread_id="thread-1", run_id="run-1")

    assert stack.size() == 1
    entry = stack.entries[0]
    assert entry.action == "feedback"
    assert entry.actor == "human"
    assert entry.priority == "critical"
    assert entry.status == "pinned"
    assert entry.thread_id == "thread-1"
    assert entry.run_id == "run-1"
    assert entry.metadata["legacy_action"] == "human_feedback"


def test_legacy_delegate_with_result_becomes_observation():
    stack = TaskMemoryStack.from_legacy_memory_stack(
        [
            {
                "timestamp": "2026-07-09T12:00:00",
                "action": "delegate",
                "agent_type": "researcher",
                "content": "Research DeerFlow ThreadState",
                "result": {"summary": "ThreadState is runtime-first"},
            }
        ]
    )

    entry = stack.entries[0]
    assert entry.action == "observe"
    assert entry.actor == "researcher"
    assert entry.metadata["legacy_action"] == "delegate"
    assert entry.metadata["legacy_result"] == {"summary": "ThreadState is runtime-first"}


def test_append_feedback_forces_pinned_critical_even_when_kwargs_disagree():
    stack = TaskMemoryStack()

    entry = stack.append_feedback("User confirmed the outline", priority="low", status="active")

    assert entry.priority == "critical"
    assert entry.status == "pinned"
    assert stack.get_pinned_entries() == [entry]


def test_condense_pops_source_entries_without_touching_pinned_feedback():
    stack = TaskMemoryStack()
    first = stack.append_think("Initial plan")
    feedback = stack.append_feedback("Keep this feedback")
    second = stack.append_observe("Research result", actor="researcher")

    summary = stack.condense([first.id, feedback.id, second.id], "Stage summary")

    assert first not in stack.entries
    assert feedback.status == "pinned"
    assert second not in stack.entries
    assert summary.action == "summarize"
    assert summary.parent_ids == [first.id, feedback.id, second.id]


def test_condense_never_removes_critical_active_memory():
    stack = TaskMemoryStack()
    critical = stack.append(
        StackMemoryEntry(
            action="user_request",
            content="Current user objective",
            priority="critical",
            status="active",
        )
    )
    observation = stack.append_observe("Verified evidence", actor="researcher")

    summary = stack.condense([critical.id, observation.id], "Keep the objective and evidence")

    assert critical in stack.entries
    assert observation not in stack.entries
    assert summary.parent_ids == [critical.id, observation.id]


def test_condense_preserves_transitive_lineage_when_compacting_a_summary():
    stack = TaskMemoryStack()
    first = stack.append_think("Initial plan")
    second = stack.append_observe("Verified result", actor="researcher")
    first_summary = stack.condense([first.id, second.id], "Planning complete")
    later = stack.append_observe("Implementation verified", actor="coder")

    final_summary = stack.condense(
        [first_summary.id, later.id],
        "Planning and implementation complete",
    )

    assert first_summary not in stack.entries
    assert later not in stack.entries
    assert final_summary.parent_ids == [
        first_summary.id,
        first.id,
        second.id,
        later.id,
    ]


def test_summary_cooldown_requires_execution_progress_not_repeated_thinking():
    stack = TaskMemoryStack()
    stack.append_summary("Stage complete", run_id="run-1")
    for index in range(6):
        stack.append_think(f"reasoning loop {index}", run_id="run-1")

    assert stack.summarization_progress_since_last(current_run_id="run-1") == 0
    assert stack.can_summarize(current_run_id="run-1") is False


def test_automatic_summary_selects_oldest_entries_before_recent_work():
    stack = TaskMemoryStack()
    entries = [stack.append_think(f"entry-{index}") for index in range(18)]

    selected = stack.select_summarization_source_ids()

    assert selected == [entry.id for entry in entries[:6]]


def test_prune_preserves_pinned_feedback():
    stack = TaskMemoryStack()
    old = stack.append_think("Old path")
    feedback = stack.append_feedback("User feedback must remain")
    stack.append_observe("Observation", actor="researcher")

    stack.prune(max_entries=1)

    assert old.status == "pruned"
    assert feedback.status == "pinned"
    assert feedback in stack.get_active_entries()


def test_serialize_round_trip_uses_thread_state_shape():
    stack = TaskMemoryStack()
    stack.append(StackMemoryEntry(id="spmem_custom", action="think", content="Keep it structured"))

    restored = TaskMemoryStack.from_dict(stack.to_dict())

    assert restored.to_dict()["version"] == 1
    assert restored.entries[0].id == "spmem_custom"
    assert restored.entries[0].content == "Keep it structured"


def test_short_term_stack_supports_designed_sp_control_flow():
    stack = TaskMemoryStack()

    think = stack.append_think("Plan the migration slice", stage="planning")
    recall = stack.append_memory_recall("Recall prior migration preferences", stage="planning")
    delegate = stack.append_delegate("Ask researcher to inspect ThreadState", stage="research")
    observe = stack.append_observe("ThreadState is the persistence boundary", actor="researcher", stage="research")
    feedback = stack.append_feedback("Keep user feedback above summaries", stage="revision")
    summary = stack.condense([think.id, recall.id, observe.id], "Planning and research summarized")
    backtrack = stack.mark_backtracked([delegate.id], "Delegation target was too broad", stage="planning")
    replan = stack.append_replan("Delegate narrower implementation tasks", stage="planning")
    finish = stack.append_finish("Short-term memory slice is locally verified")

    assert [entry.action for entry in stack.entries] == [
        "delegate",
        "feedback",
        "summarize",
        "backtrack",
        "replan",
        "finish",
    ]
    assert feedback in stack.get_pinned_entries()
    assert feedback.status == "pinned"
    assert summary.parent_ids == [think.id, recall.id, observe.id]
    assert delegate.status == "pruned"
    assert backtrack.failure_note == "Delegation target was too broad"
    assert stack.get_checkpoint("planning") == replan
    assert finish.stage == "finished"


def test_entry_bounds_large_content_and_metadata_before_thread_state_serialization():
    entry = StackMemoryEntry(
        action="observe",
        content="x" * (MAX_ENTRY_CONTENT_CHARS + 500),
        metadata={"action_id": "act-large", "raw_research": "y" * (MAX_METADATA_JSON_CHARS + 500)},
    )

    assert len(entry.content) == MAX_ENTRY_CONTENT_CHARS
    assert entry.content.endswith("...<truncated>")
    assert entry.metadata["action_id"] == "act-large"
    assert entry.metadata["_truncated"] is True
    assert len(json.dumps(entry.metadata, ensure_ascii=False)) < MAX_METADATA_JSON_CHARS


def test_checkpoint_restore_is_idempotent_after_bounds_are_applied():
    stack = TaskMemoryStack(max_size=3)
    stack.append_feedback("Latest correction")
    stack.append_observe("z" * 5000, actor="researcher")

    first_restore = TaskMemoryStack.from_dict(stack.to_dict())
    second_restore = TaskMemoryStack.from_dict(first_restore.to_dict())

    assert second_restore.to_dict() == first_restore.to_dict()
    assert second_restore.entries[0].status == "pinned"


def test_stack_bounds_inactive_audit_history_without_dropping_live_or_pinned_entries():
    stack = TaskMemoryStack(max_size=4, max_history_size=8)
    feedback = stack.append_feedback("Pinned user correction")
    for index in range(20):
        stack.append_think(f"step-{index}")

    assert len(stack.entries) <= 8
    assert feedback in stack.entries
    assert [entry.content for entry in stack.get_active_entries() if entry.status == "active"] == ["step-17", "step-18", "step-19"]

    restored = TaskMemoryStack.from_dict(stack.to_dict())
    assert restored.max_history_size == 8
    assert restored.to_dict() == stack.to_dict()
