"""Tests for SP prompt context rendering."""

from datetime import UTC, datetime, timedelta

from deerflow.sp.memory import StackMemoryEntry, TaskMemoryStack
from deerflow.sp.prompt import PromptContextBuilder


def test_context_prioritizes_pinned_human_feedback_before_recent_memory():
    stack = TaskMemoryStack()
    stack.append_think("Try the default lead-agent flow first", stage="planning")
    stack.append_feedback("User requires CentralAgent to route actions through handlers", stage="planning")

    context = PromptContextBuilder().build(stack, current_stage="planning")

    assert context.index("critical_feedback:") < context.index("recent_task_memory:")
    assert context.index("User requires CentralAgent") < context.index("Try the default")
    assert "priority_rules:" in context
    assert "Memory order: critical_feedback and recent_task_memory first" in context
    assert "Do not call sp_recall_memory" in context


def test_context_is_bounded_and_clips_entry_content():
    stack = TaskMemoryStack()
    stack.append_observe("x" * 1000, actor="researcher", result_ref="artifact://research-1")

    context = PromptContextBuilder(max_chars=1200, max_entry_chars=80).build(
        stack,
        artifact_refs={"research": "artifact://research-1"},
    )

    assert len(context) <= 1200
    assert "...<truncated>" in context
    assert "artifact://research-1" in context


def test_context_uses_refs_without_requiring_artifact_bodies():
    stack = TaskMemoryStack()
    stack.append_delegate("Ask reporter to draft from artifact refs only", result_ref="artifact://outline")

    context = PromptContextBuilder().build(
        stack,
        artifact_refs={"outline": "artifact://outline", "report": "artifact://report"},
        report_version="v1",
    )

    assert "artifact://outline" in context
    assert "artifact://report" in context
    assert "current_report_version: v1" in context
    assert "do not infer large artifact bodies" in context


def test_context_exposes_compact_workflow_status_from_artifacts_and_feedback():
    stack = TaskMemoryStack()
    stack.append_feedback("Outline approved", run_id="run-1")
    context = PromptContextBuilder().build(
        stack,
        current_run_id="run-1",
        artifact_refs={
            "outline": {"artifact_id": "outline-1", "type": "outline", "is_current": True},
            "research_observation": {
                "artifact_id": "research-1",
                "type": "research_observation",
                "is_current": True,
            },
            "_history": [
                {"artifact_id": "research-0", "type": "research_observation", "is_current": False},
            ],
        },
    )

    assert "workflow_status:" in context
    assert '"outline": 1' in context
    assert '"research_observation": 2' in context
    assert '"pinned_feedback": 1' in context


def test_context_keeps_newest_feedback_when_pinned_window_is_full():
    stack = TaskMemoryStack(max_size=30)
    base = datetime(2026, 7, 10, tzinfo=UTC)
    for index in range(15):
        stack.append_feedback(
            f"feedback-{index}",
            ts=(base + timedelta(minutes=index)).isoformat(),
        )

    context = PromptContextBuilder(recent_entry_limit=12).build(stack)

    assert "feedback-14" in context
    assert "feedback-3" in context
    assert "feedback-2" not in context
    assert "feedback-0" not in context


def test_context_renders_critical_feedback_before_large_artifact_refs():
    stack = TaskMemoryStack()
    stack.append_memory_recall("Old memory says use a narrative opening")
    stack.append_feedback("Use a conclusion-first structure")

    context = PromptContextBuilder(max_chars=1000).build(
        stack,
        artifact_refs={"research": "artifact://" + "x" * 3000},
    )

    assert len(context) <= 1000
    assert context.endswith("</sp-task-context>")
    assert "Use a conclusion-first structure" in context
    assert context.index("Use a conclusion-first structure") < context.find("current_artifact_refs:") or "current_artifact_refs:" not in context


def test_context_scopes_recent_memory_to_current_run():
    stack = TaskMemoryStack()
    stack.append_think("Old greeting should stay in history", run_id="run-old")
    stack.append_think("Current request planning", run_id="run-new")

    context = PromptContextBuilder().build(stack, current_run_id="run-new")

    assert "Current request planning" in context
    assert "Old greeting should stay in history" not in context


def test_context_carries_bounded_prior_user_updates_without_old_model_noise():
    stack = TaskMemoryStack()
    stack.append_think("Old model speculation", run_id="run-old")
    stack.append(
        StackMemoryEntry(
            action="user_request",
            actor="human",
            content="Budget changed to 90000",
            run_id="run-old",
            priority="high",
        )
    )
    stack.append_think("Current request planning", run_id="run-new")

    context = PromptContextBuilder().build(stack, current_run_id="run-new")

    assert "Budget changed to 90000" in context
    assert "Current request planning" in context
    assert "Old model speculation" not in context


def test_context_surfaces_prior_model_conclusion_for_explicit_correction():
    stack = TaskMemoryStack()
    old_observation = stack.append_observe(
        "A店总价低于B店。",
        actor="coder",
        run_id="run-old",
    )
    old_answer = stack.append_think(
        "所以应该选A店。",
        actor="central",
        run_id="run-old",
    )
    stack.append(
        StackMemoryEntry(
            action="user_request",
            actor="human",
            content="我刚问清楚，店员前面说错了。请重新检查你上次的结论。",
            run_id="run-new",
            priority="high",
        )
    )

    context = PromptContextBuilder().build(stack, current_run_id="run-new")

    assert "correction_review_required: true" in context
    assert f"id={old_observation.id}" in context
    assert f"id={old_answer.id}" in context
    assert "A店总价低于B店" in context
    assert "所以应该选A店" in context


def test_context_does_not_restore_prior_model_noise_for_non_correction_turn():
    stack = TaskMemoryStack()
    old_answer = stack.append_think(
        "Old model answer",
        actor="central",
        run_id="run-old",
    )
    stack.append(
        StackMemoryEntry(
            action="user_request",
            actor="human",
            content="继续给我一些建议。",
            run_id="run-new",
            priority="high",
        )
    )

    context = PromptContextBuilder().build(stack, current_run_id="run-new")

    assert "correction_review_required:" not in context
    assert f"id={old_answer.id}" not in context
    assert "Old model answer" not in context


def test_context_recognizes_natural_constraint_update_as_correction():
    stack = TaskMemoryStack()
    old_answer = stack.append_think(
        "小安和小陈同住。",
        actor="central",
        run_id="run-old",
    )
    stack.append(
        StackMemoryEntry(
            action="user_request",
            actor="human",
            content=("我刚收到消息，小安和小陈也不能同住，其他要求不变。请检查你上次的安排，重新排房。"),
            run_id="run-new",
            priority="high",
        )
    )

    context = PromptContextBuilder().build(stack, current_run_id="run-new")

    assert "correction_review_required: true" in context
    assert f"id={old_answer.id}" in context


def test_context_surfaces_explicit_constraints_as_an_authoritative_ledger():
    stack = TaskMemoryStack()
    stack.append(
        StackMemoryEntry(
            action="user_request",
            actor="human",
            content="补充偏好：流程轻松，不安排高强度团建；这不是可牺牲约束。",
            run_id="run-old",
            priority="high",
        )
    )
    stack.append(
        StackMemoryEntry(
            action="user_request",
            actor="human",
            content="现在输出最终 JSON。",
            run_id="run-new",
            priority="high",
        )
    )

    context = PromptContextBuilder().build(stack, current_run_id="run-new")

    assert "authoritative_constraints:" in context
    assert "流程轻松" in context
    assert "constraint_class=hard" in context
    assert context.index("authoritative_constraints:") < context.index("recent_task_memory:")


def test_context_exposes_bounded_old_candidates_for_stage_summary():
    stack = TaskMemoryStack()
    old_entries = [stack.append_think(f"old fact {index}", run_id="run-old") for index in range(17)]
    stack.append_think("current progress", run_id="run-new")

    context = PromptContextBuilder().build(stack, current_run_id="run-new")

    assert "summarization_needed: true" in context
    assert "summarization_candidates:" in context
    assert f"id={old_entries[0].id}" in context
    assert f"id={old_entries[5].id}" in context


def test_context_does_not_request_automatic_summary_below_pressure_threshold():
    stack = TaskMemoryStack()
    for index in range(17):
        stack.append_think(f"working entry {index}", run_id="run-new")

    context = PromptContextBuilder().build(stack, current_run_id="run-new")

    assert "summarization_needed: false" in context
    assert "summarization_candidates:" not in context


def test_context_carries_prior_run_summary_without_reinjecting_old_noise():
    stack = TaskMemoryStack()
    stack.append_think("old raw thought", run_id="run-old")
    stack.append_summary("verified carryover decision", run_id="run-old")
    stack.append_think("current progress", run_id="run-new")

    context = PromptContextBuilder().build(stack, current_run_id="run-new")

    assert "verified carryover decision" in context
    assert "current progress" in context
    assert "old raw thought" not in context


def test_context_reserves_latest_feedback_progress_and_control_footer_under_pressure():
    stack = TaskMemoryStack(max_size=50)
    for index in range(12):
        stack.append_feedback(
            f"feedback-{index} " + ("F" * 650),
            run_id="run-1",
        )
    for index in range(18):
        stack.append_observe(
            f"observation-{index} " + ("O" * 650),
            actor="coder",
            run_id="run-1",
        )

    context = PromptContextBuilder().build(
        stack,
        current_run_id="run-1",
        active_delegate_id="delegate-current",
        artifact_refs={
            "report": {
                "artifact_id": "report-1",
                "virtual_path": "/mnt/user-data/outputs/report.md",
                "summary": "R" * 800,
            }
        },
    )

    assert len(context) <= 6000
    assert "feedback-11" in context
    assert "observation-17" in context
    assert "summarization_pressure: true" in context
    assert "summarization_needed: true" in context
    assert "summarization_candidate_ids:" in context
    assert "active_delegate_id: delegate-current" in context
    assert "/mnt/user-data/outputs/report.md" in context
    assert context.endswith("</sp-task-context>")


def test_context_exposes_summary_cooldown_until_four_new_entries_arrive():
    stack = TaskMemoryStack()
    stack.append_summary("first stage complete", run_id="run-1")
    for index in range(3):
        stack.append_observe(
            f"new progress {index}",
            actor="researcher",
            run_id="run-1",
        )

    blocked = PromptContextBuilder().build(stack, current_run_id="run-1")
    stack.append_observe("new progress 3", actor="coder", run_id="run-1")
    ready = PromptContextBuilder().build(stack, current_run_id="run-1")

    assert "summarization_cooldown: active (3/4 new entries required)" in blocked
    assert "summarization_cooldown: ready (4 new entries since last summary)" in ready
