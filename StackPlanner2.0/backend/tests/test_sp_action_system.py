"""Tests for SP Action schema, router, and state-only handlers."""

import pytest
from langchain_core.messages import HumanMessage

from deerflow.sp.actions import ActionType, ActionValidationError, SPAction, build_default_action_router
from deerflow.sp.central import CENTRAL_AGENT_ACTION_PROMPT
from deerflow.sp.hitl import record_human_feedback
from deerflow.sp.memory import StackMemoryEntry, TaskMemoryStack
from deerflow.sp.subagents import SPSubagentResult, SPSubagentStatus, SPSubagentTask


def _action(action_type: ActionType | str, **kwargs):
    return SPAction.create(action_type, reason=kwargs.pop("reason", "unit-test reason"), **kwargs)


def test_action_schema_requires_action_id_and_normalizes_type():
    with pytest.raises(ActionValidationError, match="action_id"):
        SPAction.from_dict({"action_type": "think", "reason": "missing id"})

    action = SPAction.from_dict(
        {
            "action_id": "act-1",
            "action_type": "think",
            "reason": "Need a plan",
            "task": "Plan next step",
        }
    )

    assert action.action_type == ActionType.THINK
    assert action.idempotency_key.startswith("spidem_")


@pytest.mark.parametrize(
    ("provided", "expected"),
    [
        ("plain text", "plain text"),
        (
            {"url": "string", "title": "标题"},
            '{"title": "标题", "url": "string"}',
        ),
        (
            [{"url": "string", "title": "标题"}, "summary"],
            '[{"title": "标题", "url": "string"}, "summary"]',
        ),
    ],
)
def test_action_schema_normalizes_structured_expected_output_to_stable_text(provided, expected):
    action = SPAction.from_dict(
        {
            "action_id": "act-structured-output",
            "action_type": "delegate",
            "reason": "Need current sources",
            "target_agent": "researcher",
            "task": "Find authoritative sources",
            "expected_output": provided,
        }
    )

    assert action.expected_output == expected
    assert action.to_dict()["expected_output"] == expected


def test_delegate_schema_requires_target_agent_and_task():
    with pytest.raises(ActionValidationError, match="target_agent"):
        SPAction.create(ActionType.DELEGATE, reason="Need research", task="Research")
    with pytest.raises(ActionValidationError, match="task"):
        SPAction.create(ActionType.DELEGATE, reason="Need research", target_agent="researcher")
    with pytest.raises(ActionValidationError, match="Unsupported target_agent"):
        SPAction.create(ActionType.DELEGATE, reason="Unknown role", target_agent="invented-agent", task="Do work")


def test_finish_schema_requires_user_facing_task_text():
    with pytest.raises(ActionValidationError, match="task"):
        SPAction.create(ActionType.FINISH, reason="Done")


def test_action_schema_rejects_unknown_control_values_and_destructive_backtrack():
    with pytest.raises(ActionValidationError, match="Unsupported priority"):
        _action(ActionType.THINK, action_id="bad-priority", task="Think", priority="urgent")
    with pytest.raises(ActionValidationError, match="Unsupported stage"):
        _action(ActionType.THINK, action_id="bad-stage", task="Think", stage="mystery")
    with pytest.raises(ActionValidationError, match="backtrack_target_type"):
        _action(
            ActionType.BACKTRACK,
            action_id="bad-target",
            metadata={"backtrack_target_type": "checkpoint", "backtrack_target_id": "one"},
        )
    with pytest.raises(ActionValidationError, match="rollback_scope"):
        _action(
            ActionType.BACKTRACK,
            action_id="bad-scope",
            metadata={
                "backtrack_target_type": "entry",
                "backtrack_target_id": "one",
                "rollback_scope": "delete_everything",
            },
        )
    with pytest.raises(ActionValidationError, match="cannot delete artifact history"):
        _action(
            ActionType.BACKTRACK,
            action_id="bad-artifact-delete",
            metadata={
                "backtrack_target_type": "artifact_version",
                "backtrack_target_id": "v1",
                "preserve_artifacts": False,
            },
        )


@pytest.mark.parametrize(
    ("provided", "expected"),
    [
        ("report", "reporting"),
        ("REPORT", "reporting"),
        ("information-gathering", "research"),
        ("information_gathering", "research"),
        ("data-collection", "research"),
        ("analysis", "perception"),
        ("recognition", "perception"),
        ("thinking", "perception"),
        ("clarification", "perception"),
        ("retrieval", "research"),
        ("reinquiry", "research"),
        ("task-intake", "perception"),
        ("scope-assessment", "perception"),
        ("file-inspection", "perception"),
        ("perception-and-scope", "perception"),
        ("execution", "implementation"),
        ("parallel-delegation", "implementation"),
        ("delegation-attempt", "implementation"),
        ("plan", "planning"),
        ("outline", "planning"),
        ("implement", "implementation"),
        ("coder", "implementation"),
        ("researcher", "research"),
        ("reporter", "reporting"),
        ("verify", "verification"),
        ("backtrack", "revision"),
        ("finish", "finished"),
    ],
)
def test_action_schema_normalizes_common_stage_aliases(provided, expected):
    action = _action(ActionType.THINK, action_id=f"stage-{provided}", task="Continue", stage=provided)

    assert action.stage == expected


def test_action_schema_treats_blank_optional_target_as_absent():
    action = _action(
        ActionType.THINK,
        action_id="blank-optional-target",
        task="Continue",
        target_agent="   ",
    )

    assert action.target_agent is None


def test_delegate_still_rejects_blank_target_after_optional_normalization():
    with pytest.raises(ActionValidationError, match="target_agent"):
        _action(
            ActionType.DELEGATE,
            action_id="blank-delegate-target",
            task="Research this",
            target_agent="   ",
        )


def test_recall_memory_schema_requires_query_or_task():
    with pytest.raises(ActionValidationError, match="memory_query"):
        SPAction.create(ActionType.RECALL_MEMORY, reason="Need prior context")

    action = SPAction.create(
        ActionType.RECALL_MEMORY,
        action_id="act-recall-schema",
        reason="Need prior context",
        metadata={"memory_query": "migration preferences"},
    )

    assert action.action_type == ActionType.RECALL_MEMORY


def test_router_executes_think_and_records_events_and_thread_state():
    router = build_default_action_router()
    action = _action(ActionType.THINK, action_id="act-think", idempotency_key="idem-think", task="Inspect state", stage="planning")

    result = router.execute(action, state={}, thread_id="thread-1", run_id="run-1")

    assert result.next_step == "continue"
    assert result.state_update["sp_current_stage"] == "planning"
    assert result.state_update["sp_last_action_id"] == "act-think"
    assert result.state_update["sp_last_idempotency_key"] == "idem-think"
    assert result.state_update["sp_loop_iteration"] == 1
    stack = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    assert stack.entries[-1].action == "think"
    assert stack.entries[-1].thread_id == "thread-1"
    assert [event["event_type"] for event in result.run_events[:2]] == ["sp.action.created", "sp.handler.started"]
    assert result.run_events[-1]["event_type"] == "sp.handler.completed"


def test_router_skips_duplicate_idempotency_key():
    router = build_default_action_router()
    state = {
        "sp_last_idempotency_key": "idem-1",
        "sp_last_handler_result": {"next_step": "finish", "action_type": "THINK"},
        "sp_loop_iteration": 3,
    }
    action = _action(ActionType.THINK, action_id="act-repeat", idempotency_key="idem-1", task="Repeat")

    result = router.execute(action, state=state)

    assert result.next_step == "finish"
    assert result.state_update["sp_loop_iteration"] == 4
    assert result.state_update["sp_last_handler_result"]["next_step"] == "finish"
    assert TaskMemoryStack.from_dict(result.state_update["sp_task_memory"]).entries == []
    assert result.run_events[0]["event_type"] == "sp.action.duplicate_skipped"


def test_router_skips_idempotency_key_seen_earlier_in_the_run():
    state = {
        "sp_idempotency_ledger": {
            "idem-1": {
                "next_step": "continue",
                "action_type": "DELEGATE",
                "action_id": "act-first",
            },
            "idem-2": {
                "next_step": "continue",
                "action_type": "THINK",
                "action_id": "act-second",
            },
        },
        "sp_last_idempotency_key": "idem-2",
        "sp_last_handler_result": {"next_step": "continue", "action_type": "THINK"},
        "sp_loop_iteration": 2,
    }
    action = _action(
        ActionType.DELEGATE,
        action_id="act-repeat",
        idempotency_key="idem-1",
        target_agent="reporter",
        task="Repeat",
    )

    result = build_default_action_router().execute(action, state=state)

    assert result.next_step == "continue"
    assert result.run_events[0]["event_type"] == "sp.action.duplicate_skipped"
    assert result.state_update["sp_idempotency_ledger"]["idem-1"]["action_id"] == "act-repeat"


def test_router_retries_recoverable_action_with_same_idempotency_key():
    state = {
        "sp_last_idempotency_key": "idem-retry",
        "sp_last_handler_result": {
            "next_step": "error_recoverable",
            "action_type": "THINK",
            "error": "temporary failure",
        },
        "sp_loop_iteration": 1,
    }
    action = _action(
        ActionType.THINK,
        action_id="act-retry",
        idempotency_key="idem-retry",
        task="Retry after recovery",
    )

    result = build_default_action_router().execute(action, state=state)

    assert result.next_step == "continue"
    assert result.state_update["sp_loop_iteration"] == 2
    stack = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    assert stack.entries[-1].content == "Retry after recovery"
    assert any(event["event_type"] == "sp.handler.completed" for event in result.run_events)


def test_router_blocks_consecutive_delegation_to_same_target_until_control_action():
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="should not run",
            task_id="unexpected",
        )
    )
    state = {
        "sp_last_handler_result": {
            "next_step": "continue",
            "action_type": "DELEGATE",
            "target_agent": "researcher",
            "action_id": "previous-research",
        }
    }
    action = _action(
        ActionType.DELEGATE,
        action_id="repeat-research",
        target_agent="researcher",
        task="Search another angle",
    )

    result = build_default_action_router(delegate_executor=executor).execute(action, state=state)

    assert result.next_step == "continue"
    assert executor.tasks == []
    assert result.memory_entries[0].action == "delegate_skipped"
    assert any(event["event_type"] == "sp.delegate.policy_blocked" for event in result.run_events)


def test_same_target_policy_block_is_a_checkpoint_not_a_permanent_deadlock():
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="retry completed",
            task_id="retry-completed",
        )
    )
    router = build_default_action_router(delegate_executor=executor)
    blocked = router.execute(
        _action(
            ActionType.DELEGATE,
            action_id="repeat-coder-blocked",
            target_agent="coder",
            task="Retry the failed test.",
        ),
        state={
            "sp_last_handler_result": {
                "next_step": "continue",
                "action_type": "DELEGATE",
                "target_agent": "coder",
                "action_id": "previous-coder",
            }
        },
        run_id="run-1",
    )

    retried = router.execute(
        _action(
            ActionType.DELEGATE,
            action_id="repeat-coder-after-checkpoint",
            target_agent="coder",
            task="Retry the failed test with a different repair.",
        ),
        state=blocked.state_update,
        run_id="run-1",
    )

    assert blocked.state_update["sp_last_handler_result"]["action_type"] == "THINK"
    assert blocked.state_update["sp_last_handler_result"]["policy_checkpoint"] == ("same_target_requires_intermediate_control_action")
    assert retried.next_step == "continue"
    assert [task.task for task in executor.tasks] == ["Retry the failed test with a different repair."]


def test_router_allows_delegation_to_new_target_after_previous_delegation():
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="coder completed",
            task_id="coder-1",
        )
    )
    action = _action(
        ActionType.DELEGATE,
        action_id="coder-after-research",
        target_agent="coder",
        task="Implement the verified change",
    )

    result = build_default_action_router(delegate_executor=executor).execute(
        action,
        state={
            "sp_last_handler_result": {
                "next_step": "continue",
                "action_type": "DELEGATE",
                "target_agent": "researcher",
            }
        },
    )

    assert result.next_step == "continue"
    assert [task.subagent_type for task in executor.tasks] == ["coder"]


def test_router_forces_reflection_before_a_new_action_after_failure():
    state = {
        "sp_last_idempotency_key": "failed-action",
        "sp_last_handler_result": {
            "next_step": "error_recoverable",
            "action_id": "failed-action-id",
            "action_type": "DELEGATE",
            "idempotency_key": "failed-action",
            "error": "researcher timed out",
        },
        "sp_loop_iteration": 1,
    }
    action = _action(ActionType.FINISH, action_id="finish-too-early", task="Finish now")

    result = build_default_action_router().execute(action, state=state)

    assert result.next_step == "continue"
    assert TaskMemoryStack.from_dict(result.state_update["sp_task_memory"]).entries[-1].action == "reflect"
    assert any(event["event_type"] == "sp.action.policy_reflection_forced" for event in result.run_events)


def test_targeted_reflection_backtracks_invalid_active_memory():
    stack = TaskMemoryStack()
    wrong = stack.append_observe(
        "The first calculation says plan A is cheaper.",
        actor="coder",
        stage="implementation",
    )
    retained = stack.append_observe(
        "The receipt total is 1,240 yuan.",
        actor="coder",
        stage="implementation",
    )
    action = _action(
        ActionType.REFLECT,
        action_id="reflect-wrong-calculation",
        task="The first calculation applied the coupon before the threshold check.",
        stage="verification",
        metadata={"target_entry_ids": [wrong.id]},
    )

    result = build_default_action_router().execute(
        action,
        state={"sp_task_memory": stack.to_dict()},
        thread_id="thread-1",
        run_id="run-1",
    )

    restored = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    by_id = {entry.id: entry for entry in restored.entries}
    assert result.next_step == "continue"
    assert by_id[wrong.id].status == "pruned"
    assert by_id[retained.id].status == "active"
    assert restored.entries[-2].action == "reflect"
    assert restored.entries[-1].action == "backtrack"
    assert restored.entries[-1].parent_ids == [wrong.id]
    assert any(event["event_type"] == "sp.memory.backtracked" and event["payload"]["triggered_by"] == "reflect" for event in result.run_events)


def test_targeted_reflection_is_idempotent_for_an_already_inactive_target():
    stack = TaskMemoryStack()
    wrong = stack.append_observe(
        "The failed API request used an invalid parameter format.",
        actor="coder",
        stage="implementation",
    )
    stack.mark_backtracked(
        [wrong.id],
        "The first reflection already invalidated this attempt.",
        stage="verification",
    )
    action = _action(
        ActionType.REFLECT,
        action_id="reflect-already-inactive",
        task="Diagnose the next materially different recovery.",
        stage="verification",
        metadata={"target_entry_ids": [wrong.id]},
    )

    result = build_default_action_router().execute(
        action,
        state={"sp_task_memory": stack.to_dict()},
        thread_id="thread-1",
        run_id="run-1",
    )

    restored = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    by_id = {entry.id: entry for entry in restored.entries}
    assert result.next_step == "continue"
    assert result.error is None
    assert by_id[wrong.id].status == "pruned"
    assert restored.entries[-1].action == "reflect"
    assert restored.entries[-1].metadata["skipped_inactive_target_ids"] == [wrong.id]
    assert not any(event["event_type"] == "sp.handler.failed" for event in result.run_events)
    assert any(event["event_type"] == "sp.reflect.targets_already_inactive" and event["payload"]["source_entry_ids"] == [wrong.id] for event in result.run_events)


def test_targeted_reflection_cannot_backtrack_pinned_human_memory():
    stack = TaskMemoryStack()
    feedback = stack.append_feedback("The budget cap is 5,000 yuan.")
    action = _action(
        ActionType.REFLECT,
        action_id="reflect-pinned-feedback",
        task="Discard the budget cap.",
        metadata={"target_entry_ids": [feedback.id]},
    )

    result = build_default_action_router().execute(
        action,
        state={"sp_task_memory": stack.to_dict()},
    )

    assert result.next_step == "error_recoverable"
    assert "pinned or critical" in str(result.error)
    restored = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    assert restored.entries[0].status == "pinned"


def test_targeted_reflection_cannot_backtrack_non_pinned_human_request():
    stack = TaskMemoryStack()
    request = stack.append(
        StackMemoryEntry(
            action="user_request",
            actor="human",
            content="The budget is 5,000 yuan.",
            priority="high",
            status="active",
        )
    )
    action = _action(
        ActionType.REFLECT,
        action_id="reflect-human-request",
        task="Discard the user's budget.",
        metadata={"target_entry_ids": [request.id]},
    )

    result = build_default_action_router().execute(
        action,
        state={"sp_task_memory": stack.to_dict()},
    )

    assert result.next_step == "error_recoverable"
    assert "human-authored" in str(result.error)
    restored = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    assert restored.entries[0].status == "active"


def test_explicit_user_correction_forces_targeted_reflection_before_delegation():
    stack = TaskMemoryStack()
    old_observation = stack.append_observe(
        "Store A is cheaper.",
        actor="coder",
        run_id="run-old",
    )
    old_answer = stack.append_think(
        "Choose Store A.",
        actor="central",
        run_id="run-old",
    )
    stack.append(
        StackMemoryEntry(
            action="user_request",
            actor="human",
            content="店员前面说错了，请重新检查上次的结论。",
            run_id="run-new",
            priority="high",
        )
    )
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Store B is cheaper.",
            task_id="recalculate",
        )
    )
    action = _action(
        ActionType.DELEGATE,
        action_id="delegate-recalculation",
        target_agent="coder",
        task="Recalculate from the corrected rule.",
    )

    first = build_default_action_router(delegate_executor=executor).execute(
        action,
        state={"sp_task_memory": stack.to_dict()},
        thread_id="thread-1",
        run_id="run-new",
    )

    restored = TaskMemoryStack.from_dict(first.state_update["sp_task_memory"])
    by_id = {entry.id: entry for entry in restored.entries}
    assert executor.tasks == []
    assert by_id[old_observation.id].status == "pruned"
    assert by_id[old_answer.id].status == "pruned"
    assert restored.entries[-2].action == "reflect"
    assert restored.entries[-1].action == "backtrack"
    assert any(event["event_type"] == "sp.action.correction_reflection_forced" for event in first.run_events)

    second = build_default_action_router(delegate_executor=executor).execute(
        _action(
            ActionType.DELEGATE,
            action_id="delegate-recalculation-after-reflect",
            target_agent="coder",
            task="Recalculate from the corrected rule.",
        ),
        state=first.state_update,
        thread_id="thread-1",
        run_id="run-new",
    )

    assert second.next_step == "continue"
    assert [task.task for task in executor.tasks] == ["Recalculate from the corrected rule."]


def test_natural_user_phrase_saying_prior_statement_has_an_error_forces_reflection():
    stack = TaskMemoryStack()
    for index in range(3):
        stack.append_observe(
            f"Older report conclusion {index}.",
            actor="reporter",
            run_id="run-old",
        )
    old_answer = stack.append_think(
        "South gross margin is below the threshold.",
        actor="central",
        run_id="run-old",
    )
    stack.append(
        StackMemoryEntry(
            action="user_request",
            actor="human",
            content="我看了一下，你刚才对南区门槛的说法有错，请反查并修订刚才的文件。",
            run_id="run-new",
            priority="high",
        )
    )
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Corrected.",
            task_id="correct-report",
            artifact_content="# Corrected report",
            artifact_type="report_revision",
        )
    )
    router = build_default_action_router(delegate_executor=executor)

    result = router.execute(
        _action(
            ActionType.DELEGATE,
            action_id="delegate-correct-report",
            target_agent="reporter",
            task="Correct the report.",
            metadata={"revision_reason": "The user corrected the prior conclusion."},
        ),
        state={"sp_task_memory": stack.to_dict()},
        thread_id="thread-1",
        run_id="run-new",
    )

    restored = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    assert next(entry for entry in restored.entries if entry.id == old_answer.id).status == "pruned"
    assert any(event["event_type"] == "sp.action.correction_reflection_forced" for event in result.run_events)
    assert executor.tasks == []

    resumed = router.execute(
        _action(
            ActionType.DELEGATE,
            action_id="delegate-correct-report-after-reflection",
            target_agent="reporter",
            task="Correct the report.",
            metadata={"revision_reason": "The user corrected the prior conclusion."},
        ),
        state=result.state_update,
        thread_id="thread-1",
        run_id="run-new",
    )

    assert resumed.next_step == "continue"
    assert len(executor.tasks) == 1
    assert not any(event["event_type"] == "sp.action.correction_reflection_forced" for event in resumed.run_events)


def test_natural_user_phrase_saying_prior_answer_omitted_correction_forces_reflection():
    stack = TaskMemoryStack()
    old_answer = stack.append_think(
        "South is not ready because churn is high.",
        actor="central",
        run_id="run-old",
    )
    stack.append(
        StackMemoryEntry(
            action="user_request",
            actor="human",
            content="我发现你刚才遗漏了一个关键纠错点，请反查上一步再生成报告。",
            run_id="run-new",
            priority="high",
        )
    )

    result = build_default_action_router(
        delegate_executor=FakeSubagentExecutor(
            SPSubagentResult(
                status=SPSubagentStatus.COMPLETED,
                result="Corrected.",
                task_id="correct-omission",
            )
        )
    ).execute(
        _action(
            ActionType.DELEGATE,
            action_id="delegate-correct-omission",
            target_agent="reporter",
            task="Correct the omitted detail.",
            metadata={"revision_reason": "The prior answer omitted a correction."},
        ),
        state={"sp_task_memory": stack.to_dict()},
        thread_id="thread-1",
        run_id="run-new",
    )

    restored = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    assert next(entry for entry in restored.entries if entry.id == old_answer.id).status == "pruned"
    assert any(event["event_type"] == "sp.action.correction_reflection_forced" for event in result.run_events)


def test_ordinary_follow_up_does_not_force_correction_reflection():
    stack = TaskMemoryStack()
    stack.append_think("Choose Store A.", actor="central", run_id="run-old")
    stack.append(
        StackMemoryEntry(
            action="user_request",
            actor="human",
            content="继续给我两个购买建议。",
            run_id="run-new",
            priority="high",
        )
    )
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Two suggestions.",
            task_id="suggest",
        )
    )

    result = build_default_action_router(delegate_executor=executor).execute(
        _action(
            ActionType.DELEGATE,
            action_id="delegate-suggestions",
            target_agent="coder",
            task="Prepare two suggestions.",
        ),
        state={"sp_task_memory": stack.to_dict()},
        thread_id="thread-1",
        run_id="run-new",
    )

    assert result.next_step == "continue"
    assert len(executor.tasks) == 1
    assert not any(event["event_type"] == "sp.action.correction_reflection_forced" for event in result.run_events)


def test_user_correction_reflects_before_model_attempts_memory_revision():
    stack = TaskMemoryStack()
    old_answer = stack.append_think(
        "The old room assignment is valid.",
        actor="central",
        run_id="run-old",
    )
    stack.append(
        StackMemoryEntry(
            action="user_request",
            actor="human",
            content="我刚收到消息，上次的安排不对，请重新检查。",
            run_id="run-new",
            priority="high",
        )
    )

    result = build_default_action_router().execute(
        _action(
            ActionType.REVISE,
            action_id="revise-after-user-correction",
            task="Use the corrected arrangement.",
            metadata={"target_entry_ids": [old_answer.id]},
        ),
        state={"sp_task_memory": stack.to_dict()},
        thread_id="thread-1",
        run_id="run-new",
    )

    restored = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    assert result.next_step == "continue"
    assert next(entry for entry in restored.entries if entry.id == old_answer.id).status == "pruned"
    assert restored.entries[-2].action == "reflect"
    assert restored.entries[-1].action == "backtrack"
    assert not any(entry.action == "revise" for entry in restored.entries)


def test_router_does_not_discard_a_corrected_retry_of_the_same_action_type():
    state = {
        "sp_last_idempotency_key": "failed-delegate",
        "sp_last_handler_result": {
            "next_step": "error_recoverable",
            "action_id": "failed-delegate-id",
            "action_type": "DELEGATE",
            "idempotency_key": "failed-delegate",
            "error": "SP action requested unavailable Tools: ['markdown']",
        },
        "sp_loop_iteration": 1,
    }
    action = _action(
        ActionType.DELEGATE,
        action_id="corrected-delegate",
        target_agent="reporter",
        task="Write the report with the role's default tools",
    )

    result = build_default_action_router().execute(action, state=state)

    assert result.next_step == "error_recoverable"
    assert "No handler registered" in str(result.error)
    assert not any(event["event_type"] == "sp.action.policy_reflection_forced" for event in result.run_events)


def test_router_allows_delegate_recovery_after_finish_artifact_rejection():
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Reporter created the missing report.",
            task_id="report-recovery",
            artifact_content="# Recovered report",
            artifact_type="report_revision",
            artifact_metadata={
                "completion_status": "complete",
                "evidence_gaps": [],
            },
        )
    )
    state = {
        "sp_last_idempotency_key": "failed-finish",
        "sp_last_handler_result": {
            "next_step": "error_recoverable",
            "action_id": "failed-finish-id",
            "action_type": "FINISH",
            "idempotency_key": "failed-finish",
            "error": ("FINISH requires required artifact type 'report' and a matching final_artifact_ref"),
        },
        "sp_loop_iteration": 1,
    }
    action = _action(
        ActionType.DELEGATE,
        action_id="recover-with-reporter",
        target_agent="reporter",
        task="Create the required report.",
        stage="revision",
    )

    result = build_default_action_router(
        delegate_executor=executor,
    ).execute(action, state=state)

    assert [task.subagent_type for task in executor.tasks] == ["reporter"]
    assert not any(event["event_type"] == "sp.action.policy_reflection_forced" for event in result.run_events)


def test_router_does_not_reopen_answered_duplicate_human_request():
    state = {
        "sp_last_idempotency_key": "idem-human",
        "sp_last_handler_result": {"next_step": "interrupt", "action_type": "ASK_HUMAN"},
        "sp_pending_human_interaction": None,
        "sp_loop_iteration": 1,
    }
    action = _action(
        ActionType.ASK_HUMAN,
        action_id="act-human",
        idempotency_key="idem-human",
        task="Already answered?",
    )

    result = build_default_action_router().execute(action, state=state)

    assert result.next_step == "continue"
    assert result.state_update["sp_loop_iteration"] == 2
    assert "sp_pending_human_interaction" not in result.state_update
    assert result.run_events[0]["event_type"] == "sp.action.duplicate_skipped"


def test_router_rejects_idempotency_key_collision_across_action_types():
    state = {
        "sp_last_idempotency_key": "idem-collision",
        "sp_last_handler_result": {"next_step": "continue", "action_type": "THINK"},
    }
    action = _action(
        ActionType.SUMMARIZE,
        action_id="act-summary",
        idempotency_key="idem-collision",
        task="Do not silently reuse this key",
    )

    result = build_default_action_router().execute(action, state=state)

    assert result.next_step == "error_recoverable"
    assert "already used by THINK" in str(result.error)
    assert result.state_update["sp_loop_iteration"] == 1
    assert result.run_events[0]["event_type"] == "sp.action.idempotency_collision"


def test_summarize_handler_condenses_sources_without_touching_pinned_feedback():
    stack = TaskMemoryStack()
    first = stack.append_think("old plan")
    feedback = stack.append_feedback("Pinned feedback")
    second = stack.append_observe("research", actor="researcher", metadata={"tool_call_id": "search-1"})
    state = {"sp_task_memory": stack.to_dict()}
    action = _action(
        ActionType.SUMMARIZE,
        action_id="act-summary",
        task="Summary of useful parts",
        metadata={"source_entry_ids": [first.id, feedback.id, second.id]},
    )

    result = build_default_action_router().execute(action, state=state)
    restored = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])

    by_id = {entry.id: entry for entry in restored.entries}
    assert first.id not in by_id
    assert by_id[feedback.id].status == "pinned"
    assert second.id not in by_id
    assert restored.entries[-1].action == "summarize"
    popped_event = next(event for event in result.run_events if event["event_type"] == "sp.memory.popped")
    assert popped_event["payload"]["source_entry_ids"] == [first.id, second.id]
    condensed_event = next(event for event in result.run_events if event["event_type"] == "sp.memory.condensed")
    assert condensed_event["payload"]["source_entry_ids"] == [first.id, feedback.id, second.id]
    assert condensed_event["payload"]["popped_entry_ids"] == [first.id, second.id]
    assert condensed_event["payload"]["summary_entry_id"] == restored.entries[-1].id
    assert result.state_update["sp_consumed_tool_observation_ids"] == ["search-1"]


def test_summarize_handler_rejects_a_large_noncompressing_summary():
    stack = TaskMemoryStack()
    sources = [stack.append_observe("evidence-" + ("x" * 290), actor="researcher") for _ in range(4)]
    state = {"sp_task_memory": stack.to_dict()}
    action = _action(
        ActionType.SUMMARIZE,
        action_id="act-summary-too-long",
        task="s" * 1000,
        metadata={"source_entry_ids": [entry.id for entry in sources]},
    )

    result = build_default_action_router().execute(action, state=state)
    restored = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])

    assert result.next_step == "error_recoverable"
    assert "did not compress" in str(result.error)
    assert not any(entry.action == "summarize" for entry in restored.entries)
    assert {entry.id for entry in restored.entries} == {entry.id for entry in sources}
    assert any(event["event_type"] == "sp.memory.summary_rejected" for event in result.run_events)


def test_summarize_retains_an_explicit_requirement_omitted_by_model_summary():
    stack = TaskMemoryStack()
    update = stack.append(
        StackMemoryEntry(
            action="user_request",
            actor="human",
            content="修订：主题改为星桥，旧主题作废。只确认收到。",
        )
    )
    requirement = stack.append(
        StackMemoryEntry(
            action="user_request",
            actor="human",
            content="补充偏好：流程轻松，不安排高强度团建；这不是可牺牲约束。只确认收到。",
        )
    )
    action = _action(
        ActionType.SUMMARIZE,
        action_id="act-summary-preserve-requirement",
        task="当前主题为星桥。",
        metadata={"source_entry_ids": [update.id, requirement.id]},
    )

    result = build_default_action_router().execute(
        action,
        state={"sp_task_memory": stack.to_dict()},
    )
    summary = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"]).entries[-1]

    assert "流程轻松" in summary.content
    assert "高强度团建" in summary.content
    assert summary.metadata["retained_authoritative_source_entry_ids"] == [requirement.id]
    assert any(event["event_type"] == "sp.memory.summary_requirements_retained" for event in result.run_events)


def test_summarize_does_not_duplicate_a_semantically_covered_requirement():
    stack = TaskMemoryStack()
    requirement = stack.append(
        StackMemoryEntry(
            action="user_request",
            actor="human",
            content="永久约束A：禁酒；所有后续方案都必须满足。只确认收到。",
        )
    )
    action = _action(
        ActionType.SUMMARIZE,
        action_id="act-summary-covered-requirement",
        task="硬性约束：禁酒。",
        metadata={"source_entry_ids": [requirement.id]},
    )

    result = build_default_action_router().execute(
        action,
        state={"sp_task_memory": stack.to_dict()},
    )
    summary = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"]).entries[-1]

    assert summary.content == "硬性约束：禁酒。"
    assert "retained_authoritative_source_entry_ids" not in summary.metadata


def test_summarize_handler_without_source_ids_pops_bounded_old_entries_across_runs():
    stack = TaskMemoryStack()
    old_entries = [stack.append_think(f"old plan {index}", run_id="run-old") for index in range(8)]
    feedback = stack.append_feedback("Pinned feedback")
    recent = stack.append_observe("current research", actor="researcher", run_id="run-new")

    result = build_default_action_router().execute(
        _action(ActionType.SUMMARIZE, action_id="act-summary-fallback", task="Condensed result"),
        state={"sp_task_memory": stack.to_dict()},
    )
    restored = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    ids = {entry.id for entry in restored.entries}

    assert all(entry.id not in ids for entry in old_entries[:5])
    assert all(entry.id in ids for entry in old_entries[5:])
    assert recent.id in ids
    assert feedback.id in ids
    assert restored.entries[-1].action == "summarize"
    condensed_event = next(event for event in result.run_events if event["event_type"] == "sp.memory.condensed")
    assert condensed_event["payload"]["source_entry_count"] == 5


def test_summarize_cooldown_blocks_repeat_without_new_progress():
    stack = TaskMemoryStack()
    stack.append_think("old plan")
    first = _action(ActionType.SUMMARIZE, action_id="act-summary-once", task="Keep the verified plan")
    first_result = build_default_action_router().execute(
        first,
        state={"sp_task_memory": stack.to_dict()},
        run_id="run-1",
    )

    second = _action(ActionType.SUMMARIZE, action_id="act-summary-twice", task="Repeat summary")
    second_result = build_default_action_router().execute(
        second,
        state=first_result.state_update,
        run_id="run-1",
    )

    assert first_result.state_update["sp_summarize_committed_run_id"] == "run-1"
    assert second_result.next_step == "continue"
    assert "cooldown is active" in str(second_result.error)
    assert "sp_task_memory" not in second_result.state_update
    assert any(event["event_type"] == "sp.memory.summary_repeat_blocked" for event in second_result.run_events)


def test_summarize_can_run_again_after_meaningful_new_progress():
    stack = TaskMemoryStack()
    stack.append_think("old plan", run_id="run-1")
    first_result = build_default_action_router().execute(
        _action(ActionType.SUMMARIZE, action_id="act-summary-first", task="Keep the verified plan"),
        state={"sp_task_memory": stack.to_dict()},
        run_id="run-1",
    )
    progressed = TaskMemoryStack.from_dict(first_result.state_update["sp_task_memory"])
    for index in range(4):
        progressed.append_observe(
            f"new verified result {index}",
            actor="researcher",
            run_id="run-1",
        )

    second_result = build_default_action_router().execute(
        _action(ActionType.SUMMARIZE, action_id="act-summary-later", task="Compress the next completed stage"),
        state={**first_result.state_update, "sp_task_memory": progressed.to_dict()},
        run_id="run-1",
    )

    assert second_result.next_step == "continue"
    assert second_result.error is None
    restored = TaskMemoryStack.from_dict(second_result.state_update["sp_task_memory"])
    assert restored.entries[-1].action == "summarize"
    assert restored.entries[-1].content == "Compress the next completed stage"


def test_backtrack_marks_entries_after_target_and_preserves_artifact_history():
    stack = TaskMemoryStack()
    target = stack.append_think("safe checkpoint", stage="planning")
    doomed = stack.append_delegate("too broad delegation", stage="research")
    state = {
        "sp_task_memory": stack.to_dict(),
        "sp_active_delegate_id": "delegate-1",
        "sp_current_artifact_refs": {
            "report": {"artifact_id": "new", "type": "report", "version": 2},
            "_history": [
                {"artifact_id": "old", "type": "report", "version": 1},
                {"artifact_id": "new", "type": "report", "version": 2},
            ],
        },
    }
    action = _action(
        ActionType.BACKTRACK,
        action_id="act-backtrack",
        metadata={
            "backtrack_target_type": "entry",
            "backtrack_target_id": target.id,
            "rollback_scope": "full_working_state",
            "reason": "delegation went wide",
        },
        stage="planning",
    )

    result = build_default_action_router().execute(action, state=state)
    restored = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    by_id = {entry.id: entry for entry in restored.entries}

    assert by_id[doomed.id].status == "pruned"
    assert result.state_update["sp_active_delegate_id"] is None
    assert result.state_update["sp_current_artifact_refs"]["report"]["artifact_id"] == "new"
    assert restored.entries[-1].action == "backtrack"
    assert any(event["event_type"] == "sp.memory.backtracked" for event in result.run_events)


def test_revise_supersedes_wrong_memory_and_appends_corrected_memory():
    stack = TaskMemoryStack()
    wrong_plan = stack.append_think("Use an unofficial source", stage="research")
    wrong_observation = stack.append_observe("The result is confirmed", actor="deerflow", stage="research")
    state = {"sp_task_memory": stack.to_dict()}
    action = _action(
        ActionType.REVISE,
        action_id="act-revise",
        task="Use the official source and mark the result as unverified until checked.",
        stage="verification",
        metadata={
            "target_entry_ids": [wrong_plan.id, wrong_observation.id],
            "revision_reason": "The source was not authoritative and the result was not independently verified.",
        },
    )

    result = build_default_action_router().execute(action, state=state, thread_id="thread-1", run_id="run-1")

    restored = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    by_id = {entry.id: entry for entry in restored.entries}
    correction = restored.entries[-1]
    assert result.next_step == "continue"
    assert by_id[wrong_plan.id].status == "superseded"
    assert by_id[wrong_observation.id].status == "superseded"
    assert correction.action == "revise"
    assert correction.parent_ids == [wrong_plan.id, wrong_observation.id]
    assert correction.status == "active"
    assert [entry.id for entry in restored.get_active_entries()] == [correction.id]
    assert result.state_update["sp_current_stage"] == "verification"
    assert any(event["event_type"] == "sp.memory.revised" for event in result.run_events)


def test_revise_repairs_unique_one_character_typo_in_opaque_memory_id():
    stack = TaskMemoryStack()
    target = stack.append_backtrack(
        "The user correction invalidated the old conclusion.",
        run_id="run-2",
        metadata={"trigger": "reflect"},
    )
    mistyped_id = f"{target.id[:20]}{target.id[21:]}"
    action = _action(
        ActionType.REVISE,
        action_id="act-revise-near-memory-id",
        task="South margin already passes; only churn needs correction.",
        metadata={
            "target_entry_ids": [mistyped_id],
            "revision_reason": "Record the corrected conclusion.",
        },
    )

    result = build_default_action_router().execute(
        action,
        state={"sp_task_memory": stack.to_dict()},
        run_id="run-2",
    )

    restored = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    by_id = {entry.id: entry for entry in restored.entries}
    assert result.next_step == "continue"
    assert by_id[target.id].status == "superseded"
    assert restored.entries[-1].parent_ids == [target.id]
    normalized = next(event for event in result.run_events if event["event_type"] == "sp.action.target_normalized")
    assert normalized["payload"] == {
        "reason": "unique_one_edit_opaque_id",
        "replacements": [{"from": mistyped_id, "to": target.id}],
    }


def test_revise_cannot_pop_pinned_human_feedback():
    stack = TaskMemoryStack()
    feedback = stack.append_feedback("The final answer must use the requested format.")
    action = _action(
        ActionType.REVISE,
        action_id="act-revise-feedback",
        task="Ignore the requested format.",
        metadata={"target_entry_ids": [feedback.id]},
    )

    result = build_default_action_router().execute(action, state={"sp_task_memory": stack.to_dict()})

    assert result.next_step == "error_recoverable"
    assert "pinned or critical" in result.error
    restored = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    assert restored.entries[0].status == "pinned"
    assert len(restored.entries) == 1


def test_revise_cannot_replace_non_pinned_human_request():
    stack = TaskMemoryStack()
    request = stack.append(
        StackMemoryEntry(
            action="user_request",
            actor="human",
            content="小安只能和小陈或小丁同住。",
            priority="high",
            status="active",
        )
    )
    action = _action(
        ActionType.REVISE,
        action_id="act-revise-user-request",
        task="小安只能和小丁同住。",
        metadata={"target_entry_ids": [request.id]},
    )

    result = build_default_action_router().execute(
        action,
        state={"sp_task_memory": stack.to_dict()},
    )

    assert result.next_step == "error_recoverable"
    assert "human-authored" in str(result.error)
    restored = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    assert restored.entries[0].status == "active"
    assert len(restored.entries) == 1


def test_backtrack_restores_artifact_current_version_and_emits_change_event():
    state = {
        "sp_current_artifact_refs": {
            "report": {"artifact_id": "report-v1", "type": "report", "version": 1, "is_current": False},
            "report_revision": {"artifact_id": "report-v2", "type": "report_revision", "version": 2, "is_current": True},
            "_history": [
                {"artifact_id": "report-v1", "type": "report", "version": 1, "is_current": False},
                {"artifact_id": "report-v2", "type": "report_revision", "version": 2, "is_current": True},
            ],
        }
    }
    action = _action(
        ActionType.BACKTRACK,
        action_id="act-backtrack-report",
        metadata={
            "backtrack_target_type": "artifact_version",
            "backtrack_target_id": "report-v1",
            "rollback_scope": "artifact_refs",
            "reason": "Revision used stale evidence",
        },
    )

    result = build_default_action_router().execute(action, state=state, run_id="run-1")

    refs = result.state_update["sp_current_artifact_refs"]
    assert refs["report"]["is_current"] is True
    assert refs["report_revision"]["is_current"] is False
    assert [item["is_current"] for item in refs["_history"]] == [True, False]
    event = next(event for event in result.run_events if event["event_type"] == "sp.artifact.current_changed")
    assert event["payload"]["previous_artifact_id"] == "report-v2"
    assert event["payload"]["current_artifact_id"] == "report-v1"


def test_finish_handler_rejects_pending_human_and_accepts_final_ref():
    router = build_default_action_router()
    action = _action(ActionType.FINISH, action_id="act-finish", task="Done")

    rejected = router.execute(action, state={"sp_pending_human_interaction": {"status": "pending"}})
    assert rejected.next_step == "error_recoverable"
    assert "pending" in rejected.error

    accepted = router.execute(
        _action(ActionType.FINISH, action_id="act-finish-2", task="Done"),
        state={"sp_current_artifact_refs": {"report": {"artifact_id": "report-1"}}},
    )
    assert accepted.next_step == "finish"
    assert accepted.state_update["sp_current_stage"] == "finished"
    assert TaskMemoryStack.from_dict(accepted.state_update["sp_task_memory"]).entries[-1].action == "finish"


def test_finish_accepts_generated_output_but_not_intermediate_research():
    router = build_default_action_router()
    generated = router.execute(
        _action(ActionType.FINISH, action_id="finish-generated", task="Generated site is ready"),
        state={
            "sp_loop_run_id": "run-1",
            "sp_current_artifact_refs": {
                "generated_file": {
                    "artifact_id": "site",
                    "type": "generated_file",
                    "virtual_path": "/mnt/user-data/outputs/site/index.html",
                    "run_id": "run-1",
                }
            },
        },
        thread_id="thread-1",
        run_id="run-1",
    )
    intermediate = router.execute(
        _action(ActionType.FINISH, action_id="finish-research", task="Research is enough"),
        state={
            "sp_loop_run_id": "run-1",
            "sp_current_artifact_refs": {
                "research_observation": {
                    "artifact_id": "research",
                    "type": "research_observation",
                    "virtual_path": "/mnt/user-data/outputs/research.md",
                    "run_id": "run-1",
                }
            },
        },
        thread_id="thread-1",
        run_id="run-1",
    )

    assert generated.next_step == "finish"
    assert intermediate.next_step == "error_recoverable"
    assert "final artifact" in intermediate.error


def test_finish_uses_authoritative_result_preview_instead_of_stale_model_summary():
    preview = "A酒店总费用: 5250.48\nB酒店总费用: 5108.50\n结论: B酒店便宜141.98元\n"
    result = build_default_action_router().execute(
        _action(
            ActionType.FINISH,
            action_id="finish-grounded-result",
            task="A酒店4811.43元，应该选A酒店。",
            metadata={
                "required_artifact_type": "generated_file",
                "final_artifact_ref": "hotel-result",
            },
        ),
        state={
            "sp_loop_run_id": "run-1",
            "sp_current_artifact_refs": {
                "generated_file": {
                    "artifact_id": "hotel-result",
                    "type": "generated_file",
                    "virtual_path": "/mnt/user-data/outputs/hotel-result.txt",
                    "run_id": "run-1",
                    "metadata": {
                        "completion_status": "complete",
                        "finalization_preview": preview,
                    },
                }
            },
        },
        thread_id="thread-1",
        run_id="run-1",
    )

    assert result.next_step == "finish"
    assert result.state_update["sp_last_run_summary"] == preview.strip()
    stack = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    assert stack.entries[-1].content == preview.strip()
    assert any(event["event_type"] == "sp.finish.summary_grounded" for event in result.run_events)


def test_finish_rejects_partial_deliverable_even_with_explicit_ref():
    result = build_default_action_router().execute(
        _action(
            ActionType.FINISH,
            action_id="finish-partial-report",
            task="Report ready",
            metadata={
                "required_artifact_type": "report",
                "final_artifact_ref": "report-partial",
            },
        ),
        state={
            "sp_loop_run_id": "run-1",
            "sp_current_artifact_refs": {
                "report_revision": {
                    "artifact_id": "report-partial",
                    "type": "report_revision",
                    "virtual_path": "/mnt/user-data/outputs/report-partial.md",
                    "run_id": "run-1",
                    "is_current": True,
                    "metadata": {
                        "completion_status": "partial",
                        "evidence_gaps": ["One required marker is absent."],
                    },
                }
            },
        },
        run_id="run-1",
    )

    assert result.next_step == "error_recoverable"
    assert "required artifact type 'report'" in (result.error or "")


def test_finish_enforces_requested_artifact_family_and_explicit_ref():
    router = build_default_action_router()
    state = {
        "sp_loop_run_id": "run-1",
        "sp_current_artifact_refs": {
            "generated_file": {
                "artifact_id": "site-1",
                "type": "generated_file",
                "virtual_path": "/mnt/user-data/outputs/site.html",
                "run_id": "run-1",
            },
            "report_revision": {
                "artifact_id": "report-1",
                "type": "report_revision",
                "virtual_path": "/mnt/user-data/outputs/report.md",
                "run_id": "run-1",
                "is_current": True,
            },
        },
    }

    accepted = router.execute(
        _action(
            ActionType.FINISH,
            action_id="finish-report-family",
            task="Report ready",
            metadata={
                "required_artifact_type": "report",
                "final_artifact_ref": "report-1",
            },
        ),
        state=state,
        run_id="run-1",
    )
    rejected = router.execute(
        _action(
            ActionType.FINISH,
            action_id="finish-wrong-family",
            task="Report ready",
            metadata={
                "required_artifact_type": "report",
                "final_artifact_ref": "site-1",
            },
        ),
        state=state,
        run_id="run-1",
    )

    assert accepted.next_step == "finish"
    assert accepted.state_update["sp_last_final_artifact_ref"] == "report-1"
    assert rejected.next_step == "error_recoverable"
    assert "required artifact type 'report'" in rejected.error


def test_finish_repairs_mislabeled_type_when_exact_ref_matches_report_intent():
    result = build_default_action_router().execute(
        _action(
            ActionType.FINISH,
            action_id="finish-report-mislabeled",
            task="The Project Aurora board report is ready.",
            metadata={
                "required_artifact_type": "generated_file",
                "final_artifact_ref": "report-1",
            },
        ),
        state={
            "sp_loop_run_id": "run-1",
            "sp_current_artifact_refs": {
                "report_revision": {
                    "artifact_id": "report-1",
                    "type": "report_revision",
                    "virtual_path": "/mnt/user-data/outputs/report.md",
                    "run_id": "run-1",
                    "is_current": True,
                    "metadata": {"completion_status": "complete"},
                }
            },
        },
        run_id="run-1",
    )

    assert result.next_step == "finish"
    assert result.state_update["sp_last_final_artifact_ref"] == "report-1"
    normalized = next(event for event in result.run_events if event["event_type"] == "sp.finish.artifact_type_normalized")
    assert normalized["payload"]["from_type"] == "generated_file"
    assert normalized["payload"]["to_type"] == "report"


def test_finish_does_not_reuse_an_artifact_from_an_older_run():
    router = build_default_action_router()
    result = router.execute(
        _action(ActionType.FINISH, action_id="finish-new-run", task="Complete the new request"),
        state={
            "sp_loop_run_id": "old-run",
            "sp_current_artifact_refs": {
                "report_revision": {
                    "artifact_id": "old-greeting-report",
                    "type": "report_revision",
                    "run_id": "old-run",
                    "is_current": True,
                }
            },
        },
        run_id="new-run",
    )

    assert result.next_step == "error_recoverable"
    assert result.error == "FINISH requires a final artifact ref or allow_without_artifact=true"


def test_finish_rejects_old_report_for_an_explicit_revision_request():
    result = build_default_action_router().execute(
        _action(
            ActionType.FINISH,
            action_id="finish-stale-revision",
            task="The revised report is ready",
            metadata={
                "required_artifact_type": "report",
                "final_artifact_ref": "report-v1",
            },
        ),
        state={
            "messages": [HumanMessage(content="修订刚才的报告并交付可下载 Markdown 文件。")],
            "sp_loop_run_id": "run-2",
            "sp_current_artifact_refs": {
                "report_revision": {
                    "artifact_id": "report-v1",
                    "type": "report_revision",
                    "run_id": "run-1",
                    "is_current": True,
                }
            },
        },
        run_id="run-2",
    )

    assert result.next_step == "error_recoverable"
    assert "current revision run" in (result.error or "")
    assert any(event["event_type"] == "sp.finish.rejected" and event["payload"]["reason"] == "stale_artifact_for_revision" for event in result.run_events)


def test_finish_accepts_new_report_from_the_current_revision_run():
    result = build_default_action_router().execute(
        _action(
            ActionType.FINISH,
            action_id="finish-current-revision",
            task="The revised report is ready",
            metadata={
                "required_artifact_type": "report",
                "final_artifact_ref": "report-v2",
            },
        ),
        state={
            "messages": [HumanMessage(content="修订刚才的报告并交付可下载 Markdown 文件。")],
            "sp_loop_run_id": "run-2",
            "sp_current_artifact_refs": {
                "report_revision": {
                    "artifact_id": "report-v2",
                    "type": "report_revision",
                    "run_id": "run-2",
                    "is_current": True,
                }
            },
        },
        run_id="run-2",
    )

    assert result.next_step == "finish"
    assert result.state_update["sp_last_final_artifact_ref"] == "report-v2"


def test_finish_rejects_a_hallucinated_explicit_artifact_ref():
    result = build_default_action_router().execute(
        _action(
            ActionType.FINISH,
            action_id="finish-hallucinated-ref",
            task="Done",
            metadata={"final_artifact_ref": "spart-does-not-exist"},
        ),
        state={
            "sp_current_artifact_refs": {
                "report": {
                    "artifact_id": "spart-real",
                    "type": "report",
                    "virtual_path": "/mnt/user-data/outputs/report.md",
                }
            }
        },
    )

    assert result.next_step == "error_recoverable"
    assert result.error == "FINISH requires a final artifact ref or allow_without_artifact=true"


def test_reporter_can_start_a_new_task_when_only_an_older_report_exists(tmp_path):
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="new report",
            task_id="new-report",
            artifact_content="# New report",
            artifact_type="report_revision",
        )
    )
    action = _action(
        ActionType.DELEGATE,
        action_id="new-report-action",
        target_agent="reporter",
        task="Create a report for the new user request",
    )

    result = build_default_action_router(delegate_executor=executor).execute(
        action,
        state={
            "thread_data": {"outputs_path": str(tmp_path)},
            "sp_current_artifact_refs": {
                "report_revision": {
                    "artifact_id": "old-greeting-report",
                    "type": "report_revision",
                    "run_id": "old-run",
                    "is_current": True,
                }
            },
        },
        thread_id="thread-1",
        run_id="new-run",
    )

    assert result.next_step == "continue"
    assert [task.action_id for task in executor.tasks] == ["new-report-action"]


def test_ask_human_interrupts_and_records_pending_interaction():
    state = {"sp_current_artifact_refs": {"outline": {"artifact_id": "outline-1", "type": "outline"}}}
    action = _action(
        ActionType.ASK_HUMAN,
        action_id="act-human",
        task="Please confirm the outline",
        metadata={"interaction_type": "outline_confirmation"},
        stage="planning",
    )

    result = build_default_action_router().execute(action, state=state, thread_id="thread-1", run_id="run-1")

    assert result.next_step == "interrupt"
    pending = result.state_update["sp_pending_human_interaction"]
    assert pending["status"] == "pending"
    assert pending["interaction_type"] == "outline_confirmation"
    assert pending["artifact_refs"]["outline"]["artifact_id"] == "outline-1"
    restored = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    assert restored.entries[-1].action == "ask_human"
    assert restored.entries[-1].content == "Please confirm the outline"
    assert any(event["event_type"] == "sp.human.requested" for event in result.run_events)


def test_record_human_feedback_clears_pending_and_pins_feedback():
    stack = TaskMemoryStack()
    state = {
        "sp_task_memory": stack.to_dict(),
        "sp_current_stage": "planning",
        "sp_pending_human_interaction": {
            "interaction_id": "hitl-1",
            "interaction_type": "outline_confirmation",
            "artifact_refs": {"outline": {"artifact_id": "outline-1", "type": "outline"}},
            "status": "pending",
        },
        "sp_current_artifact_refs": {
            "outline": {"artifact_id": "outline-1", "type": "outline", "feedback_entry_ids": []},
            "_history": [{"artifact_id": "outline-1", "type": "outline", "feedback_entry_ids": []}],
        },
    }

    result = record_human_feedback(state, "以后大纲先写结论再写证据", thread_id="thread-1", run_id="run-1")

    assert result.state_update["sp_pending_human_interaction"] is None
    restored = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    feedback = restored.entries[-1]
    assert feedback.action == "feedback"
    assert feedback.priority == "critical"
    assert feedback.status == "pinned"
    assert feedback.metadata["interaction_id"] == "hitl-1"
    assert result.state_update["sp_current_artifact_refs"]["outline"]["feedback_entry_ids"] == [feedback.id]
    assert result.state_update["sp_current_artifact_refs"]["_history"][0]["feedback_entry_ids"] == [feedback.id]


def test_recall_memory_requires_memory_recaller_executor():
    action = _action(
        ActionType.RECALL_MEMORY,
        action_id="act-recall-no-executor",
        metadata={"memory_query": "migration preferences"},
    )

    result = build_default_action_router().execute(action, state={})

    assert result.next_step == "error_recoverable"
    assert "memory_recaller subagent executor" in result.error
    assert any(event["event_type"] == "sp.handler.failed" for event in result.run_events)


def test_router_rejects_unregistered_subagent_actions_until_phase_3():
    action = _action(ActionType.DELEGATE, action_id="act-delegate", target_agent="researcher", task="Research")

    result = build_default_action_router().execute(action, state={})

    assert result.next_step == "error_recoverable"
    assert "No handler registered" in result.error


class FakeSubagentExecutor:
    def __init__(self, result: SPSubagentResult):
        self.result = result
        self.tasks: list[SPSubagentTask] = []

    def execute(self, task: SPSubagentTask) -> SPSubagentResult:
        self.tasks.append(task)
        return self.result


def _thread_state_with_outputs(tmp_path):
    return {
        "thread_data": {
            "outputs_path": str(tmp_path / "threads" / "thread-1" / "user-data" / "outputs"),
        }
    }


def test_delegate_handler_calls_executor_and_externalizes_large_result(tmp_path):
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Report artifact created",
            task_id="task-1",
            artifact_content="# Draft report\n\nLarge report body",
            artifact_type="report_revision",
        )
    )
    state = _thread_state_with_outputs(tmp_path)
    action = _action(
        ActionType.DELEGATE,
        action_id="act-delegate-ok",
        target_agent="reporter",
        task="Draft the report",
        input_refs=["artifact://outline"],
        expected_output="report artifact",
        stage="reporting",
    )

    result = build_default_action_router(delegate_executor=executor).execute(action, state=state, thread_id="thread-1", run_id="run-1")

    assert result.next_step == "continue"
    assert executor.tasks[0].subagent_type == "reporter"
    assert executor.tasks[0].input_refs == ["artifact://outline"]
    assert "task_memory" in executor.tasks[0].context_refs
    restored = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    assert [entry.action for entry in restored.entries[-2:]] == ["delegate", "observe"]
    assert restored.entries[-1].actor == "reporter"
    assert result.state_update["sp_active_delegate_id"] is None
    assert result.state_update["artifacts"][0].startswith("/mnt/user-data/outputs/sp/report_revision/")
    report_ref = result.state_update["sp_current_artifact_refs"]["report_revision"]
    assert report_ref["artifact_id"] == restored.entries[-1].result_ref
    assert "Large report body" not in str(report_ref)
    assert any(event["event_type"] == "sp.delegate.completed" for event in result.run_events)
    current_event = next(event for event in result.run_events if event["event_type"] == "sp.artifact.current_changed")
    assert current_event["payload"]["previous_artifact_id"] is None
    assert current_event["payload"]["current_artifact_id"] == report_ref["artifact_id"]


def test_perception_acceptance_converts_decision_critical_gaps_to_questions(
    tmp_path,
):
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Task brief normalized.",
            task_id="perception-missing-inputs",
            artifact_content=("Objective: Create a board report.\n\nDecision-Critical Missing Information: reporting period, region metrics, and the source documents."),
            artifact_type="perception_observation",
            artifact_metadata={
                "completion_status": "complete",
                "clarification_questions": [],
            },
        )
    )

    result = build_default_action_router(delegate_executor=executor).execute(
        _action(
            ActionType.DELEGATE,
            action_id="perception-missing-inputs",
            target_agent="perception",
            task="Normalize the board report request.",
            stage="perception",
        ),
        state=_thread_state_with_outputs(tmp_path),
        thread_id="thread-1",
        run_id="run-1",
    )

    ref = result.state_update["sp_current_artifact_refs"]["perception_observation"]
    assert ref["metadata"]["task_ready"] is False
    assert ref["metadata"]["clarification_questions"] == [("Please provide or confirm the following decision-critical information before I continue: reporting period, region metrics, and the source documents.")]


def test_perception_acceptance_infers_inputs_for_unsourced_report_request(
    tmp_path,
):
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Task brief normalized.",
            task_id="perception-no-explicit-gaps",
            artifact_content=("Objective: Create a board report.\n\nDecision-Critical Missing Information: None identified yet."),
            artifact_type="perception_observation",
            artifact_metadata={
                "completion_status": "complete",
                "task_ready": True,
                "clarification_questions": [],
            },
        )
    )
    state = {
        **_thread_state_with_outputs(tmp_path),
        "messages": [HumanMessage(content="请为 Project Aurora 写一份董事会复盘报告。")],
    }

    result = build_default_action_router(delegate_executor=executor).execute(
        _action(
            ActionType.DELEGATE,
            action_id="perception-no-explicit-gaps",
            target_agent="perception",
            task="Normalize the board report request.",
            stage="perception",
        ),
        state=state,
        thread_id="thread-1",
        run_id="run-1",
    )

    ref = result.state_update["sp_current_artifact_refs"]["perception_observation"]
    questions = ref["metadata"]["clarification_questions"]
    assert ref["metadata"]["task_ready"] is False
    assert len(questions) == 1
    assert "reporting period" in questions[0]
    assert "source documents or data" in questions[0]
    assert "None identified" not in questions[0]


def test_reporter_delegate_inherits_latest_uploaded_file_refs(tmp_path):
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Report artifact created",
            task_id="task-upload-report",
            artifact_content="# Report",
            artifact_type="report_revision",
        )
    )
    state = {
        **_thread_state_with_outputs(tmp_path),
        "messages": [
            HumanMessage(
                content="Use the uploaded evidence.",
                additional_kwargs={
                    "files": [
                        {
                            "filename": "evidence.md",
                            "path": "/mnt/user-data/uploads/evidence.md",
                        }
                    ]
                },
            )
        ],
    }
    action = _action(
        ActionType.DELEGATE,
        action_id="delegate-upload-report",
        target_agent="reporter",
        task="Write the report",
        input_refs=["artifact://brief"],
        stage="reporting",
    )

    build_default_action_router(delegate_executor=executor).execute(
        action,
        state=state,
        thread_id="thread-1",
        run_id="run-1",
    )

    assert executor.tasks[0].input_refs == [
        "artifact://brief",
        "/mnt/user-data/uploads/evidence.md",
    ]


def test_perception_delegate_materializes_every_latest_text_upload(tmp_path):
    uploads = tmp_path / "threads" / "thread-1" / "user-data" / "uploads"
    uploads.mkdir(parents=True)
    (uploads / "current.md").write_text(
        "Current authoritative metrics.",
        encoding="utf-8",
    )
    (uploads / "policy.md").write_text(
        "Readiness policy.",
        encoding="utf-8",
    )
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Task brief ready.",
            task_id="task-upload-perception",
            artifact_content="# Complete task brief",
            artifact_type="perception_observation",
            artifact_metadata={
                "completion_status": "complete",
                "task_ready": True,
                "clarification_questions": [],
            },
        )
    )
    state = {
        "thread_data": {
            "outputs_path": str(tmp_path / "threads" / "thread-1" / "user-data" / "outputs"),
            "uploads_path": str(uploads),
        },
        "messages": [
            HumanMessage(
                content="Use every uploaded file.",
                additional_kwargs={
                    "files": [
                        {
                            "filename": "current.md",
                            "path": "/mnt/user-data/uploads/current.md",
                        },
                        {
                            "filename": "policy.md",
                            "path": "/mnt/user-data/uploads/policy.md",
                        },
                    ]
                },
            )
        ],
    }

    build_default_action_router(delegate_executor=executor).execute(
        _action(
            ActionType.DELEGATE,
            action_id="delegate-upload-perception",
            target_agent="perception",
            task="Normalize all supplied evidence.",
            stage="perception",
        ),
        state=state,
        thread_id="thread-1",
        run_id="run-1",
    )

    task = executor.tasks[0]
    assert task.input_refs == [
        "/mnt/user-data/uploads/current.md",
        "/mnt/user-data/uploads/policy.md",
    ]
    bundle = task.context_refs["uploaded_file_bodies"]
    assert bundle["candidate_count"] == 2
    assert bundle["materialized_count"] == 2
    assert bundle["unmaterialized_refs"] == []
    assert [item["virtual_path"] for item in bundle["items"]] == task.input_refs
    assert [item["content"] for item in bundle["items"]] == [
        "Current authoritative metrics.",
        "Readiness policy.",
    ]


def test_reporter_delegate_preserves_write_file_when_tools_are_narrowed(tmp_path):
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Report artifact created",
            task_id="task-report-tools",
            artifact_content="# Report",
            artifact_type="report_revision",
        )
    )
    action = _action(
        ActionType.DELEGATE,
        action_id="delegate-report-tools",
        target_agent="reporter",
        task="Write the report",
        stage="reporting",
        metadata={"tool_names": ["read_file"]},
    )

    build_default_action_router(delegate_executor=executor).execute(
        action,
        state=_thread_state_with_outputs(tmp_path),
        thread_id="thread-1",
        run_id="run-1",
    )

    assert executor.tasks[0].metadata["tool_names"] == ["read_file", "write_file"]


def test_coder_delegate_preserves_file_edit_and_execution_tools_when_narrowed(
    tmp_path,
):
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Script created and tests passed",
            task_id="task-coder-tools",
            artifact_content="print('ok')\n",
            artifact_type="generated_file",
        )
    )
    action = _action(
        ActionType.DELEGATE,
        action_id="delegate-coder-tools",
        target_agent="coder",
        task="Write and run a Python script, then provide the file.",
        stage="implementation",
        metadata={"tool_names": ["bash"]},
    )

    build_default_action_router(delegate_executor=executor).execute(
        action,
        state=_thread_state_with_outputs(tmp_path),
        thread_id="thread-1",
        run_id="run-1",
    )

    assert executor.tasks[0].metadata["tool_names"] == [
        "bash",
        "read_file",
        "write_file",
        "str_replace",
    ]


def test_coder_delegate_defaults_to_local_file_and_execution_tools(tmp_path):
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Exact calculation completed",
            task_id="task-coder-default-tools",
            artifact_content="6\n",
            artifact_type="generated_file",
        )
    )
    action = _action(
        ActionType.DELEGATE,
        action_id="delegate-coder-default-tools",
        target_agent="coder",
        task="Enumerate a self-contained pairing puzzle exactly.",
        stage="implementation",
    )

    build_default_action_router(delegate_executor=executor).execute(
        action,
        state=_thread_state_with_outputs(tmp_path),
        thread_id="thread-1",
        run_id="run-1",
    )

    assert executor.tasks[0].metadata["tool_names"] == [
        "read_file",
        "write_file",
        "str_replace",
        "bash",
    ]


def test_self_contained_exact_enumeration_is_rerouted_from_researcher_to_coder(
    tmp_path,
):
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Exactly six canonical assignments satisfy every constraint.",
            task_id="task-exact-enumeration",
            artifact_content="6\n",
            artifact_type="generated_file",
        )
    )
    action = _action(
        ActionType.DELEGATE,
        action_id="delegate-wrong-research-role",
        target_agent="researcher",
        task="Generate all valid room assignment configurations under the supplied constraints.",
        expected_output="Every valid configuration and the exact count.",
        stage="research",
    )

    result = build_default_action_router(delegate_executor=executor).execute(
        action,
        state=_thread_state_with_outputs(tmp_path),
        thread_id="thread-1",
        run_id="run-1",
    )

    task = executor.tasks[0]
    assert task.subagent_type == "coder"
    assert task.metadata["declared_target_agent"] == "researcher"
    assert task.metadata["tool_names"] == [
        "read_file",
        "write_file",
        "str_replace",
        "bash",
    ]
    assert any(event["event_type"] == "sp.delegate.target_normalized" and event["payload"]["from_target"] == "researcher" and event["payload"]["to_target"] == "coder" for event in result.run_events)


def test_read_only_external_api_retrieval_is_rerouted_from_coder_to_researcher(
    tmp_path,
):
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Official API values collected.",
            task_id="task-api-retrieval",
            artifact_content="[]",
            artifact_type="research_observation",
        )
    )
    action = _action(
        ActionType.DELEGATE,
        action_id="delegate-api-to-wrong-role",
        target_agent="coder",
        task=("Call the official World Bank Indicators API and retrieve the requested country values as JSON."),
        expected_output="Verified values from the official API endpoint.",
        stage="implementation",
    )

    result = build_default_action_router(delegate_executor=executor).execute(
        action,
        state=_thread_state_with_outputs(tmp_path),
        thread_id="thread-1",
        run_id="run-1",
    )

    task = executor.tasks[0]
    assert task.subagent_type == "researcher"
    assert task.metadata["declared_target_agent"] == "coder"
    delegated = next(entry for entry in TaskMemoryStack.from_dict(result.state_update["sp_task_memory"]).entries if entry.action == "delegate")
    assert delegated.stage == "research"
    assert any(
        event["event_type"] == "sp.delegate.target_normalized"
        and event["payload"]
        == {
            "from_target": "coder",
            "to_target": "researcher",
            "reason": "read_only_remote_api_retrieval",
        }
        for event in result.run_events
    )


def test_closed_user_query_rejects_ungrounded_delegated_identifier(tmp_path):
    stack = TaskMemoryStack()
    for content in (
        "初始指标代码 NY.GDP.MKTP.CD。",
        "改为 NY.GDP.PCAP.CD，旧指标作废。",
        "曾增加 SP.POP.TOTL，随后撤销。",
        "现在执行当前有效查询，只能使用已确认代码，不得出现已作废指标。",
    ):
        stack.append(
            StackMemoryEntry(
                action="user_request",
                actor="human",
                content=content,
                priority="high",
                status="active",
            )
        )
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="should not run",
        )
    )

    result = build_default_action_router(delegate_executor=executor).execute(
        _action(
            ActionType.DELEGATE,
            action_id="delegate-invented-indicator",
            target_agent="researcher",
            task=("Query the official API with invented code NY.HDI.METH.ME."),
            stage="research",
        ),
        state={
            **_thread_state_with_outputs(tmp_path),
            "sp_task_memory": stack.to_dict(),
        },
        thread_id="thread-1",
        run_id="run-1",
    )

    assert result.next_step == "error_recoverable"
    assert executor.tasks == []
    assert "NY.HDI.METH.ME" in str(result.error)
    assert "NY.GDP.PCAP.CD" in str(result.error)
    restored = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    assert restored.entries[-1].action == "delegate_skipped"
    assert any(event["event_type"] == "sp.delegate.ungrounded_identifier_rejected" for event in result.run_events)


def test_closed_user_query_allows_exact_user_provided_identifier(tmp_path):
    stack = TaskMemoryStack()
    for content in (
        "指标代码改为 NY.GDP.PCAP.CD。",
        "现在执行当前有效查询，只能使用该指标。",
    ):
        stack.append(
            StackMemoryEntry(
                action="user_request",
                actor="human",
                content=content,
                priority="high",
                status="active",
            )
        )
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Official value fetched.",
            artifact_content="[]",
            artifact_type="research_observation",
        )
    )

    result = build_default_action_router(delegate_executor=executor).execute(
        _action(
            ActionType.DELEGATE,
            action_id="delegate-grounded-indicator",
            target_agent="researcher",
            task="Query the official API with NY.GDP.PCAP.CD.",
            stage="research",
        ),
        state={
            **_thread_state_with_outputs(tmp_path),
            "sp_task_memory": stack.to_dict(),
        },
        thread_id="thread-1",
        run_id="run-1",
    )

    assert result.next_step == "continue"
    assert len(executor.tasks) == 1


def test_external_api_client_implementation_stays_with_coder(tmp_path):
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="API client implemented and tested.",
            task_id="task-api-client",
            artifact_content="print('ok')\n",
            artifact_type="generated_file",
        )
    )
    action = _action(
        ActionType.DELEGATE,
        action_id="delegate-api-client",
        target_agent="coder",
        task="Implement and test a Python client for the World Bank API.",
        expected_output="A tested Python script.",
        stage="implementation",
    )

    build_default_action_router(delegate_executor=executor).execute(
        action,
        state=_thread_state_with_outputs(tmp_path),
        thread_id="thread-1",
        run_id="run-1",
    )

    assert executor.tasks[0].subagent_type == "coder"


def test_external_evidence_enumeration_stays_with_researcher(tmp_path):
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Current sources collected.",
            task_id="task-external-research",
            artifact_content="Sources",
            artifact_type="research_observation",
        )
    )
    action = _action(
        ActionType.DELEGATE,
        action_id="delegate-real-research",
        target_agent="researcher",
        task="Search the web and list all current official room-assignment regulations.",
        expected_output="Official sources and citations.",
        stage="research",
    )

    build_default_action_router(delegate_executor=executor).execute(
        action,
        state=_thread_state_with_outputs(tmp_path),
        thread_id="thread-1",
        run_id="run-1",
    )

    assert executor.tasks[0].subagent_type == "researcher"


def test_reporter_literal_acceptance_guard_marks_missing_report_partial_and_retries_as_revision(
    tmp_path,
):
    uploads_path = tmp_path / "threads" / "thread-1" / "user-data" / "uploads"
    uploads_path.mkdir(parents=True)
    (uploads_path / "evidence.md").write_text(
        "Source label: `POLICY-V3`\nAudit marker: `AURORA-AUDIT-1`\n",
        encoding="utf-8",
    )

    class SequenceExecutor:
        def __init__(self):
            self.tasks = []
            self.results = [
                SPSubagentResult(
                    status=SPSubagentStatus.COMPLETED,
                    result="Draft report written.",
                    task_id="report-partial",
                    artifact_content="# Report\n\nUses `POLICY-V3`.",
                    artifact_type="report_revision",
                    artifact_metadata={
                        "completion_status": "complete",
                        "evidence_gaps": [],
                    },
                ),
                SPSubagentResult(
                    status=SPSubagentStatus.COMPLETED,
                    result="Corrected report written.",
                    task_id="report-complete",
                    artifact_content=("# Report\n\nUses `POLICY-V3` and preserves `AURORA-AUDIT-1`."),
                    artifact_type="report_revision",
                    artifact_metadata={
                        "completion_status": "complete",
                        "evidence_gaps": [],
                    },
                ),
            ]

        def execute(self, task):
            self.tasks.append(task)
            return self.results[len(self.tasks) - 1]

    executor = SequenceExecutor()
    router = build_default_action_router(delegate_executor=executor)
    base_state = {
        **_thread_state_with_outputs(tmp_path),
        "thread_data": {
            **_thread_state_with_outputs(tmp_path)["thread_data"],
            "uploads_path": str(uploads_path),
        },
        "messages": [
            HumanMessage(
                content="保留所有来源标签和审计标记。",
                additional_kwargs={
                    "files": [
                        {
                            "filename": "evidence.md",
                            "path": "/mnt/user-data/uploads/evidence.md",
                        }
                    ]
                },
            )
        ],
    }
    first = router.execute(
        _action(
            ActionType.DELEGATE,
            action_id="report-literal-first",
            target_agent="reporter",
            task="Write the Markdown report.",
            stage="reporting",
        ),
        state=base_state,
        thread_id="thread-1",
        run_id="run-1",
    )

    first_ref = first.state_update["sp_current_artifact_refs"]["report_revision"]
    first_observation = TaskMemoryStack.from_dict(first.state_update["sp_task_memory"]).entries[-1]
    assert first_ref["metadata"]["completion_status"] == "partial"
    assert first_ref["metadata"]["quality_checks"]["required_literal_tokens"] == {
        "required": ["POLICY-V3", "AURORA-AUDIT-1"],
        "missing": ["AURORA-AUDIT-1"],
        "passed": False,
    }
    assert first_observation.metadata["completion_status"] == "partial"

    second = router.execute(
        _action(
            ActionType.DELEGATE,
            action_id="report-literal-retry",
            target_agent="reporter",
            task="Correct the incomplete Markdown report.",
            stage="reporting",
        ),
        state={**base_state, **first.state_update},
        thread_id="thread-1",
        run_id="run-1",
    )

    second_ref = second.state_update["sp_current_artifact_refs"]["report_revision"]
    assert executor.tasks[1].metadata["revision_reason"].startswith("Complete the previous partial report")
    assert executor.tasks[1].metadata["source_artifact_ids"] == [first_ref["artifact_id"]]
    assert second_ref["version"] == 2
    assert second_ref["parent_artifact_ids"] == [first_ref["artifact_id"]]
    assert second_ref["metadata"]["completion_status"] == "complete"
    assert second_ref["metadata"]["quality_checks"]["required_literal_tokens"]["passed"] is True


def test_reporter_literal_acceptance_guard_extracts_bare_preservation_ids(tmp_path):
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Report written.",
            task_id="report-bare-literals",
            artifact_content=("# Report\n\nPreserves CORRECTION-JULY and AURORA-AUDIT-7319, but omits the historical action."),
            artifact_type="report_revision",
            artifact_metadata={
                "completion_status": "complete",
                "evidence_gaps": [],
            },
        )
    )
    state = {
        **_thread_state_with_outputs(tmp_path),
        "messages": [HumanMessage(content=("其他来源说明和历史审计都不要丢，尤其保留 CORRECTION-JULY、AURORA-AUDIT-7319、OLD-R7。"))],
    }

    result = build_default_action_router(delegate_executor=executor).execute(
        _action(
            ActionType.DELEGATE,
            action_id="report-bare-literals",
            target_agent="reporter",
            task="Revise the report while preserving every requested marker.",
            stage="revision",
            metadata={"revision_reason": "User supplied mandatory literal markers."},
        ),
        state=state,
        thread_id="thread-1",
        run_id="run-1",
    )

    ref = result.state_update["sp_current_artifact_refs"]["report_revision"]
    assert ref["metadata"]["completion_status"] == "partial"
    assert ref["metadata"]["quality_checks"]["required_literal_tokens"] == {
        "required": [
            "CORRECTION-JULY",
            "AURORA-AUDIT-7319",
            "OLD-R7",
        ],
        "missing": ["OLD-R7"],
        "passed": False,
    }


def test_reporter_tool_narrowing_keeps_read_and_write_for_safe_revision_retry(tmp_path):
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Report written.",
            task_id="report-tools",
            artifact_content="# Revised report",
            artifact_type="report_revision",
        )
    )

    build_default_action_router(delegate_executor=executor).execute(
        _action(
            ActionType.DELEGATE,
            action_id="report-tools",
            target_agent="reporter",
            task="Revise the report.",
            stage="revision",
            metadata={
                "revision_reason": "Apply the correction.",
                "tool_names": ["write_file"],
            },
        ),
        state=_thread_state_with_outputs(tmp_path),
        thread_id="thread-1",
        run_id="run-1",
    )

    assert executor.tasks[0].metadata["tool_names"] == [
        "write_file",
        "read_file",
    ]


def test_reporter_accepts_unique_tool_call_suffix_in_created_filename(tmp_path):
    outputs = tmp_path / "threads" / "thread-1" / "user-data" / "outputs"
    outputs.mkdir(parents=True)
    created = outputs / "report_revision-v3-spart_8d806b58caf7.md"
    created.write_text("# Revised report\n\nCorrected.", encoding="utf-8")
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Report written.",
            task_id="report-owned-suffix",
            artifact_type="report_revision",
            artifact_metadata={
                "completion_status": "complete",
                "evidence_gaps": [],
                "created_paths": ["/mnt/user-data/outputs/report_revision-v3-spart_8d806b58caf7.md"],
            },
        )
    )

    result = build_default_action_router(delegate_executor=executor).execute(
        _action(
            ActionType.DELEGATE,
            action_id="spact_call_8d806b58caf7",
            target_agent="reporter",
            task="Revise the report.",
            stage="revision",
            metadata={"revision_reason": "Apply the correction."},
        ),
        state=_thread_state_with_outputs(tmp_path),
        thread_id="thread-1",
        run_id="run-1",
    )

    ref = result.state_update["sp_current_artifact_refs"]["report_revision"]
    assert ref["metadata"]["completion_status"] == "complete"
    assert result.state_update["artifacts"] == ["/mnt/user-data/outputs/report_revision-v3-spart_8d806b58caf7.md"]
    assert ref["metadata"]["quality_checks"]["created_path_ownership"] == {
        "discarded_paths": [],
        "fallback_artifact_content": False,
        "passed": True,
    }


def test_reporter_acceptance_rejects_invented_and_misplaced_reference_ids(
    tmp_path,
):
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Report written.",
            task_id="report-invented-reference",
            artifact_content=("# Report\n\nUses `POLICY-V3` and invented `AUDIT-1`.\n\n| Region | Current action |\n| --- | --- |\n| South | `OLD-R7` |"),
            artifact_type="report_revision",
            artifact_metadata={
                "completion_status": "complete",
                "evidence_gaps": [],
            },
        )
    )
    state = {
        **_thread_state_with_outputs(tmp_path),
        "messages": [HumanMessage(content=("保留所有决策和审计标记 `POLICY-V3`、`OLD-R7`，禁止虚构编号。"))],
    }

    result = build_default_action_router(delegate_executor=executor).execute(
        _action(
            ActionType.DELEGATE,
            action_id="report-invented-reference",
            target_agent="reporter",
            task="Write the report.",
            stage="reporting",
        ),
        state=state,
        thread_id="thread-1",
        run_id="run-1",
    )

    ref = result.state_update["sp_current_artifact_refs"]["report_revision"]
    checks = ref["metadata"]["quality_checks"]
    assert ref["metadata"]["completion_status"] == "partial"
    assert checks["literal_token_provenance"]["introduced_tokens"] == ["AUDIT-1"]
    assert checks["literal_token_provenance"]["passed"] is False
    assert checks["source_precedence"]["passed"] is False
    assert "Superseded identifier appears" in checks["source_precedence"]["evidence_gaps"][0]


def test_reporter_acceptance_allows_superseded_id_in_labeled_history(
    tmp_path,
):
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Report written.",
            task_id="report-historical-reference",
            artifact_content=("# Report\n\nUses `POLICY-V3`.\n\n| Identifier | Status |\n| --- | --- |\n| `OLD-R7` | Superseded historical action |"),
            artifact_type="report_revision",
            artifact_metadata={
                "completion_status": "complete",
                "evidence_gaps": [],
            },
        )
    )
    state = {
        **_thread_state_with_outputs(tmp_path),
        "messages": [HumanMessage(content="保留标记 `POLICY-V3` 和历史编号 `OLD-R7`。")],
    }

    result = build_default_action_router(delegate_executor=executor).execute(
        _action(
            ActionType.DELEGATE,
            action_id="report-historical-reference",
            target_agent="reporter",
            task="Write the report.",
            stage="reporting",
        ),
        state=state,
        thread_id="thread-1",
        run_id="run-1",
    )

    ref = result.state_update["sp_current_artifact_refs"]["report_revision"]
    checks = ref["metadata"]["quality_checks"]
    assert ref["metadata"]["completion_status"] == "complete"
    assert checks["literal_token_provenance"]["passed"] is True
    assert checks["source_precedence"] == {
        "evidence_gaps": [],
        "passed": True,
    }


def test_reporter_acceptance_guard_rejects_local_filesystem_links(tmp_path):
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Report written.",
            task_id="report-local-link",
            artifact_content=("# Report\n\nSource: [evidence.md](/data/private/thread/uploads/evidence.md)"),
            artifact_type="report_revision",
            artifact_metadata={
                "completion_status": "complete",
                "evidence_gaps": [],
            },
        )
    )

    result = build_default_action_router(delegate_executor=executor).execute(
        _action(
            ActionType.DELEGATE,
            action_id="report-local-link",
            target_agent="reporter",
            task="Write the Markdown report.",
            stage="reporting",
        ),
        state=_thread_state_with_outputs(tmp_path),
        thread_id="thread-1",
        run_id="run-1",
    )

    ref = result.state_update["sp_current_artifact_refs"]["report_revision"]
    observation = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"]).entries[-1]
    portable_links = ref["metadata"]["quality_checks"]["portable_links"]
    assert ref["metadata"]["completion_status"] == "partial"
    assert portable_links["passed"] is False
    assert portable_links["unsafe_local_links"] == ["](/data/private/thread/uploads/evidence.md)"]
    assert "non-portable local filesystem links" in (observation.failure_note or "")


def test_reporter_acceptance_guard_rejects_claims_under_no_data_constraint(
    tmp_path,
):
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Report written.",
            task_id="report-no-data-fabrication",
            artifact_content=("# Report\n\n- The North Region outperformed the other regions.\n- The West Region had the highest churn rate.\n\n## Limitations\nActual values were not provided."),
            artifact_type="report_revision",
            artifact_metadata={
                "completion_status": "complete",
                "evidence_gaps": [],
            },
        )
    )
    state = {
        **_thread_state_with_outputs(tmp_path),
        "messages": [HumanMessage(content=("目前我还没有给你实际数值，也不允许联网或虚构数据。请生成报告。"))],
    }

    result = build_default_action_router(delegate_executor=executor).execute(
        _action(
            ActionType.DELEGATE,
            action_id="report-no-data-fabrication",
            target_agent="reporter",
            task="Write the Markdown report.",
            stage="reporting",
        ),
        state=state,
        thread_id="thread-1",
        run_id="run-1",
    )

    ref = result.state_update["sp_current_artifact_refs"]["report_revision"]
    observation = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"]).entries[-1]
    grounding = ref["metadata"]["quality_checks"]["no_data_grounding"]
    assert ref["metadata"]["completion_status"] == "partial"
    assert grounding["constraint_detected"] is True
    assert grounding["passed"] is False
    assert grounding["unsupported_claims"] == [
        "The North Region outperformed the other regions.",
        "The West Region had the highest churn rate.",
    ]
    assert "no actual data" in (observation.failure_note or "")


def test_reporter_acceptance_guard_allows_unknowns_under_no_data_constraint(
    tmp_path,
):
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Report written.",
            task_id="report-no-data-template",
            artifact_content=(
                "# Report\n\n"
                "| Region | Revenue | Churn |\n"
                "| --- | --- | --- |\n"
                "| North | TBD — data not provided | TBD — data not provided |\n"
                "\nWithout actual values, the highest and lowest regions "
                "cannot be determined. If data becomes available, rank the "
                "regions using the agreed definitions.\n"
                "- 建议：提供关于营收增长和毛利率改善的建议。"
            ),
            artifact_type="report_revision",
            artifact_metadata={
                "completion_status": "partial",
                "evidence_gaps": ["Actual regional values were not provided."],
            },
        )
    )
    state = {
        **_thread_state_with_outputs(tmp_path),
        "messages": [HumanMessage(content="No actual values were provided. Create a report.")],
    }

    result = build_default_action_router(delegate_executor=executor).execute(
        _action(
            ActionType.DELEGATE,
            action_id="report-no-data-template",
            target_agent="reporter",
            task="Write the Markdown report.",
            stage="reporting",
        ),
        state=state,
        thread_id="thread-1",
        run_id="run-1",
    )

    ref = result.state_update["sp_current_artifact_refs"]["report_revision"]
    grounding = ref["metadata"]["quality_checks"]["no_data_grounding"]
    assert ref["metadata"]["completion_status"] == "complete"
    assert grounding == {
        "constraint_detected": True,
        "unsupported_claims": [],
        "accepted_as_complete_with_disclosed_gaps": True,
        "passed": True,
    }


def test_reporter_acceptance_guard_rejects_english_report_for_chinese_request(
    tmp_path,
):
    english_report = (
        "# Project Report\n\n"
        "This report summarizes the supplied project evidence for the board. "
        "It explains the current operating position, the major risks, and the "
        "recommended next steps. The analysis is grounded in the supplied "
        "materials and separates observed facts from recommendations. "
        "The conclusion gives the board a clear decision path and identifies "
        "the owners responsible for follow-up actions."
    )
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Report written.",
            task_id="report-language-mismatch",
            artifact_content=english_report,
            artifact_type="report_revision",
            artifact_metadata={
                "completion_status": "complete",
                "evidence_gaps": [],
            },
        )
    )
    state = {
        **_thread_state_with_outputs(tmp_path),
        "messages": [HumanMessage(content="请根据已有材料生成中文董事会报告。")],
    }

    result = build_default_action_router(delegate_executor=executor).execute(
        _action(
            ActionType.DELEGATE,
            action_id="report-language-mismatch",
            target_agent="reporter",
            task="Write the board report.",
            stage="reporting",
        ),
        state=state,
        thread_id="thread-1",
        run_id="run-1",
    )

    ref = result.state_update["sp_current_artifact_refs"]["report_revision"]
    language = ref["metadata"]["quality_checks"]["language_alignment"]
    assert ref["metadata"]["completion_status"] == "partial"
    assert language["passed"] is False
    assert "predominantly non-Chinese" in language["evidence_gaps"][0]


def test_reporter_acceptance_guard_rejects_reversed_threshold_direction(
    tmp_path,
):
    report = """# Report

## Threshold Sensitivity Recommendations
- **Gross margin threshold (30.0%)**: The South region is 0.1% below the threshold.
- **Churn threshold (5.0%)**: The South region is 0.1% above the threshold.

| Metric | North | South | West |
| --- | ---: | ---: | ---: |
| Gross margin | 32.4% | 30.1% | 33.8% |
| Churn | 4.2% | 5.1% | 4.0% |

North is ready when gross margin ≥ 30.0% and churn ≤ 5.0%.
"""
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Report written.",
            task_id="report-threshold-direction",
            artifact_content=report,
            artifact_type="report_revision",
            artifact_metadata={
                "completion_status": "complete",
                "evidence_gaps": [],
            },
        )
    )

    result = build_default_action_router(delegate_executor=executor).execute(
        _action(
            ActionType.DELEGATE,
            action_id="report-threshold-direction",
            target_agent="reporter",
            task="Write the Markdown report.",
            stage="reporting",
        ),
        state=_thread_state_with_outputs(tmp_path),
        thread_id="thread-1",
        run_id="run-1",
    )

    ref = result.state_update["sp_current_artifact_refs"]["report_revision"]
    check = ref["metadata"]["quality_checks"]["threshold_consistency"]
    assert ref["metadata"]["completion_status"] == "partial"
    assert check["passed"] is False
    assert check["evidence_gaps"] == [("Threshold direction mismatch: South gross margin is 30.1%, which is above the 30% threshold.")]


def test_reporter_acceptance_guard_accepts_correct_threshold_direction(
    tmp_path,
):
    report = """# Report

## Threshold Sensitivity Recommendations
- **Gross margin threshold (30.0%)**: The South region is 0.1% above the threshold.
- **Churn threshold (5.0%)**: The South region is 0.1% above the threshold.
- North and West are ready, while South churn is above the 5.0% threshold.

| Metric | North | South | West |
| --- | ---: | ---: | ---: |
| Gross margin | 32.4% | 30.1% | 33.8% |
| Churn | 4.2% | 5.1% | 4.0% |

North is ready when gross margin ≥ 30.0% and churn ≤ 5.0%.
"""
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Report written.",
            task_id="report-threshold-correct",
            artifact_content=report,
            artifact_type="report_revision",
            artifact_metadata={
                "completion_status": "complete",
                "evidence_gaps": [],
            },
        )
    )

    result = build_default_action_router(delegate_executor=executor).execute(
        _action(
            ActionType.DELEGATE,
            action_id="report-threshold-correct",
            target_agent="reporter",
            task="Write the Markdown report.",
            stage="reporting",
        ),
        state=_thread_state_with_outputs(tmp_path),
        thread_id="thread-1",
        run_id="run-1",
    )

    ref = result.state_update["sp_current_artifact_refs"]["report_revision"]
    check = ref["metadata"]["quality_checks"]["threshold_consistency"]
    assert ref["metadata"]["completion_status"] == "complete"
    assert check == {"evidence_gaps": [], "passed": True}


def test_reporter_requires_explicit_user_supplied_boundary_adjustment(tmp_path):
    report = """# Project Aurora 董事会报告

南区毛利率为 30.1%，已经达到 30.0% 的门槛。南区流失率为 5.1%，
高于 5.0% 的上限，因此南区尚未准备就绪。
"""
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Report corrected.",
            task_id="report-boundary-adjustment",
            artifact_content=report,
            artifact_type="report_revision",
            artifact_metadata={
                "completion_status": "complete",
                "evidence_gaps": [],
            },
        )
    )
    state = {
        **_thread_state_with_outputs(tmp_path),
        "messages": [HumanMessage(content=("<uploaded_files>inventory only</uploaded_files>\n请修订刚才的报告。最小边界改进只是把流失率降低0.1个百分点到5.0%。"))],
        "sp_task_memory": TaskMemoryStack(
            entries=[
                StackMemoryEntry(
                    action="user_request",
                    actor="human",
                    content=("<uploaded_files>inventory only</uploaded_files>\n请修订刚才的报告。最小边界改进只是把流失率降低0.1个百分点到5.0%。"),
                    run_id="run-2",
                )
            ]
        ).to_dict(),
    }

    result = build_default_action_router(delegate_executor=executor).execute(
        _action(
            ActionType.DELEGATE,
            action_id="report-boundary-adjustment",
            target_agent="reporter",
            task="Correct the South readiness statement.",
            stage="revision",
            metadata={"revision_reason": "Apply the user's factual correction."},
        ),
        state=state,
        thread_id="thread-1",
        run_id="run-2",
    )

    requirements = executor.tasks[0].context_refs["report_requirements"]
    assert requirements["human_feedback"] == [
        {
            "entry_id": next(entry["id"] for entry in state["sp_task_memory"]["entries"] if entry["action"] == "user_request"),
            "content": ("请修订刚才的报告。最小边界改进只是把流失率降低0.1个百分点到5.0%。"),
            "run_id": "run-2",
        }
    ]
    assert requirements["required_boundary_adjustments"] == [
        {
            "metric": "流失率",
            "direction": "down",
            "delta_percentage_points": "0.1",
            "target_percent": "5.0",
        }
    ]
    ref = result.state_update["sp_current_artifact_refs"]["report_revision"]
    check = ref["metadata"]["quality_checks"]["required_boundary_adjustments"]
    assert ref["metadata"]["completion_status"] == "partial"
    assert check["passed"] is False
    assert "0.1 percentage points to 5.0%" in check["evidence_gaps"][0]


def test_reporter_accepts_explicit_user_supplied_boundary_adjustment(tmp_path):
    report = """# Project Aurora 董事会报告

南区毛利率为 30.1%，已经达到 30.0% 的门槛。南区流失率为 5.1%，
高于 5.0% 的上限；最小边界改进是将流失率降低 0.1 个百分点至 5.0%。
"""
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Report corrected.",
            task_id="report-boundary-adjustment-complete",
            artifact_content=report,
            artifact_type="report_revision",
            artifact_metadata={
                "completion_status": "complete",
                "evidence_gaps": [],
            },
        )
    )
    state = {
        **_thread_state_with_outputs(tmp_path),
        "messages": [HumanMessage(content="最小边界改进只是把流失率降低0.1个百分点到5.0%。")],
    }

    result = build_default_action_router(delegate_executor=executor).execute(
        _action(
            ActionType.DELEGATE,
            action_id="report-boundary-adjustment-complete",
            target_agent="reporter",
            task="Correct the report.",
            stage="revision",
            metadata={"revision_reason": "Apply the user's factual correction."},
        ),
        state=state,
        thread_id="thread-1",
        run_id="run-2",
    )

    ref = result.state_update["sp_current_artifact_refs"]["report_revision"]
    assert ref["metadata"]["completion_status"] == "complete"
    assert ref["metadata"]["quality_checks"]["required_boundary_adjustments"]["passed"] is True


def test_reporter_revision_discards_prior_created_path_and_persists_fallback_body(
    tmp_path,
):
    outputs = tmp_path / "threads" / "thread-1" / "user-data" / "outputs"
    outputs.mkdir(parents=True)
    old_path = outputs / "report-spact_old.md"
    old_path.write_text("# Old report\n\nMissing correction.", encoding="utf-8")
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Revised report written.",
            task_id="report-revision-fallback",
            artifact_content="# Revised report\n\nIncludes `REVIEWED-HIGH`.",
            artifact_type="report_revision",
            artifact_metadata={
                "completion_status": "complete",
                "evidence_gaps": [],
                "created_paths": ["/mnt/user-data/outputs/report-spact_old.md"],
            },
        )
    )

    result = build_default_action_router(delegate_executor=executor).execute(
        _action(
            ActionType.DELEGATE,
            action_id="spact_current",
            target_agent="reporter",
            task="Revise the Markdown report.",
            stage="revision",
            metadata={"revision_reason": "Retain the review label."},
        ),
        state=_thread_state_with_outputs(tmp_path),
        thread_id="thread-1",
        run_id="run-2",
    )

    ref = result.state_update["sp_current_artifact_refs"]["report_revision"]
    ownership = ref["metadata"]["quality_checks"]["created_path_ownership"]
    generated_path = result.state_update["artifacts"][0]
    generated_file = outputs / generated_path.removeprefix("/mnt/user-data/outputs/")
    assert ref["metadata"]["completion_status"] == "complete"
    assert ownership == {
        "discarded_paths": ["/mnt/user-data/outputs/report-spact_old.md"],
        "fallback_artifact_content": True,
        "passed": True,
    }
    assert generated_path != "/mnt/user-data/outputs/report-spact_old.md"
    assert generated_file.read_text(encoding="utf-8") == ("# Revised report\n\nIncludes `REVIEWED-HIGH`.")
    assert old_path.read_text(encoding="utf-8") == ("# Old report\n\nMissing correction.")


def test_reporter_does_not_create_unrequested_revision_when_current_report_exists():
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="should not run",
            task_id="unexpected",
            artifact_content="unexpected revision",
            artifact_type="report_revision",
        )
    )
    action = _action(
        ActionType.DELEGATE,
        action_id="act-repeat-report",
        target_agent="reporter",
        task="Generate the report again",
    )

    result = build_default_action_router(delegate_executor=executor).execute(
        action,
        state={"sp_current_artifact_refs": {"report_revision": {"artifact_id": "report-v1", "is_current": True}}},
    )

    assert result.next_step == "error_recoverable"
    assert "revision_reason" in (result.error or "")
    assert executor.tasks == []
    assert any(event["event_type"] == "sp.delegate.duplicate_skipped" for event in result.run_events)


def test_repeated_reporter_actions_keep_one_report_version(tmp_path):
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="initial report",
            task_id="report-task",
            artifact_content="# Report v1",
            artifact_type="report_revision",
        )
    )
    router = build_default_action_router(delegate_executor=executor)
    first = _action(
        ActionType.DELEGATE,
        action_id="report-first",
        target_agent="reporter",
        task="Create the report",
    )
    first_result = router.execute(
        first,
        state={"thread_data": {"outputs_path": str(tmp_path)}},
        thread_id="thread-report",
        run_id="run-report",
    )
    second = _action(
        ActionType.DELEGATE,
        action_id="report-second",
        target_agent="reporter",
        task="Create the report again",
    )

    second_result = router.execute(
        second,
        state=first_result.state_update,
        thread_id="thread-report",
        run_id="run-report",
    )

    assert second_result.next_step == "error_recoverable"
    assert "revision_reason" in (second_result.error or "")
    assert len(executor.tasks) == 1
    refs = first_result.state_update["sp_current_artifact_refs"]
    assert refs["report_revision"]["version"] == 1


def test_delegate_context_is_bounded_and_preserves_user_goal_and_pinned_feedback():
    stack = TaskMemoryStack(max_size=50)
    for index in range(30):
        stack.append_think(f"old-control-entry-{index}", metadata={"large": "x" * 1000})
    feedback = stack.append_feedback("Pinned report constraint")
    state = {
        "sp_task_memory": stack.to_dict(),
        "messages": [
            HumanMessage(content="Original report request"),
            HumanMessage(content="Internal response", additional_kwargs={"hide_from_ui": True}),
            HumanMessage(content="Latest visible refinement"),
        ],
        "sp_current_artifact_refs": {
            "report_revision": {"artifact_id": "current", "type": "report_revision"},
            "_history": [{"artifact_id": f"report-{index}"} for index in range(20)],
        },
    }
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Reporter accepted the bounded task context.",
            task_id="task-context",
        )
    )
    action = _action(
        ActionType.DELEGATE,
        action_id="act-context",
        target_agent="reporter",
        task="Revise the report",
        metadata={"revision_reason": "Pinned human feedback requires a revision."},
    )

    result = build_default_action_router(delegate_executor=executor).execute(action, state=state)

    assert result.next_step == "continue"
    refs = executor.tasks[0].context_refs
    assert refs["original_query"] == "Latest visible refinement"
    assert refs["latest_user_input"] == "Latest visible refinement"
    assert refs["thread_first_user_input"] == "Original report request"
    selected = refs["task_memory"]["entries"]
    assert len(selected) == 24
    assert selected[0]["id"] == feedback.id
    assert selected[0]["status"] == "pinned"
    assert all("metadata" not in entry for entry in selected)
    assert len(refs["artifact_refs"]["_history"]) == 12


def test_delegate_context_bounds_each_memory_entry_body():
    stack = TaskMemoryStack()
    observation = stack.append_observe(
        "x" * 1800,
        actor="researcher",
        failure_note="f" * 800,
        run_id="run-1",
    )
    state = {
        "sp_task_memory": stack.to_dict(),
        "messages": [HumanMessage(content="Use the result")],
    }
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Done",
            task_id="task-bounded-entry",
        )
    )

    build_default_action_router(delegate_executor=executor).execute(
        _action(
            ActionType.DELEGATE,
            action_id="act-bounded-entry",
            target_agent="reporter",
            task="Synthesize the result",
        ),
        state=state,
        run_id="run-1",
    )

    selected = next(item for item in executor.tasks[0].context_refs["task_memory"]["entries"] if item["id"] == observation.id)
    assert len(selected["content"]) <= 900
    assert selected["content"].endswith("...<truncated>")
    assert len(selected["failure_note"]) <= 240


def test_delegate_context_preserves_full_mandatory_human_requirements():
    stack = TaskMemoryStack()
    feedback_text = "Keep the approved scope and do not remove the comparison table. " + ("F" * 1500)
    feedback = stack.append_feedback(feedback_text, run_id="run-1")
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Requirements received.",
            task_id="requirements-task",
        )
    )

    build_default_action_router(delegate_executor=executor).execute(
        _action(
            ActionType.DELEGATE,
            action_id="requirements-delegate",
            target_agent="researcher",
            task="Research the approved scope",
        ),
        state={
            "sp_task_memory": stack.to_dict(),
            "messages": [HumanMessage(content="Prepare the comparison")],
        },
        run_id="run-1",
    )

    requirements = executor.tasks[0].context_refs["mandatory_requirements"]
    assert requirements["human_feedback"][0]["entry_id"] == feedback.id
    assert requirements["human_feedback"][0]["content"] == feedback_text
    assert len(requirements["human_feedback"][0]["content"]) > 900


@pytest.mark.parametrize(
    ("target_agent", "source_type", "source_body"),
    [
        ("outline", "perception_observation", "Approved task brief with audience and scope."),
        ("researcher", "outline", "Dependency-aware outline with evidence requirements."),
    ],
)
def test_outline_and_researcher_receive_materialized_stage_artifacts(
    tmp_path,
    target_agent,
    source_type,
    source_body,
):
    from deerflow.sp.artifacts import SPArtifactAdapter

    adapter = SPArtifactAdapter()
    state = _thread_state_with_outputs(tmp_path)
    source = adapter.write_text_artifact(
        source_body,
        artifact_type=source_type,
        state=state,
        thread_id="thread-1",
        run_id="run-1",
    )
    state = {**state, **source.state_update}
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Stage completed.",
            task_id=f"{target_agent}-task",
        )
    )

    build_default_action_router(delegate_executor=executor).execute(
        _action(
            ActionType.DELEGATE,
            action_id=f"delegate-{target_agent}-with-source",
            target_agent=target_agent,
            task=f"Run {target_agent} from the supplied source",
            input_refs=[source.metadata.artifact_id],
        ),
        state=state,
        thread_id="thread-1",
        run_id="run-1",
    )

    bundle = executor.tasks[0].context_refs["artifact_bodies"]
    assert bundle["items"][0]["artifact_id"] == source.metadata.artifact_id
    assert bundle["items"][0]["content"] == source_body


def test_reporter_receives_materialized_evidence_and_revision_lineage(tmp_path):
    from deerflow.sp.artifacts import SPArtifactAdapter

    adapter = SPArtifactAdapter()
    state = _thread_state_with_outputs(tmp_path)

    outline = adapter.write_text_artifact(
        "# Outline\n\n1. Conclusion\n2. Evidence",
        artifact_type="outline",
        state=state,
        thread_id="thread-1",
        run_id="run-1",
    )
    state = {**state, **outline.state_update}
    research_one = adapter.write_text_artifact(
        "Research body one with source https://example.com/one",
        artifact_type="research_observation",
        state=state,
        thread_id="thread-1",
        run_id="run-1",
    )
    state = {**state, **research_one.state_update}
    research_two = adapter.write_text_artifact(
        "Research body two with source https://example.com/two",
        artifact_type="research_observation",
        state=state,
        thread_id="thread-1",
        run_id="run-1",
    )
    state = {**state, **research_two.state_update}
    report_v1 = adapter.write_text_artifact(
        "# Report v1\n\nPreserve this unaffected section.",
        artifact_type="report_revision",
        state=state,
        thread_id="thread-1",
        run_id="run-1",
    )
    state = {**state, **report_v1.state_update}
    stack = TaskMemoryStack()
    feedback = stack.append_feedback(
        "Make the conclusion more direct but preserve the evidence table.",
        run_id="run-1",
    )
    state["sp_task_memory"] = stack.to_dict()
    state["messages"] = [HumanMessage(content="Revise my analytical report")]

    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Revised report completed.",
            task_id="report-v2-task",
            artifact_content="# Report v2\n\nDirect conclusion with preserved evidence.",
            artifact_type="report_revision",
        )
    )
    action = _action(
        ActionType.DELEGATE,
        action_id="report-v2",
        target_agent="reporter",
        task="Apply the pinned feedback and produce the complete revised report",
        input_refs=[report_v1.metadata.artifact_id],
        expected_output="A complete Markdown report preserving evidence and citations",
        stage="revision",
        metadata={"revision_reason": "Pinned human feedback requires a revision."},
    )

    result = build_default_action_router(delegate_executor=executor).execute(
        action,
        state=state,
        thread_id="thread-1",
        run_id="run-1",
    )

    bundle = executor.tasks[0].context_refs["artifact_bodies"]
    bodies = {item["artifact_id"]: item["content"] for item in bundle["items"]}
    assert outline.metadata.artifact_id in bodies
    assert research_one.metadata.artifact_id in bodies
    assert research_two.metadata.artifact_id in bodies
    assert report_v1.metadata.artifact_id in bodies
    assert "https://example.com/one" in bodies[research_one.metadata.artifact_id]
    assert executor.tasks[0].context_refs["report_requirements"]["human_feedback"][0]["content"] == feedback.content

    current = result.state_update["sp_current_artifact_refs"]["report_revision"]
    assert current["version"] == 2
    assert current["parent_artifact_ids"] == [report_v1.metadata.artifact_id]
    assert current["feedback_entry_ids"] == [feedback.id]
    assert set(current["metadata"]["source_artifact_ids"]) >= {
        outline.metadata.artifact_id,
        research_one.metadata.artifact_id,
        research_two.metadata.artifact_id,
        report_v1.metadata.artifact_id,
    }


def test_delegate_context_excludes_unrelated_prior_run_noise_and_artifacts():
    stack = TaskMemoryStack()
    stack.append_observe("Old unrelated research", actor="researcher", run_id="run-old")
    prior_request = stack.append(
        StackMemoryEntry(
            action="user_request",
            actor="human",
            content="Keep the event alcohol-free",
            run_id="run-old",
            priority="high",
        )
    )
    carryover = stack.append_summary("Reusable verified project decision", run_id="run-old")
    current = stack.append_think("Current task plan", run_id="run-new")
    state = {
        "sp_task_memory": stack.to_dict(),
        "messages": [HumanMessage(content="Do the new task")],
        "sp_current_artifact_refs": {
            "old_report": {
                "artifact_id": "old-report",
                "run_id": "run-old",
                "virtual_path": "/mnt/user-data/outputs/old.md",
            },
            "current_input": {
                "artifact_id": "current-input",
                "run_id": "run-new",
                "virtual_path": "/mnt/user-data/outputs/current.csv",
            },
            "_history": [
                {"artifact_id": "old-history", "run_id": "run-old"},
                {"artifact_id": "current-history", "run_id": "run-new"},
            ],
        },
    }
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Current task completed.",
            task_id="task-current",
        )
    )

    build_default_action_router(delegate_executor=executor).execute(
        _action(
            ActionType.DELEGATE,
            action_id="act-current-context",
            target_agent="coder",
            task="Process the current task",
        ),
        state=state,
        thread_id="thread-1",
        run_id="run-new",
    )

    refs = executor.tasks[0].context_refs
    selected_ids = {entry["id"] for entry in refs["task_memory"]["entries"]}
    assert prior_request.id in selected_ids
    assert carryover.id in selected_ids
    assert current.id in selected_ids
    assert all(entry["content"] != "Old unrelated research" for entry in refs["task_memory"]["entries"])
    assert "old_report" not in refs["artifact_refs"]
    assert refs["artifact_refs"]["current_input"]["artifact_id"] == "current-input"
    assert refs["artifact_refs"]["_history"] == [{"artifact_id": "current-history", "run_id": "run-new"}]


def test_delegate_handler_records_failure_as_recoverable_error():
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.FAILED,
            error="research failed",
            task_id="task-failed",
        )
    )
    action = _action(ActionType.DELEGATE, action_id="act-delegate-fail", target_agent="researcher", task="Research")

    result = build_default_action_router(delegate_executor=executor).execute(action, state={})

    assert result.next_step == "error_recoverable"
    assert result.error == "research failed"
    restored = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    assert [entry.action for entry in restored.entries[-2:]] == ["delegate", "error"]
    assert restored.entries[-1].result_ref == "task-failed"
    assert any(event["event_type"] == "sp.delegate.failed" for event in result.run_events)


def test_delegate_handler_defensively_externalizes_large_unstructured_result(tmp_path):
    large_result = "Detailed evidence. " * 200
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result=large_result,
            task_id="task-large-unstructured",
        )
    )
    action = _action(
        ActionType.DELEGATE,
        action_id="act-large-unstructured",
        target_agent="researcher",
        task="Research details",
        stage="research",
    )

    result = build_default_action_router(delegate_executor=executor).execute(
        action,
        state=_thread_state_with_outputs(tmp_path),
        thread_id="thread-1",
        run_id="run-1",
    )

    stack = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    observation = stack.entries[-1]
    artifact_ref = result.state_update["sp_current_artifact_refs"]["research_observation"]
    artifact_path = tmp_path / "threads" / "thread-1" / "user-data" / "outputs" / artifact_ref["virtual_path"].removeprefix("/mnt/user-data/outputs/")

    assert result.next_step == "continue"
    assert len(observation.content) == 700
    assert observation.content.endswith("...<truncated>")
    assert observation.result_ref == artifact_ref["artifact_id"]
    assert artifact_path.read_text(encoding="utf-8") == large_result
    assert large_result not in str(result.state_update["sp_task_memory"])


def test_delegate_handler_registers_coder_created_output_paths_as_artifacts(tmp_path):
    outputs_path = tmp_path / "threads" / "thread-1" / "user-data" / "outputs" / "generated"
    outputs_path.mkdir(parents=True)
    (outputs_path / "module.py").write_text("value = 1\n", encoding="utf-8")
    (outputs_path / "test_module.py").write_text("def test_value(): ...\n", encoding="utf-8")
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Implemented and tested the requested module.",
            task_id="task-coder-output",
            artifact_type="generated_file",
            artifact_metadata={
                "created_paths": [
                    "/mnt/user-data/outputs/generated/module.py",
                    "/mnt/user-data/outputs/generated/test_module.py",
                ]
            },
        )
    )
    action = _action(
        ActionType.DELEGATE,
        action_id="act-coder-output",
        target_agent="coder",
        task="Implement module",
        stage="implementation",
    )

    result = build_default_action_router(delegate_executor=executor).execute(
        action,
        state=_thread_state_with_outputs(tmp_path),
        thread_id="thread-1",
        run_id="run-1",
    )

    stack = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    history = result.state_update["sp_current_artifact_refs"]["_history"]
    assert result.state_update["artifacts"] == [
        "/mnt/user-data/outputs/generated/module.py",
        "/mnt/user-data/outputs/generated/test_module.py",
    ]
    assert [ref["version"] for ref in history] == [1, 2]
    assert stack.entries[-1].result_ref == history[-1]["artifact_id"]
    assert sum(event["event_type"] == "sp.artifact.registered" for event in result.run_events) == 2
    assert sum(event["event_type"] == "sp.artifact.current_changed" for event in result.run_events) == 2


def test_coder_text_from_perception_stage_is_an_observation_not_deliverable(
    tmp_path,
):
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Inspected the policy file.",
            task_id="task-coder-inspection",
            artifact_content="# Policy observations\n\nNo generated deliverable.",
            artifact_type="generated_file",
            artifact_metadata={"created_paths": ["/mnt/user-data/outputs/claimed-but-missing.md"]},
        )
    )
    result = build_default_action_router(delegate_executor=executor).execute(
        _action(
            ActionType.DELEGATE,
            action_id="act-coder-inspection",
            target_agent="coder",
            task="Inspect the policy file.",
            stage="perception",
        ),
        state=_thread_state_with_outputs(tmp_path),
        thread_id="thread-1",
        run_id="run-1",
    )

    refs = result.state_update["sp_current_artifact_refs"]
    assert "generated_file" not in refs
    assert refs["perception_observation"]["type"] == "perception_observation"
    assert result.state_update["artifacts"][0].endswith(".md")
    assert any(event["event_type"] == "sp.artifact.type_normalized" for event in result.run_events)


@pytest.mark.parametrize(
    ("filename", "original_content"),
    [
        ("module.py", b"def answer():\n    return 42\n"),
        ("icon.png", b"\x89PNG\r\n\x1a\n\x00binary-image-data"),
    ],
)
def test_delegate_created_file_is_never_overwritten_by_artifact_content(tmp_path, filename, original_content):
    outputs_path = tmp_path / "threads" / "thread-1" / "user-data" / "outputs"
    outputs_path.mkdir(parents=True)
    generated_file = outputs_path / filename
    generated_file.write_bytes(original_content)
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Generated and verified the requested file.",
            task_id="task-existing-output",
            artifact_content="# Verification summary\n\nThis text is not the generated file.",
            artifact_type="generated_file",
            artifact_metadata={"created_paths": [f"/mnt/user-data/outputs/{filename}"]},
        )
    )
    action = _action(
        ActionType.DELEGATE,
        action_id=f"act-existing-{filename}",
        target_agent="coder",
        task="Generate and verify a file",
        stage="implementation",
    )

    result = build_default_action_router(delegate_executor=executor).execute(
        action,
        state=_thread_state_with_outputs(tmp_path),
        thread_id="thread-1",
        run_id="run-1",
    )

    assert generated_file.read_bytes() == original_content
    assert result.state_update["artifacts"] == [f"/mnt/user-data/outputs/{filename}"]
    assert sum(event["event_type"] == "sp.artifact.registered" for event in result.run_events) == 1
    assert not any(event["event_type"] == "sp.artifact.created" for event in result.run_events)


def test_delegate_invalid_created_path_falls_back_to_text_artifact(tmp_path):
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Created a fallback artifact.",
            task_id="task-missing-output",
            artifact_content="fallback body",
            artifact_type="generated_file",
            artifact_metadata={"created_paths": ["/mnt/user-data/outputs/missing.py"]},
        )
    )
    action = _action(
        ActionType.DELEGATE,
        action_id="act-missing-output",
        target_agent="coder",
        task="Generate a file",
        stage="implementation",
    )

    result = build_default_action_router(delegate_executor=executor).execute(
        action,
        state=_thread_state_with_outputs(tmp_path),
        thread_id="thread-1",
        run_id="run-1",
    )

    artifact_path = result.state_update["artifacts"][0]
    assert artifact_path.startswith("/mnt/user-data/outputs/sp/generated_file/")
    actual_path = tmp_path / "threads" / "thread-1" / "user-data" / "outputs" / artifact_path.removeprefix("/mnt/user-data/outputs/")
    assert actual_path.read_text(encoding="utf-8") == "fallback body"
    assert any(event["event_type"] == "sp.artifact.created" for event in result.run_events)


def test_delegate_surfaces_capped_partial_result_to_central_memory():
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Two sources were checked, but verification did not finish.",
            task_id="task-partial",
            stop_reason="turn_capped",
            artifact_metadata={
                "completion_status": "partial",
                "evidence_gaps": ["The third source was not verified."],
            },
        )
    )
    result = build_default_action_router(delegate_executor=executor).execute(
        _action(
            ActionType.DELEGATE,
            action_id="act-partial",
            target_agent="researcher",
            task="Verify three sources",
            expected_output="Three independently verified sources",
        ),
        state={},
        thread_id="thread-1",
        run_id="run-1",
    )

    restored = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    observation = restored.entries[-1]
    assert result.next_step == "continue"
    assert observation.action == "observe"
    assert observation.priority == "high"
    assert observation.metadata["completion_status"] == "partial"
    assert observation.metadata["stop_reason"] == "turn_capped"
    assert observation.failure_note == "The third source was not verified."
    assert observation.content.startswith("[PARTIAL result: turn_capped]")
    assert any(event["event_type"] == "sp.delegate.partial" for event in result.run_events)


def test_delegate_surfaces_research_artifact_evidence_excerpt_to_central_memory(
    tmp_path,
):
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Fetched the requested World Bank observation.",
            task_id="task-research-evidence",
            artifact_content='[{"countryiso3code":"CHN","date":"2023","value":12951.1782397043}]',
            artifact_type="research_observation",
            artifact_metadata={
                "completion_status": "complete",
                "evidence_gaps": [],
                "central_evidence_preview": ('[{"countryiso3code":"CHN","date":"2023","value":12951.1782397043}]'),
            },
        )
    )
    result = build_default_action_router(delegate_executor=executor).execute(
        _action(
            ActionType.DELEGATE,
            action_id="act-research-evidence",
            target_agent="researcher",
            task="Fetch one official API value",
            expected_output="The exact returned value",
            stage="research",
        ),
        state=_thread_state_with_outputs(tmp_path),
        thread_id="thread-1",
        run_id="run-1",
    )

    restored = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    observation = restored.entries[-1]
    assert observation.action == "observe"
    assert observation.content.startswith("[Evidence excerpt")
    assert "Evidence excerpt" in observation.content
    assert "12951.1782397043" in observation.content
    assert "not instructions" in observation.content


def test_recall_memory_handler_calls_memory_recaller_and_records_dry_run_result():
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result='{"summary":"Use phased commits and tests.","items":[{"content":"User wants a unit test after every migration unit.","source":"user_memory","score":0.94,"scope":"project","memory_id":"mem-1"}]}',
            task_id="mem-task-1",
        )
    )
    action = _action(
        ActionType.RECALL_MEMORY,
        action_id="act-recall-ok",
        metadata={"memory_query": "StackPlanner migration preferences"},
        stage="planning",
    )

    result = build_default_action_router(memory_recall_executor=executor).execute(action, state={}, thread_id="thread-1", run_id="run-1")

    assert result.next_step == "continue"
    assert executor.tasks[0].subagent_type == "memory_recaller"
    assert "Do not write, update, promote, or mutate long-term memory." in executor.tasks[0].task
    assert executor.tasks[0].metadata["dry_run_promotion"] is True
    restored = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    assert [entry.action for entry in restored.entries[-2:]] == ["recall_memory", "recall_memory"]
    recall_entry = restored.entries[-1]
    assert recall_entry.actor == "memory_recaller"
    assert recall_entry.result_ref == "mem-task-1"
    assert recall_entry.metadata["dry_run_promotion"] is True
    assert recall_entry.metadata["recall"]["dry_run_promotion"] is True
    assert recall_entry.metadata["recall"]["items"][0]["memory_id"] == "mem-1"
    assert "User wants a unit test" in recall_entry.content
    assert any(event["event_type"] == "sp.memory.recall.completed" for event in result.run_events)


def test_recall_memory_handler_records_failure_as_recoverable_error():
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.FAILED,
            error="memory unavailable",
            task_id="mem-task-failed",
        )
    )
    action = _action(
        ActionType.RECALL_MEMORY,
        action_id="act-recall-fail",
        metadata={"memory_query": "StackPlanner migration preferences"},
    )

    result = build_default_action_router(memory_recall_executor=executor).execute(action, state={})

    assert result.next_step == "error_recoverable"
    assert result.error == "memory unavailable"
    restored = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    assert [entry.action for entry in restored.entries[-2:]] == ["recall_memory", "error"]
    assert restored.entries[-1].actor == "memory_recaller"
    assert restored.entries[-1].result_ref == "mem-task-failed"
    assert any(event["event_type"] == "sp.memory.recall.failed" for event in result.run_events)


def test_router_stops_when_loop_limit_is_exceeded():
    action = _action(ActionType.THINK, action_id="act-loop", task="keep thinking")

    result = build_default_action_router().execute(
        action,
        state={"sp_loop_iteration": 2, "sp_max_loop_iterations": 2},
    )

    assert result.next_step == "error_fatal"
    assert "exceeded max iterations" in result.error


def test_router_resets_loop_budget_when_dr2_run_id_changes():
    action = _action(ActionType.THINK, action_id="act-new-run", task="Continue in a new run")

    result = build_default_action_router().execute(
        action,
        state={
            "sp_loop_iteration": 20,
            "sp_loop_run_id": "old-run",
            "sp_max_loop_iterations": 20,
        },
        run_id="new-run",
    )

    assert result.next_step == "continue"
    assert result.state_update["sp_loop_iteration"] == 1
    assert result.state_update["sp_loop_run_id"] == "new-run"


def test_central_prompt_exposes_sp_controls_without_forcing_json_actions():
    prompt = CENTRAL_AGENT_ACTION_PROMPT

    assert "normal agent loop" in prompt
    assert "sp_delegate" in prompt
    assert "sp_reflect" in prompt
    assert "sp_revise" in prompt
    assert "Do not wrap" in prompt
    assert "Your only callable methods are the predefined `sp_*` actions" in prompt
    assert "belong exclusively to delegated subagents" in prompt
    assert "implicit THINK" in prompt
    assert "You are the StackPlanner 2.0 CentralAgent" in prompt
    assert "DeerFlow is not your identity" in prompt
    assert "Never introduce yourself as DeerFlow" in prompt
    assert "Direct:" in prompt
    assert "Bounded execution:" in prompt
    assert "Deliberate:" in prompt
    assert "audit-ready decision note" in prompt
    assert "observable result or stop condition" in prompt
    assert "Current-data" in prompt
    assert "read-only request to query or fetch an external HTTP/API endpoint" in prompt
    assert "call `web_fetch` on the official API before trying" in prompt
    assert 'required_artifact_type="report"' in prompt
