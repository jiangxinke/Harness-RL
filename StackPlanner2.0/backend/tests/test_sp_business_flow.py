"""Business-flow regression tests for StackPlanner-on-DeerFlow orchestration."""

from __future__ import annotations

from typing import Any

from deerflow.sp import ActionLoop, ActionType, CentralDecisionRequest, SPAction, build_default_action_router, record_human_feedback
from deerflow.sp.memory import TaskMemoryStack
from deerflow.sp.subagents import SPSubagentResult, SPSubagentStatus, SPSubagentTask


class ScriptedDecider:
    def __init__(self, actions: list[SPAction]):
        self.actions = actions
        self.requests: list[CentralDecisionRequest] = []

    def decide(self, request: CentralDecisionRequest) -> SPAction:
        self.requests.append(request)
        return self.actions[len(self.requests) - 1]


class FakeExecutor:
    def __init__(self, result: SPSubagentResult):
        self.result = result
        self.tasks: list[SPSubagentTask] = []

    def execute(self, task: SPSubagentTask) -> SPSubagentResult:
        self.tasks.append(task)
        return self.result


def _action(action_type: ActionType | str, **kwargs: Any) -> SPAction:
    return SPAction.create(action_type, reason=kwargs.pop("reason", "business-flow"), **kwargs)


def _thread_state_with_outputs(tmp_path):
    return {
        "thread_data": {
            "outputs_path": str(tmp_path / "threads" / "thread-1" / "user-data" / "outputs"),
        }
    }


def test_sp_business_flow_runs_recall_delegate_hitl_resume_finish(tmp_path):
    memory_executor = FakeExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result='{"summary":"User prefers phased migrations with tests.","items":[{"content":"Run unit tests after each migration unit.","source":"user_memory"}]}',
            task_id="memory-task-1",
        )
    )
    delegate_executor = FakeExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Report artifact created",
            task_id="delegate-task-1",
            artifact_content="# Report\n\nStackPlanner migration summary.",
            artifact_type="report_revision",
        )
    )
    router = build_default_action_router(
        delegate_executor=delegate_executor,
        memory_recall_executor=memory_executor,
    )
    first_loop = ActionLoop(
        decider=ScriptedDecider(
            [
                _action(ActionType.THINK, action_id="act-think", task="Plan migration", stage="planning"),
                _action(ActionType.RECALL_MEMORY, action_id="act-recall", metadata={"memory_query": "migration preferences"}, stage="planning"),
                _action(ActionType.DELEGATE, action_id="act-delegate", target_agent="reporter", task="Draft migration report", stage="reporting"),
                _action(ActionType.ASK_HUMAN, action_id="act-human", task="Please confirm the report", metadata={"interaction_type": "report_feedback"}, stage="reporting"),
            ]
        ),
        router=router,
        max_iterations=8,
    )

    interrupted = first_loop.run(_thread_state_with_outputs(tmp_path), thread_id="thread-1", run_id="run-1")

    assert interrupted.next_step == "interrupt"
    assert memory_executor.tasks[0].subagent_type == "memory_recaller"
    assert delegate_executor.tasks[0].subagent_type == "reporter"
    interrupted_stack = TaskMemoryStack.from_dict(interrupted.state["sp_task_memory"])
    assert [entry.action for entry in interrupted_stack.entries] == [
        "think",
        "recall_memory",
        "recall_memory",
        "delegate",
        "observe",
        "ask_human",
    ]
    assert interrupted.state["sp_pending_human_interaction"]["status"] == "pending"
    assert interrupted.state["sp_current_artifact_refs"]["report_revision"]["artifact_id"] == interrupted_stack.entries[4].result_ref

    feedback_result = record_human_feedback(
        interrupted.state,
        "以后报告先确认结论，再补证据",
        thread_id="thread-1",
        run_id="run-2",
    )
    resumed_state = {**interrupted.state, **feedback_result.state_update}
    finish_loop = ActionLoop(
        decider=ScriptedDecider([_action(ActionType.FINISH, action_id="act-finish", task="Done")]),
        router=router,
        max_iterations=2,
    )

    finished = finish_loop.run(resumed_state, thread_id="thread-1", run_id="run-2")

    assert finished.next_step == "finish"
    finished_stack = TaskMemoryStack.from_dict(finished.state["sp_task_memory"])
    assert [entry.action for entry in finished_stack.entries[-2:]] == ["feedback", "finish"]
    assert finished_stack.entries[-2].priority == "critical"
    assert finished_stack.entries[-2].status == "pinned"
    assert finished.state["sp_pending_human_interaction"] is None
    assert finished.state["sp_current_stage"] == "finished"
    assert finished.state["sp_current_artifact_refs"]["report_revision"]["feedback_entry_ids"] == [finished_stack.entries[-2].id]
