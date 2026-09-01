"""Tests for the SP CentralAgent action loop."""

from __future__ import annotations

from typing import Any

from deerflow.sp import ActionLoop, ActionType, CentralDecisionRequest, SPAction
from deerflow.sp.memory import TaskMemoryStack


class ScriptedDecider:
    def __init__(self, actions: list[SPAction | dict[str, Any]]):
        self.actions = actions
        self.requests: list[CentralDecisionRequest] = []

    def decide(self, request: CentralDecisionRequest) -> SPAction | dict[str, Any]:
        self.requests.append(request)
        if len(self.requests) <= len(self.actions):
            return self.actions[len(self.requests) - 1]
        return self.actions[-1]


def _action(action_type: ActionType | str, **kwargs):
    return SPAction.create(action_type, reason=kwargs.pop("reason", "unit-test reason"), **kwargs)


def test_action_loop_continues_until_finish_and_merges_state():
    decider = ScriptedDecider(
        [
            _action(ActionType.THINK, action_id="act-think", task="Plan the work", stage="planning"),
            _action(ActionType.FINISH, action_id="act-finish", task="Done"),
        ]
    )
    state = {"sp_current_artifact_refs": {"report": {"artifact_id": "report-1", "type": "report"}}}

    result = ActionLoop(decider=decider, max_iterations=5).run(state, thread_id="thread-1", run_id="run-1")

    assert result.next_step == "finish"
    assert result.iterations == 2
    assert len(decider.requests) == 2
    assert "CentralAgent control context" in decider.requests[0].task_context
    restored = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    assert [entry.action for entry in restored.entries] == ["think", "finish"]
    assert result.state_update["sp_current_stage"] == "finished"
    assert any(event["event_type"] == "sp.loop.completed" for event in result.run_events)


def test_action_loop_stops_on_human_interrupt():
    decider = ScriptedDecider(
        [
            _action(
                ActionType.ASK_HUMAN,
                action_id="act-human",
                task="Confirm the outline",
                metadata={"interaction_type": "outline_confirmation"},
            )
        ]
    )

    result = ActionLoop(decider=decider).run({}, thread_id="thread-1", run_id="run-1")

    assert result.next_step == "interrupt"
    assert result.iterations == 1
    assert result.state_update["sp_pending_human_interaction"]["interaction_type"] == "outline_confirmation"
    assert len(decider.requests) == 1


def test_action_loop_recovers_from_invalid_action_and_asks_again():
    decider = ScriptedDecider(
        [
            {"action_type": "THINK", "reason": "missing action id", "task": "bad"},
            _action(ActionType.FINISH, action_id="act-finish", task="Done"),
        ]
    )
    state = {"sp_current_artifact_refs": {"report": {"artifact_id": "report-1", "type": "report"}}}

    result = ActionLoop(decider=decider, max_iterations=3).run(state)

    assert result.next_step == "finish"
    assert result.iterations == 2
    assert len(decider.requests) == 2
    assert any(event["event_type"] == "sp.action.validation_failed" for event in result.run_events)


def test_action_loop_enforces_own_max_iterations():
    decider = ScriptedDecider([_action(ActionType.THINK, action_id="act-think", task="Keep thinking")])

    result = ActionLoop(decider=decider, max_iterations=2).run({})

    assert result.next_step == "error_fatal"
    assert result.iterations == 2
    assert result.error == "SP action loop exceeded max iterations: 2"
    assert len(decider.requests) == 2
    assert result.state_update["sp_last_handler_result"]["next_step"] == "error_fatal"
    assert any(event["event_type"] == "sp.loop.max_iterations_exceeded" for event in result.run_events)
