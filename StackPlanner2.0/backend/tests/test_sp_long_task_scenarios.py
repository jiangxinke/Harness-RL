"""Long-running StackPlanner business-flow regression scenarios."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage

from deerflow.sp import ActionType, SPAction, create_sp_agent_graph
from deerflow.sp.memory import TaskMemoryStack
from deerflow.sp.subagents import SPSubagentResult, SPSubagentStatus, SPSubagentTask


class ScriptedDecider:
    def __init__(self, actions: list[SPAction]):
        self.actions = actions
        self.requests = []

    def decide(self, request):
        self.requests.append(request)
        return self.actions[len(self.requests) - 1]


class ScriptedExecutor:
    def __init__(self, results: list[SPSubagentResult]):
        self.results = list(results)
        self.tasks: list[SPSubagentTask] = []

    def execute(self, task: SPSubagentTask) -> SPSubagentResult:
        self.tasks.append(task)
        if not self.results:
            raise AssertionError(f"Unexpected subagent task: {task.subagent_type}")
        return self.results.pop(0)


class FakeJournal:
    def __init__(self):
        self.events: list[dict[str, Any]] = []

    def record_custom_event(self, event_type: str, **kwargs: Any) -> None:
        self.events.append({"event_type": event_type, **kwargs})


def _action(action_type: ActionType, action_id: str, **kwargs: Any) -> SPAction:
    return SPAction.create(
        action_type,
        action_id=action_id,
        reason=kwargs.pop("reason", f"long scenario {action_id}"),
        **kwargs,
    )


def _executor_provider(executor: ScriptedExecutor):
    def provide(state, runtime):
        return executor

    return provide


def test_long_task_recovers_replans_versions_artifacts_and_resumes_after_human_feedback(tmp_path):
    research_body = "RESEARCH-BODY-MUST-STAY-IN-ARTIFACT\n" + ("evidence\n" * 400)
    report_v1 = "REPORT-V1-MUST-STAY-IN-ARTIFACT\n" + ("draft section\n" * 300)
    report_v2 = "REPORT-V2-MUST-STAY-IN-ARTIFACT\n" + ("revised section\n" * 300)
    executor = ScriptedExecutor(
        [
            SPSubagentResult(
                status=SPSubagentStatus.COMPLETED,
                task_id="memory-1",
                result=json.dumps(
                    {
                        "summary": "Prefer conclusion-first reports.",
                        "items": [
                            {
                                "content": "Put the conclusion before evidence.",
                                "source": "user_preference",
                                "score": 0.98,
                                "scope": "user",
                                "memory_id": "mem-1",
                            }
                        ],
                    }
                ),
            ),
            SPSubagentResult(
                status=SPSubagentStatus.COMPLETED,
                task_id="outline-1",
                result="Conclusion-first outline prepared.",
                artifact_content="# Outline\n1. Conclusion\n2. Evidence",
                artifact_type="outline",
            ),
            SPSubagentResult(
                status=SPSubagentStatus.TIMED_OUT,
                task_id="research-timeout",
                error="researcher timed out on an over-broad query",
                stop_reason="turn_capped",
            ),
            SPSubagentResult(
                status=SPSubagentStatus.COMPLETED,
                task_id="research-2",
                result="Narrowed research completed with three sources.",
                artifact_content=research_body,
                artifact_type="research_observation",
            ),
            SPSubagentResult(
                status=SPSubagentStatus.COMPLETED,
                task_id="report-1",
                result="First report draft completed.",
                artifact_content=report_v1,
                artifact_type="report_revision",
            ),
            SPSubagentResult(
                status=SPSubagentStatus.COMPLETED,
                task_id="report-2",
                result="Report revised from critical human feedback.",
                artifact_content=report_v2,
                artifact_type="report_revision",
            ),
        ]
    )
    first_decider = ScriptedDecider(
        [
            _action(ActionType.THINK, "think-plan", task="Plan stages", stage="planning"),
            _action(
                ActionType.RECALL_MEMORY,
                "recall-preferences",
                task="report presentation preferences",
                metadata={"memory_query": "report presentation preferences"},
            ),
            _action(
                ActionType.DELEGATE,
                "outline-first",
                target_agent="outline",
                task="Create a conclusion-first outline",
                stage="planning",
            ),
            _action(
                ActionType.DELEGATE,
                "research-broad",
                target_agent="researcher",
                task="Research every possible angle",
                stage="research",
            ),
            _action(
                ActionType.REFLECT,
                "reflect-timeout",
                task="The research scope was too broad",
                metadata={"failure_note": "The first researcher timed out"},
                stage="research",
            ),
            _action(
                ActionType.REPLAN,
                "replan-narrow",
                task="Narrow research to the three decision-critical questions",
                stage="research",
            ),
            _action(
                ActionType.DELEGATE,
                "research-narrow",
                target_agent="researcher",
                task="Research only the three decision-critical questions",
                stage="research",
            ),
            _action(
                ActionType.DELEGATE,
                "report-draft",
                target_agent="reporter",
                task="Draft the report from current artifacts",
                stage="reporting",
            ),
            _action(
                ActionType.SUMMARIZE,
                "summary-boundary",
                task="Planning and research are complete; report v1 awaits review.",
                stage="verification",
            ),
            _action(
                ActionType.ASK_HUMAN,
                "ask-review",
                task="Should the conclusion be more direct?",
                stage="verification",
                metadata={"interaction_type": "report_feedback"},
            ),
        ]
    )
    first_journal = FakeJournal()
    first_graph = create_sp_agent_graph(
        decider=first_decider,
        system_prompt="Return SP action JSON only.",
        executor_provider=_executor_provider(executor),
        max_iterations=15,
    )
    interrupted = first_graph.invoke(
        {
            "messages": [HumanMessage(content="Produce a deeply researched report", id="user-1")],
            "thread_data": {"outputs_path": str(tmp_path)},
        },
        context={"thread_id": "thread-long", "run_id": "run-1", "__run_journal": first_journal},
    )

    assert len(first_decider.requests) == 10
    assert interrupted["sp_loop_iteration"] == 10
    assert isinstance(interrupted["messages"][-1], ToolMessage)
    request = interrupted["messages"][-1].artifact["human_input"]
    assert request["source"] == "stackplanner"
    assert interrupted["sp_pending_human_interaction"]["status"] == "pending"
    first_stack = TaskMemoryStack.from_dict(interrupted["sp_task_memory"])
    assert any(entry.action == "summarize" for entry in first_stack.entries)
    assert any(event["event_type"] == "sp.delegate.failed" for event in first_journal.events)
    created_action_types = {event["content"]["payload"].get("action_type") for event in first_journal.events if event["event_type"] == "sp.action.created"}
    assert {"REFLECT", "REPLAN", "SUMMARIZE"} <= created_action_types
    serialized_control_state = json.dumps(
        {
            "memory": interrupted["sp_task_memory"],
            "refs": interrupted["sp_current_artifact_refs"],
        }
    )
    assert "RESEARCH-BODY-MUST-STAY-IN-ARTIFACT" not in serialized_control_state
    assert "REPORT-V1-MUST-STAY-IN-ARTIFACT" not in serialized_control_state

    feedback = HumanMessage(
        content="Make the conclusion more direct and preserve the evidence table.",
        id="feedback-1",
        additional_kwargs={
            "hide_from_ui": True,
            "human_input_response": {
                "version": 1,
                "kind": "human_input_response",
                "source": "stackplanner",
                "request_id": request["request_id"],
                "response_kind": "text",
                "value": "Make the conclusion more direct and preserve the evidence table.",
            },
        },
    )
    second_decider = ScriptedDecider(
        [
            _action(
                ActionType.DELEGATE,
                "report-revision",
                target_agent="reporter",
                task="Revise report v1 using the pinned human feedback",
                stage="revision",
                metadata={"revision_reason": "Pinned human feedback requires a revision."},
            ),
            _action(
                ActionType.FINISH,
                "finish-report",
                task="The revised conclusion-first report is complete.",
                stage="finished",
            ),
        ]
    )
    second_journal = FakeJournal()
    second_graph = create_sp_agent_graph(
        decider=second_decider,
        system_prompt="Return SP action JSON only.",
        executor_provider=_executor_provider(executor),
        max_iterations=15,
    )
    finished = second_graph.invoke(
        {**interrupted, "messages": [*interrupted["messages"], feedback]},
        context={"thread_id": "thread-long", "run_id": "run-2", "__run_journal": second_journal},
    )

    assert len(second_decider.requests) == 2
    assert finished["sp_loop_run_id"] == "run-2"
    assert finished["sp_loop_iteration"] == 2
    assert finished["sp_current_stage"] == "finished"
    assert finished["sp_pending_human_interaction"] is None
    final_stack = TaskMemoryStack.from_dict(finished["sp_task_memory"])
    feedback_entries = [entry for entry in final_stack.entries if entry.action == "feedback"]
    assert len(feedback_entries) == 1
    assert feedback_entries[0].priority == "critical"
    assert feedback_entries[0].status == "pinned"
    refs = finished["sp_current_artifact_refs"]
    report_history = [item for item in refs["_history"] if item["type"] == "report_revision"]
    assert [item["version"] for item in report_history] == [1, 2]
    assert refs["report_revision"]["version"] == 2
    assert refs["report_revision"]["artifact_id"] != report_history[0]["artifact_id"]
    assert "REPORT-V2-MUST-STAY-IN-ARTIFACT" not in json.dumps(finished["sp_task_memory"])
    assert "Artifact:" in finished["messages"][-1].content
    assert [task.subagent_type for task in executor.tasks] == [
        "memory_recaller",
        "outline",
        "researcher",
        "researcher",
        "reporter",
        "reporter",
    ]
    draft_bodies = [item["content"] for item in executor.tasks[4].context_refs["artifact_bodies"]["items"]]
    assert any("RESEARCH-BODY-MUST-STAY-IN-ARTIFACT" in body for body in draft_bodies)
    revision_bodies = [item["content"] for item in executor.tasks[5].context_refs["artifact_bodies"]["items"]]
    assert any("REPORT-V1-MUST-STAY-IN-ARTIFACT" in body for body in revision_bodies)
    assert executor.tasks[5].context_refs["report_requirements"]["human_feedback"][0]["content"] == ("Make the conclusion more direct and preserve the evidence table.")
    assert executor.results == []
    assert len(list(tmp_path.rglob("*.*"))) == 4
    event_types = [event["event_type"] for event in [*first_journal.events, *second_journal.events]]
    assert "sp.delegate.failed" in event_types
    assert event_types.count("sp.artifact.created") == 4
    assert "sp.human.feedback_received" in event_types
    assert event_types[-1] == "sp.loop.completed"
