"""Checkpointed business-flow tests for the SP graph running on DR2 state."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from deerflow.agents.thread_state import ThreadState
from deerflow.sp import ActionRouter, ActionType, HandlerResult, SPAction, create_sp_agent_graph
from deerflow.sp.agent_tools import SPControlActionMiddleware, build_sp_execution_tools
from deerflow.sp.central.runtime_context import SPCentralRuntimeContext
from deerflow.sp.memory import TaskMemoryStack
from deerflow.sp.middlewares import (
    SPNativeLoopGuardMiddleware,
    SPQwenActionNormalizationMiddleware,
)
from deerflow.sp.runtime import make_sp_agent


class ScriptedDecider:
    def __init__(self, actions: list[SPAction | dict[str, Any]]):
        self.actions = actions
        self.requests = []

    def decide(self, request):
        self.requests.append(request)
        return self.actions[len(self.requests) - 1]


class FakeJournal:
    def __init__(self):
        self.events: list[dict[str, Any]] = []

    def record_custom_event(self, event_type: str, **kwargs: Any) -> None:
        self.events.append({"event_type": event_type, **kwargs})


def _action(action_type: ActionType, action_id: str, **kwargs: Any) -> SPAction:
    return SPAction.create(
        action_type,
        action_id=action_id,
        reason=kwargs.pop("reason", "graph test"),
        **kwargs,
    )


def _graph(decider: ScriptedDecider, *, max_iterations: int = 8):
    return create_sp_agent_graph(
        decider=decider,
        system_prompt="Return SP action JSON only.",
        max_iterations=max_iterations,
    )


def test_graph_exposes_designed_action_loop_nodes():
    graph = _graph(
        ScriptedDecider(
            [
                _action(
                    ActionType.FINISH,
                    "finish",
                    task="Done",
                    metadata={"allow_without_artifact": True},
                )
            ]
        )
    )

    nodes = set(graph.get_graph().nodes)

    assert {
        "prepare_context",
        "central_decide",
        "validate_action",
        "execute_action",
        "route_next",
        "human_request",
        "final_response",
        "final_error",
    } <= nodes


def test_graph_runs_think_then_finish_and_emits_user_facing_message_and_events():
    decider = ScriptedDecider(
        [
            _action(ActionType.THINK, "think-1", task="Plan the task", stage="planning"),
            _action(
                ActionType.FINISH,
                "finish-1",
                task="The task is complete.",
                metadata={"allow_without_artifact": True},
            ),
        ]
    )
    journal = FakeJournal()

    result = _graph(decider).invoke(
        {"messages": [HumanMessage(content="Do the task", id="user-1")]},
        context={"thread_id": "thread-1", "run_id": "run-1", "__run_journal": journal},
    )

    stack = TaskMemoryStack.from_dict(result["sp_task_memory"])
    assert [entry.action for entry in stack.entries] == ["think", "finish"]
    assert result["sp_current_stage"] == "finished"
    assert result["sp_loop_iteration"] == 2
    assert result["sp_loop_run_id"] == "run-1"
    assert isinstance(result["messages"][-1], AIMessage)
    assert result["messages"][-1].content == "The task is complete."
    event_types = [event["event_type"] for event in journal.events]
    assert "sp.action.created" in event_types
    assert "sp.handler.completed" in event_types
    assert event_types[-1] == "sp.loop.completed"


def test_graph_starts_a_new_user_run_without_reusing_previous_summary_or_stage():
    first = _graph(
        ScriptedDecider(
            [
                _action(
                    ActionType.FINISH,
                    "finish-greeting",
                    task="你好！今天有什么我可以帮你的吗？",
                    metadata={"allow_without_artifact": True},
                )
            ]
        )
    ).invoke(
        {"messages": [HumanMessage(content="你好", id="user-1")]},
        context={"thread_id": "thread-new-task", "run_id": "run-greeting"},
    )

    second_decider = ScriptedDecider(
        [
            _action(ActionType.THINK, "plan-report", task="Plan the new report", stage="planning"),
            _action(
                ActionType.FINISH,
                "finish-report",
                task="北京大学计算机学院调研报告已完成。",
                metadata={"allow_without_artifact": True},
            ),
        ]
    )
    second = _graph(second_decider).invoke(
        {
            **first,
            "messages": [*first["messages"], HumanMessage(content="帮我生成北京大学计算机学院调研报告", id="user-2")],
        },
        context={"thread_id": "thread-new-task", "run_id": "run-report"},
    )

    assert second["sp_current_stage"] == "finished"
    assert second["sp_last_run_summary"] == "北京大学计算机学院调研报告已完成。"
    assert second["messages"][-1].content == "北京大学计算机学院调研报告已完成。"
    assert "current_stage: perception" in second_decider.requests[0].task_context


def test_graph_recovers_from_invalid_action_without_losing_iteration_budget():
    decider = ScriptedDecider(
        [
            {"action_type": "THINK", "reason": "missing action id", "task": "bad"},
            _action(
                ActionType.FINISH,
                "finish-after-invalid",
                task="Recovered and finished.",
                metadata={"allow_without_artifact": True},
            ),
        ]
    )

    result = _graph(decider, max_iterations=3).invoke(
        {"messages": [HumanMessage(content="Recover", id="user-1")]},
        context={"thread_id": "thread-1", "run_id": "run-1"},
    )

    assert len(decider.requests) == 2
    assert result["sp_loop_iteration"] == 2
    assert result["sp_last_handler_result"]["next_step"] == "finish"
    assert result["messages"][-1].content == "Recovered and finished."


def test_graph_hitl_request_feedback_resume_and_finish():
    first_graph = _graph(
        ScriptedDecider(
            [
                _action(
                    ActionType.ASK_HUMAN,
                    "ask-1",
                    task="Which report structure should I use?",
                    stage="planning",
                    metadata={
                        "interaction_type": "outline_confirmation",
                        "options": ["Conclusion first", "Evidence first"],
                    },
                )
            ]
        )
    )
    interrupted = first_graph.invoke(
        {"messages": [HumanMessage(content="Draft a report", id="user-1")]},
        context={"thread_id": "thread-1", "run_id": "run-1"},
    )

    request_message = interrupted["messages"][-1]
    assert isinstance(request_message, ToolMessage)
    request = request_message.artifact["human_input"]
    assert request["source"] == "stackplanner"
    assert request["input_mode"] == "choice_with_other"
    assert interrupted["sp_pending_human_interaction"]["status"] == "pending"

    feedback = HumanMessage(
        content="For your clarification, my answer is: Conclusion first",
        id="user-feedback",
        additional_kwargs={
            "hide_from_ui": True,
            "human_input_response": {
                "version": 1,
                "kind": "human_input_response",
                "source": "stackplanner",
                "request_id": request["request_id"],
                "response_kind": "option",
                "option_id": "option-1",
                "value": "Conclusion first",
            },
        },
    )
    second_graph = _graph(
        ScriptedDecider(
            [
                _action(
                    ActionType.FINISH,
                    "finish-after-human",
                    task="Applied the conclusion-first structure.",
                    metadata={"allow_without_artifact": True},
                )
            ]
        )
    )
    finished = second_graph.invoke(
        {**interrupted, "messages": [*interrupted["messages"], feedback]},
        context={"thread_id": "thread-1", "run_id": "run-2"},
    )

    stack = TaskMemoryStack.from_dict(finished["sp_task_memory"])
    human_feedback = [entry for entry in stack.entries if entry.action == "feedback"]
    assert len(human_feedback) == 1
    assert human_feedback[0].content == "Conclusion first"
    assert human_feedback[0].priority == "critical"
    assert human_feedback[0].status == "pinned"
    assert finished["sp_pending_human_interaction"] is None
    assert finished["sp_loop_run_id"] == "run-2"
    assert finished["sp_loop_iteration"] == 1


def test_graph_restores_validated_action_from_dr2_checkpointer():
    decider = ScriptedDecider(
        [
            _action(
                ActionType.FINISH,
                "checkpointed-finish",
                task="Finished after checkpoint restore.",
                metadata={"allow_without_artifact": True},
            )
        ]
    )
    graph = _graph(decider)
    graph.checkpointer = InMemorySaver()
    graph.interrupt_after_nodes = ["validate_action"]
    config = {"configurable": {"thread_id": "checkpoint-thread"}}
    context = {"thread_id": "checkpoint-thread", "run_id": "run-1"}

    paused = graph.invoke(
        {"messages": [HumanMessage(content="Checkpoint this", id="user-1")]},
        config=config,
        context=context,
    )

    assert paused["sp_current_action"]["action_id"] == "checkpointed-finish"
    graph.interrupt_after_nodes = []
    resumed = graph.invoke(None, config=config, context=context)

    assert resumed["sp_last_handler_result"]["next_step"] == "finish"
    assert resumed["messages"][-1].content == "Finished after checkpoint restore."
    assert len(decider.requests) == 1


def test_runtime_factory_builds_single_sp_agent_with_control_tools(monkeypatch):
    class FakeAppConfig:
        models = [SimpleNamespace(name="test-model")]
        title = SimpleNamespace(enabled=True, max_chars=60)

        @staticmethod
        def get_model_config(name):
            assert name == "test-model"
            return SimpleNamespace(supports_thinking=False)

    captured = {}

    class FakeGraph:
        metadata = {}

    def fake_make_lead_agent(config, **kwargs):
        captured.update({"config": config, **kwargs})
        return FakeGraph()

    monkeypatch.setattr("deerflow.agents.lead_agent.agent._make_lead_agent", fake_make_lead_agent)
    config = {"configurable": {"model_name": "test-model"}}

    graph = make_sp_agent(config, app_config=FakeAppConfig())

    assert graph.metadata["agent_name"] == "stackplanner"
    assert graph.metadata["model_name"] == "test-model"
    assert graph.metadata["thinking_enabled"] is False
    assert captured["identity_name"] == "StackPlanner 2.0"
    assert captured["include_base_tools"] is False
    assert captured["include_base_prompt"] is False
    assert captured["control_plane_only"] is True
    assert captured["config"]["configurable"]["subagent_enabled"] is False
    assert captured["config"]["configurable"]["is_plan_mode"] is False
    assert any(middleware.__class__.__name__ == "SPNativeLoopGuardMiddleware" for middleware in captured["extra_middlewares"])
    assert any(middleware.__class__.__name__ == "SPQwenActionNormalizationMiddleware" for middleware in captured["extra_middlewares"])
    assert {tool.name for tool in captured["extra_tools"]} == {
        "sp_think",
        "sp_delegate",
        "sp_recall_memory",
        "sp_reflect",
        "sp_revise",
        "sp_backtrack",
        "sp_replan",
        "sp_summarize",
        "sp_ask_human",
        "sp_finish",
        "sp_thought",
    }


def test_runtime_factory_passes_native_loop_limit_to_guard(monkeypatch):
    class FakeAppConfig:
        models = [SimpleNamespace(name="test-model")]

        @staticmethod
        def get_model_config(name):
            return SimpleNamespace(supports_thinking=False)

    captured = {}

    class FakeGraph:
        metadata = {}

    def fake_make_lead_agent(config, **kwargs):
        captured.update(kwargs)
        return FakeGraph()

    monkeypatch.setattr("deerflow.agents.lead_agent.agent._make_lead_agent", fake_make_lead_agent)

    make_sp_agent(
        {"configurable": {"sp_max_loop_iterations": 7}},
        app_config=FakeAppConfig(),
    )

    guard = next(middleware for middleware in captured["extra_middlewares"] if middleware.__class__.__name__ == "SPNativeLoopGuardMiddleware")
    assert guard.max_iterations == 7


def test_native_runtime_caps_undefined_tool_calls_without_graph_recursion():
    class UndefinedToolModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    model = UndefinedToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": f"bad-{index}",
                        "name": "sp_thoughtful",
                        "args": {},
                        "type": "tool_call",
                    }
                ],
            )
            for index in range(2)
        ]
    )
    graph = create_agent(
        model=model,
        tools=build_sp_execution_tools(),
        middleware=[
            SPNativeLoopGuardMiddleware(max_iterations=2),
            SPQwenActionNormalizationMiddleware(),
        ],
        state_schema=ThreadState,
    )

    result = graph.invoke(
        {"messages": [HumanMessage(content="Search this")]},
        config={"recursion_limit": 50},
        context={"run_id": "run-1"},
    )

    assert result["sp_decision_attempts"] == 2
    assert result["sp_last_handler_result"]["termination"] == "loop_capped"
    assert result["messages"][-1].additional_kwargs["stackplanner"]["status"] == ("loop_capped")


def test_native_runtime_executes_hidden_qwen_thought_alias_as_think():
    class HiddenAliasModel(FakeMessagesListChatModel):
        bound_tool_names: list[str] = []

        def bind_tools(self, tools, **kwargs):
            self.bound_tool_names = [tool.name for tool in tools]
            return self

    model = HiddenAliasModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "alias-1",
                        "name": "sp_thought",
                        "args": {
                            "task": "Plan the research",
                            "stage": "outline",
                        },
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    graph = create_agent(
        model=model,
        tools=build_sp_execution_tools(),
        middleware=[
            SPControlActionMiddleware(),
            SPNativeLoopGuardMiddleware(max_iterations=3),
            SPQwenActionNormalizationMiddleware(),
        ],
        state_schema=ThreadState,
    )

    result = graph.invoke(
        {"messages": [HumanMessage(content="Search this")]},
        context={"run_id": "run-1"},
    )

    alias_message = next(message for message in result["messages"] if isinstance(message, AIMessage) and message.tool_calls)
    assert alias_message.tool_calls[0]["name"] == "sp_thought"
    assert "sp_thought" not in model.bound_tool_names
    assert result["sp_loop_iteration"] == 1
    assert result["sp_current_stage"] == "planning"
    assert result["sp_last_handler_result"]["action_type"] == "THINK"


def test_native_runtime_resets_model_attempt_counter_for_new_run():
    class TextModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    graph = create_agent(
        model=TextModel(responses=[AIMessage(content="fresh answer")]),
        tools=build_sp_execution_tools(),
        middleware=[SPNativeLoopGuardMiddleware(max_iterations=2)],
        state_schema=ThreadState,
    )

    result = graph.invoke(
        {
            "messages": [HumanMessage(content="New request")],
            "sp_loop_run_id": "old-run",
            "sp_decision_attempts": 2,
        },
        context={"run_id": "new-run"},
    )

    assert result["messages"][-1].content == "fresh answer"
    assert result["sp_decision_attempts"] == 1


def test_runtime_factory_injects_episode_local_training_model(monkeypatch):
    class FakeAppConfig:
        models = [SimpleNamespace(name="test-model")]
        title = SimpleNamespace(enabled=False, max_chars=60)

        @staticmethod
        def get_model_config(name):
            assert name == "test-model"
            return SimpleNamespace(supports_thinking=False)

    central_model = object()
    factory_calls = []
    captured = {}

    class FakeGraph:
        metadata = {}

    def training_model_factory(**kwargs):
        factory_calls.append(kwargs)
        return central_model

    def fake_make_lead_agent(config, **kwargs):
        captured.update({"config": config, **kwargs})
        return FakeGraph()

    monkeypatch.setattr(
        "deerflow.agents.lead_agent.agent._make_lead_agent",
        fake_make_lead_agent,
    )

    make_sp_agent(
        {
            "configurable": {
                "model_name": "test-model",
                "episode_id": "episode-7",
                "thread_id": "thread-7",
                "run_id": "run-7",
                "sp_training_model_factory": training_model_factory,
            }
        },
        app_config=FakeAppConfig(),
    )

    assert captured["model"] is central_model
    assert factory_calls == [
        {
            "role": "central",
            "model_name": "test-model",
            "episode_id": "episode-7",
            "thread_id": "thread-7",
            "run_id": "run-7",
            "action_id": None,
            "task_id": None,
            "continuation_attempt": 0,
            "app_config": captured["app_config"],
        }
    ]


def test_runtime_factory_rejects_non_callable_training_model_factory():
    class FakeAppConfig:
        models = [SimpleNamespace(name="test-model")]

    with pytest.raises(
        TypeError,
        match="sp_training_model_factory must be callable",
    ):
        make_sp_agent(
            {"configurable": {"sp_training_model_factory": "not-callable"}},
            app_config=FakeAppConfig(),
        )


def test_runtime_factory_enables_route_bootstrap_only_with_training_factory(
    monkeypatch,
):
    class FakeAppConfig:
        models = [SimpleNamespace(name="test-model")]

        @staticmethod
        def get_model_config(name):
            return SimpleNamespace(supports_thinking=False)

    captured = {}

    class FakeGraph:
        metadata = {}

    def fake_make_lead_agent(config, **kwargs):
        captured.update(kwargs)
        return FakeGraph()

    monkeypatch.setattr(
        "deerflow.agents.lead_agent.agent._make_lead_agent",
        fake_make_lead_agent,
    )

    make_sp_agent(
        {
            "configurable": {
                "sp_training_model_factory": lambda **kwargs: None,
                "sp_training_force_initial_delegate": True,
            }
        },
        app_config=FakeAppConfig(),
    )

    assert any(middleware.__class__.__name__ == "SPTrainingRouteBootstrapMiddleware" for middleware in captured["extra_middlewares"])

    with pytest.raises(
        ValueError,
        match="requires sp_training_model_factory",
    ):
        make_sp_agent(
            {"configurable": {"sp_training_force_initial_delegate": True}},
            app_config=FakeAppConfig(),
        )


def test_runtime_factory_disables_automatic_long_term_memory_injection(monkeypatch):
    class FakeAppConfig:
        models = [SimpleNamespace(name="test-model")]
        title = SimpleNamespace(enabled=False, max_chars=60)

        @staticmethod
        def get_model_config(name):
            return SimpleNamespace(supports_thinking=False)

    captured = {}

    def fake_runtime_context(app_config, *, agent_name=None, user_id=None):
        captured.update({"agent_name": agent_name, "user_id": user_id})
        return SPCentralRuntimeContext(
            agent_name="sp-professor",
            system_prompt_section="<soul>Use concise academic language.</soul>",
            decision_context="<sp-long-term-context><memory>Prefer Chinese.</memory></sp-long-term-context>",
            available_skill_names=frozenset({"stackplanner-long-task"}),
        )

    monkeypatch.setattr("deerflow.sp.runtime.build_sp_central_runtime_context", fake_runtime_context)
    captured_agent = {}

    class FakeGraph:
        metadata = {}

    def fake_make_lead_agent(config, **kwargs):
        captured_agent.update({"config": config, **kwargs})
        return FakeGraph()

    monkeypatch.setattr("deerflow.agents.lead_agent.agent._make_lead_agent", fake_make_lead_agent)
    graph = make_sp_agent(
        {
            "configurable": {"model_name": "test-model"},
            "context": {"agent_name": "sp-professor", "user_id": "user-1"},
        },
        app_config=FakeAppConfig(),
    )

    assert "Use concise academic language." in captured_agent["prompt_prefix"]
    assert "Prefer Chinese." not in captured_agent["prompt_prefix"]
    assert captured_agent["inject_long_term_memory"] is False
    assert captured_agent["identity_name"] == "StackPlanner 2.0"
    assert captured_agent["include_base_tools"] is False
    assert captured_agent["include_base_prompt"] is False
    assert captured_agent["control_plane_only"] is True
    assert captured == {"agent_name": "sp-professor", "user_id": "user-1"}
    assert graph.metadata["available_skills"] == ["stackplanner-long-task"]


def test_duplicate_actions_still_consume_graph_loop_budget():
    repeated = _action(
        ActionType.THINK,
        "repeat-think",
        task="Do not run forever",
        idempotency_key="stable-repeat",
    )
    decider = ScriptedDecider([repeated, repeated, repeated])

    result = _graph(decider, max_iterations=3).invoke(
        {"messages": [HumanMessage(content="Loop safely", id="user-1")]},
        context={"thread_id": "thread-1", "run_id": "run-1"},
    )

    assert len(decider.requests) == 3
    assert result["sp_loop_iteration"] == 3
    assert result["sp_last_handler_result"]["next_step"] == "error_fatal"
    assert "exceeded max iterations" in result["messages"][-1].content
    stack = TaskMemoryStack.from_dict(result["sp_task_memory"])
    assert [entry.action for entry in stack.entries] == ["think"]


def test_broken_interrupt_without_pending_human_routes_to_final_error():
    class BrokenInterruptHandler:
        def handle(self, action, context):
            return HandlerResult(next_step="interrupt")

    router = ActionRouter({ActionType.THINK: BrokenInterruptHandler()})
    graph = create_sp_agent_graph(
        decider=ScriptedDecider([_action(ActionType.THINK, "broken", task="Interrupt incorrectly")]),
        system_prompt="Return JSON only.",
        router=router,
    )

    result = graph.invoke(
        {"messages": [HumanMessage(content="Test defensive routing", id="user-1")]},
        context={"thread_id": "thread-1", "run_id": "run-1"},
    )

    assert result["sp_last_handler_result"]["next_step"] == "error_fatal"
    assert "without pending human interaction" in result["messages"][-1].content


def test_graph_finish_emits_dry_run_promotion_without_writing_long_term_memory():
    stack = TaskMemoryStack()
    stack.append_feedback("以后默认先写结论，再给证据。")
    journal = FakeJournal()
    graph = _graph(
        ScriptedDecider(
            [
                _action(
                    ActionType.FINISH,
                    "finish-promotion",
                    task="Preference acknowledged.",
                    metadata={"allow_without_artifact": True},
                )
            ]
        )
    )

    result = graph.invoke(
        {
            "messages": [HumanMessage(content="Remember this preference", id="user-1")],
            "sp_task_memory": stack.to_dict(),
        },
        context={"thread_id": "thread-1", "run_id": "run-1", "__run_journal": journal},
    )

    promotion_event = next(event for event in journal.events if event["event_type"] == "sp.memory.promotion.dry_run")
    payload = promotion_event["content"]["payload"]
    assert payload["candidate_count"] == 1
    assert payload["approved_count"] == 1
    assert payload["written_count"] == 0
    assert payload["decisions"][0]["kind"] == "user_preference"
    assert result["sp_current_stage"] == "finished"


def test_graph_opts_into_task_memory_middleware_pruning():
    stack = TaskMemoryStack(max_size=50)
    for index in range(30):
        stack.append_think(f"working-memory-{index}")
    pinned = stack.append_feedback("Pinned feedback survives pruning")
    graph = _graph(
        ScriptedDecider(
            [
                _action(
                    ActionType.FINISH,
                    "finish-pruned",
                    task="Pruning verified.",
                    metadata={"allow_without_artifact": True},
                )
            ]
        )
    )

    journal = FakeJournal()
    result = graph.invoke(
        {
            "messages": [HumanMessage(content="Bound the context", id="user-1")],
            "sp_task_memory": stack.to_dict(),
        },
        context={"thread_id": "thread-1", "run_id": "run-1", "__run_journal": journal},
    )

    restored = TaskMemoryStack.from_dict(result["sp_task_memory"])
    active = restored.get_active_entries()
    assert len(active) == 25  # middleware limit plus the terminal FINISH entry
    assert next(entry for entry in restored.entries if entry.id == pinned.id).status == "pinned"
    assert any(entry.status == "pruned" for entry in restored.entries)
    prune_event = next(event for event in journal.events if event["event_type"] == "sp.memory.pruned")
    assert prune_event["content"]["payload"]["entry_count"] == 7
