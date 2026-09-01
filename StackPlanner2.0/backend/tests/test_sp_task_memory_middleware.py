"""Tests for SP TaskMemoryMiddleware."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from deerflow.sp.memory import TaskMemoryStack
from deerflow.sp.middlewares import TaskMemoryMiddleware
from deerflow.sp.middlewares.task_memory_middleware import SP_TASK_CONTEXT_MESSAGE_NAME


def _make_request(*, messages, state):
    request = MagicMock()
    request.messages = list(messages)
    request.state = state
    request.runtime = SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-1"}, state=state)
    request.override = lambda **updates: _override_request(request, updates)
    return request


def _override_request(request, updates):
    new = MagicMock()
    new.messages = updates.get("messages", request.messages)
    new.state = request.state
    new.runtime = request.runtime
    new.override = lambda **kw: _override_request(new, kw)
    return new


def _capture_handler():
    captured = []

    def handler(req):
        captured.append(req)
        return "response"

    return captured, handler


def test_before_agent_restores_legacy_memory_stack_into_thread_state_shape():
    middleware = TaskMemoryMiddleware()
    state = {
        "memory_stack": [
            {
                "timestamp": "2026-07-09T12:00:00",
                "action": "human_feedback",
                "agent_type": "human",
                "content": "Keep HITL feedback above summaries",
                "result": {"priority": "HIGHEST"},
            }
        ]
    }

    result = middleware.before_agent(state, Runtime(context={"thread_id": "thread-1", "run_id": "run-1"}))

    assert result is not None
    payload = result["sp_task_memory"]
    assert payload["version"] == 1
    entry = payload["entries"][0]
    assert entry["action"] == "feedback"
    assert entry["priority"] == "critical"
    assert entry["status"] == "pinned"
    assert entry["thread_id"] == "thread-1"
    assert entry["run_id"] == "run-1"


def test_before_agent_leaves_empty_state_untouched():
    result = TaskMemoryMiddleware().before_agent({}, Runtime(context={"thread_id": "thread-1"}))

    assert result is None


def test_before_agent_initializes_native_sp_run_and_anchors_current_user_request():
    stack = TaskMemoryStack()
    stack.append_summary("Previous task summary", run_id="run-old", stage="finished")
    state = {
        "messages": [HumanMessage(content="Start a new analysis", id="user-new")],
        "sp_task_memory": stack.to_dict(),
        "sp_loop_run_id": "run-old",
        "sp_loop_iteration": 9,
        "sp_current_stage": "finished",
        "sp_last_run_summary": "Old final answer",
        "sp_active_delegate_id": "stale-child",
    }

    result = TaskMemoryMiddleware().before_agent(
        state,
        Runtime(context={"thread_id": "thread-1", "run_id": "run-new"}),
    )

    assert result is not None
    assert result["sp_loop_run_id"] == "run-new"
    assert result["sp_loop_iteration"] == 0
    assert result["sp_current_stage"] == "perception"
    assert result["sp_last_run_summary"] is None
    assert result["sp_active_delegate_id"] is None
    restored = TaskMemoryStack.from_dict(result["sp_task_memory"])
    assert restored.entries[0].content == "Previous task summary"
    request_entry = restored.entries[-1]
    assert request_entry.action == "user_request"
    assert request_entry.content == "Start a new analysis"
    assert request_entry.run_id == "run-new"
    assert request_entry.metadata["source_message_id"] == "user-new"


def test_before_agent_preserves_pending_human_interaction_across_resume_run():
    stack = TaskMemoryStack()
    stack.append_think("Waiting for the user's choice", run_id="run-old", stage="planning")
    pending = {"interaction_id": "ask-1", "status": "pending"}
    state = {
        "messages": [
            HumanMessage(
                content="hidden structured response",
                id="feedback-hidden",
                additional_kwargs={"hide_from_ui": True},
            )
        ],
        "sp_task_memory": stack.to_dict(),
        "sp_loop_run_id": "run-old",
        "sp_current_stage": "planning",
        "sp_pending_human_interaction": pending,
    }

    result = TaskMemoryMiddleware().before_agent(
        state,
        Runtime(context={"thread_id": "thread-1", "run_id": "run-resume"}),
    )

    assert result is not None
    assert "sp_pending_human_interaction" not in result
    assert "sp_current_stage" not in result
    restored = TaskMemoryStack.from_dict(result["sp_task_memory"])
    assert not any(entry.action == "user_request" for entry in restored.entries)


def test_before_agent_isolates_abandoned_run_state_on_fresh_user_turn():
    stack = TaskMemoryStack()
    stack.append_observe("Abandoned raw search result", actor="researcher", run_id="run-old")
    stack.append_summary("Reusable compact decision", run_id="run-old")
    stack.append_feedback("Pinned user preference", run_id="run-old")
    state = {
        "messages": [HumanMessage(content="Do a different task", id="user-new")],
        "sp_task_memory": stack.to_dict(),
        "sp_loop_run_id": "run-old",
        "sp_pending_human_interaction": {"interaction_id": "stale"},
        "sp_current_artifact_refs": {"report": {"artifact_id": "stale-report"}},
    }
    runtime = Runtime(
        context={
            "thread_id": "thread-1",
            "run_id": "run-new",
            "fresh_user_turn_after_terminal": True,
        }
    )

    result = TaskMemoryMiddleware().before_agent(state, runtime)

    assert result is not None
    assert result["sp_pending_human_interaction"] is None
    assert result["sp_current_artifact_refs"] is None
    restored = TaskMemoryStack.from_dict(result["sp_task_memory"])
    contents = {entry.content for entry in restored.entries}
    assert "Abandoned raw search result" not in contents
    assert "Reusable compact decision" in contents
    assert "Pinned user preference" in contents
    assert "Do a different task" in contents


def test_wrap_model_call_injects_context_before_latest_user_message_only_for_request():
    stack = TaskMemoryStack()
    stack.append_feedback("User says report artifacts must stay out of ThreadState", stage="planning")
    state = {
        "sp_task_memory": stack.to_dict(),
        "sp_current_stage": "planning",
        "sp_current_artifact_refs": {"report": "artifact://report"},
    }
    request = _make_request(
        messages=[AIMessage(content="Previous answer"), HumanMessage(content="Continue the migration")],
        state=state,
    )
    captured, handler = _capture_handler()

    result = TaskMemoryMiddleware().wrap_model_call(request, handler)

    assert result == "response"
    sent = captured[0]
    assert sent is not request
    assert request.messages[-1].content == "Continue the migration"
    assert sent.messages[-2].name == SP_TASK_CONTEXT_MESSAGE_NAME
    assert sent.messages[-2].additional_kwargs["hide_from_ui"] is True
    assert "User says report artifacts" in sent.messages[-2].content
    assert sent.messages[-1].content == "Continue the migration"


def test_wrap_model_call_records_exact_context_only_when_debug_enabled():
    stack = TaskMemoryStack()
    entry = stack.append_feedback("Use the approved outline before reporting", stage="planning")
    state = {"sp_task_memory": stack.to_dict(), "sp_current_stage": "planning"}
    request = _make_request(messages=[HumanMessage(content="Continue")], state=state)
    recorded: list[dict] = []

    class FakeJournal:
        def record_custom_event(self, event_type, **kwargs):
            recorded.append({"event_type": event_type, **kwargs})

    request.runtime.context.update(
        {
            "debug_trace_enabled": True,
            "__run_journal": FakeJournal(),
        }
    )

    TaskMemoryMiddleware().wrap_model_call(request, lambda _request: "response")

    assert [event["event_type"] for event in recorded] == ["sp.central.context"]
    content = recorded[0]["content"]
    assert "approved outline" in content["prompt_context"]
    assert content["active_memory_entry_ids"] == [entry.id]
    assert content["current_stage"] == "planning"


def test_wrap_model_call_does_not_record_context_when_debug_disabled():
    stack = TaskMemoryStack()
    stack.append_think("Internal checkpoint")
    state = {"sp_task_memory": stack.to_dict()}
    request = _make_request(messages=[HumanMessage(content="Continue")], state=state)
    journal = MagicMock()
    request.runtime.context["__run_journal"] = journal

    TaskMemoryMiddleware().wrap_model_call(request, lambda _request: "response")

    journal.record_custom_event.assert_not_called()


def test_wrap_model_call_skips_empty_context():
    request = _make_request(messages=[HumanMessage(content="Hello")], state={})
    captured, handler = _capture_handler()

    TaskMemoryMiddleware().wrap_model_call(request, handler)

    assert captured[0] is request


def test_wrap_model_call_replaces_stale_persisted_task_context():
    stack = TaskMemoryStack()
    stack.append_summary("Current compressed memory", run_id="run-1")
    stale_context = HumanMessage(
        content="<sp-task-context>Old uncompressed memory</sp-task-context>",
        name=SP_TASK_CONTEXT_MESSAGE_NAME,
    )
    request = _make_request(
        messages=[
            stale_context,
            HumanMessage(content="Continue from the current state"),
        ],
        state={"sp_task_memory": stack.to_dict()},
    )
    captured, handler = _capture_handler()

    TaskMemoryMiddleware().wrap_model_call(request, handler)

    sent = captured[0]
    contexts = [message for message in sent.messages if getattr(message, "name", None) == SP_TASK_CONTEXT_MESSAGE_NAME]
    assert len(contexts) == 1
    assert "Current compressed memory" in contexts[0].content
    assert "Old uncompressed memory" not in contexts[0].content


def test_wrap_model_call_keeps_compact_summary_on_fresh_user_turn():
    stack = TaskMemoryStack()
    stack.append_summary("Previous task completed; report artifact is ready.", stage="finished")
    state = {"sp_task_memory": stack.to_dict(), "sp_current_stage": "finished"}
    request = _make_request(
        messages=[AIMessage(content="Previous answer"), HumanMessage(content="What did we decide?")],
        state=state,
    )
    request.runtime.context["fresh_user_turn_after_terminal"] = True
    captured, handler = _capture_handler()

    TaskMemoryMiddleware().wrap_model_call(request, handler)

    sent = captured[0]
    assert len(sent.messages) == 2
    assert sent.messages[-1].content == "What did we decide?"
    assert sent.messages[-2].name == SP_TASK_CONTEXT_MESSAGE_NAME
    assert "Previous task completed" in sent.messages[-2].content


def test_after_agent_prunes_active_entries_but_preserves_pinned_feedback():
    stack = TaskMemoryStack()
    stack.append_think("Old normal entry")
    feedback = stack.append_feedback("Pinned human correction")
    stack.append_observe("Another normal entry", actor="researcher")
    state = {"sp_task_memory": stack.to_dict()}

    result = TaskMemoryMiddleware(max_active_entries=1).after_agent(state, Runtime(context={"thread_id": "thread-1"}))

    assert result is not None
    restored = TaskMemoryStack.from_dict(result["sp_task_memory"])
    pinned = restored.get_pinned_entries()
    assert [entry.content for entry in pinned] == [feedback.content]
    assert all(entry.status != "active" for entry in restored.entries if entry.id != feedback.id)


def test_after_agent_auto_compacts_oldest_entries_when_central_skips_summary():
    stack = TaskMemoryStack()
    entries = [
        stack.append_think(
            f"authoritative task update {index}",
            run_id=f"run-{index}",
        )
        for index in range(18)
    ]
    state = {
        "sp_task_memory": stack.to_dict(),
        "sp_current_stage": "reporting",
    }

    result = TaskMemoryMiddleware().after_agent(
        state,
        Runtime(context={"thread_id": "thread-1", "run_id": "run-final"}),
    )

    restored = TaskMemoryStack.from_dict(result["sp_task_memory"])
    summaries = [entry for entry in restored.entries if entry.action == "summarize"]
    assert len(summaries) == 1
    assert summaries[0].parent_ids == [entry.id for entry in entries[:6]]
    assert summaries[0].metadata["automatic_fallback"] is True
    assert summaries[0].run_id == "run-final"
    assert result["sp_summarize_committed_run_id"] == "run-final"
    assert len(restored.entries) == 13
