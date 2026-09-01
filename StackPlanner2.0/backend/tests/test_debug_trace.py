"""Execution-trace aggregation for the opt-in SP2 debug surface."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def _record(**overrides):
    values = {
        "run_id": "run-1",
        "thread_id": "thread-1",
        "status": SimpleNamespace(value="success"),
        "metadata": {"debug_trace_enabled": True},
        "created_at": "2026-08-09T10:00:00+00:00",
        "updated_at": "2026-08-09T10:00:08+00:00",
        "total_input_tokens": 100,
        "total_output_tokens": 30,
        "total_tokens": 130,
        "llm_call_count": 2,
        "lead_agent_tokens": 90,
        "subagent_tokens": 40,
        "middleware_tokens": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_build_debug_trace_groups_actions_subagents_and_redacts_secrets():
    from app.gateway.debug_trace import build_debug_trace

    events = [
        {
            "seq": 1,
            "event_type": "llm.human.input",
            "created_at": "2026-08-09T10:00:00.100000+00:00",
            "content": {"type": "human", "content": "帮我查资料并回答"},
            "metadata": {"caller": "lead_agent"},
        },
        {
            "seq": 2,
            "event_type": "llm.ai.response",
            "created_at": "2026-08-09T10:00:01.500000+00:00",
            "content": {
                "type": "ai",
                "content": "",
                "additional_kwargs": {"reasoning_content": "private hidden reasoning"},
                "tool_calls": [
                    {
                        "name": "sp_delegate",
                        "args": {
                            "target_agent": "researcher",
                            "task": "查询公开资料",
                            "reason": "答案依赖当前资料",
                            "api_key": "sk-secret-value",
                        },
                    }
                ],
            },
            "metadata": {
                "caller": "lead_agent",
                "latency_ms": 500,
                "usage": {"input_tokens": 60, "output_tokens": 10, "total_tokens": 70},
            },
        },
        {
            "seq": 3,
            "event_type": "sp.action.created",
            "created_at": "2026-08-09T10:00:01.600000+00:00",
            "content": {
                "action_id": "action-1",
                "payload": {
                    "action_type": "DELEGATE",
                    "target_agent": "researcher",
                    "task": "查询公开资料",
                    "reason": "答案依赖当前资料",
                    "stage": "research",
                },
            },
            "metadata": {"action_id": "action-1", "source": "stackplanner"},
        },
        {
            "seq": 4,
            "event_type": "sp.handler.started",
            "created_at": "2026-08-09T10:00:01.700000+00:00",
            "content": {"action_id": "action-1", "payload": {"action_type": "DELEGATE"}},
            "metadata": {"action_id": "action-1"},
        },
        {
            "seq": 5,
            "event_type": "subagent.start",
            "created_at": "2026-08-09T10:00:06+00:00",
            "content": {
                "task_id": "task-1",
                "description": "查询公开资料",
                "subagent_type": "researcher",
                "occurred_at": "2026-08-09T10:00:02+00:00",
            },
            "metadata": {"task_id": "task-1"},
        },
        {
            "seq": 6,
            "event_type": "subagent.step",
            "created_at": "2026-08-09T10:00:06+00:00",
            "content": {
                "task_id": "task-1",
                "message_index": 1,
                "kind": "ai",
                "text": "",
                "tool_calls": [{"name": "web_search", "args": {"query": "测试资料", "authorization": "Bearer secret"}}],
                "occurred_at": "2026-08-09T10:00:03+00:00",
            },
            "metadata": {"task_id": "task-1"},
        },
        {
            "seq": 7,
            "event_type": "subagent.step",
            "created_at": "2026-08-09T10:00:06+00:00",
            "content": {
                "task_id": "task-1",
                "message_index": 2,
                "kind": "tool",
                "tool_name": "web_search",
                "text": "搜索结果",
                "occurred_at": "2026-08-09T10:00:05+00:00",
            },
            "metadata": {"task_id": "task-1"},
        },
        {
            "seq": 8,
            "event_type": "subagent.end",
            "created_at": "2026-08-09T10:00:06+00:00",
            "content": {
                "task_id": "task-1",
                "status": "completed",
                "result": "资料已找到",
                "subagent_type": "researcher",
                "usage": {"input_tokens": 30, "output_tokens": 10, "total_tokens": 40},
                "occurred_at": "2026-08-09T10:00:06+00:00",
            },
            "metadata": {"task_id": "task-1"},
        },
        {
            "seq": 9,
            "event_type": "sp.handler.completed",
            "created_at": "2026-08-09T10:00:06.200000+00:00",
            "content": {
                "action_id": "action-1",
                "payload": {"action_type": "DELEGATE", "next_step": "continue"},
            },
            "metadata": {"action_id": "action-1"},
        },
        {
            "seq": 10,
            "event_type": "llm.ai.response",
            "created_at": "2026-08-09T10:00:07+00:00",
            "content": {"type": "ai", "content": "这是最终回答", "tool_calls": []},
            "metadata": {
                "caller": "lead_agent",
                "latency_ms": 800,
                "usage": {"input_tokens": 20, "output_tokens": 20, "total_tokens": 40},
            },
        },
    ]

    trace = build_debug_trace(_record(), events)

    assert trace["enabled"] is True
    assert trace["duration_ms"] == 8000
    assert trace["tokens"]["total"] == 130
    assert trace["tokens"]["subagent"] == 40

    action = next(step for step in trace["steps"] if step["kind"] == "action")
    assert action["label"] == "执行动作 · DELEGATE"
    assert action["duration_ms"] == 4500
    assert action["summary"] == "答案依赖当前资料"

    subagent = next(step for step in trace["steps"] if step["kind"] == "subagent")
    assert subagent["actor"] == "researcher"
    assert subagent["duration_ms"] == 4000
    assert subagent["tokens"]["total"] == 40

    tool_request = next(step for step in trace["steps"] if step["kind"] == "tool_request")
    assert tool_request["parent_id"] == subagent["id"]
    assert tool_request["duration_ms"] == 2000

    final_answer = next(step for step in trace["steps"] if step["kind"] == "final_answer")
    assert final_answer["summary"] == "这是最终回答"

    serialized = json.dumps(trace, ensure_ascii=False)
    assert "sk-secret-value" not in serialized
    assert "Bearer secret" not in serialized
    assert "private hidden reasoning" not in serialized
    assert "[REDACTED]" in serialized


def test_build_debug_trace_exposes_exact_prompt_context_but_not_hidden_reasoning():
    from app.gateway.debug_trace import build_debug_trace

    trace = build_debug_trace(
        _record(),
        [
            {
                "seq": 1,
                "event_type": "sp.central.decided",
                "created_at": "2026-08-09T10:00:01+00:00",
                "content": {
                    "payload": {
                        "action_type": "THINK",
                        "reason": "需要核对当前约束",
                        "prompt_context": "recent_task_memory: 用户要求先研究后写报告",
                        "prompt_context_chars": 36,
                    }
                },
                "metadata": {"source": "stackplanner"},
            }
        ],
    )

    decision = next(step for step in trace["steps"] if step["kind"] == "decision_context")
    assert "先研究后写报告" in decision["detail"]["prompt_context"]
    assert trace["disclosure"]["hidden_chain_of_thought"] is False


def test_build_debug_trace_gives_repeated_subagent_message_indexes_unique_step_ids():
    from app.gateway.debug_trace import build_debug_trace

    events = [
        {
            "seq": 1,
            "event_type": "subagent.start",
            "created_at": "2026-08-09T10:00:00+00:00",
            "content": {"task_id": "task-1", "subagent_type": "researcher"},
        },
        *[
            {
                "seq": seq,
                "event_type": "subagent.step",
                "created_at": f"2026-08-09T10:00:0{seq}+00:00",
                "content": {
                    "task_id": "task-1",
                    "message_index": 1,
                    "kind": "ai",
                    "text": f"observation {seq}",
                },
            }
            for seq in (2, 3)
        ],
    ]

    trace = build_debug_trace(_record(), events)
    child_ids = [step["id"] for step in trace["steps"] if step["parent_id"] == "subagent-task-1"]

    assert len(child_ids) == 2
    assert len(set(child_ids)) == 2


def test_native_sp_event_recorder_preserves_original_event_time():
    from deerflow.sp.agent_tools import SPControlActionMiddleware

    recorded: list[dict] = []

    class FakeJournal:
        def record_custom_event(self, event_type, **kwargs):
            recorded.append({"event_type": event_type, **kwargs})

    runtime = SimpleNamespace(context={"__run_journal": FakeJournal()})
    SPControlActionMiddleware._record_events(
        runtime,
        [
            {
                "event_type": "sp.handler.started",
                "ts": "2026-08-09T10:00:02+00:00",
                "action_id": "action-1",
                "run_id": "run-1",
                "payload": {"action_type": "DELEGATE"},
            }
        ],
    )

    assert recorded[0]["content"]["occurred_at"] == "2026-08-09T10:00:02+00:00"
    assert recorded[0]["metadata"]["occurred_at"] == "2026-08-09T10:00:02+00:00"


@pytest.mark.anyio
async def test_debug_trace_endpoint_pages_events_and_checks_run_scope():
    from app.gateway.routers.thread_runs import get_run_debug_trace

    calls: list[int | None] = []
    pages = {
        None: [{"seq": 1, "event_type": "llm.human.input", "content": {"content": "hello"}, "created_at": "2026-08-09T10:00:00+00:00"}],
        1: [],
    }

    class FakeStore:
        async def list_events(self, thread_id, run_id, *, limit=500, after_seq=None, **_kwargs):
            calls.append(after_seq)
            return pages[after_seq]

    class FakeRunManager:
        async def get(self, run_id, user_id=None):
            assert user_id == "user-1"
            return _record()

    class FakeState:
        run_event_store = FakeStore()
        run_manager = FakeRunManager()

    class FakeApp:
        state = FakeState()

    class FakeUser:
        id = "user-1"

    class FakeRequest:
        app = FakeApp()
        state = SimpleNamespace(user=FakeUser(), auth_source="session")
        _deerflow_test_bypass_auth = True

    result = await get_run_debug_trace(thread_id="thread-1", run_id="run-1", request=FakeRequest())

    assert result["run_id"] == "run-1"
    assert result["steps"][0]["kind"] == "user_query"
    assert calls == [None]
