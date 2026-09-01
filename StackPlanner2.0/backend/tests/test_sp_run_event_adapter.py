"""Tests for persisting SP events through DR2 RunEventStore."""

import pytest

from deerflow.runtime.events.store.memory import MemoryRunEventStore
from deerflow.sp import SPRunEventAdapter, normalize_sp_event_for_store


def test_normalize_sp_event_for_store_preserves_action_and_task_metadata():
    record = normalize_sp_event_for_store(
        {
            "event_type": "sp.delegate.completed",
            "ts": "2026-07-09T00:00:00+00:00",
            "action_id": "act-1",
            "payload": {"task_id": "task-1", "target_agent": "researcher"},
        },
        thread_id="thread-1",
        run_id="run-1",
    )

    assert record["thread_id"] == "thread-1"
    assert record["run_id"] == "run-1"
    assert record["event_type"] == "sp.delegate.completed"
    assert record["category"] == "trace"
    assert record["metadata"]["source"] == "stackplanner"
    assert record["metadata"]["action_id"] == "act-1"
    assert record["metadata"]["task_id"] == "task-1"
    assert record["content"]["payload"]["target_agent"] == "researcher"


def test_normalize_sp_event_for_store_requires_run_id():
    with pytest.raises(ValueError, match="run_id"):
        normalize_sp_event_for_store({"event_type": "sp.action.created"}, thread_id="thread-1")


@pytest.mark.asyncio
async def test_sp_run_event_adapter_writes_events_to_dr2_store():
    store = MemoryRunEventStore()
    adapter = SPRunEventAdapter(store)

    records = await adapter.write_events(
        [
            {"event_type": "sp.action.created", "action_id": "act-1", "payload": {"action_type": "THINK"}},
            {"event_type": "sp.delegate.completed", "action_id": "act-2", "payload": {"task_id": "task-1"}},
        ],
        thread_id="thread-1",
        run_id="run-1",
    )

    assert len(records) == 2
    stored = await store.list_events("thread-1", "run-1")
    assert [event["event_type"] for event in stored] == ["sp.action.created", "sp.delegate.completed"]
    task_events = await store.list_events("thread-1", "run-1", task_id="task-1")
    assert [event["event_type"] for event in task_events] == ["sp.delegate.completed"]
