"""Run a live production StackPlanner researcher/search smoke."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HARNESS_SOURCE = str(ROOT / "StackPlanner2.0/backend/packages/harness")
sys.path.insert(0, HARNESS_SOURCE)

from deerflow.config.app_config import AppConfig  # noqa: E402
from deerflow.sp.runtime import make_sp_agent  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.errors import GraphRecursionError  # noqa: E402

sys.path.remove(HARNESS_SOURCE)


class EventJournal:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def record_custom_event(self, event_type: str, **kwargs: Any) -> None:
        self.events.append({"event_type": event_type, **kwargs})

    def record_external_llm_usage_records(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs


def _message_record(message: Any) -> dict[str, Any]:
    return {
        "type": getattr(message, "type", None),
        "name": getattr(message, "name", None),
        "content": str(getattr(message, "content", ""))[:8000],
        "tool_calls": getattr(message, "tool_calls", None),
    }


def _validate_result(result: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    entries = (result.get("task_memory") or {}).get("entries") or []
    delegated = [entry for entry in entries if entry.get("action") == "delegate"]
    completed_research = [
        entry
        for entry in entries
        if entry.get("action") == "observe"
        and entry.get("actor") == "researcher"
        and (entry.get("metadata") or {}).get("completion_status") == "complete"
    ]
    if len(delegated) != 1:
        failures.append(
            f"expected exactly one StackPlanner delegation, recorded {len(delegated)}"
        )
    if not completed_research:
        failures.append("no completed researcher observation was recorded")
    searchable = json.dumps(result, ensure_ascii=False).lower()
    if "web_search" not in searchable:
        failures.append("no web_search evidence was recorded")
    if "http://" not in searchable and "https://" not in searchable:
        failures.append("no searched result URL was recorded")
    if result.get("current_stage") != "finished":
        failures.append("StackPlanner did not reach the finished stage")
    if result.get("error"):
        failures.append(str(result["error"]))
    if any(
        "llm request failed" in str(message.get("content", "")).lower()
        for message in result.get("messages", [])
    ):
        failures.append("an LLM request failed")
    return failures


async def _run(config_path: Path) -> dict[str, Any]:
    suffix = str(int(time.time()))
    thread_id = f"prod-smoke-researcher-{suffix}"
    run_id = thread_id
    journal = EventJournal()
    runtime_context = {
        "thread_id": thread_id,
        "run_id": run_id,
        "debug_trace_enabled": True,
        "__run_journal": journal,
    }
    graph = make_sp_agent(
        {
            "configurable": {
                "model_name": "stackplanner-training",
                "thread_id": thread_id,
                "run_id": run_id,
                "max_concurrent_subagents": 1,
            },
            "context": runtime_context,
        },
        app_config=AppConfig.from_file(config_path),
    )
    prompt = (
        "This is a production routing smoke test. You must call sp_delegate "
        "exactly once with target_agent researcher. Ask the researcher to use "
        "web_search to identify the capital of France and return at least one "
        'result title and URL. Set metadata.tool_names to ["web_search"]. '
        "After the delegation, finish with the researched answer. Do not answer "
        "directly."
    )
    started = time.monotonic()
    state: dict[str, Any] = {}
    error: str | None = None
    try:
        async for snapshot in graph.astream(
            {"messages": [HumanMessage(content=prompt, id="user-smoke")]},
            config={
                "configurable": {"thread_id": thread_id},
                "recursion_limit": 100,
            },
            context=runtime_context,
            stream_mode="values",
        ):
            state = snapshot
    except GraphRecursionError as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "wall_seconds": round(time.monotonic() - started, 3),
        "thread_id": thread_id,
        "messages": [_message_record(message) for message in state.get("messages", [])],
        "task_memory": state.get("sp_task_memory"),
        "current_stage": state.get("sp_current_stage"),
        "events": journal.events,
        "error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = asyncio.run(_run(args.config.resolve()))
    result["validation_failures"] = _validate_result(result)
    result["success"] = not result["validation_failures"]
    text = json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
