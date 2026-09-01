#!/usr/bin/env python3
"""Run live Qwen action-policy checks against the StackPlanner Central prompt."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage

from deerflow.models.vllm_provider import VllmChatModel
from deerflow.sp.agent_tools import _action_payload, build_sp_control_tools
from deerflow.sp.central.prompt import CENTRAL_AGENT_ACTION_PROMPT


@dataclass(frozen=True)
class Scenario:
    name: str
    prompt: str
    validate: Callable[[list[dict[str, Any]], str], tuple[bool, str]]


def _first_call(calls: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    if not calls:
        return "", {}
    call = calls[0]
    args = call.get("args")
    return str(call.get("name") or ""), dict(args) if isinstance(args, dict) else {}


def _validate_current_research(calls: list[dict[str, Any]], _: str) -> tuple[bool, str]:
    name, args = _first_call(calls)
    passed = name == "sp_delegate" and args.get("target_agent") == "researcher" and args.get("stage") == "research"
    return passed, "expected one researcher delegation with stage=research"


def _validate_ambiguous_report(calls: list[dict[str, Any]], _: str) -> tuple[bool, str]:
    name, args = _first_call(calls)
    passed = name == "sp_ask_human" or (name == "sp_delegate" and args.get("target_agent") == "perception" and args.get("stage") == "perception")
    return passed, "expected perception task-brief delegation or one consolidated human clarification"


def _validate_report_revision(calls: list[dict[str, Any]], _: str) -> tuple[bool, str]:
    name, args = _first_call(calls)
    metadata = args.get("metadata") if isinstance(args.get("metadata"), dict) else {}
    revision_reason = args.get("revision_reason") or metadata.get("revision_reason")
    refs = args.get("input_refs") if isinstance(args.get("input_refs"), list) else []
    passed = name == "sp_delegate" and args.get("target_agent") == "reporter" and args.get("stage") == "revision" and bool(str(revision_reason or "").strip()) and "report-1" in refs
    return passed, "expected reporter revision with revision_reason and report-1 lineage"


def _validate_blocked_search(calls: list[dict[str, Any]], _: str) -> tuple[bool, str]:
    name, _args = _first_call(calls)
    passed = name in {"sp_ask_human", "sp_think", "sp_reflect", "sp_replan"}
    return passed, "expected no repeated researcher delegation and no premature finish"


def _validate_finish_report(calls: list[dict[str, Any]], _: str) -> tuple[bool, str]:
    name, args = _first_call(calls)
    metadata = args.get("metadata") if isinstance(args.get("metadata"), dict) else {}
    passed = name == "sp_finish" and metadata.get("final_artifact_ref") == "report-1" and metadata.get("required_artifact_type") == "report" and metadata.get("allow_without_artifact") is not True
    return passed, "expected explicit report-1 FINISH contract"


SCENARIOS = (
    Scenario(
        "current_information_routes_to_research",
        """<sp-task-context>
workflow_status: {"pinned_feedback": 0}
recent_task_memory: none
</sp-task-context>
用户请求：请查询今天的国际黄金价格并做两句话分析。请选择下一步动作，不要猜测实时价格。""",
        _validate_current_research,
    ),
    Scenario(
        "ambiguous_report_uses_deliberate_intake",
        """<sp-task-context>
workflow_status: {"pinned_feedback": 0}
recent_task_memory: none
</sp-task-context>
用户请求：帮我写一份新能源汽车行业报告。需求尚未说明受众、范围、时间口径或用途。请选择下一步动作。""",
        _validate_ambiguous_report,
    ),
    Scenario(
        "feedback_revises_current_report",
        """<sp-task-context>
workflow_status: {"pinned_feedback": 1, "report_revision": 1}
critical_feedback:
- id=feedback-1 (feedback, priority=critical): 把结论提前，并保留现有证据表格。
current_artifact_refs: {"report_revision":{"artifact_id":"report-1","type":"report_revision","is_current":true,"run_id":"run-1"}}
current_report_version: 1
</sp-task-context>
<available_skills>stackplanner-reporting: built-in report synthesis and revision workflow</available_skills>
用户正在审阅报告并提出了上述修改。请选择下一步动作。""",
        _validate_report_revision,
    ),
    Scenario(
        "blocked_search_does_not_loop_or_finish",
        """<sp-task-context>
workflow_status: {"pinned_feedback": 0, "partial_or_blocked": 1, "research_observation": 1}
recent_task_memory:
- id=obs-1 (observe, actor=researcher, completion_status=blocked, failure=WEB_SEARCH_UNAVAILABLE): WEB_SEARCH_ATTEMPTS_EXHAUSTED after two equivalent searches; no live result.
</sp-task-context>
原任务需要今天的实时数据，但搜索工具已明确不可用。请选择下一步动作，不要重复同类搜索。""",
        _validate_blocked_search,
    ),
    Scenario(
        "completed_report_finishes_with_contract",
        """<sp-task-context>
workflow_status: {"pinned_feedback": 0, "report_revision": 1}
recent_task_memory:
- id=obs-report (observe, actor=reporter, completion_status=complete, result_ref=report-1): 完整报告已生成并通过质量检查。
current_artifact_refs: {"report_revision":{"artifact_id":"report-1","type":"report_revision","is_current":true,"run_id":"run-1"}}
current_report_version: 1
</sp-task-context>
报告任务已经完成且无需人工审阅（one-shot delivery）。请选择最终动作。""",
        _validate_finish_report,
    ),
)


def _invoke_streaming(model: Any, prompt: str) -> tuple[list[dict[str, Any]], str, dict[str, Any] | None]:
    combined: AIMessageChunk | None = None
    for chunk in model.stream(
        [
            SystemMessage(content=CENTRAL_AGENT_ACTION_PROMPT),
            HumanMessage(content=prompt),
        ]
    ):
        combined = chunk if combined is None else combined + chunk
    if combined is None:
        return [], "", None
    content = combined.content if isinstance(combined.content, str) else json.dumps(combined.content, ensure_ascii=False)
    return list(combined.tool_calls or []), content, combined.usage_metadata


async def _invoke_async_streaming(model: Any, prompt: str) -> tuple[list[dict[str, Any]], str, dict[str, Any] | None]:
    combined: AIMessageChunk | None = None
    async for chunk in model.astream(
        [
            SystemMessage(content=CENTRAL_AGENT_ACTION_PROMPT),
            HumanMessage(content=prompt),
        ]
    ):
        combined = chunk if combined is None else combined + chunk
    if combined is None:
        return [], "", None
    content = combined.content if isinstance(combined.content, str) else json.dumps(combined.content, ensure_ascii=False)
    return list(combined.tool_calls or []), content, combined.usage_metadata


async def _invoke_async_scenarios(
    model: Any,
    scenarios: list[Scenario],
) -> dict[str, tuple[list[dict[str, Any]], str, dict[str, Any] | None] | Exception]:
    """Reuse one event loop/client pool, matching the long-lived Gateway."""
    outcomes: dict[str, tuple[list[dict[str, Any]], str, dict[str, Any] | None] | Exception] = {}
    for scenario in scenarios:
        try:
            outcomes[scenario.name] = await _invoke_async_streaming(model, scenario.prompt)
        except Exception as exc:  # pragma: no cover - live diagnostic boundary
            outcomes[scenario.name] = exc
    return outcomes


def _effective_actions(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for index, call in enumerate(calls):
        name = str(call.get("name") or "")
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        try:
            payload = _action_payload(
                name,
                args,
                tool_call_id=str(call.get("id") or f"eval-{index}"),
            )
        except (KeyError, TypeError, ValueError):
            payload = dict(args)
        actions.append({"name": name, "args": payload})
    return actions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.getenv("SP_QWEN_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=os.getenv("SP_QWEN_MODEL", "Qwen3-32B"))
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--thinking", action="store_true", help="Enable Qwen reasoning mode")
    parser.add_argument("--async-mode", action="store_true", help="Use the Gateway-style async streaming path")
    parser.add_argument("--scenario", action="append", choices=[item.name for item in SCENARIOS])
    args = parser.parse_args()

    model = VllmChatModel(
        model=args.model,
        api_key=os.getenv("SP_QWEN_API_KEY", "EMPTY"),
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=0,
        max_tokens=2048,
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": args.thinking}},
    ).bind_tools(build_sp_control_tools())

    selected_scenarios = [item for item in SCENARIOS if not args.scenario or item.name in args.scenario]
    async_outcomes = asyncio.run(_invoke_async_scenarios(model, selected_scenarios)) if args.async_mode else {}
    results: list[dict[str, Any]] = []
    for scenario in selected_scenarios:
        try:
            if args.async_mode:
                outcome = async_outcomes[scenario.name]
                if isinstance(outcome, Exception):
                    raise outcome
                calls, content, usage = outcome
            else:
                calls, content, usage = _invoke_streaming(model, scenario.prompt)
            effective_actions = _effective_actions(calls)
            passed, expectation = scenario.validate(effective_actions, content)
            results.append(
                {
                    "scenario": scenario.name,
                    "passed": passed,
                    "expectation": expectation,
                    "raw_tool_calls": calls,
                    "effective_actions": effective_actions,
                    "content": content,
                    "usage": usage,
                }
            )
        except Exception as exc:  # pragma: no cover - live diagnostic boundary
            results.append(
                {
                    "scenario": scenario.name,
                    "passed": False,
                    "expectation": "live invocation must complete",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    summary = {
        "model": args.model,
        "base_url": args.base_url,
        "thinking": args.thinking,
        "async_mode": args.async_mode,
        "passed": sum(1 for item in results if item["passed"]),
        "total": len(results),
        "results": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
