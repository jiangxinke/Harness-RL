"""Focused tests for the single-loop SP control-tool layer."""

import asyncio
import json
import threading
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deerflow.sp.agent_tools import (
    SP_ARTIFACT_PATHS_KEY,
    SPAcknowledgementMiddleware,
    SPControlActionMiddleware,
    SPFinishAvailabilityMiddleware,
    SPTerminalActionMiddleware,
    SPThinkLabelMiddleware,
    _action_payload,
    build_sp_control_tools,
    build_sp_execution_tools,
)
from deerflow.sp.memory import TaskMemoryStack
from deerflow.sp.middlewares import (
    SPNativeLoopGuardMiddleware,
    SPQwenActionNormalizationMiddleware,
)
from deerflow.sp.subagents import SPSubagentResult, SPSubagentStatus


def test_sp_control_tools_are_prefixed_and_do_not_replace_native_tools():
    tools = build_sp_control_tools()
    names = [tool.name for tool in tools]

    assert names == [
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
    ]


def test_terminal_sp_control_tools_are_return_direct_for_native_agent_loop():
    tools = {tool.name: tool for tool in build_sp_control_tools()}

    assert tools["sp_ask_human"].return_direct is True
    assert tools["sp_finish"].return_direct is False
    assert all(tool.return_direct is False for name, tool in tools.items() if name != "sp_ask_human")


def test_terminal_middleware_only_ends_an_accepted_finish():
    middleware = SPTerminalActionMiddleware()

    accepted = middleware.before_model(
        {"sp_last_handler_result": {"next_step": "finish"}},
        SimpleNamespace(context={}),
    )
    rejected = middleware.before_model(
        {
            "sp_last_handler_result": {
                "next_step": "error_recoverable",
                "error": "FINISH requires allow_without_artifact",
            }
        },
        SimpleNamespace(context={}),
    )

    assert accepted == {"jump_to": "end"}
    assert rejected is None


def test_qwen_action_normalization_hides_unambiguous_alias_from_model_schema():
    middleware = SPQwenActionNormalizationMiddleware()
    request = _ModelRequestStub(tools=build_sp_execution_tools(), state={}, messages=[])
    captured = {}

    def handler(model_request):
        captured["request"] = model_request
        return AIMessage(content="next")

    response = middleware.wrap_model_call(request, handler)

    assert response.content == "next"
    assert {tool.name for tool in captured["request"].tools} == {tool.name for tool in build_sp_control_tools()}
    assert "sp_thought" in {tool.name for tool in request.tools}
    assert "sp_thoughtful" not in {tool.name for tool in build_sp_execution_tools()}


def test_qwen_thought_alias_routes_to_think_without_rewriting_tool_name():
    payload = _action_payload(
        "sp_thought",
        {"task": "Inspect sources", "stage": "outline"},
        tool_call_id="alias-1",
    )

    assert payload["action_type"] == "THINK"
    assert payload["stage"] == "outline"


def test_qwen_thought_alias_executes_through_control_middleware():
    middleware = SPControlActionMiddleware()
    request = SimpleNamespace(
        tool_call={
            "name": "sp_thought",
            "id": "alias-1",
            "args": {
                "task": "Inspect sources",
                "stage": "outline",
            },
        },
        state={},
        runtime=SimpleNamespace(context={"run_id": "run-1"}),
    )

    command = middleware.wrap_tool_call(
        request,
        lambda _: (_ for _ in ()).throw(AssertionError("native alias handler must not run")),
    )

    assert command.update["sp_current_stage"] == "planning"
    assert command.update["sp_last_handler_result"]["action_type"] == "THINK"
    assert command.update["messages"][0].name == "sp_thought"


def test_native_loop_guard_counts_model_attempts_and_ends_at_exact_cap():
    middleware = SPNativeLoopGuardMiddleware(max_iterations=2)
    runtime = SimpleNamespace(context={"run_id": "run-1"})

    first = middleware.after_model(
        {"messages": [AIMessage(content="", tool_calls=[])]},
        runtime,
    )
    second = middleware.after_model(
        {
            "messages": [AIMessage(content="", tool_calls=[])],
            "sp_decision_attempts": first["sp_decision_attempts"],
        },
        runtime,
    )
    capped = middleware.before_model(
        {
            "messages": [HumanMessage(content="tool result")],
            "sp_decision_attempts": second["sp_decision_attempts"],
        },
        runtime,
    )

    assert first == {"sp_decision_attempts": 1}
    assert second == {"sp_decision_attempts": 2}
    assert capped["jump_to"] == "end"
    assert capped["sp_last_handler_result"]["termination"] == "loop_capped"
    assert capped["sp_last_handler_result"]["next_step"] == "error_fatal"
    assert capped["messages"][0].additional_kwargs["stackplanner"]["status"] == ("loop_capped")


def test_native_loop_guard_initializes_limit_and_preserves_accepted_finish():
    middleware = SPNativeLoopGuardMiddleware(max_iterations=3)
    assert middleware.before_agent({}, SimpleNamespace(context={})) == {"sp_max_loop_iterations": 3}
    assert middleware.before_agent(
        {"sp_loop_run_id": "old-run", "sp_decision_attempts": 3},
        SimpleNamespace(context={"run_id": "new-run"}),
    ) == {"sp_max_loop_iterations": 3, "sp_decision_attempts": 0}
    assert (
        middleware.before_model(
            {
                "sp_decision_attempts": 3,
                "sp_last_handler_result": {"next_step": "finish"},
            },
            SimpleNamespace(context={"run_id": "run-1"}),
        )
        is None
    )


class _ModelRequestStub:
    def __init__(self, *, tools, state, messages=None):
        self.tools = tools
        self.state = state
        self.messages = list(messages or [])

    def override(self, **changes):
        return _ModelRequestStub(
            tools=changes.get("tools", self.tools),
            state=changes.get("state", self.state),
            messages=changes.get("messages", self.messages),
        )


def test_finish_schema_is_hidden_until_finalizable_artifact_exists():
    middleware = SPFinishAvailabilityMiddleware()
    tools = build_sp_control_tools()
    request = _ModelRequestStub(tools=tools, state={})

    filtered = middleware._filter_request(request)

    assert "sp_finish" not in {tool.name for tool in filtered.tools}
    assert "sp_delegate" in {tool.name for tool in filtered.tools}


def test_perception_gaps_force_ask_human_before_delegation_or_search():
    middleware = SPFinishAvailabilityMiddleware()
    request = _ModelRequestStub(
        tools=build_sp_control_tools(),
        state={
            "sp_loop_run_id": "run-1",
            "sp_last_handler_result": {
                "action_type": "DELEGATE",
                "target_agent": "perception",
                "next_step": "continue",
            },
            "sp_current_artifact_refs": {
                "perception_observation": {
                    "artifact_id": "brief-1",
                    "type": "perception_observation",
                    "run_id": "run-1",
                    "metadata": {
                        "task_ready": False,
                        "clarification_questions": [
                            "What reporting period should the board report cover?",
                            "Which internal metrics are available?",
                        ],
                    },
                }
            },
        },
        messages=[HumanMessage(content="Create the board report.")],
    )

    filtered = middleware._filter_request(request)

    assert {tool.name for tool in filtered.tools} == {"sp_ask_human"}
    instruction = filtered.messages[-1]
    assert instruction.name == "sp_clarification_required"
    assert "Do not delegate, search, plan" in instruction.content
    assert "What reporting period" in instruction.content
    assert "Which internal metrics" in instruction.content


def test_perception_clarification_guard_preserves_chinese_user_language():
    middleware = SPFinishAvailabilityMiddleware()
    request = _ModelRequestStub(
        tools=build_sp_control_tools(),
        state={
            "sp_loop_run_id": "run-1",
            "sp_last_handler_result": {
                "action_type": "DELEGATE",
                "target_agent": "perception",
                "next_step": "continue",
            },
            "sp_current_artifact_refs": {
                "perception_observation": {
                    "artifact_id": "brief-zh-1",
                    "type": "perception_observation",
                    "run_id": "run-1",
                    "metadata": {
                        "task_ready": False,
                        "clarification_questions": ["What source data is available?"],
                    },
                }
            },
        },
        messages=[HumanMessage(content="请生成董事会复盘报告。")],
    )

    filtered = middleware._filter_request(request)

    instruction = filtered.messages[-1]
    assert instruction.name == "sp_clarification_required"
    assert "user wrote in Chinese, so ask in Chinese" in instruction.content


def test_completed_report_from_previous_delegate_forces_exact_finish():
    middleware = SPFinishAvailabilityMiddleware()
    request = _ModelRequestStub(
        tools=build_sp_control_tools(),
        state={
            "sp_loop_run_id": "run-1",
            "sp_last_handler_result": {
                "action_type": "DELEGATE",
                "action_id": "delegate-report-1",
                "target_agent": "reporter",
                "next_step": "continue",
            },
            "sp_current_artifact_refs": {
                "report_revision": {
                    "artifact_id": "report-final-1",
                    "type": "report_revision",
                    "run_id": "run-1",
                    "is_current": True,
                    "metadata": {
                        "completion_status": "complete",
                        "delegate_action_id": "delegate-report-1",
                    },
                }
            },
        },
        messages=[HumanMessage(content="Create the board report.")],
    )

    filtered = middleware._filter_request(request)

    assert {tool.name for tool in filtered.tools} == {"sp_finish"}
    instruction = filtered.messages[-1]
    assert instruction.name == "sp_report_finalization_required"
    assert 'final_artifact_ref="report-final-1"' in instruction.content
    assert "do not ask another question" in instruction.content


def test_completed_coder_artifact_from_previous_delegate_forces_exact_finish():
    middleware = SPFinishAvailabilityMiddleware()
    request = _ModelRequestStub(
        tools=build_sp_control_tools(),
        state={
            "sp_loop_run_id": "run-1",
            "sp_last_handler_result": {
                "action_type": "DELEGATE",
                "action_id": "delegate-code-1",
                "target_agent": "coder",
                "next_step": "continue",
            },
            "sp_current_artifact_refs": {
                "generated_file": {
                    "artifact_id": "code-final-1",
                    "type": "generated_file",
                    "virtual_path": "/mnt/user-data/outputs/ranges.py",
                    "run_id": "run-1",
                    "is_current": True,
                    "metadata": {
                        "completion_status": "complete",
                        "delegate_action_id": "delegate-code-1",
                        "finalization_preview": ("A酒店总费用: 5250.48\nB酒店总费用: 5108.50\n结论: B酒店便宜141.98元\n"),
                    },
                }
            },
        },
        messages=[HumanMessage(content="写好脚本后把文件给我下载。")],
    )

    filtered = middleware._filter_request(request)

    assert {tool.name for tool in filtered.tools} == {"sp_finish"}
    instruction = filtered.messages[-1]
    assert instruction.name == "sp_artifact_finalization_required"
    assert 'final_artifact_ref="code-final-1"' in instruction.content
    assert 'required_artifact_type="generated_file"' in instruction.content
    assert "do not delegate more work" in instruction.content
    assert "A酒店总费用: 5250.48" in instruction.content
    assert "ignore stale prior conclusions" in instruction.content


def test_completed_research_for_conversation_forces_finish_without_artifact():
    middleware = SPFinishAvailabilityMiddleware()
    stack = TaskMemoryStack()
    stack.append_observe(
        "Paris is the capital of France. Source: https://example.com/paris",
        actor="researcher",
        run_id="run-1",
        metadata={
            "target_agent": "researcher",
            "completion_status": "complete",
        },
    )
    request = _ModelRequestStub(
        tools=build_sp_control_tools(),
        state={
            "sp_loop_run_id": "run-1",
            "sp_task_memory": stack.to_dict(),
        },
        messages=[HumanMessage(content="What is the capital of France? Include a source URL.")],
    )

    filtered = middleware._filter_request(request)

    assert {tool.name for tool in filtered.tools} == {"sp_finish"}
    instruction = filtered.messages[-1]
    assert instruction.name == "sp_conversational_finalization_required"
    assert "allow_without_artifact=true" in instruction.content
    assert "latest specialist observation" in instruction.content


def test_partial_research_for_conversation_forces_bounded_redelegation():
    middleware = SPFinishAvailabilityMiddleware()
    stack = TaskMemoryStack()
    stack.append_delegate(
        "Find a source",
        run_id="run-1",
        metadata={"target_agent": "researcher"},
    )
    stack.append_observe(
        "A likely result was found, but the URL is missing.",
        actor="researcher",
        run_id="run-1",
        failure_note="No source URL was returned.",
        metadata={
            "target_agent": "researcher",
            "completion_status": "partial",
            "evidence_gaps": ["No source URL was returned."],
        },
    )
    request = _ModelRequestStub(
        tools=build_sp_control_tools(),
        state={
            "sp_loop_run_id": "run-1",
            "sp_task_memory": stack.to_dict(),
        },
        messages=[HumanMessage(content="What is the capital of France? Include a source URL.")],
    )

    filtered = middleware._filter_request(request)

    assert {tool.name for tool in filtered.tools} == {"sp_delegate"}
    instruction = filtered.messages[-1]
    assert instruction.name == "sp_artifact_recovery_required"
    assert 'target_agent="researcher"' in instruction.content
    assert "No source URL was returned" in instruction.content


def test_finish_schema_is_available_for_current_report_artifact():
    middleware = SPFinishAvailabilityMiddleware()
    tools = build_sp_control_tools()
    request = _ModelRequestStub(
        tools=tools,
        state={
            "sp_current_artifact_refs": {
                "report": {
                    "artifact_id": "report-1",
                    "type": "report",
                    "is_current": True,
                }
            }
        },
    )

    filtered = middleware._filter_request(request)

    assert "sp_finish" in {tool.name for tool in filtered.tools}


def test_finish_schema_ignores_intermediate_and_superseded_report_artifacts():
    middleware = SPFinishAvailabilityMiddleware()
    tools = build_sp_control_tools()
    request = _ModelRequestStub(
        tools=tools,
        state={
            "sp_current_artifact_refs": {
                "outline": {
                    "artifact_id": "outline-1",
                    "type": "outline",
                },
                "report": {
                    "artifact_id": "report-old",
                    "type": "report",
                    "is_current": False,
                },
            }
        },
    )

    filtered = middleware._filter_request(request)

    assert "sp_finish" not in {tool.name for tool in filtered.tools}


def test_finish_schema_ignores_partial_deliverable_artifact():
    middleware = SPFinishAvailabilityMiddleware()
    request = _ModelRequestStub(
        tools=build_sp_control_tools(),
        state={
            "sp_current_artifact_refs": {
                "report_revision": {
                    "artifact_id": "partial-report",
                    "type": "report_revision",
                    "is_current": True,
                    "metadata": {"completion_status": "partial"},
                }
            }
        },
    )

    filtered = middleware._filter_request(request)

    assert "sp_finish" not in {tool.name for tool in filtered.tools}


def test_finish_schema_requires_a_fresh_artifact_for_revision_request():
    middleware = SPFinishAvailabilityMiddleware()
    request = _ModelRequestStub(
        tools=build_sp_control_tools(),
        state={
            "messages": [HumanMessage(content="修订刚才的报告并交付可下载文件。")],
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
    )

    filtered = middleware._filter_request(request)

    assert "sp_finish" not in {tool.name for tool in filtered.tools}


def test_rejected_finish_without_artifact_forces_next_turn_to_answer_directly():
    middleware = SPFinishAvailabilityMiddleware()
    request = _ModelRequestStub(
        tools=build_sp_control_tools(),
        state={
            "sp_last_handler_result": {
                "action_type": "FINISH",
                "next_step": "error_recoverable",
                "error": "FINISH requires a final artifact",
            }
        },
        messages=[HumanMessage(content="现在输出最终 JSON。")],
    )

    filtered = middleware._filter_request(request)

    assert not {tool.name for tool in filtered.tools} & {tool.name for tool in build_sp_control_tools()}
    assert filtered.messages[-1].name == "sp_direct_answer_required"
    assert filtered.messages[-1].additional_kwargs["hide_from_ui"] is True


def test_rejected_finish_for_file_deliverable_forces_specialist_recovery():
    middleware = SPFinishAvailabilityMiddleware()
    request = _ModelRequestStub(
        tools=build_sp_control_tools(),
        state={
            "messages": [HumanMessage(content="生成一份董事会 Markdown 报告并作为可下载文件交付。")],
            "sp_last_handler_result": {
                "action_type": "FINISH",
                "next_step": "error_recoverable",
                "error": "FINISH requires a final artifact",
            },
        },
        messages=[HumanMessage(content="生成一份董事会 Markdown 报告并作为可下载文件交付。")],
    )

    filtered = middleware._filter_request(request)
    tool_names = {tool.name for tool in filtered.tools}

    assert tool_names == {"sp_delegate"}
    assert all(getattr(message, "name", None) != "sp_direct_answer_required" for message in filtered.messages)
    assert filtered.messages[-1].name == "sp_artifact_recovery_required"
    assert 'target_agent="reporter"' in filtered.messages[-1].content
    assert "invent a download URL" in filtered.messages[-1].content


def test_rejected_finish_for_partial_report_forces_revision_with_evidence_gaps():
    middleware = SPFinishAvailabilityMiddleware()
    request = _ModelRequestStub(
        tools=build_sp_control_tools(),
        state={
            "messages": [HumanMessage(content="交付修订后的可下载 Markdown 报告。")],
            "sp_current_artifact_refs": {
                "report_revision": {
                    "artifact_id": "report-partial-1",
                    "type": "report_revision",
                    "is_current": True,
                    "metadata": {
                        "completion_status": "partial",
                        "evidence_gaps": ["Missing REVIEWED-HIGH and OLD-R7"],
                    },
                }
            },
            "sp_last_handler_result": {
                "action_type": "FINISH",
                "next_step": "error_recoverable",
                "error": "FINISH requires a complete final artifact",
            },
        },
        messages=[HumanMessage(content="交付修订后的可下载 Markdown 报告。")],
    )

    filtered = middleware._filter_request(request)

    assert {tool.name for tool in filtered.tools} == {"sp_delegate"}
    recovery = filtered.messages[-1]
    assert recovery.name == "sp_artifact_recovery_required"
    assert 'target_agent="reporter"' in recovery.content
    assert 'stage="revision"' in recovery.content
    assert 'input_refs=["report-partial-1"]' in recovery.content
    assert "REVIEWED-HIGH and OLD-R7" in recovery.content


def test_partial_report_delegate_forces_revision_before_finish():
    stack = TaskMemoryStack()
    stack.append_observe(
        "[PARTIAL result: partial] Reporter did not create the file.",
        actor="reporter",
        run_id="run-2",
        failure_note="Reporter completed without creating a report artifact.",
        metadata={
            "completion_status": "partial",
            "evidence_gaps": [
                "Missing REVIEWED-HIGH",
                "Reporter completed without creating a report artifact.",
            ],
        },
    )
    middleware = SPFinishAvailabilityMiddleware()
    request = _ModelRequestStub(
        tools=build_sp_control_tools(),
        state={
            "messages": [HumanMessage(content="修订刚才的报告并交付可下载 Markdown 文件。")],
            "sp_loop_run_id": "run-2",
            "sp_task_memory": stack.to_dict(),
            "sp_last_handler_result": {
                "action_type": "DELEGATE",
                "next_step": "continue",
            },
            "sp_current_artifact_refs": {
                "report_revision": {
                    "artifact_id": "report-v1",
                    "type": "report_revision",
                    "run_id": "run-1",
                    "is_current": True,
                }
            },
        },
        messages=[HumanMessage(content="修订刚才的报告并交付可下载 Markdown 文件。")],
    )

    filtered = middleware._filter_request(request)

    assert {tool.name for tool in filtered.tools} == {"sp_delegate"}
    recovery = filtered.messages[-1]
    assert recovery.name == "sp_artifact_recovery_required"
    assert 'input_refs=["report-v1"]' in recovery.content
    assert "Missing REVIEWED-HIGH" in recovery.content
    assert "return the complete revised Markdown in artifact_content" in recovery.content


def test_current_run_partial_report_cannot_be_bypassed_by_revise():
    middleware = SPFinishAvailabilityMiddleware()
    request = _ModelRequestStub(
        tools=build_sp_control_tools(),
        state={
            "sp_loop_run_id": "run-2",
            "sp_last_handler_result": {
                "action_type": "REVISE",
                "next_step": "continue",
            },
            "sp_current_artifact_refs": {
                "report_revision": {
                    "artifact_id": "report-partial-run-2",
                    "type": "report_revision",
                    "run_id": "run-2",
                    "is_current": True,
                    "metadata": {
                        "completion_status": "partial",
                        "evidence_gaps": ["Remove the non-portable local link."],
                    },
                }
            },
        },
        messages=[HumanMessage(content="请继续完成报告。")],
    )

    filtered = middleware._filter_request(request)

    assert {tool.name for tool in filtered.tools} == {"sp_delegate"}
    recovery = filtered.messages[-1]
    assert recovery.name == "sp_artifact_recovery_required"
    assert 'input_refs=["report-partial-run-2"]' in recovery.content
    assert "Remove the non-portable local link" in recovery.content


def test_partial_report_recovery_stops_after_three_reporter_attempts():
    stack = TaskMemoryStack()
    for index in range(3):
        stack.append_delegate(
            f"Reporter attempt {index + 1}",
            run_id="run-2",
            metadata={"target_agent": "reporter"},
        )
    middleware = SPFinishAvailabilityMiddleware()
    request = _ModelRequestStub(
        tools=build_sp_control_tools(),
        state={
            "sp_loop_run_id": "run-2",
            "sp_task_memory": stack.to_dict(),
            "sp_last_handler_result": {
                "action_type": "DELEGATE",
                "target_agent": "reporter",
                "next_step": "continue",
            },
            "sp_current_artifact_refs": {
                "report_revision": {
                    "artifact_id": "report-partial-v3",
                    "type": "report_revision",
                    "version": 2,
                    "run_id": "run-2",
                    "is_current": True,
                    "metadata": {
                        "completion_status": "partial",
                        "evidence_gaps": ["The same citation gap remains unresolved."],
                    },
                }
            },
        },
        messages=[HumanMessage(content="请继续完成报告。")],
    )

    filtered = middleware._filter_request(request)

    assert {tool.name for tool in filtered.tools} == {"sp_ask_human"}
    instruction = filtered.messages[-1]
    assert instruction.name == "sp_report_recovery_exhausted"
    assert "3 reporter attempts" in instruction.content
    assert "must not be delegated again" in instruction.content
    assert "same citation gap remains unresolved" in instruction.content


def test_partial_coder_recovery_stops_after_three_attempts_even_after_reflection():
    stack = TaskMemoryStack()
    for index in range(3):
        stack.append_delegate(
            f"Coder attempt {index + 1}",
            run_id="run-2",
            metadata={"target_agent": "coder"},
        )
        stack.append_observe(
            "[PARTIAL result: partial] Script verification still fails.",
            actor="coder",
            run_id="run-2",
            metadata={
                "completion_status": "partial",
                "evidence_gaps": ["The generated Python file still fails its tests."],
                "target_agent": "coder",
            },
        )
    stack.append_reflect(
        "The previous repair strategy repeated the same failure.",
        run_id="run-2",
    )
    middleware = SPFinishAvailabilityMiddleware()
    request = _ModelRequestStub(
        tools=build_sp_control_tools(),
        state={
            "sp_loop_run_id": "run-2",
            "sp_task_memory": stack.to_dict(),
            "sp_last_handler_result": {
                "action_type": "REFLECT",
                "next_step": "continue",
            },
        },
        messages=[HumanMessage(content="请写好脚本、实际测试并给我下载。")],
    )

    filtered = middleware._filter_request(request)

    assert not {tool.name for tool in filtered.tools} & {tool.name for tool in build_sp_control_tools()}
    instruction = filtered.messages[-1]
    assert instruction.name == "sp_delegate_recovery_exhausted"
    assert "3 coder attempts" in instruction.content
    assert "must not delegate again" in instruction.content
    assert "still fails its tests" in instruction.content
    assert "do not claim that a verified file is ready" in instruction.content


def test_plain_success_claim_for_partial_coder_artifact_becomes_recovery_delegate():
    stack = TaskMemoryStack()
    stack.append_delegate(
        "Create and test the script.",
        run_id="run-2",
        metadata={"target_agent": "coder"},
    )
    stack.append_observe(
        "[PARTIAL result: partial] The negative-range test still fails.",
        actor="coder",
        run_id="run-2",
        result_ref="partial-code-1",
        metadata={
            "completion_status": "partial",
            "evidence_gaps": ["The '-3--1,2' test still fails."],
            "target_agent": "coder",
        },
    )
    middleware = SPFinishAvailabilityMiddleware()
    state = {
        "sp_loop_run_id": "run-2",
        "sp_task_memory": stack.to_dict(),
        "messages": [
            HumanMessage(content="修好后实际测试，再给我可下载的代码文件。"),
            AIMessage(content="脚本已经修好并通过全部测试。"),
        ],
    }

    update = middleware.after_model(
        state,
        SimpleNamespace(context={"run_id": "run-2"}),
    )

    assert update is not None
    guarded = update["messages"][0]
    assert isinstance(guarded, AIMessage)
    assert guarded.content == ""
    assert len(guarded.tool_calls) == 1
    recovery = guarded.tool_calls[0]
    assert recovery["name"] == "sp_delegate"
    assert recovery["args"]["target_agent"] == "coder"
    assert recovery["args"]["metadata"]["tool_names"] == [
        "read_file",
        "write_file",
        "str_replace",
        "bash",
    ]
    assert "'-3--1,2' test still fails" in recovery["args"]["task"]


def test_plain_success_claim_after_coder_recovery_limit_becomes_honest_failure():
    stack = TaskMemoryStack()
    for index in range(3):
        stack.append_delegate(
            f"Coder attempt {index + 1}",
            run_id="run-2",
            metadata={"target_agent": "coder"},
        )
        stack.append_observe(
            "[PARTIAL result: partial] The negative-range test still fails.",
            actor="coder",
            run_id="run-2",
            metadata={
                "completion_status": "partial",
                "evidence_gaps": ["The '-3--1,2' test still fails."],
                "target_agent": "coder",
            },
        )
    middleware = SPFinishAvailabilityMiddleware()
    state = {
        "sp_loop_run_id": "run-2",
        "sp_task_memory": stack.to_dict(),
        "messages": [
            HumanMessage(content="修好后实际测试，再给我可下载的代码文件。"),
            AIMessage(content="脚本已经修好并通过全部测试。"),
        ],
    }

    update = middleware.after_model(
        state,
        SimpleNamespace(context={"run_id": "run-2"}),
    )

    assert update is not None
    guarded = update["messages"][0]
    assert isinstance(guarded, AIMessage)
    assert guarded.tool_calls == []
    assert "尚未通过验证" in guarded.content
    assert "3 次恢复上限" in guarded.content
    assert "'-3--1,2' test still fails" in guarded.content
    assert "全部测试" not in guarded.content


def test_report_recovery_limit_is_scoped_to_current_run_not_global_version():
    stack = TaskMemoryStack()
    for index in range(3):
        stack.append_delegate(
            f"Earlier-run reporter attempt {index + 1}",
            run_id="run-1",
            metadata={"target_agent": "reporter"},
        )
    stack.append_delegate(
        "Current-run reporter attempt 1",
        run_id="run-2",
        metadata={"target_agent": "reporter"},
    )
    middleware = SPFinishAvailabilityMiddleware()
    request = _ModelRequestStub(
        tools=build_sp_control_tools(),
        state={
            "sp_loop_run_id": "run-2",
            "sp_task_memory": stack.to_dict(),
            "sp_last_handler_result": {
                "action_type": "DELEGATE",
                "target_agent": "reporter",
                "next_step": "continue",
            },
            "sp_current_artifact_refs": {
                "report_revision": {
                    "artifact_id": "report-global-v4-current-run-v1",
                    "type": "report_revision",
                    "version": 4,
                    "run_id": "run-2",
                    "is_current": True,
                    "metadata": {
                        "completion_status": "partial",
                        "evidence_gaps": ["Rewrite the complete report in Chinese."],
                    },
                }
            },
        },
        messages=[HumanMessage(content="请继续完成本轮报告修订。")],
    )

    filtered = middleware._filter_request(request)

    assert {tool.name for tool in filtered.tools} == {"sp_delegate"}
    instruction = filtered.messages[-1]
    assert instruction.name == "sp_artifact_recovery_required"
    assert 'input_refs=["report-global-v4-current-run-v1"]' in instruction.content
    assert "Rewrite the complete report in Chinese" in instruction.content


def test_action_payload_adds_sp_type_without_forcing_a_json_response():
    payload = _action_payload(
        "sp_reflect",
        {"task": "Check the evidence", "reason": "The result is incomplete"},
        tool_call_id="tool-1",
    )

    assert payload["action_type"] == "REFLECT"
    assert payload["action_id"] == "spact_tool-1"
    assert payload["task"] == "Check the evidence"


def test_finish_action_payload_accepts_task_alias_from_model():
    payload = _action_payload(
        "sp_finish",
        {
            "task": "报告已完成并可供下载。",
            "final_artifact_ref": "report-1",
            "required_artifact_type": "report",
        },
        tool_call_id="tool-finish-task-alias",
    )

    assert payload["task"] == "报告已完成并可供下载。"
    assert payload["metadata"]["final_artifact_ref"] == "report-1"
    assert payload["metadata"]["required_artifact_type"] == "report"


def test_action_payload_routes_pure_markdown_report_from_coder_to_reporter():
    payload = _action_payload(
        "sp_delegate",
        {
            "target_agent": "coder",
            "task": "Generate the complete board report in Markdown format.",
            "expected_output": "A downloadable board report",
            "stage": "implementation",
        },
        tool_call_id="tool-report-route",
    )

    assert payload["target_agent"] == "reporter"
    assert payload["stage"] == "reporting"
    assert payload["metadata"]["routed_from_agent"] == "coder"
    assert payload["metadata"]["routing_reason"] == "pure_report_synthesis"


def test_action_payload_does_not_treat_report_source_precedence_as_source_code():
    payload = _action_payload(
        "sp_delegate",
        {
            "target_agent": "coder",
            "task": ("Generate the final board Markdown report from the current authoritative data sources."),
            "expected_output": ("A complete report with source precedence and audit markers."),
            "stage": "implementation",
        },
        tool_call_id="tool-report-source-precedence",
    )

    assert payload["target_agent"] == "reporter"
    assert payload["stage"] == "reporting"
    assert payload["metadata"]["routing_reason"] == "pure_report_synthesis"


def test_action_payload_keeps_code_generation_with_coder():
    payload = _action_payload(
        "sp_delegate",
        {
            "target_agent": "coder",
            "task": "Generate an HTML report webpage and its source code.",
            "expected_output": "Runnable HTML, CSS, and JavaScript files",
            "stage": "implementation",
        },
        tool_call_id="tool-code-route",
    )

    assert payload["target_agent"] == "coder"
    assert payload["stage"] == "implementation"
    assert "routed_from_agent" not in payload["metadata"]


def test_action_payload_infers_perception_for_read_only_coder_delegation():
    payload = _action_payload(
        "sp_delegate",
        {
            "target_agent": "coder",
            "task": "Read the uploaded correction memo and extract its identifiers.",
            "metadata": {"tool_names": ["read_file"]},
        },
        tool_call_id="tool-read-stage",
    )

    assert payload["target_agent"] == "coder"
    assert payload["stage"] == "perception"


def test_acknowledgement_middleware_short_circuits_explicit_record_only_turn():
    middleware = SPAcknowledgementMiddleware()
    result = middleware.before_model(
        {
            "messages": [
                HumanMessage(
                    content="修订1：城市改为杭州，上海作废。只确认收到。",
                    name="user-input",
                )
            ]
        },
        SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-1"}),
    )

    assert result["jump_to"] == "end"
    assert result["messages"][0].content == "收到。"
    assert result["messages"][0].additional_kwargs["stackplanner"]["status"] == "acknowledged"


def test_acknowledgement_middleware_does_not_swallow_final_output_request():
    middleware = SPAcknowledgementMiddleware()

    result = middleware.before_model(
        {
            "messages": [
                HumanMessage(
                    content="现在输出最终结果，不要只确认收到。",
                    name="user-input",
                )
            ]
        },
        SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-2"}),
    )

    assert result is None


def test_delegate_action_payload_keeps_parent_tool_call_id_for_progress_events():
    payload = _action_payload(
        "sp_delegate",
        {
            "target_agent": "researcher",
            "task": "Search official sources",
            "metadata": {"skill_names": ["deep-research"], "__sp_parent_tool_call_id": "forged"},
        },
        tool_call_id="tool-delegate",
    )

    assert payload["metadata"]["skill_names"] == ["deep-research"]
    assert payload["metadata"]["__sp_parent_tool_call_id"] == "tool-delegate"


def test_delegate_action_payload_promotes_explicit_revision_reason_to_metadata():
    payload = _action_payload(
        "sp_delegate",
        {
            "target_agent": "reporter",
            "task": "Revise the current report",
            "input_refs": ["report-1"],
            "revision_reason": "Move the conclusion first and preserve the evidence table.",
        },
        tool_call_id="tool-revise-report",
    )

    assert "revision_reason" not in {key for key in payload if key != "metadata"}
    assert payload["metadata"]["revision_reason"] == "Move the conclusion first and preserve the evidence table."


def test_delegate_action_payload_canonicalizes_role_stage_and_recovers_revision_reason():
    perception = _action_payload(
        "sp_delegate",
        {
            "target_agent": "perception",
            "task": "Normalize the request",
            "stage": "planning",
        },
        tool_call_id="tool-perception-stage",
    )
    revision = _action_payload(
        "sp_delegate",
        {
            "target_agent": "reporter",
            "task": "Revise the current report",
            "reason": "The user asked to move the conclusion first.",
            "stage": "revision",
            "input_refs": ["report-1"],
        },
        tool_call_id="tool-report-revision-fallback",
    )

    assert perception["stage"] == "perception"
    assert revision["stage"] == "revision"
    assert revision["metadata"]["revision_reason"] == "The user asked to move the conclusion first."


def test_revise_action_payload_targets_memory_entries_and_keeps_correction_as_task():
    payload = _action_payload(
        "sp_revise",
        {
            "target_entry_ids": ["spmem-wrong"],
            "correction": "The source is from the official website.",
            "reason": "The previous observation used an unofficial mirror.",
        },
        tool_call_id="tool-revise",
    )

    assert payload["action_type"] == "REVISE"
    assert payload["task"] == "The source is from the official website."
    assert payload["metadata"]["target_entry_ids"] == ["spmem-wrong"]
    assert payload["metadata"]["revision_reason"] == payload["reason"]


def test_reflect_action_payload_carries_optional_backtrack_targets():
    payload = _action_payload(
        "sp_reflect",
        {
            "task": "The first calculation used the wrong discount order.",
            "target_entry_ids": ["spmem-wrong-calculation"],
        },
        tool_call_id="tool-reflect-backtrack",
    )

    assert payload["action_type"] == "REFLECT"
    assert payload["metadata"]["target_entry_ids"] == ["spmem-wrong-calculation"]


def test_summarize_action_payload_carries_source_entry_ids():
    payload = _action_payload(
        "sp_summarize",
        {
            "summary": "Keep only the verified implementation decision.",
            "source_entry_ids": ["spmem-old-1", "spmem-old-2"],
        },
        tool_call_id="tool-summary",
    )

    assert payload["action_type"] == "SUMMARIZE"
    assert payload["task"] == "Keep only the verified implementation decision."
    assert payload["metadata"]["source_entry_ids"] == ["spmem-old-1", "spmem-old-2"]


def test_sp_control_middleware_executes_handler_in_place():
    middleware = SPControlActionMiddleware()
    request = SimpleNamespace(
        tool_call={
            "name": "sp_reflect",
            "id": "tool-reflect",
            "args": {"task": "Inspect the failed result"},
        },
        state={},
        runtime=SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-1"}),
    )

    command = middleware.wrap_tool_call(request, lambda _: (_ for _ in ()).throw(AssertionError("native tool handler must not run")))

    assert command.update["sp_last_action_id"] == "spact_tool-reflect"
    assert command.update["sp_last_handler_result"]["action_type"] == "REFLECT"
    assert command.update["messages"][0].additional_kwargs["stackplanner"]["action_type"] == "REFLECT"


def test_sp_control_middleware_blocks_same_turn_duplicate_summarize_calls():
    middleware = SPControlActionMiddleware()
    first_request = SimpleNamespace(
        tool_call={"name": "sp_summarize", "id": "tool-summary-1", "args": {"summary": "Compact the task memory"}},
        state={},
        runtime=SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-1"}),
    )
    second_request = SimpleNamespace(
        tool_call={"name": "sp_summarize", "id": "tool-summary-2", "args": {"summary": "Compact it again"}},
        state={},
        runtime=SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-1"}),
    )

    first = middleware.wrap_tool_call(first_request, lambda _: (_ for _ in ()).throw(AssertionError("native handler must not run")))
    second = middleware.wrap_tool_call(second_request, lambda _: (_ for _ in ()).throw(AssertionError("native handler must not run")))

    assert first.update["sp_summarize_committed_run_id"] == "run-1"
    assert not second.goto
    assert second.update["sp_last_handler_result"]["next_step"] == "continue"
    assert "already executing in this model turn" in second.update["sp_last_handler_result"]["error"]
    assert "sp_task_memory" not in second.update


def test_sp_control_middleware_executes_handler_in_async_stream():
    middleware = SPControlActionMiddleware()
    request = SimpleNamespace(
        tool_call={"name": "sp_think", "id": "tool-think", "args": {"task": "Make a checkpoint"}},
        state={},
        runtime=SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-1"}),
    )

    async def run():
        async def unexpected_handler(_):
            raise AssertionError("SP control tool must not call the native handler")

        return await middleware.awrap_tool_call(request, unexpected_handler)

    command = asyncio.run(run())

    assert command.update["sp_last_handler_result"]["action_type"] == "THINK"


def test_sp_control_middleware_offloads_async_actions_from_event_loop_thread():
    middleware = SPControlActionMiddleware()
    request = SimpleNamespace(
        tool_call={"name": "sp_think", "id": "tool-thread", "args": {"task": "Checkpoint"}},
        state={},
        runtime=SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-1"}),
    )
    event_loop_thread = threading.get_ident()
    execution_threads = []
    original = middleware._execute_control_tool

    def recording_execute(req):
        execution_threads.append(threading.get_ident())
        return original(req)

    middleware._execute_control_tool = recording_execute

    async def run():
        async def unexpected_handler(_):
            raise AssertionError("SP control tool must not call the native handler")

        return await middleware.awrap_tool_call(request, unexpected_handler)

    asyncio.run(run())

    assert execution_threads
    assert execution_threads[0] != event_loop_thread


def test_sp_delegate_persists_terminal_subagent_metadata_and_artifact_paths(tmp_path):
    class FakeExecutor:
        def execute(self, task):
            return SPSubagentResult(
                status=SPSubagentStatus.COMPLETED,
                result="Found the current typhoon from official sources.",
                task_id="tool-delegate",
                artifact_content="# Typhoon report\n\nVerified details.",
                artifact_type="report_revision",
                artifact_metadata={"filename": "typhoon-report.md"},
            )

    middleware = SPControlActionMiddleware(executor_provider=lambda state, runtime: FakeExecutor())
    request = SimpleNamespace(
        tool_call={
            "name": "sp_delegate",
            "id": "tool-delegate",
            "args": {
                "target_agent": "researcher",
                "task": "Find the latest typhoon",
                "stage": "report",
            },
        },
        state={"thread_data": {"outputs_path": str(tmp_path)}},
        runtime=SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-1"}),
    )

    command = middleware.wrap_tool_call(
        request,
        lambda _: (_ for _ in ()).throw(AssertionError("native tool handler must not run")),
    )

    tool_message = command.update["messages"][0]
    assert tool_message.additional_kwargs["subagent_status"] == "completed"
    assert tool_message.additional_kwargs["subagent_result_brief"] == ("Found the current typhoon from official sources.")
    artifact_paths = tool_message.additional_kwargs[SP_ARTIFACT_PATHS_KEY]
    assert len(artifact_paths) == 1
    assert artifact_paths[0].startswith("/mnt/user-data/outputs/sp/report_revision/")
    assert artifact_paths[0].endswith("/typhoon-report.md")
    assert command.update["artifacts"] == artifact_paths
    assert next(tmp_path.rglob("*.md")).read_text(encoding="utf-8").startswith("# Typhoon report")


def test_sp_delegate_tool_message_exposes_partial_and_stop_reason_to_central():
    class FakeExecutor:
        def execute(self, task):
            return SPSubagentResult(
                status=SPSubagentStatus.COMPLETED,
                result="Only the first two checks completed.",
                task_id="task-capped",
                stop_reason="loop_capped",
                artifact_metadata={
                    "completion_status": "partial",
                    "evidence_gaps": ["The final verification is missing."],
                },
            )

    middleware = SPControlActionMiddleware(executor_provider=lambda state, runtime: FakeExecutor())
    request = SimpleNamespace(
        tool_call={
            "name": "sp_delegate",
            "id": "tool-delegate-capped",
            "args": {
                "target_agent": "researcher",
                "task": "Complete three verification checks",
                "expected_output": "All three checks verified",
            },
        },
        state={},
        runtime=SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-1"}),
    )

    command = middleware.wrap_tool_call(
        request,
        lambda _: (_ for _ in ()).throw(AssertionError("native tool handler must not run")),
    )

    tool_message = command.update["messages"][0]
    payload = json.loads(tool_message.content)
    assert payload["completion_status"] == "partial"
    assert payload["stop_reason"] == "loop_capped"
    assert payload["result"].startswith("[PARTIAL result: loop_capped]")
    assert tool_message.additional_kwargs["subagent_status"] == "completed"
    assert tool_message.additional_kwargs["subagent_stop_reason"] == "loop_capped"


def test_sp_finish_auto_presents_current_report_artifact():
    report_path = "/mnt/user-data/outputs/sp/report_revision/report-v1.md"
    middleware = SPControlActionMiddleware()
    request = SimpleNamespace(
        tool_call={
            "name": "sp_finish",
            "id": "tool-finish",
            "args": {"summary": "报告已经完成"},
        },
        state={
            "sp_current_artifact_refs": {
                "report_revision": {
                    "artifact_id": "spart-report",
                    "virtual_path": report_path,
                    "run_id": "run-1",
                }
            }
        },
        runtime=SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-1"}),
    )

    command = middleware.wrap_tool_call(
        request,
        lambda _: (_ for _ in ()).throw(AssertionError("native tool handler must not run")),
    )

    assert command.goto == "__end__"
    assert [message.name for message in command.update["messages"] if isinstance(message, ToolMessage)] == [
        "sp_finish",
        "present_files",
    ]
    present_message = command.update["messages"][1]
    assert present_message.tool_calls[0]["name"] == "present_files"
    assert present_message.tool_calls[0]["args"]["filepaths"] == [report_path]
    assert isinstance(command.update["messages"][-1], AIMessage)
    assert command.update["messages"][-1].content == "报告已经完成"


def test_sp_finish_tool_forwards_deliverable_contract():
    middleware = SPControlActionMiddleware()
    request = SimpleNamespace(
        tool_call={
            "name": "sp_finish",
            "id": "tool-finish-contract",
            "args": {
                "summary": "报告已经完成",
                "final_artifact_ref": "report-1",
                "required_artifact_type": "report",
            },
        },
        state={
            "sp_current_artifact_refs": {
                "report_revision": {
                    "artifact_id": "report-1",
                    "type": "report_revision",
                    "virtual_path": "/mnt/user-data/outputs/report.md",
                    "run_id": "run-1",
                }
            }
        },
        runtime=SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-1"}),
    )

    command = middleware.wrap_tool_call(
        request,
        lambda _: (_ for _ in ()).throw(AssertionError("native tool handler must not run")),
    )

    result = json.loads(command.update["messages"][0].content)
    assert result["next_step"] == "finish"
    assert command.update["sp_last_final_artifact_ref"] == "report-1"


def test_sp_finish_visible_answer_uses_grounded_result_preview():
    preview = "A酒店总费用: 5250.48\nB酒店总费用: 5108.50\n结论: B酒店便宜141.98元\n"
    middleware = SPControlActionMiddleware()
    request = SimpleNamespace(
        tool_call={
            "name": "sp_finish",
            "id": "tool-finish-grounded",
            "args": {
                "summary": "A酒店4811.43元，应该选A酒店。",
                "final_artifact_ref": "hotel-result",
                "required_artifact_type": "generated_file",
            },
        },
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
        runtime=SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-1"}),
    )

    command = middleware.wrap_tool_call(
        request,
        lambda _: (_ for _ in ()).throw(AssertionError("native tool handler must not run")),
    )

    assert command.update["messages"][-1].content == preview.strip()
    assert "4811.43" not in command.update["messages"][-1].content


def test_sp_finish_presents_all_current_run_generated_files_and_latest_report():
    paths = [
        "/mnt/user-data/outputs/site/index.html",
        "/mnt/user-data/outputs/site/styles.css",
        "/mnt/user-data/outputs/report-v2.md",
    ]
    middleware = SPControlActionMiddleware()
    request = SimpleNamespace(
        tool_call={
            "name": "sp_finish",
            "id": "tool-finish-bundle",
            "args": {"summary": "网页与报告均已完成"},
        },
        state={
            "sp_loop_run_id": "run-1",
            "sp_current_artifact_refs": {
                "generated_file": {
                    "artifact_id": "css",
                    "type": "generated_file",
                    "virtual_path": paths[1],
                    "run_id": "run-1",
                    "is_current": True,
                },
                "report_revision": {
                    "artifact_id": "report-v2",
                    "type": "report_revision",
                    "virtual_path": paths[2],
                    "run_id": "run-1",
                    "is_current": True,
                },
                "_history": [
                    {
                        "artifact_id": "html",
                        "type": "generated_file",
                        "virtual_path": paths[0],
                        "run_id": "run-1",
                        "is_current": False,
                    },
                    {
                        "artifact_id": "report-v1",
                        "type": "report",
                        "virtual_path": "/mnt/user-data/outputs/report-v1.md",
                        "run_id": "run-1",
                        "is_current": False,
                    },
                ],
            },
        },
        runtime=SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-1"}),
    )

    command = middleware.wrap_tool_call(
        request,
        lambda _: (_ for _ in ()).throw(AssertionError("native tool handler must not run")),
    )

    present_message = command.update["messages"][1]
    assert present_message.tool_calls[0]["args"]["filepaths"] == paths


def test_sp_finish_beside_nonterminal_action_does_not_end_the_tool_batch():
    finish_call = {
        "name": "sp_finish",
        "id": "tool-finish-early",
        "args": {"summary": "Premature finish"},
        "type": "tool_call",
    }
    delegate_call = {
        "name": "sp_delegate",
        "id": "tool-delegate-still-running",
        "args": {
            "target_agent": "researcher",
            "task": "Collect the missing evidence",
        },
        "type": "tool_call",
    }
    request = SimpleNamespace(
        tool_call=finish_call,
        state={
            "messages": [AIMessage(content="", tool_calls=[finish_call, delegate_call])],
            "sp_current_artifact_refs": {
                "report": {
                    "artifact_id": "report-1",
                    "virtual_path": "/mnt/user-data/outputs/report.md",
                    "run_id": "run-1",
                }
            },
        },
        runtime=SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-1"}),
    )

    command = SPControlActionMiddleware().wrap_tool_call(
        request,
        lambda _: (_ for _ in ()).throw(AssertionError("native tool handler must not run")),
    )

    assert command.goto != "__end__"
    assert command.update["sp_last_handler_result"]["next_step"] == "error_recoverable"
    assert "sibling result" in command.update["sp_last_handler_result"]["error"]
    assert len(command.update["messages"]) == 1


def test_sp_ask_human_emits_interactive_card_contract_and_resumes_from_reply():
    middleware = SPControlActionMiddleware()
    request = SimpleNamespace(
        tool_call={
            "name": "sp_ask_human",
            "id": "tool-human",
            "args": {
                "question": "你希望采用哪种报告结构？",
                "interaction_type": "report_structure",
                "options": ["结论优先", "证据优先"],
            },
        },
        state={},
        runtime=SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-1"}),
    )

    command = middleware.wrap_tool_call(
        request,
        lambda _: (_ for _ in ()).throw(AssertionError("native tool handler must not run")),
    )

    tool_message = command.update["messages"][0]
    card = tool_message.artifact["human_input"]
    assert command.goto == "__end__"
    assert tool_message.name == "sp_ask_human"
    assert tool_message.content == "你希望采用哪种报告结构？"
    assert card["source"] == "stackplanner"
    assert card["input_mode"] == "choice_with_other"
    assert [option["value"] for option in card["options"]] == ["结论优先", "证据优先"]

    response = HumanMessage(
        content="For your clarification, my answer is: 结论优先",
        additional_kwargs={
            "hide_from_ui": True,
            "human_input_response": {
                "version": 1,
                "kind": "human_input_response",
                "source": "stackplanner",
                "request_id": card["request_id"],
                "response_kind": "option",
                "option_id": "option-1",
                "value": "结论优先",
            },
        },
    )
    resume_state = {
        **command.update,
        "messages": [tool_message, response],
    }
    feedback_update = SPThinkLabelMiddleware().before_agent(
        resume_state,
        SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-2"}),
    )

    stack = TaskMemoryStack.from_dict(feedback_update["sp_task_memory"])
    feedback = stack.entries[-1]
    assert feedback.action == "feedback"
    assert feedback.content == "结论优先"
    assert feedback.priority == "critical"
    assert feedback.status == "pinned"
    assert feedback_update["sp_pending_human_interaction"] is None


def test_sp_control_middleware_runs_central_memory_revision_in_place():
    stack = TaskMemoryStack()
    wrong_entry = stack.append_observe("The unofficial result is authoritative.", actor="deerflow")
    middleware = SPControlActionMiddleware()
    request = SimpleNamespace(
        tool_call={
            "name": "sp_revise",
            "id": "tool-revise",
            "args": {
                "target_entry_ids": [wrong_entry.id],
                "correction": "The result requires verification against an official source.",
                "reason": "The source was unofficial.",
            },
        },
        state={"sp_task_memory": stack.to_dict()},
        runtime=SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-1"}),
    )

    command = middleware.wrap_tool_call(request, lambda _: (_ for _ in ()).throw(AssertionError("native tool handler must not run")))

    entries = command.update["sp_task_memory"]["entries"]
    assert entries[0]["status"] == "superseded"
    assert entries[-1]["action"] == "revise"
    assert entries[-1]["parent_ids"] == [wrong_entry.id]
    assert command.update["sp_last_handler_result"]["action_type"] == "REVISE"


def test_think_label_middleware_marks_normal_model_turns():
    middleware = SPThinkLabelMiddleware()
    update = middleware.after_model(
        {"messages": [AIMessage(content="I will answer directly", id="ai-1")]},
        SimpleNamespace(context={"run_id": "run-1"}),
    )

    assert update["sp_last_handler_result"] == {
        "action_type": "THINK",
        "action_id": "ai-1",
        "next_step": "continue",
    }
    entry = update["sp_task_memory"]["entries"][-1]
    assert entry["action"] == "think"
    assert entry["actor"] == "central"
    assert entry["content"] == "I will answer directly"
    assert entry["metadata"] == {
        "action_type": "THINK",
        "implicit": True,
        "control_tag": "<think>",
        "model_message_id": "ai-1",
    }


def test_think_label_middleware_enforces_explicit_strict_json_and_sort_order():
    middleware = SPThinkLabelMiddleware()
    response = """```json
[
  {"country": "China", "iso3": "CHN", "value": 2},
  {"country": "India", "iso3": "IND", "value": 3},
  {"country": "Brazil", "iso3": "BRA", "value": 1}
]
```"""
    update = middleware.after_model(
        {
            "messages": [
                HumanMessage(content=("输出严格 JSON 数组，不要解释，并按 iso3 升序排列。")),
                AIMessage(content=response, id="ai-strict-json"),
            ]
        },
        SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-1"}),
    )

    normalized = update["messages"][0]
    assert normalized.id == "ai-strict-json"
    assert not normalized.content.startswith("```")
    assert [item["iso3"] for item in json.loads(normalized.content)] == ["BRA", "CHN", "IND"]
    assert normalized.additional_kwargs["stackplanner_strict_json_normalized"] is True
    assert "```" not in update["sp_task_memory"]["entries"][-1]["content"]


def test_think_label_middleware_does_not_rewrite_json_without_strict_request():
    middleware = SPThinkLabelMiddleware()
    response = '```json\n{"status":"ok"}\n```'

    update = middleware.after_model(
        {
            "messages": [
                HumanMessage(content="请给我一个 JSON 示例。"),
                AIMessage(content=response, id="ai-normal-json"),
            ]
        },
        SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-1"}),
    )

    assert "messages" not in update
    assert update["sp_task_memory"]["entries"][-1]["content"] == ('```json {"status":"ok"} ```')


def test_think_label_middleware_deduplicates_same_model_message():
    middleware = SPThinkLabelMiddleware()
    runtime = SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-1"})
    message = AIMessage(content="Continue reasoning", id="ai-dedup")
    first = middleware.after_model({"messages": [message]}, runtime)
    second = middleware.after_model(
        {"messages": [message], "sp_task_memory": first["sp_task_memory"]},
        runtime,
    )

    assert "sp_task_memory" not in second
    entries = first["sp_task_memory"]["entries"]
    assert len(entries) == 1
    assert entries[0]["metadata"]["model_message_id"] == "ai-dedup"


def test_think_label_middleware_persists_useful_native_tool_observation():
    middleware = SPThinkLabelMiddleware()
    tool_result = ToolMessage(
        content='{"results":[{"title":"StackPlanner","url":"https://example.com"}]}',
        tool_call_id="search-1",
        name="web_search",
    )
    request = SimpleNamespace(tool_call={"name": "web_search", "id": "search-1", "args": {"query": "StackPlanner"}})

    assert middleware.wrap_tool_call(request, lambda _: tool_result) is tool_result

    update = middleware.after_model(
        {
            "messages": [
                tool_result,
                AIMessage(content="根据搜索结果继续处理", id="ai-2"),
            ]
        },
        SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-1"}),
    )

    entries = update["sp_task_memory"]["entries"]
    entry = next(entry for entry in entries if entry["action"] == "observe")
    assert entry["action"] == "observe"
    assert entry["actor"] == "deerflow"
    assert entry["metadata"]["tool_name"] == "web_search"
    assert entries[-1]["action"] == "think"
    assert update["sp_last_handler_result"]["action_type"] == "THINK"


def test_think_label_middleware_merges_multiple_native_tool_observations():
    middleware = SPThinkLabelMiddleware()
    messages = [
        ToolMessage(content="first result", tool_call_id="search-1", name="web_search"),
        ToolMessage(content="second result", tool_call_id="search-2", name="web_search"),
        AIMessage(content="已完成搜索", id="ai-2"),
    ]

    update = middleware.after_model(
        {"messages": messages},
        SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-1"}),
    )

    entries = [entry for entry in update["sp_task_memory"]["entries"] if entry["action"] == "observe"]
    assert [entry["metadata"]["tool_call_id"] for entry in entries] == ["search-1", "search-2"]


def test_think_label_middleware_does_not_persist_file_listing_or_success_noise():
    middleware = SPThinkLabelMiddleware()
    update = middleware.after_model(
        {
            "messages": [
                ToolMessage(content="-rw-r--r-- report.md", tool_call_id="ls-1", name="ls"),
                ToolMessage(content="Successfully presented files", tool_call_id="present-1", name="present_files"),
                ToolMessage(content="# Skill documentation\nUse the chart generator.", tool_call_id="read-1", name="read_file"),
                ToolMessage(content="-rw-rw-r-- 1 jxk jxk 78993 /mnt/user-data/outputs/chart.png", tool_call_id="bash-1", name="bash"),
                AIMessage(content="已完成", id="ai-3"),
            ]
        },
        SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-1"}),
    )

    entries = update["sp_task_memory"]["entries"]
    assert len(entries) == 1
    assert entries[0]["action"] == "think"
    assert entries[0]["content"] == "已完成"


def test_think_label_middleware_deduplicates_repeated_errors():
    middleware = SPThinkLabelMiddleware()
    error = "Error: Unsafe absolute paths in command: /scripts/generate.js"
    update = middleware.after_model(
        {
            "messages": [
                ToolMessage(content=error, tool_call_id="bash-1", name="bash"),
                ToolMessage(content=error, tool_call_id="bash-2", name="bash"),
                AIMessage(content="停止重复尝试", id="ai-4"),
            ]
        },
        SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-1"}),
    )

    entries = update["sp_task_memory"]["entries"]
    errors = [entry for entry in entries if entry["action"] == "error"]
    assert len(errors) == 1
    assert errors[0]["priority"] == "high"
    assert entries[-1]["action"] == "think"


def test_think_label_middleware_does_not_readd_condensed_historical_observation():
    middleware = SPThinkLabelMiddleware()
    stack = TaskMemoryStack()
    old_tool_message = ToolMessage(content="old search result", tool_call_id="search-old", name="web_search")

    update = middleware.after_model(
        {
            "messages": [old_tool_message, AIMessage(content="continue", id="ai-before")],
            "sp_task_memory": stack.to_dict(),
        },
        SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-1"}),
    )
    source_entry = next(entry for entry in TaskMemoryStack.from_dict(update["sp_task_memory"]).entries if entry.action == "observe")
    observed_state = {
        "messages": [old_tool_message, AIMessage(content="continue", id="ai-after")],
        "sp_task_memory": TaskMemoryStack.from_dict(update["sp_task_memory"]).condense([source_entry.id], "compressed").to_dict(),
        "sp_consumed_tool_observation_ids": update["sp_consumed_tool_observation_ids"],
    }

    second_update = middleware.after_model(
        observed_state,
        SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-1"}),
    )

    entries = TaskMemoryStack.from_dict(second_update["sp_task_memory"]).entries
    assert not any(entry.action == "observe" for entry in entries)
    assert entries[-1].action == "think"


def test_think_label_middleware_ignores_pre_summary_history_without_tombstone():
    middleware = SPThinkLabelMiddleware()
    stack = TaskMemoryStack()
    stack.append_summary("compressed old result")
    old_tool_message = ToolMessage(content="old search result", tool_call_id="search-old", name="web_search")
    summary_tool_message = ToolMessage(content="summary committed", tool_call_id="summary-1", name="sp_summarize")

    update = middleware.after_model(
        {
            "messages": [old_tool_message, summary_tool_message, AIMessage(content="continue", id="ai-after")],
            "sp_task_memory": stack.to_dict(),
        },
        SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-1"}),
    )

    assert update["sp_last_handler_result"]["action_type"] == "THINK"
    entries = update["sp_task_memory"]["entries"]
    assert [entry["action"] for entry in entries] == ["summarize", "think"]


def test_think_label_middleware_supports_async_native_tools():
    middleware = SPThinkLabelMiddleware()
    request = SimpleNamespace(tool_call={"name": "web_search", "id": "search-async", "args": {}})
    tool_result = ToolMessage(content="async result", tool_call_id="search-async", name="web_search")

    async def run():
        async def handler(_):
            return tool_result

        return await middleware.awrap_tool_call(request, handler)

    assert asyncio.run(run()) is tool_result


def test_think_label_middleware_stops_repeated_failed_web_searches():
    middleware = SPThinkLabelMiddleware()
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return ToolMessage(
            content='{"error":"No results found"}',
            tool_call_id=request.tool_call["id"],
            name="web_search",
        )

    def request(index):
        return SimpleNamespace(
            tool_call={"name": "web_search", "id": f"search-{index}", "args": {"query": "长沙美食"}},
            runtime=SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-search"}),
        )

    first = middleware.wrap_tool_call(request(1), handler)
    second = middleware.wrap_tool_call(request(2), handler)
    third = middleware.wrap_tool_call(request(3), handler)

    assert calls == 2
    assert "No results found" in first.content
    assert "No results found" in second.content
    assert "WEB_SEARCH_ATTEMPTS_EXHAUSTED" in third.content
