"""Tests for the SP -> DR2 SubagentExecutor adapter."""

import json
from dataclasses import dataclass

from deerflow.sp.subagents import DR2SubagentExecutorAdapter, SPSubagentStatus, SPSubagentTask, render_sp_subagent_prompt


@dataclass
class FakeDR2Result:
    status: str
    result: str | None = None
    error: str | None = None
    stop_reason: str | None = None
    task_id: str | None = None
    token_usage_records: list[dict] | None = None
    ai_messages: list[dict] | None = None


class FakeDR2Executor:
    def __init__(self, result: FakeDR2Result):
        self.result = result
        self.prompts: list[str] = []

    def execute(self, task: str) -> FakeDR2Result:
        self.prompts.append(task)
        return self.result


class SequenceDR2Executor:
    def __init__(self, results: list[FakeDR2Result]):
        self.results = list(results)
        self.prompts: list[str] = []

    def execute(self, task: str) -> FakeDR2Result:
        self.prompts.append(task)
        return self.results.pop(0)


def _sp_task() -> SPSubagentTask:
    return SPSubagentTask(
        action_id="act-1",
        subagent_type="researcher",
        task="Research migration risks",
        description="Need evidence",
        input_refs=["artifact://outline"],
        expected_output="short risk summary",
        context_refs={"task_memory": {"entries": [{"action": "think", "content": "plan"}]}},
        thread_id="thread-1",
        run_id="run-1",
        metadata={"stage": "research"},
    )


def test_render_sp_subagent_prompt_carries_structured_context():
    prompt = render_sp_subagent_prompt(_sp_task())

    assert "<sp-subagent-task>" in prompt
    assert '"action_id": "act-1"' in prompt
    assert '"subagent_type": "researcher"' in prompt
    assert '"artifact://outline"' in prompt
    assert '"task_memory"' in prompt


def test_dr2_subagent_executor_adapter_normalizes_success_result():
    executor = FakeDR2Executor(FakeDR2Result(status="completed", result="done", task_id="task-1"))
    adapter = DR2SubagentExecutorAdapter(lambda task: executor)

    result = adapter.execute(_sp_task())

    assert result.status == SPSubagentStatus.COMPLETED
    assert result.result == "done"
    assert result.task_id == "task-1"
    assert len(executor.prompts) == 1
    assert "Research migration risks" in executor.prompts[0]


def test_dr2_subagent_executor_adapter_maps_unknown_status_to_failed():
    executor = FakeDR2Executor(FakeDR2Result(status="unexpected", error="bad status", task_id="task-bad"))
    adapter = DR2SubagentExecutorAdapter(lambda task: executor)

    result = adapter.execute(_sp_task())

    assert result.status == SPSubagentStatus.FAILED
    assert result.error == "bad status"
    assert result.task_id == "task-bad"


def test_adapter_parses_structured_artifact_contract_and_usage():
    executor = FakeDR2Executor(
        FakeDR2Result(
            status="completed",
            result="""```json
{"summary":"Verified three migration risks.","artifact_content":"# Evidence\\n\\nFull research body","artifact_type":"research_observation","artifact_metadata":{"sources":3}}
```""",
            task_id="task-structured",
            token_usage_records=[{"input_tokens": 10, "output_tokens": 5}],
        )
    )

    result = DR2SubagentExecutorAdapter(lambda task: executor).execute(_sp_task())

    assert result.result == "Verified three migration risks."
    assert result.artifact_content == "# Evidence\n\nFull research body"
    assert result.artifact_type == "research_observation"
    assert result.artifact_metadata == {
        "sources": 3,
        "completion_status": "complete",
        "evidence_gaps": [],
    }
    assert result.token_usage_records == [{"input_tokens": 10, "output_tokens": 5}]


def test_adapter_parses_structured_artifact_after_natural_language_preamble():
    executor = FakeDR2Executor(
        FakeDR2Result(
            status="completed",
            result=('我已经拿到了所有数据，现在生成报告。\n\n{"summary":"报告已完成","artifact_content":"# 干净的 Markdown 报告\\n\\n正文","artifact_type":"report_revision","artifact_metadata":{}}'),
            task_id="task-prefaced-json",
        )
    )

    result = DR2SubagentExecutorAdapter(lambda task: executor).execute(_sp_task())

    assert result.result == "报告已完成"
    assert result.artifact_content == "# 干净的 Markdown 报告\n\n正文"
    assert result.artifact_type == "report_revision"


def test_adapter_cleans_fenced_markdown_and_nested_report_contract():
    nested = '{"summary":"inner","artifact_content":"```markdown\\n# Clean report\\n\\nBody\\n```","artifact_type":"report_revision","artifact_metadata":{}}'
    executor = FakeDR2Executor(
        FakeDR2Result(
            status="completed",
            result=json.dumps(
                {
                    "summary": "outer",
                    "artifact_content": nested,
                    "artifact_type": "report_revision",
                    "artifact_metadata": {},
                }
            ),
            task_id="task-nested-report",
        )
    )
    task = _sp_task()
    task.subagent_type = "reporter"

    result = DR2SubagentExecutorAdapter(lambda _: executor).execute(task)

    assert result.artifact_content == "# Clean report\n\nBody"


def test_adapter_recovers_created_output_when_final_json_has_unescaped_quotes():
    executor = FakeDR2Executor(
        FakeDR2Result(
            status="completed",
            result=('The webpage is finished.\n{"summary":"采用"Neural Noir"风格","artifact_content":null,"artifact_type":"generated_file","artifact_metadata":{"created_paths":["/mnt/user-data/outputs/index.html"]}}'),
            task_id="task-malformed-json",
        )
    )
    task = _sp_task()
    task.subagent_type = "coder"

    result = DR2SubagentExecutorAdapter(lambda _: executor).execute(task)

    assert result.artifact_content is None
    assert result.artifact_metadata["created_paths"] == ["/mnt/user-data/outputs/index.html"]


def test_adapter_recovers_output_path_from_file_tool_calls_when_result_omits_it():
    executor = FakeDR2Executor(
        FakeDR2Result(
            status="completed",
            result="Webpage created and validated.",
            task_id="task-tool-path",
            ai_messages=[
                {
                    "type": "ai",
                    "tool_calls": [
                        {
                            "name": "write_file",
                            "args": '{"path":"/mnt/user-data/outputs/site.html","content":"<html></html>"}',
                        },
                        {
                            "name": "write_file",
                            "args": {"path": "/mnt/user-data/workspace/internal.txt", "content": "private"},
                        },
                    ],
                }
            ],
        )
    )
    task = _sp_task()
    task.subagent_type = "coder"

    result = DR2SubagentExecutorAdapter(lambda _: executor).execute(task)

    assert result.artifact_metadata["created_paths"] == ["/mnt/user-data/outputs/site.html"]


def test_adapter_externalizes_unstructured_large_result_as_defensive_fallback():
    large_result = "research evidence " * 200
    executor = FakeDR2Executor(FakeDR2Result(status="completed", result=large_result, task_id="task-large"))

    result = DR2SubagentExecutorAdapter(lambda task: executor).execute(_sp_task())

    assert result.artifact_content == large_result
    assert result.artifact_type == "research_observation"
    assert result.result.endswith("...<truncated>")
    assert len(result.result) == 700


def test_adapter_preserves_timeout_status_and_observes_raw_result():
    raw = FakeDR2Result(status="timed_out", error="deadline", task_id="task-timeout")
    executor = FakeDR2Executor(raw)
    observed = []

    result = DR2SubagentExecutorAdapter(lambda task: executor, result_observer=observed.append).execute(_sp_task())

    assert result.status == SPSubagentStatus.TIMED_OUT
    assert result.error == "deadline"
    assert observed == [raw]


def test_adapter_marks_guard_capped_success_as_partial_with_an_evidence_gap():
    executor = FakeDR2Executor(
        FakeDR2Result(
            status="completed",
            result='{"summary":"Recovered partial analysis.","artifact_content":null,"artifact_type":"research_observation","artifact_metadata":{"completion_status":"complete"}}',
            stop_reason="token_capped",
            task_id="task-capped",
        )
    )

    result = DR2SubagentExecutorAdapter(lambda task: executor).execute(_sp_task())

    assert result.status == SPSubagentStatus.COMPLETED
    assert result.stop_reason == "token_capped"
    assert result.artifact_metadata["completion_status"] == "partial"
    assert result.artifact_metadata["evidence_gaps"] == ["Subagent execution ended early: token_capped"]


def test_researcher_recovers_successful_fetch_evidence_when_contract_omits_artifact():
    executor = FakeDR2Executor(
        FakeDR2Result(
            status="completed",
            result=json.dumps(
                {
                    "summary": "Fetched China; India and Brazil remain.",
                    "artifact_content": None,
                    "artifact_type": "research_observation",
                    "artifact_metadata": {
                        "completion_status": "partial",
                        "evidence_gaps": ["India and Brazil were not fetched."],
                    },
                }
            ),
            task_id="task-partial-fetch",
            ai_messages=[
                {
                    "type": "ai",
                    "tool_calls": [
                        {
                            "id": "fetch-china",
                            "name": "web_fetch",
                            "args": {"url": "https://api.worldbank.org/v2/country/CHN/indicator/NY.GDP.PCAP.CD?date=2023&format=json"},
                        }
                    ],
                },
                {
                    "type": "tool",
                    "name": "web_fetch",
                    "tool_call_id": "fetch-china",
                    "content": '[{"countryiso3code":"CHN","date":"2023","value":12951.1782397043}]',
                },
            ],
        )
    )

    result = DR2SubagentExecutorAdapter(lambda _: executor).execute(_sp_task())

    assert result.artifact_metadata["completion_status"] == "partial"
    assert result.artifact_metadata["recovered_tool_evidence"] is True
    assert result.artifact_content is not None
    assert "Untrusted source data" in result.artifact_content
    assert '"countryiso3code":"CHN"' in result.artifact_content
    assert "12951.1782397043" in result.artifact_content
    assert len(executor.prompts) == 2


def test_researcher_recovers_search_evidence_for_search_only_runtime():
    executor = FakeDR2Executor(
        FakeDR2Result(
            status="completed",
            result=json.dumps(
                {
                    "summary": "Found the capital but omitted its URL.",
                    "artifact_content": None,
                    "artifact_type": "research_observation",
                    "artifact_metadata": {
                        "completion_status": "partial",
                        "evidence_gaps": ["Source URL missing from summary."],
                    },
                }
            ),
            task_id="task-search-only",
            ai_messages=[
                {
                    "type": "ai",
                    "tool_calls": [
                        {
                            "id": "search-france",
                            "name": "web_search",
                            "args": {"query": "capital of France"},
                        }
                    ],
                },
                {
                    "type": "tool",
                    "name": "web_search",
                    "tool_call_id": "search-france",
                    "content": "Paris - Wikipedia\nhttps://en.wikipedia.org/wiki/Paris",
                },
            ],
        )
    )

    result = DR2SubagentExecutorAdapter(lambda _: executor).execute(_sp_task())

    assert result.artifact_metadata["recovered_tool_evidence"] is True
    assert result.artifact_content is not None
    assert "Evidence 1 (`web_search`)" in result.artifact_content
    assert "https://en.wikipedia.org/wiki/Paris" in result.artifact_content
    assert len(executor.prompts) == 2


def test_researcher_does_not_recover_failed_fetch_as_evidence():
    executor = FakeDR2Executor(
        FakeDR2Result(
            status="completed",
            result='{"summary":"Fetch failed.","artifact_content":null,"artifact_type":"research_observation","artifact_metadata":{"completion_status":"blocked","evidence_gaps":["API unavailable."]}}',
            task_id="task-failed-fetch",
            ai_messages=[
                {
                    "type": "tool",
                    "name": "web_fetch",
                    "tool_call_id": "fetch-failed",
                    "content": "Error: upstream timeout",
                }
            ],
        )
    )

    result = DR2SubagentExecutorAdapter(lambda _: executor).execute(_sp_task())

    assert result.artifact_content is None
    assert "recovered_tool_evidence" not in result.artifact_metadata


def test_researcher_automatically_continues_bounded_multi_item_fetches():
    def pass_result(
        *,
        country: str,
        value: float,
        status: str,
        gaps: list[str],
    ) -> FakeDR2Result:
        return FakeDR2Result(
            status="completed",
            result=json.dumps(
                {
                    "summary": f"Fetched {country}={value}.",
                    "artifact_content": json.dumps({"country": country, "value": value}),
                    "artifact_type": "research_observation",
                    "artifact_metadata": {
                        "completion_status": status,
                        "evidence_gaps": gaps,
                    },
                }
            ),
            task_id=f"task-{country.lower()}",
            ai_messages=[
                {
                    "type": "tool",
                    "name": "web_fetch",
                    "tool_call_id": f"fetch-{country.lower()}",
                    "content": json.dumps({"country": country, "value": value}),
                }
            ],
        )

    executor = SequenceDR2Executor(
        [
            pass_result(
                country="CHN",
                value=12951.1782397043,
                status="partial",
                gaps=["IND and BRA remain."],
            ),
            pass_result(
                country="IND",
                value=2434.45111237626,
                status="partial",
                gaps=["BRA remains."],
            ),
            pass_result(
                country="BRA",
                value=10377.5892792557,
                status="complete",
                gaps=[],
            ),
        ]
    )
    task = _sp_task()
    task.task = "Fetch CHN, IND, and BRA from the official API."
    task.expected_output = "All three exact values."

    result = DR2SubagentExecutorAdapter(lambda _: executor).execute(task)

    assert len(executor.prompts) == 3
    assert "resolve only the remaining items" in executor.prompts[1]
    assert "IND and BRA remain." in executor.prompts[1]
    assert "BRA remains." in executor.prompts[2]
    assert result.artifact_metadata["completion_status"] == "complete"
    assert result.artifact_metadata["research_continuation_attempts"] == 2
    assert result.artifact_metadata["research_pass_count"] == 3
    assert "2434.45111237626" in result.artifact_metadata["central_evidence_preview"]
    assert result.artifact_content is not None
    assert all(expected in result.artifact_content for expected in ("12951.1782397043", "2434.45111237626", "10377.5892792557"))
    assert result.artifact_content.index("10377.5892792557") < (result.artifact_content.index("12951.1782397043"))


def test_coder_complete_claim_is_downgraded_without_successful_execution_evidence():
    executor = FakeDR2Executor(
        FakeDR2Result(
            status="completed",
            result=json.dumps(
                {
                    "summary": "Script written and tests passed.",
                    "artifact_content": "print('ok')\n",
                    "artifact_type": "generated_file",
                    "artifact_metadata": {
                        "completion_status": "complete",
                        "evidence_gaps": [],
                    },
                }
            ),
            task_id="task-unverified-code",
            ai_messages=[],
        )
    )
    task = _sp_task()
    task.subagent_type = "coder"
    task.task = "Write a Python script and actually run its tests."
    task.expected_output = "A tested downloadable Python file."

    result = DR2SubagentExecutorAdapter(lambda _: executor).execute(task)

    assert result.artifact_metadata["completion_status"] == "partial"
    assert result.artifact_metadata["execution_verification"] == {
        "required": True,
        "passed": False,
        "successful_execution_count": 0,
        "latest_source_write_index": None,
        "latest_successful_execution_index": None,
    }
    assert result.artifact_metadata["evidence_gaps"] == ["Coder claimed execution or test verification without a successful execution-tool result after the latest source change."]


def test_coder_complete_claim_accepts_successful_execution_after_source_write():
    executor = FakeDR2Executor(
        FakeDR2Result(
            status="completed",
            result=json.dumps(
                {
                    "summary": "Script written and tests passed.",
                    "artifact_content": None,
                    "artifact_type": "generated_file",
                    "artifact_metadata": {
                        "completion_status": "complete",
                        "evidence_gaps": [],
                        "created_paths": ["/mnt/user-data/outputs/verified.py"],
                    },
                }
            ),
            task_id="task-verified-code",
            ai_messages=[
                {
                    "type": "ai",
                    "tool_calls": [
                        {
                            "id": "write-1",
                            "name": "write_file",
                            "args": {
                                "path": "/mnt/user-data/outputs/verified.py",
                                "content": "print('ok')\n",
                            },
                        }
                    ],
                },
                {
                    "type": "tool",
                    "name": "write_file",
                    "tool_call_id": "write-1",
                    "content": "OK",
                },
                {
                    "type": "ai",
                    "tool_calls": [
                        {
                            "id": "bash-1",
                            "name": "bash",
                            "args": {"command": "python /mnt/user-data/outputs/verified.py"},
                        }
                    ],
                },
                {
                    "type": "tool",
                    "name": "bash",
                    "tool_call_id": "bash-1",
                    "content": "ok",
                },
            ],
        )
    )
    task = _sp_task()
    task.subagent_type = "coder"
    task.task = "Write a Python script and actually run its tests."
    task.expected_output = "A tested downloadable Python file."

    result = DR2SubagentExecutorAdapter(lambda _: executor).execute(task)

    assert result.artifact_metadata["completion_status"] == "complete"
    assert result.artifact_metadata["execution_verification"]["passed"] is True
    assert result.artifact_metadata["execution_verification"]["successful_execution_count"] == 1


def test_coder_complete_claim_is_downgraded_when_execution_failed():
    executor = FakeDR2Executor(
        FakeDR2Result(
            status="completed",
            result=json.dumps(
                {
                    "summary": "Script written and tests passed.",
                    "artifact_content": "raise AssertionError()\n",
                    "artifact_type": "generated_file",
                    "artifact_metadata": {"completion_status": "complete"},
                }
            ),
            task_id="task-failed-code",
            ai_messages=[
                {
                    "type": "ai",
                    "tool_calls": [
                        {
                            "id": "bash-failed",
                            "name": "bash",
                            "args": {"command": "python broken.py"},
                        }
                    ],
                },
                {
                    "type": "tool",
                    "name": "bash",
                    "tool_call_id": "bash-failed",
                    "content": "AssertionError\n\nExit Code: 1",
                },
            ],
        )
    )
    task = _sp_task()
    task.subagent_type = "coder"
    task.task = "Run the Python tests and return the verified file."
    task.expected_output = "A tested Python file."

    result = DR2SubagentExecutorAdapter(lambda _: executor).execute(task)

    assert result.artifact_metadata["completion_status"] == "partial"
    assert result.artifact_metadata["execution_verification"]["passed"] is False


def test_memory_recaller_keeps_full_recall_json_for_normalization():
    recall_json = """```json
{"summary":"Use phased tests.","items":[{"content":"Test each unit.","memory_id":"mem-1"}]}
```"""
    executor = FakeDR2Executor(FakeDR2Result(status="completed", result=recall_json, task_id="memory-task"))
    task = _sp_task()
    task.subagent_type = "memory_recaller"

    result = DR2SubagentExecutorAdapter(lambda _: executor).execute(task)

    assert result.result == '{"summary":"Use phased tests.","items":[{"content":"Test each unit.","memory_id":"mem-1"}]}'
    assert result.artifact_content is None
