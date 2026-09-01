"""Tests for DR2 SOUL, long-term memory, and Skill context in SP."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from deerflow.sp.central.runtime_context import (
    DEFAULT_SKILL_INDEX_MAX_CHARS,
    _skill_index,
    build_sp_central_runtime_context,
)
from deerflow.sp.runtime import (
    DR2SPExecutorProvider,
    _apply_task_skill_policy,
    _apply_task_tool_policy,
    _infer_task_skill_selection,
    _platform_skill_secrets_for_task,
    _ProgressReportingExecutor,
    _task_stream_id,
)
from deerflow.sp.subagents import SPSubagentTask
from deerflow.sp.subagents.adapter import SPSubagentResult, SPSubagentStatus
from deerflow.subagents.config import SubagentConfig


def _app_config():
    return SimpleNamespace(
        skills=SimpleNamespace(),
        memory=SimpleNamespace(
            enabled=True,
            injection_enabled=True,
            max_injection_tokens=600,
            token_counting="char",
            guaranteed_categories=["correction"],
            guaranteed_token_budget=100,
        ),
    )


def test_runtime_context_keeps_long_term_memory_on_demand(monkeypatch):
    profile = SimpleNamespace(
        name="sp-professor",
        model="custom-model",
        skills=["stackplanner-long-task", "deep-research"],
    )
    skills = [
        SimpleNamespace(name="deep-research", description="Verify evidence from multiple sources."),
        SimpleNamespace(name="disabled-by-profile", description="Must not be visible."),
        SimpleNamespace(name="stackplanner-long-task", description="Run the SP stage and HITL workflow."),
    ]
    monkeypatch.setattr("deerflow.sp.central.runtime_context.load_agent_config", lambda name, user_id=None: profile)
    monkeypatch.setattr("deerflow.sp.central.runtime_context.load_agent_soul", lambda name, user_id=None: "Use concise group-meeting language.")
    monkeypatch.setattr("deerflow.agents.lead_agent.prompt.get_enabled_skills_for_config", lambda app_config, user_id=None: skills)
    context = build_sp_central_runtime_context(
        _app_config(),
        agent_name="sp-professor",
        user_id="user-1",
    )

    assert context.agent_name == "sp-professor"
    assert context.agent_model == "custom-model"
    assert context.available_skill_names == frozenset({"deep-research", "stackplanner-long-task"})
    assert "Use concise group-meeting language." in context.system_prompt_section
    assert "stackplanner-long-task" in context.system_prompt_section
    assert "disabled-by-profile" not in context.system_prompt_section
    assert "Prefer Chinese reports." not in context.system_prompt_section
    assert context.decision_context == ""
    assert "Long-term memory is not injected here" in context.system_prompt_section


def test_runtime_context_honors_explicit_empty_agent_skill_whitelist(monkeypatch):
    profile = SimpleNamespace(name="no-skills", model=None, skills=[])
    monkeypatch.setattr("deerflow.sp.central.runtime_context.load_agent_config", lambda name, user_id=None: profile)
    monkeypatch.setattr("deerflow.sp.central.runtime_context.load_agent_soul", lambda name, user_id=None: "No tools in CentralAgent.")
    monkeypatch.setattr(
        "deerflow.agents.lead_agent.prompt.get_enabled_skills_for_config",
        lambda app_config, user_id=None: [SimpleNamespace(name="deep-research", description="Research")],
    )
    monkeypatch.setattr("deerflow.agents.memory.get_memory_data", lambda agent_name, user_id=None: {})
    monkeypatch.setattr("deerflow.agents.memory.format_memory_for_injection", lambda memory_data, **kwargs: "")

    context = build_sp_central_runtime_context(_app_config(), agent_name="no-skills", user_id="user-1")

    assert context.available_skill_names == frozenset()
    assert "<available_skills>" not in context.system_prompt_section


def test_skill_index_keeps_all_names_visible_by_compacting_descriptions():
    skills = [SimpleNamespace(name=f"skill-{index:02d}", description="Long capability description. " * 30) for index in range(30)]
    skills.append(SimpleNamespace(name="video-generation", description="Generate videos with MiniMax. " * 30))

    index = _skill_index(skills)

    assert len(index) <= DEFAULT_SKILL_INDEX_MAX_CHARS
    assert "video-generation" in index
    assert "skill-00" in index
    assert "skill-29" in index
    assert "skill-index-truncated" not in index


def test_delegate_skill_policy_loads_only_validated_requested_skills():
    config = SubagentConfig(name="sp-researcher", description="research", skills=[])
    task = SPSubagentTask(
        action_id="delegate-1",
        subagent_type="researcher",
        task="Research the evidence",
        description="Need verified facts",
        metadata={"skill_names": ["deep-research", "deep-research", "stackplanner-long-task"]},
    )

    selected = _apply_task_skill_policy(
        config,
        task,
        available_skill_names=frozenset({"deep-research", "stackplanner-long-task"}),
    )

    assert selected.skills == ["deep-research", "stackplanner-long-task"]
    assert config.skills == []


def test_delegate_skill_policy_rejects_disabled_or_invented_skill():
    config = SubagentConfig(name="sp-researcher", description="research", skills=[])
    task = SPSubagentTask(
        action_id="delegate-1",
        subagent_type="researcher",
        task="Research the evidence",
        description="Need verified facts",
        metadata={"skill_names": ["invented-skill"]},
    )

    with pytest.raises(ValueError, match="unavailable or disabled"):
        _apply_task_skill_policy(
            config,
            task,
            available_skill_names=frozenset({"deep-research"}),
        )


def test_delegate_skill_policy_treats_none_sentinel_as_omitted_selection():
    config = SubagentConfig(name="sp-coder", description="code", skills=[])
    task = SPSubagentTask(
        action_id="delegate-no-skill",
        subagent_type="coder",
        task="Check the arithmetic",
        description="No special skill is needed",
        metadata={"skill_names": ["none"]},
    )

    selected = _apply_task_skill_policy(
        config,
        task,
        available_skill_names=frozenset({"deep-research"}),
    )

    assert selected is config
    assert "skill_names" not in task.metadata


def test_delegate_skill_policy_drops_none_sentinel_beside_real_skill():
    config = SubagentConfig(name="sp-coder", description="code", skills=[])
    task = SPSubagentTask(
        action_id="delegate-one-skill",
        subagent_type="coder",
        task="Generate a chart",
        description="Use the chart skill",
        metadata={"skill_names": ["none", "chart-visualization"]},
    )

    selected = _apply_task_skill_policy(
        config,
        task,
        available_skill_names=frozenset({"chart-visualization"}),
    )

    assert selected.skills == ["chart-visualization"]
    assert task.metadata["skill_names"] == ["chart-visualization"]


def test_delegate_without_skill_selection_preserves_specialist_default_empty_set():
    config = SubagentConfig(name="sp-reporter", description="report", skills=[])
    task = SPSubagentTask(
        action_id="delegate-1",
        subagent_type="reporter",
        task="Write the report",
        description="Synthesize artifacts",
    )

    selected = _apply_task_skill_policy(
        config,
        task,
        available_skill_names=frozenset({"stackplanner-long-task"}),
    )

    assert selected is config
    assert selected.skills == []


def test_delegate_tool_policy_applies_validated_requested_allowlist():
    config = SubagentConfig(
        name="sp-coder",
        description="code",
        tools=None,
        disallowed_tools=["present_files"],
    )
    task = SPSubagentTask(
        action_id="delegate-tools",
        subagent_type="coder",
        task="Generate a chart",
        description="Need shell execution",
        metadata={"tool_names": ["bash", "read_file", "bash"]},
    )
    available = [SimpleNamespace(name="bash"), SimpleNamespace(name="read_file"), SimpleNamespace(name="present_files")]

    selected = _apply_task_tool_policy(config, task, available_tools=available)

    assert selected.tools == ["bash", "read_file"]
    assert config.tools is None


def test_delegate_tool_policy_ignores_output_formats_misfiled_as_tool_names():
    config = SubagentConfig(
        name="sp-reporter",
        description="report",
        tools=None,
        disallowed_tools=["web_search"],
    )
    task = SPSubagentTask(
        action_id="delegate-markdown-report",
        subagent_type="reporter",
        task="Write the final Markdown report",
        description="Need a downloadable report",
        metadata={"tool_names": ["markdown"]},
    )

    selected = _apply_task_tool_policy(
        config,
        task,
        available_tools=[
            SimpleNamespace(name="read_file"),
            SimpleNamespace(name="write_file"),
        ],
    )

    assert selected is config
    assert selected.tools is None


def test_delegate_tool_policy_maps_interpreter_aliases_to_bash():
    config = SubagentConfig(
        name="sp-coder",
        description="code",
        tools=None,
        disallowed_tools=["present_files"],
    )
    task = SPSubagentTask(
        action_id="delegate-python-script",
        subagent_type="coder",
        task="Write and run a Python script",
        description="Need Python execution",
        metadata={
            "tool_names": [
                "python",
                "read_file",
                "python3",
                "write_file",
            ]
        },
    )

    selected = _apply_task_tool_policy(
        config,
        task,
        available_tools=[
            SimpleNamespace(name="bash"),
            SimpleNamespace(name="read_file"),
            SimpleNamespace(name="write_file"),
        ],
    )

    assert selected.tools == ["bash", "read_file", "write_file"]
    assert task.metadata["tool_names"] == [
        "bash",
        "read_file",
        "write_file",
    ]


@pytest.mark.parametrize("requested", [["missing_tool"], ["present_files"]])
def test_delegate_tool_policy_rejects_unavailable_or_role_forbidden_tools(requested):
    config = SubagentConfig(
        name="sp-coder",
        description="code",
        tools=None,
        disallowed_tools=["present_files"],
    )
    task = SPSubagentTask(
        action_id="delegate-tools-invalid",
        subagent_type="coder",
        task="Run a tool",
        description="Tool policy test",
        metadata={"tool_names": requested},
    )

    with pytest.raises(ValueError, match="unavailable Tools|forbidden"):
        _apply_task_tool_policy(
            config,
            task,
            available_tools=[SimpleNamespace(name="bash"), SimpleNamespace(name="present_files")],
        )


@pytest.mark.parametrize(
    "task_text",
    [
        "帮我生成三秒的关于小狗的视频",
        "Use MiniMax to create a short puppy video",
    ],
)
def test_video_generation_skill_is_auto_selected_when_central_omits_it(task_text):
    task = SPSubagentTask(
        action_id="delegate-video",
        subagent_type="coder",
        task=task_text,
        description="Complete the user's media request",
    )

    _infer_task_skill_selection(task, available_skill_names=frozenset({"video-generation", "find-skills"}))

    assert task.metadata["skill_names"] == ["video-generation"]


def test_explicit_skill_selection_is_not_overridden_by_video_fallback():
    task = SPSubagentTask(
        action_id="delegate-video-research",
        subagent_type="researcher",
        task="Research video generation providers",
        description="Compare options",
        metadata={"skill_names": ["deep-research"]},
    )

    _infer_task_skill_selection(task, available_skill_names=frozenset({"video-generation", "deep-research"}))

    assert task.metadata["skill_names"] == ["deep-research"]


def test_video_generation_replaces_redundant_find_skills_discovery():
    task = SPSubagentTask(
        action_id="delegate-find-video-skill",
        subagent_type="researcher",
        task="Find a skill to generate a three-second puppy video",
        description="Create the requested video output",
        metadata={"skill_names": ["find-skills"]},
    )

    _infer_task_skill_selection(task, available_skill_names=frozenset({"video-generation", "find-skills"}))

    assert task.metadata["skill_names"] == ["video-generation"]


@pytest.mark.parametrize(
    "task_text",
    [
        "根据最近五天北京气温生成最高温和最低温折线图",
        "Visualize the temperature data as a line chart",
        "输出数据可视化图表和分析报告",
    ],
)
def test_chart_visualization_skill_is_auto_selected_when_central_omits_it(task_text):
    task = SPSubagentTask(
        action_id="delegate-chart",
        subagent_type="coder",
        task=task_text,
        description="Create the requested visualization",
    )

    _infer_task_skill_selection(
        task,
        available_skill_names=frozenset({"chart-visualization", "find-skills"}),
    )

    assert task.metadata["skill_names"] == ["chart-visualization"]


def test_chart_visualization_replaces_redundant_find_skills_discovery():
    task = SPSubagentTask(
        action_id="delegate-find-chart-skill",
        subagent_type="researcher",
        task="Find a skill and generate a line chart from the supplied data",
        description="Produce the requested chart",
        metadata={"skill_names": ["find-skills"]},
    )

    _infer_task_skill_selection(
        task,
        available_skill_names=frozenset({"chart-visualization", "find-skills"}),
    )

    assert task.metadata["skill_names"] == ["chart-visualization"]


def test_auto_selected_skill_is_visible_in_progress_events():
    events = []
    task = SPSubagentTask(
        action_id="delegate-video",
        subagent_type="coder",
        task="生成一个小狗视频",
        description="Return the video file",
    )

    class FakeAdapter:
        def execute(self, received_task):
            assert received_task.metadata["skill_names"] == ["video-generation"]
            return SPSubagentResult(status=SPSubagentStatus.COMPLETED, result="video.mp4")

    executor = _ProgressReportingExecutor(
        adapter=FakeAdapter(),
        writer=events.append,
        task_preparer=lambda received_task: _infer_task_skill_selection(
            received_task,
            available_skill_names=frozenset({"video-generation"}),
        ),
    )

    assert executor.execute(task).is_success
    assert events[1]["message"]["name"] == "Skill: video-generation"


def test_platform_skill_secret_requires_explicit_video_skill(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "server-secret")
    selected = SPSubagentTask(
        action_id="delegate-video",
        subagent_type="coder",
        task="Generate a video",
        description="Use the video provider",
        metadata={"skill_names": ["video-generation"]},
    )
    unselected = SPSubagentTask(
        action_id="delegate-code",
        subagent_type="coder",
        task="Write code",
        description="No video needed",
    )

    assert _platform_skill_secrets_for_task(selected) == {"MINIMAX_API_KEY": "server-secret"}
    assert _platform_skill_secrets_for_task(unselected) == {}


def test_progress_executor_streams_skill_and_internal_steps_with_parent_id():
    events = []
    task = SPSubagentTask(
        action_id="spact-tool-delegate",
        subagent_type="researcher",
        task="Research the source",
        description="Need evidence",
        metadata={
            "skill_names": ["deep-research"],
            "__sp_parent_tool_call_id": "tool-delegate",
        },
    )

    class FakeAdapter:
        def execute(self, received_task):
            assert received_task is task
            events.append(
                {
                    "type": "task_running",
                    "task_id": _task_stream_id(received_task),
                    "message": {"type": "tool", "name": "web_search", "content": "done"},
                    "message_index": 2,
                }
            )
            return SPSubagentResult(status=SPSubagentStatus.COMPLETED, result="Verified", task_id="random-inner-id")

    executor = _ProgressReportingExecutor(adapter=FakeAdapter(), writer=events.append)
    result = executor.execute(task)

    assert result.is_success
    assert [event["type"] for event in events] == [
        "task_started",
        "task_running",
        "task_running",
        "task_completed",
    ]
    assert {event["task_id"] for event in events} == {"tool-delegate"}
    assert events[1]["message"]["name"] == "Skill: deep-research"
    assert events[2]["message"]["name"] == "web_search"


@pytest.mark.parametrize(
    ("status", "expected_event"),
    [
        (SPSubagentStatus.CANCELLED, "task_cancelled"),
        (SPSubagentStatus.TIMED_OUT, "task_timed_out"),
    ],
)
def test_progress_executor_preserves_terminal_stop_reason(status, expected_event):
    events = []
    task = SPSubagentTask(
        action_id="delegate-stopped",
        subagent_type="researcher",
        task="Research until stopped",
        description="Need evidence",
    )

    class FakeAdapter:
        def execute(self, received_task):
            assert received_task is task
            return SPSubagentResult(
                status=status,
                error="stopped",
                task_id="inner-task",
            )

    result = _ProgressReportingExecutor(adapter=FakeAdapter(), writer=events.append).execute(task)

    assert result.status == status
    assert events[-1]["type"] == expected_event
    assert events[-1]["task_id"] == task.action_id


def test_sp_executor_provider_falls_back_to_factory_user_and_memory_scope(monkeypatch):
    captured = {}
    usage_batches = []
    abort_event = threading.Event()
    config = SubagentConfig(
        name="sp-memory-recaller",
        description="recall",
        skills=[],
        internal=True,
    )

    class FakeExecutor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def execute(self, prompt):
            return SimpleNamespace(
                status="completed",
                result='{"summary":"No matching memory.","items":[]}',
                error=None,
                stop_reason=None,
                task_id="memory-task",
                token_usage_records=[],
            )

    monkeypatch.setattr("deerflow.subagents.get_subagent_config", lambda name, app_config=None: config)
    monkeypatch.setattr("deerflow.subagents.SubagentExecutor", FakeExecutor)
    provider = DR2SPExecutorProvider(
        app_config=SimpleNamespace(),
        parent_model="test-model",
        runnable_config={},
        available_skill_names=frozenset(),
        memory_agent_name="sp-professor",
        user_id="factory-user",
    )
    provider._tools = []
    executor = provider(
        {},
        SimpleNamespace(
            context={
                "thread_id": "thread-1",
                "run_id": "run-1",
                "__run_journal": SimpleNamespace(
                    record_external_llm_usage_records=usage_batches.append,
                ),
                "__run_abort_event": abort_event,
            },
            stream_writer=None,
        ),
    )

    result = executor.execute(
        SPSubagentTask(
            action_id="recall-1",
            subagent_type="memory_recaller",
            task="Recall preferences",
            description="Need prior preferences",
        )
    )

    assert result.is_success
    assert captured["user_id"] == "factory-user"
    assert captured["memory_agent_name"] == "sp-professor"
    assert captured["user_scoped_skills"] is True
    assert captured["force_isolated_loop"] is True
    assert captured["parent_abort_event"] is abort_event
    captured["token_usage_observer"](
        {
            "source_run_id": "usage-1",
            "caller": "subagent:sp-memory-recaller",
            "total_tokens": 10,
        }
    )
    assert usage_batches == [
        [
            {
                "source_run_id": "usage-1",
                "caller": "subagent:sp-memory-recaller",
                "total_tokens": 10,
            }
        ]
    ]


def test_sp_executor_provider_lazily_injects_isolated_model_per_delegation(
    monkeypatch,
):
    captured_executor = {}
    factory_calls = []
    injected_model = object()
    subagent_config = SubagentConfig(
        name="sp-researcher",
        description="research",
        model="inherit",
        internal=True,
    )

    class FakeExecutor:
        def __init__(self, **kwargs):
            captured_executor.update(kwargs)

        def execute(self, _prompt):
            return SimpleNamespace(
                status="completed",
                result='{"summary":"done"}',
                error=None,
                stop_reason=None,
                task_id="task-result",
                token_usage_records=[],
            )

    def training_model_factory(**kwargs):
        factory_calls.append(kwargs)
        return injected_model

    monkeypatch.setattr(
        "deerflow.subagents.get_subagent_config",
        lambda name, app_config=None: subagent_config,
    )
    monkeypatch.setattr("deerflow.subagents.SubagentExecutor", FakeExecutor)
    app_config = SimpleNamespace()
    provider = DR2SPExecutorProvider(
        app_config=app_config,
        parent_model="actor-model",
        runnable_config={},
        training_model_factory=training_model_factory,
        episode_id="episode-3",
    )
    provider._tools = []
    executor = provider(
        {},
        SimpleNamespace(
            context={"thread_id": "thread-3", "run_id": "run-3"},
            stream_writer=None,
        ),
    )

    result = executor.execute(
        SPSubagentTask(
            action_id="delegate-3",
            subagent_type="researcher",
            task="Research evidence",
            description="Need evidence",
            metadata={"automatic_research_continuation_attempt": 2},
        )
    )

    assert result.is_success
    assert captured_executor["model_factory"] is not None
    assert factory_calls == []
    assert captured_executor["model_factory"]() is injected_model
    assert factory_calls == [
        {
            "role": "researcher",
            "model_name": "actor-model",
            "episode_id": "episode-3",
            "thread_id": "thread-3",
            "run_id": "run-3",
            "action_id": "delegate-3",
            "task_id": "delegate-3",
            "continuation_attempt": 2,
            "app_config": app_config,
        }
    ]


def test_sp_executor_provider_initializes_shared_tools_once_under_parallel_load(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    call_count = 0
    expected = [SimpleNamespace(name="read_file")]

    def load_tools(**_kwargs):
        nonlocal call_count
        call_count += 1
        entered.set()
        assert release.wait(timeout=1)
        return expected

    monkeypatch.setattr("deerflow.tools.get_available_tools", load_tools)
    provider = DR2SPExecutorProvider(
        app_config=SimpleNamespace(),
        parent_model="test-model",
        runnable_config={},
    )

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(provider._available_tools) for _ in range(8)]
        assert entered.wait(timeout=1)
        release.set()
        results = [future.result(timeout=1) for future in futures]

    assert call_count == 1
    assert all(result is expected for result in results)
