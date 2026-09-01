"""Registration and isolation tests for SP-only DR2 subagents."""

from deerflow.subagents.builtins import BUILTIN_SUBAGENTS, SP_SPECIALIST_CONFIGS, SP_SPECIALIST_REGISTRY_NAMES
from deerflow.subagents.registry import get_available_subagent_names, get_subagent_config, get_subagent_names


def test_all_designed_sp_specialists_are_registered():
    expected = {"researcher", "coder", "reporter", "outline", "perception"}
    expected_registry_names = {f"sp-{name}" for name in expected}

    assert set(SP_SPECIALIST_CONFIGS) == expected
    assert set(SP_SPECIALIST_REGISTRY_NAMES.values()) == expected_registry_names
    assert expected_registry_names <= set(BUILTIN_SUBAGENTS)
    assert expected_registry_names <= set(get_subagent_names())

    for role in expected:
        config = get_subagent_config(SP_SPECIALIST_REGISTRY_NAMES[role])
        assert config is not None
        assert config.name == f"sp-{role}"
        assert config.internal is True
        assert "Return one JSON object only" in config.system_prompt
        assert "Never place a full report or research dump in the summary" in config.system_prompt
        assert "mandatory_requirements" in config.system_prompt
        assert "smallest valid request" in config.system_prompt
        assert "do not alternate guessed delimiters" in config.system_prompt
        assert "task" in config.disallowed_tools
        assert "ask_clarification" in config.disallowed_tools
        if role == "reporter":
            assert config.skills == ["stackplanner-reporting"]
            assert "write_file" not in config.disallowed_tools
            assert "artifact_bodies" in config.system_prompt
            assert "quality gate" in config.system_prompt.lower()
            assert "Never expose a host filesystem path" in config.system_prompt
            assert "Never list a created_path unless" in config.system_prompt
            assert "Never reuse a prior report's path" in config.system_prompt
            assert "requires a Chinese report" in config.system_prompt
            assert "Do not invent" in config.system_prompt
            assert "Do not invent additional audit tags" in config.system_prompt
            assert "never place it in a current decision" in config.system_prompt
            assert "unknown/TBD" in config.system_prompt
            assert 'completion_status "complete"' in config.system_prompt
            assert "Recalculate every threshold" in config.system_prompt
        else:
            assert config.skills == []


def test_sp_specialists_stay_hidden_from_default_lead_agent_task_tool(monkeypatch):
    from deerflow.subagents import registry as registry_module

    monkeypatch.setattr(registry_module, "is_host_bash_allowed", lambda: True)

    assert get_available_subagent_names() == ["general-purpose", "bash"]


def test_sp_specialists_have_role_specific_artifact_contracts():
    expected_types = {
        "researcher": "research_observation",
        "coder": "generated_file",
        "reporter": "report_revision",
        "outline": "outline",
        "perception": "perception_observation",
    }

    for name, artifact_type in expected_types.items():
        assert f'Use artifact_type "{artifact_type}"' in SP_SPECIALIST_CONFIGS[name].system_prompt


def test_sp_stage_specialists_preserve_stackplanner_1_workflow_strengths():
    perception = SP_SPECIALIST_CONFIGS["perception"].system_prompt
    outline = SP_SPECIALIST_CONFIGS["outline"].system_prompt
    researcher = SP_SPECIALIST_CONFIGS["researcher"].system_prompt

    assert "executable task brief" in perception
    assert "1-5 short, orthogonal" in perception
    assert "task_ready" in perception
    assert "Never assume" in perception
    assert "dependency-aware execution/research DAG" in outline
    assert "observable acceptance criteria" in outline
    assert "minimum set" in researcher
    assert "requirement-to-evidence coverage map" in researcher
    assert "`web_fetch` on the official endpoint before `web_search`" in researcher
    assert "single-entity official requests" in researcher
    assert "valid stopping point while other bounded requests" in researcher
    assert "both the compact summary and artifact_content" in researcher
    assert "never return `artifact_content=null` after a successful fetch" in researcher


def test_sp_coder_requires_file_tools_for_source_and_separate_execution():
    coder = SP_SPECIALIST_CONFIGS["coder"].system_prompt

    assert "Use `write_file`" in coder
    assert "Never redirect program stdout into the source file" in coder
    assert "separate tool call" in coder
    assert "assertions or exact comparisons" in coder
    assert "Printing an example output is not verification" in coder
    assert "Do not use factorial full permutations" in coder
    assert "Do not search the web for a self-contained" in coder
