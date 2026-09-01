"""Tests for the built-in StackPlanner memory_recaller subagent."""

from deerflow.subagents.builtins import BUILTIN_SUBAGENTS, MEMORY_RECALLER_CONFIG
from deerflow.subagents.registry import get_subagent_config, get_subagent_names


def test_memory_recaller_is_registered_as_builtin_subagent():
    assert BUILTIN_SUBAGENTS["sp-memory-recaller"] is MEMORY_RECALLER_CONFIG
    assert "sp-memory-recaller" in get_subagent_names()

    config = get_subagent_config("sp-memory-recaller")

    assert config is not None
    assert config.name == "sp-memory-recaller"
    assert config.max_turns == 20
    assert config.internal is True
    assert config.skills == []


def test_memory_recaller_is_read_only_and_cannot_delegate_or_write_files():
    config = get_subagent_config("sp-memory-recaller")
    assert config is not None

    assert "task" in config.disallowed_tools
    assert "bash" in config.disallowed_tools
    assert "write_file" in config.disallowed_tools
    assert "Do not write, update, promote, delete, or mutate long-term memory." in config.system_prompt
    assert "Return compact JSON only" in config.system_prompt


def test_memory_recaller_runtime_chain_includes_read_only_dynamic_context():
    from deerflow.agents.middlewares.dynamic_context_middleware import DynamicContextMiddleware
    from deerflow.agents.middlewares.memory_middleware import MemoryMiddleware
    from deerflow.agents.middlewares.tool_error_handling_middleware import build_subagent_runtime_middlewares
    from deerflow.config.app_config import AppConfig

    app_config = AppConfig.model_validate({"sandbox": {"use": "deerflow.sandbox.local:LocalSandboxProvider"}})
    middlewares = build_subagent_runtime_middlewares(
        app_config=app_config,
        agent_name="sp-memory-recaller",
        memory_agent_name="sp-professor",
    )

    dynamic_context = [middleware for middleware in middlewares if isinstance(middleware, DynamicContextMiddleware)]
    assert len(dynamic_context) == 1
    assert dynamic_context[0]._agent_name == "sp-professor"
    assert not any(isinstance(middleware, MemoryMiddleware) for middleware in middlewares)


def test_other_subagents_do_not_receive_long_term_memory_implicitly():
    from deerflow.agents.middlewares.dynamic_context_middleware import DynamicContextMiddleware
    from deerflow.agents.middlewares.tool_error_handling_middleware import build_subagent_runtime_middlewares
    from deerflow.config.app_config import AppConfig

    app_config = AppConfig.model_validate({"sandbox": {"use": "deerflow.sandbox.local:LocalSandboxProvider"}})
    middlewares = build_subagent_runtime_middlewares(
        app_config=app_config,
        agent_name="sp-researcher",
    )

    assert not any(isinstance(middleware, DynamicContextMiddleware) for middleware in middlewares)
