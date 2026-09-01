"""Factories for SP CentralAgent decision components."""

from __future__ import annotations

from typing import Any

from deerflow.config.app_config import AppConfig
from deerflow.models import create_chat_model
from deerflow.sp.actions import ActionRouter, build_default_action_router
from deerflow.sp.artifacts import SPArtifactAdapter
from deerflow.sp.central.agent import CentralAgentDecider
from deerflow.sp.loop import ActionLoop
from deerflow.sp.prompt import PromptContextBuilder
from deerflow.sp.subagents import SPSubagentExecutorProtocol


def create_sp_central_decider(
    *,
    model: Any | None = None,
    model_name: str | None = None,
    app_config: AppConfig | None = None,
    thinking_enabled: bool = False,
) -> CentralAgentDecider:
    """Create an SP CentralAgent decider with no business tools attached."""
    if model is None:
        model = create_chat_model(
            name=model_name,
            thinking_enabled=thinking_enabled,
            app_config=app_config,
            attach_tracing=False,
        )
    return CentralAgentDecider(model=model)


def create_sp_action_loop(
    *,
    model: Any | None = None,
    model_name: str | None = None,
    app_config: AppConfig | None = None,
    thinking_enabled: bool = False,
    router: ActionRouter | None = None,
    delegate_executor: SPSubagentExecutorProtocol | None = None,
    memory_recall_executor: SPSubagentExecutorProtocol | None = None,
    artifact_adapter: SPArtifactAdapter | None = None,
    context_builder: PromptContextBuilder | None = None,
    max_iterations: int = 20,
) -> ActionLoop:
    """Create an SP ActionLoop backed by a tool-less CentralAgent decider."""
    decider = create_sp_central_decider(
        model=model,
        model_name=model_name,
        app_config=app_config,
        thinking_enabled=thinking_enabled,
    )
    router = router or build_default_action_router(
        delegate_executor=delegate_executor,
        memory_recall_executor=memory_recall_executor,
        artifact_adapter=artifact_adapter,
    )
    return ActionLoop(
        decider=decider,
        router=router,
        context_builder=context_builder,
        max_iterations=max_iterations,
    )
