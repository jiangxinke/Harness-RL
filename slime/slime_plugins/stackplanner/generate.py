"""slime custom-generate hook that runs the production StackPlanner graph.

The hook is loaded by slime in each rollout worker.  It starts one local
OpenAI-compatible adapter per worker, invokes DeerFlow's production
``make_sp_agent`` graph in process, computes reward from the complete episode,
and returns the actor-generated Central/subagent trajectories as a fan-out.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import logging
import os
import secrets
import threading
from collections.abc import Mapping
from typing import Any

from .config import TrainingConfig
from .model_factory import EpisodeModelFactory, make_openai_model_builder
from .rewards import EpisodeOutcome, RewardResult

logger = logging.getLogger(__name__)


def _load_function(path: str):
    module_path, separator, name = path.replace(":", ".").rpartition(".")
    if not separator:
        raise ValueError(f"function path must include a module: {path!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)


def _load_training_config() -> TrainingConfig:
    path = os.environ.get("SP_TRAINING_RESOLVED_CONFIG")
    if not path:
        raise RuntimeError("SP_TRAINING_RESOLVED_CONFIG is required")
    return TrainingConfig.load(path)


def _episode_id(sample: Any) -> str:
    if getattr(sample, "session_id", None):
        suffix = str(sample.session_id)
    elif getattr(sample, "index", None) is not None:
        suffix = f"{sample.index}-{getattr(sample, 'group_index', 0)}"
    else:
        suffix = secrets.token_hex(12)
    run_id = os.environ.get("SP_TRAINING_RUN_ID", "run")
    return f"{run_id}.{suffix}"


def _runtime_id(kind: str, episode_id: str) -> str:
    """Return a stable StackPlanner filesystem-safe ID for one episode."""

    digest = hashlib.sha256(episode_id.encode("utf-8")).hexdigest()[:32]
    return f"{kind}-{digest}"


def _prompt_text(sample: Any, *, training_prefix: str = "") -> str:
    prompt = getattr(sample, "prompt", "")
    if isinstance(prompt, str):
        prompt_text = prompt
    elif not isinstance(prompt, list):
        prompt_text = str(prompt)
    else:
        parts: list[str] = []
        for message in prompt:
            if not isinstance(message, Mapping):
                continue
            if str(message.get("role") or "") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                parts.extend(str(block.get("text") or "") for block in content if isinstance(block, Mapping) and block.get("text"))
        prompt_text = "\n".join(parts)
    prefix = training_prefix.strip()
    return f"{prefix}\n\n{prompt_text}" if prefix else prompt_text


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(block.get("text") or block.get("content") or "") if isinstance(block, Mapping) else str(block) for block in content)
    return str(content or "")


def _final_answer(state: Mapping[str, Any]) -> str:
    for message in reversed(list(state.get("messages") or [])):
        if getattr(message, "type", None) == "ai":
            return _message_text(message).strip()
    return ""


def _artifacts(state: Mapping[str, Any]) -> tuple[str, ...]:
    refs = state.get("sp_current_artifact_refs")
    if not isinstance(refs, Mapping):
        return ()
    values: list[str] = []
    for ref in refs.values():
        if not isinstance(ref, Mapping):
            continue
        value = ref.get("virtual_path") or ref.get("artifact_url") or ref.get("artifact_id")
        if value:
            values.append(str(value))
    return tuple(dict.fromkeys(values))


def _policy_termination(state: Mapping[str, Any]) -> str | None:
    result = state.get("sp_last_handler_result")
    if not isinstance(result, Mapping):
        return None
    termination = str(result.get("termination") or "").strip()
    if termination:
        return termination
    if result.get("next_step") == "error_fatal":
        return "policy_error"
    return None


def _exception_policy_termination(exc: BaseException) -> str | None:
    """Classify only known policy-control exceptions; unknowns stay infra."""

    class_names = {cls.__name__ for cls in type(exc).__mro__}
    if "GraphRecursionError" in class_names:
        return "graph_recursion_capped"
    if "ActionValidationError" in class_names:
        return "invalid_action"
    return None


def _rollout_id(sample: Any, episode_id: str) -> int:
    if getattr(sample, "index", None) is not None:
        return int(sample.index)
    return abs(hash(episode_id)) % (2**31)


async def _finish_policy_failure(
    factory: EpisodeModelFactory,
    *,
    base_sample: Any,
    episode_id: str,
    termination: str,
) -> list[Any]:
    samples = await factory.finish(
        base_sample=base_sample,
        rollout_id=_rollout_id(base_sample, episode_id),
        group_index=getattr(base_sample, "group_index", None),
        reward=0.0,
        reward_components={"policy_completed": 0.0},
        termination=termination,
        wait_timeout=30,
    )
    if not samples:
        raise RuntimeError(f"StackPlanner {termination} episode produced no actor trajectory")
    return samples


def _reward_result(
    config: TrainingConfig,
    *,
    episode_id: str,
    prompt: str,
    state: Mapping[str, Any],
    sample: Any,
) -> RewardResult:
    reward_function = _load_function(config.reward_function)
    metadata = getattr(sample, "metadata", None)
    verifier = dict(metadata.get("verifier") or {}) if isinstance(metadata, Mapping) else {}
    result = reward_function(
        EpisodeOutcome(
            episode_id=episode_id,
            prompt=prompt,
            final_answer=_final_answer(state),
            label=getattr(sample, "label", None),
            status="completed",
            artifacts=_artifacts(state),
            verifier=verifier,
        ),
        config.reward_config,
    )
    if not isinstance(result, RewardResult):
        raise TypeError("episode reward function must return RewardResult")
    return result


def _model_kwargs(config: TrainingConfig, *, evaluation: bool = False) -> dict[str, Any]:
    return {
        "temperature": 0.0 if evaluation else config.temperature,
        "top_p": 1.0 if evaluation else config.top_p,
        "max_tokens": config.max_response_tokens,
        "timeout": float(config.episode_timeout_seconds),
        "max_retries": 0,
        "streaming": False,
    }


def _app_config(config: TrainingConfig):
    from deerflow.config.app_config import AppConfig

    path = os.environ.get("DEER_FLOW_CONFIG_PATH")
    if not path:
        raise RuntimeError("DEER_FLOW_CONFIG_PATH is required for StackPlanner training")
    app_config = AppConfig.from_file(path)
    if not app_config.models:
        raise ValueError("StackPlanner training config must contain at least one model")
    # The injected actor/frozen models perform every policy call. The config
    # model remains the schema/tool-capability source and must be resolvable.
    return app_config


@dataclasses.dataclass(slots=True)
class _WorkerAdapterService:
    adapter: Any
    app_handle: Any


_SERVICES: dict[tuple[str, int], _WorkerAdapterService] = {}
_SERVICES_LOCK = threading.Lock()


def _adapter_service(args: Any, config: TrainingConfig) -> _WorkerAdapterService:
    key = (str(args.sglang_router_ip), int(args.sglang_router_port))
    with _SERVICES_LOCK:
        existing = _SERVICES.get(key)
        if existing is not None:
            return existing

        from slime.agent.adapters import OpenAIAdapter
        from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread
        from slime.utils.processing_utils import load_tokenizer

        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        turn_annotator = None
        if config.capability_decoupling:
            turn_annotator = _load_function(config.capability_annotator_function)
        adapter = OpenAIAdapter(
            tokenizer=tokenizer,
            sglang_url=f"http://{args.sglang_router_ip}:{args.sglang_router_port}",
            tool_parser=getattr(args, "sglang_tool_call_parser", None) or None,
            reasoning_parser=getattr(args, "sglang_reasoning_parser", None) or None,
            turn_annotator=turn_annotator,
        )
        handle = run_app_in_thread(
            adapter.app,
            host=config.adapter_host,
            port=config.adapter_port,
            thread_name="stackplanner-openai-adapter",
            runner_kwargs={
                "handler_cancellation": True,
                "access_log_class": FilteredAccessLogger,
            },
        )
        service = _WorkerAdapterService(adapter, handle)
        _SERVICES[key] = service
        return service


async def _abort_factory(factory: EpisodeModelFactory | None, *, episode_id: str) -> None:
    if factory is None:
        return
    try:
        await factory.abort(wait_timeout=30)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("StackPlanner episode %s could not abort its actor trajectories", episode_id)


def _aborted_sample(sample: Any, *, episode_id: str, reason: str) -> list[Any]:
    from slime.utils.types import Sample

    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.reward = 0.0
    sample.remove_sample = True
    sample.rollout_id = _rollout_id(sample, episode_id)
    sample.status = Sample.Status.ABORTED
    sample.metadata = {
        **(getattr(sample, "metadata", None) or {}),
        "episode_id": episode_id,
        "rollout_id": sample.rollout_id,
        "infrastructure_failure": {
            "kind": reason,
            "retryable": True,
        },
    }
    return [sample]


async def generate(
    args: Any,
    base_sample: Any,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> list[Any]:
    """Run one token-correct production StackPlanner episode for slime."""

    del sampling_params
    config = _load_training_config()
    prompt_text = _prompt_text(
        base_sample,
        training_prefix=config.training_prompt_prefix,
    )
    episode_id = _episode_id(base_sample)
    # Episode IDs retain human-readable run/sample lineage and may contain
    # dots or other characters that StackPlanner deliberately rejects in
    # filesystem-backed thread IDs. The digest preserves stable isolation.
    thread_id = _runtime_id("thread", episode_id)
    run_id = _runtime_id("run", episode_id)
    factory: EpisodeModelFactory | None = None
    search_failures: list[dict[str, str]] = []
    search_failure_token = None
    try:
        from deerflow.community.search_r1.tools import SEARCH_R1_FAILURES

        search_failure_token = SEARCH_R1_FAILURES.set(search_failures)
        service = _adapter_service(args, config)
        app_config = _app_config(config)
        from deerflow.sp.runtime import SP_SUBAGENT_REGISTRY_NAMES, make_sp_agent
        from langchain_core.messages import HumanMessage

        actor_endpoint = f"http://{config.adapter_host}:{service.app_handle.port}/v1"
        actor_builder = make_openai_model_builder(
            base_url=actor_endpoint,
            model=config.actor_model_name,
            model_kwargs=_model_kwargs(config, evaluation=evaluation),
        )
        frozen_builder = make_openai_model_builder(
            base_url=config.frozen_base_url,
            model=config.frozen_model,
            api_key_env=config.frozen_api_key_env or None,
            model_kwargs=_model_kwargs(config, evaluation=evaluation),
            recover_qwen_tool_calls=True,
        )
        factory = EpisodeModelFactory(
            config=config,
            episode_id=episode_id,
            policy_roles=set(SP_SUBAGENT_REGISTRY_NAMES),
            adapter=service.adapter,
            actor_model_builder=actor_builder,
            frozen_model_builder=frozen_builder,
            model_version=str(args.hf_checkpoint),
        )
        runtime_config = {
            "configurable": {
                "model_name": app_config.models[0].name,
                "episode_id": episode_id,
                "thread_id": thread_id,
                "run_id": run_id,
                "max_concurrent_subagents": config.max_parallel_delegates,
                "sp_max_loop_iterations": config.max_iterations,
                "sp_training_model_factory": factory,
                "sp_training_force_initial_delegate": config.training_force_initial_delegate,
            },
            "context": {
                "episode_id": episode_id,
                "thread_id": thread_id,
                "run_id": run_id,
            },
        }
        graph = make_sp_agent(runtime_config, app_config=app_config)
        state = await asyncio.wait_for(
            graph.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content=prompt_text,
                            id=f"user-{episode_id}",
                        )
                    ]
                },
                config={
                    "configurable": {
                        "thread_id": thread_id,
                    },
                    "recursion_limit": max(
                        100,
                        config.max_iterations * 8,
                    ),
                },
                context={
                    "episode_id": episode_id,
                    "thread_id": thread_id,
                    "run_id": run_id,
                },
            ),
            timeout=config.episode_timeout_seconds,
        )
        if search_failures:
            raise RuntimeError(f"search infrastructure failure: {search_failures[:3]}")
        policy_termination = _policy_termination(state)
        if policy_termination is not None:
            return await _finish_policy_failure(
                factory,
                base_sample=base_sample,
                episode_id=episode_id,
                termination=policy_termination,
            )
        reward = _reward_result(
            config,
            episode_id=episode_id,
            prompt=prompt_text,
            state=state,
            sample=base_sample,
        )
        if reward.infrastructure_failure:
            await _abort_factory(factory, episode_id=episode_id)
            return _aborted_sample(
                base_sample,
                episode_id=episode_id,
                reason=reward.detail or "verifier-infrastructure",
            )
        samples = await factory.finish(
            base_sample=base_sample,
            rollout_id=_rollout_id(base_sample, episode_id),
            group_index=getattr(base_sample, "group_index", None),
            reward=reward.value,
            reward_components=reward.components,
            wait_timeout=30,
        )
        if not samples:
            raise RuntimeError("StackPlanner episode produced no actor trajectory")
        return samples
    except asyncio.CancelledError:
        await _abort_factory(factory, episode_id=episode_id)
        raise
    except Exception as exc:
        policy_termination = _exception_policy_termination(exc)
        if factory is not None and policy_termination is not None and not search_failures:
            try:
                return await _finish_policy_failure(
                    factory,
                    base_sample=base_sample,
                    episode_id=episode_id,
                    termination=policy_termination,
                )
            except asyncio.CancelledError:
                await _abort_factory(factory, episode_id=episode_id)
                raise
            except Exception:
                logger.exception(
                    "StackPlanner episode %s could not drain a %s trajectory",
                    episode_id,
                    policy_termination,
                )
        await _abort_factory(factory, episode_id=episode_id)
        logger.exception("StackPlanner episode %s failed", episode_id)
        return _aborted_sample(
            base_sample,
            episode_id=episode_id,
            reason=f"search-infrastructure:{search_failures[:3]}" if search_failures else f"{type(exc).__name__}:{str(exc)[:160]}",
        )
    except BaseException:
        await _abort_factory(factory, episode_id=episode_id)
        raise
    finally:
        if search_failure_token is not None:
            SEARCH_R1_FAILURES.reset(search_failure_token)
