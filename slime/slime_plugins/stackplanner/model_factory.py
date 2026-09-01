"""Episode-local model routing and slime adapter session lifecycle.

The StackPlanner runtime calls :class:`EpisodeModelFactory` through its optional
``sp_training_model_factory`` dependency-injection seam.  Actor-backed roles are
registered with slime before a LangChain model can issue a request; frozen roles
never enter the adapter trajectory store.  At the episode boundary, every actor
session is drained into token-correct slime samples with common rollout lineage.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import threading
import weakref
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from .config import TrainingConfig
from .routing import Backend, RoleRoute, RoleRouter
from .sessions import SessionKey, SessionManager
from .trajectory import annotate_episode_samples


class SlimeAdapter(Protocol):
    """Subset of slime's adapter API required by one StackPlanner episode."""

    def open_session(
        self,
        sid: str,
        *,
        sampling_defaults: dict | None = None,
        max_context_tokens: int = 0,
        annotation_context: dict | None = None,
    ) -> None: ...

    async def finish_session(
        self,
        sid: str,
        *,
        base_sample: Any,
        reward: float = 0.0,
        extra_metadata: dict | None = None,
        wait_timeout: float = 5.0,
    ) -> list[Any]: ...

    async def drop_session(self, sid: str, *, wait_timeout: float = 5.0) -> None: ...


ModelBuilder = Callable[..., Any]


_LOOP_ASYNC_CLIENTS: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()
_LOOP_ASYNC_CLIENTS_LOCK = threading.Lock()


def _loop_local_async_client(*, timeout: Any) -> Any:
    """Return one explicit httpx client per running event loop."""

    import httpx

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # Builders are normally invoked from an active rollout/subagent loop.
        # Keep direct synchronous callers isolated rather than entering the
        # process-global LangChain cache.
        return httpx.AsyncClient(timeout=timeout)
    with _LOOP_ASYNC_CLIENTS_LOCK:
        client = _LOOP_ASYNC_CLIENTS.get(loop)
        if client is None or client.is_closed:
            client = httpx.AsyncClient(timeout=timeout)
            _LOOP_ASYNC_CLIENTS[loop] = client
        return client


@dataclasses.dataclass(frozen=True, slots=True)
class ActorSession:
    session_id: str
    key: SessionKey
    route: RoleRoute
    requested_model: str
    thread_id: str | None
    run_id: str | None
    parent_action_id: str | None = None


def make_openai_model_builder(
    *,
    base_url: str,
    model: str | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
    default_headers: Mapping[str, str] | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
    recover_qwen_tool_calls: bool = False,
) -> ModelBuilder:
    """Create a lazy LangChain OpenAI-compatible model builder.

    For actor routes, ``EpisodeModelFactory`` passes the collision-free slime
    session id as ``session_id``; it becomes the bearer token understood by the
    slime OpenAI adapter.  Frozen endpoints use ``api_key`` or ``api_key_env``
    and never receive an actor session id. Frozen Qwen/vLLM endpoints can opt
    into DeerFlow's textual tool-call recovery; actor routes keep the slime
    adapter's native OpenAI contract. Importing LangChain stays lazy so
    contract-only tooling can import the plugin without StackPlanner's
    optional runtime dependencies installed.
    """

    if api_key is not None and api_key_env is not None:
        raise ValueError("provide api_key or api_key_env, not both")

    def build(
        *,
        requested_model: str,
        session_id: str | None,
        role: str,
        **_: Any,
    ) -> Any:
        # Shared-all-agents runs never call the frozen fallback. Keep endpoint
        # validation at route use rather than eagerly rejecting an otherwise
        # valid actor-only episode. Central-only preflight still requires a
        # configured frozen endpoint, and an unexpected frozen route fails here.
        if not base_url:
            raise ValueError("base_url must not be empty")
        if recover_qwen_tool_calls:
            from deerflow.models.vllm_provider import VllmChatModel as ModelClass
        else:
            from langchain_openai import ChatOpenAI as ModelClass

        resolved_key = session_id or api_key
        if resolved_key is None and api_key_env:
            resolved_key = os.environ.get(api_key_env)
            if not resolved_key:
                raise RuntimeError(f"required frozen endpoint credential {api_key_env} is not set")
        # Local OpenAI-compatible endpoints generally require a non-empty SDK
        # key even when they do not authenticate.  It is never logged as policy
        # metadata and actor routes always replace it with their auditable sid.
        resolved_key = resolved_key or "EMPTY"
        kwargs = dict(model_kwargs or {})
        # LangChain caches its default async httpx client by endpoint/timeout.
        # Central and delegated models intentionally execute on different
        # event loops during StackPlanner training, so that process-global
        # cache can hand a specialist a pool first used by Central. Give every
        # event loop its own explicit client; construction stays lazy and each
        # long-lived rollout/subagent loop reuses only its own pool.
        kwargs["http_async_client"] = _loop_local_async_client(timeout=kwargs.get("timeout"))
        return ModelClass(
            model=model or requested_model,
            base_url=base_url,
            api_key=resolved_key,
            default_headers=dict(default_headers or {}),
            metadata={"stackplanner_role": role},
            **kwargs,
        )

    return build


class EpisodeModelFactory:
    """Route one StackPlanner episode and own all of its actor sessions."""

    def __init__(
        self,
        *,
        config: TrainingConfig,
        episode_id: str,
        policy_roles: list[str] | tuple[str, ...] | set[str],
        adapter: SlimeAdapter,
        actor_model_builder: ModelBuilder,
        frozen_model_builder: ModelBuilder,
        model_version: str,
    ) -> None:
        self.config = config
        self.episode_id = episode_id
        self.adapter = adapter
        self.actor_model_builder = actor_model_builder
        self.frozen_model_builder = frozen_model_builder
        self.model_version = model_version
        self.router = RoleRouter(config.mode, policy_roles, config.role_overrides)
        self.sessions = SessionManager(episode_id)
        self._actor_sessions: dict[str, ActorSession] = {}
        self._attempts: dict[tuple[str, str, str, int], int] = {}
        self._lock = threading.Lock()
        self._sealed = False

    def __call__(
        self,
        *,
        role: str,
        model_name: str,
        episode_id: str | None,
        thread_id: str | None,
        run_id: str | None,
        action_id: str | None,
        task_id: str | None,
        continuation_attempt: int,
        app_config: Any,
        parent_action_id: str | None = None,
    ) -> Any:
        del app_config  # Explicitly accepted as part of StackPlanner's seam.
        if self._sealed:
            raise RuntimeError("episode model factory is already sealed")
        if episode_id is not None and episode_id != self.episode_id:
            raise ValueError(f"factory episode {self.episode_id!r} cannot serve {episode_id!r}")

        route = self.router.route(role)
        metadata = {
            "role": role,
            "route": route,
            "episode_id": self.episode_id,
            "thread_id": thread_id,
            "run_id": run_id,
            "action_id": action_id,
            "task_id": task_id,
            "parent_action_id": parent_action_id,
            "continuation_attempt": int(continuation_attempt or 0),
            "requested_model": model_name,
        }
        if route.backend is Backend.FROZEN:
            if route.trainable:
                raise ValueError(f"role {role!r} cannot be trainable through a frozen backend")
            return self.frozen_model_builder(session_id=None, **metadata)

        sid, key = self._open_actor_session(
            role=role,
            action_id=action_id,
            task_id=task_id,
            continuation=int(continuation_attempt or 0),
        )
        actor_session = ActorSession(
            session_id=sid,
            key=key,
            route=route,
            requested_model=model_name,
            thread_id=thread_id,
            run_id=run_id,
            parent_action_id=parent_action_id,
        )
        with self._lock:
            self._actor_sessions[sid] = actor_session
        try:
            return self.actor_model_builder(session_id=sid, **metadata)
        except BaseException:
            # Keep the session registered so the episode runner's exception
            # path can await abort() and cancel/drop it safely.
            raise

    def _open_actor_session(
        self,
        *,
        role: str,
        action_id: str | None,
        task_id: str | None,
        continuation: int,
    ) -> tuple[str, SessionKey]:
        counter_key = (
            role,
            action_id or "root",
            task_id or "root",
            continuation,
        )
        with self._lock:
            if self._sealed:
                raise RuntimeError("episode model factory is already sealed")
            attempt = self._attempts.get(counter_key, 0)
            self._attempts[counter_key] = attempt + 1
            sid, key = self.sessions.allocate(
                role=role,
                action_id=action_id,
                task_id=task_id,
                attempt=attempt,
                continuation=continuation,
            )
            try:
                self.adapter.open_session(
                    sid,
                    sampling_defaults={
                        "temperature": self.config.temperature,
                        "top_p": self.config.top_p,
                        "max_new_tokens": self.config.max_response_tokens,
                    },
                    max_context_tokens=self.config.max_context_tokens,
                    annotation_context={
                        "role": role,
                        "episode_id": self.episode_id,
                        "action_id": action_id,
                        "task_id": task_id,
                        "continuation_attempt": continuation,
                    },
                )
            except BaseException:
                self.sessions.close(sid)
                raise
        return sid, key

    def routing_table(self) -> list[dict[str, object]]:
        return self.router.table()

    async def finish(
        self,
        *,
        base_sample: Any,
        rollout_id: int,
        group_index: int | None,
        reward: float,
        reward_components: dict[str, float] | None = None,
        termination: str = "completed",
        wait_timeout: float = 5.0,
    ) -> list[Any]:
        """Drain all actor sessions and return one episode-owned sample fan-out."""

        with self._lock:
            self._sealed = True
            pending = list(self._actor_sessions.values())

        async def drain(session: ActorSession) -> list[Any]:
            metadata = {
                "episode_id": self.episode_id,
                "role": session.key.role,
                "session_id": session.session_id,
                "action_id": None if session.key.action_id == "root" else session.key.action_id,
                "task_id": None if session.key.task_id == "root" else session.key.task_id,
                "continuation_attempt": session.key.continuation,
                "attempt": session.key.attempt,
            }
            samples = await self.adapter.finish_session(
                session.session_id,
                base_sample=base_sample,
                reward=reward,
                extra_metadata=metadata,
                wait_timeout=wait_timeout,
            )
            return annotate_episode_samples(
                samples,
                rollout_id=rollout_id,
                group_index=group_index,
                episode_id=self.episode_id,
                role=session.key.role,
                session_id=session.session_id,
                mode=self.config.mode.value,
                model_version=self.model_version,
                trainable=session.route.trainable,
                reward=reward,
                reward_components=reward_components,
                action_id=metadata["action_id"],
                task_id=metadata["task_id"],
                parent_action_id=session.parent_action_id,
                termination=termination,
            )

        results = await asyncio.gather(
            *(drain(session) for session in pending),
            return_exceptions=True,
        )
        failed = [(session, result) for session, result in zip(pending, results, strict=True) if isinstance(result, BaseException)]
        if failed:
            await asyncio.gather(
                *(
                    self.adapter.drop_session(
                        session.session_id,
                        wait_timeout=wait_timeout,
                    )
                    for session, _ in failed
                ),
                return_exceptions=True,
            )
        with self._lock:
            for session in pending:
                self._actor_sessions.pop(session.session_id, None)
                self.sessions.close(session.session_id)
        if failed:
            raise failed[0][1]
        return [sample for session_samples in results for sample in session_samples]

    async def abort(self, *, wait_timeout: float = 5.0) -> None:
        """Drop only this episode's sessions, including partially served ones."""

        with self._lock:
            self._sealed = True
            pending = list(self._actor_sessions.values())
        await asyncio.gather(
            *(
                self.adapter.drop_session(
                    session.session_id,
                    wait_timeout=wait_timeout,
                )
                for session in pending
            ),
            return_exceptions=True,
        )
        with self._lock:
            for session in pending:
                self._actor_sessions.pop(session.session_id, None)
                self.sessions.close(session.session_id)
