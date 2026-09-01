from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from slime_plugins.stackplanner.config import TrainingConfig, TrainingMode
from slime_plugins.stackplanner.model_factory import (
    EpisodeModelFactory,
    make_openai_model_builder,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@dataclass
class FakeSample:
    tokens: list[int] = field(default_factory=lambda: [10, 20, 21])
    response_length: int = 2
    loss_mask: list[int] = field(default_factory=lambda: [1, 1])
    rollout_log_probs: list[float] = field(default_factory=lambda: [-0.1, -0.2])
    metadata: dict = field(default_factory=dict)
    reward: float = 0.0
    rollout_id: int | None = None
    group_index: int | None = None


class FakeAdapter:
    def __init__(self) -> None:
        self.opened = {}
        self.finished = []
        self.dropped = []

    def open_session(self, sid, **kwargs):
        assert sid not in self.opened
        self.opened[sid] = kwargs

    async def finish_session(self, sid, **kwargs):
        self.finished.append((sid, kwargs))
        return [FakeSample()]

    async def drop_session(self, sid, **kwargs):
        self.dropped.append((sid, kwargs))


class FailingFinishAdapter(FakeAdapter):
    def __init__(self, failing_role: str) -> None:
        super().__init__()
        self.failing_role = failing_role

    async def finish_session(self, sid, **kwargs):
        self.finished.append((sid, kwargs))
        if f".{self.failing_role}." in sid:
            raise RuntimeError("adapter disconnected")
        return [FakeSample()]


def _factory(mode=TrainingMode.CENTRAL_ONLY):
    adapter = FakeAdapter()
    built = []

    def actor_builder(**kwargs):
        built.append(("actor", kwargs))
        return SimpleNamespace(backend="actor", session_id=kwargs["session_id"])

    def frozen_builder(**kwargs):
        built.append(("frozen", kwargs))
        return SimpleNamespace(backend="frozen", session_id=kwargs["session_id"])

    factory = EpisodeModelFactory(
        config=TrainingConfig(mode=mode),
        episode_id="episode-1",
        policy_roles={"researcher", "coder"},
        adapter=adapter,
        actor_model_builder=actor_builder,
        frozen_model_builder=frozen_builder,
        model_version="checkpoint-v1",
    )
    return factory, adapter, built


def _call(factory, role, **overrides):
    values = {
        "role": role,
        "model_name": "requested-model",
        "episode_id": "episode-1",
        "thread_id": "thread-1",
        "run_id": "run-1",
        "action_id": None,
        "task_id": None,
        "continuation_attempt": 0,
        "app_config": object(),
    }
    values.update(overrides)
    return factory(**values)


def test_unused_openai_builder_allows_missing_frozen_endpoint():
    builder = make_openai_model_builder(base_url="", model="unused-frozen")

    assert callable(builder)


def test_openai_builder_rejects_missing_endpoint_when_route_is_used():
    builder = make_openai_model_builder(base_url="", model="missing-frozen")

    with pytest.raises(ValueError, match="base_url must not be empty"):
        builder(
            requested_model="missing-frozen",
            session_id=None,
            role="researcher",
        )


def test_openai_builder_can_use_qwen_vllm_recovery_for_frozen_routes():
    builder = make_openai_model_builder(
        base_url="http://localhost:18002/v1",
        model="Qwen2.5-3B-Instruct",
        recover_qwen_tool_calls=True,
    )

    model = builder(
        requested_model="ignored",
        session_id=None,
        role="researcher",
    )

    assert type(model).__name__ == "VllmChatModel"
    assert model.model_name == "Qwen2.5-3B-Instruct"
    assert model.openai_api_key.get_secret_value() == "EMPTY"


def test_openai_builder_requires_explicit_frozen_credential_only_when_configured(
    monkeypatch,
):
    monkeypatch.delenv("SP_FROZEN_API_KEY", raising=False)
    builder = make_openai_model_builder(
        base_url="http://localhost:18002/v1",
        model="Qwen2.5-3B-Instruct",
        api_key_env="SP_FROZEN_API_KEY",
    )

    with pytest.raises(
        RuntimeError, match="required frozen endpoint credential SP_FROZEN_API_KEY"
    ):
        builder(
            requested_model="ignored",
            session_id=None,
            role="researcher",
        )


def test_openai_builder_keeps_native_chat_openai_for_actor_routes():
    builder = make_openai_model_builder(
        base_url="http://localhost:18001/v1",
        model="Qwen2.5-3B-Instruct",
    )

    model = builder(
        requested_model="ignored",
        session_id="episode.central.root",
        role="central",
    )

    assert type(model).__name__ == "ChatOpenAI"


def test_openai_builder_shares_client_only_within_one_event_loop():
    builder = make_openai_model_builder(
        base_url="http://localhost:18001/v1",
        model="Qwen2.5-3B-Instruct",
        model_kwargs={"timeout": 17.0},
    )

    async def build_pair():
        first = builder(
            requested_model="ignored",
            session_id="episode.researcher.first",
            role="researcher",
        )
        second = builder(
            requested_model="ignored",
            session_id="episode.researcher.second",
            role="researcher",
        )
        assert first.http_async_client is second.http_async_client
        client = first.http_async_client
        await client.aclose()
        return client

    first_loop_client = asyncio.run(build_pair())
    second_loop_client = asyncio.run(build_pair())
    assert first_loop_client is not second_loop_client


def test_central_only_opens_only_central_actor_session():
    factory, adapter, built = _factory()
    central = _call(factory, "central")
    frozen = _call(
        factory,
        "researcher",
        action_id="action-1",
        task_id="task-1",
    )

    assert central.backend == "actor"
    assert frozen.backend == "frozen"
    assert frozen.session_id is None
    assert set(adapter.opened) == {central.session_id}
    assert [kind for kind, _ in built] == ["actor", "frozen"]


def test_shared_mode_isolates_concurrent_delegations_and_continuations():
    factory, adapter, _ = _factory(TrainingMode.SHARED_ALL_AGENTS)
    sessions = {
        _call(factory, "central").session_id,
        _call(factory, "researcher", action_id="a1", task_id="t1").session_id,
        _call(factory, "researcher", action_id="a2", task_id="t2").session_id,
        _call(
            factory,
            "researcher",
            action_id="a1",
            task_id="t1",
            continuation_attempt=1,
        ).session_id,
    }

    assert len(sessions) == 4
    assert set(adapter.opened) == sessions
    contexts = [value["annotation_context"] for value in adapter.opened.values()]
    assert {value["role"] for value in contexts} == {"central", "researcher"}
    assert {value["episode_id"] for value in contexts} == {"episode-1"}
    assert {(value["action_id"], value["task_id"]) for value in contexts} >= {
        ("a1", "t1"),
        ("a2", "t2"),
    }


def test_identical_retry_gets_a_new_attempt_session():
    factory, adapter, _ = _factory(TrainingMode.SHARED_ALL_AGENTS)
    first = _call(factory, "coder", action_id="a1", task_id="t1")
    retry = _call(factory, "coder", action_id="a1", task_id="t1")
    assert first.session_id != retry.session_id
    assert len(adapter.opened) == 2


@pytest.mark.anyio
async def test_finish_drains_actor_fanout_with_common_episode_lineage():
    factory, adapter, _ = _factory(TrainingMode.SHARED_ALL_AGENTS)
    central = _call(factory, "central")
    researcher = _call(
        factory,
        "researcher",
        action_id="action-2",
        task_id="task-2",
        parent_action_id="action-1",
    )

    samples = await factory.finish(
        base_sample=SimpleNamespace(index=3),
        rollout_id=17,
        group_index=4,
        reward=0.75,
        reward_components={"exact": 0.75},
    )

    assert len(samples) == 2
    assert {sample.metadata["episode_id"] for sample in samples} == {"episode-1"}
    assert {sample.rollout_id for sample in samples} == {17}
    assert {sample.group_index for sample in samples} == {4}
    assert {sample.metadata["role"] for sample in samples} == {"central", "researcher"}
    researcher_sample = next(
        sample for sample in samples if sample.metadata["role"] == "researcher"
    )
    assert researcher_sample.metadata["parent_action_id"] == "action-1"
    assert {sid for sid, _ in adapter.finished} == {
        central.session_id,
        researcher.session_id,
    }
    assert factory.sessions.open_sessions() == {}


@pytest.mark.anyio
async def test_abort_drops_only_episode_owned_sessions_and_seals_factory():
    factory, adapter, _ = _factory(TrainingMode.SHARED_ALL_AGENTS)
    central = _call(factory, "central")
    researcher = _call(factory, "researcher", action_id="a1", task_id="t1")

    await factory.abort()

    assert {sid for sid, _ in adapter.dropped} == {
        central.session_id,
        researcher.session_id,
    }
    assert factory.sessions.open_sessions() == {}
    with pytest.raises(RuntimeError, match="already sealed"):
        _call(factory, "central")


@pytest.mark.anyio
async def test_model_construction_failure_remains_abortable():
    adapter = FakeAdapter()

    def broken_builder(**_kwargs):
        raise RuntimeError("model construction failed")

    factory = EpisodeModelFactory(
        config=TrainingConfig(mode=TrainingMode.CENTRAL_ONLY),
        episode_id="episode-1",
        policy_roles=set(),
        adapter=adapter,
        actor_model_builder=broken_builder,
        frozen_model_builder=lambda **_: object(),
        model_version="v1",
    )
    with pytest.raises(RuntimeError, match="model construction failed"):
        _call(factory, "central")

    [sid] = adapter.opened
    await factory.abort()
    assert [item[0] for item in adapter.dropped] == [sid]
    assert factory.sessions.open_sessions() == {}


@pytest.mark.anyio
async def test_partial_finish_failure_drops_failed_session_and_cleans_all_state():
    adapter = FailingFinishAdapter("researcher")
    factory = EpisodeModelFactory(
        config=TrainingConfig(mode=TrainingMode.SHARED_ALL_AGENTS),
        episode_id="episode-1",
        policy_roles={"researcher"},
        adapter=adapter,
        actor_model_builder=lambda **kwargs: SimpleNamespace(**kwargs),
        frozen_model_builder=lambda **kwargs: SimpleNamespace(**kwargs),
        model_version="v1",
    )
    _call(factory, "central")
    researcher = _call(
        factory,
        "researcher",
        action_id="a1",
        task_id="t1",
    )

    with pytest.raises(RuntimeError, match="adapter disconnected"):
        await factory.finish(
            base_sample=SimpleNamespace(index=1),
            rollout_id=1,
            group_index=0,
            reward=0.0,
        )

    assert [item[0] for item in adapter.dropped] == [researcher.session_id]
    assert factory.sessions.open_sessions() == {}


def test_mismatched_episode_and_frozen_trainable_route_fail_closed():
    factory, _, _ = _factory()
    with pytest.raises(ValueError, match="cannot serve"):
        _call(factory, "central", episode_id="wrong-episode")

    config = TrainingConfig(
        mode=TrainingMode.CENTRAL_ONLY,
        role_overrides={"coder": {"backend": "frozen", "trainable": True}},
    )
    bad = EpisodeModelFactory(
        config=config,
        episode_id="episode-1",
        policy_roles={"coder"},
        adapter=FakeAdapter(),
        actor_model_builder=lambda **_: object(),
        frozen_model_builder=lambda **_: object(),
        model_version="v1",
    )
    with pytest.raises(ValueError, match="trainable through a frozen backend"):
        _call(bad, "coder")
