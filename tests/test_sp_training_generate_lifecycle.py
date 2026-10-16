"""Regressions for training episode failures, cancellation, and sampling."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from deerflow.community.search_r1 import tools as search_tools
from slime.utils.types import Sample
from slime_plugins.stackplanner import generate as generate_module
from slime_plugins.stackplanner.config import TrainingConfig, TrainingMode
from slime_plugins.stackplanner.ole import select_prompt_groups


class RecordingAdapter:
    def __init__(self, *, block_finish=False, block_drop=False):
        self.opened = {}
        self.finished = []
        self.dropped = []
        self.block_finish = block_finish
        self.block_drop = block_drop
        self.finish_started = asyncio.Event()
        self.drop_started = asyncio.Event()

    def open_session(self, sid, **kwargs):
        self.opened[sid] = kwargs

    async def finish_session(self, sid, **kwargs):
        self.finished.append(sid)
        if self.block_finish:
            self.finish_started.set()
            await asyncio.Future()
        return [
            Sample(
                tokens=[1, 2],
                response_length=1,
                loss_mask=[1],
                rollout_log_probs=[-0.1],
            )
        ]

    async def drop_session(self, sid, **kwargs):
        self.dropped.append(sid)
        if self.block_drop:
            self.drop_started.set()
            await asyncio.Future()


def base_sample(index=7):
    return Sample(
        prompt="question",
        label="right",
        index=index,
        group_index=2,
        session_id=f"sample-{index}",
    )


def install_harness(monkeypatch, *, behavior=None, adapter=None, config=None):
    adapter = adapter or RecordingAdapter()
    config = config or TrainingConfig()
    models = []
    factories = []

    def model_builder(**options):
        def build(**kwargs):
            model = SimpleNamespace(**options, **kwargs)
            models.append(model)
            return model

        return build

    def make_graph(runtime_config, *, app_config):
        values = runtime_config["configurable"]
        factory = values["sp_training_model_factory"]
        factories.append(factory)

        async def invoke(inputs, **_kwargs):
            for role in ("central", "researcher"):
                factory(
                    role=role,
                    model_name="configured-model",
                    episode_id=values["episode_id"],
                    thread_id=values["thread_id"],
                    run_id=values["run_id"],
                    action_id=None,
                    task_id=None,
                    continuation_attempt=0,
                    app_config=app_config,
                )
            if behavior is not None:
                return await behavior(inputs, factory)
            return {"messages": [SimpleNamespace(type="ai", content="right")]}

        return SimpleNamespace(ainvoke=invoke)

    monkeypatch.setattr(generate_module, "_load_training_config", lambda: config)
    monkeypatch.setattr(
        generate_module,
        "_adapter_service",
        lambda *_: SimpleNamespace(
            adapter=adapter, app_handle=SimpleNamespace(port=18001)
        ),
    )
    monkeypatch.setattr(
        generate_module,
        "_app_config",
        lambda _: SimpleNamespace(models=[SimpleNamespace(name="configured-model")]),
    )
    monkeypatch.setattr(generate_module, "make_openai_model_builder", model_builder)
    monkeypatch.setitem(
        sys.modules,
        "deerflow.sp.runtime",
        SimpleNamespace(
            SP_SUBAGENT_REGISTRY_NAMES={"researcher"}, make_sp_agent=make_graph
        ),
    )
    return SimpleNamespace(
        args=SimpleNamespace(hf_checkpoint="checkpoint-v1"),
        adapter=adapter,
        models=models,
        factories=factories,
    )


def failing_search(monkeypatch):
    client = MagicMock()
    client.return_value.__aenter__.return_value.post = AsyncMock(
        side_effect=httpx.ConnectError("retriever offline")
    )
    monkeypatch.setattr(search_tools, "_get_tool_config", lambda: {})
    monkeypatch.setattr(search_tools.httpx, "AsyncClient", client)


@pytest.mark.parametrize("ending", ["answer", "policy_state", "policy_exception"])
def test_hidden_search_failure_prevents_reward_and_policy_training(monkeypatch, ending):
    failing_search(monkeypatch)

    async def behavior(_inputs, _factory):
        # The specialist consumes the error; no tool message reaches Central.
        await search_tools.web_search_tool.ainvoke({"query": "test"})
        if ending == "policy_exception":
            raise type("GraphRecursionError", (RuntimeError,), {})("limit")
        if ending == "policy_state":
            return {"sp_last_handler_result": {"termination": "loop_capped"}}
        return {"messages": [SimpleNamespace(type="ai", content="right")]}

    harness = install_harness(monkeypatch, behavior=behavior)
    reward = MagicMock(wraps=generate_module._reward_result)
    monkeypatch.setattr(generate_module, "_reward_result", reward)
    [result] = asyncio.run(generate_module.generate(harness.args, base_sample(), {}))

    assert result.status == Sample.Status.ABORTED
    assert result.remove_sample is True
    assert result.reward == 0.0
    assert result.loss_mask == [0]
    assert "infrastructure_failure" in result.metadata
    assert harness.adapter.finished == []
    assert set(harness.adapter.dropped) == set(harness.adapter.opened)
    reward.assert_not_called()


def test_aborted_episode_has_lineage_and_is_excluded_from_ole(monkeypatch):
    async def behavior(*_):
        raise RuntimeError("endpoint offline")

    harness = install_harness(monkeypatch, behavior=behavior)
    [result] = asyncio.run(generate_module.generate(harness.args, base_sample(), {}))
    assert result.rollout_id == 7
    assert result.metadata["rollout_id"] == 7
    state = {}
    metrics = select_prompt_groups(
        SimpleNamespace(reward_key=None),
        TrainingConfig(ole_enabled=True),
        [[[result]]],
        state,
        rollout_id=1,
    )
    assert metrics["ole/episodes"] == 0
    assert metrics["ole/groups"] == 0


@pytest.mark.parametrize("ending", ["answer", "policy_state", "policy_exception"])
def test_normal_episode_and_policy_failures_remain_trainable(monkeypatch, ending):
    async def behavior(*_):
        if ending == "policy_exception":
            raise type("GraphRecursionError", (RuntimeError,), {})("limit")
        if ending == "policy_state":
            return {"sp_last_handler_result": {"termination": "loop_capped"}}
        return {"messages": [SimpleNamespace(type="ai", content="right")]}

    harness = install_harness(monkeypatch, behavior=behavior)
    [result] = asyncio.run(generate_module.generate(harness.args, base_sample(), {}))
    assert result.reward == (1.0 if ending == "answer" else 0.0)
    assert not result.remove_sample
    assert result.loss_mask == [1]
    assert "infrastructure_failure" not in result.metadata
    assert result.rollout_id == 7
    assert harness.adapter.dropped == []


def test_search_context_is_isolated_and_restored_for_concurrent_episodes(monkeypatch):
    failing_search(monkeypatch)

    async def behavior(_inputs, factory):
        if factory.episode_id.endswith("sample-7"):
            await search_tools.web_search_tool.ainvoke({"query": "test"})
        await asyncio.sleep(0)
        return {"messages": [SimpleNamespace(type="ai", content="right")]}

    harness = install_harness(monkeypatch, behavior=behavior)

    async def scenario():
        parent_failures = []
        token = search_tools.SEARCH_R1_FAILURES.set(parent_failures)
        try:
            [failed], [succeeded] = await asyncio.gather(
                generate_module.generate(harness.args, base_sample(7), {}),
                generate_module.generate(harness.args, base_sample(8), {}),
            )
            assert failed.remove_sample
            assert not succeeded.remove_sample
            assert succeeded.reward == 1.0
            assert search_tools.SEARCH_R1_FAILURES.get() is parent_failures
            assert parent_failures == []
        finally:
            search_tools.SEARCH_R1_FAILURES.reset(token)

    asyncio.run(scenario())


@pytest.mark.parametrize("ending", ["success", "exception", "cancel"])
def test_search_context_is_restored_on_every_exit(monkeypatch, ending):
    async def behavior(*_):
        if ending == "cancel":
            raise asyncio.CancelledError
        if ending == "exception":
            raise RuntimeError("endpoint failed")
        return {"messages": [SimpleNamespace(type="ai", content="right")]}

    harness = install_harness(monkeypatch, behavior=behavior)

    async def scenario():
        parent_failures = []
        token = search_tools.SEARCH_R1_FAILURES.set(parent_failures)
        try:
            if ending == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await generate_module.generate(harness.args, base_sample(), {})
            else:
                await generate_module.generate(harness.args, base_sample(), {})
            assert search_tools.SEARCH_R1_FAILURES.get() is parent_failures
            if ending != "success":
                assert set(harness.adapter.dropped) == set(harness.adapter.opened)
        finally:
            search_tools.SEARCH_R1_FAILURES.reset(token)

    asyncio.run(scenario())


def test_process_exit_signals_propagate_and_abort_sessions(monkeypatch):
    class RunnerShutdown(BaseException):
        pass

    async def behavior(*_):
        raise RunnerShutdown

    harness = install_harness(monkeypatch, behavior=behavior)
    with pytest.raises(RunnerShutdown):
        asyncio.run(generate_module.generate(harness.args, base_sample(), {}))
    assert set(harness.adapter.dropped) == set(harness.adapter.opened)


@pytest.mark.parametrize("ending", ["cancel", "exception"])
def test_abort_error_preserves_cancellation_or_original_failure(monkeypatch, ending):
    async def behavior(*_):
        if ending == "cancel":
            raise asyncio.CancelledError
        raise RuntimeError("original graph failure")

    harness = install_harness(monkeypatch, behavior=behavior)
    monkeypatch.setattr(
        generate_module.EpisodeModelFactory,
        "abort",
        AsyncMock(side_effect=RuntimeError("cleanup failed")),
    )
    call = generate_module.generate(harness.args, base_sample(), {})
    if ending == "cancel":
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(call)
    else:
        [result] = asyncio.run(call)
        assert (
            "original graph failure"
            in result.metadata["infrastructure_failure"]["kind"]
        )


@pytest.mark.parametrize("stage", ["policy_drain", "abort"])
def test_cancellation_during_cleanup_propagates(monkeypatch, stage):
    async def behavior(*_):
        if stage == "policy_drain":
            raise type("GraphRecursionError", (RuntimeError,), {})("limit")
        raise RuntimeError("endpoint failed")

    async def scenario():
        adapter = RecordingAdapter(
            block_finish=stage == "policy_drain", block_drop=stage == "abort"
        )
        harness = install_harness(monkeypatch, behavior=behavior, adapter=adapter)
        task = asyncio.create_task(
            generate_module.generate(harness.args, base_sample(), {})
        )
        event = (
            adapter.finish_started if stage == "policy_drain" else adapter.drop_started
        )
        await asyncio.wait_for(event.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert set(adapter.dropped) == set(adapter.opened)

    asyncio.run(scenario())


@pytest.mark.parametrize("evaluation", [False, True])
@pytest.mark.parametrize(
    "mode", [TrainingMode.CENTRAL_ONLY, TrainingMode.SHARED_ALL_AGENTS]
)
def test_evaluation_uses_greedy_sampling_for_all_roles(monkeypatch, evaluation, mode):
    from slime.agent.adapters.common import _sampling_params

    config = TrainingConfig(mode=mode, temperature=0.7, top_p=0.8)
    harness = install_harness(monkeypatch, config=config)
    asyncio.run(
        generate_module.generate(harness.args, base_sample(), {}, evaluation=evaluation)
    )
    for model in harness.models:
        assert model.model_kwargs["temperature"] == (0.0 if evaluation else 0.7)
        assert model.model_kwargs["top_p"] == (1.0 if evaluation else 0.8)
        if model.session_id is not None:
            session = SimpleNamespace(
                sampling_defaults=harness.adapter.opened[model.session_id][
                    "sampling_defaults"
                ]
            )
            effective = _sampling_params(
                session,
                model.model_kwargs,
                max_token_keys=("max_tokens",),
                stop_keys=("stop",),
            )
            assert effective["temperature"] == (0.0 if evaluation else 0.7)
            assert effective["top_p"] == (1.0 if evaluation else 0.8)
