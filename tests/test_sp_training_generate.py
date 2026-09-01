from __future__ import annotations

import asyncio
from types import SimpleNamespace

from slime_plugins.stackplanner.generate import (
    _artifacts,
    _episode_id,
    _exception_policy_termination,
    _final_answer,
    _finish_policy_failure,
    _load_function,
    _policy_termination,
    _prompt_text,
    _runtime_id,
)


def test_load_function_accepts_colon_and_dot_syntax():
    assert _load_function(
        "slime_plugins.stackplanner.rewards:deterministic_reward"
    ) is _load_function("slime_plugins.stackplanner.rewards.deterministic_reward")


def test_prompt_text_uses_only_user_messages():
    sample = SimpleNamespace(
        prompt=[
            {"role": "system", "content": "secret system"},
            {"role": "user", "content": "first"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "second"},
                    {"type": "image_url", "url": "ignored"},
                ],
            },
        ]
    )
    assert _prompt_text(sample) == "first\nsecond"


def test_prompt_text_can_add_a_recorded_training_only_prefix():
    sample = SimpleNamespace(prompt="Who wrote Hamlet?")
    assert (
        _prompt_text(sample, training_prefix="  Use a control action first.  ")
        == "Use a control action first.\n\nWho wrote Hamlet?"
    )


def test_episode_id_is_stable_for_sample_session(monkeypatch):
    monkeypatch.setenv("SP_TRAINING_RUN_ID", "formal")
    sample = SimpleNamespace(session_id="sample-17", index=1, group_index=2)
    assert _episode_id(sample) == "formal.sample-17"


def test_runtime_ids_are_stable_safe_and_episode_isolated():
    first = _runtime_id("thread", "formal.sample/17")
    repeated = _runtime_id("thread", "formal.sample/17")
    second = _runtime_id("thread", "formal.sample-17")

    assert first == repeated
    assert first != second
    assert first.startswith("thread-")
    assert first.removeprefix("thread-").isalnum()


def test_final_answer_and_artifacts_come_from_production_state():
    state = {
        "messages": [
            SimpleNamespace(type="human", content="prompt"),
            SimpleNamespace(type="ai", content="final answer"),
        ],
        "sp_current_artifact_refs": {
            "report": {"virtual_path": "/mnt/user-data/outputs/report.md"},
            "duplicate": {"virtual_path": "/mnt/user-data/outputs/report.md"},
            "evidence": {"artifact_id": "artifact-2"},
        },
    }
    assert _final_answer(state) == "final answer"
    assert _artifacts(state) == (
        "/mnt/user-data/outputs/report.md",
        "artifact-2",
    )


def test_policy_termination_classifies_native_loop_and_fatal_action_state():
    assert (
        _policy_termination(
            {
                "sp_last_handler_result": {
                    "next_step": "error_fatal",
                    "termination": "loop_capped",
                }
            }
        )
        == "loop_capped"
    )
    assert (
        _policy_termination(
            {
                "sp_last_handler_result": {
                    "next_step": "error_fatal",
                    "error": "invalid control action",
                }
            }
        )
        == "policy_error"
    )
    assert _policy_termination({}) is None


def test_exception_policy_termination_is_narrow_and_fail_closed():
    GraphRecursionError = type("GraphRecursionError", (RuntimeError,), {})
    ActionValidationError = type("ActionValidationError", (ValueError,), {})

    assert (
        _exception_policy_termination(GraphRecursionError("limit"))
        == "graph_recursion_capped"
    )
    assert (
        _exception_policy_termination(ActionValidationError("bad action"))
        == "invalid_action"
    )
    assert _exception_policy_termination(TimeoutError("endpoint timeout")) is None
    assert _exception_policy_termination(RuntimeError("unknown")) is None


def test_finish_policy_failure_drains_actor_trajectory_without_removal():
    captured = {}
    sample = SimpleNamespace(
        index=4, group_index=1, metadata={}, remove_sample=False, reward=None
    )

    class Factory:
        async def finish(self, **kwargs):
            captured.update(kwargs)
            sample.reward = kwargs["reward"]
            sample.metadata["termination"] = kwargs["termination"]
            return [sample]

    result = asyncio.run(
        _finish_policy_failure(
            Factory(),
            base_sample=sample,
            episode_id="episode-4",
            termination="loop_capped",
        )
    )

    assert result == [sample]
    assert captured["rollout_id"] == 4
    assert captured["group_index"] == 1
    assert captured["reward"] == 0.0
    assert captured["termination"] == "loop_capped"
    assert sample.remove_sample is False
    assert "infrastructure_failure" not in sample.metadata
