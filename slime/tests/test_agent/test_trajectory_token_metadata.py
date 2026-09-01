from __future__ import annotations

from slime.agent.trajectory import TrajectoryManager, TurnRecord
from slime.utils.types import Sample


def _base() -> Sample:
    return Sample(index=1, group_index=0, rollout_id=1, prompt="prompt", label="label")


def test_token_metadata_follows_exact_generated_ids_into_sample():
    manager = TrajectoryManager()
    manager.record_turn(
        "sid",
        turn=TurnRecord(
            prompt_ids=[10],
            output_ids=[20, 21],
            output_log_probs=[-0.1, -0.2],
            finish_reason="stop",
            output_token_metadata={"capability_labels": ["action_choice", "shared_syntax"]},
        ),
        prompt_messages=[{"role": "user", "content": "prompt"}],
        response_message={"role": "assistant", "content": "answer"},
    )
    [sample] = manager.get_trajectory("sid", base_sample=_base())
    assert sample.train_metadata == {
        "capability_labels": ["action_choice", "shared_syntax"]
    }


def test_realign_masks_and_clears_replayed_token_metadata():
    manager = TrajectoryManager(fork_threshold_tokens=10)
    prompt = [{"role": "user", "content": "prompt"}]
    manager.record_turn(
        "sid",
        turn=TurnRecord(
            prompt_ids=[10],
            output_ids=[20, 21],
            output_log_probs=[-0.1, -0.2],
            finish_reason="tool_calls",
            output_token_metadata={"capability_labels": ["action_choice", "shared_syntax"]},
        ),
        prompt_messages=prompt,
        response_message={"role": "assistant", "content": "old"},
    )
    manager.record_turn(
        "sid",
        turn=TurnRecord(
            prompt_ids=[10, 20, 99],
            output_ids=[30],
            output_log_probs=[-0.3],
            finish_reason="stop",
            output_token_metadata={"capability_labels": ["user_output"]},
        ),
        prompt_messages=[*prompt, {"role": "assistant", "content": "old"}],
        response_message={"role": "assistant", "content": "new"},
    )
    [sample] = manager.get_trajectory("sid", base_sample=_base())
    assert sample.loss_mask == [0, 0, 1]
    assert sample.train_metadata == {
        "capability_labels": [None, None, "user_output"]
    }


def test_misaligned_turn_token_metadata_fails_closed():
    manager = TrajectoryManager()
    try:
        manager.record_turn(
            "sid",
            turn=TurnRecord(
                prompt_ids=[10],
                output_ids=[20, 21],
                output_token_metadata={"capability_labels": ["shared_syntax"]},
                finish_reason="stop",
            ),
            prompt_messages=[{"role": "user", "content": "prompt"}],
            response_message=None,
        )
    except ValueError as exc:
        assert "output_ids length" in str(exc)
    else:
        raise AssertionError("misaligned token metadata was accepted")
