"""Numeric and observation-alignment checks for the ordinary CUAD GiGPO pilot."""
from __future__ import annotations

import json
import math

import pytest

from slime_plugins.dcc.episode import CUAD_FINAL_REVIEW, EpisodeResult, PolicyCompletion
from slime_plugins.dcc.gigpo import step_advantages


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(json.dumps(messages, sort_keys=True).encode())


def episode(reward, observations, *, final=False):
    messages = [{"role": "system", "content": "CUAD"}, {"role": "user", "content": "Find the clause"}]
    steps = []
    for index, observation in enumerate(observations):
        messages.append({"role": "user", "content": observation})
        if final and index == len(observations) - 1:
            messages.append({"role": "user", "content": CUAD_FINAL_REVIEW})
        steps.append(PolicyCompletion("sampled action", Tokenizer().apply_chat_template(messages), [6, 7]))
        messages.append({"role": "assistant", "content": "sampled action"})
    return EpisodeResult("task", "cuad", "train", [], "completed", reward, {}, steps, messages, [])


def signal(results, **kwargs):
    return step_advantages(results, tokenizer=Tokenizer(),
                           similarity_threshold=kwargs.pop("similarity_threshold", 1), **kwargs)


def test_two_level_advantages_match_hand_calculation():
    values, diagnostic = signal([episode(1, ["shared", "left"]), episode(0, ["shared", "right", "end"])])
    scale = math.sqrt(.3) + 1e-6  # Visit rewards [1, 1, 0, 0, 0], mean .4.
    local = .475 / (math.sqrt(.45125) + 1e-6)  # Discounted returns [.95, 0].
    assert values[0] == pytest.approx([.6 / scale + local, .6 / scale])
    assert values[1] == pytest.approx([-.4 / scale - local, -.4 / scale, -.4 / scale])
    assert diagnostic["groups_with_multiple_rollouts"] == 1
    assert diagnostic["nonzero_local_steps"] == 2


def test_equal_terminal_rewards_can_have_nonzero_local_credit():
    values, diagnostic = signal([episode(1, ["shared"]), episode(1, ["shared", "unique"])])
    assert all(row["episode_advantage"] == 0 for row in diagnostic["rows"])
    assert values[0][0] > 0 > values[1][0]
    assert values[1][1] == 0


def test_zero_rewards_and_singleton_groups_have_no_artificial_credit():
    values, diagnostic = signal([episode(0, ["shared", "left"]), episode(0, ["shared", "right"])])
    assert values == [[0, 0], [0, 0]]
    assert diagnostic["nonzero_local_steps"] == 0


def test_zero_weight_preserves_the_upstream_episode_component():
    values, diagnostic = signal([episode(1, ["shared", "left"]), episode(0, ["shared"])], weight=0)
    scale = math.sqrt(1 / 3) + 1e-6
    assert values[0] == pytest.approx([(1 / 3) / scale] * 2)
    assert values[1] == pytest.approx([(-2 / 3) / scale])


def test_similarity_does_not_merge_final_and_tool_phases():
    left = "Tool observation: " + "a" * 100 + " X"
    right = "Tool observation: " + "a" * 100 + " Y"
    _, diagnostic = signal([episode(1, [left]), episode(0, [right])], similarity_threshold=.9)
    assert diagnostic["group_sizes"] == [2]
    _, diagnostic = signal([episode(1, [left]), episode(0, [left], final=True)], similarity_threshold=.9)
    assert diagnostic["group_sizes"] == [1, 1]


def test_prompt_mismatch_is_rejected_instead_of_using_unseen_evidence():
    result = episode(1, ["shared"])
    result.steps[0].prompt_ids.append(1)
    with pytest.raises(ValueError, match="untruncated recorded prompts"):
        signal([result, episode(0, ["shared"])])


def test_excluded_episodes_match_the_existing_trainer():
    failed = episode(0, ["shared"])
    failed.status = "infrastructure_error"
    values, diagnostic = signal([episode(1, ["shared"]), failed])
    assert values == [[0], []]
    assert len(diagnostic["rows"]) == 1


@pytest.mark.parametrize("change", ["task", "split", "benchmark"])
def test_groups_cannot_mix_tasks_splits_or_benchmarks(change):
    result = episode(0, ["shared"])
    setattr(result, {"task": "task_id", "split": "split", "benchmark": "benchmark"}[change],
            {"task": "other", "split": "test", "benchmark": "widesearch"}[change])
    with pytest.raises(ValueError):
        signal([episode(1, ["shared"]), result])
