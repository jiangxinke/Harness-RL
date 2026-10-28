import asyncio
import json

import pytest

from slime_plugins.dcc.data import Task
from slime_plugins.dcc.episode import EpisodeRunner, PolicyCompletion
from slime_plugins.dcc.graphpo_collector import (
    ReplayMismatch, chat_tokens, collect_graphpo, expand_once,
)


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return self.encode(json.dumps(messages, ensure_ascii=False))

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]


TOKENIZER = Tokenizer()


def task():
    return Task("fixture__Document Name", "cuad", "train", "Find the name of this contract.",
                "PROMOTION AGREEMENT. This agreement governs promotion services.",
                ["PROMOTION AGREEMENT"])


def completion(messages, text):
    return PolicyCompletion(text, chat_tokens(TOKENIZER, messages), TOKENIZER.encode(text))


async def summarize(messages):
    decisions = sum(message["role"] == "assistant" for message in messages)
    return f"Observed promotion agreement; completed decisions={decisions}"


async def embed(summaries):
    return [[1.0, 0.0] for _ in summaries]


def test_actual_runner_replay_preserves_exact_prompt_and_final_control():
    runner = EpisodeRunner(max_steps=3, initial_chunks=1)
    calls = []

    async def policy(messages, **kwargs):
        calls.append(messages)
        if len(calls) == 3:
            text = '{"answer":[{"text":"PROMOTION AGREEMENT","confidence":1}]}'
        else:
            text = '{"tool":"search_document","arguments":{"query":"promotion"}}'
        return completion(messages, text)

    async def run():
        first = await expand_once(task(), runner, policy, tokenizer=TOKENIZER)
        assert first.result is None
        second = await expand_once(task(), runner, policy, tokenizer=TOKENIZER,
                                   prefix=[first.completion])
        assert second.result is None
        assert any("Final turn:" in message["content"] for message in second.next_messages)
        last = await expand_once(task(), runner, policy, tokenizer=TOKENIZER,
                                 prefix=[first.completion, second.completion])
        assert last.result.status == "completed"
        assert last.result.reward == 1
        assert last.replayed_decisions == 2
        assert len(last.result.steps) == 3
        assert last.result.steps[0] is first.completion
        assert last.result.steps[1] is second.completion
        assert len(calls) == 3
    asyncio.run(run())


def test_replay_mismatch_fails_before_new_generation():
    runner = EpisodeRunner(max_steps=2)
    bogus = PolicyCompletion('{"tool":"search_document","arguments":{"query":"promotion"}}',
                             [123], [1])

    async def policy(*args, **kwargs):
        raise AssertionError("Mismatching replay must not reach generation")

    with pytest.raises(ReplayMismatch, match="cached prefix mismatch"):
        asyncio.run(expand_once(task(), runner, policy, tokenizer=TOKENIZER, prefix=[bogus]))


def test_new_generation_truncation_is_rejected():
    runner = EpisodeRunner(max_steps=2)

    async def policy(messages, **kwargs):
        return PolicyCompletion('{"answer":[]}', [1], [2])

    with pytest.raises(ReplayMismatch, match="truncated"):
        asyncio.run(expand_once(task(), runner, policy, tokenizer=TOKENIZER))


def test_real_runner_branching_trains_each_generated_edge_once():
    runner = EpisodeRunner(max_steps=3, initial_chunks=1)
    policy_calls = []

    async def policy(messages, **kwargs):
        count = sum(message["role"] == "assistant" for message in messages)
        text = ('{"answer":[{"text":"PROMOTION AGREEMENT","confidence":1}]}' if count == 2
                else '{"tool":"search_document","arguments":{"query":"promotion"}}')
        policy_calls.append(messages)
        return completion(messages, text)

    result = asyncio.run(collect_graphpo(task(), runner, policy, tokenizer=TOKENIZER,
                                         summarize=summarize, embed=embed, branching=(2, 2, 1)))
    assert len(result.results) == 4
    assert len(policy_calls) == 10
    assert len(result.training_items) == 10
    assert sum(len(episode.steps) for episode in result.results) == 12
    assert len({id(item[0]) for item in result.training_items}) == 10
    assert result.diagnostics["budget"][1]["saved"] == 1
    assert result.diagnostics["budget"][1]["reallocated"] == 1
    assert sorted(result.diagnostics["budget"][1]["allocation"].values()) == [1, 3]
    assert result.diagnostics["costs"]["summary_calls"] == 11
    assert result.diagnostics["costs"]["generated_edges"] == 10
    assert all(path["common_success_reward"] == 10 for path in result.diagnostics["terminal_paths"])


def test_policy_can_terminate_early_without_synthetic_leaves():
    runner = EpisodeRunner(max_steps=3)

    async def policy(messages, **kwargs):
        return completion(messages, '{"answer":[]}')

    result = asyncio.run(collect_graphpo(task(), runner, policy, tokenizer=TOKENIZER,
                                         summarize=summarize, embed=embed, branching=(2, 2, 2)))
    assert len(result.results) == 2
    assert len(result.training_items) == 2
    assert len(result.diagnostics["budget"]) == 1
    assert result.diagnostics["costs"]["generated_edges"] == 2


def test_edge_budget_keeps_unfinished_evidence_unknown():
    runner = EpisodeRunner(max_steps=3)

    async def policy(messages, **kwargs):
        return completion(messages, '{"tool":"search_document","arguments":{"query":"promotion"}}')

    result = asyncio.run(collect_graphpo(task(), runner, policy, tokenizer=TOKENIZER,
                                         summarize=summarize, embed=embed, branching=(2, 2, 2),
                                         max_generated_edges=3))
    assert result.results == []
    assert result.training_items == []
    assert result.diagnostics["costs"]["generated_edges"] == 3
    assert result.diagnostics["exhaustion"] == "generated_edge_budget"
    assert all(node["score"] is None for node in result.diagnostics["nodes"])


def test_auxiliary_usage_is_separate_from_generated_actor_tokens():
    runner = EpisodeRunner(max_steps=1)

    async def policy(messages, **kwargs):
        return completion(messages, '{"answer":[]}')

    async def counted_summary(messages):
        counted_summary.usage["input_tokens"] += 100
        counted_summary.usage["output_tokens"] += 10
        return "Public observed state"

    counted_summary.usage = {"input_tokens": 17, "output_tokens": 5}
    result = asyncio.run(collect_graphpo(task(), runner, policy, tokenizer=TOKENIZER,
                                         summarize=counted_summary, embed=embed, branching=(2,)))
    assert result.diagnostics["costs"]["summarizer_usage"] == {"input_tokens": 300, "output_tokens": 30}
    assert result.diagnostics["costs"]["generated_response_tokens"] == 2 * len('{"answer":[]}')


def test_no_gold_fields_are_passed_to_summary():
    runner = EpisodeRunner(max_steps=1)
    secret_task = Task("fixture__Document Name", "cuad", "train", "Find contract name.",
                       "Observed contract content.", ["HIDDEN_REFERENCE_LABEL"])
    calls = []

    async def policy(messages, **kwargs):
        return completion(messages, '{"answer":[]}')

    async def checked_summary(messages):
        text = json.dumps(messages)
        assert "HIDDEN_REFERENCE_LABEL" not in text
        calls.append(text)
        return "Observed content only"

    asyncio.run(collect_graphpo(secret_task, runner, policy, tokenizer=TOKENIZER,
                               summarize=checked_summary, embed=embed, branching=(2,)))
    assert len(calls) == 3


def test_replay_does_not_support_live_web():
    web = Task("web", "widesearch", "train", "Search.")
    with pytest.raises(ValueError, match="deterministic CUAD"):
        asyncio.run(expand_once(web, EpisodeRunner(), None, tokenizer=TOKENIZER))


def test_audited_punctuation_spacing_success_matches_common_adapter():
    runner = EpisodeRunner(max_steps=1, initial_chunks=1, cuad_protocol="cuad-audited-v1")
    sample = Task("punctuation__Change of Control", "cuad", "train", "Find the change of control clause.",
                  "On Change of Control, CONSULTANT receives compensation.",
                  ["On Change of Control, CONSULTANT receives compensation."])

    async def policy(messages, **kwargs):
        return completion(messages, json.dumps({"answer": [{
            "text": "On Change of Control,CONSULTANT receives compensation.", "confidence": 1.0}]}))

    collected = asyncio.run(collect_graphpo(sample, runner, policy, tokenizer=TOKENIZER,
                                            summarize=summarize, embed=embed, branching=(2,)))
    result = collected.results[0]
    assert result.metrics["verbatim_span_fraction"] == 0
    assert result.metrics["f1"] == 1
    assert result.metrics["observed_token_span_fraction"] == 1
    assert result.reward == 1
    from slime_plugins.dcc.credit_adapter import build_credit_visits
    visits, _ = build_credit_visits(collected.results)
    assert all(visit.success and visit.task_reward == 10 for visit in visits)
    assert all(path["binary_success"] == 1 for path in collected.diagnostics["terminal_paths"])
    assert all(path["runner_metrics"]["verbatim_span_fraction"] == 0
               for path in collected.diagnostics["terminal_paths"])


def test_unobserved_correct_answer_is_failure_under_audited_protocol():
    runner = EpisodeRunner(max_steps=1, initial_chunks=0, cuad_protocol="cuad-audited-v1")

    async def policy(messages, **kwargs):
        return completion(messages, '{"answer":[{"text":"PROMOTION AGREEMENT","confidence":1}]}')

    collected = asyncio.run(collect_graphpo(task(), runner, policy, tokenizer=TOKENIZER,
                                            summarize=summarize, embed=embed, branching=(2,)))
    result = collected.results[0]
    assert result.metrics["verbatim_span_fraction"] == 1
    assert result.metrics["source_token_span_fraction"] == 1
    assert result.metrics["f1"] == 1
    assert result.metrics["observed_token_span_fraction"] == 0
    assert result.reward == 0
    assert all(path["binary_success"] == 0 for path in collected.diagnostics["terminal_paths"])
