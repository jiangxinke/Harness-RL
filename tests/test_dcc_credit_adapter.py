"""Real runner boundaries and archived failures in the credit adapter."""
import asyncio
import json
from pathlib import Path

import pytest

from slime_plugins.dcc.credit_adapter import build_credit_visits, public_observation
from slime_plugins.dcc.cuad_audit_protocol import parse_action, source_grounded
from slime_plugins.dcc.data import Task
from slime_plugins.dcc.episode import EpisodeRunner, EpisodeResult, PolicyCompletion, final_turn_message


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(json.dumps(messages, sort_keys=True).encode())


def run_script(outputs, *, protocol="cuad-audited-v1", gold=None, document=None):
    task = Task("example__Document Name", "cuad", "train", "What is the contract name?",
                document or "PROMOTION AGREEMENT. Details of the agreement follow.", gold or ["PROMOTION AGREEMENT"])
    class Policy:
        def __init__(self): self.index = 0
        async def __call__(self, messages, **kwargs):
            value = outputs[self.index]
            self.index += 1
            return PolicyCompletion(value, Tokenizer().apply_chat_template(messages), [self.index])
    return asyncio.run(EpisodeRunner(max_steps=len(outputs), initial_chunks=1,
                                    cuad_protocol=protocol).run(task, Policy()))


def test_actual_runner_combined_final_prompt_keeps_evidence():
    outputs = ['{"tool":"search_document","arguments":{"query":"agreement"}}',
               '{"answer":[{"text":"PROMOTION AGREEMENT","confidence":1}]}']
    for protocol in ("legacy", "cuad-audited-v1"):
        result = run_script(outputs, protocol=protocol)
        visits, _ = build_credit_visits([result], tokenizer=Tokenizer())
        assert visits[-1].observation.startswith("final\n")
        assert "PROMOTION AGREEMENT" in visits[-1].observation
        assert "Final turn:" not in visits[-1].observation
        assert visits[0].next_observation == visits[1].observation


def test_final_controls_cannot_merge_different_source_observations():
    task = Task("a", "cuad", "train", "q", "contract", [])
    base = [{"role":"system","content":"s"}]
    def observation(text):
        return public_observation(base + [{"role":"user", "content":
            'Tool observation (source material):\n' + json.dumps({"text":text}) + '\nRemaining turns: 1.'}, final_turn_message(task)])
    assert observation("evidence A") != observation("evidence B")


def test_invalid_action_is_attached_to_the_decision_that_caused_it():
    result = run_script(['{"tool":"search_document","arguments":{"query":"agreement","index":3}}',
                         '{"answer":[{"text":"PROMOTION AGREEMENT","confidence":1}]}'])
    visits, _ = build_credit_visits([result], tokenizer=Tokenizer())
    assert [v.is_action_valid for v in visits] == [False, True]
    assert [v.task_reward for v in visits] == [10, 10]
    assert [v.terminal for v in visits] == [False, True]


def test_repeat_observation_ignores_remaining_turns_but_not_action_phase():
    result = run_script(['{"tool":"search_document","arguments":{"query":"agreement"}}'] * 3
                         + ['{"answer":[{"text":"PROMOTION AGREEMENT","confidence":1}]}'])
    visits, _ = build_credit_visits([result], tokenizer=Tokenizer())
    assert visits[1].observation == visits[2].observation
    assert visits[2].observation != visits[3].observation


def test_whitespace_near_punctuation_is_not_a_zero_reward_failure():
    document = "Upon Change of Control, CONSULTANT shall receive compensation."
    span = "Upon Change of Control,CONSULTANT shall receive compensation."
    result = run_script([json.dumps({"answer":[{"text":span,"confidence":1}]})], gold=[document], document=document)
    assert result.metrics["f1"] == 1
    assert result.metrics["verbatim_span_fraction"] == 0
    assert result.metrics["source_token_span_fraction"] == 1
    assert result.reward == 1
    assert not source_grounded("Upon Change ofControl, CONSULTANT shall receive compensation.", document)
    assert not source_grounded("CONSULTANT shall not receive compensation.", document)


def test_unobserved_correct_title_does_not_count_as_grounded_success():
    document = ("contract name information. " * 500) + "SECRET AGREEMENT"
    result = run_script(['{"answer":[{"text":"SECRET AGREEMENT","confidence":1}]}'],
                        gold=["SECRET AGREEMENT"], document=document)
    assert "SECRET AGREEMENT" not in result.messages[2]["content"]
    assert result.metrics["f1"] == 1
    assert result.metrics["source_token_span_fraction"] == 1
    assert result.metrics["observed_token_span_fraction"] == 0
    assert result.reward == 0


def test_v2_accepts_only_unambiguous_final_answer_array():
    value = '[{"text":"PROMOTION AGREEMENT","confidence":1}]'
    assert parse_action(value) is None
    assert parse_action(value, allow_answer_array=True)["answer"][0]["text"] == "PROMOTION AGREEMENT"
    assert parse_action('[]', allow_answer_array=True) == {"answer": []}
    assert parse_action('["initial source", "tool request"]', allow_answer_array=True) is None
    assert parse_action('[{"tool":"search_document","arguments":{"query":"A"}}]', allow_answer_array=True) is None
    result = run_script([value], protocol="cuad-audited-v2")
    assert result.status == "completed" and result.reward == 1


def test_v2_task_wording_uses_public_question_not_gold():
    from slime_plugins.dcc.cuad_audit_protocol import task_prompt
    task = Task("contract__Document Name", "cuad", "train",
        'Highlight parts related to "Document Name" reviewed by a lawyer. Details: The name of the contract',
        "SECRET AGREEMENT", ["SECRET AGREEMENT"])
    prompt = task_prompt(task)
    assert "The name of the contract" in prompt
    assert "reviewed by a lawyer" not in prompt
    assert "SECRET AGREEMENT" not in prompt


@pytest.mark.parametrize("raw, expected", [
    ('Search: {"tool":"search_document","arguments":{"query":"agreement"}}', "search_document"),
    ('Read: {"tool":"read_document","arguments":{"start":0,"end":40}}', "read_document"),
    ('Finish: {"answer":[{"text":"PROMOTION AGREEMENT","confidence":1}]}', "answer"),
    ('```json\n{"answer":[]}\n```', "answer"),
    ('Finish: []', "answer"),
    ('Search {"tool":"search_document","arguments":{"query":"agreement"}}', "search_document"),
])
def test_v3_accepts_unambiguous_wrappers_without_rewriting_sample(raw, expected):
    assert parse_action(raw, allow_answer_array=True) is None
    action = parse_action(raw, allow_answer_array=True, allow_wrappers=True)
    assert ("answer" if "answer" in action else action["tool"]) == expected
    if expected == "answer":
        result = run_script([raw], protocol="cuad-audited-v3")
        assert result.status == "completed"
        assert result.steps[0].text == raw
        assert result.messages[-1]["content"] == raw
        assert result.metrics["protocol_aliases"][0]["step"] == 0


@pytest.mark.parametrize("raw", [
    'Search: {"answer":[]}',
    'Finish: {"tool":"search_document","arguments":{"query":"agreement"}}',
    'Read: {"tool":"search_document","arguments":{"query":"agreement"}}',
    'Here is my answer: {"answer":[]}',
    'Finish: {"answer":[]} {"answer":[]}',
    'Finish: {"answer":[],"answer":[]}',
    '```json\n{"answer":[]}\n``` trailing',
    'Search: [{"tool":"search_document","arguments":{"query":"agreement"}}]',
])
def test_v3_does_not_guess_mixed_or_conflicting_actions(raw):
    assert parse_action(raw, allow_answer_array=True, allow_wrappers=True) is None


def test_v3_examples_do_not_demonstrate_rejected_prefixes():
    from slime_plugins.dcc.cuad_audit_protocol import system_prompt
    old = system_prompt(8)
    new = system_prompt(8, version="cuad-audited-v3")
    assert all(label in old and label not in new for label in ("Search: ", "Read: ", "Finish: "))


@pytest.mark.parametrize("text", [
    '["initial retrieval", "tool request"]',
    '{"answer":[],"answer":[{"text":"A","confidence":1}]}',
    '{"answer":[{"text":"A","confidence":NaN}]}',
    '{"answer":[{"text":"A","confidence":true}]}',
    '{"answer":[],"tool":"search_document"}',
    '{"tool":"search_document","arguments":{"query":"A","index":3}}',
])
def test_invalid_protocol_is_not_silently_repaired_into_an_answer(text):
    assert parse_action(text) is None


def test_real_archived_final_states_and_prompt_mismatch():
    path = Path(__file__).parent / "fixtures/dcc_credit_regressions.json"
    observations = []
    for entry in json.loads(path.read_text(encoding="utf-8")):
        row = entry["record"]
        result = EpisodeResult(**{**row, "steps":[PolicyCompletion(**s) for s in row["steps"]]})
        visits, _ = build_credit_visits([result])
        assert visits[-1].observation.startswith("final\n")
        assert "Final turn:" not in visits[-1].observation
        observations.append(visits[-1].observation)
        with pytest.raises(ValueError, match="prompt differs"):
            build_credit_visits([result], tokenizer=Tokenizer())
    assert len(set(observations)) >= 3
