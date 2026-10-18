"""The terminal bare-object compatibility path must fail closed."""

import pytest

from slime_plugins.dcc.widesearch_metrics import JudgeError, _judge_json, _verify_entity_identity


@pytest.mark.parametrize("response", [
    '{"candidate_id": 0}',
    'The origin and reference are the same.\n{"candidate_id": 0}',
    'The "origin" has a recognized alias.\n{"candidate_id": 0}\n  ',
    'A balanced "escaped \\" quote" supports the choice.\n{"candidate_id": 0}',
    'Reasoning first.\n{\n  "candidate_id": 0\n}',
    'Reasoning first.\n```json\n{"candidate_id": 0}\n```',
])
def test_single_complete_final_object_is_accepted(response):
    assert _judge_json(lambda prompt: response, "") == {"candidate_id": 0}


def test_existing_fenced_path_still_uses_last_json_fence():
    response = '```json\n{"candidate_id": null}\n```\nThen:\n```json\n{"candidate_id": 0}\n```'
    assert _judge_json(lambda prompt: response, "") == {"candidate_id": 0}


@pytest.mark.parametrize("response", [
    'Reasoning {"candidate_id": 0} trailing garbage',
    'Reasoning {"candidate_id": 0} {"candidate_id": 1}',
    '{}\n{"candidate_id": 0}',
    'Reasoning ["data"]\n{"candidate_id": 0}',
    'Reasoning ]\n{"candidate_id": 0}',
    'Reasoning }\n{"candidate_id": 0}',
    'Reasoning "unterminated\n{"candidate_id": 0}',
    '[{"candidate_id": 0}]',
    '[malformed, {"candidate_id": 0}',
    '{"outer": [{"candidate_id": 0}',
    'Reasoning {"outer": invalid, "inner": {"candidate_id": 0}',
    'Reasoning {"candidate_id": 0',
    'Reasoning [\n{"candidate_id": 0}',
    'Reasoning without JSON',
    '"a string containing {braces}"',
    'null',
    '0',
])
def test_invalid_outer_structure_cannot_be_salvaged_as_inner_object(response):
    with pytest.raises(JudgeError, match="Malformed"):
        _judge_json(lambda prompt: response, "")


def test_new_parse_path_still_enforces_verifier_schema():
    with pytest.raises(JudgeError, match="invalid schema"):
        _verify_entity_identity("Solstice Lab", "SL", lambda prompt: 'Reasoning.\n{"same_entity":"YES","extra":true}')


def test_judge_exception_identity_and_cause_are_retained():
    original = OSError("synthetic transport failure")

    def judge(prompt):
        raise original

    with pytest.raises(JudgeError) as caught:
        _judge_json(judge, "")
    assert caught.value.__cause__ is original
