"""Real CPU grammar regressions using a byte tokenizer; no model or engine."""
from copy import deepcopy
import json

import pytest
from jsonschema import Draft202012Validator, ValidationError

from slime_plugins.dcc.cuad_structured import (
    cuad_action_schema, cuad_final_schema, cuad_backend_schema, cuad_structured_configuration,
)

EOS = 256


@pytest.fixture(scope="module")
def compiled():
    xg = pytest.importorskip("xgrammar")
    info = xg.TokenizerInfo([bytes([i]) for i in range(256)] + [b"<eos>"], stop_token_ids=[EOS])
    compiler = xg.GrammarCompiler(info, max_threads=1)
    return xg, {kind: compiler.compile_json_schema(cuad_backend_schema(factory()))
                for kind, factory in (("action", cuad_action_schema), ("final", cuad_final_schema))}


def accepts_eos(compiled, kind, text):
    xg, grammars = compiled
    matcher = xg.GrammarMatcher(grammars[kind])
    return all(matcher.accept_token(value) for value in text.encode("utf-8")) and matcher.accept_token(EOS)


@pytest.mark.parametrize("kind,value", [
    ("action", {"answer": [{"text": "Alpha", "confidence": .9}]}),
    ("action", {"answer": []}),
    ("action", {"tool": "search_document", "arguments": {"query": "example"}}),
    ("action", {"tool": "read_document", "arguments": {"start": 0, "end": None}}),
    ("action", {"answer": [{"text": 'A "quote" \\ path\nnext\t中文 α', "confidence": 1}]}),
    ("final", {"plan": "Read the document.", "checks": "Check the span.", "answer": []}),
    ("final", {"plan": "One\ntwo", "checks": 'Check "quotes" and \\ paths.',
               "answer": [{"text": "Alpha\nBeta", "confidence": 0}]}),
])
def test_complete_public_valid_json_reaches_eos(compiled, kind, value):
    public = cuad_action_schema() if kind == "action" else cuad_final_schema()
    Draft202012Validator(public).validate(value)
    assert accepts_eos(compiled, kind, json.dumps(value, ensure_ascii=False, separators=(",", ":")))


@pytest.mark.parametrize("kind,text", [
    ("action", '{"answer":[{"text":"A\nB","confidence":1}]}'),
    ("action", '{"answer":[{"text":"A\tB","confidence":1}]}'),
    ("action", '{"answer":[{"text":"A\\qB","confidence":1}]}'),
    ("action", '{"answer":[{"text":"Alpha"}]}'),
    ("action", '{"answer":[{"text":"Alpha","confidence":1,"extra":1}]}'),
    ("action", '{"answer":[{"text":"Alpha","confidence":1}]'),
    ("action", '{"answer":[]}{"answer":[]}'),
    ("action", '{"tool":"read_document"}'),
    ("final", '{"plan":"Read","answer":[]}'),
    ("final", '{"plan":"Read","checks":"Check","answer":[]}{}'),
])
def test_raw_controls_incomplete_json_and_invalid_shape_cannot_finish(compiled, kind, text):
    assert not accepts_eos(compiled, kind, text)


@pytest.mark.parametrize("value", [
    {"plan": "Read", "checks": "Check", "answer": [{"text": "   ", "confidence": 1}]},
    {"plan": "word " * 41, "checks": "Check", "answer": []},
    {"plan": "Read", "checks": "", "answer": []},
])
def test_patterns_remain_public_postvalidation_constraints(compiled, value):
    assert accepts_eos(compiled, "final", json.dumps(value, separators=(",", ":")))
    with pytest.raises(ValidationError):
        Draft202012Validator(cuad_final_schema()).validate(value)


def test_backend_schema_is_separate_and_metadata_discloses_lowering():
    public = cuad_final_schema(); snapshot = deepcopy(public)
    physical = cuad_backend_schema(public)
    assert public == snapshot
    assert "pattern" in public["properties"]["plan"]
    assert "pattern" not in physical["properties"]["plan"]
    assert physical["properties"]["answer"]["items"]["properties"]["confidence"] == {
        "type": "number", "minimum": 0, "maximum": 1}
    physical.clear()
    assert public == snapshot
    config = cuad_structured_configuration(True)
    assert config["backend_schema_matches_public"] is False
    assert config["original_public_schema_postvalidation"] is True
    assert config["backend_schema_policy"] == "structural-schema-without-string-patterns"
    altered = cuad_action_schema(); altered["anyOf"].clear()
    with pytest.raises(ValueError): cuad_backend_schema(altered)
