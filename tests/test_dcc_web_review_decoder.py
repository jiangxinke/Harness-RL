"""No-engine bounds, language-equivalence and opt-in routing contracts."""

import importlib.metadata
import json
import sys
from functools import lru_cache
from types import SimpleNamespace

import pytest

xgrammar = pytest.importorskip("xgrammar")

from slime_plugins.dcc import vllm_policy
from slime_plugins.dcc import web_review_decoder as decoder
from slime_plugins.dcc.web_review import web_explicit_review_configuration, web_review_schema
from slime_plugins.dcc.web_structured import web_action_schema, web_final_schema

URL = "https://example.org/read"


@lru_cache(maxsize=8)
def pair(state="ready"):
    if state == "final":
        base = web_final_schema(["name", "value"], [URL])
    elif state == "unread":
        base = web_final_schema(["name", "value"], [])
    else:
        base = web_action_schema(["name", "value"], [URL] if state == "ready" else [],
                                 successful_search=state == "ready")
    schema = web_review_schema(base)
    info = xgrammar.TokenizerInfo([bytes([i]) for i in range(256)] + [b"<eos>"], stop_token_ids=[256])
    compiler = xgrammar.GrammarCompiler(info, max_threads=1)
    return (compiler.compile_json_schema(schema), compiler.compile_grammar(decoder.web_review_grammar(schema)))


def accepted(compiled, text):
    matcher = xgrammar.GrammarMatcher(compiled, terminate_without_stop_token=True)
    return matcher.accept_string(text) and matcher.is_terminated()


def assert_pair(obj=None, *, text=None, expected, state="ready"):
    if text is None:
        # Constant citation URLs use escaped slash spellings in the converter.
        text = json.dumps(obj, ensure_ascii=False).replace("/", "\\/")
    observed = [accepted(grammar, text) for grammar in pair(state)]
    assert observed == [expected, expected]


@pytest.mark.parametrize("field", ["plan", "checks"])
@pytest.mark.parametrize("char", ["a", "中", "🙂"])
@pytest.mark.parametrize("length", [0, 1, 239, 240, 241])
def test_actual_xgrammar_review_character_boundaries(field, char, length):
    obj = {"plan": "p", "checks": "c", "tool": "web_search", "arguments": {"query": "q"}}
    obj[field] = char * length
    assert_pair(obj, expected=1 <= length <= 240, state="searchfirst")


@pytest.mark.parametrize("field", ["plan", "checks"])
@pytest.mark.parametrize("escaped", [r'a\"b', r'a\\b', r'a\nb', r'\u4e2d', r'\ud83d\ude42'])
def test_pinned_converter_escape_language_is_preserved(field, escaped):
    obj = {"plan": "p", "checks": "c", "tool": "web_search", "arguments": {"query": "q"}}
    text = json.dumps(obj).replace('"' + field + '": "' + obj[field] + '"',
                                   '"' + field + '": "' + escaped + '"')
    assert_pair(text=text, expected=False, state="searchfirst")


@pytest.mark.parametrize("value,expected", [("x", True), (0, True), (False, True), ("", False), (None, False)])
def test_row_nonempty_and_zero_false_semantics_are_unchanged(value, expected):
    assert_pair({"plan": "p", "checks": "c", "answer": [
        {"name": value, "value": None, "citations": [URL]}]}, expected=expected)


@pytest.mark.parametrize("citations", [[], [URL, URL], ["https://example.org/unread"]])
def test_citations_remain_one_prior_read_url(citations):
    assert_pair({"plan": "p", "checks": "c", "answer": [
        {"name": "x", "value": None, "citations": citations}]}, expected=False)


@pytest.mark.parametrize("state,kind,expected", [
    ("searchfirst", "search", True), ("searchfirst", "fetch", False), ("searchfirst", "answer", False),
    ("ready", "search", True), ("ready", "fetch", True), ("ready", "answer", True),
    ("final", "search", False), ("final", "fetch", False), ("final", "answer", True),
    ("unread", "search", False), ("unread", "fetch", False), ("unread", "answer", True),
])
def test_search_first_and_final_only_branches_remain_exact(state, kind, expected):
    action = {"search": {"tool": "web_search", "arguments": {"query": "q", "count": 8}},
              "fetch": {"tool": "fetch_url", "arguments": {"url": URL, "start": 0}},
              "answer": {"answer": []}}[kind]
    assert_pair({"plan": "p", "checks": "c", **action}, expected=expected, state=state)


@pytest.mark.parametrize("damage", ["missing-plan", "missing-checks", "extra-key", "tool-and-answer", "wrong-order"])
def test_review_single_action_and_order_rejections(damage):
    obj = {"plan": "p", "checks": "c", "tool": "web_search", "arguments": {"query": "q"}}
    if damage.startswith("missing-"):
        del obj[damage.removeprefix("missing-")]
    elif damage == "extra-key":
        obj["extra"] = "bad"
    elif damage == "tool-and-answer":
        obj["answer"] = []
    else:
        obj = {"checks": "c", "plan": "p", "tool": "web_search", "arguments": {"query": "q"}}
    assert_pair(obj, expected=False)


def test_no_other_generated_production_changes():
    schema = web_review_schema(web_action_schema(["name", "value"], [URL], successful_search=True))
    original = str(xgrammar.Grammar.from_json_schema(schema)).splitlines()
    proposed = decoder.web_review_grammar(schema).splitlines()
    retained = [line for line in proposed if not line.startswith(decoder._PREFIX)]
    for old, new in zip(original, retained, strict=True):
        assert old == new or old.startswith("string ::=") and new.startswith("string ::=")


def test_unreviewed_converter_version_fails_closed(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.1.26")
    with pytest.raises(ValueError, match="xgrammar==0.1.25"):
        decoder.web_review_grammar(web_review_schema(web_final_schema(["name"], [])))


def test_converter_shape_change_fails_closed(monkeypatch):
    decoder._cached_review_grammar.cache_clear()
    monkeypatch.setattr(decoder, "_EXPECTED_STANDALONE", decoder._EXPECTED_STANDALONE.replace("string_repeat_1", "renamed"))
    with pytest.raises(ValueError, match="converter shape"):
        decoder.web_review_grammar(web_review_schema(web_final_schema(["name"], [])))
    decoder._cached_review_grammar.cache_clear()


@pytest.mark.parametrize("change", ["renamed-helper", "missing-helper", "prefix-collision"])
def test_full_converter_naming_and_reserved_prefix_fail_closed(monkeypatch, change):
    original_converter = xgrammar.Grammar.from_json_schema
    schema = web_review_schema(web_final_schema(["name"], []))
    class ChangedGrammar:
        def __init__(self, text):
            self.text = text
        def __str__(self):
            return self.text
    def convert(value, **kwargs):
        converted = original_converter(value, **kwargs)
        if value == decoder._REVIEW_FIELD:
            return converted
        text = str(converted)
        if change == "renamed-helper":
            text = text.replace("string_repeat_1", "renamed_string_repeat_1")
        elif change == "missing-helper":
            text = "\n".join(line for line in text.splitlines() if not line.startswith("string_repeat_5 ::=")) + "\n"
        else:
            text += '\ndcc_review_finite_collision ::= ""\n'
        return ChangedGrammar(text)
    monkeypatch.setattr(xgrammar.Grammar, "from_json_schema", convert)
    decoder._cached_review_grammar.cache_clear()
    with pytest.raises(ValueError, match="naming, shape or prefix collision"):
        decoder.web_review_grammar(schema)
    decoder._cached_review_grammar.cache_clear()


@pytest.mark.parametrize("change", ["missing", "bounds", "order"])
def test_review_wrapper_shape_change_fails_closed(change):
    schema = web_review_schema(web_final_schema(["name"], []))
    if change == "missing":
        del schema["properties"]["plan"]
    elif change == "bounds":
        schema["properties"]["checks"]["maxLength"] = 241
    else:
        schema["required"] = ["checks", "plan", "answer"]
    with pytest.raises(ValueError, match="review wrapper"):
        decoder.web_review_grammar(schema)


@pytest.mark.parametrize("enabled,review", [(False, False), (False, True), (True, False), (True, True)])
def test_policy_routes_only_opted_in_review_schemas(monkeypatch, enabled, review):
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", SimpleNamespace(
        StructuredOutputsParams=lambda **kwargs: kwargs))
    policy = object.__new__(vllm_policy.VLLMPolicy)
    policy.web_structured_final = True
    policy.web_bounded_whitespace = False
    policy.provenance = {"web_explicit_review": web_explicit_review_configuration(True)} if enabled else {}
    base = web_final_schema(["name"], [])
    schema = web_review_schema(base) if review else base
    result = policy._structured_options(schema)["structured_outputs"]
    assert result["disable_fallback"] is True
    if enabled and review:
        assert set(result) == {"grammar", "disable_fallback"}
        assert decoder._PREFIX in result["grammar"]
    else:
        assert set(result) == {"json", "disable_fallback"}
        assert result["json"] is schema


def test_original_schema_validator_runs_before_review_decoder(monkeypatch):
    def reject(schema):
        raise ValueError("original schema validation failed")
    monkeypatch.setattr(vllm_policy, "validate_web_structured_schema", reject)
    monkeypatch.setattr(vllm_policy, "web_review_grammar", lambda schema: pytest.fail("decoder must not run"))
    policy = object.__new__(vllm_policy.VLLMPolicy)
    policy.web_structured_final = True
    policy.web_bounded_whitespace = False
    policy.provenance = {"web_explicit_review": web_explicit_review_configuration(True)}
    with pytest.raises(ValueError, match="original schema validation failed"):
        policy._structured_options(web_review_schema(web_final_schema(["name"], [])))


def test_decoder_configuration_is_separate_from_prompt_and_schema_versions():
    assert decoder.web_review_decoder_configuration()["version"] == "web-review-finite-right-recursive-v1"
    assert web_explicit_review_configuration(True)["schema_version"] == "web-explicit-plan-checks-v1"


def test_review_decoder_api_preflight_does_not_create_an_engine(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", SimpleNamespace(
        StructuredOutputsParams=lambda **kwargs: calls.append(kwargs)))
    configuration = decoder.preflight_web_review_decoder()
    assert configuration["version"] == decoder.WEB_REVIEW_DECODER_VERSION
    assert len(calls) == 9
    assert all(set(call) == {"grammar", "disable_fallback"} and call["disable_fallback"]
               and decoder._PREFIX in call["grammar"] for call in calls)


def test_review_decoder_preflight_fails_before_gpu_environment_validation(monkeypatch):
    monkeypatch.setattr(vllm_policy, "preflight_web_structured", lambda **kwargs: None)
    def reject():
        raise ValueError("decoder preflight failed")
    monkeypatch.setattr(vllm_policy, "preflight_web_review_decoder", reject)
    monkeypatch.setattr(vllm_policy, "_validate_launch_environment", lambda *args: pytest.fail("GPU check must not run"))
    with pytest.raises(ValueError, match="decoder preflight failed"):
        vllm_policy.load_vllm_policy("synthetic/model", web_structured_final=True,
                                     web_structured_actions=True, web_explicit_review=True)
