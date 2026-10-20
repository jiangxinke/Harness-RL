"""No-engine bounded whitespace semantics, opt-in routing and audit contracts."""
import asyncio
import copy
import importlib.metadata
import json
import sys
import runpy
from pathlib import Path
from functools import lru_cache
from types import SimpleNamespace

import pytest

xgrammar = pytest.importorskip("xgrammar")
from slime_plugins.dcc import vllm_policy
from slime_plugins.dcc import web_whitespace as decoder
from slime_plugins.dcc.web_review import web_explicit_review_configuration, web_review_schema
from slime_plugins.dcc.web_review_decoder import web_review_grammar
from slime_plugins.dcc.web_structured import web_action_schema, web_final_schema, web_structured_configuration

URL = "https://example.org/read"
CONFIG = decoder.web_bounded_whitespace_configuration(True)


def schema_for(review=False, state="ready", columns=None):
    columns = ["name", "value"] if columns is None else columns
    if state in {"final", "unread"}:
        base = web_final_schema(columns, [URL] if state == "final" else [])
    else:
        base = web_action_schema(columns, [URL] if state == "ready" else [], successful_search=state == "ready")
    return web_review_schema(base) if review else base


@lru_cache(maxsize=16)
def grammars(review=False, state="ready"):
    schema = schema_for(review, state)
    info = xgrammar.TokenizerInfo([bytes([i]) for i in range(256)] + [b"<eos>"], stop_token_ids=[256])
    compiler = xgrammar.GrammarCompiler(info, max_threads=1)
    original = web_review_grammar(schema) if review else str(xgrammar.Grammar.from_json_schema(schema))
    return (compiler.compile_grammar(original),
            compiler.compile_grammar(decoder.web_bounded_whitespace_grammar(schema, explicit_review=review)))


def accepted(grammar, text):
    matcher = xgrammar.GrammarMatcher(grammar, terminate_without_stop_token=True)
    return matcher.accept_string(text) and matcher.is_terminated()


def dump(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("/", "\\/")


@pytest.mark.parametrize("review", [False, True])
@pytest.mark.parametrize("whitespace", [" ", "\n", "\t"])
@pytest.mark.parametrize("count", [0, 1, 8, 9, 100])
def test_intertoken_whitespace_is_finite_with_both_review_modes(review, whitespace, count):
    gap = whitespace * count
    prefix = '"plan":"p","checks":"c",' if review else ""
    text = "{" + prefix + gap + '"answer":[]}'
    pair = grammars(review, "final")
    assert accepted(pair[0], text)
    assert accepted(pair[1], text) == (count <= 8)


@pytest.mark.parametrize("review", [False, True])
@pytest.mark.parametrize("value", ["a" + " " * 100 + "b", "中😀 [ \\n\\t]*", "a\nb\tc", "a\"b\\c", 0, False, None])
def test_only_insignificant_whitespace_changes_not_values(review, value):
    obj = {"answer": [{"name": "entity", "value": value, "citations": [URL]}]}
    if review:
        obj = {"plan": "p" + " " * 100 + "q", "checks": "中😀", **obj}
    text = dump(obj)
    assert json.loads(text) == obj
    assert [accepted(g, text) for g in grammars(review, "final")] == [True, True]


@pytest.mark.parametrize("review", [False, True])
@pytest.mark.parametrize("state,kind,expected", [
    ("searchfirst", "search", True), ("searchfirst", "fetch", False), ("searchfirst", "answer", False),
    ("ready", "search", True), ("ready", "fetch", True), ("ready", "answer", True),
    ("final", "search", False), ("final", "fetch", False), ("final", "answer", True),
    ("unread", "search", False), ("unread", "fetch", False), ("unread", "answer", True),
])
def test_actions_and_search_read_final_branches_unchanged(review, state, kind, expected):
    obj = {"search": {"tool": "web_search", "arguments": {"query": "q", "count": 8}},
           "fetch": {"tool": "fetch_url", "arguments": {"url": URL, "start": 0}},
           "answer": {"answer": []}}[kind]
    if review:
        obj = {"plan": "p", "checks": "c", **obj}
    assert [accepted(g, dump(obj)) for g in grammars(review, state)] == [expected, expected]


@pytest.mark.parametrize("review", [False, True])
@pytest.mark.parametrize("damage", ["missing-key", "extra-key", "empty-row", "extra-citation", "unread-url"])
def test_schema_and_citation_rejections_unchanged(review, damage):
    row = {"name": "entity", "value": None, "citations": [URL]}
    if damage == "missing-key": del row["value"]
    elif damage == "extra-key": row["extra"] = "bad"
    elif damage == "empty-row": row["name"] = None
    elif damage == "extra-citation": row["citations"].append(URL)
    else: row["citations"] = ["https://example.org/unread"]
    obj = {"answer": [row]}
    if review: obj = {"plan": "p", "checks": "c", **obj}
    assert [accepted(g, dump(obj)) for g in grammars(review, "final")] == [False, False]


@pytest.mark.parametrize("length", [0, 1, 239, 240, 241])
def test_review_character_bounds_preserved(length):
    obj = {"plan": "中" * length, "checks": "c", "tool": "web_search", "arguments": {"query": "q"}}
    assert [accepted(g, dump(obj)) for g in grammars(True, "searchfirst")] == [1 <= length <= 240] * 2


@pytest.mark.parametrize("escaped", [r'a\"b', r'a\\b', r'a\nb', r'\u4e2d'])
def test_pinned_review_escape_limitation_remains_unchanged(escaped):
    text = '{"plan":"' + escaped + '","checks":"c","tool":"web_search","arguments":{"query":"q"}}'
    assert [accepted(g, text) for g in grammars(True, "searchfirst")] == [False, False]


@pytest.mark.parametrize("review", [False, True])
def test_every_non_whitespace_production_is_retained_exactly(review):
    schema = schema_for(review)
    original = web_review_grammar(schema) if review else str(xgrammar.Grammar.from_json_schema(schema))
    bounded = decoder.web_bounded_whitespace_grammar(schema, explicit_review=review)
    retained = "\n".join(line for line in bounded.splitlines() if not line.startswith(decoder._PREFIX)) + "\n"
    assert retained.replace(decoder._PREFIX + "8", decoder._WHITESPACE_CLASS + "*") == original


def test_grammar_literals_that_look_like_whitespace_are_not_rewritten():
    grammar = 'root ::= "[ \\n\\t]*" [ \\n\\t]* "x"\n'
    bounded = decoder._bound_whitespace(grammar)
    assert bounded.startswith('root ::= "[ \\n\\t]*" dcc_json_ws_8 "x"\n')


@pytest.mark.parametrize("review", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_policy_routes_opt_in_for_off_and_review_without_changing_defaults(monkeypatch, review, enabled):
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", SimpleNamespace(StructuredOutputsParams=lambda **kwargs: kwargs))
    policy = vllm_policy.VLLMPolicy(None, SimpleNamespace(eos_token_id=0), web_structured_final=True,
                                  web_structured_actions=True, web_bounded_whitespace=enabled,
                                  provenance={"web_explicit_review": web_explicit_review_configuration(True)} if review else {})
    options = policy._structured_options(schema_for(review))["structured_outputs"]
    assert options["disable_fallback"] is True
    if enabled:
        assert "grammar" in options and decoder._PREFIX in options["grammar"]
        assert policy.provenance["web_bounded_whitespace"] == CONFIG
    else:
        assert "web_bounded_whitespace" not in policy.provenance
        assert ("grammar" in options) == review
    if review: assert "dcc_review_finite_" in options["grammar"]


@pytest.mark.parametrize("value", [None, 0, 1, "true", []])
def test_nonboolean_switch_is_rejected(value):
    with pytest.raises(ValueError, match="boolean"):
        decoder.web_bounded_whitespace_configuration(value)


def test_optional_runtime_not_needed_for_disabled_provenance(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: pytest.fail("Must not load runtime for score-only"))
    decoder.validate_web_whitespace_records([{}, {}])
    records = [{"web_bounded_whitespace": CONFIG, "web_structured_final": web_structured_configuration(True)}] * 3
    decoder.validate_web_whitespace_records(records)


@pytest.mark.parametrize("damage", ["missing", "bound", "disabled", "structured"])
def test_incomplete_mixed_or_changed_provenance_fails_closed(damage):
    records = [{"web_bounded_whitespace": copy.deepcopy(CONFIG), "web_structured_final": web_structured_configuration(True)} for _ in range(3)]
    if damage == "missing": records[1].pop("web_bounded_whitespace")
    elif damage == "bound": records[1]["web_bounded_whitespace"]["max_whitespace_chars"] = 9
    elif damage == "disabled": records[1]["web_bounded_whitespace"] = decoder.web_bounded_whitespace_configuration(False)
    else: records[1].pop("web_structured_final")
    with pytest.raises(ValueError, match="bounded whitespace"):
        decoder.validate_web_whitespace_records(records)


def test_converter_version_and_changed_grammar_shape_fail_closed(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.1.26")
    with pytest.raises(ValueError, match="xgrammar==0.1.25"):
        decoder.web_bounded_whitespace_grammar(schema_for())
    for grammar in ['root ::= "literal"\n', 'root ::= [ \\r\\n\\t]*\n', 'root ::= dcc_json_ws_collision\n', 'root ::= [ \\n\\t]* [ \\r\\n\\t]*\n']:
        with pytest.raises(ValueError): decoder._bound_whitespace(grammar)


def test_review_provenance_mismatch_fails_closed():
    with pytest.raises(ValueError, match="matching explicit-review"):
        decoder.web_bounded_whitespace_grammar(schema_for(True))
    with pytest.raises(ValueError, match="matching explicit-review"):
        decoder.web_bounded_whitespace_grammar(schema_for(), explicit_review=True)


def test_opt_in_requires_structured_web_final_before_engine_launch():
    with pytest.raises(ValueError, match="requires web_structured_final"):
        vllm_policy.load_vllm_policy("unused", web_bounded_whitespace=True)
    with pytest.raises(ValueError, match="requires web_structured_final"):
        vllm_policy.VLLMPolicy(None, SimpleNamespace(eos_token_id=0), web_bounded_whitespace=True)


@pytest.mark.parametrize("review", [False, True])
def test_loader_preflights_bounded_mode_before_constructing_engine(monkeypatch, review):
    root = Path(__file__).parents[1]
    helpers = runpy.run_path(str(root / "tests/test_dcc_vllm_policy.py"))
    helpers["runtime"].__wrapped__(monkeypatch)
    import transformers
    calls = []
    monkeypatch.setattr(vllm_policy, "preflight_web_structured", lambda **kwargs: None)
    monkeypatch.setattr(vllm_policy, "preflight_web_review_decoder", lambda: {"version": "finite"})
    monkeypatch.setattr(vllm_policy, "preflight_web_bounded_whitespace", lambda **kwargs: calls.append(kwargs) or CONFIG)
    monkeypatch.setitem(sys.modules, "vllm.config", SimpleNamespace(StructuredOutputsConfig=lambda **kwargs: SimpleNamespace(**kwargs)))
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **kw: SimpleNamespace(model_type="qwen3_moe"))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: helpers["Tokenizer"]())
    monkeypatch.setattr(transformers.GenerationConfig, "from_pretrained", lambda *a, **kw: SimpleNamespace(eos_token_id=1))
    policy = vllm_policy.load_vllm_policy("synthetic/model", web_structured_final=True,
        web_structured_actions=True, web_explicit_review=review, web_bounded_whitespace=True)
    assert calls == [{"explicit_review": review}]
    assert policy.web_bounded_whitespace is True
    assert policy.provenance["web_bounded_whitespace"] == CONFIG


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_manifest_runtime_loader_and_resume_preserve_opt_in(monkeypatch, tmp_path, enabled):
    from slime_plugins.dcc import PolicyCompletion, Task
    helpers = runpy.run_path(str(Path(__file__).parents[1] / "tests/test_dcc_model_scoring.py"))
    class Policy:
        provenance = {"backend": "vllm", "model": "example/model"}
        async def __call__(self, *args, **kwargs): return PolicyCompletion('{"answer":[]}', [1], [2])
        async def generate_structured(self, *args, **kwargs): return await self(*args, **kwargs)
        async def close(self): pass
    policy = Policy()
    if enabled:
        policy.provenance.update(web_bounded_whitespace=copy.deepcopy(CONFIG), web_structured_final=web_structured_configuration(True))
    task = Task("synthetic", "widesearch", "validation", "Find names", gold=[], metadata={"evaluation": {"required": ["name"]}})
    evaluate, args = helpers["stub_evaluation"](monkeypatch, tmp_path, policy, [task])
    args.max_steps = 1
    args.web_bounded_whitespace = enabled
    args.web_structured_final = enabled
    seen = []
    def load(*args, **kwargs):
        seen.append(kwargs)
        return policy
    monkeypatch.setitem(sys.modules, "slime_plugins.dcc.vllm_policy", SimpleNamespace(load_vllm_policy=load))
    asyncio.run(evaluate(args))
    for name in ["manifest_rank0.json", "inference_runtime.json"]:
        record = json.loads((tmp_path / name).read_text())
        if enabled: assert record["web_bounded_whitespace"] == CONFIG
        else: assert "web_bounded_whitespace" not in record
    assert bool(seen[0].get("web_bounded_whitespace")) is enabled
    if enabled:
        args.web_bounded_whitespace = False
        with pytest.raises(ValueError, match="manifest|settings|configuration"):
            asyncio.run(evaluate(args))


@pytest.mark.parametrize("damage", [None, "manifest", "runtime", "protocol", "bound"])
def test_paired_protocol_validates_matching_bounded_mode(tmp_path, damage):
    helpers = runpy.run_path(str(Path(__file__).parents[1] / "tests/test_dcc_compare_runs.py"))
    values = helpers["pair"](tmp_path, [helpers["web"]("synthetic")])
    for run, declaration in zip(values[:2], values[2:], strict=True):
        for record in (run["manifest"], run["runtime"], declaration["common"]["generation"]):
            record.update(web_bounded_whitespace=copy.deepcopy(CONFIG), web_structured_final=web_structured_configuration(True))
    if damage in {"manifest", "runtime"}: values[0][damage].pop("web_bounded_whitespace")
    elif damage == "protocol": values[2]["common"]["generation"].pop("web_bounded_whitespace")
    elif damage == "bound": values[0]["manifest"]["web_bounded_whitespace"]["max_whitespace_chars"] = 9
    if damage:
        with pytest.raises(ValueError, match="[Ww]eb bounded whitespace"):
            helpers["COMPARE"]["validate_protocols"](*values)
    else: helpers["COMPARE"]["validate_protocols"](*values)


def test_score_only_requires_matching_bounded_runtime(tmp_path):
    helpers = runpy.run_path(str(Path(__file__).parents[1] / "tests/test_dcc_model_scoring.py"))
    tasks = helpers["logical_predictions"](tmp_path)
    for path in tmp_path.glob("manifest_rank*.json"):
        record = json.loads(path.read_text())
        record.update(web_bounded_whitespace=copy.deepcopy(CONFIG), web_structured_final=web_structured_configuration(True))
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="[Ww]eb bounded whitespace"):
        helpers["cli"]()["_score_predictions"](tmp_path, tasks)


@pytest.mark.parametrize("review", [False, True])
def test_all_legal_json_token_gaps_reject_ninth_whitespace_character(review):
    obj = {"answer": [{"name": "中😀", "value": "space inside string", "citations": [URL]}]}
    if review: obj = {"plan": "p", "checks": "c", **obj}
    canonical = dump(obj)
    original, bounded = grammars(review, "final")
    tested = 0
    for position in range(len(canonical) + 1):
        candidate = canonical[:position] + " " * 9 + canonical[position:]
        try: parsed = json.loads(candidate)
        except ValueError: continue
        if parsed != obj or not accepted(original, candidate): continue
        assert not accepted(bounded, candidate), position
        tested += 1
    assert tested >= 15
