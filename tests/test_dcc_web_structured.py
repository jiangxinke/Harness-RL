"""Optional final Web schemas constrain format, without supplying answer values."""

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from slime_plugins.dcc import EpisodeRunner, Task
from slime_plugins.dcc.episode import PolicyCompletion
from slime_plugins.dcc.web_structured import (
    validate_web_structured_schema,
    web_action_schema,
    web_final_schema,
    web_structured_actions_configuration,
    web_structured_configuration,
)

URL = "https://example.org/source"


@pytest.mark.parametrize("value", ["Alpha", 3, 1.5, True, False, 0])
def test_schema_accepts_scalar_cells_without_supplying_their_values(value):
    jsonschema = pytest.importorskip("jsonschema")
    schema = web_final_schema(['name"/中文'], {URL})
    jsonschema.validate({"answer": [{'name"/中文': value, "citations": [URL]}]}, schema)
    jsonschema.validate({"answer": []}, schema)
    assert "Alpha" not in json.dumps(schema)


@pytest.mark.parametrize("cells,accepted", [
    ({"name": "", "中文": ""}, False),
    ({"name": None, "中文": ""}, False),
    ({"name": None, "中文": None}, False),
    ({"name": "", "中文": "实体"}, True),
    ({"name": "Alpha", "中文": None}, True),
    ({"name": 0, "中文": ""}, True),
    ({"name": False, "中文": None}, True),
])
def test_nonempty_row_schema_agrees_with_existing_evidence_guard(cells, accepted):
    jsonschema = pytest.importorskip("jsonschema")
    from slime_plugins.dcc.episode import _WebEvidence

    row = {**cells, "citations": [URL]}
    task = Task("web", "widesearch", "validation", "Find names",
                metadata={"evaluation": {"required": ["name", "中文"]}})
    evidence = _WebEvidence()
    evidence.observe("web_search", {"results": [{"url": URL}]})
    evidence.observe("fetch_url", {"url": URL, "text": "Observed source"})
    assert ("empty_row" not in evidence.reject_reasons(task, [row])) == accepted
    schema = web_final_schema(["name", "中文"], {URL})
    if accepted:
        jsonschema.validate({"answer": [row]}, schema)
    else:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"answer": [row]}, schema)


def test_installed_xgrammar_enforces_nonempty_cells_and_preserves_unknown_cells():
    xgrammar = pytest.importorskip("xgrammar")
    from xgrammar.testing import _is_grammar_accept_string

    grammar = xgrammar.Grammar.from_json_schema(web_final_schema(["name", "中文"], {URL}))
    for cells, accepted in [
        ({"name": "", "中文": ""}, False),
        ({"name": None, "中文": ""}, False),
        ({"name": "", "中文": "实体"}, True),
        ({"name": "Alpha", "中文": None}, True),
        ({"name": 0, "中文": ""}, True),
        ({"name": False, "中文": None}, True),
    ]:
        value = {"answer": [{**cells, "citations": [URL]}]}
        # xgrammar's constant URL grammar emits escaped slashes.
        encoded = json.dumps(value, ensure_ascii=False).replace("/", "\\/")
        assert _is_grammar_accept_string(grammar, encoded) == accepted
    assert _is_grammar_accept_string(grammar, '{"answer":[]}')


@pytest.mark.parametrize("answer", [
    [{"name": ["Alpha"], "citations": [URL]}],
    [{"name": {"value": "Alpha"}, "citations": [URL]}],
    [{"name": "Alpha", "citations": []}],
    [{"name": "Alpha", "citations": [URL, URL]}],
    [{"name": "Alpha", "citations": ["https://example.org/unread"]}],
    [{"name": "Alpha", "citations": [URL], "invented": 1}],
    [{"citations": [URL]}],
])
def test_schema_rejects_wrong_cells_columns_and_unread_citations(answer):
    jsonschema = pytest.importorskip("jsonschema")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"answer": answer}, web_final_schema(["name"], {URL}))


def test_no_successful_reads_only_allows_empty_answer():
    jsonschema = pytest.importorskip("jsonschema")
    schema = web_final_schema(["name"], set())
    jsonschema.validate({"answer": []}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"answer": [{"name": "Alpha", "citations": [URL]}]}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"answer": [], "extra": 1}, schema)


@pytest.mark.parametrize("columns,urls", [([], []), (["citations"], []), (["name"], [URL, 2])])
def test_schema_rejects_missing_public_columns_or_invalid_read_urls(columns, urls):
    with pytest.raises(ValueError):
        web_final_schema(columns, urls)


class Web:
    async def web_search(self, query):
        return {"results": [{"url": URL, "title": "Public source"}]}

    async def fetch_url(self, url):
        if url != URL:
            return {"url": url, "error": "unavailable"}
        return {"url": url, "text": "Alpha source evidence", "start": 0, "end": 21}

    def citation_validity(self, rows):
        return 1.0


class Policy:
    def __init__(self, actions):
        self.actions, self.calls = iter(actions), []

    async def __call__(self, messages, **kwargs):
        self.calls.append(("ordinary", messages, kwargs))
        return PolicyCompletion(json.dumps(next(self.actions)), [1], [2])

    async def generate_structured(self, messages, **kwargs):
        self.calls.append(("structured", messages, kwargs))
        return PolicyCompletion(json.dumps(next(self.actions)), [1], [2])


def web_task():
    return Task("web", "widesearch", "validation", "Find names", gold=[{"name": "SECRET_GOLD"}],
                metadata={"evaluation": {"required": ["name"]}})


def test_only_last_web_evaluation_call_gets_explicit_schema_of_successful_reads():
    policy = Policy([
        {"tool": "web_search", "arguments": {"query": "names"}},
        {"tool": "fetch_url", "arguments": {"url": URL}},
        {"tool": "fetch_url", "arguments": {"url": URL + "/unread"}},
        {"answer": [{"name": "Alpha", "citations": [URL]}]},
    ])
    result = asyncio.run(EpisodeRunner(max_steps=4, web_tools_factory=Web, web_structured_final=True).run(
        web_task(), policy, evaluation=True))
    assert result.status == "completed"
    assert [call[0] for call in policy.calls] == ["ordinary", "ordinary", "ordinary", "structured"]
    schema = policy.calls[-1][2]["json_schema"]
    assert schema == web_final_schema(["name"], {URL})
    assert "SECRET_GOLD" not in json.dumps(policy.calls)


@pytest.mark.parametrize("benchmark,evaluation,enabled", [
    ("cuad", True, True), ("widesearch", False, True), ("widesearch", True, False),
])
def test_training_cuad_and_default_calls_keep_ordinary_capability(benchmark, evaluation, enabled):
    task = web_task() if benchmark == "widesearch" else Task(
        "contract", "cuad", "validation", "Find clause", document="Alpha", gold=[])
    policy = Policy([{"answer": []}])
    asyncio.run(EpisodeRunner(max_steps=1, web_tools_factory=Web, web_structured_final=enabled).run(
        task, policy, evaluation=evaluation))
    assert [call[0] for call in policy.calls] == ["ordinary"]
    assert "json_schema" not in policy.calls[0][2]


def test_early_answer_is_ordinary_and_empty_structured_answer_still_fails_guard():
    early = Policy([{"tool": "web_search", "arguments": {"query": "names"}},
                    {"tool": "fetch_url", "arguments": {"url": URL}},
                    {"answer": [{"name": "Alpha", "citations": [URL]}]}])
    result = asyncio.run(EpisodeRunner(max_steps=4, web_tools_factory=Web, web_structured_final=True).run(
        web_task(), early, evaluation=True))
    assert result.status == "completed" and all(call[0] == "ordinary" for call in early.calls)
    empty = Policy([{"answer": []}])
    result = asyncio.run(EpisodeRunner(max_steps=1, web_tools_factory=Web, web_structured_final=True).run(
        web_task(), empty, evaluation=True))
    assert empty.calls[0][0] == "structured" and result.status == "evidence_failure"


def test_unsupported_policy_fails_clearly_and_nonboolean_switch_is_rejected():
    async def policy(*args, **kwargs):
        raise AssertionError("ordinary generation must not run")

    with pytest.raises(ValueError, match="generate_structured capability"):
        asyncio.run(EpisodeRunner(max_steps=1, web_tools_factory=Web, web_structured_final=True).run(
            web_task(), policy, evaluation=True))
    with pytest.raises(ValueError, match="boolean"):
        web_structured_configuration(1)


def test_unsupported_backend_schema_fails_with_exact_keys_preserved(monkeypatch):
    schema = web_final_schema(['name"/中文'], {URL})
    calls = []

    def compile_schema(encoded):
        calls.append(json.loads(encoded))
        raise RuntimeError("backend does not support this escaped property")

    monkeypatch.setitem(sys.modules, "xgrammar", SimpleNamespace(
        Grammar=SimpleNamespace(from_json_schema=compile_schema)))
    with pytest.raises(ValueError, match="public column keys and source URLs must remain exact"):
        validate_web_structured_schema(schema)
    assert calls == [schema]


@pytest.mark.parametrize("successful_search", [False, True])
def test_action_schema_search_first_and_single_object(successful_search):
    jsonschema = pytest.importorskip("jsonschema")
    schema = web_action_schema(["name"], {URL}, successful_search=successful_search)
    search = {"tool": "web_search", "arguments": {"query": 'names "quoted" / 中文', "count": 8}}
    fetch = {"tool": "fetch_url", "arguments": {"url": "https://another.example.org/public", "start": 14000}}
    final = {"answer": [{"name": "Alpha", "citations": [URL]}]}
    jsonschema.validate(search, schema)
    for value in (fetch, final, {"answer": []}):
        if successful_search:
            jsonschema.validate(value, schema)
        else:
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate(value, schema)
    for invalid in ([search, fetch], {**search, **final},
                    {"tool": "read_document", "arguments": {"start": 0}},
                    {"tool": "web_search", "arguments": {}},
                    {"tool": "web_search", "arguments": {"query": "names", "count": 11}},
                    {"tool": "fetch_url", "arguments": {"url": URL, "start": -1}},
                    {"tool": "web_search", "arguments": {"query": "names", "extra": 1}}):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, schema)


def test_action_schema_no_read_cannot_supply_nonempty_final_or_citations():
    jsonschema = pytest.importorskip("jsonschema")
    schema = web_action_schema(["name"], set(), successful_search=True)
    jsonschema.validate({"answer": []}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"answer": [{"name": "Alpha", "citations": [URL]}]}, schema)
    assert URL not in json.dumps(schema)


def test_structured_actions_follow_actual_evidence_and_retain_final_schema():
    class RetryWeb(Web):
        async def web_search(self, query):
            if query == "empty":
                return {"results": []}
            return await super().web_search(query)

    policy = Policy([
        {"tool": "web_search", "arguments": {"query": "empty"}},
        {"tool": "web_search", "arguments": {"query": "names"}},
        {"tool": "fetch_url", "arguments": {"url": URL + "/unread"}},
        {"tool": "fetch_url", "arguments": {"url": URL}},
        {"answer": [{"name": "Alpha", "citations": [URL]}]},
    ])
    runner = EpisodeRunner(max_steps=5, web_tools_factory=RetryWeb,
                           web_structured_final=True, web_structured_actions=True)
    result = asyncio.run(runner.run(web_task(), policy, evaluation=True))
    assert result.status == "completed"
    assert [call[0] for call in policy.calls] == ["structured"] * 5
    schemas = [call[2]["json_schema"] for call in policy.calls]
    assert schemas[:2] == [web_action_schema(["name"], set(), successful_search=False)] * 2
    assert schemas[2:4] == [web_action_schema(["name"], set(), successful_search=True)] * 2
    assert schemas[-1] == web_final_schema(["name"], {URL})
    assert result.metrics["evidence_guard"]["successful_searches"] == 1
    assert result.metrics["evidence_guard"]["successful_fetches"] == 1
    assert "SECRET_GOLD" not in json.dumps(policy.calls)


def test_structured_actions_allow_supported_early_final_after_actual_auto_read():
    class AutoWeb(Web):
        async def fetch_url(self, url, start=0):
            return await super().fetch_url(url)

    policy = Policy([{"tool": "web_search", "arguments": {"query": "names"}},
                     {"answer": [{"name": "Alpha", "citations": [URL]}]}])
    result = asyncio.run(EpisodeRunner(
        max_steps=4, auto_read_sources=2, web_tools_factory=AutoWeb,
        web_structured_final=True, web_structured_actions=True,
    ).run(web_task(), policy, evaluation=True))
    assert result.status == "completed" and len(policy.calls) == 2
    assert policy.calls[1][2]["json_schema"] == web_action_schema(
        ["name"], {URL}, successful_search=True)
    assert result.tool_events[1]["automatic"] is True


@pytest.mark.parametrize("benchmark,evaluation", [("cuad", True), ("widesearch", False)])
def test_structured_actions_never_change_cuad_or_training_generation(benchmark, evaluation):
    task = web_task() if benchmark == "widesearch" else Task(
        "contract", "cuad", "validation", "Find clause", document="Alpha", gold=[])
    policy = Policy([{"answer": []}])
    asyncio.run(EpisodeRunner(max_steps=1, web_tools_factory=Web,
                             web_structured_final=True, web_structured_actions=True).run(
        task, policy, evaluation=evaluation))
    assert [call[0] for call in policy.calls] == ["ordinary"]


def test_structured_actions_keep_empty_and_missing_search_rejections():
    empty = Policy([{"answer": []}])
    runner = EpisodeRunner(max_steps=1, web_tools_factory=Web,
                           web_structured_final=True, web_structured_actions=True)
    result = asyncio.run(runner.run(web_task(), empty, evaluation=True))
    assert result.status == "evidence_failure"
    assert result.metrics["evidence_guard"]["rejections"][-1]["reasons"] == [
        "missing_successful_search", "missing_successful_fetch", "empty_answer"]
    # A policy that ignores the requested grammar cannot bypass actual evidence.
    rogue = Policy([{"tool": "fetch_url", "arguments": {"url": URL}},
                    {"answer": [{"name": "Alpha", "citations": [URL]}]}])
    result = asyncio.run(EpisodeRunner(max_steps=2, web_tools_factory=Web,
                                      web_structured_final=True, web_structured_actions=True).run(
        web_task(), rogue, evaluation=True))
    assert result.status == "evidence_failure"
    assert result.metrics["evidence_guard"]["rejections"][-1]["reasons"] == ["missing_successful_search"]


def test_structured_action_switch_requires_boolean_and_final():
    with pytest.raises(ValueError, match="boolean"):
        web_structured_actions_configuration(1)
    with pytest.raises(ValueError, match="requires web_structured_final"):
        EpisodeRunner(web_structured_actions=True)
    with pytest.raises(ValueError, match="boolean"):
        web_action_schema(["name"], set(), successful_search=1)


def test_real_xgrammar_action_matcher_enforces_branches_and_single_object():
    xgr = pytest.importorskip("xgrammar")
    info = xgr.TokenizerInfo([bytes([i]) for i in range(256)], stop_token_ids=[255])
    compiler = xgr.GrammarCompiler(info, max_threads=1)
    search = {"tool": "web_search", "arguments": {"query": 'names "quoted" / 中文', "count": 8}}
    fetch = {"tool": "fetch_url", "arguments": {"url": "https://another.example.org/public", "start": 14000}}
    final = {"answer": [{"name": "Alpha", "citations": [URL]}]}
    for successful_search in (False, True):
        grammar = compiler.compile_json_schema(json.dumps(web_action_schema(
            ["name"], {URL}, successful_search=successful_search)))

        def accepts(text):
            matcher = xgr.GrammarMatcher(grammar, terminate_without_stop_token=True)
            return matcher.accept_string(text) and matcher.is_terminated()

        # xgrammar's enum serializer chooses escaped solidus for exact URLs.
        encode = lambda obj: json.dumps(obj, ensure_ascii=False).replace("/", "\\/")
        assert accepts(encode(search))
        assert accepts(encode(fetch)) is successful_search
        assert accepts(encode(final)) is successful_search
        assert accepts(encode({"answer": []})) is successful_search
        assert not accepts(encode(search) + "\n" + encode(fetch))
        assert not accepts(encode([search, fetch]))
        assert not accepts(encode({"answer": [{"name": "Alpha", "citations": [URL + "/unread"]}]}))
