from __future__ import annotations

import asyncio
import dataclasses
import json

import pytest

from slime_plugins.dcc import (
    EpisodeRunner,
    PolicyCompletion,
    Task,
    load_tasks,
    validate_disjoint,
)
from slime_plugins.dcc.episode import final_turn_message, parse_action
from slime_plugins.dcc.metrics import (
    cuad_counts,
    cuad_jaccard,
    cuad_task_metrics,
    evaluate_cuad,
)
from slime_plugins.dcc.tools import (
    DocumentTools,
    InfrastructureError,
    WebTools,
    _DuckDuckGoResults,
    visible_text,
)


def task(**kwargs):
    return Task(
        **{
            "id": "contract__Governing Law",
            "benchmark": "cuad",
            "split": "train",
            "prompt": "What is the governing law?",
            "document": "This Agreement is governed by California law.",
            "gold": ["This Agreement is governed by California law."],
            **kwargs,
        }
    )


def test_document_search_retains_verifiable_source_offsets():
    text = (
        ("Payments are due in thirty days.\n" * 20)
        + "This contract is governed by Delaware law.\n"
        + ("The service is available daily.\n" * 20)
    )
    tools = DocumentTools(text, chunk_chars=300, overlap_chars=50)
    hit = tools.search_document("Delaware governing law", top_k=1)["hits"][0]
    assert "Delaware" in hit["text"]
    assert hit["text"] == text[hit["start"] : hit["end"]]
    assert tools.read_document(hit["start"], hit["end"])["text"] == hit["text"]


def test_document_reader_bounded_and_validates_ranges():
    tools = DocumentTools("x" * 14000)
    assert len(tools.read_document(0, 20000)["text"]) == 12000
    assert "error" in tools.read_document(30000)


def test_official_cuad_matching_handles_many_to_one_and_parties_substring():
    assert cuad_jaccard("A/B, C.", "a b c") == 1
    assert cuad_counts("x", ["alpha beta", "alpha gamma"], ["alpha beta gamma"]) == (
        2,
        0,
        0,
    )
    assert cuad_counts(
        "x__Parties", ["Acme"], ["Acme entered an agreement with many unrelated words"]
    ) == (1, 0, 0)
    assert cuad_counts(
        "x__Term", ["Acme"], ["Acme entered an agreement with many unrelated words"]
    ) == (0, 1, 1)


def test_cuad_official_curve_and_no_answer_training_reward():
    report = evaluate_cuad(
        {"positive": ["alpha beta"], "negative": []},
        {"positive": [{"text": "alpha beta", "confidence": 0.9}], "negative": []},
    )
    assert report["aupr"] == 1
    assert report["prec_at_90_recall"] == 1
    assert report["micro_f1"] == 1
    assert cuad_task_metrics("negative", [], [])["f1"] == 1
    assert cuad_task_metrics("negative", [], ["wrong"])["f1"] == 0
    with pytest.raises(ValueError, match="IDs"):
        evaluate_cuad({"a": []}, {})


@pytest.mark.parametrize("evaluation, expected", [(True, 0.7), (False, 1.0)])
def test_explicit_evaluation_temperature_does_not_change_training_rollouts(evaluation, expected):
    seen = []

    async def policy(messages, **kwargs):
        seen.append(kwargs["temperature"])
        return PolicyCompletion('{"answer":[]}')

    result = asyncio.run(EpisodeRunner(temperature=1.0, evaluation_temperature=0.7).run(
        task(), policy, evaluation=evaluation))
    assert result.status == "completed" and seen == [expected]


def test_external_contracts_are_loaded_once_and_split_leaks_rejected(tmp_path):
    (tmp_path / "contract.txt").write_text("A contract", encoding="utf-8")
    rows = [
        dict(
            id=f"q{i}",
            benchmark="cuad",
            split="train",
            prompt="question",
            metadata={"document_file": "contract.txt"},
        )
        for i in range(2)
    ]
    path = tmp_path / "tasks.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    tasks = load_tasks(path)
    assert tasks[0].document is tasks[1].document
    with pytest.raises(ValueError, match="contract leaks"):
        validate_disjoint(
            tasks, [dataclasses.replace(tasks[0], id="other", split="test")]
        )


def test_gold_never_appears_in_policy_messages():
    item = task(
        gold=["hidden gold marker"], metadata={"secret_label": "another hidden marker"}
    )
    messages = EpisodeRunner().initial_messages(item)
    assert "hidden" not in json.dumps(messages)


def test_packed_contract_document_file(tmp_path):
    (tmp_path / "documents.json").write_text(
        json.dumps({"contract_1": "source text"}), encoding="utf-8"
    )
    row = {
        "id": "q",
        "benchmark": "cuad",
        "split": "train",
        "prompt": "question",
        "metadata": {"document_file": "documents.json", "document_key": "contract_1"},
    }
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps(row), encoding="utf-8")
    assert load_tasks(path)[0].document == "source text"


def test_action_parser_ignores_nested_example_in_query_and_thinking():
    assert parse_action(
        '<think>{"answer":["wrong"]}</think>```json\n{"answer":[]}\n```'
    ) == {"answer": []}
    action = {
        "tool": "search_document",
        "arguments": {"query": 'example {"answer":[]} clause'},
    }
    assert parse_action(json.dumps(action)) == action


@pytest.mark.parametrize("fenced", [False, True])
@pytest.mark.parametrize("prefix", [
    "",
    '<think>Ignore this example {"answer":["wrong </think> value"]}.</think>',
    '<think>Compare the sources.</think>Final answer:\n',
])
@pytest.mark.parametrize("action", [
    {"answer": [{"text": 'Literal </think> and "quoted" text', "confidence": 0.9}]},
    {"tool": "search_document", "arguments": {"query": 'literal </think> {"answer":[]}'}},
    {"plan": "Review </think> literally", "checks": "Read source", "answer": []},
    [{"text": "A literal </think> clause", "confidence": 1.0}],
])
def test_action_parser_preserves_literal_thinking_close_in_complete_json(action, prefix, fenced):
    payload = json.dumps(action)
    if fenced:
        payload = "```json\n" + payload + "\n```"
    expected = {"answer": action} if isinstance(action, list) else action
    for widesearch_rows in (False, True):
        assert parse_action(prefix + payload, widesearch_rows=widesearch_rows) == expected


def test_literal_thinking_close_in_contract_span_still_completes_and_scores():
    item = task(document="A literal </think> marker is part of this agreement.")
    item = dataclasses.replace(item, gold=[item.document])

    async def policy(messages, **kwargs):
        return PolicyCompletion(json.dumps({"answer": [{"text": item.document, "confidence": 1.0}]}))

    result = asyncio.run(EpisodeRunner(max_steps=1).run(item, policy))
    assert result.status == "completed"
    assert result.answer == [{"text": item.document, "confidence": 1.0}]
    assert result.reward == 1


@pytest.mark.parametrize("fenced", [False, True])
@pytest.mark.parametrize("prefix", ["", "<think>Review the evidence.</think>"])
def test_widesearch_jsonl_literal_thinking_close_keeps_rows_and_ambiguity_guard(fenced, prefix):
    rows = [
        {"name": "Alpha </think>", "citations": ["https://fixture.example/a"]},
        {"name": "Beta </think>", "citations": ["https://fixture.example/b"]},
    ]
    payload = "\n".join(json.dumps(row) for row in rows)
    complete = "```jsonl\n" + payload + "\n```" if fenced else payload
    assert parse_action(prefix + complete, widesearch_rows=True) == {"answer": rows}
    assert parse_action(prefix + complete) is None
    tool = json.dumps({"tool": "web_search", "arguments": {"query": "continue research"}})
    for mixed in (payload + "\n" + tool, tool + "\n" + payload):
        if fenced:
            mixed = "```jsonl\n" + mixed + "\n```"
        assert parse_action(prefix + mixed, widesearch_rows=True) is None


@pytest.mark.parametrize("widesearch_rows", [False, True])
def test_action_parser_does_not_promote_a_json_example_from_unclosed_thinking(widesearch_rows):
    assert parse_action('<think>Example {"answer":["literal </think> example"]}',
                        widesearch_rows=widesearch_rows) is None
    assert parse_action('<think>Example {"answer":[]} but keep thinking.',
                        widesearch_rows=widesearch_rows) is None
    assert parse_action('<think>Example {"answer":["literal </think> example"]}</think>Still reasoning.',
                        widesearch_rows=widesearch_rows) is None
    assert parse_action('For example {"answer":["literal </think> example"]} but research continues.',
                        widesearch_rows=True) is None


@pytest.mark.parametrize("fenced", [False, True])
def test_terminal_bare_answer_list_is_accepted_and_validated(fenced):
    payload = json.dumps([{"text": task().document, "confidence": 0.9}])
    if fenced:
        payload = "```json\n" + payload + "\n```"

    async def policy(messages, **kwargs):
        return PolicyCompletion(payload, [1], [2])

    result = asyncio.run(EpisodeRunner(max_steps=1).run(task(), policy))
    assert result.status == "completed"
    assert result.reward == 1
    assert parse_action('Example: [] is a possible answer, but research continues.') is None
    assert parse_action('{"tool":"search_document","arguments":{"query":"[]"}}')["tool"] == "search_document"


def test_last_turn_requests_final_answer_without_executing_tools():
    class NoLastTurnWeb(WebTools):
        async def web_search(self, **kwargs):
            pytest.fail("A final-turn tool request must not reach the network")

    async def policy(messages, **kwargs):
        assert messages[-1]["content"].startswith("Final answer now; tools are unavailable.")
        return PolicyCompletion('{"tool":"web_search","arguments":{"query":"test"}}')

    item = Task("ws", "widesearch", "test", "Find entities")
    result = asyncio.run(EpisodeRunner(max_steps=1, web_tools_factory=NoLastTurnWeb).run(item, policy))
    assert result.status == "evidence_failure"
    assert "no tool was executed" in result.tool_events[0]["observation"]["error"]


def test_policy_segments_preserve_each_prompt_and_tool_observations_are_context_only():
    calls = []

    async def policy(messages, *, max_new_tokens, temperature):
        calls.append(messages)
        text = (
            '{"tool":"read_document","arguments":{"start":0,"end":100}}'
            if len(calls) == 1
            else json.dumps({"answer": [{"text": task().document, "confidence": 0.9}]})
        )
        return PolicyCompletion(text, [10] * len(calls), [20 + len(calls)], [-0.1])

    result = asyncio.run(EpisodeRunner(max_steps=2).run(task(), policy))
    assert result.status == "completed"
    assert result.reward == 1
    assert len(result.trainable_steps) == 2
    assert result.steps[1].prompt_ids == [10, 10]
    assert result.steps[1].completion_ids == [22]
    assert "Tool observation" in calls[1][-2]["content"]
    assert "Final turn" not in calls[1][-2]["content"]
    assert "Final turn: tools are unavailable" in calls[1][-1]["content"]


def test_final_turn_control_preserves_initial_contract_source_message():
    item = task()
    runner = EpisodeRunner(max_steps=1)
    initial = runner.initial_messages(item)

    async def policy(messages, **kwargs):
        assert messages[:-1] == initial
        assert messages[-2]["content"].startswith("Initial contract retrieval (source material):")
        assert "Final turn" not in messages[-2]["content"]
        assert messages[-1]["role"] == "user"
        assert "Final turn: tools are unavailable" in messages[-1]["content"]
        return PolicyCompletion(json.dumps({"answer": [{"text": item.document}]}))

    result = asyncio.run(runner.run(item, policy))
    assert result.status == "completed"
    assert runner.initial_messages(item) == initial


@pytest.mark.parametrize("early_final", [False, True])
def test_cuad_review_preserves_early_actions_sources_and_span_scoring(early_final):
    item = task()
    runner = EpisodeRunner(max_steps=2)
    initial = runner.initial_messages(item)
    answer = [{"text": item.document, "confidence": 0.9}]
    calls = []

    async def policy(messages, **kwargs):
        calls.append(messages)
        if len(calls) == 1:
            assert messages == initial
            if not early_final:
                return PolicyCompletion(json.dumps({"tool": "read_document", "arguments": {"start": 0, "end": len(item.document)}}))
            return PolicyCompletion(json.dumps({"answer": answer}))
        assert messages[:len(initial)] == initial
        assert messages[-2]["content"].startswith("Tool observation (source material):")
        assert item.document in messages[-2]["content"]
        assert messages[-1] == final_turn_message(item)
        return PolicyCompletion(json.dumps({"plan": "Check the requested governing law.",
                                            "checks": "Observed source supports this verbatim span.", "answer": answer}))

    result = asyncio.run(runner.run(item, policy))
    assert result.status == "completed" and result.answer == answer and result.reward == 1
    assert len(calls) == (1 if early_final else 2)
    assert len(result.tool_events) == (0 if early_final else 1)
    assert runner.initial_messages(item) == initial


def test_cuad_final_review_is_independent_of_labels_and_private_metadata():
    original = task()
    changed = dataclasses.replace(original, gold=["private label marker"],
                                  metadata={"category": "private metadata marker", "is_impossible": True})
    assert final_turn_message(original) == final_turn_message(changed)
    assert "private" not in final_turn_message(changed)["content"]


def test_correct_but_nonverbatim_span_is_not_rewarded():
    async def policy(messages, **kwargs):
        return PolicyCompletion(
            '{"answer":["This Agreement is governed California law"]}', [1], [2]
        )

    result = asyncio.run(EpisodeRunner().run(task(), policy))
    assert result.metrics["f1"] == 1
    assert result.reward == 0


def test_infrastructure_failure_invalidates_all_policy_segments():
    class FailedWeb(WebTools):
        async def web_search(self, query, count=8):
            raise InfrastructureError("search unavailable")

    async def policy(messages, **kwargs):
        return PolicyCompletion(
            '{"tool":"web_search","arguments":{"query":"test"}}', [1], [2]
        )

    item = Task("ws", "widesearch", "test", "Find entities", gold=[{"secret": "label"}])
    result = asyncio.run(EpisodeRunner(web_tools_factory=FailedWeb).run(item, policy))
    assert result.status == "infrastructure_error"
    assert result.reward == 0
    assert not result.trainable_steps
    assert result.detail == "search unavailable"


class _AvailableWeb(WebTools):
    async def web_search(self, query, count=8):
        hit = {"title": "Source", "url": "https://example.org/source", "snippet": "Alpha"}
        self.sources[hit["url"]] = hit
        return {"results": [hit]}

    async def fetch_url(self, url, start=0):
        return {"url": url, "text": "Alpha source evidence"}


def _supported_web_policy():
    actions = iter([
        {"tool": "web_search", "arguments": {"query": "Alpha"}},
        {"tool": "fetch_url", "arguments": {"url": "https://example.org/source"}},
        {"answer": [{"name": "Alpha", "citations": ["https://example.org/source"]}]},
    ])

    async def policy(messages, **kwargs):
        return PolicyCompletion(json.dumps(next(actions)), [1], [2])

    return policy


def test_heldout_widesearch_gold_does_not_affect_rollout_reward():

    item = Task("ws", "widesearch", "test", "Find entities", gold=[{"name": "Alpha"}])
    result = asyncio.run(EpisodeRunner(web_tools_factory=_AvailableWeb).run(item, _supported_web_policy()))
    assert result.status == "completed"
    assert result.reward == 0
    assert "item_f1" not in result.metrics


@pytest.mark.parametrize(
    "answer", [[{}], [""], [7], [{"text": "clause", "confidence": float("nan")}]]
)
def test_malformed_span_list_cannot_earn_no_answer_reward(answer):
    async def policy(messages, **kwargs):
        return PolicyCompletion(json.dumps({"answer": answer}), [1], [2])

    result = asyncio.run(EpisodeRunner().run(task(gold=[]), policy))
    assert result.status == "invalid_answer"
    assert result.reward == 0
    assert result.metrics["valid_format"] == 0


def test_independent_web_curriculum_uses_actual_item_reward():
    item = Task(
        "independent",
        "widesearch",
        "train",
        "Find Alpha",
        gold=[{"name": "Alpha"}],
        metadata={
            "source": "independent_curriculum",
            "evaluation": {
                "required": ["name"],
                "unique_columns": ["name"],
                "eval_pipeline": {
                    "name": {"metric": ["exact_match"], "preprocess": ["norm_str"]}
                },
            },
        },
    )
    result = asyncio.run(EpisodeRunner(web_tools_factory=_AvailableWeb).run(item, _supported_web_policy()))
    assert result.reward == 1
    assert result.metrics["f1_by_item"] == 1


def test_evaluation_is_greedy_and_cancellation_is_preserved():
    seen = []

    async def policy(messages, **kwargs):
        seen.append(kwargs["temperature"])
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(EpisodeRunner().run(task(), policy, evaluation=True))
    assert seen == [0.0]


def test_invalid_policy_action_becomes_observation_and_capped_episode():
    async def policy(messages, **kwargs):
        return PolicyCompletion("not JSON", [1], [2])

    result = asyncio.run(EpisodeRunner(max_steps=2).run(task(), policy))
    assert result.status == "max_steps"
    assert result.reward == 0
    assert len(result.trainable_steps) == 2


def test_duckduckgo_results_and_visible_text():
    parser = _DuckDuckGoResults()
    parser.feed(
        '<a class="result-link" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org">Example</a><td class="result-snippet">Real source</td>'
    )
    assert parser.hits == [
        {"title": "Example", "url": "https://example.org", "snippet": "Real source"}
    ]
    assert "secret" not in visible_text(
        "<p>Hello</p><script>secret</script><p>World</p>"
    )


def test_web_reader_can_continue_long_pages_without_retrieval_again(monkeypatch):
    from slime_plugins.dcc import tools

    calls = []

    def download(url):
        calls.append(url)
        return b"abcdefghij", "text/plain", "utf-8"

    monkeypatch.setattr(tools, "_download_public", download)

    async def read():
        web = WebTools(max_page_chars=4)
        first = await web.fetch_url("https://example.org/page")
        second = await web.fetch_url(
            "https://example.org/page", start=first["next_start"]
        )
        return first, second

    first, second = asyncio.run(read())
    assert first["text"] == "abcd"
    assert second["text"] == "efgh"
    assert first["total_characters"] == 10
    assert calls == ["https://example.org/page"]


def test_completion_logprob_alignment_is_checked():
    with pytest.raises(ValueError, match="align"):
        PolicyCompletion("answer", [1], [2, 3], [-1.0])
