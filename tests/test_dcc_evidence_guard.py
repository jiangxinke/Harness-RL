from __future__ import annotations

import asyncio
import dataclasses
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from slime_plugins.dcc import EpisodeRunner, PolicyCompletion, Task
from slime_plugins.dcc import tools
from slime_plugins.dcc.episode import WIDESEARCH_EVIDENCE_GUARD_VERSION, parse_action
from slime_plugins.dcc.evaluation import evaluate_results

SOURCE = "https://example.org/source"
SEARCH = {"tool": "web_search", "arguments": {"query": "Alpha entities"}}
FETCH = {"tool": "fetch_url", "arguments": {"url": SOURCE}}
FINAL = {"answer": [{"name": "Alpha", "citations": [SOURCE]}]}


def wide():
    return Task("ws", "widesearch", "test", "Find Alpha", gold=[{"name": "Alpha"}], metadata={
        "evaluation": {"required": ["name"], "unique_columns": ["name"],
                       "eval_pipeline": {"name": {"metric": ["exact_match"], "preprocess": ["norm_str"]}}},
    })


@pytest.fixture
def public_transport(monkeypatch):
    """Exercise real tools/parsers with only HTTP transport replaced."""
    calls = []

    def request(url, **kwargs):
        calls.append(("search", url))
        return f'<a class="result-link" href="{SOURCE}">Alpha source</a><td class="result-snippet">Actual evidence</td>'

    def download(url):
        calls.append(("fetch", url))
        return b"Alpha source evidence", "text/plain", "utf-8"

    monkeypatch.setattr(tools, "_request", request)
    monkeypatch.setattr(tools, "_download_public", download)
    return calls


def run(actions, *, steps=None, item=None):
    iterator = iter(actions)
    prompts = []

    async def policy(messages, **kwargs):
        prompts.append(messages)
        action = next(iterator)
        return PolicyCompletion(action if isinstance(action, str) else json.dumps(action), [1], [2], [-0.1])

    runner = EpisodeRunner(max_steps=steps or len(actions), web_tools_factory=lambda: tools.WebTools(backend="duckduckgo_lite"))
    return asyncio.run(runner.run(item or wide(), policy, evaluation=True)), prompts


def test_early_final_can_recover_with_actual_search_fetch_and_citations(public_transport):
    result, prompts = run([FINAL, SEARCH, FETCH, FINAL])
    assert result.status == "completed"
    assert result.answer == FINAL["answer"]
    assert [call[0] for call in public_transport] == ["search", "fetch"]
    assert [event["tool"] for event in result.tool_events] == ["web_search", "fetch_url"]
    assert result.metrics["tool_calls"] == 2
    guard = result.metrics["evidence_guard"]
    assert guard["version"] == WIDESEARCH_EVIDENCE_GUARD_VERSION
    assert guard["accepted"]
    assert guard["successful_searches"] == guard["successful_fetches"] == 1
    assert len(guard["rejections"]) == 1
    feedback = prompts[1][-1]["content"]
    assert feedback.startswith("Evidence check: final answer rejected.")
    assert '"tool":"web_search"' in feedback and '"tool": "fetch_url"' in feedback
    assert "Tool observation" not in feedback
    assert "Before a final answer is accepted" in prompts[0][0]["content"]


@pytest.mark.parametrize("answer", [[], [{"name": "Alpha", "citations": ["https://example.org/invented"]}]])
def test_repeated_unsupported_finals_are_model_failures_with_full_denominator(answer):
    result, _ = run([{"answer": answer}] * 4)
    assert result.status == "evidence_failure"
    assert result.answer == [] and result.reward == 0
    assert len(result.trainable_steps) == 4
    assert result.metrics["tool_calls"] == 0 and result.tool_events == []
    assert len(result.metrics["evidence_guard"]["rejections"]) == 4
    report = evaluate_results([wide()], [result.to_dict()])
    assert report["complete"]
    assert report["status_counts"] == {"evidence_failure": 1}
    assert report["widesearch"]["metric_denominator"] == 1
    assert report["widesearch"]["score"] == 0
    assert report["widesearch"]["search_coverage"]["tasks_without_search"] == 1


def test_search_results_alone_do_not_satisfy_fetch_requirement(public_transport):
    result, _ = run([SEARCH, FINAL, FINAL])
    assert result.status == "evidence_failure"
    assert result.metrics["evidence_guard"]["successful_searches"] == 1
    assert result.metrics["evidence_guard"]["successful_fetches"] == 0
    assert "missing_successful_fetch" in result.metrics["evidence_guard"]["rejections"][-1]["reasons"]


def test_successful_fetch_without_search_does_not_satisfy_guard(public_transport):
    result, _ = run([FETCH, FINAL, FINAL])
    assert result.status == "evidence_failure"
    guard = result.metrics["evidence_guard"]
    assert guard["successful_fetches"] == 1 and guard["successful_searches"] == 0
    assert "missing_successful_search" in guard["rejections"][-1]["reasons"]


@pytest.mark.parametrize("body", [b"", b"   "])
def test_empty_fetch_body_is_not_successful_evidence(public_transport, monkeypatch, body):
    monkeypatch.setattr(tools, "_download_public", lambda url: (body, "text/plain", "utf-8"))
    result, _ = run([SEARCH, FETCH, FINAL])
    assert result.status == "evidence_failure"
    assert result.metrics["evidence_guard"]["successful_fetches"] == 0


def test_failed_fetch_is_observation_and_does_not_become_infrastructure_failure(public_transport, monkeypatch):
    def failed(url):
        raise tools.InfrastructureError("source rejected retrieval")

    monkeypatch.setattr(tools, "_download_public", failed)
    result, _ = run([SEARCH, FETCH, FINAL])
    assert result.status == "evidence_failure"
    assert result.metrics["evidence_guard"]["successful_fetches"] == 0
    assert result.tool_events[1]["observation"]["error"] == "source rejected retrieval"
    assert len(result.trainable_steps) == 3


def test_valid_empty_search_is_normal_observation_but_not_evidence(public_transport, monkeypatch):
    monkeypatch.setattr(tools, "_request", lambda *args, **kwargs: "<html>No results</html>")
    result, _ = run([SEARCH, FETCH, FINAL])
    assert result.status == "evidence_failure"
    assert result.tool_events[0]["observation"]["results"] == []
    assert result.metrics["evidence_guard"]["successful_searches"] == 0


@pytest.mark.parametrize("row,reason", [
    ({"name": "Alpha"}, "missing_row_citations"),
    ({"name": "Alpha", "citations": []}, "missing_row_citations"),
    ({"name": "Alpha", "citations": [" "]}, "missing_row_citations"),
    ({"name": "Alpha", "citations": [SOURCE, "https://example.org/invented"]}, "unobserved_citation_url"),
    ({"name": "Alpha", "citations": "wrong type"}, "invalid_answer_format"),
])
def test_every_row_must_have_only_observed_citations(public_transport, row, reason):
    result, _ = run([SEARCH, FETCH, {"answer": [FINAL["answer"][0], row]}])
    assert result.status == "evidence_failure"
    assert reason in result.metrics["evidence_guard"]["rejections"][-1]["reasons"]


def test_citation_rejection_can_be_corrected_without_fake_tool_events(public_transport):
    bad = {"answer": [{"name": "Alpha", "citations": ["https://example.org/invented"]}]}
    result, prompts = run([SEARCH, FETCH, bad, FINAL])
    assert result.status == "completed"
    assert len(result.tool_events) == 2
    assert len(result.metrics["evidence_guard"]["rejections"]) == 1
    assert "Every cited URL" in prompts[-1][-1]["content"]


def test_supported_empty_final_is_accepted_after_search_and_read(public_transport):
    result, _ = run([SEARCH, FETCH, {"answer": []}])
    assert result.status == "completed" and result.answer == []
    assert result.metrics["evidence_guard"]["accepted"]


def test_rejected_final_followed_by_budget_exhaustion_remains_model_failure(public_transport):
    result, _ = run([FINAL, SEARCH])
    assert result.status == "evidence_failure"
    assert public_transport == []  # The last-turn tool request was not executed.
    assert result.metrics["evidence_guard"]["successful_searches"] == 0


def test_evidence_failure_never_earns_empty_gold_credit_or_invokes_judge():
    item = dataclasses.replace(wide(), gold=[])

    def unexpected(prompt):
        pytest.fail("Rejected evidence must have zero score without a judge call")

    result, _ = run([{"answer": []}])
    report = evaluate_results([item], [result.to_dict()], judge=unexpected)
    assert report["widesearch"]["metric_denominator"] == 1
    assert report["widesearch"]["score"] == 0


def test_cuad_direct_final_and_prompt_are_unchanged():
    item = Task("contract", "cuad", "test", "Find clause", document="Exact clause", gold=["Exact clause"])
    result, prompts = run([{"answer": [{"text": "Exact clause", "confidence": 1}]}], item=item)
    assert result.status == "completed" and result.reward == 1
    assert "evidence_guard" not in result.metrics
    assert "web_search" not in prompts[0][0]["content"]
    assert "Before a final answer is accepted" not in prompts[0][0]["content"]


@pytest.mark.parametrize("benchmark", ["cuad", "widesearch"])
def test_private_manifest_versions_only_widesearch_without_loading_gpu_model(monkeypatch, tmp_path, benchmark):
    module = runpy.run_path(str(Path(__file__).parents[1] / "scripts/dcc_model_evaluate.py"))
    evaluate = module["evaluate"]
    item = dataclasses.replace(wide(), benchmark=benchmark, document="Example contract")
    monkeypatch.setitem(evaluate.__globals__, "load_tasks", lambda path: [item])

    def unexpected(*args, **kwargs):
        pytest.fail("Manifest/resume checks must complete without loading a model")

    monkeypatch.setitem(evaluate.__globals__, "load_policy_model", unexpected)
    for name in ("RANK", "LOCAL_RANK"):
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("DCC_SEARCH_BACKEND", "public_web")
    monkeypatch.delenv("DCC_SEARCH_URL", raising=False)
    args = SimpleNamespace(model="example/model", adapter=None, data=["tasks.jsonl"], max_steps=4,
                           max_new_tokens=1024, max_prompt_tokens=8192, initial_chunks=4,
                           seed=1, limit=0, output_dir=str(tmp_path), retry_infrastructure=False)
    (tmp_path / "predictions_rank0.jsonl").write_text(json.dumps({"task_id": item.id, "status": "completed"}) + "\n")
    asyncio.run(evaluate(args))
    manifest_path = tmp_path / "manifest_rank0.json"
    manifest = json.loads(manifest_path.read_text())
    if benchmark == "widesearch":
        assert manifest["evidence_guard_version"] == WIDESEARCH_EVIDENCE_GUARD_VERSION
        assert manifest["search_backend"] == "public_web"
        del manifest["evidence_guard_version"]
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="configuration changed"):
            asyncio.run(evaluate(args))
        manifest["evidence_guard_version"] = WIDESEARCH_EVIDENCE_GUARD_VERSION
        manifest_path.write_text(json.dumps(manifest))
        args.auto_read_sources = 1
        with pytest.raises(ValueError, match="configuration changed"):
            asyncio.run(evaluate(args))
    else:
        assert "evidence_guard_version" not in manifest
        assert "search_backend" not in manifest


def ndjson_rows():
    return '\n'.join(json.dumps(row) for row in [
        {"item": "Alpha", "value": "one", "citations": [SOURCE]},
        {"item": "Beta", "value": "two", "citations": [SOURCE]},
    ])


@pytest.mark.parametrize("fenced", [False, True])
def test_terminal_widesearch_ndjson_rows_require_explicit_parser_opt_in(fenced):
    payload = ndjson_rows()
    if fenced:
        payload = "```json\n" + payload + "\n```"
    assert parse_action(payload) is None
    action = parse_action(payload, widesearch_rows=True)
    assert [row["item"] for row in action["answer"]] == ["Alpha", "Beta"]


@pytest.mark.parametrize("payload", [
    "Reasoning about candidates:\n" + ndjson_rows(),
    ndjson_rows() + "\nThese are preliminary notes.",
    json.dumps({"item": "Alpha", "citations": [SOURCE]}) + '\n{"item":"missing citations"}',
    ndjson_rows() + '\n{}',
    ndjson_rows() + '\nnot JSON',
    '```json\n' + ndjson_rows() + '\n```\nMore reasoning',
    '```json\n' + ndjson_rows() + '\n```\n```json\n' + ndjson_rows() + '\n```',
])
def test_widesearch_ndjson_rejects_reasoning_fragments_and_partial_payloads(payload):
    assert parse_action(payload, widesearch_rows=True) is None


@pytest.mark.parametrize("tool_first", [False, True])
def test_widesearch_ndjson_never_turns_mixed_tool_actions_into_final(tool_first):
    pieces = [json.dumps(SEARCH), ndjson_rows()]
    payload = '\n'.join(pieces if tool_first else reversed(pieces))
    action = parse_action(payload, widesearch_rows=True)
    assert action == SEARCH
    assert "answer" not in action


@pytest.mark.parametrize("fenced", [False, True])
def test_real_widesearch_episode_accepts_ndjson_only_after_evidence(public_transport, fenced):
    payload = ndjson_rows()
    if fenced:
        payload = "```json\n" + payload + "\n```"
    result, _ = run([SEARCH, FETCH, payload])
    assert result.status == "completed"
    assert len(result.answer) == 2
    assert result.metrics["evidence_guard"]["accepted"]
    assert result.metrics["citation_validity"] == 1
    assert len(result.tool_events) == 2


def test_widesearch_ndjson_without_search_and_fetch_is_still_rejected():
    result, _ = run([ndjson_rows()])
    assert result.status == "evidence_failure" and result.answer == []
    assert "missing_successful_search" in result.metrics["evidence_guard"]["rejections"][0]["reasons"]
    assert result.tool_events == []


def test_widesearch_ndjson_preserves_schema_and_citation_checks(public_transport):
    bad_schema = json.dumps({"item": "Alpha", "citations": "not a list"})
    result, _ = run([SEARCH, FETCH, bad_schema])
    assert result.status == "evidence_failure"
    assert "invalid_answer_format" in result.metrics["evidence_guard"]["rejections"][0]["reasons"]
    bad_citation = ndjson_rows().replace(SOURCE, "https://example.org/unobserved")
    result, _ = run([SEARCH, FETCH, bad_citation])
    assert result.status == "evidence_failure"
    assert "unobserved_citation_url" in result.metrics["evidence_guard"]["rejections"][0]["reasons"]


def test_cuad_never_enables_widesearch_ndjson_normalization():
    item = Task("contract", "cuad", "test", "Find clause", document="Exact clause", gold=["Exact clause"])
    result, _ = run([ndjson_rows()], item=item)
    assert result.status == "max_steps" and result.answer == []
    assert "evidence_guard" not in result.metrics
    assert "Invalid action" in result.messages[-1]["content"]
