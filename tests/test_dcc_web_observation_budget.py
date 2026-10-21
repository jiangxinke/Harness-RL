"""CPU boundary tests of token-counted literal Web windows; no network/model."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slime_plugins.dcc.data import Task
from slime_plugins.dcc.episode import EpisodeRunner, PolicyCompletion
from slime_plugins.dcc.prompt_history import PromptHistoryConfig, PromptHistoryBudgetExceeded, fit_prompt_history
from slime_plugins.dcc.web_observation_budget import WebObservationBudget, web_observation_budget_configuration


URL = "https://example.org/invented-source"


class ByteTokenizer:
    """Adversarial unicode: one character can consume multiple real input IDs."""
    def encode(self, text, **kwargs):
        return list(text.encode("utf-8"))

    def apply_chat_template(self, messages, **kwargs):
        return self.encode(json.dumps(messages, ensure_ascii=False) + "assistant:")


def page(text="证据" * 7000, *, url=URL, start=23, **extra):
    return dict(url=url, text=text, start=start, end=start + len(text),
        total_characters=start + len(text) + 100, truncated=True, next_start=start + len(text), **extra)


def test_whole_observation_unicode_and_metadata_use_actual_tokenizer():
    original = page(outgoing_links=[{"url": URL + str(i), "text": "导航" * 200} for i in range(40)])
    before = deepcopy(original)
    budget = WebObservationBudget(ByteTokenizer(), 6144)
    assert len(original["text"]) == 14000 and budget.count(original) > 32768
    packed = budget.pack(original)
    assert original == before and budget.count(packed) <= 6144
    assert json.loads(json.dumps(packed, ensure_ascii=False)) == packed
    assert packed["url"] == original["url"] and packed["text"] == original["text"][:len(packed["text"])]
    assert 0 < len(packed["text"]) < len(original["text"])
    assert packed["end"] == packed["start"] + len(packed["text"]) == packed["next_start"]
    assert packed["truncated"] and packed["source_text_exposed"]
    assert packed["total_characters"] == original["total_characters"]
    assert packed["observation_budget"]["metadata_truncated"] and "outgoing_links" not in packed


def test_aggregate_budget_covers_both_auto_reads_and_preserves_errors():
    original = {"results": [{"url": URL, "snippet": "x" * 4000}],
        "auto_read_sources": [page(), {"url": URL + "/error", "error": "HTTP 403 forbidden " * 1000,
                                      "http_status": 403, "retryable": False}],
        "auto_read_attempts_remaining": 2, "auto_read_successful_sources": 1}
    budget = WebObservationBudget(ByteTokenizer(), 2048)
    packed = budget.pack(original)
    assert budget.count(packed) <= 2048
    good, error = packed["auto_read_sources"]
    assert good["url"] == URL and good["text"] and error["error"]
    assert error["error_details_truncated"] and error["http_status"] == 403 and not error["retryable"]
    assert packed["auto_read_successful_sources"] == 1
    assert packed["results"] == [{"url": URL}]


def test_archive_metadata_never_turns_a_label_prefix_into_read_evidence():
    raw = page("member label\n" + "真实内容" * 1500, start=0,
        source_text_exposed=True, archive_members=[{"status": "readable", "content_start": 2000, "content_end": 6000}])
    packed = WebObservationBudget(ByteTokenizer(), 1024).pack(raw)
    assert packed["end"] < 2000 and packed["text"] and not packed["source_text_exposed"]
    assert "archive_members" not in packed and packed["observation_budget"]["metadata_truncated"]


def test_archive_original_nonexposure_remains_authoritative():
    raw = page(start=0, source_text_exposed=False,
        archive_members=[{"status": "readable", "content_start": 0, "content_end": 14000}])
    packed = WebObservationBudget(ByteTokenizer(), 1024).pack(raw)
    assert packed["text"] and not packed["source_text_exposed"]


def test_impossible_identity_returns_recoverable_error_without_rewritten_url():
    budget = WebObservationBudget(ByteTokenizer(), 512)
    packed = budget.pack(page(url=URL + "z" * 10000))
    assert packed["error"] and "url" not in packed and "text" not in packed
    assert packed["observation_budget"]["identity_omitted"] and packed["retryable"]
    assert budget.count(packed) <= 512 and not packed["source_text_exposed"]


def test_oversized_complete_auto_read_entries_are_explicitly_omitted():
    original = {"results": [{"url": URL}], "auto_read_sources": [page(),
        {"url": URL + "z" * 10000, "error": "HTTP 403", "http_status": 403}]}
    budget = WebObservationBudget(ByteTokenizer(), 2048)
    packed = budget.pack(original)
    assert budget.count(packed) <= 2048
    assert len(packed["auto_read_sources"]) == packed["auto_read_successful_sources"] == 1
    assert packed["auto_read_sources"][0]["url"] == URL
    assert packed["observation_budget"]["omitted_auto_read_sources"] == 1
    assert packed["observation_budget"]["omitted_auto_read_errors"] == 1


def test_unaligned_source_offsets_become_recoverable_error_without_text():
    raw = page()
    raw["end"] += 1
    packed = WebObservationBudget(ByteTokenizer(), 1024).pack(raw)
    assert packed["error"] and packed["retryable"] and "text" not in packed
    assert not packed["source_text_exposed"] and packed["url"] == URL


def test_small_observation_is_byte_exact_and_unaliased():
    original = page("short source", row_view={"records": [{"text": "literal row"}]})
    packed = WebObservationBudget(ByteTokenizer(), 6144).pack(original)
    assert packed == original and packed is not original and "observation_budget" not in packed


@pytest.mark.parametrize("invalid", [0, -1, 511, 512.0, True])
def test_budget_validation(invalid):
    with pytest.raises(ValueError):
        web_observation_budget_configuration(invalid)


def test_history_really_evicts_old_docs_but_cannot_rescue_one_huge_current_turn():
    tokenizer = ByteTokenizer()
    initial = [{"role": "system", "content": "Use actual sources."}, {"role": "user", "content": "Invented task"}]
    def observation(value):
        return {"role": "user", "content": "Tool observation (source material):\n" + json.dumps(value, ensure_ascii=False)}
    old = [{"role": "assistant", "content": '{"tool":"fetch_url"}'}, observation(page("old source " * 4000))]
    latest = [{"role": "assistant", "content": '{"tool":"fetch_url"}'}, observation(page())]
    options = dict(max_prompt_tokens=32768, max_model_len=40960, max_new_tokens=8192, config=PromptHistoryConfig(True))
    with pytest.raises(PromptHistoryBudgetExceeded):
        fit_prompt_history(tokenizer, initial + old + latest, **options)
    latest[-1] = observation(WebObservationBudget(tokenizer, 6144).pack(page()))
    plan = fit_prompt_history(tokenizer, initial + old + latest, **options)
    assert plan.report["omitted_message_indices"] == [2, 3]
    assert len(plan.prompt_ids) <= 32768 and plan.messages[-2:] == latest


class InventedWeb:
    async def web_search(self, query, **kwargs):
        return {"query": query, "results": [{"url": URL}, {"url": URL + "/other"}]}

    async def fetch_url(self, url, **kwargs):
        return page(url=url, outgoing_links=[{"url": URL, "text": "metadata" * 1000}])

    def citation_validity(self, answer):
        return 1.


class OversizedSecondIdentityWeb(InventedWeb):
    async def web_search(self, query, **kwargs):
        return {"query": query, "results": [{"url": URL}, {"url": URL + "z" * 10000}]}

    async def fetch_url(self, url, **kwargs):
        if url != URL:
            return {"url": url, "error": "HTTP 403", "http_status": 403}
        return await super().fetch_url(url, **kwargs)


class InventedPolicy:
    def __init__(self):
        self.requests = []

    async def __call__(self, messages, **kwargs):
        self.requests.append(deepcopy(messages))
        action = ({"tool": "web_search", "arguments": {"query": "invented"}} if len(self.requests) == 1
            else {"answer": [{"name": "invented", "citations": [URL]}]})
        return PolicyCompletion(text=json.dumps(action), prompt_ids=[1], completion_ids=[2])


@pytest.mark.parametrize("evaluation", [False, True])
def test_real_shared_runner_commits_only_actual_packed_auto_read_windows(evaluation):
    policy = InventedPolicy()
    runner = EpisodeRunner(max_steps=3, initial_chunks=0, auto_read_sources=2,
        web_tools_factory=InventedWeb, web_observation_token_budget=2048)
    runner.set_web_observation_tokenizer(ByteTokenizer())
    result = asyncio.run(runner.run(Task("invented", "widesearch", "train", "Find invented", gold=[]),
        policy, evaluation=evaluation, score_train=False))
    assert result.status == "completed" and result.metrics["evidence_guard"]["successful_fetches"] == 2
    delivered = json.JSONDecoder().raw_decode(policy.requests[1][-1]["content"].split("\n", 1)[1])[0]
    assert runner.web_observation_budget.count(delivered) <= 2048
    assert [event["observation"] for event in result.tool_events[1:]] == delivered["auto_read_sources"]
    for source in delivered["auto_read_sources"]:
        assert source["url"] in {URL, URL + "/other"}
        assert source["text"] == page()["text"][:len(source["text"])]
        assert source["end"] == source["start"] + len(source["text"]) == source["next_start"]


def test_enabled_runner_requires_actual_tokenizer_before_any_tool_or_policy():
    runner, policy = EpisodeRunner(web_observation_token_budget=6144), InventedPolicy()
    with pytest.raises(ValueError, match="actual policy tokenizer"):
        asyncio.run(runner.run(Task("invented", "widesearch", "train", "Invented", gold=[]), policy))
    assert not policy.requests


def test_omitted_automatic_source_never_enters_real_evidence_or_navigation_events():
    runner = EpisodeRunner(max_steps=3, initial_chunks=0, auto_read_sources=2,
        web_tools_factory=OversizedSecondIdentityWeb, web_observation_token_budget=2048)
    runner.set_web_observation_tokenizer(ByteTokenizer())
    result = asyncio.run(runner.run(Task("invented", "widesearch", "train", "Find invented", gold=[]),
        InventedPolicy(), score_train=False))
    assert result.status == "completed" and result.metrics["evidence_guard"]["successful_fetches"] == 1
    assert len(result.tool_events) == 2
    assert result.tool_events[1]["observation"]["url"] == URL
    assert result.tool_events[0]["observation"]["observation_budget"]["omitted_auto_read_errors"] == 1
    assert "z" * 10000 not in json.dumps(result.tool_events)


def test_explicit_none_budget_keeps_original_runner_results_exact():
    task = Task("invented", "widesearch", "train", "Find invented", gold=[])
    def run(**extra):
        return asyncio.run(EpisodeRunner(max_steps=3, initial_chunks=0, auto_read_sources=2,
            web_tools_factory=InventedWeb, **extra).run(task, InventedPolicy(), score_train=False)).to_dict()
    assert run() == run(web_observation_token_budget=None)


def load_script(name):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    spec = importlib.util.spec_from_file_location("_observation_" + name, scripts / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shared_train_and_evaluation_configuration_same_budget_and_default_identity():
    train, evaluate = load_script("dcc_online_grpo"), load_script("dcc_online_web_evaluate")
    args = evaluate.parser().parse_args(["--model", "invented", "--data", "unused", "--output-dir", "unused",
        "--source-commit", "a" * 40])
    baseline = evaluate.runner_configuration(args)
    assert "web_observation_budget" not in baseline
    args.web_observation_token_budget = 6144
    configured = evaluate.runner_configuration(args)
    assert configured["web_observation_budget"] == web_observation_budget_configuration(6144)
    train_args = SimpleNamespace(model="invented", source_commit="a" * 40, group_size=2,
        max_steps=3, thought_tokens=4, cuad_draws_per_cycle=1, web_draws_per_cycle=1,
        enable_online_web=True, web_observation_token_budget=6144)
    config = train.configuration(train_args, [], [], 1)
    assert config["web_observation_budget"] == configured["web_observation_budget"]
    train_args.web_observation_token_budget = None
    old = train.configuration(train_args, [], [], 1)
    assert "web_observation_budget" not in old and "web_observation_token_budget" not in old["arguments"]
    del train_args.web_observation_token_budget
    assert train.configuration(train_args, [], [], 1) == old
    args.web_observation_token_budget = None
    assert evaluate.runner_configuration(args) == baseline
    del args.web_observation_token_budget
    assert evaluate.runner_configuration(args) == baseline


@pytest.mark.parametrize("name", ["dcc_online_grpo", "dcc_online_web_evaluate"])
def test_cli_invalid_budget_rejects_before_source_data_or_cuda(name, capsys):
    script = load_script(name)
    data_flag = "--train-data" if name == "dcc_online_grpo" else "--data"
    with pytest.raises(SystemExit) as error:
        script.main(["--model", "invented", data_flag, "does-not-exist", "--output-dir", "unused",
            "--source-commit", "a" * 40, "--web-observation-token-budget", "511"])
    assert error.value.code == 2 and "at least 512" in capsys.readouterr().err


def test_training_budget_requires_online_web_before_cuda(capsys):
    train = load_script("dcc_online_grpo")
    with pytest.raises(SystemExit) as error:
        train.main(["--model", "invented", "--train-data", "does-not-exist", "--output-dir", "unused",
            "--source-commit", "a" * 40, "--web-observation-token-budget", "6144"])
    assert error.value.code == 2 and "requires --enable-online-web" in capsys.readouterr().err
    with pytest.raises(ValueError, match="requires --enable-online-web"):
        asyncio.run(train.train(SimpleNamespace(web_observation_token_budget=6144), [], [], 1))
