"""Semantic CPU checks: complete messages, honest omissions, native phase IDs.

Engine outputs below are synthetic. No GPU, model loader, network or TEST data
is used. Official-tokenizer replay is a separate, explicit local CPU command.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import json

import pytest

from slime_plugins.dcc import EpisodeRunner, Task
from slime_plugins.dcc.eval_sampling import EvaluationSampling
from slime_plugins.dcc.hf_policy import encode_messages
from slime_plugins.dcc.native_budget import NativeBudget
from slime_plugins.dcc.native_thinking_eval import (
    NativeThinkingPolicy, NativeTraceJournal, validate_native_sidecars,
)
from slime_plugins.dcc.prompt_history import (
    PromptHistoryConfig, PromptHistoryBudgetExceeded, add_prompt_history_arguments,
    fit_prompt_history, prompt_history_from_arguments, validate_prompt_history_records,
)
from slime_plugins.dcc.vllm_policy import AsyncVLLMPolicy
from test_dcc_native_thinking import FakeEngine, FakeTokenizer, RECEIPT, declarations


def observation(value, remaining=2):
    return {"role": "user", "content": "Tool observation (source material):\n"
            + json.dumps(value, ensure_ascii=False) + f"\nRemaining turns: {remaining}."}


def messages_with_sources(count=5, body_chars=500):
    messages = [{"role": "system", "content": "Use only literal source evidence. Return complete JSON."},
                {"role": "user", "content": "Original task 原任务: exact columns and conditions."}]
    for index in range(count):
        url = f"https://example.org/observed/{index}?unchanged=yes#literal"
        messages += [{"role": "assistant", "content": json.dumps({
            "tool": "fetch_url", "arguments": {"url": url}})},
            observation({"url": url, "text": ('引文 \\" {完整} ' * (body_chars // 10)),
                         "start": 0, "end": body_chars})]
    messages.append({"role": "user", "content": '\nFinal turn: return exactly {"answer":[]}.'})
    return messages


def fit(tokenizer, messages, limit=1500, *, keep=1, max_model_len=None):
    return fit_prompt_history(tokenizer, messages, max_prompt_tokens=limit,
        max_new_tokens=20480, max_model_len=limit + 20480 if max_model_len is None else max_model_len,
        config=PromptHistoryConfig(True, keep))


def test_configuration_is_explicit_roundtrip_and_cli_default_off():
    for enabled in (False, True):
        value = PromptHistoryConfig(enabled)
        assert PromptHistoryConfig.from_dict(value.to_dict()) == value
    parser = argparse.ArgumentParser()
    add_prompt_history_arguments(parser)
    assert prompt_history_from_arguments(parser.parse_args([])) == PromptHistoryConfig()
    assert prompt_history_from_arguments(parser.parse_args([
        "--prompt-history-whole-turns", "--prompt-history-keep-recent-turns", "2"])) == PromptHistoryConfig(True, 2)
    with pytest.raises(ValueError):
        prompt_history_from_arguments(parser.parse_args(["--prompt-history-keep-recent-turns", "2"]))
    with pytest.raises(ValueError):
        PromptHistoryConfig.from_dict({"enabled": True})
    mistyped = PromptHistoryConfig(True).to_dict()
    mistyped["source_shortening"] = 0
    with pytest.raises(ValueError):
        PromptHistoryConfig.from_dict(mistyped)
    with pytest.raises(ValueError):
        validate_prompt_history_records(({}, {"prompt_history": PromptHistoryConfig(True).to_dict()}))


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5, "2"])
def test_keep_recent_turns_rejects_invalid_values(invalid):
    with pytest.raises(ValueError):
        PromptHistoryConfig(True, invalid)


def test_fit_request_is_identical_at_exact_boundary():
    tokenizer = FakeTokenizer()
    messages = messages_with_sources(2, 100)
    full = encode_messages(tokenizer, messages, 100000, allow_truncation=False)
    plan = fit(tokenizer, messages, len(full))
    assert plan.messages == messages
    assert plan.prompt_ids == full
    assert not plan.report["compacted"]
    assert plan.report["omitted_turns"] == []
    assert plan.report["notice"] is None


def test_whole_turn_eviction_preserves_json_current_controls_and_original_input():
    tokenizer = FakeTokenizer()
    messages = messages_with_sources()
    original = deepcopy(messages)
    plan = fit(tokenizer, messages)
    assert plan.report["compacted"] and len(plan.prompt_ids) <= 1500
    assert messages == original
    assert plan.messages[:2] == messages[:2]
    assert plan.messages[-3:] == messages[-3:]  # latest action, full observation, control
    retained = [messages[index] for index in plan.report["retained_message_indices"]]
    assert plan.messages == retained[:2] + [plan.report["notice"]] + retained[2:]
    assert all(group["message_indices"] == [2 + 2 * group["turn_index"], 3 + 2 * group["turn_index"]]
               for group in plan.report["omitted_turns"])
    assert all(row["kind"] == "read-source" for row in plan.report["omitted_observed_urls"])
    assert "https://" not in plan.report["notice"]["content"]
    assert "supplies no source evidence" in plan.report["notice"]["content"]
    assert plan.prompt_ids == encode_messages(tokenizer, plan.messages, 1500, allow_truncation=False)
    # Every source observation is an unchanged, complete parseable JSON value.
    for item in plan.messages:
        if item["content"].startswith("Tool observation (source material):\n"):
            value, end = json.JSONDecoder().raw_decode(item["content"].split("\n", 1)[1])
            assert value["url"] and value["text"]


def test_omitted_url_metadata_distinguishes_search_links_and_failed_reads():
    messages = messages_with_sources(4, 500)
    messages[3] = observation({"results": [{"url": "https://example.org/unread"}],
        "auto_read_sources": [{"url": "https://example.org/read", "text": "literal source"},
                              {"url": "https://example.org/fail", "error": "blocked"}]})
    plan = fit(FakeTokenizer(), messages, 1500)
    kinds = {(item["url"], item["kind"]) for item in plan.report["omitted_observed_urls"]}
    assert ("https://example.org/unread", "search-result; not read evidence") in kinds
    assert ("https://example.org/read", "read-source") in kinds
    assert ("https://example.org/fail", "failed-read; not evidence") in kinds


def test_recent_successful_read_survives_later_error_and_feedback():
    messages = messages_with_sources(4, 100)
    messages.pop()  # no final control until after feedback
    messages += [{"role": "assistant", "content": '{"tool":"fetch_url","arguments":{}}'},
                 observation({"url": "https://example.org/failed", "error": "timeout"}),
                 {"role": "assistant", "content": '{"answer":[]}'},
                 {"role": "user", "content": "Evidence check: final answer rejected. Keep current controls."}]
    plan = fit(FakeTokenizer(), messages, 1300)
    assert plan.report["compacted"]
    assert plan.messages[-6:] == messages[-6:]
    assert messages[-5]["content"] in [m["content"] for m in plan.messages]


def test_oversized_latest_source_fails_instead_of_cutting_json():
    tokenizer = FakeTokenizer()
    messages = messages_with_sources(3, 5000)
    original = deepcopy(messages)
    with pytest.raises(PromptHistoryBudgetExceeded) as caught:
        fit(tokenizer, messages, 1000)
    assert caught.value.reason == "minimum_complete_context_exceeds_budget"
    assert caught.value.diagnostics["minimum_prompt_tokens"] > 1000
    assert caught.value.diagnostics["actual_prompt_tokens"] is None
    assert caught.value.original_prompt_ids == encode_messages(tokenizer, original, 100000, allow_truncation=False)
    assert messages == original


def test_large_task_or_unknown_roles_fail_without_slicing_or_deleting_instructions():
    messages = messages_with_sources()
    messages[1]["content"] *= 100
    with pytest.raises(PromptHistoryBudgetExceeded):
        fit(FakeTokenizer(), messages, 1000)
    messages = messages_with_sources()
    messages.insert(4, {"role": "system", "content": "Do not delete this instruction."})
    with pytest.raises(PromptHistoryBudgetExceeded, match="unsupported_history_shape"):
        fit(FakeTokenizer(), messages, 1000)


def test_full_completion_reservation_reduces_only_available_prompt_space():
    messages = messages_with_sources()
    plan = fit(FakeTokenizer(), messages, 10000, max_model_len=20480 + 1500)
    assert plan.report["effective_prompt_limit"] == 1500
    assert plan.report["max_new_tokens_reserved"] == 20480
    assert len(plan.prompt_ids) + 20480 <= 21980


def make_native(tmp_path, values, *, limit=7500):
    pytest.importorskip("vllm")
    pytest.importorskip("xgrammar")
    tokenizer = FakeTokenizer()
    budget = NativeBudget(16384, "thinking")
    scripts = []
    for value in values:
        scripts += [{"ids": tokenizer.encode("literal synthetic thought</think>")},
                    {"ids": tokenizer.encode(json.dumps(value)) + [tokenizer.eos_token_id]}]
    engine = FakeEngine(scripts)
    base = AsyncVLLMPolicy(engine, tokenizer, max_prompt_tokens=limit,
        max_model_len=limit + budget.total_tokens, allow_truncation=False,
        eos_token_ids=[151643, 151645], evaluation_sampling=EvaluationSampling(),
        web_structured_final=True, web_structured_actions=True)
    native = NativeThinkingPolicy(base, source_receipt=RECEIPT, budget=budget,
        trace_sink=NativeTraceJournal(tmp_path / "native_thinking_traces_rank0.jsonl"),
        prompt_history_config=PromptHistoryConfig(True))
    return tokenizer, engine, native


def run_cuad_native(tmp_path, values, *, document, max_steps=3, limit=7500):
    tokenizer, engine, native = make_native(tmp_path, values, limit=limit)
    task = Task("contract", "cuad", "validation", "Extract observed confidentiality clauses.",
                document=document, gold=[])
    async def run():
        bound = native.bind_task(task.id, 0)
        runner = EpisodeRunner(max_steps=max_steps, max_new_tokens=native.budget.total_tokens,
                               initial_chunks=0, cuad_structured_actions=True)
        result = await runner.run(task, bound, evaluation=True)
        record = result.to_dict()
        record["native_thinking_traces"] = bound.references
        manifest, runtime = declarations(max_steps, budget=native.budget)
        for value in (manifest, runtime):
            value["max_prompt_tokens"] = limit
            value["prompt_history"] = native.prompt_history_config.to_dict()
        manifest["inference_parameters"]["max_model_len"] = limit + native.budget.total_tokens
        runtime["max_model_len"] = limit + native.budget.total_tokens
        await native.close()
        audit = validate_native_sidecars(tmp_path, {task.id: record}, manifest, runtime,
                                         {task.id: 0}, [task], tokenizer=tokenizer)
        return result, record, manifest, runtime, audit
    result, record, manifest, runtime, audit = asyncio.run(run())
    return tokenizer, engine, native, task, result, record, manifest, runtime, audit


def test_native_two_stages_keep_complete_thought_json_budgets_and_actual_ids(tmp_path):
    values = [{"tool": "read_document", "arguments": {"start": 0, "end": 4000}},
              {"tool": "read_document", "arguments": {"start": 4000, "end": 8000}},
              {"plan": "Review exact category", "checks": "No qualifying span", "answer": []}]
    tok, engine, native, task, result, record, manifest, runtime, audit = run_cuad_native(
        tmp_path, values, document="Literal source sentence. " * 700)
    assert result.status == "completed" and audit["traces"] == 3
    assert audit["context_budget_failures"] == 0
    assert native.traces[-1]["prompt_history"]["compacted"]
    assert len(engine.calls) == 6
    for trace, first, second in zip(native.traces, engine.calls[::2], engine.calls[1::2], strict=True):
        assert first[1].max_tokens == 16384
        assert second[1].max_tokens == 4096
        assert first[0]["prompt_token_ids"] == trace["original_prompt_ids"]
        assert second[0]["prompt_token_ids"] == first[0]["prompt_token_ids"] + trace["phases"][0]["token_ids"]
        assert len(first[0]["prompt_token_ids"]) <= 7500
    # Tampering with omissions cannot pass a self-consistency-only audit.
    path = tmp_path / "native_thinking_traces_rank0.jsonl"
    traces = [json.loads(line) for line in path.read_text().splitlines()]
    traces[-1]["prompt_history"]["omitted_message_indices"] = []
    path.write_text("".join(json.dumps(trace) + "\n" for trace in traces))
    with pytest.raises(ValueError, match="effective prompt or audit report"):
        validate_native_sidecars(tmp_path, {task.id: record}, manifest, runtime,
                                 {task.id: 0}, [task], tokenizer=tok)


def test_native_unfit_minimum_is_zero_generation_failure_with_full_auditable_request(tmp_path):
    values = [{"tool": "read_document", "arguments": {"start": 0, "end": 12000}}]
    tok, engine, native, task, result, record, manifest, runtime, audit = run_cuad_native(
        tmp_path, values, document="Literal contract source. " * 1000, limit=7500)
    assert result.status == "context_budget_failure"
    assert result.trainable_steps == []
    assert len(engine.calls) == 2 and len(result.steps) == 1
    assert audit["context_budget_failures"] == 1
    trace = native.traces[-1]
    assert trace["status"] == "preflight_failure"
    assert trace["generation_submitted"] is False and trace["phases"] == []
    assert trace["completion_ids"] == [] and trace["observed_completion_tokens"] == 0
    assert trace["prompt_history_original_messages"] == result.metrics["prompt_history_failure"]["request_messages"]
    assert trace["prompt_history"]["minimum_prompt_tokens"] > 7500
    assert len(json.JSONDecoder().raw_decode(result.messages[-1]["content"].split("\n", 1)[1])[0]["text"]) == 12000
