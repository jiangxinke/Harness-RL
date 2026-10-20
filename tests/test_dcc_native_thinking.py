"""CPU-only synthetic protocol tests; no model, official tokenizer, GPU or network.

All fixtures are local fakes. Optional real vLLM/xgrammar imports test parameter
API/schema compatibility only; no engine/model loader is ever constructed.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import re
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator, ValidationError

from slime_plugins.dcc import EpisodeRunner, PolicyCompletion, Task
from slime_plugins.dcc.cuad_structured import (
    cuad_action_schema, cuad_final_schema, cuad_structured_configuration,
    cuad_backend_schema, cuad_structured_options,
)
from slime_plugins.dcc.episode import _system_prompt, _valid_answer, final_turn_message, parse_action
from slime_plugins.dcc.eval_sampling import EvaluationSampling
from slime_plugins.dcc.native_budget import NativeBudget, BUDGET_PROTOCOL
from slime_plugins.dcc.native_thinking_eval import (
    BASE_SOURCE_COMMIT, END_THINK, NATIVE_SUFFIX, PROTOCOL, NativeThinkingPolicy,
    NativeTraceJournal, native_configuration, native_options, validate_native_sidecars,
    _strict_json, _stage2_eos_valid,
)
from slime_plugins.dcc.tools import DocumentTools
from slime_plugins.dcc.vllm_policy import AsyncVLLMPolicy
from slime_plugins.dcc.web_structured import web_final_schema

# Explicitly synthetic identity. This is not a real adopted-source commit.
RECEIPT = {"version": "dcc-native-inference-source-v1", "source_commit": "f" * 40,
           "base_source_commit": BASE_SOURCE_COMMIT, "protocol": PROTOCOL, "frozen": True}
DOCUMENT = "Alpha agrees to preserve confidential information. Beta may terminate on notice."


class FakeTokenizer:
    """Character mapping and artificial template, never an official tokenizer."""
    eos_token_id = 151645
    tags = {"<|im_start|>": 151644, "<|im_end|>": 151645,
            "<|endoftext|>": 151643, "<think>": 151667, "</think>": 151668}

    def __init__(self):
        self.message_calls = []

    def encode(self, text, *, add_special_tokens=False):
        parts = re.split("(" + "|".join(re.escape(tag) for tag in self.tags) + ")", text)
        return [token for part in parts for token in
                ([self.tags[part]] if part in self.tags else [10000 + ord(char) for char in part])]

    def decode(self, ids, *, skip_special_tokens):
        reverse = {value: key for key, value in self.tags.items()}
        return "".join(reverse[token] if token in reverse else chr(token - 10000)
                       for token in ids if not (skip_special_tokens and token in reverse))

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        self.message_calls.append(deepcopy(messages))
        text = "".join("<|im_start|>" + message["role"] + "\n" + message["content"] + "<|im_end|>\n"
                       for message in messages)
        text += NATIVE_SUFFIX if add_generation_prompt else ""
        return self.encode(text) if tokenize else text


class FakeEngine:
    def __init__(self, scripts):
        from vllm.config import StructuredOutputsConfig
        self.vllm_config = SimpleNamespace(structured_outputs_config=StructuredOutputsConfig(
            backend="xgrammar", disable_fallback=True))
        self.scripts, self.calls, self.aborts = list(scripts), [], []

    async def generate(self, prompt, params, request_id, *, lora_request):
        self.calls.append((deepcopy(prompt), params, request_id))
        script = self.scripts.pop(0)
        if script.get("block"):
            await asyncio.Event().wait()
        ids = list(script["ids"])
        output = SimpleNamespace(token_ids=ids, text="synthetic engine text",
            finish_reason=script.get("finish", "stop"),
            stop_reason=script.get("stop", ids[-1] if ids else None))
        yield SimpleNamespace(prompt_token_ids=script.get("prompt", prompt["prompt_token_ids"]),
            finished=script.get("finished", True), outputs=[output])

    async def abort(self, request_id):
        self.aborts.append(request_id)

    def shutdown(self):
        pass


def make_policy(tmp_path, *, values=None, scripts=None, review=False, budget=None):
    pytest.importorskip("vllm")
    pytest.importorskip("xgrammar")
    tokenizer = FakeTokenizer()
    budget = NativeBudget() if budget is None else budget
    max_model_len = 16384 + budget.total_tokens
    if scripts is None:
        scripts = []
        for value in values:
            scripts += [{"ids": tokenizer.encode('Ignore this thought JSON: {"answer":["POISON"]}</think>')},
                        {"ids": tokenizer.encode(json.dumps(value)) + [151645]}]
    engine = FakeEngine(scripts)
    base = AsyncVLLMPolicy(engine, tokenizer, max_prompt_tokens=16384, max_model_len=max_model_len,
        eos_token_ids=[151643, 151645], evaluation_sampling=EvaluationSampling(),
        web_structured_final=True, web_structured_actions=True,
        provenance={"source_commit": BASE_SOURCE_COMMIT, "web_explicit_review": {"enabled": review}})
    journal = NativeTraceJournal(tmp_path / "native_thinking_traces_rank0.jsonl")
    return tokenizer, engine, NativeThinkingPolicy(base, source_receipt=RECEIPT, trace_sink=journal, budget=budget)


def cuad_task():
    return Task("contract", "cuad", "validation", "Extract confidentiality clauses.",
                document=DOCUMENT, gold=[])


def declarations(max_steps, *, review=False, budget=None):
    budget = NativeBudget() if budget is None else budget
    config = native_configuration(RECEIPT, budget=budget)
    max_model_len = 16384 + budget.total_tokens
    return ({"native_thinking": config, "max_new_tokens": budget.total_tokens, "max_steps": max_steps,
             "world_size": 1, "seed": 42, "evaluation_sampling": EvaluationSampling().to_dict(),
             "max_prompt_tokens": 16384,
             "inference_parameters": {"async_engine": True, "max_model_len": max_model_len},
             "web_explicit_review": {"enabled": review}},
            {"native_thinking": config, "source_commit": RECEIPT["source_commit"],
             "eos_token_ids": [151643, 151645], "max_prompt_tokens": 16384,
             "max_model_len": max_model_len})


def run_cuad(tmp_path, *, values=None, scripts=None, max_steps=3, review=False, timeout=10, budget=None):
    tokenizer, engine, policy = make_policy(tmp_path, values=values, scripts=scripts, review=review, budget=budget)
    async def run():
        bound = policy.bind_task("contract", 0)
        runner = EpisodeRunner(max_steps=max_steps, max_new_tokens=policy.budget.total_tokens, initial_chunks=1,
            cuad_structured_actions=True, timeout_seconds=timeout)
        result = await runner.run(cuad_task(), bound, evaluation=True)
        record = result.to_dict(); record["native_thinking_traces"] = bound.references
        manifest, runtime = declarations(max_steps, review=review, budget=policy.budget)
        audit = validate_native_sidecars(tmp_path, {"contract": record}, manifest, runtime,
                                         {"contract": 0}, [cuad_task()], tokenizer=tokenizer)
        await policy.close()
        return tokenizer, engine, policy, result, record, manifest, runtime, audit
    return asyncio.run(run())


def test_cuad_public_schema_defaults_and_canonical_answers():
    assert cuad_structured_configuration() == {"enabled": False, "schema_version": None,
        "backend": None, "final_review_max_words": None}
    with pytest.raises(ValueError): cuad_structured_configuration(1)
    action = Draft202012Validator(cuad_action_schema())
    tools = DocumentTools(DOCUMENT)
    for value in [{"tool": "search_document", "arguments": {"query": "Alpha"}},
                  {"tool": "search_document", "arguments": {"query": "Alpha", "top_k": -1}},
                  {"tool": "read_document", "arguments": {}},
                  {"tool": "read_document", "arguments": {"start": 0, "end": None}}]:
        action.validate(value)
        assert "error" not in getattr(tools, value["tool"])(**value["arguments"])
    for answer in [[], [{"text": DOCUMENT, "confidence": .5}], [{"text": " span ", "confidence": 1}]]:
        final = {"plan": "Find exact clause", "checks": "Read actual source", "answer": answer}
        Draft202012Validator(cuad_final_schema()).validate(final)
        assert parse_action(json.dumps(final))["answer"] == answer
        assert _valid_answer(cuad_task(), answer)
    assert list(cuad_final_schema()["properties"]) == ["plan", "checks", "answer"]
    # A fresh schema is independent of changes to a previous return value.
    altered = cuad_final_schema(); altered["properties"]["answer"] = {}
    assert altered != cuad_final_schema()


@pytest.mark.parametrize("span", [{"text": " ", "confidence": .5}, {"text": "", "confidence": .5},
    {"text": "span", "confidence": -0.1}, {"text": "span", "confidence": 1.1},
    {"text": "span", "confidence": "0.5"}, {"text": "span", "confidence": True},
    {"text": "span"}, "legacy span"])
def test_cuad_schema_rejects_noncanonical_span_and_confidence(span):
    with pytest.raises(ValidationError):
        Draft202012Validator(cuad_final_schema()).validate({"plan": "Review", "checks": "Read", "answer": [span]})


def test_cuad_final_review_preserves_original_40_word_contract():
    schema = Draft202012Validator(cuad_final_schema())
    schema.validate({"plan": "longword " * 40, "checks": "Read", "answer": []})
    with pytest.raises(ValidationError):
        schema.validate({"plan": "longword " * 41, "checks": "Read", "answer": []})
    with pytest.raises(ValidationError): schema.validate({"answer": []})
    with pytest.raises(ValidationError):
        schema.validate({"plan": "Review", "checks": "Read", "tool": "read_document", "arguments": {}})
    assert "at most 40 words" in final_turn_message(cuad_task())["content"]


def test_native_cuad_real_ids_tools_final_and_legacy_prompts_equal(tmp_path):
    values = [{"tool": "search_document", "arguments": {"query": "confidential"}},
              {"tool": "read_document", "arguments": {"start": 0, "end": len(DOCUMENT)}},
              {"plan": "Extract confidentiality", "checks": "Observed verbatim clause",
               "answer": [{"text": DOCUMENT.split(". ")[0] + ".", "confidence": .9}]}]
    tok, engine, policy, result, record, manifest, runtime, audit = run_cuad(tmp_path, values=values)
    assert result.status == "completed" and len(engine.calls) == 6 and audit["traces"] == 3
    assert [event["tool"] for event in result.tool_events] == ["search_document", "read_document"]
    assert all("POISON" not in step.text and step.logprobs is None for step in result.steps)
    for index, step in enumerate(result.steps):
        first, second = engine.calls[2 * index:2 * index + 2]
        trace = policy.traces[index]
        assert second[0]["prompt_token_ids"] == step.prompt_ids + trace["phases"][0]["token_ids"]
        assert step.completion_ids == trace["completion_ids"]
        assert first[1].structured_outputs is None
        public_schema = cuad_final_schema() if index == 2 else cuad_action_schema()
        assert second[1].structured_outputs.json == cuad_backend_schema(public_schema)
        assert trace["json_schema"] == public_schema
        assert first[1].max_tokens == second[1].max_tokens == 4096
        assert trace["phases"][0]["token_ids"][-1] == END_THINK
        assert trace["phases"][1]["token_ids"][-1] == 151645
    class LegacyPolicy:
        def __init__(self): self.values, self.calls = iter(values), []
        async def __call__(self, messages, **kwargs):
            self.calls.append(deepcopy(messages))
            return PolicyCompletion(json.dumps(next(self.values)), [1], [2])
        async def generate_structured(self, *args, **kwargs):
            raise AssertionError("default-OFF CUAD changed route")
    legacy = LegacyPolicy()
    old = asyncio.run(EpisodeRunner(max_steps=3, max_new_tokens=8192, initial_chunks=1,
        web_structured_final=True, web_structured_actions=True, web_explicit_review=True).run(
            cuad_task(), legacy, evaluation=True))
    assert old.answer == result.answer and old.metrics == result.metrics and old.tool_events == result.tool_events
    assert legacy.calls == tok.message_calls
    assert legacy.calls[0][0]["content"] == _system_prompt(cuad_task(), 3)


def test_native_cuad_empty_early_final_uses_action_schema(tmp_path):
    tok, engine, policy, result, *rest = run_cuad(tmp_path, values=[{"answer": []}])
    assert result.status == "completed" and result.answer == [] and len(engine.calls) == 2
    assert policy.traces[0]["json_schema"] == cuad_action_schema()


def test_cuad_final_review_is_distinct_from_enabled_web_review(tmp_path):
    value = {"plan": "longword " * 40, "checks": "Observed absence", "answer": []}
    tok, engine, policy, result, *rest = run_cuad(tmp_path, values=[value], max_steps=1, review=True)
    assert result.status == "completed" and len(value["plan"]) > 240
    assert policy.traces[0]["json_schema"] == cuad_final_schema()


@pytest.mark.parametrize("damage", ["thought", "eos", "alternate_none", "duplicate", "schema", "confidence",
                                    "missing_review", "blank_span", "long_review", "empty_checks"])
def test_native_cuad_protocol_failures_never_retry_or_execute_tools(tmp_path, damage):
    tokenizer = FakeTokenizer()
    thought = {"ids": tokenizer.encode("Reason.</think>")}
    value = {"plan": "Read", "checks": "Review", "answer": []}
    final = {"ids": tokenizer.encode(json.dumps(value)) + [151645]}
    scripts = [thought, final]
    if damage == "thought": scripts = [{"ids": tokenizer.encode("unfinished"), "finish": "length", "stop": None}]
    if damage == "eos": final["stop"] = 151643
    if damage == "alternate_none": final.update(ids=tokenizer.encode(json.dumps(value)) + [151643], stop=None)
    if damage == "duplicate": final["ids"] = tokenizer.encode('{"plan":"Read","checks":"Review","answer":[],"answer":[]}') + [151645]
    if damage == "schema": final["ids"] = tokenizer.encode('{"tool":"fetch_url","arguments":{"url":"fixture"}}') + [151645]
    if damage == "confidence":
        value["answer"] = [{"text": "span", "confidence": 1.1}]
        final["ids"] = tokenizer.encode(json.dumps(value)) + [151645]
    if damage == "missing_review": final["ids"] = tokenizer.encode('{"answer":[]}') + [151645]
    if damage == "blank_span": value["answer"] = [{"text": " \t ", "confidence": .5}]
    if damage == "long_review": value["plan"] = "word " * 41
    if damage == "empty_checks": value["checks"] = ""
    if damage in {"blank_span", "long_review", "empty_checks"}:
        final["ids"] = tokenizer.encode(json.dumps(value)) + [151645]
    tok, engine, policy, result, record, manifest, runtime, audit = run_cuad(tmp_path, scripts=scripts, max_steps=1)
    assert result.status == "model_protocol_failure" and result.reward == 0
    assert len(engine.calls) == (1 if damage == "thought" else 2)
    assert result.tool_events == [] and result.trainable_steps == []
    assert len(result.steps) == 1 and result.steps[0].text == ""
    assert result.steps[0].completion_ids == policy.traces[0]["completion_ids"]
    assert audit["model_protocol_failures"] == 1


def test_native_cuad_sidecar_rejects_corrupt_raw_schema_source_and_reference(tmp_path):
    tok, engine, policy, result, record, manifest, runtime, audit = run_cuad(tmp_path, values=[{"answer": []}])
    path = tmp_path / "native_thinking_traces_rank0.jsonl"
    original = path.read_text(); trace = json.loads(original)
    for damage in ["raw", "source", "schema", "ref"]:
        changed = deepcopy(trace); saved = deepcopy(record)
        if damage == "raw": changed["phases"][1]["raw_outputs"][0]["token_ids"][0] += 1
        if damage == "source": changed["source_commit"] = "e" * 40
        if damage == "schema": changed["json_schema"] = cuad_final_schema()
        if damage == "ref": saved["native_thinking_traces"][0]["turn_index"] = 2
        path.write_text(json.dumps(changed) + "\n")
        with pytest.raises(ValueError):
            validate_native_sidecars(tmp_path, {"contract": saved}, manifest, runtime,
                {"contract": 0}, [cuad_task()], tokenizer=tok)
    path.write_text(original + original)
    with pytest.raises(ValueError):
        validate_native_sidecars(tmp_path, {"contract": record}, manifest, runtime,
            {"contract": 0}, [cuad_task()], tokenizer=tok)


def test_native_web_original_schema_and_default_off_options(tmp_path):
    assert native_options(SimpleNamespace(native_thinking=False, native_source_receipt=None), [], None, tmp_path) is None
    assert validate_native_sidecars(tmp_path, {}, {}, None, {}, []) is None
    tok, engine, policy = make_policy(tmp_path, values=[{"answer": []}])
    schema = web_final_schema(["name"], [])
    async def run():
        completion = await policy.bind_task("web", 0).generate_structured(
            [{"role": "user", "content": "Find"}], json_schema=schema, max_new_tokens=8192, temperature=0.)
        await policy.close(); return completion
    completion = asyncio.run(run())
    assert completion.text == '{"answer": []}'
    assert engine.calls[1][1].structured_outputs.json == schema


def test_native_source_budget_sampling_and_cuad_contract(tmp_path):
    pytest.importorskip("vllm"); pytest.importorskip("xgrammar")
    receipt = tmp_path / ".dcc-native-source.json"; receipt.write_text(json.dumps(RECEIPT))
    args = SimpleNamespace(native_thinking=True, native_source_receipt=str(receipt),
        inference_backend="vllm", vllm_async=True, web_structured_final=True,
        web_structured_actions=True, adapter=None, max_new_tokens=8192, retry_infrastructure=False)
    config = native_options(args, [cuad_task()], EvaluationSampling(), tmp_path)
    assert config["cuad_structured"] == cuad_structured_configuration(True)
    assert config["benchmarks"] == ["widesearch", "cuad"]
    for field, value in [("max_new_tokens", 4096), ("vllm_async", False), ("retry_infrastructure", True)]:
        wrong = deepcopy(args); setattr(wrong, field, value)
        with pytest.raises(ValueError): native_options(wrong, [cuad_task()], EvaluationSampling(), tmp_path)
    receipt.write_text(json.dumps({**RECEIPT, "source_commit": BASE_SOURCE_COMMIT}))
    with pytest.raises(ValueError): native_options(args, [cuad_task()], EvaluationSampling(), tmp_path)


def test_native_timeout_preserves_durable_trace_without_fabricated_ids(tmp_path):
    tok, engine, policy, result, record, manifest, runtime, audit = run_cuad(
        tmp_path, scripts=[{"block": True}], timeout=.01)
    assert result.status == "infrastructure_error" and result.steps == []
    assert policy.traces[0]["completion_ids"] == []
    assert "token_ids" not in policy.traces[0]["phases"][0]
    assert audit["infrastructure_errors"] == 1 and engine.aborts


def test_strict_json_and_exact_eos_boundary():
    for text in ['{"x":1,"x":2}', '{"x":{"a":1,"a":2}}', '{"x":NaN}']:
        with pytest.raises(ValueError): _strict_json(text)
    phase = {"token_ids": [10100, 151643], "finish_reason": "stop", "stop_reason": 151645}
    assert not _stage2_eos_valid(phase, [151645, 151643], 151645)
    phase["stop_reason"] = None
    assert not _stage2_eos_valid(phase, [151645, 151643], 151645)
    phase["stop_reason"] = 151643
    assert _stage2_eos_valid(phase, [151645, 151643], 151645)


@pytest.mark.parametrize("thought_tokens,observed_tokens", [(8192, 5000), (16384, 9000)])
def test_native_v3_long_thought_actual_sampling_and_exact_continuation(tmp_path, thought_tokens, observed_tokens):
    budget = NativeBudget(thought_tokens, "thinking")
    tokenizer = FakeTokenizer()
    value = {"plan": "Read the contract", "checks": "No supported clause", "answer": []}
    thought_ids = tokenizer.encode("x" * observed_tokens + "</think>")
    scripts = [{"ids": thought_ids}, {"ids": tokenizer.encode(json.dumps(value)) + [151645]}]
    tok, engine, policy, result, record, manifest, runtime, audit = run_cuad(
        tmp_path, scripts=scripts, max_steps=1, budget=budget)
    from vllm import SamplingParams
    assert result.status == "completed" and audit["traces"] == 1
    assert policy.base.max_model_len == 16384 + thought_tokens + 4096
    first, second = engine.calls
    assert isinstance(first[1], SamplingParams) and isinstance(second[1], SamplingParams)
    assert first[1].max_tokens == thought_tokens and second[1].max_tokens == 4096
    assert (first[1].temperature, first[1].top_p, first[1].top_k) == (.6, .95, 20)
    # vLLM 0.11.2 normalizes configured greedy top_k=-1 to effective 0 (disabled).
    assert (second[1].temperature, second[1].top_p, second[1].top_k) == (0., 1., 0)
    assert first[1].structured_outputs is None
    assert second[1].structured_outputs.json == cuad_backend_schema(cuad_final_schema())
    trace = policy.traces[0]
    assert trace["json_schema"] == cuad_final_schema()
    assert trace["protocol"] == BUDGET_PROTOCOL
    assert trace["phases"][0]["token_ids"] == thought_ids and len(thought_ids) > 4096
    assert second[0]["prompt_token_ids"] == result.steps[0].prompt_ids + thought_ids
    assert result.steps[0].completion_ids == thought_ids + scripts[1]["ids"]
    assert trace["native_thinking"] == manifest["native_thinking"] == runtime["native_thinking"]
    assert trace["phases"][0]["effective_sampling"] == trace["native_thinking"]["thought_effective_sampling"]
    assert trace["phases"][1]["effective_sampling"] == trace["native_thinking"]["json_effective_sampling"]
    assert manifest["max_new_tokens"] == thought_tokens + 4096
    assert runtime["max_model_len"] == manifest["inference_parameters"]["max_model_len"]


@pytest.mark.parametrize("damage", ["no_close", "duplicate_close", "ordinary_eos"])
def test_native_v3_real_close_failure_keeps_original_ids_without_json_or_retry(tmp_path, damage):
    tokenizer = FakeTokenizer()
    budget = NativeBudget(16384, "thinking")
    script = {"ids": tokenizer.encode("x" * budget.thought_tokens), "finish": "length", "stop": None}
    if damage == "duplicate_close": script = {"ids": tokenizer.encode("Reason.</think></think>")}
    if damage == "ordinary_eos": script = {"ids": tokenizer.encode("Reason.<|im_end|>"), "stop": 151645}
    tok, engine, policy, result, record, manifest, runtime, audit = run_cuad(
        tmp_path, scripts=[script], max_steps=1, budget=budget)
    assert result.status == "model_protocol_failure" and len(engine.calls) == 1
    assert result.tool_events == [] and result.trainable_steps == []
    assert policy.traces[0]["failure_reason"] == "thought_did_not_stop_on_real_close"
    assert policy.traces[0]["completion_ids"] == script["ids"] == result.steps[0].completion_ids
    assert record["metrics"]["model_protocol_failure"]["protocol"] == BUDGET_PROTOCOL
    assert audit["model_protocol_failures"] == 1


@pytest.mark.parametrize("damage", ["phase_budget", "sampling", "effective_sampling", "config", "context", "total"])
def test_native_v3_sidecar_rejects_budget_sampling_and_context_changes(tmp_path, damage):
    budget = NativeBudget(16384, "thinking")
    tok, engine, policy, result, record, manifest, runtime, audit = run_cuad(
        tmp_path, values=[{"answer": []}], budget=budget)
    assert audit["traces"] == 1
    path = tmp_path / "native_thinking_traces_rank0.jsonl"
    trace = json.loads(path.read_text())
    if damage == "phase_budget": trace["phases"][0]["max_tokens"] = 4096
    if damage == "sampling": trace["phases"][0]["sampling"]["temperature"] = 0.
    if damage == "effective_sampling": trace["phases"][0]["effective_sampling"]["top_k"] = 0
    if damage == "config":
        manifest = deepcopy(manifest)
        manifest["native_thinking"]["thought_max_tokens"] = 8192
    if damage == "context": runtime["max_model_len"] -= 1
    if damage == "total": manifest["max_new_tokens"] = 8192
    path.write_text(json.dumps(trace) + "\n")
    with pytest.raises(ValueError):
        validate_native_sidecars(tmp_path, {"contract": record}, manifest, runtime,
            {"contract": 0}, [cuad_task()], tokenizer=tok)
