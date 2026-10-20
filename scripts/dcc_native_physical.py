"""Shared completed native TRAIN physical integrity, without answer acceptance.

SFT applies its existing answer/grounding/polarity gates AFTER this helper.
DPO applies trusted public-prefix ownership and separate offline TRAIN rewards.
Every physical validation condition and exception remains unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from jsonschema import Draft202012Validator
from slime_plugins.dcc.eval_sampling import EvaluationSampling
from slime_plugins.dcc.native_budget import effective_sampling
from slime_plugins.dcc.hf_policy import encode_messages
from dcc_native_sft import (
    CONTRACT, END_THINK, NATIVE_SUFFIX, PHASE_TOKENS, NativeIdentity,
    _ids, _json_key, _require, _strict_json, validate_source_receipt,
)


@dataclass(frozen=True)
class PhysicalNativeCompletion:
    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]
    phase_lengths: tuple[int, int]
    public_json: str


def validate_native_physical(bundle, tokenizer, *, expected_identity: NativeIdentity,
                             original_schema: dict):
    """Validate actual two-phase token ownership against caller approval.

    This checks completion integrity only; it imposes no SFT correctness or
    polarity gate and never constructs, crops, repairs or retemplates targets.
    original_schema must come from trusted public runtime reconstruction.
    """
    _require(isinstance(bundle, dict) and bundle.get("contract") == CONTRACT,
             "Explicit native TRAIN contract required")
    _require(set(bundle) == {"contract", "record", "identity", "collector", "native_trace"},
             "Unexpected native bundle fields")
    _require(_json_key(bundle["identity"]) == _json_key(expected_identity.manifest(tokenizer)),
             "Native artifact/tokenizer/runtime identity mismatch")
    collector = bundle["collector"]
    _require(isinstance(collector, dict), "Recorded TRAIN collector provenance required")
    for key, expected in {
        "split": "train", "task_input": "public-train-task-without-gold",
        "gold_visible_to_policy": False, "recorded_messages": True,
        "synthetic_thought": False, "inserted_close": False, "inserted_eos": False,
    }.items():
        _require(type(collector.get(key)) is type(expected) and collector[key] == expected,
                 "Invalid TRAIN collector assertion: " + key)
    _require(isinstance(collector.get("episode_id"), str) and bool(collector["episode_id"].strip())
             and type(collector.get("turn")) is int and collector["turn"] >= 0,
             "Actual episode ID and zero-based turn are required")

    record, trace = bundle["record"], bundle["native_trace"]
    _require(isinstance(record, dict) and isinstance(trace, dict), "Record and trace must be objects")
    _require(record.get("split") == "train", "Native training accepts TRAIN only")
    _require(isinstance(original_schema, dict) and bool(original_schema),
             "Trusted original action schema is required")
    Draft202012Validator.check_schema(original_schema)
    budget = expected_identity.budget
    _require(trace.get("protocol") == budget.protocol
             and trace.get("source_commit") == expected_identity.source_commit,
             "Frozen native protocol/source mismatch")
    _require(trace.get("task_id") == record.get("task_id")
             and type(trace.get("turn_index")) is int and trace["turn_index"] == collector["turn"]
             and type(trace.get("rank")) is int and trace["rank"] >= 0
             and type(collector.get("rank")) is int and collector["rank"] == trace["rank"]
             and isinstance(trace.get("trace_id"), str) and bool(trace["trace_id"].strip())
             and collector.get("trace_id") == trace["trace_id"],
             "Actual trace/task/turn/collection-rank ownership mismatch")
    native = trace.get("native_thinking")
    _require(isinstance(native, dict) and native.get("protocol") == budget.protocol,
             "Actual native runtime configuration required")
    validate_source_receipt(native.get("source_receipt"), expected_identity.source_commit)
    if not budget.legacy:
        from dcc_historical_native_config import HISTORICAL_TRAIN_SOURCE, historical_native_configuration
        if expected_identity.source_commit == HISTORICAL_TRAIN_SOURCE:
            approved_configuration = historical_native_configuration(native["source_receipt"], budget=budget)
        else:
            from slime_plugins.dcc.native_thinking_eval import native_configuration
            approved_configuration = native_configuration(native["source_receipt"], budget=budget)
        _require(_json_key(native) == _json_key(approved_configuration),
                 "Actual budgeted TRAIN configuration differs from caller approval")
    _require(trace.get("status") == "completed" and trace.get("failure_reason") is None
             and trace.get("json_complete") is True and trace.get("json_schema_valid") is True,
             "Only completed schema-valid native rollouts can be SFT targets")
    _require(trace.get("public_text_contains_thought") is False,
             "Public completion must contain only phase-two JSON")
    for key, limit in (("declared_max_new_tokens", budget.total_tokens),
                       ("thought_max_tokens", budget.thought_tokens), ("json_max_tokens", PHASE_TOKENS)):
        _require(type(trace.get(key)) is int and trace[key] == limit, "Native budget mismatch: " + key)
    _require(_json_key(trace.get("json_schema")) == _json_key(original_schema),
             "Recorded schema differs from trusted original schema")

    vocabulary = len(tokenizer)
    prompt = _ids(trace.get("original_prompt_ids"), vocabulary=vocabulary, name="original prompt")
    if expected_identity.prompt_history is not None:
        from slime_plugins.dcc.prompt_history import PromptHistoryConfig, fit_prompt_history
        config = PromptHistoryConfig.from_dict(expected_identity.prompt_history)
    else:
        config = None
    if config is not None and config.enabled:
        _require(_json_key(trace.get("prompt_history_original_messages")) == _json_key(record["messages"]),
                 "Prompt history audit differs from complete original record messages")
        plan = fit_prompt_history(tokenizer, record["messages"],
            max_prompt_tokens=expected_identity.max_prompt_tokens,
            max_new_tokens=budget.total_tokens, max_model_len=expected_identity.max_model_len,
            config=config)
        _require(_json_key(trace.get("prompt_history")) == _json_key(plan.report),
                 "Prompt history receipt differs from trusted deterministic replay")
        _require(prompt == plan.prompt_ids,
                 "Recorded prompt differs from exact effective whole-turn encoding")
    else:
        _require("prompt_history" not in trace and "prompt_history_original_messages" not in trace,
                 "Unapproved prompt history audit in unmanaged native record")
        actual_prompt = encode_messages(tokenizer, record["messages"],
                                        expected_identity.max_prompt_tokens, allow_truncation=False)
        _require(prompt == actual_prompt, "Recorded prompt differs from exact original message encoding")
    suffix = tokenizer.encode(NATIVE_SUFFIX, add_special_tokens=False)
    _require(suffix and prompt[-len(suffix):] == suffix,
             "Prompt must use the native open-thinking generation suffix")
    _require(len(prompt) + budget.total_tokens <= expected_identity.max_model_len,
             "Recorded prompt cannot reserve the full native protocol")
    phases = trace.get("phases")
    _require(isinstance(phases, list) and len(phases) == 2, "Exactly two recorded phases required")
    eos = expected_identity.eos_token_ids
    phase_targets, request_ids = [], set()
    for index, name in enumerate(("thought", "json")):
        phase = phases[index]
        _require(isinstance(phase, dict) and phase.get("name") == name, "Native phase order mismatch")
        ids = _ids(phase.get("token_ids"), vocabulary=vocabulary, name=name)
        expected_prompt = prompt if index == 0 else prompt + phase_targets[0]
        for key in ("prompt_ids", "response_prompt_ids"):
            _require(_ids(phase.get(key), vocabulary=vocabulary, name=name + " " + key) == expected_prompt,
                     name + " must continue exact real IDs without retemplating")
        _require(phase.get("response_finished") is True and phase.get("finish_reason") == "stop",
                 name + " lacks a terminal stop response")
        phase_limit = budget.thought_tokens if index == 0 else PHASE_TOKENS
        _require(type(phase.get("max_tokens")) is int and phase["max_tokens"] == phase_limit
                 and len(ids) <= phase_limit, name + " approved token budget mismatch")
        _require(type(phase.get("n")) is int and phase["n"] == 1
                 and type(phase.get("seed")) is int and phase["seed"] == expected_identity.generation_seed,
                 name + " actual generation count/seed mismatch")
        for key, value in {"ignore_eos": False, "skip_special_tokens": False,
                           "include_stop_str_in_output": True,
                           "structured_outputs": index == 1}.items():
            _require(phase.get(key) is value, name + " invalid " + key)
        _require(phase.get("output_kind") == "FINAL_ONLY", name + " must record FINAL_ONLY output")
        sampling = budget.thought_sampling if index == 0 else EvaluationSampling()
        _require(_json_key(phase.get("sampling")) == _json_key(sampling.to_dict()),
                 name + " must use approved phase-specific sampling")
        if not budget.legacy:
            _require(phase.get("effective_sampling") == effective_sampling(sampling),
                     name + " actual effective sampling differs from caller approval")
        stops = sorted(set(eos + (END_THINK,))) if index == 0 else list(eos)
        _require(_ids(phase.get("stop_token_ids"), vocabulary=vocabulary, name=name + " stops") == stops,
                 name + " configured stops differ")
        request_id = phase.get("request_id")
        _require(isinstance(request_id, str) and bool(request_id.strip()) and request_id not in request_ids,
                 "Distinct real phase request IDs required")
        request_ids.add(request_id)
        outputs = phase.get("raw_outputs")
        _require(isinstance(outputs, list) and len(outputs) == 1 and isinstance(outputs[0], dict),
                 name + " requires exactly one raw response")
        for key in ("token_ids", "engine_output_text", "finish_reason", "stop_reason"):
            _require(key in phase and key in outputs[0]
                     and _json_key(phase[key]) == _json_key(outputs[0][key]),
                     name + " raw/flattened response mismatch: " + key)
        _require(isinstance(phase["engine_output_text"], str), name + " raw engine text required")
        _require(phase.get("decoded_text_raw") == tokenizer.decode(ids, skip_special_tokens=False),
                 name + " recorded decode mismatch")
        phase_targets.append(ids)

    thought, final = phase_targets
    thought_stop = phases[0]["stop_reason"]
    _require(type(thought_stop) is int and thought_stop == END_THINK
             and thought[-1] == END_THINK and thought.count(END_THINK) == 1
             and not any(token in eos for token in thought),
             "Thought must stop on its single actually generated closing token")
    _require(bool(tokenizer.decode(thought[:-1], skip_special_tokens=False).strip()),
             "Native TRAIN proposal requires a nonempty generated thought")
    final_stop = phases[1]["stop_reason"]
    _require(final[-1] in eos and not any(token in eos for token in final[:-1])
             and ((final_stop is None and final[-1] == tokenizer.eos_token_id)
                  or (type(final_stop) is int and final_stop == final[-1])),
             "JSON requires its single actually generated ordinary EOS")
    public_json = tokenizer.decode(final[:-1], skip_special_tokens=False)
    _require(record.get("completion") == public_json and trace.get("final_text") == public_json,
             "Public projection must exactly decode actual JSON IDs minus only terminal EOS")
    Draft202012Validator(original_schema).validate(_strict_json(public_json))
    completion = thought + final
    _require(_ids(trace.get("completion_ids"), vocabulary=vocabulary, name="combined completion") == completion,
             "Combined target differs from the two actual generated phases")
    _require(type(trace.get("observed_completion_tokens")) is int
             and trace["observed_completion_tokens"] == len(completion),
             "Observed completion count mismatch")

    return PhysicalNativeCompletion(tuple(prompt), tuple(completion),
                                    (len(thought), len(final)), public_json)
