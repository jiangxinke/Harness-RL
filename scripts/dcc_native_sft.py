"""Opt-in native Thinking SFT data validation and sampler integration.

The adopted dcc_tool_sft module must already be on PYTHONPATH. This module never
loads a tokenizer/model, changes source globals, starts collection, or uses CUDA
on import. Explicit CPU preflight can load only the local tokenizer.
Recorded token IDs are targets; JSON text is only a validated public projection.
"""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import dcc_tool_sft as legacy
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from slime_plugins.dcc.eval_sampling import EvaluationSampling
from slime_plugins.dcc.native_budget import NativeBudget, effective_sampling
from slime_plugins.dcc.hf_policy import PROMPT_ENCODING_VERSION, encode_messages

BASE_SOURCE_COMMIT = "5bed6188feaaf6c79fac53eead30cfbad83b0beb"
MODEL_REPOSITORY = "Qwen/Qwen3-30B-A3B-Thinking-2507"
MODEL_REVISION = "144afc2f379b542fdd4e85a1fcd5e1f79112d95d"
PROTOCOL = "native-thinking2507-real-close-two-stage-4096-4096-v2"
CONTRACT = "native-thinking-train-record-v2"
OPEN_THINK, END_THINK = 151667, 151668
PHASE_TOKENS, TOTAL_TOKENS = 4096, 8192
NATIVE_SUFFIX = "<|im_start|>assistant\n<think>\n"


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _json_key(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def _strict_json(text):
    def nonfinite(value):
        raise ValueError("Non-finite JSON constant: " + value)

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON object key")
            result[key] = value
        return result

    return json.loads(text, parse_constant=nonfinite, object_pairs_hook=unique_object)


def _ids(value, *, vocabulary, name):
    _require(isinstance(value, list) and bool(value), name + " must be nonempty recorded IDs")
    _require(all(type(token) is int and 0 <= token < vocabulary for token in value),
             name + " contains invalid token IDs")
    return list(value)


@dataclass(frozen=True)
class NativeIdentity:
    """Caller-approved artifact identity, separate from untrusted row contents.

    derivative_id identifies the ACTUAL pruned/base artifact, not a server path.
    This is a provenance assertion, not a proof of its weights or trace origin.
    """

    derivative_id: str
    eos_token_ids: tuple[int, ...]
    source_commit: str
    max_prompt_tokens: int = 16384
    max_model_len: int = 24576
    generation_seed: int = 42
    thought_tokens: int = 4096
    thought_sampling_profile: str = "greedy"
    prompt_history: dict | None = None

    @property
    def budget(self):
        return NativeBudget(self.thought_tokens, self.thought_sampling_profile)

    def manifest(self, tokenizer):
        _require(isinstance(self.source_commit, str)
                 and re.fullmatch(r"[0-9a-f]{40}", self.source_commit) is not None
                 and self.source_commit != BASE_SOURCE_COMMIT,
                 "Caller-approved actual native integration source is required")
        _require(isinstance(self.derivative_id, str) and bool(self.derivative_id.strip()),
                 "Approved derivative artifact identity is required")
        _require(isinstance(self.eos_token_ids, tuple) and bool(self.eos_token_ids)
                 and all(type(token) is int and 0 <= token < len(tokenizer)
                         for token in self.eos_token_ids)
                 and len(set(self.eos_token_ids)) == len(self.eos_token_ids)
                 and END_THINK not in self.eos_token_ids and OPEN_THINK not in self.eos_token_ids,
                 "Configured ordinary EOS IDs are required")
        _require(type(tokenizer.eos_token_id) is int and tokenizer.eos_token_id in self.eos_token_ids,
                 "Tokenizer primary EOS must be one configured integer EOS")
        _require(type(self.max_prompt_tokens) is int and self.max_prompt_tokens > 0
                 and type(self.max_model_len) is int
                 and self.max_prompt_tokens + self.budget.total_tokens <= self.max_model_len,
                 "Reserve the full approved native completion budget")
        _require(type(self.generation_seed) is int and self.generation_seed >= 0,
                 "Actual collection generation seed is required")
        _require(tokenizer.encode("<think>", add_special_tokens=False) == [OPEN_THINK]
                 and tokenizer.encode("</think>", add_special_tokens=False) == [END_THINK],
                 "Native opening/closing token metadata mismatch")
        _require(isinstance(tokenizer.chat_template, str) and bool(tokenizer.chat_template),
                 "A single exact native chat template is required")
        result = {
            "source_commit": self.source_commit,
            "model_repository": MODEL_REPOSITORY,
            "model_revision": MODEL_REVISION,
            "derivative_id": self.derivative_id,
            "tokenizer_revision": MODEL_REVISION,
            "tokenizer_vocabulary_size": len(tokenizer),
            "chat_template": tokenizer.chat_template,
            "prompt_encoding_version": PROMPT_ENCODING_VERSION,
            "protocol": self.budget.protocol,
            "eos_token_ids": list(self.eos_token_ids),
            "primary_eos_token_id": tokenizer.eos_token_id,
            "max_prompt_tokens": self.max_prompt_tokens,
            "max_model_len": self.max_model_len,
            "generation_seed": self.generation_seed,
        }
        if not self.budget.legacy:
            result.update(thought_tokens=self.thought_tokens,
                          thought_sampling_profile=self.thought_sampling_profile)
        if self.prompt_history is not None:
            from dcc_historical_native_config import HISTORICAL_TRAIN_SOURCE
            _require(self.source_commit != HISTORICAL_TRAIN_SOURCE,
                     "Historical TRAIN source cannot approve a new context policy")
            from slime_plugins.dcc.prompt_history import PromptHistoryConfig
            result["prompt_history"] = PromptHistoryConfig.from_dict(self.prompt_history).to_dict()
        return result


def identity_from_manifest(value, expected_source):
    """Reconstruct only caller-approved values; manifest equality rejects extras."""
    return NativeIdentity(value["derivative_id"], tuple(value["eos_token_ids"]), expected_source,
        value["max_prompt_tokens"], value["max_model_len"], value["generation_seed"],
        value.get("thought_tokens", 4096), value.get("thought_sampling_profile", "greedy"),
        copy.deepcopy(value.get("prompt_history")))


def validate_native_record(bundle, tokenizer, *, expected_identity: NativeIdentity,
                           original_schema: dict, web_schema_profile=None):
    """Validate physical native IDs, then preserve all accepted-SFT gates."""
    # Local import lets the shared helper reuse the stable NativeIdentity API
    # without changing module-import order or introducing a circular import.
    from dcc_native_physical import validate_native_physical
    physical = validate_native_physical(bundle, tokenizer,
        expected_identity=expected_identity, original_schema=original_schema)
    record = bundle["record"]
    prompt, completion = list(physical.prompt_ids), list(physical.completion_ids)
    public_json = physical.public_json
    # Original validators remain unchanged, including tool replay and grounding.
    if record.get("source") == "cuad_train":
        legacy.validate_cuad_record(record)
        domain, stage = "cuad_tool", "cuad_tool_" + record["stage"]
        category = record["metadata"]["category"]
    else:
        if web_schema_profile is not None and web_schema_profile.explicit_review:
            legacy.validate_record(record, _native_web_profile=web_schema_profile)
        else:
            legacy.validate_record(record)
        domain, stage, category = "web", record["stage"], "independent_web"
    if record["stage"] == "final":
        visible = " ".join(tokenizer.decode(prompt, skip_special_tokens=False).split()).replace('""', '"')
        fields = (record["grounding_fields"] if domain == "cuad_tool" else
                  [field for row in _strict_json(public_json)["answer"]
                   for name, field in row.items() if name != "citations"])
        _require(all(" ".join(field.split()) in visible or
                     " ".join(json.dumps(field, ensure_ascii=False)[1:-1].split()) in visible
                     for field in fields), "Final grounding is absent from actual encoded prompt")
    return legacy.Example(record["task_id"], stage, category, prompt, completion,
                          native_phase_lengths=physical.phase_lengths)


def encode_training_record(record, tokenizer, *, max_prompt, max_response,
                           batch_tokens, skips: Counter, native_thinking=False,
                           expected_identity=None, original_schema=None):
    """Default-off path delegates verbatim; native path never creates target IDs."""
    _require(type(native_thinking) is bool, "native_thinking must be a boolean")
    if not native_thinking:
        return legacy.encode_record(record, tokenizer, max_prompt=max_prompt,
                                    max_response=max_response, batch_tokens=batch_tokens, skips=skips)
    _require(isinstance(expected_identity, NativeIdentity), "Explicit native identity is required")
    example = validate_native_record(record, tokenizer, expected_identity=expected_identity,
                                     original_schema=original_schema)
    return apply_training_budgets(example, max_prompt=max_prompt, max_response=max_response,
                                  batch_tokens=batch_tokens, skips=skips)


def apply_training_budgets(example, *, max_prompt, max_response, batch_tokens, skips):
    for count, limit, suffix in ((len(example.prompt), max_prompt, "prompt_too_long"),
                                  (len(example.completion), max_response, "target_too_long"),
                                  (example.tokens, batch_tokens, "batch_budget")):
        _require(type(limit) is int and limit > 0, "Training token budgets must be positive integers")
        if count > limit:
            prefix = "cuad_tool_" if example.stage.startswith("cuad_tool_") else "web_"
            stage = example.stage.removeprefix("cuad_tool_")
            skips[prefix + stage + "_" + suffix] += 1
            return None
    return example


def _iter_validated_native(path, tokenizer, *, expected_identity: NativeIdentity,
                           schema_resolver: Callable[[dict, dict], dict]):
    """Strict TRAIN-only JSONL iteration with the legacy duplicate identities.

    A trusted resolver receives the pure-JSON original record, never its trace.
    Do not implement a resolver by returning native_trace['json_schema'].
    """
    _require(callable(schema_resolver), "Trusted original-schema resolver required")
    seen, any_rows = set(), False
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                bundle = _strict_json(line)
                _require(isinstance(bundle, dict) and isinstance(bundle.get("record"), dict),
                         "Native bundle and original record required")
                record = bundle["record"]
                example = validate_native_record(bundle, tokenizer, expected_identity=expected_identity,
                                                 original_schema=schema_resolver(record, bundle.get("collector")),
                                                 web_schema_profile=(getattr(schema_resolver.__self__, "web", None)
                                                     if isinstance(getattr(schema_resolver, "__self__", None), NativeSchemaProfile) else None))
                if record.get("source") == "cuad_train":
                    key = ("cuad", record["task_id"], record["metadata"].get("plan_id"),
                           record["stage"], record["completion"])
                else:
                    key = ("web", record["task_id"], record["stage"], record["completion"])
                _require(key not in seen, "Duplicate native teacher segment")
                seen.add(key)
            except (ValueError, TypeError, KeyError, ValidationError, SchemaError) as error:
                raise ValueError(f"Invalid native TRAIN record at line {number}: {error}") from error
            any_rows = True
            yield bundle, example
    _require(any_rows, "Native TRAIN records must be nonempty")


def iter_native_records(path, tokenizer, *, expected_identity: NativeIdentity,
                        schema_resolver: Callable[[dict, dict], dict]) -> Iterator[dict]:
    for bundle, _ in _iter_validated_native(path, tokenizer, expected_identity=expected_identity,
                                           schema_resolver=schema_resolver):
        yield bundle


def check_native_training_options(*, native_thinking=False, warm_start=None,
                                  raw_cuad_auxiliary=False):
    """Integration guard; this proposal has no native warm-start provenance path."""
    _require(type(native_thinking) is bool, "native_thinking must be a boolean")
    if native_thinking:
        _require(not warm_start, "Native proposal rejects preexisting adapters")
        _require(raw_cuad_auxiliary is False, "Native proposal rejects legacy gold-only CUAD auxiliary")


def validate_source_receipt(value, expected_source):
    """Match a trusted actual source boundary, never the row's own assertion."""
    _require(isinstance(expected_source, str)
             and re.fullmatch(r"[0-9a-f]{40}", expected_source) is not None
             and expected_source != BASE_SOURCE_COMMIT,
             "Explicit caller-approved actual integration source required")
    expected = {"version": "dcc-native-inference-source-v1", "source_commit": expected_source,
                "base_source_commit": BASE_SOURCE_COMMIT, "protocol": PROTOCOL, "frozen": True}
    _require(isinstance(value, dict) and _json_key(value) == _json_key(expected),
             "Actual frozen source receipt differs from caller-approved source")
    return copy.deepcopy(expected)


@dataclass(frozen=True)
class WebSchemaProfile:
    """Trusted collection configuration; no value comes from target answers."""

    max_steps: int
    explicit_review: bool = False
    cached_rows: bool = False
    navigation_ledger: bool = False
    profile_version: str = "dcc-native-train-profile-v1"
    data_discovery_guidance: bool = False
    multi_citation: bool = False

    def validate(self):
        _require(type(self.max_steps) is int and self.max_steps > 0,
                 "Trusted collection max_steps required")
        _require(type(self.explicit_review) is bool and type(self.cached_rows) is bool,
                 "Trusted Web schema flags must be booleans")
        from dcc_review_training import LEGACY_PROFILE, REVIEW20_PROFILE, WEB32K_PROFILE
        _require(type(self.navigation_ledger) is bool, "Trusted navigation flag must be boolean")
        _require(type(self.data_discovery_guidance) is bool,
                 "Trusted discovery guidance flag must be boolean")
        _require(type(self.multi_citation) is bool,
                 "Trusted multi-citation flag must be boolean")
        _require(not self.data_discovery_guidance or
                 self.profile_version in {REVIEW20_PROFILE, WEB32K_PROFILE},
                 "Discovery guidance requires an explicit reviewed Web TRAIN profile")
        _require(not self.multi_citation or
                 self.profile_version in {REVIEW20_PROFILE, WEB32K_PROFILE},
                 "Multi-citation requires an explicit reviewed Web TRAIN profile")
        if self.profile_version in {REVIEW20_PROFILE, WEB32K_PROFILE}:
            _require(self.max_steps == 20 and self.explicit_review and self.navigation_ledger
                     and not self.cached_rows, "Explicit TRAIN20 requires review/navigation on and cached rows off")
        else:
            _require(self.profile_version == LEGACY_PROFILE and not self.explicit_review
                     and not self.navigation_ledger, "Reviewed final controls require the separate explicit TRAIN20 profile")

    def original_schema(self, record, collector):
        """Rebuild the actual early/last-turn Web schema from public history."""
        from slime_plugins.dcc.episode import Task, _WebEvidence, final_turn_message
        from slime_plugins.dcc.web_structured import web_action_schema, web_final_schema

        self.validate()
        if self.explicit_review:
            from dcc_review_training import review_request_state
            schema, _ = review_request_state(self, record, collector)
            legacy.validate_record(record, _native_web_profile=self)
            return schema
        _require(record.get("source") == "independent_curriculum",
                 "Trusted Web profile requires independent_curriculum records")
        legacy.validate_record(record)  # Preserve every existing replay/grounding condition.
        _require(isinstance(collector, dict) and type(collector.get("turn")) is int
                 and 0 <= collector["turn"] < self.max_steps,
                 "Actual turn is outside trusted collection profile")
        messages = record["messages"]
        turn = sum(message["role"] == "assistant" for message in messages)
        _require(turn == collector["turn"], "Turn differs from actual historical action count")
        marker = "Required answer columns: "
        _require(marker in messages[0]["content"], "Original public answer columns are missing")
        columns, _ = json.JSONDecoder().raw_decode(messages[0]["content"].split(marker, 1)[1])
        _require(isinstance(columns, list) and columns
                 and all(isinstance(column, str) and column for column in columns),
                 "Invalid original public answer columns")
        evidence, pending = _WebEvidence(), None
        for message in messages[2:]:
            if message["role"] == "assistant":
                pending = _strict_json(message["content"])
            elif pending is not None:
                text = message["content"][len(legacy._OBSERVATION):].rsplit("\nRemaining turns:", 1)[0]
                evidence.observe(pending["tool"], _strict_json(text))
                pending = None
        task = Task(record["task_id"], "widesearch", "train", messages[1]["content"],
                    metadata={"evaluation": {"required": columns}})
        final_control = final_turn_message(task, evidence)
        is_last = turn == self.max_steps - 1
        _require((messages[-1] == final_control) == is_last,
                 "Last-turn control differs from trusted collection max_steps")
        if is_last:
            return web_final_schema(columns, evidence.fetched_sources)
        return web_action_schema(columns, evidence.fetched_sources,
                                 successful_search=bool(evidence.searches), cached_rows=self.cached_rows)


@dataclass(frozen=True)
class CuadSchemaProfile:
    """Separately approved public CUAD route; never derived from target spans."""

    max_steps: int
    structured_actions: bool
    schema_version: str

    def validate(self):
        from slime_plugins.dcc.cuad_structured import CUAD_STRUCTURED_SCHEMA_VERSION

        _require(type(self.max_steps) is int and self.max_steps > 0,
                 "Trusted CUAD collection max_steps required")
        _require(self.structured_actions is True
                 and self.schema_version == CUAD_STRUCTURED_SCHEMA_VERSION,
                 "Trusted CUAD canonical structured route/version is required")

    def original_schema(self, record, collector):
        """Use the same zero-label early/last schemas as actual collection."""
        from slime_plugins.dcc.cuad_structured import cuad_action_schema, cuad_final_schema
        from slime_plugins.dcc.episode import CUAD_FINAL_REVIEW, FINAL_TURN

        self.validate()
        _require(record.get("source") == "cuad_train",
                 "Trusted CUAD profile requires cuad_train records")
        legacy.validate_cuad_record(record)
        _require(isinstance(collector, dict) and type(collector.get("turn")) is int
                 and 0 <= collector["turn"] < self.max_steps,
                 "Actual CUAD turn is outside trusted collection profile")
        messages = record["messages"]
        turn = sum(message["role"] == "assistant" for message in messages)
        _require(turn == collector["turn"], "CUAD turn differs from historical action count")
        final_control = {"role": "user", "content": FINAL_TURN + CUAD_FINAL_REVIEW}
        is_last = turn == self.max_steps - 1
        # The original validator recognizes a historical bare FINAL_TURN too.
        # New native collection uses exactly the full public CUAD final review.
        has_control = legacy._is_final_control(record, messages, messages[-1])
        _require(has_control == is_last and (not is_last or messages[-1] == final_control),
                 "CUAD last-turn control differs from trusted collection max_steps")
        return cuad_final_schema() if is_last else cuad_action_schema()


@dataclass(frozen=True)
class NativeSchemaProfile:
    web: WebSchemaProfile
    cuad: CuadSchemaProfile | None = None

    def original_schema(self, record, collector):
        if record.get("source") == "cuad_train":
            _require(self.cuad is not None, "Trusted CUAD collection profile is required")
            return self.cuad.original_schema(record, collector)
        return self.web.original_schema(record, collector)


def _segment_key(record):
    if record.get("source") == "cuad_train":
        return ("cuad", record["task_id"], record["metadata"].get("plan_id"),
                record["stage"], record["completion"])
    return ("web", record["task_id"], record["stage"], record["completion"])


@dataclass
class NativeTrainingData:
    """Fully validated files, original flat indexes, and exact token targets."""

    records: list[dict]
    cuad_records: list[dict]
    identity: NativeIdentity
    manifest: dict
    examples: dict
    record_json: dict
    configuration: dict | None = None
    cuad_identity: NativeIdentity | None = None
    cuad_manifest: dict | None = None
    strict_training_budgets: bool = False

    def encode_record(self, record, tokenizer, *, max_prompt, max_response, batch_tokens, skips):
        _require(_json_key(self.identity.manifest(tokenizer)) == _json_key(self.manifest),
                 "Loaded training tokenizer differs from fully validated data tokenizer")
        if self.cuad_identity is not None:
            _require(_json_key(self.cuad_identity.manifest(tokenizer)) == _json_key(self.cuad_manifest),
                     "Loaded training tokenizer differs from approved CUAD data tokenizer")
        key = _segment_key(record)
        _require(key in self.examples and _json_key(record) == self.record_json[key],
                 "Flat training record differs from fully validated native data")
        example = self.examples[key]
        # Return fresh lists: the sampler never gains mutable access to cached targets.
        example = legacy.Example(example.task_id, example.stage, example.category,
                                 list(example.prompt), list(example.completion),
                                 native_phase_lengths=example.native_phase_lengths)
        if self.strict_training_budgets:
            for count, limit, name in ((len(example.prompt), max_prompt, "prompt"),
                    (len(example.completion), max_response, "completion"),
                    (example.tokens, batch_tokens, "cumulative batch")):
                _require(type(limit) is int and limit > 0 and count <= limit,
                         "Approved native sample exceeds training " + name + " budget; refusing to skip")
        return apply_training_budgets(example, max_prompt=max_prompt, max_response=max_response,
                                      batch_tokens=batch_tokens, skips=skips)


def load_native_training_data(web_path, tokenizer, *, expected_identity, source_receipt,
                              schema_resolver, cuad_path=None, expected_cuad_identity=None,
                              approved_cuad_identity_differences=None, strict_training_budgets=False):
    """Validate ALL input files before returning any sampler-visible record.

    The schema resolver is independent of row targets/traces. The production
    CLI uses the same pure Web/CUAD schemas as the actual inference route.
    """
    validate_source_receipt(source_receipt, expected_identity.source_commit)
    _require(type(strict_training_budgets) is bool, "Strict training budget flag must be boolean")
    if expected_cuad_identity is not None:
        _require(cuad_path is not None, "Separate CUAD identity requires an explicit CUAD input file")
        web_manifest = expected_identity.manifest(tokenizer)
        cuad_manifest = expected_cuad_identity.manifest(tokenizer)
        differences = {"source_commit", "generation_seed", "max_prompt_tokens", "max_model_len", "prompt_history"}
        actual_differences = sorted(k for k in set(web_manifest) | set(cuad_manifest)
                                    if web_manifest.get(k) != cuad_manifest.get(k))
        _require(isinstance(approved_cuad_identity_differences, list)
                 and approved_cuad_identity_differences == actual_differences
                 and set(actual_differences) <= differences,
                 "Web/CUAD identity differences must be exactly approved; same model/tokenizer/native protocol required")
    else:
        _require(approved_cuad_identity_differences is None,
                 "CUAD identity differences require a separately approved identity")
    files = [(web_path, "independent_curriculum", expected_identity)]
    if cuad_path is not None:
        files.append((cuad_path, "cuad_train", expected_cuad_identity or expected_identity))
    validated = []
    for path, domain, approved_identity in files:
        rows = list(_iter_validated_native(path, tokenizer, expected_identity=approved_identity,
                                           schema_resolver=schema_resolver))
        _require(all(bundle["record"]["source"] == domain for bundle, _ in rows),
                 "Native input file contains the wrong domain")
        validated.extend(rows)
    records, cuad_records, examples, record_json, trace_ids, requests = [], [], {}, {}, set(), set()
    for bundle, example in validated:
        record, trace = copy.deepcopy(bundle["record"]), bundle["native_trace"]
        key = _segment_key(record)
        _require(key not in examples and trace["trace_id"] not in trace_ids,
                 "Native segment or real trace reused across input files")
        trace_ids.add(trace["trace_id"])
        for phase in trace["phases"]:
            _require(phase["request_id"] not in requests, "Real native generation request reused")
            requests.add(phase["request_id"])
        examples[key], record_json[key] = example, _json_key(record)
        (cuad_records if record["source"] == "cuad_train" else records).append(record)
    return NativeTrainingData(records, cuad_records, expected_identity,
                              expected_identity.manifest(tokenizer), examples, record_json,
                              cuad_identity=expected_cuad_identity,
                              cuad_manifest=(expected_cuad_identity.manifest(tokenizer)
                                  if expected_cuad_identity is not None else None),
                              strict_training_budgets=strict_training_budgets)


def prepare_native_training(args, *, source_root, tokenizer=None):
    """CPU preflight for the proposed opt-in source patch, before model/CUDA.

    The trusted profile is separate from the rollout files. The source receipt
    is read only from the actual frozen source root, and must match an explicit
    caller-approved expected commit. No Git command or model load is performed.
    """
    check_native_training_options(native_thinking=True, warm_start=args.adapter,
                                  raw_cuad_auxiliary=bool(args.cuad_train_data))
    expected_source = getattr(args, "native_expected_source", None)
    receipt_path = Path(source_root) / ".dcc-native-source.json"
    receipt = validate_source_receipt(_strict_json(receipt_path.read_text()), expected_source)
    profile_path = getattr(args, "native_train_profile", None)
    _require(profile_path is not None, "--native-train-profile is required")
    profile = _strict_json(Path(profile_path).read_text())
    from dcc_review_training import approved_web_profile, bind_profile_source, WEB32K_PROFILE
    web_profile = approved_web_profile(profile, WebSchemaProfile)
    bind_profile_source(profile, source_root)
    cuad = profile.get("cuad")
    _require(not args.cuad_tool_records or cuad is not None,
             "Native CUAD records require a separate trusted CUAD collection profile")
    cuad_profile = None
    if cuad is not None:
        _require(isinstance(cuad, dict)
                 and set(cuad) == {"max_steps", "structured_actions", "schema_version"},
                 "Trusted CUAD collection profile is invalid")
        cuad_profile = CuadSchemaProfile(**cuad)
        cuad_profile.validate()
    approved = profile["identity"]
    _require(isinstance(approved, dict) and approved.get("source_commit") == expected_source,
             "Trusted profile differs from explicit expected source")
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    identity = identity_from_manifest(approved, expected_source)
    _require(_json_key(approved) == _json_key(identity.manifest(tokenizer)),
             "Trusted profile and actual training tokenizer differ")
    cuad_identity = None
    if "cuad_identity" in profile:
        approved_cuad = profile["cuad_identity"]
        _require(isinstance(approved_cuad, dict), "Trusted CUAD identity must be an object")
        cuad_identity = identity_from_manifest(approved_cuad, approved_cuad.get("source_commit"))
        _require(_json_key(approved_cuad) == _json_key(cuad_identity.manifest(tokenizer)),
                 "Trusted CUAD profile and actual training tokenizer differ")
    web = profile["web"]
    schema_profile = NativeSchemaProfile(web_profile, cuad_profile)
    data = load_native_training_data(args.records, tokenizer, expected_identity=identity,
                                     source_receipt=receipt, schema_resolver=schema_profile.original_schema,
                                     cuad_path=args.cuad_tool_records, expected_cuad_identity=cuad_identity,
                                     approved_cuad_identity_differences=profile.get("cuad_identity_differences"),
                                     strict_training_budgets=profile["version"] == WEB32K_PROFILE)
    data.configuration = {"enabled": True, "protocol": identity.budget.protocol, "source_receipt": receipt,
                          "derivative_id": identity.derivative_id, "model_revision": MODEL_REVISION,
                          "web": copy.deepcopy(web), "validated_records": len(data.records),
                          "cuad": copy.deepcopy(cuad), "validated_cuad_records": len(data.cuad_records),
                          "all_files_validated_before_model_load": True,
                          "warm_start": False, "raw_cuad_auxiliary": False}
    if profile["version"] != "dcc-native-train-profile-v1":
        data.configuration["training_profile_version"] = profile["version"]
        data.configuration["collection_overlay_id"] = profile["collection_overlay_id"]
    if not identity.budget.legacy:
        from dcc_historical_native_config import HISTORICAL_TRAIN_SOURCE, historical_native_configuration
        if identity.source_commit == HISTORICAL_TRAIN_SOURCE:
            data.configuration["rollout_native_thinking"] = historical_native_configuration(receipt, budget=identity.budget)
        else:
            from slime_plugins.dcc.native_thinking_eval import native_configuration
            data.configuration["rollout_native_thinking"] = native_configuration(receipt, budget=identity.budget)
    if profile["version"] == WEB32K_PROFILE:
        data.configuration["domain_identities"] = {"independent_curriculum": copy.deepcopy(approved),
            "cuad_train": copy.deepcopy(profile.get("cuad_identity", approved))}
        data.configuration["prompt_history"] = copy.deepcopy(identity.prompt_history)
        if cuad_identity is not None:
            data.configuration["cuad_identity_differences"] = copy.deepcopy(profile["cuad_identity_differences"])
        data.configuration["training_budget_overflow"] = "fail; never skip approved samples"
    return data
