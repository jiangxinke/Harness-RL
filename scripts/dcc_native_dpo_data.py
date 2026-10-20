"""Prospective TRAIN-only final-decision preferences; no collection or targets.

The shared physical validator also serves existing native SFT. Accepted-SFT
answer/polarity gates remain in validate_native_record after physical integrity.
DPO instead requires trusted public-state ownership and separate offline rewards.
Caller approvals establish provenance; these assertions are not cryptographic
proof that an untrusted producer generated a trace or withheld its labels.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math

from jsonschema import Draft202012Validator
from slime_plugins.dcc.eval_sampling import EvaluationSampling
from slime_plugins.dcc.native_budget import effective_sampling
from slime_plugins.dcc.hf_policy import encode_messages
from dcc_native_physical import validate_native_physical
from dcc_native_sft import (
    CONTRACT, END_THINK, NATIVE_SUFFIX, PHASE_TOKENS, NativeIdentity,
    _ids, _json_key, _require, _strict_json, validate_source_receipt,
)

PREFERENCE_CONTRACT = "native-final-preference-candidate-v1"
PUBLIC_PREFIX_CONTRACT = "native-final-public-prefix-v1"


def validate_final_only_schema(schema):
    """Keep the actual approved final schema; never rewrite action anyOf."""
    _require(isinstance(schema, dict) and schema.get("type") == "object"
             and isinstance(schema.get("properties"), dict)
             and "answer" in schema["properties"]
             and isinstance(schema.get("required"), list) and "answer" in schema["required"]
             and schema.get("additionalProperties") is False
             and not {"tool", "arguments"}.intersection(schema["properties"])
             and not {"anyOf", "oneOf", "allOf", "$ref", "patternProperties", "if", "then", "else", "not"}.intersection(schema),
             "Trusted prefix requires an unchanged final-only root object schema")
    Draft202012Validator.check_schema(schema)
    return schema


@dataclass(frozen=True)
class PublicFinalPrefix:
    """Created from the trusted public runner state, never from candidate rows."""
    prefix_id: str
    task_id: str
    benchmark: str
    source: str
    messages_json: str
    schema_json: str
    harness_id: str
    native_identity_json: str
    split: str = "train"

    def __post_init__(self):
        _require(self.split == "train", "Preferences accept TRAIN only")
        _require((self.benchmark, self.source) in {
            ("cuad", "cuad_train"), ("widesearch", "independent_curriculum")},
            "Official WideSearch and unknown sources cannot enter preferences")
        _require(all(isinstance(v, str) and v.strip() for v in (
            self.prefix_id, self.task_id, self.harness_id)),
            "Frozen public prefix and harness identities required")
        messages, schema = _strict_json(self.messages_json), _strict_json(self.schema_json)
        _require(isinstance(messages, list) and bool(messages), "Public messages required")
        validate_final_only_schema(schema)
        identity = _strict_json(self.native_identity_json)
        _require(isinstance(identity, dict) and bool(identity) and "generation_seed" not in identity,
                 "Frozen public native identity must exclude candidate generation seed")

    @classmethod
    def from_dict(cls, value):
        _require(isinstance(value, dict) and set(value) == {"contract", "prefix_id", "task_id",
            "benchmark", "source", "split", "harness_id", "messages", "json_schema", "native_identity"}
            and value["contract"] == PUBLIC_PREFIX_CONTRACT, "Explicit public-prefix contract required")
        return cls(value["prefix_id"], value["task_id"], value["benchmark"], value["source"],
            _json_key(value["messages"]), json.dumps(value["json_schema"], ensure_ascii=False,
                allow_nan=False, sort_keys=False, separators=(",", ":")), value["harness_id"],
            _json_key(value["native_identity"]), value["split"])

    def to_dict(self):
        return {"contract": PUBLIC_PREFIX_CONTRACT, "prefix_id": self.prefix_id,
            "task_id": self.task_id, "benchmark": self.benchmark, "source": self.source,
            "split": self.split, "harness_id": self.harness_id,
            "messages": _strict_json(self.messages_json), "json_schema": _strict_json(self.schema_json),
            "native_identity": _strict_json(self.native_identity_json)}


@dataclass(frozen=True)
class TrainRewardReceipt:
    """Produced offline after all candidate generation/journals have closed."""
    task_id: str
    trace_id: str
    prefix_id: str
    harness_id: str
    reward_version: str
    value: float
    generation_closed_before_scoring: bool
    split: str = "train"

    def __post_init__(self):
        _require(self.split == "train" and self.generation_closed_before_scoring is True,
                 "TRAIN reward requires generation closure before label access")
        _require(type(self.value) in (int, float) and math.isfinite(self.value)
                 and 0 <= self.value <= 1, "Finite bounded offline TRAIN reward required")


@dataclass(frozen=True)
class NativeSequence:
    prompt: tuple[int, ...]
    completion: tuple[int, ...]
    phase_lengths: tuple[int, int]

    def __post_init__(self):
        _require(isinstance(self.prompt, tuple) and bool(self.prompt)
                 and isinstance(self.completion, tuple) and bool(self.completion)
                 and all(type(token) is int and token >= 0 for token in self.prompt + self.completion),
                 "Immutable recorded prompt/completion IDs required")
        _require(isinstance(self.phase_lengths, tuple) and len(self.phase_lengths) == 2
                 and all(type(length) is int and length > 0 for length in self.phase_lengths)
                 and sum(self.phase_lengths) == len(self.completion), "Exact native phase boundaries required")
        thought = self.completion[:self.phase_lengths[0]]
        _require(thought[-1] == END_THINK and thought.count(END_THINK) == 1,
                 "Physical native thought closing-token ownership required")


@dataclass(frozen=True)
class NativeFinalCandidate:
    prefix: PublicFinalPrefix
    sequence: NativeSequence
    trace_id: str
    episode_id: str
    generation_seed: int
    frozen_identity_json: str
    reward: float
    reward_version: str

    def __post_init__(self):
        _require(type(self.generation_seed) is int and self.generation_seed >= 0,
                 "Actual per-candidate generation seed required")
        _require(all(isinstance(value, str) and value.strip() for value in (
            self.trace_id, self.episode_id, self.frozen_identity_json, self.reward_version)), "Candidate ownership required")
        _require(type(self.reward) in (int, float) and math.isfinite(self.reward)
                 and 0 <= self.reward <= 1, "Finite bounded offline TRAIN reward required")

    @property
    def group_key(self):
        return self.prefix, self.sequence.prompt, self.frozen_identity_json, self.reward_version


@dataclass(frozen=True)
class PreferencePair:
    chosen: NativeFinalCandidate
    rejected: NativeFinalCandidate

    def __post_init__(self):
        _require(self.chosen.group_key == self.rejected.group_key,
                 "Preference pair must share exact public prefix/schema/harness/budgets/identity")
        _require(self.chosen.trace_id != self.rejected.trace_id
                 and self.chosen.episode_id != self.rejected.episode_id
                 and self.chosen.generation_seed != self.rejected.generation_seed,
                 "Distinct genuine sampled candidate identities and seeds required")
        _require(self.chosen.sequence.completion != self.rejected.sequence.completion,
                 "Identical completions cannot be a preference contrast")
        _require(self.chosen.reward > self.rejected.reward,
                 "Preference pair requires a strict offline TRAIN reward gap")

    @property
    def key(self):
        # Includes full exact IDs and ownership; no path or sampled-text hash.
        return self.chosen, self.rejected


def best_worst_pair(candidates, *, minimum_reward_gap=0.05):
    """One deterministic best/worst pair per public prefix; tied groups omitted."""
    _require(type(minimum_reward_gap) in (int, float) and math.isfinite(minimum_reward_gap)
             and minimum_reward_gap > 0, "Positive finite pair gap required")
    _require(len(candidates) >= 2, "At least two sampled candidates required")
    _require(len({candidate.group_key for candidate in candidates}) == 1,
             "Do not pair unrelated tasks, evidence states or schemas")
    for field in ("trace_id", "episode_id", "generation_seed"):
        _require(len({getattr(c, field) for c in candidates}) == len(candidates),
                 "Duplicate candidate " + field)
    ordered = sorted(candidates, key=lambda c: (c.reward, c.trace_id))
    if ordered[-1].reward - ordered[0].reward <= minimum_reward_gap:
        return None
    return PreferencePair(ordered[-1], ordered[0])


def validate_final_candidate(envelope, tokenizer, *, prefix, expected_identity, reward):
    """Physical validation uses trusted public state; reward is a separate receipt.

    Wrong complete answers remain eligible, including positive-task abstentions
    and negative-task false positives. No gold target, synthetic thought, EOS,
    close token, retemplating, or completion cropping is introduced here.
    """
    _require(isinstance(envelope, dict) and set(envelope) == {
        "contract", "prefix_id", "native_bundle"}
        and envelope["contract"] == PREFERENCE_CONTRACT
        and envelope["prefix_id"] == prefix.prefix_id, "Explicit preference contract required")
    bundle = envelope["native_bundle"]
    record = bundle["record"]
    _require(record.get("task_id") == prefix.task_id and record.get("stage") == "final"
             and record.get("split") == "train" and record.get("source") == prefix.source
             and _json_key(record.get("messages")) == _json_key(_strict_json(prefix.messages_json)),
             "Candidate differs from trusted public TRAIN final state")
    _require(isinstance(reward, TrainRewardReceipt)
             and reward.task_id == prefix.task_id and reward.prefix_id == prefix.prefix_id
             and reward.harness_id == prefix.harness_id and bool(reward.reward_version.strip())
             and reward.trace_id == bundle["collector"]["trace_id"],
             "Offline TRAIN reward receipt ownership mismatch")
    physical = validate_native_physical(bundle, tokenizer, expected_identity=expected_identity,
                                        original_schema=_strict_json(prefix.schema_json))
    sequence = NativeSequence(physical.prompt_ids, physical.completion_ids, physical.phase_lengths)
    identity = expected_identity.manifest(tokenizer)
    identity.pop("generation_seed")
    _require(_json_key(identity) == _json_key(_strict_json(prefix.native_identity_json)),
             "Candidate runtime/budget/identity differs from frozen public prefix")
    return NativeFinalCandidate(prefix, sequence, bundle["collector"]["trace_id"],
        bundle["collector"]["episode_id"], expected_identity.generation_seed,
        _json_key(identity), float(reward.value), reward.reward_version)
