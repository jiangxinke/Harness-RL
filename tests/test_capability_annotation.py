from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from slime_plugins.stackplanner.capability.all_agent_output import (
    ALL_AGENT_OUTPUT_CAPABILITIES,
    annotate_turn as annotate_all_agent_output_turn,
)
from slime_plugins.stackplanner.capability.annotator import annotate_turn
from slime_plugins.stackplanner.capability.sample_transform import (
    fan_out_capability_samples,
)
from slime_plugins.stackplanner.capability.taxonomy import CAPABILITIES


class CharacterTokenizer:
    def decode(self, token_ids, **_kwargs):
        return "".join(chr(token_id) for token_id in token_ids)

    def __call__(self, text, **_kwargs):
        return {
            "input_ids": [ord(char) for char in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def _annotate(text: str) -> list[str]:
    result = annotate_turn(
        tokenizer=CharacterTokenizer(),
        output_ids=[ord(char) for char in text],
        raw_output=text,
        parsed=SimpleNamespace(),
        tools_schema=[],
    )
    return result["capability_labels"]


def test_exact_annotation_maps_every_direct_answer_token_to_action_output():
    text = "用户可见答案 ✓"
    labels = _annotate(text)
    assert len(labels) == len(text)
    assert set(labels) == {"action_output"}


def test_delegate_json_fields_unicode_and_multiple_calls_use_only_two_routes():
    calls = [
        {
            "name": "sp_delegate",
            "arguments": {
                "target_agent": "researcher",
                "stage": "research",
                "task": '查找 café "证据"',
                "expected_output": "摘要",
                "metadata": {"skill_names": ["web-search"]},
            },
        },
        {
            "name": "sp_backtrack",
            "arguments": {"target_id": "entry-1", "reason": "invalid"},
        },
    ]
    text = "".join(
        f"<tool_call>\n{json.dumps(call, ensure_ascii=False)}\n</tool_call>"
        for call in calls
    )
    labels = _annotate(text)
    assert set(labels) == set(CAPABILITIES)
    for call in calls:
        name = call["name"]
        name_start = text.index(name)
        assert set(labels[name_start : name_start + len(name)]) == {"action_choice"}
    task = calls[0]["arguments"]["task"]
    encoded_task = json.dumps(task, ensure_ascii=False)
    task_start = text.index(encoded_task)
    assert set(labels[task_start : task_start + len(encoded_task)]) == {"action_output"}
    assert all(label in CAPABILITIES for label in labels)


class NonLosslessTokenizer(CharacterTokenizer):
    def decode(self, token_ids, **_kwargs):
        return "?" * len(token_ids)


def test_annotation_fails_closed_when_per_id_decode_is_not_lossless():
    with pytest.raises(ValueError, match="refuses to guess"):
        annotate_turn(
            tokenizer=NonLosslessTokenizer(),
            output_ids=[1],
            raw_output="x",
            parsed=SimpleNamespace(),
            tools_schema=[],
        )


def test_annotation_accepts_exact_noncanonical_byte_bpe_sequence():
    class NonCanonicalTokenizer(CharacterTokenizer):
        def decode(self, token_ids, **_kwargs):
            return "sp_delegate"

        def __call__(self, text, **_kwargs):
            assert text == "sp_delegate"
            return {"input_ids": [99], "offset_mapping": [(0, len(text))]}

        def convert_ids_to_tokens(self, token_ids):
            assert token_ids == [1, 2, 3]
            return ["sp", "_de", "legate"]

    labels = annotate_turn(
        tokenizer=NonCanonicalTokenizer(),
        output_ids=[1, 2, 3],
        raw_output="sp_delegate",
        parsed=SimpleNamespace(),
        tools_schema=[],
    )["capability_labels"]
    assert labels == ["action_output"] * 3


def test_token_crossing_action_name_boundary_deterministically_uses_choice():
    text = '<tool_call>{"name":"sp_finish","arguments":{}}</tool_call>'

    class BoundaryTokenizer(CharacterTokenizer):
        def decode(self, token_ids, **_kwargs):
            assert token_ids == [1, 2, 3]
            return text

        def __call__(self, raw_text, **_kwargs):
            assert raw_text == text
            name_start = text.index("sp_finish")
            return {
                "input_ids": [1, 2, 3],
                "offset_mapping": [
                    (0, name_start + 2),
                    (name_start + 2, name_start + len("sp_finish")),
                    (name_start + len("sp_finish"), len(text)),
                ],
            }

    labels = annotate_turn(
        tokenizer=BoundaryTokenizer(),
        output_ids=[1, 2, 3],
        raw_output=text,
        parsed=SimpleNamespace(),
        tools_schema=[],
    )["capability_labels"]
    assert labels == ["action_choice", "action_choice", "action_output"]


def test_unknown_tool_name_is_concrete_action_output():
    text = '<tool_call>{"name":"not_a_stackplanner_action","arguments":{}}</tool_call>'
    assert set(_annotate(text)) == {"action_output"}


def test_annotation_fails_closed_when_noncanonical_vocab_bytes_do_not_match():
    class InvalidTokenizer(CharacterTokenizer):
        def decode(self, token_ids, **_kwargs):
            return "ab"

        def __call__(self, text, **_kwargs):
            return {"input_ids": [99], "offset_mapping": [(0, 2)]}

        def convert_ids_to_tokens(self, token_ids):
            return ["a", "c"]

    with pytest.raises(ValueError, match="do not exactly reproduce"):
        annotate_turn(
            tokenizer=InvalidTokenizer(),
            output_ids=[1, 2],
            raw_output="ab",
            parsed=SimpleNamespace(),
            tools_schema=[],
        )


def test_real_qwen_offsets_cover_unicode_escaping_and_field_boundaries():
    transformers = pytest.importorskip("transformers")
    model_value = os.environ.get("SP_TEST_TOKENIZER_PATH")
    if not model_value:
        pytest.skip("SP_TEST_TOKENIZER_PATH is not configured")
    model = Path(model_value)
    if not model.is_dir():
        pytest.skip("configured Qwen tokenizer is unavailable")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model, trust_remote_code=True, use_fast=True
    )
    payloads = [
        {
            "name": "sp_delegate",
            "arguments": {
                "target_agent": "researcher",
                "stage": "检索",
                "task": '查找 café "证据" 🙂',
                "expected_output": "Unicode 摘要",
                "metadata": {
                    "skill_names": ["web-search"],
                    "tool_names": ["search"],
                },
            },
        },
        {
            "name": "sp_finish",
            "arguments": {
                "summary": "答案： 完成 ✓",
                "allow_without_artifact": True,
                "final_artifact_ref": None,
                "required_artifact_type": None,
            },
        },
        {
            "name": "sp_replan",
            "arguments": {
                "task": "重新检索",
                "reason": "证据冲突",
                "stage": "recovery",
            },
        },
    ]
    text = "preface 🙂\n" + "".join(
        f"<tool_call>\n{json.dumps(call, ensure_ascii=False)}\n</tool_call>"
        for call in payloads
    )
    output_ids = tokenizer.encode(text, add_special_tokens=False)
    labels = annotate_turn(
        tokenizer=tokenizer,
        output_ids=output_ids,
        raw_output=text,
        parsed=SimpleNamespace(),
        tools_schema=[],
    )["capability_labels"]
    assert len(labels) == len(output_ids)
    assert set(labels) == set(CAPABILITIES)

    # The same decoded text can be represented by a deliberately split byte-BPE
    # sequence.  It is not the exact canonical mapping used for offsets.
    split_ids = [
        token_id
        for character in "sp_delegate"
        for token_id in tokenizer.encode(character, add_special_tokens=False)
    ]
    canonical_ids = tokenizer.encode("sp_delegate", add_special_tokens=False)
    assert split_ids != canonical_ids
    assert (
        tokenizer.decode(
            split_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        == "sp_delegate"
    )
    split_labels = annotate_turn(
        tokenizer=tokenizer,
        output_ids=split_ids,
        raw_output="sp_delegate",
        parsed=SimpleNamespace(),
        tools_schema=[],
    )["capability_labels"]
    assert len(split_labels) == len(split_ids)


@dataclass
class FakeSample:
    response_length: int
    loss_mask: list[int]
    train_metadata: dict
    rollout_id: int = 7
    reward: float = 1.0
    metadata: dict = field(default_factory=dict)


def test_capability_fanout_is_exact_partition_of_actor_mask():
    labels = ["action_choice", None, "action_output", "action_choice", "action_output"]
    original = FakeSample(5, [1, 0, 1, 1, 1], {"capability_labels": labels})
    children = fan_out_capability_samples(SimpleNamespace(), [original])
    assert len(children) == len(CAPABILITIES)
    assert {child.rollout_id for child in children} == {7}
    assert {child.train_metadata["microbatch_group"] for child in children} == set(
        CAPABILITIES
    )
    reconstructed = [
        sum(child.loss_mask[index] for child in children) for index in range(5)
    ]
    assert reconstructed == original.loss_mask
    assert all(
        not enabled or original.loss_mask[index]
        for child in children
        for index, enabled in enumerate(child.loss_mask)
    )
    assert sum(sum(child.loss_mask) for child in children) == sum(original.loss_mask)
    assert all(
        sum(child.loss_mask[index] for child in children) <= 1
        for index in range(original.response_length)
    )


def test_two_capability_fanout_preserves_episode_denominator_across_siblings():
    samples = [
        FakeSample(
            4,
            [1, 1, 0, 1],
            {
                "capability_labels": [
                    "action_choice",
                    "action_output",
                    None,
                    "action_output",
                ]
            },
            rollout_id=9,
        ),
        FakeSample(
            3,
            [1, 0, 1],
            {"capability_labels": ["action_output", None, "action_choice"]},
            rollout_id=9,
        ),
    ]
    original_episode_tokens = sum(sum(sample.loss_mask) for sample in samples)
    children = fan_out_capability_samples(SimpleNamespace(), samples)
    assert {child.rollout_id for child in children} == {9}
    assert len(children) == len(samples) * len(CAPABILITIES)
    assert sum(sum(child.loss_mask) for child in children) == original_episode_tokens


def test_capability_fanout_rejects_missing_label_on_trainable_token():
    original = FakeSample(2, [0, 1], {"capability_labels": [None, None]})
    with pytest.raises(ValueError, match="unknown capability labels"):
        fan_out_capability_samples(SimpleNamespace(), [original])


def test_all_agent_action_output_annotation_is_exact_disjoint_and_role_tagged():
    call = {
        "name": "sp_delegate",
        "arguments": {
            "target_agent": "researcher",
            "task": "find evidence",
        },
    }
    text = f"<tool_call>{json.dumps(call)}</tool_call>"
    result = annotate_all_agent_output_turn(
        tokenizer=CharacterTokenizer(),
        output_ids=[ord(char) for char in text],
        raw_output=text,
        parsed=SimpleNamespace(),
        tools_schema=[],
        annotation_context={"role": "central"},
    )
    assert len(result["capability_labels"]) == len(text)
    assert set(result["capability_labels"]) == set(ALL_AGENT_OUTPUT_CAPABILITIES)
    assert set(result["actor_roles"]) == {"central"}
    name_start = text.index("sp_delegate")
    assert set(
        result["capability_labels"][name_start : name_start + len("sp_delegate")]
    ) == {"action_choice"}
    argument_start = text.index("{", text.index('"arguments"'))
    argument_end = text.rindex("}")
    assert set(result["capability_labels"][argument_start:argument_end]) == {
        "action_output"
    }


def test_all_agent_action_output_annotation_requires_explicit_role():
    with pytest.raises(ValueError, match="explicit session role"):
        annotate_all_agent_output_turn(
            tokenizer=CharacterTokenizer(),
            output_ids=[ord("x")],
            raw_output="x",
            parsed=SimpleNamespace(),
            tools_schema=[],
        )


def test_all_agent_action_output_recognizes_specialist_tool_choice():
    call = {"name": "web_search", "arguments": {"query": "evidence"}}
    text = f"<tool_call>{json.dumps(call)}</tool_call>"
    result = annotate_all_agent_output_turn(
        tokenizer=CharacterTokenizer(),
        output_ids=[ord(char) for char in text],
        raw_output=text,
        parsed=SimpleNamespace(),
        tools_schema=[],
        annotation_context={"role": "researcher"},
    )
    name_start = text.index("web_search")
    assert set(
        result["capability_labels"][name_start : name_start + len("web_search")]
    ) == {"action_choice"}
    assert set(result["actor_roles"]) == {"researcher"}


def test_all_agent_action_output_fanout_uses_b_taxonomy(monkeypatch):
    monkeypatch.setenv("SP_DECOUPLING_SCOPE", "all-agent-action-output")
    labels = list(ALL_AGENT_OUTPUT_CAPABILITIES)
    original = FakeSample(2, [1, 1], {"capability_labels": labels})
    children = fan_out_capability_samples(SimpleNamespace(), [original])
    assert len(children) == 2
    assert {child.train_metadata["capability"] for child in children} == set(
        ALL_AGENT_OUTPUT_CAPABILITIES
    )
