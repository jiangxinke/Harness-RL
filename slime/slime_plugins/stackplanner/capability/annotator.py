"""Exact-token capability annotation for StackPlanner Central outputs."""

from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from collections.abc import Mapping
from typing import Any

from .taxonomy import (
    ACTION_CHOICE_TOOLS,
    CAPABILITIES,
)


def _byte_decoder() -> dict[str, int]:
    """Return the reversible GPT-2/Qwen byte-level BPE alphabet."""

    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    unicode_values = list(byte_values)
    offset = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            unicode_values.append(256 + offset)
            offset += 1
    return {chr(codepoint): value for value, codepoint in zip(byte_values, unicode_values, strict=True)}


_QWEN_BYTE_DECODER = _byte_decoder()


def _served_byte_bpe_offsets(tokenizer: Any, output_ids: list[int], raw_output: str) -> list[tuple[int, int]]:
    """Align a non-canonical served Qwen byte-BPE sequence exactly.

    SGLang sampling can legally emit two adjacent vocabulary entries whose
    concatenated bytes equal one longer vocabulary entry. Re-encoding the text
    then produces different ids even though the served sequence is exact. The
    vocabulary symbols remain a reversible byte representation, so use those
    bytes instead of guessing offsets from decoded token fragments. Tokens that
    split one UTF-8 codepoint intentionally overlap that character.
    """

    try:
        pieces = tokenizer.convert_ids_to_tokens(output_ids)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("non-canonical served ids require a reversible byte-BPE tokenizer") from exc
    if not isinstance(pieces, list) or len(pieces) != len(output_ids):
        raise ValueError("tokenizer did not expose one vocabulary symbol per served id")

    encoded_pieces: list[bytes] = []
    for piece in pieces:
        if not isinstance(piece, str) or not piece:
            raise ValueError(f"invalid vocabulary symbol for exact alignment: {piece!r}")
        try:
            encoded_pieces.append(bytes(_QWEN_BYTE_DECODER[char] for char in piece))
        except KeyError as exc:
            raise ValueError("non-canonical served ids are not reversible Qwen byte-BPE symbols") from exc
    raw_bytes = raw_output.encode("utf-8")
    if b"".join(encoded_pieces) != raw_bytes:
        raise ValueError("served vocabulary bytes do not exactly reproduce the decoded output")

    boundaries = [0]
    for char in raw_output:
        boundaries.append(boundaries[-1] + len(char.encode("utf-8")))
    offsets: list[tuple[int, int]] = []
    byte_start = 0
    for piece in encoded_pieces:
        byte_end = byte_start + len(piece)
        char_start = max(0, bisect_right(boundaries, byte_start) - 1)
        char_end = bisect_left(boundaries, byte_end)
        if not (0 <= char_start < char_end <= len(raw_output)):
            raise ValueError(f"served token bytes have no exact character overlap: {(byte_start, byte_end)!r}")
        offsets.append((char_start, char_end))
        byte_start = byte_end
    return offsets


def _exact_token_offsets(tokenizer: Any, output_ids: list[int], raw_output: str) -> list[tuple[int, int]]:
    """Map served ids to characters only through a verified fast-tokenizer encoding.

    Decoding individual byte-level BPE tokens is not compositional for every
    Unicode sequence.  Offsets therefore come from one whole-string encoding,
    and that encoding is accepted only when it exactly reproduces the ids that
    SGLang served.  A non-canonical generated token sequence fails closed rather
    than receiving an approximate semantic mask.
    """

    decoded = tokenizer.decode(
        output_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded != raw_output:
        raise ValueError("served token ids do not decode exactly to the adapter output; capability annotation refuses to guess spans")
    try:
        encoded = tokenizer(
            raw_output,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        reencoded_ids = list(encoded["input_ids"])
        raw_offsets = list(encoded["offset_mapping"])
    except (KeyError, TypeError, ValueError, NotImplementedError) as exc:
        raise ValueError("capability annotation requires a fast tokenizer with exact return_offsets_mapping support") from exc
    if reencoded_ids != output_ids:
        return _served_byte_bpe_offsets(tokenizer, output_ids, raw_output)
    if len(raw_offsets) != len(output_ids):
        raise ValueError(f"offset count {len(raw_offsets)} != served token count {len(output_ids)}")

    offsets: list[tuple[int, int]] = []
    covered = [False] * len(raw_output)
    previous_start = 0
    for value in raw_offsets:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"invalid tokenizer offset: {value!r}")
        start, end = int(value[0]), int(value[1])
        if not (0 <= start < end <= len(raw_output)) or start < previous_start:
            raise ValueError(f"invalid or non-monotonic tokenizer offset {(start, end)!r}")
        offsets.append((start, end))
        for index in range(start, end):
            covered[index] = True
        previous_start = start
    if raw_output and not all(covered):
        missing = [index for index, value in enumerate(covered) if not value]
        raise ValueError(f"tokenizer offsets do not cover the exact decoded output; first uncovered character index={missing[0]}")
    return offsets


def _tool_payload_spans(raw_output: str) -> list[tuple[int, int, Mapping[str, Any]]]:
    spans: list[tuple[int, int, Mapping[str, Any]]] = []
    decoder = json.JSONDecoder()
    cursor = 0
    while True:
        start_tag = raw_output.find("<tool_call>", cursor)
        if start_tag < 0:
            break
        end_tag = raw_output.find("</tool_call>", start_tag + 11)
        if end_tag < 0:
            break
        body_start = start_tag + len("<tool_call>")
        while body_start < end_tag and raw_output[body_start].isspace():
            body_start += 1
        try:
            payload, length = decoder.raw_decode(raw_output[body_start:end_tag])
        except json.JSONDecodeError:
            cursor = end_tag + len("</tool_call>")
            continue
        if isinstance(payload, Mapping):
            spans.append((body_start, body_start + length, payload))
        cursor = end_tag + len("</tool_call>")
    return spans


def _tool_block_ranges(raw_output: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = raw_output.find("<tool_call>", cursor)
        if start < 0:
            break
        closing = raw_output.find("</tool_call>", start + len("<tool_call>"))
        if closing < 0:
            break
        end = closing + len("</tool_call>")
        ranges.append((start, end))
        cursor = end
    return ranges


def _mark_chars(labels: list[set[str]], start: int, end: int, capability: str) -> None:
    for index in range(max(start, 0), min(end, len(labels))):
        labels[index].add(capability)


def _locate_json_value(text: str, key: str, start: int, end: int) -> tuple[int, int] | None:
    encoded_key = json.dumps(key, ensure_ascii=False)
    key_at = text.find(encoded_key, start, end)
    if key_at < 0:
        return None
    colon = text.find(":", key_at + len(encoded_key), end)
    if colon < 0:
        return None
    value_start = colon + 1
    while value_start < end and text[value_start].isspace():
        value_start += 1
    decoder = json.JSONDecoder()
    try:
        _, length = decoder.raw_decode(text[value_start:end])
    except json.JSONDecodeError:
        return None
    return value_start, value_start + length


def annotate_turn(
    *,
    tokenizer: Any,
    output_ids: list[int],
    raw_output: str,
    parsed: Any,
    tools_schema: list[dict] | None,
    annotation_context: Mapping[str, Any] | None = None,
) -> dict[str, list[Any]]:
    """Return one exclusive capability label for every exact output token.

    Labels are computed while the adapter still owns SGLang's original token
    ids.  Character spans are promoted to token spans by overlap, so JSON
    escaping, Unicode, and tokens crossing field boundaries remain covered.
    """

    del parsed, tools_schema, annotation_context
    offsets = _exact_token_offsets(tokenizer, output_ids, raw_output)
    # C deliberately has only two routes. Every served character defaults to
    # concrete output; only the exact JSON value of a recognized StackPlanner
    # action name is promoted to action choice. This makes JSON syntax, action
    # arguments, execution/recovery contracts, plain text, and final answers
    # one coherent action_output capability with no shared/fallback bucket.
    action_choice_chars = [False for _ in raw_output]
    for payload_start, payload_end, payload in _tool_payload_spans(raw_output):
        name = str(payload.get("name") or "")
        name_span = _locate_json_value(raw_output, "name", payload_start, payload_end)
        if name_span and name in ACTION_CHOICE_TOOLS:
            # StackPlanner action names are fixed ASCII identifiers. Keep the
            # surrounding JSON quotes in action_output; only the identifier
            # itself is semantic action choice.
            choice_start, choice_end = name_span
            if choice_end - choice_start >= 2 and raw_output[choice_start] == '"' and raw_output[choice_end - 1] == '"':
                choice_start += 1
                choice_end -= 1
            for index in range(choice_start, choice_end):
                action_choice_chars[index] = True

    labels: list[str] = []
    for start, end in offsets:
        # Served tokens are indivisible. A token that crosses an action-name
        # boundary deterministically follows the choice route if it overlaps
        # any character of that exact name value; every other token is output.
        labels.append("action_choice" if any(action_choice_chars[start:end]) else "action_output")
    if len(labels) != len(output_ids) or any(label not in CAPABILITIES for label in labels):
        raise AssertionError("invalid capability annotation")
    return {"capability_labels": labels}
