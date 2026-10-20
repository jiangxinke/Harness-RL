"""Opt-in, token-measured eviction of whole interaction turns.

No source, assistant JSON, task, or token array is sliced. The caller retains
the complete request; the plan records the exact original message indices and
observed structured URLs of every omitted turn. URL metadata is audit data,
never an evidence summary injected into the model prompt.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from urllib.parse import urlsplit


PROMPT_HISTORY_VERSION = "whole-interaction-turns-v1"
_OBSERVATION_PREFIXES = (
    "Initial contract retrieval (source material):\n",
    "Tool observation (source material):\n",
)


@dataclass(frozen=True)
class PromptHistoryConfig:
    enabled: bool = False
    keep_recent_turns: int = 1

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("prompt history enabled must be a boolean")
        if type(self.keep_recent_turns) is not int or self.keep_recent_turns < 1:
            raise ValueError("keep_recent_turns must be a positive integer")

    def to_dict(self):
        return {"enabled": self.enabled,
                "version": PROMPT_HISTORY_VERSION if self.enabled else None,
                "keep_recent_turns": self.keep_recent_turns if self.enabled else None,
                "source_shortening": False,
                "url_metadata_scope": "structured observed metadata; not answer evidence"}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or type(value.get("enabled")) is not bool:
            raise ValueError("A complete prompt history declaration is required")
        configuration = cls(value["enabled"], value.get("keep_recent_turns") if value["enabled"] else 1)
        if not prompt_history_json_equal(value, configuration.to_dict()):
            raise ValueError("Prompt history declaration differs from the supported configuration")
        return configuration


def prompt_history_json_equal(left, right):
    """Strict JSON equality: bool/int, array order and exact text matter."""
    try:
        return (json.dumps(left, sort_keys=True, ensure_ascii=False, allow_nan=False)
                == json.dumps(right, sort_keys=True, ensure_ascii=False, allow_nan=False))
    except (ValueError, TypeError):
        return False


def add_prompt_history_arguments(parser):
    parser.add_argument("--prompt-history-whole-turns", action="store_true",
                        help="Opt in to token-measured whole-turn history eviction; no JSON/text slicing")
    parser.add_argument("--prompt-history-keep-recent-turns", type=int, default=1,
                        help="Minimum latest complete interaction turns to retain (default: 1)")


def prompt_history_from_arguments(args):
    configuration = PromptHistoryConfig(
        getattr(args, "prompt_history_whole_turns", False),
        getattr(args, "prompt_history_keep_recent_turns", 1))
    if not configuration.enabled and configuration.keep_recent_turns != 1:
        raise ValueError("--prompt-history-keep-recent-turns requires --prompt-history-whole-turns")
    return configuration


def validate_prompt_history_records(records):
    declarations = [PromptHistoryConfig.from_dict(value["prompt_history"])
                    if "prompt_history" in value else PromptHistoryConfig()
                    for value in records]
    if any(value != declarations[0] for value in declarations[1:]):
        raise ValueError("Prompt history differs between manifest and runtime")
    return declarations[0]


class PromptHistoryBudgetExceeded(ValueError):
    """No request was submitted; even protected complete context cannot fit."""

    def __init__(self, reason, *, diagnostics, original_prompt_ids):
        super().__init__(
            f"{reason}: full={diagnostics['original_prompt_tokens']}, "
            f"minimum={diagnostics['minimum_prompt_tokens']}, "
            f"limit={diagnostics['effective_prompt_limit']}; no generation submitted")
        self.reason = reason
        self.diagnostics = copy.deepcopy(diagnostics)
        self.original_prompt_ids = list(original_prompt_ids)


@dataclass
class PromptHistoryPlan:
    messages: list[dict]
    prompt_ids: list[int]
    report: dict


def _encode(tokenizer, messages):
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    return list(encoded["input_ids"] if hasattr(encoded, "keys") else encoded)


def _source_observation(message):
    content = message["content"]
    if message["role"] == "tool":
        body = content
    else:
        prefix = next((p for p in _OBSERVATION_PREFIXES if content.startswith(p)), None)
        if prefix is None:
            return None
        body = content[len(prefix):]
    try:
        value, end = json.JSONDecoder().raw_decode(body)
    except (ValueError, RecursionError):
        return None
    # Only runner controls may follow the complete JSON value. Never repair a
    # cut JSON string or interpret arbitrary prose as a successful tool read.
    tail = body[end:]
    if tail and not tail.startswith(("\nRemaining turns:", "\nFinal turn:")):
        return None
    return value if isinstance(value, dict) else None


def _observed_urls(message, index):
    observation = _source_observation(message)
    if observation is None:
        return []
    entries = []

    def append(url, kind):
        if not isinstance(url, str):
            return
        try:
            parsed = urlsplit(url)
            valid = parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username and not parsed.password
        except ValueError:
            valid = False
        if valid:
            entry = {"message_index": index, "url": url, "kind": kind}
            if entry not in entries:
                entries.append(entry)

    def page(value):
        if not isinstance(value, dict):
            return
        if value.get("error"):
            append(value.get("url"), "failed-read; not evidence")
            return
        text = value.get("text")
        append(value.get("url"), "read-source" if isinstance(text, str) and text.strip()
               else "observed-url; no read claim")
        for row in value.get("results", []) if isinstance(value.get("results"), list) else []:
            if isinstance(row, dict):
                append(row.get("url"), "search-result; not read evidence")
        for row in value.get("outgoing_links", []) if isinstance(value.get("outgoing_links"), list) else []:
            if isinstance(row, dict):
                append(row.get("url"), "outgoing-link; not read evidence")

    page(observation)
    sources = observation.get("auto_read_sources")
    for source in sources if isinstance(sources, list) else []:
        page(source)
    return entries


def _notice(count):
    return {"role": "user", "content": (
        f"History budget notice: {count} earlier complete interaction turn(s) "
        "were omitted from this request to fit the context limit. "
        "This notice supplies no source evidence. Use the retained source "
        "observations; reread an omitted source if its text is needed. "
        "The original task and current controls remain in force.")}


def _read_source(message):
    value = _source_observation(message)
    if value is None or value.get("error"):
        return False
    if isinstance(value.get("text"), str) and value["text"].strip():
        return True
    # CUAD ranked hits and Web auto-reads are literal source content too.
    return any(isinstance(row, dict) and not row.get("error")
               and isinstance(row.get("text"), str) and row["text"].strip()
               for field in ("hits", "auto_read_sources")
               for row in (value.get(field) if isinstance(value.get(field), list) else []))


def fit_prompt_history(tokenizer, messages, *, max_prompt_tokens, max_new_tokens,
                       max_model_len, config):
    """Return the largest recent whole-turn suffix that actually fits.

    Protect the initial system/task, all messages before the first assistant,
    and at least ``keep_recent_turns`` latest assistant+reply groups. Current
    controls appended after the latest reply are part of that protected group.
    The full caller request is never mutated. A single oversized latest source
    fails explicitly; this function does not shorten source text or JSON.
    """
    if not isinstance(config, PromptHistoryConfig) or not config.enabled:
        raise ValueError("An explicitly enabled PromptHistoryConfig is required")
    for name, value in (("max_prompt_tokens", max_prompt_tokens),
                        ("max_new_tokens", max_new_tokens), ("max_model_len", max_model_len)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    limit = min(max_prompt_tokens, max_model_len - max_new_tokens)
    if limit < 1:
        raise ValueError("No context remains after reserving the complete generation budget")
    if (not isinstance(messages, list) or len(messages) < 2
            or any(not isinstance(m, dict) or not isinstance(m.get("content"), str) for m in messages)
            or [m.get("role") for m in messages[:2]] != ["system", "user"]):
        raise ValueError("Whole-turn history requires an initial system message and original user task")

    original = copy.deepcopy(messages)
    full_ids = _encode(tokenizer, original)
    report = {"configuration": config.to_dict(), "max_prompt_tokens": max_prompt_tokens,
              "max_new_tokens_reserved": max_new_tokens, "max_model_len": max_model_len,
              "effective_prompt_limit": limit, "original_prompt_tokens": len(full_ids),
              "actual_prompt_tokens": len(full_ids), "minimum_prompt_tokens": None,
              "compacted": False, "omitted_turns": [], "omitted_message_indices": [],
              "omitted_observed_urls": [], "retained_message_indices": list(range(len(original))),
              "notice": None}
    if len(full_ids) <= limit:
        return PromptHistoryPlan(original, full_ids, report)

    starts = [i for i in range(2, len(original)) if original[i].get("role") == "assistant"]
    prefix_end = starts[0] if starts else len(original)
    if any(m.get("role") not in {"user", "tool"} for m in original[2:prefix_end]):
        raise PromptHistoryBudgetExceeded("unsupported_history_shape", diagnostics=report,
                                          original_prompt_ids=full_ids)
    groups = [list(range(start, starts[j + 1] if j + 1 < len(starts) else len(original)))
              for j, start in enumerate(starts)]
    # Fail closed on unsupported structure instead of deleting an instruction,
    # an unpaired assistant action, or an unknown role to manufacture a fit.
    if any(len(group) < 2 or any(original[i].get("role") not in {"user", "tool"}
                               for i in group[1:]) for group in groups):
        raise PromptHistoryBudgetExceeded("unsupported_history_shape", diagnostics=report,
                                          original_prompt_ids=full_ids)

    removable = max(0, len(groups) - config.keep_recent_turns)
    # Later feedback or a failed tool call must not evict the latest actual
    # source body. Preserve a suffix starting no later than that source pair.
    source_groups = [j for j, group in enumerate(groups)
                     if any(_read_source(original[i]) for i in group[1:])]
    observation_groups = [j for j, group in enumerate(groups)
                          if any(original[i]["role"] == "tool" or _source_observation(original[i]) is not None
                                 for i in group[1:])]
    if source_groups:
        removable = min(removable, source_groups[-1])
    if observation_groups:
        removable = min(removable, observation_groups[-1])
    minimum_count = len(full_ids)
    for omitted_count in range(1, removable + 1):
        omitted = groups[:omitted_count]
        retained_indices = list(range(prefix_end)) + [i for group in groups[omitted_count:] for i in group]
        notice = _notice(omitted_count)
        candidate = (original[:prefix_end] + [notice]
                     + [original[i] for i in retained_indices[prefix_end:]])
        ids = _encode(tokenizer, candidate)
        minimum_count = min(minimum_count, len(ids))
        if len(ids) <= limit:
            omitted_indices = [i for group in omitted for i in group]
            report.update(compacted=True, actual_prompt_tokens=len(ids),
                          minimum_prompt_tokens=None,
                          omitted_turns=[{"turn_index": j, "message_indices": group}
                                         for j, group in enumerate(omitted)],
                          omitted_message_indices=omitted_indices,
                          retained_message_indices=retained_indices, notice=notice,
                          omitted_observed_urls=[entry for i in omitted_indices
                                                 for entry in _observed_urls(original[i], i)])
            return PromptHistoryPlan(copy.deepcopy(candidate), ids, report)

    report.update(actual_prompt_tokens=None, minimum_prompt_tokens=minimum_count,
                  protected_message_indices=(list(range(prefix_end))
                      + [i for group in groups[removable:] for i in group]),
                  removable_turns=removable)
    raise PromptHistoryBudgetExceeded("minimum_complete_context_exceeds_budget",
                                      diagnostics=report, original_prompt_ids=full_ids)
