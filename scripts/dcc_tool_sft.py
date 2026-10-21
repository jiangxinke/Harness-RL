#!/usr/bin/env python3
"""Train attention LoRA on independent web teachers and grounded CUAD spans.

Only completion tokens receive loss. Teacher observations stay in the prompt.
Use torchrun for disjoint per-rank samples and averaged adapter gradients.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "slime"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.distributed as dist
from dcc_train import supervised_example
from dcc_training_io import TrainingJournal, finalize_training, prepare_training_output, publish_checkpoint
from dcc_native_loss import (
    LOSS_TELEMETRY_CONFIGURATION,
    MIXTURE_LOSS_CONFIGURATION,
    backward_native_mixture_batch,
    merge_native_loss_summaries,
    native_sample_metadata,
)
from slime_plugins.dcc import EpisodeRunner, Task, load_tasks
from slime_plugins.dcc.episode import CUAD_FINAL_REVIEW, EPISODE_PROMPT_VERSION, FINAL_TURN, _WebEvidence, final_turn_message
from slime_plugins.dcc.hf_policy import (
    PROMPT_ENCODING_VERSION,
    completion_logprobs,
    encode_messages,
    load_policy_model,
)
from slime_plugins.dcc.retrieval import load_retrieval_profile
from slime_plugins.dcc.tools import WEB_READER_VERSION

STAGES = ("search", "fetch", "final")
CUAD_STAGES = ("search", "read", "final")
_OBSERVATION = "Tool observation (source material):\n"


@dataclass(frozen=True)
class Example:
    task_id: str
    stage: str
    category: str
    prompt: list[int]
    completion: list[int]
    # Set only after native validation of both actual recorded phase ID lists.
    native_phase_lengths: tuple[int, int] | None = None

    @property
    def tokens(self):
        return len(self.prompt) + len(self.completion)


def _is_final_control(value, messages, message, evidence=None):
    """Recognize the exact runner suffix without consuming a tool observation."""
    if value.get("stage") != "final":
        return False
    if evidence is None:
        return message in (
            {"role": "user", "content": FINAL_TURN},
            {"role": "user", "content": FINAL_TURN + CUAD_FINAL_REVIEW},
        )
    metadata = {}
    if evidence is not None:
        marker = "Required answer columns: "
        system = messages[0]["content"]
        if marker not in system:
            return False
        try:
            columns, _ = json.JSONDecoder().raw_decode(system.split(marker, 1)[1])
        except json.JSONDecodeError:
            return False
        if not isinstance(columns, list) or any(not isinstance(column, str) for column in columns):
            return False
        metadata = {"evaluation": {"required": columns}}
    task = Task(value["task_id"], "widesearch", "train", messages[1]["content"], metadata=metadata)
    if message == final_turn_message(task, evidence):
        return True
    # Keep exact compatibility with previously prepared full Web controls.
    # Columns and read URLs are reconstructed from the validated history;
    # arbitrary suffixes and tool observations are never controls.
    legacy = (
        FINAL_TURN
        + " Recheck the original task below before writing rows. Exclude entities outside any "
        "requested filter, remove duplicate entities, and finish the JSON when the verified "
        "set is exhausted. Keep supported partial rows rather than discarding them. "
        "Original task: " + task.prompt
        + "\nExact answer column keys: " + json.dumps(columns, ensure_ascii=False)
        + ". Add citations to each row. Successfully read source URLs: "
        + json.dumps(sorted(evidence.fetched_sources)[:8])
        + ". Use each row's actual supporting read URL in its citations list."
    )
    return message == {"role": "user", "content": legacy}


def validate_record(value: dict, *, _native_web_profile=None, _native_web_collector=None,
                    _native_cached_sources=None) -> dict:
    """Reject split leakage and replay real tool observations for finals."""
    if not isinstance(value, dict) or value.get("source") != "independent_curriculum" or value.get("split") != "train":
        raise ValueError("Web records must be independent_curriculum train rows")
    if not isinstance(value.get("task_id"), str) or not value["task_id"].strip() or value.get("stage") not in STAGES:
        raise ValueError("Web record requires a task_id and supported stage")
    messages = value.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("Web teacher requires historical messages")
    if any(not isinstance(m, dict) or set(m) != {"role", "content"} or m["role"] not in {"system", "user", "assistant"}
           or not isinstance(m["content"], str) or not m["content"].strip() for m in messages):
        raise ValueError("Invalid historical message")
    if messages[0]["role"] != "system" or messages[1]["role"] != "user" or messages[-1]["role"] != "user":
        raise ValueError("Teacher prompt must end before the target assistant response")
    control_indices = set()
    if _native_web_profile is not None:
        from dcc_review_training import review_request_state
        _, control_indices = review_request_state(_native_web_profile, value, _native_web_collector,
                                                  cached_sources=_native_cached_sources)
    evidence, fetched = _WebEvidence(), []
    pending = None
    for index, message in enumerate(messages[2:], 2):
        if index in control_indices:
            if pending is not None:
                raise ValueError("Native control cannot replace a pending tool observation")
            continue
        if (index == len(messages) - 1 and pending is None
                and _is_final_control(value, messages, message, evidence)):
            continue
        if message["role"] == "assistant":
            try:
                pending = json.loads(message["content"])
            except json.JSONDecodeError as exc:
                raise ValueError("Teacher history must contain exact tool JSON") from exc
            if not isinstance(pending, dict) or "answer" in pending or pending.get("tool") not in {"web_search", "fetch_url"}:
                raise ValueError("Teacher history contains a final target or unsupported tool")
            if index + 1 >= len(messages) or messages[index + 1]["role"] != "user":
                raise ValueError("Every historical tool needs its observation")
        elif message["role"] == "user" and pending is not None:
            if not message["content"].startswith(_OBSERVATION):
                raise ValueError("Missing actual tool observation")
            payload = message["content"][len(_OBSERVATION):].rsplit("\nRemaining turns:", 1)[0]
            try:
                observation = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError("Malformed historical tool observation") from exc
            evidence.observe(pending["tool"], observation)
            if pending["tool"] == "fetch_url" and isinstance(observation, dict) and not observation.get("error"):
                text = observation.get("text")
                if isinstance(text, str) and text.strip():
                    fetched.append(text)
                if _native_web_profile is not None and _native_web_profile.cached_rows:
                    fetched.extend(row["text"] for row in observation.get("row_view", {}).get("records", []))
            pending = None
        else:
            raise ValueError("Teacher history must alternate tools and observations")
    try:
        action = json.loads(value["completion"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Teacher completion must be exact action JSON") from exc
    if not isinstance(action, dict):
        raise ValueError("Teacher completion must be an action object")
    stage = value["stage"]
    if stage == "search":
        arguments = action.get("arguments", {})
        if action.get("tool") != "web_search" or "answer" in action or not isinstance(arguments, dict) or not isinstance(arguments.get("query"), str) or not arguments["query"].strip():
            raise ValueError("Search target must call web_search with a query")
    elif stage == "fetch":
        arguments = action.get("arguments", {})
        if action.get("tool") != "fetch_url" or "answer" in action or not isinstance(arguments, dict) or not _WebEvidence._source_url(arguments.get("url")) or not evidence.searches:
            raise ValueError("Fetch target requires a successful historical search")
    else:
        rows = action.get("answer")
        if "tool" in action or not isinstance(rows, list) or not rows or not evidence.searches or not evidence.fetches:
            raise ValueError("Final target requires successful search and fetch sources")
        fields = []
        for row in rows:
            if not isinstance(row, dict) or not row:
                raise ValueError("Final teacher rows must be nonempty objects")
            citations = row.get("citations")
            if not isinstance(citations, list) or not citations or any(not isinstance(url, str) or url not in evidence.sources for url in citations):
                raise ValueError("Final citation lacks an observed successful source")
            values = [field for name, field in row.items() if name != "citations"]
            if not values or any(not isinstance(field, str) or not field.strip() for field in values):
                raise ValueError("Final teacher cells must be nonempty strings")
            fields.extend(values)
        declared = value.get("grounding_fields", fields)
        if not isinstance(declared, list) or Counter(declared) != Counter(fields):
            raise ValueError("Grounding fields do not match final target cells")
        sources = [" ".join(text.split()).replace('""', '"') for text in fetched]
        if any(not any(" ".join(field.split()) in source for source in sources) for field in fields):
            raise ValueError("Final target is unsupported by fetched source text")
    return value


def load_records(path):
    records, seen = [], set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = validate_record(json.loads(line))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid teacher record at line {line_number}: {exc}") from exc
            key = (record["task_id"], record["stage"], record["completion"])
            if key in seen:
                raise ValueError("Duplicate teacher segment")
            seen.add(key)
            records.append(record)
    if not records:
        raise ValueError("Teacher records must be nonempty")
    return records


def validate_cuad_record(value, *, _native_cuad_sources=None):
    if not isinstance(value, dict) or value.get("source") != "cuad_train" or value.get("benchmark") != "cuad" or value.get("split") != "train":
        raise ValueError("CUAD tool records require cuad_train benchmark=cuad split=train")
    if not isinstance(value.get("task_id"), str) or not value["task_id"].strip() or value.get("stage") not in CUAD_STAGES:
        raise ValueError("CUAD tool record requires task_id and search/read/final stage")
    metadata, fields, messages = value.get("metadata", {}), value.get("grounding_fields"), value.get("messages")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("category"), str) or not metadata["category"].strip() or type(metadata.get("positive")) is not bool:
        raise ValueError("CUAD tool metadata requires category and positive boolean")
    if not isinstance(fields, list) or any(not isinstance(field, str) or not field.strip() for field in fields) or bool(fields) != metadata["positive"]:
        raise ValueError("CUAD grounding_fields must match positive/negative provenance")
    if not isinstance(messages, list) or len(messages) < 2 or any(not isinstance(message, dict) or set(message) != {"role", "content"}
            or message["role"] not in {"system", "user", "assistant"} or not isinstance(message["content"], str) for message in messages):
        raise ValueError("Invalid CUAD teacher messages")
    if messages[0]["role"] != "system" or messages[1]["role"] != "user" or messages[-1]["role"] != "user":
        raise ValueError("CUAD teacher prompt must end before its target")
    source_texts, ranges, successes = [], [], 0

    def observe(observation):
        if not isinstance(observation, dict) or observation.get("error"):
            return False
        chunks = observation.get("hits", [observation])
        valid = False
        for chunk in chunks if isinstance(chunks, list) else []:
            if not isinstance(chunk, dict):
                continue
            start, end, text = chunk.get("start"), chunk.get("end"), chunk.get("text")
            if type(start) is int and type(end) is int and 0 <= start < end and isinstance(text, str) and text.strip() and end - start == len(text):
                if "hits" in observation:
                    ranges.append((start, end))
                source_texts.append(text)
                valid = True
        return valid

    def check_read(arguments):
        from slime_plugins.dcc.retrieval import document_read_windows

        if not isinstance(arguments, dict):
            raise ValueError("CUAD read arguments must be an object")
        if any(type(value) is not int for value in arguments.values()) or not any(
            arguments in document_read_windows({"start": left, "end": right}) for left, right in ranges
        ):
            raise ValueError("CUAD read must use a hit boundary or a public reader window exposed by previous source hits")

    pending = None
    for index, message in enumerate(messages[2:], 2):
        content = message["content"]
        if (index == len(messages) - 1 and pending is None
                and _is_final_control(value, messages, message)):
            continue
        if message["role"] == "user" and pending is None and content.startswith("Initial contract retrieval (source material):\n"):
            # The runner exposes real source text before its first policy call.
            # Count that evidence without inventing a historical model tool.
            if index != 2:
                raise ValueError("CUAD initial retrieval must precede all historical actions")
            successes += int(observe(json.loads(content.split("\n", 1)[1])))
        elif message["role"] == "assistant" and pending is None:
            pending = json.loads(content)
            if not isinstance(pending, dict) or "answer" in pending or pending.get("tool") not in {"search_document", "read_document"}:
                raise ValueError("CUAD teacher history contains an invalid tool or leaked final")
            if pending["tool"] == "read_document":
                check_read(pending.get("arguments", {}))
        elif message["role"] == "user" and pending is not None and content.startswith(_OBSERVATION):
            payload = content[len(_OBSERVATION):].split("\nRemaining turns:", 1)[0]
            observation = json.loads(payload)
            if pending["tool"] == "read_document" and isinstance(observation, dict) and not observation.get("error"):
                arguments = pending["arguments"]
                start = arguments["start"]
                limit = min(arguments.get("end", start + 8000), start + 12000)
                if observation.get("start") != start or type(observation.get("end")) is not int or observation["end"] > limit:
                    raise ValueError("CUAD read observation differs from the requested range")
            successes += int(observe(observation))
            pending = None
        else:
            raise ValueError("Invalid CUAD tool/observation history")
    if pending is not None:
        raise ValueError("CUAD historical tool lacks its observation")
    action = json.loads(value["completion"])
    if not isinstance(action, dict):
        raise ValueError("CUAD teacher completion must be exact action JSON")
    stage = value["stage"]
    if stage in {"search", "read"}:
        arguments = action.get("arguments", {})
        if "answer" in action or action.get("tool") != ("search_document" if stage == "search" else "read_document") or not isinstance(arguments, dict):
            raise ValueError("CUAD target tool disagrees with its stage")
        if stage == "search" and (not isinstance(arguments.get("query"), str) or not arguments["query"].strip()):
            raise ValueError("CUAD search requires a nonempty query")
        if stage == "read":
            check_read(arguments)
    else:
        rows = action.get("answer")
        if "tool" in action or not isinstance(rows, list) or not successes:
            raise ValueError("CUAD tool final needs successful historical source observations")
        spans = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError("CUAD final requires verbatim span objects")
            confidence = row.get("confidence", 1.0)
            if isinstance(confidence, bool) or not isinstance(confidence, (float, int)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("CUAD final confidence must be finite and in [0, 1]")
            spans.append(row["text"])
        if Counter(spans) != Counter(fields):
            raise ValueError("CUAD final target differs from declared grounding spans")
        sources = [" ".join(text.split()) for text in
                   (source_texts if _native_cuad_sources is None else _native_cuad_sources)]
        if any(not any(" ".join(span.split()) in source for source in sources) for span in spans):
            raise ValueError("CUAD final span is absent from actual source observations")
    return value


def load_cuad_records(path):
    records, seen = [], set()
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = validate_cuad_record(json.loads(line))
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"Invalid CUAD tool record at line {number}: {exc}") from exc
            key = (value["task_id"], value["metadata"].get("plan_id"), value["stage"], value["completion"])
            if key in seen:
                raise ValueError("Duplicate CUAD teacher segment")
            seen.add(key)
            records.append(value)
    if not records:
        raise ValueError("CUAD tool records must be nonempty")
    return records


def encode_record(record, tokenizer, *, max_prompt, max_response, batch_tokens, skips):
    domain = "cuad_tool" if record.get("source") == "cuad_train" else "web"
    try:
        prompt = encode_messages(tokenizer, record["messages"], max_prompt, allow_truncation=False)
    except ValueError as exc:
        if "exceeds its context budget" not in str(exc):
            raise
        skips[f"{domain}_{record['stage']}_prompt_too_long"] += 1
        return None
    completion = tokenizer.encode(record["completion"], add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(completion) > max_response:
        skips[f"{domain}_{record['stage']}_target_too_long"] += 1
        return None
    if len(prompt) + len(completion) > batch_tokens:
        skips[f"{domain}_{record['stage']}_batch_budget"] += 1
        return None
    if record["stage"] == "final":
        visible = " ".join(tokenizer.decode(prompt, skip_special_tokens=False).split()).replace('""', '"')
        fields = record["grounding_fields"] if domain == "cuad_tool" else [field for row in json.loads(record["completion"])["answer"] for name, field in row.items() if name != "citations"]
        if any(" ".join(field.split()) not in visible and " ".join(json.dumps(field, ensure_ascii=False)[1:-1].split()) not in visible for field in fields):
            skips[f"{domain}_final_not_visible_after_encoding"] += 1
            return None
    stage = f"cuad_tool_{record['stage']}" if domain == "cuad_tool" else record["stage"]
    category = record["metadata"]["category"] if domain == "cuad_tool" else "independent_web"
    return Example(record["task_id"], stage, category, prompt, completion)


class MixedSampler:
    def __init__(self, records, cuad_tasks, tokenizer, runner, *, rank, world, seed, web_fraction,
                 negative_fraction, max_prompt, max_response, batch_tokens, cuad_records=(), cuad_tool_fraction=0.75,
                 cuad_tool_partial_stages=False, record_encoder=None, recorded_cuad_stages=False,
                 web_stage_profile=None):
        encoder = encode_record if record_encoder is None else record_encoder
        self.rng = random.Random(seed + rank)
        self.tokenizer, self.runner = tokenizer, runner
        self.max_prompt, self.max_response, self.batch_tokens = max_prompt, max_response, batch_tokens
        self.skips, self.web, self.cuad = Counter(), defaultdict(list), defaultdict(lambda: defaultdict(list))
        self.cuad_tools = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.cuad_missing_stages = []
        ids = sorted({record["task_id"] for record in records})
        assigned = set(ids[rank::world])
        for record in records:
            if record["task_id"] in assigned:
                example = encoder(record, tokenizer, max_prompt=max_prompt, max_response=max_response,
                                        batch_tokens=batch_tokens, skips=self.skips)
                if example is not None:
                    self.web[example.stage].append(example)
        for task in cuad_tasks:
            if task.benchmark != "cuad" or task.split != "train":
                raise ValueError("CUAD auxiliary accepts CUAD train tasks only")
            if not isinstance(task.gold, list) or any(not isinstance(span, str) or not span.strip() for span in task.gold):
                raise ValueError("CUAD gold must contain only nonempty spans")
        cuad_ids = sorted({task.id for task in cuad_tasks} | {record["task_id"] for record in cuad_records})
        assigned_cuad = set(cuad_ids[rank::world])
        for task in cuad_tasks:
            if task.id in assigned_cuad:
                self.cuad[bool(task.gold)][str(task.metadata.get("category", task.prompt))].append(task)
        for record in cuad_records:
            if record["task_id"] in assigned_cuad:
                example = encoder(record, tokenizer, max_prompt=max_prompt, max_response=max_response,
                                        batch_tokens=batch_tokens, skips=self.skips)
                if example is not None:
                    self.cuad_tools[record["metadata"]["positive"]][example.category][record["stage"]].append(example)
        for positive, categories in self.cuad_tools.items():
            for category, stages in categories.items():
                if set(stages) != set(CUAD_STAGES):
                    if not (recorded_cuad_stages and set(stages) <= set(CUAD_STAGES)
                            or cuad_tool_partial_stages and positive and set(stages) == {"search", "read"}):
                        raise ValueError(f"CUAD tool category {category!r} lost a full stage; raise budgets; skipped={dict(self.skips)}")
                    self.cuad_missing_stages.append({"positive": positive, "category": category,
                        "available_stages": [stage for stage in CUAD_STAGES if stage in stages],
                        "missing_stages": [stage for stage in CUAD_STAGES if stage not in stages]})
        if not self.web and not self.cuad and not self.cuad_tools:
            raise ValueError(f"No eligible local training examples fit the token budget; skipped={dict(self.skips)}")
        self.web_fraction = web_fraction if self.cuad or self.cuad_tools else 1.0
        if not self.web:
            if web_fraction > 0:
                raise ValueError(f"No local web teacher fits; increase budgets or reduce worker count; skipped={dict(self.skips)}")
            self.web_fraction = 0.0
        if self.web_fraction > 0 and set(self.web) != set(STAGES):
            raise ValueError(f"Every local web stage must retain an eligible example; skipped={dict(self.skips)}")
        self.negative_fraction = negative_fraction
        self.cuad_tool_fraction = cuad_tool_fraction
        self.web_stage_cycle = STAGES
        if web_stage_profile is not None:
            from dcc_review_training import WEB_STAGE_PROFILE, WEB_STAGE_CYCLE
            if web_stage_profile != WEB_STAGE_PROFILE:
                raise ValueError("Unsupported explicit Web stage profile")
            self.web_stage_cycle = WEB_STAGE_CYCLE
        self.stage_index, self.cuad_stage_index = rank, rank
        self.pending = None
        self.available = {stage: len(values) for stage, values in self.web.items()}

    def sample(self):
        if self.pending is not None:
            example, self.pending = self.pending, None
            return example
        if self.web and self.rng.random() < self.web_fraction:
            stage = self.web_stage_cycle[self.stage_index % len(self.web_stage_cycle)]
            self.stage_index += 1
            return self.rng.choice(self.web[stage])
        for _ in range(300):
            positive = self.rng.random() >= self.negative_fraction
            if positive not in self.cuad and positive not in self.cuad_tools:
                positive = not positive
            categories = self.cuad.get(positive, {})
            tool_categories = self.cuad_tools.get(positive, {})
            category = self.rng.choice(sorted(set(categories) | set(tool_categories)))
            if category in tool_categories and (category not in categories or self.rng.random() < self.cuad_tool_fraction):
                available_stages = [stage for stage in CUAD_STAGES if stage in tool_categories[category]]
                stage = available_stages[self.cuad_stage_index % len(available_stages)]
                self.cuad_stage_index += 1
                return self.rng.choice(tool_categories[category][stage])
            task = self.rng.choice(categories[category])
            example = supervised_example(task, self.runner, self.tokenizer, self.max_prompt, self.max_response)
            if example is None:
                self.skips["cuad_ungrounded_or_target_too_long"] += 1
                continue
            prompt, completion = example
            if len(prompt) + len(completion) > self.batch_tokens:
                self.skips["cuad_batch_budget"] += 1
                continue
            return Example(task.id, "cuad_positive" if positive else "cuad_negative", category, prompt, completion)
        raise RuntimeError("No grounded CUAD example fits after 300 draws; inspect budgets and retrieval")


def sample_batch(sampler, max_examples):
    examples, tokens = [], 0
    for _ in range(max_examples):
        example = sampler.sample()
        if examples and tokens + example.tokens > sampler.batch_tokens:
            # Preserve the selected stage rather than systematically dropping
            # long final answers when only part of a batch budget remains.
            sampler.pending = example
            break
        examples.append(example)
        tokens += example.tokens
    return examples


def native_phase_normalized_nll(logprobs, example):
    """Fixed phase mass, preserving every validated native completion token."""
    lengths = example.native_phase_lengths
    if (not isinstance(lengths, tuple) or len(lengths) != 2
            or any(type(length) is not int or length < 1 for length in lengths)
            or sum(lengths) != len(example.completion)
            or logprobs.ndim != 1 or logprobs.numel() != sum(lengths)):
        raise ValueError("Phase-normalized SFT requires validated native phase lengths")
    thought_length, _ = lengths
    # The boundary comes from validated real phase IDs, never text extraction.
    if example.completion[thought_length - 1] != 151668:
        raise ValueError("Native thought phase must retain its actual closing ID")
    return -(0.2 * logprobs[:thought_length].mean()
             + 0.8 * logprobs[thought_length:].mean())


def backward_batch(model, examples, *, native_phase_normalized_loss=False,
                   native_completion_json_mixture_loss=False, native_loss_telemetry=None,
                   native_telemetry_metadata=None):
    """Equal example weight; observations and historical assistants are context."""
    if native_phase_normalized_loss and native_completion_json_mixture_loss:
        raise ValueError("Native loss modes are mutually exclusive")
    if native_completion_json_mixture_loss:
        return backward_native_mixture_batch(model, examples, logprob_fn=completion_logprobs,
                                            telemetry=native_loss_telemetry,
                                            sample_metadata=native_telemetry_metadata)
    if not examples:
        raise ValueError("Cannot train an empty token batch")
    total = 0.0
    for example in examples:
        logprobs = completion_logprobs(model, example.prompt, example.completion)
        loss = (native_phase_normalized_nll(logprobs, example) if native_phase_normalized_loss
                else -logprobs.mean()) / len(examples)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite supervised loss; refusing adapter update")
        loss.backward()
        total += loss.item()
    return total


def average_example_gradients(parameters, local_examples, world):
    """Weight local mean gradients by the actual global example count.

    The token budget can give ranks different batch sizes. Averaging rank
    means would otherwise give examples in a smaller batch more weight.
    """
    if type(local_examples) is not int or local_examples < 1:
        raise ValueError("Gradient averaging requires a positive local example count")
    if world == 1:
        return local_examples
    total = torch.tensor(local_examples, dtype=torch.int64, device=parameters[0].device)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    global_examples = total.item()
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        parameter.grad.mul_(local_examples)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(global_examples)
    return global_examples


def train(args):
    phase_normalized = getattr(args, "native_phase_normalized_loss", False)
    mixture_loss = getattr(args, "native_completion_json_mixture_loss", False)
    task_coverage = getattr(args, "native_global_stage_task_coverage", False)
    task_first = getattr(args, "native_task_first_coverage", False)
    positive_final_first = getattr(args, "native_positive_final_first", False)
    if task_first and task_coverage:
        raise ValueError("Native task coverage modes are mutually exclusive")
    if positive_final_first and not task_first:
        raise ValueError("Positive-final-first curriculum requires native task-first coverage")
    if task_first and (not getattr(args, "native_thinking", False) or args.adapter):
        raise ValueError("Native task-first coverage requires fresh native LoRA training")
    if task_first and getattr(args, "web_stage_profile", None) is not None:
        raise ValueError("Native task-first coverage owns its explicit per-task stage schedule")
    if phase_normalized and mixture_loss:
        raise ValueError("Native loss modes are mutually exclusive")
    if (phase_normalized or mixture_loss) and not getattr(args, "native_thinking", False):
        raise ValueError("Native loss modes require --native-thinking")
    if task_coverage and not getattr(args, "native_thinking", False):
        raise ValueError("Global task coverage requires --native-thinking")
    sampler_factory = MixedSampler
    if task_coverage:
        from dcc_stage_task_coverage import GlobalStageTaskCoverageSampler, STAGE_TASK_COVERAGE_CONFIGURATION

        def sampler_factory(*values, **options):
            return GlobalStageTaskCoverageSampler(
                *values, sampler_class=MixedSampler, batch_sampler=sample_batch, **options)
    if task_first:
        from dcc_task_first_coverage import GlobalTaskFirstSampler, TASK_FIRST_CONFIGURATION

        def sampler_factory(*values, **options):
            return GlobalTaskFirstSampler(*values, sampler_class=MixedSampler,
                batch_sampler=sample_batch, positive_final_first=positive_final_first, **options)
    native_data = None
    if getattr(args, "native_thinking", False):
        from dcc_native_sft import prepare_native_training

        native_data = prepare_native_training(args, source_root=Path(__file__).resolve().parents[1])
    rank, local, world = (int(os.environ.get(key, default)) for key, default in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1")))
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
    torch.manual_seed(args.seed)
    records = native_data.records if native_data is not None else load_records(args.records)
    cuad_records = (native_data.cuad_records if native_data is not None else
                    load_cuad_records(args.cuad_tool_records) if args.cuad_tool_records else [])
    tasks = [] if native_data is not None else load_tasks(args.cuad_train_data) if args.cuad_train_data else []
    runner = EpisodeRunner(max_steps=3, initial_chunks=args.initial_chunks,
                           retrieval_profile=load_retrieval_profile(args.retrieval_profile),
                           retrieval_context_chars=getattr(args, "retrieval_context_chars", 0),
                           retrieval_max_context_chars=getattr(args, "retrieval_max_context_chars", 16000))
    telemetry_metadata = native_sample_metadata(records, cuad_records) if mixture_loss else None
    output = Path(args.output_dir)
    prepare_training_output(output, rank=rank, world_size=world, distributed=dist)
    config = {key: value for key, value in vars(args).items()
              if key not in {"model", "adapter", "records", "cuad_train_data", "cuad_tool_records", "output_dir", "retrieval_profile",
                             "native_thinking", "native_train_profile", "native_expected_source", "web_stage_profile",
                             "native_phase_normalized_loss", "native_completion_json_mixture_loss",
                             "native_global_stage_task_coverage", "native_task_first_coverage",
                             "native_positive_final_first"}}
    if task_coverage:
        config["sampler_configuration"] = dict(STAGE_TASK_COVERAGE_CONFIGURATION)
    if task_first:
        config["sampler_configuration"] = dict(TASK_FIRST_CONFIGURATION,
                                               positive_final_first_enabled=positive_final_first)
    if getattr(args, "web_stage_profile", None) is not None:
        config["web_stage_profile"] = args.web_stage_profile
    if native_data is not None:
        config["native_thinking"] = native_data.configuration
    if phase_normalized or mixture_loss:
        config["training_method"] = "accepted-trajectory supervised fine-tuning; not RL"
        config["loss_configuration"] = (dict(MIXTURE_LOSS_CONFIGURATION) if mixture_loss else {
            "version": "native-phase-normalized-completion-sft-v1",
            "thought_weight": 0.2, "json_weight": 0.8,
            "phase_boundaries": "validated-real-native-phase-ids",
            "example_weighting": "global-equal-example",
        })
        if mixture_loss:
            config["loss_telemetry_configuration"] = dict(LOSS_TELEMETRY_CONFIGURATION)
    config.update(warm_start=bool(args.adapter), retrieval_profile_enabled=bool(args.retrieval_profile),
                  episode_prompt_version=EPISODE_PROMPT_VERSION,
                  prompt_encoding_version=PROMPT_ENCODING_VERSION,
                  web_reader_version=WEB_READER_VERSION)
    started, counts, ids = time.monotonic(), Counter(), set()
    loss_summary = None
    with TrainingJournal(output, rank, sync=getattr(args, "sync_artifacts", False)) as log:
        def record(**value):
            line = log.record(**value)
            if rank == 0:
                print(line, flush=True)

        record(event="start", world_size=world, algorithm="mixed tool/span SFT", **config)
        model, tokenizer = load_policy_model(args.model, adapter=args.adapter, trainable=True, rank=local,
                                             lora_rank=args.lora_rank, quantization=args.quantization)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters or any("lora_" not in name for name, parameter in model.named_parameters() if parameter.requires_grad):
            raise ValueError("Tool SFT requires only LoRA parameters to be trainable")
        if any(isinstance(module, torch.nn.Dropout) and module.p != 0 for name, module in model.named_modules() if "lora_dropout" in name):
            raise ValueError("Tool SFT requires LoRA dropout=0")
        if hasattr(model, "dcc_quantization_report"):
            record(event="quantization", **model.dcc_quantization_report)
        sampler = sampler_factory(records, tasks, tokenizer, runner, rank=rank, world=world, seed=args.seed,
                               web_fraction=args.web_fraction, negative_fraction=args.negative_fraction,
                               max_prompt=args.max_prompt_tokens, max_response=args.max_new_tokens, batch_tokens=args.batch_tokens,
                               cuad_records=cuad_records, cuad_tool_fraction=args.cuad_tool_fraction,
                               cuad_tool_partial_stages=getattr(args, "cuad_tool_partial_stages", False),
                               record_encoder=native_data.encode_record if native_data is not None else None,
                               recorded_cuad_stages=native_data is not None,
                               web_stage_profile=getattr(args, "web_stage_profile", None))
        record(event="eligible", web_stages=sampler.available, cuad_tasks=sum(len(rows) for categories in sampler.cuad.values() for rows in categories.values()),
               cuad_tool_stages=dict(Counter({stage: sum(len(stages.get(stage, [])) for categories in sampler.cuad_tools.values() for stages in categories.values()) for stage in CUAD_STAGES})),
               cuad_missing_stages=sampler.cuad_missing_stages,
               effective_web_fraction=sampler.web_fraction, skipped=dict(sampler.skips), trainable_parameters=sum(p.numel() for p in parameters),
               **({"sampler_buckets": sampler.bucket_summary()} if task_coverage or task_first else {}))
        optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)

        def save(name, step):
            log.checkpoint_boundary()
            if world > 1:
                dist.barrier()
            if rank == 0:
                def write_checkpoint(destination):
                    model.save_pretrained(destination)
                    tokenizer.save_pretrained(destination)
                    torch.save({"optimizer": optimizer.state_dict(), "optimizer_steps": step,
                                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                                "sampler_rng": sampler.rng.getstate()}, destination / "training_state.pt")
                    (destination / "training_config.json").write_text(json.dumps(config, indent=2) + "\n")

                publish_checkpoint(output, name, step, write_checkpoint, sync=log.sync)
            if world > 1:
                dist.barrier()

        if getattr(args, "save_initial", False):
            save("initial", 0)

        for step in range(1, args.steps + 1):
            begin = time.monotonic()
            examples = (sampler.next_batch(args.samples_per_step) if task_coverage or task_first
                        else sample_batch(sampler, args.samples_per_step))
            model.train()
            optimizer.zero_grad(set_to_none=True)
            step_telemetry = {} if mixture_loss else None
            loss = backward_batch(model, examples, native_phase_normalized_loss=phase_normalized,
                                  native_completion_json_mixture_loss=mixture_loss,
                                  native_loss_telemetry=step_telemetry,
                                  native_telemetry_metadata=telemetry_metadata)
            if mixture_loss:
                loss_summary = merge_native_loss_summaries(loss_summary, step_telemetry["summary"])
            global_examples = average_example_gradients(parameters, len(examples), world)
            norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            if not torch.isfinite(norm) or norm <= 0:
                raise FloatingPointError("Expected finite nonzero adapter gradient; refusing update")
            optimizer.step()
            counts.update(example.stage for example in examples)
            ids.update(example.task_id for example in examples)
            record(event="step", step=step, loss=loss, gradient_norm=norm.item(), seconds=time.monotonic() - begin,
                   global_examples=global_examples,
                   samples=[{"task_id": e.task_id, "stage": e.stage, "category": e.category,
                             "prompt_tokens": len(e.prompt), "completion_tokens": len(e.completion)} for e in examples],
                   stage_counts=dict(counts), skipped=dict(sampler.skips),
                   **({"global_sampler_coverage": sampler.coverage_summary()} if task_first else {}),
                   **({"native_loss_telemetry": step_telemetry} if mixture_loss else {}))
            if args.save_every and step % args.save_every == 0:
                save(f"step_{step:04d}", step)
        save("final", args.steps)
        record(event="finished", optimizer_steps=args.steps, stage_counts=dict(counts), unique_task_ids=len(ids),
               skipped=dict(sampler.skips), elapsed=time.monotonic() - started,
               **({"global_sampler_coverage": sampler.coverage_summary()} if task_first else {}),
               **({"native_loss_summary": loss_summary} if mixture_loss else {}))
        finalize_training(log, args.steps, world_size=world, distributed=dist)
    if world > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-thinking", action="store_true",
                        help="Train fresh LoRA on fully validated recorded native Thinking Web/CUAD rollouts")
    samplers = parser.add_mutually_exclusive_group()
    samplers.add_argument("--native-global-stage-task-coverage", action="store_true",
                        help="Opt in to global domain/polarity/stage task decks; reweights categories; fresh runs only")
    samplers.add_argument("--native-task-first-coverage", action="store_true",
                        help="Uniform native task decks before stage selection, with explicit domain/polarity quotas")
    parser.add_argument("--native-positive-final-first", action="store_true",
                        help="With task-first coverage, expose each positive CUAD task's real final on its first draw")
    native_losses = parser.add_mutually_exclusive_group()
    native_losses.add_argument("--native-phase-normalized-loss", action="store_true",
                               help="Opt in to SFT loss 0.2*thought_mean + 0.8*JSON_mean on real native IDs")
    native_losses.add_argument("--native-completion-json-mixture-loss", action="store_true",
                               help="Opt in to SFT loss 0.8*completion_mean + 0.2*JSON_mean and detached phase telemetry")
    parser.add_argument("--native-train-profile",
                        help="Separately approved native identity and public collection schema profile")
    parser.add_argument("--native-expected-source",
                        help="Caller-approved actual frozen integration commit; must match source receipt")
    parser.add_argument("--model", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--cuad-train-data")
    parser.add_argument("--cuad-tool-records")
    parser.add_argument("--cuad-tool-fraction", type=float, default=0.75)
    parser.add_argument("--cuad-tool-partial-stages", action="store_true",
                        help="Allow positive search/read-only category buckets; negatives still require final")
    parser.add_argument("--output-dir", required=True, help="New or empty directory; existing runs are never resumed")
    parser.add_argument("--sync-artifacts", action="store_true",
                        help="Request filesystem fsync only at checkpoint/terminal boundaries")
    parser.add_argument("--adapter")
    parser.add_argument("--retrieval-profile")
    parser.add_argument("--retrieval-context-chars", type=int, default=0)
    parser.add_argument("--retrieval-max-context-chars", type=int, default=16000)
    parser.add_argument("--quantization", choices=("none", "nf4"), default="none")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--web-stage-profile", choices=("final50-search25-fetch25",), default=None,
                        help="Opt in to exact final/search/final/fetch Web-stage cycle; default preserves original sampler")
    parser.add_argument("--web-fraction", type=float, default=0.5)
    parser.add_argument("--negative-fraction", type=float, default=1 / 3)
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--batch-tokens", type=int, default=12288)
    parser.add_argument("--samples-per-step", type=int, default=2)
    parser.add_argument("--initial-chunks", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=12)
    parser.add_argument("--save-initial", action="store_true", help="Save the initial adapter before any optimizer update")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (args.native_phase_normalized_loss or args.native_completion_json_mixture_loss) and not args.native_thinking:
        parser.error("Native loss modes require --native-thinking")
    if args.native_global_stage_task_coverage and not args.native_thinking:
        parser.error("Global task coverage requires --native-thinking")
    if args.native_task_first_coverage and (not args.native_thinking or args.adapter):
        parser.error("Task-first coverage requires fresh --native-thinking training")
    if args.native_positive_final_first and not args.native_task_first_coverage:
        parser.error("Positive-final-first curriculum requires --native-task-first-coverage")
    if args.native_task_first_coverage and args.web_stage_profile is not None:
        parser.error("Task-first coverage owns its per-task stage schedule")
    if min(args.steps, args.lora_rank, args.max_prompt_tokens, args.max_new_tokens, args.batch_tokens,
           args.samples_per_step, args.initial_chunks) < 1 or args.save_every < 0:
        parser.error("training steps, ranks and token budgets must be positive; save-every must be nonnegative")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in (args.web_fraction, args.negative_fraction, args.cuad_tool_fraction)):
        parser.error("mixture fractions must be finite and between zero and one")
    if any(not math.isfinite(value) or value <= 0 for value in (args.learning_rate, args.max_grad_norm)):
        parser.error("learning rate and gradient clipping must be finite and positive")
    train(args)


if __name__ == "__main__":
    main()
