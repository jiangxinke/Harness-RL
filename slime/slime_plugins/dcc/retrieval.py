"""Optional query expansion fitted exclusively from CUAD training answers.

Profiles are external experiment artifacts. Inference reads only the question,
the profile, and the source contract; it never reads task labels or metadata.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any
from collections.abc import Iterable

from .data import Task
from .tools import DocumentTools, _terms

SCHEMA = "harness-rl.dcc.retrieval"
VERSION = 1
METHOD = "train_gold_log_lift_rrf"
_BOILERPLATE = {
    "party", "parties", "such", "have", "has", "will", "not", "may", "other",
    "its", "their", "each", "all", "including", "hereunder", "thereof", "therein",
    "been", "but", "than", "any", "date", "agreement", "section",
}


def _question(value: str) -> str:
    return " ".join(value.split())


def _category(value: str) -> str:
    value = _question(value)
    match = re.search(r'related to ["“]([^"”]+)["”]', value, re.IGNORECASE)
    return match.group(1) if match else _question(value)


@dataclass(frozen=True)
class RetrievalProfile:
    questions: dict[str, tuple[str, ...]]
    rrf_constant: float


def fit_retrieval_profile(
    tasks: Iterable[Task], *, max_terms: int = 10, min_examples: int = 3,
    min_fraction: float = 0.08, rrf_constant: float = 60.0,
) -> dict[str, Any]:
    """Return a portable profile, rejecting every non-training input row."""
    if isinstance(max_terms, bool) or not isinstance(max_terms, int) or max_terms < 1:
        raise ValueError("max_terms must be a positive integer")
    if isinstance(min_examples, bool) or not isinstance(min_examples, int) or min_examples < 1:
        raise ValueError("min_examples must be a positive integer")
    if isinstance(min_fraction, bool) or not isinstance(min_fraction, (int, float)) or not math.isfinite(min_fraction) or not 0 <= min_fraction <= 1:
        raise ValueError("min_fraction must be between zero and one")
    if isinstance(rrf_constant, bool) or not isinstance(rrf_constant, (int, float)) or not math.isfinite(rrf_constant) or rrf_constant <= 0:
        raise ValueError("rrf_constant must be finite and positive")
    counts: dict[str, Counter] = defaultdict(Counter)
    positives: Counter = Counter()
    all_terms: Counter = Counter()
    questions: dict[str, str] = {}
    ids: set[str] = set()
    contracts: set[str] = set()
    documents: set[str] = set()
    for task in tasks:
        if task.benchmark != "cuad" or task.split != "train":
            raise ValueError("retrieval fitting accepts CUAD train rows only")
        if task.id in ids:
            raise ValueError(f"duplicate training task id: {task.id}")
        ids.add(task.id)
        contracts.add(task.id.rsplit("__", 1)[0])
        documents.add(task.document)
        if not isinstance(task.gold, list) or not all(isinstance(span, str) and span.strip() for span in task.gold):
            raise ValueError("training gold must be a list of nonempty span strings")
        question, category = _question(task.prompt), _category(task.prompt)
        questions[question] = category
        if task.gold:
            words = {
                word for word in _terms(" ".join(task.gold))
                if len(word) > 2 and word not in _BOILERPLATE and word.isalpha()
            }
            counts[category].update(words)
            all_terms.update(words)
            positives[category] += 1
    if not ids or not positives:
        raise ValueError("retrieval fitting requires nonempty training data with positive answers")
    total = sum(positives.values())
    expansions: dict[str, list[str]] = {}
    for category, frequencies in counts.items():
        cutoff = max(min_examples, positives[category] * min_fraction)
        scores = {
            word: math.log((frequency + 1) / (positives[category] + 2))
            - math.log((all_terms[word] + 1) / (total + 2))
            for word, frequency in frequencies.items() if frequency >= cutoff
        }
        expansions[category] = sorted(scores, key=lambda word: (-scores[word], word))[:max_terms]
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "method": METHOD,
        "parameters": {
            "max_terms": max_terms, "min_examples": min_examples,
            "min_fraction": min_fraction, "rrf_constant": rrf_constant,
        },
        "training": {
            "benchmark": "cuad", "split": "train", "tasks": len(ids),
            "positive_tasks": total, "contracts": len(contracts),
            "unique_documents": len(documents),
            "task_ids_sha256": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
        },
        "questions": {
            question: {"category": category, "terms": expansions.get(category, [])}
            for question, category in sorted(questions.items())
        },
    }


def parse_retrieval_profile(value: Any) -> RetrievalProfile:
    """The runtime profile validator, also used by portable TRAIN bindings."""
    try:
        if not isinstance(value, dict) or value.get("schema") != SCHEMA or type(value.get("version")) is not int or value["version"] != VERSION or value.get("method") != METHOD:
            raise ValueError("unsupported retrieval profile schema, version, or method")
        training = value["training"]
        if training["benchmark"] != "cuad" or training["split"] != "train":
            raise ValueError("retrieval profile must declare CUAD train provenance")
        for name in ("tasks", "positive_tasks", "contracts"):
            if type(training[name]) is not int or training[name] < 1:
                raise ValueError("invalid retrieval training summary")
        if training["positive_tasks"] > training["tasks"] or training["contracts"] > training["tasks"]:
            raise ValueError("inconsistent retrieval training summary")
        if not re.fullmatch(r"[0-9a-f]{64}", training["task_ids_sha256"]):
            raise ValueError("invalid retrieval training ID summary")
        parameters = value["parameters"]
        max_terms, constant = parameters["max_terms"], parameters["rrf_constant"]
        if type(max_terms) is not int or max_terms < 1 or isinstance(constant, bool) or not isinstance(constant, (int, float)) or not math.isfinite(constant) or constant <= 0:
            raise ValueError("invalid retrieval parameters")
        if type(parameters["min_examples"]) is not int or parameters["min_examples"] < 1 or isinstance(parameters["min_fraction"], bool) or not isinstance(parameters["min_fraction"], (int, float)) or not math.isfinite(parameters["min_fraction"]) or not 0 <= parameters["min_fraction"] <= 1:
            raise ValueError("invalid retrieval fitting parameters")
        queries = value["questions"]
        if not isinstance(queries, dict) or not queries:
            raise ValueError("retrieval profile requires known training questions")
        normalized = {}
        for question, entry in queries.items():
            if not isinstance(question, str) or not question.strip() or question != _question(question) or not isinstance(entry, dict) or entry.get("category") != _category(question):
                raise ValueError("invalid retrieval question/category mapping")
            words = entry["terms"]
            if not isinstance(words, list) or len(words) > max_terms or len(set(words)) != len(words) or not all(isinstance(word, str) and re.fullmatch(r"[a-z]{3,}", word) for word in words):
                raise ValueError("invalid retrieval expansion terms")
            normalized[question] = tuple(words)
        return RetrievalProfile(normalized, float(constant))
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("malformed retrieval profile") from exc


def load_retrieval_profile(path: str | Path | None) -> RetrievalProfile | None:
    """None disables the feature; explicit missing or malformed paths fail."""
    if path is None:
        return None
    try:
        return parse_retrieval_profile(json.loads(Path(path).read_text(encoding="utf-8")))
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("malformed retrieval profile") from exc


def _rank(document_tools: DocumentTools, query: str) -> list[int]:
    """The existing DocumentTools BM25 formula, retaining the complete rank."""
    query_terms = sorted(set(_terms(query)))
    scores = []
    for index, counts in enumerate(document_tools.counts):
        length, score = sum(counts.values()), 0.0
        for term in query_terms:
            frequency = counts[term]
            if frequency:
                df = document_tools.document_frequency[term]
                inverse = math.log(1 + (len(document_tools.counts) - df + 0.5) / (df + 0.5))
                score += inverse * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * length / max(document_tools.average_length, 1)))
        scores.append((score, index))
    return [index for _, index in sorted(scores, key=lambda item: (-item[0], item[1]))]


def document_read_windows(hit: dict[str, Any]) -> list[dict[str, int]]:
    """Fixed public read plans derived only from an observed search hit."""
    start, end = hit["start"], hit["end"]
    if type(start) is not int or type(end) is not int or not 0 <= start < end:
        raise ValueError("read window requires valid observed hit offsets")
    windows = [
        {"start": start, "end": end}, {"start": start},
        {"start": start, "end": start + 12000},
        {"start": max(0, start - 800), "end": end + 800},
        {"start": max(0, start - 1600), "end": end + 1600},
        {"start": max(0, start - 4000), "end": start + 8000},
    ]
    return list({tuple(window.items()): window for window in windows}.values())


def search_document(
    document_tools: DocumentTools, query: str, *, top_k: int = 4,
    profile: RetrievalProfile | None = None,
    context_chars: int = 0, max_context_chars: int = 16000,
) -> dict[str, Any]:
    """Retrieve the same bounded number of original source chunks.

    Unknown questions and categories use the unmodified baseline. Two complete
    BM25 rankings are fused; each chunk appears at most once, with index ties.
    """
    words = profile.questions.get(_question(query), ()) if profile is not None else ()
    if not words:
        observation = document_tools.search_document(query, top_k=top_k)
        return expand_document_context(document_tools, observation, context_chars=context_chars,
                                       max_chars=max_context_chars)
    fused: Counter = Counter()
    for order in (_rank(document_tools, query), _rank(document_tools, " ".join(words))):
        for position, index in enumerate(order):
            fused[index] += 1 / (profile.rrf_constant + position)
    order = sorted(fused, key=lambda index: (-fused[index], index))
    hits = [
        {
            "start": document_tools.chunks[index].start,
            "end": document_tools.chunks[index].end,
            "text": document_tools.chunks[index].text,
            "score": round(fused[index], 4),
        }
        for index in order[:max(1, min(int(top_k), 8))]
    ]
    return expand_document_context(document_tools, {"document_length": len(document_tools.document), "hits": hits},
                                   context_chars=context_chars, max_chars=max_context_chars)


def expand_document_context(
    document_tools: DocumentTools, observation: dict[str, Any], *,
    context_chars: int = 800, max_chars: int = 16000,
) -> dict[str, Any]:
    """Expand ranked hits on both sides, merge overlap, and cap source characters.

    This opt-in repair for clauses crossing chunk boundaries uses source offsets
    only. It never looks at answers or task metadata. The character cap counts
    the returned source text, not JSON/chat formatting or tokenizer overhead;
    callers must still enforce their actual token budget. Zero context preserves
    the historical observation byte-for-byte in value.
    """
    if type(context_chars) is not int or context_chars < 0:
        raise ValueError("context_chars must be a nonnegative integer")
    if type(max_chars) is not int or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    if context_chars == 0:
        return observation
    source = document_tools.document
    windows: list[dict[str, Any]] = []
    for hit in observation["hits"]:
        start, end = hit["start"], hit["end"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(source) or hit["text"] != source[start:end]:
            raise ValueError("retrieval hit must identify an exact source interval")
        lower, upper = max(0, start - context_chars), min(len(source), end + context_chars)
        touching = [i for i, window in enumerate(windows) if lower <= window["end"] and upper >= window["start"]]
        merged_lower = min([lower] + [windows[i]["start"] for i in touching])
        merged_upper = max([upper] + [windows[i]["end"] for i in touching])
        used = sum(window["end"] - window["start"] for i, window in enumerate(windows) if i not in touching)
        if used + merged_upper - merged_lower > max_chars:
            # Prefer the higher-ranked windows already returned. A later hit
            # may still fit through overlap, so continue rather than stopping.
            continue
        merged = {"start": merged_lower, "end": merged_upper,
                  "text": source[merged_lower:merged_upper], "score": hit["score"]}
        if touching:
            first = touching[0]
            merged["score"] = windows[first]["score"]
            windows[first] = merged
            windows = [window for i, window in enumerate(windows) if i not in touching[1:]]
        else:
            windows.append(merged)
    return {**observation, "hits": windows}
