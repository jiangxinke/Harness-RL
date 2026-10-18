from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import string
import tempfile
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Iterable

from .core import Harness, RolloutTree
from .http_search import HttpSearch
from .slime_adapter import SGLangHarnessModel

logger = logging.getLogger(__name__)

NORMAL_MAX_STEPS = 32
FORCED_ANSWER_MAX_TOKENS = 256
FORCED_ANSWER_PREFIX = "<think>I must now give the best short final answer using the evidence already collected.</think><answer>"

DATASET_FILES: dict[str, tuple[str, ...]] = {
    "2wikimultihopqa": ("FlashRAG_datasets/2wikimultihopqa/dev.jsonl",),
    "bamboogle": ("FlashRAG_datasets/bamboogle/test.jsonl",),
    "frames": ("FRAMES/test.jsonl",),
    "hotpotqa": ("FlashRAG_datasets/hotpotqa/dev.jsonl",),
    "musique": ("FlashRAG_datasets/musique/dev.jsonl",),
    "nq": (
        "FlashRAG_datasets/nq/dev.jsonl",
        "FlashRAG_datasets/nq/test.jsonl",
    ),
    "triviaqa": (
        "FlashRAG_datasets/triviaqa/dev.jsonl",
        "FlashRAG_datasets/triviaqa/test.jsonl",
    ),
}

_RESULT_LOCK = threading.Lock()
_RESULT_CACHE: dict[str, dict[str, Any]] | None = None


def normalize_answer(value: Any) -> str:
    """SQuAD/FlashRAG-style normalization used by lexical QA metrics."""

    text = "" if value is None else str(value)
    text = text.lower()
    text = "".join(character for character in text if character not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_recall(prediction: Any, reference: Any) -> float:
    """Return reference-token coverage (reported under the legacy ``f1`` key)."""

    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not reference_tokens:
        return float(not prediction_tokens)
    overlap = Counter(prediction_tokens) & Counter(reference_tokens)
    return sum(overlap.values()) / len(reference_tokens)


def score_prediction(prediction: Any, references: Iterable[Any]) -> tuple[float, float]:
    """Return token recall and lexical exact match; ACC is replaced by LLM judging."""

    references = [reference for reference in references if reference is not None]
    if not references:
        return 0.0, 0.0
    prediction_normalized = normalize_answer(prediction)
    lexical_accuracy = float(any(prediction_normalized == normalize_answer(reference) for reference in references))
    recall = max(token_recall(prediction, reference) for reference in references)
    return recall, lexical_accuracy


def included_in_metrics(record: dict[str, Any]) -> bool:
    """Exclude incomplete/empty rollouts from aggregate evaluation metrics."""

    explicit = record.get("included_in_metrics")
    if explicit is not None:
        return bool(explicit)
    if record.get("status") not in {None, "completed"}:
        return False
    if "prediction" in record and not str(record["prediction"]).strip():
        return False
    return True


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def original_dataset_weights(data_root: str | Path) -> dict[str, int]:
    root = Path(data_root)
    weights: dict[str, int] = {}
    for source, relative_paths in DATASET_FILES.items():
        paths = [root / relative_path for relative_path in relative_paths]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing original dev/test files for {source}: {missing}")
        weights[source] = sum(_line_count(path) for path in paths)
    return weights


def _stable_row_key(row: dict[str, Any]) -> str:
    payload = json.dumps(
        [row.get("source"), row.get("split"), row.get("query"), row.get("answer")],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_eval_rows(rows: list[dict[str, Any]], total_samples: int) -> list[dict[str, Any]]:
    """Select a deterministic source-balanced subset (630 or 315 in production)."""

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source"])].append(row)
    sources = sorted(by_source)
    if not sources or total_samples <= 0 or total_samples % len(sources) != 0:
        raise ValueError(f"total_samples must be a positive multiple of {len(sources)}, got {total_samples}")
    per_source = total_samples // len(sources)
    selected: list[dict[str, Any]] = []
    for source in sources:
        candidates = sorted(by_source[source], key=_stable_row_key)
        if len(candidates) < per_source:
            raise ValueError(f"dataset {source} contains {len(candidates)} rows, need {per_source}")
        selected.extend(candidates[:per_source])
    return sorted(selected, key=lambda row: (str(row["source"]), _stable_row_key(row)))


def prepare_eval_parquet(input_path: str | Path, output_path: str | Path, total_samples: int) -> dict[str, int]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = pq.read_table(input_path).to_pylist()
    selected = select_eval_rows(rows, total_samples)
    prepared = []
    counts: Counter[str] = Counter()
    for index, row in enumerate(selected):
        source = str(row["source"])
        counts[source] += 1
        prepared.append(
            {
                "query": str(row["query"]),
                "answer": [str(value) for value in row["answer"]],
                "metadata": {
                    "source": source,
                    "split": str(row["split"]),
                    "eval_index": index,
                },
            }
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    pq.write_table(pa.Table.from_pylist(prepared), temporary)
    os.replace(temporary, output)
    return dict(sorted(counts.items()))


def _answer_nodes(tree: RolloutTree) -> list[Any]:
    return [tree.nodes[node_id] for node_id in sorted(tree.nodes) if tree.nodes[node_id].kind == "answer"]


def prediction_from_tree(tree: RolloutTree) -> str:
    forced = [node for node in _answer_nodes(tree) if node.metadata.get("forced_final_answer")]
    if forced:
        return forced[-1].text
    accepted = [node for node in _answer_nodes(tree) if node.metadata.get("accepted") is not False]
    return accepted[-1].text if accepted else ""


def _terminated_active_node(tree: RolloutTree) -> int | None:
    terminal = [node for node in tree.nodes.values() if node.kind == "backtrack" and node.metadata.get("termination") is True]
    if terminal:
        reflect = tree.nodes[terminal[-1].parent]
        return reflect.parent
    direct = [node for node in tree.nodes.values() if node.kind == "termination" and node.metadata.get("termination") is True]
    if direct:
        return direct[-1].parent

    # A few controller failures historically set ``tree.status`` directly
    # without appending a harness termination node.  Preserve the useful
    # prefix in those cases so evaluation can still request one final answer.
    # For an update-writer failure, the audited resume checkpoint is the exact
    # prefix from which the rewrite would have continued.
    for audit in reversed(tree.resume_audit):
        selected = audit.get("selected_checkpoint")
        if isinstance(selected, int) and selected in tree.nodes:
            return selected

    # Invalid-action retry exhaustion leaves a harness backtrack as the newest
    # leaf but does not mark it as the terminal branch.  Exclude that harness
    # bookkeeping and retain the model's preceding thinking/evidence path.
    for leaf in sorted(tree.leaves(), reverse=True):
        node = tree.nodes[leaf]
        if node.kind == "backtrack" and node.parent is not None:
            parent = tree.nodes[node.parent]
            if parent.kind == "reflect" and parent.parent is not None:
                return parent.parent
        if node.kind == "termination":
            return node.parent
    return tree.root_id if tree.status == "terminated" else None


def force_final_answer(tree: RolloutTree, model: SGLangHarnessModel, *, max_tokens: int = FORCED_ANSWER_MAX_TOKENS) -> bool:
    """Add one gold-blind final-answer turn after any answerless termination."""

    active = _terminated_active_node(tree)
    if active is None or tree.status != "terminated" or prediction_from_tree(tree):
        return False
    prior_reason = tree.termination_reason

    generation_prompt, generation_context = model._fit_generation_prompt(
        _trace_without_query(tree.render_path(active)),
        FORCED_ANSWER_PREFIX,
        max_tokens,
    )
    generated = model._generate(
        generation_prompt,
        max_tokens=max_tokens,
        stop=["</answer>"],
        temperature=0.0,
    )
    raw = generated.raw
    match = re.search(r"(.*?)</answer>", raw, re.DOTALL)
    answer = (match.group(1) if match else raw).strip()
    answer = re.sub(r"<[^>]+>", " ", answer)
    answer = " ".join(answer.split())[:200].strip() or "unknown"

    think_id = tree.append(
        active,
        "think",
        "I must now give the best short final answer using the evidence already collected.",
        {
            "forced_final_answer": True,
            "model_turn_id": tree.model_turns,
            "forced_prefix": FORCED_ANSWER_PREFIX,
            "forced_from_termination_reason": prior_reason,
            "generated_output": generated.raw,
            "generation_prompt": generation_prompt,
            "generation_context": generation_context,
            "finish_reason": generated.finish_reason,
            "usage": {"completion_tokens": len(generated.token_ids)},
        },
    )
    tree.append(
        think_id,
        "answer",
        answer,
        {"accepted": True, "forced_final_answer": True},
    )
    tree.model_turns += 1
    tree.output_tokens += len(generated.token_ids)
    tree.status = "completed"
    tree.termination_reason = "forced_answer_after_max_steps" if str(prior_reason).startswith("max_steps_reached") else "forced_answer_after_termination"
    return True


def _trace_without_query(trace: str) -> str:
    end = trace.find("</query>")
    return trace[end + len("</query>") :] if end >= 0 else trace


def _references(label: Any) -> list[str]:
    values = label if isinstance(label, (list, tuple)) else [label]
    return [str(value) for value in values if value is not None]


def _eval_uid(sample: Any) -> str:
    payload = json.dumps(
        [
            sample.metadata.get("source"),
            sample.metadata.get("split"),
            sample.prompt,
            _references(sample.label),
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _results_path() -> Path | None:
    output_dir = os.environ.get("REFLECT_EVAL_OUTPUT_DIR")
    return Path(output_dir) / "predictions.jsonl" if output_dir else None


def _load_result_cache() -> dict[str, dict[str, Any]]:
    global _RESULT_CACHE
    with _RESULT_LOCK:
        if _RESULT_CACHE is not None:
            return _RESULT_CACHE
        _RESULT_CACHE = {}
        path = _results_path()
        if path and path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("eval_uid"):
                    _RESULT_CACHE[record["eval_uid"]] = record
        return _RESULT_CACHE


def _persist_record(record: dict[str, Any]) -> None:
    path = _results_path()
    if path is None:
        return
    with _RESULT_LOCK:
        cache = _RESULT_CACHE if _RESULT_CACHE is not None else {}
        if record["eval_uid"] in cache:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        cache[record["eval_uid"]] = record


def _record_to_sample(sample: Any, record: dict[str, Any]) -> Any:
    from slime.utils.types import Sample

    sample.response = record["prediction"]
    sample.response_length = 0
    sample.reward = float(record["f1"])
    sample.status = Sample.Status.COMPLETED
    sample.metadata = {**dict(sample.metadata or {}), "eval_result": record}
    return sample


async def generate(args: Any, sample: Any, sampling_params: dict[str, Any], evaluation: bool = False) -> Any:
    """Gold-blind Reflect evaluation with 32 normal turns and one forced answer turn."""

    if not evaluation:
        raise ValueError("slime_plugins.loongreflect.evaluation.generate is evaluation-only")
    if not isinstance(sample.prompt, str):
        raise TypeError("Reflect evaluation expects a raw string query")

    uid = _eval_uid(sample)
    cached = _load_result_cache().get(uid)
    if cached is not None:
        return _record_to_sample(sample, cached)

    from slime.rollout.sglang_rollout import GenerateState

    started = time.monotonic()
    state = GenerateState(args)
    model = SGLangHarnessModel(args, state.tokenizer, sample.prompt)
    search = HttpSearch(
        os.environ.get(
            "REFLECT_SEARCH_URL",
            getattr(args, "reflect_search_url", "http://127.0.0.1:8002/retrieve"),
        ),
        top_k=getattr(args, "reflect_search_top_k", 3),
    )
    harness = Harness(
        actor=model,
        search=search,
        backtrack_judge=model,
        update_writer=model,
        max_steps=NORMAL_MAX_STEPS,
        max_output_tokens=args.rollout_max_response_len,
        max_tokens_per_turn=min(
            getattr(args, "reflect_max_tokens_per_turn", 1024),
            args.rollout_max_response_len,
        ),
        max_protocol_retries=1,
        answer_validator=None,
        max_answer_retries=0,
        allow_reflect=(getattr(args, "reflect_action_protocol", "full") == "full"),
    )
    forced = False
    try:
        tree = await asyncio.to_thread(harness.run, sample.prompt)
        if tree.status == "terminated" and not prediction_from_tree(tree):
            forced = await asyncio.to_thread(force_final_answer, tree, model)
    finally:
        model.close()
        search.close()

    prediction = prediction_from_tree(tree)
    references = _references(sample.label)
    f1, lexical_accuracy = score_prediction(prediction, references)
    included = tree.status == "completed" and bool(prediction.strip())
    record = {
        "eval_uid": uid,
        "eval_index": sample.metadata.get("eval_index"),
        "source": str(sample.metadata.get("source", "unknown")),
        "split": str(sample.metadata.get("split", "unknown")),
        "query": sample.prompt,
        "references": references,
        "prediction": prediction,
        "f1": f1,
        "accuracy": None,
        "lexical_accuracy": lexical_accuracy,
        "included_in_metrics": included,
        "normal_model_turns": min(tree.model_turns, NORMAL_MAX_STEPS),
        "total_model_turns": tree.model_turns,
        "forced_answer_turn": forced,
        "termination_reason": tree.termination_reason,
        "status": tree.status,
        "output_tokens": tree.output_tokens,
        "wall_seconds": time.monotonic() - started,
        "tree": tree.to_dict(),
    }
    _persist_record(record)
    return _record_to_sample(sample, record)


async def eval_reward(args: Any, sample: Any, **kwargs: Any) -> float:
    """Return the already computed gold-blind-generation QA F1 to slime."""

    del args, kwargs
    return float(sample.metadata["eval_result"]["f1"])


@dataclass(frozen=True)
class DatasetMetrics:
    count: int
    f1: float
    accuracy: float | None
    original_dev_test_size: int


def summarize_records(records: Iterable[dict[str, Any]], weights: dict[str, int]) -> dict[str, Any]:
    all_records = list(records)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in all_records:
        grouped[str(record["source"])].append(record)
    missing = sorted(set(weights) - set(grouped))
    unknown = sorted(set(grouped) - set(weights))
    if missing or unknown:
        raise ValueError(f"dataset mismatch: missing={missing}, unknown={unknown}")

    per_dataset: dict[str, dict[str, Any]] = {}
    for source in sorted(grouped):
        all_values = grouped[source]
        values = [record for record in all_values if included_in_metrics(record)]
        if not values:
            raise ValueError(f"dataset {source} has no completed predictions")
        judged = [record for record in values if record.get("accuracy") is not None]
        per_dataset[source] = {
            "total_count": len(all_values),
            "included_count": len(values),
            "excluded_terminated_count": len(all_values) - len(values),
            "f1": sum(float(item["f1"]) for item in values) / len(values),
            "accuracy": (sum(float(item["accuracy"]) for item in judged) / len(judged) if len(judged) == len(values) else None),
            "original_dev_test_size": int(weights[source]),
        }

    total_weight = sum(weights.values())
    weighted_f1 = sum(weights[source] * per_dataset[source]["f1"] for source in per_dataset) / total_weight
    has_complete_judgments = all(per_dataset[source]["accuracy"] is not None for source in per_dataset)
    weighted_accuracy = sum(weights[source] * per_dataset[source]["accuracy"] for source in per_dataset) / total_weight if has_complete_judgments else None
    records = [record for record in all_records if included_in_metrics(record)]
    count = len(records)
    wall_values = [float(record.get("wall_seconds", 0.0)) for record in records]
    return {
        "total_count": len(all_records),
        "included_count": count,
        "excluded_terminated_count": len(all_records) - count,
        "per_dataset": per_dataset,
        "sample_micro_f1": sum(float(item["f1"]) for item in records) / count,
        "sample_micro_accuracy": (sum(float(item["accuracy"]) for item in records) / count if has_complete_judgments else None),
        "original_size_weighted_f1": weighted_f1,
        "original_size_weighted_accuracy": weighted_accuracy,
        "original_dev_test_total_size": total_weight,
        "forced_answer_count": sum(bool(item.get("forced_answer_turn")) for item in records),
        "mean_sample_wall_seconds": sum(wall_values) / len(wall_values),
        "max_sample_wall_seconds": max(wall_values),
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def log_eval_results(rollout_id: int, args: Any, data: dict[str, Any], extra_metrics: dict[str, Any] | None) -> bool:
    """Write canonical prediction JSONL and the requested weighted summary."""

    del rollout_id, extra_metrics
    records = [sample.metadata["eval_result"] for info in data.values() for sample in info["samples"]]
    records.sort(key=lambda record: (record["source"], record.get("eval_index", -1)))
    output_dir = Path(os.environ["REFLECT_EVAL_OUTPUT_DIR"])
    weights = original_dataset_weights(os.environ["REFLECT_EVAL_DATA_ROOT"])
    summary = summarize_records(records, weights)
    summary.update(
        {
            "checkpoint": str(args.load),
            "checkpoint_iteration": 99,
            "normal_max_steps": NORMAL_MAX_STEPS,
            "forced_answer_step": NORMAL_MAX_STEPS + 1,
            "temperature": float(args.rollout_temperature),
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = output_dir / "predictions.jsonl"
    temporary = predictions.with_name(f".{predictions.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, predictions)
    _atomic_json(output_dir / "generation_summary.json", summary)
    logger.info("Reflect generation summary (LLM judging pending): %s", summary)
    return True


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and summarize Reflect step100 evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--total-samples", type=int, required=True)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--predictions", type=Path, required=True)
    summarize.add_argument("--data-root", type=Path, required=True)
    summarize.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.command == "prepare":
        counts = prepare_eval_parquet(args.input, args.output, args.total_samples)
        print(json.dumps({"output": str(args.output), "counts": counts}, sort_keys=True))
        return
    result = summarize_records(_read_jsonl(args.predictions), original_dataset_weights(args.data_root))
    if args.output:
        _atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
