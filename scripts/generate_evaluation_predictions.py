"""Generate restart-safe predictions with the production StackPlanner graph.

Every attempt is appended and fsynced. A restart reuses successful records and
retries only missing samples. The strict 630-row predictions file is published
atomically only after every manifest ID has a non-empty answer.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Prefer the source tree that owns this entrypoint over an editable slime install
# left by a different clone or worktree. Evaluation metadata already records the
# owning Git commit; this makes the imported implementation match that identity.
ROOT = Path(__file__).resolve().parents[1]
SLIME_SOURCE = str(ROOT / "slime")
sys.path.insert(0, SLIME_SOURCE)

from slime_plugins.stackplanner.config import TrainingMode  # noqa: E402
from slime_plugins.stackplanner.cli import (  # noqa: E402
    _search_r1_service_identity,
)
from slime_plugins.stackplanner.model_factory import (  # noqa: E402
    make_openai_model_builder,
)
from slime_plugins.stackplanner.routing import Backend, RoleRouter  # noqa: E402

sys.path.remove(SLIME_SOURCE)

GRAPH_SUPERSTEPS_PER_LOOP = 16
MINIMUM_GRAPH_RECURSION_LIMIT = 256
SEARCH_RESULT_POLICY = "fail-retryable-search-errors-v1"

EXPECTED_DATASETS = {
    "2WikiMultiHopQA",
    "HotpotQA",
    "Bamboogle",
    "FRAMES",
    "MuSiQue",
    "NQ",
    "TriviaQA",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluation_config_digests(
    training_config: Path, inference_config: Path
) -> dict[str, str]:
    """Hash the immutable training and inference configuration payloads."""

    missing = [
        str(path) for path in (training_config, inference_config) if not path.is_file()
    ]
    if missing:
        raise ValueError("evaluation config file is missing: " + ", ".join(missing))
    return {
        "training_config_sha256": _sha256(training_config),
        "inference_config_sha256": _sha256(inference_config),
    }


def checkpoint_digest(root: Path) -> str:
    """Hash the exact HF safetensors payload using stable relative names."""

    root = root.resolve()
    index = root / "model.safetensors.index.json"
    if index.is_file():
        try:
            payload = json.loads(index.read_text(encoding="utf-8"))
            filenames = sorted(set(payload["weight_map"].values()))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"invalid safetensors index: {index}") from exc
    elif (root / "model.safetensors").is_file():
        filenames = ["model.safetensors"]
    else:
        raise ValueError(f"checkpoint contains no indexed safetensors: {root}")
    lines: list[str] = []
    for filename in filenames:
        if not isinstance(filename, str):
            raise ValueError(f"invalid safetensors filename in {index}")
        candidate = (root / filename).resolve()
        if candidate.parent != root or not candidate.is_file():
            raise ValueError(f"missing or unsafe checkpoint shard: {filename}")
        lines.append(f"{filename}\t{candidate.stat().st_size}\t{_sha256(candidate)}\n")
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def load_manifest(path: Path, *, samples_per_dataset: int = 90) -> list[dict[str, Any]]:
    if samples_per_dataset < 1:
        raise ValueError("samples_per_dataset must be positive")
    rows = _read_jsonl(path)
    counts = Counter(str(row.get("dataset")) for row in rows)
    if (
        len(rows) != len(EXPECTED_DATASETS) * samples_per_dataset
        or set(counts) != EXPECTED_DATASETS
        or any(counts[name] != samples_per_dataset for name in EXPECTED_DATASETS)
    ):
        raise ValueError(
            "evaluation manifest must contain exactly seven datasets x "
            f"{samples_per_dataset}: {counts}"
        )
    seen: set[str] = set()
    for row in rows:
        sample_id = row.get("sample_id")
        question = row.get("question")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("every manifest row needs a non-empty string sample_id")
        if sample_id in seen:
            raise ValueError(f"duplicate manifest sample_id: {sample_id}")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"manifest sample {sample_id} has no question")
        seen.add(sample_id)
    return rows


def successful_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        sample_id = row.get("sample_id")
        prediction = row.get("prediction")
        if (
            row.get("status") == "succeeded"
            and isinstance(sample_id, str)
            and isinstance(prediction, str)
            and prediction.strip()
        ):
            completed[sample_id] = row
    return completed


def _attempt_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if path.is_file():
        for row in _read_jsonl(path):
            sample_id = row.get("sample_id")
            attempt = row.get("attempt")
            if isinstance(sample_id, str) and isinstance(attempt, int):
                counts[sample_id] = max(counts.get(sample_id, 0), attempt)
    return counts


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_resume_identity(
    metadata_path: Path, records_path: Path, identity: Mapping[str, Any]
) -> None:
    """Reject records produced for another manifest, model, or policy."""

    if not records_path.exists():
        return
    if not metadata_path.is_file():
        raise ValueError(
            f"cannot trust existing records without metadata: {records_path}"
        )
    try:
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read resume metadata: {metadata_path}") from exc
    previous_identity = dict(previous)
    previous_search = previous_identity.get("search_r1")
    if isinstance(previous_search, Mapping):
        previous_identity["search_r1"] = {
            key: value for key, value in previous_search.items() if key != "verified_at"
        }
    requested_identity = dict(identity)
    requested_search = requested_identity.get("search_r1")
    if isinstance(requested_search, Mapping):
        requested_identity["search_r1"] = {
            key: value
            for key, value in requested_search.items()
            if key != "verified_at"
        }
    mismatches = {
        key: {"previous": previous_identity.get(key), "requested": value}
        for key, value in requested_identity.items()
        if previous_identity.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "existing prediction records have a different run identity: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def materialize_predictions(
    manifest: list[dict[str, Any]],
    completed: Mapping[str, Mapping[str, Any]],
    output: Path,
) -> None:
    manifest_ids = [str(row["sample_id"]) for row in manifest]
    if set(completed) != set(manifest_ids):
        missing = [
            sample_id for sample_id in manifest_ids if sample_id not in completed
        ]
        raise ValueError(
            f"cannot publish incomplete predictions: missing={missing[:10]}"
        )
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for sample_id in manifest_ids:
            handle.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "prediction": str(completed[sample_id]["prediction"]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)


def _endpoint_models(base_url: str, timeout: float = 30) -> list[str]:
    request = urllib.request.Request(base_url.rstrip("/") + "/models")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"endpoint is not healthy: {base_url}: {exc}") from exc
    try:
        return [str(item["id"]) for item in payload["data"]]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"endpoint returned an invalid model list: {base_url}"
        ) from exc


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text") or block.get("content") or "")
            if isinstance(block, Mapping)
            else str(block)
            for block in content
        )
    return str(content or "")


def _final_answer(state: Mapping[str, Any]) -> str:
    for message in reversed(list(state.get("messages") or [])):
        if getattr(message, "type", None) == "ai":
            answer = _message_text(message).strip()
            if answer:
                return answer
    return ""


def _search_infrastructure_errors(state: Mapping[str, Any]) -> list[str]:
    """Return retryable Search-R1 failures that invalidate a prediction."""

    from langchain_core.messages import ToolMessage

    errors: list[str] = []
    for message in list(state.get("messages") or []):
        if (
            not isinstance(message, ToolMessage)
            or str(message.name or "") != "web_search"
        ):
            continue
        content = _message_text(message).strip()
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, Mapping) or payload.get("retryable") is not True:
            continue
        code = str(payload.get("error") or "WEB_SEARCH_ERROR").strip()
        detail = str(payload.get("message") or "retryable search failure").strip()
        errors.append(f"{code}: {detail}")
    return errors


class InferenceModelFactory:
    """Apply a formal arm role route without opening training sessions."""

    def __init__(
        self,
        *,
        mode: TrainingMode,
        policy_roles: set[str],
        actor_builder: Any,
        frozen_builder: Any,
    ) -> None:
        self.router = RoleRouter(mode, policy_roles)
        self.actor_builder = actor_builder
        self.frozen_builder = frozen_builder

    def __call__(self, *, role: str, **kwargs: Any) -> Any:
        route = self.router.route(role)
        builder = (
            self.actor_builder
            if route.backend is Backend.ACTOR
            else self.frozen_builder
        )
        try:
            requested_model = str(kwargs.pop("model_name"))
        except KeyError as exc:
            raise ValueError("StackPlanner model factory requires model_name") from exc
        return builder(
            role=role,
            session_id=None,
            requested_model=requested_model,
            **kwargs,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class RunSettings:
    actor_base_url: str
    actor_model: str
    frozen_base_url: str
    frozen_model: str
    mode: TrainingMode
    deer_flow_config: Path
    max_response_tokens: int
    max_iterations: int
    timeout: float
    inference_profile: str = "stackplanner"


async def _predict(row: Mapping[str, Any], settings: RunSettings) -> str:
    from deerflow.config.app_config import AppConfig
    from deerflow.sp.runtime import (
        SP_SUBAGENT_REGISTRY_NAMES,
        make_sp_agent,
        make_sp_search_agent,
    )
    from langchain_core.messages import HumanMessage

    sample_id = str(row["sample_id"])
    identity = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:32]
    episode_id = f"eval-{identity}"
    thread_id = f"thread-{identity}"
    run_id = f"run-{identity}"
    model_kwargs = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": settings.max_response_tokens,
        "timeout": settings.timeout,
        "max_retries": 0,
        "streaming": False,
    }
    actor_builder = make_openai_model_builder(
        base_url=settings.actor_base_url,
        model=settings.actor_model,
        model_kwargs=model_kwargs,
        recover_qwen_tool_calls=True,
    )
    frozen_builder = make_openai_model_builder(
        base_url=settings.frozen_base_url,
        model=settings.frozen_model,
        model_kwargs=model_kwargs,
        recover_qwen_tool_calls=True,
    )
    factory = InferenceModelFactory(
        mode=settings.mode,
        policy_roles=set(SP_SUBAGENT_REGISTRY_NAMES),
        actor_builder=actor_builder,
        frozen_builder=frozen_builder,
    )
    app_config = AppConfig.from_file(settings.deer_flow_config)
    if not app_config.models:
        raise RuntimeError("DeerFlow config has no model definition")
    context = {
        "episode_id": episode_id,
        "thread_id": thread_id,
        "run_id": run_id,
    }
    runtime_config = {
        "configurable": {
            "model_name": app_config.models[0].name,
            **context,
            "max_concurrent_subagents": 1,
            "sp_max_loop_iterations": settings.max_iterations,
            "sp_training_model_factory": factory,
        },
        "context": context,
    }
    graph_factory = (
        make_sp_search_agent
        if settings.inference_profile == "search-direct-v1"
        else make_sp_agent
    )
    graph = graph_factory(runtime_config, app_config=app_config)
    state = await asyncio.wait_for(
        graph.ainvoke(
            {
                "messages": [
                    HumanMessage(content=str(row["question"]), id=f"user-{identity}")
                ]
            },
            config={
                "configurable": {"thread_id": thread_id},
                # A single native StackPlanner decision can traverse more than
                # eight LangGraph nodes (model, validation, memory, tool and
                # routing middleware). Keep this generic graph guard above the
                # explicit native loop cap so the latter can return a final
                # policy answer instead of leaking GraphRecursionError.
                "recursion_limit": max(
                    MINIMUM_GRAPH_RECURSION_LIMIT,
                    settings.max_iterations * GRAPH_SUPERSTEPS_PER_LOOP,
                ),
            },
            context=context,
        ),
        timeout=settings.timeout,
    )
    search_errors = _search_infrastructure_errors(state)
    if search_errors:
        raise RuntimeError(
            "retryable Search-R1 infrastructure failure: " + "; ".join(search_errors)
        )
    prediction = _final_answer(state)
    if not prediction:
        raise RuntimeError("StackPlanner graph returned no final answer")
    return prediction


async def _run(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.resolve()
    training_config_path = args.training_config.resolve()
    inference_config_path = args.deer_flow_config.resolve()
    manifest = load_manifest(
        manifest_path, samples_per_dataset=args.samples_per_dataset
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "records.jsonl"
    predictions_path = args.output_dir / "predictions.jsonl"
    metadata_path = args.output_dir / "metadata.json"
    actor_models = await asyncio.to_thread(_endpoint_models, args.actor_base_url)
    frozen_models = await asyncio.to_thread(_endpoint_models, args.frozen_base_url)
    if args.actor_model not in actor_models:
        raise RuntimeError(
            f"actor endpoint does not serve {args.actor_model!r}: {actor_models}"
        )
    if args.frozen_model not in frozen_models:
        raise RuntimeError(
            f"frozen endpoint does not serve {args.frozen_model!r}: {frozen_models}"
        )
    started_at = time.time()
    manifest_sha256 = _sha256(manifest_path)
    config_digests = await asyncio.to_thread(
        evaluation_config_digests,
        training_config_path,
        inference_config_path,
    )
    search_r1 = await asyncio.to_thread(
        _search_r1_service_identity,
        SimpleNamespace(deer_flow_config_path=str(inference_config_path)),
    )
    search_r1_run_identity = {
        key: value for key, value in search_r1.items() if key != "verified_at"
    }
    actual_checkpoint_digest = await asyncio.to_thread(
        checkpoint_digest, args.checkpoint
    )
    if actual_checkpoint_digest != args.checkpoint_digest:
        raise ValueError(
            "checkpoint digest mismatch: "
            f"expected {args.checkpoint_digest}, got {actual_checkpoint_digest}"
        )
    git_commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    run_identity = {
        "actor_model": args.actor_model,
        "checkpoint_digest": actual_checkpoint_digest,
        **config_digests,
        "frozen_model": args.frozen_model,
        "git_commit": git_commit,
        "manifest_sha256": manifest_sha256,
        "mode": args.mode,
        "inference_profile": args.inference_profile,
        "search_result_policy": SEARCH_RESULT_POLICY,
        "samples_per_dataset": args.samples_per_dataset,
        "search_r1": search_r1_run_identity,
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_response_tokens": args.max_response_tokens,
            "max_iterations": args.max_iterations,
        },
    }
    validate_resume_identity(metadata_path, records_path, run_identity)
    completed = successful_records(records_path)
    attempt_counts = _attempt_counts(records_path)
    manifest_ids = {str(row["sample_id"]) for row in manifest}
    completed = {key: value for key, value in completed.items() if key in manifest_ids}
    settings = RunSettings(
        actor_base_url=args.actor_base_url,
        actor_model=args.actor_model,
        frozen_base_url=args.frozen_base_url,
        frozen_model=args.frozen_model,
        mode=TrainingMode(args.mode),
        deer_flow_config=inference_config_path,
        max_response_tokens=args.max_response_tokens,
        max_iterations=args.max_iterations,
        timeout=args.timeout,
        inference_profile=args.inference_profile,
    )
    metadata: dict[str, Any] = {
        "actor_base_url": args.actor_base_url,
        "actor_model": args.actor_model,
        "actor_models_reported": actor_models,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_digest": actual_checkpoint_digest,
        "completed_samples": len(completed),
        **config_digests,
        "deer_flow_config": str(settings.deer_flow_config),
        "frozen_base_url": args.frozen_base_url,
        "frozen_model": args.frozen_model,
        "frozen_models_reported": frozen_models,
        "git_commit": git_commit,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "mode": settings.mode.value,
        "inference_profile": settings.inference_profile,
        "search_result_policy": SEARCH_RESULT_POLICY,
        "samples_per_dataset": args.samples_per_dataset,
        "search_r1": search_r1_run_identity,
        "search_r1_verification": search_r1,
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_response_tokens": args.max_response_tokens,
            "max_iterations": args.max_iterations,
        },
        "started_at_unix": started_at,
        "status": "running",
        "target_samples": len(manifest),
        "training_config": str(training_config_path),
    }
    _atomic_json(metadata_path, metadata)

    append_lock = asyncio.Lock()
    completed_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(args.concurrency)
    queue = [row for row in manifest if str(row["sample_id"]) not in completed]
    if args.max_new:
        queue = queue[: args.max_new]

    async def append_record(record: Mapping[str, Any]) -> None:
        rendered = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        async with append_lock:
            with records_path.open("a", encoding="utf-8") as handle:
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())

    async def worker(row: Mapping[str, Any]) -> None:
        sample_id = str(row["sample_id"])
        async with semaphore:
            for local_attempt in range(1, args.max_attempts + 1):
                attempt = attempt_counts.get(sample_id, 0) + local_attempt
                attempt_started = time.time()
                try:
                    prediction = await _predict(row, settings)
                    record = {
                        "attempt": attempt,
                        "elapsed_seconds": time.time() - attempt_started,
                        "prediction": prediction,
                        "sample_id": sample_id,
                        "status": "succeeded",
                        "time_unix": time.time(),
                    }
                    await append_record(record)
                    async with completed_lock:
                        completed[sample_id] = record
                    return
                except Exception as exc:
                    await append_record(
                        {
                            "attempt": attempt,
                            "elapsed_seconds": time.time() - attempt_started,
                            "error": f"{type(exc).__name__}: {exc}",
                            "sample_id": sample_id,
                            "status": "failed",
                            "time_unix": time.time(),
                        }
                    )
                    if local_attempt < args.max_attempts:
                        await asyncio.sleep(min(30.0, 2.0**local_attempt))

    await asyncio.gather(*(worker(row) for row in queue))
    metadata["completed_samples"] = len(completed)
    metadata["elapsed_seconds"] = time.time() - started_at
    if len(completed) == len(manifest):
        materialize_predictions(manifest, completed, predictions_path)
        metadata["predictions_sha256"] = _sha256(predictions_path)
        metadata["status"] = "complete"
        metadata["finished_at_unix"] = time.time()
        _atomic_json(metadata_path, metadata)
        return 0
    metadata["status"] = "partial"
    metadata["missing_samples"] = [
        str(row["sample_id"])
        for row in manifest
        if str(row["sample_id"]) not in completed
    ]
    _atomic_json(metadata_path, metadata)
    return 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-digest", required=True)
    parser.add_argument(
        "--training-config",
        type=Path,
        required=True,
        help="immutable resolved_config.yaml (or immutable base source config)",
    )
    parser.add_argument("--actor-base-url", required=True)
    parser.add_argument("--actor-model", required=True)
    parser.add_argument("--frozen-base-url", required=True)
    parser.add_argument("--frozen-model", required=True)
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in TrainingMode],
        required=True,
    )
    parser.add_argument(
        "--inference-profile",
        choices=("stackplanner", "search-direct-v1"),
        default="stackplanner",
        help="production SP graph or bounded evidence-first Search-R1 profile",
    )
    parser.add_argument(
        "--deer-flow-config",
        type=Path,
        default=ROOT / "configs/deerflow_search_r1_only.yaml",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-new", type=int, default=0)
    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=90,
        help="expected balanced manifest size; formal evaluation keeps the default 90",
    )
    parser.add_argument("--max-response-tokens", type=int, default=1024)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if (
        args.concurrency < 1
        or args.max_attempts < 1
        or args.max_new < 0
        or args.samples_per_dataset < 1
    ):
        parser.error(
            "concurrency/max-attempts/samples-per-dataset must be positive "
            "and max-new nonnegative"
        )
    if not args.checkpoint.exists():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if not args.training_config.is_file():
        parser.error(f"training config does not exist: {args.training_config}")
    if not args.deer_flow_config.is_file():
        parser.error(f"inference config does not exist: {args.deer_flow_config}")
    if len(args.checkpoint_digest) != 64 or any(
        char not in "0123456789abcdef" for char in args.checkpoint_digest
    ):
        parser.error("checkpoint-digest must be a lowercase SHA-256")
    os.environ["DEER_FLOW_CONFIG_PATH"] = str(args.deer_flow_config.resolve())
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
