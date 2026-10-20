"""Top-level command line and preflight for StackPlanner/slime training."""

from __future__ import annotations

import argparse
import ast
import dataclasses
import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any
from collections.abc import Sequence

from .config import DecouplingScope, Experiment, TrainingConfig, TrainingMode
from .routing import Backend, RoleRouter

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SENSITIVE_OPTION_MARKERS = ("api-key", "password", "secret", "credential", "auth-token")
_SEARCH_R1_PROVIDER = "deerflow.community.search_r1.tools:web_search_tool"


class PreflightError(RuntimeError):
    """Raised when a run would be unsafe or cannot start as configured."""


@dataclasses.dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


@dataclasses.dataclass(frozen=True, slots=True)
class LaunchPlan:
    config: TrainingConfig
    policy_roles: tuple[str, ...]
    route_table: tuple[dict[str, Any], ...]
    command: tuple[str, ...]
    environment: dict[str, str]
    checks: tuple[CheckResult, ...]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (_REPO_ROOT / path).resolve()


def _read_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    # JSON is a YAML subset and gives the launcher a dependency-free config
    # format even when the filename uses the documented .yaml suffix.
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(f"{path} is not JSON; install PyYAML to read YAML configuration") from exc
        raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError(f"configuration root in {path} must be an object")
    return raw


def _json_object(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train StackPlanner 2.0 through a slime-managed SGLang actor.",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", help="JSON or YAML configuration; explicit CLI options win")
    parser.add_argument("--print-command", action="store_true", help="print the resolved slime command")

    groups: dict[str, argparse._ArgumentGroup] = {
        "mode": parser.add_argument_group("training mode and routing"),
        "paths": parser.add_argument_group("repositories and run paths"),
        "models": parser.add_argument_group("models"),
        "data": parser.add_argument_group("data"),
        "train": parser.add_argument_group("training"),
        "rollout": parser.add_argument_group("rollout"),
        "stackplanner": parser.add_argument_group("StackPlanner episode limits"),
        "services": parser.add_argument_group("adapter, SGLang, and Ray"),
        "hardware": parser.add_argument_group("distributed hardware"),
        "reward": parser.add_argument_group("reward and verifier"),
        "debug": parser.add_argument_group("debug, replay, and resume"),
    }

    def add(group: str, name: str, **kwargs: Any) -> None:
        kwargs.setdefault("default", argparse.SUPPRESS)
        groups[group].add_argument(name, **kwargs)

    add("mode", "--mode", choices=[mode.value for mode in TrainingMode])
    add("mode", "--experiment", choices=[item.value for item in Experiment])
    add("mode", "--capability-decoupling", action=argparse.BooleanOptionalAction)
    add(
        "mode",
        "--decoupling-scope",
        choices=[scope.value for scope in DecouplingScope],
    )
    add("mode", "--role-overrides", type=_json_object, metavar="JSON")
    add("paths", "--run-id")
    add("paths", "--stackplanner-path")
    add("paths", "--slime-path")
    add("paths", "--megatron-path")
    add("paths", "--deer-flow-config-path")
    add("paths", "--output-root")
    add("models", "--hf-checkpoint")
    add("models", "--reference-checkpoint")
    add("models", "--base-checkpoint-sha256")
    add("models", "--tokenizer")
    add("models", "--frozen-model")
    add("models", "--frozen-base-url")
    add("models", "--actor-model-name")
    add("models", "--actor-base-url")
    add("rollout", "--custom-generate-function")
    add("reward", "--reward-post-process-function")
    add("data", "--data-path")
    add("data", "--input-key")
    add("data", "--label-key")
    add("data", "--data-start", type=int)
    add("data", "--data-limit", type=int)
    add("data", "--seed", type=int)
    add("train", "--train-steps", type=int)
    add("train", "--algorithm", choices=["grpo", "gspo", "cispo", "ppo"])
    add("train", "--learning-rate", type=float)
    add("train", "--kl-coefficient", type=float)
    add("train", "--clip-epsilon", type=float)
    add("train", "--optimizer")
    add("train", "--save-interval", type=int)
    add("train", "--eval-interval", type=int)
    add("train", "--ole-enabled", action=argparse.BooleanOptionalAction)
    add("train", "--ole-threshold-percent", type=float)
    add("train", "--ole-k", type=float)
    add("train", "--ole-init-threshold", type=float)
    add("rollout", "--samples-per-prompt", type=int)
    add("rollout", "--rollout-batch-size", type=int)
    add("rollout", "--temperature", type=float)
    add("rollout", "--top-p", type=float)
    add("rollout", "--max-context-tokens", type=int)
    add("rollout", "--max-response-tokens", type=int)
    add("rollout", "--episode-timeout-seconds", type=int)
    add("stackplanner", "--max-iterations", type=int)
    add("stackplanner", "--max-parallel-delegates", type=int)
    add("stackplanner", "--subagent-max-turns", type=int)
    add("stackplanner", "--subagent-timeout-seconds", type=int)
    add("services", "--adapter-host")
    add("services", "--adapter-port", type=int)
    add("services", "--sglang-router-host")
    add("services", "--sglang-router-port", type=int)
    add("services", "--sglang-engine-port-base", type=int)
    add("services", "--ray-address")
    add("hardware", "--num-nodes", type=int)
    add("hardware", "--gpus-per-node", type=int)
    add("hardware", "--tensor-parallel-size", type=int)
    add("hardware", "--pipeline-parallel-size", type=int)
    add("hardware", "--context-parallel-size", type=int)
    add("hardware", "--data-parallel-size", type=int)
    add("hardware", "--expert-parallel-size", type=int)
    add("hardware", "--colocate", action=argparse.BooleanOptionalAction)
    add("hardware", "--offload", action=argparse.BooleanOptionalAction)
    add("hardware", "--fully-async", action=argparse.BooleanOptionalAction)
    add("reward", "--reward-function")
    add("reward", "--reward-config", type=_json_object, metavar="JSON")
    add("reward", "--infrastructure-failure-policy", choices=["remove", "retry", "fail"])
    add("debug", "--dry-run", action=argparse.BooleanOptionalAction)
    add("debug", "--rollout-only", action=argparse.BooleanOptionalAction)
    add("debug", "--train-only", action=argparse.BooleanOptionalAction)
    add("debug", "--replay-path")
    add("debug", "--resume-from")
    add("debug", "--dump-trajectories", action=argparse.BooleanOptionalAction)
    add("debug", "--log-level")
    add("debug", "--capability-taxonomy-path")
    add("debug", "--capability-probe-manifest")
    add("debug", "--capability-layer-map")
    add("debug", "--capability-annotator-function")
    add("debug", "--capability-loss-function")
    add("debug", "--capability-gradient-router-function")
    add("debug", "--capability-probe-bootstrap", action=argparse.BooleanOptionalAction)
    add(
        "debug",
        "--extra-slime-arg",
        action="append",
        dest="extra_slime_args",
        help="one literal slime argument; repeat for flags and values",
    )
    return parser


def resolve_config(argv: Sequence[str] | None = None) -> tuple[TrainingConfig, list[str], bool]:
    parser = build_parser()
    namespace, forwarded = parser.parse_known_args(argv)
    explicit = vars(namespace)
    config_path = explicit.pop("config", None)
    print_command = bool(explicit.pop("print_command", False))
    base = _read_mapping(_path(config_path)) if config_path else {}
    # A repeated --extra-slime-arg is an explicit replacement. Unknown options
    # are appended and passed to slime, enabling its full native CLI surface.
    explicit_extra = explicit.pop("extra_slime_args", None)
    merged = {**base, **explicit}
    extra = list(merged.get("extra_slime_args", []))
    if explicit_extra is not None:
        extra = list(explicit_extra)
    extra.extend(forwarded)
    merged["extra_slime_args"] = extra
    config = TrainingConfig.from_mapping(merged)
    return config, forwarded, print_command


def _discover_policy_roles(stackplanner: Path) -> tuple[str, ...]:
    runtime = stackplanner / "backend/packages/harness/deerflow/sp/runtime.py"
    tree = ast.parse(runtime.read_text(encoding="utf-8"), filename=str(runtime))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "SP_SUBAGENT_REGISTRY_NAMES" for target in targets):
                value = ast.literal_eval(node.value)
                if isinstance(value, dict) and value:
                    return tuple(sorted(str(role) for role in value))
    raise ValueError(f"SP_SUBAGENT_REGISTRY_NAMES not found in {runtime}")


def _path_check(name: str, value: str, *, directory: bool = False) -> CheckResult:
    if not value:
        return CheckResult(name, False, "not configured")
    path = _path(value)
    ok = path.is_dir() if directory else path.exists()
    return CheckResult(name, ok, str(path) if ok else f"not found: {path}")


def _module_check(name: str, module: str, search_root: Path) -> CheckResult:
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(search_root))
        found = importlib.util.find_spec(module) is not None
    except (ImportError, AttributeError, ValueError) as exc:
        return CheckResult(name, False, str(exc))
    finally:
        sys.path[:] = old_path
    return CheckResult(name, found, f"{module} import spec {'found' if found else 'missing'}")


def _function_check(name: str, dotted_path: str, *search_roots: Path) -> CheckResult:
    normalized = dotted_path.replace(":", ".")
    if not normalized or "." not in normalized:
        return CheckResult(name, False, f"invalid import path: {dotted_path!r}")
    module_name, _, attribute = normalized.rpartition(".")
    old_path = list(sys.path)
    try:
        sys.path[:0] = [str(root) for root in search_roots]
        module = __import__(module_name, fromlist=[attribute])
        function = getattr(module, attribute)
    except (ImportError, AttributeError, RuntimeError) as exc:
        return CheckResult(name, False, str(exc))
    finally:
        sys.path[:] = old_path
    return CheckResult(name, callable(function), f"{dotted_path} is {'callable' if callable(function) else 'not callable'}")


def _data_check(config: TrainingConfig) -> CheckResult:
    if not config.data_path:
        return CheckResult("training data", False, "data_path is not configured")
    path = _path(config.data_path)
    if not path.is_file():
        return CheckResult("training data", False, f"not found: {path}")
    try:
        row = next((json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()), None)
    except (OSError, json.JSONDecodeError) as exc:
        return CheckResult("training data", False, str(exc))
    missing = [key for key in (config.input_key, config.label_key) if key and (not isinstance(row, dict) or key not in row)]
    return CheckResult("training data", not missing, f"{path}; missing fields: {missing}" if missing else str(path))


def _search_r1_identity(config: TrainingConfig) -> dict[str, Any]:
    """Resolve the only retrieval backend accepted by training/evaluation."""

    path = _path(config.deer_flow_config_path)
    raw = _read_mapping(path)
    tools = raw.get("tools")
    if not isinstance(tools, list):
        raise ValueError("DeerFlow config must contain a tools list")
    web_tools = [tool for tool in tools if isinstance(tool, dict) and tool.get("name") == "web_search"]
    if len(web_tools) != 1 or len(tools) != 1:
        raise ValueError("training requires exactly one web_search tool and no fallback retrieval providers")
    tool = web_tools[0]
    if tool.get("use") != _SEARCH_R1_PROVIDER:
        raise ValueError(f"training requires Search-R1 provider {_SEARCH_R1_PROVIDER!r}")
    endpoint = str(tool.get("base_url") or "").rstrip("/")
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("training Search-R1 endpoint must be an HTTP(S) URL")
    digests: dict[str, str | None] = {}
    for name in ("index_sha256", "corpus_sha256"):
        digest = tool.get(name)
        if digest in (None, ""):
            digests[name] = None
        elif isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest):
            digests[name] = digest
        else:
            raise ValueError(f"training Search-R1 config has an invalid {name}")
    return {
        "config_path": str(path),
        "provider": _SEARCH_R1_PROVIDER,
        "endpoint": endpoint,
        **digests,
    }


def _search_r1_config_check(config: TrainingConfig) -> CheckResult:
    try:
        identity = _search_r1_identity(config)
    except (OSError, RuntimeError, ValueError) as exc:
        return CheckResult("Search-R1 configuration", False, str(exc))
    return CheckResult(
        "Search-R1 configuration",
        True,
        f"provider={identity['provider']}; endpoint={identity['endpoint']}; index_sha256={identity['index_sha256'] or 'not pinned'}; corpus_sha256={identity['corpus_sha256'] or 'not pinned'}",
    )


def _search_r1_service_identity(config: TrainingConfig) -> dict[str, Any]:
    identity = _search_r1_identity(config)
    request_body = json.dumps(
        {
            "queries": ["StackPlanner Search-R1 preflight Ada Lovelace"],
            "topk": 1,
            "return_scores": True,
        }
    ).encode()
    request = urllib.request.Request(
        f"{identity['endpoint']}/retrieve",
        data=request_body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, urllib.error.URLError) as exc:
        raise RuntimeError(f"Search-R1 live retrieval failed: {exc}") from exc
    batches = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(batches, list) or len(batches) != 1 or not isinstance(batches[0], list) or not batches[0] or not isinstance(batches[0][0], dict) or not isinstance(batches[0][0].get("document"), dict):
        raise RuntimeError("Search-R1 live retrieval returned an invalid or empty result batch")
    return {
        **identity,
        "verified_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "health_query": "StackPlanner Search-R1 preflight Ada Lovelace",
        "health_result_count": len(batches[0]),
    }


def _model_check(name: str, value: str, *, require_hf_config: bool = True) -> CheckResult:
    if not value:
        return CheckResult(name, False, "not configured")
    path = _path(value)
    if path.exists():
        required = path / "config.json" if path.is_dir() and require_hf_config else path
        ok = required.exists()
        return CheckResult(name, ok, str(path) if ok else f"config.json missing in {path}")
    # A Hub identifier is configurable but is not proof that weights are locally
    # accessible. Fail before reserving GPUs and tell the operator what to stage.
    return CheckResult(name, False, f"local checkpoint not found: {path} (stage Hub id {value!r} first)")


def _tokenizer_check(config: TrainingConfig) -> CheckResult:
    value = config.tokenizer or config.hf_checkpoint
    if not value:
        return CheckResult("tokenizer/chat template", False, "tokenizer is not configured")
    path = _path(value)
    if not path.is_dir():
        return CheckResult("tokenizer/chat template", False, f"local tokenizer not found: {path}")
    tokenizer_files = [path / "tokenizer.json", path / "tokenizer_config.json", path / "tokenizer.model"]
    if not any(candidate.exists() for candidate in tokenizer_files):
        return CheckResult("tokenizer/chat template", False, f"tokenizer files missing in {path}")
    template = path / "chat_template.jinja"
    if template.exists():
        return CheckResult("tokenizer/chat template", True, str(template))
    config_file = path / "tokenizer_config.json"
    try:
        raw = json.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raw = {}
    ok = bool(raw.get("chat_template"))
    return CheckResult("tokenizer/chat template", ok, str(config_file) if ok else "chat template missing")


def _port_check(name: str, host: str, port: int) -> CheckResult:
    if port == 0:
        return CheckResult(name, True, f"{host}:ephemeral (allocated per rollout worker)")
    if not (1 <= port <= 65535):
        return CheckResult(name, False, f"invalid port: {port}")
    bind_host = host if host not in {"0.0.0.0", "::"} else "0.0.0.0"
    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        sock.bind((bind_host, port))
    except OSError as exc:
        return CheckResult(name, False, f"{host}:{port} unavailable: {exc}")
    finally:
        sock.close()
    return CheckResult(name, True, f"{host}:{port} available")


def _gpu_check(config: TrainingConfig) -> CheckResult:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return CheckResult("GPU topology", False, "nvidia-smi is unavailable")
    command = [
        executable,
        "--query-gpu=index,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10).stdout
        rows = [line.strip() for line in output.splitlines() if line.strip()]
        local_required = config.gpus_per_node
        valid = [row for row in rows if int(row.rsplit(",", 1)[1].strip()) >= 40000]
        ok = len(valid) >= local_required
        detail = f"visible={len(rows)}, NVIDIA>=40GiB={len(valid)}, required_on_this_node={local_required}"
        if config.num_nodes > 1:
            detail += f"; remaining {config.num_nodes - 1} node(s) must be registered with Ray"
        return CheckResult("GPU topology", ok, detail)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return CheckResult("GPU topology", False, str(exc))


def _output_check(config: TrainingConfig) -> CheckResult:
    run_dir = config.run_dir
    if run_dir.exists():
        return CheckResult("run output", False, f"refusing to overwrite existing run: {run_dir}")
    parent = run_dir.parent
    probe = parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    ok = probe.is_dir() and os.access(probe, os.W_OK | os.X_OK)
    return CheckResult("run output", ok, f"{run_dir}; writable ancestor={probe}")


def _topology_check(config: TrainingConfig) -> CheckResult:
    total = config.num_nodes * config.gpus_per_node
    model_parallel = config.tensor_parallel_size * config.pipeline_parallel_size * config.context_parallel_size
    divisible = model_parallel > 0 and total % model_parallel == 0
    expected_dp = total // model_parallel if divisible else None
    sizes_valid = all(1 <= size <= total for size in (model_parallel, config.data_parallel_size, config.expert_parallel_size))
    rollout_tp_valid = config.gpus_per_node % config.tensor_parallel_size == 0
    detail = f"total={total}, TPxPPxCP={model_parallel}, configured_DP={config.data_parallel_size}, expected_DP={expected_dp}, EP={config.expert_parallel_size}"
    return CheckResult(
        "parallel topology",
        sizes_valid and divisible and config.data_parallel_size == expected_dp and rollout_tp_valid,
        detail,
    )


def _routing_checks(config: TrainingConfig, router: RoleRouter) -> list[CheckResult]:
    routes = router.table()
    central = next(route for route in routes if route["role"] == "central")
    central_ok = central["backend"] is Backend.ACTOR and central["trainable"] is True
    checks = [CheckResult("Central route", central_ok, str(central))]
    subagents = [route for route in routes if route["role"] != "central"]
    if config.mode is TrainingMode.CENTRAL_ONLY:
        expected = all(route["backend"] is Backend.FROZEN and route["trainable"] is False for route in subagents)
        complete = bool(config.frozen_model and config.frozen_base_url)
        checks.append(CheckResult("Central-only subagent isolation", expected and complete, f"routes={subagents}; frozen backend configured={complete}"))
    else:
        expected = all(route["backend"] is Backend.ACTOR and route["trainable"] is True for route in subagents)
        configured_actor = config.actor_base_url.rstrip("/")
        expected_actor = f"http://{config.adapter_host}:{config.adapter_port}/v1"
        # With port zero each rollout worker allocates a collision-free local
        # adapter and EpisodeModelFactory rewrites actor calls to its actual
        # port. The static actor_base_url is then only documentary.
        same_actor = config.adapter_port == 0 or configured_actor == expected_actor
        checks.append(CheckResult("shared actor invariant", expected and same_actor, f"routes={subagents}; per-worker ephemeral adapter={config.adapter_port == 0}; static actor endpoint matches={configured_actor == expected_actor}"))
    return checks


def build_slime_command(config: TrainingConfig) -> tuple[str, ...]:
    slime = _path(config.slime_path)
    entry = slime / ("train_async.py" if config.fully_async else "train.py")
    run_dir = config.run_dir
    start_rollout_id = _checkpoint_next_rollout_id(_path(config.resume_from)) if config.resume_from else 0
    prompt_data = str(_path(config.data_path))
    if config.data_start or config.data_limit:
        end = config.data_start + config.data_limit if config.data_limit else ""
        prompt_data = f"{prompt_data}@[{config.data_start}:{end}]"
    # Group-relative estimators apply reference KL in the policy loss;
    # PPO consumes its KL coefficient when shaping token rewards.
    uses_kl_loss = config.algorithm in {"grpo", "gspo", "cispo"}
    args = [
        sys.executable,
        str(entry),
        "--actor-num-nodes",
        str(config.num_nodes),
        "--actor-num-gpus-per-node",
        str(config.gpus_per_node),
        "--num-gpus-per-node",
        str(config.gpus_per_node),
        "--rollout-num-gpus-per-engine",
        str(config.tensor_parallel_size),
        "--sglang-engine-port-base",
        str(config.sglang_engine_port_base),
        "--hf-checkpoint",
        str(_path(config.hf_checkpoint)),
        "--ref-load",
        str(_path(config.reference_checkpoint)),
        "--load",
        str(_path(config.resume_from or config.reference_checkpoint)),
        "--start-rollout-id",
        str(start_rollout_id),
        "--save",
        str(run_dir / "checkpoints"),
        "--save-hf",
        str(run_dir / "checkpoints/hf_step_{rollout_id}"),
        "--save-interval",
        str(config.save_interval),
        "--prompt-data",
        prompt_data,
        "--custom-generate-function-path",
        config.custom_generate_function,
        "--custom-reward-post-process-path",
        config.reward_post_process_function,
        "--input-key",
        config.input_key,
        "--label-key",
        config.label_key,
        "--num-rollout",
        str(start_rollout_id + config.train_steps),
        "--rollout-batch-size",
        str(config.rollout_batch_size),
        "--n-samples-per-prompt",
        str(config.samples_per_prompt),
        "--num-steps-per-rollout",
        "1",
        "--global-batch-size",
        str(config.rollout_batch_size * config.samples_per_prompt),
        "--rollout-max-context-len",
        str(config.max_context_tokens),
        "--rollout-max-response-len",
        str(config.max_response_tokens),
        "--rollout-temperature",
        str(config.temperature),
        "--rollout-top-p",
        str(config.top_p),
        "--rollout-seed",
        str(config.seed),
        "--seed",
        str(config.seed),
        "--advantage-estimator",
        config.algorithm,
        "--lr",
        str(config.learning_rate),
        "--kl-coef",
        str(0.0 if uses_kl_loss else config.kl_coefficient),
        "--eps-clip",
        str(config.clip_epsilon),
        "--optimizer",
        config.optimizer,
        "--tensor-model-parallel-size",
        str(config.tensor_parallel_size),
        "--pipeline-model-parallel-size",
        str(config.pipeline_parallel_size),
        "--context-parallel-size",
        str(config.context_parallel_size),
        "--expert-model-parallel-size",
        str(config.expert_parallel_size),
        "--dump-details",
        str(run_dir / "rollouts"),
        "--save-debug-rollout-data",
        str(run_dir / "rollouts/{rollout_id}.pt"),
        "--use-tensorboard",
    ]
    if uses_kl_loss and config.kl_coefficient != 0:
        args.extend(("--use-kl-loss", "--kl-loss-coef", str(config.kl_coefficient)))
    if config.eval_interval > 0:
        args.extend(("--eval-interval", str(config.eval_interval)))
    if config.ole_enabled:
        args.extend(
            (
                "--rollout-function-path",
                "slime_plugins.stackplanner.ole.generate_rollout",
                "--rollout-data-postprocess-path",
                "slime_plugins.stackplanner.ole.apply_selection_to_advantages",
            )
        )
    if config.capability_decoupling and not config.capability_probe_bootstrap:
        args.extend(
            (
                "--custom-train-sample-transform-path",
                config.capability_loss_function,
                "--custom-megatron-before-train-step-hook-path",
                config.capability_gradient_router_function,
                "--custom-megatron-train-batch-hook-path",
                "slime_plugins.stackplanner.capability.gradient_router.attach_microbatch_capability",
                "--no-gradient-accumulation-fusion",
            )
        )
    if config.colocate:
        args.append("--colocate")
    if config.offload:
        args.append("--offload")
    if config.rollout_only:
        args.append("--debug-rollout-only")
    if config.train_only:
        args.extend(("--debug-train-only", "--load-debug-rollout-data", str(_path(config.replay_path))))
    if config.resume_from:
        # A resumed invocation may request only the remaining updates, so its
        # derived ``num_rollout`` (and freshly constructed scheduler horizon)
        # can differ from the original run. Restore the checkpoint scheduler
        # contract and consumed-step counter rather than resetting either.
        args.append("--use-checkpoint-opt-param-scheduler")
    args.extend(config.extra_slime_args)
    return tuple(args)


def build_plan(config: TrainingConfig) -> LaunchPlan:
    stackplanner = _path(config.stackplanner_path)
    slime = _path(config.slime_path)
    megatron = _path(config.megatron_path)
    checks = [
        _path_check("StackPlanner repository", config.stackplanner_path, directory=True),
        _path_check("slime repository", config.slime_path, directory=True),
        _path_check("Megatron repository", config.megatron_path, directory=True),
        _path_check("StackPlanner runtime config", config.deer_flow_config_path),
        _search_r1_config_check(config),
        _module_check("StackPlanner Python package", "deerflow", stackplanner / "backend/packages/harness"),
        _module_check("slime Python package", "slime", slime),
        _module_check("Megatron training package", "megatron.training", megatron),
        _function_check("StackPlanner custom-generate hook", config.custom_generate_function, slime),
        _function_check("StackPlanner reward function", config.reward_function, slime),
        _function_check("StackPlanner reward post-process hook", config.reward_post_process_function, slime),
        _data_check(config),
        _model_check("actor checkpoint", config.hf_checkpoint),
        _model_check("reference checkpoint", config.reference_checkpoint, require_hf_config=False),
        _tokenizer_check(config),
        _port_check("adapter port", config.adapter_host, config.adapter_port),
        _port_check("SGLang/router port", config.sglang_router_host, config.sglang_router_port),
        _gpu_check(config),
        _topology_check(config),
        _output_check(config),
    ]
    if config.ole_enabled:
        for name in ("generate_rollout", "apply_selection_to_advantages"):
            checks.append(_function_check(f"OLE {name} hook", f"slime_plugins.stackplanner.ole.{name}", slime))
    try:
        roles = _discover_policy_roles(stackplanner)
        role_check = CheckResult("runtime policy registry", True, ", ".join(roles))
    except (OSError, SyntaxError, ValueError) as exc:
        roles = ()
        role_check = CheckResult("runtime policy registry", False, str(exc))
    checks.append(role_check)
    router = RoleRouter(config.mode, roles, config.role_overrides)
    checks.extend(_routing_checks(config, router))
    cuda_library_path = os.pathsep.join(
        (
            str(Path(os.environ.get("CUDA_HOME", sys.prefix)) / "lib"),
            str(Path(os.environ.get("CUDA_HOME", sys.prefix)) / "targets/x86_64-linux/lib"),
            str(Path(os.environ.get("CUDA_HOME", sys.prefix)) / "targets/x86_64-linux/lib/stubs"),
            os.environ.get("LIBRARY_PATH", ""),
        )
    )
    env = {
        "SP_TRAINING_MODE": config.mode.value,
        "SP_TRAINING_RUN_ID": config.run_id,
        "SP_TRAINING_RESOLVED_CONFIG": str(config.run_dir / "resolved_config.yaml"),
        "SP_TRAINING_REWARD_FUNCTION": config.reward_function,
        "SP_TRAINING_REWARD_CONFIG": json.dumps(config.reward_config, sort_keys=True),
        "SP_TRAINING_MAX_ITERATIONS": str(config.max_iterations),
        "SP_TRAINING_MAX_PARALLEL_DELEGATES": str(config.max_parallel_delegates),
        "SP_TRAINING_EPISODE_TIMEOUT_SECONDS": str(config.episode_timeout_seconds),
        "SP_CAPABILITY_LAYER_MAP": str(_path(config.capability_layer_map)) if config.capability_layer_map else "",
        "SP_DECOUPLING_SCOPE": config.decoupling_scope.value,
        "SP_BASE_CHECKPOINT_SHA256": config.base_checkpoint_sha256,
        "SP_METRICS_JSONL": str(config.run_dir / "metrics/metrics.jsonl"),
        "TENSORBOARD_DIR": str(config.run_dir / "metrics/tensorboard"),
        "RAY_ADDRESS": config.ray_address,
        "DEER_FLOW_CONFIG_PATH": str(_path(config.deer_flow_config_path)),
        # FlashInfer JIT invokes the host linker directly. LD_LIBRARY_PATH is
        # runtime-only; LIBRARY_PATH must name conda CUDA's libcudart/libcuda
        # directories for the link step, including inside Ray actors.
        "LIBRARY_PATH": cuda_library_path,
        "PYTHONPATH": os.pathsep.join(
            (
                str(megatron),
                str(_REPO_ROOT),
                str(slime),
                str(stackplanner / "backend/packages/harness"),
                os.environ.get("PYTHONPATH", ""),
            )
        ),
    }
    return LaunchPlan(config, roles, tuple(router.table()), build_slime_command(config), env, tuple(checks))


def _redacted_command(command: Sequence[str]) -> str:
    redacted: list[str] = []
    hide_next = False
    for part in command:
        if hide_next:
            redacted.append("<redacted>")
            hide_next = False
            continue
        redacted.append(part)
        option = part.lstrip("-").lower()
        if part.startswith("--") and any(marker in option for marker in _SENSITIVE_OPTION_MARKERS):
            hide_next = "=" not in part
            if "=" in part:
                redacted[-1] = part.split("=", 1)[0] + "=<redacted>"
    return shlex.join(redacted)


def _print_plan(plan: LaunchPlan) -> None:
    print(f"experiment: {plan.config.experiment.value}")
    print(f"mode: {plan.config.mode.value}")
    print(f"run directory: {plan.config.run_dir}")
    print(f"resources: {plan.config.num_nodes} node(s) x {plan.config.gpus_per_node} GPU(s)")
    print("routes:")
    for route in plan.route_table:
        backend = route["backend"].value if isinstance(route["backend"], Backend) else route["backend"]
        print(f"  {route['role']}: backend={backend}, trainable={route['trainable']}")
    print("preflight:")
    for check in plan.checks:
        print(f"  [{'OK' if check.ok else 'FAIL'}] {check.name}: {check.detail}")
    print("slime command:")
    print(f"  {_redacted_command(plan.command)}")


def _environment_record(plan: LaunchPlan, search_r1_identity: dict[str, Any]) -> dict[str, Any]:
    def git_head(path: Path) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        except subprocess.SubprocessError:
            return None

    return {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": sys.version,
        "platform": sys.platform,
        "repository_head": git_head(_REPO_ROOT),
        "stackplanner_tree": git_head(_path(plan.config.stackplanner_path)),
        "slime_tree": git_head(_path(plan.config.slime_path)),
        "megatron_tree": git_head(_path(plan.config.megatron_path)),
        "container_image": os.environ.get("CONTAINER_IMAGE_NAME"),
        "policy_roles": list(plan.policy_roles),
        "route_table": [{**route, "backend": route["backend"].value if isinstance(route["backend"], Backend) else route["backend"]} for route in plan.route_table],
        "search_r1": search_r1_identity,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_identity(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    if path.is_file():
        return {
            "path": str(path),
            "exists": True,
            "kind": "file",
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
    preferred = (
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "latest_checkpointed_iteration.txt",
    )
    files = []
    for name in preferred:
        candidate = path / name
        if candidate.is_file():
            files.append(
                {
                    "relative_path": name,
                    "size": candidate.stat().st_size,
                    "sha256": _sha256(candidate),
                }
            )
    return {
        "path": str(path),
        "exists": True,
        "kind": "directory",
        "identity_files": files,
    }


def _checkpoint_next_rollout_id(path: Path) -> int:
    tracker = path / "latest_checkpointed_iteration.txt"
    try:
        value = tracker.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError(f"resume checkpoint tracker is unreadable: {tracker}") from exc
    if value == "release":
        raise ValueError("resume_from must be a trained checkpoint, not a release checkpoint")
    try:
        iteration = int(value)
    except ValueError as exc:
        raise ValueError(f"invalid resume checkpoint iteration: {value!r}") from exc
    if iteration < 0:
        raise ValueError(f"invalid resume checkpoint iteration: {iteration}")
    return iteration + 1


def _prepare_run(plan: LaunchPlan) -> None:
    # This network gate runs before the output directory, GPU monitor, Ray
    # actors, or trainer process exists. A configured-but-dead retriever can
    # therefore never silently turn into a formal run.
    search_r1_identity = _search_r1_service_identity(plan.config)
    run_dir = plan.config.run_dir
    # Preflight has already established collision freedom; exist_ok=False also
    # closes the race between checking and creating the formal run directory.
    run_dir.mkdir(parents=True, exist_ok=False)
    for relative in ("logs", "rollouts", "checkpoints/start", "gpu", "metrics"):
        (run_dir / relative).mkdir(parents=True)
    (run_dir / "resolved_config.yaml").write_text(json.dumps(plan.config.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(_redacted_command(plan.command) + "\n", encoding="utf-8")
    (run_dir / "environment.json").write_text(
        json.dumps(_environment_record(plan, search_r1_identity), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    start_checkpoint = _path(plan.config.resume_from or plan.config.reference_checkpoint)
    data_path = _path(plan.config.data_path)
    tokenizer_path = _path(plan.config.tokenizer or plan.config.hf_checkpoint)
    (run_dir / "checkpoints/start/source.json").write_text(
        json.dumps(
            {
                "hf_checkpoint": str(_path(plan.config.hf_checkpoint)),
                "reference_checkpoint": str(_path(plan.config.reference_checkpoint)),
                "base_checkpoint_sha256": plan.config.base_checkpoint_sha256,
                "resume_from": str(_path(plan.config.resume_from)) if plan.config.resume_from else None,
                "start_checkpoint_identity": _checkpoint_identity(start_checkpoint),
                "tokenizer_identity": _checkpoint_identity(tokenizer_path),
                "training_data": {
                    "path": str(data_path),
                    "size": data_path.stat().st_size,
                    "sha256": _sha256(data_path),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _gpu_monitor(path: Path, stop: threading.Event) -> None:
    fields = "timestamp,index,name,memory.total,memory.used,utilization.gpu,utilization.memory,power.draw"
    query = fields.removeprefix("timestamp,")
    with path.open("w", encoding="utf-8") as handle:
        handle.write(fields + "\n")
        while not stop.is_set():
            timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
            try:
                result = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                for line in result.stdout.splitlines():
                    handle.write(f"{timestamp},{line}\n")
                handle.flush()
            except OSError as exc:
                handle.write(f"{timestamp},ERROR,{json.dumps(str(exc))}\n")
                handle.flush()
            stop.wait(15)


def execute(plan: LaunchPlan) -> int:
    if not plan.ok:
        failed = "; ".join(f"{item.name}: {item.detail}" for item in plan.checks if not item.ok)
        raise PreflightError(failed)
    _prepare_run(plan)
    run_dir = plan.config.run_dir
    stop = threading.Event()
    monitor = threading.Thread(target=_gpu_monitor, args=(run_dir / "gpu/nvidia-smi.csv", stop), daemon=True)
    monitor.start()
    env = os.environ.copy()
    env.update(plan.environment)
    log_path = run_dir / "logs/train.log"
    started = time.monotonic()
    child: subprocess.Popen[str] | None = None
    try:
        with log_path.open("w", encoding="utf-8") as log:
            child = subprocess.Popen(
                plan.command,
                cwd=_path(plan.config.slime_path),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert child.stdout is not None
            for line in child.stdout:
                sys.stdout.write(line)
                log.write(line)
                log.flush()
            return_code = child.wait()
    except KeyboardInterrupt:
        if child and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.terminate()
        return_code = 130
    finally:
        stop.set()
        monitor.join(timeout=20)
    summary = {
        "run_id": plan.config.run_id,
        "mode": plan.config.mode.value,
        "requested_optimizer_steps": plan.config.train_steps,
        "process_return_code": return_code,
        "wall_time_seconds": round(time.monotonic() - started, 3),
        "completion_verified": False,
        "note": "Successful optimizer updates must be populated from slime metrics before formal acceptance.",
    }
    (run_dir / "metrics/summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if return_code == 0 and not (plan.config.rollout_only or plan.config.dry_run):
        from .audit import AuditError, audit_run

        try:
            audit = audit_run(run_dir, plan.config.train_steps)
        except AuditError as exc:
            summary["completion_audit_error"] = str(exc)
            summary["process_return_code"] = 3
            return_code = 3
        else:
            summary.update(
                {
                    "completion_verified": True,
                    "successful_optimizer_updates": audit["successful_optimizer_updates"],
                    "checkpoint_iteration": audit["checkpoint"]["iteration"],
                }
            )
            (run_dir / "metrics/completion_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (run_dir / "metrics/summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return return_code


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config, _forwarded, print_command = resolve_config(argv)
        plan = build_plan(config)
        _print_plan(plan)
        if config.dry_run or print_command:
            return 0 if plan.ok else 2
        return execute(plan)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
