"""Resolved configuration for the StackPlanner plugin running on slime."""

from __future__ import annotations

import dataclasses
import enum
import json
import math
import re
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class TrainingMode(str, enum.Enum):
    CENTRAL_ONLY = "central-only"
    SHARED_ALL_AGENTS = "shared-all-agents"


class Experiment(str, enum.Enum):
    CENTRAL_BASELINE = "A"
    ALL_AGENTS = "B"
    CAPABILITY_DECOUPLED = "C"


class DecouplingScope(str, enum.Enum):
    NONE = "none"
    CENTRAL_CAPABILITY = "central-capability"
    ALL_AGENT_ACTION_OUTPUT = "all-agent-action-output"


@dataclasses.dataclass(slots=True)
class TrainingConfig:
    experiment: Experiment = Experiment.CENTRAL_BASELINE
    mode: TrainingMode = TrainingMode.CENTRAL_ONLY
    run_id: str = ""
    stackplanner_path: str = "StackPlanner2.0"
    slime_path: str = "slime"
    megatron_path: str = "../Megatron-LM"
    deer_flow_config_path: str = "configs/deerflow_search_r1_only.yaml"
    data_path: str = ""
    output_root: str = "outputs"
    hf_checkpoint: str = ""
    reference_checkpoint: str = ""
    base_checkpoint_sha256: str = ""
    tokenizer: str = ""
    frozen_model: str = ""
    frozen_base_url: str = ""
    # Leave empty for an unauthenticated local OpenAI-compatible endpoint.
    # When set, the named environment variable is required only when a frozen
    # route is actually constructed.
    frozen_api_key_env: str = ""
    actor_model_name: str = "slime-actor"
    actor_base_url: str = "http://127.0.0.1:18001/v1"
    custom_generate_function: str = "slime_plugins.stackplanner.generate.generate"
    reward_post_process_function: str = "slime_plugins.stackplanner.rewards.normalize_by_episode"
    train_steps: int = 100
    algorithm: str = "grpo"
    learning_rate: float = 1e-6
    kl_coefficient: float = 0.0
    clip_epsilon: float = 0.2
    optimizer: str = "adam"
    save_interval: int = 20
    eval_interval: int = 0
    # Optional OLE prompt-group selection; defaults match ../OLE.
    ole_enabled: bool = False
    ole_threshold_percent: float = 0.4
    ole_k: float = 1.0
    ole_init_threshold: float = 0.0
    samples_per_prompt: int = 4
    rollout_batch_size: int = 4
    temperature: float = 0.8
    top_p: float = 1.0
    max_context_tokens: int = 8192
    max_response_tokens: int = 1024
    episode_timeout_seconds: int = 900
    max_iterations: int = 20
    max_parallel_delegates: int = 1
    subagent_max_turns: int = 20
    subagent_timeout_seconds: int = 600
    adapter_host: str = "127.0.0.1"
    # Zero asks the rollout worker to bind an ephemeral local port. This is the
    # safe default when several colocated rollout workers share one host.
    adapter_port: int = 0
    sglang_router_host: str = "127.0.0.1"
    sglang_router_port: int = 30000
    # Independent RolloutManagers on one node must use disjoint 1024-port
    # engine blocks. The router has its own separately configured port.
    sglang_engine_port_base: int = 15000
    ray_address: str = "auto"
    num_nodes: int = 1
    gpus_per_node: int = 8
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    context_parallel_size: int = 1
    data_parallel_size: int = 8
    expert_parallel_size: int = 1
    colocate: bool = True
    offload: bool = False
    fully_async: bool = False
    reward_function: str = "slime_plugins.stackplanner.rewards:deterministic_reward"
    reward_config: dict[str, Any] = dataclasses.field(default_factory=dict)
    infrastructure_failure_policy: str = "remove"
    role_overrides: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)
    input_key: str = "prompt"
    label_key: str = "label"
    # Optional training-only instruction prepended to the user task before a
    # production StackPlanner rollout. It is recorded in resolved_config.yaml
    # and never used by the fixed evaluation path.
    training_prompt_prefix: str = ""
    # Training-only runtime gate that narrows Central's first action to the
    # canonical sp_delegate tool. Fixed evaluation never consumes this field.
    training_force_initial_delegate: bool = False
    seed: int = 42
    data_start: int = 0
    data_limit: int = 0
    dry_run: bool = False
    rollout_only: bool = False
    train_only: bool = False
    replay_path: str = ""
    resume_from: str = ""
    log_level: str = "INFO"
    dump_trajectories: bool = True
    extra_slime_args: list[str] = dataclasses.field(default_factory=list)
    capability_decoupling: bool = False
    decoupling_scope: DecouplingScope = DecouplingScope.NONE
    capability_taxonomy_path: str = ""
    capability_probe_manifest: str = ""
    capability_layer_map: str = ""
    capability_annotator_function: str = ""
    capability_loss_function: str = ""
    capability_gradient_router_function: str = ""
    # B may explicitly collect exact all-agent production trajectories before
    # its target-model probe exists. This mode is rollout-only and can never
    # install a loss or optimizer hook.
    capability_probe_bootstrap: bool = False

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> TrainingConfig:
        values = dict(raw)
        # Accept the pre-plugin prototype key while making the resolved config
        # and every new preset describe slime's actual extension point.
        legacy_generate = values.pop("rollout_function", None)
        if legacy_generate and "custom_generate_function" not in values:
            values["custom_generate_function"] = legacy_generate
        if "mode" in values:
            values["mode"] = TrainingMode(values["mode"])
        if "experiment" in values:
            values["experiment"] = Experiment(str(values["experiment"]).upper())
        if "decoupling_scope" in values:
            values["decoupling_scope"] = DecouplingScope(values["decoupling_scope"])
        valid = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(values) - valid)
        if unknown:
            raise ValueError(f"Unknown training config keys: {', '.join(unknown)}")
        config = cls(**values)
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> TrainingConfig:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError(f"{path} is not JSON; install PyYAML to load YAML configuration") from exc
            raw = yaml.safe_load(text)
        if not isinstance(raw, dict):
            raise ValueError("Training configuration root must be an object")
        return cls.from_mapping(raw)

    def validate(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if self.train_steps < 1:
            raise ValueError("train_steps must be positive")
        if not isinstance(self.ole_enabled, bool):
            raise ValueError("ole_enabled must be a boolean")
        for name in ("ole_threshold_percent", "ole_k", "ole_init_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if not 0 < self.ole_threshold_percent <= 1:
            raise ValueError("ole_threshold_percent must be in (0, 1]")
        if self.ole_k < 0:
            raise ValueError("ole_k must not be negative")
        if self.ole_enabled:
            reserved = {"--rollout-function-path", "--rollout-data-postprocess-path", "--data-source-path", "--disable-rollout-global-dataset"}
            if any(arg.split("=", 1)[0] in reserved for arg in self.extra_slime_args):
                raise ValueError("OLE owns the rollout and postprocess hooks and requires the checkpointed global data source")
        if self.base_checkpoint_sha256 and not re.fullmatch(r"[0-9a-f]{64}", self.base_checkpoint_sha256):
            raise ValueError("base_checkpoint_sha256 must be a lowercase SHA-256 digest")
        if self.num_nodes < 1 or self.gpus_per_node < 1:
            raise ValueError("num_nodes and gpus_per_node must be positive")
        engine_port_limit = self.sglang_engine_port_base + 1024
        if self.sglang_engine_port_base < 1024 or engine_port_limit > 65536:
            raise ValueError("sglang_engine_port_base must reserve a complete 1024-port block within 1024..65535")
        if self.sglang_engine_port_base <= self.sglang_router_port < engine_port_limit:
            raise ValueError("sglang_router_port must not overlap this run's reserved SGLang engine-port block")
        if self.max_parallel_delegates < 1:
            raise ValueError("max_parallel_delegates must be positive")
        if self.data_start < 0 or self.data_limit < 0:
            raise ValueError("data_start and data_limit must not be negative")
        if self.rollout_batch_size < 1 or self.samples_per_prompt < 1:
            raise ValueError("rollout_batch_size and samples_per_prompt must be positive")
        if not isinstance(self.training_prompt_prefix, str):
            raise ValueError("training_prompt_prefix must be a string")
        if not isinstance(self.training_force_initial_delegate, bool):
            raise ValueError("training_force_initial_delegate must be a boolean")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if (self.custom_generate_function == "slime_plugins.stackplanner.generate.generate"
                and not self.train_only and self.top_p != 1.0):
            raise ValueError(
                "The built-in StackPlanner rollout currently requires top_p=1.0; "
                "it does not preserve the exact nucleus token metadata needed for training. "
                "A custom_generate_function may implement its own top-p replay support."
            )
        if self.rollout_only and self.train_only:
            raise ValueError("rollout_only and train_only are mutually exclusive")
        if self.train_only and not self.replay_path:
            raise ValueError("train_only requires replay_path")
        if self.capability_probe_bootstrap and not (self.experiment is Experiment.ALL_AGENTS and self.rollout_only):
            raise ValueError("capability_probe_bootstrap is allowed only for experiment B rollout-only collection")
        if self.infrastructure_failure_policy not in {"remove", "retry", "fail"}:
            raise ValueError("infrastructure_failure_policy must be remove, retry, or fail")
        expected = {
            Experiment.CENTRAL_BASELINE: (
                TrainingMode.CENTRAL_ONLY,
                False,
                DecouplingScope.NONE,
            ),
            # Experiment B was revised by the experiment owner on 2026-08-11:
            # every policy role shares the actor and the actor output is routed
            # through the two-route action/output runtime. Keep this in the
            # executable contract so an old non-decoupled B preset cannot be
            # launched accidentally after a handoff.
            Experiment.ALL_AGENTS: (
                TrainingMode.SHARED_ALL_AGENTS,
                True,
                DecouplingScope.ALL_AGENT_ACTION_OUTPUT,
            ),
            Experiment.CAPABILITY_DECOUPLED: (
                TrainingMode.CENTRAL_ONLY,
                True,
                DecouplingScope.CENTRAL_CAPABILITY,
            ),
        }[self.experiment]
        actual = (self.mode, self.capability_decoupling, self.decoupling_scope)
        if actual != expected:
            raise ValueError(f"experiment {self.experiment.value} requires mode={expected[0].value}, capability_decoupling={expected[1]}, and decoupling_scope={expected[2].value}")
        if self.capability_decoupling:
            from .capability.contracts import require_capability_runtime

            require_capability_runtime(self)

    def as_dict(self) -> dict[str, Any]:
        raw = dataclasses.asdict(self)
        raw["mode"] = self.mode.value
        raw["experiment"] = self.experiment.value
        raw["decoupling_scope"] = self.decoupling_scope.value
        return raw

    @property
    def run_dir(self) -> Path:
        if not self.run_id:
            raise ValueError("run_id is required before resolving run_dir")
        root = Path(self.output_root).expanduser()
        if not root.is_absolute():
            root = _REPOSITORY_ROOT / root
        return root.resolve() / self.run_id
