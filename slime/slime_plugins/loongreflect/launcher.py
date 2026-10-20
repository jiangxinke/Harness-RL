"""Validate prepared data and launch the optional full-parameter RL plugin."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[3]


def validate_data(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("query"), str) or not row["query"].strip():
                raise ValueError(f"line {line_number}: query must be a nonempty string")
            metadata = row.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError(f"line {line_number}: metadata must be an object")
            split = str(row.get("split", metadata.get("split", ""))).lower()
            if split in {"test", "dev", "validation", "eval", "evaluation"}:
                raise ValueError(f"line {line_number}: evaluation split cannot be used for training")
            answer = row.get("answer")
            aliases = answer if isinstance(answer, list) else [answer]
            if not aliases or not all(isinstance(alias, str) and alias.strip() for alias in aliases):
                raise ValueError(f"line {line_number}: answer must contain nonempty reference strings")
            count += 1
    if not count:
        raise ValueError("training data is empty")
    return count


def model_arguments(path: Path) -> list[str]:
    """Read the literal MODEL_ARGS array from a checked-in Slime preset."""
    text = path.read_text(encoding="utf-8")
    if "MODEL_ARGS=(" not in text:
        raise ValueError("model preset must define a literal MODEL_ARGS array")
    array = text.split("MODEL_ARGS=(", 1)[1].split(")", 1)[0]
    if any(character in array for character in ("$", "`", "(", ")")):
        raise ValueError("model preset must use literal arguments; expressions are unsupported")
    return shlex.split(array, comments=True)


def build_command(args: argparse.Namespace) -> tuple[list[str], dict]:
    if min(args.steps, args.actor_gpus, args.rollout_gpus, args.tensor_parallel, args.rollout_batch_size, args.samples_per_prompt, args.sequence_length, args.max_response_tokens, args.save_interval) <= 0:
        raise ValueError("step, resource, batch and token counts must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning rate must be finite and positive")
    if args.actor_gpus % args.tensor_parallel:
        raise ValueError("actor GPU count must be divisible by tensor parallel size")
    if args.max_response_tokens >= args.sequence_length:
        raise ValueError("response budget must be smaller than sequence length")
    plugin = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(plugin, dict) or plugin.get("reflect_self_distill") is not True:
        raise ValueError("config must enable reflect_self_distill")
    plugin["reflect_train_mode"] = args.mode or plugin.get("reflect_train_mode", "joint")
    stage_override = getattr(args, "require_post_search_reflection", None)
    if stage_override is not None:
        plugin["reflect_require_post_search_reflection"] = stage_override
    stage_enabled = plugin.get("reflect_require_post_search_reflection", False)
    if type(stage_enabled) is not bool:
        raise ValueError("reflect_require_post_search_reflection must be a boolean")
    if stage_enabled and plugin.get("reflect_action_protocol", "full") != "full":
        raise ValueError("post-search reflection requires the full action protocol")
    if args.fast_updates is not None:
        plugin["reflect_fast_updates_per_step"] = args.fast_updates
    if int(plugin.get("reflect_fast_updates_per_step", 3)) < 1:
        raise ValueError("Fast-Slow requires at least one fast update")
    if args.search_url:
        plugin["reflect_search_url"] = args.search_url
    batch = args.rollout_batch_size * args.samples_per_prompt
    command = [sys.executable, str(ROOT / "slime/train.py"), *model_arguments(args.model_args)]
    options = {
        "actor-num-nodes": 1,
        "actor-num-gpus-per-node": args.actor_gpus,
        "rollout-num-gpus": args.rollout_gpus,
        "rollout-num-gpus-per-engine": 1,
        "tensor-model-parallel-size": args.tensor_parallel,
        "context-parallel-size": 1,
        "hf-checkpoint": args.hf_checkpoint,
        "load": args.resume or args.megatron_checkpoint,
        "ref-load": args.reference_checkpoint,
        "save": args.save_dir / "checkpoints",
        "save-interval": args.save_interval,
        "prompt-data": args.data,
        "input-key": "query",
        "label-key": "answer",
        "num-rollout": args.steps,
        "rollout-batch-size": args.rollout_batch_size,
        "n-samples-per-prompt": args.samples_per_prompt,
        "global-batch-size": batch,
        "num-steps-per-rollout": 1,
        "micro-batch-size": 1,
        "seq-length": args.sequence_length,
        "max-tokens-per-gpu": args.sequence_length,
        "rollout-max-context-len": args.sequence_length,
        "rollout-max-prompt-len": args.sequence_length - 1,
        "rollout-max-response-len": args.max_response_tokens,
        "rollout-temperature": 1.0,
        "rollout-top-p": 1.0,
        "advantage-estimator": "grpo",
        "lr": args.learning_rate,
        "lr-decay-style": "constant",
        "weight-decay": 0.01,
        "adam-beta1": 0.9,
        "adam-beta2": 0.98,
        "clip-grad": 1.0,
        "eps-clip": 0.2,
        "eps-clip-high": 0.28,
        "kl-loss-coef": 0.001,
        "kl-loss-type": "low_var_kl",
        "loss-type": "custom_loss",
        "custom-advantage-function-path": "slime_plugins.loongreflect.self_distill.dense_tree_advantages",
        "custom-loss-function-path": "slime_plugins.loongreflect.self_distill.reflect_joint_loss",
        "rollout-data-postprocess-path": "slime_plugins.loongreflect.self_distill.schedule_rollout_loss",
        "custom-generate-function-path": "slime_plugins.loongreflect.slime_adapter.generate",
        "custom-rm-path": "slime_plugins.loongreflect.slime_adapter.reward_func",
        "custom-config-path": args.save_dir / "loongreflect.yaml",
    }
    for name, value in options.items():
        command.extend((f"--{name}", str(value)))
    command.extend(("--bf16", "--use-dynamic-batch-size", "--rollout-shuffle", "--disable-rewards-normalization", "--use-kl-loss"))
    if args.resume is None:
        command.extend(("--finetune", "--no-load-optim", "--no-load-rng", "--start-rollout-id", "0"))
    return command, plugin


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional LoongReflect RL; validates and prints by default")
    for name in ("data", "hf-checkpoint", "megatron-checkpoint", "reference-checkpoint", "save-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--model-args", type=Path, default=ROOT / "slime/scripts/models/qwen2.5-3B.sh")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/loongreflect/joint.yaml")
    parser.add_argument("--mode", choices=("joint", "fast_slow", "policy_only", "distill_only"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--actor-gpus", type=int, default=4)
    parser.add_argument("--rollout-gpus", type=int, default=4)
    parser.add_argument("--tensor-parallel", type=int, default=2)
    parser.add_argument("--rollout-batch-size", type=int, default=8)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=16384)
    parser.add_argument("--max-response-tokens", type=int, default=8192)
    parser.add_argument("--fast-updates", type=int)
    parser.add_argument(
        "--require-post-search-reflection",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional training-only full-protocol stage: require a real model reflection after search evidence before accepting an answer",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--search-url")
    parser.add_argument("--resume", type=Path, help="Existing checkpoint directory; steps remains the total target")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    for name in ("data", "hf_checkpoint", "megatron_checkpoint", "reference_checkpoint", "save_dir", "model_args", "config", "resume"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.absolute())
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = validate_data(args.data)
    command, plugin = build_command(args)
    if not args.execute:
        print(f"validated_rows={rows}")
        print(shlex.join(command))
        print(yaml.safe_dump(plugin, sort_keys=True))
        return 0
    if not args.search_url:
        raise ValueError("--search-url is required for execution")
    for path in (args.hf_checkpoint, args.megatron_checkpoint, args.reference_checkpoint, args.resume):
        if path is not None and not path.is_dir():
            raise FileNotFoundError(path)
    if rows < args.rollout_batch_size:
        raise ValueError("training data must contain at least one rollout batch")
    if args.save_dir.exists() and any(args.save_dir.iterdir()) and args.resume is None:
        raise ValueError("output directory is nonempty; choose a new directory or use --resume")
    args.save_dir.mkdir(parents=True, exist_ok=True)
    (args.save_dir / "loongreflect.yaml").write_text(yaml.safe_dump(plugin, sort_keys=True), encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT / "slime"), environment.get("PYTHONPATH", "")))
    return subprocess.run(command, cwd=ROOT, env=environment, check=False).returncode
