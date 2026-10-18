"""Optional LoongReflect lifecycle hooks for the existing Megatron actor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from .self_distill import build_update_objectives, ema_tau


def validate_runtime(args: Any, parallel: dict[str, int]) -> None:
    if parallel["cp_size"] != 1:
        raise ValueError("LoongReflect private teacher scoring requires context-parallel-size=1")
    if not args.compute_advantages_and_returns or args.loss_type != "custom_loss":
        raise ValueError("LoongReflect requires advantage computation and its custom loss")
    if getattr(args, "use_opd", False) or getattr(args, "use_critic", False):
        raise ValueError("LoongReflect EMA self-distillation requires the actor-only training path")
    if getattr(args, "use_routing_replay", False):
        raise ValueError("LoongReflect does not replay student routing under the private teacher prompt")
    if getattr(args, "rollout_top_p", 1.0) != 1.0:
        raise ValueError("LoongReflect currently requires rollout-top-p=1")
    if getattr(args, "reset_optimizer_states", False):
        raise ValueError("LoongReflect must preserve optimizer state between Fast-Slow substeps")
    if getattr(args, "reflect_lightweight_checkpoint", False):
        raise ValueError("This integration uses standard full checkpoints, not the reference LoRA format")
    for name in ("reflect_ema_tau_base", "reflect_ema_tau_final"):
        if not 0.0 <= float(getattr(args, name, 0.996 if name.endswith("base") else 1.0)) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    build_update_objectives(args)


def teacher_checkpoint_path(base: str, rollout_id: int, rank: int) -> Path:
    return Path(base) / f"iter_{rollout_id:07d}" / f"reflect_ema_rank_{rank:05d}.pt"


def initialize_teacher(actor: Any, loaded_rollout_id: int) -> None:
    actor.weights_backuper.backup("ema_teacher")
    path = teacher_checkpoint_path(actor.args.load or actor.args.ref_load, loaded_rollout_id, dist.get_rank())
    if not path.is_file():
        if getattr(actor.args, "start_rollout_id", None) != 0:
            raise FileNotFoundError(f"LoongReflect resume requires the EMA teacher checkpoint: {path}")
        return
    state = torch.load(path, map_location="cpu", weights_only=True)
    destination = actor.weights_backuper.get("ema_teacher")
    if state.keys() != destination.keys():
        raise ValueError("EMA checkpoint tensor names do not match the actor")
    for name, value in state.items():
        if value.shape != destination[name].shape or value.dtype != destination[name].dtype:
            raise ValueError(f"EMA checkpoint tensor shape/dtype mismatch: {name}")
    for name, value in state.items():
        destination[name].copy_(value)


def update_teacher(actor: Any, rollout_id: int) -> None:
    actor.weights_backuper.ema_update(
        src_tag="actor",
        dst_tag="ema_teacher",
        tau=ema_tau(
            rollout_id,
            actor.args.num_rollout,
            float(getattr(actor.args, "reflect_ema_tau_base", 0.996)),
            float(getattr(actor.args, "reflect_ema_tau_final", 1.0)),
        ),
    )


def save_teacher(actor: Any, rollout_id: int) -> None:
    path = teacher_checkpoint_path(actor.args.save, rollout_id, dist.get_rank())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(actor.weights_backuper.get("ema_teacher"), temporary)
    temporary.replace(path)
