#!/usr/bin/env python3
"""Opt-in fresh TRAIN-only native final-decision DPO with one NF4 base.

Generation closure is validated before any offline reward receipt is opened.
Rewards must have a separately approved explicit version. This entrypoint never
collects, scores labels, resumes, imports an adapter, or evaluates benchmarks.
The original completion scorer and all existing SFT entrypoints stay unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import timedelta
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import random
import re
import subprocess
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parent),
                str(Path(__file__).resolve().parents[1] / "slime")]

import torch
import torch.distributed as dist

from dcc_native_dpo import OfflineNativeDPOTrainer, ReferenceIdentity, validate_deterministic_scoring
from dcc_native_dpo_data import TrainRewardReceipt, best_worst_pair, validate_final_candidate
from dcc_native_sft import _strict_json
from dcc_training_io import TrainingJournal, finalize_training, prepare_training_output, publish_checkpoint
from slime_plugins.dcc.hf_policy import completion_logprobs, load_policy_model
from slime_plugins.dcc.hf_chunked_logprobs import resolve_logprob_backend
from slime_plugins.dcc.native_dpo_adapter import DPO_METHOD, PAIR_CONTRACT
from slime_plugins.dcc.native_dpo_sampling import (
    DOMAIN_BALANCED_PAIR_SAMPLING, LEGACY_PAIR_SAMPLING, DomainBalancedPairDeck,
    balanced_pair_sampling_configuration,
)
from slime_plugins.dcc.episode import WIDESEARCH_EVIDENCE_GUARD_VERSION
from slime_plugins.dcc_train_literal_reward_contract import (
    PROTOCOL as LITERAL_PROTOCOL, LITERAL_RECEIPT, LITERAL_DPO_COLLECTIONS, PER_COLLECTION_RECEIPTS,
    configuration as literal_configuration, validate_approval, validate_literal_receipt,
)

ALGORITHM = "offline_native_final_DPO; not RL"
QUANTIZATION_ID = "qwen3-moe-nf4-double-quant-bf16-compute-transformers-4.57.6-v1"
REWARD_CONTRACT = "native-final-train-reward-receipt-v1"
COLLECTIONS_CONTRACT = "native-final-dpo-collections-v1"
PUBLIC_OWNERSHIP_FIELDS = ("frozen_prefix_ids", "frozen_task_ids", "attempt_trace_ids", "phase_request_ids")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def public_id(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/+@-]{0,199}", value)
            and all(part not in {"", ".", ".."} for part in value.split("/")),
            "Use an explicit public identity, without a filesystem path")
    return value


def training_source_identity(expected_source):
    """A new trainer requires its actual committed clean training runtime.

    This is separate from the approved source that generated the offline IDs.
    A patch applied to an old checkout cannot claim the old commit unchanged.
    """
    root = Path(__file__).resolve().parents[1]
    require(isinstance(expected_source, str) and re.fullmatch(r"[0-9a-f]{40}", expected_source),
            "Explicit approved 40-hex training source commit required")
    def git(*arguments):
        result = subprocess.run(["git", "-C", str(root), *arguments], capture_output=True, text=True)
        require(result.returncode == 0, "Training runtime must be inside its actual committed source checkout")
        return result.stdout.strip()
    require(git("rev-parse", "HEAD") == expected_source,
            "Actual training runtime commit differs from caller-approved training source")
    git("ls-files", "--error-unmatch", "scripts/dcc_native_dpo_train.py", "scripts/dcc_native_dpo.py",
        "scripts/dcc_native_dpo_data.py", "scripts/dcc_native_physical.py")
    require(not git("status", "--porcelain", "--untracked-files=normal", "--", "scripts", "slime/slime_plugins/dcc"),
            "Training runtime source has modified or untracked code; commit and approve the actual new source")
    return {"version": "dcc-offline-native-dpo-training-source-v1", "source_commit": expected_source,
            "tracked_runtime_clean": True, "training_entrypoint": "dcc_native_dpo_train.py"}


def read_train_rewards(path, *, reward_version, train_reward=None):
    """Call only after read_closed_g4 returned successfully on ALL ranks."""
    literal = reward_version == LITERAL_PROTOCOL
    if literal:
        validate_approval(train_reward, expected_protocol=reward_version)
    else:
        require(train_reward is None, "Legacy reward receipts exclude new configuration metadata")
    indexed = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            require(line.endswith("\n"), "Incomplete offline reward receipt journal")
            row = _strict_json(line)
            if literal:
                fields = validate_literal_receipt(row, reward_version=reward_version, train_reward=train_reward,
                    evidence_guard_version=WIDESEARCH_EVIDENCE_GUARD_VERSION)
                receipt = TrainRewardReceipt(**fields)
            else:
                require(isinstance(row, dict) and set(row) == {"contract", "task_id", "trace_id", "prefix_id",
                    "harness_id", "reward_version", "value", "generation_closed_before_scoring", "split"}
                    and row["contract"] == REWARD_CONTRACT and row["reward_version"] == reward_version,
                    "Explicit separately approved reward receipt version required")
                receipt = TrainRewardReceipt(**{key: value for key, value in row.items() if key != "contract"})
            require(receipt.trace_id not in indexed, "Duplicate offline reward receipt")
            indexed[receipt.trace_id] = receipt
    return indexed


def read_approved_collection(args, tokenizer):
    """Only public freeze and complete generation journals; never rewards."""
    from dcc_collect_native_preferences import load_frozen, read_closed_g4
    freeze, prefixes, identity = load_frozen(args.approved_freeze, tokenizer)
    require(identity.derivative_id == args.base_derivative_id,
            "Frozen collection derivative differs from the approved training/reference base")
    closed = read_closed_g4(args.generation_dir, freeze=freeze, prefixes=prefixes,
        tokenizer=tokenizer, world_size=args.generation_world_size, expected_source=args.expected_source)
    require(closed.get("generation_closed_before_scoring") is True,
            "Whole generation closure required before reward receipt access")
    return freeze, prefixes, identity, closed


def pairs_from_closed_collection(args, tokenizer, prepared):
    """Called only after every approved collection has passed its closure."""
    freeze, prefixes, identity, closed = prepared
    if args.reward_version == LITERAL_PROTOCOL:
        validate_approval(getattr(args, "train_reward", None), expected_protocol=args.reward_version)
        require({prefix.benchmark for prefix in prefixes} == {"widesearch"},
            "Literal TRAIN rewards require independently frozen WideSearch domain")
    rewards = read_train_rewards(args.train_rewards, reward_version=args.reward_version,
        train_reward=getattr(args, "train_reward", None))
    envelopes = closed["candidates"]
    actual_ids = [e["native_bundle"]["collector"]["trace_id"] for e in envelopes]
    require(len(set(actual_ids)) == len(actual_ids) and set(rewards) == set(actual_ids),
            "Offline rewards must exactly cover valid closed-generation candidates")
    by_prefix = {p.prefix_id: [] for p in prefixes}
    public = {p.prefix_id: p for p in prefixes}
    for envelope in envelopes:
        collector = envelope["native_bundle"]["collector"]
        candidate = validate_final_candidate(envelope, tokenizer,
            prefix=public[envelope["prefix_id"]],
            expected_identity=replace(identity, generation_seed=collector["generation_seed"]),
            reward=rewards[collector["trace_id"]])
        by_prefix[candidate.prefix.prefix_id].append(candidate)
    pairs, omitted = [], {"fewer_than_two_valid_candidates": 0, "reward_gap_at_most_0.05": 0}
    for prefix in prefixes:
        candidates = by_prefix[prefix.prefix_id]
        if len(candidates) < 2:
            omitted["fewer_than_two_valid_candidates"] += 1
            continue
        pair = best_worst_pair(candidates, minimum_reward_gap=0.05)
        if pair is None:
            omitted["reward_gap_at_most_0.05"] += 1
        else:
            pairs.append(pair)
    require(bool(pairs), "No strict-gap TRAIN preference pairs; refusing an empty training run")
    metadata = {"experiment_id": public_id(freeze["experiment_id"]),
        "harness_id": public_id(freeze["harness_id"]), "schema_policy": freeze["schema_policy"],
        "collection_source_commit": args.expected_source, "reward_version": args.reward_version,
        "collection_native_identity": freeze["native_identity"],
        "collection_source_receipt": freeze["source_receipt"],
        "generation_world_size": args.generation_world_size, "generation_closed_before_scoring": True,
        "attempt_count": len(closed["attempts"]), "valid_candidates": len(envelopes),
        "eligible_pairs": len(pairs), "omitted_prefixes": omitted,
        "pair_selection": "one-best-worst-per-frozen-prefix; reward-gap-strictly-greater-than-0.05",
        "training_sources": sorted({p.chosen.prefix.source for p in pairs})}
    if args.reward_version == LITERAL_PROTOCOL:
        metadata.update(reward_contract=LITERAL_RECEIPT, train_reward=literal_configuration(),
            train_metric="train_exact_row_f1")
    metadata.update(frozen_prefix_ids=list(freeze["prefix_ids"]), frozen_task_ids=list(freeze["task_ids"]),
        public_freeze=freeze,
        attempt_trace_ids=[row["trace_id"] for row in closed["attempts"]],
        phase_request_ids=list(closed["phase_request_ids"]),
        selected_pairs=[{"prefix_id": p.chosen.prefix.prefix_id,
            "chosen_trace_id": p.chosen.trace_id, "rejected_trace_id": p.rejected.trace_id,
            "chosen_seed": p.chosen.generation_seed, "rejected_seed": p.rejected.generation_seed}
            for p in pairs])
    return pairs, metadata


def common_collection_identity(native):
    # Context, thought sampling, history and collection source stay per domain.
    # Model/tokenizer/template/protocol describe the same actual training base.
    return {key: value for key, value in native.items() if key not in {
        "source_commit", "max_prompt_tokens", "max_model_len", "thought_tokens",
        "thought_sampling_profile", "prompt_history"}}


def validate_public_collection_ownership(profiles):
    """Validate every public owner and common base before any reward access."""
    require(bool(profiles), "At least one separately approved collection required")
    seen = {key: set() for key in ("collection_id", *PUBLIC_OWNERSHIP_FIELDS)}
    base_identity = None
    for profile in profiles:
        collection_id = public_id(profile["collection_id"])
        require(collection_id not in seen["collection_id"], "Duplicate approved collection identity")
        seen["collection_id"].add(collection_id)
        for field in PUBLIC_OWNERSHIP_FIELDS:
            values = profile[field]
            require(isinstance(values, list) and all(isinstance(value, str) and bool(value.strip()) for value in values)
                and len(set(values)) == len(values) and not seen[field].intersection(values),
                "Reused public prefix/task/attempt/phase identity across approved collections")
            require(bool(values) or field == "phase_request_ids", "Missing approved public collection identities")
            seen[field].update(values)
        native = profile["collection_native_identity"]
        require(isinstance(native, dict), "Approved public native collection identity required")
        common = common_collection_identity(native)
        require(base_identity is None or common == base_identity,
                "Approved collections differ in base/tokenizer/template/protocol identity")
        base_identity = common


def validate_prepared_collection_ownership(approved):
    profiles = []
    for collection_id, _, prepared in approved:
        freeze, _, _, closed = prepared
        profiles.append({"collection_id": collection_id, "collection_native_identity": freeze["native_identity"],
            "frozen_prefix_ids": freeze["prefix_ids"], "frozen_task_ids": freeze["task_ids"],
            "attempt_trace_ids": [row["trace_id"] for row in closed["attempts"]],
            "phase_request_ids": closed["phase_request_ids"]})
    validate_public_collection_ownership(profiles)


def load_dpo_pairs(args, tokenizer):
    prepared = read_approved_collection(args, tokenizer)
    validate_prepared_collection_ownership([("single-approved-collection", args, prepared)])
    return pairs_from_closed_collection(args, tokenizer, prepared)


def merge_training_collections(collections):
    """Retain each approved public collection, never erase its context/source."""
    validate_public_collection_ownership([dict(metadata, collection_id=collection_id)
        for collection_id, _, metadata in collections])
    pairs, profiles = [], []
    for collection_id, local_pairs, metadata in collections:
        profiles.append(dict(metadata, collection_id=collection_id))
        pairs.extend(local_pairs)
    require(len({p.key for p in pairs}) == len(pairs), "Duplicate exact pairs across approved collections")
    literal = any(p["reward_version"] == LITERAL_PROTOCOL for p in profiles)
    return pairs, {"collection_merge_contract": LITERAL_DPO_COLLECTIONS if literal else COLLECTIONS_CONTRACT,
        "collections": profiles,
        "generation_closed_before_scoring": True, "eligible_pairs": len(pairs),
        "valid_candidates": sum(p["valid_candidates"] for p in profiles),
        "attempt_count": sum(p["attempt_count"] for p in profiles),
        "training_sources": sorted({source for p in profiles for source in p["training_sources"]}),
        "pair_selection": "one-best-worst-per-frozen-prefix; reward-gap-strictly-greater-than-0.05"}


def load_training_pairs(args, tokenizer):
    """A caller-approved multi-collection manifest supplies runtime locations.

    Every collection is independently closed/validated/paired. Its complete
    public identity and selected pair ownership are retained in training metadata.
    Runtime locations from this manifest never enter the public configuration.
    """
    if args.approved_collections is None:
        require(all(getattr(args, field, None) is not None for field in (
            "approved_freeze", "generation_dir", "train_rewards", "reward_version", "expected_source")),
            "Single collection requires its freeze/generation/reward/source arguments")
        pairs, metadata = load_dpo_pairs(args, tokenizer)
        return merge_training_collections([("single-approved-collection", pairs, metadata)])
    require(all(getattr(args, field, None) is None for field in (
        "approved_freeze", "generation_dir", "train_rewards", "reward_version", "expected_source", "train_reward")),
        "Multi-collection approval excludes single-collection runtime arguments")
    manifest = _strict_json(Path(args.approved_collections).read_text(encoding="utf-8"))
    require(isinstance(manifest, dict) and set(manifest) == {
        "version", "base_derivative_id", "tokenizer_id", "collections"}
        and manifest["version"] in {COLLECTIONS_CONTRACT, LITERAL_DPO_COLLECTIONS}
        and manifest["base_derivative_id"] == args.base_derivative_id
        and manifest["tokenizer_id"] == args.tokenizer_id
        and isinstance(manifest["collections"], list) and bool(manifest["collections"]),
        "Explicit independently approved multi-collection/base/tokenizer contract required")
    approved = []
    literal_manifest = manifest["version"] == LITERAL_DPO_COLLECTIONS
    literal_collections = 0
    for row in manifest["collections"]:
        literal = isinstance(row, dict) and row.get("reward_version") == LITERAL_PROTOCOL
        expected_fields = {"collection_id", "approved_freeze", "generation_dir", "generation_world_size",
            "train_rewards", "expected_source", "reward_version"} | ({"train_reward"} if literal else set())
        require(isinstance(row, dict) and set(row) == expected_fields and (not literal or literal_manifest)
            and type(row["generation_world_size"]) is int and row["generation_world_size"] > 0
            and isinstance(row["expected_source"], str) and re.fullmatch(r"[0-9a-f]{40}", row["expected_source"]),
            "Each collection needs its own explicit freeze/generation/reward/source approval")
        public_id(row["reward_version"])
        if literal:
            validate_approval(row["train_reward"], expected_protocol=row["reward_version"])
            literal_collections += 1
        local_args = SimpleNamespace(**{key: value for key, value in row.items() if key != "collection_id"},
                                     base_derivative_id=args.base_derivative_id)
        prepared = read_approved_collection(local_args, tokenizer)
        approved.append((row["collection_id"], local_args, prepared))
    require(not literal_manifest or literal_collections > 0, "New literal manifest must declare the new reward protocol")
    # Validate ALL closures AND cross-collection ownership before ANY reward access.
    validate_prepared_collection_ownership(approved)
    return merge_training_collections([(collection_id, *pairs_from_closed_collection(local_args, tokenizer, prepared))
        for collection_id, local_args, prepared in approved])


def collective_pair_count(local_count, *, device, world_size=1, distributed=dist):
    require(type(local_count) is int and local_count >= 0, "Nonnegative local pair count required")
    count = torch.tensor(local_count, dtype=torch.int64, device=device)
    if world_size > 1:
        require(distributed.is_initialized() and distributed.get_world_size() == world_size,
                "Initialized process group must match the training world size")
        distributed.all_reduce(count, op=distributed.ReduceOp.SUM)
    return int(count.item())


def synchronize_pair_gradients(parameters, world_size, *, distributed=dist):
    """Every rank calls every collective, even with no local pair/gradient."""
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        distributed.all_reduce(parameter.grad, op=distributed.ReduceOp.SUM)
        parameter.grad.div_(world_size)


def collective_dpo_step(trainer, pairs, *, distributed=dist):
    """Derive the denominator from the actual batch; callers cannot supply it."""
    pairs = tuple(pairs)
    count = collective_pair_count(len(pairs), device=trainer.parameters[0].device,
                                 world_size=trainer.world_size, distributed=distributed)
    result = trainer.step(pairs, global_pair_count=count)
    if count:
        keys = ("dpo_loss_sum", "reference_adjusted_margin_sum", "chosen_raw_sum", "rejected_raw_sum")
        values = torch.tensor([result[key] for key in keys], dtype=torch.float64,
                              device=trainer.parameters[0].device)
        if trainer.world_size > 1:
            distributed.all_reduce(values, op=distributed.ReduceOp.SUM)
        result.update({"global_" + key: float(value) for key, value in zip(keys, values.tolist())})
        result["global_mean_dpo_loss"] = float(values[0] / count)
    return result


def validate_fresh_nf4_lora(model):
    """Production entrypoint accepts only the approved loader's fresh adapter."""
    validate_deterministic_scoring(model)
    configs = getattr(model, "peft_config", {})
    require(set(configs) == {"default"} and model.active_adapters == ["default"],
            "Exactly one fresh default LoRA adapter required")
    config = configs["default"]
    require(str(config.peft_type).split(".")[-1] == "LORA" and config.r == 16
            and config.lora_alpha == 32 and config.lora_dropout == 0
            and set(config.target_modules) == {"q_proj", "v_proj", "o_proj"}
            and config.bias == "none" and not config.modules_to_save
            and not getattr(config, "use_dora", False) and not getattr(config, "use_rslora", False)
            and not getattr(config, "rank_pattern", {}) and not getattr(config, "alpha_pattern", {}),
            "Production DPO requires fresh rank16/alpha32 q/v/o LoRA with zero dropout")
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    require(bool(parameters) and all("lora_" in name
        and any("." + target + "." in name for target in ("q_proj", "v_proj", "o_proj"))
        for name, _ in parameters), "Only approved q/v/o LoRA parameters may be trainable")
    require(any("lora_B" in name for name, _ in parameters)
        and all(torch.count_nonzero(p).item() == 0 for name, p in parameters if "lora_B" in name),
        "Fresh zero-initialized LoRA B required; imported/warm adapters are unsupported")
    report = getattr(model, "dcc_quantization_report", {})
    require(report.get("mode") == "nf4" and report.get("transformers_version") == "4.57.6"
        and report.get("expert_4bit_coverage") == 1.0
        and report.get("expert_parameters", 0) > 0
        and report.get("expert_parameters") == report.get("expert_4bit_parameters")
        and report.get("expert_4bit_matrices", 0) > 0,
        "Approved complete NF4 expert-coverage report required")
    return report


class GlobalPairDeck:
    """A fresh shuffled deck, sampled without duplicate pairs in an update."""
    def __init__(self, count, *, seed):
        require(type(count) is int and count > 0, "Nonempty validated pair dataset required")
        self.count, self.rng, self.deck = count, random.Random(seed), []

    def next(self, requested):
        require(type(requested) is int and requested > 0, "Positive global pairs per step required")
        if not self.deck:
            self.deck = list(range(self.count))
            self.rng.shuffle(self.deck)
        size = min(requested, len(self.deck))
        selected, self.deck = self.deck[:size], self.deck[size:]
        return selected


def run_dpo_loop(model, tokenizer, pairs, reference_identity, *, output, steps,
                 pairs_per_step=4, learning_rate=1e-6, beta=0.1, seed=42,
                 rank=0, world_size=1, distributed=dist, config=None,
                 save_every=0, save_initial=False, sync_artifacts=False,
                 logprob_fn=completion_logprobs, pair_sampling=LEGACY_PAIR_SAMPLING):
    """Shared whole loop; CLI validates production NF4 before entering here.

    Tiny CPU fixtures exercise this same optimizer/journal/checkpoint path.
    There is no CLI switch that disables the production NF4/fresh-LoRA gates.
    """
    require(type(steps) is int and steps > 0 and type(save_every) is int and save_every >= 0,
            "Positive update count and nonnegative save interval required")
    require(type(learning_rate) in (int, float) and math.isfinite(learning_rate) and learning_rate > 0,
            "Positive finite learning rate required")
    require(reference_identity.mode == "base" and reference_identity.adapter_snapshot_id is None,
            "Fresh production loop requires same-base disabled-adapter reference")
    pairs = tuple(pairs)
    require(pair_sampling in {LEGACY_PAIR_SAMPLING, DOMAIN_BALANCED_PAIR_SAMPLING},
            "Unknown explicit pair sampling mode")
    balanced_pairs = pair_sampling == DOMAIN_BALANCED_PAIR_SAMPLING
    deck = DomainBalancedPairDeck(pairs, seed=seed) if balanced_pairs else GlobalPairDeck(len(pairs), seed=seed)
    if balanced_pairs:
        balanced_pair_sampling_configuration(seed=seed, global_draws=pairs_per_step)
    require(type(pairs_per_step) is int and pairs_per_step > 0, "Positive global pair batch required")
    require("pair_sampling" not in (config or {}),
            "Pair sampler metadata is owned by the explicit training-loop option")
    output = Path(output)
    prepare_training_output(output, rank=rank, world_size=world_size, distributed=distributed)
    public_config = dict(config or {})
    public_config.update(algorithm=ALGORITHM, reference=asdict(reference_identity), beta=beta,
        learning_rate=learning_rate, weight_decay=0.0, max_grad_norm=1.0, steps=steps,
        pairs_per_step=pairs_per_step, seed=seed, lora_rank=16, lora_alpha=32,
        lora_target_modules=["q_proj", "v_proj", "o_proj"], lora_dropout=0.0,
        sequence_score="sum-all-actual-native-thought-close-JSON-EOS-token-logprobs",
        gradient_weighting="global-equal-pair", fresh_run=True, resume_supported=False,
        reference_models_loaded=0, warm_start=False, measured_KL=False,
        training_method=DPO_METHOD, pair_contract=PAIR_CONTRACT, minimum_reward_gap=0.05,
        candidate_contract="native-final-preference-candidate-v1",
        reward_contract=PER_COLLECTION_RECEIPTS if any(p.get("reward_version") == LITERAL_PROTOCOL
            for p in public_config.get("collections", [])) else REWARD_CONTRACT)
    if balanced_pairs:
        public_config["pair_sampling"] = balanced_pair_sampling_configuration(
            seed=seed, global_draws=pairs_per_step)
    started = time.monotonic()
    with TrainingJournal(output, rank, sync=sync_artifacts) as journal:
        def record(**value):
            line = journal.record(**value)
            if rank == 0:
                print(line, flush=True)
        record(event="start", world_size=world_size, **public_config)
        try:
            if world_size > 1:
                layout = [(n, tuple(p.shape), str(p.dtype)) for n, p in model.named_parameters() if p.requires_grad]
                layouts = [None] * world_size
                distributed.all_gather_object(layouts, layout)
                require(all(value == layout for value in layouts), "Trainable LoRA layouts differ across ranks")
                for parameter in model.parameters():
                    if parameter.requires_grad:
                        distributed.broadcast(parameter.data, src=0)
            owned = pairs[rank::world_size]
            sync = (lambda parameters, world: synchronize_pair_gradients(parameters, world,
                    distributed=distributed)) if world_size > 1 else None
            trainer = OfflineNativeDPOTrainer(model, owned, reference_identity,
                logprob_fn=logprob_fn, beta=beta, world_size=world_size, gradient_sync=sync,
                optimizer_factory=lambda values: torch.optim.AdamW(values, lr=learning_rate, weight_decay=0.0),
                allow_repeated_pairs=balanced_pairs)
            record(event="reference_cached", local_pairs=len(owned),
                   reference_mode=reference_identity.mode, cache_before_optimizer_updates=True,
                   second_base_model_loaded=False)

            def save(name, step):
                journal.checkpoint_boundary()
                status = [None]
                if world_size > 1:
                    distributed.barrier()
                if rank == 0:
                    try:
                        def writer(destination):
                            model.save_pretrained(destination)
                            tokenizer.save_pretrained(destination)
                            state = {"optimizer": trainer.optimizer.state_dict(), "optimizer_steps": step,
                                "torch_rng": torch.get_rng_state(),
                                **({"sampler_state": deck.state_dict()} if balanced_pairs else {
                                    "sampler_rng": deck.rng.getstate(), "sampler_deck": list(deck.deck)}),
                                "resume_supported": False}
                            if next(model.parameters()).device.type == "cuda":
                                state["cuda_rng"] = torch.cuda.get_rng_state()
                            torch.save(state, destination / "training_state.pt")
                            (destination / "training_config.json").write_text(
                                json.dumps(public_config, allow_nan=False, indent=2) + "\n", encoding="utf-8")
                        publish_checkpoint(output, name, step, writer, sync=sync_artifacts)
                    except Exception as error:
                        status[0] = type(error).__name__
                if world_size > 1:
                    distributed.broadcast_object_list(status, src=0)
                if status[0] is not None:
                    raise RuntimeError("DPO checkpoint publication failed: " + status[0])

            if save_initial:
                save("initial", 0)
            for step in range(1, steps + 1):
                begin = time.monotonic()
                selected = [deck.next(pairs_per_step) if rank == 0 else None]
                if world_size > 1:
                    distributed.broadcast_object_list(selected, src=0)
                local_pairs = [pairs[index] for index in selected[0] if index % world_size == rank]
                report = collective_dpo_step(trainer, local_pairs, distributed=distributed)
                require(not report["optimizer_step_skipped"] and trainer.optimizer_steps == step,
                        "Nonempty sampled global update must advance exactly once")
                record(event="step", step=step, seconds=time.monotonic() - begin, **report,
                       **({"pair_sampling_statistics": deck.describe(selected[0])} if balanced_pairs else {}))
                if save_every and step % save_every == 0:
                    save(f"step_{step:04d}", step)
            save("final", steps)
            record(event="finished", optimizer_steps=trainer.optimizer_steps,
                   elapsed=time.monotonic() - started)
            finalize_training(journal, steps, world_size=world_size, distributed=distributed)
            return trainer
        except BaseException as error:
            if not journal.stream.closed:
                record(event="failed", exception_type=type(error).__name__)
            raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", required=True)
    collection_mode = result.add_mutually_exclusive_group(required=True)
    collection_mode.add_argument("--approved-freeze", help="Independently approved single freeze directory")
    collection_mode.add_argument("--approved-collections", help="Explicit approved multi-collection manifest JSON")
    result.add_argument("--generation-dir", help="Already-closed G4 generation journals")
    result.add_argument("--generation-world-size", type=int, default=1)
    result.add_argument("--train-rewards", help="Separately approved offline TRAIN receipts")
    result.add_argument("--reward-version", type=public_id)
    result.add_argument("--train-reward", type=_strict_json,
        help="Complete JSON literal TRAIN reward configuration; requires explicit new reward version")
    result.add_argument("--expected-source")
    result.add_argument("--training-source-commit", required=True,
                        help="Separately approved actual committed clean training runtime")
    result.add_argument("--base-derivative-id", required=True, type=public_id)
    result.add_argument("--tokenizer-id", required=True, type=public_id)
    result.add_argument("--output-dir", required=True, help="Fresh or empty artifact directory; no resume")
    result.add_argument("--steps", type=int, default=20)
    result.add_argument("--pairs-per-step", type=int, default=4,
                        help="Global pairs per update; balanced mode counts draws including explicit repetitions")
    result.add_argument("--pair-sampling", choices=(LEGACY_PAIR_SAMPLING, DOMAIN_BALANCED_PAIR_SAMPLING),
                        default=LEGACY_PAIR_SAMPLING,
                        help="Opt in to half-CUAD/half-Web fixed draw batches; carries tails and can repeat pairs")
    result.add_argument("--learning-rate", type=float, default=1e-6)
    result.add_argument("--beta", type=float, default=0.1)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--save-every", type=int, default=0)
    result.add_argument("--save-initial", action="store_true")
    result.add_argument("--sync-artifacts", action="store_true")
    result.add_argument("--logprob-backend", choices=("original", "qwen3-moe-chunked"),
                        default="original", help="Experimental opt-in bounded-head scorer")
    result.add_argument("--logprob-head-chunk-tokens", type=int, default=None,
                        help="Positive head chunk size; requires explicit chunked backend (default 128)")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    logprob_fn, scoring_backend = resolve_logprob_backend(
        args.logprob_backend, args.logprob_head_chunk_tokens)
    training_source = training_source_identity(args.training_source_commit)
    require(args.approved_collections is not None or (isinstance(args.expected_source, str)
        and re.fullmatch(r"[0-9a-f]{40}", args.expected_source)), "Explicit approved 40-hex collection source required")
    require(type(args.generation_world_size) is int and args.generation_world_size > 0,
            "Positive generation world size required")
    require(args.steps > 0 and args.pairs_per_step > 0 and args.save_every >= 0
            and math.isfinite(args.beta) and args.beta > 0
            and math.isfinite(args.learning_rate) and args.learning_rate > 0,
            "Positive finite optimizer configuration required")
    if args.pair_sampling == DOMAIN_BALANCED_PAIR_SAMPLING:
        balanced_pair_sampling_configuration(seed=args.seed, global_draws=args.pairs_per_step)
    rank, local, world = (int(os.environ.get(key, default)) for key, default in
                         (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1")))
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
    try:
        torch.manual_seed(args.seed)
        payload = [None]
        if rank == 0:
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(args.model)
                pairs, metadata = load_training_pairs(args, tokenizer)
                require(metadata["eligible_pairs"] > 0, "Nonempty preference dataset required")
                payload[0] = {"pairs": pairs, "metadata": metadata}
            except Exception as error:
                payload[0] = {"error_type": type(error).__name__}
        if world > 1:
            dist.broadcast_object_list(payload, src=0)
        require("error_type" not in payload[0],
                "TRAIN freeze/closure/reward validation failed: " + payload[0].get("error_type", ""))
        if args.pair_sampling == DOMAIN_BALANCED_PAIR_SAMPLING:
            DomainBalancedPairDeck(payload[0]["pairs"], seed=args.seed)
        model, tokenizer = load_policy_model(args.model, adapter=None, trainable=True, rank=local,
                                             lora_rank=16, quantization="nf4")
        report = validate_fresh_nf4_lora(model)
        from dcc_native_sft import identity_from_manifest
        for collection in payload[0]["metadata"]["collections"]:
            frozen_identity = dict(collection["collection_native_identity"], generation_seed=0)
            loaded_identity = identity_from_manifest(frozen_identity, collection["collection_source_commit"])
            require(loaded_identity.manifest(tokenizer) == frozen_identity,
                    "Actually loaded training tokenizer differs from a frozen collection identity")
        identity = ReferenceIdentity(args.base_derivative_id, args.tokenizer_id, QUANTIZATION_ID, "base")
        run_dpo_loop(model, tokenizer, payload[0]["pairs"], identity,
            output=args.output_dir, steps=args.steps, pairs_per_step=args.pairs_per_step,
            learning_rate=args.learning_rate, beta=args.beta, seed=args.seed,
            rank=rank, world_size=world, config=dict(payload[0]["metadata"], quantization=report,
                training_source=training_source, scoring_backend=scoring_backend),
            logprob_fn=logprob_fn, pair_sampling=args.pair_sampling,
            save_every=args.save_every, save_initial=args.save_initial, sync_artifacts=args.sync_artifacts)
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
