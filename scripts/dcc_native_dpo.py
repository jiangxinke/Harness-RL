"""Prospective offline DPO LoRA trainer: one base, immutable scalar reference.

The opt-in production CLI is dcc_native_dpo_train.py. This math module has no
collector, generation, model loading, evaluation, or second base model.
This is offline preference optimization, not RL or on-policy importance sampling.
Sequence scores are SUMS over every physical completion ID, including native
thought, real </think>, JSON and its actual EOS. No length normalization.
"""
from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from dcc_native_dpo_data import PreferencePair

SCORING_VERSION = "raw-causal-full-completion-token-loglikelihood-sum-v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_deterministic_scoring(model):
    """Eval/no-grad previews and train-mode checkpoint replay must agree."""
    dropout_types = (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d,
                     torch.nn.Dropout3d, torch.nn.AlphaDropout, torch.nn.FeatureAlphaDropout)
    for name, module in model.named_modules():
        if isinstance(module, dropout_types) and module.p != 0:
            raise ValueError("Offline DPO replay requires zero dropout: " + name)
        if getattr(module, "attention_dropout", 0.0) != 0:
            raise ValueError("Offline DPO replay requires zero attention dropout: " + name)


@dataclass(frozen=True)
class ReferenceIdentity:
    """Caller-approved identities, never server paths or a base/SFT ambiguity."""
    base_derivative_id: str
    tokenizer_id: str
    quantization_id: str
    mode: str  # base OR initial_adapter; disable_adapter() implements base only.
    adapter_snapshot_id: str | None = None
    scoring_version: str = SCORING_VERSION

    def __post_init__(self):
        _require(all(isinstance(v, str) and v.strip() for v in (
            self.base_derivative_id, self.tokenizer_id, self.quantization_id)),
            "Explicit base derivative, tokenizer and quantization identities required")
        _require(self.scoring_version == SCORING_VERSION, "Raw sum reference scoring required")
        _require(self.mode in {"base", "initial_adapter"}, "Explicit reference mode required")
        _require((self.mode == "base" and self.adapter_snapshot_id is None)
                 or (self.mode == "initial_adapter" and isinstance(self.adapter_snapshot_id, str)
                     and bool(self.adapter_snapshot_id.strip())),
                 "Initial adapter reference requires its snapshot identity; base reference has none")


@dataclass(frozen=True)
class CachedReference:
    pair_key: tuple
    identity: ReferenceIdentity
    chosen_sum: float
    rejected_sum: float

    def __post_init__(self):
        _require(math.isfinite(self.chosen_sum) and math.isfinite(self.rejected_sum),
                 "Finite immutable reference sums required")


def _sequence_sum(model, sequence, logprob_fn):
    values = logprob_fn(model, list(sequence.prompt), list(sequence.completion))
    _require(values.ndim == 1 and values.numel() == len(sequence.completion),
             "Exact recorded completion-token scores required")
    score = values.float().sum()
    if not torch.isfinite(score):
        raise FloatingPointError("Non-finite sequence likelihood; refusing adapter update")
    return score


def cache_reference_scores(model, pairs, identity, *, logprob_fn):
    """Cache before any optimizer mutation; initial_adapter keeps adapter enabled.

    References use the actual same quantized base as training. A warm SFT
    reference is scored with that original adapter enabled, not disable_adapter.
    A persisted cache must retain this identity AND the full exact pair key.
    """
    validate_deterministic_scoring(model)
    pairs = tuple(pairs)
    _require(len({pair.key for pair in pairs}) == len(pairs), "Duplicate preference pairs")
    if identity.mode == "base":
        _require(callable(getattr(model, "disable_adapter", None)),
                 "Base reference requires explicit PEFT disable_adapter context")
        context = model.disable_adapter()
    else:
        _require(bool(getattr(model, "peft_config", None)),
                 "Initial adapter reference requires the enabled loaded PEFT adapter")
        _require(not any(getattr(module, "disable_adapters", False) is True
                         for module in model.modules()),
                 "Initial adapter reference cannot silently score a disabled adapter")
        context = nullcontext()
    was_training = model.training
    model.eval()
    observed = []
    try:
        with context, torch.no_grad():
            for pair in pairs:
                observed.append(torch.stack([
                    _sequence_sum(model, pair.chosen.sequence, logprob_fn),
                    _sequence_sum(model, pair.rejected.sequence, logprob_fn),
                ]))
        rows = torch.stack(observed).cpu().tolist() if observed else []
    finally:
        model.train(was_training)
    return {pair.key: CachedReference(pair.key, identity, row[0], row[1])
            for pair, row in zip(pairs, rows)}


def dpo_loss(chosen_sum, rejected_sum, reference, *, beta):
    _require(type(beta) in (int, float) and math.isfinite(beta) and beta > 0,
             "DPO beta must be positive and finite")
    margin = beta * ((chosen_sum - reference.chosen_sum)
                     - (rejected_sum - reference.rejected_sum))
    return -F.logsigmoid(margin)


def backward_pair_low_memory(model, pair, reference, *, beta, logprob_fn, scale=1.0):
    """Exact DPO chain rule; only one completion activation graph is live.

    Under unchanged dropout-free parameters, c=beta*sigmoid(-margin). Replaying
    chosen with -c and rejected with +c produces the direct DPO gradient. The
    detached preview and two replay forwards trade compute for activation memory.
    Callers MUST delay optimizer mutation until all pair replay has finished.
    """
    _require(reference.pair_key == pair.key, "Reference score belongs to different exact IDs")
    _require(type(scale) in (int, float) and math.isfinite(scale) and scale > 0,
             "Positive finite global pair scale required")
    _require(type(beta) in (int, float) and math.isfinite(beta) and beta > 0,
             "DPO beta must be positive and finite")
    model.eval()
    with torch.no_grad():
        chosen = _sequence_sum(model, pair.chosen.sequence, logprob_fn)
        rejected = _sequence_sum(model, pair.rejected.sequence, logprob_fn)
        margin = beta * ((chosen - reference.chosen_sum) - (rejected - reference.rejected_sum))
        coefficient = beta * torch.sigmoid(-margin)
        loss = -F.logsigmoid(margin)
        observation = torch.stack([loss, margin, chosen, rejected]).detach()
    # Train mode enables the existing non-reentrant checkpoint implementation;
    # its deterministic policy equality is guarded before training/cache setup.
    model.train()
    (-coefficient * scale * _sequence_sum(model, pair.chosen.sequence, logprob_fn)).backward()
    (coefficient * scale * _sequence_sum(model, pair.rejected.sequence, logprob_fn)).backward()
    return observation


class OfflineNativeDPOTrainer:
    """Minimal owner of one LoRA model, an immutable cache and update boundary.

    Multi-rank caller supplies a collectively verified global_pair_count and
    gradient_sync(parameters, world_size), which SUMS then DIVIDES by world.
    Coefficients world/global_pairs produce the global equal-pair objective,
    including zero-local-pair ranks. The constructor caches the approved reference
    before optimizer construction. This class does not load or generate data.
    """
    def __init__(self, model, pairs, reference_identity, *, logprob_fn,
                 optimizer_factory, beta=0.1, gradient_sync=None, world_size=1,
                 allow_repeated_pairs=False):
        _require(type(world_size) is int and world_size >= 1, "Positive world size required")
        _require(world_size == 1 or callable(gradient_sync), "Distributed gradients require explicit sync")
        _require(type(beta) in (int, float) and math.isfinite(beta) and beta > 0,
                 "DPO beta must be positive and finite")
        validate_deterministic_scoring(model)
        self.model, self.logprob_fn, self.beta = model, logprob_fn, beta
        self.reference_identity, self.world_size = reference_identity, world_size
        self.gradient_sync = gradient_sync
        _require(type(allow_repeated_pairs) is bool, "Explicit repeated-pair mode must be boolean")
        self.allow_repeated_pairs = allow_repeated_pairs
        pairs = tuple(pairs)
        _require(all(isinstance(pair, PreferencePair) for pair in pairs), "Validated native pairs required")
        self.references = cache_reference_scores(model, pairs, reference_identity, logprob_fn=logprob_fn)
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        _require(bool(self.parameters), "Trainable LoRA parameters required")
        self.optimizer = optimizer_factory(self.parameters)
        self.optimizer_steps = 0

    def step(self, pairs, *, global_pair_count=None):
        pairs = tuple(pairs)
        multiplicities = Counter(pairs) if self.allow_repeated_pairs else None
        if not self.allow_repeated_pairs:
            _require(len({pair.key for pair in pairs}) == len(pairs), "Duplicate pair in optimizer batch")
        if global_pair_count is None:
            _require(self.world_size == 1, "Every distributed rank must supply a verified global pair count")
            global_pair_count = len(pairs)
        _require(type(global_pair_count) is int and global_pair_count >= len(pairs)
                 and (self.world_size != 1 or global_pair_count == len(pairs)),
                 "Incorrect global pair denominator")
        self.optimizer.zero_grad(set_to_none=True)
        if global_pair_count == 0:
            return {"algorithm": "offline_native_final_DPO; not RL", "optimizer_step_skipped": True,
                    "local_pairs": 0, "global_pairs": 0}
        observations = []
        weighted_pairs = multiplicities.items() if self.allow_repeated_pairs else ((pair, 1) for pair in pairs)
        for pair, multiplicity in weighted_pairs:
            _require(pair.key in self.references, "Pair is absent from immutable reference cache")
            reference = self.references[pair.key]
            _require(reference.identity == self.reference_identity, "Reference identity mismatch")
            observations.append(backward_pair_low_memory(self.model, pair, reference,
                beta=self.beta, logprob_fn=self.logprob_fn,
                scale=multiplicity * self.world_size / global_pair_count) * multiplicity)
        if self.gradient_sync is not None:
            self.gradient_sync(self.parameters, self.world_size)
        norm = torch.nn.utils.clip_grad_norm_(self.parameters, 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("Non-finite DPO adapter gradient; refusing update")
        self.optimizer.step()
        self.optimizer_steps += 1
        rows = torch.stack(observations).cpu().tolist() if observations else []
        # Sums + counts allow global logging without averaging rank means.
        return {"algorithm": "offline_native_final_DPO; not RL", "optimizer_step_skipped": False,
                "optimizer_steps": self.optimizer_steps, "local_pairs": len(pairs),
                "global_pairs": global_pair_count, "gradient_norm": float(norm),
                "dpo_loss_sum": sum(row[0] for row in rows),
                "reference_adjusted_margin_sum": sum(row[1] for row in rows),
                "chosen_raw_sum": sum(row[2] for row in rows),
                "rejected_raw_sum": sum(row[3] for row in rows),
                "measured_KL": False,
                **({"local_pair_draws": len(pairs), "local_unique_pairs": len(multiplicities)}
                   if self.allow_repeated_pairs else {})}
