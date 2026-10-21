# Optional balanced DPO pair draws

The default `--pair-sampling global-deck` keeps the original shuffled deck,
partial epoch tail, reference ownership and logging. Existing frozen runs keep
that default. This option does not change accepted-trajectory SFT.

For a future independently configured DPO experiment, add:

```text
--pair-sampling domain-balanced-fixed-batch --pairs-per-step 8
```

This requires both CUAD and WideSearch TRAIN preference pairs. Each update has
four CUAD draws and four WideSearch draws. Each domain maintains an independently
seeded shuffled cycle. When a cycle ends, its remaining pairs are consumed and a
new cycle fills the rest of the batch. No pair is dropped. Each pair appears once
per complete cycle within its domain; smaller domains therefore repeat more often.

A domain with three pairs still supplies four draws. Those four draws include a
repeated pair and are not four independent examples. Repetitions are explicitly
combined into one loss/gradient replay per unique pair, with its multiplicity.
The result equals the mean DPO objective evaluated separately for all eight draws.
Raw full-completion sequence sums, the disabled-adapter base reference, optimizer,
clipping and LoRA configuration retain their existing definitions.

Dataset index modulo world size continues to own each pair and its immutable
reference cache. A rank may receive several draws or none. Every rank participates
in gradient synchronization. Local gradients use `multiplicity * world / draws`;
the existing sum-and-divide-by-world synchronization gives each global draw weight
`1 / draws`. Replicas receive the same synchronized gradient and Adam update.
Reference caching still scores only the rank's original unique owned pairs.
The existing `gradient_weighting="global-equal-pair"` field therefore means
equal weights per draw in this mode; a repeated pair receives its multiplicity.

The saved configuration includes the explicit sampler version, weights, seed,
batch size and repetition policy. Every step records global draw count, unique
pair count, domain counts and pair-index multiplicities in
`pair_sampling_statistics`; `global_pairs` remains the legacy field name and
counts draws in this mode. `local_pair_draws` and `local_unique_pairs` distinguish
local repetition. Checkpoints store the domain RNG/deck state for inspection;
resume remains unsupported. Default configurations, step fields and checkpoint
sampler fields retain their existing form.
Adapter adoption checks the optional `pair_sampling` object against the canonical
configuration reconstructed from the saved seed and global batch size, including
domain weights and the draw denominator. Legacy approvals without this object
retain their existing behavior.

This is an experimental sampling strategy, not evidence of improved benchmark
scores. Prefer collecting more unique Web TRAIN pairs to repeatedly reusing a
small Web subset. Any experiment needs a fresh source/configuration and the same
base-versus-trained evaluation protocol; existing frozen experiments are unchanged.

CPU regression:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest -q -p no:cacheprovider tests/test_dcc_balanced_dpo_sampling.py
```

The tests use the observed G5 domain counts (14 CUAD / 3 Web) with invented tokens.
They check coverage, determinism, fixed batches, explicit repetitions, direct
per-draw gradient/loss/Adam-moment equivalence, owner-only reference caching,
empty-rank participation, whole-loop logging/checkpointing and legacy behavior.
The distributed arithmetic test runs the real synchronization helper against an
in-process collective boundary; it starts no GPU, worker process or network service.
