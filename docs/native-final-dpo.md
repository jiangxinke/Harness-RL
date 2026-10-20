# Native final-decision DPO

`scripts/dcc_native_dpo_train.py` is a separate opt-in offline TRAIN preference
trainer. Existing accepted-trajectory SFT entrypoints, sampler defaults and
accepted-answer/polarity gates retain their previous behavior. Both methods
reuse `validate_native_physical` for original prompt IDs, phase ownership, actual
closing/EOS IDs, unchanged public final schema and source identity checks.

Generation must use independently approved public TRAIN final prefixes and
close every G4 rank journal before any offline reward file is opened. All
collections must close and pass common base/tokenizer/template checks and
cross-collection prefix, task, attempt and actual phase-request ownership
checks before the first reward file is opened. Failed recorded phase requests
are included in that ownership check. The trainer calls `load_frozen` and `read_closed_g4` from the prospective collector
before reading reward receipts. The collection freeze and adjacent public
prefix file must both have been independently approved. No official WideSearch
or held-out split enters training. An explicit final-decision schema
intervention, when approved, remains identified by its collection harness and
schema policy.

Each valid closed-generation candidate needs exactly one separate receipt:

```json
{"contract":"native-final-train-reward-receipt-v1","task_id":"task-1","trace_id":"trace-1","prefix_id":"prefix-1","harness_id":"approved-harness","reward_version":"separately-approved-version","value":0.8,"generation_closed_before_scoring":true,"split":"train"}
```

The receipt journal uses one complete JSON object per newline. The trainer
does not implement or select a reward scorer/version. A proposal or synthetic
CPU reward test is not approval of a reward profile. Receipt identities must
match the public prefix and physical trace exactly; all rewards must be finite
and in `[0,1]`. Valid candidates alone enter pairing. Fewer than two valid
candidates omit that prefix. Best/worst rewards must differ by strictly more
than `0.05`; ties and smaller gaps omit the prefix. Missing/failed attempts are
retained by the collector and never fabricated or refilled by reward.

The production loader is the existing approved full-expert NF4 loader: one
base, double quantization, BF16 compute, fresh rank16/alpha32 q/v/o LoRA and
zero dropout. The fixed reference is the same quantized base with the adapter
disabled, cached before optimizer updates. No second model, warm adapter,
resume, synthetic thought, target rewrite, truncation or length normalization
is supported. Sequence scores sum raw log probabilities over every recorded
thought, real closing token, JSON and actual terminal EOS ID. The initial
defaults are beta `0.1`, AdamW learning rate `1e-6`, no weight decay and gradient
norm clipping at `1.0`. DPO loss/margin is not a measured KL estimate.

The low-memory chain rule previews each chosen/rejected score without a graph,
then replays each completion separately with the exact detached DPO derivative.
Only one completion activation graph is live. This relies on unchanged model
parameters and zero dropout; it trades compute for activation memory. The
original `completion_logprobs` scorer is used. There is no chunked scorer flag.

The batch size is a maximum number of distinct global pairs. A shared shuffled
deck consumes its final short batch before starting another epoch. Static
pair-to-rank ownership permits caching only local references. An actual
collective sums current local pair counts; gradients are summed on every rank
and divided by this global denominator. Empty ranks materialize zero gradients
and take the same collectives and optimizer update. A globally empty step
skips Adam entirely. The production deck rejects an empty dataset.

Run from a committed, clean source checkout using a newly approved training
commit. `--expected-source` identifies the actual source that collected the
offline sequences. `--training-source-commit` independently identifies the
actual new training runtime. The latter is checked against HEAD, tracked new
trainer/helper files and an unchanged runtime source tree. An old checkout
plus an uncommitted patch cannot claim the old commit as unchanged.

```bash
torchrun --standalone --nproc_per_node=8 scripts/dcc_native_dpo_train.py \
  --model "$DPO_MODEL" --approved-freeze "$DPO_FREEZE" \
  --generation-dir "$DPO_GENERATION" --generation-world-size 8 \
  --train-rewards "$DPO_TRAIN_REWARDS" --reward-version "$DPO_REWARD_VERSION" \
  --expected-source "$DPO_COLLECTION_SOURCE" \
  --training-source-commit "$DPO_TRAINING_SOURCE" \
  --base-derivative-id "$DPO_BASE_ARTIFACT_ID" --tokenizer-id "$DPO_TOKENIZER_ARTIFACT_ID" \
  --output-dir "$DPO_FRESH_ARTIFACT_DIR" --steps 20 --pairs-per-step 4
```

For separately approved CUAD16K and Web32K collections, pass
`--approved-collections "$DPO_COLLECTIONS_MANIFEST"` in place of the single
freeze/generation/reward/source arguments. The runtime manifest is:

```json
{
  "version": "native-final-dpo-collections-v1",
  "base_derivative_id": "approved-base",
  "tokenizer_id": "approved-tokenizer",
  "collections": [
    {
      "collection_id": "cuad-16k",
      "approved_freeze": "runtime-cuad-freeze",
      "generation_dir": "runtime-cuad-generation",
      "generation_world_size": 8,
      "train_rewards": "runtime-cuad-reward-receipts",
      "expected_source": "0000000000000000000000000000000000000000",
      "reward_version": "independently-approved-CUAD-reward"
    },
    {
      "collection_id": "web-32k",
      "approved_freeze": "runtime-web-freeze",
      "generation_dir": "runtime-web-generation",
      "generation_world_size": 8,
      "train_rewards": "runtime-web-reward-receipts",
      "expected_source": "1111111111111111111111111111111111111111",
      "reward_version": "independently-approved-WEB-reward"
    }
  ]
}
```

The example commits and reward versions are placeholders, not approvals. Each
collection retains its complete public freeze, context limits, harness/schema,
source receipt, reward version, all attempt/phase request IDs and selected
pair ownership. CUAD and Web contexts are independently frozen; the merger
preserves them and checks the shared base/tokenizer/template/protocol.

Model/data/output locations are runtime arguments and are excluded from public
training configuration. Supply public artifact identities explicitly. Outputs
use the existing fresh-directory ownership, one JSONL journal per rank,
staged checkpoint publication, closed-log validation and terminal completion
receipt. They include real adapter files, tokenizer, optimizer/RNG/deck state
and the complete public training configuration. Stored state supports audit;
the CLI does not resume it.

For native evaluation, DPO requires a separately approved JSON profile with
exactly three keys: `version: native-final-dpo-adapter-approval-v1`,
`training_configuration` equal to the complete actual saved configuration, and
`checkpoint_optimizer_steps` equal to the actual selected checkpoint step.
Review and approve this profile independently; the trainer never automatically
approves its own checkpoint. Pass it as `--native-lora-dpo-approval` alongside
the ordinary native evaluation/source/derivative options. This opt-in is
mutually exclusive with the accepted-SFT training source/protocol flags and
requires explicit native v3 evaluation. The native evaluation protocol and
its source remain independent of both collection and DPO training sources.

The existing adapter metadata checker and exact safetensors attention
shape/dtype/completeness checks are reused. The DPO branch additionally checks
the honest method, same-base reference, reward and pair contracts, TRAIN-only
closure, exact approved configuration and the approved actual checkpoint step.
For example, a step8 adapter from a configured16-step run is adopted only when
`checkpoint_optimizer_steps` is `8` and the unchanged saved configuration retains
`steps: 16`. Approving step16 cannot adopt step8, and approving step8 cannot
adopt the final step16 adapter. Initial step0 audit checkpoints are rejected. Manifests, runtime bindings, native trace sidecars and paired comparison
reports carry `native_lora_dpo_approval` separately from SFT approval. Omitting
the DPO flag preserves the strict accepted-SFT validator; it rejects DPO.

CPU regressions use invented fixtures and tiny random Qwen MoE+PEFT models.
They verify real optimizer updates/checkpoint journals against direct DPO,
actual eight-rank Gloo normalization with empty ranks, a globally empty Adam
skip, original SFT/header guards, explicit DPO native preflight/wrapping and
paired sidecar validation. Synthetic NF4 metadata in CPU adoption fixtures
does not establish actual NF4 loading, GPU memory, throughput or benchmark
quality. Real GPU/NF4 validation and independently approved rewards remain
required before production training or any score claim.
