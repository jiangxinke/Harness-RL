# Offline native final TRAIN rewards

`scripts/dcc_score_native_preferences.py` closes the gap between native G4
collection and `dcc_native_dpo_train.py`. It loads only the actual tokenizer,
never model weights, and scores only separately frozen CUAD TRAIN or independent
Web TRAIN collections. It does not generate candidates or change either
benchmark evaluator.

Approve a runtime manifest using this structure:

```json
{
  "version": "native-final-train-reward-inputs-v1",
  "base_derivative_id": "approved-base",
  "tokenizer_id": "approved-tokenizer",
  "collections": [
    {
      "collection_id": "cuad-16k",
      "approved_freeze": "runtime-cuad-freeze",
      "generation_dir": "runtime-cuad-g4",
      "generation_world_size": 8,
      "expected_source": "0000000000000000000000000000000000000000",
      "original_source_dir": "runtime-original-cuad-collection",
      "public_train": "runtime-public-train",
      "train_labels": "runtime-TRAIN-labels",
      "reward_version": "cuad-native-final-existing-f1-verbatim-v1"
    },
    {
      "collection_id": "web-32k",
      "approved_freeze": "runtime-web-freeze",
      "generation_dir": "runtime-web-g4",
      "generation_world_size": 8,
      "expected_source": "1111111111111111111111111111111111111111",
      "original_source_dir": "runtime-original-web-collection",
      "public_train": "runtime-public-train",
      "train_labels": "runtime-TRAIN-labels",
      "reward_version": "web-native-final-row75-item25-original-profile-evidence-v1"
    }
  ]
}
```

Runtime locations and example commits are placeholders. `public_train` may be
the same full public task file for both domains. `original_source_dir` must be
the original raw closed collection used by that freeze's export. Its expected
source profile, receipt and world size come from the independently approved
`export_source_approval.json` adjacent to the freeze.

Before any TRAIN label file is opened, every G4 rank and every original raw
collection must close. Cross-collection prefix/task/attempt/actual phase-request
ownership and shared native base identities must also validate. The scorer
joins each frozen prefix through `lineage.jsonl` to the original raw source
request and its actual `episode.public_task`. That public task must equal the
selected task from `public_train` exactly, including the full CUAD document.
The frozen messages, original schema, final schema and actual fetched source
URLs are replayed through the existing public request validator. Original
completion correctness, reward and success status never select or score the
new candidates; rejected original finals remain eligible when their approved
public history is valid.

CUAD uses the existing `cuad_task_metrics(...)["f1"]` multiplied by the same
whitespace-normalized verbatim-document span fraction as `EpisodeRunner`.
Empty predictions have grounding factor1: a negative empty/empty answer scores1,
a positive empty answer scores0, and a negative false positive scores0.
Confidence does not change the existing F1 calculation.

A separate experiment can explicitly select
`cuad-native-final-f1-verbatim-brier25-v2` in the CUAD collection manifest.
It uses `F1 * grounding * (1 - 0.25 * mean((confidence - target)^2))`.
Each predicted span's target is 1 exactly when the existing official CUAD
span matcher matches a gold span, including its Parties substring rule.
Empty predictions have zero Brier penalty. The factor remains in [0.75, 1],
so zero-reward answers stay zero and negative empty answers keep reward1.
The audit records the original reward, Brier value and applied factor.
This provides confidence-sensitive TRAIN preferences; it is not the global
benchmark AUPR and does not establish a quality improvement. The default
version and both benchmark evaluators are unchanged. Freeze a new reward
manifest and declare the actual scorer source when comparing versions;
never replace an ongoing experiment's reward receipts. Pass the same explicit
version to DPO, whose loader rejects receipts from a different reward version.

Web replays actual historical search/fetch observations into the existing
`_WebEvidence` guard. A guard rejection scores0 without repairing the answer.
Otherwise the reward is `0.75 * f1_by_row + 0.25 * f1_by_item` from the existing
`evaluate_widesearch` and `web_train_evaluation`. A complete original TRAIN
scoring profile retains its exact entity key and pipeline. Required-only TRAIN
profiles retain the existing strict whole-row compound key across all required
columns. In that case row and item F1 coincide; their equality is reported
honestly. No entity key is inferred and no held-out evaluator is changed.

```bash
python scripts/dcc_score_native_preferences.py \
  --approved-collections "$DPO_REWARD_INPUTS" \
  --tokenizer "$DPO_TOKENIZER" --output-dir "$DPO_FRESH_REWARD_ARTIFACT_DIR"
```

The output directory must be new. Each numeric `collection_000` subdirectory
contains a complete `receipts.jsonl`, component-only `audit.jsonl` and
`reward_complete.json`. The top-level completion file maps public collection IDs
to relative receipt files and reports full attempt/valid/failed/reward/guard
counts. Files contain one complete JSON object per newline. Only physically
valid candidates receive receipts; failed G4 attempts remain in the original
generation audit and full counts. There are no retries, replacement candidates
or reward-dependent prefix selection. Pass each receipt file and its exact
fixed reward version into the corresponding independently approved DPO
collection manifest. Public receipts/provenance exclude runtime locations.

CPU regression fixtures use invented TRAIN labels, actual native fake-engine
original/G4 closure, and the unchanged pair loader. They are protocol and
arithmetic evidence, not real model or benchmark quality results.
