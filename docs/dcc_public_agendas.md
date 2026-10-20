# Public CUAD TRAIN agendas

`dcc_prepare_cuad_public_agendas.py` collects actual CUAD tool trajectories from
an exact public projection containing only task ID, question, and document.
The planner rejects a full labeled task. Optional `--max-tasks` selection uses
only a seeded task-ID digest with a separate `public-subset-v1` salt. Selection
IDs are written before the entire public agenda journal is collected and closed;
only then can TRAIN labels attach targets. Official splits are rejected.

The fixed agenda family uses search/read, search/read/read, or
search/search/read, with real public retrieval windows and no fabricated padding.
On short contracts, duplicate actual source ranges are replaced by a genuine
alternate search or omitted. Every agenda is replayed through `EpisodeRunner`
with exact prompt and tool-observation comparisons. Final targets require all
gold spans in actual observations and in the encoded prompt. Unsupported
positive examples retain their real search/read supervision; they receive no
negative final target. Real negatives retain their actual empty-answer final.

Example using portable, explicitly configured locations:

```sh
python scripts/dcc_prepare_cuad_public_agendas.py \
  --data "$CUAD_TRAIN_DATA" --retrieval-profile "$CUAD_RETRIEVAL_PROFILE" \
  --tokenizer "$TOKENIZER" --output-dir "$PUBLIC_AGENDA_OUTPUT" \
  --max-tasks 4096 --seed 47 --passes 1
```

## Partial-stage buckets are opt in

The trainer keeps its existing strict requirement by default: every local
(label, category) tool bucket must contain search, read, and final records.
`--cuad-tool-partial-stages` permits a positive bucket with exactly search and
read records. A negative bucket still needs all three stages; a missing search
or read remains an error. The original global stage index selects cyclically
from the selected bucket's available stages in the original order. Complete
buckets keep the original three-stage cycle and seeded choices. Eligible logs
list each partial bucket's missing stages, and step logs report actual stage
counts.

The flag changes the sampling recipe: positive tool-only buckets contribute
search/read supervision without an invented final answer. It does not change
completion-only loss, equal example weighting, optimizer, task sharding,
negative sampling fraction, or Web examples. Enable it explicitly for both
training and preflight when retaining unsupported-positive tool supervision:

```sh
python scripts/dcc_preflight_public_agendas.py \
  --cuad-records "$PUBLIC_AGENDA_OUTPUT/teacher_train.jsonl" \
  --web-records "$WEB_TEACHER_RECORDS" --tokenizer "$TOKENIZER" \
  --feature-audit "$PUBLIC_AGENDA_OUTPUT/final-state-feature-audit.json" \
  --cuad-tool-partial-stages --output "$PREFLIGHT_OUTPUT"
```

Preflight uses the actual `MixedSampler` for eight ranks and 128 steps, with up
to four examples per rank/step, reporting budget skips, retained and sampled
stages, missing-stage buckets, and TRAIN feature correlations. `--rank` may be
repeated to inspect selected actual ranks. Feature correlations describe TRAIN
only and do not establish benchmark improvement or independent validation.
The unchanged final target uses constant confidence 0.9; this data construction
does not address confidence calibration.
