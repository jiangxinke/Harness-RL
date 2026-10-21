# Task-first sampling for native SFT

`--native-task-first-coverage` selects tasks before their recorded action stage.
It is an explicit new training objective. Existing default and global stage/task
samplers keep their original behavior. This does not change rollout collection,
selection, recorded tokens, native/source validation, loss, or gradient averaging.

The previous per-stage decks can overexpose tasks with more available stages.
For two positive tasks in one category, one with search/read/final records and
one with a final alone, the synthetic 60-draw control gives 50/10 task exposure
with the original global stage sampler and 30/30 with task-first sampling.
That demonstrates a weighting difference, not a measured benchmark improvement.

The new sampler uses deterministic weighted-deficit domain/polarity quotas,
then a shuffled task deck within each requested pool. Each task cycles its
available final/search/final/read-or-fetch stages, and variants within a stage
use another shuffled deck. Categories have task-uniform weight. Every requested
pool must contain accepted tasks and every accepted task must retain a real
native final. Missing pools fail instead of falling back to another polarity.

`--native-positive-final-first` optionally exposes each positive CUAD task's
real final on its first draw. This changes early stage weights and is recorded
separately. `--web-fraction 0` supports a CUAD-specific fresh adapter; `1`
supports Web-specific training. Both original native input files and the Web
cache sidecar still pass the complete original loader before sampling.

All processes replay the same global planner and return their own logical
rank's batch. Original packing retains deferred examples. Coverage logs count
consumed examples separately from reserved lookahead and are emitted after
successful optimizer updates. Input order, world size, seed and budgets must
match across ranks. Full native examples must fit without truncation or skips.

This mode permits fresh native LoRA training only. Its saved RNG is not a full
planner resume state. It is mutually exclusive with the old global stage/task
mode and `--web-stage-profile`. CPU controls do not establish distributed or
GPU training acceptance, memory sufficiency, or quality gains.
