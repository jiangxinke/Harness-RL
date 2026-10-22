# Opt-in online Web TRAIN groups

The native HF LoRA trainer can mix its existing CUAD TRAIN groups with an
independent, public-only Web TRAIN curriculum with the WideSearch task schema. The default remains
CUAD-only. This path uses TRAIN-only exact row rewards; it does not report a WideSearch
benchmark score or the published judge metric.

Use `--enable-online-web` with all three independent inputs:

```sh
python scripts/dcc_online_grpo.py \
  --model MODEL --train-data cuad_train.jsonl --output-dir OUTPUT \
  --source-commit ACTUAL_COMMITTED_RUNTIME \
  --enable-online-web --web-train-data public_web_train.jsonl \
  --web-train-labels private_web_train_labels.jsonl \
  --web-binding reviewed_web_binding.json \
  --cuad-draws-per-cycle 1 --web-draws-per-cycle 1
```

Public Web records must contain only public task fields, `gold: null` if present,
and required-column metadata. Labels, provenance URLs and source answers cannot
be attached to rollout tasks. The Web deck cycles through shuffled public IDs.
The CUAD deck keeps its existing TRAIN gold positive/negative stratification, so
the complete mixed plan is not label-blind.

A separately reviewed binding uses version
`dcc-hf-online-web-reviewed-binding-v1` and contains:

- `source_commit`: the actual clean runtime commit/export identity;
- `protocols`: the exact object returned by
  `slime_plugins.dcc.online_web.binding_protocols()`;
- `public_train` and `private_train_labels`: each file's exact `{sha256, bytes}`
  identity, as returned by `file_identity(path)`;
- `public_task_ids`: every public TRAIN task ID in input order;
- `source_evidence`: `reviewed: true`, a nonempty `review_record`, positive
  `receipt_pairs` and `lineage_cells`, and a nonempty `inventory` identifying the
  externally reviewed acquisition receipts and exact cell lineage;
- `root_accepted: true`, `training_reward_admission_approved: true`, and a
  nonempty `review_record` referring to the actual review of this binding.

These fields are explicit external review assertions, not authenticated producer
proof. Historical acquisition receipts do not demonstrate native rollout fetches
or runtime admission. A pending manifest or an approval for another source,
reward wrapper, public file or private label file is rejected. A new runtime
commit needs its own review binding. Keep private data and bindings outside Git.
CPU controls do not constitute model/GPU admission.

Each Web group creates fresh `native_traces`, `request_owners`, and
`public_outcomes` JSONL journals under the output directory. The owner records
bind actual completion UUIDs to task, iteration, rank, sample and turn, preserving
original messages, schemas, token IDs and streamed raw behavior thought logprobs.
The trainer must enable the native streaming behavior-score capture; replayed
teacher-forced scores cannot substitute for the actual sampled behavior scores. The trainer drains all episode/model
work and closes all three durable writers, then rereads the closed bytes and
validates exact ownership, original native schemas, real thinking close/EOS,
message encoding and final decoding. Only then is `generation_closed.json`
written and the private label file parsed. Parsed labels are not cached across
subsequent rollouts. Preflight only hashes the label file as opaque bytes.

Reward uses the unchanged literal text/null scorer, public evidence guard,
nonempty string cells, support in actual fetched observations, and support in
the actual encoded final prompt. Substring support is a limited deterministic
check, not semantic entailment. Numeric JSON cells are not converted into strings.
Model failures retain sampled thoughts at zero reward. Any infrastructure or
context failure retains every outcome and discards the whole gradient group.
Timed-out native generations are drained and their actual late completions
remain owned in the closed journals. No retry, sibling filtering or denominator
change is performed.

`--web-train-reward-variant` defaults to `table-grounded-exact-v1`, preserving
the original whole-table grounding gate and scoring receipt. To opt into
per-row grounding for online Web TRAIN, add
`--web-train-reward-variant per-row-grounded-exact-v1` together with
`--enable-online-web`. This option does not change benchmark evaluation, SFT,
DPO, the strict literal scorer, or the original literal label format.

In the per-row variant, the complete answer must first pass strict literal
validation and the unchanged public schema/citation guard. Any malformed table
still receives zero for the entire table. Each predicted row then passes through
the same existing fetched-text and actual-final-prompt grounding function,
using the original full tool history and actual decoded final prompt. Only
eligible rows can be true positives, matched as complete required-column tuples
against gold with strict multiset matching. The reward is
`2 * true_positives / (all_predicted_rows + all_gold_rows)`: ineligible rows
remain in the denominator, and excess duplicate predictions remain false
positives. No cell normalization or numeric conversion is introduced. Empty
tables remain rejected. Failed outcomes, unreliable-group discard, and label
access only after the group's durable close retain their existing behavior.

The selected composition is recorded in `online_web.online_train_reward`.
Per-row receipts use their own protocol and the actual training metric
`train_per_row_grounded_exact_f1`. Their `raw_literal_diagnostic` records the
unchanged literal scorer before grounding; it is not the actual training
reward. The labels continue to declare that original raw literal profile.
Neither metric is a benchmark score.

Grounding inherits the existing substring check across the fetched-text pool;
it does not establish semantic entailment or connect each cell to its particular
cited source. Equal rewards within a group can still produce zero advantages,
even when those rewards are positive. Recomputed scores from old closed TRAIN
outputs are diagnostics; fresh sampling, nonzero gradients and a matched
base/adapter evaluation are needed to establish training improvement.

`--native-batch-size N` sets the maximum queued requests per physical native
rollout batch on each rank. It accepts a positive integer and defaults to `4`,
preserving the existing behavior. For example, `--group-size 4 --native-batch-size 2`
still collects all four episodes, including model failures, using batches of at
most two requests. It does not change the group advantage calculation, fixed
GRPO denominator, thought/JSON/turn budgets, or replay batching. The value is
saved in the training arguments. Changing physical batches can change padding,
RNG consumption, trajectories and queue time; CPU queue tests do not establish
that a particular GPU configuration fits in memory or runs faster.

For a separate throughput experiment, `--native-mixed-schema-batching` lets
queued native requests with different public schemas share a thought batch
and a JSON batch. The pinned xgrammar processor creates a separate grammar
matcher for each row. Each row keeps its own required columns and read-source
citation enum, followed by validation against that row's original public
schema. Prompt-budget rejections and thoughts without a real closing token
retain their row ownership when the remaining rows enter the JSON phase.
The default remains schema-separated batching. The flag is recorded in the
training arguments. Different padding, RNG consumption and concurrent KV
storage can change sampled trajectories and peak memory; verify the selected
configuration on the actual GPU before scaling it.

The episode deadline covers queue waiting as well as execution, but is not a
hard wall-clock limit on native drain. On cancellation, owned Web requests mark
their thread-safe native cancellation control expired. The worker skips expired
queued requests before submission, retaining an explicit cancelled request error
and the episode's infrastructure outcome without inventing tokens or behavior
scores. A submitted thought phase drains its actual IDs and scores; if cancelled
before the JSON phase starts, it produces a `request_cancelled_before_json`
trace with no JSON or synthetic EOS. A JSON phase already started drains its
actual output. Such traces are accepted only for cancelled delivery and an
infrastructure-error episode. All group members remain represented and the
existing unreliable-group gradient discard is unchanged.

Training and matched Web evaluation use the same cancellation control. Policies
without that capability retain the previous complete-request drain. Repeated
outer cancellations do not cancel submitted native work. Allow outer watchdog
time for drain: a running kernel or phase still has no hard wall-clock cutoff.
Submission means the worker's check and mark immediately before dispatching
`to_thread`; JSON uses the corresponding locked check at its phase boundary.
These are submission boundaries, not a guarantee that no GPU instruction can
execute after the episode deadline.
Normal requests keep the same sampling, token budgets, schemas and loss. This
timeout fix changes future runtime behavior; already frozen evaluations keep
their original source, and comparisons using the fix need a new source identity
and matched control.

`--replay-diagnostic full|first|off` controls only the extra evaluation-mode,
no-gradient decoder forwards used to compare teacher-forced replay with the
actual sampled behavior. The default `full` retains the existing forwards for
every nonzero-advantage segment. `first` measures only the first such segment
per group; `off` performs none. Every mode still scores every nonzero-advantage
thought with gradients, checks its complete actual behavior/current alignment
and finite importance ratios, and applies the same clipped policy loss.
Teacher-forced replay never becomes the old-policy denominator.

Training arguments and configuration retain the diagnostic mode. Step records
also identify its scope and `diagnostic_scored_segments`/`diagnostic_scored_tokens`,
separately from all loss-scored segments/tokens. Unmeasured diagnostic maxima
are null; `first` maxima describe that one segment, not the entire group.
Actual behavior-versus-current maxima and ratio ranges/clip fractions retain
their complete loss-scored coverage in every mode. This saves diagnostic work;
it does not establish a training-quality improvement or measured GPU speedup.

`--cuad-concise-reasoning` is an optional CUAD prompt intervention, disabled
by default. It asks the model to locate candidate clauses using source IDs or
character ranges already shown in observations, avoid copying long contract
passages in reasoning, and copy qualifying spans only in the final answer.
It does not calculate new offsets, change retrieval, change the JSON schema,
force a thinking close, or increase the token budget. Web prompts are unchanged.
The training configuration and evaluation comparison record its enabled state
and prompt version. A matched base/adapter evaluation must use the same setting;
an adapter evaluated with this option cannot use an old baseline without it.
Treat it as an experimental option until a separate TRAIN-only comparison
establishes its effect on completion, correctness and generated length.

`--web-train-thought-budget-close` is a separate default-off thought-only
token-GRPO policy that samples an actual singleton `</think>` at the final
in-budget Web thought position. CUAD rows retain raw sampling. Captured,
current and frozen-reference singleton logprobs are zero, its policy gradient
is masked, and the original fixed budget denominator is retained. See
[Web TRAIN thought budget close](dcc_web_train_thought_budget_close.md) for
mode restrictions, ownership, history composition and validation limits.
