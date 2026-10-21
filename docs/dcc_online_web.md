# Opt-in online Web TRAIN groups

The native HF LoRA trainer can mix its existing CUAD TRAIN groups with an
independent, public-only Web TRAIN curriculum with the WideSearch task schema. The default remains
CUAD-only. This path uses TRAIN literal row F1; it does not report a WideSearch
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

The episode deadline covers queue waiting as well as execution. Allow enough
outer watchdog time for the configured group size, and use the explicit
thought/JSON/turn budgets to bound policy work. A queue timeout alone does not
show a model error. Requests cancelled while waiting for another schema group
are checked again before submission and do not start new generation. Already
running work is still drained and retained with its actual ownership.

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
