# Native SFT task coverage (opt in)

`dcc_tool_sft.py --native-thinking --native-global-stage-task-coverage`
uses global task decks during accepted-trajectory LoRA supervised fine-tuning.
It is not reinforcement learning. The flag defaults to off; existing sampler
code, default sample sequence, loss, optimizer, learning rate, token budgets,
checkpoints and training/evaluation protocols retain their existing behavior.
The disabled flag is omitted from the saved configuration.

## Selection law

The planner constructs the original `MixedSampler` on all eligible native TRAIN
records. It keeps that sampler's domain, polarity, category and available-stage
decisions and consumes the original example-choice RNG draw. At that point it
replaces the selected example with one from a broader bucket:

* Web: actual stage.
* CUAD: polarity and actual stage, across all categories.

Each bucket shuffles task IDs without replacement; multiple examples of the
same task cycle separately. The output category belongs to the selected task.
This **intentionally removes category-uniform example weighting**. The original
category decision remains an input to stage availability; it does not determine
the output task's category. A CPU shadow control draws a uniform task with
replacement and then a uniform variant from the same broader bucket, using an
independent selection RNG.

Global availability differs from the default rank-local pools. Consequently,
domain fallback, categories and available-stage distributions can differ from
the default even with the same seed. Both new arms preserve the decision stream
of the **global** original sampler, not the original rank-local stream. For
fixed logical draw counts, shadow/coverage decisions consume identical RNG;
when different lengths change packing, per-step draw counts may differ.

## Packing, state and telemetry

Every process deterministically plans each optimizer step in logical-rank order,
returning its own packed batch. Global pools are encoded once per process;
logical ranks share Example and token objects. Planning uses validated cached
lengths, with no repeated tokenization, tensor copies or new collectives. State
is O(records + tasks + world size), with no saved schedule or growing history.
Per-step work simulates all ranks' sampling and packing.

Host memory needs separate accounting: the native encoder initially copies
prompt/completion token lists, and global construction retains all records per
real process. Sampler-owned token-list storage can therefore approach world
size times the old rank-local amount. Logical-rank copies share those lists;
there are no additional token-list copies per rank or training step. The cached
length-only CPU replay does not measure these real token-list allocations.

The original one-example pending lookahead remains unchanged. Coverage refers
to reservation order, including pending examples, rather than wall-clock GPU
consumption order. A pending reservation may be consumed after another bucket
cycle has begun. All ranks require identical validated records, order, seed,
world size, example count and budgets.

`eligible.sampler_buckets` reports global bucket sizes. `skipped` is the shared
global-pool construction Counter, not rank-local skip telemetry. For an actual
matched four-example experiment, require zero skips and `batch_tokens >= 4*M`,
where M bounds every validated example's total tokens; also keep prompt and
response caps large enough to retain every example. This prevents a sampler
change from becoming a budget filtering or variable-batch intervention.

The trainer accepts fresh runs only. Its saved decision RNG does not contain
the task-deck, variant, pending and logical-rank state needed for a full resume.

## CPU evidence and limits

The CPU suite covers a small frozen default trace, the existing original sampler
guard, default configuration, mixed Web/CUAD planning across eight logical
ranks and two independent emulated processes, stage/polarity decisions,
task/variant coverage, exact object reuse, pending packing and bounded state.

```
python -m unittest discover -s tests -p test_dcc_stage_task_coverage.py -v
```

A cached CUAD-only TRAIN length replay used 378 tasks / 428 segments, eight
ranks, 128 steps, four examples per rank/step, seed 47. All 4,096 examples fit.
Shadow and coverage matched all 1,024 rank-step decision states and bucket
traces. This is neither the mixed Web/CUAD training schedule nor an evaluation
of model quality. Effective task size is `(sum exposures)^2 / sum(exposure^2)`.

| Exposure statistic | Default local replacement | Global task replacement | Global stage task coverage |
|---|---:|---:|---:|
| Positive effective task size | 61.00 | 65.89 | 68.15 |
| Positive maximum task exposures | 87 | 66 | 53 |
| Negative tasks covered (of 300) | 286 | 298 | 300 |
| Negative effective task size | 201.54 | 184.23 | 211.13 |
| Negative maximum task exposures | 14 | 29 | 26 |

The negative maximum worsens because the global stage law draws 22 negative
read examples from a pool containing only one task; the old local law draws
six. Stage decks cannot remove that source imbalance while preserving the
global stage decision law. All four additional fixed seeds (11, 23, 71, 101)
improved positive and negative effective size versus default, covered all 300
negative tasks, and retained this higher negative maximum. These exposure
results justify a prospective controlled comparison, not a performance claim.
Training quality still requires the registered matched development evaluation.
