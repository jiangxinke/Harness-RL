# Mixed tool and contract supervision

`scripts/dcc_tool_sft.py` warm-starts or creates an attention LoRA adapter using
independent web-tool teachers and grounded CUAD training examples. It uses the
existing model loader and completion log probabilities; the original training
entry points remain available. Use the separate NF4 environment described in
[dcc_training.md](dcc_training.md) when enabling `--quantization nf4`.

Web records must declare `source=independent_curriculum` and `split=train`.
Search, fetch and final targets are exact action JSON. The loader reconstructs
successful searches and reads from the historical tool observations, rejects
unsupported citations, and checks every final cell against actual fetched text
and the encoded prompt. A `source_urls` declaration alone is insufficient.
Official WideSearch tasks and validation records never become training targets.

CUAD tool teachers are optional through `--cuad-tool-records`. They must declare
`source=cuad_train`, `benchmark=cuad`, `split=train`, a category and a positive
boolean in `metadata`, and the final spans in `grounding_fields`. Their stages
are search, read and final. Read starts must be exposed by prior search hits;
ends use the hit boundary or a public reader window. Final spans must each
appear within an actual source observation and remain visible after encoding.
Teachers may use training labels to select a successful plan from public
question/profile queries. They must never inject gold phrases into queries or
choose arbitrary offsets from answer locations.

Prepare these records on CPU with `scripts/dcc_prepare_cuad_tools.py`. The
default `--episodes 96` selects hidden positive training questions. Optional
`--negative-episodes` adds category-balanced questions whose training gold is
empty; its default is zero, preserving positive-only generation. Negative
teachers execute an actual search and a read anchored to one of its returned
hits before producing `{"answer":[]}`. Their `metadata.positive` is false and
`grounding_fields` is empty. Validation and test tasks are rejected, including
when only negative episodes are requested. The manifest reports positive and
negative counts and categories separately.

Set `--episode-max-steps` to the deployed CUAD runner turn budget. Its default
is 3 and values below 3 are rejected. A three-turn teacher includes the runtime
forced-final control before its final target. With 4 or more turns, the same
real search/read trace finishes normally on the third turn, with accurate
remaining-turn observations and no forced-final control. This does not add
tool actions or train a forced fourth-turn review. Targets and source evidence
stay the same; prepare a separate teacher file for each turn budget.

```bash
python scripts/dcc_prepare_cuad_tools.py \
  --data "$DATA_ROOT/cuad/train.jsonl" \
  --retrieval-profile "$RUN_ROOT/retrieval/profile.json" \
  --tokenizer "$MODEL_CACHE" \
  --output "$RUN_ROOT/teachers/cuad-train.jsonl" \
  --episodes 128 --negative-episodes 48 --episode-max-steps 4 \
  --max-prompt-tokens 8192 --max-response-tokens 1024 --batch-tokens 12288
```

Use `--episodes 0 --negative-episodes 48` to prepare a separate negative file
for an existing positive curriculum. Concatenate the complete files without
dropping stages, then validate with the SFT loader and the training tokenizer.
Each distributed worker must retain search, read and final stages after
task-level sharding and token-budget filtering. The preparation command checks
all three stages without truncation and refuses to overwrite existing outputs.

Web and CUAD tool prompts are never truncated. Targets are never truncated.
Overlong prompts, responses and token-budget failures are counted explicitly.
Every local tool stage must retain eligible examples, so a run cannot silently
discard the final stage and train only tool requests. Direct CUAD targets reuse
the existing grounded-span helper: positive answers must be visible in the
actual encoded prompt. Its existing prompt truncation remains available.

Within web examples, stages cycle evenly. CUAD samples use approximately two
positive examples for each negative example and choose categories uniformly,
instead of letting large categories dominate. Optional CUAD tool examples
replace direct examples within their selected category with probability
`--cuad-tool-fraction` (default 0.75); their stages also cycle evenly. This
probability applies only when that category has local eligible tool records.
With `--cuad-train-data`, a category lacking tool records can still produce a
direct initial-prompt final target, even at tool fraction 1. Short empty direct
targets receive the same example weight as longer tool and positive targets,
which can teach premature empty answers before tools run. To train only tool
trajectories, omit `--cuad-train-data` and supply complete CUAD tool records.
Sampling fractions describe draws; actual trained stage counts are logged.
Token batches accumulate sequential examples up to `--batch-tokens` and
`--samples-per-step`; a long example that does not fit the remainder is retained
for the next batch. Each example receives equal loss weight. Only its generated
completion and EOS are targets; historical assistants and tool observations
receive no direct loss.

Workers receive disjoint task IDs and independent sampling RNGs. All replicas
start with the same adapter initialization, keep LoRA dropout at zero, and
average adapter gradients before clipping and updating. Frozen base weights
receive no optimizer updates. Non-finite losses or zero/non-finite gradients
abort the run. Per-rank logs record actual task IDs, categories, stages, token
counts, skips, gradients and quantization coverage. Checkpoints and logs always
use the explicit output directory.

For a fresh tool-only attention LoRA, start with a conservative learning rate
and balanced positive/negative CUAD tool records:

```bash
torchrun --standalone --nproc-per-node="$WORKERS" scripts/dcc_tool_sft.py \
  --model "$MODEL_CACHE" --quantization none \
  --records "$CURRICULUM_ROOT/teacher_train.jsonl" \
  --cuad-tool-records "$CUAD_TEACHER_ROOT/teacher_train.jsonl" \
  --cuad-tool-fraction 1.0 --web-fraction 0.5 --negative-fraction 0.3333333333333333 \
  --retrieval-profile "$RUN_ROOT/retrieval/profile.json" \
  --output-dir "$RUN_ROOT/tool-sft-fresh" \
  --steps 128 --learning-rate 1e-5 --lora-rank 16 --save-every 32 \
  --max-prompt-tokens 16384 --max-new-tokens 4096 \
  --batch-tokens 32768 --samples-per-step 4
```

This example creates a fresh adapter and uses CUAD tool supervision without
`--cuad-train-data`. If both CUAD direct data and CUAD tool records are absent,
the run uses only independent web examples. Supply `--adapter` only to restore
existing adapter weights; optimizer and step state are saved for inspection
and are not automatically resumed. Use `--quantization nf4` in its supported
environment when reducing frozen-weight memory, and report that precision
with results. Keep complete stages on every worker, check actual sampler
counts and TRAIN behavior, and evaluate the fixed checkpoints with a matched
base protocol. Keep official test labels out of parameter selection.

Run the CPU checks with
`CUDA_VISIBLE_DEVICES='' PYTHONPATH=slime python -m pytest tests/test_dcc_tool_sft.py tests/test_dcc_prepare_cuad_tools.py`.
