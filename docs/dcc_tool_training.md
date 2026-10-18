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
`--cuad-tool-fraction` (default 0.75); their stages also cycle evenly. Sampling
fractions describe draws; actual trained stage counts are logged. Token batches
accumulate sequential examples up to `--batch-tokens` and
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

```bash
torchrun --standalone --nproc-per-node="$WORKERS" scripts/dcc_tool_sft.py \
  --model "$MODEL_CACHE" --adapter "$START_ADAPTER" --quantization nf4 \
  --records "$CURRICULUM_ROOT/teacher_train.jsonl" \
  --cuad-train-data "$DATA_ROOT/cuad/train.jsonl" \
  --cuad-tool-records "$CUAD_TEACHER_ROOT/teacher_train.jsonl" \
  --retrieval-profile "$RUN_ROOT/retrieval/profile.json" \
  --output-dir "$RUN_ROOT/tool-sft-round1" \
  --steps 24 --learning-rate 1e-5 --batch-tokens 12288
```

Omit `--adapter` for a new adapter or `--cuad-tool-records` before those teachers
are available. Without CUAD data, the run uses only independent web examples.
`--adapter` restores adapter weights only; optimizer and step state are saved
for inspection and are not automatically resumed. Choose each short training
round using independent web validation and CUAD validation. Keep official test
labels out of parameter selection and report quantized precision with scores.

Run the CPU checks with
`CUDA_VISIBLE_DEVICES='' PYTHONPATH=slime python -m pytest tests/test_dcc_tool_sft.py`.
