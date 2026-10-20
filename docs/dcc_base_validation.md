# Validate and freeze the base harness

Develop the harness on independent validation tasks before training an adapter.
WideSearch's official full split is test only: its reference rows must not become
teacher data, retrieval queries, prompts, or checkpoint-selection criteria.
Generate teachers only from independent tasks marked `train`.

Check both English and Chinese tasks, enumeration lengths, and column counts.
The official tasks span 3–42 columns; a development set containing only 2–3
columns cannot check realistic output complexity. Include independent public
tables with 6–12 columns and separate wide-schema regression fixtures. The
runner uses `metadata.evaluation.required`, with `answer_columns` as a fallback,
as exact JSON keys. It preserves the question's requested language and cell
formats while using the harness JSON response schema for Markdown questions.
Rows with extra fields are rejected with feedback before scoring. Row citations
are evidence metadata, removed from the benchmark table before either scorer.

HTML observations include bounded `outgoing_links`, with structured-data and
download links prioritized. These links describe the whole downloaded page,
including content outside the current text window; they are unread references.
The agent must fetch a linked page before using its contents or citing it.
Keep the reader and prompt versions in generation manifests when comparing
harness revisions. Review both entity filtering and duplicate rows: successful
source reads and evidence acceptance do not establish answer correctness.

The table reader protocol `csv-inference-v2` gives native JSON rows and Markdown
tables the same CSV type inference as the pinned upstream evaluator
(`9825ba7b140b71d81b364793f86dabe4cfed6749`,
[data loader](https://github.com/ByteDance-Seed/WideSearch/blob/9825ba7b140b71d81b364793f86dabe4cfed6749/src/evaluation/data_loader.py)).
This includes numeric strings, leading zeros, booleans and missing cells.
Rows and columns are retained. It fixes the harness's additional JSON input path;
the `same_entity_v9` judge prompts and entity/cell rules remain frozen. Record
`widesearch.table_input_protocol` and rerun both policies when this protocol changes.
Judge-score merging checks this version against the strict report and every
shard and task. Missing historical declarations remain explicit: the merged
version is null, provenance completeness is false, and observed versions and
legacy shard counts are retained. Re-score historical predictions to establish
current provenance before comparing policies.

Run CPU regressions, review the changes, then freeze a clean commit before the
base GPU run:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/test_dcc_evidence_guard.py tests/test_dcc_web_recovery.py \
  tests/test_dcc_auto_read.py tests/test_dcc_core.py \
  tests/test_dcc_search_provider.py tests/test_dcc_web_unicode.py \
  tests/test_dcc_tools.py tests/test_dcc_hf_runtime.py
git rev-parse HEAD
```

Set portable paths to the frozen model, independent validation JSONL, Python
environment, and an external durable output directory. The scheduler must assign
`WORKER_GPU`; the command does not select or reserve a device. This base run has
no adapter:

```bash
RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 CUDA_VISIBLE_DEVICES="$WORKER_GPU" \
  DCC_SEARCH_BACKEND=public_web DCC_WEB_TIMEOUT=15 \
  "$VLLM_ENV/bin/python" scripts/dcc_model_evaluate.py \
  --model "$MODEL" --model-name "$MODEL_NAME" \
  --inference-backend vllm --vllm-async --quantization none \
  --vllm-tensor-parallel-size 1 --vllm-memory-utilization 0.88 \
  --vllm-batched-tokens 8192 --seed 42 \
  --data "$WIDE_DEV" --output-dir "$EVAL_ROOT/base-dev-read0" \
  --auto-read-sources 0 --max-steps 12 --max-new-tokens 4096 \
  --max-prompt-tokens 16384 --batch-size 4 --timeout 900

CUDA_VISIBLE_DEVICES='' RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
  "$CPU_ENV/bin/python" scripts/dcc_model_evaluate.py \
  --model "$MODEL" --model-name "$MODEL_NAME" --score-only \
  --data "$WIDE_DEV" --output-dir "$EVAL_ROOT/base-dev-read0"
```

Compare `--auto-read-sources 1` in a fresh directory while keeping every other
setting fixed. Then compare a 12,288-token prompt budget if context cost matters.
Change one factor at a time. For independent engines, give each worker the same
complete input and add `--eval-shard-index "$SHARD_INDEX" --eval-num-shards
"$SHARD_COUNT"`; score only after every shard finishes, without shard options.

For a 34-task development set, useful provisional engineering gates are:

- All 34 tasks have predictions, with no infrastructure or model-runtime errors.
- At least 31 tasks have a search returning source URLs and a successful fetch
  containing nonempty page text. A search event alone does not satisfy this gate.
- At least 28 tasks finish with nonempty, evidence-accepted tables; each language
  reaches at least 75%. Accepted rows contain every required key and cite pages
  actually read. Review failures separately for each source family.
- Inspect every completion ending with `finish_reason=length`, especially final
  tables. Confirm that output truncation or schema errors are not hiding useful
  rows. Report row counts and strict row/item F1 alongside these behavior gates.

These gates test basic research behavior, not benchmark quality. Empty tables
and failed episodes stay in the full denominator. Strict exact-match scores are
development proxies. Official WideSearch scoring uses its published evaluation
and judge configuration; replacing that judge with a local model must be labeled
as a custom-judge result, which is not directly comparable to the official
leaderboard. See [the evaluation runtime guide](dcc_vllm.md) for judge commands.

Twelve model turns allow at most eleven tool-action turns and one final turn.
Automatic reading adds at most four attempted downloads per episode. A
`--batch-size 4` engine admits up to eight episodes, while each public search
backend serializes requests within its process. Native async vLLM can abort GPU
requests when the episode times out. `DCC_WEB_TIMEOUT` bounds individual socket
operations; it is not a whole-shard or strict total DNS/download deadline.
Monitor task latency and actual search/read coverage before increasing workers.
If search queues dominate, reduce batch size to two and keep other assigned GPUs
busy with independently scheduled work. Use a fresh directory for changed
budgets. For the much wider official tables, validate an 8,192-token completion
budget separately; do not assume the smaller development budget suffices.

Freeze the selected harness, datasets, runtime, and inference settings before
training. Compare the frozen base and adapter with identical base weights,
precision, seed, task order, search backend, auto-read setting, context/output
budgets, timeouts, batching, and scorer. Keep separate outputs and record the
adapter identity. A harness change requires rerunning both policies; an adapter
comparison must not inherit results generated by an older harness.

Before starting those runs, require basic instruction following with
`scripts/dcc_base_sanity.py`. Its default eager mode is useful for separating
model quality from runtime compilation. Repeat with
`--no-vllm-enforce-eager` to check compiled execution, keeping the model and
other arguments fixed. A successful model load or nonempty text is insufficient.
If compiled execution fails this check, use `--vllm-enforce-eager` consistently
for both policies and record that setting. Also run
`scripts/dcc_policy_smoke.py --vllm-enforce-eager` to verify concurrent requests
and native timeout cancellation before launching asynchronous evaluation.
