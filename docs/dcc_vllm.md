# Optional vLLM evaluation

The vLLM backend accelerates benchmark generation with fused MoE
inference and continuous batching. HF remains the default for generation and
the training runtime. This backend does not compute trainable log probabilities
and uses greedy decoding by default. Explicit evaluation sampling is optional;
it never changes the HF trainer's rollout probability calculation.

The evaluation CLI accepts `--eval-temperature`, `--eval-top-p`, `--eval-top-k`,
`--eval-min-p`, `--eval-presence-penalty`, `--eval-frequency-penalty` and
`--eval-repetition-penalty`. Nondefault values require the vLLM backend. For
example, the Qwen3-30B-A3B-Instruct-2507 model card recommends
`--eval-temperature 0.7 --eval-top-p 0.8 --eval-top-k 20 --eval-min-p 0`.
Presence penalties are an additional choice to validate on development data.
These settings alone do not establish a quality improvement.

Generation manifests and inference runtimes retain the full evaluation
sampling configuration. Use a fresh output directory when changing it. A paired
base/adapter comparison must declare the same full `generation.sampling`
dictionary and matching `generation.temperature` in both protocol files.
Legacy runs without sampling metadata remain greedy-only. Direct Python users
pass an `EvaluationSampling` from `slime_plugins.dcc.eval_sampling` as
`evaluation_sampling` to `load_vllm_policy`, and pass its temperature to each
policy call; mismatches are rejected.

`--web-structured-final` optionally constrains only the final-budget Web
evaluation call using vLLM's xgrammar backend, with fallback disabled. The JSON
schema contains the public required column names, scalar string/number/boolean/
null cells, and exactly one citation per row drawn from actually read source URLs,
matching the final review prompt.
Every row must contain at least one nonempty string, number or boolean cell;
other cells may remain unknown. This enforces the existing empty-row rejection
during generation and does not remove or repair rows after generation.
It supplies no answer values or target row count. Empty answers remain possible
and still fail the evidence guard. Earlier turns, early answers, CUAD and
training generation retain their existing behavior. Format constraints do not
establish factual correctness or a quality improvement.

The flag defaults off and requires vLLM. Startup preflights the installed
structured-output API and JSON-schema grammar on CPU. Enabled generation
requests also validate their actual schemas; unsupported escaped public column
keys fail clearly instead of being renamed or weakening the schema.
Enabled generation manifests and runtimes record `web_structured_final` as
`{"enabled": true, "schema_version": "web-final-nonempty-row-read-citations-v2", "backend": "xgrammar"}`.
Use a fresh output directory and declare the identical dictionary under
`common.generation.web_structured_final` in both paired protocol files. Missing
legacy metadata attests only disabled constraints. Direct Python callers must
enable `web_structured_final=True` on both `load_vllm_policy` and `EpisodeRunner`;
the runner calls the policy's explicit `generate_structured(..., json_schema=...)`
capability only for the eligible final call.

`--web-structured-actions` additionally constrains every earlier Web evaluation
turn to one JSON action. It defaults off and requires `--web-structured-final`.
Before an actual successful search, only `web_search` is permitted. Afterward,
the schema allows a search, a fetch, or the existing final answer. The final
budget turn still uses the final-only schema. This prevents consecutive tool
objects from being parsed as just the last action. The evidence guard and
scoring rules still apply, including rejection of unsupported or empty answers.
Enabled manifests and runtimes record `web_structured_actions` as
`{"enabled": true, "schema_version": "web-single-action-search-first-v1", "backend": "xgrammar"}`;
paired protocols must declare the same dictionary under `common.generation`.
Direct Python callers enable the switch on both the policy loader and runner.
CUAD and training generation retain their existing behavior.

`--web-explicit-review` optionally adds brief `plan` and `checks` JSON strings
before every structured Web action. It defaults off and requires both structured
switches and vLLM. Each string is limited to 240 Unicode characters. The model
reviews the original filters, unique keys, missing-field rules, evidence still
needed and finite stopping. This uses public task instructions and observed
sources. Review text grants no search, read or citation evidence; the existing
answer guard and scorer still apply. CUAD and training prompts are unchanged.
Direct Python callers pass `web_explicit_review=True` to the loader and runner.

Enabled manifests and inference runtimes record `web_explicit_review` with
`enabled=true`, schema and prompt version `web-explicit-plan-checks-v1`,
`max_chars=240`, scope `every-structured-web-action` and backend `xgrammar`.
Paired protocols require that same dictionary in `common.generation`.
Scoring and resume reject mismatched configurations; missing legacy metadata
means off. The reviewed schema preserves the existing tool and row restrictions.
Compare fresh full task sets with the same sampling, tools and token budgets.
The additional text consumes those budgets, and fluent checks do not establish
correctness or an improvement in benchmark scores.

The Web reader also reports when a returned text window exactly repeats an
earlier read in the same episode. It preserves the full text, navigation and
citation URL; the progress note helps the model choose an unseen section or a
different search. Equal windows do not establish that complete sources are
identical or that the source lacks useful information.

`--web-browser-reader` optionally renders successful public HTML on CPU. It
defaults off and currently requires the vLLM evaluation backend. Install the
separate optional dependency and its Chromium binary in the evaluation
environment before enabling it:

```bash
"$VLLM_ENV/bin/python" -m pip install -r configs/dcc/requirements-browser.txt
"$VLLM_ENV/bin/python" -m playwright install chromium
```

The evaluator preflights Playwright 1.56.0 and Chromium before writing the
generation manifest or loading the GPU model. One reader is shared across the
process with at most two concurrent pages. Requests, responses, bytes, visible
text and links have bounded budgets; each read has a 45-second total timeout.
Browser cleanup runs after all episodes and the policy have stopped, including
on setup or generation failures.

The browser is used only after an ordinary successful HTML read. HTTP errors,
access challenges, PDF, spreadsheets and JSON retain their existing paths.
Browser failures retain the successful HTTP observation. Reads expose rendered
visible text and public links. Observed public GET JSON collection URLs are
navigation metadata: their contents have not been read or cited until an
explicit fetch succeeds. Redirects, private destinations, downloads and
non-GET/HEAD requests are blocked by the optional reader.

Enabled manifests and inference runtimes include `web_browser_reader` with
the complete configuration, reader version, Playwright version, Chromium
version and CPU execution setting. Copy that exact object into
`common.generation.web_browser_reader` in both paired protocols. Scoring and
paired comparison reject missing, unsupported or mismatched enabled metadata.
Default-off runs add no browser metadata and do not import Playwright. Use a
fresh output directory and a fresh matched base when enabling this option;
neither the reader nor format constraints establish factual correctness.

Add `--vllm-async` to the evaluation CLI to use native V1 per-request
scheduling. Short requests can finish while a longer request remains active;
episode cancellation invokes `AsyncLLM.abort` on the engine's owning event
loop. This mode avoids the synchronous batch-delivery barrier and never calls
a synchronous engine's abort method from another thread. The generation
manifest and runtime record this mode, so use a new output directory when
switching it. The synchronous backend remains available by omitting the flag.

Use a separate environment with a CUDA-compatible driver:

```bash
python -m venv "$VLLM_ENV"
"$VLLM_ENV/bin/python" -m pip install -r configs/dcc/requirements-vllm.txt
CUDA_VISIBLE_DEVICES='' "$VLLM_ENV/bin/python" -c \
  'from vllm import LLM, SamplingParams; from vllm.lora.request import LoRARequest'
CUDA_VISIBLE_DEVICES='' PYTHONPATH=slime "$VLLM_ENV/bin/python" -m pytest -q \
  tests/test_dcc_vllm_policy.py
```

The environment pins vLLM 0.11.2 and Transformers 4.57.6. vLLM pins its own
compatible PyTorch version; installing it into the BF16 or NF4 training
environment would replace those dependencies. Reuse the existing base model
cache and local adapter checkpoint. No model merge or duplicate base checkpoint
is needed. The upstream Qwen3 MoE implementation supports LoRA and maps q/v
adapters into its fused qkv projection while retaining the o projection.
This backend accepts local attention-only PEFT adapters with ranks up to 64;
the current training configuration uses q/v/o with rank 8.

The Python integration API is:

```python
from slime_plugins.dcc.vllm_policy import load_vllm_policy

policy = load_vllm_policy(
    model_name,
    adapter=adapter_path,  # None for the base policy
    batch_size=8,
    max_prompt_tokens=16384,
    max_model_len=24576,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.88,
    max_num_batched_tokens=8192,
    seed=42,
)
# Inside an async function:
completion = await policy(messages, max_new_tokens=8192, temperature=0)
await policy.close()
```

Use one launcher process, without multi-rank `torchrun`. Select exactly as many
visible GPUs as `tensor_parallel_size`: TP=1 requires one visible GPU; a single
TP engine may use several explicitly assigned GPUs. The loader rejects a
nonzero launcher/local rank, a launcher world size other than one, mismatched
GPU visibility, or an existing distributed process group. It temporarily clears
launcher rendezvous variables during engine creation, restores them afterward,
uses spawned workers, and explicitly selects the pinned V1 uni/MP executor.
These executors assign worker ranks and a fresh TCP rendezvous internally;
the `external_launcher` executor is not used.

For data parallel evaluation, launch independent engines with
`--eval-shard-index INDEX --eval-num-shards COUNT`. Supply both options, with
`0 <= INDEX < COUNT`, and give every engine the same complete `--data` list and
shared `--output-dir`. Each process keeps launcher `RANK=0`, `LOCAL_RANK=0` and
`WORLD_SIZE=1`; select its own GPU with `CUDA_VISIBLE_DEVICES`. The logical
shard only controls task assignment, per-rank output names and the PyTorch seed.
Model settings, sampling configuration and token budgets remain the same.

For example, invoke this once per assigned GPU, setting `SHARD_INDEX` from zero
through `SHARD_COUNT - 1`:

```bash
RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 CUDA_VISIBLE_DEVICES="$WORKER_GPU" \
  "$VLLM_ENV/bin/python" scripts/dcc_model_evaluate.py \
  --model "$MODEL" --adapter "$ADAPTER" --inference-backend vllm \
  --eval-shard-index "$SHARD_INDEX" --eval-num-shards "$SHARD_COUNT" \
  --data "$TASKS" --output-dir "$EVAL_OUTPUT"
```

This writes disjoint `predictions_rankN.jsonl`, `manifest_rankN.json` and
`rankN.done` files. Multiple shards also use `inference_runtime_rankN.json`;
a single engine retains `inference_runtime.json`. All existing rank manifests
must agree on the full generation configuration, including shard count and
backend. Repeating a worker resumes its own predictions. Wait for every worker
to finish, then run the existing `--score-only` command against the full data
and shared output directory, omitting both shard options. Scoring requires
complete task coverage across all prediction files. Explicit shard options
are rejected for HF generation and for scoring or judge modes. Omitting them
preserves the existing HF launcher rank/world behavior.

For a complete two-benchmark run, define portable paths to the model, adapter,
full held-out datasets and an external output root. Use a fresh output root for
each candidate. Invoke this command once for each assigned worker, with
`SHARD_COUNT=8` and `SHARD_INDEX` from 0 through 7 for eight independent engines:

```bash
SHARD_COUNT=8
RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 CUDA_VISIBLE_DEVICES="$WORKER_GPU" \
  "$VLLM_ENV/bin/python" scripts/dcc_model_evaluate.py \
  --model "$MODEL" --model-name "$MODEL_NAME" --adapter "$ADAPTER" \
  --inference-backend vllm --quantization none \
  --eval-shard-index "$SHARD_INDEX" --eval-num-shards "$SHARD_COUNT" \
  --vllm-tensor-parallel-size 1 --vllm-memory-utilization 0.88 \
  --vllm-batched-tokens 8192 --seed 42 \
  --data "$CUAD_TEST" --output-dir "$EVAL_ROOT/cuad" \
  --retrieval-profile "$RETRIEVAL_PROFILE" --initial-chunks 8 \
  --max-steps 3 --max-new-tokens 1024 --max-prompt-tokens 8192 \
  --batch-size 8 --timeout 1800

RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 CUDA_VISIBLE_DEVICES="$WORKER_GPU" \
  "$VLLM_ENV/bin/python" scripts/dcc_model_evaluate.py \
  --model "$MODEL" --model-name "$MODEL_NAME" --adapter "$ADAPTER" \
  --inference-backend vllm --quantization none \
  --eval-shard-index "$SHARD_INDEX" --eval-num-shards "$SHARD_COUNT" \
  --vllm-tensor-parallel-size 1 --vllm-memory-utilization 0.88 \
  --vllm-batched-tokens 8192 --seed 42 \
  --data "$WIDE_TEST" --output-dir "$EVAL_ROOT/widesearch" \
  --auto-read-sources 0 --max-steps 12 --max-new-tokens 8192 \
  --max-prompt-tokens 16384 --batch-size 4 --timeout 1800
```

Do not partition the input files before invoking the workers: logical task
assignment uses the same complete ordered task list. After every worker exits,
score each benchmark on CPU, without logical shard options:

```bash
for BENCHMARK in cuad widesearch; do
  if [ "$BENCHMARK" = cuad ]; then TASKS="$CUAD_TEST"; else TASKS="$WIDE_TEST"; fi
  CUDA_VISIBLE_DEVICES='' RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
    "$CPU_ENV/bin/python" scripts/dcc_model_evaluate.py \
    --model "$MODEL" --model-name "$MODEL_NAME" --score-only \
    --data "$TASKS" --output-dir "$EVAL_ROOT/$BENCHMARK"
done
```

Scoring checks complete ID coverage, disjoint shard ownership, identical
manifests and runtimes, known episode statuses, and every logical shard's
matching completion marker. It keeps the latest retry within one shard but
rejects the same task in different shards. Reports copy actual generation
budgets from the saved manifest, independent of scoring CLI defaults. Older
manifests without logical shard fields remain scoreable; their historical
completion metadata cannot provide the newer strict logical-shard audit.

For an optional frozen HF judge, run the following after strict scoring, using
a separate HF environment and explicitly assigned judge GPUs. The judge loads
the base model without the trained adapter. `JUDGE_PROCESSES` must match the
number of visible judge GPUs; `--judge-workers 1` serializes requests within
each process. Neither judging nor merging takes logical shard options:

```bash
CUDA_VISIBLE_DEVICES="$JUDGE_GPUS" "$HF_ENV/bin/python" -m torch.distributed.run \
  --standalone --nproc-per-node "$JUDGE_PROCESSES" scripts/dcc_model_evaluate.py \
  --model "$MODEL" --model-name "$MODEL_NAME" --score-only --local-judge \
  --judge-name "$MODEL_NAME" --quantization none \
  --judge-batch-size 1 --judge-workers 1 --judge-context 65536 \
  --judge-tokens 4096 --judge-batch-tokens 69632 \
  --data "$WIDE_TEST" --output-dir "$EVAL_ROOT/widesearch"
CUDA_VISIBLE_DEVICES='' RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
  "$CPU_ENV/bin/python" scripts/dcc_model_evaluate.py \
  --model "$MODEL" --model-name "$MODEL_NAME" --score-only --merge-judge \
  --judge-name "$MODEL_NAME" --quantization none \
  --data "$WIDE_TEST" --output-dir "$EVAL_ROOT/widesearch"
```

The strict WideSearch report is labeled `strict_exact_proxy`; the local-judge
report is `official_protocol_custom_judge`. The latter follows the official
matching protocol with a custom judge and is not an official-judge score.
Report `status_counts`, `complete`, WideSearch `metric_denominator` and
`judge_failures` with scores. Model `evidence_failure` outcomes receive zero
within the full task denominator. Infrastructure errors make `complete=false`;
judge errors are listed explicitly, excluded from the judge metric denominator,
and also make the report incomplete.

Each judge shard records its frozen model, adapter absence, quantization,
context and completion budgets, request batching, truncation policy and runtime
versions in `judge_configuration`. Merge checks that these configurations agree
across shards and match the selected judge. It also checks generation manifest
and inference runtime provenance against the saved predictions, along with
complete and unique task coverage. Legacy reports without judge configuration
or generation provenance remain readable; `judge_configuration_complete` is
false if any shard lacks its configuration. Configurations that are present
must still agree, including in a mixed legacy/new merge.

For uneven answer sizes, `--judge-scheduling balanced` optionally assigns
saved answer tables using a deterministic size-based schedule after validating
the original generation ownership. Pass the same option to every local-judge
worker and the merge command. This changes judge task ownership only;
generation files and original generation ranks remain intact. The default
is `round_robin`. Use a fresh scoring directory when changing the strategy.

A failed judge shard must be rerun with its original judge rank, world size,
scheduling strategy and the same data, output directory and judge
configuration. This re-evaluates the
whole judge shard and replaces that rank's report; judging does not resume
individual tasks, and `--retry-infrastructure` applies only to generation. For
one-rank recovery, launch Python directly with the original `RANK` and
`WORLD_SIZE`, `LOCAL_RANK=0` and one visible GPU. A one-process `torchrun` would
change the world size and task assignment. If the judge context, quantization
or other recorded settings change, use a fresh judge output and recompute all
judge shards with the same configuration. Retain every full gold reference;
reduce response batches or increase the context if needed rather than clipping
reference vocabularies. Context planning must reserve both the response text
inside the judge prompt and the judge's completion budget.

Set GPU visibility before creating an engine. Guard standalone launchers with
`if __name__ == "__main__":` so worker subprocesses cannot rerun the launcher.
The queue serializes engine access and batches requests. `close()` drains the
queue, then calls V1 `EngineCoreClient.shutdown()` to finalize its background
resources and engine/worker subprocesses. It is idempotent. Run each development
comparison in a dedicated process and verify exit before reassigning its GPUs.
Allow the run owner to schedule GPU memory and concurrency.

The evaluation CLI also closes the policy after setup or episode failures. It
cancels and awaits unfinished episodes before draining the policy and closing
the prediction stream. The original failure propagates; cleanup failures are
attached as exception notes, and a failed run does not write its rank completion
marker. Persisted predictions remain available for a matching resumed run.

Prompt tokenization and truncation call the same `encode_messages` helper as
HFPolicy, with thinking disabled. The engine receives those exact token IDs.
Prompt plus requested completion must fit `max_model_len`; an insufficient
context budget raises an error and never silently shortens generation. KV
cache uses the BF16 model dtype. LoRA dtype is explicitly `auto`, which resolves
to BF16 with this base. HF/PEFT can store and compute its adapter branch in
FP32; the vLLM adapter computation can therefore differ even with the same
checkpoint. Validate on development tasks and retain that precision provenance.
Prefix caching and chunked prefill are enabled;
`max_num_batched_tokens` limits each scheduling iteration. The memory fraction
includes the base weights, engine allocations and KV cache. Measure actual
memory and throughput before increasing concurrency.

Save `policy.provenance` beside the prediction manifest in a new external output
directory. It records model/tokenizer revisions, resolved model revision when
available, runtime versions, adapter configuration/path/size/modification time,
precision, seed, context/token budgets, memory fraction and cache settings.
The evaluation CLI identifies the small LoRA checkpoint by path, configuration,
filename, byte size and a content digest. Local ctime, inode, device and mtime
invalidate the digest cache when a checkpoint changes; they are not the
semantic identity, so an identical checkpoint copy can be evaluated on another
worker. The unchanged local checkpoint is not repeatedly hashed, and base model
weights are not hashed. Resume checks the saved identity
before reusing completed predictions, and refuses inconsistent runtimes rather
than overwriting them. Logical generation records identity in its manifest
before loading and verifies it again afterward. Keep checkpoints immutable
while evaluations run; changed checkpoints require a fresh output directory.
Older records retain only the identity fields originally available.
Actual prompt/completion IDs and finish reasons are retained
in `PolicyCompletion`; log probabilities are absent. Changing the backend or
adapter requires a new output directory. Runtime paths belong in external
artifacts, not committed configuration.

Validate base and adapter generation on independent web or CUAD development
tasks with identical semantic budgets before a full run. Compare wall-clock
time, generated tokens, answer validity and actual evidence use. Fused kernels
can produce different greedy outputs from HF BF16. Evaluating an NF4-trained
adapter on this BF16 base also changes base precision and must be reported as
such; it is not a continuation of an NF4 evaluation. CPU adapter and batching
tests do not establish GPU compatibility or a speedup.

Upstream implementation reference:
[vLLM 0.11.2 Qwen3 MoE](https://github.com/vllm-project/vllm/blob/v0.11.2/vllm/model_executor/models/qwen3_moe.py).
The executor rank/rendezvous and shutdown behavior are defined in
[UniProcExecutor](https://github.com/vllm-project/vllm/blob/v0.11.2/vllm/v1/executor/uniproc_executor.py),
[MultiprocExecutor](https://github.com/vllm-project/vllm/blob/v0.11.2/vllm/v1/executor/multiproc_executor.py),
and [EngineCoreClient](https://github.com/vllm-project/vllm/blob/v0.11.2/vllm/v1/engine/core_client.py).


The HTTP reader also accepts bounded ZIP text exports (CSV, TSV, XML, JSON and TXT). `archive_members` lists literal previews and source-wide character offsets; fetch the source URL at a readable member’s `start` before citing its contents. Downloads are limited to 5 MB, 128 members and one million decoded text characters. Archives are read in memory; encrypted, nested, unsafe, malformed or oversized archives fail without publishing evidence. XLSX keeps its existing reader. Legacy binary XLS is unsupported.

Automatic reading seeks up to `--auto-read-sources` successful new pages after each search, while all successful, empty and failed reads count toward the unchanged four-attempt episode limit. Errors remain visible in the observation and grant no evidence. Sources exceeding the 5 MB download limit are rejected explicitly instead of returning an incomplete byte prefix.


Structured Web evaluation can opt into `--web-bounded-whitespace`. It bounds insignificant JSON space, LF and TAB runs to eight characters using the pinned xgrammar 0.1.25 EBNF converter, with fallback disabled. It supports both ordinary structured actions and finite explicit review. The flag defaults off; JSON string values, public schema and evidence/scoring guards stay unchanged. Generation, resume, scoring and paired comparison require matching opt-in provenance. This formatting change needs fresh GPU validation and does not by itself establish better answers.

Optional FP8-Marlin weights

Use `--inference-backend vllm --quantization fp8` to reduce weight memory
while keeping BF16 activations, BF16 router gates and the existing KV cache
dtype. The default remains BF16. This path supports unquantized 125-expert
Qwen3 MoE checkpoints on SM80 with vLLM 0.11.2 and tensor parallel size 1.
Other model layouts and prequantized checkpoints are rejected.

Standard attention-only LoRA with rank 1 through 64 is supported. Before
loading the model, the worker validates every adapter A/B pair, shape, dtype
and offset using only its safetensors header. It excludes fused experts from
the LoRA manager, keeps router gates unquantized and verifies the actual
FP8-Marlin expert and attention methods after loading. The original manager
filter is restored even if loading fails. Base-only runs use the same router
and expert checks with LoRA disabled.

Use fresh output directories for both a quantized base and its trained
comparison. FP8 runs write create-once header and worker receipts under
`fp8_validation_rank{rank}` and record the full contract in manifests and
inference provenance; failed or partial validation directories are not reused.
The evaluator checks the actual receipts before generating benchmark requests
and closes the engine if that check fails. Completed predictions are scored
with the existing separate scoring commands.

For direct `load_vllm_policy` calls, pass `quantization="fp8"` and an absolute
fresh `fp8_validation_dir`; GPU ownership remains the caller's responsibility.
Native thinking budgets, stop tokens, schema checks and training are unchanged.
