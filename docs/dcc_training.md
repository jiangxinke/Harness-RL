# Local MoE training and evaluation

The optional Transformers runtime trains attention LoRA adapters using the same
DCC episode tools and rewards as evaluation. It leaves the existing Slime,
StackPlanner, OLE and Megatron training defaults intact. The historical upstream
model is `Qwen/Qwen3-30B-A3B-Instruct-2507`: nominal 30B class,
**30,532,122,624 base parameters**, about 3.3B activated per token. Its
rank-8 attention LoRA adds 5,701,632 parameters, giving **30,537,824,256** total.
A strict total-parameter cap of 30B excludes this original model.

For a strict parameter cap, the optional expert-pruning exporter creates a
distinct, physically smaller Qwen3 MoE checkpoint. On the original 48-layer
model, retaining 125 of 128 experts per layer yields **29,852,350,464** base
parameters, or **29,858,052,096** including the rank-8 attention adapters.
Routing still selects eight experts per token. Exporting changes the model;
the original model's scores do not establish this derivative's scores.

Collect actual router top-k counts using `RouterCounter` from
`scripts/dcc_prune_experts.py` on training examples only, before choosing experts.
Its `report` requires the source model ID/revision and training task IDs; use
`counter.valid_tokens(attention_mask)` around padded forward passes and call
`counter.close()` afterwards. The exporter selects the most-used experts in
each layer, retaining original expert order and remapping every router row and
expert projection consistently. Its metadata plan reads checkpoint headers;
the export audits saved shapes and logical parameter counts before publication.

```bash
python scripts/dcc_prune_experts.py plan \
  --source "$MODEL_CACHE" --calibration "$RUN_ROOT/routing-counts.json" \
  --keep-experts 125 --max-parameters 30000000000
python scripts/dcc_prune_experts.py export \
  --source "$MODEL_CACHE" --calibration "$RUN_ROOT/routing-counts.json" \
  --keep-experts 125 --max-parameters 30000000000 --workers 2 \
  --model-id DCC-Qwen3MoE-29.85B-125E-traincal \
  --output "$RUN_ROOT/base-125"
```

Use the exported directory as `--model` for training and evaluation. A compatible
attention adapter may warm-start it; subsequent adapter checkpoints record the
explicitly loaded base instead of the adapter's historical base path. Validate
the derivative with the intended HF and inference runtimes before benchmarking.

Install `configs/dcc/requirements-training.txt` in a separate environment with
a CUDA-compatible PyTorch build. The grouped MoE implementation requires
PyTorch 2.10 or newer and an SM80-or-newer GPU. One BF16 model replica must fit
on each device; this is data parallel adapter training, not tensor parallelism.
Choose the number of workers for the hardware available to you.

```bash
python -m torch.distributed.run --standalone --nproc-per-node="$WORKERS" scripts/dcc_train.py \
  --model "$MODEL_CACHE" --train-data "$DATA_ROOT/cuad/train.jsonl" \
  --output-dir "$RUN_ROOT/training" \
  --sft-steps 32 --rl-steps 24 --group-size 2
```

The warmup balances positive and negative CUAD questions and uses only exact
answers visible in the actual encoded prompt. Overlong examples are skipped.
GRPO samples independent trajectories with temperature 1, top-p 1 and top-k 0.
Inherited sampling filters, repetition penalties, n-gram constraints and forced
or suppressed tokens are disabled, so rollout probabilities match the raw
model log probabilities used for both old and current policy scores. Greedy
evaluation is unchanged. The earliest trial used temperature 0.8 with nucleus
and top-k sampling; its sampling policy did not exactly match those raw scores.

Training computes fixed old policy log probabilities and normalizes rewards
within each valid group. Each rollout batch receives one optimizer update,
with no policy epochs or rollout reuse, so the initial probability ratio is
one and clipping normally does not activate. Its grounded CUAD SFT auxiliary
has coefficient 0.1 by default. Groups with equal rewards have zero policy
gradient; the auxiliary may still update the adapter. Infrastructure failures
are excluded. Only generated assistant tokens receive loss, including tool
requests. Tool observations never become targets. LoRA dropout is disabled;
frozen model weights are never synchronized or updated. Adapter gradients are
averaged across all workers before one AdamW step.

Training accepts the train split only. All 200 official WideSearch tasks are
held out. Independent web curricula must identify their source as
`independent_curriculum`; they can participate in RL, while the span-based SFT
auxiliary applies only to CUAD. An all-web run should set `--sft-steps 0`.

Adapter checkpoints, optimizer/RNG state, exact token trajectories and per-rank
training logs are written under the explicitly configured output directory.
`--adapter` warm-starts adapter weights for a new run; it does not restore the
optimizer, step counter or RNG state from `training_state.pt`. To continue with
pure RL, use `--sft-steps 0 --sft-coefficient 0` and a new output directory.
Equal-reward groups contribute zero RL gradient. If all workers have zero
reward variance, there is no new policy-gradient signal, although existing Adam
momentum can still move parameters. Inspect reward variance and gradient norms
before describing such a run as successful RL learning. Keep runtime paths and
credentials outside the repository.

The optional mixed SFT continuation trains direct grounded CUAD answers and
actual search/read/final teacher segments together. Prepare its retrieval
profile and teachers using training data only:

```bash
python scripts/dcc_fit_retrieval.py \
  --data "$DATA_ROOT/cuad/train.jsonl" \
  --output "$RUN_ROOT/cuad-retrieval-profile.json"
python scripts/dcc_prepare_web_curriculum.py \
  --output-dir "$RUN_ROOT/web-curriculum" --teachers
python scripts/dcc_prepare_cuad_tools.py \
  --data "$DATA_ROOT/cuad/train.jsonl" \
  --retrieval-profile "$RUN_ROOT/cuad-retrieval-profile.json" \
  --tokenizer "$MODEL_CACHE" --episodes 128 --negative-episodes 48 \
  --output "$RUN_ROOT/cuad-tool-teachers.jsonl"
```

The following reproduces the second mixed phase's configuration. Set
`WARMSTART_ADAPTER` to the first mixed phase's saved adapter, whose actual
lineage and validation results are recorded in [dcc_experiment.md](dcc_experiment.md).
This creates a fresh optimizer; it does not restore optimizer state.

```bash
python -m torch.distributed.run --standalone --nproc-per-node="$WORKERS" \
  scripts/dcc_tool_sft.py \
  --model "$MODEL_CACHE" --adapter "$WARMSTART_ADAPTER" \
  --records "$RUN_ROOT/web-curriculum/teacher_train.jsonl" \
  --cuad-train-data "$DATA_ROOT/cuad/train.jsonl" \
  --cuad-tool-records "$RUN_ROOT/cuad-tool-teachers.jsonl" \
  --retrieval-profile "$RUN_ROOT/cuad-retrieval-profile.json" \
  --output-dir "$RUN_ROOT/tool-sft-round2" \
  --steps 64 --save-every 16 --learning-rate 5e-6 \
  --web-fraction .25 --cuad-tool-fraction .3 --negative-fraction .25 \
  --max-prompt-tokens 8192 --max-new-tokens 2048 \
  --batch-tokens 32768 --samples-per-step 4 --initial-chunks 4 --seed 45
```

The measured run used eight workers. Each worker accumulates gradients over
individual examples; `--batch-tokens` is a sampling budget, not a padded
forward batch allocation. Averaging weights each actual example equally across
workers, including steps where workers sample unequal counts. Dynamic direct
CUAD examples are skipped when gold text is not visible in the encoded prompt
or the target exceeds its budget. Inspect the saved sampled IDs, stages and
skip counts rather than assuming every eligible teacher segment was used.
The held-out WideSearch tasks and either benchmark's official test labels must
not enter teacher construction, retrieval-profile fitting or checkpoint selection.

The historical third mixed phase starts from round-2 step 32, using the same
training data and a fresh optimizer. Its eight-worker, seed-46 configuration is:

```bash
python -m torch.distributed.run --standalone --nproc-per-node="$WORKERS" \
  scripts/dcc_tool_sft.py \
  --model "$MODEL_CACHE" --adapter "$RUN_ROOT/tool-sft-round2/step_0032" \
  --records "$RUN_ROOT/web-curriculum/teacher_train.jsonl" \
  --cuad-train-data "$DATA_ROOT/cuad/train.jsonl" \
  --cuad-tool-records "$RUN_ROOT/cuad-tool-teachers.jsonl" \
  --retrieval-profile "$RUN_ROOT/cuad-retrieval-profile.json" \
  --output-dir "$RUN_ROOT/tool-sft-round3" \
  --steps 32 --save-every 8 --learning-rate 2e-6 \
  --web-fraction .25 --cuad-tool-fraction .15 --negative-fraction .10 \
  --max-prompt-tokens 8192 --max-new-tokens 2048 \
  --batch-tokens 32768 --samples-per-step 4 --initial-chunks 8 --seed 46
```

Its four checkpoints were compared on all 820 validation questions and the
independent web validation. The historical selection rule and measured results
are recorded in [dcc_experiment.md](dcc_experiment.md).

## Strict-30B continuation (seed 47)

The separate 125-expert experiment warm-starts the historical round-3 step-8
adapter on the physically pruned base, with a fresh optimizer and seed 47.
Its total, including LoRA, is **29,858,052,096 parameters**. Set `PRUNED_MODEL`
to the audited export, `WARMSTART_ADAPTER` to that historical adapter,
`STRICT_RUN_ROOT` to a new artifact directory and `WORKERS=8`:

```bash
python -m torch.distributed.run --standalone --nproc-per-node="$WORKERS" \
  scripts/dcc_tool_sft.py \
  --model "$PRUNED_MODEL" --adapter "$WARMSTART_ADAPTER" \
  --records "$RUN_ROOT/web-curriculum/teacher_train.jsonl" \
  --cuad-train-data "$DATA_ROOT/cuad/train.jsonl" \
  --cuad-tool-records "$RUN_ROOT/cuad-tool-teachers.jsonl" \
  --retrieval-profile "$RUN_ROOT/cuad-retrieval-profile.json" \
  --output-dir "$STRICT_RUN_ROOT/tool-sft-strict" \
  --quantization none --lora-rank 8 \
  --steps 32 --save-every 8 --learning-rate 2e-6 \
  --web-fraction .25 --cuad-tool-fraction .15 --negative-fraction .10 \
  --max-prompt-tokens 8192 --max-new-tokens 2048 \
  --batch-tokens 32768 --samples-per-step 4 --initial-chunks 8 --seed 47
```

Routing calibration uses 1,024 training-only examples and seed 47; remove the
three least-used experts per layer with deterministic expert-ID ties. The
continuation completed 32 steps and 1,024 examples. Validation of steps
8/16/24/32 is complete on all 820 CUAD questions and the independent web8,
with full denominators and zero infrastructure errors. The frozen rule
maximizes the harmonic mean of CUAD micro F1 and web strict row F1, then CUAD
AUPR, then the earlier step. It selected step 24 and froze that adapter at
`2026-10-19T10:40:45Z`. The selected weights inherit **24 new continuation
steps and 768 sampled examples**, plus historical warm-start training; the
last eight steps of the complete run are absent from them. The four-candidate
results are recorded in [dcc_experiment.md](dcc_experiment.md). This is
separate from the seed-46 run and its historical eligibility floors.

Fresh official generation started at `2026-10-19T10:41:11Z` with this same
frozen adapter for both benchmarks; generation and strict-proxy scoring finished
at `2026-10-19T11:25:40Z`. Both tests have full coverage and zero infrastructure
failures. CUAD full-test aggregation is complete for
all 4,182 questions (coverage 1.0): 4,180 completed, two reached `max_steps`
and remain in the denominators, and none had infrastructure failures. Its AUPR
is 23.1103% and micro F1 is 42.1842%; the independent final tensor, optimizer
and scoring audit passes all 284 checks. See [dcc_experiment.md](dcc_experiment.md)
for the other metrics and artifact record. WideSearch retains all 200 tasks:
75 completed and 125 evidence failures. Its strict-proxy macro row F1 is 0%,
macro item F1 is 0.089230% and complete-table accuracy is 0%.
The full-200 custom-v9 judge completed at `2026-10-19T11:33:22Z` with zero
judge failures: macro row F1 is 0.0250%, macro item F1 is 0.120044%, and
complete-table accuracy is 0%. These are custom-Qwen judge scores. Of all 200
outputs, 195 are empty: 70 of the 75 completed episodes and all 125 evidence
failures. This exposes a severe answer-coverage and citation problem; it is
not evidence of improved WideSearch performance. The independent final audit
passes all 284 checks, including replay of all 25 fresh captured judge calls.
The final verified record is `strict30b-v1/final-result-verified.json` within
the run artifact directory; original pre-audit records remain unchanged.

In this frozen runtime, `--timeout 1800` is an episode limit starting after
the concurrency semaphore is acquired. WideSearch allows 8,192 generated
tokens per turn and up to 12 turns. Synchronous vLLM `LLM.generate` returns
only after the entire batch finishes; cancellation of an awaiting episode
cannot abort a `generate` already running through `asyncio.to_thread`. GPU
execution can outlast the episode deadline, and slow batch members can delay
other results. Direct cross-thread aborts are avoided because vLLM V1 is not
thread-safe for that use. The frozen official execution source remains
`b84a211d`.

Earlier official-test results have already been seen; this repeat is not an
untouched first-use holdout. Old and new official scores must not choose
experts, hyperparameters or the checkpoint. Keep the unchanged custom-v9 judge
and its disclosed development-calibration limitations; it is separate from
the strict-30B policy and is not the official GPT-4.1 leaderboard judge.

## Evaluation

Evaluate each benchmark in a separate directory, with a budget appropriate to
its answer size. These HF examples illustrate CLI usage; the strict-30B
experiment uses the frozen validation-selected adapter and graph vLLM as
described in [dcc_vllm.md](dcc_vllm.md):

```bash
python -m torch.distributed.run --standalone --nproc-per-node="$WORKERS" scripts/dcc_model_evaluate.py \
  --model "$MODEL_CACHE" --adapter "$RUN_ROOT/training/final" \
  --data "$DATA_ROOT/cuad/test.jsonl" --output-dir "$RUN_ROOT/cuad" \
  --max-steps 3 --max-new-tokens 1024 --max-prompt-tokens 8192 --batch-size 8

python -m torch.distributed.run --standalone --nproc-per-node="$WORKERS" scripts/dcc_model_evaluate.py \
  --model "$MODEL_CACHE" --adapter "$RUN_ROOT/training/final" \
  --data "$DATA_ROOT/widesearch/test.jsonl" --output-dir "$RUN_ROOT/widesearch" \
  --max-steps 12 --max-new-tokens 8192 --max-prompt-tokens 16384 --batch-size 1
```

Long-prompt MoE prefill can require much more memory than decoding. Start
WideSearch with a per-worker batch size of one and increase concurrency only
after measuring memory use. An out-of-memory error aborts the run rather than
becoming a model score; lower the batch size and resume with unchanged semantic
budgets. Apply the same caution to local judge batching.

Full-expert NF4 is an explicit alternative when the BF16 base weights leave too
little memory for activations. Install
`configs/dcc/requirements-quantized.txt` in a **different environment** and add
`--quantization nf4` to training or evaluation. The default remains BF16 with
Transformers 5.20.1 and grouped MoE. The quantized environment pins Transformers
4.57.6, whose Qwen3 MoE experts use individual linear modules that bitsandbytes
can quantize. It uses NF4 weights, double quantization, BF16 computation and
attention LoRA. Expert computation uses that version's individual expert
implementation; it does not use grouped MoE and may have lower throughput.
Do not replace the dependency versions in an environment with running jobs.

Transformers 5.20.1 stores experts as fused three-dimensional parameters.
Its native bitsandbytes integration only converts linear modules, so simply
setting `load_in_4bit` there leaves almost all expert weights unquantized.
PEFT preparation can then upcast those frozen weights to FP32. The optional
loader rejects that environment, requires every expert matrix to be NF4 before
preparing the model for k-bit training, and prepares the base **before** adding
or loading trainable adapters. Router and output-head weights stay unquantized;
PEFT preparation preserves quantized experts and casts other floating base
weights to FP32. Quantized inference uses the same base dtypes as training,
without enabling gradient checkpointing. Only adapters receive gradients and
optimizer updates.

Training logs and quantized evaluation reports record the original logical base
parameter count, expert coverage, unquantized FP32 fraction, packed parameter
bytes and quantization metadata bytes. Packed tensor element counts alone do
not describe the original model size. These storage figures exclude activations,
KV cache, CUDA workspaces and allocator reservations; use actual memory results
to choose sequence lengths and batches. Quantization changes predictions, so
use separate output directories and report quantized precision for both policy
and any local judge. It does not change the model's original parameter count.

Run the CPU tiny-model checks in that separate environment with
`CUDA_VISIBLE_DEVICES='' PYTHONPATH=slime python -m pytest tests/test_dcc_quantization.py`.
They load and quantize real tiny Qwen3 MoE experts, verify finite nonzero adapter
gradients through frozen NF4 experts, check that frozen weights stay unchanged,
and round-trip an adapter checkpoint. Before scheduling a full run, measure one
model on an available GPU:

```bash
python scripts/dcc_quantization_smoke.py \
  --model "$MODEL_CACHE" --output-dir "$RUN_ROOT/nf4-smoke" \
  --prompt-tokens 4096 --max-new-tokens 16 --train-step
```

The smoke records actual loading, full prefill and synthetic training-step
memory and timing. Its synthetic prompt never reads benchmark labels. A short
synthetic step verifies runtime correctness; it does not establish training
quality. Raise the prefill budget in a separate run to test the intended context
length. Historical validation on the original 128-expert model quantized all
18,432 expert matrices. Loaded allocation was 16.79 GiB and full 16,384-token
prefill plus 16 generated tokens peaked at 21.17 GiB, compared with 60.65 GiB
for the BF16 implementation.
A synthetic LoRA optimizer step produced finite, nonzero gradients.
The unfused NF4 runtime trades speed for memory: matched single-request decode
measured 2.12 tokens/s versus 16.22 tokens/s for fused BF16. Choose batch size
using actual memory and throughput measurements; quantization alone does not
establish faster training or better benchmark scores.

The evaluator resumes saved task IDs and rejects changed generation settings
in the same directory. Use different directories for base and trained policies.
`--retry-infrastructure` retries infrastructure failures. Set a reliable search
provider as described in [dcc.md](dcc.md); record the actual provider and any
fallback coverage restrictions with the scores.

After every prediction has been persisted, run the same evaluation command
with `--score-only` in a single process for CUAD or a strict WideSearch proxy.
Add `--local-judge` under `torchrun` to score WideSearch with the **frozen base
Qwen model** across all workers, then run `--score-only --merge-judge` in a single
process. This uses the official metric pipeline with a custom judge, not the
published GPT-4.1 judge. Judge input is never silently truncated. Full coverage,
infrastructure failure counts, judge failures, row F1 and item F1 are retained
in the reports. Reports and predictions belong in the external artifact root.

For uneven answer sizes, optionally pass `--judge-scheduling balanced` to both
the local-judge workers and the merge command. After validating the original
generation manifests and task ownership, this assigns large saved answer tables
first using a deterministic, count-capped schedule. It uses answer sizes only,
keeps every task, and leaves prediction files and generation ownership intact.
The reports record the scoring assignment; merge recomputes and verifies it.
The default remains `round_robin`. Use a fresh scoring directory when changing
the scheduling mode, and never combine reports from different modes or sources.
