The optional online CUAD entrypoint samples a fresh group of thoughts from the
current policy before every update. It uses the native opening/closing thinking
tokens and the existing public CUAD action/final schemas. JSON is generated
greedily with the structural xgrammar backend and then checked against the
original public schema. Only the sampled thought tokens receive GRPO loss.
There is no SFT auxiliary, offline preference pair, or DPO loss.

This is an independent HF LoRA runtime with optional NF4 loading, separate from Slime/Megatron and the
existing BF16 vLLM native evaluation protocol. Start with a bounded pilot and
compare base and trained adapters under this exact HF precision, harness, token
budgets and evaluation sampling. Scores from older vLLM runs are not matched
controls for this runtime. The matched CUAD evaluator rejects Web. The trainer offers separately reviewed
Web TRAIN groups through the opt-in switch described in `dcc_online_web.md`.

Install `configs/dcc/requirements-online-grpo.txt` in the separate quantized
environment. Set paths locally; none is embedded in the code:

```bash
python scripts/dcc_online_grpo.py \
  --model "$MODEL_CACHE" --train-data "$CUAD_TRAIN" \
  --source-commit "$SOURCE_COMMIT" --output-dir "$RUN_ROOT/pilot" \
  --steps 4 --group-size 2 --max-steps 2 \
  --thought-tokens 4096 --json-tokens 4096 \
  --initial-chunks 8 --retrieval-context-chars 800 \
  --retrieval-max-context-chars 16000 \
  --retrieval-profile "$RETRIEVAL_PROFILE" \
  --quantization nf4 --lora-rank 16 --learning-rate 1e-5 \
  --cpu-preflight --plan-world-size 1
```

After CPU preflight, omit its last two options for a single GPU, or use
`torchrun` and specify the intended world size during preflight. The source
must be its actual clean commit or an explicitly identified Git export. The
output directory must be fresh. Production artifacts belong in the operator's
configured durable storage. The default four-step pilot is a runtime check;
it is not evidence of benchmark improvement.

Thought sampling uses raw softmax at temperature 1, without nucleus/top-k or
inherited penalties. The old-policy denominator is recorded directly from the
raw generation scores for each actually sampled thought token. A streaming
processor retains one vocabulary score matrix and token-level history, without
changing scores or consuming RNG. Missing, nonfinite or misaligned behavior
scores reject the group; teacher-forced replay is never a fallback. The replay
is diagnostic because cached sampling and full-sequence replay can differ
numerically even before an update. Logs distinguish behavior-versus-replay,
behavior-versus-current and replay-versus-current differences, plus actual
importance-ratio ranges and clip fractions. Unscored groups record null
diagnostics. Each group receives one replay and at most one optimizer update. Group advantages retain reward-standard-deviation
normalization; the token loss uses a fixed group/turn/thought-budget denominator
instead of dividing by observed response lengths. This is fixed-budget GRPO,
not the complete Dr.GRPO algorithm.

Model truncation and invalid JSON retain their actual sampled thoughts and
zero rewards in the group. Version `online-native-HF-thought-only-GRPO-v2`
also retains a trajectory's prior sampled thoughts with zero reward when a
later full request fails the actual raw-HF prompt preflight. Successful Web
siblings keep their original TRAIN literal rewards. Initial prompt overflow,
unverified context failures and infrastructure failures discard the whole group.
The shared TRAIN classifier requires actual retained token boundaries, native
completion ownership and aligned streamed behavior logprobs; it never supplies
missing tokens or scores. The fixed objective denominator is unchanged.

Training configurations and reviewed Web bindings record
`hf-native-context-sampled-prefix-v2`. Raw training capture journals include
the rejected request's preflight receipt without a generated completion.
Capture-disabled evaluation keeps its completion-only trace stream and its
existing statuses/rewards. Serialized outcomes require verification against
the actual native generation/error journals and re-encoded failed request
before `restore_verified_native_context` restores their eligibility; plain
diagnostic strings do not authorize gradients. Episode diagnostics gain source,
stage and turn metadata; the legacy `trainable_steps` property is unchanged.
Previously frozen v1 sources and adapters retain their original whole-group
context-discard semantics and must not be labeled as v2 training. This rule is
the new plugin default and has no separate switch.

Constant rewards give exact zero advantages.
All replicas participate in gradient averaging; a zero global gradient skips
Adam entirely, including its momentum update. Logs distinguish rollout
iterations, applied optimizer updates, changed adapter tensors, and whether
the run had any policy-gradient updates. A complete run with zero updates is
never reported as effective learning.

CPU checks include real tiny Qwen3MoE LoRA replay, an analytic categorical
gradient oracle, failure coverage, and real Gloo synchronization. The policy
checks include actual Hugging Face generation and xgrammar. These checks do
not establish GPU correctness or benchmark improvement; both require the
actual pilot and a matched base comparison.

The optional `--compute-profile bf16-autocast-v1` applies the same BF16
autocast context to sampling, old/current replay and evaluation. The default
`legacy` preserves the original computation. Base storage follows the loader,
LoRA master weights remain FP32, and log-softmax uses FP32. Checkpoints record
the selected context and actual registered parameter dtypes; adapter reload
and matched evaluation reject changed profiles or precision. Older adapters
without profile metadata retain `legacy`. Use the same option for training
and both evaluation arms. This profile does not promise BF16 KV tensors or
lower memory for every quantized operator: bitsandbytes may return its input
dtype. Real GPU throughput and memory must be measured before scaling a run.

Native generation explicitly disables inherited checkpoint penalties, forced
tokens and minimum lengths. Transformers versions exposing `use_model_defaults`
receive `False`; versions that removed it receive every public generation field
as an explicit argument, including disabled optional fields set to `None`.
The implementation checks the base HF API rather than PEFT's generic wrapper.
Actual CPU generation/replay tests cover both pinned HF environments; a new
package version is still a distinct runtime for matched evaluation.

With nonzero retrieval context, the source-character budget reserves original
ranked hit cores before expanding their surroundings. If the core union exceeds
the cap, whole cores are retained greedily by rank; later overlapping or shorter
cores can still fit. Expansion can no longer displace an already retained core.
Zero context keeps the original retrieval behavior. This changes visible
evidence and must be held fixed between base and adapter evaluation; evidence
coverage alone is not a benchmark score.

The optional positive integer `--web-text-window-chars` limits only the body
text returned by ordinary `fetch_url` reads. When omitted, the window inherits
the original `max_page_chars` value, which is 14000 in the shared native runner.
Downloaded text remains complete in the episode cache. Source offsets,
`next_start`, navigation metadata, citation checks and scoring are unchanged.
Cached-row selection retains its original 14000-character serialized JSON
budget, including literal records and provenance, even with a smaller body
window. The trainer and native Web evaluator record the effective body window
separately from that original budget. Base and adapter Web evaluation must use
the same window; the comparison rejects a mismatch. A 6000-character window is
a TRAIN pilot candidate, not a demonstrated improvement or a new default.


The matching evaluator uses the same HF loader for base and adapter, including
NF4 preparation and nonquantized base dtypes. Use a fresh output for each arm.
The following settings match the example pilot above; set all paths locally:

```bash
EVAL_ARGS=(
  --model "$MODEL_CACHE" --data "$CUAD_DEV"
  --source-commit "$EVALUATION_SOURCE_COMMIT"
  --quantization nf4 --profile thinking --seed 47
  --max-steps 2 --thought-tokens 4096 --json-tokens 4096
  --initial-chunks 8 --retrieval-context-chars 800
  --retrieval-max-context-chars 16000
  --retrieval-profile "$RETRIEVAL_PROFILE"
)
python scripts/dcc_online_evaluate.py "${EVAL_ARGS[@]}" \
  --output-dir "$RUN_ROOT/base" --generate-only
python scripts/dcc_online_evaluate.py "${EVAL_ARGS[@]}" \
  --adapter "$FINAL_ADAPTER" --output-dir "$RUN_ROOT/adapter" --generate-only
python scripts/dcc_online_evaluate.py "${EVAL_ARGS[@]}" \
  --output-dir "$RUN_ROOT/base" --score-only
python scripts/dcc_online_evaluate.py "${EVAL_ARGS[@]}" \
  --adapter "$FINAL_ADAPTER" --output-dir "$RUN_ROOT/adapter" \
  --base-output-dir "$RUN_ROOT/base" --score-only
```

Generation can run in parallel on separately reserved GPUs. For sharded
generation, add the same `--shards N` to every invocation and assign each
generation process a unique `--shard-index I`. Score only after every shard
has closed. Adapter scoring requires a completed base report with identical
public tasks, scoring references, packages, retrieval, sampling, token budgets
and observed base precision. Missing, duplicate, extra or inconsistent shards
are rejected. The real split is preserved; rollout copies have no gold labels.
Every failed episode remains in the denominator, with a separate
`failure_zero_macro_f1` that assigns zero even to failures on negative tasks.

Evaluation and training source commits are recorded separately. An adapter
with no actual online optimizer updates is never marked as having recorded
policy-gradient updates. Reports retain failures and deltas; they do not
automatically assert a capability gain or a statistically reliable improvement.

The shared `NativeHFPolicy` also accepts the exact public WideSearch action,
final-answer and review schemas, including dynamic read-source citations and
cached-row arguments. It reuses the existing review and bounded-whitespace
grammars and validates the original public schema after decoding. Unsupported
quoted or backslash-containing column keys are rejected before sampling, as in
the existing pinned backend. Web TRAIN groups are explicitly opt-in and require a source/data/reward binding
as documented in `dcc_online_web.md`. The matching CUAD evaluator remains CUAD-only.

TRAIN task selection defaults to `--cuad-task-sampling stratified-3to1`, retaining
the original seeded positive/negative decks and the exact three-to-one
positive-to-negative draw schedule when both exist. This uses TRAIN gold labels
for stratification and is recorded explicitly.

The optional `--cuad-task-sampling uniform` samples every CUAD TRAIN task index
once in a seeded shuffled deck before starting the next deck. Selection does
not inspect labels, rewards or outcomes, and it supports an all-negative TRAIN
pool. The training entrypoint still requires explicit TRAIN reward labels for
scoring. The option also applies to the CUAD draws in mixed CUAD/Web training;
Web selection and the CUAD/Web draw ratio are unchanged. The configuration
records the selected mode, whether it uses labels, and the actual planned task
IDs. Uniform sampling is a distribution-matching experiment; it is not evidence
of improved benchmark performance. Compare it with the default while holding
the seed, budgets, learning rate and all other training choices fixed.

Neither mode resamples tasks based on outcomes. Phase events separately record collection,
reward signal, replay, optimizer application and checkpoint publication. Closed
native traces include batch thought/JSON elapsed time; absent JSON is null.

Distributed training uses a separate full-world Gloo CPU rendezvous after
rollout and replay, before submitting gradient all-reduces to NCCL. Each rank
first synchronizes its own CUDA work. This prevents an early rank's GPU
collective from waiting throughout another rank's longer generation. Readiness
has a finite two-hour timeout; NCCL watchdog monitoring remains enabled.
Readiness timing is recorded separately from optimizer application. Gradient
averaging, clipping, zero-signal skipping and the objective are unchanged.
This is a synchronization safeguard, not evidence of improved model quality
or proof that every NCCL watchdog failure has the same cause.
