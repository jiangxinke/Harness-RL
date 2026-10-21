The optional online CUAD entrypoint samples a fresh group of thoughts from the
current policy before every update. It uses the native opening/closing thinking
tokens and the existing public CUAD action/final schemas. JSON is generated
greedily with the structural xgrammar backend and then checked against the
original public schema. Only the sampled thought tokens receive GRPO loss.
There is no SFT auxiliary, offline preference pair, or DPO loss.

This is an independent HF/NF4 LoRA runtime, separate from Slime/Megatron and the
existing BF16 vLLM native evaluation protocol. Start with a bounded pilot and
compare base and trained adapters under this exact HF precision, harness, token
budgets and evaluation sampling. Scores from older vLLM runs are not matched
controls for this runtime. Web is rejected before loading weights; its native
schema backend needs a separate integration.

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
inherited penalties. Old scores are fixed before one replay and one possible
optimizer update. Group advantages retain reward-standard-deviation
normalization; the token loss uses a fixed group/turn/thought-budget denominator
instead of dividing by observed response lengths. This is fixed-budget GRPO,
not the complete Dr.GRPO algorithm.

Model truncation and invalid JSON retain their actual sampled thoughts and
zero rewards in the group. Infrastructure/context failures discard the whole
group and are logged explicitly. Constant rewards give exact zero advantages.
All replicas participate in gradient averaging; a zero global gradient skips
Adam entirely, including its momentum update. Logs distinguish rollout
iterations, applied optimizer updates, changed adapter tensors, and whether
the run had any policy-gradient updates. A complete run with zero updates is
never reported as effective learning.

CPU checks include real tiny Qwen3MoE LoRA replay, an analytic categorical
gradient oracle, failure coverage, and real Gloo synchronization. The policy
checks include actual Hugging Face generation and xgrammar. These checks do
not establish NF4 GPU correctness or benchmark improvement; both require the
actual pilot and a matched base comparison.


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
