# Local MoE training and evaluation

The optional Transformers runtime trains attention LoRA adapters using the same
DCC episode tools and rewards as evaluation. It leaves the existing Slime,
StackPlanner, OLE and Megatron training defaults intact. The selected model is
`Qwen/Qwen3-30B-A3B-Instruct-2507`: nominal 30B class, **30.5B actual total
parameters**, about 3.3B activated per token. A strict total-parameter cap of
30B excludes this model.

Install `configs/dcc/requirements-training.txt` in a separate environment with
a CUDA-compatible PyTorch build. The grouped MoE implementation requires
PyTorch 2.10 or newer and an SM80-or-newer GPU. One BF16 model replica must fit
on each device; this is data parallel adapter training, not tensor parallelism.
Choose the number of workers for the hardware available to you.

```bash
torchrun --standalone --nproc-per-node="$WORKERS" scripts/dcc_train.py \
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

Evaluate each benchmark in a separate directory, with a budget appropriate to
its answer size. For example:

```bash
torchrun --standalone --nproc-per-node="$WORKERS" scripts/dcc_model_evaluate.py \
  --model "$MODEL_CACHE" --adapter "$RUN_ROOT/training/final" \
  --data "$DATA_ROOT/cuad/test.jsonl" --output-dir "$RUN_ROOT/cuad" \
  --max-steps 3 --max-new-tokens 1024 --max-prompt-tokens 8192 --batch-size 8

torchrun --standalone --nproc-per-node="$WORKERS" scripts/dcc_model_evaluate.py \
  --model "$MODEL_CACHE" --adapter "$RUN_ROOT/training/final" \
  --data "$DATA_ROOT/widesearch/test.jsonl" --output-dir "$RUN_ROOT/widesearch" \
  --max-steps 12 --max-new-tokens 8192 --max-prompt-tokens 16384 --batch-size 4
```

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
