# Recorded native Thinking training

Native training is opt-in with `--native-thinking`. Ordinary training keeps its
existing path. Native targets contain the actual generated thought IDs through
the real closing token, followed by the schema-constrained JSON IDs and ordinary
EOS. Neither phase is re-encoded, repaired, or replaced by a teacher answer.

The separate approved TRAIN profile binds the model derivative, tokenizer,
source commit, prompt/context limits, generation seed, and action schemas. Its
default identity uses 4096 thought tokens, 4096 JSON tokens, and greedy sampling,
retaining the original v2 manifest. An explicit identity with `thought_tokens`
and `thought_sampling_profile` selects budgeted v3. Supported thought limits are
4096, 8192, and 16384. The `thinking` sampling profile uses temperature 0.6,
top-p 0.95, and top-k 20; JSON always uses neutral greedy sampling.

`scripts/dcc_collect_native_train.py generate` accepts public TRAIN inputs with
no gold and closes all journals. Only a later `select` command opens the separate
TRAIN labels. The default threshold is perfect TRAIN correctness. A required-only
Web TRAIN schema uses all required columns as an exact compound key; the actual
profile is recorded in accepted decisions. Held-out scoring is unchanged. CUAD
final targets may use the runner's real initial retrieval without invented tool
calls; every span must remain grounded in the observed source and encoded prompt.

`generate` accepts `--vllm-enforce-eager`, which is **off by default**
(`False`). Supplying it disables vLLM compilation/CUDA graphs for that fresh
collection and forwards `enforce_eager=True` to the engine. Record the chosen
execution mode with the collection configuration.

Fresh Web collection can also opt into
[whole-turn prompt history](dcc_prompt_history.md). Its approved identity and
runtime flags must agree, and the full original request remains available for
replay. Use a separate experiment when increasing the prompt budget; preserve
older CUAD data under its original identity.

Pass the selected Web/CUAD files and the same approved profile to
`scripts/dcc_tool_sft.py`, using `--native-train-profile` and
`--native-expected-source`. This validates all bundles before model loading.
Training limits must accommodate the actual recorded sequences. Legacy profiles
count overlong examples as skipped, never truncated. The separate Web32K profile
requires every recorded sequence to fit and fails before training on an
insufficient budget. Optimizer, sampling, gradient averaging, and checkpoint
handling retain the ordinary trainer's path.
Native CUAD sampling uses only stages actually observed; it creates no missing
search/read examples. `--quantization nf4` can reduce weight memory using the
separate pinned environment described in [dcc_training.md](dcc_training.md).

The default loss averages all completion tokens with equal example weight.
Two mutually exclusive options change the native SFT objective:
`--native-completion-json-mixture-loss` uses 80% completion mean plus 20% JSON
mean, and `--native-phase-normalized-loss` uses 20% thought mean plus 80% JSON
mean. Both require validated native phase IDs and retain every generated token.
The mixture option logs detached phase losses by domain, stage, and TRAIN
polarity, with counts and sums for aggregation across ranks. These are supervised
fine-tuning objectives; enabling them does not make the run reinforcement learning
or establish a benchmark gain. Record the option in the experiment declaration
and compare checkpoints under identical evaluation settings.

For matched native evaluation, both policies must use the same evaluation source,
budget, sampling, retrieval, tools, and scoring. Approve the actual training
source and protocol separately with `--native-lora-training-source` and
`--native-lora-training-protocol`. The adapter binding records the actual v3
training rollout configuration; the saved training configuration and update audit
must also match the frozen training plan. Protocol failures remain zero-score
observations in the full evaluation denominator. Runtime checks establish
compatibility; benchmark improvement requires a completed matched comparison.
