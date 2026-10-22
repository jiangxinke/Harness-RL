# Web TRAIN thought budget close

`scripts/dcc_online_grpo.py --web-train-thought-budget-close` opts into a
separate Web TRAIN policy. It is disabled by default and requires
`--enable-online-web`, raw behavior capture, thought-only token-GRPO, and the
existing greedy JSON phase. Joint JSON training and episode-GSPO are rejected
before model or CUDA initialization. The evaluation-only
`--web-thought-budget-close` remains a separate mode.

For a thought budget `B`, the controlled Web policy uses the unchanged raw
softmax at generated positions `0` through `B-2`. At position `B-1`, if reached,
its distribution is a parameter-independent singleton on the actual native
`</think>` token ID `151668`. A row-selective logits processor applies this
distribution immediately before the streaming behavior tracker and actual HF
multinomial sampling. No token is appended, no additional token is budgeted,
and no cached old probability is replaced. Natural earlier closes and ordinary
early EOS retain their actual raw probabilities and outcomes.

The processor selects only Web and Web-review rows. A mixed physical batch can
contain CUAD rows, whose logits remain raw at every thought position. TRAIN
does not set HF's global `forced_eos_token_id`. JSON still uses the existing
per-row xgrammar, greedy decoding, schema postvalidation and zero policy-loss
mask. A successfully closed thought, including its real closing token, remains
the actual JSON prompt prefix.

## Replay and objective

All old thought logprobs come from the actual sampled distributions. The new
tracker records exactly `0` at an observed singleton-close position. Current,
diagnostic and adapter-disabled frozen-base replay use the same controlled
policy: raw model logprobs before that position, then logprob `0` and gradient
`0` at that position. The full actual token sequence remains in the decoder
context. Natural earlier closes remain trainable raw tokens.

The token-GRPO surrogate explicitly masks the singleton position out of its
numerator. Its denominator remains `group_size * max_steps * B`, including
when `B=1` or an episode ends early; it is never replaced with `B-1` or observed
length. Frozen-base K3 KL is zero at the singleton position because both
controlled policies assign probability one to the same token. The existing
zero-global-gradient rule prevents an Adam step when a group supplies only
deterministic positions or otherwise has zero gradient.

Step reports separate `forced_close_scored_tokens` and
`stochastic_scored_tokens`. Importance-ratio ranges and clipping fractions
cover stochastic positions only. Behavior/current/diagnostic alignment still
checks the complete actual sequence. Reports and training configuration carry
the distinct policy identity rather than describing forced positions as raw
model samples.

## Ownership and history

New Web completions use `BudgetClosedNativeHFCompletion`, protocol
`hf-native-web-thought-budget-close-training-v1`, behavior source
`streaming-generation-raw-prefix-singleton-close-v1`, an exact typed policy
profile, and actual `forced_thought_positions`. Closed owner/native journals,
delivered completions, native preflight and retained-context restoration must
agree on this identity, budget, token IDs and captured probabilities. Legacy
raw and evaluation traces cannot be converted into this training mode.

`--web-train-prompt-history-whole-turns` and
`--web-train-prompt-history-keep-recent-turns 2` may compose with this mode.
Whole-turn prompt planning is unchanged; its provenance records the composed
policy. Unfittable requests retain their original full-request native preflight
proof. A verified actual sampled prefix may be retained under the existing
context policy; an initial overflow has no invented completion. Existing
cancellation draining and journal ownership continue to apply to actual
completions, including sampled singleton closes.

With the flag absent or false, policy configuration, completion serialization,
native sources, generation settings, objective and random-number consumption
retain their existing behavior. Synthetic CPU controls cover processed-score
capture, mixed rows, natural close, early EOS, `B=1`, RNG parity, default parity,
fixed-denominator gradients, actual tiny LoRA/reference replay, closed journal
ownership, whole-turn history, context restoration and cancelled late
completions. These controls establish implementation behavior; a fresh matched
base/adapter evaluation is needed to establish capability improvement or GPU
memory/throughput effects.
