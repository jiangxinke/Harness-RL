# Native HF Web whole-turn history

`scripts/dcc_online_web_evaluate.py` supports the default-off flags
`--prompt-history-whole-turns` and `--prompt-history-keep-recent-turns N`
(default `N=1`). Base and adapter comparisons, including `--score-only`, must
use the same flags. This native HF integration supports Web evaluation only.
CUAD schemas, joint JSON policies, training behavior capture and source
truncation are rejected before model submission. Training support is deferred.

The existing `fit_prompt_history` algorithm measures the complete request
with the checkpoint's unchanged chat template and reserves both complete
generation phase budgets. If necessary, it removes the earliest complete
assistant/action plus observation turns, preserving the original system/task,
at least the requested number of latest turns, the latest actual source,
complete JSON/text and current controls. A notice identifies omitted history
without providing source evidence. Oversized protected context still fails
before generation. There is no retry or source summarization.

Every enabled native trace stores the full original request, actual compacted
prompt token IDs, the versioned policy declaration, and the complete omission
report, including original message indices and observed URL metadata. Failed
preflight owners retain the full original token IDs and an explicit
`generation_submitted=false`. Timeout-late failures remain owned and keep
their infrastructure outcome. Generation closure and enabled score-only
reconstruct the plan from the original full request, then compare the exact
report, retained prompt IDs and native open-thinking suffix. Score-only loads
the same local tokenizer and context configuration with downloads disabled;
it never loads model weights. Stored reports are not replay inputs.

The native real thinking close, JSON EOS/schema validation, evidence guards,
scorer, task denominators and zero scores for failures are unchanged. Disabled
runs retain their existing serialized fields and tokenizer-free score-only
behavior. This option changes the model's available history, so it requires a
fresh matched experiment; fitting an old failed request proves only that its
context can fit, not successful completion or an improved score.

CPU checks use invented requests and outputs, a real tiny local tokenizer,
and the real native policy/ownership code. They cover suffix preservation,
whole-turn omissions, mixed batches, unsupported modes, protected-context
failure, cancelled-request draining, metadata/token tampering, and matched
base/adapter and score-only configuration mismatches.
