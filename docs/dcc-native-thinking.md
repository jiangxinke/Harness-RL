# Optional native Thinking evaluation

`--native-thinking` enables a two-stage Qwen3 Thinking-2507 protocol for CUAD
and WideSearch. It defaults off, preserving the existing evaluation path.

Use the vLLM async backend, `--web-structured-final --web-structured-actions`,
`--max-new-tokens 8192`, neutral greedy sampling, and a fresh output directory.
The two stages allow 4096 thought tokens followed by 4096 JSON tokens. The model
must generate its own closing thought token. The JSON stage continues the exact
observed token sequence using the public task schema. No forced close, extracted
answer, repair, or automatic retry is used.

CUAD exposes its existing document search/read tools and canonical span/confidence
answer format. The last turn requires the existing brief plan/checks review.
Empty answers are allowed. Original tools, evidence guards and scoring remain
responsible for answer correctness. Complete JSON Schema validation follows grammar
decoding, including confidence bounds.

A launcher must bind a verified committed source to `.dcc-native-source.json` in
its frozen source root, then pass its path through `--native-source-receipt`:

```json
{
  "version": "dcc-native-inference-source-v1",
  "source_commit": "ACTUAL_NEW_FROZEN_40_HEX_COMMIT",
  "base_source_commit": "5bed6188feaaf6c79fac53eead30cfbad83b0beb",
  "protocol": "native-thinking2507-real-close-two-stage-4096-4096-v2",
  "frozen": true
}
```

The receipt format check does not establish Git provenance by itself. Preserve the
actual source identity independently. Native manifests, runtime reports, predictions
and per-rank raw token journals must agree. Scoring verifies their consistency.
Protocol failures remain in the denominator with zero reward; timeouts retain only
actually observed outputs. Journals include unredacted model thoughts and should
be stored according to the deployment's data policy.

Native adapter evaluation additionally requires `--native-lora-derivative-id`
matching the caller-approved derivative and the adapter's recorded native TRAIN
configuration. Local attention adapter shapes, source identity and actual vLLM
request are checked before use. Scores obtained
with this protocol cannot establish a training gain over a different base harness
or token budget. CPU protocol tests and runtime smoke tests establish compatibility,
not benchmark quality.

Run the portable tests with `PYTHONPATH=slime python -m pytest -q
 tests/test_dcc_native_thinking.py`. Optional vLLM/xgrammar tests skip when their
runtime dependencies are absent; all tokenizer/engine fixtures are synthetic.

## Recorded native LoRA targets

`dcc_tool_sft.py --native-thinking` accepts independently approved TRAIN rollout
bundles through `--records` and optional `--cuad-tool-records`. Supply an explicit
`--native-expected-source` and a separate `--native-train-profile` binding the
actual frozen source, derivative, tokenizer, public schema settings and budgets.
Both complete input files are checked before any CUDA/model/optimizer work.
Targets contain only the actual generated thought and JSON token IDs, including
their generated close/EOS. Prompt, observation and history tokens receive no
direct target loss. Original example weighting, loss and optimizer are unchanged.

Native mode requires a fresh adapter and rejects the raw gold-only CUAD auxiliary.
Failed/incomplete generations belong in an audit stream. The collector must
independently establish TRAIN provenance and close its generation journals before
label-based selection; bundle assertions alone cannot prove this boundary.
Existing Web/CUAD validators still apply. Histories with unsupported feedback or
review controls and default/nullable CUAD reader windows may therefore be valid
at runtime while ineligible for this training stream. Do not rewrite their
messages to bypass validation. The opt-in code establishes data compatibility;
actual rollout, GPU training and paired quality evidence are separate checks.

## Closed-journal native TRAIN collection

`dcc_collect_native_train.py generate` accepts a public-only TRAIN file with null
gold and no answer/category metadata. It records exact requests and actual native
traces, closes every rank journal, then `select` separately loads TRAIN labels and
exports only complete, validated episodes meeting the declared correctness
threshold. Failed and ineligible episodes remain in the audit stream. No held-out
labels or repaired model output belongs in this process.

The collection profile uses initial retrieval of four chunks, no auto read, no
Web review/navigation controls, and explicit caller-approved step/token budgets.
This is a separate TRAIN sampling profile. Any evaluation of training gain must
compare the resulting adapter and its own unadapted base with an identical
evaluation harness and runtime budget.

The runner retains its default TRAIN scoring. The collector explicitly defers it
until journals are closed, so unlabeled public tasks never need placeholder gold.
Unsupported feedback and reader windows are retained as ineligible failures;
messages and original validators are never rewritten to make a sample eligible.

Recorded native CUAD episodes may answer directly from initial retrieval or skip
a reader call. Native training samples only their observed, validated stages and
reports unavailable category stages. Legacy teacher training retains its existing
full-stage requirement. No missing tool action is synthesized for native data.

## Explicit v3 thought budget and sampling

Defaults retain the v2 protocol exactly: 4096 thought tokens, 4096 JSON tokens,
and neutral greedy sampling. Explicit `--native-thought-tokens 8192` or `16384`,
or `--native-thought-sampling thinking`, enables the separate budgeted v3 protocol.
JSON remains 4096 tokens and neutral greedy. The thinking preset applies only to
the thought phase: temperature 0.6, top-p 0.95, top-k 20, with neutral penalties.
Keep the global evaluation sampler neutral so runner and JSON requests agree.

For a 16384-token thought budget, add these options to the same approved async
structured evaluation command, replacing its original total budget:

```bash
--native-thought-tokens 16384 \
--native-thought-sampling thinking \
--max-new-tokens 20480
```

`max_new_tokens` must equal thought+4096. The engine must reserve
`max_model_len >= max_prompt_tokens + max_new_tokens`; the evaluation entry point
sets this sum automatically. For example, a 32768-token prompt budget requires
53248 model tokens, while a 16384-token prompt budget requires 36864. Use fresh
outputs and compare base and adapter with identical benchmark inputs, harness
settings, phase sampling and budgets. A protocol change alone does not establish
an improvement or lower the original evidence/quality gates.

The model must still generate its own real closing thought token. Missing or
invalid close remains a terminal protocol failure with the observed IDs; no
forced close, repair, extraction or retry is added. JSON continues the exact
original prompt plus actual thought IDs. Manifest, runtime and native traces bind
the complete v3 configuration and per-phase limits/sampling. V3 also records the
actual effective sampling: pinned vLLM 0.11.2 normalizes greedy top-k -1 to 0,
both meaning disabled, while the configured dictionary retains -1. The source
receipt retains its existing v2 foundation marker and identifies the actual
frozen evaluation commit; the requested v3 generation protocol is recorded
separately in the complete native configuration.

To evaluate an actual native v2 adapter under v3, additionally supply
`--native-lora-training-source ACTUAL_APPROVED_40_HEX_TRAINING_COMMIT` and
`--native-lora-training-protocol native-thinking2507-real-close-two-stage-4096-4096-v2`,
alongside `--adapter` and `--native-lora-derivative-id`. Approval comes from the
caller. The actual adapter training configuration must match that independently
approved training source/protocol, derivative and model revision; actual LoRA
header shapes and request binding remain checked. Training and evaluation sources
may differ and both are recorded. The v2 TRAIN collector and training validators
retain their existing protocol and constraints.

## Episode diagnostics

Saved predictions and compact evaluation console records include an additive
`terminal_summary`. It distinguishes episode deadlines, dependency failures,
exhausted model turns, evidence rejections and native protocol failures using
fixed reason labels and counts. It contains no prompt, answer, source URL or
raw exception text. Full native traces and existing validators remain the
authority for inspecting a failure; the summary does not alter scoring or
generation budgets.

The existing `token_usage` fields keep their original meaning: token IDs in
returned policy completions. `token_accounting.logical` repeats those counts;
`token_accounting.native` separately records both phase-input prompts and
observed completion IDs, including a cancelled request without a returned
policy completion. Native prompt counts describe recorded inputs, without
adjusting for engine caching or billing. Zero observed completion IDs does not
prove zero generated tokens. Unobserved suffixes and incomplete or conflicting
references leave `completion_tokens_total` and `completion_tokens_unobserved`
as `null`; no missing work is estimated.
