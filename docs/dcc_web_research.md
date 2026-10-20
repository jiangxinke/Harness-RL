# Optional Web research checkpoints

`--web-research-checkpoints` is an evaluation option, disabled by default. It
requires vLLM, `--web-structured-final`, and `--web-structured-actions`. It applies
only to WideSearch episodes evaluated by `EpisodeRunner`; CUAD and training
requests are unchanged. It is independent of explicit review, navigation-ledger,
and bounded-whitespace options and can be combined with them.

```bash
python scripts/dcc_model_evaluate.py \
  --model "$MODEL_DIR" --data "$WEB_VALIDATION_JSONL" \
  --output-dir "$EVAL_OUTPUT_DIR" --inference-backend vllm \
  --web-structured-final --web-structured-actions --web-research-checkpoints
```

Before each existing policy request, the runner inserts one latest controller
message containing only the original public required answer column labels,
remaining model/research turns, and generic guidance. It asks the policy to check
whether observed source fields can support the task, verify returned records
against the original conditions instead of trusting query parameters, and
organize actually supported rows before paging further. When at most three
research turns remain, it emphasizes stopping unrelated navigation and checking
missing required fields, unique keys, and all original filters. It never supplies
answer cells, source evidence, or verified candidate claims.

Turn indices are zero based. For a twelve-turn run, turn 0 reports twelve model
turns, eleven research turns including the current request, and one reserved final
model turn. Turn 8 reports four model turns and three research turns. Turn 11
reports one model turn and zero research turns. Early final answers remain
allowed. These messages do not increase the turn, tool-call or token budgets or
add an extra review model; existing limits and timeouts still apply. They consume
the existing prompt budget and can displace older excerpts under the unchanged
prompt-truncation rules. The original final
controller message stays last, after any navigation ledger and research message.

Only the latest checkpoint is included in a policy request. Checkpoints do not
accumulate in persistent episode `messages`; their exact content is retained in
`metrics.web_research_checkpoints` as `{turn, content}` records. A policy request
that fails before yielding a completion can leave one more checkpoint than
completed `steps`, while retaining the original infrastructure-error status.

The controller does not inspect task labels, references, or benchmark answers.
Source previews and headers remain navigation material unless actually read.
Answers still use the unchanged structured schema, source-read citations,
missing-field rules, and evidence guard. Empty output remains an abstention;
the option never forces a nonempty answer or manufactures evidence.

Enabled runs declare the full `web_research_checkpoints` configuration in their
generation manifest and inference runtime. Score-only and paired-run validation
require identical supported versions, configuration, structured modes and vLLM
backend. A paired protocol must place the same full configuration in
`common.generation.web_research_checkpoints`. Both policies must share the same
mode. Missing legacy configuration means disabled; malformed declarations or
mixed enabled/disabled records are rejected. Changed configuration requires a
new output directory, preserving resume provenance. The version is
`public-task-web-research-checkpoints-v1`; changes to guidance, placement, phase
threshold, budget semantics or public inputs require a version/configuration
update and a new matched experiment.

This option is a proposed harness intervention, not an established improvement.
Use matched development runs with unchanged denominators, model/adapter,
sampling, tool/turn/token budgets, evidence rules and scoring. Keep all registered
candidates and failures; do not use partial results to select a configuration or
copy evaluation questions/answers into training data.
