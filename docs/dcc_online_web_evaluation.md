# Matched native HF Web evaluation

`scripts/dcc_online_web_evaluate.py` evaluates prepared Web task JSONL with the
same `dcc_online_grpo.make_runner` used by online training. It is separate from
the existing CUAD evaluator and vLLM benchmark protocol. It does not change
public Web schemas, citation guards, review instructions or tool behavior.

The score is a **strict row/item F1 proxy** using the existing table evaluator
without a semantic judge. A column configured with `llm_judge` uses exact cell
matching here. This is not GPT-4.1 semantic scoring, even when the input is the
prepared original WideSearch task set. The `widesearch` backend field alone
does not establish dataset provenance: reports retain task count, splits,
original source/revision declarations and a public-input identity. Independent
Web DEV34 must be described as independent Web DEV34. Original prepared
WideSearch200 can be evaluated without rewriting tasks, but prior exposure
must not be relabeled as an untouched holdout.

Use validation data for harness decisions. Use the prepared original TEST only
for a final base/adapter comparison. This command performs no training,
automatic hyperparameter search, task selection or test-based checkpoint
selection, and rejects TRAIN task inputs.

## Generate, close, then score

Use the same committed source, base model, runtime packages, precision,
sampling profile, seed and harness arguments for both runs. The example uses
operator-supplied paths; put logs/checkpoints in the designated artifact store.

```bash
export DCC_SEARCH_BACKEND=public_web DCC_WEB_TIMEOUT=15

python scripts/dcc_online_web_evaluate.py \
  --model "$MODEL" --data "$WEB_DEV_DATA" --split validation \
  --source-commit "$SOURCE_COMMIT" --output-dir "$BASE_OUTPUT" \
  --quantization none --compute-profile bf16-autocast-v1 \
  --profile thinking --max-steps 20 --thought-tokens 4096 \
  --json-tokens 4096 --max-prompt-tokens 16384 --seed 47 \
  --generate-only

python scripts/dcc_online_web_evaluate.py \
  --model "$MODEL" --data "$WEB_DEV_DATA" --split validation \
  --source-commit "$SOURCE_COMMIT" --output-dir "$ADAPTER_OUTPUT" \
  --adapter "$ADAPTER" --base-output-dir "$BASE_OUTPUT" \
  --quantization none --compute-profile bf16-autocast-v1 \
  --profile thinking --max-steps 20 --thought-tokens 4096 \
  --json-tokens 4096 --max-prompt-tokens 16384 --seed 47 \
  --generate-only
```

Repeat the respective complete command with `--score-only` replacing
`--generate-only`, first for base and then for adapter. Scoring does not load a
model or tokenizer. Base and adapter generation can run independently before
the base report exists. To shard across GPUs, pass the same positive `--shards`
and a different `--shard-index` per process; `--device-index` selects that
process's visible device. Base and adapter may use different shard counts.
Per-task seeds are independent of sharding. All other comparison settings
must match. `--profile raw` matches training's raw thought sampling; `thinking`
and `greedy` are also supported when identical across the comparison.

The evaluator uses the existing complete Web runner configuration: structured
actions/final, explicit review, navigation ledger, research checkpoints, data
discovery guidance, multiple citations and cached rows. It records every
versioned guard and all turn/token/time/tool settings. Web search backend,
reader timeout, actual tool/search/query implementation identities, default and
maximum search result counts, query/fetch/extraction budgets, language headers,
provider routing, cache/cooldown settings, and digests of any configured search
endpoint/effective urllib proxies/TLS environment enter the matching
configuration. The package manifest also records installed `pypdf`,
`trafilatura` and `lxml` versions, or `null` when a distribution is absent;
these entries must match across generation shards and both comparison arms.
Optional packages may be absent from both arms; a matching manifest does not
establish that a package imports or extracts a particular page successfully.
Credentials and API keys are never recorded. There are no
additional locale or search-budget environment knobs in this implementation;
the fixed limits are tied to the recorded implementation source. The example
explicitly selects public_web with a 15-second reader timeout for both arms
and scoring. The underlying tools otherwise default to 30 seconds; an old
15-second run and a new default-timeout run do not match. Preserve these
environment settings during score-only replay as well. Live pages, DNS/network
conditions and process-local cache/cooldown state can still change between
runs; matching configuration does not freeze the web.

## What a completed run establishes

Generation removes gold and private metadata before constructing policy tasks;
only the public question and required output columns reach the runner. The
original split is preserved. Scoring rules and reference rows are reloaded in
the separate scoring pass, after local generation returns. Generation itself
rejects reference metrics or nonzero reward.

Each shard writes predictions, native traces and request ownership journals.
Timeout-late native outputs are drained and retained with cancelled delivery.
Every delivered step and every native trace UUID must match its request,
original public schema, prompt IDs, actual thought/JSON IDs and status. Before
publishing the done marker, the actual tokenizer verifies prompt encoding and
JSON decoding. All three durable writers and the native worker are closed.
Score-only rechecks all persisted cross-links and every configured shard;
missing/extra shards, duplicate traces, foreign outputs, changed task seeds or
changed configuration fail closed. Generation never resumes or retries into
an existing shard.

All noncompleted outcomes, including protocol, evidence, context and policy
timeout failures, receive zero and remain in the full task denominator. The
report retains failure statuses and the per-task proxy metrics. An adapter
comparison requires an actual closed and scored base run with the same public
tasks, complete evaluation references/rules, runtime, model configuration and
base storage precision. The saved adapter content identity is checked again
at scoring; its small weights are hashed once per process with the existing
identity cache. Base model weight files are not scanned or repeatedly hashed.
The loader's model revision/config and tokenizer provenance are recorded;
operators must keep the selected base checkpoint immutable.

`matched_base` reports paired mean deltas without automatically claiming an
improvement. A CPU test, runtime smoke or one favorable DEV comparison does not
establish an official benchmark improvement or statistical confidence.
