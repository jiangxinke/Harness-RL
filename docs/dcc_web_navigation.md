# Optional Web navigation ledger

`dcc_model_evaluate.py --web-navigation-ledger` adds navigation advice and a
bounded summary before each Web evaluation policy call. It requires
`--inference-backend vllm --web-structured-actions --web-structured-final` and
can be combined with `--web-explicit-review`. The default is disabled. CUAD and
training episode prompts keep their existing behavior.

The summary derives candidate URLs and character offsets only from actual
`results`, `outgoing_links`, `worksheets`, readable `archive_members`, and
`next_start` tool metadata. It records successful source text windows, observed
unread targets, repeated exact windows, and recent errors. A source preview,
archive directory entry, observed link, or failed read never counts as answer
evidence. The existing evidence guard, schemas, and scorer remain unchanged.

A `fetch_url` start or `next_start` is a character offset in the extracted text
of that same URL. It is not an API row offset or page parameter. The guidance
prefers observed public structured exports and their documentation; it allows
legitimate constructed API URLs and necessary rereads. It asks the policy to
preserve literal cells and verify every original public condition, exact column,
unique key, and missing-field rule against actually read evidence. It does not
supply endpoints, entities, target rows, unavailable data, or query execution.

The ledger retains at most 48 candidates, 32 read windows, and 12 errors. Each
snapshot, including its prefix, is at most 6000 characters and remains complete
JSON. URLs longer than 2048 characters are omitted from this optional summary;
this does not restrict the tools or evidence. Candidate labels are limited to
120 characters, errors to 160. Evicted-record and display-omission counts make
the bounds explicit. Summary absence never proves a source is unread or
unavailable. Each request gets only the latest snapshot; older summaries do not
accumulate in its message history. Exact per-turn snapshots are persisted in
`metrics.web_navigation_ledger` for audit.

Enabled runs record the full version, scope, candidate-source declaration, and
limits in both generation manifests and inference runtime metadata. Paired run
protocols must put that same `web_navigation_ledger` object in
`common.generation` for both runs. Scoring and paired comparisons reject missing,
unsupported, or inconsistent declarations and changed structured/backend
prerequisites. Switching the mode requires a new output directory. Legacy runs
without the optional key retain their disabled configuration.

CPU verification uses synthetic public sources and the real EpisodeRunner,
evaluator, scoring-provenance, vLLM-loader doubles, and paired-protocol validator:

```bash
PYTHONPATH=slime python -m pytest -q tests/test_dcc_web_navigation.py
```

The checks establish integration and provenance contracts. They do not establish
a benchmark score improvement or base-model readiness; that requires a fresh,
matched evaluation retaining every registered task and outcome.
