# Optional Web navigation ledger

`dcc_model_evaluate.py --web-navigation-ledger` adds navigation advice and a
bounded summary before each Web evaluation policy call. It requires
`--inference-backend vllm --web-structured-actions --web-structured-final` and
can be combined with `--web-explicit-review`. The default is disabled. CUAD and
training episode prompts keep their existing behavior.

The summary derives candidate URLs and character offsets only from actual
`results`, `outgoing_links`, `worksheets`, readable `archive_members`, and
`next_start` tool metadata, plus successful browser `browser_read.observed_json_urls` navigation metadata. It records successful source text windows, observed
unread targets, repeated exact windows, and recent errors. Failed fetch targets are marked
`failed` and sort after untried targets, even when a failed target is a structured
export. Typed HTTP status and retryability are retained; raw exception text is
not copied because it may echo credentials or response contents. Failures do not
restrict tool actions or create a permanent ban: a later actual successful fetch
clears failure state, records its real read span, and exposes any next window. A source preview,
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
this does not restrict the tools or evidence. The summary omits malformed URLs,
non-HTTP(S) schemes, host credentials, and credential-bearing query parameters.
Fragments are removed consistently with `fetch_url`, so fragment aliases share
one target; distinct query strings remain distinct. Candidate labels are limited to
120 characters, errors to 160. Evicted-record and display-omission counts make
the bounds explicit. Summary absence never proves a source is unread or
unavailable. Each request gets only the latest snapshot; older summaries do not
accumulate in its message history. Exact per-turn snapshots are persisted in
`metrics.web_navigation_ledger` for audit.

Version `observed-web-frontier-ledger-v2` also admits only JSON navigation records
from the current visible-DOM browser reader with integer status 200 and
`navigation_only=true`. Those URLs remain unread and never add API contents,
source text, or citation evidence. They share the existing URL, candidate and
snapshot limits. No alternative endpoint or query is synthesized. Documented API
construction and necessary rereads remain allowed by the tool schema.

Enabled runs record the full version, scope, candidate-source declaration, and
limits in both generation manifests and inference runtime metadata. Paired run
protocols must put that same `web_navigation_ledger` object in
`common.generation` for both runs. Scoring and paired comparisons reject missing,
unsupported, or inconsistent declarations and changed structured/backend
prerequisites. Switching the mode requires a new output directory. Legacy runs
without the optional key retain their disabled configuration. The disabled
configuration is unchanged. Enabled v1 and v2 snapshots/provenance must not be
mixed: retain original frozen source for old runs and use a new source receipt
and matching v2 configuration for every arm of a new native evaluation. Native
model, thinking protocol and token budgets do not change.

CPU verification uses synthetic public sources and the real EpisodeRunner,
evaluator, scoring-provenance, vLLM-loader doubles, and paired-protocol validator:

```bash
PYTHONPATH=slime python -m pytest -q tests/test_dcc_web_navigation.py
```

The checks establish integration and provenance contracts. They do not establish
a benchmark score improvement or base-model readiness; that requires a fresh,
matched evaluation retaining every registered task and outcome.
