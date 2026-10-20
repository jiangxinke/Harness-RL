# DCC: contract and web research

The DCC harness supports CUAD clause extraction and the
[ByteDance-Seed WideSearch benchmark](https://github.com/ByteDance-Seed/WideSearch).
It provides bounded document retrieval, HTTP web search and reading, structured
answers, per-call token trajectories, and evaluation separate from rollout.

The optional [Web research checkpoints](dcc_web_research.md) add public-column
and remaining-turn guidance to structured Web evaluation. They are disabled by
default and preserve the existing evidence rules and generation budgets.

For native Thinking evaluation, [the native reference harness](dcc_native_base_harness.md)
records the selected `cuad_native16k` and `web_steps20_32k` presets. Web reference
selection used 34 independently authored development tasks; the 200 released
WideSearch tasks form a separate TEST set.

## Data boundaries

Task JSONL records contain `id`, `benchmark` (`cuad` or `widesearch`), `split`,
`prompt`, `document`, `gold`, and optional `metadata`. CUAD source contracts can
instead live in files referenced by `metadata.document_file`, relative to the
JSONL file. `load_tasks` hydrates and caches those files. `validate_disjoint`
rejects task-ID overlap and contracts reused across training and evaluation.
Packed contract files additionally use `metadata.document_key`, avoiding one
filesystem operation per contract on remote artifact storage.

CUAD uses its official contract-level test split. Training and validation must
come exclusively from the remaining contracts. WideSearch has 200 released
tasks and no official training split; all 200 remain held out. Any web training
curriculum must be authored independently and identified as such in metadata.
Agent messages never contain `gold` or arbitrary metadata.

Install the small scoring dependencies and prepare the pinned public datasets:

```bash
python -m pip install -r configs/dcc/requirements.txt
python scripts/dcc_prepare_data.py --output-dir "$DATA_ROOT"
```

Set `DATA_ROOT` outside the checkout. To reuse previously downloaded raw data,
also pass `--source-dir "$RAW_DATA_ROOT"`. The generated manifests record source
revisions and split counts. The default CUAD validation split contains 20
contracts, selected from its 408 training contracts.

## Episode and training interface

With the repository's `slime` directory on `PYTHONPATH`:

```python
from slime_plugins.dcc import EpisodeRunner, PolicyCompletion, load_tasks

# async policy(messages, *, max_new_tokens, temperature) must return:
# PolicyCompletion(text, exact_prompt_token_ids, generated_token_ids, logprobs)
runner = EpisodeRunner(max_steps=4, max_new_tokens=1024)
result = await runner.run(load_tasks("data/cuad/train.jsonl")[0], policy)
segments, reward = result.trainable_steps, result.reward
```

Every model call is an independent prompt/completion pair. The policy loss must
cover only that pair's generated tokens. Tool observations and chat-template
tokens belong to the prompt and receive no loss. Do not concatenate the pairs
into a flattened episode. Infrastructure failures have no trainable segments;
policy formatting failures remain valid zero-reward training examples.

CUAD starts with BM25-ranked contract excerpts and offers `search_document` and
`read_document` tools with exact character offsets. Final answers are lists of
`{"text": "verbatim span", "confidence": 0.9}`; an empty list means no clause.
Training reward combines CUAD's official span-match F1 with verbatim grounding.
The no-answer reward does not create an official true positive. Balance positive
and negative questions when sampling training batches.

WideSearch uses `web_search` and `fetch_url` and returns table-row objects with
the requested column names and a `citations` URL list. Citation validity means
the URLs were observed by the agent; it does not establish that a source entails
every cell. Held-out answers are consulted only after predictions are saved.
The web reader extracts HTML article bodies and tables, supports PDF text,
and returns `next_start` to page through long documents without redownloading.

WideSearch accepts a final answer only after an executed `web_search` returns
source URLs and an executed `fetch_url` returns nonempty source text. Each row
must have a nonempty `citations` list containing only source URLs actually read
with nonempty text. Empty answer lists are rejected; successful search and read
steps alone do not establish a completed table.
This checks observed provenance; semantic support still needs source review.
An unsupported final receives explicit correction with tool JSON templates and
uses the remaining model turns. Corrections are recorded separately in
`metrics.evidence_guard.rejections`, never as tool calls. Exhausting the turn
budget without an accepted final produces `evidence_failure`: a model failure
with zero reward and zero evaluation score that remains in the full task
denominator. Search-service outages remain `infrastructure_error` and exclude
the episode's policy segments from training. Empty results and unreadable
individual pages are normal tool observations, and do not satisfy the evidence
requirements. The private WideSearch evaluation manifest includes the guard
version, preventing resume from mixing predictions across guard versions.
CUAD retains its existing extraction protocol.

## Web providers

Set `DCC_SEARCH_BACKEND` to `public_web` (default), `duckduckgo_wikipedia`,
`wikipedia`, `duckduckgo_lite`, `seznam`, `baidu_mobile`, `serper`, `brave`,
`searxng`, or `bing_rss`. The default tries DuckDuckGo, then Seznam for
non-Chinese queries, then Baidu's mobile web index, then English/Chinese
Wikipedia's official search API. Seznam and Baidu return direct source URLs;
individual source sites may still reject retrieval. Live validation covered
multiple English and Chinese topics, actual source-page retrieval, and eight
simultaneous queries. Provider availability and ranking can change.

Public providers serialize and limit requests, cache query results in the
process, and start a process-local cooldown of 300 seconds for challenge or
CAPTCHA errors and 10 seconds for other errors. Challenges are detected and
skipped. The compatibility setting
`duckduckgo_wikipedia` (also `auto`) retains the earlier DuckDuckGo-to-Wikipedia
chain. Every observation records the configured and actual `backend`,
`search_scope`, and `fallback_reason`. Wikipedia covers encyclopedia articles and does not provide
the coverage of a general web index. Its reader retrieves article HTML through
the official parse API and preserves tables. MediaWiki's plaintext extracts
omit tables and would discard many enumeration answers.

API providers require `SERPER_API_KEY` or
`BRAVE_SEARCH_API_KEY`; SearXNG requires `DCC_SEARCH_URL`. Requests use
`DCC_WEB_TIMEOUT` seconds (default 30). Configure these outside version control.
Standalone `duckduckgo_lite` treats bot challenges as infrastructure failures.
If every provider in `public_web` fails, the episode is invalidated instead of
creating a misleading training reward. A valid empty response remains a normal
observation even if later providers fail. A reliable search API is
recommended for broad coverage in the full WideSearch run. Record provider
coverage alongside scores. Validate query relevance before using Bing RSS.
WideSearch reports include `search_coverage`: actual and configured backend
counts, scope counts, fallback categories, the fraction of recorded search
calls using the encyclopedia, and the fraction of all evaluation tasks with
encyclopedia search. Tasks without recorded search remain visible. Raw fallback
errors and operator-configured URLs are never copied into these statistics.

## Scoring

CUAD reports the official word-set Jaccard matching protocol, including the
Parties substring rule, AUPR, precision at 80% and 90% recall, and additional
operating-point micro-F1. AUPR uses the model's generated span confidences;
these are not calibrated classifier probabilities.

WideSearch scoring must disclose its protocol and judge model. Its official
pipeline performs semantic table normalization/alignment before cell scoring;
using a different judge is a custom-judge evaluation, and disabling the judge
produces a strict matching proxy. Never label that proxy as the official score.
Report the task denominator, failed episodes, and coverage alongside scores.

Score previously saved `EpisodeResult.to_dict()` JSONL records with:

```bash
python scripts/dcc_evaluate.py \
  --tasks "$DATA_ROOT/cuad/test.jsonl" \
  --predictions "$RUN_ROOT/cuad_predictions.jsonl" \
  --report "$RUN_ROOT/cuad_report.json"
```

The same command accepts WideSearch tasks. Add `--judge-base-url` and
`--judge-model` to use the semantic evaluation protocol. Without those options,
WideSearch results are labeled `strict_exact_proxy`. Missing predictions remain
in the score denominator; judge failures make the report incomplete and are
reported separately. Add `--base-url` and `--model` to generate missing episodes
using an OpenAI-compatible endpoint. Generation resumes from existing task IDs.
Use the local Hugging Face policy for training so exact token IDs are retained.

Store generated data, trajectories, logs, checkpoints, and reports in the run's
externally configured artifact directory. Runtime paths, credentials, hostnames,
and hardware identifiers must stay out of committed configuration and reports.

Optional downloaded-source row filtering is documented in
[dcc_web_cached_rows.md](dcc_web_cached_rows.md).
