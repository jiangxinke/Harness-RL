# Optional source reading after web search

`EpisodeRunner(auto_read_sources=1)` lets a WideSearch agent read actual source
pages during the same model turn as its search. The default is `0` (disabled);
supported values are `0`, `1`, and `2`. CUAD behavior is unchanged.
The local evaluation and RL entry points expose the same switch through
`--auto-read-sources 1`. Evaluation manifests reject resuming predictions with
a changed setting; use separate directories for comparisons.

After a successful nonterminal `web_search`, the runner visits returned HTTP(S)
URLs in result order and calls the real `fetch_url(url, start=0)` for at most the
configured number of new sources. Every episode allows at most four automatic
attempts, including failed or empty reads. Previously attempted automatic URLs
and pages already read successfully by the agent are skipped. The agent can
still fetch any source itself or use `next_start` to continue a truncated page.

Search events retain their original observations. Each automatic read has a
separate `fetch_url` event with `automatic=true`, its actual arguments and its
actual observation. The model receives those page observations in the search
response's `auto_read_sources` field, together with continuation guidance and
the remaining attempt budget. No labels or benchmark reference answers select
URLs. Returned errors stay visible and cannot satisfy the successful-fetch
evidence check. A valid final still needs search, nonempty fetched text, and
observed citations; failed episodes remain in the evaluation denominator.

The online GRPO and HF evaluation entry points also accept
`--auto-read-follow-docs`, disabled by default. With `--auto-read-sources 2`, a
successfully read page can contribute one observed, same-origin Help, API, or
documentation link ahead of the next search result. The link is read through
the normal fetcher and does not recursively trigger another follow-up. This
uses the existing successful-page and four-attempt budgets; failed follow-ups
also consume an attempt. These events use
`trigger=observed_documentation_link` and record `parent_url`. Enable the same
setting for base and adapter comparisons. This is a navigation candidate;
its availability does not establish a benchmark improvement.

The option adds page text to the prompt and can increase context use and HTTP
latency. Start with one source and check a separate development set before
enabling it for an evaluation. Automatic reads do not consume model turns, but
they count as tool calls. No tools run on the terminal model turn.

CPU transport tests exercise the real search parser and page reader:

```bash
python -m pytest -q tests/test_dcc_auto_read.py
```
