# DCC experiment

## Model and held-out data

The model is `Qwen/Qwen3-30B-A3B-Instruct-2507`, revision
`0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`. Its nominal size is 30B, but
its actual total parameter count is approximately **30.53B**, with about 3.3B
activated per token. It exceeds a strict 30B total-parameter limit.

CUAD has 15,908 training questions from 388 contracts, 820 validation questions
from 20 contracts, and 4,182 official test questions from 102 contracts. Its
source is `The-Atticus-Project/cuad` revision
`67faa0e6023b04fcaae6cc09497ab00e5d63a2a2`. Official WideSearch is
`ByteDance-Seed/WideSearch` revision
`6531a7e5b497d44c8912407e0cb3dc95bd98cc09`: all 200 tasks, 100 English and
100 Chinese, remain held out. No official WideSearch question or reference
answer becomes a training target or checkpoint-selection example. The reference
WideSearch evaluator revision is `9825ba7b140b71d81b364793f86dabe4cfed6749`.

The independent web curriculum has 58 bilingual training tasks from three IANA
registries and eight validation tasks from two different registries. All 58
teachers execute real search/read/cite checks, yielding 174 assistant segments.
Grounded CUAD teachers use training questions only: 128 positive episodes and
48 negative episodes yield 528 search/read/final segments across all 41 clause
categories. Queries use question/profile terms; read offsets come from actual
search hits. Training gold selects valid teachers but never enters evaluation
queries or source selection.

## Runtime and training lineage

Only attention `q_proj`, `v_proj` and `o_proj` adapters are trainable: LoRA rank
8, alpha 16, dropout zero, 288 tensors and **5,701,632 trainable parameters**.
Base expert weights stay frozen. Gradients give each actual example equal
weight across workers before clipping and AdamW. Mixed SFT accumulates individual
forwards/backwards; its token budget does not allocate a padded batch at once.

| Phase | Optimizer steps | Workers | Actual data / observations |
| --- | ---: | ---: | --- |
| Initial adapter | 32 SFT + 24 RL | 8 | Earlier sampling/log-probability discrepancy, described below |
| Corrected pure-RL continuation | 8 | 8 | 256 trajectories, 248 completed, 8 turn-budget failures, 0 infrastructure failures |
| Mixed SFT round 1 | 24 | 1 | 93 examples from 78 task IDs |
| Mixed SFT round 2 | 64 | 8 | 2,048 examples from 1,305 task IDs |
| Mixed SFT round 3 | 32 | 8 | 1,024 examples from 742 task IDs |

The initial RL phase sampled at temperature 0.8, top-p 0.95 and top-k 20 while
scoring raw model log probabilities. Its continuation corrects this to
temperature 1, top-p 1 and top-k 0, disables inherited sampling penalties and
uses a fresh optimizer without SFT auxiliary. Both phases contribute to later
weights; the original discrepancy is not retroactively removed. Each rollout
batch gets one optimizer update, without policy epochs or rollout reuse.
Fifteen of 64 corrected groups have nonzero reward variance; five of eight
optimizer steps have nonzero global gradients. All 288 adapter tensors changed
and remain finite. See [dcc_training.md](dcc_training.md) for objective limits.

Round 1 warm-starts that adapter at learning rate 0.00001. Its actual examples
are 44 web segments, 26 CUAD tool segments, 21 direct negatives and two direct
positives; three dynamic CUAD candidates were skipped. All losses and gradient
norms are finite and nonzero, all adapter tensors changed, and training takes
255.6 seconds including setup. The 558 eligible teacher segments at this stage
are a dataset size, not a claim that all were sampled.

Round 2 warm-starts round 1 with a fresh optimizer, learning rate 0.000005,
eight workers, four examples per worker, seed 45, prompt budget 8,192 and
completion budget 2,048. It saves every 16 steps. Its actual 2,048 examples are
885 direct CUAD positives, 484 direct negatives, 502 web segments (167 search,
168 fetch, 167 final), and 177 CUAD tool segments (58 search, 59 read, 60 final).
Dynamic preparation skips 486 ungrounded or overlong candidates. All 512
per-worker step records have finite loss and nonzero finite gradient norms,
and synchronized norms agree. All 288 tensors change and remain finite in
every checkpoint. Training logs record 748.88 seconds.

Round 3 warm-starts round-2 step 32 with a fresh optimizer, learning rate
0.000002, eight initial chunks, negative draw fraction 0.10, web fraction 0.25,
CUAD tool fraction 0.15 and seed 46. It completes 32 steps with four examples
per worker, saving every eight steps. Actual examples are 657 direct CUAD
positives, 102 direct negatives, 223 web segments (75 search, 74 fetch, 74 final)
and 42 CUAD tool segments (14 search, 15 read, 13 final). The tool samples
include 40 positive segments and two negative read segments; negative tool
search/final teachers were available but were not sampled. Preparation skips
261 direct candidates and no tool records. All losses are finite, gradients
are finite and nonzero, and synchronized norms agree. All five saved adapters
have 288 finite changed tensors, correct optimizer steps and the intended
configuration; final equals step 32. Training logs record 1,093.8 seconds,
including substantial filesystem waits during checkpoint creation.

A full-expert NF4 option covers all 18,432 expert matrices. Actual loaded
allocation is 16.79 GiB; 16,384-token prefill plus 16 generated tokens peaks at
21.17 GiB. Real quantized adapter training/reload checks pass. Matched mixed
steps take about 86 seconds under this unfused runtime, versus 4.6–5.0 seconds
under fused BF16. BF16 fits, so experiments retain it; NF4 is available when
frozen base weights leave insufficient memory. Quantization changes predictions
and does not reduce the original logical parameter count.

## Validation and further optimization

Selection uses CUAD validation and the independent eight-task web validation,
not either official test set. The original 82-question sample and all 820
validation questions share the same 20 contracts; they are not independent
contract holdouts. Small-sample checkpoint rankings failed to generalize.

All current selection runs use graph vLLM 0.11.2, PyTorch 2.9.0,
Transformers 4.57.6, BF16 policy/LoRA/KV computation, TP1, greedy decoding and
seed 42. The actual manifests record memory utilization 0.88 and an
8,192-token prefill scheduling budget. HF and vLLM can produce different greedy
answers, so their scores are kept separate. Earlier matched single-worker HF
validation improved from F1 0.340426 without query expansion to 0.433333 with
it, and round-1 final reached 0.467290. Those HF scores are not substituted for
graph runtime results.

Current graph CUAD validation uses the training-only retrieval profile, eight
initial chunks, 8,192 prompt tokens, 1,024 generated tokens per turn, at most
three turns, and batch size eight.

| Complete CUAD validation, 820 questions | AUPR | Micro precision | Micro recall | Micro F1 | Engines |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen base | 0.230205 | 0.305451 | 0.523349 | 0.385757 | 1 |
| Early corrected-GRPO adapter | 0.273384 | 0.433036 | 0.468599 | 0.450116 | 1 |
| Early adapter, matched four-engine control | 0.277331 | 0.453435 | 0.478261 | 0.465517 | 4 |
| Round 2, step 32 | 0.225080 | 0.521028 | 0.359098 | 0.425167 | 4 |
| Round 2, step 48 | 0.207676 | 0.589577 | 0.291465 | 0.390086 | 4 |
| Selected round 3, step 8 | 0.251273 | 0.576623 | 0.357488 | 0.441352 | 2 |

All 820 questions complete with zero infrastructure failures in each run.
Engine counts change batch composition and can cause numerical differences,
as the matched early-adapter control demonstrates. Round-2 step 48
had the best 82-question graph score (F1 0.451613, AUPR 0.276915), but its full
validation recall regresses. More steps do not establish better performance.

Independent web validation keeps 12 turns, 16,384 prompt tokens, 8,192 generated
tokens per turn, and batch size four. The following are strict exact-match
proxies on eight independent tasks, not official WideSearch scores.

| Policy | Row F1, auto-read off | Row F1, auto-read on |
| --- | ---: | ---: |
| Frozen base | 0.848744 | 0.701289 |
| Early corrected-GRPO adapter | 0.458028 | 0.397727 |
| Round 2, step 32 | 0.831909 | 0.728861 |
| Round 2, step 48 | 0.771398 | 0.782102 |

All tasks remain in each denominator. Step 32 with auto-read on has seven
completed tasks and one evidence failure; the other settings complete all eight.
Actual searches use the public-web provider's Seznam fallback. Automatic page
reads are optional and do not consistently help this validation set.

All four round-3 checkpoints complete all 820 CUAD questions and all eight
independent web tasks with zero infrastructure failures. Each CUAD run uses two
logical engines; the round-2 comparison uses four, so batch composition differs.
Grounding skips raised round 2's actual direct-negative fraction to about 35%,
above its configured 25% draw probability. Separately saved LoRA-factor
interpolation candidates combine the early adapter and step 32 without changing
either source checkpoint; factor interpolation is not equivalent to a linear
interpolation of effective model functions. Factors 0.25 and 0.5 give independent
web row F1 0.650400 and 0.532705, respectively, below step 32; neither is promoted.

The final selection rule is fixed before official testing: require independent
web row F1 at least 0.78 with auto-read off and complete CUAD validation micro F1
at least 0.4251668255, then maximize CUAD AUPR among eligible trained policies,
breaking ties by micro F1. Round-2 step 32 is the existing eligible fallback.
This balances extraction and web behavior, rather than selecting on one task
alone. No additional training rounds or official-test-based tuning are planned.

| Candidate | CUAD AUPR | CUAD micro F1 | Web row F1 | Eligible |
| --- | ---: | ---: | ---: | --- |
| Round 2, step 32 | 0.225080 | 0.425167 | 0.831909 | Yes |
| **Round 3, step 8** | **0.251273** | **0.441352** | **0.848744** | **Selected** |
| Round 3, step 16 | 0.235500 | 0.418828 | 0.720798 | No |
| Round 3, step 24 | 0.233861 | 0.422680 | 0.860174 | No |
| Round 3, step 32 | 0.239564 | 0.433010 | 0.831909 | Yes |

Selection freezes at `2026-10-18T20:10:45Z`, before new official-test generation,
in external `dcc-run/final-selection.json`. The selected adapter is
`dcc-run/tool-sft-round3/step_0008`: it inherits round-2 step 32 and incorporates
the first eight round-3 steps, or 256 global examples. The 1,024-example round-3
diagnostic describes the complete candidate-training run. Step 24's stronger
web score does not relax the predeclared CUAD floor. Step 8 has the highest
AUPR among eligible candidates and is used unchanged for both final benchmarks.
The selected weight lineage is initial adapter → corrected RL continuation →
round-1 final → round-2 step 32 → round-3 step 8. Later candidate steps are not
part of these weights; totals for entire training runs must not be interpreted
as the number of updates inherited by the selected adapter.

## Harness and test protocol

Training-only query expansion fits category terms from CUAD train answers and
combines original/expanded searches with reciprocal-rank fusion. Evaluation
queries match the public question and never read gold labels or category
metadata. On complete validation, requiring a full unnormalized gold string
inside one actual chunk gives visibility of 407/621 spans with four initial
chunks and 481/621 with eight. The longest eight-chunk encoded prompt has
4,008 tokens. This is source visibility, not answer quality.

WideSearch requires successful search, a nonempty page read and citations to
observed URLs before accepting a final answer. Unsupported finals receive
feedback within the same turn budget; exhausted guards are model failures
scored zero. Whole terminal NDJSON answer payloads are accepted only when their
schema and citations pass the same checks. Page text is sanitized at the web
cache boundary for tokenizer compatibility. Optional automatic reads fetch
actual search URLs, retain source events and have a cumulative attempt budget.

A CPU replay of six selected WideSearch evidence failures and one successful
control exactly reproduces the frozen parser and guard decisions. Observed
causes include searching without reading, missing row citations, truncated
final JSON, invalid final formats and external HTTP retrieval failures. This
is a small diagnostic sample, not an estimate of the final distribution. The
saved HTTP failures occur before PDF extraction and retain only the exception
class, so their HTTP status codes cannot be recovered. Zero task-level
`infrastructure_error` does not mean all external pages were accessible, and
these evidence failures cannot all be attributed to model behavior alone.

CUAD official test scoring is complete; full WideSearch generation and judging
are **in progress**. Both final runs use source snapshot `b25a4d63`, graph
vLLM/BF16 and eight independent TP1 engines. CUAD uses the training-only profile, eight initial chunks,
8,192/1,024 prompt/completion budgets, three turns and batch size eight.
WideSearch uses 16,384/8,192 budgets, 12 turns, batch size four, automatic page
reads off and the same evidence guard. Each engine has memory utilization 0.88,
8,192 scheduled prefill tokens and seed 42. No test result changes selection.

The frozen adapter scores **AUPR 0.228265** and **micro F1 0.422087** on all
4,182 official CUAD questions from 102 contracts. Micro precision is 0.553035
and recall is 0.341279; precision at 80% and 90% recall is zero. Coverage is
100%: 4,179 tasks complete and three exhaust the turn budget, with zero
infrastructure failures. All questions, including budget failures, remain in
the denominator. Span confidence comes from model output and is not separately
calibrated. The eight-engine run takes 390.3 seconds from first launch to final
generation exit, including model loading and compilation, consuming 11,878,691
prompt tokens and 168,150 completion tokens. This CUAD run makes zero tool
calls: it answers from the eight retrieved initial chunks. The harness supports
additional retrieval tools, but these scores do not demonstrate tool use.

The earlier adapter's original CUAD test run is retained as historical context:
4,182/4,182 questions, AUPR 0.182906, micro F1 0.370070, precision 0.457637,
recall 0.310632, 4,180 completed, two exhausted budgets and zero infrastructure
failures. It used HF, four initial chunks and no training-derived retrieval
profile. These are not scores for the new candidates. The old HF WideSearch
run was explicitly aborted after 104 durable predictions and is not scored;
its partial files are never merged into a new policy/backend run.

CUAD uses the official word-set Jaccard threshold of 0.5 and Parties substring
exception; correctly empty answers do not create official true positives.
Reports include AUPR, precision at 80%/90% recall and additional micro F1.
WideSearch will report a strict proxy and the official metric pipeline with a
**custom frozen base-Qwen judge**, not the published GPT-4.1 leaderboard judge.
All 200 tasks remain in the denominator. Infrastructure and judge failures are
explicit, and judge references are never silently truncated. The same frozen
final adapter is used for both benchmarks.

An independent, synthetic three-case semantic check found that the original
alignment prompt allowed the local judge to map two unrelated institutions.
The raw judge response and a CPU replay confirmed a model judgment error, not
a parser-generated mapping. Formal custom judging was held before it started.
Scoring revision `same_entity_v2` distinguishes schema labels from entity names,
requires the same specific entity and explicitly abstains on unsupported
identity. It also rejects duplicate aligned headers, stably keeps the first
row when aliases or normalization collapse a unique key, and combines multiple
criteria for one cell before counting item accuracy. These guards prevent
duplicate matches or multiple criteria from inflating a metric above one.
The adapted scorer is versioned separately from the unchanged `b25a4d63`
generation snapshot. Its strict proxy must also be recomputed from the saved
predictions; the original scoring files are retained separately. These are
scoring corrections based on artificial examples, not policy selection or
tuning against official test answers.

## Checks, reproduction and artifacts

The `45a0857f` checkpoint passed 494 CPU regression checks with two optional
checks skipped. Those checks were subsequently covered by three NF4 checks
and one real-Qwen tokenizer check in their appropriate environments. BF16/NF4
preflights pass. Sharding, scoring audit, resume and slow-mount fixes at
`513f41f8` pass 111 targeted checks, Ruff and diff checks. Logical-shard complete
820-question runs also exercise the real GPU path.

Scoring revision `same_entity_v2` passes 98 targeted CPU checks across metrics,
evaluation and sharded-judge provenance, plus fatal-error Ruff rules and diff
checks. Independent review covers header and entity collisions, per-cell
multiple criteria, exact-identity anchors, conflicting explicit versions and
missing or null legacy versions. No official test item is used for these
regressions.

Judge provenance changes at `360971f5` pass 90 targeted checks and reject
mixed judge models, budgets, runtime versions, adapters or generation records.
A real frozen-base BF16 judge smoke completes one independent web task with
all seven reference rows intact, batch size one, context 65,536 and generation
budget 4,096, without judge failures. CPU planning checks all 200 official
WideSearch references using the real tokenizer: 200 header, 285 entity-alignment
and 1,168 cell prompts. The longest reference prompt has 9,384 tokens; adding
8,192 prediction tokens and 4,096 completion tokens totals 21,672, below the
65,536 limit. A synthetic-only BF16 pressure run also completes the full planned worst-case
budget on one GPU: 17,576 input tokens plus exactly 4,096 generated tokens,
batch size one, adapter disabled and truncation disabled. All 4,096 output-logit
checks are finite; peak allocated/reserved memory is 61.0/61.7 GiB. Generation
takes 267.0 seconds, or 287.0 seconds including setup, without OOM. The input
contains no official task or reference. These are runtime/capacity checks, not
benchmark scores.

The completed official CUAD run passes 30 additional CPU audit checks: all
4,182 unique test IDs appear exactly once on the expected shards, frozen
selection/runtime provenance agrees, and independently recomputed aggregate
and per-task metrics match the final report. There are 902 true-positive,
729 false-positive and 1,741 false-negative answer spans. The three exhausted
budgets remain in the evaluation population with empty answers.

See [dcc.md](dcc.md), [dcc_training.md](dcc_training.md),
[dcc_tool_training.md](dcc_tool_training.md), [dcc_vllm.md](dcc_vllm.md) and
[dcc_runtime_setup.md](dcc_runtime_setup.md) for portable installation, data,
training, generation, frozen judging, merge and resume commands. Model weights,
credentials, machine paths, generated data, predictions, checkpoints and logs
stay outside Git in the configured artifact root.

Artifact paths relative to that root include `dcc-run/training/final`,
`dcc-run/training-raw/final`, `dcc-run/tool-sft-round1/final` and
`dcc-run/tool-sft-round2/step_0032`. Training diagnostics are in the corresponding
`tool-sft-round*-diagnostics.json`; validation tables are retained in
`round2-graph-dev82-summary.json`, `single-validation-graph-summary.json` and
`full-validation-graph-summary.json` under `dcc-run/`. The frozen selection is
`dcc-run/final-selection.json`; final predictions and scores use the new
`dcc-run/cuad-final-selected` and `dcc-run/widesearch-final-selected` directories.
CUAD scores are in `dcc-run/cuad-final-selected/scores_strict.json`. WideSearch
scores will be recorded after full generation and judging finish.

Final CUAD coverage and scoring audits are retained under
`dcc-run/scoring-audit/final-cuad-score-audit.json` and
`dcc-run/scoring-audit/final-selection-runtime-audit.json`.

The selected WideSearch trajectory replay is retained in
`dcc-run/scoring-audit/widesearch-evidence-sample-audit.json`.

The real synthetic BF16 judge budget check is retained in
`dcc-run/judge-long-context-stress-result.json`.
