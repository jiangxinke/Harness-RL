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

Round 3 is complete and its four checkpoints are under complete validation.
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

Full official scoring of the final optimized policy is **pending selection**.
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
final adapter will be used for both benchmarks.

## Checks, reproduction and artifacts

The `45a0857f` checkpoint passed 494 CPU regression checks with two optional
checks skipped. Those checks were subsequently covered by three NF4 checks
and one real-Qwen tokenizer check in their appropriate environments. BF16/NF4
preflights pass. Sharding, scoring audit, resume and slow-mount fixes at
`513f41f8` pass 111 targeted checks, Ruff and diff checks. Logical-shard complete
820-question runs also exercise the real GPU path.

Judge provenance changes at `360971f5` pass 90 targeted checks and reject
mixed judge models, budgets, runtime versions, adapters or generation records.
A real frozen-base BF16 judge smoke completes one independent web task with
all seven reference rows intact, batch size one, context 65,536 and generation
budget 4,096, without judge failures. CPU planning checks all 200 official
WideSearch references using the real tokenizer: 200 header, 285 entity-alignment
and 1,168 cell prompts. The longest reference prompt has 9,384 tokens; adding
8,192 prediction tokens and 4,096 completion tokens totals 21,672, below the
65,536 limit. These are runtime/capacity checks, not benchmark scores.

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
`full-validation-graph-summary.json` under `dcc-run/`. Final selection, test
artifact locations and complete scores will be recorded after the remaining
validation and full evaluations finish.
