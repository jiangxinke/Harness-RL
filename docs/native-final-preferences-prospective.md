Prospective native final-decision preferences
============================================

This is an independent, opt-in TRAIN experiment. It does not alter existing SFT
acceptance rules, rollout stopping rules, held-out evaluators, or default loss.
The code has CPU mock coverage; that is not model-quality evidence or a completed
GPU pilot. A caller must separately approve the actual source and artifact.

`dcc_collect_native_preferences.py export` accepts the already-closed **raw**
native collection directory and its independently approved source profile and
receipt. It reuses `dcc_collect_native_train.read_closed` to require every shard
closure and exact episode/request/response/native-trace joins. It does not use a
selected-SFT file. Public tasks must be TRAIN, have no gold, and contain only
allowed public metadata. It neither reads label files nor calls a reward scorer.

Eligibility requires a physically complete original native final decision,
including its original thought closing token, JSON EOS, exact token IDs, approved
budget/sampling/tokenizer identity, successful response ownership, and exact
original request schema reconstructed from public runner state. The original
answer need not be correct, accepted by the evidence guard, grounded for SFT, or
in a completed episode. The current answer is inspected only to recognize a final
JSON object and validate its original physical/schema integrity. No original
completion, thought, gold, category, polarity, reward, or SFT acceptance decision
enters the new generation input. Earlier final actions and any rejected-final
feedback in the prompt are rejected. Thus a poor first final may supply a prefix;
a prompt after its rejection cannot.

The exporter replays exact public anchors, tool/observation pairs, remaining-turn
counters, and terminal controls. CUAD uses the canonical fixed final schema. Web
citations derive from actual successful fetch observations; review/navigation
histories must pass the exact existing public replay. Candidate-reported URLs or
candidate-selected schemas cannot establish provenance. Source observations are
trusted closed logs, not cryptographic proof against an untrusted log producer.

`original-final-only` is the default: original action schemas that permit tools
are rejected. `final-decision-only-v1` is an explicitly named intervention. At the
same unchanged public history it replaces the original action schema with the
canonical final schema constructed from that history. It does not add a final
control message, rewrite the history, or claim the runner was already final-only.
`lineage.jsonl` retains both schemas, the actual source trace/episode/turn, fetched
sources, schema provenance, and whether the intervention was applied. The frozen
profile records this policy and the actual Web review decoder. Existing SFT data
and its contracts remain unchanged.

`export_source_approval.json` preserves the separately approved original source
profile/receipt and expected raw-collection world size, without original answers
or rewards. Generation does not load that provenance sidecar or original logs.

Selection starts with the first eligible final per task. Within each explicit
benchmark scope, sort those public task IDs, shuffle using a separately fixed
`prefix_selection_seed` (default 47), then apply the predeclared per-domain cap.
Each domain uses its own `random.Random(seed)` instance, so adding another domain
cannot change its selection. No category, polarity, original answer quality, or
reward enters that selection. The freeze retains `benchmarks`, the selection
policy/version, seed, and cap. `freeze.json` and adjacent `prefixes.jsonl`
are the jointly approved immutable export artifact; approve both files together.
IDs, public messages, schema, harness/native identity, four distinct seeds, and
selection policy are fixed before new generation. Replacing the public-prefix
file after approval creates a new experiment even if task IDs are unchanged.
No task may be chosen by prospective rewards. Accepted SFT finals alone can test
the mechanics but do not establish representative reward contrast.

The public prefix contract is `native-final-public-prefix-v1`; the candidate
contract is `native-final-preference-candidate-v1`. `PublicFinalPrefix.from_dict`
must preserve schema property order. The exporter writes public prefixes without
sorting nested keys because the actual Web review decoder requires canonical
`plan`, `checks`, `answer` order. Other journals may use canonical sorted JSON;
their schemas are checked structurally and never replayed as decoder inputs.

Generation submits exactly four native attempts per frozen prefix. The thought
route must explicitly use stochastic `thinking`; JSON remains the existing
greedy structured phase. Calls for multiple prefixes may overlap within a seed.
All such calls finish both phases before `base.seed` changes, avoiding a thought/
JSON seed race. Distinct seeds do not guarantee distinct answers or useful gaps.
There are no synthetic rejects, retries, inserted closing/EOS tokens, completion
cropping, retemplating of generated thoughts, or repaired stops. Failed attempts
retain their actual request/response/native trace and a failure audit, without a
fabricated candidate. Successful candidates retain the actual combined thought
and JSON physical IDs; their record contains only public identity/messages/JSON.

Every request, candidate, audit, and native-trace journal closes before its rank
closure marker. `read_closed_g4` requires all declared ranks, including empty
shards, and verifies the exact seed-by-prefix attempts and ownership joins before
returning `generation_closed_before_scoring:true`. The separate offline TRAIN
scorer must call this helper before opening any label file. The generation CLI
has no reward/label argument. The pair builder subsequently uses genuine distinct
seeds with exact common prefix/schema/native identity/reward version and a strict
best/worst reward gap **greater than 0.05**. All tied/insufficient-contrast prefixes
are reported; they must not be replaced by reward-chosen IDs or synthetic rejects.

The stable integration functions are:

```python
freeze, prefixes, identity = load_frozen(approved_export_directory, tokenizer)
closed = read_closed_g4(
    generation_directory, freeze=freeze, prefixes=prefixes, tokenizer=tokenizer,
    world_size=expected_world_size, expected_source=caller_approved_source,
)
# Only now may a separate process open TRAIN labels/reward receipts.
```

`closed` contains `prefixes`, `candidates`, `audits` (also `attempts`), `freeze`,
and a `closure_receipt`. Approval of the export/profile is a caller responsibility;
loading a candidate's own identity is not independent approval. Runtime paths are
CLI inputs and are not embedded in public configuration examples.

Separate domain budgets and subsequent joint DPO training
------------------------------------------------------

`export --benchmark cuad` and `export --benchmark widesearch` independently
scope the same already-closed mixed raw collection. Repeat `--benchmark` to
declare a multi-domain scope; omission defaults to both. Different actual context
budgets require separate exports, candidate profiles, unique experiment IDs, and
independent approvals of both freeze and public-prefix files. The original raw
source/profile can be shared; the prospective merged source receipt may also be
shared when it matches the actual new candidate runtime. Generation consumes each
export separately and retains every prefix's original `native_identity`.

A joint DPO input must explicitly name each independently approved collection,
its freeze/public prefix artifact, generation directory, expected generation
world size/source, reward version, and independent TRAIN reward receipts. First
validate every collection's freeze and all generation shard closures. Only after
**all** collections close may any TRAIN reward receipt be opened. Reject cross-
collection task, prefix, trace, and actual phase-request identity reuse. Construct
pairs within each original prefix/schema/harness/native identity/reward version,
then combine validated pairs without replacing identities or reassigning seeds.
The base/reference derivative and actual tokenizer must agree across collections;
approved context budgets may differ and stay attached to their original domain.
Training metadata/adoption approval must preserve each full freeze, public
identity, closure receipt, reward version, and counts, omitting private paths.
Concatenating raw JSONL or relabelling Web examples with CUAD's context identity
does not satisfy this contract.

Reward proposal, pending independent approval
-------------------------------------------

The following is a pre-registered **proposal**, not an adopted reward version or
an automatic scorer change. No private TRAIN labels or held-out results were read
to choose it. Root must freeze a distinct reward version and its full evaluation
profile before using any new rewards or comparing pilot arms.

For independent Web TRAIN, run the unchanged public evidence guard and actual
public final schema validation first. A valid physical candidate whose final is
rejected by that guard receives reward 0; it remains a genuine candidate. The
guard verifies successful search/fetch, nonempty answer shape, required columns,
and read citations; it does not prove factual entailment of every cell. For a
guard-passing final, call the existing deterministic `evaluate_widesearch` with
`judge=None`, its full unchanged TRAIN evaluation profile, and full denominators:

```
proposed_reward = 0.75 * f1_by_row + 0.25 * f1_by_item
```

The proposed name is `independent-web-train-guarded-strict-row75-item25-v1`
(not frozen or official). Retain both component metrics and the guard decision.
It is a strict TRAIN proxy and does not reproduce the official semantic judge.
No guessed entity key, aliasing, new preprocessing, row truncation, or changed
matching rule is introduced.

There are two materially different existing TRAIN evaluation paths:

* **Required-only profile:** `web_train_evaluation` uses **all** required columns
  as the compound unique key, no preprocessing, exact matching. Every matched row
  therefore has all cells marked correct: row F1 and item F1 are mathematically
  equal. The proposed mixture is numerically identical to either component on
  this path and cannot be claimed to newly encourage complete rows.
* **Complete original profile:** an explicit smaller original entity key can
  align a row with some wrong non-key cells, so row F1 and item F1 may differ.
  The proposed mixture puts 75% on complete rows and 25% on partial cells. Preserve
  the actual original key/pipeline; never infer one from candidate answers.

Synthetic CPU boundary tests verify both paths, the fact that higher item F1 can
coincide with worse complete-row F1, duplicates, empty answers, unread citations,
and proposal boundedness. Coefficients must not be adjusted using an unclosed DEV
run. Held-out reporting should continue to show row F1 as the main Web measure
and item F1 as a diagnostic under its separately declared evaluator.

For CUAD, preserve the existing runner reward exactly: `cuad_task_metrics` F1
times the fraction of predicted spans that appear verbatim in the normalized
public document (1 for an empty span list). Negative-gold/empty prediction scores
1; positive-gold/empty prediction scores 0; negative false positives score 0.
This proposal changes no Jaccard/Parties matching, confidence interpretation, or
negative-empty rule. Synthetic tests establish those boundaries only.
