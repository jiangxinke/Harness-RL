# CUAD task-bound retrieval and contiguous source evidence

The optional TRAIN profile `dcc-native-train-profile-cuad-taskbound-contiguous-v2`
changes two CUAD harness behaviors: followup search uses the current public task's
fitted expansion while retaining the original model query for BM25, and final
span grounding can use overlapping or adjacent observed intervals from that
same public document. Default and v1 configurations retain their behavior.

Create the binding from the exact original TRAIN retrieval profile bytes:

```python
from dcc_cuad_retrieval import make_retrieval_binding
from dcc_review_training import CUAD_TASKBOUND_UNION_PROFILE

profile["version"] = CUAD_TASKBOUND_UNION_PROFILE
profile["web_profile_version"] = "dcc-native-train-profile-web20-review-navigation-v1"
profile["cuad"]["retrieval"] = make_retrieval_binding(
    original_profile_bytes,
    initial_chunks=8,
    retrieval_context_chars=0,
    retrieval_max_context_chars=16000,
    taskbound_contiguous=True,
)
```

Use the original approved model/tokenizer identity and CUAD structured schema;
the new binding does not rewrite either. The binding version is
`cuad-train-retrieval-taskbound-contiguous-v2`. The source overlay's
`explicit_training_profile` must match the new wrapper profile. The unchanged
Web route can also be the separately approved whole-turn Web32k v2 profile;
its existing identity requirements still apply. The existing joint cached16k
Web/CUAD v3 profile requires the original CUAD v1 binding and does not
automatically support this new route.

Collect a **new** public TRAIN rollout with `dcc_collect_native_train.py generate`
using the approved profile and a fresh output/collection identity. Its runner,
public request replay, closed journal checks, strict selector and native SFT
loader share the binding. Never retrofit the new marker onto historical
rollouts or reinterpret earlier selections as v2 results. Selection score and
confidence rules are unchanged; the native loader preserves actual prompt IDs,
completion IDs and phase boundaries. Collection and native SFT are covered by
synthetic CPU integration tests; no new DPO training result is claimed.

Each contributing hit must exactly equal the bound public document's original
`[start:end]` slice. Only overlapping or adjacent intervals merge. Gaps are not
filled, and separate documents cannot share evidence. The native validator first
reconstructs the exact physical prompt IDs from the effective approved messages,
then requires the **complete original source user message**, including its source
header and JSON hit offsets/text, to appear verbatim in the actual decoded prompt.
Matching a quotation alone, offsets elsewhere, or another identical quotation at
a different source position is insufficient. Final answer containment keeps the
existing whitespace folding only; no punctuation, case or Unicode normalization
is added.

This is a harness fix with synthetic CPU validation. New real collection,
training and benchmark evaluation are required before claiming any score gain.
