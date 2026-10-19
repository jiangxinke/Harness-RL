# Optional CUAD retrieval profile

The baseline ranks contract chunks with BM25 over the complete question. The
optional profile learns at most ten discriminative query terms per CUAD clause
category from training answer spans, then fuses the original-question and
expanded-query rankings with reciprocal rank fusion (constant 60). It returns
the same bounded number of original chunks, preserving source offsets. Ties
use source chunk order, and a chunk appears once. Scores in enhanced results
are fusion scores rather than BM25 scores.

Fit a profile outside the checkout:

```bash
python scripts/dcc_fit_retrieval.py \
  --data /path/to/prepared/cuad/train.jsonl \
  --output /path/to/persistent/artifacts/cuad-retrieval.json
```

The fitter rejects every non-CUAD or non-training row, including negative
validation rows, and refuses to overwrite an existing profile. It groups
questions using the category written in the question itself. Task metadata
does not determine categories. Each candidate term must occur in at least
three positive training examples and eight percent of the category's positive
examples. Terms are ranked by smoothed log frequency lift against all positive
training questions. The profile records its schema, version, parameters, train
row/positive/contract/distinct-document counts and a digest of training IDs. It contains learned
terms and known question texts, without answer records, contract text or source paths.
Treat profiles as learned artifacts; do not commit the generated terms.

Runtime integration uses the existing document tool:

```python
from slime_plugins.dcc.retrieval import load_retrieval_profile, search_document

profile = load_retrieval_profile(profile_path)  # None selects the baseline
observation = search_document(document_tools, question, top_k=4, profile=profile)
```

An explicit missing or malformed profile raises an error before evaluation.
An unknown question/category, or a category without learned terms, falls back
to baseline retrieval. Matching uses question text with whitespace normalized;
inference never reads evaluation labels or arbitrary task metadata. A free-form
follow-up search therefore usually uses the ordinary document tool. The
enhancement changes ranking only, with the existing chunking and top-k cap.

Measure retrieval visibility on held-out development contracts before model
comparisons. Freeze profile fitting, chunking and top-k settings for both the
base policy and trained adapters. Reserve development contracts before fitting
or training; question-disjoint subsets from the same contracts do not create
an independent contract holdout.
