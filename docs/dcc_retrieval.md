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

The recipe was selected after CPU diagnostics on CUAD validation data. On the
820 validation questions, complete gold-span visibility in the initial four
chunks rose from 307/621 to 411/621 while mean source context stayed near 5.3k
characters. This is a retrieval diagnostic, not a model benchmark score. The
82-question development set and the remaining 738 questions share the same 20
validation contracts; the latter are question-disjoint, not an independent
contract holdout. Parameters and the recipe must be frozen before the next
model comparison, using identical model and generation settings. A clean new
contract-level development split requires reserving additional training
contracts and excluding them from both profile fitting and any new training.
