# Exact text rewards for independent WideSearch training tasks

Independent TRAIN tasks can require the exact text of source cells, including
leading zeros, decimal precision, whitespace and UTC suffixes. The benchmark's
CSV inference protocol can change those spellings. Such tasks can explicitly
select `dcc-web-train-native-text-null-exact-v1`; benchmark evaluation and the
default training selector keep their existing behavior.

Each new label must declare `label_protocol` as
`dcc-web-train-native-text-null-label-v1` and `train_reward` as the complete
dictionary returned by `scripts/dcc_train_web_literal_reward.py:configuration`.
Its evaluation must contain only the same `required` columns as the public
TRAIN task. New curricula need separate dataset versions, task IDs, label files
and selection destinations. Do not edit historical labels or their results.

The selector also requires the explicit option:

```text
--web-train-reward-protocol dcc-web-train-native-text-null-exact-v1
```

A missing or mismatched label declaration or selector option rejects the
configuration before selection outputs are created. Closed-journal validation
still precedes the first label read. Existing completion, citation, schema,
native token, source identity and exported-data validation remain required.

Content cells must be JSON strings or null, with exactly the required columns.
No numeric conversion, normalization or missing-value substitution occurs.
All required cells form the row key. Row order is ignored; repeated prediction
rows count as false positives beyond their matches. Duplicate gold rows reject
the label. Invalid prediction rows invalidate the whole prediction. Optional
valid citation metadata is excluded from content comparison; existing grounding
checks still apply independently.

The result is `train_exact_row_f1`, with precision, recall and table-match
diagnostics. It is a TRAIN reward, not the official WideSearch item score or a
benchmark result. The selector records the full new reward declaration and a
separate summary version. The offline reward/DPO route does not yet support
this protocol; do not send these labels through its legacy scorer or reuse old
reward receipts.
