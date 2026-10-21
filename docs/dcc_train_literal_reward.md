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
separate summary version.

The offline reward scorer accepts this protocol only with the new manifest
version `native-final-train-literal-row-reward-inputs-v1`. Each literal Web
collection must declare its exact `reward_version` and full `train_reward`.
All original and candidate generation journals and cross-collection ownership
must close before labels are opened. The original evidence guard still applies.

Literal reward receipts use
`native-final-train-literal-row-reward-receipt-v1`, retaining the full reward
configuration, label protocol, TRAIN-only metric and evidence-guard outcome.
DPO requires the explicitly approved manifest version
`native-final-dpo-literal-row-collections-v1` and the same complete declaration.
It validates receipt metadata before constructing training pairs and retains
each collection's configuration in checkpoint and adapter approval metadata.
The strict reward-gap threshold and fresh LoRA/reference behavior are unchanged.
This is offline DPO, not online RL. Legacy receipts, labels and manifest
versions cannot implicitly select the new reward.
