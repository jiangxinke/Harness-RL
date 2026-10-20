# Whole-turn prompt history

Native Thinking evaluation and Web TRAIN collection can enable
`--prompt-history-whole-turns`. The default remains off. This option measures the
rendered prompt with the actual tokenizer and removes the oldest complete
action/observation turns only when the prompt exceeds its budget.

The original system instruction, task, initial context, current controls, latest
interaction, and latest successful source read are retained. Assistant JSON and
source text are never sliced. An omission notice tells the policy to reread an
omitted source when needed. URL metadata in the audit distinguishes source
reads, unread search results, outgoing links, and failed reads; it adds no
evidence to the policy prompt.

For a separate 32K Web experiment, use these flags with the usual approved
native source receipt, model, data, and output arguments:

```text
--native-thinking
--native-thought-tokens 16384
--native-thought-sampling thinking
--max-new-tokens 20480
--max-prompt-tokens 32768
--prompt-history-whole-turns
```

The engine must reserve at least 53,248 tokens: 32,768 prompt tokens plus the
unchanged 16,384-token thought and 4,096-token JSON budgets. The second generation
stage receives every actually generated thought token. Increasing the prompt
budget does not reduce either output budget. Optionally set
`--prompt-history-keep-recent-turns` to retain more than one recent interaction.

Full original requests remain in the collection journal and native trace.
Traces also record effective prompt IDs, omitted message indices, and measured
token counts. Validation reconstructs the original request and repeats the
packing operation; a self-consistent edited trace cannot replace that replay.

When the minimum protected context still cannot fit, the episode records a
`context_budget_failure` with a zero-generation preflight trace. It contributes
zero to Web scores, remains in the full task denominator, and supplies no SFT
targets. The run is incomplete and cannot support an improvement claim.

Base and trained evaluations must use identical history declarations and
budgets. Keep results from older 16K protocols separate. Fresh Web collection
also requires the exact history declaration in its approved TRAIN identity;
preserved CUAD records retain their original identity and token IDs.
