# Online Web TRAIN whole-turn prompt history

The default online GRPO request and training identity are unchanged. To enable
token-measured history compaction for online Web training, add:

```text
--enable-online-web
--web-train-prompt-history-whole-turns
--web-train-prompt-history-keep-recent-turns 2
```

The recent-turn option defaults to 1 when enabled. It requires the enable flag;
the training flag requires online Web. The identity is
`hf-native-web-whole-turns-training-v1`. This is separate from the existing
evaluation-only `--prompt-history-whole-turns` option and does not relax its
restrictions.

Before each Web request, the unchanged whole-turn planner measures the complete
chat template with the actual tokenizer and both reserved generation budgets.
It removes only the oldest complete action/observation pairs when necessary,
preserving the system instruction, task, initial context, protected recent
turns, latest source read and current controls. It never slices source text or
JSON. An omission notice asks the model to reread missing sources; diagnostic
URLs are not added as evidence. CUAD requests retain their original path,
including within mixed native batches.

The native trace saves the original full messages, recomputable omission report
and actual submitted prompt IDs. Group closure recomputes the plan using the
owner's full request, actual policy tokenizer and budgets, then checks the
trace, delivered completion and streamed raw behavior scores. A trace cannot
enable this policy by declaring it. Rewards continue to use the actual final
prompt; omitted evidence is not silently restored.

Optimizer replay uses each segment's saved actual prompt and sampled thought
IDs. Original sampled behavior scores remain the importance-ratio denominator.
JSON stays greedy with zero policy loss, and the fixed group/step/thought-budget
denominator is unchanged. Joint JSON training, forced thought closing,
truncation and the evaluation history configuration cannot be combined with
this training option.

If protected history cannot fit, the original full-request
`NativeHFPromptBudgetExceeded` and native preflight trace remain unchanged.
The owner separately retains the failed history plan, which closure also
recomputes. Original `hf-native-context-sampled-prefix-v2` handling still
discards an initial overflow group and retains a verified previously sampled
thought prefix with zero episode reward after a later overflow. No completion
or behavior score is invented for the failed request. This option requires
fresh, reviewed runtime identity for an actual experiment; CPU controls alone
do not establish benchmark quality or authorize replay of earlier traces.
