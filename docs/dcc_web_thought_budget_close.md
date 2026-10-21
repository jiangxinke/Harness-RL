# Optional Web evaluation thought closing

`--web-thought-budget-close` in `dcc_online_web_evaluate.py` enables an
experimental decoding policy. On the last available thought position, HF's
`forced_eos_token_id` masks the support to the real `END_THINK` token. HF then
selects and returns that token through its normal generation loop. Earlier
positions and natural early stops keep their existing behavior. No token is
appended and the thought budget does not increase. JSON decoding, its budget,
public schemas, evidence rules and full failure denominators stay unchanged.

The switch defaults off; disabled configuration, completion and trace
serialization retain the previous shape. Enabled runs use protocol
`hf-native-web-thought-budget-close-evaluation-v1` and record the versioned
policy in `comparison.harness.web_thought_budget_close` and each native trace.
Traces also record the actual `forced_thought_positions`, using zero-based
thought positions. Every shard, score-only invocation and matched base/adapter
must use the same policy. Re-generate both arms when changing the switch.

This option supports exact Web schemas only and refuses behavior-logprob
capture and joint JSON training. At the constrained position the conditional
distribution is a singleton, with logprob zero; this is not a raw-model
logprob. Evaluation records no training behavior scores. Training support
would require a new behavior identity and the same position support in current
and reference replay; it is intentionally unavailable here.

This candidate targets exhausted thought budgets that prevent the JSON phase
from starting. It differs from concise prompt guidance and does not establish
better answers or a benchmark gain. Closing an unfinished thought can still
produce bad JSON, evidence failures or incorrect answers. Assess it on a fresh
matched development/control run before making any performance claim.
