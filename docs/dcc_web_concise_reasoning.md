# Optional concise Web reasoning

`--web-concise-reasoning` adds public, generic system guidance to Web episodes
in `dcc_online_grpo.py` and `dcc_online_web_evaluate.py`. It asks the model to
focus on the next real search/read action, avoid repeated table enumeration or
empty-answer deliberation, and avoid simulating requests or unobserved values.
The reserved final turn and all original evidence rules still apply.

The switch defaults off. Disabled prompts and serialized configuration retain
the previous behavior; CUAD prompts and CUAD evaluation configuration are
unchanged even when it is enabled. Enabled Web evaluation records
`web_concise_reasoning` with version `web-concise-next-action-v1` in
`comparison.harness`; every shard, score-only invocation and compared policy
must use the same setting. Online Web training records the same versioned
option. Set it in both the training and evaluation commands when comparing a
policy trained with this guidance.

This is an unvalidated behavioral intervention for long thought enumeration and
self-debate, not a demonstrated bug fix or benchmark improvement. It changes
only the prompt: thought/JSON budgets, real closing-token requirements,
stop/mask behavior, grammar, evidence guards, rewards and scores are unchanged.
It injects no closing token, fabricated JSON, source URL, answer value or retry.
Use fresh TRAIN/DEV rollouts to assess it before a final benchmark comparison.
