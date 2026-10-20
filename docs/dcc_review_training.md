# Review and navigation in native TRAIN trajectories

The optional `dcc-native-train-profile-web20-review-navigation-v1` profile records
the same 20-turn review and navigation workflow used by the Web harness. The
existing `dcc-native-train-profile-v1` profile and default sampler keep their
original behavior.

The new profile requires `web.max_steps=20`, `explicit_review=true`,
`navigation_ledger=true`, and `cached_rows=false`. Its `collection_overlay_id`
must match the actual source declaration in `.dcc-training-code-overlay.json`;
the declaration also records `explicit_training_profile` and
`baseline_source_unchanged=false`. Model, tokenizer, source identity, native
thinking budgets, and optional CUAD settings retain their existing checks.

Collection stores the actual requests, tool observations, completion tokens,
and terminal journals. Validation replays the public request state through the
real harness, including intermediate review schemas, navigation state, and the
final-turn instruction. It does not edit recorded tokens or insert controls.

Selection reads labels only after all shard journals close. The optional
`select --maximum-episode-requests 12` rejects an entire longer trajectory;
it never truncates it. Failed episodes remain in the selection audit. Keep
training labels on the selecting host and use only TRAIN examples here.

Training can opt into `--web-stage-profile final50-search25-fetch25`. This uses
the deterministic final/search/final/fetch cycle, with stage proportions
50%/25%/25%. Omitting this flag preserves the original stage cycle and random
sampling sequence.

Regression coverage lives in `tests/test_dcc_review_training.py`, including
real harness request replay at turns 3 and 20, tampered controls and schemas,
closed-journal label gating, native record loading, and unchanged default
sampling. These CPU checks establish compatibility; gains require a completed
matched evaluation of the trained model against the base model.
