# Explicit cached rows v3 for independent Web TRAIN

Use the separate profile `dcc-native-train-profile-web32k-cached-rows-v3`.
It requires the existing Web32k identity (32768 prompt tokens, native 16384
Thinking tokens plus 4096 JSON tokens, whole-turn prompt history), 20 steps,
explicit review and navigation. In `web`, set `cached_rows: true` and set
`cached_rows_configuration` to the complete current
`web_cached_rows_configuration(True)` declaration. The trusted source overlay
must declare this new profile version and the same full `web_cached_rows`
configuration. Existing profiles retain their original cache restrictions.

Generation additionally requires `--web-cached-rows`. Existing
`--native-phase-timing` and `--timeout-seconds 1800` are compatible. They keep
the original native token/closing checks and never authorize changes to a
previously frozen collection. No DPO support is added by this TRAIN profile.

New collection creates `web_cached_sources_rankN.jsonl`. Each episode/URL
publishes its actual full extracted source and navigation metadata once;
requests carry only ordered active source IDs and their content/metadata digests.
A digest is calculated once at source publication and checked once at sidecar
loading; unchanged caches are compared directly without per-turn rehashing. A source may contain at most
5,000,000 characters, and each tool observation retains the original 14000
character budget. Sources must begin empty and remain immutable during the
episode: replacement, eviction or alias cache keys fail closed. There can be at
most one new source per preceding fetch and at most 19 sources per episode.
Source sidecars remain private training artifacts; do not commit them.

Before opening TRAIN labels, the selector validates every cache source and
request reference, then rebuilds historical observations with the original
`WebTools` implementation without performing I/O. Selected native data has a
required sibling `native_web_cached_sources.jsonl`; the training loader checks
its identity, all references and orphan records before returning any samples.
Full replay includes literal cells, offsets, bounds, padding and read-progress.
Native prompt/completion IDs, schema equality, final grounding and loss masks
are unchanged. Missing sources, stale configurations and cross-episode/source
references are errors rather than a fallback to a generic loader.

## Separate static 8192-token thought profile

`dcc-native-train-profile-web32k-cached-rows-8k-thought-v3` is a separate explicit
profile for a static 8192-token Thinking budget plus the unchanged 4096-token
JSON budget. It requires `thought_tokens: 8192`, `thought_sampling_profile:
"thinking"`, 32768 prompt tokens and `max_model_len >= 45056` in the full native
identity. The existing v3 16k profile continues to require exactly 16384 thought
tokens and at least 53248 model tokens. Neither version adapts budgets during
an episode or permits substituting one arm's identity, source overlay or
source-sidecar declarations for the other.

Use a separately approved profile and matching frozen source overlay for each
arm. The overlay's `explicit_training_profile` and every source sidecar's
`profile_version` must match that arm's exact version. All cache configuration,
source publication, actual native closing/token validation, selector and loader
gates remain the same. `--native-phase-timing` and `--timeout-seconds 1800` work
with either explicitly selected static identity.
