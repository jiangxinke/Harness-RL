# Token-budgeted Web observations

`--web-observation-token-budget 6144` is an optional candidate for the native
online Web trainer and the matched native HF Web evaluator. Omit the option to
retain the original tool observations and configuration identity. In training,
it requires `--enable-online-web` and affects only Web episodes in the mixed
CUAD/Web runner. Set the same cap for base and adapter evaluation.

The shared `EpisodeRunner` measures the complete serialized tool observation
JSON with the actual policy tokenizer, without special tokens. The cap includes
search results, every automatic source read, source text, errors and metadata.
The versioned configuration is `web-observation-token-windows-v1`. Direct users
of the shared runner must install their actual policy tokenizer with
`runner.set_web_observation_tokenizer(tokenizer)` before a Web rollout.

When the original observation fits, its content is unchanged. Otherwise, the
runner omits source-wide metadata, reduces search entries to whole returned
URLs, and exposes source text only as literal character-prefix windows. It
never summarizes source text or abbreviates a URL. `start`, `end` and
`next_start` are character offsets in the same downloaded source, including
Unicode text; they are not tokenizer offsets or API pagination parameters.
Continue a truncated source with `fetch_url` using the same URL and its
`next_start`. Previously exposed source content may be downloaded/cached, but
the evidence guard and navigation events receive only the packed windows.

An `observation_budget` notice identifies omitted metadata and whole search or
automatic-read entries. Error text remains an error and is explicitly marked
when shortened. Invalid source offsets or an identity too large to retain
produce a recoverable error with no source-text exposure. Archive member labels
alone never establish read evidence. The minimum accepted cap is 512 tokens;
a cap that cannot fit the explicit error with the supplied tokenizer fails.

The original strict whole-prompt checks still apply. A per-observation cap does
not guarantee that an arbitrarily large task, accumulated conversation or
current control message fits the model context. The separate whole-turn
history option remains evaluation-only; this option does not enable history
compaction during training. No benchmark improvement is established by the
cap or by a CPU boundary test.

Use the same optional argument in the existing commands:

```sh
python scripts/dcc_online_grpo.py \
  --model "$MODEL" --train-data "$CUAD_TRAIN" \
  --source-commit "$SOURCE_COMMIT" --output-dir "$TRAIN_OUTPUT" \
  --enable-online-web --web-train-data "$WEB_TRAIN" \
  --web-train-labels "$WEB_TRAIN_LABELS" --web-binding "$WEB_BINDING" \
  --web-observation-token-budget 6144

python scripts/dcc_online_web_evaluate.py \
  --model "$MODEL" --data "$WEB_DEV_DATA" --split validation \
  --source-commit "$SOURCE_COMMIT" --output-dir "$EVAL_OUTPUT" \
  --web-observation-token-budget 6144
```

These examples add only the observation cap. Keep all other precision,
generation, source, input, durable-storage and comparison settings matched to
the intended experiment; the online Web input binding requirements still apply.
