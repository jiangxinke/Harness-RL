# Optional multiple Web citations

`scripts/dcc_model_evaluate.py --web-multi-citation` allows one Web result row
to cite up to four actually fetched sources, for example official API
documentation and a separate data response. It requires
`--inference-backend vllm --web-structured-final --web-structured-actions`.
It works with explicit review, bounded whitespace and native Thinking evaluation.

The switch is off by default. Existing single-citation schemas, schema versions,
requests and omitted configuration fields stay unchanged when it is off.
Enabled runs record the following in their manifest, runtime and paired
protocol's `common.generation`:

```json
"web_multi_citation": {
  "enabled": true,
  "version": "web-multi-read-citations-v1",
  "max_citations": 4
}
```

Use a fresh output directory and the same setting for both policies in a paired
comparison. Resume, score-only, native sidecar and comparison checks reject
missing, altered or mismatched declarations. Historical single-citation runs
continue to reconstruct the original schemas.

Every cited URL must still be an actual successful read in that episode.
Search snippets, listed outgoing links and failed fetches do not qualify.
Multiple citations permit evidence from separate sources; they do not prove
that the model's claims are supported. The existing evidence guard and scorers
retain responsibility for acceptance and quality.

The Web reader also rejects short robot-verification pages before optional
browser rendering or recording a successful read. This includes
`verify that you're not a robot` and its typographic-apostrophe variant.

The current guard rejects an empty Web table even though the structured schema
can encode abstention. Task prompts that allow empty results therefore have a
controller contract conflict. This change preserves the existing guard rather
than changing benchmark acceptance rules.
