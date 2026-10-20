# Optional public Web data discovery guidance

`dcc_model_evaluate.py --web-data-discovery-guidance` appends one fixed generic discovery and API-use instruction block to the system message for structured WideSearch evaluation. The default is off; CUAD and training requests keep their existing prompts. This option requires `--inference-backend vllm --web-structured-final --web-structured-actions`.

The guidance asks the policy to reformulate unrelated searches, consult authoritative format documentation, correct failed API requests, distinguish text windows from API pages, and retain only verified rows. It supplies no URLs, records or answers. Original tools, parsers, provider selection, action schemas, budgets, evidence rules and scorers remain responsible for execution and acceptance. It never requires a nonempty answer.

For example, add the flag to an otherwise unchanged evaluation command and use a fresh output directory:

```bash
python scripts/dcc_model_evaluate.py \
  --model "$DCC_MODEL" --data "$DCC_WEB_DATA" \
  --output-dir "$DCC_NEW_OUTPUT" --inference-backend vllm \
  --web-structured-final --web-structured-actions \
  --web-data-discovery-guidance
```

The enabled manifest and inference runtime record the complete `public-web-data-discovery-guidance-v1` configuration. Resume, saved-result validation, native sidecar auditing and paired comparisons reject missing or mixed declarations. Enabled saved Web results must retain the exact guidance once in their first system message. Comparing an off run with an on run requires a separately preregistered harness experiment; the original paired model-gain comparator rejects mixed harness settings.

This implementation supports evaluation only. It does not revise frozen training examples, collectors or training profiles. New training collection would require an explicit versioned collection profile and new source provenance. CPU tests use synthetic tasks and mocked engines. They do not establish live vLLM compatibility, benchmark improvement or GPU success.
