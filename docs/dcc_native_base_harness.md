# Native CUAD and WideSearch reference harness

The selected reference configurations are `cuad_native16k` and `web_steps20_32k`. The remaining WideSearch presets retain the other declared candidate commands. `configs/dcc/native_base_harness_candidates.json` contains complete generation/strict-score templates, with `finalized: true` and `selected_web_preset: "web_steps20_32k"`. Selection used the complete reference evaluations and the unchanged quality gate. This configuration selection makes no training-gain claim.

The configuration was validated against reference source commit `76b57e227b943fb34a446226fa67cc1420f40e52`; that identifier records the actual reference evaluation source. Portable execution uses the caller's actual committed code and a fresh source receipt, as shown below. A new commit requires its own full evaluation and source provenance; it does not inherit the reference source's results or claim identical behavior. The paired comparator separately uses `4bd41a2060fbdead6ad3e135bfd3db4d5e41a895`; its source role does not change the actual generation/scoring source.

The model is the caller-supplied `DCC-Qwen3-30B-A3B-Thinking-2507-125E-traincal` derivative of `Qwen/Qwen3-30B-A3B-Thinking-2507`, revision `144afc2f379b542fdd4e85a1fcd5e1f79112d95d`, with 29,852,350,464 logical base parameters. These presets evaluate BF16 weights. NF4 training is a separate configuration and does not change the BF16 evaluation commands.

| Configuration | CUAD | Web research | Web research + cached rows | Web 16 turns | Web 20 turns |
| --- | --- | --- | --- | --- | --- |
| Full development tasks | 820 | 34 | 34 | 34 | 34 |
| Physical shards | 16 | 16 | 16 | 16 | 16 |
| Prompt budget | 16,384 | 32,768 | 32,768 | 32,768 | 32,768 |
| Thought + JSON budget | 16,384 + 4,096 | 16,384 + 4,096 | 16,384 + 4,096 | 16,384 + 4,096 | 16,384 + 4,096 |
| Engine context | 36,864 | 53,248 | 53,248 | 53,248 | 53,248 |
| Model turns | 4 | 12 | 12 | 16 | 20 |
| Initial retrieval chunks | 8 | 8 | 8 | 8 | 8 |
| Batch size | 8 | 4 | 4 | 4 | 4 |
| Episode timeout (seconds) | 1,800 | 3,600 | 3,600 | 3,600 | 3,600 |
| Auto-read sources | default/off | 2 | 2 | 2 | 2 |
| Research checkpoints | default/off | on | on | off | off |
| Cached row selection | default/off | off | on | off | off |

All five use async vLLM, eager execution, 8,192 batched tokens and seed 42. Thinking sampling applies only to the thought phase: temperature 0.6, top-p 0.95 and top-k 20. The global evaluation/JSON sampler retains the frozen source's neutral greedy defaults. Native generation must produce its own closing thought token; JSON continues the actual prompt/thought token sequence under the original schema. Total new tokens remain 20,480 per model call. No forced close, repair, label/scorer changes or automatic retry is added.

Provide deployment-owned values for `DCC_PYTHON`, `DCC_SCORE_PYTHON`, `DCC_REPOSITORY`, a new `DCC_SOURCE_ROOT`, `DCC_MODEL`, the benchmark data, separate fresh output directories and `DCC_SHARD_INDEX`. Set `DCC_SHARD_INDEX` to each value 0 through 15 to cover a full group; keep all 16 original processes and artifacts. CUAD additionally requires `DCC_CUAD_RETRIEVAL_PROFILE`, with the same frozen retrieval configuration across the group. Search/browser configuration and credentials belong in the deployment environment described in [dcc.md](dcc.md); keep them identical across Web candidates. The portable commands contain no deployment-specific browser wrapper, endpoint or directory.

First commit the reviewed implementation in `DCC_REPOSITORY`. Set `DCC_SOURCE_ROOT` to a new export directory owned by the caller. The following commands export that repository's actual committed `HEAD`; uncommitted changes are excluded. They then exclusively create `.dcc-native-source.json` inside the exported source root. Run all generation shards and strict scoring from this same unchanged export.

```bash
set -euo pipefail
: "${DCC_PYTHON:?Set the evaluation Python executable}"
: "${DCC_REPOSITORY:?Set the repository checkout}"
: "${DCC_SOURCE_ROOT:?Set a new source export directory}"
DCC_SOURCE_COMMIT="$(git -C "$DCC_REPOSITORY" rev-parse --verify 'HEAD^{commit}')"
export DCC_SOURCE_COMMIT
"$DCC_PYTHON" - "$DCC_SOURCE_ROOT" "$DCC_SOURCE_COMMIT" <<'PY_CREATE_ROOT'
from pathlib import Path
import re
import sys

root, commit = Path(sys.argv[1]), sys.argv[2]
if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
    raise ValueError("Expected the actual full 40-character Git commit")
root.mkdir(parents=True, exist_ok=False)
PY_CREATE_ROOT
git -C "$DCC_REPOSITORY" archive --format=tar "$DCC_SOURCE_COMMIT" | tar -xf - -C "$DCC_SOURCE_ROOT"
DCC_NATIVE_SOURCE_RECEIPT="$DCC_SOURCE_ROOT/.dcc-native-source.json"
export DCC_NATIVE_SOURCE_RECEIPT
"$DCC_PYTHON" - "$DCC_NATIVE_SOURCE_RECEIPT" "$DCC_SOURCE_COMMIT" <<'PY_CREATE_RECEIPT'
import json
from pathlib import Path
import re
import sys

path, commit = Path(sys.argv[1]), sys.argv[2]
anchor = "5bed6188feaaf6c79fac53eead30cfbad83b0beb"
if re.fullmatch(r"[0-9a-f]{40}", commit) is None or commit == anchor:
    raise ValueError("Expected the actual new native implementation commit")
receipt = {
    "version": "dcc-native-inference-source-v1",
    "source_commit": commit,
    "base_source_commit": anchor,
    "protocol": "native-thinking2507-real-close-two-stage-4096-4096-v2",
    "frozen": True,
}
with path.open("x", encoding="utf-8") as stream:
    json.dump(receipt, stream, indent=2)
    stream.write("\n")
PY_CREATE_RECEIPT
```

`source_commit` must be the actual commit just exported, not the reference snapshot identifier. `base_source_commit` is the fixed native-protocol anchor shown above, not the exported commit's immediate Git parent. The receipt's v2 protocol identifies that source-receipt contract; the explicit thought budget and sampling arguments select the budgeted v3 generation protocol separately. The runtime requires the receipt at `DCC_SOURCE_ROOT/.dcc-native-source.json` and records the actual receipt in manifests and native traces. Keep those new records with the complete new evaluation; do not relabel them as the reference snapshot. Do not edit the export after creating its receipt.

CUAD retains document search/read tools, retrieval 8, the original structured span/confidence answer format and final review. Its retrieval context budget stays 0 with a 16,000-character maximum. All Web candidates enable structured actions/final, bounded whitespace, navigation ledger, explicit review and auto-read 2. The research variants enable public-task research checkpoints; the cached variant adds only literal-record selection from already downloaded sources under the unchanged evidence guard. The 16/20-turn pair leaves both research checkpoints and cached-row selection disabled. Those two commands differ only in `--max-steps` and their fresh output directories; their other arguments exactly match the declared 12-turn baseline.

Run each chosen command separately. The examples show one physical shard, not a complete result.

cuad_native16k

```bash
"$DCC_PYTHON" "$DCC_SOURCE_ROOT/scripts/dcc_model_evaluate.py" \
  --model "$DCC_MODEL" \
  --model-name DCC-Qwen3-30B-A3B-Thinking-2507-125E-traincal \
  --inference-backend vllm \
  --vllm-async \
  --vllm-enforce-eager \
  --vllm-batched-tokens 8192 \
  --data "$DCC_CUAD_DATA" \
  --output-dir "$DCC_CUAD_NATIVE16K_OUTPUT" \
  --eval-shard-index "$DCC_SHARD_INDEX" \
  --eval-num-shards 16 \
  --max-steps 4 \
  --max-new-tokens 20480 \
  --max-prompt-tokens 16384 \
  --initial-chunks 8 \
  --seed 42 \
  --batch-size 8 \
  --timeout 1800 \
  --web-structured-final \
  --web-structured-actions \
  --native-thinking \
  --native-source-receipt "$DCC_NATIVE_SOURCE_RECEIPT" \
  --retrieval-profile "$DCC_CUAD_RETRIEVAL_PROFILE" \
  --retrieval-context-chars 0 \
  --retrieval-max-context-chars 16000 \
  --native-thought-tokens 16384 \
  --native-thought-sampling thinking
```

web_research32k

```bash
"$DCC_PYTHON" "$DCC_SOURCE_ROOT/scripts/dcc_model_evaluate.py" \
  --model "$DCC_MODEL" \
  --model-name DCC-Qwen3-30B-A3B-Thinking-2507-125E-traincal \
  --inference-backend vllm \
  --vllm-async \
  --vllm-enforce-eager \
  --vllm-batched-tokens 8192 \
  --data "$DCC_WEB_DATA" \
  --output-dir "$DCC_WEB_RESEARCH32K_OUTPUT" \
  --eval-shard-index "$DCC_SHARD_INDEX" \
  --eval-num-shards 16 \
  --max-steps 12 \
  --max-new-tokens 20480 \
  --max-prompt-tokens 32768 \
  --initial-chunks 8 \
  --seed 42 \
  --batch-size 4 \
  --timeout 3600 \
  --web-structured-final \
  --web-structured-actions \
  --native-thinking \
  --native-source-receipt "$DCC_NATIVE_SOURCE_RECEIPT" \
  --auto-read-sources 2 \
  --web-bounded-whitespace \
  --web-navigation-ledger \
  --web-explicit-review \
  --native-thought-tokens 16384 \
  --native-thought-sampling thinking \
  --web-research-checkpoints
```

web_research_cachedrows32k

```bash
"$DCC_PYTHON" "$DCC_SOURCE_ROOT/scripts/dcc_model_evaluate.py" \
  --model "$DCC_MODEL" \
  --model-name DCC-Qwen3-30B-A3B-Thinking-2507-125E-traincal \
  --inference-backend vllm \
  --vllm-async \
  --vllm-enforce-eager \
  --vllm-batched-tokens 8192 \
  --data "$DCC_WEB_DATA" \
  --output-dir "$DCC_WEB_RESEARCH_CACHEDROWS32K_OUTPUT" \
  --eval-shard-index "$DCC_SHARD_INDEX" \
  --eval-num-shards 16 \
  --max-steps 12 \
  --max-new-tokens 20480 \
  --max-prompt-tokens 32768 \
  --initial-chunks 8 \
  --seed 42 \
  --batch-size 4 \
  --timeout 3600 \
  --web-structured-final \
  --web-structured-actions \
  --native-thinking \
  --native-source-receipt "$DCC_NATIVE_SOURCE_RECEIPT" \
  --auto-read-sources 2 \
  --web-bounded-whitespace \
  --web-navigation-ledger \
  --web-explicit-review \
  --native-thought-tokens 16384 \
  --native-thought-sampling thinking \
  --web-research-checkpoints \
  --web-cached-rows
```

web_steps16_32k

```bash
"$DCC_PYTHON" "$DCC_SOURCE_ROOT/scripts/dcc_model_evaluate.py" \
  --model "$DCC_MODEL" \
  --model-name DCC-Qwen3-30B-A3B-Thinking-2507-125E-traincal \
  --inference-backend vllm \
  --vllm-async \
  --vllm-enforce-eager \
  --vllm-batched-tokens 8192 \
  --data "$DCC_WEB_DATA" \
  --output-dir "$DCC_WEB_STEPS16_32K_OUTPUT" \
  --eval-shard-index "$DCC_SHARD_INDEX" \
  --eval-num-shards 16 \
  --max-steps 16 \
  --max-new-tokens 20480 \
  --max-prompt-tokens 32768 \
  --initial-chunks 8 \
  --seed 42 \
  --batch-size 4 \
  --timeout 3600 \
  --web-structured-final \
  --web-structured-actions \
  --native-thinking \
  --native-source-receipt "$DCC_NATIVE_SOURCE_RECEIPT" \
  --auto-read-sources 2 \
  --web-bounded-whitespace \
  --web-navigation-ledger \
  --web-explicit-review \
  --native-thought-tokens 16384 \
  --native-thought-sampling thinking
```

web_steps20_32k

```bash
"$DCC_PYTHON" "$DCC_SOURCE_ROOT/scripts/dcc_model_evaluate.py" \
  --model "$DCC_MODEL" \
  --model-name DCC-Qwen3-30B-A3B-Thinking-2507-125E-traincal \
  --inference-backend vllm \
  --vllm-async \
  --vllm-enforce-eager \
  --vllm-batched-tokens 8192 \
  --data "$DCC_WEB_DATA" \
  --output-dir "$DCC_WEB_STEPS20_32K_OUTPUT" \
  --eval-shard-index "$DCC_SHARD_INDEX" \
  --eval-num-shards 16 \
  --max-steps 20 \
  --max-new-tokens 20480 \
  --max-prompt-tokens 32768 \
  --initial-chunks 8 \
  --seed 42 \
  --batch-size 4 \
  --timeout 3600 \
  --web-structured-final \
  --web-structured-actions \
  --native-thinking \
  --native-source-receipt "$DCC_NATIVE_SOURCE_RECEIPT" \
  --auto-read-sources 2 \
  --web-bounded-whitespace \
  --web-navigation-ledger \
  --web-explicit-review \
  --native-thought-tokens 16384 \
  --native-thought-sampling thinking
```

After every physical shard is terminal and the complete original raw/native ownership audit passes, run the original strict score-only command for that same group. Set `DCC_DATA` and `DCC_OUTPUT` to that group's already-complete data/output:

```bash
"$DCC_SCORE_PYTHON" "$DCC_SOURCE_ROOT/scripts/dcc_model_evaluate.py" \
  --model "$DCC_MODEL" \
  --model-name DCC-Qwen3-30B-A3B-Thinking-2507-125E-traincal \
  --data "$DCC_DATA" \
  --output-dir "$DCC_OUTPUT" \
  --score-only
```

CUAD keeps the original scorer; WideSearch keeps its original `strict_exact_proxy` table protocol with no local judge. Protocol failures remain in the full task/status denominator. Infrastructure errors retain their original ineligible status.

The reference Web selection completed after all declared candidate groups were terminal, preserved, audited and scored. `web_steps20_32k` passed the unchanged Web gate: at least 31 tasks with successful search and read, at least 28 accepted nonempty outputs, accepted fraction at least 0.75 within each language, zero infrastructure errors, full coverage and telemetry. Eligible candidates were ranked by accepted nonempty count, then strict row F1, then fewer model turns. New source evaluations must retain the same full-group barrier, gate and complete failures. No partial result or held-out/test metric selects the harness.

The matched adapter plan must retain the complete original CUAD and selected Web declarations separately, together with their exact baseline commands, caches and provenance. Generation may change only the approved adapter policy fields and output path. Evaluate both preregistered learning rates (1e-5 and 2e-6) at steps 8/16/32/64/128: 10 checkpoints, 320 physical generation jobs, 20 full strict scores and 20 pairs. All 20 groups must close/audit/score before any pair or checkpoint selection. Pair all 20 with 10,000 bootstrap resamples, seed 47 and confidence 0.95 using the actual comparison-only source. Keep generation, TRAIN and comparator identities separate. An actual selected-harness receipt must still bind the complete baseline commands, caches and provenance; this portable file alone does not release a trained adapter.
