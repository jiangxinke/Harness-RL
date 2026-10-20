# Independent web tool curriculum

The optional web curriculum teaches search, reading, complete table extraction,
and row citations from independent public sources. The default mixed curriculum
is `independent-public-tables-v3`, with **176 training tasks and 34 validation
tasks**. Training questions have English and Chinese variants; both variants
remain in the training split. Validation keeps one variant of each selection.
Task IDs, source-family identifiers, and source URLs are disjoint across splits.
Separate IANA registries count as separate source families; sharing a domain
does not imply sharing a registry.

| Split | Sources | Tasks | Answer columns, excluding citations |
| --- | --- | ---: | ---: |
| Train | IANA HTTP status codes, IP protocol numbers, DNS record types | 58 | 2–3 |
| Train | PubChem periodic table | 24 | 8 |
| Train | Chinese prefecture-level administrative divisions | 46 | 3 |
| Train | Nobel Physics and Chemistry laureates | 48 | 6 |
| Validation | Independent IANA ICMP types and ARP operations | 8 | 2 |
| Validation | World Bank countries and economies | 22 | 8 |
| Validation | GBIF bird-order taxonomy | 4 | 9 |

The richer schemas contain real source fields: PubChem includes atomic mass,
electron configuration, group, state, and discovery year; Nobel includes
laureate identifiers, prize share, and motivation. World Bank includes capital,
region, income group, lending type, longitude, and latitude. GBIF includes
taxonomic rank, ancestry, identifiers, and status. Each question declares its
required columns and selection rule. Smaller IANA and administrative tables
retain their natural schemas.

Official WideSearch and CUAD test questions, answers, predictions, and labels
must not enter curriculum construction, teacher supervision, retrieval-profile
fitting, or checkpoint selection. Independent validation tasks never become
teacher training segments.

Set `DATA_ROOT` to an artifact directory outside the checkout. Preparation uses
CPU and public web requests; teacher collection runs the actual tools.

```bash
: "${DATA_ROOT:?Set DATA_ROOT to your artifact directory}"
python scripts/dcc_prepare_web_curriculum.py \
  --output-dir "$DATA_ROOT/web-curriculum-v3" \
  --source-mode mixed --teachers --workers 4
```

The script caches complete official CSV or JSON exports in `sources/`, creates
deterministic range, prefix, or province-selection questions, and records source
URLs, source families, preparation time, schemas, and split counts. PubChem uses
its compact official CSV export, preserving the full source rather than
cropping it around answer rows. Parsers reject incomplete API pages and
duplicate primary keys. Reserved, unassigned, unused, or incomplete entries are
excluded only by the explicit rules for that source, also stated in the
question. The manifest records the prepared snapshot; public sources may
change. Use `--no-download` to require existing cached exports, or
`--source-mode iana` / `--source-mode public` to prepare a subset.

Teacher trajectories run the real `EpisodeRunner` and web tools. They search
for a source, read its complete export, follow pagination if necessary, and
return the extracted table with citations. Successive fetches use a fixed
**1,000-character overlap** based only on the observed next-page offset, so a
field crossing a page boundary remains intact in an observation. Pagination
and source formatting never depend on gold-answer positions.

An empty search triggers alternate queries; no synthetic successful search is
inserted. Before the final answer, the source is reparsed and the complete
selected rows must match the task. Every target cell must occur in actual
fetched evidence, and each accepted teacher passes the evidence, answer-schema,
and metric checks used by a policy episode. Unavailable or changed sources
produce explicit failed-teacher records. Successful supervised segments are
written to `teacher_train.jsonl`; complete episodes, failures, tasks, source
exports, and the manifest are retained alongside it. These structured tasks
exercise tools and enumeration; they do not establish general open-web research
quality.

Each supervised segment contains only its historical chat messages and the
next assistant completion. Search/read observations stay in the prompt, and
the target completion never appears in its own prompt. Terminal segments also
record grounding fields for checking visibility after tokenization. Training
must skip overlong examples without truncating source evidence or targets.

Use a starting budget of **12,288 prompt tokens, 4,096 completion tokens, and
16,384 batch tokens**. Check these limits with the actual policy tokenizer and
track accepted final segments by source family, not just accepted search
segments. The batch limit is a sampler budget; it does not imply one padded
forward batch. For example, an independent-web-only LoRA run can use:

```bash
: "${MODEL_CACHE:?Set MODEL_CACHE to your policy model directory}"
: "${RUN_ROOT:?Set RUN_ROOT to your output directory}"
: "${WORKERS:?Set WORKERS to your chosen process count}"
python -m torch.distributed.run --standalone --nproc-per-node="$WORKERS" \
  scripts/dcc_tool_sft.py \
  --model "$MODEL_CACHE" \
  --records "$DATA_ROOT/web-curriculum-v3/teacher_train.jsonl" \
  --output-dir "$RUN_ROOT/web-tool-sft" --web-fraction 1 \
  --max-prompt-tokens 12288 --max-new-tokens 4096 --batch-tokens 16384
```

Keep generated tasks, source exports, teacher records, logs, and checkpoints in
the configured artifact root. See [dcc_training.md](dcc_training.md)
for model, quantization, mixed-CUAD training, and runtime requirements.
