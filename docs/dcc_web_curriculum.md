# Independent web tool curriculum

The optional web curriculum teaches search, reading, complete table extraction,
and row citations from public IANA registries. It does not use WideSearch
benchmark questions, answers, or predictions. Training uses HTTP status codes,
IP protocol numbers, and DNS resource-record types. Validation uses separate
ICMP and ARP registries. Registry sources are disjoint across the two splits;
English and Chinese training variants of a question remain in the same split.

```bash
python scripts/dcc_prepare_web_curriculum.py \
  --output-dir "$DATA_ROOT/web-curriculum" --teachers
```

The script caches the original CSV sources outside the checkout, generates
deterministic numeric-range questions, and records source URLs, preparation
time, and split counts. Names marked Reserved, Unassigned, or (Unused), range
allocations, and rows missing required fields are excluded by explicit rules
that also appear in each question. The current snapshot yields 58 training
and eight validation tasks. This small curriculum tests tool execution and
enumeration; it does not establish general open-web research quality.

Teacher trajectories run the real `EpisodeRunner` and web tools. They search
for the registry, read its CSV export, follow pagination if necessary, and
return the extracted table with citations. An empty search triggers alternate
queries; no synthetic successful search is inserted. Every target cell must
occur in the fetched source, and each accepted teacher must pass the same
evidence, answer-schema, and metric checks as a policy episode. Unavailable or
changed sources produce explicit failed-teacher records rather than invented
observations. Successful supervised segments are written to
`teacher_train.jsonl`; full episodes and failures are retained alongside it.
No validation task becomes a teacher training segment.

Each supervised segment contains only its historical chat messages and the
next assistant completion. Search/read observations stay in the prompt, and
the target completion never appears in its own prompt. Terminal segments also
record grounding fields for checking visibility after tokenization. Training
must skip overlong examples without truncating source evidence or targets.

Keep generated tasks, source exports, teacher records, logs, and checkpoints in
the configured artifact root. See [dcc_training.md](dcc_training.md)
for the optional MoE training runtime.
