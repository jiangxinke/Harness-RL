# Optional cached source row selection

`--web-cached-rows` is **off by default**. It requires vLLM and
`--web-structured-actions --web-structured-final`, and applies only to structured
Web evaluation. Contract evaluation and training retain their existing behavior.
The tool uses no model, reference answer, benchmark-specific mapping, new network
request, extra model turn or extra tool-call allowance.
Prompt and completion token limits and the original context truncation stay the
same; the optional instructions and row provenance consume existing prompt space
and may displace older excerpts.

After a normal `fetch_url` has downloaded a source in the current episode, the
same tool can select records from that URL's cached extracted text:

```json
{"tool":"fetch_url","arguments":{"url":"https://example.org/export.csv","select_rows":{"format":"csv","columns":["code","name","region"],"where":[{"column":"region","op":"eq","value":"Example region"}],"limit":32,"after":0}}}
```

CSV and TSV require an explicit format and exact, distinct, nonempty header
names. `header_row` is a zero-based CSV record index, defaults to zero, and is
never guessed. Quoted delimiters, embedded newlines, CRLF, Unicode and an initial
BOM retain correct source character offsets. Width mismatches, duplicate column
names and malformed records reject the entire selection, including when the bad
record follows a requested first page. Unknown columns fail explicitly.

Cached-row protocol v3 adds an explicit per-selection CSV/TSV option,
`csv_trailing_empty_columns`. Its default `"reject"` retains the strict behavior
above. A source whose header ends in unnamed padding columns may be selected
with `"verified_empty_padding"`:

```json
{"tool":"fetch_url","arguments":{"url":"https://example.org/export.csv","select_rows":{"format":"csv","columns":["code","name"],"csv_trailing_empty_columns":"verified_empty_padding"}}}
```

This option applies to CSV and TSV, including explicitly selected worksheet or
archive sections. Only a trailing suffix of header cells equal to the empty
string may be excluded from column mapping. Before predicate filtering, the
`after` cursor or any returned page, the entire bounded table must have the
header's original full width and every excluded cell must equal the empty
string. A quoted empty CSV cell is empty; a space, `"null"`, `"0"` or any other
nonempty text is not. Empty names inside the content header, duplicate content
names, all-empty headers, ragged records and later malformed records still
reject the whole selection. No columns are guessed, renamed, trimmed or filled.
The original header and row text, delimiters, numeric spelling and offsets
remain intact. The policy field in the returned view is navigation metadata,
not extra row evidence. JSON requests explicitly carrying this option fail.
All original parser limits, observation budgets and read-provenance rules apply.

Enabled v3 manifests, inference runtime and paired protocol declarations record
the same complete supported padding policy configuration. Missing or old
enabled configurations cannot resume as v3. Disabled configuration and ordinary
tool schemas retain their previous serialized fields. Existing frozen v1/v2
sources keep their original identities and are not relabeled or rewritten.

JSON requires an array at the explicit RFC 6901 `path`; the empty pointer selects
a root array. Each selected column or predicate column is a pointer relative to
a record. Array indices must be explicit nonnegative decimal indices without
leading zeros. There is no dot notation, recursive search or inferred path.

```json
{"tool":"fetch_url","arguments":{"url":"https://example.org/records.json","select_rows":{"format":"json","path":"/results","columns":["/name","/location/latitude","/identifiers/0"],"where":[{"column":"/rank","op":"eq","value":"species"}]}}}
```

JSON duplicate keys anywhere in the full cached document, nonfinite numbers,
unpaired Unicode surrogates, malformed structure, and selected non-scalar cells
fail explicitly. Missing record fields return only `{"present":false}`. No empty
string, null or other replacement value is invented. Numeric cell `value` is the
original JSON numeric **lexeme as a string**, with `kind:"number"`; precision,
trailing zeros and exponent spelling are preserved. Actual string, boolean and
null kinds remain distinct. Each present JSON cell also has exact source offsets
covering its original scalar representation.

All predicates are ANDed. `eq` and `ne` compare exact strings, booleans or null
without cross-type coercion. `contains`, `starts_with`, `lex_ge` and `lex_le`
require source strings. Lexical comparisons use Python Unicode code-point order;
there is no locale, normalization, case folding or implicit date interpretation.
`num_eq`, `num_ge` and `num_le` explicitly request finite decimal comparison of a
JSON number lexeme or a CSV/JSON string containing a JSON decimal spelling. Their
argument must itself be a string. No floating-point round trip occurs, and an
invalid numeric operand rejects the selection by default. `exists` has no value argument.
Missing operands fail every predicate except `exists`, which tests presence.
There are no joins, regexes, sorting, aggregation, field aliases or answer filling.
The model must still verify all conditions and the original missing-field rule.

For an XLSX source, `format:"csv"` and an exact observed `worksheet` select the
existing extracted CSV section. For a ZIP text export, an exact observed
`archive_member` and matching format select the recorded member content bounds.
Names with truncation, duplicates, absent or unsupported members, or inconsistent
boundaries fail. Labels are excluded by their recorded boundaries; source text is
never split by guessing marker-looking content. XLSX values are the existing
reader's displayed/cached cell values, not evaluated formulas. Nested archives,
XML tables and arbitrary HTML tables are outside this selector's scope.

The returned `row_view.kind` is `literal-record-windows`. Every record contains
its exact `start`, `end`, `text`, zero-based `record_index`, selected `cells` and
actual `read_progress`. All offsets address the **same URL's cached extracted
text**. Each row is a separate observed literal window. Top-level `text/start/end`
cover only the first returned row; they never pretend that disjoint rows are one
continuous page. The navigation ledger records the additional row spans with the
selector version. Headers and selector metadata are navigation only and add no
data-row read. Zero matches publish no top-level text, add no read progress or
successful evidence, and preserve the downloaded cache. Existing citation and
final-answer guards are unchanged.

The entire JSON observation, including duplicated primary text, selected cells,
provenance and read telemetry, fits the original `max_page_chars` allowance.
A single row or fixed metadata that cannot fit produces an error without changing
read history. `limit` is 1–64 (default 32). `after` and returned `next_after` are
zero-based **cached data-record cursors**, independent of character `start` and
remote API pagination; combining a text start with selection fails. The full
bounded table is validated before any row is published. The implementation limits
sources to 5,000,000 cached characters, 100,000 data records, 300,000 JSON nodes,
depth 32, 32 selected columns, 8 predicates and 64 returned records. Decimal
spellings have a 256-character/exponent bound. No completeness claim extends
beyond this cached source. API query parameters, record offsets and remote next
pages require actual source documentation or observed links as before.

Enabled runs save the complete `web_cached_rows` configuration and version in the
generation manifest and inference runtime. Score-only, completed resume and paired
comparison reject missing, mismatched or unsupported declarations. Paired
`common.generation` must declare the same complete configuration. Legacy absence
means OFF. OFF retains the original action schema, persistent prompts, runtime and
manifest fields, tool behavior for ordinary calls, evidence guard and scoring.

Synthetic parser, tool, decoder, actual episode, runtime/resume and paired tests
live in `tests/test_dcc_web_rows.py`. They require no live source or benchmark
answers. These tests establish implementation behavior, not a benchmark score or
an improvement over another harness.

Additional padding controls live in `tests/test_dcc_web_rows_padding.py`.


Cached-row protocol v2 adds optional `select_rows.numeric_source_policy`:
`error` is the default; explicit `exclude_uncomparable` treats present scalar
source operands without an accepted finite decimal spelling as nonmatches for
numeric predicates. It does not parse ranges, trim strings or infer values.
Missing cells remain ordinary nonmatches. Invalid predicate arguments and
malformed or nonscalar source data still fail.

The opt-in returns `uncomparable_records_after` and `predicate_diagnostics`
with independent `match_count` and `uncomparable_count` per predicate, scoped to
cached records at or after the cursor. Predicates remain ANDed; independent
matches need not intersect. Use `ne` to exclude an exact scalar value. Exclusion
does not prove that an original task condition fails: inspect incomparable
source records separately when their count is positive. These counts are
navigation metadata, not observed row evidence or remote completeness. All
metadata shares the original observation budget; literal spans and read
telemetry retain their existing rules.
