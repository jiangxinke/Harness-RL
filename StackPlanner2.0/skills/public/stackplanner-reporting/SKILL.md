---
name: stackplanner-reporting
description: Use inside the StackPlanner Reporter specialist to turn supplied artifact bodies into an evidence-grounded Markdown report, preserve citations and human feedback across revisions, and verify report quality before creating a versioned deliverable.
---

# StackPlanner Reporting

## Boundary

The CentralAgent owns planning, research, and human interaction. The Reporter
owns synthesis only. Use the delegated task, expected output,
`context_refs.artifact_bodies`, task memory, and report requirements as the
complete evidence boundary. Do not search for missing facts and do not delegate.

## Evidence Pass

Before drafting:

1. Read every materialized artifact body and identify its artifact ID and type.
2. Build a private claim ledger separating verified facts, source-backed
   analysis, recommendations, conflicts, and unresolved gaps.
3. Treat pinned human feedback and the latest visible user request as mandatory.
4. For revisions, compare the prior report with the feedback. Preserve
   unaffected content, sources, tables, and images; change only what the request
   or corrected evidence requires.
5. If an artifact is marked truncated, do not imply that its omitted content was
   checked. State the limitation when it matters.

## Report Design

Choose the smallest structure that fully serves the request. A substantial
analytical report normally contains:

- one H1 title;
- 4-6 answer-first key findings;
- a short overview defining scope and context;
- detailed sections organized around the user's decisions or questions;
- concise Markdown tables for comparisons, metrics, timelines, or options;
- conclusions and actionable recommendations only when supported;
- limitations, unresolved questions, or missing information;
- a final source list preserving supplied titles and URLs.

Do not force this template onto a short memo, code explanation, or narrowly
requested format. Follow explicit audience, language, tone, style, and length
requirements first.

## Integrity Rules

- Use only supplied facts and artifact bodies. Never invent a statistic,
  quotation, event, URL, image, or citation.
- Clearly distinguish facts from inference and recommendations.
- Preserve units, dates, denominators, comparison baselines, and uncertainty.
- Represent conflicts instead of silently choosing a convenient source.
- Say that information was not provided when a required answer is unsupported.
- Reuse images only from supplied Markdown image references and place them near
  the relevant analysis. Never create an image URL.
- Produce clean Markdown without an outer code fence, transport JSON, process
  narration, or claims about filesystem limitations.

## Revision Contract

A revision is a complete new report version, not a patch. Apply every feedback
item, preserve valid citations and unaffected sections, and link the result to
the prior report through the artifact metadata supplied by the harness. If a
feedback request needs evidence that is absent, return `partial` with a precise
evidence gap so Central can delegate research before trying again.

## Delivery and Quality Gate

For long reports, write a unique Markdown file under
`/mnt/user-data/outputs/report-<action_id>.md`, set `artifact_content` to null,
and list the path in `created_paths`. Never overwrite a prior version. A short
report may be returned directly in `artifact_content`.

Before marking the task complete, verify:

- every delegated requirement and mandatory feedback item is addressed;
- key claims map to supplied evidence and source URLs are preserved;
- tables, numbers, dates, and units are internally consistent;
- limitations and conflicts are visible;
- the artifact is a polished final report rather than notes or an outline;
- `artifact_metadata` contains `source_artifact_ids`, `quality_checks`, and any
  remaining `evidence_gaps`.
