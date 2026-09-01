---
name: stackplanner-long-task
description: Use for long-horizon StackPlanner research, report, or implementation work that needs staged delegation, artifact lineage, failure recovery, and human review. The workflow is perception and scope, outline, optional approval, evidence or implementation, reflection and replan on failure, versioned deliverable, human feedback, revision, verification, then finish.
---

# StackPlanner Long Task

## Scope

Use this Skill only inside a StackPlanner specialist delegation. The CentralAgent owns control flow; complete only the delegated stage and never delegate again or ask the user directly.

## Workflow Profile

Use the smallest profile that satisfies the delegated contract:

- **Direct**: no staged workflow for a simple explanation or already-resolved answer.
- **Bounded execution**: one precise specialist task followed by verification.
- **Deliberate**: ambiguous, high-impact, multi-hop, or substantial deliverable work. Use the stage contract below.

Do not recreate a completed stage. Inspect `context_refs.artifact_bodies`,
`context_refs.mandatory_requirements`, and the supplied artifact IDs first.

## Deliberate Stage Contract

1. **Perception and scope**: normalize the objective, deliverable, audience,
   constraints, acceptance criteria, assumptions, and only decision-critical
   unresolved questions. Return at most one batch of 1-5 orthogonal questions.
2. **Outline or plan**: define the minimum useful sections or dependency DAG,
   required evidence, acceptance criteria, and decisions that need human approval.
3. **Evidence or implementation**: execute the bounded task, preserve source URLs or test evidence, and distinguish facts from inference.
4. **Recovery**: if blocked, return a compact failure diagnosis and the smallest viable retry scope. Do not hide partial failure.
5. **Deliverable**: put the full result in `artifact_content` or a created output file; keep the summary under 700 characters.
6. **Revision**: apply pinned human feedback before older plans or recalled memory, preserve artifact lineage, and identify the prior version in metadata when available.
7. **Verification**: check the delegated acceptance criteria, citations or tests, unresolved gaps, and output location before reporting success.

The `stage` value describes the delegated work: perception, planning, research,
implementation, reporting, revision, or verification. For example, a researcher
performing a current-information lookup always uses `research`.

## Artifact Rules

- Reports, outlines, research bodies, generated files, evidence bundles, and multimodal observations are artifacts.
- Task memory receives only a compact summary and artifact reference.
- Never overwrite an earlier report version. Produce a new version linked to its parent.
- Preserve feedback bindings and source references across revisions.

## Output

Return one JSON object only:

```json
{
  "summary": "compact result for the CentralAgent",
  "artifact_content": "full artifact body or null when files were created",
  "artifact_type": "outline | research_observation | perception_observation | report_revision | generated_file",
  "artifact_metadata": {
    "created_paths": [],
    "sources": [],
    "verification": [],
    "unresolved_gaps": []
  }
}
```
