"""Tests for SP dry-run long-term memory promotion candidates."""

from deerflow.sp.memory import MemoryCandidate, MemoryCandidateExtractor, MemoryPromotionHook, MemoryPromotionJudge, StackMemoryEntry, TaskMemoryStack


def test_temporary_human_feedback_is_not_promoted():
    stack = TaskMemoryStack()
    stack.append_feedback("这次报告先写宏观部分")

    candidates = MemoryCandidateExtractor().extract(stack)

    assert candidates == []


def test_temporary_feedback_candidate_is_rejected_by_judge():
    candidate = MemoryCandidate(
        candidate_id="spcand_temp",
        kind="user_preference",
        content="这次报告先写宏观部分",
        source_entry_ids=["entry-1"],
        confidence=0.95,
    )

    decision = MemoryPromotionJudge().judge(candidate)

    assert decision.status == "rejected"
    assert "temporary" in decision.reason


def test_stable_feedback_becomes_dry_run_preference_candidate():
    stack = TaskMemoryStack()
    feedback = stack.append_feedback("以后报告默认先写宏观产业链，再写公司案例")

    candidates = MemoryCandidateExtractor().extract(stack)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.kind == "user_preference"
    assert candidate.scope == "user"
    assert candidate.dry_run is True
    assert candidate.source_entry_ids == [feedback.id]
    assert candidate.confidence >= 0.9


def test_stable_correction_feedback_becomes_high_risk_candidate():
    stack = TaskMemoryStack()
    stack.append_feedback("以后不要把大文本放进 ThreadState，改成 artifact refs")

    candidate = MemoryCandidateExtractor().extract(stack)[0]

    assert candidate.kind == "correction"
    assert candidate.risk == "high"
    assert candidate.dry_run is True


def test_reflect_failure_note_can_be_extracted_as_failure_pattern():
    stack = TaskMemoryStack()
    entry = StackMemoryEntry(
        action="reflect",
        content="Reporter used stale evidence",
        failure_note="When evidence is stale, recall memory and re-run researcher before reporting",
        promotion_candidate=True,
    )
    stack.append(entry)

    candidates = MemoryCandidateExtractor().extract(stack)

    assert len(candidates) == 1
    assert candidates[0].kind == "failure_pattern"
    assert candidates[0].scope == "project"
    assert candidates[0].source_entry_ids == [entry.id]
    assert candidates[0].dry_run is True


def test_artifact_refs_only_promote_summary_not_workspace_path():
    artifact_refs = {
        "report": {
            "artifact_id": "spart_report",
            "type": "report",
            "version": 2,
            "summary": "Reusable report structure: macro, evidence, risks, action plan",
            "workspace_path": "/tmp/private/report.md",
            "metadata": {"promotion_candidate": True, "candidate_kind": "sop", "scope": "project"},
        }
    }

    candidate = MemoryCandidateExtractor().extract(TaskMemoryStack(), artifact_refs=artifact_refs)[0]
    payload = candidate.to_dict()

    assert candidate.kind == "sop"
    assert candidate.source_artifact_ids == ["spart_report"]
    assert "/tmp/private" not in str(payload)
    assert "Reusable report structure" in candidate.content


def test_all_candidates_default_to_dry_run():
    stack = TaskMemoryStack()
    stack.append_feedback("以后默认用中文解释迁移风险")
    stack.append(StackMemoryEntry(action="reflect", content="failure", failure_note="Reusable failure", promotion_candidate=True))

    candidates = MemoryCandidateExtractor().extract(
        stack,
        event_refs=[{"event_id": "evt-1", "promotion_candidate": True, "summary": "Repeated delegate timeout"}],
    )

    assert candidates
    assert all(candidate.dry_run for candidate in candidates)


def test_repeated_stable_feedback_merges_provenance_into_one_candidate():
    stack = TaskMemoryStack()
    first = stack.append_feedback("以后默认用中文解释迁移风险")
    second = stack.append_feedback("  以后默认用中文解释迁移风险  ")

    candidates = MemoryCandidateExtractor().extract(stack)

    assert len(candidates) == 1
    assert candidates[0].source_entry_ids == [first.id, second.id]
    assert candidates[0].metadata["duplicate_signal_count"] == 2


def test_promotion_hook_is_dry_run_by_default_and_does_not_write():
    stack = TaskMemoryStack()
    stack.append_feedback("以后默认用中文解释迁移风险")
    candidate = MemoryCandidateExtractor().extract(stack)[0]
    writes = []

    result = MemoryPromotionHook(writer=writes.append).promote([candidate])

    assert result.approved_count == 1
    assert result.written_count == 0
    assert writes == []
    assert result.decisions[0].dry_run is True


def test_promotion_hook_requires_explicit_non_dry_run_to_write():
    candidate = MemoryCandidate(
        candidate_id="spcand_write",
        kind="user_preference",
        content="以后默认用中文解释迁移风险",
        source_entry_ids=["entry-1"],
        confidence=0.96,
        dry_run=False,
    )
    writes = []

    result = MemoryPromotionHook(writer=writes.append, dry_run=False).promote([candidate])

    assert result.written_count == 1
    assert writes == [candidate]


def test_report_like_content_is_rejected_without_explicit_long_term_signal():
    candidate = MemoryCandidate(
        candidate_id="spcand_report",
        kind="fact",
        content="# 调研报告\n\n这是一整篇报告正文，不应该默认写入长期记忆。",
        source_artifact_ids=["report-v1"],
        confidence=0.93,
    )

    decision = MemoryPromotionJudge().judge(candidate)

    assert decision.status == "rejected"
    assert "report-like" in decision.reason
