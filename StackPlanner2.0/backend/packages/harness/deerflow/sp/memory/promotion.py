"""Dry-run long-term memory promotion candidates for SP task memory."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from deerflow.sp.memory.entry import StackMemoryEntry
from deerflow.sp.memory.stack import TaskMemoryStack

MemoryCandidateKind = Literal["user_preference", "fact", "correction", "behavior", "sop", "agent_style", "failure_pattern"]
MemoryCandidateScope = Literal["user", "agent", "project", "global"]
PromotionStatus = Literal["approved", "rejected", "written"]

_STABLE_MARKERS = (
    "以后",
    "后续",
    "以后都",
    "以后不要",
    "默认",
    "长期",
    "记住",
    "偏好",
    "总是",
    "不要再",
    "from now on",
    "always",
    "never",
    "prefer",
    "preference",
    "by default",
)
_CORRECTION_MARKERS = ("纠正", "错", "不对", "不要", "改成", "instead", "wrong", "correct", "correction")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    return repr(value)


def _stable_id(kind: str, content: str, source_ids: Iterable[str]) -> str:
    payload = "|".join([kind, content, *source_ids])
    return f"spcand_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _is_stable_feedback(content: str) -> bool:
    lowered = content.lower()
    return any(marker in lowered for marker in _STABLE_MARKERS)


def _feedback_kind(content: str) -> MemoryCandidateKind:
    lowered = content.lower()
    if any(marker in lowered for marker in _CORRECTION_MARKERS):
        return "correction"
    return "user_preference"


def _confidence_from_entry(entry: StackMemoryEntry, base: float) -> float:
    if entry.priority == "critical" or entry.status == "pinned":
        return min(0.98, base + 0.08)
    return base


@dataclass(slots=True)
class MemoryCandidate:
    """A candidate for DR2 long-term memory promotion.

    Candidates are dry-run by default. Writing to DR2 memory remains a later,
    explicitly gated step through DR2's existing MemoryUpdater/storage path.
    """

    candidate_id: str
    kind: MemoryCandidateKind
    content: str
    source_entry_ids: list[str] = field(default_factory=list)
    source_artifact_ids: list[str] = field(default_factory=list)
    source_event_ids: list[str] = field(default_factory=list)
    scope: MemoryCandidateScope = "user"
    confidence: float = 0.7
    risk: str = "medium"
    dry_run: bool = True
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "kind": self.kind,
            "content": self.content,
            "source_entry_ids": list(self.source_entry_ids),
            "source_artifact_ids": list(self.source_artifact_ids),
            "source_event_ids": list(self.source_event_ids),
            "scope": self.scope,
            "confidence": self.confidence,
            "risk": self.risk,
            "dry_run": self.dry_run,
            "reason": self.reason,
            "metadata": _json_safe(self.metadata),
        }


@dataclass(slots=True)
class MemoryPromotionDecision:
    """Code-gated decision for a long-term memory promotion candidate."""

    candidate: MemoryCandidate
    status: PromotionStatus
    reason: str
    dry_run: bool = True
    write_result: Any | None = None

    @property
    def approved(self) -> bool:
        return self.status in {"approved", "written"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "status": self.status,
            "reason": self.reason,
            "dry_run": self.dry_run,
            "write_result": _json_safe(self.write_result),
        }


@dataclass(slots=True)
class MemoryPromotionResult:
    """Dry-run promotion hook result."""

    decisions: list[MemoryPromotionDecision] = field(default_factory=list)

    @property
    def approved_count(self) -> int:
        return sum(1 for decision in self.decisions if decision.approved)

    @property
    def written_count(self) -> int:
        return sum(1 for decision in self.decisions if decision.status == "written")

    def to_dict(self) -> dict[str, Any]:
        return {
            "decisions": [decision.to_dict() for decision in self.decisions],
            "approved_count": self.approved_count,
            "written_count": self.written_count,
        }


class MemoryPromotionJudge:
    """Deterministic gate before any SP signal may become DR2 long-term memory."""

    def __init__(self, *, min_confidence: float = 0.72):
        self._min_confidence = min_confidence

    def judge(self, candidate: MemoryCandidate) -> MemoryPromotionDecision:
        content = candidate.content.strip()
        if not content:
            return self._reject(candidate, "Candidate content is empty.")

        if candidate.confidence < self._min_confidence:
            return self._reject(candidate, "Candidate confidence is below the promotion threshold.")

        if candidate.kind in {"user_preference", "correction"} and not _is_stable_feedback(content):
            return self._reject(candidate, "Human feedback is temporary; it lacks stable future/default/preference markers.")

        if candidate.kind == "fact" and not candidate.source_entry_ids and not candidate.source_artifact_ids and not candidate.source_event_ids:
            return self._reject(candidate, "Fact candidate has no SP provenance.")

        if self._looks_like_large_report(content) and not self._has_explicit_long_term_signal(candidate):
            return self._reject(candidate, "Large report-like content is not promoted by default.")

        if candidate.kind == "correction" and candidate.scope == "global":
            return self._reject(candidate, "Corrections require a narrower user, agent, or project scope.")

        return MemoryPromotionDecision(
            candidate=candidate,
            status="approved",
            reason="Candidate passed deterministic long-term memory promotion gates.",
            dry_run=candidate.dry_run,
        )

    @staticmethod
    def _reject(candidate: MemoryCandidate, reason: str) -> MemoryPromotionDecision:
        return MemoryPromotionDecision(candidate=candidate, status="rejected", reason=reason, dry_run=True)

    @staticmethod
    def _looks_like_large_report(content: str) -> bool:
        lowered = content.lower()
        if len(content) > 2000:
            return True
        report_markers = ("# ", "executive summary", "final report", "研究报告", "调研报告", "报告正文")
        return any(marker in lowered for marker in report_markers)

    @staticmethod
    def _has_explicit_long_term_signal(candidate: MemoryCandidate) -> bool:
        if candidate.kind in {"user_preference", "correction"} and _is_stable_feedback(candidate.content):
            return True
        return bool(candidate.metadata.get("promotion_candidate") or candidate.metadata.get("explicit_long_term"))


class MemoryPromotionHook:
    """Default dry-run bridge from SP promotion candidates to DR2 memory APIs.

    The hook never owns storage. Production wiring can inject a writer that uses
    DeerFlow's existing MemoryUpdater/client APIs, but writes remain disabled
    unless ``dry_run`` is explicitly set to False and each candidate is also
    non-dry-run.
    """

    def __init__(
        self,
        *,
        judge: MemoryPromotionJudge | None = None,
        writer: Callable[[MemoryCandidate], Any] | None = None,
        dry_run: bool = True,
    ) -> None:
        self._judge = judge or MemoryPromotionJudge()
        self._writer = writer
        self._dry_run = dry_run

    def promote(self, candidates: Iterable[MemoryCandidate]) -> MemoryPromotionResult:
        decisions: list[MemoryPromotionDecision] = []
        for candidate in candidates:
            decision = self._judge.judge(candidate)
            if not decision.approved:
                decisions.append(decision)
                continue
            if self._dry_run or candidate.dry_run or self._writer is None:
                decisions.append(decision)
                continue

            write_result = self._writer(candidate)
            decisions.append(
                MemoryPromotionDecision(
                    candidate=candidate,
                    status="written",
                    reason="Candidate was written through the injected DR2 memory writer.",
                    dry_run=False,
                    write_result=write_result,
                )
            )
        return MemoryPromotionResult(decisions=decisions)


class MemoryCandidateExtractor:
    """Extract dry-run long-term memory candidates from SP runtime signals."""

    def __init__(self, *, max_candidates: int = 20):
        self._max_candidates = max_candidates

    def extract(
        self,
        stack: TaskMemoryStack,
        *,
        artifact_refs: Mapping[str, Any] | None = None,
        event_refs: Iterable[Mapping[str, Any]] | None = None,
    ) -> list[MemoryCandidate]:
        candidates: list[MemoryCandidate] = []
        candidates.extend(self._from_stack_entries(stack.entries))
        candidates.extend(self._from_artifact_refs(artifact_refs or {}))
        candidates.extend(self._from_event_refs(event_refs or []))
        return self._dedupe(candidates)[: self._max_candidates]

    def _from_stack_entries(self, entries: Iterable[StackMemoryEntry]) -> list[MemoryCandidate]:
        candidates: list[MemoryCandidate] = []
        for entry in entries:
            if entry.action == "feedback" and _is_stable_feedback(entry.content):
                kind = _feedback_kind(entry.content)
                candidates.append(
                    MemoryCandidate(
                        candidate_id=_stable_id(kind, entry.content, [entry.id]),
                        kind=kind,
                        content=entry.content,
                        source_entry_ids=[entry.id],
                        scope="user",
                        confidence=_confidence_from_entry(entry, 0.88),
                        risk="medium" if kind == "user_preference" else "high",
                        reason="Stable human feedback contains future/default/preference markers.",
                        metadata={"entry_action": entry.action, "entry_stage": entry.stage},
                    )
                )
                continue

            if entry.action == "reflect" and (entry.promotion_candidate or entry.failure_note):
                content = entry.failure_note or entry.content
                candidates.append(
                    MemoryCandidate(
                        candidate_id=_stable_id("failure_pattern", content, [entry.id]),
                        kind="failure_pattern",
                        content=content,
                        source_entry_ids=[entry.id],
                        scope="project",
                        confidence=_confidence_from_entry(entry, 0.76),
                        risk="high",
                        reason="Reflect entry captured a reusable failure or recovery note.",
                        metadata={"entry_stage": entry.stage},
                    )
                )
                continue

            if entry.action == "finish" and entry.promotion_candidate:
                candidates.append(
                    MemoryCandidate(
                        candidate_id=_stable_id("behavior", entry.content, [entry.id]),
                        kind="behavior",
                        content=entry.content,
                        source_entry_ids=[entry.id],
                        scope="agent",
                        confidence=0.74,
                        risk="medium",
                        reason="Finish entry was explicitly marked as reusable behavior.",
                        metadata={"entry_stage": entry.stage},
                    )
                )
        return candidates

    def _from_artifact_refs(self, artifact_refs: Mapping[str, Any]) -> list[MemoryCandidate]:
        candidates: list[MemoryCandidate] = []
        history = artifact_refs.get("_history")
        refs = history if isinstance(history, list) else [value for key, value in artifact_refs.items() if key != "_history"]
        for ref in refs:
            if not isinstance(ref, Mapping):
                continue
            metadata = ref.get("metadata") if isinstance(ref.get("metadata"), Mapping) else {}
            if not metadata.get("promotion_candidate"):
                continue
            artifact_id = str(ref.get("artifact_id") or "")
            summary = str(ref.get("summary") or "")
            if not artifact_id or not summary:
                continue
            kind = str(metadata.get("candidate_kind") or "sop")
            if kind not in MemoryCandidateKind.__args__:
                kind = "sop"
            scope = str(metadata.get("scope") or "project")
            if scope not in MemoryCandidateScope.__args__:
                scope = "project"
            candidates.append(
                MemoryCandidate(
                    candidate_id=_stable_id(kind, summary, [artifact_id]),
                    kind=kind,  # type: ignore[arg-type]
                    content=summary,
                    source_artifact_ids=[artifact_id],
                    scope=scope,  # type: ignore[arg-type]
                    confidence=float(metadata.get("confidence") or 0.7),
                    risk=str(metadata.get("risk") or "high"),
                    reason="Artifact metadata explicitly marked this ref as a promotion candidate.",
                    metadata={"artifact_type": ref.get("type"), "artifact_version": ref.get("version")},
                )
            )
        return candidates

    def _from_event_refs(self, event_refs: Iterable[Mapping[str, Any]]) -> list[MemoryCandidate]:
        candidates: list[MemoryCandidate] = []
        for event in event_refs:
            if not event.get("promotion_candidate"):
                continue
            event_id = str(event.get("event_id") or event.get("id") or "")
            content = str(event.get("summary") or event.get("content") or "")
            if not event_id or not content:
                continue
            candidates.append(
                MemoryCandidate(
                    candidate_id=_stable_id("failure_pattern", content, [event_id]),
                    kind="failure_pattern",
                    content=content,
                    source_event_ids=[event_id],
                    scope="project",
                    confidence=float(event.get("confidence") or 0.7),
                    risk=str(event.get("risk") or "high"),
                    reason="Run/Event signal explicitly marked this pattern as reusable.",
                    metadata={"event_type": event.get("type")},
                )
            )
        return candidates

    @staticmethod
    def _dedupe(candidates: list[MemoryCandidate]) -> list[MemoryCandidate]:
        by_semantic_key: dict[tuple[str, str, str], MemoryCandidate] = {}
        order: list[tuple[str, str, str]] = []
        for candidate in candidates:
            normalized_content = " ".join(candidate.content.casefold().split())
            semantic_key = (candidate.kind, candidate.scope, normalized_content)
            existing = by_semantic_key.get(semantic_key)
            if existing is None:
                order.append(semantic_key)
                by_semantic_key[semantic_key] = candidate
                continue

            existing.source_entry_ids = list(dict.fromkeys([*existing.source_entry_ids, *candidate.source_entry_ids]))
            existing.source_artifact_ids = list(dict.fromkeys([*existing.source_artifact_ids, *candidate.source_artifact_ids]))
            existing.source_event_ids = list(dict.fromkeys([*existing.source_event_ids, *candidate.source_event_ids]))
            existing.confidence = max(existing.confidence, candidate.confidence)
            existing.metadata = {
                **existing.metadata,
                **candidate.metadata,
                "duplicate_signal_count": int(existing.metadata.get("duplicate_signal_count", 1)) + 1,
            }
            existing.candidate_id = _stable_id(
                existing.kind,
                existing.content,
                [*existing.source_entry_ids, *existing.source_artifact_ids, *existing.source_event_ids],
            )
        return [by_semantic_key[key] for key in order]
