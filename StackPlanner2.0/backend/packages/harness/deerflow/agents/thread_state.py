from collections.abc import Mapping
from typing import Annotated, Any, NotRequired, TypedDict

from langchain.agents import AgentState

from deerflow.agents.goal_state import GoalState
from deerflow.subagents.status_contract import SUBAGENT_STATUS_VALUES


class SandboxState(TypedDict):
    sandbox_id: NotRequired[str | None]


class ThreadDataState(TypedDict):
    workspace_path: NotRequired[str | None]
    uploads_path: NotRequired[str | None]
    outputs_path: NotRequired[str | None]


class ViewedImageData(TypedDict):
    base64: str
    mime_type: str


def merge_sandbox(existing: SandboxState | None, new: SandboxState | None) -> SandboxState | None:
    """Reducer for sandbox state - accepts idempotent writes only.

    Multiple sandbox tools can initialize lazily in the same graph step and
    emit the same sandbox_id via Command(update=...). LangGraph needs an
    explicit reducer for that shared state key. Different sandbox ids in the
    same thread indicate a lifecycle/isolation bug, so fail closed instead of
    choosing one silently.
    """
    if new is None:
        return existing
    if existing is None:
        return new

    existing_id = existing.get("sandbox_id")
    new_id = new.get("sandbox_id")
    if existing_id == new_id:
        return existing
    raise ValueError(f"Conflicting sandbox state updates: {existing_id!r} != {new_id!r}")


SandboxStateField = Annotated[NotRequired[SandboxState | None], merge_sandbox]


def merge_artifacts(existing: list[str] | None, new: list[str] | None) -> list[str]:
    """Reducer for artifacts list - merges and deduplicates artifacts."""
    if existing is None:
        return new or []
    if new is None:
        return existing
    # Use dict.fromkeys to deduplicate while preserving order
    return list(dict.fromkeys(existing + new))


def merge_viewed_images(existing: dict[str, ViewedImageData] | None, new: dict[str, ViewedImageData] | None) -> dict[str, ViewedImageData]:
    """Reducer for viewed_images dict - merges image dictionaries.

    Special case: If new is an empty dict {}, it clears the existing images.
    This allows middlewares to clear the viewed_images state after processing.
    """
    if existing is None:
        return new or {}
    if new is None:
        return existing
    # Special case: empty dict means clear all viewed images
    if len(new) == 0:
        return {}
    # Merge dictionaries, new values override existing ones for same keys
    return {**existing, **new}


def merge_todos(existing: list | None, new: list | None) -> list | None:
    """Reducer for todos list - keeps the last non-None value.

    Semantics:
    - If `new` is None (node didn't touch todos), preserve `existing`.
    - If `new` is provided (even empty list), it represents an explicit
      update and wins over `existing`.
    """
    if new is None:
        return existing
    return new


def merge_goal(existing: GoalState | None, new: GoalState | None) -> GoalState | None:
    """Reducer for goal state - preserves existing when a node does not touch it."""
    if new is None:
        return existing
    return new


SPTaskMemoryState = dict[str, Any]
SPArtifactRefs = dict[str, Any]
SPHumanInteraction = dict[str, Any]
MAX_CONSUMED_TOOL_OBSERVATION_IDS = 2000
MAX_SP_ARTIFACT_HISTORY = 50
MAX_SP_IDEMPOTENCY_LEDGER_ENTRIES = 100
SP_CONCURRENT_MERGE_MODE_KEY = "_sp_concurrent_merge"


def merge_sp_task_memory(existing: SPTaskMemoryState | None, new: SPTaskMemoryState | None) -> SPTaskMemoryState | None:
    """Merge parallel SP action stacks without breaking replacement semantics.

    Ordinary middleware/graph updates replace the serialized stack exactly,
    which is required for fresh-turn filtering and explicit resets. SP control
    actions add ``SP_CONCURRENT_MERGE_MODE_KEY`` because several tool calls from
    one model turn all start from the same checkpoint. Those stale-base
    snapshots are merged by entry id so completed parallel delegations do not
    overwrite each other's observations.

    Summary entries remain true pop operations: their ``parent_ids`` act as
    tombstones during a concurrent merge, preventing a sibling's stale snapshot
    from reintroducing entries that SUMMARIZE just condensed.
    """
    if new is None:
        return existing
    if not isinstance(new, dict):
        return new
    merge_concurrently = bool(new.get(SP_CONCURRENT_MERGE_MODE_KEY))
    normalized_new = {key: value for key, value in new.items() if key != SP_CONCURRENT_MERGE_MODE_KEY}
    if not merge_concurrently or existing is None or not isinstance(existing, dict):
        return normalized_new
    if normalized_new == {}:
        return {}

    existing_entries = existing.get("entries")
    new_entries = normalized_new.get("entries")
    if not isinstance(existing_entries, list) or not isinstance(new_entries, list):
        return normalized_new

    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    anonymous: list[Any] = []
    for raw_entry in [*existing_entries, *new_entries]:
        if not isinstance(raw_entry, dict) or not raw_entry.get("id"):
            anonymous.append(raw_entry)
            continue
        entry_id = str(raw_entry["id"])
        if entry_id not in by_id:
            order.append(entry_id)
        by_id[entry_id] = dict(raw_entry)

    # Re-apply every retained summary tombstone after unioning branches. This
    # makes the result independent of LangGraph's branch-reducer order.
    condensed_ids: set[str] = set()
    for entry in by_id.values():
        if entry.get("action") != "summarize":
            continue
        parent_ids = entry.get("parent_ids")
        if isinstance(parent_ids, list):
            condensed_ids.update(str(parent_id) for parent_id in parent_ids)
    for entry_id in condensed_ids:
        entry = by_id.get(entry_id)
        if entry is None:
            continue
        if entry.get("status") == "pinned" or entry.get("priority") == "critical":
            continue
        by_id.pop(entry_id, None)
        if entry_id in order:
            order.remove(entry_id)

    return {
        **existing,
        **normalized_new,
        "entries": [by_id[entry_id] for entry_id in order if entry_id in by_id] + anonymous,
    }


def _sp_artifact_family(artifact_type: str) -> str:
    if artifact_type in {"report", "report_revision", "final_report"}:
        return "report"
    return artifact_type


def _sp_artifact_identity(ref: Mapping[str, Any]) -> str:
    artifact_id = ref.get("artifact_id")
    if artifact_id:
        return f"id:{artifact_id}"
    return "|".join(
        [
            str(ref.get("type") or ""),
            str(ref.get("version") or ""),
            str(ref.get("virtual_path") or ""),
        ]
    )


def _sp_int(value: Any, *, default: int = 0) -> int:
    """Parse persisted numeric SP metadata without crashing state reduction."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def merge_sp_artifact_refs(existing: SPArtifactRefs | None, new: SPArtifactRefs | None) -> SPArtifactRefs | None:
    """Merge artifact refs produced by sibling delegations.

    Updates without the internal merge marker remain exact replacements so
    BACKTRACK and fresh-turn resets can intentionally restore an older current
    artifact. Parallel DELEGATE updates retain every artifact in ``_history``
    and select one deterministic current ref per version family.
    """
    if new is None:
        return None
    if not isinstance(new, dict):
        return new
    merge_concurrently = bool(new.get(SP_CONCURRENT_MERGE_MODE_KEY))
    normalized_new = {key: value for key, value in new.items() if key != SP_CONCURRENT_MERGE_MODE_KEY}
    if not merge_concurrently or existing is None or not isinstance(existing, dict):
        return normalized_new

    refs_by_identity: dict[str, dict[str, Any]] = {}
    passthrough: dict[str, Any] = {}

    def collect(value: Mapping[str, Any]) -> None:
        history = value.get("_history")
        candidates = [item for item in history if isinstance(item, Mapping)] if isinstance(history, list) else []
        candidates.extend(item for key, item in value.items() if key not in {"_history", SP_CONCURRENT_MERGE_MODE_KEY} and isinstance(item, Mapping))
        for candidate in candidates:
            normalized = dict(candidate)
            identity = _sp_artifact_identity(normalized)
            previous = refs_by_identity.get(identity)
            refs_by_identity[identity] = {**(previous or {}), **normalized}
        for key, item in value.items():
            if key in {"_history", SP_CONCURRENT_MERGE_MODE_KEY} or isinstance(item, Mapping):
                continue
            passthrough[str(key)] = item

    collect(existing)
    collect(normalized_new)

    refs = list(refs_by_identity.values())
    current_by_family: dict[str, dict[str, Any]] = {}
    for ref in refs:
        artifact_type = str(ref.get("type") or "generated_file")
        family = _sp_artifact_family(artifact_type)
        candidate_rank = (
            _sp_int(ref.get("version")),
            str(ref.get("artifact_id") or ""),
            str(ref.get("virtual_path") or ""),
        )
        current = current_by_family.get(family)
        if current is None:
            current_by_family[family] = ref
            continue
        current_rank = (
            _sp_int(current.get("version")),
            str(current.get("artifact_id") or ""),
            str(current.get("virtual_path") or ""),
        )
        if candidate_rank > current_rank:
            current_by_family[family] = ref

    for ref in refs:
        artifact_type = str(ref.get("type") or "generated_file")
        ref["is_current"] = current_by_family.get(_sp_artifact_family(artifact_type)) is ref

    refs.sort(
        key=lambda ref: (
            _sp_artifact_family(str(ref.get("type") or "generated_file")),
            _sp_int(ref.get("version")),
            str(ref.get("artifact_id") or ""),
        )
    )
    merged: dict[str, Any] = dict(passthrough)
    for ref in refs:
        artifact_type = str(ref.get("type") or "generated_file")
        previous = merged.get(artifact_type)
        if not isinstance(previous, Mapping) or (
            _sp_int(ref.get("version")),
            bool(ref.get("is_current")),
            str(ref.get("artifact_id") or ""),
        ) >= (
            _sp_int(previous.get("version")),
            bool(previous.get("is_current")),
            str(previous.get("artifact_id") or ""),
        ):
            merged[artifact_type] = ref
    merged["_history"] = refs[-MAX_SP_ARTIFACT_HISTORY:]
    return merged


def merge_sp_idempotency_ledger(
    existing: dict[str, dict[str, Any]] | None,
    new: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]] | None:
    """Union idempotency records emitted by parallel SP actions."""
    if new is None:
        return None
    if not isinstance(new, dict):
        return new
    merged = {**(existing or {}), **new}
    return dict(list(merged.items())[-MAX_SP_IDEMPOTENCY_LEDGER_ENTRIES:])


def merge_sp_last_value(existing: Any, new: Any) -> Any:
    """Permit concurrent writes while retaining normal last-value semantics."""
    del existing
    return new


def merge_sp_counter(existing: int | None, new: int | None) -> int | None:
    """Keep the furthest counter value, while allowing an explicit reset to 0."""
    if new is None:
        return None
    if int(new) == 0:
        return 0
    if existing is None:
        return int(new)
    return max(int(existing), int(new))


def merge_sp_report_version(existing: str | None, new: str | None) -> str | None:
    """Keep the highest report version across sibling artifact writers."""
    if new is None:
        return None
    if existing is None:
        return str(new)
    try:
        return str(max(int(existing), int(new)))
    except (TypeError, ValueError):
        return str(new)


_HANDLER_STEP_RANK = {
    # FINISH is valid only when every sibling action is also terminal. If a
    # model emits FINISH beside a DELEGATE/THINK in one tool batch, the
    # non-terminal action must win so Central can inspect the new result.
    "finish": -1,
    "continue": 0,
    "error_recoverable": 2,
    "interrupt": 3,
    "error_fatal": 4,
}


def merge_sp_handler_result(existing: dict[str, Any] | None, new: dict[str, Any] | None) -> dict[str, Any] | None:
    """Combine results from multiple actions in the same CentralAgent turn."""
    if new is None:
        return None
    if existing is None or not isinstance(existing, dict):
        return new
    if not isinstance(new, dict):
        return new
    if "loop_iteration" not in existing or "loop_iteration" not in new:
        # Legacy/native graph nodes use ordinary sequential last-value updates
        # and do not attach the action-loop marker used by control-tool
        # branches. Preserve that exact replacement behavior.
        return new

    existing_iteration = _sp_int(existing.get("loop_iteration"), default=-1)
    new_iteration = _sp_int(new.get("loop_iteration"), default=-1)
    if new_iteration > existing_iteration:
        return new
    if new_iteration < existing_iteration:
        return existing

    existing_run = existing.get("run_id")
    new_run = new.get("run_id")
    if existing_run != new_run or existing.get("action_id") == new.get("action_id"):
        return new

    results: list[dict[str, Any]] = []
    for value in (existing, new):
        parallel = value.get("parallel_action_results")
        if isinstance(parallel, list):
            results.extend(dict(item) for item in parallel if isinstance(item, Mapping))
        else:
            results.append(dict(value))
    deduped: dict[str, dict[str, Any]] = {}
    for result in results:
        action_id = str(result.get("action_id") or len(deduped))
        deduped[action_id] = result
    ordered_results = [deduped[key] for key in sorted(deduped)]
    worst = max(
        ordered_results,
        key=lambda result: _HANDLER_STEP_RANK.get(str(result.get("next_step") or "continue"), 0),
    )
    errors = list(dict.fromkeys(str(result["error"]) for result in ordered_results if result.get("error")))
    return {
        **worst,
        "action_type": "BATCH",
        "action_id": ",".join(str(result.get("action_id") or "") for result in ordered_results),
        "parallel_action_results": ordered_results,
        "parallel_action_count": len(ordered_results),
        "error": "; ".join(errors) if errors else None,
    }


def merge_consumed_tool_observation_ids(existing: list[str] | None, new: list[str] | None) -> list[str]:
    """Persist tool-message tombstones even after their memory entries are condensed."""
    merged = list(dict.fromkeys([*(existing or []), *(new or [])]))
    return merged[-MAX_CONSUMED_TOOL_OBSERVATION_IDS:]


ConsumedToolObservationIdsField = Annotated[NotRequired[list[str]], merge_consumed_tool_observation_ids]


class PromotedTools(TypedDict):
    catalog_hash: str
    names: list[str]


def merge_promoted(existing: PromotedTools | None, new: PromotedTools | None) -> PromotedTools | None:
    """Reducer for deferred-tool promotions, scoped by catalog hash.

    - new None/empty -> preserve existing (node didn't touch promotions).
    - catalog_hash changed -> replace wholesale, dropping stale names (prevents a
      persisted bare name from exposing a different tool after catalog drift).
    - same catalog_hash -> union names, dedupe, preserve order.
    """
    if not new:
        return existing
    if existing is None or existing.get("catalog_hash") != new["catalog_hash"]:
        return {
            "catalog_hash": new["catalog_hash"],
            "names": list(dict.fromkeys(new["names"])),
        }
    return {
        "catalog_hash": existing["catalog_hash"],
        "names": list(dict.fromkeys(existing["names"] + new["names"])),
    }


TERMINAL_STATUSES: frozenset[str] = frozenset(SUBAGENT_STATUS_VALUES)
_DELEGATION_LEDGER_MAX_ENTRIES = 50


class DelegationEntry(TypedDict):
    id: str
    description: str
    subagent_type: str
    status: str
    result_brief: NotRequired[str]
    result_sha256: NotRequired[str]
    result_ref: NotRequired[str]
    # Why a guardrail cap ended the run early (#3875 Phase 2): token_capped /
    # turn_capped / loop_capped. The status stays completed/failed; this field
    # is the additive signal that distinguishes a capped run from a clean one.
    stop_reason: NotRequired[str]
    created_at: str


def merge_delegations(existing: list[DelegationEntry] | None, new: list[DelegationEntry] | None) -> list[DelegationEntry]:
    """Reducer for the delegation ledger.

    - new None/empty -> preserve existing.
    - append entries, replacing same id with the latest version while preserving
      first-seen order.
    - terminal status is never overwritten by a non-terminal status.
    """
    if not new:
        return existing or []

    by_id: dict[str, DelegationEntry] = {}
    order: list[str] = []
    for entry in [*(existing or []), *new]:
        entry_id = entry["id"]
        previous = by_id.get(entry_id)
        if previous is not None and previous["status"] in TERMINAL_STATUSES and entry["status"] not in TERMINAL_STATUSES:
            continue
        if entry_id not in by_id:
            order.append(entry_id)
        elif previous.get("created_at"):
            entry = {**entry, "created_at": previous["created_at"]}
        by_id[entry_id] = entry
    merged = [by_id[entry_id] for entry_id in order]
    if len(merged) > _DELEGATION_LEDGER_MAX_ENTRIES:
        merged = merged[-_DELEGATION_LEDGER_MAX_ENTRIES:]
    return merged


_SKILL_CONTEXT_MAX_ENTRIES = 8
_SKILL_DESCRIPTION_MAX_CHARS = 500


class SkillEntry(TypedDict):
    name: str
    path: str
    description: str
    loaded_at: int


def _normalize_skill_entry(entry: Mapping[str, object]) -> SkillEntry:
    """Drop legacy payload keys before storing skill_context back to state."""
    description = entry.get("description")
    loaded_at = entry.get("loaded_at")
    return {
        "name": str(entry.get("name") or ""),
        "path": str(entry["path"]),
        "description": " ".join(description.split())[:_SKILL_DESCRIPTION_MAX_CHARS] if isinstance(description, str) else "",
        "loaded_at": loaded_at if isinstance(loaded_at, int) else 0,
    }


def merge_skill_context(existing: list[SkillEntry] | None, new: list[SkillEntry] | None) -> list[SkillEntry]:
    """Reducer for the skill-context channel.

    - new None/empty -> preserve existing.
    - legacy entries are normalized to references; verbatim body keys are dropped.
    - dedup by ``path``; later reads refresh recency and replace the reference.
    - cap by keeping the most recently read entries. ``loaded_at`` is
      observational only because message indices reset after compaction.
    """
    normalized_existing = [_normalize_skill_entry(entry) for entry in existing or []]
    if not new:
        return normalized_existing

    by_path: dict[str, SkillEntry] = {}
    order: list[str] = []
    for entry in normalized_existing:
        path = entry["path"]
        if path not in by_path:
            order.append(path)
        by_path[path] = entry

    for entry in (_normalize_skill_entry(entry) for entry in new):
        path = entry["path"]
        if path in by_path:
            order.remove(path)
        order.append(path)
        by_path[path] = entry

    merged = [by_path[path] for path in order]
    if len(merged) > _SKILL_CONTEXT_MAX_ENTRIES:
        merged = merged[-_SKILL_CONTEXT_MAX_ENTRIES:]
    return merged


class ThreadState(AgentState):
    sandbox: SandboxStateField
    thread_data: NotRequired[ThreadDataState | None]
    title: NotRequired[str | None]
    artifacts: Annotated[list[str], merge_artifacts]
    todos: Annotated[list | None, merge_todos]
    goal: Annotated[GoalState | None, merge_goal]
    uploaded_files: NotRequired[list[dict] | None]
    viewed_images: Annotated[dict[str, ViewedImageData], merge_viewed_images]  # image_path -> {base64, mime_type}
    promoted: Annotated[PromotedTools | None, merge_promoted]
    delegations: Annotated[list[DelegationEntry], merge_delegations]
    skill_context: Annotated[list[SkillEntry], merge_skill_context]
    summary_text: NotRequired[str | None]
    sp_task_memory: Annotated[SPTaskMemoryState | None, merge_sp_task_memory]
    sp_consumed_tool_observation_ids: ConsumedToolObservationIdsField
    sp_current_stage: Annotated[NotRequired[str | None], merge_sp_last_value]
    sp_active_delegate_id: Annotated[NotRequired[str | None], merge_sp_last_value]
    sp_pending_human_interaction: Annotated[NotRequired[SPHumanInteraction | None], merge_sp_last_value]
    sp_current_artifact_refs: Annotated[NotRequired[SPArtifactRefs | None], merge_sp_artifact_refs]
    sp_current_report_version: Annotated[NotRequired[str | None], merge_sp_report_version]
    sp_last_run_summary: Annotated[NotRequired[str | None], merge_sp_last_value]
    sp_summarize_committed_run_id: Annotated[NotRequired[str | None], merge_sp_last_value]
    sp_last_final_artifact_ref: Annotated[NotRequired[str | None], merge_sp_last_value]
    sp_current_action_id: Annotated[NotRequired[str | None], merge_sp_last_value]
    sp_current_action: Annotated[NotRequired[dict[str, Any] | None], merge_sp_last_value]
    sp_last_action_id: Annotated[NotRequired[str | None], merge_sp_last_value]
    sp_last_idempotency_key: Annotated[NotRequired[str | None], merge_sp_last_value]
    sp_idempotency_ledger: Annotated[NotRequired[dict[str, dict[str, Any]] | None], merge_sp_idempotency_ledger]
    sp_last_handler_result: Annotated[NotRequired[dict[str, Any] | None], merge_sp_handler_result]
    sp_loop_iteration: Annotated[NotRequired[int | None], merge_sp_counter]
    sp_loop_run_id: Annotated[NotRequired[str | None], merge_sp_last_value]
    sp_decision_attempts: Annotated[NotRequired[int | None], merge_sp_counter]
    sp_max_loop_iterations: NotRequired[int | None]
