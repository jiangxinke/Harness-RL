"""Additive evaluation diagnostics; no policy, scoring or budget decisions.

Only fixed reason labels and counts enter the compact terminal summary. Raw
exception messages, model text, tool arguments, source URLs and pages stay out.
Native counts describe retained reference observations, never billed work or an
estimate of tokens the engine might have generated after its last response.
"""

from collections import Counter


_TOOLS = {"search_document", "read_document", "web_search", "fetch_url"}
_INFRASTRUCTURE_REASONS = {
    "episode_deadline_exceeded", "dependency_timeout",
    "dependency_connection_error", "dependency_infrastructure_error",
}
_EVIDENCE_REASONS = {
    "missing_successful_search", "missing_successful_fetch", "invalid_answer_format",
    "empty_answer", "missing_required_columns", "unexpected_answer_columns",
    "empty_row", "missing_row_citations", "unobserved_citation_url", "unread_citation_url",
}
_NATIVE_REASONS = {
    "thought_did_not_stop_on_real_close", "json_did_not_finish_with_real_eos",
    "json_not_complete_or_schema_valid",
} | {
    stage + "_" + reason
    for stage in ("thought", "json")
    for reason in ("no_response", "output_count_mismatch", "prompt_id_mismatch",
                   "nonterminal_response", "invalid_token_ids", "unknown_finish_reason")
}


def _count(value):
    return value if type(value) is int and value >= 0 else None


def _label(value, allowed, default):
    return value if isinstance(value, str) and value in allowed else default


def terminal_summary(result, *, max_steps):
    """Summarize existing outcomes without consulting prompts, answers or gold."""
    status = _label(result.status, {
        "completed", "invalid_answer", "max_steps", "evidence_failure",
        "infrastructure_error", "model_protocol_failure",
    }, "unknown_status")
    summary = {
        "version": "dcc-episode-terminal-summary-v1", "reason": status,
        "model_turn_budget_exhausted": len(result.steps) >= max_steps,
    }
    infrastructure = result.metrics.get("infrastructure_error")
    if status == "infrastructure_error":
        infrastructure = infrastructure if isinstance(infrastructure, dict) else {}
        summary["reason"] = _label(infrastructure.get("reason"), _INFRASTRUCTURE_REASONS,
                                   "infrastructure_error_unclassified")
        operation = infrastructure.get("operation")
        summary["infrastructure"] = {
            "operation": _label(operation, {"initialization", "policy", "auto_read_sources"}
                                | {"tool:" + name for name in _TOOLS}, "other_operation"),
            "turn_index": _count(infrastructure.get("turn_index")),
        }
    if status == "model_protocol_failure":
        failure = result.metrics.get("model_protocol_failure", {})
        failure = failure if isinstance(failure, dict) else {}
        reason = _label(failure.get("reason"), _NATIVE_REASONS, "other_protocol_failure")
        finish = _label(result.steps[-1].finish_reason if result.steps else None,
                        {"stop", "length", "unknown"}, "unknown")
        category = "schema_validation" if reason == "json_not_complete_or_schema_valid" else "protocol"
        if finish == "length" and reason in {
                "thought_did_not_stop_on_real_close", "json_did_not_finish_with_real_eos"}:
            category = "generation_budget"
        summary["model_protocol"] = {"reason": reason, "category": category, "finish_reason": finish}
    evidence = result.metrics.get("evidence_guard")
    if isinstance(evidence, dict):
        rejections = evidence.get("rejections", [])
        rejections = rejections if isinstance(rejections, list) else []
        counts = Counter()
        final_reasons = []
        for rejection in rejections:
            if not isinstance(rejection, dict):
                continue
            reasons = rejection.get("reasons", [])
            if not isinstance(reasons, list):
                continue
            safe_reasons = list(dict.fromkeys(_label(reason, _EVIDENCE_REASONS, "other_evidence_rejection")
                                             for reason in reasons))
            counts.update(safe_reasons)
            if rejection.get("turn") == len(result.steps) - 1:
                final_reasons = safe_reasons
        summary["evidence"] = {
            "successful_searches": _count(evidence.get("successful_searches")),
            "successful_fetches": _count(evidence.get("successful_fetches")),
            "observed_source_count": _count(evidence.get("observed_source_count")),
            "rejection_count": len(rejections), "rejection_counts": dict(sorted(counts.items())),
            "final_rejection_reasons": final_reasons,
        }
        if status == "evidence_failure":
            summary["reason"] = "evidence_rejected" if final_reasons else "model_turn_budget_exhausted"
    if status == "max_steps":
        summary["reason"] = "model_turn_budget_exhausted"
    errors = Counter()
    events = result.tool_events
    for event in events:
        observation = event.get("observation") if isinstance(event, dict) else None
        # An empty TimeoutError string is still a recorded tool error.
        if isinstance(observation, dict) and "error" in observation:
            errors[_label(event.get("tool"), _TOOLS, "other_tool")] += 1
    summary["tools"] = {
        "recorded_event_count": len(events), "observation_error_count": sum(errors.values()),
        "observation_error_counts_by_tool": dict(sorted(errors.items())),
        "scope": "recorded_tool_events_only",
    }
    return summary


def token_accounting(result, *, native_references=None):
    """Keep legacy logical counts; add native phase-input/observed-ID counts.

    A cancelled FINAL_ONLY call may have no PolicyCompletion and zero observed
    IDs while its actual generated total is unknown. Native references retain
    this extra call. Count each trace once. Conservatively report the generated
    total as unknown for non-completed, missing or conflicting references.
    """
    logical = {
        "prompt": sum(len(step.prompt_ids) for step in result.steps),
        "completion": sum(len(step.completion_ids) for step in result.steps),
    }
    result_value = {
        "version": "dcc-episode-token-accounting-v1", "logical": logical,
        "logical_scope": "returned_policy_completions_including_protocol_failures",
        "native": None,
    }
    if native_references is None:
        return result_value
    seen, unique = {}, []
    duplicates = invalid = conflicts = 0
    for ref in native_references:
        if not isinstance(ref, dict) or not isinstance(ref.get("trace_id"), str) or not ref["trace_id"]:
            invalid += 1
            continue
        trace_id = ref["trace_id"]
        if trace_id in seen:
            duplicates += 1
            conflicts += int(ref != seen[trace_id])
            continue
        seen[trace_id] = ref
        unique.append(ref)
    prompts = 0
    observed = Counter()
    statuses = Counter()
    unknown_counts = 0
    for ref in unique:
        status = _label(ref.get("status"), {"completed", "failed", "interrupted", "runtime_error"},
                        "unknown_status")
        statuses[status] += 1
        prompt_count = _count(ref.get("physical_prompt_tokens"))
        stages = ref.get("completion_tokens_by_stage")
        counts_known = prompt_count is not None and isinstance(stages, dict) and bool(stages)
        prompts += prompt_count or 0
        if isinstance(stages, dict):
            for stage, count in stages.items():
                count = _count(count)
                if count is None:
                    counts_known = False
                else:
                    observed[_label(stage, {"thought", "json"}, "other_stage")] += count
        unknown_counts += int(not counts_known)
    # Existing native integrity validation remains authoritative. This flag is
    # diagnostic only and never raises, drops a prediction or changes its score.
    trailing = result.status == "infrastructure_error" and len(unique) == len(result.steps) + 1
    coverage_mismatch = len(unique) != len(result.steps) and not trailing
    complete = bool(unique) and not (invalid or conflicts or unknown_counts or coverage_mismatch)
    complete = complete and all(ref.get("status") == "completed" for ref in unique)
    completion_observed = sum(observed.values())
    result_value["native"] = {
        "physical_prompt_tokens_requested": prompts,
        "physical_prompt_scope": "recorded_phase_inputs_not_billed_or_cache_adjusted_work",
        "completion_tokens_observed": completion_observed,
        "completion_tokens_by_stage_observed": dict(sorted(observed.items())),
        "completion_observation_scope": "retained_phase_token_ids_unobserved_suffix_unknown",
        "completion_tokens_total": completion_observed if complete else None,
        "completion_tokens_unobserved": 0 if complete else None,
        "reference_count": len(unique), "reference_status_counts": dict(sorted(statuses.items())),
        "duplicate_reference_count": duplicates, "conflicting_reference_count": conflicts,
        "invalid_reference_count": invalid, "unknown_count_reference_count": unknown_counts,
        "reference_coverage_mismatch": coverage_mismatch,
    }
    return result_value
