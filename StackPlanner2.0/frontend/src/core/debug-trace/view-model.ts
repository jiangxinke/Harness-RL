import { type DebugTraceFilter, type DebugTraceStep } from "./types";

export function formatTraceDuration(durationMs: number | null): string {
  if (durationMs === null || !Number.isFinite(durationMs)) return "—";
  if (durationMs < 1000) return `${Math.round(durationMs)} ms`;
  if (durationMs < 60_000) {
    const seconds = Math.round(durationMs / 100) / 10;
    return `${seconds.toFixed(1)} s`;
  }
  const minutes = Math.floor(durationMs / 60_000);
  const seconds = Math.round((durationMs % 60_000) / 1000);
  return `${minutes}m ${seconds}s`;
}

export function formatTraceTokens(tokens: number): string {
  if (tokens < 1000) return String(tokens);
  if (tokens < 1_000_000)
    return `${(Math.round(tokens / 100) / 10).toFixed(1)}k`;
  return `${(Math.round(tokens / 100_000) / 10).toFixed(1)}m`;
}

export function filterTraceSteps(
  steps: DebugTraceStep[],
  filter: DebugTraceFilter,
): DebugTraceStep[] {
  if (filter === "all") return steps;
  return steps.filter((step) => {
    switch (filter) {
      case "central":
        return step.actor === "central" || step.kind === "final_answer";
      case "subagents":
        return (
          step.kind === "subagent" ||
          (step.parent_id?.startsWith("subagent-") ?? false)
        );
      case "tools":
        return step.kind === "tool_request" || step.kind === "tool_result";
      case "middleware":
        return step.kind === "middleware" || step.actor === "middleware";
      case "errors":
        return step.status === "failed" || step.kind === "error";
    }
  });
}
