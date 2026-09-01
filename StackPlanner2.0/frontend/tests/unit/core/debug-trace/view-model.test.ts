import { describe, expect, it } from "@rstest/core";

import { type DebugTraceStep } from "@/core/debug-trace/types";
import {
  filterTraceSteps,
  formatTraceDuration,
  formatTraceTokens,
} from "@/core/debug-trace/view-model";

const steps: DebugTraceStep[] = [
  {
    id: "central",
    seq: 1,
    parent_id: null,
    kind: "decision",
    label: "中枢决策",
    actor: "central",
    status: "completed",
    started_at: null,
    ended_at: null,
    duration_ms: 1200,
    offset_ms: 0,
    tokens: { input: 10, output: 2, total: 12 },
    summary: "delegate",
    detail: null,
    error: null,
  },
  {
    id: "tool",
    seq: 2,
    parent_id: "subagent-1",
    kind: "tool_request",
    label: "请求工具 · web_search",
    actor: "researcher",
    status: "completed",
    started_at: null,
    ended_at: null,
    duration_ms: 50,
    offset_ms: 100,
    tokens: null,
    summary: "gold price",
    detail: null,
    error: null,
  },
];

describe("debug trace view model", () => {
  it("formats durations across millisecond and minute ranges", () => {
    expect(formatTraceDuration(850)).toBe("850 ms");
    expect(formatTraceDuration(12_400)).toBe("12.4 s");
    expect(formatTraceDuration(125_000)).toBe("2m 5s");
    expect(formatTraceDuration(null)).toBe("—");
  });

  it("formats compact token counts", () => {
    expect(formatTraceTokens(999)).toBe("999");
    expect(formatTraceTokens(12_450)).toBe("12.5k");
  });

  it("filters central and tool steps without losing order", () => {
    expect(filterTraceSteps(steps, "central").map((step) => step.id)).toEqual([
      "central",
    ]);
    expect(filterTraceSteps(steps, "tools").map((step) => step.id)).toEqual([
      "tool",
    ]);
    expect(filterTraceSteps(steps, "all")).toEqual(steps);
  });
});
