import { fetch } from "@/core/api/fetcher";
import { getBackendBaseURL } from "@/core/config";

import { type DebugTraceResponse } from "./types";

export async function fetchDebugTrace(
  threadId: string,
  runId: string,
  signal?: AbortSignal,
): Promise<DebugTraceResponse> {
  const url = `${getBackendBaseURL()}/api/threads/${encodeURIComponent(
    threadId,
  )}/runs/${encodeURIComponent(runId)}/debug-trace`;
  const response = await fetch(url, { signal });
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as {
      detail?: string;
    } | null;
    throw new Error(
      payload?.detail ?? `Failed to load trace (${response.status})`,
    );
  }
  return (await response.json()) as DebugTraceResponse;
}
