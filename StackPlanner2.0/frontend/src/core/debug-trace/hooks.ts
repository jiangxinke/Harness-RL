import { useQuery } from "@tanstack/react-query";

import { fetchDebugTrace } from "./api";
import { type DebugTraceResponse } from "./types";

export function useDebugTrace(threadId?: string, runId?: string) {
  return useQuery<DebugTraceResponse>({
    queryKey: ["debug-trace", threadId, runId],
    queryFn: ({ signal }) => fetchDebugTrace(threadId!, runId!, signal),
    enabled: Boolean(threadId && runId),
    refetchOnWindowFocus: false,
    refetchInterval(query) {
      const status = query.state.data?.status;
      return status === "pending" || status === "running" ? 1500 : false;
    },
  });
}
