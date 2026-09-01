"use client";

import { WaypointsIcon } from "lucide-react";
import { useEffect, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { getAPIClient } from "@/core/api";
import { PRODUCT_ASSISTANT_ID, RUNTIME_NAME } from "@/core/branding";
import type { AgentThreadState } from "@/core/threads/types";

import { Tooltip } from "./tooltip";

export function StackPlannerRuntimeBadge({
  actionType,
  threadId,
  isMock = false,
}: {
  actionType?: string | null;
  threadId?: string;
  isMock?: boolean;
}) {
  const [checkpointActionType, setCheckpointActionType] = useState<
    string | null
  >(null);

  useEffect(() => {
    if (actionType || !threadId || isMock) {
      return;
    }

    let active = true;
    void getAPIClient(isMock)
      .threads.getState<AgentThreadState>(threadId)
      .then((state) => {
        if (active) {
          setCheckpointActionType(
            state.values.sp_last_handler_result?.action_type ?? null,
          );
        }
      })
      .catch(() => {
        // The live stream remains the primary source while a checkpoint loads.
      });

    return () => {
      active = false;
    };
  }, [actionType, isMock, threadId]);

  const normalizedActionType = (
    actionType ?? checkpointActionType
  )?.toUpperCase();
  const actionLabel = normalizedActionType
    ? `SP action: ${normalizedActionType}`
    : "SP Runtime";
  const accessibleLabel = `${actionLabel}: assistant_id=${PRODUCT_ASSISTANT_ID} on ${RUNTIME_NAME}`;

  return (
    <Tooltip content={accessibleLabel}>
      <Badge
        aria-label={accessibleLabel}
        className="bg-background/80 h-6 border-dashed font-mono"
        data-assistant-id={PRODUCT_ASSISTANT_ID}
        data-testid="sp-runtime-indicator"
        tabIndex={0}
        variant="outline"
      >
        <WaypointsIcon />
        <span className="hidden sm:inline">{actionLabel}</span>
        <span className="sm:hidden">SP</span>
      </Badge>
    </Tooltip>
  );
}
