"use client";

import { BrainCircuit, RefreshCw } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { useThread } from "@/components/workspace/messages/context";
import { Tooltip } from "@/components/workspace/tooltip";
import { getAPIClient } from "@/core/api";
import type { AgentThreadState } from "@/core/threads/types";

type MemoryEntry = {
  id?: string;
  action?: string;
  actor?: string;
  content?: string;
  priority?: string;
  stage?: string;
  status?: string;
  result_ref?: string;
};

type MemoryState = {
  entries?: MemoryEntry[];
  version?: number;
};

function memoryStateFromValues(values: AgentThreadState): MemoryState {
  const memory = values.sp_task_memory;
  if (!memory || typeof memory !== "object") {
    return { entries: [] };
  }
  const entries = Array.isArray(memory.entries)
    ? memory.entries.filter(
        (entry): entry is MemoryEntry =>
          typeof entry === "object" && entry !== null,
      )
    : [];
  return {
    entries,
    version: typeof memory.version === "number" ? memory.version : undefined,
  };
}

export function SPMemoryTrigger({ threadId }: { threadId: string }) {
  const { thread, isMock } = useThread();
  const [open, setOpen] = useState(false);
  const [snapshot, setSnapshot] = useState<AgentThreadState | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const liveValuesRef = useRef(thread.values);
  liveValuesRef.current = thread.values;

  useEffect(() => {
    if (!open) {
      setSnapshot(null);
      return;
    }

    let active = true;
    setSnapshot(liveValuesRef.current);
    if (isMock || !threadId) {
      return () => {
        active = false;
      };
    }

    setIsRefreshing(true);
    void getAPIClient(isMock)
      .threads.getState<AgentThreadState>(threadId)
      .then((state) => {
        if (active) {
          setSnapshot(state.values);
        }
      })
      .catch(() => {
        // The live stream values remain available if the checkpoint request fails.
      })
      .finally(() => {
        if (active) {
          setIsRefreshing(false);
        }
      });

    return () => {
      active = false;
    };
  }, [isMock, open, threadId]);

  const values = useMemo(
    () => ({ ...snapshot, ...thread.values }),
    [snapshot, thread.values],
  );
  const memory = memoryStateFromValues(values);
  const entries = [...(memory.entries ?? [])].reverse();
  const stage =
    typeof values.sp_current_stage === "string"
      ? values.sp_current_stage
      : "unknown";
  const activeDelegate =
    typeof values.sp_active_delegate_id === "string"
      ? values.sp_active_delegate_id
      : "none";

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <Tooltip content="View SP central memory">
        <DialogTrigger asChild>
          <Button
            aria-label="View SP central memory"
            className="text-muted-foreground hover:text-foreground"
            data-testid="sp-memory-trigger"
            size="sm"
            variant="ghost"
          >
            <BrainCircuit />
            <span className="hidden sm:inline">SP Memory</span>
          </Button>
        </DialogTrigger>
      </Tooltip>
      <DialogContent className="grid h-[min(80vh,720px)] max-h-[min(80vh,720px)] min-h-0 max-w-2xl grid-rows-[auto_auto_minmax(0,1fr)] overflow-hidden">
        <DialogHeader>
          <DialogTitle>SP Central Memory</DialogTitle>
          <DialogDescription className="sr-only">
            Current checkpoint view of the StackPlanner central task memory.
          </DialogDescription>
        </DialogHeader>
        <div className="grid grid-cols-2 gap-2 text-sm sm:grid-cols-3">
          <div className="bg-muted/40 rounded-md border px-3 py-2">
            <div className="text-muted-foreground text-xs">Stage</div>
            <div className="truncate font-medium">{stage}</div>
          </div>
          <div className="bg-muted/40 rounded-md border px-3 py-2">
            <div className="text-muted-foreground text-xs">Active delegate</div>
            <div className="truncate font-medium">{activeDelegate}</div>
          </div>
          <div className="bg-muted/40 rounded-md border px-3 py-2">
            <div className="text-muted-foreground text-xs">Entries</div>
            <div className="font-medium">{entries.length}</div>
          </div>
        </div>
        <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-hidden">
          <div className="text-muted-foreground flex items-center justify-between text-xs">
            <span>Newest entries first</span>
            {isRefreshing && <RefreshCw className="size-3 animate-spin" />}
          </div>
          <div className="min-h-0 overflow-y-auto overscroll-contain rounded-md border">
            {entries.length === 0 ? (
              <div className="text-muted-foreground p-6 text-center text-sm">
                No SP task memory in the latest checkpoint.
              </div>
            ) : (
              <div className="divide-y">
                {entries.map((entry, index) => (
                  <div className="space-y-1 px-3 py-3" key={entry.id ?? index}>
                    <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs">
                      <span className="font-semibold">
                        {entry.action ?? "memory"}
                      </span>
                      {entry.actor && (
                        <span className="text-muted-foreground">
                          {entry.actor}
                        </span>
                      )}
                      {entry.stage && (
                        <span className="text-muted-foreground">
                          stage={entry.stage}
                        </span>
                      )}
                      {entry.priority && (
                        <span className="text-muted-foreground">
                          priority={entry.priority}
                        </span>
                      )}
                      {entry.status && (
                        <span className="text-muted-foreground">
                          status={entry.status}
                        </span>
                      )}
                    </div>
                    <p className="text-sm whitespace-pre-wrap">
                      {entry.content ?? ""}
                    </p>
                    {entry.result_ref && (
                      <div className="text-muted-foreground truncate text-xs">
                        ref={entry.result_ref}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
