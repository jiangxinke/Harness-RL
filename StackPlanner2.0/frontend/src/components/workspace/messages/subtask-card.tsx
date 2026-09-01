import {
  CheckCircleIcon,
  ChevronUp,
  ClipboardListIcon,
  Loader2Icon,
  SparklesIcon,
  WrenchIcon,
  XCircleIcon,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import {
  ChainOfThought,
  ChainOfThoughtContent,
  ChainOfThoughtStep,
} from "@/components/ai-elements/chain-of-thought";
import { Shimmer } from "@/components/ai-elements/shimmer";
import { Button } from "@/components/ui/button";
import { ShineBorder } from "@/components/ui/shine-border";
import { useI18n } from "@/core/i18n/hooks";
import { hasToolCalls } from "@/core/messages/utils";
import { useRehypeSplitWordsIntoSpans } from "@/core/rehype";
import { streamdownPluginsWithWordAnimation } from "@/core/streamdown";
import { SafeStreamdown } from "@/core/streamdown/components";
import { fetchSubtaskSteps } from "@/core/tasks/api";
import { useSubtask, useUpdateSubtask } from "@/core/tasks/context";
import {
  latestActiveSubtaskToolCall,
  stepsForDisplay,
} from "@/core/tasks/steps";
import { explainLastToolCall, explainToolCall } from "@/core/tools/utils";
import { cn } from "@/lib/utils";

import { ArtifactFileList } from "../artifacts/artifact-file-list";
import { CitationLink } from "../citations/citation-link";
import { FlipDisplay } from "../flip-display";

import { MarkdownContent } from "./markdown-content";

export function SubtaskCard({
  className,
  taskId,
  threadId,
  runId,
  isLoading,
}: {
  className?: string;
  taskId: string;
  threadId?: string;
  runId?: string;
  isLoading: boolean;
}) {
  const { t } = useI18n();
  const [collapsed, setCollapsed] = useState(true);
  const rehypePlugins = useRehypeSplitWordsIntoSpans(isLoading);
  const task = useSubtask(taskId)!;
  const updateSubtask = useUpdateSubtask();

  // The card shows the subagent's step timeline (#3779): its reasoning turns
  // (AI text) interleaved with the tools it ran (by name). See stepsForDisplay
  // for what is kept/dropped.
  const displaySteps = stepsForDisplay(task.steps, task.status);
  const activeToolCall = latestActiveSubtaskToolCall(task.steps);
  const runningActivity = activeToolCall
    ? explainToolCall(activeToolCall, t)
    : task.latestMessage && hasToolCalls(task.latestMessage)
      ? explainLastToolCall(task.latestMessage, t)
      : t.subtasks[task.status];

  // Backfill step history on expand for historical runs (#3779). Live runs
  // already have steps from SSE, so the `steps.length` guard skips the fetch.
  const stepsCount = task.steps?.length ?? 0;
  const backfilledRef = useRef(false);
  useEffect(() => {
    if (collapsed || backfilledRef.current || stepsCount > 0) {
      return;
    }
    if (!threadId || !runId) {
      return;
    }
    backfilledRef.current = true;
    fetchSubtaskSteps(threadId, runId, taskId)
      .then((steps) => {
        if (steps.length > 0) {
          updateSubtask({ id: taskId, steps });
        }
      })
      .catch(() => {
        // Allow a retry on the next expand if the fetch failed.
        backfilledRef.current = false;
      });
  }, [collapsed, stepsCount, threadId, runId, taskId, updateSubtask]);
  const icon = useMemo(() => {
    if (task.status === "completed") {
      return <CheckCircleIcon className="size-3" />;
    } else if (task.status === "failed") {
      return <XCircleIcon className="size-3 text-red-500" />;
    } else if (task.status === "in_progress") {
      return <Loader2Icon className="size-3 animate-spin" />;
    }
  }, [task.status]);
  return (
    <ChainOfThought
      className={cn(
        "relative w-full max-w-full min-w-0 gap-2 overflow-hidden rounded-lg border py-0",
        className,
      )}
      open={!collapsed}
    >
      <div
        className={cn(
          "ambilight z-[-1]",
          task.status === "in_progress" ? "enabled" : "",
        )}
      ></div>
      {task.status === "in_progress" && (
        <>
          <ShineBorder
            borderWidth={1.5}
            shineColor={["#A07CFE", "#FE8FB5", "#FFBE7B"]}
          />
        </>
      )}
      <div className="bg-background/95 flex w-full flex-col rounded-lg">
        <div className="flex w-full items-center justify-between p-0.5">
          <Button
            className="h-auto w-full min-w-0 items-start justify-start overflow-hidden text-left whitespace-normal"
            variant="ghost"
            onClick={() => setCollapsed(!collapsed)}
          >
            <div className="flex w-full min-w-0 items-start justify-between gap-2">
              <ChainOfThoughtStep
                className="min-w-0 flex-1 font-normal"
                label={
                  <div className="max-w-full [overflow-wrap:anywhere] whitespace-normal">
                    {task.status === "in_progress" ? (
                      <Shimmer
                        as="span"
                        className="max-w-full [overflow-wrap:anywhere] whitespace-normal"
                        duration={3}
                        spread={3}
                      >
                        {task.description}
                      </Shimmer>
                    ) : (
                      task.description
                    )}
                  </div>
                }
                icon={<ClipboardListIcon />}
              ></ChainOfThoughtStep>
              <div className="flex shrink-0 items-center gap-1">
                {collapsed && (
                  <div
                    className={cn(
                      "text-muted-foreground flex items-center gap-1 text-xs font-normal",
                      task.status === "failed" ? "text-red-500 opacity-67" : "",
                    )}
                  >
                    {icon}
                    <FlipDisplay
                      className="line-clamp-2 max-w-[420px] min-w-0 pb-1 text-right [overflow-wrap:anywhere] whitespace-normal"
                      uniqueKey={
                        task.status === "in_progress"
                          ? runningActivity
                          : (task.latestMessage?.id ?? task.status)
                      }
                    >
                      {task.status === "in_progress"
                        ? runningActivity
                        : t.subtasks[task.status]}
                    </FlipDisplay>
                  </div>
                )}
                <ChevronUp
                  className={cn(
                    "text-muted-foreground size-4",
                    !collapsed ? "" : "rotate-180",
                  )}
                />
              </div>
            </div>
          </Button>
        </div>
        <ChainOfThoughtContent className="px-4 pb-4">
          {task.prompt && (
            <ChainOfThoughtStep
              label={
                <SafeStreamdown
                  {...streamdownPluginsWithWordAnimation}
                  components={{ a: CitationLink }}
                >
                  {task.prompt}
                </SafeStreamdown>
              }
            ></ChainOfThoughtStep>
          )}
          {displaySteps.map((step, i) => {
            const isLastWhileRunning =
              task.status === "in_progress" && i === displaySteps.length - 1;
            const hasRequestedTools =
              step.kind === "ai" && Boolean(step.tool_calls?.length);
            const icon = isLastWhileRunning ? (
              <Loader2Icon className="size-4 animate-spin" />
            ) : step.kind === "tool" || hasRequestedTools ? (
              <WrenchIcon className="size-4" />
            ) : (
              <SparklesIcon className="size-4" />
            );
            return (
              <ChainOfThoughtStep
                key={`${step.message_index}-${i}`}
                label={
                  step.kind === "tool" ? (
                    (step.tool_name ?? t.subtasks[task.status])
                  ) : hasRequestedTools ? (
                    <div className="flex max-w-full min-w-0 flex-col gap-1">
                      {step.text.trim() !== "" && (
                        <div className="text-muted-foreground line-clamp-3 text-sm">
                          <MarkdownContent
                            content={step.text}
                            isLoading={false}
                            rehypePlugins={rehypePlugins}
                          />
                        </div>
                      )}
                      {step.tool_calls!.map((toolCall, toolCallIndex) => (
                        <div
                          className="max-w-full min-w-0 text-sm [overflow-wrap:anywhere] whitespace-normal"
                          key={`${toolCall.name ?? "tool"}-${toolCallIndex}`}
                        >
                          {explainToolCall(toolCall, t)}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-muted-foreground line-clamp-3 text-sm">
                      <MarkdownContent
                        content={step.text}
                        isLoading={false}
                        rehypePlugins={rehypePlugins}
                      />
                    </div>
                  )
                }
                icon={icon}
              />
            );
          })}
          {task.status === "completed" && (
            <>
              <ChainOfThoughtStep
                label={t.subtasks.completed}
                icon={<CheckCircleIcon className="size-4" />}
              ></ChainOfThoughtStep>
              <ChainOfThoughtStep
                label={
                  task.result ? (
                    <MarkdownContent
                      content={task.result}
                      isLoading={false}
                      rehypePlugins={rehypePlugins}
                    />
                  ) : null
                }
              ></ChainOfThoughtStep>
            </>
          )}
          {task.status === "failed" && (
            <ChainOfThoughtStep
              label={<div className="text-red-500">{task.error}</div>}
              icon={<XCircleIcon className="size-4 text-red-500" />}
            ></ChainOfThoughtStep>
          )}
        </ChainOfThoughtContent>
        {threadId && task.artifacts && task.artifacts.length > 0 && (
          <ArtifactFileList
            className="border-t px-4 py-4"
            files={task.artifacts}
            threadId={threadId}
          />
        )}
      </div>
    </ChainOfThought>
  );
}
