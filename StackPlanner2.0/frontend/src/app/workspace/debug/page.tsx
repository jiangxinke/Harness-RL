"use client";

import {
  ActivityIcon,
  AlertTriangleIcon,
  BotIcon,
  BrainCircuitIcon,
  Clock3Icon,
  DownloadIcon,
  RefreshCwIcon,
  WrenchIcon,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import {
  WorkspaceBody,
  WorkspaceContainer,
  WorkspaceHeader,
} from "@/components/workspace/workspace-container";
import {
  filterTraceSteps,
  formatTraceDuration,
  formatTraceTokens,
  type DebugTraceFilter,
  type DebugTraceResponse,
  type DebugTraceStep,
  useDebugTrace,
} from "@/core/debug-trace";
import { useI18n } from "@/core/i18n/hooks";
import { useLocalSettings } from "@/core/settings";
import { useInfiniteThreads, useThreadRuns } from "@/core/threads/hooks";
import { titleOfThread } from "@/core/threads/utils";
import { cn } from "@/lib/utils";

const TRACE_FILTERS: { value: DebugTraceFilter; label: string }[] = [
  { value: "all", label: "全部步骤" },
  { value: "central", label: "中枢" },
  { value: "subagents", label: "子智能体" },
  { value: "tools", label: "工具" },
  { value: "middleware", label: "中间件" },
  { value: "errors", label: "错误" },
];

function statusVariant(status: string) {
  if (status === "failed" || status === "error") return "destructive";
  if (status === "running" || status === "pending") return "secondary";
  return "outline";
}

function formatTimestamp(value: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function StepIcon({ step }: { step: DebugTraceStep }) {
  if (step.status === "failed" || step.kind === "error") {
    return <AlertTriangleIcon className="text-destructive size-4" />;
  }
  if (step.kind === "tool_request" || step.kind === "tool_result") {
    return <WrenchIcon className="size-4 text-sky-600" />;
  }
  if (step.kind === "subagent" || step.parent_id) {
    return <BotIcon className="size-4 text-violet-600" />;
  }
  if (step.actor === "central") {
    return <BrainCircuitIcon className="text-primary size-4" />;
  }
  return <ActivityIcon className="text-muted-foreground size-4" />;
}

function TraceStepRow({ step }: { step: DebugTraceStep }) {
  const hasDetails = step.detail !== null || Boolean(step.error);
  return (
    <details
      className={cn(
        "group bg-background/80 rounded-lg border",
        step.parent_id && "ml-6 border-l-2",
      )}
    >
      <summary
        className={cn(
          "flex list-none items-start gap-3 p-3",
          hasDetails ? "cursor-pointer" : "cursor-default",
        )}
        onClick={(event) => {
          if (!hasDetails) event.preventDefault();
        }}
      >
        <div className="bg-muted mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-full">
          <StepIcon step={step} />
        </div>
        <div className="min-w-0 flex-1 space-y-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-medium">{step.label}</span>
            <Badge variant={statusVariant(step.status)}>{step.status}</Badge>
            <span className="text-muted-foreground text-xs">{step.actor}</span>
          </div>
          {step.summary && (
            <p className="text-muted-foreground line-clamp-3 text-sm break-words">
              {step.summary}
            </p>
          )}
          <div className="text-muted-foreground flex flex-wrap gap-x-4 gap-y-1 text-xs">
            <span>{formatTimestamp(step.started_at)}</span>
            <span>{formatTraceDuration(step.duration_ms)}</span>
            {step.tokens && (
              <span>{formatTraceTokens(step.tokens.total)} tokens</span>
            )}
            {step.offset_ms !== null && (
              <span>+{formatTraceDuration(step.offset_ms)}</span>
            )}
          </div>
        </div>
      </summary>
      {hasDetails && (
        <div className="border-t px-4 py-3">
          {step.error && (
            <p className="text-destructive mb-3 text-sm break-words">
              {step.error}
            </p>
          )}
          {step.detail !== null && (
            <pre className="bg-muted max-h-96 overflow-auto rounded-md p-3 text-xs leading-relaxed break-words whitespace-pre-wrap">
              {JSON.stringify(step.detail, null, 2)}
            </pre>
          )}
        </div>
      )}
    </details>
  );
}

function exportTrace(trace: DebugTraceResponse) {
  const blob = new Blob([JSON.stringify(trace, null, 2)], {
    type: "application/json;charset=utf-8",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `sp2-trace-${trace.run_id}.json`;
  anchor.click();
  URL.revokeObjectURL(url);
}

export default function DebugTracePage() {
  const { t } = useI18n();
  const [settings, setSettings] = useLocalSettings();
  const [selectedThreadId, setSelectedThreadId] = useState("");
  const [selectedRunId, setSelectedRunId] = useState("");
  const [filter, setFilter] = useState<DebugTraceFilter>("all");
  const { data: threadPages } = useInfiniteThreads();
  const threads = useMemo(() => threadPages?.pages.flat() ?? [], [threadPages]);
  const { data: runs = [], isLoading: runsLoading } = useThreadRuns(
    selectedThreadId,
    { enabled: Boolean(selectedThreadId) },
  );
  const {
    data: trace,
    error,
    isLoading: traceLoading,
    isFetching,
    refetch,
  } = useDebugTrace(selectedThreadId, selectedRunId);

  useEffect(() => {
    document.title = `${t.sidebar.executionTrace} - ${t.pages.appName}`;
  }, [t.pages.appName, t.sidebar.executionTrace]);

  useEffect(() => {
    if (selectedThreadId || threads.length === 0) return;
    const params = new URLSearchParams(window.location.search);
    const requested = params.get("thread_id");
    const next = requested ?? threads[0]?.thread_id ?? "";
    setSelectedThreadId(next);
  }, [selectedThreadId, threads]);

  useEffect(() => {
    if (!selectedThreadId || runs.length === 0) return;
    const params = new URLSearchParams(window.location.search);
    const requested = params.get("run_id");
    const next =
      (requested && runs.some((run) => run.run_id === requested)
        ? requested
        : runs[0]?.run_id) ?? "";
    if (!selectedRunId || !runs.some((run) => run.run_id === selectedRunId)) {
      setSelectedRunId(next);
    }
  }, [runs, selectedRunId, selectedThreadId]);

  useEffect(() => {
    if (!selectedThreadId) return;
    const params = new URLSearchParams();
    params.set("thread_id", selectedThreadId);
    if (selectedRunId) params.set("run_id", selectedRunId);
    window.history.replaceState(null, "", `/workspace/debug?${params}`);
  }, [selectedRunId, selectedThreadId]);

  const visibleSteps = useMemo(
    () => filterTraceSteps(trace?.steps ?? [], filter),
    [filter, trace?.steps],
  );

  return (
    <WorkspaceContainer>
      <WorkspaceHeader />
      <WorkspaceBody className="items-stretch overflow-hidden">
        <ScrollArea className="size-full">
          <div className="mx-auto flex w-full max-w-6xl flex-col gap-5 p-4 pb-12 md:p-8">
            <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
              <div>
                <h1 className="text-2xl font-semibold">执行追踪</h1>
                <p className="text-muted-foreground mt-1 max-w-3xl text-sm">
                  查看一次 query
                  从进入系统到最终回答之间的中枢动作、子智能体、工具、中间件、耗时与
                  token。
                </p>
              </div>
              <div className="flex items-center gap-3 rounded-lg border px-4 py-3">
                <div>
                  <div className="text-sm font-medium">Debug 模式</div>
                  <div className="text-muted-foreground text-xs">
                    对下一次提交生效
                  </div>
                </div>
                <Switch
                  aria-label="切换执行追踪 Debug 模式"
                  checked={Boolean(settings.context.debug_trace_enabled)}
                  onCheckedChange={(checked) =>
                    setSettings("context", { debug_trace_enabled: checked })
                  }
                />
              </div>
            </div>

            <div className="border-border bg-muted/40 flex gap-3 rounded-lg border p-4 text-sm">
              <AlertTriangleIcon className="mt-0.5 size-4 shrink-0" />
              <p>
                这里展示可观察的模型输出、显式动作理由、中枢实际收到的任务记忆上下文，以及工具输入/输出；不会记录模型提供商未公开的隐藏思维链。敏感字段会在后端脱敏。
              </p>
            </div>

            <Card className="gap-4 py-5">
              <CardHeader className="px-5">
                <CardTitle>选择运行</CardTitle>
                <CardDescription>
                  历史运行也能查看基础事件；打开 Debug
                  后的新运行会额外保存中枢决策上下文。
                </CardDescription>
              </CardHeader>
              <CardContent className="grid gap-3 px-5 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto]">
                <Select
                  value={selectedThreadId}
                  onValueChange={(value) => {
                    setSelectedThreadId(value);
                    setSelectedRunId("");
                  }}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="选择会话" />
                  </SelectTrigger>
                  <SelectContent>
                    {selectedThreadId &&
                      !threads.some(
                        (thread) => thread.thread_id === selectedThreadId,
                      ) && (
                        <SelectItem value={selectedThreadId}>
                          {selectedThreadId}
                        </SelectItem>
                      )}
                    {threads.map((thread) => (
                      <SelectItem
                        key={thread.thread_id}
                        value={thread.thread_id}
                      >
                        {titleOfThread(thread)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Select
                  value={selectedRunId}
                  onValueChange={setSelectedRunId}
                  disabled={!selectedThreadId || runsLoading}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue
                      placeholder={runsLoading ? "加载运行…" : "选择 Run"}
                    />
                  </SelectTrigger>
                  <SelectContent>
                    {runs.map((run) => (
                      <SelectItem key={run.run_id} value={run.run_id}>
                        {String(run.created_at ?? "")
                          .replace("T", " ")
                          .slice(0, 19)}{" "}
                        · {run.status}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Button
                  variant="outline"
                  onClick={() => void refetch()}
                  disabled={!selectedRunId || isFetching}
                >
                  <RefreshCwIcon
                    className={cn("size-4", isFetching && "animate-spin")}
                  />
                  刷新
                </Button>
              </CardContent>
            </Card>

            {error && (
              <div className="border-destructive/40 bg-destructive/5 text-destructive rounded-lg border p-4 text-sm">
                {error.message}
              </div>
            )}

            {traceLoading && selectedRunId && (
              <div className="text-muted-foreground py-16 text-center text-sm">
                正在加载执行追踪…
              </div>
            )}

            {trace && (
              <>
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                  <Card className="gap-2 py-4">
                    <CardContent className="px-4">
                      <div className="text-muted-foreground flex items-center gap-2 text-xs">
                        <Clock3Icon className="size-3.5" /> 总耗时
                      </div>
                      <div className="mt-2 text-xl font-semibold">
                        {formatTraceDuration(trace.duration_ms)}
                      </div>
                    </CardContent>
                  </Card>
                  <Card className="gap-2 py-4">
                    <CardContent className="px-4">
                      <div className="text-muted-foreground text-xs">
                        总 token
                      </div>
                      <div className="mt-2 text-xl font-semibold">
                        {formatTraceTokens(trace.tokens.total)}
                      </div>
                      <div className="text-muted-foreground mt-1 text-xs">
                        中枢 {formatTraceTokens(trace.tokens.lead_agent)} ·
                        子智能体 {formatTraceTokens(trace.tokens.subagent)}
                      </div>
                    </CardContent>
                  </Card>
                  <Card className="gap-2 py-4">
                    <CardContent className="px-4">
                      <div className="text-muted-foreground text-xs">
                        模型调用
                      </div>
                      <div className="mt-2 text-xl font-semibold">
                        {trace.tokens.llm_calls}
                      </div>
                      <div className="text-muted-foreground mt-1 text-xs">
                        {trace.event_count} 个原始事件
                      </div>
                    </CardContent>
                  </Card>
                  <Card className="gap-2 py-4">
                    <CardContent className="px-4">
                      <div className="text-muted-foreground text-xs">
                        运行状态
                      </div>
                      <div className="mt-2 flex items-center gap-2">
                        <Badge variant={statusVariant(trace.status)}>
                          {trace.status}
                        </Badge>
                        <Badge variant="outline">
                          {trace.enabled ? "增强追踪" : "基础追踪"}
                        </Badge>
                      </div>
                    </CardContent>
                  </Card>
                </div>

                <Card className="gap-4 py-5">
                  <CardHeader className="px-5">
                    <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                      <div>
                        <CardTitle>执行时间线</CardTitle>
                        <CardDescription className="mt-1">
                          共 {trace.steps.length} 步，当前显示{" "}
                          {visibleSteps.length} 步
                          {trace.truncated ? "；事件过多，结果已截断" : ""}
                        </CardDescription>
                      </div>
                      <div className="flex items-center gap-2">
                        <Select
                          value={filter}
                          onValueChange={(value) =>
                            setFilter(value as DebugTraceFilter)
                          }
                        >
                          <SelectTrigger className="w-36">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {TRACE_FILTERS.map((item) => (
                              <SelectItem key={item.value} value={item.value}>
                                {item.label}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        <Button
                          variant="outline"
                          onClick={() => exportTrace(trace)}
                        >
                          <DownloadIcon className="size-4" /> 导出 JSON
                        </Button>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-2 px-5">
                    {visibleSteps.map((step) => (
                      <TraceStepRow key={step.id} step={step} />
                    ))}
                    {visibleSteps.length === 0 && (
                      <div className="text-muted-foreground py-12 text-center text-sm">
                        当前筛选条件下没有步骤。
                      </div>
                    )}
                  </CardContent>
                </Card>
              </>
            )}
          </div>
        </ScrollArea>
      </WorkspaceBody>
    </WorkspaceContainer>
  );
}
