import type { ToolCall } from "@langchain/core/messages";
import type { AIMessage } from "@langchain/langgraph-sdk";

import type { Translations } from "../i18n";
import { hasToolCalls } from "../messages/utils";

import { getSPActionType, isSPActionToolName } from "./sp-actions";

export interface ExplainableToolCall {
  name?: string;
  args?: unknown;
}

function toolArgs(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function toolName(value: unknown): string {
  if (typeof value !== "string") return "unknown";
  const normalized = value.trim();
  if (!normalized) return "unknown";
  return normalized;
}

export function explainLastToolCall(message: AIMessage, t: Translations) {
  if (hasToolCalls(message)) {
    const lastToolCall = message.tool_calls![message.tool_calls!.length - 1]!;
    return explainToolCall(lastToolCall, t);
  }
  return t.common.thinking;
}

export function explainToolCall(
  toolCall: ExplainableToolCall | ToolCall,
  t: Translations,
) {
  const name = toolName(toolCall.name);
  const args = toolArgs(toolCall.args);
  const query = typeof args.query === "string" ? args.query.trim() : "";
  if (isSPActionToolName(name)) {
    return t.toolCalls.spAction(getSPActionType(name));
  } else if (name === "web_search" || name === "image_search") {
    if (query) {
      return t.toolCalls.searchFor(query);
    }
    return name === "image_search"
      ? t.toolCalls.searchForRelatedImages
      : t.toolCalls.searchForRelatedInfo;
  } else if (name === "web_fetch") {
    return t.toolCalls.viewWebPage;
  } else if (name === "present_files") {
    return t.toolCalls.presentFiles;
  } else if (name === "write_todos") {
    return t.toolCalls.writeTodos;
  } else if (typeof args.description === "string" && args.description.trim()) {
    return args.description;
  } else {
    return t.toolCalls.useTool(name);
  }
}
