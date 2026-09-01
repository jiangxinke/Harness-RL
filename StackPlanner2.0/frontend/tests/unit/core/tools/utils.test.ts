import { describe, expect, it } from "@rstest/core";

import type { Translations } from "@/core/i18n";
import { explainToolCall } from "@/core/tools/utils";

const translations = {
  toolCalls: {
    spAction: (name: string) => `action:${name}`,
    searchFor: (query: string) => `search:${query}`,
    searchForRelatedImages: "search:images",
    searchForRelatedInfo: "search:related",
    viewWebPage: "view:web",
    presentFiles: "present:files",
    writeTodos: "write:todos",
    useTool: (name: string) => `tool:${name}`,
  },
} as unknown as Translations;

describe("explainToolCall", () => {
  it("shows the exact safe web-search query", () => {
    expect(
      explainToolCall(
        { name: "web_search", args: { query: "Qwen tool calling" } },
        translations,
      ),
    ).toBe("search:Qwen tool calling");
  });

  it("falls back safely when search args are unavailable or malformed", () => {
    expect(
      explainToolCall(
        { name: "web_search", args: "truncated private payload" },
        translations,
      ),
    ).toBe("search:related");
  });
});
