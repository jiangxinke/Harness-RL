import path from "node:path";

import { describe, expect, it } from "@rstest/core";

import {
  getArtifactContentType,
  resolveDemoArtifactPath,
} from "@/core/artifacts/demo-path";

describe("resolveDemoArtifactPath", () => {
  const root = path.resolve("/srv/sp2/public/demo/threads");

  it("resolves sandbox-style paths inside the selected demo thread", () => {
    expect(
      resolveDemoArtifactPath({
        artifactPath: "mnt/user-data/outputs/chart.png",
        demoThreadsRoot: root,
        threadId: "thread-123",
      }),
    ).toBe(
      path.resolve(root, "thread-123", "user-data", "outputs", "chart.png"),
    );
  });

  it("rejects traversal and malformed thread identifiers", () => {
    expect(
      resolveDemoArtifactPath({
        artifactPath: "mnt/../../other-thread/private.txt",
        demoThreadsRoot: root,
        threadId: "thread-123",
      }),
    ).toBeNull();
    expect(
      resolveDemoArtifactPath({
        artifactPath: "mnt/user-data/outputs/report.md",
        demoThreadsRoot: root,
        threadId: "../other-thread",
      }),
    ).toBeNull();
  });

  it("requires the sandbox mnt prefix", () => {
    expect(
      resolveDemoArtifactPath({
        artifactPath: "user-data/outputs/report.md",
        demoThreadsRoot: root,
        threadId: "thread-123",
      }),
    ).toBeNull();
  });
});

describe("getArtifactContentType", () => {
  it("returns renderable types for common SP artifacts", () => {
    expect(getArtifactContentType("chart.PNG")).toBe("image/png");
    expect(getArtifactContentType("report.pdf")).toBe("application/pdf");
    expect(getArtifactContentType("site.html")).toBe(
      "text/html; charset=utf-8",
    );
  });

  it("falls back to a binary content type", () => {
    expect(getArtifactContentType("model.bin")).toBe(
      "application/octet-stream",
    );
  });
});
