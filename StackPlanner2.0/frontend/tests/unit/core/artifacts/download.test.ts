import { beforeEach, expect, rs, test } from "@rstest/core";

const fetchWithAuth = rs.fn();

rs.mock("@/core/api/fetcher", () => ({
  fetch: fetchWithAuth,
}));

beforeEach(() => {
  fetchWithAuth.mockReset();
});

test("downloads an artifact through the authenticated fetcher", async () => {
  const { downloadArtifact } = await import("@/core/artifacts/download");
  fetchWithAuth.mockResolvedValue({
    ok: true,
    blob: async () => new Blob(["artifact content"], { type: "text/markdown" }),
  });

  const previousWindow = (globalThis as { window?: unknown }).window;
  const previousDocument = (globalThis as { document?: unknown }).document;
  const click = rs.fn();
  const remove = rs.fn();
  const appendChild = rs.fn();
  const anchor = { href: "", download: "", click, remove };
  Object.defineProperty(globalThis, "document", {
    configurable: true,
    value: {
      body: { appendChild },
      createElement: rs.fn().mockReturnValue(anchor),
    },
  });
  const setTimeout = ((callback: TimerHandler) => {
    if (typeof callback === "function") callback();
    return 0;
  }) as typeof globalThis.setTimeout;
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: { setTimeout },
  });

  const createObjectURL = rs
    .spyOn(URL, "createObjectURL")
    .mockReturnValue("blob:http://localhost/artifact");
  const revokeObjectURL = rs.spyOn(URL, "revokeObjectURL");

  try {
    await downloadArtifact({
      url: "/api/threads/thread-1/artifacts/report.md?download=true",
      filename: "report.md",
    });

    expect(fetchWithAuth).toHaveBeenCalledWith(
      "/api/threads/thread-1/artifacts/report.md?download=true",
    );
    expect(createObjectURL).toHaveBeenCalledTimes(1);
    expect(appendChild).toHaveBeenCalledWith(anchor);
    expect(anchor.download).toBe("report.md");
    expect(click).toHaveBeenCalledTimes(1);
    expect(remove).toHaveBeenCalledTimes(1);
    expect(revokeObjectURL).toHaveBeenCalledWith(
      "blob:http://localhost/artifact",
    );
  } finally {
    createObjectURL.mockRestore();
    revokeObjectURL.mockRestore();
    if (previousWindow === undefined) {
      delete (globalThis as { window?: unknown }).window;
    } else {
      Object.defineProperty(globalThis, "window", {
        configurable: true,
        value: previousWindow,
      });
    }
    if (previousDocument === undefined) {
      delete (globalThis as { document?: unknown }).document;
    } else {
      Object.defineProperty(globalThis, "document", {
        configurable: true,
        value: previousDocument,
      });
    }
  }
});
