import fs from "node:fs/promises";
import path from "node:path";

import type { NextRequest } from "next/server";

import {
  getArtifactContentType,
  resolveDemoArtifactPath,
} from "@/core/artifacts/demo-path";

const DEMO_THREADS_ROOT = path.resolve(
  process.cwd(),
  "public",
  "demo",
  "threads",
);

export async function GET(
  request: NextRequest,
  {
    params,
  }: {
    params: Promise<{
      thread_id: string;
      artifact_path?: string[] | undefined;
    }>;
  },
) {
  const resolvedParams = await params;
  const artifactPath = resolveDemoArtifactPath({
    artifactPath: resolvedParams.artifact_path?.join("/") ?? "",
    demoThreadsRoot: DEMO_THREADS_ROOT,
    threadId: resolvedParams.thread_id,
  });
  if (!artifactPath) {
    return new Response("File not found", { status: 404 });
  }

  try {
    const fileStat = await fs.stat(artifactPath);
    if (!fileStat.isFile()) {
      return new Response("File not found", { status: 404 });
    }
    const headers = new Headers({
      "Cache-Control": "no-store",
      "Content-Type": getArtifactContentType(artifactPath),
    });
    if (request.nextUrl.searchParams.get("download") === "true") {
      const filename = path.basename(artifactPath).replace(/["\\\r\n]/g, "_");
      headers.set(
        "Content-Disposition",
        `attachment; filename="${filename}"; filename*=UTF-8''${encodeURIComponent(filename)}`,
      );
    }
    return new Response(await fs.readFile(artifactPath), {
      status: 200,
      headers,
    });
  } catch {
    return new Response("File not found", { status: 404 });
  }
}
