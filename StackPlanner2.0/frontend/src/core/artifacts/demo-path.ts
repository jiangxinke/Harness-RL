import path from "node:path";

const SAFE_THREAD_ID = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

const CONTENT_TYPES: Readonly<Record<string, string>> = {
  ".css": "text/css; charset=utf-8",
  ".csv": "text/csv; charset=utf-8",
  ".gif": "image/gif",
  ".html": "text/html; charset=utf-8",
  ".jpeg": "image/jpeg",
  ".jpg": "image/jpeg",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".md": "text/markdown; charset=utf-8",
  ".mp4": "video/mp4",
  ".pdf": "application/pdf",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".txt": "text/plain; charset=utf-8",
  ".webm": "video/webm",
  ".webp": "image/webp",
};

export function resolveDemoArtifactPath({
  artifactPath,
  demoThreadsRoot,
  threadId,
}: {
  artifactPath: string;
  demoThreadsRoot: string;
  threadId: string;
}) {
  if (!SAFE_THREAD_ID.test(threadId) || threadId === "." || threadId === "..") {
    return null;
  }

  const segments = artifactPath.replace(/^\/+/, "").split("/");
  if (segments.shift() !== "mnt" || segments.length === 0) {
    return null;
  }

  const threadRoot = path.resolve(demoThreadsRoot, threadId);
  const candidate = path.resolve(threadRoot, ...segments);
  if (
    candidate === threadRoot ||
    !candidate.startsWith(`${threadRoot}${path.sep}`)
  ) {
    return null;
  }
  return candidate;
}

export function getArtifactContentType(filePath: string) {
  return (
    CONTENT_TYPES[path.extname(filePath).toLowerCase()] ??
    "application/octet-stream"
  );
}
