import { getBackendBaseURL } from "../config";
import { isStaticWebsiteOnly } from "../static-mode";
import type { AgentThread } from "../threads";

export function normalizeArtifactURL(
  href: string,
  threadId: string,
): string | null {
  if (href.startsWith("/mnt/")) {
    return resolveArtifactURL(href, threadId);
  }

  if (typeof window === "undefined") {
    return null;
  }

  let parsed: URL;
  try {
    parsed = new URL(href, window.location.origin);
  } catch {
    return null;
  }

  const artifactPrefix = `/api/threads/${encodeURIComponent(threadId)}/artifacts/`;
  if (!parsed.pathname.startsWith(artifactPrefix)) {
    return null;
  }

  const isCurrentOrigin = parsed.origin === window.location.origin;
  let isConfiguredBackendOrigin = false;
  try {
    isConfiguredBackendOrigin =
      parsed.origin ===
      new URL(getBackendBaseURL(), window.location.origin).origin;
  } catch {
    isConfiguredBackendOrigin = false;
  }
  const isLocalOrigin =
    parsed.hostname === "localhost" || parsed.hostname === "127.0.0.1";
  if (!isCurrentOrigin && !isConfiguredBackendOrigin && !isLocalOrigin) {
    return null;
  }

  return `${parsed.pathname}${parsed.search}`;
}

export function artifactDownloadURL(url: string): string {
  const parsed = new URL(
    url,
    typeof window === "undefined" ? "http://localhost" : window.location.origin,
  );
  parsed.searchParams.set("download", "true");
  return parsed.toString();
}

export function urlOfArtifact({
  filepath,
  threadId,
  download = false,
  isMock = false,
}: {
  filepath: string;
  threadId: string;
  download?: boolean;
  isMock?: boolean;
}) {
  if (isStaticWebsiteOnly()) {
    return staticDemoArtifactURL({ filepath, threadId, download });
  }
  if (isMock) {
    return `${getBackendBaseURL()}/mock/api/threads/${threadId}/artifacts${filepath}${download ? "?download=true" : ""}`;
  }
  return `${getBackendBaseURL()}/api/threads/${threadId}/artifacts${filepath}${download ? "?download=true" : ""}`;
}

export function extractArtifactsFromThread(thread: AgentThread) {
  return thread.values.artifacts ?? [];
}

export function resolveArtifactURL(absolutePath: string, threadId: string) {
  if (isStaticWebsiteOnly()) {
    return staticDemoArtifactURL({ filepath: absolutePath, threadId });
  }
  return `${getBackendBaseURL()}/api/threads/${threadId}/artifacts${absolutePath}`;
}

function staticDemoArtifactURL({
  filepath,
  threadId,
  download = false,
}: {
  filepath: string;
  threadId: string;
  download?: boolean;
}) {
  const demoPath = filepath.replace(/^\/mnt\//, "/");
  return `${getBackendBaseURL()}/demo/threads/${threadId}${demoPath}${download ? "?download=true" : ""}`;
}
