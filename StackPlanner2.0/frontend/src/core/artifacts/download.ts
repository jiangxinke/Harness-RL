import { fetch as fetchWithAuth } from "@/core/api/fetcher";

export async function downloadArtifact({
  url,
  filename,
}: {
  url: string;
  filename: string;
}): Promise<void> {
  const response = await fetchWithAuth(url);
  if (!response.ok) {
    throw new Error(`Artifact download failed with status ${response.status}`);
  }

  const objectURL = URL.createObjectURL(await response.blob());
  const link = document.createElement("a");
  link.href = objectURL;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  // Let the browser start the download before releasing the Blob URL.
  window.setTimeout(() => URL.revokeObjectURL(objectURL), 0);
}
