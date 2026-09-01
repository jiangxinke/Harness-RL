import { DownloadIcon, LoaderIcon, PackageIcon } from "lucide-react";
import { useCallback, useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardAction,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { downloadArtifact } from "@/core/artifacts/download";
import { artifactDownloadURL, urlOfArtifact } from "@/core/artifacts/utils";
import { useAuth } from "@/core/auth/AuthProvider";
import { useI18n } from "@/core/i18n/hooks";
import { installSkill, SkillRequestError } from "@/core/skills/api";
import {
  getFileExtensionDisplayName,
  getFileIcon,
  getFileName,
  isBrowserImageFile,
} from "@/core/utils/files";
import { cn } from "@/lib/utils";

import { useArtifacts } from "./context";

export function ArtifactFileList({
  className,
  files,
  threadId,
}: {
  className?: string;
  files: string[];
  threadId: string;
}) {
  const { t } = useI18n();
  const { user } = useAuth();
  const isAdmin = user?.system_role === "admin";
  const { select: selectArtifact, setOpen } = useArtifacts();
  const [installingFile, setInstallingFile] = useState<string | null>(null);

  const handleClick = useCallback(
    (filepath: string) => {
      selectArtifact(filepath);
      setOpen(true);
    },
    [selectArtifact, setOpen],
  );

  const handleInstallSkill = useCallback(
    async (e: React.MouseEvent, filepath: string) => {
      e.stopPropagation();
      e.preventDefault();

      if (installingFile) return;

      setInstallingFile(filepath);
      try {
        const result = await installSkill({
          thread_id: threadId,
          path: filepath,
        });
        if (result.success) {
          toast.success(result.message);
        } else {
          toast.error(result.message || "Failed to install skill");
        }
      } catch (error) {
        console.error("Failed to install skill:", error);
        if (error instanceof SkillRequestError && error.isAdminRequired) {
          toast.error(t.settings.skills.installAdminRequired);
        } else {
          toast.error("Failed to install skill");
        }
      } finally {
        setInstallingFile(null);
      }
    },
    [threadId, installingFile, t],
  );

  return (
    <ul className={cn("flex w-full flex-col gap-4", className)}>
      {files.map((file) => {
        const isImage = isBrowserImageFile(file);
        const previewUrl = isImage
          ? urlOfArtifact({ filepath: file, threadId })
          : undefined;
        return (
          <Card
            key={file}
            className="relative cursor-pointer gap-3 p-3"
            onClick={() => handleClick(file)}
          >
            {previewUrl && (
              <div className="bg-muted/20 border-border/40 flex max-h-72 w-full items-center justify-center overflow-hidden rounded-md border">
                <img
                  src={previewUrl}
                  alt={getFileName(file)}
                  className="max-h-72 w-full object-contain"
                  loading="lazy"
                />
              </div>
            )}
            <CardHeader className="grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1 pr-2 pl-1">
              <CardTitle className="relative min-w-0 pl-8 leading-tight [overflow-wrap:anywhere] break-words">
                <div className="min-w-0">{getFileName(file)}</div>
                <div className="absolute top-2 -left-0.5">
                  {getFileIcon(file, "size-6")}
                </div>
              </CardTitle>
              <CardDescription className="min-w-0 pl-8 text-xs">
                {getFileExtensionDisplayName(file)} file
              </CardDescription>
              <CardAction className="row-span-1 self-center">
                {file.endsWith(".skill") && isAdmin && (
                  <Button
                    variant="ghost"
                    disabled={installingFile === file}
                    onClick={(e) => handleInstallSkill(e, file)}
                  >
                    {installingFile === file ? (
                      <LoaderIcon className="size-4 animate-spin" />
                    ) : (
                      <PackageIcon className="size-4" />
                    )}
                    {t.common.install}
                  </Button>
                )}
                <Button
                  variant="ghost"
                  onClick={(e) => {
                    e.stopPropagation();
                    void downloadArtifact({
                      url: artifactDownloadURL(
                        urlOfArtifact({
                          filepath: file,
                          threadId,
                          download: true,
                        }),
                      ),
                      filename: getFileName(file),
                    }).catch(() => toast.error("Artifact download failed"));
                  }}
                >
                  <DownloadIcon className="size-4" />
                  {t.common.download}
                </Button>
              </CardAction>
            </CardHeader>
          </Card>
        );
      })}
    </ul>
  );
}
