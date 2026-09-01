import type { AnchorHTMLAttributes } from "react";
import { toast } from "sonner";

import { downloadArtifact } from "@/core/artifacts/download";
import {
  artifactDownloadURL,
  normalizeArtifactURL,
} from "@/core/artifacts/utils";
import { cn } from "@/lib/utils";

import { CitationLink } from "../citations/citation-link";

function isExternalUrl(href: string | undefined): boolean {
  return !!href && /^https?:\/\//.test(href);
}

function isSafeHref(href: string | undefined): boolean {
  if (!href) return false;
  if (href.startsWith("/") || href.startsWith("#")) return true;
  return /^(https?:|mailto:)/i.test(href);
}

/**
 * Builds the `a` renderer shared by message content and generic markdown.
 * Passing a `threadId` also resolves `/mnt/` artifact links; without it those
 * links fall through to the default external-link handling.
 */
export function createMarkdownLinkComponent(threadId?: string) {
  return function MarkdownLink({
    href,
    ...props
  }: AnchorHTMLAttributes<HTMLAnchorElement>) {
    if (typeof props.children === "string") {
      const match = /^citation:(.+)$/.exec(props.children);
      if (match && isSafeHref(href)) {
        const [, text] = match;
        return (
          <CitationLink {...props} href={href}>
            {text}
          </CitationLink>
        );
      }
    }
    const artifactURL =
      threadId && href ? normalizeArtifactURL(href, threadId) : null;
    if (artifactURL) {
      const filename = decodeURIComponent(
        artifactURL.split("/").pop()?.split("?")[0] ?? "artifact",
      );
      return (
        <a
          {...props}
          href={artifactDownloadURL(artifactURL)}
          onClick={(event) => {
            event.preventDefault();
            void downloadArtifact({
              url: artifactDownloadURL(artifactURL),
              filename,
            }).catch(() => toast.error("Artifact download failed"));
          }}
        />
      );
    }
    const { className, target, rel, ...rest } = props;
    const external = isExternalUrl(href);
    const safeHref = isSafeHref(href) ? href : undefined;
    return (
      <a
        {...rest}
        href={safeHref}
        className={cn(
          "text-primary decoration-primary/30 hover:decoration-primary/60 underline underline-offset-2 transition-colors",
          className,
        )}
        target={target ?? (external ? "_blank" : undefined)}
        rel={rel ?? (external ? "noopener noreferrer" : undefined)}
      />
    );
  };
}
