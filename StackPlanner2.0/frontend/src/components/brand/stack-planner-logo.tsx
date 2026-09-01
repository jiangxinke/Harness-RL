import Image from "next/image";

import { PRODUCT_MARK_PATH, PRODUCT_NAME } from "@/core/branding";
import { cn } from "@/lib/utils";

export function StackPlannerLogo({
  className,
  compact = false,
  markClassName,
}: {
  className?: string;
  compact?: boolean;
  markClassName?: string;
}) {
  return (
    <span
      className={cn("inline-flex min-w-0 items-center gap-2", className)}
      data-testid="stackplanner-logo"
    >
      <Image
        alt=""
        aria-hidden="true"
        className={cn("size-7 shrink-0", markClassName)}
        height={28}
        priority
        src={PRODUCT_MARK_PATH}
        width={28}
      />
      {compact ? (
        <span className="sr-only">{PRODUCT_NAME}</span>
      ) : (
        <span className="truncate font-serif">{PRODUCT_NAME}</span>
      )}
    </span>
  );
}
