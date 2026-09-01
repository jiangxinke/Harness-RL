import { describe, expect, test } from "@rstest/core";

import { canBrowserPreviewFile, isBrowserImageFile } from "@/core/utils/files";

describe("browser artifact file classification", () => {
  test.each(["chart.png", "photo.JPG", "plot.webp", "frame.avif"])(
    "recognizes image artifact %s",
    (filepath) => {
      expect(isBrowserImageFile(filepath)).toBe(true);
      expect(canBrowserPreviewFile(filepath)).toBe(true);
    },
  );

  test("keeps non-image browser previews separate", () => {
    expect(isBrowserImageFile("report.pdf")).toBe(false);
    expect(canBrowserPreviewFile("report.pdf")).toBe(true);
    expect(isBrowserImageFile("report.md")).toBe(false);
  });
});
