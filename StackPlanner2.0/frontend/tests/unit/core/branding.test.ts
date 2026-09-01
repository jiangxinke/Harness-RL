import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, test } from "@rstest/core";

import {
  PRODUCT_ASSISTANT_ID,
  PRODUCT_MARK_PATH,
  PRODUCT_NAME,
  PRODUCT_REPOSITORY_URL,
  RUNTIME_NAME,
} from "@/core/branding";

describe("StackPlanner product branding", () => {
  test("points the UI at the SP2 product and repository", () => {
    expect(PRODUCT_NAME).toBe("StackPlanner 2.0");
    expect(PRODUCT_REPOSITORY_URL).toBe(
      "https://github.com/chichuxwx/StackPlanner2",
    );
    expect(RUNTIME_NAME).toBe("DeerFlow 2.0");
    expect(PRODUCT_ASSISTANT_ID).toBe("stackplanner");
  });

  test("ships an accessible, gradient-free product mark", () => {
    const svgPath = resolve(
      process.cwd(),
      "public",
      PRODUCT_MARK_PATH.slice(1),
    );
    const svg = readFileSync(svgPath, "utf8");

    expect(svg).toContain('<title id="title">StackPlanner 2.0</title>');
    expect(svg).toContain('<desc id="description">');
    expect(svg).not.toContain("linearGradient");
  });
});
