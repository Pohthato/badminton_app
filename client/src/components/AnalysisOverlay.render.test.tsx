import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { AnalysisOverlay } from "./AnalysisOverlay";

describe("AnalysisOverlay completed-session render", () => {
  it("renders verified, error, and no-data states from a completed worker frame", () => {
    const markup = renderToStaticMarkup(
      <AnalysisOverlay
        currentTimeMs={1_000}
        frames={[{
          timeMs: 1_000,
          errors: { Racket: "occluded by body" },
          skeleton: [{ x: 50, y: 45, confidence: 0.92 }],
          shuttle: { x: 72, y: 25, confidence: 0.8 },
        }]}
        layers={{ Skeleton: true, Shuttle: true, Racket: true, "Court map": true }}
      />,
    );

    expect(markup).toContain("POSE VERIFIED");
    expect(markup).toContain("SHUTTLE VERIFIED");
    expect(markup).toContain("RACKET ERROR");
    expect(markup).toContain("COURT NO DATA");
  });
});
