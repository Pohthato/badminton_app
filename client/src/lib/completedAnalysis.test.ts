import { describe, expect, it } from "vitest";
import { getCompletedOverlayFrames } from "./completedAnalysis";

describe("completed Home analysis flow", () => {
  it("passes completed worker overlays to the review renderer and blocks incomplete sessions", () => {
    const overlays = [{ timeMs: 1_000, errors: { Racket: "occluded" }, skeleton: [{ x: 50, y: 50, confidence: 0.9 }] }];
    expect(getCompletedOverlayFrames({ status: "queued", result: { overlays } })).toEqual([]);
    expect(getCompletedOverlayFrames({ status: "completed", result: { overlays } })).toEqual(overlays);
    expect(getCompletedOverlayFrames({ status: "completed", result: null })).toEqual([]);
  });
});
