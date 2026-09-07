import { describe, expect, it } from "vitest";
import { getOverlayLayerStates } from "./AnalysisOverlay";

describe("completed worker overlay rendering contract", () => {
  it("maps verified result packets into explicit per-layer states", () => {
    const states = getOverlayLayerStates(
      {
        timeMs: 1_000,
        errors: { Racket: "occluded by body" },
        skeleton: [{ x: 50, y: 45, confidence: 0.92 }],
        shuttle: { x: 72, y: 25, confidence: 0.8 },
      },
      [{ x: 50, y: 45, confidence: 0.92 }],
      { x: 72, y: 25, confidence: 0.8 },
      undefined,
      undefined,
    );

    expect(states.find((state) => state.key === "Skeleton")?.status).toBe("VERIFIED");
    expect(states.find((state) => state.key === "Shuttle")?.status).toBe("VERIFIED");
    expect(states.find((state) => state.key === "Racket")?.status).toBe("ERROR");
    expect(states.find((state) => state.key === "Racket")?.error).toBe("occluded by body");
    expect(states.find((state) => state.key === "Court map")?.status).toBe("NO DATA");
  });

  it("keeps low-confidence worker points visible as uncertainty rather than promoting them to verified evidence", () => {
    const states = getOverlayLayerStates(
      { timeMs: 1_000, skeleton: [{ x: 50, y: 45, confidence: 0.2 }] },
      [{ x: 50, y: 45, confidence: 0.2 }],
    );
    expect(states.find((state) => state.key === "Skeleton")?.status).toBe("UNCERTAIN");
  });
});
