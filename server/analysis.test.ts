import { describe, expect, it } from "vitest";
import { buildCoachingPrompt, computeBiomechanicsMetrics, deriveProvisionalFourthCorner, validateCalibration } from "./analysis";
import { validateWorkerResult } from "./analysisWorker";

const completeCorners = [
  { label: "nearLeft" as const, x: 12, y: 86 },
  { label: "nearRight" as const, x: 88, y: 86 },
  { label: "farRight" as const, x: 73, y: 18 },
  { label: "farLeft" as const, x: 27, y: 18 },
];

describe("court calibration contracts", () => {
  it("derives a provisional fourth corner from rectangle topology without claiming it is validated geometry", () => {
    const inferred = deriveProvisionalFourthCorner([
      { label: "nearLeft", x: 12, y: 86 },
      { label: "nearRight", x: 88, y: 86 },
      { label: "farRight", x: 73, y: 18 },
    ]);
    expect(inferred).toEqual({ label: "farLeft", x: -3, y: 18 });
  });
  it("requires four named correspondences before it accepts metre-based court mapping", () => {
    const calibration = validateCalibration(completeCorners);
    expect(calibration.confidence).toBe("validated");
    expect(calibration.supportsCourtMapping).toBe(true);
  });

  it("preserves three-corner calibration as provisional rather than pretending the missing correspondence is known", () => {
    const calibration = validateCalibration(completeCorners.slice(0, 3));
    expect(calibration.confidence).toBe("provisional");
    expect(calibration.supportsCourtMapping).toBe(false);
  });
});

describe("frame-level biomechanics metrics", () => {
  it("computes confidence-gated posture and court coverage metrics from worker tracks", () => {
    const keypoints = Array.from({ length: 17 }, (_, index) => ({ x: index, y: index + 1, confidence: 0.9 }));
    const metrics = computeBiomechanicsMetrics({
      source: "validated_worker",
      fps: 30,
      skeletonFrames: [
        { frame: 1, timeMs: 33, confidence: 0.92, keypoints },
        { frame: 2, timeMs: 66, confidence: 0.91, keypoints: keypoints.map((point) => ({ ...point, x: point.x + 1 })) },
      ],
      courtFrames: [
        { frame: 1, timeMs: 33, confidence: 0.8, xMeters: 1, yMeters: 1 },
        { frame: 2, timeMs: 66, confidence: 0.8, xMeters: 2, yMeters: 2 },
      ],
    });
    expect(metrics.map((metric) => metric.metric)).toEqual(expect.arrayContaining(["base width", "trunk angle", "knee loading", "court coverage"]));
    expect(metrics.find((metric) => metric.metric === "court coverage")?.value).toBeCloseTo(Math.sqrt(2));
  });

  it("emits split-step and recovery evidence only for confident worker events", () => {
    const keypoints = Array.from({ length: 17 }, (_, index) => ({ x: index, y: index + 1, confidence: 0.9 }));
    const metrics = computeBiomechanicsMetrics({
      source: "validated_worker",
      fps: 30,
      skeletonFrames: [{ frame: 1, timeMs: 33, confidence: 0.9, keypoints }],
      events: [
        { type: "split_step", frame: 12, timeMs: 400, confidence: 0.91 },
        { type: "contact", frame: 30, timeMs: 1000, confidence: 0.9 },
        { type: "recovery_complete", frame: 45, timeMs: 1500, confidence: 0.88 },
        { type: "split_step", frame: 50, timeMs: 1700, confidence: 0.2 },
      ],
    });
    expect(metrics.find((metric) => metric.metric === "split-step timing")?.value).toBe(400);
    const recovery = metrics.find((metric) => metric.metric === "recovery time");
    expect(recovery?.value).toBe(500);
    expect(recovery?.evidenceFrames.map((frame) => frame.frame)).toEqual([30, 45]);
  });

  it("returns no biomechanics metrics when tracks are not explicitly accepted by the worker", () => {
    const keypoints = Array.from({ length: 17 }, (_, index) => ({ x: index, y: index + 1, confidence: 0.9 }));
    expect(computeBiomechanicsMetrics({ source: "unverified" as "validated_worker", fps: 30, skeletonFrames: [{ frame: 1, timeMs: 33, confidence: 0.9, keypoints }] })).toEqual([]);
  });
});

describe("completed worker result contract", () => {
  it("preserves verified overlay packets and explicit layer errors for the review renderer", () => {
    const calibration = validateCalibration(completeCorners);
    const result = validateWorkerResult({
      processingVersion: "cv-1.0.0",
      calibration,
      quality: { usableFrameRatio: 0.91, poseTrackConfidence: 0.88, shuttleTrackConfidence: 0.61 },
      metrics: [],
      shotDistribution: {},
      overlays: [{ timeMs: 1000, errors: { Racket: "occluded by body" }, skeleton: [{ x: 50, y: 50, confidence: 0.9 }] }],
    });
    expect(result.overlays?.[0].skeleton?.[0].confidence).toBe(0.9);
    expect(result.overlays?.[0].errors?.Racket).toBe("occluded by body");
  });
});

describe("calibration state transitions and fallback gating", () => {
  it("moves from provisional to validated and explicitly supports image-space-only rejection", () => {
    const provisional = validateCalibration(completeCorners.slice(0, 3));
    const validated = validateCalibration(completeCorners);
    expect(provisional.confidence).toBe("provisional");
    expect(validated.confidence).toBe("validated");
    const rejected = { ...provisional, confidence: "image_space_only" as const, supportsCourtMapping: false, guidance: "Court-line evidence was not reliable enough." };
    const prompt = buildCoachingPrompt({ player: "near", calibration: rejected, result: { processingVersion: "cv-1.0.0", calibration: rejected, quality: { usableFrameRatio: 0.7, poseTrackConfidence: 0.85, shuttleTrackConfidence: 0.4 }, metrics: [{ metric: "court coverage", value: 12, unit: "metres travelled", direction: "contextual", confidence: 0.95, evidenceFrames: [{ frame: 1, timeMs: 33, confidence: 0.95, source: "court" }] }], shotDistribution: {} } });
    expect(prompt).toContain("Image-space-only mode");
    expect(prompt).not.toContain("metres travelled");
  });
});

describe("coaching prompt construction", () => {
  it("keeps only high-confidence, timestamped metric evidence in the coaching payload", () => {
    const calibration = validateCalibration(completeCorners);
    const prompt = buildCoachingPrompt({
      player: "near",
      calibration,
      result: {
        processingVersion: "cv-1.0.0",
        calibration,
        quality: { usableFrameRatio: 0.88, poseTrackConfidence: 0.92, shuttleTrackConfidence: 0.72 },
        metrics: [
          { metric: "split step timing", value: 105, unit: "ms", direction: "contextual", confidence: 0.9, evidenceFrames: [{ frame: 72, timeMs: 2400, confidence: 0.94, source: "pose" }] },
          { metric: "racket path", value: 0, unit: "score", direction: "contextual", confidence: 0.43, evidenceFrames: [{ frame: 22, timeMs: 733, confidence: 0.43, source: "racket" }] },
        ],
        shotDistribution: { clear: 4 },
        overlays: [{ timeMs: 2400, skeleton: [{ x: 50, y: 34, confidence: 0.92 }], shuttle: { x: 74, y: 23, confidence: 0.77 } }],
      },
      question: "Where should I focus?",
    });

    expect(prompt).toContain("split step timing");
    expect(prompt).not.toContain("racket path");
    expect(prompt).toContain("frame\":72");
    expect(prompt).toContain("Do not diagnose injuries");
  });
});
