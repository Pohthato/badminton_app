import { describe, expect, it } from "vitest";
import { buildCoachingPrompt, buildRallySequenceModel, collectFailureCases, computeBiomechanicsMetrics, deriveProvisionalFourthCorner, evaluateAnalysisQuality, evaluateByVideoCondition, summarizeRallyPatterns, validateCalibration } from "./analysis";
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
    expect(metrics.map((metric) => metric.metric)).toEqual(expect.arrayContaining(["base width", "trunk angle", "knee flexion angle (2D)", "court coverage"]));
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

  it("summarizes session evidence quality and tactical pattern before giving coaching guidance", () => {
    const calibration = validateCalibration(completeCorners);
    const prompt = buildCoachingPrompt({
      player: "near",
      calibration,
      result: {
        processingVersion: "cv-1.0.0",
        calibration,
        quality: { usableFrameRatio: 0.83, poseTrackConfidence: 0.9, shuttleTrackConfidence: 0.77, courtReprojectionErrorPx: 14 },
        metrics: [
          { metric: "court coverage", value: 9.4, unit: "metres travelled", direction: "contextual", confidence: 0.92, evidenceFrames: [{ frame: 10, timeMs: 330, confidence: 0.92, source: "court" }] },
          { metric: "split-step timing", value: 220, unit: "ms", direction: "contextual", confidence: 0.89, evidenceFrames: [{ frame: 17, timeMs: 560, confidence: 0.89, source: "pose" }] },
        ],
        shots: [
          { frame: 1, timeMs: 200, confidence: 0.8, label: "clear", verified: true, direction: "cross", depth: "back", rallyIndex: 0 },
          { frame: 2, timeMs: 600, confidence: 0.82, label: "smash", verified: true, direction: "cross", depth: "back", rallyIndex: 0 },
          { frame: 3, timeMs: 850, confidence: 0.79, label: "drop", verified: true, direction: "straight", depth: "short", rallyIndex: 0 },
        ],
        shotDistribution: { clear: 1, smash: 1, drop: 1 },
        rallies: [{ index: 0, startMs: 0, endMs: 1100, contactCount: 3, verifiedShots: 3 }],
      },
      question: "What is the main tactical pattern?",
    });

    expect(prompt).toContain("Evidence quality");
    expect(prompt).toContain("Tactical pattern");
    expect(prompt).toContain("clear");
    expect(prompt).toContain("smash");
    expect(prompt).toContain("main tactical pattern");
  });

  it("derives rally-level summaries that reflect actual shot sequences and attack balance", () => {
    const calibration = validateCalibration(completeCorners);
    const summary = summarizeRallyPatterns({
      processingVersion: "cv-1.0.0",
      calibration,
      quality: { usableFrameRatio: 0.8, poseTrackConfidence: 0.8, shuttleTrackConfidence: 0.8 },
      metrics: [],
      shotDistribution: { clear: 2, smash: 1 },
      rallies: [
        { index: 0, startMs: 0, endMs: 900, contactCount: 2, verifiedShots: 2 },
        { index: 1, startMs: 1000, endMs: 1800, contactCount: 3, verifiedShots: 3 },
      ],
      shots: [
        { frame: 1, timeMs: 100, confidence: 0.8, label: "clear", verified: true, direction: "cross", depth: "back", rallyIndex: 0 },
        { frame: 2, timeMs: 200, confidence: 0.8, label: "smash", verified: true, direction: "cross", depth: "back", rallyIndex: 0 },
        { frame: 3, timeMs: 1200, confidence: 0.8, label: "clear", verified: true, direction: "straight", depth: "mid", rallyIndex: 1 },
        { frame: 4, timeMs: 1300, confidence: 0.8, label: "drop", verified: true, direction: "straight", depth: "short", rallyIndex: 1 },
        { frame: 5, timeMs: 1500, confidence: 0.8, label: "clear", verified: true, direction: "cross", depth: "back", rallyIndex: 1 },
      ],
    });

    expect(summary.rallyCount).toBe(2);
    expect(summary.dominantStroke).toBe("clear");
    expect(summary.attackRatio).toBeGreaterThan(0.5);
    expect(summary.averageRallyLengthMs).toBe(850);
  });

  it("produces a benchmark-ready evaluation summary with measurable strengths and risks", () => {
    const calibration = validateCalibration(completeCorners);
    const evaluation = evaluateAnalysisQuality({
      processingVersion: "cv-1.0.0",
      calibration,
      quality: { usableFrameRatio: 0.86, poseTrackConfidence: 0.9, shuttleTrackConfidence: 0.8, courtReprojectionErrorPx: 18 },
      metrics: [
        { metric: "court coverage", value: 9.4, unit: "metres travelled", direction: "contextual", confidence: 0.92, evidenceFrames: [{ frame: 10, timeMs: 330, confidence: 0.92, source: "court" }] },
        { metric: "split-step timing", value: 220, unit: "ms", direction: "contextual", confidence: 0.89, evidenceFrames: [{ frame: 17, timeMs: 560, confidence: 0.89, source: "pose" }] },
      ],
      shots: [
        { frame: 1, timeMs: 150, confidence: 0.8, label: "clear", verified: true, direction: "cross", depth: "back", rallyIndex: 0 },
        { frame: 2, timeMs: 620, confidence: 0.82, label: "smash", verified: true, direction: "cross", depth: "back", rallyIndex: 0 },
      ],
      shotDistribution: { clear: 1, smash: 1 },
      rallies: [{ index: 0, startMs: 0, endMs: 900, contactCount: 2, verifiedShots: 2 }],
    });

    expect(evaluation.overallScore).toBeGreaterThan(70);
    expect(evaluation.strengths.some((strength) => strength.toLowerCase().includes("pose"))).toBe(true);
    expect(evaluation.riskFlags.some((flag) => flag.toLowerCase().includes("reprojection"))).toBe(true);
    expect(evaluation.evidenceStatus).toMatch(/strong|moderate|weak/i);
  });

  it("evaluates the session by camera angle, lighting, and skill level with scenario-specific scoring", () => {
    const calibration = validateCalibration(completeCorners);
    const evaluation = evaluateByVideoCondition({
      processingVersion: "cv-1.0.0",
      calibration,
      quality: { usableFrameRatio: 0.82, poseTrackConfidence: 0.88, shuttleTrackConfidence: 0.74 },
      metrics: [
        { metric: "court coverage", value: 7.6, unit: "metres travelled", direction: "contextual", confidence: 0.81, evidenceFrames: [{ frame: 10, timeMs: 330, confidence: 0.81, source: "court" }] },
      ],
      shotDistribution: { clear: 2 },
      shots: [{ frame: 1, timeMs: 150, confidence: 0.72, label: "clear", verified: true, direction: "cross", depth: "back", rallyIndex: 0 }],
      rallies: [{ index: 0, startMs: 0, endMs: 900, contactCount: 1, verifiedShots: 1 }],
    }, {
      cameraAngle: "rear",
      lighting: "mixed",
      skillLevel: "intermediate",
    });

    expect(evaluation.cameraAngle.score).toBeGreaterThan(0);
    expect(evaluation.lighting.score).toBeGreaterThan(0);
    expect(evaluation.skillLevel.score).toBeGreaterThan(0);
    expect(evaluation.overallScore).toBeGreaterThan(0);
    expect(evaluation.factorSummary).toContain("camera angle");
  });

  it("builds a rally sequence model and extracts failure cases from weak evidence", () => {
    const calibration = validateCalibration(completeCorners.slice(0, 3));
    const result = {
      processingVersion: "cv-1.0.0",
      calibration,
      quality: { usableFrameRatio: 0.45, poseTrackConfidence: 0.55, shuttleTrackConfidence: 0.52 },
      metrics: [],
      shotDistribution: { clear: 1 },
      shots: [
        { frame: 1, timeMs: 100, confidence: 0.7, label: "clear", verified: true, direction: "cross", depth: "back", rallyIndex: 0 },
        { frame: 2, timeMs: 300, confidence: 0.68, label: "drop", verified: true, direction: "straight", depth: "short", rallyIndex: 0 },
        { frame: 3, timeMs: 650, confidence: 0.74, label: "smash", verified: true, direction: "cross", depth: "back", rallyIndex: 0 },
      ],
      rallies: [{ index: 0, startMs: 0, endMs: 700, contactCount: 3, verifiedShots: 3 }],
    };

    const sequence = buildRallySequenceModel(result);
    expect(sequence.sequence.length).toBe(3);
    expect(sequence.patterns.length).toBeGreaterThan(0);
    const failures = collectFailureCases(result);
    expect(failures.some((failure) => failure.type.includes("court") || failure.type.includes("pose") || failure.type.includes("shuttle"))).toBe(true);
  });
});
