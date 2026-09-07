export const courtCornerLabels = ["nearLeft", "nearRight", "farRight", "farLeft"] as const;
export type CourtCornerLabel = typeof courtCornerLabels[number];
export type CalibrationConfidence = "unverified" | "provisional" | "validated" | "image_space_only";
export type AnalysisStatus = "draft" | "queued" | "processing" | "completed" | "failed";

export type CourtCorner = {
  label: CourtCornerLabel;
  x: number;
  y: number;
};

export type Calibration = {
  corners: CourtCorner[];
  confidence: CalibrationConfidence;
  supportsCourtMapping: boolean;
  guidance: string;
};

export type FrameEvidence = {
  frame: number;
  timeMs: number;
  confidence: number;
  source: "pose" | "shuttle" | "racket" | "court";
};

export type MetricSummary = {
  metric: string;
  value: number;
  unit: string;
  direction: "higher_is_better" | "lower_is_better" | "contextual";
  confidence: number;
  evidenceFrames: FrameEvidence[];
};

export type AnalysisResult = {
  processingVersion: string;
  calibration: Calibration;
  quality: {
    usableFrameRatio: number;
    poseTrackConfidence: number;
    shuttleTrackConfidence: number;
    courtReprojectionErrorPx?: number;
  };
  metrics: MetricSummary[];
  shotDistribution: Record<string, number>;
  annotatedVideoStorageKey?: string;
  overlays?: Array<{
    timeMs: number;
    errors?: Partial<Record<"Skeleton" | "Shuttle" | "Racket" | "Court map", string>>;
    skeleton?: Array<{ x: number; y: number; confidence: number }>;
    shuttle?: { x: number; y: number; confidence: number };
    racket?: { grip: { x: number; y: number; confidence: number }; head: { x: number; y: number; confidence: number } };
    courtPosition?: { x: number; y: number; confidence: number };
  }>;
};

export type SkeletonFrame = {
  frame: number;
  timeMs: number;
  confidence: number;
  keypoints: Array<{ x: number; y: number; confidence: number }>;
};

export type CourtMappedFrame = {
  frame: number;
  timeMs: number;
  confidence: number;
  xMeters: number;
  yMeters: number;
};

export type AnalysisEvent = {
  type: "split_step" | "contact" | "recovery_complete";
  frame: number;
  timeMs: number;
  confidence: number;
};

export type MetricComputationInput = {
  source: "validated_worker";
  skeletonFrames: SkeletonFrame[];
  courtFrames?: CourtMappedFrame[];
  events?: AnalysisEvent[];
  fps: number;
};

export type ComputedMetric = {
  metric: string;
  value: number;
  unit: string;
  direction: MetricSummary["direction"];
  confidence: number;
  evidenceFrames: FrameEvidence[];
};

export function deriveProvisionalFourthCorner(corners: CourtCorner[]): CourtCorner | null {
  if (corners.length !== 3) return null;
  const byLabel = new Map(corners.map((corner) => [corner.label, corner]));
  const nearLeft = byLabel.get("nearLeft");
  const nearRight = byLabel.get("nearRight");
  const farRight = byLabel.get("farRight");
  const farLeft = byLabel.get("farLeft");

  // A fourth corner is not uniquely determined by three image points alone.
  // This rectangle-topology estimate is deliberately provisional: the worker
  // must validate it against court-line evidence before metre-based analysis.
  if (nearLeft && nearRight && farRight && !farLeft) {
    return { label: "farLeft", x: nearLeft.x + farRight.x - nearRight.x, y: nearLeft.y + farRight.y - nearRight.y };
  }
  if (nearLeft && nearRight && farLeft && !farRight) {
    return { label: "farRight", x: nearRight.x + farLeft.x - nearLeft.x, y: nearRight.y + farLeft.y - nearLeft.y };
  }
  if (nearLeft && farRight && farLeft && !nearRight) {
    return { label: "nearRight", x: nearLeft.x + farRight.x - farLeft.x, y: nearLeft.y + farRight.y - farLeft.y };
  }
  if (nearRight && farRight && farLeft && !nearLeft) {
    return { label: "nearLeft", x: nearRight.x + farLeft.x - farRight.x, y: nearRight.y + farLeft.y - farRight.y };
  }
  return null;
}

function distance(a: { x: number; y: number }, b: { x: number; y: number }) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function angleDegrees(a: { x: number; y: number }, b: { x: number; y: number }, c: { x: number; y: number }) {
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };
  const denominator = Math.hypot(ab.x, ab.y) * Math.hypot(cb.x, cb.y);
  if (!denominator) return null;
  return Math.acos(Math.max(-1, Math.min(1, (ab.x * cb.x + ab.y * cb.y) / denominator))) * 180 / Math.PI;
}

function frameEvidence(frame: SkeletonFrame, source: FrameEvidence["source"] = "pose"): FrameEvidence {
  return { frame: frame.frame, timeMs: frame.timeMs, confidence: frame.confidence, source };
}

export function computeBiomechanicsMetrics(input: MetricComputationInput): ComputedMetric[] {
  if (input.source !== "validated_worker") return [];
  const usable = input.skeletonFrames.filter((frame) => frame.confidence >= 0.55 && frame.keypoints.length >= 17);
  if (!usable.length) return [];

  const baseWidths = usable.map((frame) => distance(frame.keypoints[15], frame.keypoints[16])).filter(Number.isFinite);
  const trunkAngles = usable.map((frame) => angleDegrees(frame.keypoints[5], frame.keypoints[11], frame.keypoints[12])).filter((value): value is number => value !== null);
  const kneeAngles = usable.flatMap((frame) => [angleDegrees(frame.keypoints[11], frame.keypoints[13], frame.keypoints[15]), angleDegrees(frame.keypoints[12], frame.keypoints[14], frame.keypoints[16])]).filter((value): value is number => value !== null);

  const metrics: ComputedMetric[] = [];
  if (baseWidths.length) metrics.push({ metric: "base width", value: baseWidths.reduce((a, b) => a + b, 0) / baseWidths.length, unit: "normalized body-width", direction: "contextual", confidence: usable.reduce((a, frame) => a + frame.confidence, 0) / usable.length, evidenceFrames: usable.slice(0, 4).map((frame) => frameEvidence(frame)) });
  if (trunkAngles.length) metrics.push({ metric: "trunk angle", value: trunkAngles.reduce((a, b) => a + b, 0) / trunkAngles.length, unit: "degrees", direction: "contextual", confidence: usable.reduce((a, frame) => a + frame.confidence, 0) / usable.length, evidenceFrames: usable.slice(0, 4).map((frame) => frameEvidence(frame)) });
  if (kneeAngles.length) metrics.push({ metric: "knee loading", value: kneeAngles.reduce((a, b) => a + b, 0) / kneeAngles.length, unit: "degrees", direction: "contextual", confidence: usable.reduce((a, frame) => a + frame.confidence, 0) / usable.length, evidenceFrames: usable.slice(0, 4).map((frame) => frameEvidence(frame)) });

  const events = input.events?.filter((event) => event.confidence >= 0.65) ?? [];
  const splitSteps = events.filter((event) => event.type === "split_step");
  if (splitSteps.length) {
    metrics.push({ metric: "split-step timing", value: splitSteps.reduce((total, event) => total + event.timeMs, 0) / splitSteps.length, unit: "ms from rally origin", direction: "contextual", confidence: splitSteps.reduce((total, event) => total + event.confidence, 0) / splitSteps.length, evidenceFrames: splitSteps.slice(0, 6).map((event) => ({ frame: event.frame, timeMs: event.timeMs, confidence: event.confidence, source: "pose" })) });
  }

  const contacts = events.filter((event) => event.type === "contact");
  const recoveries = events.filter((event) => event.type === "recovery_complete");
  const recoveryDurations = contacts.flatMap((contact) => {
    const recovery = recoveries.find((event) => event.timeMs > contact.timeMs);
    return recovery ? [recovery.timeMs - contact.timeMs] : [];
  });
  if (recoveryDurations.length) {
    metrics.push({ metric: "recovery time", value: recoveryDurations.reduce((total, value) => total + value, 0) / recoveryDurations.length, unit: "ms", direction: "lower_is_better", confidence: Math.min(contacts.length, recoveries.length) / Math.max(contacts.length, recoveries.length), evidenceFrames: contacts.slice(0, 4).flatMap((contact) => { const recovery = recoveries.find((event) => event.timeMs > contact.timeMs); return recovery ? [{ frame: contact.frame, timeMs: contact.timeMs, confidence: contact.confidence, source: "pose" as const }, { frame: recovery.frame, timeMs: recovery.timeMs, confidence: recovery.confidence, source: "pose" as const }] : []; }) });
  }

  const courtFrames = input.courtFrames?.filter((frame) => frame.confidence >= 0.65) ?? [];
  if (courtFrames.length > 1) {
    const distances = courtFrames.slice(1).map((frame, index) => distance({ x: frame.xMeters, y: frame.yMeters }, { x: courtFrames[index].xMeters, y: courtFrames[index].yMeters })).filter(Number.isFinite);
    const coverage = distances.reduce((a, b) => a + b, 0);
    metrics.push({ metric: "court coverage", value: coverage, unit: "metres travelled", direction: "contextual", confidence: courtFrames.reduce((a, frame) => a + frame.confidence, 0) / courtFrames.length, evidenceFrames: courtFrames.slice(0, 4).map((frame) => ({ frame: frame.frame, timeMs: frame.timeMs, confidence: frame.confidence, source: "court" })) });
  }

  return metrics;
}

export type CoachingContext = {
  player: "near" | "far";
  sessionGoal?: string;
  calibration: Calibration;
  result: AnalysisResult;
  question?: string;
};

function isFinitePoint(point: CourtCorner) {
  return Number.isFinite(point.x) && Number.isFinite(point.y) && point.x >= 0 && point.x <= 100 && point.y >= 0 && point.y <= 100;
}

function signedArea(corners: CourtCorner[]) {
  return corners.reduce((total, point, index) => {
    const next = corners[(index + 1) % corners.length];
    return total + point.x * next.y - next.x * point.y;
  }, 0) / 2;
}

export function validateCalibration(corners: CourtCorner[]): Calibration {
  if (corners.length < 3 || corners.length > 4) {
    throw new Error("Calibration requires three or four named court intersections.");
  }
  if (new Set(corners.map((corner) => corner.label)).size !== corners.length) {
    throw new Error("Each calibration corner must have a distinct semantic label.");
  }
  if (!corners.every(isFinitePoint)) {
    throw new Error("Court intersections must be finite points within the video frame.");
  }
  if (Math.abs(signedArea(corners)) < 1) {
    throw new Error("Court intersections are too close together to define meaningful geometry.");
  }

  const hasAllCorners = courtCornerLabels.every((label) => corners.some((corner) => corner.label === label));
  return {
    corners,
    confidence: hasAllCorners ? "validated" : "provisional",
    supportsCourtMapping: hasAllCorners,
    guidance: hasAllCorners
      ? "Four labelled correspondences can support a planar court homography. Reprojection error still determines whether the worker accepts it."
      : "Three intersections preserve a provisional calibration. The worker may infer the missing corner from court-line and vanishing-point evidence; metre-based movement metrics require an accepted reprojection error and camera-confidence result.",
  };
}

export function buildCoachingPrompt(context: CoachingContext) {
  const result = context.result;
  const trustedMetrics = result.metrics.filter((metric) => metric.confidence >= 0.7 && metric.evidenceFrames.length > 0 && !(context.calibration.confidence === "image_space_only" && metric.metric === "court coverage"));
  const metricEvidence = trustedMetrics.map((metric) => ({
    metric: metric.metric,
    value: metric.value,
    unit: metric.unit,
    direction: metric.direction,
    confidence: metric.confidence,
    frames: metric.evidenceFrames.map((frame) => ({ frame: frame.frame, timeMs: frame.timeMs, confidence: frame.confidence, source: frame.source })),
  }));

  return [
    "You are OmniCourt Coach, a high-performance badminton analyst working from computer-vision evidence. Your job is to convert evidence into precise, practical and intellectually honest coaching guidance.",
    "Do not diagnose injuries, make medical claims, claim 3D certainty from a monocular recording, invent shots or events, or compare the player to a professional norm unless that norm is included in the supplied evidence. If evidence is insufficient, say exactly what could not be verified and what capture setup would improve it.",
    "Distinguish observation from interpretation. A recommendation must be tied to a named metric and at least one evidence timestamp. Weight high-confidence evidence more heavily. Never use a player bounding box as evidence; the usable visual tracks are skeleton, shuttle, racket, and court-plane mapping.",
    "Return plain text only, with exactly these titled sections: Evidence quality; Highest-leverage finding; Technique; Movement and recovery; Tactical pattern; Two drills; Next capture. Use short complete paragraphs, no Markdown syntax, no motivational filler, no more than two drills.",
    `Selected player: ${context.player}. Session goal: ${context.sessionGoal ?? "general singles performance"}.`,
    `Calibration confidence: ${context.calibration.confidence}. Court mapping accepted: ${context.calibration.supportsCourtMapping}. Calibration note: ${context.calibration.guidance}`,
    context.calibration.confidence === "image_space_only" ? "Image-space-only mode: do not make exact metre, court-position, or camera-pose claims. Restrict movement feedback to visible pose timing, relative direction, and frame timestamps." : "Court-space claims may be used only when supported by accepted worker calibration and confidence-gated evidence.",
    `Processing quality: usable-frame ratio ${result.quality.usableFrameRatio}; pose track confidence ${result.quality.poseTrackConfidence}; shuttle track confidence ${result.quality.shuttleTrackConfidence}; court reprojection error px ${result.quality.courtReprojectionErrorPx ?? "not available"}.`,
    `Trusted metric evidence: ${JSON.stringify(metricEvidence)}.`,
    `Shot distribution: ${JSON.stringify(result.shotDistribution)}.`,
    context.question ? `Athlete question: ${context.question}` : "Athlete question: Provide a first coaching assessment from this session.",
  ].join("\n\n");
}
