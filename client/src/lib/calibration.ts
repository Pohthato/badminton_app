export const courtCornerKeys = ["nearLeft", "nearRight", "farRight", "farLeft"] as const;
export type CourtCornerKey = typeof courtCornerKeys[number];

export type CalibrationPoint = {
  x: number;
  y: number;
};

export type ResolvedCalibrationPoint = CalibrationPoint & {
  key: CourtCornerKey;
  label: string;
  inferred?: boolean;
};

const displayLabels: Record<CourtCornerKey, string> = {
  nearLeft: "Near left",
  nearRight: "Near right",
  farRight: "Far right",
  farLeft: "Far left",
};

function clamp(value: number) {
  return Math.max(5, Math.min(95, value));
}

function angleAt(point: CalibrationPoint, first: CalibrationPoint, second: CalibrationPoint) {
  const a = { x: first.x - point.x, y: first.y - point.y };
  const b = { x: second.x - point.x, y: second.y - point.y };
  const denominator = Math.hypot(a.x, a.y) * Math.hypot(b.x, b.y);
  if (!denominator) return 0;
  return Math.acos(Math.max(-1, Math.min(1, (a.x * b.x + a.y * b.y) / denominator)));
}

/**
 * Three points do not uniquely determine a projective quadrilateral. This
 * intentionally conservative estimate uses the most right-angled observed
 * vertex as the shared corner and marks the fourth point as provisional. The
 * GPU worker must validate it against court-line/vanishing-point evidence.
 */
export function inferMissingCorner(points: CalibrationPoint[]): CalibrationPoint | null {
  if (points.length !== 3) return null;
  const candidates = points.map((point, index) => {
    const others = points.filter((_, otherIndex) => otherIndex !== index);
    return { index, angle: angleAt(point, others[0], others[1]) };
  });
  const sharedIndex = candidates.sort((a, b) => b.angle - a.angle)[0]?.index ?? 1;
  const shared = points[sharedIndex];
  const adjacent = points.filter((_, index) => index !== sharedIndex);
  if (!shared || !adjacent[0] || !adjacent[1]) return null;
  return {
    x: clamp(adjacent[0].x + adjacent[1].x - shared.x),
    y: clamp(adjacent[0].y + adjacent[1].y - shared.y),
  };
}

function assignSemanticLabels(points: Array<CalibrationPoint & { inferred?: boolean }>): ResolvedCalibrationPoint[] {
  const sortedByDepth = [...points].sort((a, b) => a.y - b.y);
  const far = sortedByDepth.slice(0, 2).sort((a, b) => a.x - b.x);
  const near = sortedByDepth.slice(-2).sort((a, b) => a.x - b.x);
  const assignments: Array<[CourtCornerKey, CalibrationPoint & { inferred?: boolean } | undefined]> = [
    ["nearLeft", near[0]],
    ["nearRight", near[1]],
    ["farRight", far[1]],
    ["farLeft", far[0]],
  ];
  return assignments.flatMap(([key, point]) => point ? [{
    ...point,
    key,
    label: point.inferred ? `${displayLabels[key]} · estimated` : displayLabels[key],
  }] : []);
}

export function resolveCalibrationPoints(points: CalibrationPoint[]): ResolvedCalibrationPoint[] {
  if (points.length < 3) return [];
  const estimated = points.length === 3 ? inferMissingCorner(points) : null;
  return assignSemanticLabels([...points, ...(estimated ? [{ ...estimated, inferred: true }] : [])]);
}

export function getObservedCalibrationPoints(points: CalibrationPoint[]): ResolvedCalibrationPoint[] {
  return resolveCalibrationPoints(points).filter((point) => !point.inferred);
}

export function getProvisionalFourthCorner(points: CalibrationPoint[]): ResolvedCalibrationPoint | null {
  return resolveCalibrationPoints(points).find((point) => point.inferred) ?? null;
}

export function getCalibrationPointLabel(index: number) {
  return `Observed point ${index + 1}`;
}

export function getEstimatedCornerLabel() {
  return "Estimated missing corner";
}

export { displayLabels };
