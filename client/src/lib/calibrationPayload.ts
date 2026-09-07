import { getObservedCalibrationPoints, type CalibrationPoint } from "./calibration";

export function buildCalibrationPayload(points: CalibrationPoint[]) {
  return getObservedCalibrationPoints(points)
    .map((point) => ({ label: point.key, x: point.x, y: point.y }))
    .sort((a, b) => a.label.localeCompare(b.label));
}
