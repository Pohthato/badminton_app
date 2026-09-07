import { describe, expect, it } from "vitest";
import { getCalibrationPointLabel, getEstimatedCornerLabel, getObservedCalibrationPoints, getProvisionalFourthCorner, resolveCalibrationPoints } from "./calibration";

describe("arbitrary three-point court calibration", () => {
  it("keeps click markers neutral instead of assigning Near/Far labels", () => {
    expect(getCalibrationPointLabel(0)).toBe("Observed point 1");
    expect(getCalibrationPointLabel(1)).toBe("Observed point 2");
    expect(getEstimatedCornerLabel()).toBe("Estimated missing corner");
  });
  it("accepts three visible points in any order and infers the missing semantic corner", () => {
    const points = [
      { x: 20, y: 25 },
      { x: 25, y: 75 },
      { x: 70, y: 30 },
    ];
    const resolved = resolveCalibrationPoints(points);
    const observed = getObservedCalibrationPoints(points);
    const estimate = getProvisionalFourthCorner(points);

    expect(observed).toHaveLength(3);
    expect(new Set(observed.map((point) => point.key)).size).toBe(3);
    expect(estimate?.key).toBe("nearRight");
    expect(resolved).toHaveLength(4);
    expect(estimate?.inferred).toBe(true);
  });

  it("does not require near-left and handles a cropped far-left corner", () => {
    const points = [
      { x: 85, y: 80 },
      { x: 70, y: 30 },
      { x: 25, y: 75 },
    ];
    const estimate = getProvisionalFourthCorner(points);

    expect(estimate?.key).toBe("farLeft");
    expect(estimate?.label).toContain("estimated");
  });
});
