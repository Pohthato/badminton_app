import { describe, expect, it } from "vitest";
import { buildCalibrationPayload } from "./calibrationPayload";

describe("calibration submission payload", () => {
  it("produces the same semantic payload regardless of click order", () => {
    const visiblePoints = [
      { x: 20, y: 25 },
      { x: 25, y: 75 },
      { x: 70, y: 30 },
    ];
    const firstOrder = buildCalibrationPayload(visiblePoints);
    const differentOrder = buildCalibrationPayload([visiblePoints[1], visiblePoints[2], visiblePoints[0]]);

    expect(differentOrder).toEqual(firstOrder);
    expect(firstOrder.map((point) => point.label)).toEqual(["farLeft", "farRight", "nearLeft"]);
    expect(firstOrder).not.toContainEqual({ label: "nearRight", x: 25, y: 75 });
  });
});
