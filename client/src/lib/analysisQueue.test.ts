import { describe, expect, it } from "vitest";
import {
  ANALYSIS_QUEUE_WARNING_MS,
  getAnalysisQueueNotice,
} from "./analysisQueue";

describe("analysis queue notice", () => {
  const now = Date.parse("2026-09-04T21:20:00.000Z");

  it("does not warn during a normal queued startup", () => {
    expect(
      getAnalysisQueueNotice("queued", new Date(now - 2 * 60_000), now)
    ).toEqual({ delayed: false, ageMinutes: 2 });
  });

  it("warns after the queue threshold", () => {
    expect(
      getAnalysisQueueNotice(
        "queued",
        new Date(now - ANALYSIS_QUEUE_WARNING_MS),
        now
      )
    ).toEqual({ delayed: true, ageMinutes: 5 });
    expect(
      getAnalysisQueueNotice("processing", new Date(now - 8 * 60_000), now)
    ).toEqual({ delayed: true, ageMinutes: 8 });
  });

  it("does not warn for completed sessions or invalid timestamps", () => {
    expect(
      getAnalysisQueueNotice("completed", new Date(now - 30 * 60_000), now)
    ).toEqual({ delayed: false, ageMinutes: 0 });
    expect(getAnalysisQueueNotice("queued", "not-a-date", now)).toEqual({
      delayed: false,
      ageMinutes: 0,
    });
  });
});
