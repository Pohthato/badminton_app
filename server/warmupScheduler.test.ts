import { afterEach, describe, expect, it, vi } from "vitest";
import {
  isKeepWarmRunning,
  startKeepWarmScheduler,
  stopKeepWarmScheduler,
} from "./warmupScheduler";
import { resetWorkerWarmupCooldown } from "./analysisWorker";

describe("keep-warm scheduler", () => {
  afterEach(() => {
    stopKeepWarmScheduler();
    resetWorkerWarmupCooldown();
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it("starts a timer that submits a warmup job on the first interval", async () => {
    vi.stubEnv("CV_WORKER_URL", "https://api.runpod.ai/v2/endpoint-1");
    vi.stubEnv("CV_WORKER_TOKEN", "runpod-test-key");
    vi.stubEnv("CV_WORKER_KEEP_WARM_ENABLED", "true");
    vi.stubEnv("CV_WORKER_KEEP_WARM_INTERVAL_MS", "20");
    const fetchMock = vi
      .fn()
      .mockImplementation(() =>
        new Response(JSON.stringify({ id: "warmup-1", status: "IN_QUEUE" }), { status: 200 })
      );
    vi.stubGlobal("fetch", fetchMock);

    vi.useFakeTimers();
    startKeepWarmScheduler();
    expect(isKeepWarmRunning()).toBe(true);

    // Advance past one interval to let the timer fire.
    await vi.advanceTimersByTimeAsync(25);
    expect(fetchMock).toHaveBeenCalled();
  });

  it("does not start when disabled", () => {
    vi.stubEnv("CV_WORKER_KEEP_WARM_ENABLED", "false");
    startKeepWarmScheduler();
    expect(isKeepWarmRunning()).toBe(false);
  });

  it("starts only one timer even when called twice", () => {
    vi.stubEnv("CV_WORKER_URL", "https://api.runpod.ai/v2/endpoint-1");
    vi.stubEnv("CV_WORKER_TOKEN", "runpod-test-key");
    vi.stubEnv("CV_WORKER_KEEP_WARM_ENABLED", "true");
    startKeepWarmScheduler();
    startKeepWarmScheduler();
    expect(isKeepWarmRunning()).toBe(true);
  });

  it("stops cleanly", () => {
    vi.stubEnv("CV_WORKER_KEEP_WARM_ENABLED", "true");
    startKeepWarmScheduler();
    stopKeepWarmScheduler();
    expect(isKeepWarmRunning()).toBe(false);
  });
});