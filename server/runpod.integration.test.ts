import { describe, expect, it } from "vitest";

describe("configured RunPod worker", () => {
  it("accepts the configured credential at the endpoint health operation", async () => {
    const workerUrl = process.env.CV_WORKER_URL;
    const token = process.env.CV_WORKER_TOKEN;
    expect(workerUrl, "CV_WORKER_URL must be configured").toBeTruthy();
    expect(token, "CV_WORKER_TOKEN must be configured").toBeTruthy();

    const response = await fetch(`${workerUrl!.replace(/\/$/, "")}/health`, {
      headers: { Authorization: `Bearer ${token}` },
    });

    expect(response.status).not.toBe(401);
    expect(response.status).not.toBe(403);
    expect(response.status).toBeLessThan(500);
  }, 20_000);
});
