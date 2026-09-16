import { describe, expect, it } from "vitest";

describe("configured RunPod worker", () => {
  const workerUrl = process.env.CV_WORKER_URL;
  const token = process.env.CV_WORKER_TOKEN;
  it(
    "accepts the configured credential at the endpoint health operation",
    { skip: !workerUrl || !token },
    async () => {
      const response = await fetch(`${workerUrl!.replace(/\/$/, "")}/health`, {
        headers: { Authorization: `Bearer ${token}` },
      });

      expect(response.status).not.toBe(401);
      expect(response.status).not.toBe(403);
      expect(response.status).toBeLessThan(500);
    },
    20_000
  );
});
