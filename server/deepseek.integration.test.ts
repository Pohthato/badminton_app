import { describe, expect, it } from "vitest";

describe("DeepSeek coaching integration", () => {
  const apiKey = process.env.DEEPSEEK_API_KEY;
  it(
    "authenticates against the model catalog with the configured server secret",
    { skip: !apiKey },
    async () => {
      const response = await fetch("https://api.deepseek.com/v1/models", {
        headers: {
          Authorization: `Bearer ${apiKey}`,
        },
      });

      expect(response.status, "DeepSeek rejected the configured server credential").toBe(200);
      const payload = await response.json() as { data?: unknown[] };
      expect(Array.isArray(payload.data), "DeepSeek returned an unexpected models payload").toBe(true);
    },
    20_000
  );
});
