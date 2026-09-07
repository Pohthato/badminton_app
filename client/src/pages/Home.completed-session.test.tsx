import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";

const completedSession = {
  id: "session-123",
  status: "completed",
  result: {
    calibration: { confidence: "image_space_only", supportsCourtMapping: false },
    overlays: [{ timeMs: 0, errors: { Racket: "occluded by body" }, skeleton: [{ x: 50, y: 45, confidence: 0.9 }] }],
  },
};

vi.mock("@/_core/hooks/useAuth", () => ({ useAuth: () => ({ isAuthenticated: true }) }));
vi.mock("@/components/AIChatBox", () => ({ AIChatBox: () => React.createElement("div", null, "DeepSeek coach") }));
vi.mock("@/lib/trpc", () => {
  const mutation = () => ({ mutateAsync: vi.fn(), isPending: false });
  return {
    trpc: {
      upload: { prepareVideo: { useMutation: mutation } },
      analysis: {
        createDraft: { useMutation: mutation },
        submit: { useMutation: mutation },
        refresh: { useMutation: mutation },
        get: { useQuery: () => ({ data: completedSession, refetch: vi.fn() }) },
      },
      coach: { chat: { useMutation: mutation } },
    },
  };
});

import Home from "./Home";

describe("Home completed-session integration", () => {
  beforeEach(() => {
    vi.stubGlobal("URL", { createObjectURL: vi.fn(() => "blob:source"), revokeObjectURL: vi.fn() });
  });

  it("renders the completed annotated-review path from session result overlays", () => {
    const markup = renderToStaticMarkup(<Home />);
    expect(markup).toContain("Image-space only");
    expect(markup).toContain("POSE VERIFIED");
    expect(markup).toContain("RACKET ERROR");
    expect(markup).toContain("COURT NO DATA");
    expect(markup).toContain("DeepSeek coach");
  });
});
