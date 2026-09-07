import type { VerifiedOverlayFrame } from "@/components/AnalysisOverlay";

export type CompletedAnalysisSession = {
  status?: string;
  result?: { overlays?: VerifiedOverlayFrame[] } | null;
};

export function getCompletedOverlayFrames(session: CompletedAnalysisSession | null | undefined) {
  if (session?.status !== "completed") return [];
  return session.result?.overlays ?? [];
}
