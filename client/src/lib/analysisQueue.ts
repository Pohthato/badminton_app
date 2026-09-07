export type AnalysisQueueStatus =
  | "draft"
  | "queued"
  | "processing"
  | "completed"
  | "failed";

export const ANALYSIS_QUEUE_WARNING_MS = 5 * 60 * 1000;

export function getAnalysisQueueNotice(
  status: AnalysisQueueStatus,
  createdAt: Date | string | null | undefined,
  now = Date.now()
) {
  if ((status !== "queued" && status !== "processing") || !createdAt) {
    return { delayed: false, ageMinutes: 0 };
  }

  const createdAtMs = new Date(createdAt).getTime();
  if (!Number.isFinite(createdAtMs)) return { delayed: false, ageMinutes: 0 };

  const ageMs = Math.max(0, now - createdAtMs);
  return {
    delayed: ageMs >= ANALYSIS_QUEUE_WARNING_MS,
    ageMinutes: Math.floor(ageMs / 60_000),
  };
}
