import { getAnalysisWorkerJob, validateWorkerResult } from "./analysisWorker";
import { getAnalysisSessionById, updateAnalysisSessionFromWorker } from "./db";

function safeFailureReason(value: string | undefined) {
  if (!value) return "Computer-vision worker failed before returning a result.";
  // Do not persist request URLs, bearer headers, or a potentially enormous
  // traceback in a customer-facing session record.
  return value
    .replace(/https?:\/\/\S+/gi, "[redacted URL]")
    .replace(/bearer\s+[\w.-]+/gi, "Bearer [redacted]")
    .replace(/\s+/g, " ")
    .slice(0, 480);
}

/**
 * Polls the source of truth (RunPod), then persists only a validated result.
 * It is intentionally shared by the customer refresh action and webhook
 * completion route so webhook bodies are never trusted as analysis evidence.
 */
export async function reconcileAnalysisWorkerJob(analysisId: string) {
  const session = await getAnalysisSessionById(analysisId);
  if (!session) return { found: false as const, status: "failed" as const };
  if (!session.workerJobId) return { found: true as const, status: session.status, updated: false };

  const worker = await getAnalysisWorkerJob(session.workerJobId);
  const now = new Date();
  if (worker.status === "completed") {
    await updateAnalysisSessionFromWorker(session.id, session.workerJobId, {
      status: "completed",
      result: validateWorkerResult(worker.result!),
      failureReason: null,
      lastWorkerStatusAt: now,
    });
  } else if (worker.status === "failed") {
    await updateAnalysisSessionFromWorker(session.id, session.workerJobId, {
      status: "failed",
      failureReason: safeFailureReason(worker.error),
      lastWorkerStatusAt: now,
    });
  } else {
    await updateAnalysisSessionFromWorker(session.id, session.workerJobId, {
      status: worker.status,
      lastWorkerStatusAt: now,
    });
  }
  return { found: true as const, status: worker.status, updated: true, error: worker.status === "failed" ? safeFailureReason(worker.error) : undefined };
}
