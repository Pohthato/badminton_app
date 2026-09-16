import { submitWorkerWarmup } from "./analysisWorker";

/**
 * Keep-warm scheduler for the serverless GPU endpoint.
 *
 * RunPod cold starts take minutes because a worker image must boot, import
 * torch, and load the .pt weights. A short, cheap warmup job that only loads
 * models keeps ONE worker warm so the first real analysis skips the cold start
 * while the GPU idles at the (low) min-worker cost. The module-level cooldown
 * in analysisWorker.ts is shared with the client-triggered warmup, so the
 * scheduler and the UI never double-pay for back-to-back warmup jobs.
 *
 * Schedule:
 * - warmup interval is shorter than the endpoint idle timeout (default 12s vs
 *   a 10-15 min idle timeout). Each warmup job runs in seconds because models
 *   are already loaded in that worker.
 * - Disable with CV_WORKER_KEEP_WARM_ENABLED=false (e.g. overnight) to stop
 *   idle GPU cost. Remove that env flag at product hours to warm back up.
 */

let keepWarmTimer: ReturnType<typeof setInterval> | null = null;

function keepWarmEnabled(): boolean {
  return process.env.CV_WORKER_KEEP_WARM_ENABLED !== "false";
}

function keepWarmIntervalMs(): number {
  return Number(process.env.CV_WORKER_KEEP_WARM_INTERVAL_MS ?? 12_000);
}

export function isKeepWarmRunning(): boolean {
  return keepWarmTimer !== null;
}

/**
 * Starts the keep-warm loop. Safe to call multiple times (idempotent). Cancels
 * the previous timer when invoked again so tests can restart cleanly.
 */
export function startKeepWarmScheduler(): void {
  if (keepWarmTimer !== null) return;
  if (!keepWarmEnabled()) {
    console.log("[Keep-warm] Disabled (CV_WORKER_KEEP_WARM_ENABLED=false).");
    return;
  }
  const intervalMs = keepWarmIntervalMs();

  // Fire once immediately so deployments without an active warm worker warm up
  // as soon as the app process starts.
  keepWarmTimer = setInterval(() => {
    void submitWorkerWarmup({ force: true }).catch((error) => {
      // A failing warmup (no WORKER configured yet) must never crash the web
      // process. Retry on the next interval.
      console.warn("[Keep-warm] Warmup request failed:", error instanceof Error ? error.message : String(error));
    });
  }, intervalMs);
  console.log(`[Keep-warm] Every ${intervalMs} ms (one warm GPU during product hours).`);
}

/** Stops the keep-warm loop (used by tests and shutdown hooks). */
export function stopKeepWarmScheduler(): void {
  if (keepWarmTimer !== null) {
    clearInterval(keepWarmTimer);
    keepWarmTimer = null;
  }
}