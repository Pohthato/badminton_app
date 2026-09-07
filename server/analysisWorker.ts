import { AnalysisResult, Calibration } from "./analysis";

export type WorkerJobRequest = {
  analysisId: string;
  videoStorageKey: string;
  selectedPlayer: "near" | "far";
  calibration: Calibration;
  requestedLayers: Array<"skeleton" | "shuttle" | "racket" | "courtMap">;
};

export type WorkerJobResponse = { jobId: string };

export type WorkerJobStatusResponse = {
  status: "queued" | "processing" | "completed" | "failed";
  result?: AnalysisResult;
  error?: string;
};

function isRunpodEndpoint(workerUrl: string) {
  return workerUrl.includes("api.runpod.ai/v2/");
}

function runpodRoot(workerUrl: string) {
  return workerUrl
    .replace(/\/(runsync|run|status(?:\/[^/]+)?)\/?$/, "")
    .replace(/\/$/, "");
}

function mapRunpodStatus(
  status: unknown
): WorkerJobStatusResponse["status"] | undefined {
  if (status === "COMPLETED") return "completed";
  if (status === "FAILED" || status === "CANCELLED" || status === "TIMED_OUT")
    return "failed";
  if (status === "IN_QUEUE") return "queued";
  if (status === "IN_PROGRESS" || status === "RUNNING") return "processing";
  return undefined;
}

async function responseDetail(response: Response) {
  try {
    const raw = await response.text();
    if (!raw) return "";
    const parsed = JSON.parse(raw) as { detail?: unknown; message?: unknown };
    const detail =
      typeof parsed.detail === "string"
        ? parsed.detail
        : typeof parsed.message === "string"
          ? parsed.message
          : raw;
    return detail.replace(/\s+/g, " ").slice(0, 240);
  } catch {
    return "";
  }
}

function workerEndpointHint(status: number, route: string, detail: string) {
  const suffix = detail ? ` Response: ${detail}` : "";
  if (status === 404)
    return `Computer-vision worker rejected the job (404) at ${route}. Check that CV_WORKER_URL points to the worker root and that POST ${route} exists.${suffix}`;
  return `Computer-vision worker rejected the job (${status}) at ${route}.${suffix}`;
}

export async function submitAnalysisWorkerJob(
  request: WorkerJobRequest
): Promise<WorkerJobResponse> {
  const workerUrl = process.env.CV_WORKER_URL;
  if (!workerUrl)
    throw new Error(
      "No production computer-vision worker has been configured."
    );

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (process.env.CV_WORKER_TOKEN)
    headers.Authorization = `Bearer ${process.env.CV_WORKER_TOKEN}`;

  if (isRunpodEndpoint(workerUrl)) {
    const endpoint = `${runpodRoot(workerUrl)}/run`;
    const response = await fetch(endpoint, {
      method: "POST",
      headers,
      body: JSON.stringify({ input: request }),
    });
    if (!response.ok) {
      const detail = await responseDetail(response);
      throw new Error(
        `RunPod worker rejected the job (${response.status}).${detail ? ` Response: ${detail}` : ""}`
      );
    }
    const body = (await response.json()) as { id?: unknown };
    if (typeof body.id !== "string")
      throw new Error("RunPod returned an invalid asynchronous job response.");
    return { jobId: body.id };
  }

  const endpoint = `${workerUrl.replace(/\/$/, "")}/v1/analysis-jobs`;
  const route = "/v1/analysis-jobs";
  const response = await fetch(endpoint, {
    method: "POST",
    headers,
    body: JSON.stringify(request),
  });
  if (!response.ok)
    throw new Error(
      workerEndpointHint(response.status, route, await responseDetail(response))
    );
  const body = (await response.json()) as Partial<WorkerJobResponse>;
  if (!body.jobId || typeof body.jobId !== "string")
    throw new Error("Computer-vision worker returned an invalid job response.");
  return { jobId: body.jobId };
}

export async function getAnalysisWorkerJob(
  jobId: string
): Promise<WorkerJobStatusResponse> {
  const workerUrl = process.env.CV_WORKER_URL;
  if (!workerUrl)
    throw new Error(
      "No production computer-vision worker has been configured."
    );

  const headers: Record<string, string> = {};
  if (process.env.CV_WORKER_TOKEN)
    headers.Authorization = `Bearer ${process.env.CV_WORKER_TOKEN}`;

  if (isRunpodEndpoint(workerUrl)) {
    const endpoint = `${runpodRoot(workerUrl)}/status/${encodeURIComponent(jobId)}`;
    const response = await fetch(endpoint, { headers });
    if (!response.ok) {
      const detail = await responseDetail(response);
      throw new Error(
        `RunPod worker status check failed (${response.status}).${detail ? ` Response: ${detail}` : ""}`
      );
    }
    const body = (await response.json()) as {
      status?: unknown;
      output?: AnalysisResult;
      error?: unknown;
    };
    const status = mapRunpodStatus(body.status);
    if (!status) throw new Error("RunPod returned an invalid job status.");
    if (status === "failed")
      return {
        status,
        error:
          typeof body.error === "string"
            ? body.error
            : "RunPod analysis job failed.",
      };
    if (status === "completed" && !body.output)
      throw new Error("RunPod completed without an analysis result.");
    return { status, result: body.output };
  }

  const endpoint = `${workerUrl.replace(/\/$/, "")}/v1/analysis-jobs/${encodeURIComponent(jobId)}`;
  const response = await fetch(endpoint, { headers });
  if (!response.ok) {
    const detail = await responseDetail(response);
    throw new Error(
      `Computer-vision worker status check failed (${response.status}) at /v1/analysis-jobs/{jobId}.${detail ? ` Response: ${detail}` : ""}`
    );
  }
  const body = (await response.json()) as Partial<WorkerJobStatusResponse>;
  if (
    !["queued", "processing", "completed", "failed"].includes(body.status ?? "")
  ) {
    throw new Error("Computer-vision worker returned an invalid job status.");
  }
  if (body.status === "completed" && !body.result)
    throw new Error(
      "Computer-vision worker completed without an analysis result."
    );
  return body as WorkerJobStatusResponse;
}

export function validateWorkerResult(result: AnalysisResult) {
  if (!result.processingVersion || !Array.isArray(result.metrics))
    throw new Error(
      "Worker result is missing a processing version or metrics."
    );
  if (
    result.quality.usableFrameRatio < 0 ||
    result.quality.usableFrameRatio > 1
  )
    throw new Error("Worker result contains an invalid usable frame ratio.");
  return result;
}
