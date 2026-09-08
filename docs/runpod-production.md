# RunPod production configuration

OmniCourt uses a queue-based endpoint (`/run`) because a video analysis can
take longer than an HTTP request. The web application sends a short-lived
read-only source URL, never storage credentials. RunPod returns the job ID;
the app polls `/status/{jobId}` and also registers a signed completion webhook.
The webhook does not trust its body: it re-reads the RunPod job with the server
credential and validates the result before it is persisted.

## Before deploying

1. Put evaluated private weight files in `worker/weights/` before building:
   `badminton_pose.pt`, `shuttlecock_yolov8n.pt`, and, when available,
   `badminton_racket.pt`. The worker refuses to pretend an unavailable layer was
   detected.
2. Build and push from the repository root:

   ```bash
   docker build -f worker/Dockerfile -t your-registry/omnicourt-worker:1.0.0 .
   docker push your-registry/omnicourt-worker:1.0.0
   ```

3. Create a **queue-based** RunPod Serverless endpoint using that immutable
   image. Select one 24 GB GPU first (L4/A5000/3090-class), followed by an
   available compatible fallback; do not allocate three GPUs to one small-video
   job before benchmarking throughput on one GPU.
4. Set the following application secrets. Do not place any in the browser or
   source tree.

   ```text
   CV_WORKER_URL=https://api.runpod.ai/v2/YOUR_ENDPOINT_ID
   CV_WORKER_TOKEN=RUNPOD_API_KEY
   PUBLIC_APP_URL=https://your-omnicourt-domain
   WORKER_CALLBACK_SECRET=a-long-random-secret
   CV_WORKER_EXECUTION_TIMEOUT_MS=900000
   CV_WORKER_TTL_MS=3600000
   ```

5. Apply database migration `0001_worker_observability.sql` before deploying
   the application that includes this code.

## Low-latency / low-cost operating mode

Use active workers `0` for development and sparse free-tier traffic. For a
paid launch or known busy hours, keep **one** worker warm, maximum `3–5`, and
use a 10–20 minute idle timeout. This is the efficient middle ground: first
customers avoid a cold start, while idle capacity is capped. Increase the warm
minimum only after the p95 queue-delay data warrants it.

Enable FlashBoot. Prefer RunPod model caching or bake private weights into the
image; do not download weights inside `handler()`. A network volume is useful
for very large shared model sets, but pins the endpoint to a region and can add
latency. Keep multiple compatible GPU types enabled for availability. Use a
request TTL that covers expected queue delay plus active processing time.

## Required dashboards and alerts

Track p50/p95 queue delay, cold-start duration, model-load time, execution
seconds per video-minute, failure rate, GPU seconds per completed analysis,
and the fraction of sessions whose pose/shuttle/court quality clears the
coaching threshold. Alert on worker initialization over five minutes, jobs
queued over two minutes in warm mode, completion failures, and a sudden drop in
shuttle confidence.

The worker returns stage timings and model versions in `result.diagnostics`.
Those are customer-visible analysis notes, not a replacement for endpoint
metrics. Keep RunPod's metrics tab as the source of truth for infrastructure
queue/cold-start measurements.
