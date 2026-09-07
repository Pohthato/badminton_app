import { COOKIE_NAME } from "@shared/const";
import { TRPCError } from "@trpc/server";
import { nanoid } from "nanoid";
import { z } from "zod";
import { AnalysisResult, CourtCorner, buildCoachingPrompt, validateCalibration } from "./analysis";
import { getAnalysisWorkerJob, submitAnalysisWorkerJob, validateWorkerResult } from "./analysisWorker";
import { getSessionCookieOptions } from "./_core/cookies";
import { systemRouter } from "./_core/systemRouter";
import { protectedProcedure, publicProcedure, router } from "./_core/trpc";
import { createAnalysisSession, getAnalysisSessionForUser, listAnalysisSessionsForUser, updateAnalysisSessionForUser } from "./db";
import { requestDeepSeekCoaching } from "./deepseek";
import { storageCreateUploadUrl } from "./storage";

const cornerSchema = z.object({
  label: z.enum(["nearLeft", "nearRight", "farRight", "farLeft"]),
  x: z.number().finite().min(0).max(100),
  y: z.number().finite().min(0).max(100),
});

const resultSchema = z.object({
  processingVersion: z.string().min(1),
  calibration: z.object({
    corners: z.array(cornerSchema).min(3).max(4),
    confidence: z.enum(["unverified", "provisional", "validated", "image_space_only"]),
    supportsCourtMapping: z.boolean(),
    guidance: z.string().min(1),
  }),
  quality: z.object({
    usableFrameRatio: z.number().min(0).max(1),
    poseTrackConfidence: z.number().min(0).max(1),
    shuttleTrackConfidence: z.number().min(0).max(1),
    courtReprojectionErrorPx: z.number().nonnegative().optional(),
  }),
  metrics: z.array(z.object({
    metric: z.string().min(1), value: z.number().finite(), unit: z.string().min(1), direction: z.enum(["higher_is_better", "lower_is_better", "contextual"]), confidence: z.number().min(0).max(1),
    evidenceFrames: z.array(z.object({ frame: z.number().int().nonnegative(), timeMs: z.number().nonnegative(), confidence: z.number().min(0).max(1), source: z.enum(["pose", "shuttle", "racket", "court"]) })),
  })),
  shotDistribution: z.record(z.string(), z.number().nonnegative()),
  annotatedVideoStorageKey: z.string().optional(),
  overlays: z.array(z.object({
    timeMs: z.number().nonnegative(),
    errors: z.record(z.enum(["Skeleton", "Shuttle", "Racket", "Court map"]), z.string().min(1)).optional(),
    skeleton: z.array(z.object({ x: z.number().min(0).max(100), y: z.number().min(0).max(100), confidence: z.number().min(0).max(1) })).optional(),
    shuttle: z.object({ x: z.number().min(0).max(100), y: z.number().min(0).max(100), confidence: z.number().min(0).max(1) }).optional(),
    racket: z.object({ grip: z.object({ x: z.number().min(0).max(100), y: z.number().min(0).max(100), confidence: z.number().min(0).max(1) }), head: z.object({ x: z.number().min(0).max(100), y: z.number().min(0).max(100), confidence: z.number().min(0).max(1) }) }).optional(),
    courtPosition: z.object({ x: z.number().min(0).max(100), y: z.number().min(0).max(100), confidence: z.number().min(0).max(1) }).optional(),
  })).optional(),
});

const permittedVideoTypes = ["video/mp4", "video/quicktime", "video/webm"] as const;

function safeFilename(filename: string) {
  return filename.replace(/[^a-zA-Z0-9._-]/g, "_").slice(0, 180) || "source-video";
}

export const appRouter = router({
    // if you need to use socket.io, read and register route in server/_core/index.ts, all api should start with '/api/' so that the gateway can route correctly
  system: systemRouter,
  auth: router({
    me: publicProcedure.query(opts => opts.ctx.user),
    logout: publicProcedure.mutation(({ ctx }) => {
      const cookieOptions = getSessionCookieOptions(ctx.req);
      ctx.res.clearCookie(COOKIE_NAME, { ...cookieOptions, maxAge: -1 });
      return {
        success: true,
      } as const;
    }),
  }),
  upload: router({
    prepareVideo: protectedProcedure.input(z.object({
      filename: z.string().trim().min(1).max(255),
      mimeType: z.enum(permittedVideoTypes),
      bytes: z.number().int().positive().max(5_000_000_000),
    })).mutation(async ({ ctx, input }) => {
      const filename = safeFilename(input.filename);
      return storageCreateUploadUrl(
        `analysis-sources/${ctx.user.id}/${nanoid(12)}-${filename}`,
        input.mimeType,
      );
    }),
  }),
  analysis: router({
    list: protectedProcedure.query(({ ctx }) => listAnalysisSessionsForUser(ctx.user.id)),
    get: protectedProcedure.input(z.object({ id: z.string().min(6).max(28) })).query(async ({ ctx, input }) => {
      const session = await getAnalysisSessionForUser(input.id, ctx.user.id);
      if (!session) throw new TRPCError({ code: "NOT_FOUND", message: "Analysis session not found." });
      return session;
    }),
    createDraft: protectedProcedure.input(z.object({
      sourceName: z.string().trim().min(1).max(255),
      sourceStorageKey: z.string().trim().min(1).max(768),
      sourceDurationMs: z.number().int().positive().max(7_200_000).optional(),
      selectedPlayer: z.enum(["near", "far"]),
      corners: z.array(cornerSchema).min(3).max(4),
    })).mutation(async ({ ctx, input }) => {
      const calibration = validateCalibration(input.corners as CourtCorner[]);
      const id = nanoid(18);
      await createAnalysisSession({ id, userId: ctx.user.id, sourceName: input.sourceName, sourceStorageKey: input.sourceStorageKey, sourceDurationMs: input.sourceDurationMs, selectedPlayer: input.selectedPlayer, calibration });
      return { id, calibration };
    }),
    submit: protectedProcedure.input(z.object({
      id: z.string().min(6).max(28),
      requestedLayers: z.array(z.enum(["skeleton", "shuttle", "racket", "courtMap"])).min(1),
    })).mutation(async ({ ctx, input }) => {
      const session = await getAnalysisSessionForUser(input.id, ctx.user.id);
      if (!session) throw new TRPCError({ code: "NOT_FOUND", message: "Analysis session not found." });
      const calibration = session.calibration as ReturnType<typeof validateCalibration>;
      const worker = await submitAnalysisWorkerJob({ analysisId: session.id, videoStorageKey: session.sourceStorageKey, selectedPlayer: session.selectedPlayer, calibration, requestedLayers: input.requestedLayers });
      await updateAnalysisSessionForUser(session.id, ctx.user.id, { status: "queued", workerJobId: worker.jobId });
      return worker;
    }),
    refresh: protectedProcedure.input(z.object({ id: z.string().min(6).max(28) })).mutation(async ({ ctx, input }) => {
      const session = await getAnalysisSessionForUser(input.id, ctx.user.id);
      if (!session) throw new TRPCError({ code: "NOT_FOUND", message: "Analysis session not found." });
      if (!session.workerJobId) return { status: session.status, updated: false };

      const worker = await getAnalysisWorkerJob(session.workerJobId);
      if (worker.status === "completed") {
        await updateAnalysisSessionForUser(session.id, ctx.user.id, { status: "completed", result: validateWorkerResult(worker.result!) });
      } else if (worker.status === "failed") {
        await updateAnalysisSessionForUser(session.id, ctx.user.id, { status: "failed" });
      } else {
        await updateAnalysisSessionForUser(session.id, ctx.user.id, { status: worker.status });
      }
      return { status: worker.status, updated: true, error: worker.error };
    }),
    acceptResult: protectedProcedure.input(z.object({ id: z.string().min(6).max(28), result: resultSchema })).mutation(async ({ ctx, input }) => {
      const session = await getAnalysisSessionForUser(input.id, ctx.user.id);
      if (!session) throw new TRPCError({ code: "NOT_FOUND", message: "Analysis session not found." });
      await updateAnalysisSessionForUser(session.id, ctx.user.id, { status: "completed", result: input.result });
      return { success: true };
    }),
  }),
  coach: router({
    previewPrompt: protectedProcedure.input(z.object({ player: z.enum(["near", "far"]), result: resultSchema, question: z.string().trim().max(1_200).optional() })).query(({ input }) => {
      return buildCoachingPrompt({ player: input.player, calibration: input.result.calibration, result: input.result as AnalysisResult, question: input.question });
    }),
    chat: protectedProcedure.input(z.object({ player: z.enum(["near", "far"]), result: resultSchema, question: z.string().trim().min(1).max(1_200) })).mutation(async ({ input }) => {
      const result = input.result as AnalysisResult;
      const feedback = await requestDeepSeekCoaching({ player: input.player, calibration: result.calibration, result, question: input.question });
      return { feedback };
    }),
  }),
});

export type AppRouter = typeof appRouter;
