import { and, desc, eq } from "drizzle-orm";
import { drizzle } from "drizzle-orm/mysql2";
import { analysisSessions, InsertAnalysisSession, InsertUser, users } from "../drizzle/schema";
import { ENV } from './_core/env';

let _db: ReturnType<typeof drizzle> | null = null;

// Lazily create the drizzle instance so local tooling can run without a DB.
export async function getDb() {
  if (!_db && process.env.DATABASE_URL) {
    try {
      _db = drizzle(process.env.DATABASE_URL);
    } catch (error) {
      console.warn("[Database] Failed to connect:", error);
      _db = null;
    }
  }
  return _db;
}

export async function upsertUser(user: InsertUser): Promise<void> {
  if (!user.openId) {
    throw new Error("User openId is required for upsert");
  }

  const db = await getDb();
  if (!db) {
    console.warn("[Database] Cannot upsert user: database not available");
    return;
  }

  try {
    const values: InsertUser = {
      openId: user.openId,
    };
    const updateSet: Record<string, unknown> = {};

    const textFields = ["name", "email", "loginMethod"] as const;
    type TextField = (typeof textFields)[number];

    const assignNullable = (field: TextField) => {
      const value = user[field];
      if (value === undefined) return;
      const normalized = value ?? null;
      values[field] = normalized;
      updateSet[field] = normalized;
    };

    textFields.forEach(assignNullable);

    if (user.lastSignedIn !== undefined) {
      values.lastSignedIn = user.lastSignedIn;
      updateSet.lastSignedIn = user.lastSignedIn;
    }
    if (user.role !== undefined) {
      values.role = user.role;
      updateSet.role = user.role;
    } else if (user.openId === ENV.ownerOpenId) {
      values.role = 'admin';
      updateSet.role = 'admin';
    }

    if (!values.lastSignedIn) {
      values.lastSignedIn = new Date();
    }

    if (Object.keys(updateSet).length === 0) {
      updateSet.lastSignedIn = new Date();
    }

    await db.insert(users).values(values).onDuplicateKeyUpdate({
      set: updateSet,
    });
  } catch (error) {
    console.error("[Database] Failed to upsert user:", error);
    throw error;
  }
}

export async function getUserByOpenId(openId: string) {
  const db = await getDb();
  if (!db) {
    console.warn("[Database] Cannot get user: database not available");
    return undefined;
  }

  const result = await db.select().from(users).where(eq(users.openId, openId)).limit(1);

  return result.length > 0 ? result[0] : undefined;
}

export async function createAnalysisSession(session: InsertAnalysisSession) {
  const db = await getDb();
  if (!db) throw new Error("Database is not available for analysis session storage");

  await db.insert(analysisSessions).values(session);
  return session.id;
}

export async function getAnalysisSessionForUser(id: string, userId: number) {
  const db = await getDb();
  if (!db) throw new Error("Database is not available for analysis session lookup");

  const rows = await db.select().from(analysisSessions).where(and(eq(analysisSessions.id, id), eq(analysisSessions.userId, userId))).limit(1);
  return rows[0];
}

/** Only server-to-server completion code may use this lookup. */
export async function getAnalysisSessionById(id: string) {
  const db = await getDb();
  if (!db) throw new Error("Database is not available for analysis session lookup");
  const rows = await db.select().from(analysisSessions).where(eq(analysisSessions.id, id)).limit(1);
  return rows[0];
}

export async function listAnalysisSessionsForUser(userId: number) {
  const db = await getDb();
  if (!db) throw new Error("Database is not available for analysis session lookup");

  return db.select().from(analysisSessions).where(eq(analysisSessions.userId, userId)).orderBy(desc(analysisSessions.createdAt)).limit(24);
}

export async function updateAnalysisSessionForUser(
  id: string,
  userId: number,
  update: Pick<InsertAnalysisSession, "status" | "workerJobId" | "result" | "failureReason" | "lastWorkerStatusAt">
) {
  const db = await getDb();
  if (!db) throw new Error("Database is not available for analysis session updates");

  await db.update(analysisSessions).set(update).where(and(eq(analysisSessions.id, id), eq(analysisSessions.userId, userId)));
}

export async function updateAnalysisSessionFromWorker(
  id: string,
  workerJobId: string,
  update: Pick<InsertAnalysisSession, "status" | "result" | "failureReason" | "lastWorkerStatusAt">
) {
  const db = await getDb();
  if (!db) throw new Error("Database is not available for analysis session updates");
  // Match the job id as well as the session. This makes stale RunPod
  // callbacks unable to overwrite a retried analysis.
  await db.update(analysisSessions).set(update).where(and(eq(analysisSessions.id, id), eq(analysisSessions.workerJobId, workerJobId)));
}
