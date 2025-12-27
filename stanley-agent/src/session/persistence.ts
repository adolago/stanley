/**
 * Session Persistence
 *
 * Handles saving and restoring session state to/from disk.
 */

import * as fs from "fs";
import * as path from "path";
import {
  SessionState,
  validateState,
  SESSION_STATE_VERSION,
  createInitialState,
} from "./state";

/**
 * Persistence configuration
 */
export interface PersistenceConfig {
  sessionDir: string;
  autoSaveInterval?: number; // milliseconds
  maxSessionAge?: number; // days
  compressOldSessions?: boolean;
}

/**
 * Default persistence configuration
 */
export const DEFAULT_PERSISTENCE_CONFIG: PersistenceConfig = {
  sessionDir: getDefaultSessionDir(),
  autoSaveInterval: 30000, // 30 seconds
  maxSessionAge: 30, // 30 days
  compressOldSessions: false,
};

/**
 * Get default session directory
 */
function getDefaultSessionDir(): string {
  const homeDir = process.env.HOME || process.env.USERPROFILE || "";
  return path.join(homeDir, ".stanley", "sessions");
}

/**
 * Session file info
 */
export interface SessionFileInfo {
  sessionId: string;
  filePath: string;
  createdAt: number;
  lastModified: number;
  size: number;
}

/**
 * Persistence manager for session state
 */
export class SessionPersistence {
  private config: PersistenceConfig;
  private autoSaveTimer?: ReturnType<typeof setInterval>;

  constructor(config: Partial<PersistenceConfig> = {}) {
    this.config = { ...DEFAULT_PERSISTENCE_CONFIG, ...config };
    this.ensureSessionDir();
  }

  /**
   * Ensure session directory exists
   */
  private ensureSessionDir(): void {
    if (!fs.existsSync(this.config.sessionDir)) {
      fs.mkdirSync(this.config.sessionDir, { recursive: true });
    }
  }

  /**
   * Get session file path
   */
  private getSessionPath(sessionId: string): string {
    return path.join(this.config.sessionDir, `${sessionId}.json`);
  }

  /**
   * Save session state to disk
   */
  async save(state: SessionState): Promise<void> {
    const filePath = this.getSessionPath(state.metadata.id);
    const content = JSON.stringify(state, null, 2);

    await fs.promises.writeFile(filePath, content, "utf-8");
  }

  /**
   * Load session state from disk
   */
  async load(sessionId: string): Promise<SessionState | null> {
    const filePath = this.getSessionPath(sessionId);

    if (!fs.existsSync(filePath)) {
      return null;
    }

    try {
      const content = await fs.promises.readFile(filePath, "utf-8");
      const parsed = JSON.parse(content);

      if (!validateState(parsed)) {
        console.warn(`Invalid session state in ${filePath}, attempting migration`);
        return this.migrateState(parsed, sessionId);
      }

      return parsed;
    } catch (error) {
      console.error(`Failed to load session ${sessionId}:`, error);
      return null;
    }
  }

  /**
   * Delete session from disk
   */
  async delete(sessionId: string): Promise<boolean> {
    const filePath = this.getSessionPath(sessionId);

    if (!fs.existsSync(filePath)) {
      return false;
    }

    try {
      await fs.promises.unlink(filePath);
      return true;
    } catch (error) {
      console.error(`Failed to delete session ${sessionId}:`, error);
      return false;
    }
  }

  /**
   * List all saved sessions
   */
  async listSessions(): Promise<SessionFileInfo[]> {
    this.ensureSessionDir();

    const files = await fs.promises.readdir(this.config.sessionDir);
    const sessions: SessionFileInfo[] = [];

    for (const file of files) {
      if (!file.endsWith(".json")) continue;

      const filePath = path.join(this.config.sessionDir, file);
      const stats = await fs.promises.stat(filePath);

      try {
        const content = await fs.promises.readFile(filePath, "utf-8");
        const parsed = JSON.parse(content);

        sessions.push({
          sessionId: file.replace(".json", ""),
          filePath,
          createdAt: parsed.metadata?.createdAt || stats.birthtimeMs,
          lastModified: stats.mtimeMs,
          size: stats.size,
        });
      } catch {
        // Skip invalid files
        continue;
      }
    }

    // Sort by last modified, most recent first
    return sessions.sort((a, b) => b.lastModified - a.lastModified);
  }

  /**
   * Get most recent session
   */
  async getMostRecentSession(): Promise<SessionState | null> {
    const sessions = await this.listSessions();

    if (sessions.length === 0) {
      return null;
    }

    return this.load(sessions[0].sessionId);
  }

  /**
   * Clean up old sessions
   */
  async cleanupOldSessions(): Promise<number> {
    if (!this.config.maxSessionAge) return 0;

    const sessions = await this.listSessions();
    const cutoffDate = Date.now() - this.config.maxSessionAge * 24 * 60 * 60 * 1000;
    let deletedCount = 0;

    for (const session of sessions) {
      if (session.lastModified < cutoffDate) {
        const deleted = await this.delete(session.sessionId);
        if (deleted) deletedCount++;
      }
    }

    return deletedCount;
  }

  /**
   * Export session to file
   */
  async exportSession(sessionId: string, exportPath: string): Promise<boolean> {
    const state = await this.load(sessionId);
    if (!state) return false;

    try {
      const content = JSON.stringify(state, null, 2);
      await fs.promises.writeFile(exportPath, content, "utf-8");
      return true;
    } catch (error) {
      console.error(`Failed to export session ${sessionId}:`, error);
      return false;
    }
  }

  /**
   * Import session from file
   */
  async importSession(importPath: string): Promise<SessionState | null> {
    try {
      const content = await fs.promises.readFile(importPath, "utf-8");
      const parsed = JSON.parse(content);

      if (!validateState(parsed)) {
        console.warn("Invalid session state in import file");
        return null;
      }

      // Save to session directory
      await this.save(parsed);
      return parsed;
    } catch (error) {
      console.error("Failed to import session:", error);
      return null;
    }
  }

  /**
   * Start auto-save timer
   */
  startAutoSave(getState: () => SessionState): void {
    if (!this.config.autoSaveInterval) return;

    this.stopAutoSave();

    this.autoSaveTimer = setInterval(async () => {
      try {
        const state = getState();
        if (state.userPreferences.autoSave) {
          await this.save(state);
        }
      } catch (error) {
        console.error("Auto-save failed:", error);
      }
    }, this.config.autoSaveInterval);
  }

  /**
   * Stop auto-save timer
   */
  stopAutoSave(): void {
    if (this.autoSaveTimer) {
      clearInterval(this.autoSaveTimer);
      this.autoSaveTimer = undefined;
    }
  }

  /**
   * Migrate old session state to current version
   */
  private migrateState(
    oldState: unknown,
    sessionId: string
  ): SessionState | null {
    if (!oldState || typeof oldState !== "object") {
      return null;
    }

    const old = oldState as Record<string, unknown>;

    try {
      // Attempt to extract what we can from old state
      const metadata = old.metadata as Record<string, unknown> | undefined;
      const provider = metadata?.provider as string || "unknown";
      const model = metadata?.model as string || "unknown";

      // Create fresh state with old data where possible
      const newState = createInitialState(sessionId, provider, model);

      // Migrate conversation history - cast to preserve Message type compatibility
      if (Array.isArray(old.conversationHistory)) {
        const validMessages = old.conversationHistory.filter(
          (msg) =>
            msg &&
            typeof msg === "object" &&
            typeof (msg as Record<string, unknown>).role === "string"
        );
        // Cast through unknown to preserve the Message type structure
        newState.conversationHistory = validMessages as unknown as SessionState["conversationHistory"];
      }

      // Migrate tool call history
      if (Array.isArray(old.toolCallHistory)) {
        newState.toolCallHistory = old.toolCallHistory.filter(
          (call): call is SessionState["toolCallHistory"][0] =>
            call &&
            typeof call === "object" &&
            typeof (call as Record<string, unknown>).toolName === "string"
        );
      }

      // Migrate token usage
      if (Array.isArray(old.tokenUsage)) {
        newState.tokenUsage = old.tokenUsage.filter(
          (usage): usage is SessionState["tokenUsage"][0] =>
            usage &&
            typeof usage === "object" &&
            typeof (usage as Record<string, unknown>).totalTokens === "number"
        );
      }

      // Preserve timestamps
      if (metadata?.createdAt && typeof metadata.createdAt === "number") {
        newState.metadata.createdAt = metadata.createdAt;
      }
      if (metadata?.lastActiveAt && typeof metadata.lastActiveAt === "number") {
        newState.metadata.lastActiveAt = metadata.lastActiveAt;
      }

      // Mark as migrated
      newState.metadata.version = SESSION_STATE_VERSION;

      return newState;
    } catch (error) {
      console.error("Migration failed:", error);
      return null;
    }
  }

  /**
   * Create backup of session
   */
  async backup(sessionId: string): Promise<string | null> {
    const state = await this.load(sessionId);
    if (!state) return null;

    const backupDir = path.join(this.config.sessionDir, "backups");
    if (!fs.existsSync(backupDir)) {
      fs.mkdirSync(backupDir, { recursive: true });
    }

    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const backupPath = path.join(backupDir, `${sessionId}_${timestamp}.json`);

    try {
      await fs.promises.writeFile(
        backupPath,
        JSON.stringify(state, null, 2),
        "utf-8"
      );
      return backupPath;
    } catch (error) {
      console.error("Backup failed:", error);
      return null;
    }
  }

  /**
   * Get session directory path
   */
  getSessionDir(): string {
    return this.config.sessionDir;
  }
}
