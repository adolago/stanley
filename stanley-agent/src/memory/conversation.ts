/**
 * Conversation History Manager
 *
 * Features:
 * - Store full conversations with tool calls
 * - Summarization for long conversations
 * - Session restoration
 * - Export/import capability
 */

import { Database } from "bun:sqlite";
import { existsSync, mkdirSync } from "fs";
import { dirname, join } from "path";
import type { Message } from "../agents/stanley";

/**
 * Tool call record within a conversation
 */
export interface ToolCallRecord {
  toolName: string;
  args: unknown;
  result: unknown;
  timestamp: number;
  durationMs?: number;
}

/**
 * Conversation message with tool calls
 */
export interface ConversationMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  toolCalls?: ToolCallRecord[];
  timestamp: number;
  tokenCount?: number;
}

/**
 * Conversation session
 */
export interface ConversationSession {
  id: string;
  title?: string;
  createdAt: number;
  updatedAt: number;
  messageCount: number;
  totalTokens?: number;
  summary?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Full conversation with messages
 */
export interface Conversation extends ConversationSession {
  messages: ConversationMessage[];
}

/**
 * Summarization options
 */
export interface SummarizationOptions {
  maxMessages?: number;
  maxTokens?: number;
  preserveRecent?: number;
}

/**
 * Export format
 */
export interface ConversationExport {
  version: string;
  exportedAt: number;
  sessions: Conversation[];
}

/**
 * Conversation History Manager
 */
export class ConversationManager {
  private db: Database;

  constructor(dbPath?: string) {
    const path = dbPath || this.getDefaultDbPath();
    this.ensureDirectory(path);
    this.db = new Database(path);
    this.initializeSchema();
  }

  /**
   * Get default database path
   */
  private getDefaultDbPath(): string {
    const xdgData = process.env.XDG_DATA_HOME || join(process.env.HOME || "", ".local", "share");
    return join(xdgData, "stanley-agent", "conversations.db");
  }

  /**
   * Ensure directory exists
   */
  private ensureDirectory(dbPath: string): void {
    const dir = dirname(dbPath);
    if (!existsSync(dir)) {
      mkdirSync(dir, { recursive: true });
    }
  }

  /**
   * Initialize database schema
   */
  private initializeSchema(): void {
    this.db.run(`
      CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        title TEXT,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        total_tokens INTEGER DEFAULT 0,
        summary TEXT,
        metadata TEXT
      )
    `);

    this.db.run(`
      CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        tool_calls TEXT,
        timestamp INTEGER NOT NULL,
        token_count INTEGER,
        FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
      )
    `);

    this.db.run(`
      CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)
    `);

    this.db.run(`
      CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)
    `);

    this.db.run(`
      CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC)
    `);
  }

  /**
   * Generate unique ID
   */
  private generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
  }

  /**
   * Create a new conversation session
   */
  createSession(title?: string, metadata?: Record<string, unknown>): ConversationSession {
    const id = this.generateId();
    const now = Date.now();

    this.db.run(
      `
      INSERT INTO sessions (id, title, created_at, updated_at, metadata)
      VALUES (?, ?, ?, ?, ?)
      `,
      [id, title || null, now, now, metadata ? JSON.stringify(metadata) : null]
    );

    return {
      id,
      title,
      createdAt: now,
      updatedAt: now,
      messageCount: 0,
      metadata,
    };
  }

  /**
   * Add a message to a session
   */
  addMessage(
    sessionId: string,
    role: "user" | "assistant" | "system",
    content: string,
    options?: {
      toolCalls?: ToolCallRecord[];
      tokenCount?: number;
    }
  ): ConversationMessage {
    const id = this.generateId();
    const now = Date.now();

    const message: ConversationMessage = {
      id,
      role,
      content,
      toolCalls: options?.toolCalls,
      timestamp: now,
      tokenCount: options?.tokenCount,
    };

    this.db.run(
      `
      INSERT INTO messages (id, session_id, role, content, tool_calls, timestamp, token_count)
      VALUES (?, ?, ?, ?, ?, ?, ?)
      `,
      [
        id,
        sessionId,
        role,
        content,
        options?.toolCalls ? JSON.stringify(options.toolCalls) : null,
        now,
        options?.tokenCount || null,
      ]
    );

    // Update session
    this.db.run(
      `
      UPDATE sessions
      SET updated_at = ?,
          total_tokens = total_tokens + COALESCE(?, 0)
      WHERE id = ?
      `,
      [now, options?.tokenCount || 0, sessionId]
    );

    return message;
  }

  /**
   * Add multiple messages from a standard conversation
   */
  addMessages(sessionId: string, messages: Message[]): void {
    for (const msg of messages) {
      const content = typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content);
      this.addMessage(sessionId, msg.role as "user" | "assistant" | "system", content);
    }
  }

  /**
   * Get a session by ID
   */
  getSession(sessionId: string): ConversationSession | null {
    const row = this.db
      .query<
        {
          id: string;
          title: string | null;
          created_at: number;
          updated_at: number;
          total_tokens: number;
          summary: string | null;
          metadata: string | null;
        },
        [string]
      >("SELECT * FROM sessions WHERE id = ?")
      .get(sessionId);

    if (!row) return null;

    const messageCount = this.db
      .query<{ count: number }, [string]>(
        "SELECT COUNT(*) as count FROM messages WHERE session_id = ?"
      )
      .get(sessionId);

    return {
      id: row.id,
      title: row.title || undefined,
      createdAt: row.created_at,
      updatedAt: row.updated_at,
      messageCount: messageCount?.count || 0,
      totalTokens: row.total_tokens,
      summary: row.summary || undefined,
      metadata: row.metadata ? JSON.parse(row.metadata) : undefined,
    };
  }

  /**
   * Get full conversation with messages
   */
  getConversation(sessionId: string): Conversation | null {
    const session = this.getSession(sessionId);
    if (!session) return null;

    const rows = this.db
      .query<
        {
          id: string;
          role: string;
          content: string;
          tool_calls: string | null;
          timestamp: number;
          token_count: number | null;
        },
        [string]
      >("SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC")
      .all(sessionId);

    const messages: ConversationMessage[] = rows.map((row) => ({
      id: row.id,
      role: row.role as "user" | "assistant" | "system",
      content: row.content,
      toolCalls: row.tool_calls ? JSON.parse(row.tool_calls) : undefined,
      timestamp: row.timestamp,
      tokenCount: row.token_count || undefined,
    }));

    return {
      ...session,
      messages,
    };
  }

  /**
   * Get messages in a format suitable for the AI SDK
   */
  getMessagesForAI(sessionId: string): Message[] {
    const rows = this.db
      .query<{ role: string; content: string }, [string]>(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp ASC"
      )
      .all(sessionId);

    return rows.map((row) => ({
      role: row.role as "user" | "assistant" | "system",
      content: row.content,
    }));
  }

  /**
   * Get recent messages (useful for context window management)
   */
  getRecentMessages(sessionId: string, limit: number): ConversationMessage[] {
    const rows = this.db
      .query<
        {
          id: string;
          role: string;
          content: string;
          tool_calls: string | null;
          timestamp: number;
          token_count: number | null;
        },
        [string, number]
      >(
        `
        SELECT * FROM messages
        WHERE session_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
        `
      )
      .all(sessionId, limit);

    return rows
      .map((row) => ({
        id: row.id,
        role: row.role as "user" | "assistant" | "system",
        content: row.content,
        toolCalls: row.tool_calls ? JSON.parse(row.tool_calls) : undefined,
        timestamp: row.timestamp,
        tokenCount: row.token_count || undefined,
      }))
      .reverse();
  }

  /**
   * List all sessions
   */
  listSessions(options?: {
    limit?: number;
    offset?: number;
    orderBy?: "created" | "updated";
  }): ConversationSession[] {
    const { limit = 50, offset = 0, orderBy = "updated" } = options || {};
    const orderColumn = orderBy === "created" ? "created_at" : "updated_at";

    const rows = this.db
      .query<
        {
          id: string;
          title: string | null;
          created_at: number;
          updated_at: number;
          total_tokens: number;
          summary: string | null;
          metadata: string | null;
        },
        [number, number]
      >(`SELECT * FROM sessions ORDER BY ${orderColumn} DESC LIMIT ? OFFSET ?`)
      .all(limit, offset);

    return rows.map((row) => {
      const messageCount = this.db
        .query<{ count: number }, [string]>(
          "SELECT COUNT(*) as count FROM messages WHERE session_id = ?"
        )
        .get(row.id);

      return {
        id: row.id,
        title: row.title || undefined,
        createdAt: row.created_at,
        updatedAt: row.updated_at,
        messageCount: messageCount?.count || 0,
        totalTokens: row.total_tokens,
        summary: row.summary || undefined,
        metadata: row.metadata ? JSON.parse(row.metadata) : undefined,
      };
    });
  }

  /**
   * Update session title
   */
  updateTitle(sessionId: string, title: string): void {
    this.db.run("UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?", [
      title,
      Date.now(),
      sessionId,
    ]);
  }

  /**
   * Update session summary
   */
  updateSummary(sessionId: string, summary: string): void {
    this.db.run("UPDATE sessions SET summary = ?, updated_at = ? WHERE id = ?", [
      summary,
      Date.now(),
      sessionId,
    ]);
  }

  /**
   * Delete a session and all its messages
   */
  deleteSession(sessionId: string): boolean {
    this.db.run("DELETE FROM messages WHERE session_id = ?", [sessionId]);
    const result = this.db.run("DELETE FROM sessions WHERE id = ?", [sessionId]);
    return result.changes > 0;
  }

  /**
   * Search messages across all sessions
   */
  searchMessages(
    query: string,
    options?: { sessionId?: string; limit?: number }
  ): Array<ConversationMessage & { sessionId: string }> {
    const { sessionId, limit = 50 } = options || {};

    let sql = `
      SELECT m.*, m.session_id as session_id
      FROM messages m
      WHERE m.content LIKE ?
    `;
    const params: (string | number)[] = [`%${query}%`];

    if (sessionId) {
      sql += " AND m.session_id = ?";
      params.push(sessionId);
    }

    sql += " ORDER BY m.timestamp DESC LIMIT ?";
    params.push(limit);

    const rows = this.db
      .query<
        {
          id: string;
          session_id: string;
          role: string;
          content: string;
          tool_calls: string | null;
          timestamp: number;
          token_count: number | null;
        },
        (string | number)[]
      >(sql)
      .all(...params);

    return rows.map((row) => ({
      id: row.id,
      sessionId: row.session_id,
      role: row.role as "user" | "assistant" | "system",
      content: row.content,
      toolCalls: row.tool_calls ? JSON.parse(row.tool_calls) : undefined,
      timestamp: row.timestamp,
      tokenCount: row.token_count || undefined,
    }));
  }

  /**
   * Get conversation context with optional summarization
   */
  getContext(
    sessionId: string,
    options: SummarizationOptions = {}
  ): { messages: Message[]; summary?: string; truncated: boolean } {
    const { maxMessages = 50, preserveRecent = 10 } = options;

    const conversation = this.getConversation(sessionId);
    if (!conversation) {
      return { messages: [], truncated: false };
    }

    const totalMessages = conversation.messages.length;

    if (totalMessages <= maxMessages) {
      return {
        messages: conversation.messages.map((m) => ({
          role: m.role,
          content: m.content,
        })),
        truncated: false,
      };
    }

    // Need to truncate - keep recent and add summary context
    const recentMessages = conversation.messages.slice(-preserveRecent);

    return {
      messages: recentMessages.map((m) => ({
        role: m.role,
        content: m.content,
      })),
      summary: conversation.summary,
      truncated: true,
    };
  }

  /**
   * Export conversations
   */
  exportConversations(sessionIds?: string[]): ConversationExport {
    let sessions: Conversation[];

    if (sessionIds && sessionIds.length > 0) {
      sessions = sessionIds
        .map((id) => this.getConversation(id))
        .filter((c): c is Conversation => c !== null);
    } else {
      const allSessions = this.listSessions({ limit: 1000 });
      sessions = allSessions
        .map((s) => this.getConversation(s.id))
        .filter((c): c is Conversation => c !== null);
    }

    return {
      version: "1.0",
      exportedAt: Date.now(),
      sessions,
    };
  }

  /**
   * Import conversations
   */
  importConversations(data: ConversationExport): { imported: number; errors: string[] } {
    const errors: string[] = [];
    let imported = 0;

    for (const session of data.sessions) {
      try {
        // Check if session already exists
        const existing = this.getSession(session.id);
        if (existing) {
          errors.push(`Session ${session.id} already exists, skipping`);
          continue;
        }

        // Insert session
        this.db.run(
          `
          INSERT INTO sessions (id, title, created_at, updated_at, total_tokens, summary, metadata)
          VALUES (?, ?, ?, ?, ?, ?, ?)
          `,
          [
            session.id,
            session.title || null,
            session.createdAt,
            session.updatedAt,
            session.totalTokens || 0,
            session.summary || null,
            session.metadata ? JSON.stringify(session.metadata) : null,
          ]
        );

        // Insert messages
        for (const msg of session.messages) {
          this.db.run(
            `
            INSERT INTO messages (id, session_id, role, content, tool_calls, timestamp, token_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            `,
            [
              msg.id,
              session.id,
              msg.role,
              msg.content,
              msg.toolCalls ? JSON.stringify(msg.toolCalls) : null,
              msg.timestamp,
              msg.tokenCount || null,
            ]
          );
        }

        imported++;
      } catch (error) {
        errors.push(`Failed to import session ${session.id}: ${error}`);
      }
    }

    return { imported, errors };
  }

  /**
   * Get statistics
   */
  getStats(): {
    totalSessions: number;
    totalMessages: number;
    totalTokens: number;
    oldestSession: number | null;
    newestSession: number | null;
  } {
    const sessions = this.db
      .query<{ count: number; tokens: number }, []>(
        "SELECT COUNT(*) as count, COALESCE(SUM(total_tokens), 0) as tokens FROM sessions"
      )
      .get();

    const messages = this.db
      .query<{ count: number }, []>("SELECT COUNT(*) as count FROM messages")
      .get();

    const dates = this.db
      .query<{ oldest: number | null; newest: number | null }, []>(
        "SELECT MIN(created_at) as oldest, MAX(created_at) as newest FROM sessions"
      )
      .get();

    return {
      totalSessions: sessions?.count || 0,
      totalMessages: messages?.count || 0,
      totalTokens: sessions?.tokens || 0,
      oldestSession: dates?.oldest || null,
      newestSession: dates?.newest || null,
    };
  }

  /**
   * Close the database connection
   */
  close(): void {
    this.db.close();
  }
}

/**
 * Create a conversation manager instance
 */
export function createConversationManager(dbPath?: string): ConversationManager {
  return new ConversationManager(dbPath);
}

/**
 * Singleton instance for shared access
 */
let defaultManager: ConversationManager | null = null;

export function getDefaultConversationManager(): ConversationManager {
  if (!defaultManager) {
    defaultManager = createConversationManager();
  }
  return defaultManager;
}

export function closeDefaultConversationManager(): void {
  if (defaultManager) {
    defaultManager.close();
    defaultManager = null;
  }
}
