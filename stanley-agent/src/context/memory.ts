/**
 * Persistent Memory Store
 *
 * Manages persistent storage of context and insights across sessions.
 * Supports both in-memory and file-based persistence.
 */

import * as fs from "fs";
import * as path from "path";
import type {
  ContextItem,
  ContextSourceType,
  ContextPriority,
} from "./types";

// =============================================================================
// Memory Store Types
// =============================================================================

export interface MemoryEntry {
  id: string;
  key: string;
  value: unknown;
  category: string;
  createdAt: Date;
  updatedAt: Date;
  expiresAt?: Date;
  accessCount: number;
  lastAccessedAt: Date;
  tags: string[];
  priority: ContextPriority;
}

export interface InsightEntry {
  id: string;
  type: "observation" | "pattern" | "recommendation" | "warning";
  content: string;
  symbols: string[];
  confidence: number;
  source: string;
  createdAt: Date;
  validUntil?: Date;
  metadata: Record<string, unknown>;
}

export interface SessionState {
  sessionId: string;
  startedAt: Date;
  lastActivityAt: Date;
  contextItems: ContextItem[];
  insights: InsightEntry[];
  conversationSummary?: string;
  activeSymbols: string[];
}

export interface MemoryStoreConfig {
  enablePersistence: boolean;
  storagePath: string;
  maxEntries: number;
  maxInsights: number;
  defaultTtlMs: number;
  autoSaveIntervalMs: number;
}

const DEFAULT_CONFIG: MemoryStoreConfig = {
  enablePersistence: true,
  storagePath: ".stanley/memory",
  maxEntries: 1000,
  maxInsights: 200,
  defaultTtlMs: 7 * 24 * 60 * 60 * 1000, // 7 days
  autoSaveIntervalMs: 5 * 60 * 1000, // 5 minutes
};

// =============================================================================
// Memory Store Implementation
// =============================================================================

export class MemoryStore {
  private entries: Map<string, MemoryEntry> = new Map();
  private insights: InsightEntry[] = [];
  private sessions: Map<string, SessionState> = new Map();
  private currentSession: SessionState | null = null;
  private config: MemoryStoreConfig;
  private autoSaveTimer: ReturnType<typeof setInterval> | null = null;
  private dirty = false;

  constructor(config: Partial<MemoryStoreConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };

    if (this.config.enablePersistence) {
      this.ensureStorageDirectory();
      this.loadFromDisk();
      this.startAutoSave();
    }
  }

  // ===========================================================================
  // Memory Entry Operations
  // ===========================================================================

  /**
   * Store a value in memory
   */
  set(
    key: string,
    value: unknown,
    options: {
      category?: string;
      tags?: string[];
      priority?: ContextPriority;
      ttlMs?: number;
    } = {}
  ): MemoryEntry {
    const now = new Date();
    const existing = this.entries.get(key);

    const entry: MemoryEntry = {
      id: existing?.id || this.generateId(),
      key,
      value,
      category: options.category || "general",
      createdAt: existing?.createdAt || now,
      updatedAt: now,
      expiresAt: options.ttlMs
        ? new Date(now.getTime() + options.ttlMs)
        : undefined,
      accessCount: existing?.accessCount || 0,
      lastAccessedAt: now,
      tags: options.tags || [],
      priority: options.priority || "medium",
    };

    this.entries.set(key, entry);
    this.dirty = true;
    this.enforceMaxEntries();

    return entry;
  }

  /**
   * Get a value from memory
   */
  get<T = unknown>(key: string): T | null {
    const entry = this.entries.get(key);

    if (!entry) {
      return null;
    }

    // Check expiration
    if (entry.expiresAt && entry.expiresAt < new Date()) {
      this.entries.delete(key);
      this.dirty = true;
      return null;
    }

    // Update access stats
    entry.accessCount++;
    entry.lastAccessedAt = new Date();
    this.dirty = true;

    return entry.value as T;
  }

  /**
   * Delete a memory entry
   */
  delete(key: string): boolean {
    const deleted = this.entries.delete(key);
    if (deleted) {
      this.dirty = true;
    }
    return deleted;
  }

  /**
   * Check if a key exists
   */
  has(key: string): boolean {
    const entry = this.entries.get(key);
    if (!entry) return false;

    // Check expiration
    if (entry.expiresAt && entry.expiresAt < new Date()) {
      this.entries.delete(key);
      this.dirty = true;
      return false;
    }

    return true;
  }

  /**
   * Get all entries matching a category
   */
  getByCategory(category: string): MemoryEntry[] {
    return Array.from(this.entries.values()).filter(
      (e) => e.category === category && !this.isExpired(e)
    );
  }

  /**
   * Get all entries matching tags
   */
  getByTags(tags: string[]): MemoryEntry[] {
    return Array.from(this.entries.values()).filter(
      (e) => !this.isExpired(e) && tags.some((t) => e.tags.includes(t))
    );
  }

  /**
   * Search entries by key pattern
   */
  search(pattern: string): MemoryEntry[] {
    const regex = new RegExp(pattern, "i");
    return Array.from(this.entries.values()).filter(
      (e) => !this.isExpired(e) && regex.test(e.key)
    );
  }

  // ===========================================================================
  // Insight Operations
  // ===========================================================================

  /**
   * Store an insight
   */
  addInsight(insight: Omit<InsightEntry, "id" | "createdAt">): InsightEntry {
    const entry: InsightEntry = {
      ...insight,
      id: this.generateId(),
      createdAt: new Date(),
    };

    this.insights.unshift(entry);
    this.dirty = true;
    this.enforceMaxInsights();

    return entry;
  }

  /**
   * Get insights for symbols
   */
  getInsightsForSymbols(symbols: string[]): InsightEntry[] {
    const now = new Date();
    return this.insights.filter(
      (i) =>
        (!i.validUntil || i.validUntil > now) &&
        i.symbols.some((s) => symbols.includes(s))
    );
  }

  /**
   * Get recent insights
   */
  getRecentInsights(limit = 10): InsightEntry[] {
    const now = new Date();
    return this.insights
      .filter((i) => !i.validUntil || i.validUntil > now)
      .slice(0, limit);
  }

  /**
   * Get insights by type
   */
  getInsightsByType(type: InsightEntry["type"]): InsightEntry[] {
    const now = new Date();
    return this.insights.filter(
      (i) => i.type === type && (!i.validUntil || i.validUntil > now)
    );
  }

  // ===========================================================================
  // Session Management
  // ===========================================================================

  /**
   * Start a new session
   */
  startSession(): SessionState {
    const session: SessionState = {
      sessionId: this.generateId(),
      startedAt: new Date(),
      lastActivityAt: new Date(),
      contextItems: [],
      insights: [],
      activeSymbols: [],
    };

    this.currentSession = session;
    this.sessions.set(session.sessionId, session);
    this.dirty = true;

    return session;
  }

  /**
   * Get the current session
   */
  getCurrentSession(): SessionState | null {
    return this.currentSession;
  }

  /**
   * Update session activity
   */
  updateSessionActivity(): void {
    if (this.currentSession) {
      this.currentSession.lastActivityAt = new Date();
      this.dirty = true;
    }
  }

  /**
   * Add context items to current session
   */
  addContextToSession(items: ContextItem[]): void {
    if (!this.currentSession) {
      this.startSession();
    }

    if (this.currentSession) {
      this.currentSession.contextItems.push(...items);
      this.currentSession.lastActivityAt = new Date();

      // Track active symbols
      for (const item of items) {
        const symbols = item.metadata.symbols as string[] | undefined;
        if (symbols) {
          for (const symbol of symbols) {
            if (!this.currentSession.activeSymbols.includes(symbol)) {
              this.currentSession.activeSymbols.push(symbol);
            }
          }
        }
      }

      this.dirty = true;
    }
  }

  /**
   * Add insight to current session
   */
  addInsightToSession(insight: InsightEntry): void {
    if (this.currentSession) {
      this.currentSession.insights.push(insight);
      this.dirty = true;
    }
  }

  /**
   * Set conversation summary for session
   */
  setConversationSummary(summary: string): void {
    if (this.currentSession) {
      this.currentSession.conversationSummary = summary;
      this.dirty = true;
    }
  }

  /**
   * End the current session
   */
  endSession(): SessionState | null {
    const session = this.currentSession;
    this.currentSession = null;

    if (session) {
      // Store important context from session for future reference
      this.preserveSessionInsights(session);
      this.dirty = true;
    }

    return session;
  }

  /**
   * Get a previous session by ID
   */
  getSession(sessionId: string): SessionState | null {
    return this.sessions.get(sessionId) || null;
  }

  /**
   * Get recent sessions
   */
  getRecentSessions(limit = 5): SessionState[] {
    return Array.from(this.sessions.values())
      .sort((a, b) => b.startedAt.getTime() - a.startedAt.getTime())
      .slice(0, limit);
  }

  // ===========================================================================
  // Symbol-Specific Memory
  // ===========================================================================

  /**
   * Store symbol-specific context
   */
  setSymbolContext(
    symbol: string,
    context: Record<string, unknown>
  ): MemoryEntry {
    return this.set(`symbol:${symbol}`, context, {
      category: "symbol",
      tags: [symbol],
      priority: "high",
    });
  }

  /**
   * Get symbol-specific context
   */
  getSymbolContext<T = Record<string, unknown>>(symbol: string): T | null {
    return this.get<T>(`symbol:${symbol}`);
  }

  /**
   * Get all symbol contexts
   */
  getAllSymbolContexts(): Map<string, unknown> {
    const symbolEntries = this.getByCategory("symbol");
    const result = new Map<string, unknown>();

    for (const entry of symbolEntries) {
      const symbol = entry.key.replace("symbol:", "");
      result.set(symbol, entry.value);
    }

    return result;
  }

  // ===========================================================================
  // Persistence
  // ===========================================================================

  /**
   * Save memory to disk
   */
  async save(): Promise<void> {
    if (!this.config.enablePersistence || !this.dirty) {
      return;
    }

    const data = {
      version: 1,
      savedAt: new Date().toISOString(),
      entries: Array.from(this.entries.entries()),
      insights: this.insights,
      sessions: Array.from(this.sessions.entries()),
    };

    const filePath = this.getStorageFilePath();

    try {
      await fs.promises.writeFile(filePath, JSON.stringify(data, null, 2));
      this.dirty = false;
    } catch (error) {
      console.error("Failed to save memory store:", error);
    }
  }

  /**
   * Load memory from disk
   */
  private loadFromDisk(): void {
    const filePath = this.getStorageFilePath();

    try {
      if (!fs.existsSync(filePath)) {
        return;
      }

      const content = fs.readFileSync(filePath, "utf-8");
      const data = JSON.parse(content);

      if (data.version !== 1) {
        console.warn("Memory store version mismatch, starting fresh");
        return;
      }

      // Restore entries
      this.entries = new Map(
        data.entries.map(([key, entry]: [string, MemoryEntry]) => [
          key,
          {
            ...entry,
            createdAt: new Date(entry.createdAt),
            updatedAt: new Date(entry.updatedAt),
            lastAccessedAt: new Date(entry.lastAccessedAt),
            expiresAt: entry.expiresAt ? new Date(entry.expiresAt) : undefined,
          },
        ])
      );

      // Restore insights
      this.insights = data.insights.map((i: InsightEntry) => ({
        ...i,
        createdAt: new Date(i.createdAt),
        validUntil: i.validUntil ? new Date(i.validUntil) : undefined,
      }));

      // Restore sessions (limited to recent ones)
      this.sessions = new Map(
        data.sessions
          .slice(0, 20)
          .map(([id, session]: [string, SessionState]) => [
            id,
            {
              ...session,
              startedAt: new Date(session.startedAt),
              lastActivityAt: new Date(session.lastActivityAt),
            },
          ])
      );

      // Clean up expired entries
      this.cleanupExpired();
    } catch (error) {
      console.error("Failed to load memory store:", error);
    }
  }

  /**
   * Clear all memory
   */
  clear(): void {
    this.entries.clear();
    this.insights = [];
    this.sessions.clear();
    this.currentSession = null;
    this.dirty = true;
  }

  /**
   * Get memory statistics
   */
  getStats(): {
    entryCount: number;
    insightCount: number;
    sessionCount: number;
    oldestEntry: Date | null;
    newestEntry: Date | null;
  } {
    const entriesArray = Array.from(this.entries.values());
    const dates = entriesArray.map((e) => e.createdAt.getTime());

    return {
      entryCount: entriesArray.length,
      insightCount: this.insights.length,
      sessionCount: this.sessions.size,
      oldestEntry: dates.length > 0 ? new Date(Math.min(...dates)) : null,
      newestEntry: dates.length > 0 ? new Date(Math.max(...dates)) : null,
    };
  }

  /**
   * Cleanup and close the memory store
   */
  async close(): Promise<void> {
    if (this.autoSaveTimer) {
      clearInterval(this.autoSaveTimer);
      this.autoSaveTimer = null;
    }

    await this.save();
  }

  // ===========================================================================
  // Private Helpers
  // ===========================================================================

  private generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
  }

  private isExpired(entry: MemoryEntry): boolean {
    return entry.expiresAt !== undefined && entry.expiresAt < new Date();
  }

  private enforceMaxEntries(): void {
    if (this.entries.size <= this.config.maxEntries) {
      return;
    }

    // Remove oldest, lowest priority entries first
    const entries = Array.from(this.entries.entries())
      .sort((a, b) => {
        // Priority order: critical > high > medium > low
        const priorityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
        const priorityDiff =
          priorityOrder[b[1].priority] - priorityOrder[a[1].priority];
        if (priorityDiff !== 0) return priorityDiff;

        return a[1].lastAccessedAt.getTime() - b[1].lastAccessedAt.getTime();
      });

    const toRemove = entries.slice(this.config.maxEntries);
    for (const [key] of toRemove) {
      this.entries.delete(key);
    }
  }

  private enforceMaxInsights(): void {
    if (this.insights.length > this.config.maxInsights) {
      this.insights = this.insights.slice(0, this.config.maxInsights);
    }
  }

  private cleanupExpired(): void {
    const now = new Date();

    const entriesArray = Array.from(this.entries.entries());
    for (const [key, entry] of entriesArray) {
      if (entry.expiresAt && entry.expiresAt < now) {
        this.entries.delete(key);
      }
    }

    this.insights = this.insights.filter(
      (i) => !i.validUntil || i.validUntil > now
    );
  }

  private preserveSessionInsights(session: SessionState): void {
    // Add session insights to long-term storage
    for (const insight of session.insights) {
      const exists = this.insights.some((i) => i.id === insight.id);
      if (!exists && insight.confidence > 0.7) {
        this.insights.unshift(insight);
      }
    }

    // Store session summary if present
    if (session.conversationSummary) {
      this.set(`session:${session.sessionId}:summary`, {
        summary: session.conversationSummary,
        symbols: session.activeSymbols,
        insightCount: session.insights.length,
      }, {
        category: "session_summary",
        tags: session.activeSymbols,
        priority: "low",
        ttlMs: 30 * 24 * 60 * 60 * 1000, // 30 days
      });
    }
  }

  private ensureStorageDirectory(): void {
    const dir = path.dirname(this.getStorageFilePath());
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
  }

  private getStorageFilePath(): string {
    return path.join(this.config.storagePath, "memory.json");
  }

  private startAutoSave(): void {
    if (this.autoSaveTimer) {
      return;
    }

    this.autoSaveTimer = setInterval(() => {
      this.save().catch(console.error);
    }, this.config.autoSaveIntervalMs);
  }
}

// =============================================================================
// Factory Function
// =============================================================================

export function createMemoryStore(
  config?: Partial<MemoryStoreConfig>
): MemoryStore {
  return new MemoryStore(config);
}
