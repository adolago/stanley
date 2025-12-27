/**
 * Insights Extraction and Management
 *
 * Features:
 * - Track key findings from research sessions
 * - Link insights to symbols/topics
 * - Priority/importance scoring
 * - Recall relevant insights for new queries
 */

import { Database } from "bun:sqlite";
import { existsSync, mkdirSync } from "fs";
import { dirname, join } from "path";

/**
 * Insight category types
 */
export type InsightCategory =
  | "market_signal"
  | "institutional_activity"
  | "technical_pattern"
  | "fundamental_observation"
  | "risk_alert"
  | "opportunity"
  | "macro_trend"
  | "sector_rotation"
  | "earnings_insight"
  | "valuation_finding"
  | "general";

/**
 * Insight importance levels
 */
export type InsightImportance = "low" | "medium" | "high" | "critical";

/**
 * Insight entity
 */
export interface Insight {
  id: string;
  content: string;
  category: InsightCategory;
  importance: InsightImportance;
  symbols: string[];
  topics: string[];
  source?: string;
  sessionId?: string;
  confidence: number;
  createdAt: number;
  updatedAt: number;
  expiresAt?: number;
  metadata?: Record<string, unknown>;
}

/**
 * Insight creation options
 */
export interface CreateInsightOptions {
  content: string;
  category?: InsightCategory;
  importance?: InsightImportance;
  symbols?: string[];
  topics?: string[];
  source?: string;
  sessionId?: string;
  confidence?: number;
  ttlDays?: number;
  metadata?: Record<string, unknown>;
}

/**
 * Insight search options
 */
export interface SearchInsightOptions {
  symbols?: string[];
  topics?: string[];
  categories?: InsightCategory[];
  minImportance?: InsightImportance;
  minConfidence?: number;
  limit?: number;
  includeExpired?: boolean;
}

/**
 * Insight recall result
 */
export interface InsightRecall {
  insight: Insight;
  relevanceScore: number;
  matchedSymbols: string[];
  matchedTopics: string[];
}

/**
 * Importance score mapping
 */
const IMPORTANCE_SCORES: Record<InsightImportance, number> = {
  low: 1,
  medium: 2,
  high: 3,
  critical: 4,
};

/**
 * Insights Manager
 */
export class InsightsManager {
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
    return join(xdgData, "stanley-agent", "insights.db");
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
      CREATE TABLE IF NOT EXISTS insights (
        id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        category TEXT NOT NULL DEFAULT 'general',
        importance TEXT NOT NULL DEFAULT 'medium',
        source TEXT,
        session_id TEXT,
        confidence REAL NOT NULL DEFAULT 0.5,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        expires_at INTEGER,
        metadata TEXT
      )
    `);

    this.db.run(`
      CREATE TABLE IF NOT EXISTS insight_symbols (
        insight_id TEXT NOT NULL,
        symbol TEXT NOT NULL,
        PRIMARY KEY (insight_id, symbol),
        FOREIGN KEY (insight_id) REFERENCES insights(id) ON DELETE CASCADE
      )
    `);

    this.db.run(`
      CREATE TABLE IF NOT EXISTS insight_topics (
        insight_id TEXT NOT NULL,
        topic TEXT NOT NULL,
        PRIMARY KEY (insight_id, topic),
        FOREIGN KEY (insight_id) REFERENCES insights(id) ON DELETE CASCADE
      )
    `);

    this.db.run(`
      CREATE INDEX IF NOT EXISTS idx_insights_category ON insights(category)
    `);

    this.db.run(`
      CREATE INDEX IF NOT EXISTS idx_insights_importance ON insights(importance)
    `);

    this.db.run(`
      CREATE INDEX IF NOT EXISTS idx_insights_created ON insights(created_at DESC)
    `);

    this.db.run(`
      CREATE INDEX IF NOT EXISTS idx_symbols_symbol ON insight_symbols(symbol)
    `);

    this.db.run(`
      CREATE INDEX IF NOT EXISTS idx_topics_topic ON insight_topics(topic)
    `);

    // Full-text search for content
    this.db.run(`
      CREATE VIRTUAL TABLE IF NOT EXISTS insights_fts USING fts5(
        content,
        content_rowid='id'
      )
    `);
  }

  /**
   * Generate unique ID
   */
  private generateId(): string {
    return `ins-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
  }

  /**
   * Create a new insight
   */
  create(options: CreateInsightOptions): Insight {
    const id = this.generateId();
    const now = Date.now();

    const insight: Insight = {
      id,
      content: options.content,
      category: options.category || "general",
      importance: options.importance || "medium",
      symbols: options.symbols || [],
      topics: options.topics || [],
      source: options.source,
      sessionId: options.sessionId,
      confidence: options.confidence ?? 0.5,
      createdAt: now,
      updatedAt: now,
      expiresAt: options.ttlDays ? now + options.ttlDays * 24 * 60 * 60 * 1000 : undefined,
      metadata: options.metadata,
    };

    // Insert main insight
    this.db.run(
      `
      INSERT INTO insights (id, content, category, importance, source, session_id, confidence, created_at, updated_at, expires_at, metadata)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `,
      [
        id,
        insight.content,
        insight.category,
        insight.importance,
        insight.source || null,
        insight.sessionId || null,
        insight.confidence,
        insight.createdAt,
        insight.updatedAt,
        insight.expiresAt || null,
        insight.metadata ? JSON.stringify(insight.metadata) : null,
      ]
    );

    // Insert symbols
    for (const symbol of insight.symbols) {
      this.db.run(
        "INSERT OR IGNORE INTO insight_symbols (insight_id, symbol) VALUES (?, ?)",
        [id, symbol.toUpperCase()]
      );
    }

    // Insert topics
    for (const topic of insight.topics) {
      this.db.run(
        "INSERT OR IGNORE INTO insight_topics (insight_id, topic) VALUES (?, ?)",
        [id, topic.toLowerCase()]
      );
    }

    // Update FTS index
    this.db.run("INSERT INTO insights_fts (rowid, content) VALUES (?, ?)", [id, insight.content]);

    return insight;
  }

  /**
   * Get an insight by ID
   */
  get(id: string): Insight | null {
    const row = this.db
      .query<
        {
          id: string;
          content: string;
          category: string;
          importance: string;
          source: string | null;
          session_id: string | null;
          confidence: number;
          created_at: number;
          updated_at: number;
          expires_at: number | null;
          metadata: string | null;
        },
        [string]
      >("SELECT * FROM insights WHERE id = ?")
      .get(id);

    if (!row) return null;

    return this.hydrateInsight(row);
  }

  /**
   * Hydrate insight with symbols and topics
   */
  private hydrateInsight(row: {
    id: string;
    content: string;
    category: string;
    importance: string;
    source: string | null;
    session_id: string | null;
    confidence: number;
    created_at: number;
    updated_at: number;
    expires_at: number | null;
    metadata: string | null;
  }): Insight {
    const symbols = this.db
      .query<{ symbol: string }, [string]>(
        "SELECT symbol FROM insight_symbols WHERE insight_id = ?"
      )
      .all(row.id)
      .map((r) => r.symbol);

    const topics = this.db
      .query<{ topic: string }, [string]>(
        "SELECT topic FROM insight_topics WHERE insight_id = ?"
      )
      .all(row.id)
      .map((r) => r.topic);

    return {
      id: row.id,
      content: row.content,
      category: row.category as InsightCategory,
      importance: row.importance as InsightImportance,
      symbols,
      topics,
      source: row.source || undefined,
      sessionId: row.session_id || undefined,
      confidence: row.confidence,
      createdAt: row.created_at,
      updatedAt: row.updated_at,
      expiresAt: row.expires_at || undefined,
      metadata: row.metadata ? JSON.parse(row.metadata) : undefined,
    };
  }

  /**
   * Update an insight
   */
  update(id: string, updates: Partial<CreateInsightOptions>): Insight | null {
    const existing = this.get(id);
    if (!existing) return null;

    const now = Date.now();

    // Update main fields
    const fields: string[] = ["updated_at = ?"];
    const values: (string | number | null)[] = [now];

    if (updates.content !== undefined) {
      fields.push("content = ?");
      values.push(updates.content);
    }
    if (updates.category !== undefined) {
      fields.push("category = ?");
      values.push(updates.category);
    }
    if (updates.importance !== undefined) {
      fields.push("importance = ?");
      values.push(updates.importance);
    }
    if (updates.confidence !== undefined) {
      fields.push("confidence = ?");
      values.push(updates.confidence);
    }
    if (updates.metadata !== undefined) {
      fields.push("metadata = ?");
      values.push(JSON.stringify(updates.metadata));
    }

    values.push(id);
    this.db.run(`UPDATE insights SET ${fields.join(", ")} WHERE id = ?`, values);

    // Update symbols if provided
    if (updates.symbols !== undefined) {
      this.db.run("DELETE FROM insight_symbols WHERE insight_id = ?", [id]);
      for (const symbol of updates.symbols) {
        this.db.run(
          "INSERT OR IGNORE INTO insight_symbols (insight_id, symbol) VALUES (?, ?)",
          [id, symbol.toUpperCase()]
        );
      }
    }

    // Update topics if provided
    if (updates.topics !== undefined) {
      this.db.run("DELETE FROM insight_topics WHERE insight_id = ?", [id]);
      for (const topic of updates.topics) {
        this.db.run(
          "INSERT OR IGNORE INTO insight_topics (insight_id, topic) VALUES (?, ?)",
          [id, topic.toLowerCase()]
        );
      }
    }

    // Update FTS if content changed
    if (updates.content !== undefined) {
      this.db.run("DELETE FROM insights_fts WHERE rowid = ?", [id]);
      this.db.run("INSERT INTO insights_fts (rowid, content) VALUES (?, ?)", [id, updates.content]);
    }

    return this.get(id);
  }

  /**
   * Delete an insight
   */
  delete(id: string): boolean {
    this.db.run("DELETE FROM insight_symbols WHERE insight_id = ?", [id]);
    this.db.run("DELETE FROM insight_topics WHERE insight_id = ?", [id]);
    this.db.run("DELETE FROM insights_fts WHERE rowid = ?", [id]);
    const result = this.db.run("DELETE FROM insights WHERE id = ?", [id]);
    return result.changes > 0;
  }

  /**
   * Search insights
   */
  search(options: SearchInsightOptions = {}): Insight[] {
    const {
      symbols,
      topics,
      categories,
      minImportance,
      minConfidence,
      limit = 50,
      includeExpired = false,
    } = options;

    const now = Date.now();
    let sql = "SELECT DISTINCT i.* FROM insights i";
    const joins: string[] = [];
    const conditions: string[] = [];
    const params: (string | number)[] = [];

    // Join for symbol filtering
    if (symbols && symbols.length > 0) {
      joins.push("INNER JOIN insight_symbols s ON i.id = s.insight_id");
      conditions.push(`s.symbol IN (${symbols.map(() => "?").join(", ")})`);
      params.push(...symbols.map((s) => s.toUpperCase()));
    }

    // Join for topic filtering
    if (topics && topics.length > 0) {
      joins.push("INNER JOIN insight_topics t ON i.id = t.insight_id");
      conditions.push(`t.topic IN (${topics.map(() => "?").join(", ")})`);
      params.push(...topics.map((t) => t.toLowerCase()));
    }

    // Category filter
    if (categories && categories.length > 0) {
      conditions.push(`i.category IN (${categories.map(() => "?").join(", ")})`);
      params.push(...categories);
    }

    // Importance filter
    if (minImportance) {
      const importanceValues = Object.entries(IMPORTANCE_SCORES)
        .filter(([, score]) => score >= IMPORTANCE_SCORES[minImportance])
        .map(([imp]) => imp);
      conditions.push(`i.importance IN (${importanceValues.map(() => "?").join(", ")})`);
      params.push(...importanceValues);
    }

    // Confidence filter
    if (minConfidence !== undefined) {
      conditions.push("i.confidence >= ?");
      params.push(minConfidence);
    }

    // Expiration filter
    if (!includeExpired) {
      conditions.push("(i.expires_at IS NULL OR i.expires_at > ?)");
      params.push(now);
    }

    // Build final query
    sql += " " + joins.join(" ");
    if (conditions.length > 0) {
      sql += " WHERE " + conditions.join(" AND ");
    }
    sql += " ORDER BY i.created_at DESC LIMIT ?";
    params.push(limit);

    const rows = this.db
      .query<
        {
          id: string;
          content: string;
          category: string;
          importance: string;
          source: string | null;
          session_id: string | null;
          confidence: number;
          created_at: number;
          updated_at: number;
          expires_at: number | null;
          metadata: string | null;
        },
        (string | number)[]
      >(sql)
      .all(...params);

    return rows.map((row) => this.hydrateInsight(row));
  }

  /**
   * Full-text search on content
   */
  searchContent(query: string, options: SearchInsightOptions = {}): Insight[] {
    const { limit = 50 } = options;

    const rows = this.db
      .query<{ id: string }, [string, number]>(
        `
        SELECT i.id
        FROM insights i
        INNER JOIN insights_fts f ON i.id = f.rowid
        WHERE insights_fts MATCH ?
        ORDER BY rank
        LIMIT ?
        `
      )
      .all(query, limit);

    return rows
      .map((r) => this.get(r.id))
      .filter((i): i is Insight => i !== null);
  }

  /**
   * Recall relevant insights for a query context
   */
  recall(context: {
    symbols?: string[];
    topics?: string[];
    query?: string;
  }): InsightRecall[] {
    const { symbols = [], topics = [], query } = context;
    const results: Map<string, InsightRecall> = new Map();

    // Search by symbols
    if (symbols.length > 0) {
      const symbolInsights = this.search({ symbols, limit: 20 });
      for (const insight of symbolInsights) {
        const matched = insight.symbols.filter((s) =>
          symbols.map((sym) => sym.toUpperCase()).includes(s)
        );
        const relevance = (matched.length / Math.max(symbols.length, 1)) * 0.5 +
          IMPORTANCE_SCORES[insight.importance] * 0.125 +
          insight.confidence * 0.25;

        results.set(insight.id, {
          insight,
          relevanceScore: relevance,
          matchedSymbols: matched,
          matchedTopics: [],
        });
      }
    }

    // Search by topics
    if (topics.length > 0) {
      const topicInsights = this.search({ topics, limit: 20 });
      for (const insight of topicInsights) {
        const matchedTopics = insight.topics.filter((t) =>
          topics.map((top) => top.toLowerCase()).includes(t)
        );
        const existing = results.get(insight.id);

        if (existing) {
          existing.matchedTopics = matchedTopics;
          existing.relevanceScore += (matchedTopics.length / Math.max(topics.length, 1)) * 0.3;
        } else {
          const relevance =
            (matchedTopics.length / Math.max(topics.length, 1)) * 0.5 +
            IMPORTANCE_SCORES[insight.importance] * 0.125 +
            insight.confidence * 0.25;

          results.set(insight.id, {
            insight,
            relevanceScore: relevance,
            matchedSymbols: [],
            matchedTopics,
          });
        }
      }
    }

    // Full-text search on query
    if (query) {
      try {
        const contentInsights = this.searchContent(query, { limit: 10 });
        for (const insight of contentInsights) {
          const existing = results.get(insight.id);
          if (existing) {
            existing.relevanceScore += 0.2;
          } else {
            results.set(insight.id, {
              insight,
              relevanceScore:
                0.3 + IMPORTANCE_SCORES[insight.importance] * 0.1 + insight.confidence * 0.2,
              matchedSymbols: [],
              matchedTopics: [],
            });
          }
        }
      } catch {
        // FTS query might fail for complex queries, ignore
      }
    }

    // Sort by relevance and return
    return Array.from(results.values())
      .sort((a, b) => b.relevanceScore - a.relevanceScore)
      .slice(0, 20);
  }

  /**
   * Get insights for a specific symbol
   */
  getBySymbol(symbol: string, options?: { limit?: number }): Insight[] {
    return this.search({ symbols: [symbol], limit: options?.limit || 20 });
  }

  /**
   * Get insights for a specific topic
   */
  getByTopic(topic: string, options?: { limit?: number }): Insight[] {
    return this.search({ topics: [topic], limit: options?.limit || 20 });
  }

  /**
   * Get insights by category
   */
  getByCategory(category: InsightCategory, options?: { limit?: number }): Insight[] {
    return this.search({ categories: [category], limit: options?.limit || 20 });
  }

  /**
   * Get high priority insights
   */
  getHighPriority(options?: { limit?: number }): Insight[] {
    return this.search({
      minImportance: "high",
      limit: options?.limit || 20,
    });
  }

  /**
   * Get recent insights
   */
  getRecent(limit = 20): Insight[] {
    return this.search({ limit });
  }

  /**
   * Clean up expired insights
   */
  cleanupExpired(): number {
    const now = Date.now();

    // Get IDs to delete
    const toDelete = this.db
      .query<{ id: string }, [number]>(
        "SELECT id FROM insights WHERE expires_at IS NOT NULL AND expires_at < ?"
      )
      .all(now);

    let deleted = 0;
    for (const { id } of toDelete) {
      if (this.delete(id)) {
        deleted++;
      }
    }

    return deleted;
  }

  /**
   * Get statistics
   */
  getStats(): {
    totalInsights: number;
    byCategory: Record<string, number>;
    byImportance: Record<string, number>;
    uniqueSymbols: number;
    uniqueTopics: number;
    avgConfidence: number;
  } {
    const total = this.db
      .query<{ count: number }, []>("SELECT COUNT(*) as count FROM insights")
      .get();

    const byCategory = this.db
      .query<{ category: string; count: number }, []>(
        "SELECT category, COUNT(*) as count FROM insights GROUP BY category"
      )
      .all();

    const byImportance = this.db
      .query<{ importance: string; count: number }, []>(
        "SELECT importance, COUNT(*) as count FROM insights GROUP BY importance"
      )
      .all();

    const symbols = this.db
      .query<{ count: number }, []>(
        "SELECT COUNT(DISTINCT symbol) as count FROM insight_symbols"
      )
      .get();

    const topics = this.db
      .query<{ count: number }, []>(
        "SELECT COUNT(DISTINCT topic) as count FROM insight_topics"
      )
      .get();

    const avgConfidence = this.db
      .query<{ avg: number | null }, []>(
        "SELECT AVG(confidence) as avg FROM insights"
      )
      .get();

    return {
      totalInsights: total?.count || 0,
      byCategory: Object.fromEntries(byCategory.map((r) => [r.category, r.count])),
      byImportance: Object.fromEntries(byImportance.map((r) => [r.importance, r.count])),
      uniqueSymbols: symbols?.count || 0,
      uniqueTopics: topics?.count || 0,
      avgConfidence: avgConfidence?.avg ?? 0,
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
 * Create an insights manager instance
 */
export function createInsightsManager(dbPath?: string): InsightsManager {
  return new InsightsManager(dbPath);
}

/**
 * Singleton instance for shared access
 */
let defaultInsightsManager: InsightsManager | null = null;

export function getDefaultInsightsManager(): InsightsManager {
  if (!defaultInsightsManager) {
    defaultInsightsManager = createInsightsManager();
  }
  return defaultInsightsManager;
}

export function closeDefaultInsightsManager(): void {
  if (defaultInsightsManager) {
    defaultInsightsManager.close();
    defaultInsightsManager = null;
  }
}
