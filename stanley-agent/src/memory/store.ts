/**
 * Memory Store - SQLite-based key-value storage with TTL and namespacing
 *
 * Features:
 * - Namespace support (user, session, global)
 * - TTL for automatic expiration
 * - Pattern-based key search
 * - Atomic operations
 * - Type-safe value serialization
 */

import { Database } from "bun:sqlite";
import { existsSync, mkdirSync } from "fs";
import { dirname, join } from "path";

/**
 * Namespace types for organizing memory entries
 */
export type MemoryNamespace = "user" | "session" | "global" | string;

/**
 * Memory entry metadata
 */
export interface MemoryEntry<T = unknown> {
  key: string;
  namespace: MemoryNamespace;
  value: T;
  createdAt: number;
  updatedAt: number;
  expiresAt: number | null;
  metadata?: Record<string, unknown>;
}

/**
 * Store options for set operations
 */
export interface SetOptions {
  namespace?: MemoryNamespace;
  ttlSeconds?: number;
  metadata?: Record<string, unknown>;
}

/**
 * Query options for retrieval
 */
export interface QueryOptions {
  namespace?: MemoryNamespace;
  includeExpired?: boolean;
}

/**
 * Search result with relevance
 */
export interface SearchResult<T = unknown> {
  entry: MemoryEntry<T>;
  relevance: number;
}

/**
 * Memory Store class
 */
export class MemoryStore {
  private db: Database;
  private cleanupInterval: ReturnType<typeof setInterval> | null = null;

  constructor(dbPath?: string) {
    const path = dbPath || this.getDefaultDbPath();
    this.ensureDirectory(path);
    this.db = new Database(path);
    this.initializeSchema();
    this.startCleanupTask();
  }

  /**
   * Get default database path
   */
  private getDefaultDbPath(): string {
    const xdgData = process.env.XDG_DATA_HOME || join(process.env.HOME || "", ".local", "share");
    return join(xdgData, "stanley-agent", "memory.db");
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
      CREATE TABLE IF NOT EXISTS memory_store (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        key TEXT NOT NULL,
        namespace TEXT NOT NULL DEFAULT 'global',
        value TEXT NOT NULL,
        metadata TEXT,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        expires_at INTEGER,
        UNIQUE(key, namespace)
      )
    `);

    this.db.run(`
      CREATE INDEX IF NOT EXISTS idx_memory_namespace ON memory_store(namespace)
    `);

    this.db.run(`
      CREATE INDEX IF NOT EXISTS idx_memory_expires ON memory_store(expires_at)
    `);

    this.db.run(`
      CREATE INDEX IF NOT EXISTS idx_memory_key_pattern ON memory_store(key)
    `);
  }

  /**
   * Start periodic cleanup of expired entries
   */
  private startCleanupTask(): void {
    // Run cleanup every 5 minutes
    this.cleanupInterval = setInterval(() => {
      this.cleanupExpired();
    }, 5 * 60 * 1000);
  }

  /**
   * Clean up expired entries
   */
  cleanupExpired(): number {
    const now = Date.now();
    const result = this.db.run(
      "DELETE FROM memory_store WHERE expires_at IS NOT NULL AND expires_at < ?",
      [now]
    );
    return result.changes;
  }

  /**
   * Set a value in the store
   */
  set<T>(key: string, value: T, options: SetOptions = {}): void {
    const { namespace = "global", ttlSeconds, metadata } = options;
    const now = Date.now();
    const expiresAt = ttlSeconds ? now + ttlSeconds * 1000 : null;

    const serializedValue = JSON.stringify(value);
    const serializedMetadata = metadata ? JSON.stringify(metadata) : null;

    this.db.run(
      `
      INSERT INTO memory_store (key, namespace, value, metadata, created_at, updated_at, expires_at)
      VALUES (?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(key, namespace) DO UPDATE SET
        value = excluded.value,
        metadata = excluded.metadata,
        updated_at = excluded.updated_at,
        expires_at = excluded.expires_at
      `,
      [key, namespace, serializedValue, serializedMetadata, now, now, expiresAt]
    );
  }

  /**
   * Get a value from the store
   */
  get<T>(key: string, options: QueryOptions = {}): T | null {
    const { namespace = "global", includeExpired = false } = options;
    const now = Date.now();

    let query = "SELECT value FROM memory_store WHERE key = ? AND namespace = ?";
    const params: (string | number)[] = [key, namespace];

    if (!includeExpired) {
      query += " AND (expires_at IS NULL OR expires_at > ?)";
      params.push(now);
    }

    const row = this.db.query<{ value: string }, (string | number)[]>(query).get(...params);

    if (!row) return null;

    try {
      return JSON.parse(row.value) as T;
    } catch {
      return null;
    }
  }

  /**
   * Get entry with metadata
   */
  getEntry<T>(key: string, options: QueryOptions = {}): MemoryEntry<T> | null {
    const { namespace = "global", includeExpired = false } = options;
    const now = Date.now();

    let query = `
      SELECT key, namespace, value, metadata, created_at, updated_at, expires_at
      FROM memory_store
      WHERE key = ? AND namespace = ?
    `;
    const params: (string | number)[] = [key, namespace];

    if (!includeExpired) {
      query += " AND (expires_at IS NULL OR expires_at > ?)";
      params.push(now);
    }

    const row = this.db
      .query<
        {
          key: string;
          namespace: string;
          value: string;
          metadata: string | null;
          created_at: number;
          updated_at: number;
          expires_at: number | null;
        },
        (string | number)[]
      >(query)
      .get(...params);

    if (!row) return null;

    try {
      return {
        key: row.key,
        namespace: row.namespace as MemoryNamespace,
        value: JSON.parse(row.value) as T,
        createdAt: row.created_at,
        updatedAt: row.updated_at,
        expiresAt: row.expires_at,
        metadata: row.metadata ? JSON.parse(row.metadata) : undefined,
      };
    } catch {
      return null;
    }
  }

  /**
   * Delete a value from the store
   */
  delete(key: string, namespace: MemoryNamespace = "global"): boolean {
    const result = this.db.run(
      "DELETE FROM memory_store WHERE key = ? AND namespace = ?",
      [key, namespace]
    );
    return result.changes > 0;
  }

  /**
   * Check if a key exists
   */
  has(key: string, options: QueryOptions = {}): boolean {
    return this.get(key, options) !== null;
  }

  /**
   * List all keys in a namespace
   */
  keys(namespace: MemoryNamespace = "global", includeExpired = false): string[] {
    const now = Date.now();

    let query = "SELECT key FROM memory_store WHERE namespace = ?";
    const params: (string | number)[] = [namespace];

    if (!includeExpired) {
      query += " AND (expires_at IS NULL OR expires_at > ?)";
      params.push(now);
    }

    const rows = this.db.query<{ key: string }, (string | number)[]>(query).all(...params);
    return rows.map((r) => r.key);
  }

  /**
   * Search keys by pattern (glob-style: * for any, ? for single char)
   */
  searchKeys(pattern: string, options: QueryOptions = {}): string[] {
    const { namespace, includeExpired = false } = options;
    const now = Date.now();

    // Convert glob pattern to SQL LIKE pattern
    const sqlPattern = pattern.replace(/\*/g, "%").replace(/\?/g, "_");

    let query = "SELECT key FROM memory_store WHERE key LIKE ?";
    const params: (string | number)[] = [sqlPattern];

    if (namespace) {
      query += " AND namespace = ?";
      params.push(namespace);
    }

    if (!includeExpired) {
      query += " AND (expires_at IS NULL OR expires_at > ?)";
      params.push(now);
    }

    const rows = this.db.query<{ key: string }, (string | number)[]>(query).all(...params);
    return rows.map((r) => r.key);
  }

  /**
   * Search entries by key pattern with full data
   */
  search<T>(pattern: string, options: QueryOptions = {}): MemoryEntry<T>[] {
    const { namespace, includeExpired = false } = options;
    const now = Date.now();

    const sqlPattern = pattern.replace(/\*/g, "%").replace(/\?/g, "_");

    let query = `
      SELECT key, namespace, value, metadata, created_at, updated_at, expires_at
      FROM memory_store
      WHERE key LIKE ?
    `;
    const params: (string | number)[] = [sqlPattern];

    if (namespace) {
      query += " AND namespace = ?";
      params.push(namespace);
    }

    if (!includeExpired) {
      query += " AND (expires_at IS NULL OR expires_at > ?)";
      params.push(now);
    }

    const rows = this.db
      .query<
        {
          key: string;
          namespace: string;
          value: string;
          metadata: string | null;
          created_at: number;
          updated_at: number;
          expires_at: number | null;
        },
        (string | number)[]
      >(query)
      .all(...params);

    const results: MemoryEntry<T>[] = [];
    for (const row of rows) {
      try {
        results.push({
          key: row.key,
          namespace: row.namespace as MemoryNamespace,
          value: JSON.parse(row.value) as T,
          createdAt: row.created_at,
          updatedAt: row.updated_at,
          expiresAt: row.expires_at,
          metadata: row.metadata ? JSON.parse(row.metadata) : undefined,
        });
      } catch {
        // Skip invalid entries
      }
    }
    return results;
  }

  /**
   * Clear all entries in a namespace
   */
  clearNamespace(namespace: MemoryNamespace): number {
    const result = this.db.run("DELETE FROM memory_store WHERE namespace = ?", [namespace]);
    return result.changes;
  }

  /**
   * Clear all entries
   */
  clearAll(): number {
    const result = this.db.run("DELETE FROM memory_store");
    return result.changes;
  }

  /**
   * Get statistics about the store
   */
  getStats(): {
    totalEntries: number;
    byNamespace: Record<string, number>;
    expiredCount: number;
    totalSize: number;
  } {
    const now = Date.now();

    const total = this.db
      .query<{ count: number }, []>("SELECT COUNT(*) as count FROM memory_store")
      .get();

    const expired = this.db
      .query<{ count: number }, [number]>(
        "SELECT COUNT(*) as count FROM memory_store WHERE expires_at IS NOT NULL AND expires_at < ?"
      )
      .get(now);

    const byNamespace = this.db
      .query<{ namespace: string; count: number }, []>(
        "SELECT namespace, COUNT(*) as count FROM memory_store GROUP BY namespace"
      )
      .all();

    const size = this.db
      .query<{ total: number }, []>(
        "SELECT SUM(LENGTH(value) + LENGTH(key) + COALESCE(LENGTH(metadata), 0)) as total FROM memory_store"
      )
      .get();

    return {
      totalEntries: total?.count ?? 0,
      byNamespace: Object.fromEntries(byNamespace.map((r) => [r.namespace, r.count])),
      expiredCount: expired?.count ?? 0,
      totalSize: size?.total ?? 0,
    };
  }

  /**
   * Atomic increment operation
   */
  increment(key: string, amount = 1, options: SetOptions = {}): number {
    const { namespace = "global" } = options;
    const current = this.get<number>(key, { namespace });
    const newValue = (current ?? 0) + amount;
    this.set(key, newValue, options);
    return newValue;
  }

  /**
   * Atomic decrement operation
   */
  decrement(key: string, amount = 1, options: SetOptions = {}): number {
    return this.increment(key, -amount, options);
  }

  /**
   * Set with conditional (only if key doesn't exist)
   */
  setNx<T>(key: string, value: T, options: SetOptions = {}): boolean {
    if (this.has(key, { namespace: options.namespace })) {
      return false;
    }
    this.set(key, value, options);
    return true;
  }

  /**
   * Get and delete atomically
   */
  getDelete<T>(key: string, namespace: MemoryNamespace = "global"): T | null {
    const value = this.get<T>(key, { namespace });
    if (value !== null) {
      this.delete(key, namespace);
    }
    return value;
  }

  /**
   * Touch an entry to update its timestamp
   */
  touch(key: string, namespace: MemoryNamespace = "global"): boolean {
    const now = Date.now();
    const result = this.db.run(
      "UPDATE memory_store SET updated_at = ? WHERE key = ? AND namespace = ?",
      [now, key, namespace]
    );
    return result.changes > 0;
  }

  /**
   * Extend TTL for an entry
   */
  extendTtl(key: string, additionalSeconds: number, namespace: MemoryNamespace = "global"): boolean {
    const entry = this.getEntry(key, { namespace, includeExpired: true });
    if (!entry) return false;

    const newExpiresAt = (entry.expiresAt ?? Date.now()) + additionalSeconds * 1000;
    const result = this.db.run(
      "UPDATE memory_store SET expires_at = ? WHERE key = ? AND namespace = ?",
      [newExpiresAt, key, namespace]
    );
    return result.changes > 0;
  }

  /**
   * Close the database connection
   */
  close(): void {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
      this.cleanupInterval = null;
    }
    this.db.close();
  }
}

/**
 * Create a memory store instance
 */
export function createMemoryStore(dbPath?: string): MemoryStore {
  return new MemoryStore(dbPath);
}

/**
 * Singleton instance for shared access
 */
let defaultStore: MemoryStore | null = null;

export function getDefaultStore(): MemoryStore {
  if (!defaultStore) {
    defaultStore = createMemoryStore();
  }
  return defaultStore;
}

export function closeDefaultStore(): void {
  if (defaultStore) {
    defaultStore.close();
    defaultStore = null;
  }
}
