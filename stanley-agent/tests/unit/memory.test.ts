/**
 * Memory Persistence Tests
 *
 * Tests for the memory store system that provides persistent storage
 * for agent state, conversation history, and user preferences.
 */

import { describe, it, expect, beforeEach, afterEach, mock } from "bun:test";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";

// Mock implementations for testing
interface MemoryEntry<T = unknown> {
  key: string;
  value: T;
  createdAt: number;
  updatedAt: number;
  expiresAt?: number;
  metadata?: Record<string, unknown>;
}

interface MemoryQuery {
  prefix?: string;
  namespace?: string;
  limit?: number;
  offset?: number;
  includeExpired?: boolean;
}

// Mock MemoryStore class
class MockMemoryStore {
  private store: Map<string, MemoryEntry> = new Map();
  private namespace: string;
  private persistPath?: string;
  private autoSave: boolean;

  constructor(options?: {
    namespace?: string;
    persistPath?: string;
    autoSave?: boolean;
  }) {
    this.namespace = options?.namespace ?? "default";
    this.persistPath = options?.persistPath;
    this.autoSave = options?.autoSave ?? false;
  }

  private makeKey(key: string): string {
    return `${this.namespace}:${key}`;
  }

  async set<T>(
    key: string,
    value: T,
    options?: { ttl?: number; metadata?: Record<string, unknown> }
  ): Promise<void> {
    const now = Date.now();
    const fullKey = this.makeKey(key);
    const existing = this.store.get(fullKey);

    const entry: MemoryEntry<T> = {
      key,
      value,
      createdAt: existing?.createdAt ?? now,
      updatedAt: now,
      expiresAt: options?.ttl ? now + options.ttl : undefined,
      metadata: options?.metadata,
    };

    this.store.set(fullKey, entry);

    if (this.autoSave && this.persistPath) {
      await this.save();
    }
  }

  async get<T>(key: string): Promise<T | undefined> {
    const fullKey = this.makeKey(key);
    const entry = this.store.get(fullKey);

    if (!entry) {
      return undefined;
    }

    // Check expiration
    if (entry.expiresAt && Date.now() > entry.expiresAt) {
      this.store.delete(fullKey);
      return undefined;
    }

    return entry.value as T;
  }

  async has(key: string): Promise<boolean> {
    const value = await this.get(key);
    return value !== undefined;
  }

  async delete(key: string): Promise<boolean> {
    const fullKey = this.makeKey(key);
    return this.store.delete(fullKey);
  }

  async clear(): Promise<void> {
    const keysToDelete: string[] = [];
    for (const key of this.store.keys()) {
      if (key.startsWith(`${this.namespace}:`)) {
        keysToDelete.push(key);
      }
    }
    for (const key of keysToDelete) {
      this.store.delete(key);
    }
  }

  async keys(query?: MemoryQuery): Promise<string[]> {
    const prefix = query?.prefix ?? "";
    const limit = query?.limit ?? Infinity;
    const offset = query?.offset ?? 0;
    const includeExpired = query?.includeExpired ?? false;
    const now = Date.now();

    const matchingKeys: string[] = [];
    const namespacePrefix = `${this.namespace}:`;

    for (const [fullKey, entry] of this.store.entries()) {
      if (!fullKey.startsWith(namespacePrefix)) continue;

      const key = fullKey.slice(namespacePrefix.length);
      if (!key.startsWith(prefix)) continue;

      if (!includeExpired && entry.expiresAt && now > entry.expiresAt) continue;

      matchingKeys.push(key);
    }

    return matchingKeys.slice(offset, offset + limit);
  }

  async entries(query?: MemoryQuery): Promise<MemoryEntry[]> {
    const keys = await this.keys(query);
    const entries: MemoryEntry[] = [];

    for (const key of keys) {
      const fullKey = this.makeKey(key);
      const entry = this.store.get(fullKey);
      if (entry) {
        entries.push(entry);
      }
    }

    return entries;
  }

  async save(): Promise<void> {
    if (!this.persistPath) {
      throw new Error("No persist path configured");
    }

    const data: Record<string, MemoryEntry> = {};
    const namespacePrefix = `${this.namespace}:`;

    for (const [key, entry] of this.store.entries()) {
      if (key.startsWith(namespacePrefix)) {
        data[key] = entry;
      }
    }

    await fs.promises.writeFile(
      this.persistPath,
      JSON.stringify(data, null, 2)
    );
  }

  async load(): Promise<void> {
    if (!this.persistPath) {
      throw new Error("No persist path configured");
    }

    try {
      const content = await fs.promises.readFile(this.persistPath, "utf-8");
      const data = JSON.parse(content) as Record<string, MemoryEntry>;

      for (const [key, entry] of Object.entries(data)) {
        this.store.set(key, entry);
      }
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
        throw error;
      }
      // File doesn't exist, start with empty store
    }
  }

  getNamespace(): string {
    return this.namespace;
  }

  size(): number {
    let count = 0;
    const namespacePrefix = `${this.namespace}:`;
    for (const key of this.store.keys()) {
      if (key.startsWith(namespacePrefix)) {
        count++;
      }
    }
    return count;
  }
}

describe("Memory Store", () => {
  let memoryStore: MockMemoryStore;

  beforeEach(() => {
    memoryStore = new MockMemoryStore();
  });

  describe("Basic Operations", () => {
    describe("set/get", () => {
      it("should store and retrieve a value", async () => {
        await memoryStore.set("test-key", "test-value");
        const value = await memoryStore.get<string>("test-key");

        expect(value).toBe("test-value");
      });

      it("should store and retrieve complex objects", async () => {
        const obj = {
          name: "Stanley",
          symbols: ["AAPL", "GOOGL"],
          settings: { theme: "dark", notifications: true },
        };

        await memoryStore.set("complex", obj);
        const value = await memoryStore.get<typeof obj>("complex");

        expect(value).toEqual(obj);
      });

      it("should return undefined for non-existent key", async () => {
        const value = await memoryStore.get("nonexistent");
        expect(value).toBeUndefined();
      });

      it("should overwrite existing value", async () => {
        await memoryStore.set("key", "value1");
        await memoryStore.set("key", "value2");

        const value = await memoryStore.get<string>("key");
        expect(value).toBe("value2");
      });

      it("should preserve creation time on update", async () => {
        await memoryStore.set("key", "value1");

        // Wait a bit to ensure different timestamps
        await new Promise((resolve) => setTimeout(resolve, 10));

        await memoryStore.set("key", "value2");

        const entries = await memoryStore.entries({ prefix: "key" });
        expect(entries[0].createdAt).toBeLessThan(entries[0].updatedAt);
      });
    });

    describe("has", () => {
      it("should return true for existing key", async () => {
        await memoryStore.set("exists", "value");
        const exists = await memoryStore.has("exists");

        expect(exists).toBe(true);
      });

      it("should return false for non-existent key", async () => {
        const exists = await memoryStore.has("nonexistent");
        expect(exists).toBe(false);
      });
    });

    describe("delete", () => {
      it("should delete existing key", async () => {
        await memoryStore.set("to-delete", "value");
        const deleted = await memoryStore.delete("to-delete");

        expect(deleted).toBe(true);
        expect(await memoryStore.has("to-delete")).toBe(false);
      });

      it("should return false when deleting non-existent key", async () => {
        const deleted = await memoryStore.delete("nonexistent");
        expect(deleted).toBe(false);
      });
    });

    describe("clear", () => {
      it("should clear all keys in namespace", async () => {
        await memoryStore.set("key1", "value1");
        await memoryStore.set("key2", "value2");
        await memoryStore.set("key3", "value3");

        await memoryStore.clear();

        expect(await memoryStore.has("key1")).toBe(false);
        expect(await memoryStore.has("key2")).toBe(false);
        expect(await memoryStore.has("key3")).toBe(false);
      });
    });
  });

  describe("TTL (Time-To-Live)", () => {
    it("should expire entries after TTL", async () => {
      await memoryStore.set("expiring", "value", { ttl: 50 }); // 50ms TTL

      // Should exist immediately
      expect(await memoryStore.get("expiring")).toBe("value");

      // Wait for expiration
      await new Promise((resolve) => setTimeout(resolve, 100));

      // Should be expired now
      expect(await memoryStore.get("expiring")).toBeUndefined();
    });

    it("should not expire entries without TTL", async () => {
      await memoryStore.set("permanent", "value");

      await new Promise((resolve) => setTimeout(resolve, 50));

      expect(await memoryStore.get("permanent")).toBe("value");
    });

    it("should update expiration on set", async () => {
      await memoryStore.set("key", "value1", { ttl: 50 });
      await memoryStore.set("key", "value2", { ttl: 500 });

      await new Promise((resolve) => setTimeout(resolve, 100));

      // Should still exist with new TTL
      expect(await memoryStore.get("key")).toBe("value2");
    });
  });

  describe("Namespaces", () => {
    it("should isolate keys by namespace", async () => {
      const store1 = new MockMemoryStore({ namespace: "ns1" });
      const store2 = new MockMemoryStore({ namespace: "ns2" });

      await store1.set("key", "value1");
      await store2.set("key", "value2");

      expect(await store1.get("key")).toBe("value1");
      expect(await store2.get("key")).toBe("value2");
    });

    it("should return namespace name", () => {
      const store = new MockMemoryStore({ namespace: "custom" });
      expect(store.getNamespace()).toBe("custom");
    });

    it("should clear only keys in current namespace", async () => {
      const store1 = new MockMemoryStore({ namespace: "ns1" });
      const store2 = new MockMemoryStore({ namespace: "ns2" });

      await store1.set("key1", "value1");
      await store2.set("key2", "value2");

      await store1.clear();

      expect(await store1.has("key1")).toBe(false);
      expect(await store2.has("key2")).toBe(true);
    });
  });

  describe("Queries", () => {
    beforeEach(async () => {
      await memoryStore.set("user:profile", { name: "John" });
      await memoryStore.set("user:settings", { theme: "dark" });
      await memoryStore.set("session:current", { id: "abc" });
      await memoryStore.set("cache:data1", { value: 1 });
      await memoryStore.set("cache:data2", { value: 2 });
    });

    it("should list all keys", async () => {
      const keys = await memoryStore.keys();
      expect(keys).toHaveLength(5);
    });

    it("should filter keys by prefix", async () => {
      const keys = await memoryStore.keys({ prefix: "user:" });

      expect(keys).toHaveLength(2);
      expect(keys).toContain("user:profile");
      expect(keys).toContain("user:settings");
    });

    it("should limit number of results", async () => {
      const keys = await memoryStore.keys({ limit: 2 });
      expect(keys).toHaveLength(2);
    });

    it("should support pagination with offset", async () => {
      const page1 = await memoryStore.keys({ limit: 2, offset: 0 });
      const page2 = await memoryStore.keys({ limit: 2, offset: 2 });

      expect(page1).toHaveLength(2);
      expect(page2).toHaveLength(2);

      // Pages should not overlap
      for (const key of page1) {
        expect(page2).not.toContain(key);
      }
    });

    it("should return entries with metadata", async () => {
      await memoryStore.set("with-meta", { data: "test" }, {
        metadata: { source: "api", important: true },
      });

      const entries = await memoryStore.entries({ prefix: "with-meta" });

      expect(entries).toHaveLength(1);
      expect(entries[0].metadata).toEqual({ source: "api", important: true });
    });
  });

  describe("Persistence", () => {
    let tempDir: string;
    let persistPath: string;

    beforeEach(async () => {
      tempDir = await fs.promises.mkdtemp(path.join(os.tmpdir(), "memory-test-"));
      persistPath = path.join(tempDir, "memory.json");
    });

    afterEach(async () => {
      try {
        await fs.promises.rm(tempDir, { recursive: true });
      } catch {
        // Ignore cleanup errors
      }
    });

    it("should save memory to file", async () => {
      const store = new MockMemoryStore({ persistPath });

      await store.set("key1", "value1");
      await store.set("key2", "value2");
      await store.save();

      const content = await fs.promises.readFile(persistPath, "utf-8");
      const data = JSON.parse(content);

      expect(Object.keys(data)).toHaveLength(2);
    });

    it("should load memory from file", async () => {
      // First, save some data
      const store1 = new MockMemoryStore({ persistPath, namespace: "test" });
      await store1.set("key", "value");
      await store1.save();

      // Then load in new store
      const store2 = new MockMemoryStore({ persistPath, namespace: "test" });
      await store2.load();

      expect(await store2.get("key")).toBe("value");
    });

    it("should handle missing persist file gracefully", async () => {
      const store = new MockMemoryStore({
        persistPath: path.join(tempDir, "nonexistent.json"),
      });

      // Should not throw
      await expect(store.load()).resolves.toBeUndefined();
    });

    it("should auto-save when enabled", async () => {
      const store = new MockMemoryStore({
        persistPath,
        autoSave: true,
      });

      await store.set("key", "value");

      // File should exist after set
      const content = await fs.promises.readFile(persistPath, "utf-8");
      expect(content).toContain("key");
    });
  });

  describe("Size and Stats", () => {
    it("should return size of store", async () => {
      expect(memoryStore.size()).toBe(0);

      await memoryStore.set("key1", "value1");
      expect(memoryStore.size()).toBe(1);

      await memoryStore.set("key2", "value2");
      expect(memoryStore.size()).toBe(2);

      await memoryStore.delete("key1");
      expect(memoryStore.size()).toBe(1);
    });
  });
});

describe("Memory Store Specialized Types", () => {
  describe("Conversation History", () => {
    let memoryStore: MockMemoryStore;

    beforeEach(() => {
      memoryStore = new MockMemoryStore({ namespace: "conversations" });
    });

    it("should store conversation messages", async () => {
      const messages = [
        { role: "user", content: "What is AAPL trading at?" },
        { role: "assistant", content: "AAPL is trading at $150.25" },
        { role: "user", content: "What about GOOGL?" },
      ];

      await memoryStore.set("session-123", messages);
      const retrieved = await memoryStore.get<typeof messages>("session-123");

      expect(retrieved).toEqual(messages);
      expect(retrieved).toHaveLength(3);
    });

    it("should append to conversation history", async () => {
      const initial = [{ role: "user", content: "Hello" }];
      await memoryStore.set("session-123", initial);

      const current = await memoryStore.get<typeof initial>("session-123");
      current?.push({ role: "assistant", content: "Hi there!" });

      await memoryStore.set("session-123", current);
      const updated = await memoryStore.get<typeof initial>("session-123");

      expect(updated).toHaveLength(2);
    });
  });

  describe("User Preferences", () => {
    let memoryStore: MockMemoryStore;

    beforeEach(() => {
      memoryStore = new MockMemoryStore({ namespace: "preferences" });
    });

    it("should store user preferences", async () => {
      const preferences = {
        theme: "dark",
        defaultSymbols: ["AAPL", "GOOGL", "MSFT"],
        notifications: true,
        timezone: "America/New_York",
      };

      await memoryStore.set("user-456", preferences);
      const retrieved = await memoryStore.get<typeof preferences>("user-456");

      expect(retrieved).toEqual(preferences);
    });
  });

  describe("Cache Entries", () => {
    let memoryStore: MockMemoryStore;

    beforeEach(() => {
      memoryStore = new MockMemoryStore({ namespace: "cache" });
    });

    it("should cache API responses with TTL", async () => {
      const apiResponse = {
        symbol: "AAPL",
        price: 150.25,
        timestamp: Date.now(),
      };

      await memoryStore.set("market:AAPL", apiResponse, { ttl: 60000 });
      const cached = await memoryStore.get<typeof apiResponse>("market:AAPL");

      expect(cached).toEqual(apiResponse);
    });

    it("should expire cached data", async () => {
      await memoryStore.set("short-lived", { data: "test" }, { ttl: 10 });

      await new Promise((resolve) => setTimeout(resolve, 50));

      const cached = await memoryStore.get("short-lived");
      expect(cached).toBeUndefined();
    });
  });
});

describe("Memory Store Error Handling", () => {
  it("should handle JSON serialization errors gracefully", async () => {
    const store = new MockMemoryStore();

    // Circular references can't be serialized
    const circular: Record<string, unknown> = { name: "test" };
    circular.self = circular;

    // This should handle the error gracefully
    await expect(
      store.set("circular", circular)
    ).rejects.toThrow();
  });

  it("should handle corrupted persist file", async () => {
    const tempDir = await fs.promises.mkdtemp(path.join(os.tmpdir(), "memory-test-"));
    const persistPath = path.join(tempDir, "memory.json");

    // Write invalid JSON
    await fs.promises.writeFile(persistPath, "not valid json{{{");

    const store = new MockMemoryStore({ persistPath });

    await expect(store.load()).rejects.toThrow();

    await fs.promises.rm(tempDir, { recursive: true });
  });
});
