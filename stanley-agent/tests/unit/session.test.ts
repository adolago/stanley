/**
 * Session Management Tests
 *
 * Tests for the session management system that handles
 * session lifecycle, persistence, restoration, and state.
 */

import { describe, it, expect, beforeEach, afterEach, mock } from "bun:test";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";

// Mock Session interfaces
interface SessionMetadata {
  id: string;
  createdAt: number;
  updatedAt: number;
  expiresAt?: number;
  tags?: string[];
  title?: string;
}

interface SessionMessage {
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: number;
  metadata?: Record<string, unknown>;
}

interface SessionState {
  metadata: SessionMetadata;
  messages: SessionMessage[];
  context: Record<string, unknown>;
  toolHistory: Array<{
    name: string;
    args: unknown;
    result: unknown;
    timestamp: number;
  }>;
}

// Mock SessionManager class
class MockSessionManager {
  private sessions: Map<string, SessionState> = new Map();
  private activeSessionId: string | null = null;
  private storagePath?: string;
  private maxSessions: number;
  private sessionTimeout: number;

  constructor(options?: {
    storagePath?: string;
    maxSessions?: number;
    sessionTimeout?: number;
  }) {
    this.storagePath = options?.storagePath;
    this.maxSessions = options?.maxSessions ?? 100;
    this.sessionTimeout = options?.sessionTimeout ?? 86400000; // 24 hours
  }

  generateId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`;
  }

  create(options?: {
    title?: string;
    tags?: string[];
    initialContext?: Record<string, unknown>;
  }): SessionState {
    const id = this.generateId();
    const now = Date.now();

    const session: SessionState = {
      metadata: {
        id,
        createdAt: now,
        updatedAt: now,
        expiresAt: now + this.sessionTimeout,
        title: options?.title,
        tags: options?.tags,
      },
      messages: [],
      context: options?.initialContext ?? {},
      toolHistory: [],
    };

    this.sessions.set(id, session);
    this.activeSessionId = id;

    // Enforce max sessions limit
    this.pruneOldSessions();

    return session;
  }

  get(id: string): SessionState | undefined {
    return this.sessions.get(id);
  }

  getActive(): SessionState | undefined {
    if (!this.activeSessionId) return undefined;
    return this.sessions.get(this.activeSessionId);
  }

  setActive(id: string): boolean {
    if (!this.sessions.has(id)) return false;
    this.activeSessionId = id;
    return true;
  }

  list(options?: { limit?: number; tags?: string[] }): SessionMetadata[] {
    let sessions = Array.from(this.sessions.values())
      .map((s) => s.metadata)
      .sort((a, b) => b.updatedAt - a.updatedAt);

    if (options?.tags) {
      sessions = sessions.filter((s) =>
        options.tags!.some((tag) => s.tags?.includes(tag))
      );
    }

    if (options?.limit) {
      sessions = sessions.slice(0, options.limit);
    }

    return sessions;
  }

  update(id: string, updates: Partial<SessionState>): boolean {
    const session = this.sessions.get(id);
    if (!session) return false;

    if (updates.metadata) {
      Object.assign(session.metadata, updates.metadata);
    }
    if (updates.messages) {
      session.messages = updates.messages;
    }
    if (updates.context) {
      Object.assign(session.context, updates.context);
    }
    if (updates.toolHistory) {
      session.toolHistory = updates.toolHistory;
    }

    session.metadata.updatedAt = Date.now();

    return true;
  }

  addMessage(id: string, message: Omit<SessionMessage, "timestamp">): boolean {
    const session = this.sessions.get(id);
    if (!session) return false;

    session.messages.push({
      ...message,
      timestamp: Date.now(),
    });

    session.metadata.updatedAt = Date.now();

    return true;
  }

  addToolCall(
    id: string,
    toolCall: { name: string; args: unknown; result: unknown }
  ): boolean {
    const session = this.sessions.get(id);
    if (!session) return false;

    session.toolHistory.push({
      ...toolCall,
      timestamp: Date.now(),
    });

    session.metadata.updatedAt = Date.now();

    return true;
  }

  delete(id: string): boolean {
    if (this.activeSessionId === id) {
      this.activeSessionId = null;
    }
    return this.sessions.delete(id);
  }

  clear(): void {
    this.sessions.clear();
    this.activeSessionId = null;
  }

  async save(): Promise<void> {
    if (!this.storagePath) {
      throw new Error("No storage path configured");
    }

    const data: Record<string, SessionState> = {};
    for (const [id, session] of this.sessions) {
      data[id] = session;
    }

    await fs.promises.mkdir(path.dirname(this.storagePath), { recursive: true });
    await fs.promises.writeFile(
      this.storagePath,
      JSON.stringify(
        { sessions: data, activeSessionId: this.activeSessionId },
        null,
        2
      )
    );
  }

  async load(): Promise<void> {
    if (!this.storagePath) {
      throw new Error("No storage path configured");
    }

    try {
      const content = await fs.promises.readFile(this.storagePath, "utf-8");
      const data = JSON.parse(content);

      this.sessions.clear();
      for (const [id, session] of Object.entries(data.sessions)) {
        this.sessions.set(id, session as SessionState);
      }

      this.activeSessionId = data.activeSessionId || null;

      // Remove expired sessions
      this.pruneExpiredSessions();
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
        throw error;
      }
    }
  }

  private pruneOldSessions(): void {
    if (this.sessions.size <= this.maxSessions) return;

    const sorted = Array.from(this.sessions.entries()).sort(
      (a, b) => a[1].metadata.updatedAt - b[1].metadata.updatedAt
    );

    const toRemove = sorted.slice(0, this.sessions.size - this.maxSessions);
    for (const [id] of toRemove) {
      this.sessions.delete(id);
    }
  }

  private pruneExpiredSessions(): void {
    const now = Date.now();
    for (const [id, session] of this.sessions) {
      if (session.metadata.expiresAt && now > session.metadata.expiresAt) {
        this.sessions.delete(id);
      }
    }
  }

  getSessionCount(): number {
    return this.sessions.size;
  }

  export(id: string): SessionState | undefined {
    return this.sessions.get(id);
  }

  import(session: SessionState): boolean {
    if (!session.metadata.id) return false;
    this.sessions.set(session.metadata.id, session);
    return true;
  }
}

describe("Session Manager", () => {
  let sessionManager: MockSessionManager;

  beforeEach(() => {
    sessionManager = new MockSessionManager();
  });

  describe("Session Creation", () => {
    it("should create a new session", () => {
      const session = sessionManager.create();

      expect(session).toBeDefined();
      expect(session.metadata.id).toBeDefined();
      expect(session.metadata.createdAt).toBeDefined();
      expect(session.messages).toEqual([]);
      expect(session.toolHistory).toEqual([]);
    });

    it("should create session with title", () => {
      const session = sessionManager.create({ title: "AAPL Analysis" });

      expect(session.metadata.title).toBe("AAPL Analysis");
    });

    it("should create session with tags", () => {
      const session = sessionManager.create({ tags: ["research", "tech"] });

      expect(session.metadata.tags).toEqual(["research", "tech"]);
    });

    it("should create session with initial context", () => {
      const session = sessionManager.create({
        initialContext: { symbol: "AAPL", type: "analysis" },
      });

      expect(session.context).toEqual({ symbol: "AAPL", type: "analysis" });
    });

    it("should set new session as active", () => {
      const session = sessionManager.create();
      const active = sessionManager.getActive();

      expect(active?.metadata.id).toBe(session.metadata.id);
    });

    it("should generate unique session IDs", () => {
      const session1 = sessionManager.create();
      const session2 = sessionManager.create();

      expect(session1.metadata.id).not.toBe(session2.metadata.id);
    });
  });

  describe("Session Retrieval", () => {
    it("should get session by ID", () => {
      const created = sessionManager.create({ title: "Test" });
      const retrieved = sessionManager.get(created.metadata.id);

      expect(retrieved).toBeDefined();
      expect(retrieved?.metadata.title).toBe("Test");
    });

    it("should return undefined for non-existent session", () => {
      const session = sessionManager.get("nonexistent");
      expect(session).toBeUndefined();
    });

    it("should get active session", () => {
      sessionManager.create({ title: "Session 1" });
      const session2 = sessionManager.create({ title: "Session 2" });

      const active = sessionManager.getActive();

      expect(active?.metadata.title).toBe("Session 2");
    });

    it("should return undefined when no active session", () => {
      const active = sessionManager.getActive();
      expect(active).toBeUndefined();
    });
  });

  describe("Session Listing", () => {
    beforeEach(() => {
      sessionManager.create({ title: "First", tags: ["research"] });
      sessionManager.create({ title: "Second", tags: ["trading"] });
      sessionManager.create({ title: "Third", tags: ["research", "tech"] });
    });

    it("should list all sessions", () => {
      const list = sessionManager.list();

      expect(list).toHaveLength(3);
    });

    it("should list sessions sorted by updatedAt descending", () => {
      const list = sessionManager.list();

      expect(list[0].title).toBe("Third"); // Most recent
    });

    it("should limit number of sessions returned", () => {
      const list = sessionManager.list({ limit: 2 });

      expect(list).toHaveLength(2);
    });

    it("should filter sessions by tags", () => {
      const list = sessionManager.list({ tags: ["research"] });

      expect(list).toHaveLength(2);
      expect(list.every((s) => s.tags?.includes("research"))).toBe(true);
    });
  });

  describe("Session Updates", () => {
    it("should update session metadata", () => {
      const session = sessionManager.create();
      const id = session.metadata.id;

      sessionManager.update(id, {
        metadata: { ...session.metadata, title: "Updated Title" },
      });

      const updated = sessionManager.get(id);
      expect(updated?.metadata.title).toBe("Updated Title");
    });

    it("should update session context", () => {
      const session = sessionManager.create();
      const id = session.metadata.id;

      sessionManager.update(id, {
        context: { newKey: "newValue" },
      });

      const updated = sessionManager.get(id);
      expect(updated?.context.newKey).toBe("newValue");
    });

    it("should update updatedAt timestamp", async () => {
      const session = sessionManager.create();
      const id = session.metadata.id;
      const originalUpdatedAt = session.metadata.updatedAt;

      await new Promise((resolve) => setTimeout(resolve, 10));

      sessionManager.update(id, { context: { updated: true } });

      const updated = sessionManager.get(id);
      expect(updated?.metadata.updatedAt).toBeGreaterThan(originalUpdatedAt);
    });

    it("should return false when updating non-existent session", () => {
      const result = sessionManager.update("nonexistent", { context: {} });
      expect(result).toBe(false);
    });
  });

  describe("Message Management", () => {
    it("should add message to session", () => {
      const session = sessionManager.create();
      const id = session.metadata.id;

      sessionManager.addMessage(id, {
        role: "user",
        content: "What is AAPL trading at?",
      });

      const updated = sessionManager.get(id);
      expect(updated?.messages).toHaveLength(1);
      expect(updated?.messages[0].role).toBe("user");
      expect(updated?.messages[0].content).toBe("What is AAPL trading at?");
    });

    it("should add timestamp to message", () => {
      const before = Date.now();
      const session = sessionManager.create();

      sessionManager.addMessage(session.metadata.id, {
        role: "user",
        content: "Test",
      });

      const after = Date.now();
      const updated = sessionManager.get(session.metadata.id);

      expect(updated?.messages[0].timestamp).toBeGreaterThanOrEqual(before);
      expect(updated?.messages[0].timestamp).toBeLessThanOrEqual(after);
    });

    it("should maintain message order", () => {
      const session = sessionManager.create();
      const id = session.metadata.id;

      sessionManager.addMessage(id, { role: "user", content: "First" });
      sessionManager.addMessage(id, { role: "assistant", content: "Second" });
      sessionManager.addMessage(id, { role: "user", content: "Third" });

      const updated = sessionManager.get(id);

      expect(updated?.messages[0].content).toBe("First");
      expect(updated?.messages[1].content).toBe("Second");
      expect(updated?.messages[2].content).toBe("Third");
    });

    it("should include message metadata", () => {
      const session = sessionManager.create();

      sessionManager.addMessage(session.metadata.id, {
        role: "assistant",
        content: "Response",
        metadata: { tokens: 150, model: "claude-3" },
      });

      const updated = sessionManager.get(session.metadata.id);
      expect(updated?.messages[0].metadata).toEqual({
        tokens: 150,
        model: "claude-3",
      });
    });
  });

  describe("Tool History", () => {
    it("should add tool call to history", () => {
      const session = sessionManager.create();
      const id = session.metadata.id;

      sessionManager.addToolCall(id, {
        name: "get_market_data",
        args: { symbol: "AAPL" },
        result: { price: 150.25 },
      });

      const updated = sessionManager.get(id);
      expect(updated?.toolHistory).toHaveLength(1);
      expect(updated?.toolHistory[0].name).toBe("get_market_data");
    });

    it("should add timestamp to tool call", () => {
      const session = sessionManager.create();
      const before = Date.now();

      sessionManager.addToolCall(session.metadata.id, {
        name: "get_research",
        args: { symbol: "GOOGL" },
        result: {},
      });

      const after = Date.now();
      const updated = sessionManager.get(session.metadata.id);

      expect(updated?.toolHistory[0].timestamp).toBeGreaterThanOrEqual(before);
      expect(updated?.toolHistory[0].timestamp).toBeLessThanOrEqual(after);
    });
  });

  describe("Session Deletion", () => {
    it("should delete session", () => {
      const session = sessionManager.create();
      const id = session.metadata.id;

      const deleted = sessionManager.delete(id);

      expect(deleted).toBe(true);
      expect(sessionManager.get(id)).toBeUndefined();
    });

    it("should clear active session when deleting active", () => {
      const session = sessionManager.create();
      sessionManager.delete(session.metadata.id);

      expect(sessionManager.getActive()).toBeUndefined();
    });

    it("should return false when deleting non-existent session", () => {
      const deleted = sessionManager.delete("nonexistent");
      expect(deleted).toBe(false);
    });

    it("should clear all sessions", () => {
      sessionManager.create();
      sessionManager.create();
      sessionManager.create();

      sessionManager.clear();

      expect(sessionManager.getSessionCount()).toBe(0);
    });
  });

  describe("Active Session Management", () => {
    it("should set active session", () => {
      const session1 = sessionManager.create({ title: "Session 1" });
      const session2 = sessionManager.create({ title: "Session 2" });

      sessionManager.setActive(session1.metadata.id);

      const active = sessionManager.getActive();
      expect(active?.metadata.title).toBe("Session 1");
    });

    it("should return false when setting non-existent session as active", () => {
      const result = sessionManager.setActive("nonexistent");
      expect(result).toBe(false);
    });
  });

  describe("Session Limits", () => {
    it("should prune old sessions when exceeding max", () => {
      const manager = new MockSessionManager({ maxSessions: 3 });

      manager.create({ title: "Session 1" });
      manager.create({ title: "Session 2" });
      manager.create({ title: "Session 3" });
      manager.create({ title: "Session 4" });

      expect(manager.getSessionCount()).toBe(3);
    });
  });
});

describe("Session Persistence", () => {
  let tempDir: string;
  let storagePath: string;

  beforeEach(async () => {
    tempDir = await fs.promises.mkdtemp(path.join(os.tmpdir(), "session-test-"));
    storagePath = path.join(tempDir, "sessions.json");
  });

  afterEach(async () => {
    try {
      await fs.promises.rm(tempDir, { recursive: true });
    } catch {
      // Ignore cleanup errors
    }
  });

  it("should save sessions to file", async () => {
    const manager = new MockSessionManager({ storagePath });

    manager.create({ title: "Test Session" });
    await manager.save();

    const content = await fs.promises.readFile(storagePath, "utf-8");
    const data = JSON.parse(content);

    expect(Object.keys(data.sessions)).toHaveLength(1);
  });

  it("should load sessions from file", async () => {
    // Save first
    const manager1 = new MockSessionManager({ storagePath });
    const session = manager1.create({ title: "Persisted" });
    manager1.addMessage(session.metadata.id, {
      role: "user",
      content: "Hello",
    });
    await manager1.save();

    // Load in new manager
    const manager2 = new MockSessionManager({ storagePath });
    await manager2.load();

    const loaded = manager2.get(session.metadata.id);
    expect(loaded?.metadata.title).toBe("Persisted");
    expect(loaded?.messages).toHaveLength(1);
  });

  it("should restore active session ID", async () => {
    const manager1 = new MockSessionManager({ storagePath });
    const session = manager1.create({ title: "Active" });
    await manager1.save();

    const manager2 = new MockSessionManager({ storagePath });
    await manager2.load();

    const active = manager2.getActive();
    expect(active?.metadata.id).toBe(session.metadata.id);
  });

  it("should handle missing storage file", async () => {
    const manager = new MockSessionManager({ storagePath: "/nonexistent/path.json" });

    // Should not throw
    await expect(manager.load()).resolves.toBeUndefined();
  });
});

describe("Session Export/Import", () => {
  it("should export session", () => {
    const manager = new MockSessionManager();
    const session = manager.create({ title: "Export Me" });
    manager.addMessage(session.metadata.id, { role: "user", content: "Test" });

    const exported = manager.export(session.metadata.id);

    expect(exported).toBeDefined();
    expect(exported?.metadata.title).toBe("Export Me");
    expect(exported?.messages).toHaveLength(1);
  });

  it("should import session", () => {
    const manager = new MockSessionManager();

    const sessionData: SessionState = {
      metadata: {
        id: "imported_123",
        createdAt: Date.now(),
        updatedAt: Date.now(),
        title: "Imported Session",
      },
      messages: [{ role: "user", content: "Imported message", timestamp: Date.now() }],
      context: { imported: true },
      toolHistory: [],
    };

    const imported = manager.import(sessionData);

    expect(imported).toBe(true);
    expect(manager.get("imported_123")).toBeDefined();
    expect(manager.get("imported_123")?.context.imported).toBe(true);
  });
});

// =============================================================================
// Real Session Module Tests (using src/session)
// =============================================================================

import {
  SessionManager as RealSessionManager,
  SessionPersistence as RealSessionPersistence,
  SessionAnalytics as RealSessionAnalytics,
  createInitialState,
  validateState,
  isSessionExpired,
  calculateTotalTokenUsage,
  getToolCallFrequency,
  extractTopicsFromConversation,
  recordTokenUsage,
  recordToolCall,
  addInsight,
  addPendingAction,
  completePendingAction,
  SESSION_STATE_VERSION,
  type SessionState as RealSessionState,
} from "../../src/session";

// Test directory for real session persistence
const REAL_TEST_SESSION_DIR = path.join(os.tmpdir(), "stanley-real-session-tests");

function cleanupRealTestDir(): void {
  if (fs.existsSync(REAL_TEST_SESSION_DIR)) {
    fs.rmSync(REAL_TEST_SESSION_DIR, { recursive: true, force: true });
  }
}

describe("Real Session State Module", () => {
  describe("createInitialState", () => {
    it("creates valid initial state with metadata", () => {
      const state = createInitialState("test-session-1", "anthropic", "claude-3-opus");

      expect(state.metadata.id).toBe("test-session-1");
      expect(state.metadata.provider).toBe("anthropic");
      expect(state.metadata.model).toBe("claude-3-opus");
      expect(state.metadata.status).toBe("active");
      expect(state.metadata.version).toBe(SESSION_STATE_VERSION);
      expect(state.metadata.createdAt).toBeLessThanOrEqual(Date.now());
    });

    it("initializes empty collections", () => {
      const state = createInitialState("test-session-2", "openai", "gpt-4");

      expect(state.conversationHistory).toEqual([]);
      expect(state.toolCallHistory).toEqual([]);
      expect(state.tokenUsage).toEqual([]);
      expect(state.researchFocusAreas).toEqual([]);
    });

    it("sets default user preferences", () => {
      const state = createInitialState("test-session-3", "openrouter", "model-x");

      expect(state.userPreferences.verbosity).toBe("detailed");
      expect(state.userPreferences.riskTolerance).toBe("moderate");
      expect(state.userPreferences.analysisDepth).toBe("standard");
      expect(state.userPreferences.autoSave).toBe(true);
      expect(state.userPreferences.sessionTimeout).toBe(60);
    });
  });

  describe("validateState", () => {
    it("validates correct state structure", () => {
      const state = createInitialState("test", "provider", "model");
      expect(validateState(state)).toBe(true);
    });

    it("rejects null", () => {
      expect(validateState(null)).toBe(false);
    });

    it("rejects missing metadata", () => {
      const state = createInitialState("test", "provider", "model");
      delete (state as any).metadata;
      expect(validateState(state)).toBe(false);
    });
  });

  describe("isSessionExpired", () => {
    it("returns false for fresh session", () => {
      const state = createInitialState("test", "provider", "model");
      expect(isSessionExpired(state)).toBe(false);
    });

    it("returns true for old session", () => {
      const state = createInitialState("test", "provider", "model");
      state.metadata.lastActiveAt = Date.now() - 2 * 60 * 60 * 1000;
      expect(isSessionExpired(state, 60)).toBe(true);
    });
  });

  describe("calculateTotalTokenUsage", () => {
    it("calculates totals correctly", () => {
      const state = createInitialState("test", "provider", "model");
      state.tokenUsage = [
        { promptTokens: 100, completionTokens: 50, totalTokens: 150, timestamp: Date.now() },
        { promptTokens: 200, completionTokens: 100, totalTokens: 300, timestamp: Date.now() },
      ];

      const totals = calculateTotalTokenUsage(state);
      expect(totals.totalPromptTokens).toBe(300);
      expect(totals.totalCompletionTokens).toBe(150);
      expect(totals.totalTokens).toBe(450);
    });
  });

  describe("getToolCallFrequency", () => {
    it("counts tool calls correctly", () => {
      const state = createInitialState("test", "provider", "model");
      state.toolCallHistory = [
        { id: "1", toolName: "get_market_data", args: {}, result: {}, timestamp: Date.now(), durationMs: 100, success: true },
        { id: "2", toolName: "get_market_data", args: {}, result: {}, timestamp: Date.now(), durationMs: 100, success: true },
        { id: "3", toolName: "get_research", args: {}, result: {}, timestamp: Date.now(), durationMs: 100, success: true },
      ];

      const frequency = getToolCallFrequency(state);
      expect(frequency.get("get_market_data")).toBe(2);
      expect(frequency.get("get_research")).toBe(1);
    });
  });

  describe("state mutation helpers", () => {
    it("recordTokenUsage adds token entry", () => {
      const state = createInitialState("test", "provider", "model");
      recordTokenUsage(state, 100, 50);

      expect(state.tokenUsage.length).toBe(1);
      expect(state.tokenUsage[0].promptTokens).toBe(100);
      expect(state.tokenUsage[0].completionTokens).toBe(50);
    });

    it("recordToolCall adds tool call entry", () => {
      const state = createInitialState("test", "provider", "model");
      recordToolCall(state, "get_market_data", { symbol: "AAPL" }, { price: 150 }, 100, true);

      expect(state.toolCallHistory.length).toBe(1);
      expect(state.toolCallHistory[0].toolName).toBe("get_market_data");
    });

    it("addInsight adds unique insights", () => {
      const state = createInitialState("test", "provider", "model");
      addInsight(state, "AAPL shows strong momentum");
      addInsight(state, "Market volatility elevated");
      addInsight(state, "AAPL shows strong momentum"); // duplicate

      expect(state.activeContext.insights.length).toBe(2);
    });

    it("addPendingAction and completePendingAction", () => {
      const state = createInitialState("test", "provider", "model");
      addPendingAction(state, "Review earnings");
      addPendingAction(state, "Check portfolio");

      expect(state.activeContext.pendingActions.length).toBe(2);

      completePendingAction(state, "Review earnings");
      expect(state.activeContext.pendingActions.length).toBe(1);
    });
  });
});

describe("Real Session Persistence", () => {
  beforeEach(() => {
    cleanupRealTestDir();
    fs.mkdirSync(REAL_TEST_SESSION_DIR, { recursive: true });
  });

  afterEach(() => {
    cleanupRealTestDir();
  });

  it("saves and loads session state", async () => {
    const persistence = new RealSessionPersistence({ sessionDir: REAL_TEST_SESSION_DIR });
    const state = createInitialState("test-persist-1", "anthropic", "claude");
    state.conversationHistory = [
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi there!" },
    ];

    await persistence.save(state);
    const loaded = await persistence.load("test-persist-1");

    expect(loaded).not.toBeNull();
    expect(loaded!.metadata.id).toBe("test-persist-1");
    expect(loaded!.conversationHistory.length).toBe(2);
  });

  it("returns null for non-existent session", async () => {
    const persistence = new RealSessionPersistence({ sessionDir: REAL_TEST_SESSION_DIR });
    const loaded = await persistence.load("non-existent");
    expect(loaded).toBeNull();
  });

  it("lists all saved sessions", async () => {
    const persistence = new RealSessionPersistence({ sessionDir: REAL_TEST_SESSION_DIR });

    await persistence.save(createInitialState("session-a", "p1", "m1"));
    await persistence.save(createInitialState("session-b", "p2", "m2"));

    const sessions = await persistence.listSessions();
    expect(sessions.length).toBe(2);
  });
});

describe("Real Session Analytics", () => {
  it("generates comprehensive analytics", () => {
    const state = createInitialState("analytics-test", "anthropic", "claude");

    state.conversationHistory = [
      { role: "user", content: "What is AAPL price?" },
      { role: "assistant", content: "AAPL is trading at $150" },
    ];

    state.tokenUsage = [
      { promptTokens: 100, completionTokens: 50, totalTokens: 150, timestamp: Date.now() },
    ];

    state.toolCallHistory = [
      { id: "1", toolName: "get_market_data", args: { symbol: "AAPL" }, result: {}, timestamp: Date.now(), durationMs: 150, success: true },
    ];

    const analytics = RealSessionAnalytics.analyze(state);

    expect(analytics.sessionId).toBe("analytics-test");
    expect(analytics.conversation.totalMessages).toBe(2);
    expect(analytics.tokens.totalTokens).toBe(150);
    expect(analytics.tools.totalCalls).toBe(1);
    expect(analytics.tools.successRate).toBe(1);
  });

  it("formats duration correctly", () => {
    expect(RealSessionAnalytics.formatDuration(5000)).toBe("5s");
    expect(RealSessionAnalytics.formatDuration(125000)).toBe("2m 5s");
    expect(RealSessionAnalytics.formatDuration(3725000)).toBe("1h 2m 5s");
  });
});

describe("Real Session Manager", () => {
  beforeEach(() => {
    cleanupRealTestDir();
    fs.mkdirSync(REAL_TEST_SESSION_DIR, { recursive: true });
  });

  afterEach(() => {
    cleanupRealTestDir();
  });

  it("creates new session with metadata", async () => {
    const manager = new RealSessionManager({
      persistence: { sessionDir: REAL_TEST_SESSION_DIR },
    });

    const session = await manager.createSession("anthropic", "claude-3-opus");

    expect(session.metadata.id).toMatch(/^stanley-/);
    expect(session.metadata.provider).toBe("anthropic");
    expect(session.metadata.model).toBe("claude-3-opus");
    expect(session.metadata.status).toBe("active");

    await manager.shutdown();
  });

  it("manages conversation history", async () => {
    const manager = new RealSessionManager({
      persistence: { sessionDir: REAL_TEST_SESSION_DIR },
    });

    await manager.createSession("provider", "model");

    manager.addMessage({ role: "user", content: "Hello" });
    manager.addMessage({ role: "assistant", content: "Hi there!" });

    const history = manager.getConversationHistory();
    expect(history.length).toBe(2);
    expect(history[0].content).toBe("Hello");

    await manager.shutdown();
  });

  it("records token usage", async () => {
    const manager = new RealSessionManager({
      persistence: { sessionDir: REAL_TEST_SESSION_DIR },
    });

    await manager.createSession("provider", "model");
    manager.recordTokens(100, 50);
    manager.recordTokens(200, 75);

    const analytics = manager.getAnalytics();
    expect(analytics!.tokens.totalTokens).toBe(425);

    await manager.shutdown();
  });

  it("resumes existing session", async () => {
    const manager = new RealSessionManager({
      persistence: { sessionDir: REAL_TEST_SESSION_DIR },
      autoResumeLastSession: false,
    });

    const original = await manager.createSession("provider", "model");
    manager.addMessage({ role: "user", content: "Remember this" });
    await manager.endSession();

    const resumed = await manager.resumeSession(original.metadata.id);

    expect(resumed).not.toBeNull();
    expect(resumed!.metadata.id).toBe(original.metadata.id);
    expect(resumed!.conversationHistory.length).toBe(1);

    await manager.shutdown();
  });
});
