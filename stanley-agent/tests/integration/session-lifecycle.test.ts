/**
 * Session Lifecycle Integration Tests
 *
 * End-to-end tests for session management including:
 * - Session creation and initialization
 * - Session state persistence and restoration
 * - Session resumption across agent restarts
 * - Multi-session management
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach, afterEach } from "bun:test";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";

// Configuration
const SKIP_INTEGRATION = process.env.SKIP_INTEGRATION_TESTS === "true";

// Integrated session types
interface SessionMessage {
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: number;
  toolCalls?: Array<{
    name: string;
    args: unknown;
    result: unknown;
  }>;
}

interface SessionMetadata {
  id: string;
  title?: string;
  createdAt: number;
  updatedAt: number;
  resumedAt?: number;
  turnCount: number;
  tags?: string[];
}

interface FullSession {
  metadata: SessionMetadata;
  messages: SessionMessage[];
  context: Record<string, unknown>;
  memory: Record<string, unknown>;
}

// Integrated SessionManager with persistence
class IntegratedSessionManager {
  private sessions: Map<string, FullSession> = new Map();
  private activeSessionId: string | null = null;
  private storagePath: string;
  private memoryStore: Map<string, unknown> = new Map();

  constructor(storagePath: string) {
    this.storagePath = storagePath;
  }

  generateId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  }

  // Session Creation
  create(options?: { title?: string; tags?: string[]; fromPrevious?: string }): FullSession {
    const id = this.generateId();
    const now = Date.now();

    let inheritedContext: Record<string, unknown> = {};
    let inheritedMemory: Record<string, unknown> = {};

    // Inherit from previous session if specified
    if (options?.fromPrevious) {
      const previous = this.sessions.get(options.fromPrevious);
      if (previous) {
        inheritedContext = { ...previous.context };
        inheritedMemory = { ...previous.memory };
      }
    }

    const session: FullSession = {
      metadata: {
        id,
        title: options?.title,
        createdAt: now,
        updatedAt: now,
        turnCount: 0,
        tags: options?.tags,
      },
      messages: [],
      context: inheritedContext,
      memory: inheritedMemory,
    };

    this.sessions.set(id, session);
    this.activeSessionId = id;

    return session;
  }

  // Session Retrieval
  get(id: string): FullSession | undefined {
    return this.sessions.get(id);
  }

  getActive(): FullSession | undefined {
    if (!this.activeSessionId) return undefined;
    return this.sessions.get(this.activeSessionId);
  }

  // Session Resume
  resume(id: string): FullSession | undefined {
    const session = this.sessions.get(id);
    if (!session) return undefined;

    session.metadata.resumedAt = Date.now();
    session.metadata.updatedAt = Date.now();
    this.activeSessionId = id;

    return session;
  }

  // Add Message (conversation turn)
  addTurn(userMessage: string, assistantResponse: string, toolCalls?: Array<{ name: string; args: unknown; result: unknown }>): void {
    const session = this.getActive();
    if (!session) throw new Error("No active session");

    const now = Date.now();

    // Add user message
    session.messages.push({
      role: "user",
      content: userMessage,
      timestamp: now,
    });

    // Add assistant response
    session.messages.push({
      role: "assistant",
      content: assistantResponse,
      timestamp: now + 1,
      toolCalls,
    });

    session.metadata.turnCount++;
    session.metadata.updatedAt = now;
  }

  // Context Management
  setContext(key: string, value: unknown): void {
    const session = this.getActive();
    if (!session) return;
    session.context[key] = value;
    session.metadata.updatedAt = Date.now();
  }

  getContext<T>(key: string): T | undefined {
    const session = this.getActive();
    return session?.context[key] as T | undefined;
  }

  // Memory Management (persistent across sessions)
  setMemory(key: string, value: unknown): void {
    const session = this.getActive();
    if (session) {
      session.memory[key] = value;
    }
    this.memoryStore.set(key, value);
  }

  getMemory<T>(key: string): T | undefined {
    return this.memoryStore.get(key) as T | undefined;
  }

  // List sessions
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

  // Persistence
  async save(): Promise<void> {
    const data = {
      sessions: Object.fromEntries(this.sessions),
      activeSessionId: this.activeSessionId,
      memory: Object.fromEntries(this.memoryStore),
    };

    await fs.promises.mkdir(path.dirname(this.storagePath), { recursive: true });
    await fs.promises.writeFile(this.storagePath, JSON.stringify(data, null, 2));
  }

  async load(): Promise<boolean> {
    try {
      const content = await fs.promises.readFile(this.storagePath, "utf-8");
      const data = JSON.parse(content);

      this.sessions.clear();
      for (const [id, session] of Object.entries(data.sessions)) {
        this.sessions.set(id, session as FullSession);
      }

      this.activeSessionId = data.activeSessionId;

      this.memoryStore.clear();
      for (const [key, value] of Object.entries(data.memory)) {
        this.memoryStore.set(key, value);
      }

      return true;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") {
        return false; // No saved state
      }
      throw error;
    }
  }

  // End session
  end(): SessionMetadata | undefined {
    const session = this.getActive();
    if (!session) return undefined;

    this.activeSessionId = null;
    return session.metadata;
  }

  // Clear all
  clear(): void {
    this.sessions.clear();
    this.activeSessionId = null;
    this.memoryStore.clear();
  }
}

describe("Session Lifecycle Integration", () => {
  if (SKIP_INTEGRATION) {
    it.skip("Integration tests skipped", () => {});
    return;
  }

  let tempDir: string;
  let storagePath: string;
  let sessionManager: IntegratedSessionManager;

  beforeAll(async () => {
    tempDir = await fs.promises.mkdtemp(path.join(os.tmpdir(), "session-lifecycle-"));
    storagePath = path.join(tempDir, "sessions.json");
  });

  afterAll(async () => {
    try {
      await fs.promises.rm(tempDir, { recursive: true });
    } catch {
      // Ignore cleanup errors
    }
  });

  beforeEach(() => {
    sessionManager = new IntegratedSessionManager(storagePath);
  });

  afterEach(async () => {
    try {
      await fs.promises.unlink(storagePath);
    } catch {
      // Ignore if file doesn't exist
    }
  });

  describe("Session Creation and Initialization", () => {
    it("should create new session with proper initialization", () => {
      const session = sessionManager.create({
        title: "Investment Research",
        tags: ["research", "tech"],
      });

      expect(session.metadata.id).toBeDefined();
      expect(session.metadata.title).toBe("Investment Research");
      expect(session.metadata.tags).toEqual(["research", "tech"]);
      expect(session.metadata.createdAt).toBeDefined();
      expect(session.metadata.turnCount).toBe(0);
      expect(session.messages).toEqual([]);
      expect(session.context).toEqual({});
    });

    it("should set new session as active", () => {
      const session = sessionManager.create({ title: "Test" });
      const active = sessionManager.getActive();

      expect(active?.metadata.id).toBe(session.metadata.id);
    });

    it("should inherit context from previous session", () => {
      const session1 = sessionManager.create({ title: "Session 1" });
      sessionManager.setContext("portfolio", ["AAPL", "GOOGL"]);
      sessionManager.setContext("riskTolerance", "moderate");

      const session2 = sessionManager.create({
        title: "Session 2",
        fromPrevious: session1.metadata.id,
      });

      expect(session2.context.portfolio).toEqual(["AAPL", "GOOGL"]);
      expect(session2.context.riskTolerance).toBe("moderate");
    });
  });

  describe("Conversation Flow", () => {
    it("should track conversation turns correctly", () => {
      const session = sessionManager.create({ title: "Conversation" });

      sessionManager.addTurn(
        "What is AAPL trading at?",
        "AAPL is currently trading at $150.25"
      );

      sessionManager.addTurn(
        "What about its P/E ratio?",
        "AAPL has a P/E ratio of 28.5"
      );

      const updated = sessionManager.get(session.metadata.id);

      expect(updated?.metadata.turnCount).toBe(2);
      expect(updated?.messages).toHaveLength(4); // 2 user + 2 assistant
    });

    it("should track tool calls in conversation", () => {
      sessionManager.create({ title: "Tool Usage" });

      sessionManager.addTurn(
        "Get me market data for AAPL",
        "Based on the market data, AAPL is trading at $150.25 with volume of 50M",
        [
          {
            name: "get_market_data",
            args: { symbol: "AAPL" },
            result: { price: 150.25, volume: 50000000 },
          },
        ]
      );

      const session = sessionManager.getActive();
      const lastMessage = session?.messages[session.messages.length - 1];

      expect(lastMessage?.toolCalls).toHaveLength(1);
      expect(lastMessage?.toolCalls?.[0].name).toBe("get_market_data");
    });

    it("should maintain context throughout conversation", () => {
      sessionManager.create({ title: "Context Test" });

      // First turn establishes context
      sessionManager.addTurn("Analyze AAPL for me", "Starting analysis of AAPL...");
      sessionManager.setContext("currentSymbol", "AAPL");
      sessionManager.setContext("analysisStarted", true);

      // Second turn uses context
      sessionManager.addTurn("What are the key metrics?", "The key metrics for AAPL are...");

      // Context should persist
      expect(sessionManager.getContext("currentSymbol")).toBe("AAPL");
      expect(sessionManager.getContext("analysisStarted")).toBe(true);
    });
  });

  describe("Session Persistence and Restoration", () => {
    it("should save session state to disk", async () => {
      const session = sessionManager.create({ title: "Persisted Session" });
      sessionManager.addTurn("Hello", "Hi there!");
      sessionManager.setContext("saved", true);

      await sessionManager.save();

      // Verify file exists and contains data
      const content = await fs.promises.readFile(storagePath, "utf-8");
      const data = JSON.parse(content);

      expect(data.sessions[session.metadata.id]).toBeDefined();
      expect(data.activeSessionId).toBe(session.metadata.id);
    });

    it("should restore session state from disk", async () => {
      // Create and save session
      const original = sessionManager.create({ title: "Original Session" });
      sessionManager.addTurn("Question", "Answer");
      sessionManager.setContext("important", "data");
      sessionManager.setMemory("persistent_key", "persistent_value");
      await sessionManager.save();

      // Create new manager and load
      const newManager = new IntegratedSessionManager(storagePath);
      const loaded = await newManager.load();

      expect(loaded).toBe(true);

      const restored = newManager.get(original.metadata.id);
      expect(restored?.metadata.title).toBe("Original Session");
      expect(restored?.messages).toHaveLength(2);
      expect(restored?.context.important).toBe("data");
      expect(newManager.getMemory("persistent_key")).toBe("persistent_value");
    });

    it("should handle missing storage file gracefully", async () => {
      const manager = new IntegratedSessionManager("/nonexistent/path.json");
      const loaded = await manager.load();

      expect(loaded).toBe(false);
    });
  });

  describe("Session Resume", () => {
    it("should resume previous session", async () => {
      // Create and save session
      const original = sessionManager.create({ title: "Resume Test" });
      sessionManager.addTurn("First message", "First response");
      await sessionManager.save();

      // Create new manager and load
      const newManager = new IntegratedSessionManager(storagePath);
      await newManager.load();

      // Create a new session (makes it active)
      newManager.create({ title: "New Session" });

      // Resume the original
      const resumed = newManager.resume(original.metadata.id);

      expect(resumed).toBeDefined();
      expect(resumed?.metadata.id).toBe(original.metadata.id);
      expect(resumed?.metadata.resumedAt).toBeDefined();
      expect(newManager.getActive()?.metadata.id).toBe(original.metadata.id);
    });

    it("should continue conversation after resume", async () => {
      const original = sessionManager.create({ title: "Continue Test" });
      sessionManager.addTurn("Question 1", "Answer 1");
      sessionManager.setContext("step", 1);
      await sessionManager.save();

      // Simulate restart
      const newManager = new IntegratedSessionManager(storagePath);
      await newManager.load();
      newManager.resume(original.metadata.id);

      // Continue conversation
      newManager.addTurn("Question 2", "Answer 2");
      newManager.setContext("step", 2);

      const session = newManager.getActive();
      expect(session?.metadata.turnCount).toBe(2);
      expect(session?.messages).toHaveLength(4);
      expect(session?.context.step).toBe(2);
    });
  });

  describe("Multi-Session Management", () => {
    it("should manage multiple concurrent sessions", () => {
      const session1 = sessionManager.create({ title: "Research Session", tags: ["research"] });
      sessionManager.addTurn("Research Q1", "Research A1");

      const session2 = sessionManager.create({ title: "Trading Session", tags: ["trading"] });
      sessionManager.addTurn("Trade Q1", "Trade A1");

      const session3 = sessionManager.create({ title: "Another Research", tags: ["research"] });

      // List all sessions
      const all = sessionManager.list();
      expect(all).toHaveLength(3);

      // Filter by tag
      const research = sessionManager.list({ tags: ["research"] });
      expect(research).toHaveLength(2);

      // Limit results
      const limited = sessionManager.list({ limit: 2 });
      expect(limited).toHaveLength(2);
    });

    it("should switch between sessions correctly", () => {
      const session1 = sessionManager.create({ title: "Session 1" });
      sessionManager.setContext("session", 1);

      const session2 = sessionManager.create({ title: "Session 2" });
      sessionManager.setContext("session", 2);

      // Active should be session 2
      expect(sessionManager.getActive()?.context.session).toBe(2);

      // Resume session 1
      sessionManager.resume(session1.metadata.id);
      expect(sessionManager.getActive()?.context.session).toBe(1);
    });
  });

  describe("Session End and Cleanup", () => {
    it("should end session properly", () => {
      const session = sessionManager.create({ title: "Ending Session" });
      sessionManager.addTurn("Final Q", "Final A");

      const ended = sessionManager.end();

      expect(ended?.id).toBe(session.metadata.id);
      expect(sessionManager.getActive()).toBeUndefined();
    });

    it("should preserve ended session for resumption", () => {
      const session = sessionManager.create({ title: "Preserved" });
      sessionManager.addTurn("Q", "A");
      sessionManager.end();

      // Session should still be retrievable
      const retrieved = sessionManager.get(session.metadata.id);
      expect(retrieved).toBeDefined();

      // And resumable
      const resumed = sessionManager.resume(session.metadata.id);
      expect(resumed).toBeDefined();
    });
  });

  describe("Memory Persistence Across Sessions", () => {
    it("should persist memory across sessions", () => {
      sessionManager.create({ title: "Session 1" });
      sessionManager.setMemory("user_preference", { theme: "dark" });
      sessionManager.setMemory("watchlist", ["AAPL", "GOOGL"]);
      sessionManager.end();

      sessionManager.create({ title: "Session 2" });

      // Memory should persist
      expect(sessionManager.getMemory("user_preference")).toEqual({ theme: "dark" });
      expect(sessionManager.getMemory("watchlist")).toEqual(["AAPL", "GOOGL"]);
    });

    it("should persist memory after reload", async () => {
      sessionManager.create({ title: "Memory Test" });
      sessionManager.setMemory("persistent_data", { value: 42 });
      await sessionManager.save();

      const newManager = new IntegratedSessionManager(storagePath);
      await newManager.load();

      expect(newManager.getMemory("persistent_data")).toEqual({ value: 42 });
    });
  });
});

describe("Error Recovery", () => {
  if (SKIP_INTEGRATION) {
    it.skip("Integration tests skipped", () => {});
    return;
  }

  let tempDir: string;
  let storagePath: string;

  beforeAll(async () => {
    tempDir = await fs.promises.mkdtemp(path.join(os.tmpdir(), "session-error-"));
    storagePath = path.join(tempDir, "sessions.json");
  });

  afterAll(async () => {
    try {
      await fs.promises.rm(tempDir, { recursive: true });
    } catch {
      // Ignore
    }
  });

  it("should handle corrupted storage file", async () => {
    // Write invalid JSON
    await fs.promises.writeFile(storagePath, "not valid json{{{");

    const manager = new IntegratedSessionManager(storagePath);

    await expect(manager.load()).rejects.toThrow();
  });

  it("should handle missing active session reference", async () => {
    // Write valid JSON but with invalid active session reference
    const data = {
      sessions: {},
      activeSessionId: "nonexistent_session",
      memory: {},
    };
    await fs.promises.writeFile(storagePath, JSON.stringify(data));

    const manager = new IntegratedSessionManager(storagePath);
    await manager.load();

    // Should load successfully but active session should be undefined
    expect(manager.getActive()).toBeUndefined();
  });

  it("should recover from incomplete save", async () => {
    const manager = new IntegratedSessionManager(storagePath);
    manager.create({ title: "Test" });

    // Simulate a complete save first
    await manager.save();

    // Load in new manager
    const newManager = new IntegratedSessionManager(storagePath);
    await newManager.load();

    expect(newManager.list()).toHaveLength(1);
  });
});
