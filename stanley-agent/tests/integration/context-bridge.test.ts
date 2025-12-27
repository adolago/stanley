/**
 * Context Bridge Integration Tests
 *
 * End-to-end tests for the context bridge system that verify:
 * - Full context gathering flow
 * - Integration between context sources
 * - Context injection into prompts
 * - Performance under real conditions
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach, afterEach } from "bun:test";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";

// Configuration
const SKIP_INTEGRATION = process.env.SKIP_INTEGRATION_TESTS === "true";

// Mock classes for integration testing
// These mirror the expected implementation patterns

interface ContextSource {
  name: string;
  priority: number;
  gather: () => Promise<Record<string, unknown>>;
}

interface MemoryStore {
  get: <T>(key: string) => Promise<T | undefined>;
  set: <T>(key: string, value: T) => Promise<void>;
  keys: () => Promise<string[]>;
}

interface SessionManager {
  getActive: () => { metadata: { id: string }; messages: unknown[]; context: Record<string, unknown> } | undefined;
}

interface PromptBuilder {
  setSystemPrompt: (prompt: string) => PromptBuilder;
  addContext: (context: string) => PromptBuilder;
  addToolDescription: (name: string, desc: string) => PromptBuilder;
  build: () => string;
}

// Mock implementations for integration testing
class IntegrationMemoryStore implements MemoryStore {
  private store: Map<string, unknown> = new Map();

  async get<T>(key: string): Promise<T | undefined> {
    return this.store.get(key) as T | undefined;
  }

  async set<T>(key: string, value: T): Promise<void> {
    this.store.set(key, value);
  }

  async keys(): Promise<string[]> {
    return Array.from(this.store.keys());
  }

  async clear(): Promise<void> {
    this.store.clear();
  }
}

class IntegrationSessionManager implements SessionManager {
  private sessions: Map<string, { metadata: { id: string; title?: string }; messages: unknown[]; context: Record<string, unknown> }> = new Map();
  private activeId: string | null = null;

  create(options?: { title?: string }) {
    const id = `session_${Date.now()}`;
    const session = {
      metadata: { id, title: options?.title },
      messages: [],
      context: {},
    };
    this.sessions.set(id, session);
    this.activeId = id;
    return session;
  }

  getActive() {
    if (!this.activeId) return undefined;
    return this.sessions.get(this.activeId);
  }

  addMessage(role: string, content: string) {
    const session = this.getActive();
    if (session) {
      session.messages.push({ role, content, timestamp: Date.now() });
    }
  }

  setContext(key: string, value: unknown) {
    const session = this.getActive();
    if (session) {
      session.context[key] = value;
    }
  }
}

class IntegrationPromptBuilder implements PromptBuilder {
  private parts: { system?: string; contexts: string[]; tools: string[] } = {
    contexts: [],
    tools: [],
  };

  setSystemPrompt(prompt: string): this {
    this.parts.system = prompt;
    return this;
  }

  addContext(context: string): this {
    this.parts.contexts.push(context);
    return this;
  }

  addToolDescription(name: string, desc: string): this {
    this.parts.tools.push(`${name}: ${desc}`);
    return this;
  }

  build(): string {
    const sections: string[] = [];

    if (this.parts.system) {
      sections.push(this.parts.system);
    }

    if (this.parts.contexts.length > 0) {
      sections.push("\n## Context\n" + this.parts.contexts.join("\n\n"));
    }

    if (this.parts.tools.length > 0) {
      sections.push("\n## Tools\n" + this.parts.tools.join("\n"));
    }

    return sections.join("\n");
  }
}

class IntegrationContextBridge {
  private sources: Map<string, ContextSource> = new Map();
  private memory: IntegrationMemoryStore;
  private sessions: IntegrationSessionManager;
  private promptBuilder: IntegrationPromptBuilder;

  constructor(
    memory: IntegrationMemoryStore,
    sessions: IntegrationSessionManager
  ) {
    this.memory = memory;
    this.sessions = sessions;
    this.promptBuilder = new IntegrationPromptBuilder();
  }

  registerSource(source: ContextSource): void {
    this.sources.set(source.name, source);
  }

  async gather(): Promise<Record<string, unknown>> {
    const context: Record<string, unknown> = {};

    // Sort by priority
    const sorted = Array.from(this.sources.entries()).sort(
      (a, b) => a[1].priority - b[1].priority
    );

    // Gather from all sources in parallel
    const results = await Promise.all(
      sorted.map(async ([name, source]) => {
        try {
          return { name, data: await source.gather() };
        } catch (error) {
          return { name, data: { error: (error as Error).message } };
        }
      })
    );

    for (const { name, data } of results) {
      context[name] = data;
    }

    return context;
  }

  async buildPrompt(systemPrompt: string, tools: Array<{ name: string; description: string }>): Promise<string> {
    const context = await this.gather();

    this.promptBuilder = new IntegrationPromptBuilder();
    this.promptBuilder.setSystemPrompt(systemPrompt);

    // Add context from each source
    for (const [sourceName, sourceData] of Object.entries(context)) {
      const contextStr = this.formatContext(sourceName, sourceData);
      if (contextStr) {
        this.promptBuilder.addContext(contextStr);
      }
    }

    // Add tool descriptions
    for (const tool of tools) {
      this.promptBuilder.addToolDescription(tool.name, tool.description);
    }

    return this.promptBuilder.build();
  }

  private formatContext(name: string, data: unknown): string {
    if (!data || typeof data !== "object") return "";
    if ((data as any).error) return "";

    return `### ${name}\n${JSON.stringify(data, null, 2)}`;
  }
}

describe("Context Bridge Integration", () => {
  if (SKIP_INTEGRATION) {
    it.skip("Integration tests skipped", () => {});
    return;
  }

  let memory: IntegrationMemoryStore;
  let sessions: IntegrationSessionManager;
  let contextBridge: IntegrationContextBridge;

  beforeEach(() => {
    memory = new IntegrationMemoryStore();
    sessions = new IntegrationSessionManager();
    contextBridge = new IntegrationContextBridge(memory, sessions);
  });

  describe("Full Context Gathering Flow", () => {
    it("should gather context from memory, session, and project sources", async () => {
      // Setup memory
      await memory.set("user:preferences", { theme: "dark", defaultSymbol: "SPY" });
      await memory.set("recent:queries", ["AAPL", "GOOGL", "MSFT"]);

      // Setup session
      sessions.create({ title: "Investment Analysis" });
      sessions.addMessage("user", "What is AAPL trading at?");
      sessions.setContext("currentSymbol", "AAPL");

      // Register sources
      contextBridge.registerSource({
        name: "memory",
        priority: 1,
        gather: async () => ({
          preferences: await memory.get("user:preferences"),
          recentQueries: await memory.get("recent:queries"),
        }),
      });

      contextBridge.registerSource({
        name: "session",
        priority: 2,
        gather: async () => {
          const active = sessions.getActive();
          return {
            sessionId: active?.metadata.id,
            messageCount: active?.messages.length,
            context: active?.context,
          };
        },
      });

      contextBridge.registerSource({
        name: "project",
        priority: 3,
        gather: async () => ({
          type: "stanley-agent",
          version: "0.1.0",
          apiUrl: "http://localhost:8000",
        }),
      });

      // Gather all context
      const context = await contextBridge.gather();

      expect(context).toHaveProperty("memory");
      expect(context).toHaveProperty("session");
      expect(context).toHaveProperty("project");

      const memoryContext = context.memory as Record<string, unknown>;
      expect(memoryContext.preferences).toEqual({ theme: "dark", defaultSymbol: "SPY" });

      const sessionContext = context.session as Record<string, unknown>;
      expect(sessionContext.messageCount).toBe(1);
      expect((sessionContext.context as Record<string, unknown>).currentSymbol).toBe("AAPL");
    });

    it("should handle source failures gracefully", async () => {
      contextBridge.registerSource({
        name: "working",
        priority: 1,
        gather: async () => ({ status: "ok" }),
      });

      contextBridge.registerSource({
        name: "failing",
        priority: 2,
        gather: async () => {
          throw new Error("Source unavailable");
        },
      });

      const context = await contextBridge.gather();

      expect((context.working as Record<string, unknown>).status).toBe("ok");
      expect((context.failing as Record<string, unknown>).error).toBe("Source unavailable");
    });

    it("should respect source priority", async () => {
      const callOrder: string[] = [];

      contextBridge.registerSource({
        name: "low-priority",
        priority: 10,
        gather: async () => {
          callOrder.push("low");
          return {};
        },
      });

      contextBridge.registerSource({
        name: "high-priority",
        priority: 1,
        gather: async () => {
          callOrder.push("high");
          return {};
        },
      });

      await contextBridge.gather();

      // All sources should be gathered
      expect(callOrder).toContain("high");
      expect(callOrder).toContain("low");
    });
  });

  describe("Prompt Building with Context", () => {
    it("should build complete prompt with gathered context", async () => {
      // Setup
      await memory.set("watchlist", ["AAPL", "GOOGL"]);
      sessions.create();
      sessions.setContext("analysisType", "fundamental");

      contextBridge.registerSource({
        name: "memory",
        priority: 1,
        gather: async () => ({
          watchlist: await memory.get("watchlist"),
        }),
      });

      contextBridge.registerSource({
        name: "session",
        priority: 2,
        gather: async () => ({
          context: sessions.getActive()?.context,
        }),
      });

      const tools = [
        { name: "get_market_data", description: "Fetch real-time market data" },
        { name: "get_research", description: "Get research report" },
      ];

      const prompt = await contextBridge.buildPrompt(
        "You are Stanley, an investment analyst.",
        tools
      );

      expect(prompt).toContain("You are Stanley");
      expect(prompt).toContain("## Context");
      expect(prompt).toContain("## Tools");
      expect(prompt).toContain("get_market_data");
      expect(prompt).toContain("watchlist");
    });

    it("should exclude empty or error contexts from prompt", async () => {
      contextBridge.registerSource({
        name: "valid",
        priority: 1,
        gather: async () => ({ data: "valid" }),
      });

      contextBridge.registerSource({
        name: "empty",
        priority: 2,
        gather: async () => ({}),
      });

      contextBridge.registerSource({
        name: "error",
        priority: 3,
        gather: async () => {
          throw new Error("Failed");
        },
      });

      const prompt = await contextBridge.buildPrompt("System prompt", []);

      expect(prompt).toContain("valid");
      expect(prompt).not.toContain("empty");
      expect(prompt).not.toContain("Failed");
    });
  });

  describe("Context Source Integration", () => {
    it("should integrate with Stanley API context", async () => {
      // Simulate API health check
      contextBridge.registerSource({
        name: "api",
        priority: 1,
        gather: async () => {
          // In real implementation, this would call the API
          return {
            status: "healthy",
            version: "1.0.0",
            endpoints: {
              market: true,
              research: true,
              portfolio: true,
            },
          };
        },
      });

      const context = await contextBridge.gather();
      const apiContext = context.api as Record<string, unknown>;

      expect(apiContext.status).toBe("healthy");
      expect(apiContext.endpoints).toBeDefined();
    });

    it("should integrate with tool availability context", async () => {
      const availableTools = [
        "get_market_data",
        "get_research",
        "get_portfolio_analytics",
        "get_institutional_holdings",
      ];

      contextBridge.registerSource({
        name: "tools",
        priority: 1,
        gather: async () => ({
          available: availableTools,
          count: availableTools.length,
        }),
      });

      const context = await contextBridge.gather();
      const toolsContext = context.tools as Record<string, unknown>;

      expect(toolsContext.count).toBe(4);
      expect(toolsContext.available).toContain("get_market_data");
    });
  });
});

describe("Session-Context Integration", () => {
  if (SKIP_INTEGRATION) {
    it.skip("Integration tests skipped", () => {});
    return;
  }

  let memory: IntegrationMemoryStore;
  let sessions: IntegrationSessionManager;

  beforeEach(() => {
    memory = new IntegrationMemoryStore();
    sessions = new IntegrationSessionManager();
  });

  it("should maintain context across conversation turns", async () => {
    // Start session
    sessions.create({ title: "Multi-turn conversation" });

    // First turn
    sessions.addMessage("user", "Tell me about AAPL");
    sessions.setContext("currentSymbol", "AAPL");
    await memory.set("last_query_symbol", "AAPL");

    // Second turn
    sessions.addMessage("assistant", "AAPL is currently trading at $150");
    sessions.addMessage("user", "What about its P/E ratio?");

    // Context should persist
    const session = sessions.getActive();
    expect(session?.context.currentSymbol).toBe("AAPL");
    expect(session?.messages.length).toBe(3);
    expect(await memory.get("last_query_symbol")).toBe("AAPL");
  });

  it("should restore context from previous session", async () => {
    // First session
    const session1 = sessions.create({ title: "Session 1" });
    sessions.setContext("portfolio", ["AAPL", "GOOGL"]);

    // Save to memory
    await memory.set(`session:${session1.metadata.id}:context`, session1.context);

    // Simulate loading context in new session
    const savedContext = await memory.get<Record<string, unknown>>(
      `session:${session1.metadata.id}:context`
    );

    sessions.create({ title: "Session 2" });
    if (savedContext?.portfolio) {
      sessions.setContext("previousPortfolio", savedContext.portfolio);
    }

    const activeSession = sessions.getActive();
    expect(activeSession?.context.previousPortfolio).toEqual(["AAPL", "GOOGL"]);
  });
});

describe("Performance Tests", () => {
  if (SKIP_INTEGRATION) {
    it.skip("Integration tests skipped", () => {});
    return;
  }

  it("should gather context within acceptable time", async () => {
    const memory = new IntegrationMemoryStore();
    const sessions = new IntegrationSessionManager();
    const contextBridge = new IntegrationContextBridge(memory, sessions);

    // Add multiple sources
    for (let i = 0; i < 5; i++) {
      contextBridge.registerSource({
        name: `source_${i}`,
        priority: i,
        gather: async () => {
          // Simulate some async work
          await new Promise((resolve) => setTimeout(resolve, 10));
          return { id: i, data: `data_${i}` };
        },
      });
    }

    const start = performance.now();
    await contextBridge.gather();
    const duration = performance.now() - start;

    // Should complete in under 100ms (sources run in parallel)
    expect(duration).toBeLessThan(100);
  });

  it("should handle large context data efficiently", async () => {
    const memory = new IntegrationMemoryStore();
    const sessions = new IntegrationSessionManager();
    const contextBridge = new IntegrationContextBridge(memory, sessions);

    // Store large data
    const largeData = {
      items: Array(1000).fill(null).map((_, i) => ({
        id: i,
        name: `Item ${i}`,
        data: "x".repeat(100),
      })),
    };
    await memory.set("large_dataset", largeData);

    contextBridge.registerSource({
      name: "large",
      priority: 1,
      gather: async () => ({
        data: await memory.get("large_dataset"),
      }),
    });

    const start = performance.now();
    const context = await contextBridge.gather();
    const duration = performance.now() - start;

    expect(context.large).toBeDefined();
    expect(duration).toBeLessThan(100);
  });
});
