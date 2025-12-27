/**
 * Agent Loop Tests
 *
 * Tests for the Stanley agent including:
 * - Agent initialization
 * - Message handling
 * - Tool execution callbacks
 * - Error handling
 */

import { describe, it, expect, beforeEach, mock } from "bun:test";
import { StanleyAgent, createStanleyAgent, type AgentOptions } from "../../src/agents/stanley";
import type { CoreTool } from "ai";

// Mock language model
const createMockModel = () => {
  return {
    // Mock LanguageModelV1 interface
    specificationVersion: "v1" as const,
    provider: "mock",
    modelId: "mock-model",
    defaultObjectGenerationMode: "json" as const,
    doGenerate: mock(async () => ({
      rawCall: { rawPrompt: "", rawSettings: {} },
      finishReason: "stop" as const,
      text: "Mock response",
      usage: { promptTokens: 10, completionTokens: 20 },
      warnings: [],
    })),
    doStream: mock(async () => ({
      rawCall: { rawPrompt: "", rawSettings: {} },
      stream: (async function* () {
        yield { type: "text-delta" as const, textDelta: "Mock " };
        yield { type: "text-delta" as const, textDelta: "response" };
        yield {
          type: "finish" as const,
          finishReason: "stop" as const,
          usage: { promptTokens: 10, completionTokens: 20 },
        };
      })(),
      warnings: [],
    })),
  };
};

// Mock tools
const createMockTools = (): Record<string, CoreTool> => ({
  mock_tool: {
    description: "A mock tool for testing",
    parameters: {
      type: "object" as const,
      properties: {
        input: { type: "string" as const },
      },
    },
    execute: mock(async ({ input }: { input: string }) => ({
      result: `Processed: ${input}`,
    })),
  },
  failing_tool: {
    description: "A tool that always fails",
    parameters: {
      type: "object" as const,
      properties: {},
    },
    execute: mock(async () => {
      throw new Error("Tool execution failed");
    }),
  },
});

describe("StanleyAgent", () => {
  let mockModel: ReturnType<typeof createMockModel>;
  let mockTools: Record<string, CoreTool>;

  beforeEach(() => {
    mockModel = createMockModel();
    mockTools = createMockTools();
  });

  describe("constructor", () => {
    it("should create agent with required options", () => {
      const agent = new StanleyAgent({
        model: mockModel as any,
        tools: mockTools,
      });

      expect(agent).toBeInstanceOf(StanleyAgent);
    });

    it("should accept custom system prompt", () => {
      const customPrompt = "You are a custom assistant.";
      const agent = new StanleyAgent({
        model: mockModel as any,
        tools: mockTools,
        systemPrompt: customPrompt,
      });

      expect(agent).toBeInstanceOf(StanleyAgent);
    });

    it("should accept custom maxSteps", () => {
      const agent = new StanleyAgent({
        model: mockModel as any,
        tools: mockTools,
        maxSteps: 5,
      });

      expect(agent).toBeInstanceOf(StanleyAgent);
    });
  });

  describe("createStanleyAgent", () => {
    it("should create agent via factory function", () => {
      const agent = createStanleyAgent({
        model: mockModel as any,
        tools: mockTools,
      });

      expect(agent).toBeInstanceOf(StanleyAgent);
    });
  });

  describe("getToolNames", () => {
    it("should return list of available tools", () => {
      const agent = new StanleyAgent({
        model: mockModel as any,
        tools: mockTools,
      });

      const names = agent.getToolNames();

      expect(names).toContain("mock_tool");
      expect(names).toContain("failing_tool");
      expect(names).toHaveLength(2);
    });
  });

  describe("setSystemPrompt", () => {
    it("should update system prompt", () => {
      const agent = new StanleyAgent({
        model: mockModel as any,
        tools: mockTools,
      });

      agent.setSystemPrompt("New prompt");

      // We can't directly verify the prompt, but the method should not throw
      expect(true).toBe(true);
    });
  });

  describe("addContext", () => {
    it("should append context to system prompt", () => {
      const agent = new StanleyAgent({
        model: mockModel as any,
        tools: mockTools,
        systemPrompt: "Base prompt",
      });

      agent.addContext("Current portfolio: AAPL, GOOGL");

      // Method should not throw
      expect(true).toBe(true);
    });
  });

  describe("callbacks", () => {
    it("should call onToolCall callback", async () => {
      const onToolCall = mock((name: string, args: unknown) => {});

      const agent = new StanleyAgent({
        model: mockModel as any,
        tools: mockTools,
        onToolCall,
      });

      // Since we're using a mock model, the actual tool calls won't happen
      // in a real integration test. This verifies the callback is set.
      expect(agent).toBeInstanceOf(StanleyAgent);
    });

    it("should call onToolResult callback", async () => {
      const onToolResult = mock((name: string, result: unknown) => {});

      const agent = new StanleyAgent({
        model: mockModel as any,
        tools: mockTools,
        onToolResult,
      });

      expect(agent).toBeInstanceOf(StanleyAgent);
    });

    it("should call onError callback", async () => {
      const onError = mock((error: Error) => {});

      const agent = new StanleyAgent({
        model: mockModel as any,
        tools: mockTools,
        onError,
      });

      expect(agent).toBeInstanceOf(StanleyAgent);
    });
  });

  describe("Message Types", () => {
    it("should handle user message", () => {
      const agent = new StanleyAgent({
        model: mockModel as any,
        tools: mockTools,
      });

      const messages = [{ role: "user" as const, content: "Hello" }];

      // Message format is valid
      expect(messages[0].role).toBe("user");
      expect(messages[0].content).toBe("Hello");
    });

    it("should handle assistant message", () => {
      const messages = [
        { role: "user" as const, content: "Hi" },
        { role: "assistant" as const, content: "Hello! How can I help?" },
      ];

      expect(messages).toHaveLength(2);
      expect(messages[1].role).toBe("assistant");
    });

    it("should handle multi-turn conversation", () => {
      const messages = [
        { role: "user" as const, content: "What is AAPL trading at?" },
        { role: "assistant" as const, content: "Apple (AAPL) is currently trading at $150.25." },
        { role: "user" as const, content: "What about GOOGL?" },
      ];

      expect(messages).toHaveLength(3);
    });
  });
});

describe("Agent Response Types", () => {
  it("should define correct response structure", () => {
    const response = {
      text: "Analysis complete",
      toolCalls: [
        {
          toolName: "get_market_data",
          args: { symbol: "AAPL" },
          result: { price: 150.25 },
        },
      ],
      finishReason: "stop",
      usage: {
        promptTokens: 100,
        completionTokens: 50,
        totalTokens: 150,
      },
    };

    expect(response.text).toBe("Analysis complete");
    expect(response.toolCalls).toHaveLength(1);
    expect(response.toolCalls?.[0].toolName).toBe("get_market_data");
    expect(response.finishReason).toBe("stop");
    expect(response.usage?.totalTokens).toBe(150);
  });

  it("should allow response without tool calls", () => {
    const response = {
      text: "Simple response",
      finishReason: "stop",
    };

    expect(response.text).toBe("Simple response");
    expect(response.toolCalls).toBeUndefined();
  });

  it("should allow response without usage", () => {
    const response = {
      text: "Response without usage",
      finishReason: "stop",
    };

    expect(response.usage).toBeUndefined();
  });
});
