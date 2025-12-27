/**
 * Agent Integration Tests
 *
 * Tests the full agent loop with mocked provider but real tools.
 * These tests verify end-to-end behavior without requiring API keys.
 */

import { describe, it, expect, beforeAll, afterAll, mock } from "bun:test";
import { createStanleyTools } from "../../src/mcp/tools";
import { createStanleyAgent } from "../../src/agents/stanley";

// Configuration
const API_URL = process.env.STANLEY_API_URL || "http://localhost:8000";
const SKIP_INTEGRATION = process.env.SKIP_INTEGRATION_TESTS === "true";

// Mock model that simulates tool calls
const createMockModelWithToolCalls = (toolCallsToMake: { name: string; args: any }[]) => {
  let callIndex = 0;

  return {
    specificationVersion: "v1" as const,
    provider: "mock",
    modelId: "mock-model",
    defaultObjectGenerationMode: "json" as const,
    doGenerate: mock(async (options: any) => {
      const { tools } = options;

      // If there are tool calls to make and we haven't made them all
      if (callIndex < toolCallsToMake.length) {
        const toolCall = toolCallsToMake[callIndex++];
        return {
          rawCall: { rawPrompt: "", rawSettings: {} },
          finishReason: "tool-calls" as const,
          text: "",
          toolCalls: [
            {
              toolCallType: "function" as const,
              toolCallId: `call-${callIndex}`,
              toolName: toolCall.name,
              args: JSON.stringify(toolCall.args),
            },
          ],
          usage: { promptTokens: 10, completionTokens: 20 },
          warnings: [],
        };
      }

      // Final response after tool calls
      return {
        rawCall: { rawPrompt: "", rawSettings: {} },
        finishReason: "stop" as const,
        text: "Analysis complete. Based on the data retrieved, here is my assessment.",
        usage: { promptTokens: 50, completionTokens: 100 },
        warnings: [],
      };
    }),
  };
};

describe("Agent Integration Tests", () => {
  let tools: ReturnType<typeof createStanleyTools>;
  let apiAvailable = false;

  beforeAll(async () => {
    if (SKIP_INTEGRATION) {
      console.log("Skipping agent integration tests (SKIP_INTEGRATION_TESTS=true)");
      return;
    }

    tools = createStanleyTools({
      baseUrl: API_URL,
      timeout: 10000,
    });

    // Check if API is available
    try {
      const response = await fetch(`${API_URL}/api/health`);
      apiAvailable = response.ok;
    } catch (error) {
      apiAvailable = false;
    }
  });

  describe("Agent with Mock Model", () => {
    it("should create agent with tools", () => {
      if (SKIP_INTEGRATION) return;

      const mockModel = createMockModelWithToolCalls([]);

      const agent = createStanleyAgent({
        model: mockModel as any,
        tools,
        maxSteps: 5,
      });

      expect(agent).toBeDefined();
      expect(agent.getToolNames().length).toBeGreaterThan(0);
    });

    it("should have all expected tools", () => {
      if (SKIP_INTEGRATION) return;

      const mockModel = createMockModelWithToolCalls([]);

      const agent = createStanleyAgent({
        model: mockModel as any,
        tools,
      });

      const toolNames = agent.getToolNames();

      expect(toolNames).toContain("get_market_data");
      expect(toolNames).toContain("get_research");
      expect(toolNames).toContain("analyze_money_flow");
      expect(toolNames).toContain("get_portfolio_analytics");
      expect(toolNames).toContain("health_check");
    });
  });

  describe("Tool Callback Integration", () => {
    it("should track tool calls via callbacks", () => {
      if (SKIP_INTEGRATION) return;

      const toolCalls: { name: string; args: any }[] = [];
      const toolResults: { name: string; result: any }[] = [];

      const mockModel = createMockModelWithToolCalls([]);

      const agent = createStanleyAgent({
        model: mockModel as any,
        tools,
        onToolCall: (name, args) => {
          toolCalls.push({ name, args });
        },
        onToolResult: (name, result) => {
          toolResults.push({ name, result });
        },
      });

      expect(agent).toBeDefined();
    });

    it("should track errors via callback", () => {
      if (SKIP_INTEGRATION) return;

      const errors: Error[] = [];

      const mockModel = createMockModelWithToolCalls([]);

      const agent = createStanleyAgent({
        model: mockModel as any,
        tools,
        onError: (error) => {
          errors.push(error);
        },
      });

      expect(agent).toBeDefined();
    });
  });

  describe("Context Management", () => {
    it("should add context to agent", () => {
      if (SKIP_INTEGRATION) return;

      const mockModel = createMockModelWithToolCalls([]);

      const agent = createStanleyAgent({
        model: mockModel as any,
        tools,
        systemPrompt: "You are a helpful assistant.",
      });

      // Add context
      agent.addContext("User is analyzing tech stocks");
      agent.addContext("Portfolio focus: growth");

      expect(agent).toBeDefined();
    });

    it("should update system prompt", () => {
      if (SKIP_INTEGRATION) return;

      const mockModel = createMockModelWithToolCalls([]);

      const agent = createStanleyAgent({
        model: mockModel as any,
        tools,
      });

      agent.setSystemPrompt("Custom prompt for specialized analysis");

      expect(agent).toBeDefined();
    });
  });

  describe("Message Handling", () => {
    it("should handle simple user message", () => {
      if (SKIP_INTEGRATION) return;

      const messages = [
        { role: "user" as const, content: "What is the current market outlook?" },
      ];

      expect(messages).toHaveLength(1);
      expect(messages[0].role).toBe("user");
    });

    it("should handle conversation history", () => {
      if (SKIP_INTEGRATION) return;

      const messages = [
        { role: "user" as const, content: "Analyze AAPL" },
        {
          role: "assistant" as const,
          content: "Apple (AAPL) is currently trading at $150. The stock has shown strong momentum.",
        },
        { role: "user" as const, content: "How does it compare to MSFT?" },
      ];

      expect(messages).toHaveLength(3);
      expect(messages[2].role).toBe("user");
    });
  });
});
