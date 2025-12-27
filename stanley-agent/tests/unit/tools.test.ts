/**
 * MCP Tools Tests
 *
 * Tests for Stanley MCP tools with mocked API responses.
 * These tests verify tool behavior without requiring a running API server.
 */

import { describe, it, expect, beforeEach, afterEach, mock, spyOn } from "bun:test";
import { createStanleyTools, getToolNames } from "../../src/mcp/tools";

// Mock fetch globally
const originalFetch = globalThis.fetch;

describe("Stanley MCP Tools", () => {
  let mockFetch: ReturnType<typeof mock>;
  let tools: ReturnType<typeof createStanleyTools>;

  beforeEach(() => {
    mockFetch = mock(() => Promise.resolve(new Response()));
    globalThis.fetch = mockFetch;
    tools = createStanleyTools({
      baseUrl: "http://localhost:8000",
      timeout: 5000,
    });
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  describe("getToolNames", () => {
    it("should return all available tool names", () => {
      const names = getToolNames(tools);
      expect(names).toContain("get_market_data");
      expect(names).toContain("get_institutional_holdings");
      expect(names).toContain("analyze_money_flow");
      expect(names).toContain("get_portfolio_analytics");
      expect(names).toContain("get_research");
      expect(names).toContain("health_check");
    });

    it("should have at least 15 tools", () => {
      const names = getToolNames(tools);
      expect(names.length).toBeGreaterThanOrEqual(15);
    });
  });

  describe("Tool Definitions", () => {
    it("should have description for each tool", () => {
      for (const [name, tool] of Object.entries(tools)) {
        expect(tool.description).toBeDefined();
        expect(typeof tool.description).toBe("string");
        expect(tool.description.length).toBeGreaterThan(10);
      }
    });

    it("should have parameters for each tool", () => {
      for (const [name, tool] of Object.entries(tools)) {
        expect(tool.parameters).toBeDefined();
      }
    });

    it("should have execute function for each tool", () => {
      for (const [name, tool] of Object.entries(tools)) {
        expect(tool.execute).toBeDefined();
        expect(typeof tool.execute).toBe("function");
      }
    });
  });

  describe("get_market_data", () => {
    it("should call API with correct endpoint", async () => {
      mockFetch.mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            success: true,
            data: {
              symbol: "AAPL",
              price: 150.25,
              change: 2.50,
              changePercent: 1.69,
              volume: 50000000,
            },
          }),
          { status: 200 }
        )
      );

      const result = await tools.get_market_data.execute({ symbol: "AAPL" });

      expect(mockFetch).toHaveBeenCalledTimes(1);
      const [url, options] = mockFetch.mock.calls[0];
      expect(url).toBe("http://localhost:8000/api/market/AAPL");
      expect(result).toHaveProperty("symbol", "AAPL");
      expect(result).toHaveProperty("price", 150.25);
    });

    it("should uppercase symbol", async () => {
      mockFetch.mockResolvedValueOnce(
        new Response(JSON.stringify({ success: true, data: {} }), { status: 200 })
      );

      await tools.get_market_data.execute({ symbol: "aapl" });

      const [url] = mockFetch.mock.calls[0];
      expect(url).toBe("http://localhost:8000/api/market/AAPL");
    });

    it("should handle API error", async () => {
      mockFetch.mockResolvedValueOnce(
        new Response("Not Found", { status: 404 })
      );

      const result = await tools.get_market_data.execute({ symbol: "INVALID" });

      expect(result).toHaveProperty("error");
      expect(result.error).toContain("404");
    });
  });

  describe("get_institutional_holdings", () => {
    it("should call API with correct endpoint and limit", async () => {
      mockFetch.mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            success: true,
            data: [
              { managerName: "BlackRock", sharesHeld: 1000000 },
              { managerName: "Vanguard", sharesHeld: 900000 },
            ],
          }),
          { status: 200 }
        )
      );

      const result = await tools.get_institutional_holdings.execute({
        symbol: "MSFT",
        limit: 5,
      });

      const [url] = mockFetch.mock.calls[0];
      expect(url).toBe("http://localhost:8000/api/institutional/MSFT?limit=5");
      expect(Array.isArray(result)).toBe(true);
    });

    it("should use default limit", async () => {
      mockFetch.mockResolvedValueOnce(
        new Response(JSON.stringify({ success: true, data: [] }), { status: 200 })
      );

      await tools.get_institutional_holdings.execute({ symbol: "GOOGL" });

      const [url] = mockFetch.mock.calls[0];
      expect(url).toContain("limit=10");
    });
  });

  describe("analyze_money_flow", () => {
    it("should POST with correct body", async () => {
      mockFetch.mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            success: true,
            data: {
              sectors: ["XLK", "XLF"],
              analysis: {},
            },
          }),
          { status: 200 }
        )
      );

      await tools.analyze_money_flow.execute({
        sectors: ["XLK", "XLF", "XLE"],
        lookback_days: 30,
      });

      const [url, options] = mockFetch.mock.calls[0];
      expect(url).toBe("http://localhost:8000/api/money-flow");
      expect(options.method).toBe("POST");
      const body = JSON.parse(options.body);
      expect(body.sectors).toEqual(["XLK", "XLF", "XLE"]);
      expect(body.lookback_days).toBe(30);
    });

    it("should use default lookback days", async () => {
      mockFetch.mockResolvedValueOnce(
        new Response(JSON.stringify({ success: true, data: {} }), { status: 200 })
      );

      await tools.analyze_money_flow.execute({
        sectors: ["XLK"],
      });

      const body = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(body.lookback_days).toBe(63);
    });
  });

  describe("get_portfolio_analytics", () => {
    it("should POST portfolio holdings correctly", async () => {
      mockFetch.mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            success: true,
            data: {
              totalValue: 100000,
              beta: 1.1,
              var95: 5000,
            },
          }),
          { status: 200 }
        )
      );

      const holdings = [
        { symbol: "AAPL", shares: 100, average_cost: 150 },
        { symbol: "GOOGL", shares: 50 },
      ];

      const result = await tools.get_portfolio_analytics.execute({ holdings });

      const [url, options] = mockFetch.mock.calls[0];
      expect(url).toBe("http://localhost:8000/api/portfolio-analytics");
      expect(options.method).toBe("POST");
      expect(result).toHaveProperty("totalValue");
      expect(result).toHaveProperty("beta");
    });
  });

  describe("get_dark_pool", () => {
    it("should call correct endpoint", async () => {
      mockFetch.mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            success: true,
            data: { symbol: "NVDA", darkPoolVolume: 5000000 },
          }),
          { status: 200 }
        )
      );

      await tools.get_dark_pool.execute({ symbol: "nvda" });

      const [url] = mockFetch.mock.calls[0];
      expect(url).toBe("http://localhost:8000/api/dark-pool/NVDA");
    });
  });

  describe("get_research", () => {
    it("should return comprehensive research data", async () => {
      mockFetch.mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            success: true,
            data: {
              symbol: "AMZN",
              valuation: {},
              earnings: {},
              peers: [],
            },
          }),
          { status: 200 }
        )
      );

      const result = await tools.get_research.execute({ symbol: "AMZN" });

      expect(result).toHaveProperty("symbol", "AMZN");
      expect(result).toHaveProperty("valuation");
      expect(result).toHaveProperty("earnings");
    });
  });

  describe("get_commodities", () => {
    it("should call commodities endpoint without parameters", async () => {
      mockFetch.mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            success: true,
            data: {
              gold: { price: 2000 },
              oil: { price: 80 },
            },
          }),
          { status: 200 }
        )
      );

      await tools.get_commodities.execute({});

      const [url] = mockFetch.mock.calls[0];
      expect(url).toBe("http://localhost:8000/api/commodities");
    });
  });

  describe("get_commodity", () => {
    it("should call specific commodity endpoint", async () => {
      mockFetch.mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            success: true,
            data: { symbol: "GC", price: 2000, name: "Gold" },
          }),
          { status: 200 }
        )
      );

      const result = await tools.get_commodity.execute({ symbol: "gc" });

      const [url] = mockFetch.mock.calls[0];
      expect(url).toBe("http://localhost:8000/api/commodities/GC");
      expect(result).toHaveProperty("symbol", "GC");
    });
  });

  describe("get_options_flow", () => {
    it("should call options flow endpoint", async () => {
      mockFetch.mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            success: true,
            data: { symbol: "TSLA", unusualActivity: [] },
          }),
          { status: 200 }
        )
      );

      await tools.get_options_flow.execute({ symbol: "TSLA" });

      const [url] = mockFetch.mock.calls[0];
      expect(url).toBe("http://localhost:8000/api/options/TSLA/flow");
    });
  });

  describe("search_notes", () => {
    it("should encode search query in URL", async () => {
      mockFetch.mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            success: true,
            data: [{ name: "thesis-aapl", content: "Apple thesis" }],
          }),
          { status: 200 }
        )
      );

      await tools.search_notes.execute({ query: "apple stock thesis" });

      const [url] = mockFetch.mock.calls[0];
      expect(url).toBe(
        "http://localhost:8000/api/notes/search?q=apple%20stock%20thesis"
      );
    });
  });

  describe("save_note", () => {
    it("should PUT note with content", async () => {
      mockFetch.mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            success: true,
            data: { name: "my-note", saved: true },
          }),
          { status: 200 }
        )
      );

      await tools.save_note.execute({
        name: "my-note",
        content: "# My Research\n\nThis is my analysis.",
      });

      const [url, options] = mockFetch.mock.calls[0];
      expect(url).toBe("http://localhost:8000/api/notes/my-note");
      expect(options.method).toBe("PUT");
      const body = JSON.parse(options.body);
      expect(body.content).toContain("My Research");
    });

    it("should URL-encode note names", async () => {
      mockFetch.mockResolvedValueOnce(
        new Response(JSON.stringify({ success: true, data: {} }), { status: 200 })
      );

      await tools.save_note.execute({
        name: "thesis/apple/2024",
        content: "Content",
      });

      const [url] = mockFetch.mock.calls[0];
      expect(url).toBe("http://localhost:8000/api/notes/thesis%2Fapple%2F2024");
    });
  });

  describe("health_check", () => {
    it("should call health endpoint", async () => {
      mockFetch.mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            success: true,
            data: { status: "healthy", version: "1.0.0" },
          }),
          { status: 200 }
        )
      );

      const result = await tools.health_check.execute({});

      const [url] = mockFetch.mock.calls[0];
      expect(url).toBe("http://localhost:8000/api/health");
      expect(result).toHaveProperty("status", "healthy");
    });
  });

  describe("Error Handling", () => {
    it("should handle network errors", async () => {
      mockFetch.mockRejectedValueOnce(new Error("Network error"));

      const result = await tools.get_market_data.execute({ symbol: "AAPL" });

      expect(result).toHaveProperty("error");
      expect(result.error).toContain("Network error");
    });

    it("should handle timeout", async () => {
      // Create tools with very short timeout
      const fastTools = createStanleyTools({
        baseUrl: "http://localhost:8000",
        timeout: 1, // 1ms timeout
      });

      // Mock a slow response
      mockFetch.mockImplementationOnce(
        () =>
          new Promise((resolve) =>
            setTimeout(() => resolve(new Response(JSON.stringify({}))), 100)
          )
      );

      const result = await fastTools.get_market_data.execute({ symbol: "AAPL" });

      expect(result).toHaveProperty("error");
      // Note: AbortError might manifest differently in different runtimes
    });

    it("should handle malformed JSON response", async () => {
      mockFetch.mockResolvedValueOnce(
        new Response("not json", { status: 200 })
      );

      const result = await tools.get_market_data.execute({ symbol: "AAPL" });

      // Should handle the JSON parse error gracefully
      expect(result).toHaveProperty("error");
    });

    it("should handle 500 server error", async () => {
      mockFetch.mockResolvedValueOnce(
        new Response("Internal Server Error", { status: 500 })
      );

      const result = await tools.get_research.execute({ symbol: "FAIL" });

      expect(result).toHaveProperty("error");
      expect(result.error).toContain("500");
    });
  });
});
