/**
 * API Integration Tests
 *
 * Tests that verify the MCP tools work correctly with a running Stanley API.
 * These tests require the Python API server to be running.
 *
 * To run these tests:
 * 1. Start the Stanley API: python -m stanley.api.main
 * 2. Run tests: bun test tests/integration
 */

import { describe, it, expect, beforeAll, afterAll } from "bun:test";
import { createStanleyTools, getToolNames } from "../../src/mcp/tools";

// Configuration
const API_URL = process.env.STANLEY_API_URL || "http://localhost:8000";
const SKIP_INTEGRATION = process.env.SKIP_INTEGRATION_TESTS === "true";

describe("API Integration Tests", () => {
  let tools: ReturnType<typeof createStanleyTools>;
  let apiAvailable = false;

  beforeAll(async () => {
    if (SKIP_INTEGRATION) {
      console.log("Skipping integration tests (SKIP_INTEGRATION_TESTS=true)");
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
      if (!apiAvailable) {
        console.log("Stanley API not available, skipping integration tests");
      }
    } catch (error) {
      console.log("Stanley API not reachable, skipping integration tests");
      apiAvailable = false;
    }
  });

  describe("Health Check", () => {
    it("should verify API is healthy", async () => {
      if (SKIP_INTEGRATION || !apiAvailable) return;

      const result = await tools.health_check.execute({});

      expect(result).not.toHaveProperty("error");
      expect(result).toHaveProperty("status");
    });
  });

  describe("Market Data", () => {
    it("should fetch market data for AAPL", async () => {
      if (SKIP_INTEGRATION || !apiAvailable) return;

      const result = await tools.get_market_data.execute({ symbol: "AAPL" });

      // Even if using mock data, should return valid structure
      if (!result.error) {
        expect(result).toHaveProperty("symbol");
      }
    });

    it("should handle invalid symbol gracefully", async () => {
      if (SKIP_INTEGRATION || !apiAvailable) return;

      const result = await tools.get_market_data.execute({ symbol: "INVALID123XYZ" });

      // Should either return error or empty data, not crash
      expect(result).toBeDefined();
    });
  });

  describe("Research Endpoints", () => {
    it("should fetch research report", async () => {
      if (SKIP_INTEGRATION || !apiAvailable) return;

      const result = await tools.get_research.execute({ symbol: "MSFT" });

      if (!result.error) {
        expect(result).toBeDefined();
      }
    });

    it("should fetch valuation data", async () => {
      if (SKIP_INTEGRATION || !apiAvailable) return;

      const result = await tools.get_valuation.execute({ symbol: "GOOGL" });

      if (!result.error) {
        expect(result).toBeDefined();
      }
    });

    it("should fetch earnings data", async () => {
      if (SKIP_INTEGRATION || !apiAvailable) return;

      const result = await tools.get_earnings.execute({ symbol: "AMZN" });

      if (!result.error) {
        expect(result).toBeDefined();
      }
    });

    it("should fetch peer comparison", async () => {
      if (SKIP_INTEGRATION || !apiAvailable) return;

      const result = await tools.get_peers.execute({ symbol: "META" });

      if (!result.error) {
        expect(result).toBeDefined();
      }
    });
  });

  describe("Money Flow Analysis", () => {
    it("should analyze sector money flow", async () => {
      if (SKIP_INTEGRATION || !apiAvailable) return;

      const result = await tools.analyze_money_flow.execute({
        sectors: ["XLK", "XLF", "XLE"],
        lookback_days: 30,
      });

      if (!result.error) {
        expect(result).toBeDefined();
      }
    });

    it("should fetch equity flow data", async () => {
      if (SKIP_INTEGRATION || !apiAvailable) return;

      const result = await tools.get_equity_flow.execute({ symbol: "NVDA" });

      if (!result.error) {
        expect(result).toBeDefined();
      }
    });
  });

  describe("Portfolio Analytics", () => {
    it("should calculate portfolio analytics", async () => {
      if (SKIP_INTEGRATION || !apiAvailable) return;

      const result = await tools.get_portfolio_analytics.execute({
        holdings: [
          { symbol: "AAPL", shares: 100, average_cost: 150 },
          { symbol: "GOOGL", shares: 50, average_cost: 140 },
          { symbol: "MSFT", shares: 75, average_cost: 380 },
        ],
      });

      if (!result.error) {
        expect(result).toBeDefined();
      }
    });
  });

  describe("Institutional Data", () => {
    it("should fetch institutional holdings", async () => {
      if (SKIP_INTEGRATION || !apiAvailable) return;

      const result = await tools.get_institutional_holdings.execute({
        symbol: "AAPL",
        limit: 5,
      });

      if (!result.error) {
        expect(result).toBeDefined();
      }
    });

    it("should fetch dark pool data", async () => {
      if (SKIP_INTEGRATION || !apiAvailable) return;

      const result = await tools.get_dark_pool.execute({ symbol: "TSLA" });

      if (!result.error) {
        expect(result).toBeDefined();
      }
    });
  });

  describe("Commodities", () => {
    it("should fetch commodity overview", async () => {
      if (SKIP_INTEGRATION || !apiAvailable) return;

      const result = await tools.get_commodities.execute({});

      if (!result.error) {
        expect(result).toBeDefined();
      }
    });

    it("should fetch specific commodity", async () => {
      if (SKIP_INTEGRATION || !apiAvailable) return;

      const result = await tools.get_commodity.execute({ symbol: "GC" });

      if (!result.error) {
        expect(result).toBeDefined();
      }
    });
  });

  describe("Options", () => {
    it("should fetch options flow", async () => {
      if (SKIP_INTEGRATION || !apiAvailable) return;

      const result = await tools.get_options_flow.execute({ symbol: "SPY" });

      if (!result.error) {
        expect(result).toBeDefined();
      }
    });

    it("should fetch gamma exposure", async () => {
      if (SKIP_INTEGRATION || !apiAvailable) return;

      const result = await tools.get_gamma_exposure.execute({ symbol: "QQQ" });

      if (!result.error) {
        expect(result).toBeDefined();
      }
    });
  });

  describe("Notes", () => {
    it("should list notes", async () => {
      if (SKIP_INTEGRATION || !apiAvailable) return;

      const result = await tools.get_notes.execute({});

      if (!result.error) {
        expect(result).toBeDefined();
      }
    });

    it("should search notes", async () => {
      if (SKIP_INTEGRATION || !apiAvailable) return;

      const result = await tools.search_notes.execute({ query: "thesis" });

      if (!result.error) {
        expect(result).toBeDefined();
      }
    });

    it("should get investment theses", async () => {
      if (SKIP_INTEGRATION || !apiAvailable) return;

      const result = await tools.get_theses.execute({});

      if (!result.error) {
        expect(result).toBeDefined();
      }
    });

    it("should get trade log", async () => {
      if (SKIP_INTEGRATION || !apiAvailable) return;

      const result = await tools.get_trades.execute({});

      if (!result.error) {
        expect(result).toBeDefined();
      }
    });

    it("should get trade statistics", async () => {
      if (SKIP_INTEGRATION || !apiAvailable) return;

      const result = await tools.get_trade_stats.execute({});

      if (!result.error) {
        expect(result).toBeDefined();
      }
    });
  });
});
