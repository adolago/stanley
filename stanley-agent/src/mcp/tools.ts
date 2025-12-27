/**
 * Stanley MCP Tools
 *
 * MCP (Model Context Protocol) tools for interacting with the Stanley API.
 * These tools are exposed to the AI model for investment research tasks.
 */

import { z } from "zod";
import { tool, zodSchema } from "ai";

/**
 * Tool result wrapper
 */
interface ToolResult<T = unknown> {
  success: boolean;
  data?: T;
  error?: string;
}

/**
 * Stanley API client configuration
 */
interface ApiClientConfig {
  baseUrl: string;
  timeout?: number;
}

/**
 * Create HTTP client for Stanley API
 */
function createApiClient(config: ApiClientConfig) {
  const { baseUrl, timeout = 30000 } = config;

  return async function fetchApi<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ToolResult<T>> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
      const response = await fetch(`${baseUrl}${endpoint}`, {
        ...options,
        signal: controller.signal,
        headers: {
          "Content-Type": "application/json",
          ...options.headers,
        },
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        return {
          success: false,
          error: `API error ${response.status}: ${errorText}`,
        };
      }

      const data = (await response.json()) as { data?: T } | T;
      return { success: true, data: (data && typeof data === "object" && "data" in data ? data.data : data) as T };
    } catch (error) {
      clearTimeout(timeoutId);

      if (error instanceof Error) {
        if (error.name === "AbortError") {
          return { success: false, error: "Request timeout" };
        }
        return { success: false, error: error.message };
      }
      return { success: false, error: "Unknown error" };
    }
  };
}

// Define Zod schemas for tool inputs
const symbolSchema = z.object({
  symbol: z.string().describe("Stock ticker symbol (e.g., AAPL, MSFT)"),
});

const institutionalSchema = z.object({
  symbol: z.string().describe("Stock ticker symbol"),
  limit: z.number().optional().default(10).describe("Maximum number of institutions to return"),
});

const moneyFlowSchema = z.object({
  sectors: z.array(z.string()).describe("List of sector ETF symbols (e.g., XLK, XLF, XLE)"),
  lookback_days: z.number().optional().default(63).describe("Analysis period in days"),
});

const portfolioSchema = z.object({
  holdings: z.array(z.object({
    symbol: z.string(),
    shares: z.number(),
    average_cost: z.number().optional(),
  })).describe("List of portfolio holdings"),
});

const noteSchema = z.object({
  name: z.string().describe("Note name/identifier"),
});

const saveNoteSchema = z.object({
  name: z.string().describe("Note name/identifier"),
  content: z.string().describe("Note content in markdown format"),
});

const searchSchema = z.object({
  query: z.string().describe("Search query"),
});

const emptySchema = z.object({});

/**
 * Create Stanley MCP tools
 */
export function createStanleyTools(config: ApiClientConfig) {
  const api = createApiClient(config);

  return {
    get_market_data: tool({
      description: "Get current market data including price, volume, and change for a stock symbol",
      inputSchema: zodSchema(symbolSchema),
      execute: async ({ symbol }: z.infer<typeof symbolSchema>) => {
        const result = await api(`/api/market/${symbol.toUpperCase()}`);
        if (!result.success) return { error: result.error };
        return result.data;
      },
    }),

    get_institutional_holdings: tool({
      description: "Get institutional investor holdings from 13F filings for a stock",
      inputSchema: zodSchema(institutionalSchema),
      execute: async ({ symbol, limit }: z.infer<typeof institutionalSchema>) => {
        const result = await api(`/api/institutional/${symbol.toUpperCase()}?limit=${limit}`);
        if (!result.success) return { error: result.error };
        return result.data;
      },
    }),

    analyze_money_flow: tool({
      description: "Analyze money flow patterns across market sectors",
      inputSchema: zodSchema(moneyFlowSchema),
      execute: async ({ sectors, lookback_days }: z.infer<typeof moneyFlowSchema>) => {
        const result = await api("/api/money-flow", {
          method: "POST",
          body: JSON.stringify({ sectors, lookback_days }),
        });
        if (!result.success) return { error: result.error };
        return result.data;
      },
    }),

    get_portfolio_analytics: tool({
      description: "Calculate portfolio analytics including VaR, beta, and sector exposure",
      inputSchema: zodSchema(portfolioSchema),
      execute: async ({ holdings }: z.infer<typeof portfolioSchema>) => {
        const result = await api("/api/portfolio-analytics", {
          method: "POST",
          body: JSON.stringify({ holdings }),
        });
        if (!result.success) return { error: result.error };
        return result.data;
      },
    }),

    get_dark_pool: tool({
      description: "Get dark pool trading activity for a stock",
      inputSchema: zodSchema(symbolSchema),
      execute: async ({ symbol }: z.infer<typeof symbolSchema>) => {
        const result = await api(`/api/dark-pool/${symbol.toUpperCase()}`);
        if (!result.success) return { error: result.error };
        return result.data;
      },
    }),

    get_equity_flow: tool({
      description: "Get equity money flow data for a stock",
      inputSchema: zodSchema(symbolSchema),
      execute: async ({ symbol }: z.infer<typeof symbolSchema>) => {
        const result = await api(`/api/equity-flow/${symbol.toUpperCase()}`);
        if (!result.success) return { error: result.error };
        return result.data;
      },
    }),

    get_research: tool({
      description: "Get comprehensive research report for a stock including valuation, earnings, and peer comparison",
      inputSchema: zodSchema(symbolSchema),
      execute: async ({ symbol }: z.infer<typeof symbolSchema>) => {
        const result = await api(`/api/research/${symbol.toUpperCase()}`);
        if (!result.success) return { error: result.error };
        return result.data;
      },
    }),

    get_valuation: tool({
      description: "Get detailed valuation analysis with DCF model",
      inputSchema: zodSchema(symbolSchema),
      execute: async ({ symbol }: z.infer<typeof symbolSchema>) => {
        const result = await api(`/api/valuation/${symbol.toUpperCase()}`);
        if (!result.success) return { error: result.error };
        return result.data;
      },
    }),

    get_earnings: tool({
      description: "Get earnings analysis and surprise history",
      inputSchema: zodSchema(symbolSchema),
      execute: async ({ symbol }: z.infer<typeof symbolSchema>) => {
        const result = await api(`/api/earnings/${symbol.toUpperCase()}`);
        if (!result.success) return { error: result.error };
        return result.data;
      },
    }),

    get_peers: tool({
      description: "Get peer comparison analysis",
      inputSchema: zodSchema(symbolSchema),
      execute: async ({ symbol }: z.infer<typeof symbolSchema>) => {
        const result = await api(`/api/peers/${symbol.toUpperCase()}`);
        if (!result.success) return { error: result.error };
        return result.data;
      },
    }),

    get_commodities: tool({
      description: "Get commodity market overview with prices and trends",
      inputSchema: zodSchema(emptySchema),
      execute: async () => {
        const result = await api("/api/commodities");
        if (!result.success) return { error: result.error };
        return result.data;
      },
    }),

    get_commodity: tool({
      description: "Get detailed data for a specific commodity",
      inputSchema: zodSchema(z.object({
        symbol: z.string().describe("Commodity symbol (e.g., GC for gold, CL for oil)"),
      })),
      execute: async ({ symbol }: { symbol: string }) => {
        const result = await api(`/api/commodities/${symbol.toUpperCase()}`);
        if (!result.success) return { error: result.error };
        return result.data;
      },
    }),

    get_options_flow: tool({
      description: "Get options trading flow and unusual activity",
      inputSchema: zodSchema(symbolSchema),
      execute: async ({ symbol }: z.infer<typeof symbolSchema>) => {
        const result = await api(`/api/options/${symbol.toUpperCase()}/flow`);
        if (!result.success) return { error: result.error };
        return result.data;
      },
    }),

    get_gamma_exposure: tool({
      description: "Get options gamma exposure analysis",
      inputSchema: zodSchema(symbolSchema),
      execute: async ({ symbol }: z.infer<typeof symbolSchema>) => {
        const result = await api(`/api/options/${symbol.toUpperCase()}/gamma`);
        if (!result.success) return { error: result.error };
        return result.data;
      },
    }),

    search_notes: tool({
      description: "Search research notes and memos",
      inputSchema: zodSchema(searchSchema),
      execute: async ({ query }: z.infer<typeof searchSchema>) => {
        const result = await api(`/api/notes/search?q=${encodeURIComponent(query)}`);
        if (!result.success) return { error: result.error };
        return result.data;
      },
    }),

    get_notes: tool({
      description: "Get all research notes",
      inputSchema: zodSchema(emptySchema),
      execute: async () => {
        const result = await api("/api/notes");
        if (!result.success) return { error: result.error };
        return result.data;
      },
    }),

    get_note: tool({
      description: "Get a specific research note by name",
      inputSchema: zodSchema(noteSchema),
      execute: async ({ name }: z.infer<typeof noteSchema>) => {
        const result = await api(`/api/notes/${encodeURIComponent(name)}`);
        if (!result.success) return { error: result.error };
        return result.data;
      },
    }),

    save_note: tool({
      description: "Create or update a research note",
      inputSchema: zodSchema(saveNoteSchema),
      execute: async ({ name, content }: z.infer<typeof saveNoteSchema>) => {
        const result = await api(`/api/notes/${encodeURIComponent(name)}`, {
          method: "PUT",
          body: JSON.stringify({ content }),
        });
        if (!result.success) return { error: result.error };
        return result.data;
      },
    }),

    get_theses: tool({
      description: "Get all investment theses",
      inputSchema: zodSchema(emptySchema),
      execute: async () => {
        const result = await api("/api/theses");
        if (!result.success) return { error: result.error };
        return result.data;
      },
    }),

    get_trades: tool({
      description: "Get trade log and history",
      inputSchema: zodSchema(emptySchema),
      execute: async () => {
        const result = await api("/api/trades");
        if (!result.success) return { error: result.error };
        return result.data;
      },
    }),

    get_trade_stats: tool({
      description: "Get trading performance statistics",
      inputSchema: zodSchema(emptySchema),
      execute: async () => {
        const result = await api("/api/trades/stats");
        if (!result.success) return { error: result.error };
        return result.data;
      },
    }),

    health_check: tool({
      description: "Check if the Stanley API is healthy and responsive",
      inputSchema: zodSchema(emptySchema),
      execute: async () => {
        const result = await api("/api/health");
        if (!result.success) return { error: result.error };
        return result.data;
      },
    }),
  };
}

/**
 * Get tool names for logging
 */
export function getToolNames(tools: Record<string, unknown>): string[] {
  return Object.keys(tools);
}
