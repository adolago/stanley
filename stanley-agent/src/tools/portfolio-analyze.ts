/**
 * Portfolio Analyze Tool
 *
 * MCP tool for portfolio analytics including VaR, beta, sector exposure,
 * and comprehensive risk metrics.
 */

import { z } from "zod";
import { stanleyFetch, formatToolOutput, buildQueryString } from "./api-client";
import type {
  ToolContext,
  PortfolioAnalytics,
  RiskMetrics,
  StanleyApiResponse,
} from "./types";

const TOOL_DESCRIPTION = `Analyze investment portfolios for risk and performance metrics.

This tool provides comprehensive portfolio analytics including:
- Total value, returns, and performance attribution
- Risk metrics: VaR (Value at Risk) at 95% and 99% confidence levels
- Beta (market sensitivity) and alpha (excess returns)
- Sharpe and Sortino ratios for risk-adjusted returns
- Sector exposure breakdown
- Correlation analysis and diversification scoring

Use this tool when:
- A user wants to analyze their portfolio's risk profile
- You need to calculate VaR or other risk metrics
- Sector exposure or concentration analysis is needed
- Performance attribution or benchmark comparison is required

The tool connects to Stanley's portfolio analyzer backend which uses
historical market data for calculations.`;

// Input schema for portfolio analysis
const PortfolioAnalyzeArgs = z.object({
  action: z
    .enum(["analyze", "risk", "correlation", "attribution", "sector-exposure"])
    .describe(
      "Type of analysis: 'analyze' for full analytics, 'risk' for risk metrics, 'correlation' for correlation matrix, 'attribution' for performance attribution, 'sector-exposure' for sector breakdown"
    ),
  holdings: z
    .array(
      z.object({
        symbol: z.string().describe("Stock ticker symbol (e.g., AAPL)"),
        shares: z.number().positive().describe("Number of shares"),
        average_cost: z.number().optional().describe("Average cost per share"),
      })
    )
    .min(1)
    .describe("Portfolio holdings to analyze"),
  benchmark: z
    .string()
    .optional()
    .default("SPY")
    .describe("Benchmark symbol for comparison (default: SPY)"),
  confidence_level: z
    .number()
    .min(0.9)
    .max(0.99)
    .optional()
    .default(0.95)
    .describe("VaR confidence level (default: 0.95)"),
  method: z
    .enum(["historical", "parametric"])
    .optional()
    .default("historical")
    .describe("VaR calculation method"),
  lookback_days: z
    .number()
    .min(30)
    .max(756)
    .optional()
    .default(252)
    .describe("Historical days for calculations (default: 252)"),
  period: z
    .enum(["1M", "3M", "6M", "1Y"])
    .optional()
    .default("1M")
    .describe("Attribution period (for attribution action)"),
});

type PortfolioAnalyzeInput = z.infer<typeof PortfolioAnalyzeArgs>;

async function execute(
  args: PortfolioAnalyzeInput,
  ctx: ToolContext
): Promise<string> {
  const { action, holdings, benchmark, confidence_level, method, lookback_days, period } =
    args;

  // Convert holdings to API format
  const holdingsPayload = holdings.map((h) => ({
    symbol: h.symbol.toUpperCase(),
    shares: h.shares,
    average_cost: h.average_cost ?? 0,
  }));

  try {
    switch (action) {
      case "analyze": {
        const response = await stanleyFetch<PortfolioAnalytics>(
          "/api/portfolio/analytics",
          {
            method: "POST",
            body: { holdings: holdingsPayload, benchmark },
            signal: ctx.abort,
          }
        );

        if (!response.success || !response.data) {
          return `Portfolio analysis failed: ${response.error || "No data returned"}`;
        }

        const data = response.data;
        return formatPortfolioAnalytics(data);
      }

      case "risk": {
        const response = await stanleyFetch<RiskMetrics>("/api/portfolio/risk", {
          method: "POST",
          body: {
            holdings: holdingsPayload,
            confidence_level,
            method,
            lookback_days,
          },
          signal: ctx.abort,
        });

        if (!response.success || !response.data) {
          return `Risk analysis failed: ${response.error || "No data returned"}`;
        }

        return formatRiskMetrics(response.data);
      }

      case "correlation": {
        const response = await stanleyFetch<{
          symbols: string[];
          matrix: number[][];
          highly_correlated: Array<{
            symbol1: string;
            symbol2: string;
            correlation: number;
          }>;
          diversification_score: number;
        }>("/api/portfolio/correlation", {
          method: "POST",
          body: { holdings: holdingsPayload, lookback_days },
          signal: ctx.abort,
        });

        if (!response.success || !response.data) {
          return `Correlation analysis failed: ${response.error || "No data returned"}`;
        }

        const data = response.data;
        let output = `## Portfolio Correlation Analysis\n\n`;
        output += `**Diversification Score:** ${data.diversification_score}/100\n\n`;

        if (data.highly_correlated.length > 0) {
          output += `### Highly Correlated Pairs (>0.7)\n`;
          for (const pair of data.highly_correlated) {
            output += `- ${pair.symbol1} / ${pair.symbol2}: ${pair.correlation.toFixed(3)}\n`;
          }
        } else {
          output += `No highly correlated pairs found (>0.7 threshold).\n`;
        }

        return output;
      }

      case "attribution": {
        const response = await stanleyFetch<{
          period: string;
          total_return: number;
          benchmark_return: number;
          active_return: number;
          by_sector: Array<{ sector: string; weight: number; contribution: number }>;
          by_holding: Array<{
            symbol: string;
            weight: number;
            return_pct: number;
            contribution: number;
            sector: string;
          }>;
        }>("/api/portfolio/attribution", {
          method: "POST",
          body: { holdings: holdingsPayload, period, benchmark },
          signal: ctx.abort,
        });

        if (!response.success || !response.data) {
          return `Attribution analysis failed: ${response.error || "No data returned"}`;
        }

        const data = response.data;
        let output = `## Performance Attribution (${data.period})\n\n`;
        output += `**Total Return:** ${data.total_return.toFixed(2)}%\n`;
        output += `**Benchmark Return:** ${data.benchmark_return.toFixed(2)}%\n`;
        output += `**Active Return:** ${data.active_return.toFixed(2)}%\n\n`;

        if (data.by_sector.length > 0) {
          output += `### By Sector\n`;
          for (const s of data.by_sector) {
            output += `- ${s.sector}: ${s.contribution.toFixed(2)}% contribution (${s.weight.toFixed(1)}% weight)\n`;
          }
        }

        if (data.by_holding.length > 0) {
          output += `\n### Top Contributors\n`;
          const sorted = [...data.by_holding].sort(
            (a, b) => b.contribution - a.contribution
          );
          for (const h of sorted.slice(0, 5)) {
            output += `- ${h.symbol}: ${h.contribution.toFixed(2)}% (return: ${h.return_pct.toFixed(2)}%)\n`;
          }
        }

        return output;
      }

      case "sector-exposure": {
        const response = await stanleyFetch<{
          portfolio_weights: Record<string, number>;
          benchmark_weights: Record<string, number>;
          active_weights: Record<string, number>;
        }>("/api/portfolio/sector-exposure", {
          method: "POST",
          body: { holdings: holdingsPayload, benchmark },
          signal: ctx.abort,
        });

        if (!response.success || !response.data) {
          return `Sector exposure failed: ${response.error || "No data returned"}`;
        }

        const data = response.data;
        let output = `## Sector Exposure Analysis\n\n`;
        output += `| Sector | Portfolio | Benchmark | Active |\n`;
        output += `|--------|-----------|-----------|--------|\n`;

        const sectors = Object.keys(data.portfolio_weights);
        for (const sector of sectors.sort()) {
          const portfolio = data.portfolio_weights[sector] ?? 0;
          const benchmark_w = data.benchmark_weights[sector] ?? 0;
          const active = data.active_weights[sector] ?? 0;
          const activeSign = active >= 0 ? "+" : "";
          output += `| ${sector} | ${portfolio.toFixed(1)}% | ${benchmark_w.toFixed(1)}% | ${activeSign}${active.toFixed(1)}% |\n`;
        }

        return output;
      }

      default:
        return `Unknown action: ${action}`;
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return `Portfolio analysis error: ${message}`;
  }
}

function formatPortfolioAnalytics(data: PortfolioAnalytics): string {
  let output = `## Portfolio Analysis Summary\n\n`;

  // Value and Returns
  output += `### Value & Returns\n`;
  output += `- **Total Value:** $${data.totalValue.toLocaleString()}\n`;
  output += `- **Total Cost:** $${data.totalCost.toLocaleString()}\n`;
  output += `- **Total Return:** $${data.totalReturn.toLocaleString()} (${data.totalReturnPercent.toFixed(2)}%)\n\n`;

  // Risk Metrics
  output += `### Risk Metrics\n`;
  output += `- **Beta:** ${data.beta.toFixed(3)}\n`;
  output += `- **Alpha:** ${data.alpha.toFixed(3)}\n`;
  output += `- **Sharpe Ratio:** ${data.sharpeRatio.toFixed(3)}\n`;
  output += `- **Sortino Ratio:** ${data.sortinoRatio.toFixed(3)}\n`;
  output += `- **Volatility:** ${data.volatility.toFixed(2)}%\n`;
  output += `- **Max Drawdown:** ${data.maxDrawdown.toFixed(2)}%\n\n`;

  // VaR
  output += `### Value at Risk\n`;
  output += `- **VaR 95%:** $${data.var95.toLocaleString()} (${data.var95Percent.toFixed(2)}%)\n`;
  output += `- **VaR 99%:** $${data.var99.toLocaleString()} (${data.var99Percent.toFixed(2)}%)\n\n`;

  // Sector Exposure
  output += `### Sector Exposure\n`;
  const sectors = Object.entries(data.sectorExposure)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);
  for (const [sector, weight] of sectors) {
    output += `- ${sector}: ${weight.toFixed(1)}%\n`;
  }

  // Top Holdings
  if (data.topHoldings.length > 0) {
    output += `\n### Top Holdings\n`;
    for (const h of data.topHoldings.slice(0, 5)) {
      output += `- ${h.symbol}: $${h.marketValue.toLocaleString()} (${h.weight.toFixed(1)}%)\n`;
    }
  }

  return output;
}

function formatRiskMetrics(data: RiskMetrics): string {
  let output = `## Risk Metrics Analysis\n\n`;
  output += `**Method:** ${data.method} | **Lookback:** ${data.lookbackDays} days\n\n`;

  output += `### Value at Risk\n`;
  output += `- **VaR 95%:** $${data.var95.toLocaleString()} (${data.var95Percent.toFixed(2)}%)\n`;
  output += `- **VaR 99%:** $${data.var99.toLocaleString()} (${data.var99Percent.toFixed(2)}%)\n`;
  output += `- **CVaR 95% (Expected Shortfall):** $${data.cvar95.toLocaleString()}\n`;
  output += `- **CVaR 99%:** $${data.cvar99.toLocaleString()}\n\n`;

  output += `### Performance Metrics\n`;
  output += `- **Volatility:** ${data.volatility.toFixed(2)}%\n`;
  output += `- **Max Drawdown:** ${data.maxDrawdown.toFixed(2)}%\n`;
  output += `- **Sharpe Ratio:** ${data.sharpeRatio.toFixed(3)}\n`;
  output += `- **Sortino Ratio:** ${data.sortinoRatio.toFixed(3)}\n`;
  output += `- **Beta:** ${data.beta.toFixed(3)}\n`;

  return output;
}

export const portfolioAnalyze = {
  description: TOOL_DESCRIPTION,
  args: PortfolioAnalyzeArgs,
  execute,
};
