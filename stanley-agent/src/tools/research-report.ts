/**
 * Research Report Tool
 *
 * MCP tool for comprehensive equity research including valuation analysis,
 * earnings analysis, DCF models, and peer comparisons.
 */

import { z } from "zod";
import { stanleyFetch, buildQueryString } from "./api-client";
import type {
  ToolContext,
  ResearchReport,
  ValuationData,
  EarningsAnalysis,
  StanleyApiResponse,
} from "./types";

const TOOL_DESCRIPTION = `Generate comprehensive equity research reports and analysis.

This tool provides fundamental research capabilities including:
- Full research reports with valuation, earnings, and thesis
- Valuation analysis (multiples, DCF, sensitivity)
- Earnings analysis (trends, surprises, quality)
- Peer comparison (relative valuation)
- DCF models with customizable assumptions

Use this tool when:
- A user wants a complete research report on a stock
- Valuation analysis or fair value estimates are needed
- Earnings quality or trend analysis is required
- Peer group comparison for relative valuation
- DCF model with sensitivity analysis

The tool connects to Stanley's research analyzer which uses
real-time market data and financial statements.`;

// Input schema for research analysis
const ResearchReportArgs = z.object({
  action: z
    .enum(["report", "valuation", "earnings", "peers", "dcf", "summary"])
    .describe(
      "Type of analysis: 'report' for full research, 'valuation' for multiples/DCF, 'earnings' for earnings analysis, 'peers' for peer comparison, 'dcf' for detailed DCF model, 'summary' for quick overview"
    ),
  symbol: z.string().describe("Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"),
  include_dcf: z
    .boolean()
    .optional()
    .default(true)
    .describe("Include DCF analysis in valuation"),
  quarters: z
    .number()
    .min(1)
    .max(40)
    .optional()
    .default(12)
    .describe("Number of quarters for earnings analysis"),
  peers: z
    .string()
    .optional()
    .describe("Comma-separated peer symbols (auto-detected if not provided)"),
  discount_rate: z
    .number()
    .min(0.01)
    .max(0.30)
    .optional()
    .describe("Override discount rate (WACC) for DCF"),
  terminal_growth: z
    .number()
    .min(0)
    .max(0.05)
    .optional()
    .default(0.025)
    .describe("Terminal growth rate for DCF"),
  projection_years: z
    .number()
    .min(3)
    .max(10)
    .optional()
    .default(5)
    .describe("Projection years for DCF"),
});

type ResearchReportInput = z.infer<typeof ResearchReportArgs>;

async function execute(
  args: ResearchReportInput,
  ctx: ToolContext
): Promise<string> {
  const {
    action,
    symbol,
    include_dcf,
    quarters,
    peers,
    discount_rate,
    terminal_growth,
    projection_years,
  } = args;

  const upperSymbol = symbol.toUpperCase();

  try {
    switch (action) {
      case "report": {
        const response = await stanleyFetch<ResearchReport>(
          `/api/research/${upperSymbol}`,
          { signal: ctx.abort }
        );

        if (!response.success || !response.data) {
          return `Research report failed: ${response.error || "No data returned"}`;
        }

        return formatResearchReport(response.data);
      }

      case "valuation": {
        const query = buildQueryString({ include_dcf, method: "dcf" });
        const response = await stanleyFetch<ValuationData>(
          `/api/valuation/${upperSymbol}${query}`,
          { signal: ctx.abort }
        );

        if (!response.success || !response.data) {
          return `Valuation analysis failed: ${response.error || "No data returned"}`;
        }

        return formatValuation(response.data);
      }

      case "earnings": {
        const query = buildQueryString({ quarters, period: "quarterly" });
        const response = await stanleyFetch<EarningsAnalysis>(
          `/api/earnings/${upperSymbol}${query}`,
          { signal: ctx.abort }
        );

        if (!response.success || !response.data) {
          return `Earnings analysis failed: ${response.error || "No data returned"}`;
        }

        return formatEarnings(response.data);
      }

      case "peers": {
        const query = peers ? buildQueryString({ peers }) : "";
        const response = await stanleyFetch<{
          target: Record<string, unknown>;
          peerAverages: Record<string, number>;
          premiumDiscount: Record<string, number>;
          peers: Array<Record<string, unknown>>;
          fairValueRange?: { low: number; high: number };
        }>(`/api/peers/${upperSymbol}${query}`, { signal: ctx.abort });

        if (!response.success || !response.data) {
          return `Peer comparison failed: ${response.error || "No data returned"}`;
        }

        return formatPeerComparison(upperSymbol, response.data);
      }

      case "dcf": {
        const query = buildQueryString({
          discount_rate,
          terminal_growth,
          projection_years,
        });
        const response = await stanleyFetch<{
          symbol: string;
          dcf: {
            intrinsicValue: number;
            currentPrice: number;
            upsidePercentage: number;
            marginOfSafety: number;
            discountRate: number;
            terminalGrowthRate: number;
            projectionYears: number;
            pvCashFlows: number;
            pvTerminalValue: number;
            netDebt: number;
            sharesOutstanding: number;
          };
          sensitivity?: Record<string, unknown>;
          assumptions: Record<string, unknown>;
        }>(`/api/research/${upperSymbol}/dcf${query}`, { signal: ctx.abort });

        if (!response.success || !response.data) {
          return `DCF analysis failed: ${response.error || "No data returned"}`;
        }

        return formatDCF(response.data);
      }

      case "summary": {
        const response = await stanleyFetch<{
          symbol: string;
          companyName: string;
          sector: string;
          industry: string;
          currentPrice: number;
          marketCap: number;
          valuationRating: string;
          overallScore: number;
          keyMetrics: Record<string, number | null>;
          growth: Record<string, number>;
          margins: Record<string, number>;
          fairValueRange: { low: number; high: number };
          topStrengths: string[];
          topRisks: string[];
        }>(`/api/research/${upperSymbol}/summary`, { signal: ctx.abort });

        if (!response.success || !response.data) {
          return `Research summary failed: ${response.error || "No data returned"}`;
        }

        return formatSummary(response.data);
      }

      default:
        return `Unknown action: ${action}`;
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return `Research analysis error: ${message}`;
  }
}

function formatResearchReport(data: ResearchReport): string {
  let output = `## Research Report: ${data.symbol}\n`;
  output += `**${data.companyName}** | ${data.sector} > ${data.industry}\n\n`;

  // Key Metrics
  output += `### Key Metrics\n`;
  output += `- **Current Price:** $${data.currentPrice.toFixed(2)}\n`;
  output += `- **Market Cap:** $${(data.marketCap / 1e9).toFixed(2)}B\n`;
  output += `- **Valuation Rating:** ${data.valuationRating}\n`;
  output += `- **Overall Score:** ${data.overallScore.toFixed(1)}/100\n\n`;

  // Fair Value
  output += `### Fair Value Range\n`;
  output += `- **Low:** $${data.fairValueRange.low.toFixed(2)}\n`;
  output += `- **High:** $${data.fairValueRange.high.toFixed(2)}\n`;
  const midpoint = (data.fairValueRange.low + data.fairValueRange.high) / 2;
  const upside = ((midpoint - data.currentPrice) / data.currentPrice) * 100;
  output += `- **Implied Upside:** ${upside >= 0 ? "+" : ""}${upside.toFixed(1)}%\n\n`;

  // Profitability
  output += `### Profitability\n`;
  output += `- **Gross Margin:** ${data.grossMargin.toFixed(1)}%\n`;
  output += `- **Operating Margin:** ${data.operatingMargin.toFixed(1)}%\n`;
  output += `- **Net Margin:** ${data.netMargin.toFixed(1)}%\n`;
  output += `- **ROE:** ${data.roe.toFixed(1)}%\n`;
  output += `- **ROIC:** ${data.roic.toFixed(1)}%\n\n`;

  // Growth
  output += `### Growth (5Y)\n`;
  output += `- **Revenue Growth:** ${data.revenueGrowth5yr.toFixed(1)}%\n`;
  output += `- **EPS Growth:** ${data.epsGrowth5yr.toFixed(1)}%\n`;
  output += `- **Earnings Quality Score:** ${data.earningsQualityScore.toFixed(1)}/100\n\n`;

  // Financial Health
  output += `### Financial Health\n`;
  output += `- **Debt-to-Equity:** ${data.debtToEquity.toFixed(2)}\n`;
  output += `- **Current Ratio:** ${data.currentRatio.toFixed(2)}\n\n`;

  // Investment Thesis
  if (data.strengths.length > 0) {
    output += `### Strengths\n`;
    data.strengths.slice(0, 4).forEach((s) => {
      output += `- ${s}\n`;
    });
    output += `\n`;
  }

  if (data.weaknesses.length > 0) {
    output += `### Weaknesses\n`;
    data.weaknesses.slice(0, 4).forEach((w) => {
      output += `- ${w}\n`;
    });
    output += `\n`;
  }

  if (data.catalysts.length > 0) {
    output += `### Catalysts\n`;
    data.catalysts.slice(0, 3).forEach((c) => {
      output += `- ${c}\n`;
    });
    output += `\n`;
  }

  if (data.risks.length > 0) {
    output += `### Risks\n`;
    data.risks.slice(0, 3).forEach((r) => {
      output += `- ${r}\n`;
    });
  }

  return output;
}

function formatValuation(data: ValuationData): string {
  let output = `## Valuation Analysis: ${data.symbol}\n\n`;
  output += `**Method:** ${data.method}\n\n`;

  // Trading Multiples
  if (data.valuation) {
    output += `### Trading Multiples\n`;
    const v = data.valuation as Record<string, unknown>;
    if (v.peRatio) output += `- **P/E Ratio:** ${Number(v.peRatio).toFixed(1)}x\n`;
    if (v.forwardPe) output += `- **Forward P/E:** ${Number(v.forwardPe).toFixed(1)}x\n`;
    if (v.pegRatio) output += `- **PEG Ratio:** ${Number(v.pegRatio).toFixed(2)}\n`;
    if (v.evToEbitda) output += `- **EV/EBITDA:** ${Number(v.evToEbitda).toFixed(1)}x\n`;
    if (v.priceToSales) output += `- **P/S Ratio:** ${Number(v.priceToSales).toFixed(1)}x\n`;
    if (v.priceToBook) output += `- **P/B Ratio:** ${Number(v.priceToBook).toFixed(1)}x\n`;
    if (v.fcfYield) output += `- **FCF Yield:** ${Number(v.fcfYield).toFixed(2)}%\n`;
    output += `\n`;
  }

  // DCF Summary
  if (data.dcf) {
    const dcf = data.dcf as Record<string, unknown>;
    output += `### DCF Valuation\n`;
    if (dcf.intrinsicValue) output += `- **Intrinsic Value:** $${Number(dcf.intrinsicValue).toFixed(2)}\n`;
    if (dcf.currentPrice) output += `- **Current Price:** $${Number(dcf.currentPrice).toFixed(2)}\n`;
    if (dcf.upsidePercentage !== undefined) {
      const upside = Number(dcf.upsidePercentage);
      output += `- **Upside:** ${upside >= 0 ? "+" : ""}${upside.toFixed(1)}%\n`;
    }
    if (dcf.marginOfSafety !== undefined) {
      output += `- **Margin of Safety:** ${Number(dcf.marginOfSafety).toFixed(1)}%\n`;
    }
  }

  // Fair Value Summary
  if (data.fairValue && data.currentPrice) {
    output += `\n### Conclusion\n`;
    const upside = data.upsidePercent ?? ((data.fairValue - data.currentPrice) / data.currentPrice * 100);
    if (upside > 15) {
      output += `Stock appears **undervalued** with ${upside.toFixed(1)}% upside potential.\n`;
    } else if (upside < -15) {
      output += `Stock appears **overvalued** with ${upside.toFixed(1)}% downside.\n`;
    } else {
      output += `Stock appears **fairly valued** near current levels.\n`;
    }
  }

  return output;
}

function formatEarnings(data: EarningsAnalysis): string {
  let output = `## Earnings Analysis: ${data.symbol}\n\n`;

  // Key Metrics
  output += `### Earnings Quality\n`;
  output += `- **EPS Growth (YoY):** ${data.epsGrowthYoy.toFixed(1)}%\n`;
  output += `- **EPS Growth (3Y CAGR):** ${data.epsGrowth3yrCagr.toFixed(1)}%\n`;
  output += `- **Earnings Consistency:** ${data.earningsConsistency.toFixed(1)}%\n`;
  output += `- **Earnings Volatility:** ${data.earningsVolatility.toFixed(1)}%\n\n`;

  // Beat/Miss Record
  output += `### Surprise Record\n`;
  output += `- **Beat Rate:** ${(data.beatRate * 100).toFixed(0)}%\n`;
  output += `- **Avg Surprise:** ${data.avgEpsSurprisePercent >= 0 ? "+" : ""}${data.avgEpsSurprisePercent.toFixed(1)}%\n`;
  output += `- **Consecutive Beats:** ${data.consecutiveBeats}\n\n`;

  // Recent Quarters
  if (data.quarters && data.quarters.length > 0) {
    output += `### Recent Quarters\n`;
    output += `| Quarter | EPS Actual | EPS Est | Surprise |\n`;
    output += `|---------|------------|---------|----------|\n`;
    for (const q of data.quarters.slice(0, 6)) {
      const quarter = q.fiscalQuarter || "N/A";
      const actual = typeof q.epsActual === "number" ? `$${q.epsActual.toFixed(2)}` : "N/A";
      const est = typeof q.epsEstimate === "number" ? `$${q.epsEstimate.toFixed(2)}` : "N/A";
      const surprise = typeof q.epsSurprisePercent === "number"
        ? `${q.epsSurprisePercent >= 0 ? "+" : ""}${q.epsSurprisePercent.toFixed(1)}%`
        : "N/A";
      output += `| ${quarter} | ${actual} | ${est} | ${surprise} |\n`;
    }
  }

  return output;
}

function formatPeerComparison(
  symbol: string,
  data: {
    target: Record<string, unknown>;
    peerAverages: Record<string, number>;
    premiumDiscount: Record<string, number>;
    peers: Array<Record<string, unknown>>;
    fairValueRange?: { low: number; high: number };
  }
): string {
  let output = `## Peer Comparison: ${symbol}\n\n`;

  // Premium/Discount
  output += `### Relative Valuation\n`;
  const metrics = ["peRatio", "evToEbitda", "priceToSales", "priceToBook"];
  for (const metric of metrics) {
    const premium = data.premiumDiscount[metric];
    if (premium !== undefined) {
      const label = metric
        .replace(/([A-Z])/g, " $1")
        .replace(/^./, (s) => s.toUpperCase());
      output += `- **${label}:** ${premium >= 0 ? "+" : ""}${premium.toFixed(1)}% vs peers\n`;
    }
  }
  output += `\n`;

  // Peer Table
  if (data.peers && data.peers.length > 0) {
    output += `### Peer Multiples\n`;
    output += `| Symbol | P/E | EV/EBITDA | P/S |\n`;
    output += `|--------|-----|-----------|-----|\n`;

    // Add target company
    const t = data.target;
    output += `| **${symbol}** | ${formatNum(t.peRatio)} | ${formatNum(t.evToEbitda)} | ${formatNum(t.priceToSales)} |\n`;

    // Add peers
    for (const p of data.peers.slice(0, 6)) {
      const sym = p.symbol || "N/A";
      output += `| ${sym} | ${formatNum(p.peRatio)} | ${formatNum(p.evToEbitda)} | ${formatNum(p.priceToSales)} |\n`;
    }

    // Add averages
    output += `| *Avg* | ${formatNum(data.peerAverages.peRatio)} | ${formatNum(data.peerAverages.evToEbitda)} | ${formatNum(data.peerAverages.priceToSales)} |\n`;
  }

  // Fair Value Range
  if (data.fairValueRange) {
    output += `\n### Implied Fair Value\n`;
    output += `- **Low:** $${data.fairValueRange.low.toFixed(2)}\n`;
    output += `- **High:** $${data.fairValueRange.high.toFixed(2)}\n`;
  }

  return output;
}

function formatDCF(data: {
  symbol: string;
  dcf: {
    intrinsicValue: number;
    currentPrice: number;
    upsidePercentage: number;
    marginOfSafety: number;
    discountRate: number;
    terminalGrowthRate: number;
    projectionYears: number;
    pvCashFlows: number;
    pvTerminalValue: number;
    netDebt: number;
    sharesOutstanding: number;
  };
  sensitivity?: Record<string, unknown>;
  assumptions: Record<string, unknown>;
}): string {
  const dcf = data.dcf;
  let output = `## DCF Valuation: ${data.symbol}\n\n`;

  // Key Results
  output += `### Valuation Summary\n`;
  output += `- **Intrinsic Value:** $${dcf.intrinsicValue.toFixed(2)}\n`;
  output += `- **Current Price:** $${dcf.currentPrice.toFixed(2)}\n`;
  output += `- **Upside/Downside:** ${dcf.upsidePercentage >= 0 ? "+" : ""}${dcf.upsidePercentage.toFixed(1)}%\n`;
  output += `- **Margin of Safety:** ${dcf.marginOfSafety.toFixed(1)}%\n\n`;

  // Model Components
  output += `### Model Components\n`;
  output += `- **PV of Cash Flows:** $${(dcf.pvCashFlows / 1e9).toFixed(2)}B\n`;
  output += `- **PV of Terminal Value:** $${(dcf.pvTerminalValue / 1e9).toFixed(2)}B\n`;
  output += `- **Net Debt:** $${(dcf.netDebt / 1e9).toFixed(2)}B\n`;
  output += `- **Shares Outstanding:** ${(dcf.sharesOutstanding / 1e6).toFixed(1)}M\n\n`;

  // Assumptions
  output += `### Key Assumptions\n`;
  output += `- **Discount Rate (WACC):** ${(dcf.discountRate * 100).toFixed(1)}%\n`;
  output += `- **Terminal Growth Rate:** ${(dcf.terminalGrowthRate * 100).toFixed(1)}%\n`;
  output += `- **Projection Period:** ${dcf.projectionYears} years\n`;

  return output;
}

function formatSummary(data: {
  symbol: string;
  companyName: string;
  sector: string;
  industry: string;
  currentPrice: number;
  marketCap: number;
  valuationRating: string;
  overallScore: number;
  keyMetrics: Record<string, number | null>;
  growth: Record<string, number>;
  margins: Record<string, number>;
  fairValueRange: { low: number; high: number };
  topStrengths: string[];
  topRisks: string[];
}): string {
  let output = `## Quick Summary: ${data.symbol}\n`;
  output += `**${data.companyName}** | ${data.sector}\n\n`;

  output += `**Price:** $${data.currentPrice.toFixed(2)} | **Market Cap:** $${(data.marketCap / 1e9).toFixed(1)}B\n`;
  output += `**Rating:** ${data.valuationRating} | **Score:** ${data.overallScore.toFixed(0)}/100\n\n`;

  output += `**Fair Value Range:** $${data.fairValueRange.low.toFixed(2)} - $${data.fairValueRange.high.toFixed(2)}\n\n`;

  if (data.topStrengths.length > 0) {
    output += `**Strengths:** ${data.topStrengths.join("; ")}\n\n`;
  }

  if (data.topRisks.length > 0) {
    output += `**Risks:** ${data.topRisks.join("; ")}\n`;
  }

  return output;
}

function formatNum(val: unknown): string {
  if (val === null || val === undefined) return "N/A";
  if (typeof val === "number") return val.toFixed(1);
  return String(val);
}

export const researchReport = {
  description: TOOL_DESCRIPTION,
  args: ResearchReportArgs,
  execute,
};
