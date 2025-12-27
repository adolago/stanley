/**
 * Money Flow Tool
 *
 * MCP tool for sector money flow analysis, equity flow, sector rotation,
 * and market breadth indicators.
 */

import { z } from "zod";
import { stanleyFetch, buildQueryString } from "./api-client";
import type { ToolContext, StanleyApiResponse } from "./types";

const TOOL_DESCRIPTION = `Analyze money flow across sectors and individual equities.

This tool provides comprehensive money flow analytics including:
- Sector money flow: Net flows, institutional changes, smart money sentiment
- Equity flow: Money flow score, accumulation/distribution for individual stocks
- Sector rotation: Momentum rankings, relative strength, rotation signals
- Market breadth: Advance/decline ratio, McClellan Oscillator, breadth thrust

Sector ETFs covered:
- XLK (Technology), XLF (Financials), XLE (Energy)
- XLV (Healthcare), XLY (Consumer Discretionary), XLP (Consumer Staples)
- XLI (Industrials), XLB (Materials), XLU (Utilities)
- XLRE (Real Estate), XLC (Communications)

Use this tool when:
- Analyzing sector rotation and relative strength
- Tracking institutional money flow into/out of sectors
- Assessing market breadth and internal market health
- Identifying accumulation or distribution in individual stocks
- Detecting smart money positioning signals`;

// Input schema for money flow analysis
const MoneyFlowArgs = z.object({
  action: z
    .enum([
      "sector-flow",
      "equity-flow",
      "sector-rotation",
      "market-breadth",
      "smart-money",
      "unusual-volume",
      "flow-momentum",
      "comprehensive",
    ])
    .describe(
      "Type of analysis: 'sector-flow' for sector ETF flows, 'equity-flow' for individual stock flow, 'sector-rotation' for rotation signals, 'market-breadth' for breadth indicators, 'smart-money' for smart money tracking, 'unusual-volume' for volume anomalies, 'flow-momentum' for momentum indicators, 'comprehensive' for full analysis"
    ),
  symbol: z
    .string()
    .optional()
    .describe(
      "Stock ticker symbol (required for equity-flow, smart-money, unusual-volume, flow-momentum, comprehensive)"
    ),
  sectors: z
    .array(z.string())
    .optional()
    .default(["XLK", "XLF", "XLE", "XLV", "XLI"])
    .describe("Sector ETF symbols for sector-flow analysis"),
  lookback_days: z
    .number()
    .min(1)
    .max(365)
    .optional()
    .default(63)
    .describe("Historical days for analysis (default: 63 ~ 3 months)"),
  period: z
    .enum(["1W", "1M", "3M", "6M"])
    .optional()
    .default("1M")
    .describe("Analysis period for sector flow"),
});

type MoneyFlowInput = z.infer<typeof MoneyFlowArgs>;

// Response types
interface MoneyFlowData {
  symbol: string;
  netFlow1m: number;
  netFlow3m: number;
  institutionalChange: number;
  smartMoneySentiment: number;
  flowAcceleration: number;
  confidenceScore: number;
}

interface MoneyFlowResponse {
  sectors: Record<string, MoneyFlowData>;
  net_flows: Record<string, number>;
  momentum: Record<string, number>;
  timestamp: string;
}

interface EquityFlowData {
  symbol: string;
  moneyFlowScore: number;
  institutionalSentiment: number;
  smartMoneyActivity: number;
  shortPressure: number;
  accumulationDistribution: number;
  confidence: number;
}

interface SectorRotationData {
  sector: string;
  sectorName: string;
  relativeStrength: number;
  momentumScore: number;
  flowScore: number;
  rotationPhase: string;
  recommendation: string;
  oneMonthReturn: number;
  threeMonthReturn: number;
}

interface MarketBreadthData {
  advanceDeclineRatio: number;
  advancingVolume: number;
  decliningVolume: number;
  newHighsNewLows: number;
  percentAbove50DMA: number;
  percentAbove200DMA: number;
  mcclellanOscillator: number;
  breadthThrust: number;
  interpretation: string;
  timestamp: string;
}

interface SmartMoneyData {
  symbol: string;
  smartMoneyScore: number;
  institutionalAccumulation: number;
  darkPoolSentiment: number;
  blockTradeActivity: number;
  insiderSentiment: number;
  signalStrength: string;
  recommendation: string;
}

interface UnusualVolumeData {
  symbol: string;
  currentVolume: number;
  averageVolume: number;
  volumeRatio: number;
  zScore: number;
  percentile: number;
  isUnusual: boolean;
  direction: string;
  interpretation: string;
}

interface FlowMomentumData {
  symbol: string;
  flowMomentum: number;
  flowAcceleration: number;
  trendDirection: string;
  maCrossover: string;
  signalStrength: number;
  interpretation: string;
}

async function execute(
  args: MoneyFlowInput,
  ctx: ToolContext
): Promise<string> {
  const { action, symbol, sectors, lookback_days, period } = args;

  try {
    switch (action) {
      case "sector-flow": {
        const response = await stanleyFetch<MoneyFlowResponse>(
          "/api/money-flow",
          {
            method: "POST",
            body: { sectors, lookback_days, period },
            signal: ctx.abort,
          }
        );

        if (!response.success || !response.data) {
          return `Sector flow analysis failed: ${response.error || "No data returned"}`;
        }

        return formatSectorFlow(response.data);
      }

      case "equity-flow": {
        if (!symbol) {
          return "Error: Symbol is required for equity flow analysis. Provide a stock ticker like AAPL, MSFT, or NVDA.";
        }

        const query = buildQueryString({ lookback_days });
        const response = await stanleyFetch<EquityFlowData>(
          `/api/equity-flow/${symbol.toUpperCase()}${query}`,
          { signal: ctx.abort }
        );

        if (!response.success || !response.data) {
          return `Equity flow analysis failed: ${response.error || "No data returned"}`;
        }

        return formatEquityFlow(response.data);
      }

      case "sector-rotation": {
        const query = buildQueryString({ lookback_days });
        const response = await stanleyFetch<SectorRotationData[]>(
          `/api/sector-rotation${query}`,
          { signal: ctx.abort }
        );

        if (!response.success || !response.data) {
          return `Sector rotation analysis failed: ${response.error || "No data returned"}`;
        }

        return formatSectorRotation(response.data);
      }

      case "market-breadth": {
        const response = await stanleyFetch<MarketBreadthData>(
          "/api/market-breadth",
          { signal: ctx.abort }
        );

        if (!response.success || !response.data) {
          return `Market breadth analysis failed: ${response.error || "No data returned"}`;
        }

        return formatMarketBreadth(response.data);
      }

      case "smart-money": {
        if (!symbol) {
          return "Error: Symbol is required for smart money tracking.";
        }

        const query = buildQueryString({ lookback_days });
        const response = await stanleyFetch<SmartMoneyData>(
          `/api/smart-money/${symbol.toUpperCase()}${query}`,
          { signal: ctx.abort }
        );

        if (!response.success || !response.data) {
          return `Smart money tracking failed: ${response.error || "No data returned"}`;
        }

        return formatSmartMoney(response.data);
      }

      case "unusual-volume": {
        if (!symbol) {
          return "Error: Symbol is required for unusual volume detection.";
        }

        const query = buildQueryString({ lookback_days });
        const response = await stanleyFetch<UnusualVolumeData>(
          `/api/unusual-volume/${symbol.toUpperCase()}${query}`,
          { signal: ctx.abort }
        );

        if (!response.success || !response.data) {
          return `Unusual volume detection failed: ${response.error || "No data returned"}`;
        }

        return formatUnusualVolume(response.data);
      }

      case "flow-momentum": {
        if (!symbol) {
          return "Error: Symbol is required for flow momentum analysis.";
        }

        const query = buildQueryString({ lookback_days });
        const response = await stanleyFetch<FlowMomentumData>(
          `/api/flow-momentum/${symbol.toUpperCase()}${query}`,
          { signal: ctx.abort }
        );

        if (!response.success || !response.data) {
          return `Flow momentum analysis failed: ${response.error || "No data returned"}`;
        }

        return formatFlowMomentum(response.data);
      }

      case "comprehensive": {
        if (!symbol) {
          return "Error: Symbol is required for comprehensive analysis.";
        }

        const query = buildQueryString({ lookback_days });
        const response = await stanleyFetch<{
          symbol: string;
          equityFlow: EquityFlowData;
          smartMoney: SmartMoneyData;
          unusualVolume: UnusualVolumeData;
          flowMomentum: FlowMomentumData;
          alerts: Array<{ type: string; message: string; severity: string }>;
        }>(`/api/comprehensive/${symbol.toUpperCase()}${query}`, {
          signal: ctx.abort,
        });

        if (!response.success || !response.data) {
          return `Comprehensive analysis failed: ${response.error || "No data returned"}`;
        }

        return formatComprehensive(response.data);
      }

      default:
        return `Unknown action: ${action}`;
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return `Money flow analysis error: ${message}`;
  }
}

function formatSectorFlow(data: MoneyFlowResponse): string {
  let output = `## Sector Money Flow Analysis\n\n`;

  // Summary
  const flows = Object.values(data.sectors);
  const totalNetFlow = flows.reduce((sum, s) => sum + s.netFlow1m, 0);
  const avgSentiment =
    flows.reduce((sum, s) => sum + s.smartMoneySentiment, 0) / flows.length;

  output += `**Total Net Flow (1M):** $${(totalNetFlow / 1e6).toFixed(1)}M\n`;
  output += `**Avg Smart Money Sentiment:** ${avgSentiment >= 0 ? "+" : ""}${(avgSentiment * 100).toFixed(1)}%\n\n`;

  // Sector breakdown
  output += `### By Sector\n`;
  output += `| Sector | Net Flow 1M | Net Flow 3M | Inst. Change | Sentiment | Signal |\n`;
  output += `|--------|-------------|-------------|--------------|-----------|--------|\n`;

  const sortedSectors = Object.entries(data.sectors).sort(
    ([, a], [, b]) => b.netFlow1m - a.netFlow1m
  );

  for (const [, sector] of sortedSectors) {
    const signal =
      sector.confidenceScore > 0.3
        ? "Bullish"
        : sector.confidenceScore < -0.3
          ? "Bearish"
          : "Neutral";
    output += `| ${sector.symbol} | $${(sector.netFlow1m / 1e6).toFixed(1)}M | $${(sector.netFlow3m / 1e6).toFixed(1)}M | ${(sector.institutionalChange * 100).toFixed(2)}% | ${(sector.smartMoneySentiment * 100).toFixed(1)}% | ${signal} |\n`;
  }

  // Leaders and laggards
  output += `\n### Key Observations\n`;
  const leaders = sortedSectors.slice(0, 2);
  const laggards = sortedSectors.slice(-2).reverse();

  output += `**Inflows:** ${leaders.map(([, s]) => s.symbol).join(", ")}\n`;
  output += `**Outflows:** ${laggards.map(([, s]) => s.symbol).join(", ")}\n`;

  return output;
}

function formatEquityFlow(data: EquityFlowData): string {
  let output = `## Equity Money Flow: ${data.symbol}\n\n`;

  // Overall score
  const signalText =
    data.moneyFlowScore > 0.3
      ? "Bullish"
      : data.moneyFlowScore < -0.3
        ? "Bearish"
        : "Neutral";
  output += `**Money Flow Score:** ${(data.moneyFlowScore * 100).toFixed(1)}% (${signalText})\n`;
  output += `**Confidence:** ${(data.confidence * 100).toFixed(0)}%\n\n`;

  output += `### Flow Indicators\n`;
  output += `- **Institutional Sentiment:** ${data.institutionalSentiment >= 0 ? "+" : ""}${(data.institutionalSentiment * 100).toFixed(1)}%\n`;
  output += `- **Smart Money Activity:** ${data.smartMoneyActivity >= 0 ? "+" : ""}${(data.smartMoneyActivity * 100).toFixed(1)}%\n`;
  output += `- **Short Pressure:** ${(data.shortPressure * 100).toFixed(1)}%\n`;
  output += `- **Accumulation/Distribution:** ${data.accumulationDistribution >= 0 ? "+" : ""}${(data.accumulationDistribution * 100).toFixed(1)}%\n`;

  // Interpretation
  output += `\n### Interpretation\n`;
  if (data.moneyFlowScore > 0.3 && data.smartMoneyActivity > 0.2) {
    output += `Strong institutional buying pressure with smart money accumulation.\n`;
  } else if (data.moneyFlowScore < -0.3 && data.smartMoneyActivity < -0.2) {
    output += `Significant distribution pattern with institutional selling.\n`;
  } else if (data.shortPressure > 0.3) {
    output += `Elevated short interest may create squeeze potential.\n`;
  } else {
    output += `Mixed signals - wait for clearer flow direction.\n`;
  }

  return output;
}

function formatSectorRotation(data: SectorRotationData[]): string {
  let output = `## Sector Rotation Analysis\n\n`;

  // Group by phase
  const leaders = data.filter((s) => s.rotationPhase === "leading");
  const improving = data.filter((s) => s.rotationPhase === "improving");
  const weakening = data.filter((s) => s.rotationPhase === "weakening");
  const lagging = data.filter((s) => s.rotationPhase === "lagging");

  output += `### Rotation Phases\n`;
  if (leaders.length > 0) {
    output += `**Leading:** ${leaders.map((s) => s.sector).join(", ")}\n`;
  }
  if (improving.length > 0) {
    output += `**Improving:** ${improving.map((s) => s.sector).join(", ")}\n`;
  }
  if (weakening.length > 0) {
    output += `**Weakening:** ${weakening.map((s) => s.sector).join(", ")}\n`;
  }
  if (lagging.length > 0) {
    output += `**Lagging:** ${lagging.map((s) => s.sector).join(", ")}\n`;
  }

  // Full table
  output += `\n### Sector Rankings\n`;
  output += `| Sector | Name | RS | Momentum | 1M | 3M | Phase | Rec |\n`;
  output += `|--------|------|-----|----------|-----|-----|-------|-----|\n`;

  for (const sector of data) {
    output += `| ${sector.sector} | ${sector.sectorName} | ${sector.relativeStrength.toFixed(2)} | ${sector.momentumScore.toFixed(2)} | ${(sector.oneMonthReturn * 100).toFixed(1)}% | ${(sector.threeMonthReturn * 100).toFixed(1)}% | ${sector.rotationPhase} | ${sector.recommendation} |\n`;
  }

  // Recommendations
  output += `\n### Recommendations\n`;
  const overweight = data.filter((s) => s.recommendation === "overweight");
  const underweight = data.filter((s) => s.recommendation === "underweight");

  if (overweight.length > 0) {
    output += `**Overweight:** ${overweight.map((s) => `${s.sector} (${s.sectorName})`).join(", ")}\n`;
  }
  if (underweight.length > 0) {
    output += `**Underweight:** ${underweight.map((s) => `${s.sector} (${s.sectorName})`).join(", ")}\n`;
  }

  return output;
}

function formatMarketBreadth(data: MarketBreadthData): string {
  let output = `## Market Breadth Analysis\n\n`;

  output += `**Overall Interpretation:** ${data.interpretation.charAt(0).toUpperCase() + data.interpretation.slice(1)}\n\n`;

  output += `### Breadth Indicators\n`;
  output += `- **Advance/Decline Ratio:** ${data.advanceDeclineRatio.toFixed(2)}\n`;
  output += `- **Advancing Volume:** ${data.advancingVolume.toFixed(1)}%\n`;
  output += `- **Declining Volume:** ${data.decliningVolume.toFixed(1)}%\n`;
  output += `- **New Highs - New Lows:** ${data.newHighsNewLows.toFixed(0)}\n\n`;

  output += `### Moving Average Position\n`;
  output += `- **% Above 50 DMA:** ${data.percentAbove50DMA.toFixed(1)}%\n`;
  output += `- **% Above 200 DMA:** ${data.percentAbove200DMA.toFixed(1)}%\n\n`;

  output += `### Technical Indicators\n`;
  output += `- **McClellan Oscillator:** ${data.mcclellanOscillator.toFixed(2)}\n`;
  output += `- **Breadth Thrust:** ${data.breadthThrust.toFixed(4)}\n`;

  // Interpretation
  output += `\n### Market Health Assessment\n`;
  if (data.interpretation === "bullish") {
    output += `Market internals are healthy with broad participation. `;
    output += `A/D ratio > 1 and strong % above moving averages support uptrend.\n`;
  } else if (data.interpretation === "bearish") {
    output += `Market internals show deterioration. `;
    output += `Declining participation suggests caution despite any index-level strength.\n`;
  } else {
    output += `Mixed breadth signals. Market in transition - watch for directional confirmation.\n`;
  }

  return output;
}

function formatSmartMoney(data: SmartMoneyData): string {
  let output = `## Smart Money Tracking: ${data.symbol}\n\n`;

  output += `**Smart Money Score:** ${(data.smartMoneyScore * 100).toFixed(1)}%\n`;
  output += `**Signal Strength:** ${data.signalStrength}\n`;
  output += `**Recommendation:** ${data.recommendation}\n\n`;

  output += `### Components\n`;
  output += `- **Institutional Accumulation:** ${data.institutionalAccumulation >= 0 ? "+" : ""}${(data.institutionalAccumulation * 100).toFixed(1)}%\n`;
  output += `- **Dark Pool Sentiment:** ${data.darkPoolSentiment >= 0 ? "+" : ""}${(data.darkPoolSentiment * 100).toFixed(1)}%\n`;
  output += `- **Block Trade Activity:** ${(data.blockTradeActivity * 100).toFixed(1)}%\n`;
  output += `- **Insider Sentiment:** ${data.insiderSentiment >= 0 ? "+" : ""}${(data.insiderSentiment * 100).toFixed(1)}%\n`;

  return output;
}

function formatUnusualVolume(data: UnusualVolumeData): string {
  let output = `## Unusual Volume: ${data.symbol}\n\n`;

  const volumeStatus = data.isUnusual ? "UNUSUAL" : "Normal";
  output += `**Status:** ${volumeStatus}\n`;
  output += `**Direction:** ${data.direction}\n\n`;

  output += `### Volume Metrics\n`;
  output += `- **Current Volume:** ${data.currentVolume.toLocaleString()}\n`;
  output += `- **Average Volume:** ${data.averageVolume.toLocaleString()}\n`;
  output += `- **Volume Ratio:** ${data.volumeRatio.toFixed(2)}x\n`;
  output += `- **Z-Score:** ${data.zScore.toFixed(2)}\n`;
  output += `- **Percentile:** ${data.percentile.toFixed(1)}%\n\n`;

  output += `### Interpretation\n`;
  output += `${data.interpretation}\n`;

  return output;
}

function formatFlowMomentum(data: FlowMomentumData): string {
  let output = `## Flow Momentum: ${data.symbol}\n\n`;

  output += `**Trend Direction:** ${data.trendDirection}\n`;
  output += `**MA Crossover:** ${data.maCrossover}\n`;
  output += `**Signal Strength:** ${(data.signalStrength * 100).toFixed(0)}%\n\n`;

  output += `### Momentum Metrics\n`;
  output += `- **Flow Momentum:** ${data.flowMomentum >= 0 ? "+" : ""}${(data.flowMomentum * 100).toFixed(2)}%\n`;
  output += `- **Flow Acceleration:** ${data.flowAcceleration >= 0 ? "+" : ""}${(data.flowAcceleration * 100).toFixed(4)}%\n\n`;

  output += `### Interpretation\n`;
  output += `${data.interpretation}\n`;

  return output;
}

function formatComprehensive(data: {
  symbol: string;
  equityFlow: EquityFlowData;
  smartMoney: SmartMoneyData;
  unusualVolume: UnusualVolumeData;
  flowMomentum: FlowMomentumData;
  alerts: Array<{ type: string; message: string; severity: string }>;
}): string {
  let output = `## Comprehensive Money Flow Analysis: ${data.symbol}\n\n`;

  // Summary scores
  const ef = data.equityFlow;
  const sm = data.smartMoney;

  output += `### Summary Scores\n`;
  output += `| Metric | Score | Signal |\n`;
  output += `|--------|-------|--------|\n`;
  output += `| Money Flow | ${(ef.moneyFlowScore * 100).toFixed(1)}% | ${ef.moneyFlowScore > 0.3 ? "Bullish" : ef.moneyFlowScore < -0.3 ? "Bearish" : "Neutral"} |\n`;
  output += `| Smart Money | ${(sm.smartMoneyScore * 100).toFixed(1)}% | ${sm.recommendation} |\n`;
  output += `| Inst. Sentiment | ${(ef.institutionalSentiment * 100).toFixed(1)}% | ${ef.institutionalSentiment > 0.2 ? "Buying" : ef.institutionalSentiment < -0.2 ? "Selling" : "Mixed"} |\n`;
  output += `| Volume | ${data.unusualVolume.volumeRatio.toFixed(2)}x | ${data.unusualVolume.isUnusual ? "Unusual" : "Normal"} |\n\n`;

  // Flow momentum
  output += `### Flow Momentum\n`;
  output += `- **Trend:** ${data.flowMomentum.trendDirection}\n`;
  output += `- **Momentum:** ${(data.flowMomentum.flowMomentum * 100).toFixed(2)}%\n`;
  output += `- **MA Signal:** ${data.flowMomentum.maCrossover}\n\n`;

  // Alerts
  if (data.alerts && data.alerts.length > 0) {
    output += `### Active Alerts\n`;
    for (const alert of data.alerts) {
      const emoji =
        alert.severity === "high"
          ? "[!]"
          : alert.severity === "medium"
            ? "[*]"
            : "[-]";
      output += `${emoji} **${alert.type}:** ${alert.message}\n`;
    }
  }

  return output;
}

export const moneyFlow = {
  description: TOOL_DESCRIPTION,
  args: MoneyFlowArgs,
  execute,
};
