/**
 * Commodities Data Tool
 *
 * MCP tool for commodity market analysis including prices, correlations,
 * macro linkages, and futures curve analysis.
 */

import { z } from "zod";
import { stanleyFetch, buildQueryString } from "./api-client";
import type {
  ToolContext,
  CommoditySummary,
  MacroLinkage,
  CorrelationMatrix,
  StanleyApiResponse,
} from "./types";

const TOOL_DESCRIPTION = `Access commodity market data and analysis.

This tool provides comprehensive commodity analytics including:
- Market overview across all commodity categories
- Individual commodity detail (prices, trends, volatility)
- Correlation matrices between commodities
- Macro-commodity linkages (USD, inflation, growth)
- Futures curve analysis (contango/backwardation)

Commodity categories covered:
- Energy: Crude Oil (CL), Natural Gas (NG), Gasoline (RB)
- Precious Metals: Gold (GC), Silver (SI), Platinum (PL)
- Base Metals: Copper (HG), Aluminum (ALI)
- Agriculture: Corn (ZC), Wheat (ZW), Soybeans (ZS)
- Softs: Coffee (KC), Sugar (SB), Cocoa (CC)

Use this tool when:
- A user asks about commodity prices or trends
- Correlation analysis between commodities is needed
- Understanding macro-commodity relationships
- Futures curve shape analysis for storage/supply signals`;

// Input schema for commodities analysis
const CommoditiesDataArgs = z.object({
  action: z
    .enum(["overview", "detail", "correlations", "macro-linkage", "futures-curve"])
    .describe(
      "Type of analysis: 'overview' for market overview, 'detail' for specific commodity, 'correlations' for correlation matrix, 'macro-linkage' for economic linkages, 'futures-curve' for term structure"
    ),
  symbol: z
    .string()
    .optional()
    .describe(
      "Commodity symbol (e.g., CL for crude oil, GC for gold). Required for detail, macro-linkage, and futures-curve actions."
    ),
  commodities: z
    .string()
    .optional()
    .describe(
      "Comma-separated list of commodity symbols for correlation analysis (e.g., 'CL,GC,NG')"
    ),
  lookback_days: z
    .number()
    .min(30)
    .max(756)
    .optional()
    .default(252)
    .describe("Historical days for analysis (default: 252)"),
  num_contracts: z
    .number()
    .min(2)
    .max(12)
    .optional()
    .default(6)
    .describe("Number of futures contracts for curve analysis"),
});

type CommoditiesDataInput = z.infer<typeof CommoditiesDataArgs>;

async function execute(
  args: CommoditiesDataInput,
  ctx: ToolContext
): Promise<string> {
  const { action, symbol, commodities, lookback_days, num_contracts } = args;

  try {
    switch (action) {
      case "overview": {
        const response = await stanleyFetch<{
          timestamp: string;
          sentiment: string;
          avgChange: number;
          categories: Record<
            string,
            {
              category: string;
              count: number;
              avgChange: number;
              leader?: {
                symbol: string;
                name: string;
                price: number;
                change: number;
                changePercent: number;
              };
              laggard?: {
                symbol: string;
                name: string;
                price: number;
                change: number;
                changePercent: number;
              };
              commodities: Array<{
                symbol: string;
                name: string;
                price: number;
                change: number;
                changePercent: number;
              }>;
            }
          >;
        }>("/api/commodities", { signal: ctx.abort });

        if (!response.success || !response.data) {
          return `Commodities overview failed: ${response.error || "No data returned"}`;
        }

        return formatMarketOverview(response.data);
      }

      case "detail": {
        if (!symbol) {
          return "Error: Symbol is required for commodity detail. Provide a commodity symbol like CL (crude oil), GC (gold), or NG (natural gas).";
        }

        const query = buildQueryString({ lookback_days });
        const response = await stanleyFetch<CommoditySummary>(
          `/api/commodities/${symbol.toUpperCase()}${query}`,
          { signal: ctx.abort }
        );

        if (!response.success || !response.data) {
          return `Commodity detail failed: ${response.error || "No data returned"}`;
        }

        return formatCommodityDetail(response.data);
      }

      case "correlations": {
        const query = buildQueryString({
          commodities,
          lookback_days,
        });
        const response = await stanleyFetch<CorrelationMatrix>(
          `/api/commodities/correlations${query}`,
          { signal: ctx.abort }
        );

        if (!response.success || !response.data) {
          return `Correlation analysis failed: ${response.error || "No data returned"}`;
        }

        return formatCorrelationMatrix(response.data);
      }

      case "macro-linkage": {
        if (!symbol) {
          return "Error: Symbol is required for macro-linkage analysis.";
        }

        const query = buildQueryString({ lookback_days });
        const response = await stanleyFetch<MacroLinkage>(
          `/api/commodities/${symbol.toUpperCase()}/macro${query}`,
          { signal: ctx.abort }
        );

        if (!response.success || !response.data) {
          return `Macro-linkage analysis failed: ${response.error || "No data returned"}`;
        }

        return formatMacroLinkage(response.data);
      }

      case "futures-curve": {
        if (!symbol) {
          return "Error: Symbol is required for futures curve analysis.";
        }

        const query = buildQueryString({ num_contracts });
        const response = await stanleyFetch<{
          symbol: string;
          name: string;
          curveShape: string;
          frontMonth: {
            contract: string;
            expiry: string;
            price: number;
            openInterest: number;
            volume: number;
          };
          curve: Array<{
            contract: string;
            expiry: string;
            price: number;
            openInterest: number;
            volume: number;
          }>;
          rollYield: number;
          timestamp: string;
        }>(`/api/commodities/futures-curve/${symbol.toUpperCase()}${query}`, {
          signal: ctx.abort,
        });

        if (!response.success || !response.data) {
          return `Futures curve analysis failed: ${response.error || "No data returned"}`;
        }

        return formatFuturesCurve(response.data);
      }

      default:
        return `Unknown action: ${action}`;
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return `Commodities analysis error: ${message}`;
  }
}

function formatMarketOverview(data: {
  timestamp: string;
  sentiment: string;
  avgChange: number;
  categories: Record<
    string,
    {
      category: string;
      count: number;
      avgChange: number;
      leader?: {
        symbol: string;
        name: string;
        price: number;
        changePercent: number;
      };
      laggard?: {
        symbol: string;
        name: string;
        price: number;
        changePercent: number;
      };
      commodities: Array<{
        symbol: string;
        name: string;
        price: number;
        changePercent: number;
      }>;
    }
  >;
}): string {
  let output = `## Commodity Market Overview\n\n`;
  output += `**Market Sentiment:** ${data.sentiment}\n`;
  output += `**Average Change:** ${data.avgChange >= 0 ? "+" : ""}${data.avgChange.toFixed(2)}%\n\n`;

  for (const [categoryName, cat] of Object.entries(data.categories)) {
    output += `### ${categoryName}\n`;
    output += `Avg Change: ${cat.avgChange >= 0 ? "+" : ""}${cat.avgChange.toFixed(2)}%\n`;

    if (cat.leader) {
      output += `- **Leader:** ${cat.leader.name} (${cat.leader.symbol}): $${cat.leader.price.toFixed(2)} (${cat.leader.changePercent >= 0 ? "+" : ""}${cat.leader.changePercent.toFixed(2)}%)\n`;
    }
    if (cat.laggard) {
      output += `- **Laggard:** ${cat.laggard.name} (${cat.laggard.symbol}): $${cat.laggard.price.toFixed(2)} (${cat.laggard.changePercent >= 0 ? "+" : ""}${cat.laggard.changePercent.toFixed(2)}%)\n`;
    }
    output += `\n`;
  }

  return output;
}

function formatCommodityDetail(data: CommoditySummary): string {
  let output = `## ${data.name} (${data.symbol})\n`;
  output += `**Category:** ${data.category}\n\n`;

  output += `### Price\n`;
  output += `- **Current:** $${data.price.toFixed(2)}\n`;
  output += `- **1D Change:** ${data.change1d >= 0 ? "+" : ""}${data.change1d.toFixed(2)}%\n`;
  output += `- **1W Change:** ${data.change1w >= 0 ? "+" : ""}${data.change1w.toFixed(2)}%\n`;
  output += `- **1M Change:** ${data.change1m >= 0 ? "+" : ""}${data.change1m.toFixed(2)}%\n`;
  output += `- **YTD Change:** ${data.changeYtd >= 0 ? "+" : ""}${data.changeYtd.toFixed(2)}%\n\n`;

  output += `### Technicals\n`;
  output += `- **30D Volatility:** ${data.volatility30d.toFixed(2)}%\n`;
  output += `- **Trend:** ${data.trend}\n`;
  output += `- **Relative Strength:** ${data.relativeStrength.toFixed(2)}\n`;

  return output;
}

function formatCorrelationMatrix(data: CorrelationMatrix): string {
  let output = `## Commodity Correlation Matrix\n\n`;

  if (data.commodities.length === 0) {
    return output + "No correlation data available.\n";
  }

  // Create table header
  output += `| | ${data.commodities.join(" | ")} |\n`;
  output += `|${"-".repeat(3)}|${data.commodities.map(() => "-".repeat(6)).join("|")}|\n`;

  // Create table rows
  for (const commodity of data.commodities) {
    const row = data.matrix[commodity];
    if (!row) continue;

    const values = data.commodities.map((c) => {
      const val = row[c];
      if (val === undefined || val === null) return "N/A";
      return val.toFixed(2);
    });
    output += `| **${commodity}** | ${values.join(" | ")} |\n`;
  }

  // Identify high correlations
  output += `\n### Notable Correlations\n`;
  const pairs: Array<{ c1: string; c2: string; corr: number }> = [];
  for (let i = 0; i < data.commodities.length; i++) {
    for (let j = i + 1; j < data.commodities.length; j++) {
      const c1 = data.commodities[i];
      const c2 = data.commodities[j];
      const corr = data.matrix[c1]?.[c2];
      if (corr !== undefined && Math.abs(corr) > 0.5) {
        pairs.push({ c1, c2, corr });
      }
    }
  }

  pairs.sort((a, b) => Math.abs(b.corr) - Math.abs(a.corr));
  if (pairs.length === 0) {
    output += "No strongly correlated pairs (>0.5) found.\n";
  } else {
    for (const { c1, c2, corr } of pairs.slice(0, 5)) {
      const direction = corr > 0 ? "positive" : "negative";
      output += `- ${c1} / ${c2}: ${corr.toFixed(2)} (${direction})\n`;
    }
  }

  return output;
}

function formatMacroLinkage(data: MacroLinkage): string {
  let output = `## Macro-Commodity Linkages: ${data.name} (${data.commodity})\n`;
  output += `**Category:** ${data.category}\n`;
  if (data.primaryDriver) {
    output += `**Primary Driver:** ${data.primaryDriver}\n`;
  }
  output += `\n`;

  if (data.linkages.length === 0) {
    return output + "No macro linkages identified.\n";
  }

  output += `### Economic Relationships\n`;
  output += `| Indicator | Correlation | Lead/Lag | Relationship | Strength |\n`;
  output += `|-----------|-------------|----------|--------------|----------|\n`;

  for (const link of data.linkages) {
    const lag =
      link.leadLagDays === 0
        ? "Concurrent"
        : link.leadLagDays > 0
          ? `+${link.leadLagDays}d`
          : `${link.leadLagDays}d`;
    output += `| ${link.macroIndicator} | ${link.correlation.toFixed(2)} | ${lag} | ${link.relationship} | ${link.strength} |\n`;
  }

  output += `\n### Interpretation\n`;
  const strongLinks = data.linkages.filter((l) => l.strength === "strong" || l.strength === "high");
  if (strongLinks.length > 0) {
    output += `Strong relationships with: ${strongLinks.map((l) => l.macroIndicator).join(", ")}\n`;
  }

  return output;
}

function formatFuturesCurve(data: {
  symbol: string;
  name: string;
  curveShape: string;
  frontMonth: {
    contract: string;
    expiry: string;
    price: number;
    openInterest: number;
    volume: number;
  };
  curve: Array<{
    contract: string;
    expiry: string;
    price: number;
    openInterest: number;
    volume: number;
  }>;
  rollYield: number;
  timestamp: string;
}): string {
  let output = `## Futures Curve: ${data.name} (${data.symbol})\n\n`;

  output += `### Curve Shape\n`;
  output += `**Shape:** ${data.curveShape.charAt(0).toUpperCase() + data.curveShape.slice(1)}\n`;
  output += `**Roll Yield (Annualized):** ${data.rollYield >= 0 ? "+" : ""}${data.rollYield.toFixed(2)}%\n\n`;

  // Interpretation
  output += `### Interpretation\n`;
  if (data.curveShape === "contango") {
    output += `- Curve is in **contango** (futures > spot)\n`;
    output += `- Implies storage costs and adequate supply\n`;
    output += `- Negative roll yield for long positions\n`;
  } else if (data.curveShape === "backwardation") {
    output += `- Curve is in **backwardation** (futures < spot)\n`;
    output += `- Implies tight near-term supply\n`;
    output += `- Positive roll yield for long positions\n`;
  } else {
    output += `- Curve is **flat**\n`;
    output += `- No significant term structure premium\n`;
  }
  output += `\n`;

  // Curve Table
  output += `### Term Structure\n`;
  output += `| Contract | Expiry | Price | Open Interest | Volume |\n`;
  output += `|----------|--------|-------|---------------|--------|\n`;

  for (const contract of data.curve) {
    output += `| ${contract.contract} | ${contract.expiry} | $${contract.price.toFixed(2)} | ${contract.openInterest.toLocaleString()} | ${contract.volume.toLocaleString()} |\n`;
  }

  return output;
}

export const commoditiesData = {
  description: TOOL_DESCRIPTION,
  args: CommoditiesDataArgs,
  execute,
};
