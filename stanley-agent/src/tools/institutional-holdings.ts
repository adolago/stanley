/**
 * Institutional Holdings Tool
 *
 * MCP tool for institutional ownership analysis including 13F filings,
 * ownership breakdown, smart money tracking, and sentiment scoring.
 */

import { z } from "zod";
import { stanleyFetch, buildQueryString } from "./api-client";
import type {
  ToolContext,
  InstitutionalHolding,
  OwnershipBreakdown,
  InstitutionalSentiment,
  SmartMoneyFlow,
  StanleyApiResponse,
} from "./types";

const TOOL_DESCRIPTION = `Analyze institutional ownership and 13F filings.

This tool provides comprehensive institutional analytics including:
- 13F holdings from SEC filings (top institutional holders)
- Ownership breakdown (institutional vs retail vs insider)
- Position changes and conviction scoring
- Whale accumulation detection (large position changes)
- Multi-factor sentiment scoring
- Smart money flow tracking
- Cross-filing pattern analysis
- Coordinated buying/selling alerts

Use this tool when:
- A user asks about institutional ownership of a stock
- 13F filing analysis or position changes are needed
- Smart money activity tracking is required
- Whale accumulation/distribution detection
- Institutional sentiment scoring

Data is sourced from SEC 13F filings (quarterly, ~45 day delay).`;

// Input schema for institutional analysis
const InstitutionalHoldingsArgs = z.object({
  action: z
    .enum([
      "holdings",
      "ownership",
      "changes",
      "whales",
      "sentiment",
      "clusters",
      "momentum",
      "smart-money-flow",
      "new-positions",
      "coordinated-buying",
      "conviction-picks",
    ])
    .describe(
      "Type of analysis: 'holdings' for 13F data, 'ownership' for breakdown, 'changes' for position changes, 'whales' for large accumulation, 'sentiment' for scoring, 'clusters' for position grouping, 'momentum' for trend tracking, 'smart-money-flow' for net buying/selling, 'new-positions' for recent initiations, 'coordinated-buying' for multi-manager activity, 'conviction-picks' for high-weight positions"
    ),
  symbol: z
    .string()
    .optional()
    .describe(
      "Stock ticker symbol. Required for symbol-specific actions (holdings, ownership, changes, whales, sentiment, clusters, momentum, smart-money-flow)."
    ),
  conviction_threshold: z
    .number()
    .min(0)
    .max(1)
    .optional()
    .default(0.05)
    .describe("Minimum change to consider significant (default: 5%)"),
  min_position_change: z
    .number()
    .min(0)
    .max(1)
    .optional()
    .default(0.1)
    .describe("Minimum position change for whale detection (default: 10%)"),
  min_aum: z
    .number()
    .optional()
    .default(1e9)
    .describe("Minimum AUM to consider 'whale' (default: $1B)"),
  n_clusters: z
    .number()
    .min(2)
    .max(10)
    .optional()
    .default(4)
    .describe("Number of clusters for position analysis"),
  window_quarters: z
    .number()
    .min(1)
    .max(12)
    .optional()
    .default(4)
    .describe("Quarters for momentum calculation"),
  lookback_days: z
    .number()
    .min(1)
    .max(365)
    .optional()
    .default(45)
    .describe("Days to look back for alerts"),
  min_buyers: z
    .number()
    .min(2)
    .optional()
    .default(3)
    .describe("Minimum buyers for coordinated buying alert"),
  min_weight: z
    .number()
    .min(0)
    .max(1)
    .optional()
    .default(0.05)
    .describe("Minimum portfolio weight for conviction picks"),
});

type InstitutionalHoldingsInput = z.infer<typeof InstitutionalHoldingsArgs>;

async function execute(
  args: InstitutionalHoldingsInput,
  ctx: ToolContext
): Promise<string> {
  const {
    action,
    symbol,
    conviction_threshold,
    min_position_change,
    min_aum,
    n_clusters,
    window_quarters,
    lookback_days,
    min_buyers,
    min_weight,
  } = args;

  // Symbol is required for most actions
  const symbolRequired = [
    "holdings",
    "ownership",
    "changes",
    "whales",
    "sentiment",
    "clusters",
    "momentum",
    "smart-money-flow",
  ];
  if (symbolRequired.includes(action) && !symbol) {
    return `Error: Symbol is required for ${action} action.`;
  }

  const upperSymbol = symbol?.toUpperCase();

  try {
    switch (action) {
      case "holdings": {
        const response = await stanleyFetch<InstitutionalHolding[]>(
          `/api/institutional/${upperSymbol}`,
          { signal: ctx.abort }
        );

        if (!response.success || !response.data) {
          return `Institutional holdings failed: ${response.error || "No data returned"}`;
        }

        return formatHoldings(upperSymbol!, response.data);
      }

      case "ownership": {
        const response = await stanleyFetch<OwnershipBreakdown>(
          `/api/institutional/${upperSymbol}/ownership`,
          { signal: ctx.abort }
        );

        if (!response.success || !response.data) {
          return `Ownership breakdown failed: ${response.error || "No data returned"}`;
        }

        return formatOwnership(response.data);
      }

      case "changes": {
        const query = buildQueryString({ conviction_threshold });
        const response = await stanleyFetch<
          Array<{
            managerName: string;
            changeType: string;
            sharesCurrent: number;
            sharesPrevious: number;
            changeMagnitude: number;
            changePercentage: number;
            convictionScore: number;
          }>
        >(`/api/institutional/${upperSymbol}/changes${query}`, {
          signal: ctx.abort,
        });

        if (!response.success || !response.data) {
          return `Position changes failed: ${response.error || "No data returned"}`;
        }

        return formatChanges(upperSymbol!, response.data);
      }

      case "whales": {
        const query = buildQueryString({
          min_position_change,
          min_aum,
        });
        const response = await stanleyFetch<
          Array<{
            institutionName: string;
            changeType: string;
            magnitude: number;
            estimatedAum: number;
            alertLevel: string;
            sharesChanged: number;
          }>
        >(`/api/institutional/${upperSymbol}/whales${query}`, {
          signal: ctx.abort,
        });

        if (!response.success || !response.data) {
          return `Whale detection failed: ${response.error || "No data returned"}`;
        }

        return formatWhales(upperSymbol!, response.data);
      }

      case "sentiment": {
        const response = await stanleyFetch<InstitutionalSentiment>(
          `/api/institutional/${upperSymbol}/sentiment`,
          { signal: ctx.abort }
        );

        if (!response.success || !response.data) {
          return `Sentiment scoring failed: ${response.error || "No data returned"}`;
        }

        return formatSentiment(response.data);
      }

      case "clusters": {
        const query = buildQueryString({ n_clusters });
        const response = await stanleyFetch<{
          symbol: string;
          clusterLabels: Array<{
            managerName: string;
            cluster: number;
            clusterName: string;
          }>;
          clusterStats: Record<string, unknown>;
          smartMoneyDirection: string;
          clusterSummary: string;
        }>(`/api/institutional/${upperSymbol}/clusters${query}`, {
          signal: ctx.abort,
        });

        if (!response.success || !response.data) {
          return `Position clustering failed: ${response.error || "No data returned"}`;
        }

        return formatClusters(response.data);
      }

      case "momentum": {
        const query = buildQueryString({
          window_quarters,
          weight_by_performance: true,
        });
        const response = await stanleyFetch<{
          symbol: string;
          momentumScore: number;
          trendDirection: string;
          acceleration: number;
          quarterlyMomentum: Record<string, number>;
          topMovers: Array<Record<string, unknown>>;
          signalStrength: number;
          windowQuarters: number;
        }>(`/api/institutional/${upperSymbol}/momentum${query}`, {
          signal: ctx.abort,
        });

        if (!response.success || !response.data) {
          return `Momentum tracking failed: ${response.error || "No data returned"}`;
        }

        return formatMomentum(response.data);
      }

      case "smart-money-flow": {
        const response = await stanleyFetch<SmartMoneyFlow>(
          `/api/institutional/${upperSymbol}/smart-money-flow`,
          { signal: ctx.abort }
        );

        if (!response.success || !response.data) {
          return `Smart money flow failed: ${response.error || "No data returned"}`;
        }

        return formatSmartMoneyFlow(response.data);
      }

      case "new-positions": {
        const query = buildQueryString({
          lookback_days,
          min_value: 1e7,
        });
        const response = await stanleyFetch<
          Array<{
            symbol: string;
            managerName: string;
            managerCik: string;
            positionValue: number;
            shares: number;
            weight: number;
            managerPerformanceScore: number;
            alertType: string;
            significance: number;
          }>
        >(`/api/institutional/alerts/new-positions${query}`, {
          signal: ctx.abort,
        });

        if (!response.success || !response.data) {
          return `New positions alert failed: ${response.error || "No data returned"}`;
        }

        return formatNewPositions(response.data);
      }

      case "coordinated-buying": {
        const query = buildQueryString({
          min_buyers,
          lookback_days,
        });
        const response = await stanleyFetch<
          Array<{
            symbol: string;
            buyersCount: number;
            totalValueAdded: number;
            avgBuyerPerformance: number;
            signalStrength: number;
            buyers: string[];
          }>
        >(`/api/institutional/alerts/coordinated-buying${query}`, {
          signal: ctx.abort,
        });

        if (!response.success || !response.data) {
          return `Coordinated buying alert failed: ${response.error || "No data returned"}`;
        }

        return formatCoordinatedBuying(response.data);
      }

      case "conviction-picks": {
        const query = buildQueryString({
          min_weight,
          top_n_managers: 50,
        });
        const response = await stanleyFetch<
          Array<{
            symbol: string;
            holders: string[];
            avgWeight: number;
            maxWeight: number;
            holderCount: number;
            totalValue: number;
            avgManagerScore: number;
          }>
        >(`/api/institutional/conviction-picks${query}`, {
          signal: ctx.abort,
        });

        if (!response.success || !response.data) {
          return `Conviction picks failed: ${response.error || "No data returned"}`;
        }

        return formatConvictionPicks(response.data);
      }

      default:
        return `Unknown action: ${action}`;
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return `Institutional analysis error: ${message}`;
  }
}

function formatHoldings(
  symbol: string,
  data: InstitutionalHolding[]
): string {
  let output = `## Institutional Holdings: ${symbol}\n\n`;

  if (data.length === 0) {
    return output + "No institutional holdings data available.\n";
  }

  output += `| Manager | Shares | Value | Ownership |\n`;
  output += `|---------|--------|-------|----------|\n`;

  for (const h of data.slice(0, 15)) {
    const value =
      h.valueHeld >= 1e9
        ? `$${(h.valueHeld / 1e9).toFixed(2)}B`
        : `$${(h.valueHeld / 1e6).toFixed(1)}M`;
    output += `| ${h.managerName.substring(0, 30)} | ${h.sharesHeld.toLocaleString()} | ${value} | ${h.ownershipPercentage.toFixed(2)}% |\n`;
  }

  if (data.length > 15) {
    output += `\n*Showing top 15 of ${data.length} institutional holders.*\n`;
  }

  return output;
}

function formatOwnership(data: OwnershipBreakdown): string {
  let output = `## Ownership Breakdown: ${data.symbol}\n\n`;

  output += `### Ownership Structure\n`;
  output += `- **Institutional:** ${data.institutionalOwnership.toFixed(1)}%\n`;
  output += `- **Retail:** ${data.retailOwnership.toFixed(1)}%\n`;
  output += `- **Insider:** ${data.insiderOwnership.toFixed(1)}%\n\n`;

  output += `### Concentration\n`;
  output += `- **Top 10 Holders:** ${data.top10Concentration.toFixed(1)}% of shares\n`;
  output += `- **Total Holders:** ${data.totalHolders.toLocaleString()}\n`;
  output += `- **Shares Outstanding:** ${(data.sharesOutstanding / 1e6).toFixed(1)}M\n`;
  output += `- **Float Shares:** ${(data.floatShares / 1e6).toFixed(1)}M\n`;

  return output;
}

function formatChanges(
  symbol: string,
  data: Array<{
    managerName: string;
    changeType: string;
    sharesCurrent: number;
    sharesPrevious: number;
    changeMagnitude: number;
    changePercentage: number;
    convictionScore: number;
  }>
): string {
  let output = `## 13F Position Changes: ${symbol}\n\n`;

  if (data.length === 0) {
    return output + "No significant position changes detected.\n";
  }

  // Group by change type
  const newPositions = data.filter((d) => d.changeType === "new");
  const increases = data.filter((d) => d.changeType === "increase");
  const decreases = data.filter((d) => d.changeType === "decrease");
  const closed = data.filter((d) => d.changeType === "closed");

  if (newPositions.length > 0) {
    output += `### New Positions (${newPositions.length})\n`;
    for (const p of newPositions.slice(0, 5)) {
      output += `- ${p.managerName}: ${p.sharesCurrent.toLocaleString()} shares (conviction: ${p.convictionScore.toFixed(2)})\n`;
    }
    output += `\n`;
  }

  if (increases.length > 0) {
    output += `### Increases (${increases.length})\n`;
    for (const p of increases.slice(0, 5)) {
      output += `- ${p.managerName}: +${p.changeMagnitude.toLocaleString()} shares (+${(p.changePercentage * 100).toFixed(1)}%)\n`;
    }
    output += `\n`;
  }

  if (decreases.length > 0) {
    output += `### Decreases (${decreases.length})\n`;
    for (const p of decreases.slice(0, 5)) {
      output += `- ${p.managerName}: ${p.changeMagnitude.toLocaleString()} shares (${(p.changePercentage * 100).toFixed(1)}%)\n`;
    }
    output += `\n`;
  }

  if (closed.length > 0) {
    output += `### Closed Positions (${closed.length})\n`;
    for (const p of closed.slice(0, 5)) {
      output += `- ${p.managerName}: sold ${Math.abs(p.sharesPrevious).toLocaleString()} shares\n`;
    }
  }

  return output;
}

function formatWhales(
  symbol: string,
  data: Array<{
    institutionName: string;
    changeType: string;
    magnitude: number;
    estimatedAum: number;
    alertLevel: string;
    sharesChanged: number;
  }>
): string {
  let output = `## Whale Activity: ${symbol}\n\n`;

  if (data.length === 0) {
    return output + "No significant whale activity detected.\n";
  }

  for (const whale of data) {
    const aum = `$${(whale.estimatedAum / 1e9).toFixed(1)}B AUM`;
    const alertEmoji =
      whale.alertLevel === "high"
        ? "[HIGH]"
        : whale.alertLevel === "medium"
          ? "[MED]"
          : "[LOW]";

    output += `### ${alertEmoji} ${whale.institutionName}\n`;
    output += `- **Action:** ${whale.changeType}\n`;
    output += `- **Shares Changed:** ${whale.sharesChanged.toLocaleString()}\n`;
    output += `- **Magnitude:** ${(whale.magnitude * 100).toFixed(1)}%\n`;
    output += `- **Manager AUM:** ${aum}\n\n`;
  }

  return output;
}

function formatSentiment(data: InstitutionalSentiment): string {
  let output = `## Institutional Sentiment: ${data.symbol}\n\n`;

  output += `### Sentiment Score\n`;
  output += `**Score:** ${data.score.toFixed(2)} (${data.classification})\n`;
  output += `**Confidence:** ${(data.confidence * 100).toFixed(0)}%\n\n`;

  output += `### Contributing Factors\n`;
  for (const [factor, value] of Object.entries(data.contributingFactors)) {
    const sign = value >= 0 ? "+" : "";
    output += `- ${factor}: ${sign}${value.toFixed(3)}\n`;
  }

  return output;
}

function formatClusters(data: {
  symbol: string;
  clusterLabels: Array<{
    managerName: string;
    cluster: number;
    clusterName: string;
  }>;
  smartMoneyDirection: string;
  clusterSummary: string;
}): string {
  let output = `## Position Clustering: ${data.symbol}\n\n`;

  output += `**Smart Money Direction:** ${data.smartMoneyDirection}\n`;
  output += `**Summary:** ${data.clusterSummary}\n\n`;

  // Group by cluster
  const clusters = new Map<string, string[]>();
  for (const label of data.clusterLabels) {
    const name = label.clusterName;
    if (!clusters.has(name)) {
      clusters.set(name, []);
    }
    clusters.get(name)!.push(label.managerName);
  }

  for (const [clusterName, managers] of clusters) {
    output += `### ${clusterName}\n`;
    for (const m of managers.slice(0, 5)) {
      output += `- ${m}\n`;
    }
    if (managers.length > 5) {
      output += `- *...and ${managers.length - 5} more*\n`;
    }
    output += `\n`;
  }

  return output;
}

function formatMomentum(data: {
  symbol: string;
  momentumScore: number;
  trendDirection: string;
  acceleration: number;
  signalStrength: number;
  windowQuarters: number;
}): string {
  let output = `## Smart Money Momentum: ${data.symbol}\n\n`;

  output += `**Momentum Score:** ${data.momentumScore.toFixed(2)}\n`;
  output += `**Trend Direction:** ${data.trendDirection}\n`;
  output += `**Acceleration:** ${data.acceleration >= 0 ? "+" : ""}${data.acceleration.toFixed(3)}\n`;
  output += `**Signal Strength:** ${(data.signalStrength * 100).toFixed(0)}%\n`;
  output += `**Window:** ${data.windowQuarters} quarters\n`;

  return output;
}

function formatSmartMoneyFlow(data: SmartMoneyFlow): string {
  let output = `## Smart Money Flow: ${data.symbol}\n\n`;

  output += `### Flow Summary\n`;
  output += `**Net Flow:** ${data.netFlow >= 0 ? "+" : ""}${data.netFlow.toFixed(2)}\n`;
  output += `**Weighted Flow:** ${data.weightedFlow >= 0 ? "+" : ""}${data.weightedFlow.toFixed(2)}\n`;
  output += `**Signal:** ${data.signal.toUpperCase()} (${(data.signalStrength * 100).toFixed(0)}% strength)\n\n`;

  output += `### Activity\n`;
  output += `- **Buyers:** ${data.buyersCount}\n`;
  output += `- **Sellers:** ${data.sellersCount}\n`;
  output += `- **Coordinated Buying:** ${data.coordinatedBuying ? "Yes" : "No"}\n`;
  output += `- **Coordinated Selling:** ${data.coordinatedSelling ? "Yes" : "No"}\n`;

  return output;
}

function formatNewPositions(
  data: Array<{
    symbol: string;
    managerName: string;
    positionValue: number;
    shares: number;
    weight: number;
    managerPerformanceScore: number;
    significance: number;
  }>
): string {
  let output = `## New Position Alerts\n\n`;

  if (data.length === 0) {
    return output + "No new position alerts.\n";
  }

  output += `| Symbol | Manager | Value | Weight | Score |\n`;
  output += `|--------|---------|-------|--------|-------|\n`;

  for (const p of data.slice(0, 15)) {
    const value = `$${(p.positionValue / 1e6).toFixed(1)}M`;
    output += `| ${p.symbol} | ${p.managerName.substring(0, 20)} | ${value} | ${(p.weight * 100).toFixed(1)}% | ${p.managerPerformanceScore.toFixed(2)} |\n`;
  }

  return output;
}

function formatCoordinatedBuying(
  data: Array<{
    symbol: string;
    buyersCount: number;
    totalValueAdded: number;
    avgBuyerPerformance: number;
    signalStrength: number;
    buyers: string[];
  }>
): string {
  let output = `## Coordinated Buying Alerts\n\n`;

  if (data.length === 0) {
    return output + "No coordinated buying detected.\n";
  }

  for (const alert of data.slice(0, 10)) {
    const value = `$${(alert.totalValueAdded / 1e6).toFixed(1)}M`;
    output += `### ${alert.symbol}\n`;
    output += `- **Buyers:** ${alert.buyersCount}\n`;
    output += `- **Total Value Added:** ${value}\n`;
    output += `- **Signal Strength:** ${(alert.signalStrength * 100).toFixed(0)}%\n`;
    output += `- **Managers:** ${alert.buyers.slice(0, 3).join(", ")}${alert.buyers.length > 3 ? "..." : ""}\n\n`;
  }

  return output;
}

function formatConvictionPicks(
  data: Array<{
    symbol: string;
    avgWeight: number;
    maxWeight: number;
    holderCount: number;
    totalValue: number;
    avgManagerScore: number;
  }>
): string {
  let output = `## High Conviction Picks\n\n`;

  if (data.length === 0) {
    return output + "No conviction picks found.\n";
  }

  output += `| Symbol | Avg Weight | Max Weight | Holders | Manager Score |\n`;
  output += `|--------|------------|------------|---------|---------------|\n`;

  for (const pick of data.slice(0, 15)) {
    output += `| ${pick.symbol} | ${(pick.avgWeight * 100).toFixed(1)}% | ${(pick.maxWeight * 100).toFixed(1)}% | ${pick.holderCount} | ${pick.avgManagerScore.toFixed(2)} |\n`;
  }

  return output;
}

export const institutionalHoldings = {
  description: TOOL_DESCRIPTION,
  args: InstitutionalHoldingsArgs,
  execute,
};
