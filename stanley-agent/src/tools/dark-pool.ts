/**
 * Dark Pool Tool
 *
 * MCP tool for dark pool activity analysis, institutional block trades,
 * and hidden liquidity tracking.
 */

import { z } from "zod";
import { stanleyFetch, buildQueryString } from "./api-client";
import type { ToolContext, StanleyApiResponse } from "./types";

const TOOL_DESCRIPTION = `Analyze dark pool activity and institutional block trades.

This tool provides dark pool analytics including:
- Dark pool volume as percentage of total trading
- Large block trade detection and analysis
- Dark pool sentiment signals (accumulation vs distribution)
- Historical dark pool activity trends
- Summary statistics for institutional positioning

Dark pools are private exchanges where institutional investors trade
large blocks of shares away from public exchanges. High dark pool
activity often signals institutional accumulation or distribution.

Key signals to watch:
- Dark pool % > 40%: Significant institutional activity
- Large block spikes: Potential position building/unwinding
- Sentiment shifts: Changes in buy/sell pressure in dark pools

Use this tool when:
- Analyzing institutional positioning in a stock
- Detecting potential accumulation before price moves
- Understanding hidden liquidity and large block activity
- Confirming technical signals with institutional flow data`;

// Input schema for dark pool analysis
const DarkPoolArgs = z.object({
  action: z
    .enum(["activity", "summary", "trend", "alerts"])
    .describe(
      "Type of analysis: 'activity' for daily dark pool data, 'summary' for aggregate statistics, 'trend' for trend analysis, 'alerts' for unusual activity alerts"
    ),
  symbol: z
    .string()
    .describe("Stock ticker symbol (e.g., AAPL, MSFT, NVDA)"),
  lookback_days: z
    .number()
    .min(1)
    .max(90)
    .optional()
    .default(20)
    .describe("Number of days to analyze (default: 20, max: 90)"),
});

type DarkPoolInput = z.infer<typeof DarkPoolArgs>;

// Response types
interface DarkPoolEntry {
  symbol: string;
  date: string | null;
  darkPoolVolume: number;
  totalVolume: number;
  darkPoolPercentage: number;
  largeBlockActivity: number;
  darkPoolSignal: number; // -1: bearish, 0: neutral, 1: bullish
}

interface DarkPoolResponse {
  symbol: string;
  data: DarkPoolEntry[];
  summary: {
    averageDarkPoolPercentage: number;
    averageBlockActivity: number;
    totalDarkPoolVolume: number;
    signalBias: number;
  };
  timestamp: string;
}

async function execute(
  args: DarkPoolInput,
  ctx: ToolContext
): Promise<string> {
  const { action, symbol, lookback_days } = args;

  if (!symbol) {
    return "Error: Symbol is required for dark pool analysis.";
  }

  const upperSymbol = symbol.toUpperCase();

  try {
    // All actions use the same base endpoint
    const query = buildQueryString({ lookback_days });
    const response = await stanleyFetch<DarkPoolResponse>(
      `/api/dark-pool/${upperSymbol}${query}`,
      { signal: ctx.abort }
    );

    if (!response.success || !response.data) {
      return `Dark pool analysis failed: ${response.error || "No data returned"}`;
    }

    const data = response.data;

    switch (action) {
      case "activity":
        return formatDarkPoolActivity(data);

      case "summary":
        return formatDarkPoolSummary(data);

      case "trend":
        return formatDarkPoolTrend(data);

      case "alerts":
        return formatDarkPoolAlerts(data);

      default:
        return `Unknown action: ${action}`;
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return `Dark pool analysis error: ${message}`;
  }
}

function formatDarkPoolActivity(data: DarkPoolResponse): string {
  let output = `## Dark Pool Activity: ${data.symbol}\n\n`;

  // Summary stats
  const summary = data.summary;
  output += `### Summary Statistics\n`;
  output += `- **Average Dark Pool %:** ${(summary.averageDarkPoolPercentage * 100).toFixed(2)}%\n`;
  output += `- **Total Dark Pool Volume:** ${summary.totalDarkPoolVolume.toLocaleString()}\n`;
  output += `- **Avg Block Activity:** ${(summary.averageBlockActivity * 100).toFixed(2)}%\n`;
  output += `- **Signal Bias:** ${formatSignalBias(summary.signalBias)}\n\n`;

  // Daily activity table
  if (data.data.length > 0) {
    output += `### Daily Dark Pool Activity\n`;
    output += `| Date | DP Volume | Total Volume | DP % | Block Activity | Signal |\n`;
    output += `|------|-----------|--------------|------|----------------|--------|\n`;

    // Show last 10 days
    const recentData = data.data.slice(0, 10);
    for (const entry of recentData) {
      const dateStr = entry.date ? entry.date.split("T")[0] : "N/A";
      const signal = formatSignal(entry.darkPoolSignal);
      output += `| ${dateStr} | ${formatVolume(entry.darkPoolVolume)} | ${formatVolume(entry.totalVolume)} | ${(entry.darkPoolPercentage * 100).toFixed(1)}% | ${(entry.largeBlockActivity * 100).toFixed(1)}% | ${signal} |\n`;
    }

    if (data.data.length > 10) {
      output += `\n*Showing 10 of ${data.data.length} days*\n`;
    }
  }

  return output;
}

function formatDarkPoolSummary(data: DarkPoolResponse): string {
  let output = `## Dark Pool Summary: ${data.symbol}\n\n`;

  const summary = data.summary;
  const entries = data.data;

  // Key metrics
  output += `### Key Metrics\n`;
  output += `- **Average Dark Pool %:** ${(summary.averageDarkPoolPercentage * 100).toFixed(2)}%\n`;
  output += `- **Total Dark Pool Volume:** ${summary.totalDarkPoolVolume.toLocaleString()}\n`;
  output += `- **Average Block Activity:** ${(summary.averageBlockActivity * 100).toFixed(2)}%\n`;
  output += `- **Net Signal Bias:** ${formatSignalBias(summary.signalBias)}\n\n`;

  // Statistical analysis
  if (entries.length > 0) {
    const dpPercentages = entries.map((e) => e.darkPoolPercentage);
    const maxDp = Math.max(...dpPercentages);
    const minDp = Math.min(...dpPercentages);
    const latestDp = dpPercentages[0];

    output += `### Statistical Range\n`;
    output += `- **Current DP %:** ${(latestDp * 100).toFixed(2)}%\n`;
    output += `- **Max DP % (${data.data.length}d):** ${(maxDp * 100).toFixed(2)}%\n`;
    output += `- **Min DP % (${data.data.length}d):** ${(minDp * 100).toFixed(2)}%\n`;
    output += `- **Range:** ${((maxDp - minDp) * 100).toFixed(2)}%\n\n`;

    // Signal distribution
    const bullish = entries.filter((e) => e.darkPoolSignal > 0).length;
    const bearish = entries.filter((e) => e.darkPoolSignal < 0).length;
    const neutral = entries.filter((e) => e.darkPoolSignal === 0).length;

    output += `### Signal Distribution\n`;
    output += `- **Bullish Days:** ${bullish} (${((bullish / entries.length) * 100).toFixed(0)}%)\n`;
    output += `- **Bearish Days:** ${bearish} (${((bearish / entries.length) * 100).toFixed(0)}%)\n`;
    output += `- **Neutral Days:** ${neutral} (${((neutral / entries.length) * 100).toFixed(0)}%)\n`;
  }

  // Interpretation
  output += `\n### Interpretation\n`;
  const avgDp = summary.averageDarkPoolPercentage;
  const bias = summary.signalBias;

  if (avgDp > 0.45) {
    output += `**High institutional activity** - Dark pool % above 45% indicates significant off-exchange trading.\n`;
  } else if (avgDp > 0.35) {
    output += `**Moderate institutional activity** - Normal range of dark pool trading.\n`;
  } else {
    output += `**Low institutional activity** - Limited dark pool engagement.\n`;
  }

  if (bias > 3) {
    output += `**Accumulation signal** - Net positive bias suggests institutional buying.\n`;
  } else if (bias < -3) {
    output += `**Distribution signal** - Net negative bias suggests institutional selling.\n`;
  } else {
    output += `**Mixed signals** - No clear directional bias in dark pool activity.\n`;
  }

  return output;
}

function formatDarkPoolTrend(data: DarkPoolResponse): string {
  let output = `## Dark Pool Trend Analysis: ${data.symbol}\n\n`;

  const entries = data.data;
  if (entries.length < 5) {
    return output + "Insufficient data for trend analysis (need at least 5 days).\n";
  }

  // Calculate trends
  const dpPercentages = entries.map((e) => e.darkPoolPercentage);
  const blockActivity = entries.map((e) => e.largeBlockActivity);
  const signals = entries.map((e) => e.darkPoolSignal);

  // Recent vs earlier comparison
  const recentDp = dpPercentages.slice(0, 5).reduce((a, b) => a + b, 0) / 5;
  const earlierDp = dpPercentages.slice(-5).reduce((a, b) => a + b, 0) / 5;
  const dpTrend = recentDp - earlierDp;

  const recentBlock = blockActivity.slice(0, 5).reduce((a, b) => a + b, 0) / 5;
  const earlierBlock = blockActivity.slice(-5).reduce((a, b) => a + b, 0) / 5;
  const blockTrend = recentBlock - earlierBlock;

  const recentSignal = signals.slice(0, 5).reduce((a, b) => a + b, 0);
  const earlierSignal = signals.slice(-5).reduce((a, b) => a + b, 0);

  output += `### Trend Indicators\n`;
  output += `| Metric | Recent (5d) | Earlier (5d) | Trend |\n`;
  output += `|--------|-------------|--------------|-------|\n`;
  output += `| Dark Pool % | ${(recentDp * 100).toFixed(1)}% | ${(earlierDp * 100).toFixed(1)}% | ${formatTrend(dpTrend * 100)} |\n`;
  output += `| Block Activity | ${(recentBlock * 100).toFixed(1)}% | ${(earlierBlock * 100).toFixed(1)}% | ${formatTrend(blockTrend * 100)} |\n`;
  output += `| Signal Sum | ${recentSignal} | ${earlierSignal} | ${recentSignal > earlierSignal ? "Improving" : recentSignal < earlierSignal ? "Weakening" : "Stable"} |\n\n`;

  // Trend interpretation
  output += `### Trend Interpretation\n`;

  if (dpTrend > 0.02 && blockTrend > 0) {
    output += `**Increasing institutional engagement** - Rising dark pool % with higher block activity suggests building positions.\n`;
  } else if (dpTrend < -0.02 && blockTrend < 0) {
    output += `**Decreasing institutional engagement** - Falling dark pool activity may indicate reduced institutional interest.\n`;
  } else if (dpTrend > 0.02 && recentSignal > 0) {
    output += `**Accumulation trend** - Increasing dark pool activity with bullish signals.\n`;
  } else if (dpTrend > 0.02 && recentSignal < 0) {
    output += `**Distribution trend** - Increasing dark pool activity with bearish signals.\n`;
  } else {
    output += `**Stable activity** - No significant trend in dark pool behavior.\n`;
  }

  // 5-day momentum
  output += `\n### 5-Day Momentum\n`;
  for (let i = 0; i < Math.min(5, entries.length); i++) {
    const entry = entries[i];
    const dateStr = entry.date ? entry.date.split("T")[0] : `Day ${i + 1}`;
    const bar = generateBar(entry.darkPoolPercentage, 0.5);
    const signal = formatSignal(entry.darkPoolSignal);
    output += `${dateStr}: ${bar} ${(entry.darkPoolPercentage * 100).toFixed(1)}% [${signal}]\n`;
  }

  return output;
}

function formatDarkPoolAlerts(data: DarkPoolResponse): string {
  let output = `## Dark Pool Alerts: ${data.symbol}\n\n`;

  const entries = data.data;
  const summary = data.summary;
  const alerts: Array<{ severity: string; type: string; message: string }> = [];

  // Check for unusual activity
  const avgDp = summary.averageDarkPoolPercentage;
  const latestDp = entries.length > 0 ? entries[0].darkPoolPercentage : 0;
  const latestBlock = entries.length > 0 ? entries[0].largeBlockActivity : 0;

  // High dark pool percentage alert
  if (latestDp > 0.5) {
    alerts.push({
      severity: "HIGH",
      type: "High Dark Pool %",
      message: `Current dark pool trading at ${(latestDp * 100).toFixed(1)}% (>50%) indicates exceptional institutional activity.`,
    });
  } else if (latestDp > 0.4) {
    alerts.push({
      severity: "MEDIUM",
      type: "Elevated Dark Pool %",
      message: `Dark pool trading at ${(latestDp * 100).toFixed(1)}% (>40%) shows significant institutional interest.`,
    });
  }

  // Large block activity alert
  if (latestBlock > 0.3) {
    alerts.push({
      severity: "HIGH",
      type: "Large Block Activity",
      message: `Block trade activity at ${(latestBlock * 100).toFixed(1)}% indicates large institutional orders.`,
    });
  }

  // Signal bias alerts
  if (summary.signalBias > 5) {
    alerts.push({
      severity: "MEDIUM",
      type: "Strong Accumulation Signal",
      message: `Net signal bias of +${summary.signalBias} suggests sustained institutional buying.`,
    });
  } else if (summary.signalBias < -5) {
    alerts.push({
      severity: "MEDIUM",
      type: "Strong Distribution Signal",
      message: `Net signal bias of ${summary.signalBias} suggests sustained institutional selling.`,
    });
  }

  // Trend change detection
  if (entries.length >= 5) {
    const recent3 = entries.slice(0, 3);
    const prior3 = entries.slice(3, 6);

    const recentAvg = recent3.reduce((sum, e) => sum + e.darkPoolPercentage, 0) / 3;
    const priorAvg = prior3.length > 0
      ? prior3.reduce((sum, e) => sum + e.darkPoolPercentage, 0) / prior3.length
      : recentAvg;

    const change = (recentAvg - priorAvg) / priorAvg;
    if (Math.abs(change) > 0.2) {
      alerts.push({
        severity: "MEDIUM",
        type: change > 0 ? "Dark Pool % Spike" : "Dark Pool % Drop",
        message: `${change > 0 ? "+" : ""}${(change * 100).toFixed(0)}% change in dark pool activity over recent days.`,
      });
    }
  }

  // Consecutive signal days
  if (entries.length >= 3) {
    const recent3Signals = entries.slice(0, 3).map((e) => e.darkPoolSignal);
    if (recent3Signals.every((s) => s > 0)) {
      alerts.push({
        severity: "LOW",
        type: "Consecutive Bullish Signals",
        message: "3 consecutive bullish dark pool signals detected.",
      });
    } else if (recent3Signals.every((s) => s < 0)) {
      alerts.push({
        severity: "LOW",
        type: "Consecutive Bearish Signals",
        message: "3 consecutive bearish dark pool signals detected.",
      });
    }
  }

  // Output alerts
  if (alerts.length === 0) {
    output += `No significant alerts detected. Dark pool activity is within normal ranges.\n\n`;
  } else {
    output += `### Active Alerts (${alerts.length})\n\n`;

    // Sort by severity
    const severityOrder = { HIGH: 0, MEDIUM: 1, LOW: 2 };
    alerts.sort((a, b) => severityOrder[a.severity as keyof typeof severityOrder] - severityOrder[b.severity as keyof typeof severityOrder]);

    for (const alert of alerts) {
      const icon =
        alert.severity === "HIGH"
          ? "[!!!]"
          : alert.severity === "MEDIUM"
            ? "[!!]"
            : "[!]";
      output += `${icon} **${alert.type}** (${alert.severity})\n`;
      output += `${alert.message}\n\n`;
    }
  }

  // Current status
  output += `### Current Status\n`;
  output += `- **Dark Pool %:** ${(latestDp * 100).toFixed(1)}%\n`;
  output += `- **Block Activity:** ${(latestBlock * 100).toFixed(1)}%\n`;
  output += `- **Period Average:** ${(avgDp * 100).toFixed(1)}%\n`;
  output += `- **Signal Bias:** ${formatSignalBias(summary.signalBias)}\n`;

  return output;
}

// Helper functions
function formatVolume(vol: number): string {
  if (vol >= 1e9) return `${(vol / 1e9).toFixed(1)}B`;
  if (vol >= 1e6) return `${(vol / 1e6).toFixed(1)}M`;
  if (vol >= 1e3) return `${(vol / 1e3).toFixed(1)}K`;
  return vol.toString();
}

function formatSignal(signal: number): string {
  if (signal > 0) return "Bullish";
  if (signal < 0) return "Bearish";
  return "Neutral";
}

function formatSignalBias(bias: number): string {
  if (bias > 3) return `+${bias} (Bullish)`;
  if (bias < -3) return `${bias} (Bearish)`;
  return `${bias} (Neutral)`;
}

function formatTrend(change: number): string {
  if (change > 1) return `+${change.toFixed(1)}% (Up)`;
  if (change < -1) return `${change.toFixed(1)}% (Down)`;
  return `${change.toFixed(1)}% (Flat)`;
}

function generateBar(value: number, max: number): string {
  const normalized = Math.min(value / max, 1);
  const filled = Math.round(normalized * 20);
  return "[" + "#".repeat(filled) + "-".repeat(20 - filled) + "]";
}

export const darkPool = {
  description: TOOL_DESCRIPTION,
  args: DarkPoolArgs,
  execute,
};
