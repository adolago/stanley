/**
 * Macro Indicators Tool
 *
 * MCP tool for macroeconomic analysis including economic indicators,
 * regime detection, yield curve analysis, and recession probability.
 */

import { z } from "zod";
import { stanleyFetch, buildQueryString } from "./api-client";
import type {
  ToolContext,
  EconomicIndicator,
  CountrySnapshot,
  RegimeState,
  YieldCurve,
  StanleyApiResponse,
} from "./types";

const TOOL_DESCRIPTION = `Access macroeconomic indicators and regime analysis.

This tool provides comprehensive macro analytics including:
- Economic indicators (GDP, inflation, unemployment, policy rates)
- Market regime detection (goldilocks, stagflation, risk-on/off)
- Yield curve analysis and recession signals
- Fed policy expectations and rate probabilities
- Cross-asset correlations and risk regimes
- Global economic overview

Use this tool when:
- A user asks about economic conditions or indicators
- Market regime classification is needed
- Yield curve analysis or recession probability
- Fed policy expectations or rate forecasts
- Cross-asset correlation regime analysis

Supported countries: USA, DEU, GBR, JPN, CHN, FRA, ITA, CAN, AUS, KOR`;

// Input schema for macro analysis
const MacroIndicatorsArgs = z.object({
  action: z
    .enum([
      "indicators",
      "regime",
      "yield-curve",
      "recession",
      "fed-watch",
      "cross-asset",
      "global",
      "compare",
    ])
    .describe(
      "Type of analysis: 'indicators' for economic data, 'regime' for market regime, 'yield-curve' for curve analysis, 'recession' for probability model, 'fed-watch' for policy expectations, 'cross-asset' for correlations, 'global' for overview, 'compare' for country comparison"
    ),
  country: z
    .string()
    .optional()
    .default("USA")
    .describe("ISO country code (e.g., USA, DEU, JPN)"),
  countries: z
    .string()
    .optional()
    .describe("Comma-separated country codes for comparison (e.g., 'USA,DEU,JPN')"),
  correlation_window: z
    .number()
    .min(20)
    .max(252)
    .optional()
    .default(60)
    .describe("Rolling window for correlations (days)"),
  lookback_days: z
    .number()
    .min(60)
    .max(756)
    .optional()
    .default(252)
    .describe("Historical days for analysis"),
  include_snapshot: z
    .boolean()
    .optional()
    .default(true)
    .describe("Include economic snapshot in indicators response"),
});

type MacroIndicatorsInput = z.infer<typeof MacroIndicatorsArgs>;

async function execute(
  args: MacroIndicatorsInput,
  ctx: ToolContext
): Promise<string> {
  const {
    action,
    country,
    countries,
    correlation_window,
    lookback_days,
    include_snapshot,
  } = args;

  try {
    switch (action) {
      case "indicators": {
        const query = buildQueryString({ country, include_snapshot });
        const response = await stanleyFetch<{
          country: string;
          indicators: EconomicIndicator[];
          snapshot?: CountrySnapshot;
          timestamp: string;
        }>(`/api/macro/indicators${query}`, { signal: ctx.abort });

        if (!response.success || !response.data) {
          return `Economic indicators failed: ${response.error || "No data returned"}`;
        }

        return formatIndicators(response.data);
      }

      case "regime": {
        const query = buildQueryString({ country });
        const response = await stanleyFetch<RegimeState>(
          `/api/macro/regime${query}`,
          { signal: ctx.abort }
        );

        if (!response.success || !response.data) {
          return `Regime detection failed: ${response.error || "No data returned"}`;
        }

        return formatRegimeState(response.data);
      }

      case "yield-curve": {
        const query = buildQueryString({ country });
        const response = await stanleyFetch<YieldCurve>(
          `/api/macro/yield-curve${query}`,
          { signal: ctx.abort }
        );

        if (!response.success || !response.data) {
          return `Yield curve analysis failed: ${response.error || "No data returned"}`;
        }

        return formatYieldCurve(response.data);
      }

      case "recession": {
        const query = buildQueryString({ country });
        const response = await stanleyFetch<{
          country: string;
          probability12m: number;
          probability6m: number;
          riskLevel: string;
          riskScore: number;
          factors: Array<{
            factor: string;
            severity: string;
            description: string;
            contribution: number;
          }>;
          modelVersion: string;
          confidence: number;
          timestamp: string;
        }>(`/api/macro/recession-probability${query}`, { signal: ctx.abort });

        if (!response.success || !response.data) {
          return `Recession probability failed: ${response.error || "No data returned"}`;
        }

        return formatRecessionProbability(response.data);
      }

      case "fed-watch": {
        const response = await stanleyFetch<{
          currentRate: number;
          targetRange: string;
          nextMeeting: {
            date: string;
            probNoChange: number;
            probHike25bp: number;
            probCut25bp: number;
            probCut50bp: number;
            impliedRate: number;
          };
          upcomingMeetings: Array<{
            date: string;
            probNoChange: number;
            probCut25bp: number;
            probCut50bp: number;
            impliedRate: number;
          }>;
          terminalRate: number;
          marketExpectation: string;
          timestamp: string;
        }>("/api/macro/fed-watch", { signal: ctx.abort });

        if (!response.success || !response.data) {
          return `Fed watch failed: ${response.error || "No data returned"}`;
        }

        return formatFedWatch(response.data);
      }

      case "cross-asset": {
        const query = buildQueryString({ correlation_window, lookback_days });
        const response = await stanleyFetch<{
          regime: string;
          stockBondCorrelation: number;
          stockCommodityCorrelation: number;
          usdRiskCorrelation: number;
          riskOnOffScore: number;
          correlationStability: number;
          confidence: number;
          assetMomentum: Record<string, number>;
          correlationMatrix: Record<string, Record<string, number>>;
          timestamp: string;
        }>(`/api/macro/cross-asset${query}`, { signal: ctx.abort });

        if (!response.success || !response.data) {
          return `Cross-asset analysis failed: ${response.error || "No data returned"}`;
        }

        return formatCrossAsset(response.data);
      }

      case "global": {
        const response = await stanleyFetch<{
          globalGrowth: number | null;
          globalInflation: number | null;
          regions: Record<
            string,
            {
              avgGrowth: number | null;
              avgInflation: number | null;
              countries: Record<
                string,
                {
                  gdpGrowth: number | null;
                  inflation: number | null;
                  regime: string | null;
                }
              >;
            }
          >;
          timestamp: string;
        }>("/api/macro/global-overview", { signal: ctx.abort });

        if (!response.success || !response.data) {
          return `Global overview failed: ${response.error || "No data returned"}`;
        }

        return formatGlobalOverview(response.data);
      }

      case "compare": {
        const countryList = countries || "USA,DEU,JPN,GBR,CHN";
        const query = buildQueryString({ countries: countryList });
        const response = await stanleyFetch<{
          countries: string[];
          comparison: Record<
            string,
            {
              gdpGrowth: number | null;
              inflation: number | null;
              unemployment: number | null;
              policyRate: number | null;
              currentAccount: number | null;
              regime: string | null;
            }
          >;
          timestamp: string;
        }>(`/api/macro/compare-countries${query}`, { signal: ctx.abort });

        if (!response.success || !response.data) {
          return `Country comparison failed: ${response.error || "No data returned"}`;
        }

        return formatCountryComparison(response.data);
      }

      default:
        return `Unknown action: ${action}`;
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return `Macro indicators error: ${message}`;
  }
}

function formatIndicators(data: {
  country: string;
  indicators: EconomicIndicator[];
  snapshot?: CountrySnapshot;
  timestamp: string;
}): string {
  let output = `## Economic Indicators: ${data.country}\n\n`;

  // Snapshot
  if (data.snapshot) {
    const s = data.snapshot;
    output += `### Economic Snapshot\n`;
    if (s.gdpGrowth !== null && s.gdpGrowth !== undefined)
      output += `- **GDP Growth:** ${s.gdpGrowth.toFixed(1)}%\n`;
    if (s.inflation !== null && s.inflation !== undefined)
      output += `- **Inflation:** ${s.inflation.toFixed(1)}%\n`;
    if (s.unemployment !== null && s.unemployment !== undefined)
      output += `- **Unemployment:** ${s.unemployment.toFixed(1)}%\n`;
    if (s.policyRate !== null && s.policyRate !== undefined)
      output += `- **Policy Rate:** ${s.policyRate.toFixed(2)}%\n`;
    if (s.currentAccount !== null && s.currentAccount !== undefined)
      output += `- **Current Account:** ${s.currentAccount.toFixed(1)}% of GDP\n`;
    if (s.regime) output += `- **Economic Regime:** ${s.regime}\n`;
    output += `\n`;
  }

  // Indicators Table
  if (data.indicators.length > 0) {
    output += `### Key Indicators\n`;
    output += `| Indicator | Value | Unit | Frequency |\n`;
    output += `|-----------|-------|------|----------|\n`;

    for (const ind of data.indicators) {
      output += `| ${ind.name} | ${ind.value.toFixed(2)} | ${ind.unit} | ${ind.frequency} |\n`;
    }
  }

  return output;
}

function formatRegimeState(data: RegimeState): string {
  let output = `## Market Regime Analysis\n\n`;

  // Current Regime
  output += `### Current Regime\n`;
  output += `**Regime:** ${data.currentRegime.toUpperCase()}\n`;
  output += `**Confidence:** ${data.confidence}\n`;
  output += `**Score:** ${data.regimeScore.toFixed(2)}\n`;
  output += `**Duration:** ${data.regimeDurationDays} days\n\n`;

  // Positioning
  output += `### Recommended Positioning\n`;
  output += `- **Equity:** ${data.positioning.equity}\n`;
  output += `- **Duration:** ${data.positioning.duration}\n`;
  output += `- **Credit:** ${data.positioning.credit}\n`;
  output += `- **Volatility:** ${data.positioning.volatility}\n\n`;

  // Components
  output += `### Regime Components\n`;
  for (const [component, status] of Object.entries(data.components)) {
    output += `- **${component}:** ${status}\n`;
  }
  output += `\n`;

  // Key Signals
  if (data.signals.length > 0) {
    output += `### Key Signals\n`;
    for (const signal of data.signals.slice(0, 5)) {
      output += `- ${signal.source}: ${signal.signal} (${signal.strength.toFixed(2)} strength, ${(signal.confidence * 100).toFixed(0)}% confidence)\n`;
    }
  }

  return output;
}

function formatYieldCurve(data: YieldCurve): string {
  let output = `## Yield Curve Analysis: ${data.country}\n\n`;

  // Shape and Spreads
  output += `### Curve Shape\n`;
  output += `**Shape:** ${data.shape.charAt(0).toUpperCase() + data.shape.slice(1)}\n`;
  output += `**Dynamic:** ${data.dynamic}\n`;
  if (data.spread2y10y !== null && data.spread2y10y !== undefined)
    output += `**2Y-10Y Spread:** ${data.spread2y10y.toFixed(2)}%\n`;
  if (data.spread3m10y !== null && data.spread3m10y !== undefined)
    output += `**3M-10Y Spread:** ${data.spread3m10y.toFixed(2)}%\n`;
  output += `\n`;

  // Recession Signal
  output += `### Recession Signal\n`;
  output += `**Signal Strength:** ${data.recessionSignal}\n`;
  output += `**12-Month Probability:** ${(data.recessionProbability12m * 100).toFixed(1)}%\n`;
  if (data.inversionDurationDays > 0) {
    output += `**Inversion Duration:** ${data.inversionDurationDays} days\n`;
  }
  output += `\n`;

  // Curve Points
  if (data.curve.length > 0) {
    output += `### Term Structure\n`;
    output += `| Tenor | Yield | Change |\n`;
    output += `|-------|-------|--------|\n`;
    for (const point of data.curve) {
      const change =
        point.change !== null && point.change !== undefined
          ? `${point.change >= 0 ? "+" : ""}${point.change.toFixed(3)}%`
          : "N/A";
      output += `| ${point.tenor} | ${point.yield.toFixed(3)}% | ${change} |\n`;
    }
  }

  return output;
}

function formatRecessionProbability(data: {
  country: string;
  probability12m: number;
  probability6m: number;
  riskLevel: string;
  riskScore: number;
  factors: Array<{
    factor: string;
    severity: string;
    description: string;
    contribution: number;
  }>;
  modelVersion: string;
  confidence: number;
  timestamp: string;
}): string {
  let output = `## Recession Probability Model: ${data.country}\n\n`;

  // Probabilities
  output += `### Probability Assessment\n`;
  output += `**6-Month Probability:** ${(data.probability6m * 100).toFixed(1)}%\n`;
  output += `**12-Month Probability:** ${(data.probability12m * 100).toFixed(1)}%\n`;
  output += `**Risk Level:** ${data.riskLevel.toUpperCase()}\n`;
  output += `**Risk Score:** ${data.riskScore}/100\n`;
  output += `**Model Confidence:** ${(data.confidence * 100).toFixed(0)}%\n\n`;

  // Risk Factors
  if (data.factors.length > 0) {
    output += `### Risk Factors\n`;
    for (const factor of data.factors) {
      const severity =
        factor.severity === "high"
          ? "[HIGH]"
          : factor.severity === "medium"
            ? "[MEDIUM]"
            : "[LOW]";
      output += `- ${severity} **${factor.factor}:** ${factor.description}\n`;
    }
  }

  return output;
}

function formatFedWatch(data: {
  currentRate: number;
  targetRange: string;
  nextMeeting: {
    date: string;
    probNoChange: number;
    probHike25bp: number;
    probCut25bp: number;
    probCut50bp: number;
    impliedRate: number;
  };
  upcomingMeetings: Array<{
    date: string;
    probNoChange: number;
    probCut25bp: number;
    probCut50bp: number;
    impliedRate: number;
  }>;
  terminalRate: number;
  marketExpectation: string;
  timestamp: string;
}): string {
  let output = `## Fed Watch\n\n`;

  // Current State
  output += `### Current Policy\n`;
  output += `**Target Range:** ${data.targetRange}\n`;
  output += `**Current Rate:** ${data.currentRate.toFixed(2)}%\n`;
  output += `**Market Expectation:** ${data.marketExpectation.toUpperCase()}\n`;
  output += `**Terminal Rate:** ${data.terminalRate.toFixed(2)}%\n\n`;

  // Next Meeting
  if (data.nextMeeting) {
    const m = data.nextMeeting;
    output += `### Next Meeting (${m.date})\n`;
    output += `- **No Change:** ${(m.probNoChange * 100).toFixed(1)}%\n`;
    output += `- **25bp Cut:** ${(m.probCut25bp * 100).toFixed(1)}%\n`;
    output += `- **50bp Cut:** ${(m.probCut50bp * 100).toFixed(1)}%\n`;
    output += `- **25bp Hike:** ${(m.probHike25bp * 100).toFixed(1)}%\n`;
    output += `- **Implied Rate:** ${m.impliedRate.toFixed(2)}%\n\n`;
  }

  // Upcoming Meetings
  if (data.upcomingMeetings.length > 1) {
    output += `### Rate Path Expectations\n`;
    output += `| Meeting | Implied Rate | Cut Probability |\n`;
    output += `|---------|--------------|----------------|\n`;
    for (const m of data.upcomingMeetings.slice(0, 6)) {
      const cutProb = (m.probCut25bp + m.probCut50bp) * 100;
      output += `| ${m.date} | ${m.impliedRate.toFixed(2)}% | ${cutProb.toFixed(0)}% |\n`;
    }
  }

  return output;
}

function formatCrossAsset(data: {
  regime: string;
  stockBondCorrelation: number;
  stockCommodityCorrelation: number;
  usdRiskCorrelation: number;
  riskOnOffScore: number;
  correlationStability: number;
  confidence: number;
  assetMomentum: Record<string, number>;
  timestamp: string;
}): string {
  let output = `## Cross-Asset Correlation Analysis\n\n`;

  // Regime
  output += `### Correlation Regime\n`;
  output += `**Regime:** ${data.regime}\n`;
  output += `**Risk-On/Off Score:** ${data.riskOnOffScore.toFixed(2)} (${data.riskOnOffScore > 0 ? "Risk-On" : "Risk-Off"})\n`;
  output += `**Stability:** ${data.correlationStability.toFixed(2)}\n`;
  output += `**Confidence:** ${(data.confidence * 100).toFixed(0)}%\n\n`;

  // Key Correlations
  output += `### Key Correlations\n`;
  output += `- **Stock-Bond:** ${data.stockBondCorrelation.toFixed(3)}\n`;
  output += `- **Stock-Commodity:** ${data.stockCommodityCorrelation.toFixed(3)}\n`;
  output += `- **USD-Risk:** ${data.usdRiskCorrelation.toFixed(3)}\n\n`;

  // Interpretation
  output += `### Interpretation\n`;
  if (data.stockBondCorrelation > 0.3) {
    output += `- Positive stock-bond correlation suggests inflation concerns or liquidity stress\n`;
  } else if (data.stockBondCorrelation < -0.3) {
    output += `- Negative stock-bond correlation indicates normal risk-off hedging behavior\n`;
  }

  // Asset Momentum
  if (Object.keys(data.assetMomentum).length > 0) {
    output += `\n### Asset Momentum\n`;
    const sorted = Object.entries(data.assetMomentum).sort((a, b) => b[1] - a[1]);
    for (const [asset, momentum] of sorted) {
      output += `- ${asset}: ${momentum >= 0 ? "+" : ""}${(momentum * 100).toFixed(2)}%\n`;
    }
  }

  return output;
}

function formatGlobalOverview(data: {
  globalGrowth: number | null;
  globalInflation: number | null;
  regions: Record<
    string,
    {
      avgGrowth: number | null;
      avgInflation: number | null;
      countries: Record<
        string,
        {
          gdpGrowth: number | null;
          inflation: number | null;
          regime: string | null;
        }
      >;
    }
  >;
  timestamp: string;
}): string {
  let output = `## Global Economic Overview\n\n`;

  // Global Aggregates
  output += `### Global Averages\n`;
  if (data.globalGrowth !== null)
    output += `- **Global Growth:** ${data.globalGrowth.toFixed(1)}%\n`;
  if (data.globalInflation !== null)
    output += `- **Global Inflation:** ${data.globalInflation.toFixed(1)}%\n`;
  output += `\n`;

  // Regional Breakdown
  for (const [region, rData] of Object.entries(data.regions)) {
    output += `### ${region}\n`;
    if (rData.avgGrowth !== null)
      output += `Average Growth: ${rData.avgGrowth.toFixed(1)}% | `;
    if (rData.avgInflation !== null)
      output += `Average Inflation: ${rData.avgInflation.toFixed(1)}%`;
    output += `\n`;

    for (const [country, cData] of Object.entries(rData.countries)) {
      const growth =
        cData.gdpGrowth !== null ? `${cData.gdpGrowth.toFixed(1)}%` : "N/A";
      const inflation =
        cData.inflation !== null ? `${cData.inflation.toFixed(1)}%` : "N/A";
      const regime = cData.regime || "N/A";
      output += `- ${country}: Growth ${growth}, Inflation ${inflation}, Regime: ${regime}\n`;
    }
    output += `\n`;
  }

  return output;
}

function formatCountryComparison(data: {
  countries: string[];
  comparison: Record<
    string,
    {
      gdpGrowth: number | null;
      inflation: number | null;
      unemployment: number | null;
      policyRate: number | null;
      currentAccount: number | null;
      regime: string | null;
    }
  >;
  timestamp: string;
}): string {
  let output = `## Country Comparison\n\n`;

  output += `| Country | GDP Growth | Inflation | Unemployment | Policy Rate | Regime |\n`;
  output += `|---------|------------|-----------|--------------|-------------|--------|\n`;

  for (const country of data.countries) {
    const c = data.comparison[country];
    if (!c) continue;

    const growth = c.gdpGrowth !== null ? `${c.gdpGrowth.toFixed(1)}%` : "N/A";
    const inflation =
      c.inflation !== null ? `${c.inflation.toFixed(1)}%` : "N/A";
    const unemployment =
      c.unemployment !== null ? `${c.unemployment.toFixed(1)}%` : "N/A";
    const rate = c.policyRate !== null ? `${c.policyRate.toFixed(2)}%` : "N/A";
    const regime = c.regime || "N/A";

    output += `| ${country} | ${growth} | ${inflation} | ${unemployment} | ${rate} | ${regime} |\n`;
  }

  return output;
}

export const macroIndicators = {
  description: TOOL_DESCRIPTION,
  args: MacroIndicatorsArgs,
  execute,
};
