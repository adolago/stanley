/**
 * Stanley MCP Tools - Shared Types
 */

import { z } from "zod";

// =============================================================================
// Tool Context
// =============================================================================

export type ToolContext = {
  sessionID: string;
  messageID: string;
  agent: string;
  abort: AbortSignal;
};

// =============================================================================
// API Response Types
// =============================================================================

export interface StanleyApiResponse<T = unknown> {
  success: boolean;
  data: T | null;
  error: string | null;
  timestamp: string;
}

// =============================================================================
// Portfolio Types
// =============================================================================

export const PortfolioHoldingSchema = z.object({
  symbol: z.string().describe("Stock ticker symbol (e.g., AAPL, MSFT)"),
  shares: z.number().positive().describe("Number of shares held"),
  averageCost: z.number().optional().describe("Average cost per share"),
});

export type PortfolioHolding = z.infer<typeof PortfolioHoldingSchema>;

export interface PortfolioAnalytics {
  totalValue: number;
  totalCost: number;
  totalReturn: number;
  totalReturnPercent: number;
  beta: number;
  alpha: number;
  sharpeRatio: number;
  sortinoRatio: number;
  var95: number;
  var99: number;
  var95Percent: number;
  var99Percent: number;
  volatility: number;
  maxDrawdown: number;
  sectorExposure: Record<string, number>;
  topHoldings: Array<{
    symbol: string;
    shares: number;
    averageCost: number;
    currentPrice: number;
    marketValue: number;
    weight: number;
  }>;
}

export interface RiskMetrics {
  var95: number;
  var99: number;
  var95Percent: number;
  var99Percent: number;
  cvar95: number;
  cvar99: number;
  maxDrawdown: number;
  volatility: number;
  sharpeRatio: number;
  sortinoRatio: number;
  beta: number;
  method: string;
  lookbackDays: number;
}

// =============================================================================
// Research Types
// =============================================================================

export interface ResearchReport {
  symbol: string;
  companyName: string;
  sector: string;
  industry: string;
  currentPrice: number;
  marketCap: number;
  valuation: Record<string, unknown>;
  dcf?: Record<string, unknown>;
  fairValueRange: { low: number; high: number };
  valuationRating: string;
  earnings: Record<string, unknown>;
  earningsQualityScore: number;
  revenueGrowth5yr: number;
  epsGrowth5yr: number;
  grossMargin: number;
  operatingMargin: number;
  netMargin: number;
  roe: number;
  roic: number;
  debtToEquity: number;
  currentRatio: number;
  overallScore: number;
  strengths: string[];
  weaknesses: string[];
  catalysts: string[];
  risks: string[];
}

export interface ValuationData {
  symbol: string;
  method: string;
  valuation: Record<string, unknown>;
  dcf?: Record<string, unknown>;
  sensitivity?: Record<string, unknown>;
  fairValue?: number;
  currentPrice?: number;
  upsidePercent?: number;
  assumptions?: Record<string, unknown>;
}

export interface EarningsAnalysis {
  symbol: string;
  quarters: Array<Record<string, unknown>>;
  epsGrowthYoy: number;
  epsGrowth3yrCagr: number;
  avgEpsSurprisePercent: number;
  beatRate: number;
  consecutiveBeats: number;
  earningsVolatility: number;
  earningsConsistency: number;
}

// =============================================================================
// Commodities Types
// =============================================================================

export interface CommodityPrice {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume?: number;
  timestamp: string;
}

export interface CommoditySummary {
  symbol: string;
  name: string;
  category: string;
  price: number;
  change1d: number;
  change1w: number;
  change1m: number;
  changeYtd: number;
  volatility30d: number;
  trend: string;
  relativeStrength: number;
}

export interface MacroLinkage {
  commodity: string;
  name: string;
  category: string;
  linkages: Array<{
    commodity: string;
    macroIndicator: string;
    correlation: number;
    leadLagDays: number;
    relationship: string;
    strength: string;
  }>;
  primaryDriver?: string;
}

export interface CorrelationMatrix {
  commodities: string[];
  matrix: Record<string, Record<string, number>>;
}

// =============================================================================
// Macro Types
// =============================================================================

export interface EconomicIndicator {
  code: string;
  name: string;
  value: number;
  previousValue?: number;
  change?: number;
  unit: string;
  frequency: string;
  lastUpdate: string;
  source: string;
}

export interface CountrySnapshot {
  country: string;
  gdpGrowth?: number;
  inflation?: number;
  unemployment?: number;
  policyRate?: number;
  currentAccount?: number;
  regime?: string;
  timestamp: string;
}

export interface RegimeState {
  currentRegime: string;
  confidence: string;
  regimeScore: number;
  components: Record<string, string>;
  metrics: Record<string, number | null>;
  risk: Record<string, unknown>;
  positioning: {
    equity: string;
    duration: string;
    credit: string;
    volatility: string;
  };
  signals: Array<{
    source: string;
    signal: string;
    strength: number;
    confidence: number;
    details: Record<string, unknown>;
  }>;
  regimeDurationDays: number;
  timestamp: string;
}

export interface YieldCurve {
  country: string;
  shape: string;
  spread2y10y?: number;
  spread3m10y?: number;
  recessionSignal: string;
  recessionProbability12m: number;
  inversionDurationDays: number;
  curve: Array<{
    tenor: string;
    yield: number;
    priorYield?: number;
    change?: number;
  }>;
  dynamic: string;
  timestamp: string;
}

// =============================================================================
// Institutional Types
// =============================================================================

export interface InstitutionalHolding {
  managerName: string;
  managerCik: string;
  sharesHeld: number;
  valueHeld: number;
  ownershipPercentage: number;
  changeFromLastQuarter?: number;
}

export interface OwnershipBreakdown {
  symbol: string;
  institutionalOwnership: number;
  retailOwnership: number;
  insiderOwnership: number;
  top10Concentration: number;
  totalHolders: number;
  sharesOutstanding: number;
  floatShares: number;
}

export interface InstitutionalSentiment {
  symbol: string;
  score: number;
  classification: string;
  confidence: number;
  contributingFactors: Record<string, number>;
  weightsUsed: Record<string, number>;
}

export interface SmartMoneyFlow {
  symbol: string;
  netFlow: number;
  weightedFlow: number;
  signal: string;
  signalStrength: number;
  buyersCount: number;
  sellersCount: number;
  buyingActivity: Array<Record<string, unknown>>;
  sellingActivity: Array<Record<string, unknown>>;
  coordinatedBuying: boolean;
  coordinatedSelling: boolean;
}

// =============================================================================
// Money Flow Types
// =============================================================================

export interface MoneyFlowData {
  symbol: string;
  netFlow1m: number;
  netFlow3m: number;
  institutionalChange: number;
  smartMoneySentiment: number;
  flowAcceleration: number;
  confidenceScore: number;
}

export interface SectorRotation {
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

export interface MarketBreadth {
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

// =============================================================================
// Dark Pool Types
// =============================================================================

export interface DarkPoolData {
  symbol: string;
  date?: string;
  darkPoolVolume: number;
  totalVolume: number;
  darkPoolPercentage: number;
  largeBlockActivity: number;
  darkPoolSignal: number;
}

export interface DarkPoolSummary {
  symbol: string;
  data: DarkPoolData[];
  summary: {
    averageDarkPoolPercentage: number;
    averageBlockActivity: number;
    totalDarkPoolVolume: number;
    signalBias: number;
  };
  timestamp: string;
}
