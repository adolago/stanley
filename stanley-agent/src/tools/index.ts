/**
 * Stanley MCP Tools
 *
 * MCP tool definitions for the Stanley investment analysis platform.
 * These tools connect to Stanley's Python API endpoints for:
 * - Portfolio analytics (VaR, beta, sector exposure)
 * - Research reports (valuation, earnings, DCF)
 * - Commodities data (prices, correlations, macro linkages)
 * - Macro indicators (economic data, regime detection)
 * - Institutional holdings (13F filings analysis)
 * - Money flow (sector money flow analysis)
 * - Dark pool (dark pool activity)
 */

export { portfolioAnalyze } from "./portfolio-analyze";
export { researchReport } from "./research-report";
export { commoditiesData } from "./commodities-data";
export { macroIndicators } from "./macro-indicators";
export { institutionalHoldings } from "./institutional-holdings";
export { moneyFlow } from "./money-flow";
export { darkPool } from "./dark-pool";

// Re-export types
export type { StanleyApiResponse, ToolContext } from "./types";
