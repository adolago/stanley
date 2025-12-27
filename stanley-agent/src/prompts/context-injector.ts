/**
 * Context Injector
 *
 * Injects relevant context into prompts based on available data
 * and query intent. Manages context prioritization and token budgets.
 */

import type { PortfolioHolding, RegimeState } from "../tools/types";

/**
 * Context types that can be injected
 */
export interface ContextData {
  /** Current portfolio positions */
  portfolio?: PortfolioPosition[];
  /** Recent research findings */
  research?: ResearchFinding[];
  /** User's watchlist symbols */
  watchlist?: string[];
  /** Current market regime */
  regime?: RegimeState | MarketRegimeSummary;
  /** Relevant notes/memos */
  notes?: NoteSnippet[];
  /** Recent tool results */
  toolResults?: ToolResult[];
  /** Session history summary */
  sessionSummary?: string;
  /** Custom context */
  custom?: Record<string, unknown>;
}

/**
 * Portfolio position with context-relevant fields
 */
export interface PortfolioPosition {
  symbol: string;
  shares: number;
  averageCost?: number;
  currentPrice?: number;
  marketValue?: number;
  weight?: number;
  sector?: string;
  dayChange?: number;
  unrealizedPnL?: number;
}

/**
 * Research finding summary
 */
export interface ResearchFinding {
  symbol: string;
  date: string;
  type: "report" | "valuation" | "earnings" | "note";
  summary: string;
  keyMetrics?: Record<string, number | string>;
  rating?: string;
}

/**
 * Market regime summary (simplified)
 */
export interface MarketRegimeSummary {
  regime: string;
  confidence: string;
  positioning: {
    equity: string;
    duration: string;
    credit: string;
    volatility: string;
  };
  timestamp: string;
}

/**
 * Note snippet from research memos
 */
export interface NoteSnippet {
  id: string;
  title: string;
  excerpt: string;
  symbols?: string[];
  tags?: string[];
  date: string;
}

/**
 * Tool result from recent execution
 */
export interface ToolResult {
  toolName: string;
  timestamp: string;
  symbols?: string[];
  summary: string;
}

/**
 * Context injection options
 */
export interface InjectionOptions {
  /** Maximum tokens for context (approximate) */
  maxTokens?: number;
  /** Priority order for context types */
  priority?: ContextPriority[];
  /** Whether to include metadata about context */
  includeMetadata?: boolean;
  /** Symbols to emphasize in context */
  focusSymbols?: string[];
}

/**
 * Context priority levels
 */
export type ContextPriority =
  | "portfolio"
  | "regime"
  | "research"
  | "watchlist"
  | "notes"
  | "toolResults"
  | "sessionSummary";

/**
 * Default priority order
 */
const DEFAULT_PRIORITY: ContextPriority[] = [
  "portfolio",
  "regime",
  "research",
  "watchlist",
  "notes",
  "toolResults",
  "sessionSummary",
];

/**
 * Injected context result
 */
export interface InjectedContext {
  /** Formatted context string to append to prompt */
  contextString: string;
  /** Estimated token count */
  estimatedTokens: number;
  /** Which context types were included */
  includedTypes: ContextPriority[];
  /** Which context types were truncated */
  truncatedTypes: ContextPriority[];
  /** Metadata about the injection */
  metadata: {
    portfolioPositions?: number;
    researchCount?: number;
    watchlistCount?: number;
    notesCount?: number;
    toolResultsCount?: number;
  };
}

/**
 * Estimate token count (rough approximation: 4 chars = 1 token)
 */
function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

/**
 * Format portfolio context
 */
function formatPortfolioContext(
  positions: PortfolioPosition[],
  maxPositions: number = 10
): string {
  if (!positions.length) return "";

  const sorted = [...positions]
    .sort((a, b) => (b.marketValue ?? 0) - (a.marketValue ?? 0))
    .slice(0, maxPositions);

  let context = "### Current Portfolio Positions\n\n";
  context += "| Symbol | Shares | Weight | Sector | Day Chg |\n";
  context += "|--------|--------|--------|--------|--------|\n";

  for (const pos of sorted) {
    const weight = pos.weight ? `${pos.weight.toFixed(1)}%` : "-";
    const dayChg = pos.dayChange ? `${pos.dayChange >= 0 ? "+" : ""}${pos.dayChange.toFixed(2)}%` : "-";
    context += `| ${pos.symbol} | ${pos.shares.toLocaleString()} | ${weight} | ${pos.sector ?? "-"} | ${dayChg} |\n`;
  }

  if (positions.length > maxPositions) {
    context += `\n*Showing top ${maxPositions} of ${positions.length} positions*\n`;
  }

  return context + "\n";
}

/**
 * Format research context
 */
function formatResearchContext(
  findings: ResearchFinding[],
  focusSymbols?: string[],
  maxFindings: number = 5
): string {
  if (!findings.length) return "";

  // Prioritize findings for focus symbols
  let sorted = [...findings];
  if (focusSymbols?.length) {
    sorted.sort((a, b) => {
      const aFocus = focusSymbols.includes(a.symbol) ? 0 : 1;
      const bFocus = focusSymbols.includes(b.symbol) ? 0 : 1;
      return aFocus - bFocus;
    });
  }
  sorted = sorted.slice(0, maxFindings);

  let context = "### Recent Research Findings\n\n";

  for (const finding of sorted) {
    context += `**${finding.symbol}** (${finding.type}, ${finding.date})\n`;
    context += `${finding.summary}\n`;
    if (finding.rating) {
      context += `Rating: ${finding.rating}\n`;
    }
    if (finding.keyMetrics) {
      const metrics = Object.entries(finding.keyMetrics)
        .slice(0, 3)
        .map(([k, v]) => `${k}: ${v}`)
        .join(" | ");
      context += `Key Metrics: ${metrics}\n`;
    }
    context += "\n";
  }

  return context;
}

/**
 * Format watchlist context
 */
function formatWatchlistContext(symbols: string[]): string {
  if (!symbols.length) return "";

  return `### Watchlist\n\nUser is tracking: ${symbols.join(", ")}\n\n`;
}

/**
 * Format regime context
 */
function formatRegimeContext(regime: RegimeState | MarketRegimeSummary): string {
  let context = "### Current Market Regime\n\n";

  const regimeName = "currentRegime" in regime ? regime.currentRegime : regime.regime;
  const confidence = regime.confidence;
  const positioning = regime.positioning;

  context += `**Regime:** ${regimeName} (${confidence} confidence)\n\n`;
  context += `**Positioning Guidance:**\n`;
  context += `- Equity: ${positioning.equity}\n`;
  context += `- Duration: ${positioning.duration}\n`;
  context += `- Credit: ${positioning.credit}\n`;
  context += `- Volatility: ${positioning.volatility}\n`;

  return context + "\n";
}

/**
 * Format notes context
 */
function formatNotesContext(
  notes: NoteSnippet[],
  focusSymbols?: string[],
  maxNotes: number = 3
): string {
  if (!notes.length) return "";

  // Prioritize notes mentioning focus symbols
  let sorted = [...notes];
  if (focusSymbols?.length) {
    sorted.sort((a, b) => {
      const aRelevant = a.symbols?.some((s) => focusSymbols.includes(s)) ? 0 : 1;
      const bRelevant = b.symbols?.some((s) => focusSymbols.includes(s)) ? 0 : 1;
      return aRelevant - bRelevant;
    });
  }
  sorted = sorted.slice(0, maxNotes);

  let context = "### Relevant Research Notes\n\n";

  for (const note of sorted) {
    context += `**${note.title}** (${note.date})\n`;
    context += `${note.excerpt}\n`;
    if (note.symbols?.length) {
      context += `Symbols: ${note.symbols.join(", ")}\n`;
    }
    context += "\n";
  }

  return context;
}

/**
 * Format tool results context
 */
function formatToolResultsContext(
  results: ToolResult[],
  maxResults: number = 3
): string {
  if (!results.length) return "";

  const recent = results.slice(0, maxResults);

  let context = "### Recent Analysis Results\n\n";

  for (const result of recent) {
    context += `**${result.toolName}** (${result.timestamp})\n`;
    if (result.symbols?.length) {
      context += `Symbols: ${result.symbols.join(", ")}\n`;
    }
    context += `${result.summary}\n\n`;
  }

  return context;
}

/**
 * Format session summary context
 */
function formatSessionSummaryContext(summary: string): string {
  if (!summary) return "";

  return `### Session Context\n\n${summary}\n\n`;
}

/**
 * Context injector class
 */
export class ContextInjector {
  private data: ContextData;
  private options: InjectionOptions;

  constructor(data: ContextData = {}, options: InjectionOptions = {}) {
    this.data = data;
    this.options = {
      maxTokens: options.maxTokens ?? 2000,
      priority: options.priority ?? DEFAULT_PRIORITY,
      includeMetadata: options.includeMetadata ?? true,
      focusSymbols: options.focusSymbols,
    };
  }

  /**
   * Update context data
   */
  setContext(data: Partial<ContextData>): void {
    this.data = { ...this.data, ...data };
  }

  /**
   * Update options
   */
  setOptions(options: Partial<InjectionOptions>): void {
    this.options = { ...this.options, ...options };
  }

  /**
   * Set focus symbols for context prioritization
   */
  setFocusSymbols(symbols: string[]): void {
    this.options.focusSymbols = symbols;
  }

  /**
   * Build context string based on available data and options
   */
  inject(): InjectedContext {
    const { maxTokens, priority, includeMetadata, focusSymbols } = this.options;
    const includedTypes: ContextPriority[] = [];
    const truncatedTypes: ContextPriority[] = [];
    const metadata: InjectedContext["metadata"] = {};

    let contextString = "\n## Current Context\n\n";
    let currentTokens = estimateTokens(contextString);

    const contextBuilders: Record<ContextPriority, () => string> = {
      portfolio: () => {
        if (!this.data.portfolio?.length) return "";
        metadata.portfolioPositions = this.data.portfolio.length;
        return formatPortfolioContext(this.data.portfolio);
      },
      regime: () => {
        if (!this.data.regime) return "";
        return formatRegimeContext(this.data.regime);
      },
      research: () => {
        if (!this.data.research?.length) return "";
        metadata.researchCount = this.data.research.length;
        return formatResearchContext(this.data.research, focusSymbols);
      },
      watchlist: () => {
        if (!this.data.watchlist?.length) return "";
        metadata.watchlistCount = this.data.watchlist.length;
        return formatWatchlistContext(this.data.watchlist);
      },
      notes: () => {
        if (!this.data.notes?.length) return "";
        metadata.notesCount = this.data.notes.length;
        return formatNotesContext(this.data.notes, focusSymbols);
      },
      toolResults: () => {
        if (!this.data.toolResults?.length) return "";
        metadata.toolResultsCount = this.data.toolResults.length;
        return formatToolResultsContext(this.data.toolResults);
      },
      sessionSummary: () => {
        if (!this.data.sessionSummary) return "";
        return formatSessionSummaryContext(this.data.sessionSummary);
      },
    };

    // Build context in priority order
    for (const contextType of priority!) {
      const builder = contextBuilders[contextType];
      if (!builder) continue;

      const section = builder();
      if (!section) continue;

      const sectionTokens = estimateTokens(section);

      if (currentTokens + sectionTokens <= maxTokens!) {
        contextString += section;
        currentTokens += sectionTokens;
        includedTypes.push(contextType);
      } else {
        truncatedTypes.push(contextType);
      }
    }

    // Add metadata if requested
    if (includeMetadata && includedTypes.length > 0) {
      contextString += `---\n*Context includes: ${includedTypes.join(", ")}*\n`;
    }

    return {
      contextString: includedTypes.length > 0 ? contextString : "",
      estimatedTokens: currentTokens,
      includedTypes,
      truncatedTypes,
      metadata,
    };
  }

  /**
   * Extract symbols mentioned in a query for focus
   */
  static extractSymbols(query: string): string[] {
    // Match common stock ticker patterns (1-5 uppercase letters)
    const tickerPattern = /\b[A-Z]{1,5}\b/g;
    const matches = query.match(tickerPattern) || [];

    // Filter out common words that look like tickers
    const commonWords = new Set([
      "A", "I", "AM", "PM", "CEO", "CFO", "IPO", "ETF", "EPS", "PE",
      "US", "UK", "EU", "GDP", "CPI", "PPI", "VIX", "THE", "AND", "FOR",
      "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HAD", "HER", "WAS",
      "ONE", "OUR", "OUT", "DAY", "GET", "HAS", "HIM", "HIS", "HOW",
      "MAN", "NEW", "NOW", "OLD", "SEE", "WAY", "WHO", "BOY", "DID",
      "ITS", "LET", "PUT", "SAY", "SHE", "TOO", "USE", "VS", "YTD",
    ]);

    return [...new Set(matches.filter((m) => !commonWords.has(m)))];
  }
}

/**
 * Create a context injector with data
 */
export function createContextInjector(
  data?: ContextData,
  options?: InjectionOptions
): ContextInjector {
  return new ContextInjector(data, options);
}
