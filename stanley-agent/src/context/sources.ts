/**
 * Context Source Adapters
 *
 * Adapters for gathering context from various sources:
 * - Portfolio holdings and performance
 * - Research notes and theses
 * - Recent tool call results
 * - User preferences and watchlists
 * - Market alerts and signals
 */

import type {
  ContextItem,
  ContextSourceType,
  ContextPriority,
  PortfolioContext,
  PortfolioHolding,
  ResearchNote,
  ThesisContext,
  ToolCallResult,
  UserPreferences,
  Alert,
  SignalContext,
} from "./types";

// =============================================================================
// Base Context Source Interface
// =============================================================================

export interface ContextSource {
  readonly type: ContextSourceType;
  readonly name: string;

  /**
   * Fetch fresh context from this source
   */
  fetch(): Promise<ContextItem[]>;

  /**
   * Check if the source needs refresh
   */
  needsRefresh(lastFetch: Date): boolean;

  /**
   * Get the default priority for items from this source
   */
  getDefaultPriority(): ContextPriority;
}

// =============================================================================
// API Client Helper
// =============================================================================

interface ApiResponse<T> {
  success: boolean;
  data: T | null;
  error: string | null;
  timestamp: string;
}

async function fetchFromApi<T>(
  baseUrl: string,
  endpoint: string
): Promise<T | null> {
  try {
    const response = await fetch(`${baseUrl}${endpoint}`, {
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      console.error(`API error: ${response.status} ${response.statusText}`);
      return null;
    }

    const result = (await response.json()) as ApiResponse<T>;
    return result.success ? result.data : null;
  } catch (error) {
    console.error(`Failed to fetch from ${endpoint}:`, error);
    return null;
  }
}

// =============================================================================
// Portfolio Context Source
// =============================================================================

export class PortfolioSource implements ContextSource {
  readonly type: ContextSourceType = "portfolio";
  readonly name = "Portfolio Holdings";

  private refreshIntervalMs = 5 * 60 * 1000; // 5 minutes

  constructor(
    private readonly apiUrl: string,
    private readonly holdings?: PortfolioHolding[]
  ) {}

  async fetch(): Promise<ContextItem[]> {
    const items: ContextItem[] = [];

    // If we have cached holdings, use those
    if (this.holdings && this.holdings.length > 0) {
      items.push(this.createPortfolioContextItem(this.holdings));
      return items;
    }

    // Otherwise fetch from API (would need stored portfolio endpoint)
    // For now, return empty - portfolio context comes from user's holdings
    return items;
  }

  needsRefresh(lastFetch: Date): boolean {
    return Date.now() - lastFetch.getTime() > this.refreshIntervalMs;
  }

  getDefaultPriority(): ContextPriority {
    return "high";
  }

  private createPortfolioContextItem(holdings: PortfolioHolding[]): ContextItem {
    const totalValue = holdings.reduce((sum, h) => sum + (h.marketValue || 0), 0);
    const topHoldings = holdings
      .sort((a, b) => (b.marketValue || 0) - (a.marketValue || 0))
      .slice(0, 5);

    const content = this.formatPortfolioContent(topHoldings, totalValue);

    return {
      id: `portfolio-${Date.now()}`,
      source: "portfolio",
      priority: "high",
      timestamp: new Date(),
      content,
      metadata: {
        holdingsCount: holdings.length,
        totalValue,
        symbols: holdings.map((h) => h.symbol),
      },
    };
  }

  private formatPortfolioContent(
    holdings: PortfolioHolding[],
    totalValue: number
  ): string {
    const lines = [
      `Portfolio Value: $${totalValue.toLocaleString()}`,
      "Top Holdings:",
    ];

    for (const h of holdings) {
      const weight = h.weight?.toFixed(1) || "N/A";
      const pnl = h.pnlPercent ? `${h.pnlPercent >= 0 ? "+" : ""}${h.pnlPercent.toFixed(1)}%` : "";
      lines.push(`  - ${h.symbol}: ${weight}% weight ${pnl}`);
    }

    return lines.join("\n");
  }

  /**
   * Update holdings for this source
   */
  updateHoldings(holdings: PortfolioHolding[]): void {
    // Use Object.assign to update the private property
    Object.assign(this, { holdings });
  }
}

// =============================================================================
// Research Notes Context Source
// =============================================================================

export class NotesSource implements ContextSource {
  readonly type: ContextSourceType = "notes";
  readonly name = "Research Notes";

  private refreshIntervalMs = 10 * 60 * 1000; // 10 minutes

  constructor(private readonly apiUrl: string) {}

  async fetch(): Promise<ContextItem[]> {
    const items: ContextItem[] = [];

    // Fetch recent notes
    const notes = await fetchFromApi<ResearchNote[]>(this.apiUrl, "/api/notes?limit=20");

    if (notes) {
      for (const note of notes) {
        items.push(this.createNoteContextItem(note));
      }
    }

    return items;
  }

  needsRefresh(lastFetch: Date): boolean {
    return Date.now() - lastFetch.getTime() > this.refreshIntervalMs;
  }

  getDefaultPriority(): ContextPriority {
    return "medium";
  }

  private createNoteContextItem(note: ResearchNote): ContextItem {
    const priority = this.determinePriority(note);

    return {
      id: `note-${note.id || note.name}`,
      source: "notes",
      priority,
      timestamp: new Date(note.updatedAt || note.createdAt),
      content: this.formatNoteContent(note),
      metadata: {
        noteType: note.noteType,
        symbol: note.symbol,
        tags: note.tags,
        conviction: note.conviction,
      },
    };
  }

  private determinePriority(note: ResearchNote): ContextPriority {
    if (note.noteType === "thesis" && note.conviction === "high") {
      return "critical";
    }
    if (note.noteType === "thesis" || note.noteType === "trade") {
      return "high";
    }
    return "medium";
  }

  private formatNoteContent(note: ResearchNote): string {
    const lines = [`[${note.noteType.toUpperCase()}] ${note.name}`];

    if (note.symbol) {
      lines.push(`Symbol: ${note.symbol}`);
    }

    if (note.conviction) {
      lines.push(`Conviction: ${note.conviction}`);
    }

    if (note.tags.length > 0) {
      lines.push(`Tags: ${note.tags.join(", ")}`);
    }

    // Include first 200 chars of content
    const preview = note.content.substring(0, 200);
    lines.push(`Content: ${preview}${note.content.length > 200 ? "..." : ""}`);

    return lines.join("\n");
  }
}

// =============================================================================
// Theses Context Source
// =============================================================================

export class ThesesSource implements ContextSource {
  readonly type: ContextSourceType = "research";
  readonly name = "Investment Theses";

  private refreshIntervalMs = 15 * 60 * 1000; // 15 minutes

  constructor(private readonly apiUrl: string) {}

  async fetch(): Promise<ContextItem[]> {
    const items: ContextItem[] = [];

    // Fetch active theses
    const theses = await fetchFromApi<ThesisContext[]>(this.apiUrl, "/api/theses?status=active");

    if (theses) {
      for (const thesis of theses) {
        items.push(this.createThesisContextItem(thesis));
      }
    }

    return items;
  }

  needsRefresh(lastFetch: Date): boolean {
    return Date.now() - lastFetch.getTime() > this.refreshIntervalMs;
  }

  getDefaultPriority(): ContextPriority {
    return "high";
  }

  private createThesisContextItem(thesis: ThesisContext): ContextItem {
    const priority: ContextPriority =
      thesis.conviction === "high" ? "critical" : thesis.conviction === "medium" ? "high" : "medium";

    return {
      id: `thesis-${thesis.symbol}`,
      source: "research",
      priority,
      timestamp: new Date(),
      content: this.formatThesisContent(thesis),
      metadata: {
        symbol: thesis.symbol,
        direction: thesis.direction,
        status: thesis.status,
        conviction: thesis.conviction,
        entryPrice: thesis.entryPrice,
        targetPrice: thesis.targetPrice,
      },
    };
  }

  private formatThesisContent(thesis: ThesisContext): string {
    const lines = [
      `[THESIS] ${thesis.symbol} - ${thesis.direction.toUpperCase()}`,
      `Status: ${thesis.status} | Conviction: ${thesis.conviction}`,
    ];

    if (thesis.entryPrice) {
      lines.push(`Entry: $${thesis.entryPrice}`);
    }
    if (thesis.targetPrice) {
      lines.push(`Target: $${thesis.targetPrice}`);
    }
    if (thesis.stopLoss) {
      lines.push(`Stop: $${thesis.stopLoss}`);
    }

    lines.push(`Summary: ${thesis.summary}`);

    if (thesis.catalysts.length > 0) {
      lines.push(`Catalysts: ${thesis.catalysts.join(", ")}`);
    }

    return lines.join("\n");
  }
}

// =============================================================================
// Tool Results Context Source
// =============================================================================

export class ToolResultsSource implements ContextSource {
  readonly type: ContextSourceType = "tool_results";
  readonly name = "Recent Tool Results";

  private results: ToolCallResult[] = [];
  private maxResults = 10;
  private maxAgeMs = 30 * 60 * 1000; // 30 minutes

  constructor(maxAgeMs?: number) {
    if (maxAgeMs) {
      this.maxAgeMs = maxAgeMs;
    }
  }

  async fetch(): Promise<ContextItem[]> {
    // Filter out expired results
    const now = Date.now();
    this.results = this.results.filter(
      (r) => now - r.timestamp.getTime() < this.maxAgeMs
    );

    return this.results.map((r) => this.createToolResultContextItem(r));
  }

  needsRefresh(_lastFetch: Date): boolean {
    // Tool results are maintained in-memory, no refresh needed
    return false;
  }

  getDefaultPriority(): ContextPriority {
    return "medium";
  }

  /**
   * Add a new tool result to the context
   */
  addResult(result: ToolCallResult): void {
    // Add to front of array
    this.results.unshift(result);

    // Trim to max size
    if (this.results.length > this.maxResults) {
      this.results = this.results.slice(0, this.maxResults);
    }
  }

  /**
   * Clear all results
   */
  clearResults(): void {
    this.results = [];
  }

  private createToolResultContextItem(result: ToolCallResult): ContextItem {
    return {
      id: `tool-${result.toolName}-${result.timestamp.getTime()}`,
      source: "tool_results",
      priority: "medium",
      timestamp: result.timestamp,
      expiresAt: new Date(result.timestamp.getTime() + this.maxAgeMs),
      content: this.formatToolResultContent(result),
      metadata: {
        toolName: result.toolName,
        args: result.args,
        symbols: result.symbols,
      },
    };
  }

  private formatToolResultContent(result: ToolCallResult): string {
    const lines = [`[TOOL: ${result.toolName}]`];

    if (result.summary) {
      lines.push(result.summary);
    } else {
      // Create a brief summary from the result
      const resultStr = JSON.stringify(result.result);
      const preview = resultStr.substring(0, 300);
      lines.push(`Result: ${preview}${resultStr.length > 300 ? "..." : ""}`);
    }

    if (result.symbols && result.symbols.length > 0) {
      lines.push(`Symbols: ${result.symbols.join(", ")}`);
    }

    return lines.join("\n");
  }
}

// =============================================================================
// User Preferences Context Source
// =============================================================================

export class PreferencesSource implements ContextSource {
  readonly type: ContextSourceType = "preferences";
  readonly name = "User Preferences";

  private preferences: UserPreferences | null = null;
  private refreshIntervalMs = 60 * 60 * 1000; // 1 hour

  constructor(private readonly apiUrl: string) {}

  async fetch(): Promise<ContextItem[]> {
    // Try to fetch from API (if preferences endpoint exists)
    const prefs = await fetchFromApi<UserPreferences>(
      this.apiUrl,
      "/api/settings/preferences"
    );

    if (prefs) {
      this.preferences = prefs;
    }

    if (!this.preferences) {
      return [];
    }

    return [this.createPreferencesContextItem(this.preferences)];
  }

  needsRefresh(lastFetch: Date): boolean {
    return Date.now() - lastFetch.getTime() > this.refreshIntervalMs;
  }

  getDefaultPriority(): ContextPriority {
    return "low";
  }

  /**
   * Set preferences directly (for local state)
   */
  setPreferences(prefs: UserPreferences): void {
    this.preferences = prefs;
  }

  private createPreferencesContextItem(prefs: UserPreferences): ContextItem {
    return {
      id: "user-preferences",
      source: "preferences",
      priority: "low",
      timestamp: new Date(),
      content: this.formatPreferencesContent(prefs),
      metadata: {
        watchlist: prefs.watchlist,
        riskTolerance: prefs.riskTolerance,
        investmentHorizon: prefs.investmentHorizon,
      },
    };
  }

  private formatPreferencesContent(prefs: UserPreferences): string {
    const lines = ["[USER PREFERENCES]"];

    if (prefs.watchlist.length > 0) {
      lines.push(`Watchlist: ${prefs.watchlist.join(", ")}`);
    }

    if (prefs.favoriteSymbols.length > 0) {
      lines.push(`Favorites: ${prefs.favoriteSymbols.join(", ")}`);
    }

    lines.push(`Risk Tolerance: ${prefs.riskTolerance}`);
    lines.push(`Investment Horizon: ${prefs.investmentHorizon}`);

    if (prefs.preferredSectors.length > 0) {
      lines.push(`Preferred Sectors: ${prefs.preferredSectors.join(", ")}`);
    }

    return lines.join("\n");
  }
}

// =============================================================================
// Watchlist Context Source
// =============================================================================

export class WatchlistSource implements ContextSource {
  readonly type: ContextSourceType = "watchlist";
  readonly name = "Watchlist";

  private watchlist: string[] = [];
  private refreshIntervalMs = 5 * 60 * 1000; // 5 minutes

  constructor(private readonly apiUrl: string) {}

  async fetch(): Promise<ContextItem[]> {
    // Fetch watchlist data with current prices
    // For now, return the watchlist symbols
    if (this.watchlist.length === 0) {
      return [];
    }

    return [this.createWatchlistContextItem(this.watchlist)];
  }

  needsRefresh(lastFetch: Date): boolean {
    return Date.now() - lastFetch.getTime() > this.refreshIntervalMs;
  }

  getDefaultPriority(): ContextPriority {
    return "medium";
  }

  /**
   * Set the watchlist symbols
   */
  setWatchlist(symbols: string[]): void {
    this.watchlist = symbols;
  }

  /**
   * Add a symbol to the watchlist
   */
  addSymbol(symbol: string): void {
    if (!this.watchlist.includes(symbol)) {
      this.watchlist.push(symbol);
    }
  }

  /**
   * Remove a symbol from the watchlist
   */
  removeSymbol(symbol: string): void {
    this.watchlist = this.watchlist.filter((s) => s !== symbol);
  }

  private createWatchlistContextItem(symbols: string[]): ContextItem {
    return {
      id: "user-watchlist",
      source: "watchlist",
      priority: "medium",
      timestamp: new Date(),
      content: `[WATCHLIST]\nSymbols: ${symbols.join(", ")}`,
      metadata: {
        symbols,
        count: symbols.length,
      },
    };
  }
}

// =============================================================================
// Alerts Context Source
// =============================================================================

export class AlertsSource implements ContextSource {
  readonly type: ContextSourceType = "alerts";
  readonly name = "Market Alerts";

  private alerts: Alert[] = [];

  constructor(private readonly apiUrl: string) {}

  async fetch(): Promise<ContextItem[]> {
    // Fetch active alerts from API
    const alerts = await fetchFromApi<Alert[]>(
      this.apiUrl,
      "/api/signals/alerts?active=true"
    );

    if (alerts) {
      this.alerts = alerts;
    }

    // Only return unacknowledged alerts
    const activeAlerts = this.alerts.filter((a) => !a.acknowledged);
    return activeAlerts.map((a) => this.createAlertContextItem(a));
  }

  needsRefresh(_lastFetch: Date): boolean {
    // Alerts should be checked frequently
    return true;
  }

  getDefaultPriority(): ContextPriority {
    return "high";
  }

  /**
   * Add a local alert
   */
  addAlert(alert: Alert): void {
    this.alerts.push(alert);
  }

  /**
   * Acknowledge an alert
   */
  acknowledgeAlert(alertId: string): void {
    const alert = this.alerts.find((a) => a.id === alertId);
    if (alert) {
      alert.acknowledged = true;
    }
  }

  private createAlertContextItem(alert: Alert): ContextItem {
    const priority: ContextPriority =
      alert.severity === "critical" ? "critical" : alert.severity === "warning" ? "high" : "medium";

    return {
      id: `alert-${alert.id}`,
      source: "alerts",
      priority,
      timestamp: alert.triggeredAt || new Date(),
      content: this.formatAlertContent(alert),
      metadata: {
        alertType: alert.type,
        symbol: alert.symbol,
        severity: alert.severity,
        triggered: alert.triggered,
      },
    };
  }

  private formatAlertContent(alert: Alert): string {
    const lines = [`[ALERT: ${alert.type.toUpperCase()}]`];

    if (alert.symbol) {
      lines.push(`Symbol: ${alert.symbol}`);
    }

    lines.push(`Severity: ${alert.severity}`);
    lines.push(`Message: ${alert.message}`);

    return lines.join("\n");
  }
}

// =============================================================================
// Signals Context Source
// =============================================================================

export class SignalsSource implements ContextSource {
  readonly type: ContextSourceType = "signals";
  readonly name = "Investment Signals";

  private signals: SignalContext[] = [];
  private refreshIntervalMs = 15 * 60 * 1000; // 15 minutes

  constructor(private readonly apiUrl: string) {}

  async fetch(): Promise<ContextItem[]> {
    // Fetch recent signals from API
    const signals = await fetchFromApi<SignalContext[]>(
      this.apiUrl,
      "/api/signals/recent?limit=10"
    );

    if (signals) {
      this.signals = signals;
    }

    return this.signals.map((s) => this.createSignalContextItem(s));
  }

  needsRefresh(lastFetch: Date): boolean {
    return Date.now() - lastFetch.getTime() > this.refreshIntervalMs;
  }

  getDefaultPriority(): ContextPriority {
    return "high";
  }

  /**
   * Add a signal directly
   */
  addSignal(signal: SignalContext): void {
    this.signals.unshift(signal);
    // Keep only recent signals
    if (this.signals.length > 20) {
      this.signals = this.signals.slice(0, 20);
    }
  }

  private createSignalContextItem(signal: SignalContext): ContextItem {
    const priority: ContextPriority =
      signal.strength === "strong" ? "critical" : signal.strength === "moderate" ? "high" : "medium";

    return {
      id: `signal-${signal.symbol}-${signal.generatedAt.getTime()}`,
      source: "signals",
      priority,
      timestamp: signal.generatedAt,
      content: this.formatSignalContent(signal),
      metadata: {
        symbol: signal.symbol,
        signalType: signal.signalType,
        strength: signal.strength,
        conviction: signal.conviction,
        factors: signal.factors,
      },
    };
  }

  private formatSignalContent(signal: SignalContext): string {
    const lines = [
      `[SIGNAL] ${signal.symbol} - ${signal.signalType.toUpperCase()}`,
      `Strength: ${signal.strength} | Conviction: ${(signal.conviction * 100).toFixed(0)}%`,
      `Price at Signal: $${signal.priceAtSignal}`,
    ];

    if (signal.targetPrice) {
      lines.push(`Target: $${signal.targetPrice}`);
    }
    if (signal.stopLoss) {
      lines.push(`Stop: $${signal.stopLoss}`);
    }

    // Add top factors
    const topFactors = Object.entries(signal.factors)
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
      .slice(0, 3)
      .map(([name, score]) => `${name}: ${score >= 0 ? "+" : ""}${score.toFixed(2)}`)
      .join(", ");

    lines.push(`Key Factors: ${topFactors}`);

    return lines.join("\n");
  }
}

// =============================================================================
// Source Factory
// =============================================================================

export interface ContextSourceOptions {
  apiUrl: string;
  toolResultsMaxAgeMs?: number;
}

export function createContextSources(options: ContextSourceOptions): ContextSource[] {
  return [
    new PortfolioSource(options.apiUrl),
    new NotesSource(options.apiUrl),
    new ThesesSource(options.apiUrl),
    new ToolResultsSource(options.toolResultsMaxAgeMs),
    new PreferencesSource(options.apiUrl),
    new WatchlistSource(options.apiUrl),
    new AlertsSource(options.apiUrl),
    new SignalsSource(options.apiUrl),
  ];
}
