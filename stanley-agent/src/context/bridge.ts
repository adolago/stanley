/**
 * Context Bridge
 *
 * The main context bridge class that maintains continuous awareness
 * between the AI agent and the user's investment research context.
 *
 * Features:
 * - Automatic context gathering from multiple sources
 * - Sliding window of relevant context
 * - Persistent memory across sessions
 * - Context injection into agent prompts
 * - Real-time context updates
 */

import type {
  ContextItem,
  ContextSourceType,
  ContextPriority,
  ContextBridgeConfig,
  ContextSummary,
  ContextEvent,
  ContextEventHandler,
  ToolCallResult,
  PortfolioHolding,
  UserPreferences,
  SignalContext,
  Alert,
} from "./types";
import { DEFAULT_CONTEXT_CONFIG } from "./types";
import {
  type ContextSource,
  createContextSources,
  PortfolioSource,
  ToolResultsSource,
  PreferencesSource,
  WatchlistSource,
  AlertsSource,
  SignalsSource,
} from "./sources";
import { MemoryStore, createMemoryStore, type InsightEntry } from "./memory";

// =============================================================================
// Context Bridge Class
// =============================================================================

export class ContextBridge {
  private config: ContextBridgeConfig;
  private sources: Map<ContextSourceType, ContextSource> = new Map();
  private contextItems: ContextItem[] = [];
  private lastFetch: Map<ContextSourceType, Date> = new Map();
  private memory: MemoryStore;
  private eventHandlers: ContextEventHandler[] = [];
  private refreshTimer: ReturnType<typeof setInterval> | null = null;
  private initialized = false;

  constructor(config: Partial<ContextBridgeConfig> = {}) {
    this.config = { ...DEFAULT_CONTEXT_CONFIG, ...config };

    // Initialize memory store
    this.memory = createMemoryStore({
      enablePersistence: this.config.enablePersistence,
      storagePath: this.config.storagePath || ".stanley/memory",
    });

    // Initialize context sources
    const sourcesList = createContextSources({
      apiUrl: this.config.stanleyApiUrl,
      toolResultsMaxAgeMs: this.config.toolResultsMaxAgeMs,
    });

    for (const source of sourcesList) {
      this.sources.set(source.type, source);
    }
  }

  // ===========================================================================
  // Initialization
  // ===========================================================================

  /**
   * Initialize the context bridge and start background refresh
   */
  async initialize(): Promise<void> {
    if (this.initialized) {
      return;
    }

    // Start a new session
    this.memory.startSession();

    // Perform initial context fetch
    await this.refreshAllSources();

    // Start background refresh timer
    this.startBackgroundRefresh();

    this.initialized = true;
    this.emitEvent({ type: "context_updated", timestamp: new Date() });
  }

  /**
   * Shutdown the context bridge
   */
  async shutdown(): Promise<void> {
    // Stop background refresh
    if (this.refreshTimer) {
      clearInterval(this.refreshTimer);
      this.refreshTimer = null;
    }

    // End session and save memory
    this.memory.endSession();
    await this.memory.close();

    this.initialized = false;
  }

  // ===========================================================================
  // Context Retrieval
  // ===========================================================================

  /**
   * Get context for agent prompt injection
   *
   * Returns formatted context string optimized for the sliding window
   */
  getContextForPrompt(options: {
    maxTokens?: number;
    relevantSymbols?: string[];
    priorityThreshold?: ContextPriority;
  } = {}): string {
    const {
      maxTokens = this.config.maxContextTokens,
      relevantSymbols = [],
      priorityThreshold = "low",
    } = options;

    // Score and filter items
    const scoredItems = this.scoreAndFilterItems(relevantSymbols, priorityThreshold);

    // Build context string within token budget
    return this.buildContextString(scoredItems, maxTokens);
  }

  /**
   * Get raw context items
   */
  getContextItems(options: {
    sources?: ContextSourceType[];
    minPriority?: ContextPriority;
    symbols?: string[];
  } = {}): ContextItem[] {
    let items = [...this.contextItems];

    if (options.sources) {
      items = items.filter((i) => options.sources!.includes(i.source));
    }

    if (options.minPriority) {
      const priorityOrder: Record<ContextPriority, number> = {
        critical: 4,
        high: 3,
        medium: 2,
        low: 1,
      };
      const threshold = priorityOrder[options.minPriority];
      items = items.filter((i) => priorityOrder[i.priority] >= threshold);
    }

    if (options.symbols && options.symbols.length > 0) {
      items = items.filter((i) => {
        const itemSymbols = (i.metadata.symbols as string[]) || [];
        const symbol = i.metadata.symbol as string | undefined;
        const allSymbols = symbol ? [...itemSymbols, symbol] : itemSymbols;
        return options.symbols!.some((s) => allSymbols.includes(s));
      });
    }

    return items;
  }

  /**
   * Get context summary
   */
  getContextSummary(): ContextSummary {
    const itemsBySource: Record<ContextSourceType, number> = {
      portfolio: 0,
      notes: 0,
      research: 0,
      tool_results: 0,
      preferences: 0,
      watchlist: 0,
      alerts: 0,
      signals: 0,
    };

    let oldest: Date | null = null;
    let newest: Date | null = null;

    for (const item of this.contextItems) {
      itemsBySource[item.source]++;

      if (!oldest || item.timestamp < oldest) {
        oldest = item.timestamp;
      }
      if (!newest || item.timestamp > newest) {
        newest = item.timestamp;
      }
    }

    return {
      totalItems: this.contextItems.length,
      itemsBySource,
      oldestItem: oldest,
      newestItem: newest,
      estimatedTokens: this.estimateTokens(this.contextItems),
      lastRefresh: new Date(),
    };
  }

  // ===========================================================================
  // Context Updates
  // ===========================================================================

  /**
   * Refresh all context sources
   */
  async refreshAllSources(): Promise<void> {
    const newItems: ContextItem[] = [];

    const sourceEntries = Array.from(this.sources.entries());
    for (const [type, source] of sourceEntries) {
      try {
        const items = await source.fetch();
        newItems.push(...items);
        this.lastFetch.set(type, new Date());
        this.emitEvent({
          type: "source_refreshed",
          timestamp: new Date(),
          source: type,
        });
      } catch (error) {
        console.error(`Failed to fetch from source ${type}:`, error);
      }
    }

    this.updateContextItems(newItems);
  }

  /**
   * Refresh a specific source
   */
  async refreshSource(type: ContextSourceType): Promise<void> {
    const source = this.sources.get(type);
    if (!source) {
      return;
    }

    try {
      const items = await source.fetch();

      // Remove old items from this source
      this.contextItems = this.contextItems.filter((i) => i.source !== type);

      // Add new items
      this.contextItems.push(...items);
      this.lastFetch.set(type, new Date());

      this.emitEvent({
        type: "source_refreshed",
        timestamp: new Date(),
        source: type,
      });
    } catch (error) {
      console.error(`Failed to refresh source ${type}:`, error);
    }
  }

  /**
   * Add a tool call result to context
   */
  addToolResult(result: ToolCallResult): void {
    const source = this.sources.get("tool_results") as ToolResultsSource | undefined;
    if (source) {
      source.addResult(result);
      this.refreshSource("tool_results").catch(console.error);
    }
  }

  /**
   * Update portfolio holdings
   */
  updatePortfolio(holdings: PortfolioHolding[]): void {
    const source = this.sources.get("portfolio") as PortfolioSource | undefined;
    if (source) {
      source.updateHoldings(holdings);
      this.refreshSource("portfolio").catch(console.error);
    }
  }

  /**
   * Update user preferences
   */
  updatePreferences(preferences: UserPreferences): void {
    const source = this.sources.get("preferences") as PreferencesSource | undefined;
    if (source) {
      source.setPreferences(preferences);
      this.refreshSource("preferences").catch(console.error);
    }
  }

  /**
   * Update watchlist
   */
  updateWatchlist(symbols: string[]): void {
    const source = this.sources.get("watchlist") as WatchlistSource | undefined;
    if (source) {
      source.setWatchlist(symbols);
      this.refreshSource("watchlist").catch(console.error);
    }
  }

  /**
   * Add an alert
   */
  addAlert(alert: Alert): void {
    const source = this.sources.get("alerts") as AlertsSource | undefined;
    if (source) {
      source.addAlert(alert);
      this.refreshSource("alerts").catch(console.error);
    }
  }

  /**
   * Add a signal
   */
  addSignal(signal: SignalContext): void {
    const source = this.sources.get("signals") as SignalsSource | undefined;
    if (source) {
      source.addSignal(signal);
      this.refreshSource("signals").catch(console.error);
    }
  }

  // ===========================================================================
  // Memory / Insights
  // ===========================================================================

  /**
   * Store an insight in persistent memory
   */
  storeInsight(insight: Omit<InsightEntry, "id" | "createdAt">): InsightEntry {
    const entry = this.memory.addInsight(insight);
    this.memory.addInsightToSession(entry);
    return entry;
  }

  /**
   * Get insights for symbols
   */
  getInsightsForSymbols(symbols: string[]): InsightEntry[] {
    return this.memory.getInsightsForSymbols(symbols);
  }

  /**
   * Get recent insights
   */
  getRecentInsights(limit = 10): InsightEntry[] {
    return this.memory.getRecentInsights(limit);
  }

  /**
   * Store symbol-specific context
   */
  storeSymbolContext(symbol: string, context: Record<string, unknown>): void {
    this.memory.setSymbolContext(symbol, context);
  }

  /**
   * Get symbol-specific context
   */
  getSymbolContext<T = Record<string, unknown>>(symbol: string): T | null {
    return this.memory.getSymbolContext<T>(symbol);
  }

  /**
   * Set conversation summary for current session
   */
  setConversationSummary(summary: string): void {
    this.memory.setConversationSummary(summary);
  }

  /**
   * Get memory statistics
   */
  getMemoryStats(): ReturnType<MemoryStore["getStats"]> {
    return this.memory.getStats();
  }

  // ===========================================================================
  // Event Handling
  // ===========================================================================

  /**
   * Subscribe to context events
   */
  onEvent(handler: ContextEventHandler): () => void {
    this.eventHandlers.push(handler);
    return () => {
      this.eventHandlers = this.eventHandlers.filter((h) => h !== handler);
    };
  }

  // ===========================================================================
  // Private Methods
  // ===========================================================================

  private updateContextItems(newItems: ContextItem[]): void {
    // Merge with existing items, removing duplicates by ID
    const existingIds = new Set(newItems.map((i) => i.id));
    const retained = this.contextItems.filter((i) => !existingIds.has(i.id));

    this.contextItems = [...retained, ...newItems];

    // Remove expired items
    const now = new Date();
    this.contextItems = this.contextItems.filter(
      (i) => !i.expiresAt || i.expiresAt > now
    );

    // Enforce max items
    if (this.contextItems.length > this.config.maxContextItems) {
      this.contextItems = this.prioritizeItems(this.contextItems).slice(
        0,
        this.config.maxContextItems
      );
    }

    // Update session
    this.memory.addContextToSession(newItems);

    this.emitEvent({ type: "context_updated", timestamp: new Date() });
  }

  private prioritizeItems(items: ContextItem[]): ContextItem[] {
    const priorityOrder: Record<ContextPriority, number> = {
      critical: 4,
      high: 3,
      medium: 2,
      low: 1,
    };

    return items.sort((a, b) => {
      // First by priority
      const priorityDiff =
        priorityOrder[b.priority] - priorityOrder[a.priority];
      if (priorityDiff !== 0) return priorityDiff;

      // Then by timestamp (newer first)
      return b.timestamp.getTime() - a.timestamp.getTime();
    });
  }

  private scoreAndFilterItems(
    relevantSymbols: string[],
    priorityThreshold: ContextPriority
  ): ContextItem[] {
    const priorityOrder: Record<ContextPriority, number> = {
      critical: 4,
      high: 3,
      medium: 2,
      low: 1,
    };
    const threshold = priorityOrder[priorityThreshold];

    // Filter by priority
    let items = this.contextItems.filter(
      (i) => priorityOrder[i.priority] >= threshold
    );

    // Score items based on relevance
    const scoredItems = items.map((item) => {
      let score = priorityOrder[item.priority];

      // Boost for matching symbols
      if (relevantSymbols.length > 0) {
        const itemSymbols = (item.metadata.symbols as string[]) || [];
        const symbol = item.metadata.symbol as string | undefined;
        const allSymbols = symbol ? [...itemSymbols, symbol] : itemSymbols;

        const matches = relevantSymbols.filter((s) =>
          allSymbols.includes(s)
        ).length;
        score += matches * 2;
      }

      // Apply source weight
      const sourceWeight = this.config.sourceWeights[item.source] || 1.0;
      score *= sourceWeight;

      // Recency bonus (items from last hour get boost)
      const ageMs = Date.now() - item.timestamp.getTime();
      if (ageMs < 60 * 60 * 1000) {
        score *= 1.2;
      }

      return { item, score };
    });

    // Sort by score and return items
    return scoredItems
      .sort((a, b) => b.score - a.score)
      .map((s) => s.item);
  }

  private buildContextString(items: ContextItem[], maxTokens: number): string {
    const sections: string[] = [];
    let estimatedTokens = 0;
    const tokensPerChar = 0.25; // Rough estimate

    // Add header
    const header = "=== CURRENT CONTEXT ===\n";
    sections.push(header);
    estimatedTokens += header.length * tokensPerChar;

    // Group by source for better organization
    const bySource = new Map<ContextSourceType, ContextItem[]>();
    for (const item of items) {
      const existing = bySource.get(item.source) || [];
      existing.push(item);
      bySource.set(item.source, existing);
    }

    // Order of sources in output
    const sourceOrder: ContextSourceType[] = [
      "alerts",
      "signals",
      "portfolio",
      "research",
      "notes",
      "tool_results",
      "watchlist",
      "preferences",
    ];

    for (const source of sourceOrder) {
      const sourceItems = bySource.get(source);
      if (!sourceItems || sourceItems.length === 0) continue;

      const sourceHeader = `\n--- ${source.toUpperCase()} ---\n`;
      const headerTokens = sourceHeader.length * tokensPerChar;

      if (estimatedTokens + headerTokens > maxTokens) {
        break;
      }

      sections.push(sourceHeader);
      estimatedTokens += headerTokens;

      for (const item of sourceItems) {
        const content = item.content + "\n";
        const contentTokens = content.length * tokensPerChar;

        if (estimatedTokens + contentTokens > maxTokens) {
          sections.push("... (truncated)\n");
          break;
        }

        sections.push(content);
        estimatedTokens += contentTokens;
      }
    }

    // Add insights if space permits
    const insights = this.memory.getRecentInsights(3);
    if (insights.length > 0 && estimatedTokens < maxTokens * 0.9) {
      sections.push("\n--- RECENT INSIGHTS ---\n");

      for (const insight of insights) {
        const content = `[${insight.type.toUpperCase()}] ${insight.content}\n`;
        const contentTokens = content.length * tokensPerChar;

        if (estimatedTokens + contentTokens > maxTokens) {
          break;
        }

        sections.push(content);
        estimatedTokens += contentTokens;
      }
    }

    sections.push("\n=== END CONTEXT ===\n");

    return sections.join("");
  }

  private estimateTokens(items: ContextItem[]): number {
    const totalChars = items.reduce((sum, i) => sum + i.content.length, 0);
    return Math.ceil(totalChars * 0.25); // Rough estimate
  }

  private startBackgroundRefresh(): void {
    if (this.refreshTimer) {
      return;
    }

    this.refreshTimer = setInterval(() => {
      // Check each source if it needs refresh
      const sourceEntries = Array.from(this.sources.entries());
      for (const [type, source] of sourceEntries) {
        const lastFetch = this.lastFetch.get(type) || new Date(0);
        if (source.needsRefresh(lastFetch)) {
          this.refreshSource(type).catch(console.error);
        }
      }
    }, this.config.refreshIntervalMs);
  }

  private emitEvent(event: ContextEvent): void {
    for (const handler of this.eventHandlers) {
      try {
        handler(event);
      } catch (error) {
        console.error("Error in context event handler:", error);
      }
    }
  }
}

// =============================================================================
// Factory Function
// =============================================================================

export function createContextBridge(
  config?: Partial<ContextBridgeConfig>
): ContextBridge {
  return new ContextBridge(config);
}

// =============================================================================
// Integration with Stanley Agent
// =============================================================================

/**
 * Create a system prompt enhancer that injects context
 */
export function createContextEnhancer(
  bridge: ContextBridge,
  options: {
    maxContextTokens?: number;
    alwaysInclude?: ContextSourceType[];
  } = {}
): (basePrompt: string, relevantSymbols?: string[]) => string {
  return (basePrompt: string, relevantSymbols?: string[]): string => {
    const context = bridge.getContextForPrompt({
      maxTokens: options.maxContextTokens || 4000,
      relevantSymbols: relevantSymbols || [],
    });

    if (!context || context.trim() === "") {
      return basePrompt;
    }

    return `${basePrompt}

${context}`;
  };
}
