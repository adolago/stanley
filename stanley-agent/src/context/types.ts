/**
 * Context Bridge Types
 *
 * Shared type definitions for the context bridge system.
 */

// =============================================================================
// Context Item Types
// =============================================================================

/**
 * Types of context sources available
 */
export type ContextSourceType =
  | "portfolio"
  | "notes"
  | "research"
  | "tool_results"
  | "preferences"
  | "watchlist"
  | "alerts"
  | "signals";

/**
 * Priority levels for context items
 */
export type ContextPriority = "critical" | "high" | "medium" | "low";

/**
 * Base context item structure
 */
export interface ContextItem {
  id: string;
  source: ContextSourceType;
  priority: ContextPriority;
  timestamp: Date;
  expiresAt?: Date;
  content: string;
  metadata: Record<string, unknown>;
  relevanceScore?: number;
}

// =============================================================================
// Portfolio Context Types
// =============================================================================

export interface PortfolioHolding {
  symbol: string;
  shares: number;
  averageCost?: number;
  currentPrice?: number;
  marketValue?: number;
  weight?: number;
  sector?: string;
  pnl?: number;
  pnlPercent?: number;
}

export interface PortfolioContext {
  holdings: PortfolioHolding[];
  totalValue: number;
  totalCost: number;
  totalReturn: number;
  totalReturnPercent: number;
  beta?: number;
  sharpeRatio?: number;
  var95?: number;
  sectorExposure: Record<string, number>;
  lastUpdated: Date;
}

// =============================================================================
// Research Context Types
// =============================================================================

export interface ResearchNote {
  id: string;
  name: string;
  noteType: "thesis" | "research" | "trade" | "event" | "daily";
  symbol?: string;
  content: string;
  tags: string[];
  createdAt: Date;
  updatedAt: Date;
  conviction?: "low" | "medium" | "high";
  status?: string;
}

export interface ThesisContext {
  symbol: string;
  direction: "long" | "short";
  status: "research" | "watchlist" | "active" | "closed";
  conviction: "low" | "medium" | "high";
  entryPrice?: number;
  targetPrice?: number;
  stopLoss?: number;
  summary: string;
  catalysts: string[];
}

// =============================================================================
// Tool Results Context Types
// =============================================================================

export interface ToolCallResult {
  toolName: string;
  args: Record<string, unknown>;
  result: unknown;
  timestamp: Date;
  symbols?: string[];
  summary?: string;
}

// =============================================================================
// User Preferences Types
// =============================================================================

export interface UserPreferences {
  watchlist: string[];
  favoriteSymbols: string[];
  preferredSectors: string[];
  riskTolerance: "conservative" | "moderate" | "aggressive";
  investmentHorizon: "short" | "medium" | "long";
  alertPreferences: {
    priceAlerts: boolean;
    volumeAlerts: boolean;
    newsAlerts: boolean;
    earningsAlerts: boolean;
  };
}

// =============================================================================
// Alert Types
// =============================================================================

export interface Alert {
  id: string;
  type: "price" | "volume" | "institutional" | "earnings" | "signal" | "custom";
  symbol?: string;
  message: string;
  severity: "info" | "warning" | "critical";
  triggered: boolean;
  triggeredAt?: Date;
  acknowledged: boolean;
}

// =============================================================================
// Signal Context Types
// =============================================================================

export interface SignalContext {
  symbol: string;
  signalType: "buy" | "sell" | "hold";
  strength: "weak" | "moderate" | "strong";
  conviction: number;
  factors: Record<string, number>;
  priceAtSignal: number;
  targetPrice?: number;
  stopLoss?: number;
  generatedAt: Date;
}

// =============================================================================
// Context Bridge Configuration
// =============================================================================

export interface ContextBridgeConfig {
  /** Maximum number of context items to maintain */
  maxContextItems: number;

  /** Default TTL for context items in milliseconds */
  defaultTtlMs: number;

  /** How often to refresh context sources (ms) */
  refreshIntervalMs: number;

  /** Maximum age of tool results to include (ms) */
  toolResultsMaxAgeMs: number;

  /** Maximum total tokens for context window */
  maxContextTokens: number;

  /** Stanley API base URL */
  stanleyApiUrl: string;

  /** Enable persistence to local storage */
  enablePersistence: boolean;

  /** Path for persistent storage */
  storagePath?: string;

  /** Priority weights for context sources */
  sourceWeights: Partial<Record<ContextSourceType, number>>;
}

export const DEFAULT_CONTEXT_CONFIG: ContextBridgeConfig = {
  maxContextItems: 100,
  defaultTtlMs: 60 * 60 * 1000, // 1 hour
  refreshIntervalMs: 5 * 60 * 1000, // 5 minutes
  toolResultsMaxAgeMs: 30 * 60 * 1000, // 30 minutes
  maxContextTokens: 8000,
  stanleyApiUrl: "http://localhost:8000",
  enablePersistence: true,
  storagePath: undefined,
  sourceWeights: {
    portfolio: 1.5,
    notes: 1.2,
    research: 1.3,
    tool_results: 1.0,
    preferences: 0.8,
    watchlist: 1.0,
    alerts: 1.4,
    signals: 1.5,
  },
};

// =============================================================================
// Context Summary Types
// =============================================================================

export interface ContextSummary {
  totalItems: number;
  itemsBySource: Record<ContextSourceType, number>;
  oldestItem: Date | null;
  newestItem: Date | null;
  estimatedTokens: number;
  lastRefresh: Date;
}

// =============================================================================
// Context Events
// =============================================================================

export type ContextEventType =
  | "context_updated"
  | "source_refreshed"
  | "item_added"
  | "item_removed"
  | "item_expired"
  | "memory_persisted"
  | "memory_restored";

export interface ContextEvent {
  type: ContextEventType;
  timestamp: Date;
  source?: ContextSourceType;
  itemId?: string;
  data?: unknown;
}

export type ContextEventHandler = (event: ContextEvent) => void;
