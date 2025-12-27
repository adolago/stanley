/**
 * Context Bridge Module
 *
 * Maintains continuous awareness between the AI agent and the user's
 * investment research context.
 *
 * @example
 * ```typescript
 * import { createContextBridge, createContextEnhancer } from "./context";
 *
 * // Create and initialize the context bridge
 * const bridge = createContextBridge({
 *   stanleyApiUrl: "http://localhost:8000",
 *   enablePersistence: true,
 * });
 * await bridge.initialize();
 *
 * // Update portfolio context
 * bridge.updatePortfolio([
 *   { symbol: "AAPL", shares: 100, averageCost: 150 },
 *   { symbol: "MSFT", shares: 50, averageCost: 350 },
 * ]);
 *
 * // Add tool results as they happen
 * bridge.addToolResult({
 *   toolName: "research_report",
 *   args: { symbol: "AAPL" },
 *   result: { ... },
 *   timestamp: new Date(),
 *   symbols: ["AAPL"],
 * });
 *
 * // Get context for agent prompts
 * const context = bridge.getContextForPrompt({
 *   maxTokens: 4000,
 *   relevantSymbols: ["AAPL", "MSFT"],
 * });
 *
 * // Store insights for persistence
 * bridge.storeInsight({
 *   type: "observation",
 *   content: "AAPL showing strong institutional accumulation",
 *   symbols: ["AAPL"],
 *   confidence: 0.85,
 *   source: "institutional_analysis",
 *   metadata: {},
 * });
 *
 * // Cleanup
 * await bridge.shutdown();
 * ```
 */

// Main bridge
export { ContextBridge, createContextBridge, createContextEnhancer } from "./bridge";

// Context sources
export {
  type ContextSource,
  PortfolioSource,
  NotesSource,
  ThesesSource,
  ToolResultsSource,
  PreferencesSource,
  WatchlistSource,
  AlertsSource,
  SignalsSource,
  createContextSources,
} from "./sources";

// Memory store
export {
  MemoryStore,
  createMemoryStore,
  type MemoryEntry,
  type InsightEntry,
  type SessionState,
  type MemoryStoreConfig,
} from "./memory";

// Types
export {
  type ContextSourceType,
  type ContextPriority,
  type ContextItem,
  type ContextBridgeConfig,
  type ContextSummary,
  type ContextEvent,
  type ContextEventHandler,
  type ContextEventType,
  type PortfolioHolding,
  type PortfolioContext,
  type ResearchNote,
  type ThesisContext,
  type ToolCallResult,
  type UserPreferences,
  type Alert,
  type SignalContext,
  DEFAULT_CONTEXT_CONFIG,
} from "./types";
