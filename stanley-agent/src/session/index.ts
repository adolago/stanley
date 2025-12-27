/**
 * Session Module Exports
 *
 * Provides session management capabilities for the Stanley agent:
 * - Session lifecycle (create, resume, end)
 * - State persistence to disk
 * - Session analytics and metrics
 * - Multi-session support
 */

// Main session manager
export { SessionManager } from "./manager";
export type { SessionManagerConfig } from "./manager";

// Session state types and utilities
export {
  createInitialState,
  cloneState,
  validateState,
  isSessionExpired,
  calculateTotalTokenUsage,
  getToolCallFrequency,
  extractTopicsFromConversation,
  SESSION_STATE_VERSION,
  DEFAULT_USER_PREFERENCES,
  DEFAULT_ACTIVE_CONTEXT,
} from "./state";

export type {
  SessionState,
  SessionMetadata,
  ToolCallRecord,
  TokenUsage,
  ResearchFocus,
  UserPreferences,
  ActiveContext,
} from "./state";

// Persistence
export { SessionPersistence, DEFAULT_PERSISTENCE_CONFIG } from "./persistence";
export type { PersistenceConfig, SessionFileInfo } from "./persistence";

// Analytics
export {
  SessionAnalytics,
  recordTokenUsage,
  recordToolCall,
  addInsight,
  addPendingAction,
  completePendingAction,
} from "./analytics";

export type {
  ToolPerformanceMetrics,
  TimeBasedAnalytics,
  SessionComparison,
} from "./analytics";
