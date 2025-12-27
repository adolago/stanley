/**
 * Session State Container
 *
 * Defines the session state structure and provides state management utilities.
 */

import type { Message } from "../agents/stanley";

/**
 * Tool call record for tracking
 */
export interface ToolCallRecord {
  id: string;
  toolName: string;
  args: unknown;
  result: unknown;
  timestamp: number;
  durationMs: number;
  success: boolean;
  error?: string;
}

/**
 * Token usage tracking
 */
export interface TokenUsage {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  timestamp: number;
}

/**
 * Research focus area
 */
export interface ResearchFocus {
  topic: string;
  symbols: string[];
  sectors: string[];
  addedAt: number;
  priority: "high" | "medium" | "low";
  notes?: string;
}

/**
 * User preferences
 */
export interface UserPreferences {
  verbosity: "concise" | "detailed" | "verbose";
  preferredSectors: string[];
  watchlist: string[];
  riskTolerance: "conservative" | "moderate" | "aggressive";
  analysisDepth: "quick" | "standard" | "comprehensive";
  autoSave: boolean;
  sessionTimeout: number; // in minutes
}

/**
 * Active context state
 */
export interface ActiveContext {
  currentSymbol?: string;
  currentSector?: string;
  portfolioLoaded: boolean;
  lastAnalysis?: {
    type: string;
    symbol?: string;
    timestamp: number;
  };
  pendingActions: string[];
  insights: string[];
}

/**
 * Session metadata
 */
export interface SessionMetadata {
  id: string;
  createdAt: number;
  lastActiveAt: number;
  endedAt?: number;
  version: string;
  provider: string;
  model: string;
  status: "active" | "paused" | "ended" | "expired";
}

/**
 * Complete session state
 */
export interface SessionState {
  metadata: SessionMetadata;
  conversationHistory: Message[];
  activeContext: ActiveContext;
  toolCallHistory: ToolCallRecord[];
  tokenUsage: TokenUsage[];
  userPreferences: UserPreferences;
  researchFocusAreas: ResearchFocus[];
}

/**
 * Default user preferences
 */
export const DEFAULT_USER_PREFERENCES: UserPreferences = {
  verbosity: "detailed",
  preferredSectors: [],
  watchlist: [],
  riskTolerance: "moderate",
  analysisDepth: "standard",
  autoSave: true,
  sessionTimeout: 60,
};

/**
 * Default active context
 */
export const DEFAULT_ACTIVE_CONTEXT: ActiveContext = {
  portfolioLoaded: false,
  pendingActions: [],
  insights: [],
};

/**
 * Session state version for migration support
 */
export const SESSION_STATE_VERSION = "1.0.0";

/**
 * Create initial session state
 */
export function createInitialState(
  sessionId: string,
  provider: string,
  model: string
): SessionState {
  const now = Date.now();

  return {
    metadata: {
      id: sessionId,
      createdAt: now,
      lastActiveAt: now,
      version: SESSION_STATE_VERSION,
      provider,
      model,
      status: "active",
    },
    conversationHistory: [],
    activeContext: { ...DEFAULT_ACTIVE_CONTEXT },
    toolCallHistory: [],
    tokenUsage: [],
    userPreferences: { ...DEFAULT_USER_PREFERENCES },
    researchFocusAreas: [],
  };
}

/**
 * Calculate total token usage for session
 */
export function calculateTotalTokenUsage(state: SessionState): {
  totalPromptTokens: number;
  totalCompletionTokens: number;
  totalTokens: number;
} {
  return state.tokenUsage.reduce(
    (acc, usage) => ({
      totalPromptTokens: acc.totalPromptTokens + usage.promptTokens,
      totalCompletionTokens: acc.totalCompletionTokens + usage.completionTokens,
      totalTokens: acc.totalTokens + usage.totalTokens,
    }),
    { totalPromptTokens: 0, totalCompletionTokens: 0, totalTokens: 0 }
  );
}

/**
 * Get tool call frequency map
 */
export function getToolCallFrequency(
  state: SessionState
): Map<string, number> {
  const frequency = new Map<string, number>();

  for (const call of state.toolCallHistory) {
    const count = frequency.get(call.toolName) || 0;
    frequency.set(call.toolName, count + 1);
  }

  return frequency;
}

/**
 * Extract discussed topics from conversation
 */
export function extractTopicsFromConversation(state: SessionState): string[] {
  const topics = new Set<string>();

  // Extract from research focus areas
  for (const focus of state.researchFocusAreas) {
    topics.add(focus.topic);
  }

  // Extract from active context
  if (state.activeContext.currentSymbol) {
    topics.add(`Symbol: ${state.activeContext.currentSymbol}`);
  }
  if (state.activeContext.currentSector) {
    topics.add(`Sector: ${state.activeContext.currentSector}`);
  }

  // Extract symbols from tool calls
  for (const call of state.toolCallHistory) {
    if (call.args && typeof call.args === "object") {
      const args = call.args as Record<string, unknown>;
      if (typeof args.symbol === "string") {
        topics.add(`Symbol: ${args.symbol}`);
      }
      if (Array.isArray(args.sectors)) {
        for (const sector of args.sectors) {
          if (typeof sector === "string") {
            topics.add(`Sector: ${sector}`);
          }
        }
      }
    }
  }

  return Array.from(topics);
}

/**
 * Clone session state for immutable updates
 */
export function cloneState(state: SessionState): SessionState {
  return JSON.parse(JSON.stringify(state));
}

/**
 * Validate session state structure
 */
export function validateState(state: unknown): state is SessionState {
  if (!state || typeof state !== "object") return false;

  const s = state as Record<string, unknown>;

  // Check required top-level properties
  if (!s.metadata || typeof s.metadata !== "object") return false;
  if (!Array.isArray(s.conversationHistory)) return false;
  if (!s.activeContext || typeof s.activeContext !== "object") return false;
  if (!Array.isArray(s.toolCallHistory)) return false;
  if (!Array.isArray(s.tokenUsage)) return false;
  if (!s.userPreferences || typeof s.userPreferences !== "object") return false;
  if (!Array.isArray(s.researchFocusAreas)) return false;

  // Check metadata
  const meta = s.metadata as Record<string, unknown>;
  if (typeof meta.id !== "string") return false;
  if (typeof meta.createdAt !== "number") return false;
  if (typeof meta.lastActiveAt !== "number") return false;
  if (typeof meta.version !== "string") return false;
  if (typeof meta.status !== "string") return false;

  return true;
}

/**
 * Check if session is expired based on timeout
 */
export function isSessionExpired(
  state: SessionState,
  timeoutMinutes?: number
): boolean {
  const timeout = timeoutMinutes ?? state.userPreferences.sessionTimeout;
  const now = Date.now();
  const lastActive = state.metadata.lastActiveAt;
  const elapsed = (now - lastActive) / 1000 / 60; // minutes

  return elapsed > timeout;
}
