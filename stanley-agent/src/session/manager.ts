/**
 * Session Manager
 *
 * Handles session lifecycle: create, resume, end, and manage concurrent sessions.
 */

import type { Message } from "../agents/stanley";
import {
  SessionState,
  createInitialState,
  cloneState,
  isSessionExpired,
  UserPreferences,
  ResearchFocus,
  ActiveContext,
} from "./state";
import {
  SessionPersistence,
  PersistenceConfig,
  SessionFileInfo,
} from "./persistence";
import {
  SessionAnalytics,
  recordTokenUsage,
  recordToolCall,
} from "./analytics";

/**
 * Session manager configuration
 */
export interface SessionManagerConfig {
  persistence: Partial<PersistenceConfig>;
  maxConcurrentSessions?: number;
  autoResumeLastSession?: boolean;
  onSessionStart?: (sessionId: string) => void;
  onSessionEnd?: (sessionId: string, analytics: ReturnType<typeof SessionAnalytics.analyze>) => void;
  onSessionExpired?: (sessionId: string) => void;
}

/**
 * Default session manager configuration
 */
const DEFAULT_CONFIG: SessionManagerConfig = {
  persistence: {},
  maxConcurrentSessions: 5,
  autoResumeLastSession: true,
};

/**
 * Generate unique session ID
 */
function generateSessionId(): string {
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).substring(2, 8);
  return `stanley-${timestamp}-${random}`;
}

/**
 * Session manager for handling session lifecycle
 */
export class SessionManager {
  private config: SessionManagerConfig;
  private persistence: SessionPersistence;
  private activeSessions: Map<string, SessionState>;
  private currentSessionId: string | null;

  constructor(config: Partial<SessionManagerConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.persistence = new SessionPersistence(this.config.persistence);
    this.activeSessions = new Map();
    this.currentSessionId = null;
  }

  /**
   * Create a new session
   */
  async createSession(
    provider: string,
    model: string,
    preferences?: Partial<UserPreferences>
  ): Promise<SessionState> {
    // Check concurrent session limit
    if (
      this.config.maxConcurrentSessions &&
      this.activeSessions.size >= this.config.maxConcurrentSessions
    ) {
      // End oldest session
      const oldestId = this.getOldestActiveSession();
      if (oldestId) {
        await this.endSession(oldestId);
      }
    }

    const sessionId = generateSessionId();
    const state = createInitialState(sessionId, provider, model);

    // Apply custom preferences
    if (preferences) {
      state.userPreferences = { ...state.userPreferences, ...preferences };
    }

    // Store in active sessions
    this.activeSessions.set(sessionId, state);
    this.currentSessionId = sessionId;

    // Start auto-save
    this.persistence.startAutoSave(() => this.getCurrentState()!);

    // Save initial state
    await this.persistence.save(state);

    // Callback
    this.config.onSessionStart?.(sessionId);

    return state;
  }

  /**
   * Resume an existing session
   */
  async resumeSession(sessionId: string): Promise<SessionState | null> {
    // Check if already active
    if (this.activeSessions.has(sessionId)) {
      this.currentSessionId = sessionId;
      return this.activeSessions.get(sessionId)!;
    }

    // Load from persistence
    const state = await this.persistence.load(sessionId);
    if (!state) {
      return null;
    }

    // Check if expired
    if (isSessionExpired(state)) {
      state.metadata.status = "expired";
      this.config.onSessionExpired?.(sessionId);
    } else {
      state.metadata.status = "active";
    }

    // Update last active time
    state.metadata.lastActiveAt = Date.now();

    // Add to active sessions
    this.activeSessions.set(sessionId, state);
    this.currentSessionId = sessionId;

    // Start auto-save
    this.persistence.startAutoSave(() => this.getCurrentState()!);

    // Save updated state
    await this.persistence.save(state);

    return state;
  }

  /**
   * Resume most recent session or create new one
   */
  async resumeOrCreate(
    provider: string,
    model: string,
    preferences?: Partial<UserPreferences>
  ): Promise<SessionState> {
    if (this.config.autoResumeLastSession) {
      const recentSession = await this.persistence.getMostRecentSession();
      if (recentSession && !isSessionExpired(recentSession)) {
        const resumed = await this.resumeSession(recentSession.metadata.id);
        if (resumed) {
          return resumed;
        }
      }
    }

    return this.createSession(provider, model, preferences);
  }

  /**
   * End a session
   */
  async endSession(sessionId?: string): Promise<void> {
    const id = sessionId || this.currentSessionId;
    if (!id) return;

    const state = this.activeSessions.get(id);
    if (!state) return;

    // Update state
    state.metadata.status = "ended";
    state.metadata.endedAt = Date.now();

    // Generate analytics
    const analytics = SessionAnalytics.analyze(state);

    // Save final state
    await this.persistence.save(state);

    // Stop auto-save if this is current session
    if (id === this.currentSessionId) {
      this.persistence.stopAutoSave();
      this.currentSessionId = null;
    }

    // Remove from active sessions
    this.activeSessions.delete(id);

    // Callback
    this.config.onSessionEnd?.(id, analytics);
  }

  /**
   * Pause current session
   */
  async pauseSession(): Promise<void> {
    const state = this.getCurrentState();
    if (!state) return;

    state.metadata.status = "paused";
    state.metadata.lastActiveAt = Date.now();

    await this.persistence.save(state);
    this.persistence.stopAutoSave();
  }

  /**
   * Get current session state
   */
  getCurrentState(): SessionState | null {
    if (!this.currentSessionId) return null;
    return this.activeSessions.get(this.currentSessionId) || null;
  }

  /**
   * Get current session ID
   */
  getCurrentSessionId(): string | null {
    return this.currentSessionId;
  }

  /**
   * Get all active sessions
   */
  getActiveSessions(): SessionState[] {
    return Array.from(this.activeSessions.values());
  }

  /**
   * Get oldest active session ID
   */
  private getOldestActiveSession(): string | null {
    let oldestId: string | null = null;
    let oldestTime = Infinity;

    for (const [id, state] of this.activeSessions) {
      if (state.metadata.lastActiveAt < oldestTime) {
        oldestTime = state.metadata.lastActiveAt;
        oldestId = id;
      }
    }

    return oldestId;
  }

  /**
   * Switch to a different active session
   */
  switchSession(sessionId: string): boolean {
    if (!this.activeSessions.has(sessionId)) {
      return false;
    }

    this.currentSessionId = sessionId;
    return true;
  }

  /**
   * Add message to conversation history
   */
  addMessage(message: Message): void {
    const state = this.getCurrentState();
    if (!state) return;

    state.conversationHistory.push(message);
    state.metadata.lastActiveAt = Date.now();
  }

  /**
   * Get conversation history
   */
  getConversationHistory(): Message[] {
    const state = this.getCurrentState();
    return state?.conversationHistory || [];
  }

  /**
   * Clear conversation history
   */
  clearConversation(): void {
    const state = this.getCurrentState();
    if (!state) return;

    state.conversationHistory = [];
    state.metadata.lastActiveAt = Date.now();
  }

  /**
   * Record token usage for current session
   */
  recordTokens(promptTokens: number, completionTokens: number): void {
    const state = this.getCurrentState();
    if (!state) return;

    recordTokenUsage(state, promptTokens, completionTokens);
    state.metadata.lastActiveAt = Date.now();
  }

  /**
   * Record tool call for current session
   */
  recordTool(
    toolName: string,
    args: unknown,
    result: unknown,
    durationMs: number,
    success: boolean,
    error?: string
  ): void {
    const state = this.getCurrentState();
    if (!state) return;

    recordToolCall(state, toolName, args, result, durationMs, success, error);
    state.metadata.lastActiveAt = Date.now();
  }

  /**
   * Update user preferences
   */
  updatePreferences(preferences: Partial<UserPreferences>): void {
    const state = this.getCurrentState();
    if (!state) return;

    state.userPreferences = { ...state.userPreferences, ...preferences };
    state.metadata.lastActiveAt = Date.now();
  }

  /**
   * Add research focus area
   */
  addResearchFocus(focus: Omit<ResearchFocus, "addedAt">): void {
    const state = this.getCurrentState();
    if (!state) return;

    state.researchFocusAreas.push({
      ...focus,
      addedAt: Date.now(),
    });
    state.metadata.lastActiveAt = Date.now();
  }

  /**
   * Remove research focus area
   */
  removeResearchFocus(topic: string): void {
    const state = this.getCurrentState();
    if (!state) return;

    state.researchFocusAreas = state.researchFocusAreas.filter(
      (f) => f.topic !== topic
    );
    state.metadata.lastActiveAt = Date.now();
  }

  /**
   * Update active context
   */
  updateContext(context: Partial<ActiveContext>): void {
    const state = this.getCurrentState();
    if (!state) return;

    state.activeContext = { ...state.activeContext, ...context };
    state.metadata.lastActiveAt = Date.now();
  }

  /**
   * Set current symbol being analyzed
   */
  setCurrentSymbol(symbol: string | undefined): void {
    this.updateContext({ currentSymbol: symbol });
  }

  /**
   * Set current sector being analyzed
   */
  setCurrentSector(sector: string | undefined): void {
    this.updateContext({ currentSector: sector });
  }

  /**
   * Mark portfolio as loaded
   */
  setPortfolioLoaded(loaded: boolean): void {
    this.updateContext({ portfolioLoaded: loaded });
  }

  /**
   * Get session analytics
   */
  getAnalytics(sessionId?: string): ReturnType<typeof SessionAnalytics.analyze> | null {
    const id = sessionId || this.currentSessionId;
    if (!id) return null;

    const state = this.activeSessions.get(id);
    if (!state) return null;

    return SessionAnalytics.analyze(state);
  }

  /**
   * List all saved sessions
   */
  async listSessions(): Promise<SessionFileInfo[]> {
    return this.persistence.listSessions();
  }

  /**
   * Delete a session
   */
  async deleteSession(sessionId: string): Promise<boolean> {
    // End if active
    if (this.activeSessions.has(sessionId)) {
      await this.endSession(sessionId);
    }

    return this.persistence.delete(sessionId);
  }

  /**
   * Export session to file
   */
  async exportSession(sessionId: string, exportPath: string): Promise<boolean> {
    // Save current state first if active
    const state = this.activeSessions.get(sessionId);
    if (state) {
      await this.persistence.save(state);
    }

    return this.persistence.exportSession(sessionId, exportPath);
  }

  /**
   * Import session from file
   */
  async importSession(importPath: string): Promise<SessionState | null> {
    return this.persistence.importSession(importPath);
  }

  /**
   * Merge context from another session
   */
  async mergeFromSession(sourceSessionId: string): Promise<boolean> {
    const currentState = this.getCurrentState();
    if (!currentState) return false;

    const sourceState = await this.persistence.load(sourceSessionId);
    if (!sourceState) return false;

    // Merge research focus areas
    for (const focus of sourceState.researchFocusAreas) {
      if (!currentState.researchFocusAreas.some((f) => f.topic === focus.topic)) {
        currentState.researchFocusAreas.push(focus);
      }
    }

    // Merge watchlist
    const watchlistSet = new Set(currentState.userPreferences.watchlist);
    for (const symbol of sourceState.userPreferences.watchlist) {
      watchlistSet.add(symbol);
    }
    currentState.userPreferences.watchlist = Array.from(watchlistSet);

    // Merge insights
    const insightSet = new Set(currentState.activeContext.insights);
    for (const insight of sourceState.activeContext.insights) {
      insightSet.add(insight);
    }
    currentState.activeContext.insights = Array.from(insightSet);

    currentState.metadata.lastActiveAt = Date.now();
    return true;
  }

  /**
   * Create backup of current session
   */
  async backupCurrentSession(): Promise<string | null> {
    if (!this.currentSessionId) return null;

    // Save latest state first
    const state = this.getCurrentState();
    if (state) {
      await this.persistence.save(state);
    }

    return this.persistence.backup(this.currentSessionId);
  }

  /**
   * Clean up old sessions
   */
  async cleanupOldSessions(): Promise<number> {
    return this.persistence.cleanupOldSessions();
  }

  /**
   * Save current session state immediately
   */
  async saveCurrentSession(): Promise<void> {
    const state = this.getCurrentState();
    if (state) {
      await this.persistence.save(state);
    }
  }

  /**
   * Shutdown manager and save all sessions
   */
  async shutdown(): Promise<void> {
    this.persistence.stopAutoSave();

    for (const [id, state] of this.activeSessions) {
      state.metadata.status = "paused";
      await this.persistence.save(state);
    }

    this.activeSessions.clear();
    this.currentSessionId = null;
  }
}
