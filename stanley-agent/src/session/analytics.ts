/**
 * Session Analytics
 *
 * Tracks and analyzes session metrics and patterns.
 */

import {
  SessionState,
  ToolCallRecord,
  TokenUsage,
  calculateTotalTokenUsage,
  getToolCallFrequency,
  extractTopicsFromConversation,
} from "./state";

/**
 * Session analytics summary
 */
export interface SessionAnalytics {
  sessionId: string;
  duration: {
    startTime: number;
    endTime: number;
    durationMs: number;
    durationFormatted: string;
  };
  tokens: {
    totalPromptTokens: number;
    totalCompletionTokens: number;
    totalTokens: number;
    averageTokensPerMessage: number;
    tokensPerMinute: number;
  };
  conversation: {
    totalMessages: number;
    userMessages: number;
    assistantMessages: number;
    averageMessageLength: number;
  };
  tools: {
    totalCalls: number;
    uniqueTools: number;
    successRate: number;
    averageDurationMs: number;
    mostUsedTools: Array<{ tool: string; count: number }>;
    failedCalls: number;
  };
  topics: {
    discussed: string[];
    symbolsAnalyzed: string[];
    sectorsAnalyzed: string[];
  };
  insights: {
    generated: number;
    researchAreas: number;
    pendingActions: number;
  };
}

/**
 * Tool performance metrics
 */
export interface ToolPerformanceMetrics {
  toolName: string;
  callCount: number;
  successCount: number;
  failureCount: number;
  successRate: number;
  averageDurationMs: number;
  minDurationMs: number;
  maxDurationMs: number;
  totalDurationMs: number;
}

/**
 * Time-based analytics
 */
export interface TimeBasedAnalytics {
  messagesByHour: Map<number, number>;
  toolCallsByHour: Map<number, number>;
  tokensByHour: Map<number, number>;
  peakActivityHour: number;
  activityDistribution: Array<{ hour: number; activity: number }>;
}

/**
 * Session comparison
 */
export interface SessionComparison {
  sessions: string[];
  totalTokens: number[];
  totalMessages: number[];
  toolCalls: number[];
  durations: number[];
  averageTokensPerMessage: number[];
}

/**
 * Analytics tracker for session metrics
 */
export class SessionAnalytics {
  /**
   * Generate comprehensive analytics for a session
   */
  static analyze(state: SessionState): SessionAnalytics {
    const now = Date.now();
    const startTime = state.metadata.createdAt;
    const endTime = state.metadata.endedAt || state.metadata.lastActiveAt;
    const durationMs = endTime - startTime;

    // Token analytics
    const tokenTotals = calculateTotalTokenUsage(state);
    const durationMinutes = durationMs / 1000 / 60;

    // Conversation analytics
    const userMessages = state.conversationHistory.filter(
      (m) => m.role === "user"
    ).length;
    const assistantMessages = state.conversationHistory.filter(
      (m) => m.role === "assistant"
    ).length;
    const totalMessageLength = state.conversationHistory.reduce(
      (sum, m) => sum + (typeof m.content === "string" ? m.content.length : 0),
      0
    );

    // Tool analytics
    const toolFrequency = getToolCallFrequency(state);
    const successfulCalls = state.toolCallHistory.filter((c) => c.success).length;
    const totalDuration = state.toolCallHistory.reduce(
      (sum, c) => sum + c.durationMs,
      0
    );

    // Topics
    const topics = extractTopicsFromConversation(state);
    const symbols = topics
      .filter((t) => t.startsWith("Symbol: "))
      .map((t) => t.replace("Symbol: ", ""));
    const sectors = topics
      .filter((t) => t.startsWith("Sector: "))
      .map((t) => t.replace("Sector: ", ""));

    // Most used tools
    const sortedTools = Array.from(toolFrequency.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([tool, count]) => ({ tool, count }));

    return {
      sessionId: state.metadata.id,
      duration: {
        startTime,
        endTime,
        durationMs,
        durationFormatted: this.formatDuration(durationMs),
      },
      tokens: {
        ...tokenTotals,
        averageTokensPerMessage:
          state.conversationHistory.length > 0
            ? tokenTotals.totalTokens / state.conversationHistory.length
            : 0,
        tokensPerMinute:
          durationMinutes > 0 ? tokenTotals.totalTokens / durationMinutes : 0,
      },
      conversation: {
        totalMessages: state.conversationHistory.length,
        userMessages,
        assistantMessages,
        averageMessageLength:
          state.conversationHistory.length > 0
            ? totalMessageLength / state.conversationHistory.length
            : 0,
      },
      tools: {
        totalCalls: state.toolCallHistory.length,
        uniqueTools: toolFrequency.size,
        successRate:
          state.toolCallHistory.length > 0
            ? successfulCalls / state.toolCallHistory.length
            : 1,
        averageDurationMs:
          state.toolCallHistory.length > 0
            ? totalDuration / state.toolCallHistory.length
            : 0,
        mostUsedTools: sortedTools,
        failedCalls: state.toolCallHistory.length - successfulCalls,
      },
      topics: {
        discussed: topics,
        symbolsAnalyzed: symbols,
        sectorsAnalyzed: sectors,
      },
      insights: {
        generated: state.activeContext.insights.length,
        researchAreas: state.researchFocusAreas.length,
        pendingActions: state.activeContext.pendingActions.length,
      },
    };
  }

  /**
   * Get detailed tool performance metrics
   */
  static getToolPerformance(state: SessionState): ToolPerformanceMetrics[] {
    const toolMetrics = new Map<
      string,
      {
        calls: ToolCallRecord[];
        successCount: number;
        failureCount: number;
      }
    >();

    for (const call of state.toolCallHistory) {
      let metrics = toolMetrics.get(call.toolName);
      if (!metrics) {
        metrics = { calls: [], successCount: 0, failureCount: 0 };
        toolMetrics.set(call.toolName, metrics);
      }

      metrics.calls.push(call);
      if (call.success) {
        metrics.successCount++;
      } else {
        metrics.failureCount++;
      }
    }

    return Array.from(toolMetrics.entries()).map(([toolName, metrics]) => {
      const durations = metrics.calls.map((c) => c.durationMs);
      const totalDuration = durations.reduce((sum, d) => sum + d, 0);

      return {
        toolName,
        callCount: metrics.calls.length,
        successCount: metrics.successCount,
        failureCount: metrics.failureCount,
        successRate:
          metrics.calls.length > 0
            ? metrics.successCount / metrics.calls.length
            : 1,
        averageDurationMs:
          metrics.calls.length > 0 ? totalDuration / metrics.calls.length : 0,
        minDurationMs: durations.length > 0 ? Math.min(...durations) : 0,
        maxDurationMs: durations.length > 0 ? Math.max(...durations) : 0,
        totalDurationMs: totalDuration,
      };
    });
  }

  /**
   * Get time-based analytics
   */
  static getTimeBasedAnalytics(state: SessionState): TimeBasedAnalytics {
    const messagesByHour = new Map<number, number>();
    const toolCallsByHour = new Map<number, number>();
    const tokensByHour = new Map<number, number>();

    // Initialize all hours
    for (let h = 0; h < 24; h++) {
      messagesByHour.set(h, 0);
      toolCallsByHour.set(h, 0);
      tokensByHour.set(h, 0);
    }

    // Count tool calls by hour
    for (const call of state.toolCallHistory) {
      const hour = new Date(call.timestamp).getHours();
      toolCallsByHour.set(hour, (toolCallsByHour.get(hour) || 0) + 1);
    }

    // Count tokens by hour
    for (const usage of state.tokenUsage) {
      const hour = new Date(usage.timestamp).getHours();
      tokensByHour.set(hour, (tokensByHour.get(hour) || 0) + usage.totalTokens);
    }

    // Find peak activity hour
    let peakActivityHour = 0;
    let peakActivity = 0;

    for (let h = 0; h < 24; h++) {
      const activity =
        (toolCallsByHour.get(h) || 0) +
        (messagesByHour.get(h) || 0) * 2;
      if (activity > peakActivity) {
        peakActivity = activity;
        peakActivityHour = h;
      }
    }

    // Create activity distribution
    const activityDistribution: Array<{ hour: number; activity: number }> = [];
    for (let h = 0; h < 24; h++) {
      activityDistribution.push({
        hour: h,
        activity:
          (toolCallsByHour.get(h) || 0) + (messagesByHour.get(h) || 0) * 2,
      });
    }

    return {
      messagesByHour,
      toolCallsByHour,
      tokensByHour,
      peakActivityHour,
      activityDistribution,
    };
  }

  /**
   * Compare multiple sessions
   */
  static compareSessions(sessions: SessionState[]): SessionComparison {
    return {
      sessions: sessions.map((s) => s.metadata.id),
      totalTokens: sessions.map((s) => calculateTotalTokenUsage(s).totalTokens),
      totalMessages: sessions.map((s) => s.conversationHistory.length),
      toolCalls: sessions.map((s) => s.toolCallHistory.length),
      durations: sessions.map(
        (s) =>
          (s.metadata.endedAt || s.metadata.lastActiveAt) - s.metadata.createdAt
      ),
      averageTokensPerMessage: sessions.map((s) => {
        const tokens = calculateTotalTokenUsage(s).totalTokens;
        return s.conversationHistory.length > 0
          ? tokens / s.conversationHistory.length
          : 0;
      }),
    };
  }

  /**
   * Format duration for display
   */
  static formatDuration(ms: number): string {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) {
      return `${hours}h ${minutes % 60}m ${seconds % 60}s`;
    }
    if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`;
    }
    return `${seconds}s`;
  }

  /**
   * Generate analytics summary text
   */
  static generateSummary(analytics: SessionAnalytics): string {
    const lines: string[] = [
      `Session Analytics: ${analytics.sessionId}`,
      `---`,
      `Duration: ${analytics.duration.durationFormatted}`,
      ``,
      `Conversation:`,
      `  - Total messages: ${analytics.conversation.totalMessages}`,
      `  - User messages: ${analytics.conversation.userMessages}`,
      `  - Assistant messages: ${analytics.conversation.assistantMessages}`,
      ``,
      `Token Usage:`,
      `  - Total tokens: ${analytics.tokens.totalTokens.toLocaleString()}`,
      `  - Prompt tokens: ${analytics.tokens.totalPromptTokens.toLocaleString()}`,
      `  - Completion tokens: ${analytics.tokens.totalCompletionTokens.toLocaleString()}`,
      `  - Avg per message: ${analytics.tokens.averageTokensPerMessage.toFixed(1)}`,
      ``,
      `Tool Usage:`,
      `  - Total calls: ${analytics.tools.totalCalls}`,
      `  - Unique tools: ${analytics.tools.uniqueTools}`,
      `  - Success rate: ${(analytics.tools.successRate * 100).toFixed(1)}%`,
      `  - Avg duration: ${analytics.tools.averageDurationMs.toFixed(0)}ms`,
    ];

    if (analytics.tools.mostUsedTools.length > 0) {
      lines.push(`  - Most used:`);
      for (const { tool, count } of analytics.tools.mostUsedTools.slice(0, 5)) {
        lines.push(`    - ${tool}: ${count} calls`);
      }
    }

    if (analytics.topics.symbolsAnalyzed.length > 0) {
      lines.push(``);
      lines.push(`Symbols Analyzed: ${analytics.topics.symbolsAnalyzed.join(", ")}`);
    }

    if (analytics.topics.sectorsAnalyzed.length > 0) {
      lines.push(`Sectors Analyzed: ${analytics.topics.sectorsAnalyzed.join(", ")}`);
    }

    return lines.join("\n");
  }
}

/**
 * Record a token usage event
 */
export function recordTokenUsage(
  state: SessionState,
  promptTokens: number,
  completionTokens: number
): void {
  state.tokenUsage.push({
    promptTokens,
    completionTokens,
    totalTokens: promptTokens + completionTokens,
    timestamp: Date.now(),
  });
}

/**
 * Record a tool call
 */
export function recordToolCall(
  state: SessionState,
  toolName: string,
  args: unknown,
  result: unknown,
  durationMs: number,
  success: boolean,
  error?: string
): void {
  state.toolCallHistory.push({
    id: `${toolName}-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
    toolName,
    args,
    result,
    timestamp: Date.now(),
    durationMs,
    success,
    error,
  });
}

/**
 * Add insight to session
 */
export function addInsight(state: SessionState, insight: string): void {
  if (!state.activeContext.insights.includes(insight)) {
    state.activeContext.insights.push(insight);
  }
}

/**
 * Add pending action
 */
export function addPendingAction(state: SessionState, action: string): void {
  if (!state.activeContext.pendingActions.includes(action)) {
    state.activeContext.pendingActions.push(action);
  }
}

/**
 * Complete pending action
 */
export function completePendingAction(state: SessionState, action: string): void {
  const index = state.activeContext.pendingActions.indexOf(action);
  if (index !== -1) {
    state.activeContext.pendingActions.splice(index, 1);
  }
}
