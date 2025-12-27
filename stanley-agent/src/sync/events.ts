/**
 * Stanley Sync Events
 *
 * Event emitter system for real-time context synchronization between
 * Stanley agent and GUI components.
 */

import { EventEmitter } from "events";

// =============================================================================
// Event Types
// =============================================================================

/**
 * Types of sync events that can be emitted
 */
export enum SyncEventType {
  // Portfolio events
  PORTFOLIO_UPDATE = "portfolio_update",
  PORTFOLIO_HOLDING_ADDED = "portfolio_holding_added",
  PORTFOLIO_HOLDING_REMOVED = "portfolio_holding_removed",

  // Note events
  NOTE_SAVED = "note_saved",
  NOTE_DELETED = "note_deleted",
  NOTE_UPDATED = "note_updated",
  THESIS_CREATED = "thesis_created",
  TRADE_OPENED = "trade_opened",
  TRADE_CLOSED = "trade_closed",

  // Research events
  RESEARCH_COMPLETE = "research_complete",
  RESEARCH_STARTED = "research_started",
  RESEARCH_PROGRESS = "research_progress",

  // Alert events
  ALERT_TRIGGERED = "alert_triggered",
  ALERT_ACKNOWLEDGED = "alert_acknowledged",
  PRICE_ALERT = "price_alert",
  FLOW_ALERT = "flow_alert",

  // View events (GUI -> Agent)
  VIEW_OPENED = "view_opened",
  VIEW_CLOSED = "view_closed",
  SYMBOL_SELECTED = "symbol_selected",
  SYMBOL_DESELECTED = "symbol_deselected",

  // Agent events
  AGENT_QUERY_START = "agent_query_start",
  AGENT_QUERY_COMPLETE = "agent_query_complete",
  AGENT_TOOL_CALL = "agent_tool_call",
  AGENT_TOOL_RESULT = "agent_tool_result",
  AGENT_ERROR = "agent_error",

  // Connection events
  CLIENT_CONNECTED = "client_connected",
  CLIENT_DISCONNECTED = "client_disconnected",
  SYNC_ERROR = "sync_error",
}

/**
 * Base event payload
 */
export interface SyncEventPayload {
  /** Unique event ID */
  id: string;
  /** Event type */
  type: SyncEventType;
  /** Timestamp of event creation */
  timestamp: string;
  /** Source of the event (agent, gui, api) */
  source: "agent" | "gui" | "api";
  /** Optional correlation ID for tracking related events */
  correlationId?: string;
}

/**
 * Portfolio update event payload
 */
export interface PortfolioUpdatePayload extends SyncEventPayload {
  type: SyncEventType.PORTFOLIO_UPDATE;
  data: {
    totalValue: number;
    holdings: Array<{
      symbol: string;
      shares: number;
      value: number;
      weight: number;
      change?: number;
    }>;
    sectorExposure: Record<string, number>;
    riskMetrics?: {
      var95: number;
      beta: number;
      sharpeRatio: number;
    };
  };
}

/**
 * Note saved event payload
 */
export interface NoteSavedPayload extends SyncEventPayload {
  type: SyncEventType.NOTE_SAVED;
  data: {
    noteId: string;
    notePath: string;
    noteType: "thesis" | "trade" | "event" | "person" | "sector" | "general";
    title: string;
    symbol?: string;
    tags: string[];
    preview: string;
  };
}

/**
 * Research complete event payload
 */
export interface ResearchCompletePayload extends SyncEventPayload {
  type: SyncEventType.RESEARCH_COMPLETE;
  data: {
    queryId: string;
    symbol: string;
    researchType: "valuation" | "earnings" | "peer_comparison" | "full_report";
    summary: string;
    keyFindings: string[];
    recommendation?: string;
    confidence: number;
  };
}

/**
 * Alert triggered event payload
 */
export interface AlertTriggeredPayload extends SyncEventPayload {
  type: SyncEventType.ALERT_TRIGGERED;
  data: {
    alertId: string;
    alertType: "price" | "flow" | "institutional" | "dark_pool" | "custom";
    symbol: string;
    condition: string;
    currentValue: number;
    threshold: number;
    message: string;
    severity: "info" | "warning" | "critical";
  };
}

/**
 * View opened event payload (from GUI)
 */
export interface ViewOpenedPayload extends SyncEventPayload {
  type: SyncEventType.VIEW_OPENED;
  data: {
    viewName: string;
    viewType:
      | "dashboard"
      | "portfolio"
      | "research"
      | "notes"
      | "commodities"
      | "comparison"
      | "agent";
    context?: {
      selectedSymbol?: string;
      selectedSector?: string;
      filters?: Record<string, unknown>;
    };
  };
}

/**
 * Symbol selected event payload
 */
export interface SymbolSelectedPayload extends SyncEventPayload {
  type: SyncEventType.SYMBOL_SELECTED;
  data: {
    symbol: string;
    viewContext: string;
    previousSymbol?: string;
  };
}

/**
 * Agent tool call event payload
 */
export interface AgentToolCallPayload extends SyncEventPayload {
  type: SyncEventType.AGENT_TOOL_CALL;
  data: {
    toolName: string;
    toolId: string;
    arguments: Record<string, unknown>;
    status: "pending" | "running" | "complete" | "error";
  };
}

/**
 * Union type of all event payloads
 */
export type SyncEvent =
  | PortfolioUpdatePayload
  | NoteSavedPayload
  | ResearchCompletePayload
  | AlertTriggeredPayload
  | ViewOpenedPayload
  | SymbolSelectedPayload
  | AgentToolCallPayload
  | SyncEventPayload;

// =============================================================================
// Event Emitter
// =============================================================================

/**
 * Type-safe event emitter for sync events
 */
export class SyncEventEmitter extends EventEmitter {
  private eventHistory: SyncEvent[] = [];
  private maxHistorySize: number = 100;

  constructor(options?: { maxHistorySize?: number }) {
    super();
    this.maxHistorySize = options?.maxHistorySize ?? 100;
  }

  /**
   * Emit a sync event
   */
  emitSyncEvent<T extends SyncEvent>(event: T): boolean {
    // Add to history
    this.eventHistory.push(event);
    if (this.eventHistory.length > this.maxHistorySize) {
      this.eventHistory.shift();
    }

    // Emit on specific type channel
    this.emit(event.type, event);

    // Also emit on wildcard channel for subscribers who want all events
    this.emit("*", event);

    return true;
  }

  /**
   * Subscribe to a specific event type
   */
  onSyncEvent<T extends SyncEvent>(
    eventType: SyncEventType,
    handler: (event: T) => void
  ): this {
    return this.on(eventType, handler);
  }

  /**
   * Subscribe to all events
   */
  onAllEvents(handler: (event: SyncEvent) => void): this {
    return this.on("*", handler);
  }

  /**
   * Unsubscribe from a specific event type
   */
  offSyncEvent(eventType: SyncEventType, handler: (...args: unknown[]) => void): this {
    return this.off(eventType, handler);
  }

  /**
   * Get recent event history
   */
  getEventHistory(limit?: number): SyncEvent[] {
    const slice = limit ? -limit : undefined;
    return slice ? this.eventHistory.slice(slice) : [...this.eventHistory];
  }

  /**
   * Get events by type
   */
  getEventsByType(eventType: SyncEventType, limit?: number): SyncEvent[] {
    const filtered = this.eventHistory.filter((e) => e.type === eventType);
    return limit ? filtered.slice(-limit) : filtered;
  }

  /**
   * Clear event history
   */
  clearHistory(): void {
    this.eventHistory = [];
  }
}

// =============================================================================
// Event Factory
// =============================================================================

/**
 * Factory for creating sync events with proper structure
 */
export class SyncEventFactory {
  private source: "agent" | "gui" | "api";
  private eventCounter: number = 0;

  constructor(source: "agent" | "gui" | "api") {
    this.source = source;
  }

  /**
   * Generate unique event ID
   */
  private generateId(): string {
    const timestamp = Date.now();
    const counter = ++this.eventCounter;
    return `${this.source}-${timestamp}-${counter}`;
  }

  /**
   * Get current ISO timestamp
   */
  private getTimestamp(): string {
    return new Date().toISOString();
  }

  /**
   * Create base event payload
   */
  private createBase(
    type: SyncEventType,
    correlationId?: string
  ): SyncEventPayload {
    return {
      id: this.generateId(),
      type,
      timestamp: this.getTimestamp(),
      source: this.source,
      correlationId,
    };
  }

  /**
   * Create portfolio update event
   */
  portfolioUpdate(
    data: PortfolioUpdatePayload["data"],
    correlationId?: string
  ): PortfolioUpdatePayload {
    return {
      ...this.createBase(SyncEventType.PORTFOLIO_UPDATE, correlationId),
      type: SyncEventType.PORTFOLIO_UPDATE,
      data,
    };
  }

  /**
   * Create note saved event
   */
  noteSaved(
    data: NoteSavedPayload["data"],
    correlationId?: string
  ): NoteSavedPayload {
    return {
      ...this.createBase(SyncEventType.NOTE_SAVED, correlationId),
      type: SyncEventType.NOTE_SAVED,
      data,
    };
  }

  /**
   * Create research complete event
   */
  researchComplete(
    data: ResearchCompletePayload["data"],
    correlationId?: string
  ): ResearchCompletePayload {
    return {
      ...this.createBase(SyncEventType.RESEARCH_COMPLETE, correlationId),
      type: SyncEventType.RESEARCH_COMPLETE,
      data,
    };
  }

  /**
   * Create alert triggered event
   */
  alertTriggered(
    data: AlertTriggeredPayload["data"],
    correlationId?: string
  ): AlertTriggeredPayload {
    return {
      ...this.createBase(SyncEventType.ALERT_TRIGGERED, correlationId),
      type: SyncEventType.ALERT_TRIGGERED,
      data,
    };
  }

  /**
   * Create view opened event
   */
  viewOpened(
    data: ViewOpenedPayload["data"],
    correlationId?: string
  ): ViewOpenedPayload {
    return {
      ...this.createBase(SyncEventType.VIEW_OPENED, correlationId),
      type: SyncEventType.VIEW_OPENED,
      data,
    };
  }

  /**
   * Create symbol selected event
   */
  symbolSelected(
    data: SymbolSelectedPayload["data"],
    correlationId?: string
  ): SymbolSelectedPayload {
    return {
      ...this.createBase(SyncEventType.SYMBOL_SELECTED, correlationId),
      type: SyncEventType.SYMBOL_SELECTED,
      data,
    };
  }

  /**
   * Create agent tool call event
   */
  agentToolCall(
    data: AgentToolCallPayload["data"],
    correlationId?: string
  ): AgentToolCallPayload {
    return {
      ...this.createBase(SyncEventType.AGENT_TOOL_CALL, correlationId),
      type: SyncEventType.AGENT_TOOL_CALL,
      data,
    };
  }

  /**
   * Create generic event
   */
  generic(type: SyncEventType, correlationId?: string): SyncEventPayload {
    return this.createBase(type, correlationId);
  }
}

// =============================================================================
// Singleton Instance
// =============================================================================

/**
 * Global sync event emitter instance
 */
export const syncEvents = new SyncEventEmitter({ maxHistorySize: 500 });

/**
 * Event factory for agent-sourced events
 */
export const agentEventFactory = new SyncEventFactory("agent");
