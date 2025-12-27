/**
 * Stanley Polling Fallback
 *
 * Polling-based fallback for API change detection when WebSocket
 * connection is unavailable or unreliable.
 */

import {
  syncEvents,
  agentEventFactory,
  SyncEventType,
  type SyncEvent,
} from "./events";

// =============================================================================
// Types
// =============================================================================

/**
 * Polling target configuration
 */
export interface PollingTarget {
  /** Unique identifier for this target */
  id: string;
  /** API endpoint to poll */
  endpoint: string;
  /** Polling interval in milliseconds */
  interval: number;
  /** Event type to emit on change */
  eventType: SyncEventType;
  /** Optional transform function for response data */
  transform?: (data: unknown) => unknown;
  /** Optional comparison function to detect changes */
  compare?: (prev: unknown, curr: unknown) => boolean;
  /** Whether this target is currently enabled */
  enabled: boolean;
}

/**
 * Polling state for a target
 */
interface PollingState {
  target: PollingTarget;
  lastValue: unknown;
  lastPollTime: string | null;
  lastError: string | null;
  pollCount: number;
  errorCount: number;
  intervalId: ReturnType<typeof setInterval> | null;
}

/**
 * Polling service configuration
 */
export interface PollingServiceConfig {
  /** Base URL for API requests */
  baseUrl: string;
  /** Default polling interval in milliseconds */
  defaultInterval?: number;
  /** Maximum consecutive errors before disabling target */
  maxConsecutiveErrors?: number;
  /** Request timeout in milliseconds */
  requestTimeout?: number;
  /** Whether to emit events on initial poll */
  emitOnInitialPoll?: boolean;
}

// =============================================================================
// Default Polling Targets
// =============================================================================

/**
 * Default polling targets for Stanley API
 */
export const DEFAULT_POLLING_TARGETS: PollingTarget[] = [
  {
    id: "portfolio",
    endpoint: "/api/portfolio-analytics",
    interval: 60000, // 1 minute
    eventType: SyncEventType.PORTFOLIO_UPDATE,
    enabled: true,
  },
  {
    id: "notes",
    endpoint: "/api/notes/recent",
    interval: 30000, // 30 seconds
    eventType: SyncEventType.NOTE_UPDATED,
    enabled: true,
  },
  {
    id: "trades",
    endpoint: "/api/trades?status=open",
    interval: 60000, // 1 minute
    eventType: SyncEventType.TRADE_OPENED,
    enabled: true,
  },
  {
    id: "alerts",
    endpoint: "/api/alerts/active",
    interval: 15000, // 15 seconds
    eventType: SyncEventType.ALERT_TRIGGERED,
    enabled: true,
  },
];

// =============================================================================
// Polling Service
// =============================================================================

/**
 * Polling service for API change detection
 */
export class PollingService {
  private config: Required<PollingServiceConfig>;
  private states: Map<string, PollingState> = new Map();
  private isRunning: boolean = false;

  constructor(config: PollingServiceConfig) {
    this.config = {
      baseUrl: config.baseUrl.replace(/\/$/, ""),
      defaultInterval: config.defaultInterval ?? 30000,
      maxConsecutiveErrors: config.maxConsecutiveErrors ?? 5,
      requestTimeout: config.requestTimeout ?? 10000,
      emitOnInitialPoll: config.emitOnInitialPoll ?? false,
    };
  }

  /**
   * Add a polling target
   */
  addTarget(target: PollingTarget): void {
    if (this.states.has(target.id)) {
      console.warn(`[Polling] Target already exists: ${target.id}`);
      return;
    }

    const state: PollingState = {
      target,
      lastValue: null,
      lastPollTime: null,
      lastError: null,
      pollCount: 0,
      errorCount: 0,
      intervalId: null,
    };

    this.states.set(target.id, state);

    // Start polling if service is running
    if (this.isRunning && target.enabled) {
      this.startTargetPolling(target.id);
    }
  }

  /**
   * Remove a polling target
   */
  removeTarget(targetId: string): boolean {
    const state = this.states.get(targetId);
    if (!state) {
      return false;
    }

    // Stop polling
    if (state.intervalId) {
      clearInterval(state.intervalId);
    }

    this.states.delete(targetId);
    return true;
  }

  /**
   * Enable/disable a target
   */
  setTargetEnabled(targetId: string, enabled: boolean): void {
    const state = this.states.get(targetId);
    if (!state) {
      console.warn(`[Polling] Unknown target: ${targetId}`);
      return;
    }

    state.target.enabled = enabled;

    if (this.isRunning) {
      if (enabled && !state.intervalId) {
        this.startTargetPolling(targetId);
      } else if (!enabled && state.intervalId) {
        clearInterval(state.intervalId);
        state.intervalId = null;
      }
    }
  }

  /**
   * Start polling for a specific target
   */
  private startTargetPolling(targetId: string): void {
    const state = this.states.get(targetId);
    if (!state || state.intervalId) {
      return;
    }

    // Initial poll
    this.pollTarget(targetId);

    // Set up interval
    state.intervalId = setInterval(
      () => this.pollTarget(targetId),
      state.target.interval
    );
  }

  /**
   * Poll a specific target
   */
  private async pollTarget(targetId: string): Promise<void> {
    const state = this.states.get(targetId);
    if (!state || !state.target.enabled) {
      return;
    }

    const { target } = state;
    const url = `${this.config.baseUrl}${target.endpoint}`;

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(
        () => controller.abort(),
        this.config.requestTimeout
      );

      const response = await fetch(url, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      const transformedData = target.transform ? target.transform(data) : data;

      state.pollCount++;
      state.lastPollTime = new Date().toISOString();
      state.errorCount = 0;
      state.lastError = null;

      // Check for changes
      const hasChanged = this.detectChange(state, transformedData);

      if (hasChanged || (state.pollCount === 1 && this.config.emitOnInitialPoll)) {
        state.lastValue = transformedData;
        this.emitChangeEvent(target, transformedData);
      }
    } catch (error) {
      state.errorCount++;
      state.lastError = error instanceof Error ? error.message : "Unknown error";

      console.error(`[Polling] Error polling ${targetId}:`, state.lastError);

      // Disable target if too many consecutive errors
      if (state.errorCount >= this.config.maxConsecutiveErrors) {
        console.warn(
          `[Polling] Disabling ${targetId} after ${state.errorCount} consecutive errors`
        );
        this.setTargetEnabled(targetId, false);

        // Emit error event
        syncEvents.emitSyncEvent({
          ...agentEventFactory.generic(SyncEventType.SYNC_ERROR),
          data: {
            targetId,
            error: state.lastError,
            errorCount: state.errorCount,
          },
        } as SyncEvent & { data: { targetId: string; error: string; errorCount: number } });
      }
    }
  }

  /**
   * Detect if data has changed
   */
  private detectChange(state: PollingState, newValue: unknown): boolean {
    if (state.lastValue === null) {
      return true;
    }

    // Use custom compare function if provided
    if (state.target.compare) {
      return !state.target.compare(state.lastValue, newValue);
    }

    // Default: JSON comparison
    return JSON.stringify(state.lastValue) !== JSON.stringify(newValue);
  }

  /**
   * Emit change event
   */
  private emitChangeEvent(target: PollingTarget, data: unknown): void {
    const event = {
      ...agentEventFactory.generic(target.eventType),
      data,
      source: "api" as const,
    };

    syncEvents.emitSyncEvent(event as SyncEvent);
  }

  /**
   * Start the polling service
   */
  start(): void {
    if (this.isRunning) {
      console.warn("[Polling] Service already running");
      return;
    }

    this.isRunning = true;

    for (const [targetId, state] of this.states) {
      if (state.target.enabled) {
        this.startTargetPolling(targetId);
      }
    }

    console.log(`[Polling] Service started with ${this.states.size} targets`);
  }

  /**
   * Stop the polling service
   */
  stop(): void {
    if (!this.isRunning) {
      return;
    }

    this.isRunning = false;

    for (const state of this.states.values()) {
      if (state.intervalId) {
        clearInterval(state.intervalId);
        state.intervalId = null;
      }
    }

    console.log("[Polling] Service stopped");
  }

  /**
   * Force poll all targets immediately
   */
  async pollAll(): Promise<void> {
    const promises = Array.from(this.states.keys()).map((id) =>
      this.pollTarget(id)
    );
    await Promise.allSettled(promises);
  }

  /**
   * Force poll a specific target immediately
   */
  async pollNow(targetId: string): Promise<void> {
    await this.pollTarget(targetId);
  }

  /**
   * Get polling status for all targets
   */
  getStatus(): Record<string, PollingTargetStatus> {
    const status: Record<string, PollingTargetStatus> = {};

    for (const [id, state] of this.states) {
      status[id] = {
        id,
        endpoint: state.target.endpoint,
        enabled: state.target.enabled,
        interval: state.target.interval,
        lastPollTime: state.lastPollTime,
        lastError: state.lastError,
        pollCount: state.pollCount,
        errorCount: state.errorCount,
        isPolling: state.intervalId !== null,
      };
    }

    return status;
  }

  /**
   * Check if service is running
   */
  isActive(): boolean {
    return this.isRunning;
  }
}

/**
 * Polling target status
 */
export interface PollingTargetStatus {
  id: string;
  endpoint: string;
  enabled: boolean;
  interval: number;
  lastPollTime: string | null;
  lastError: string | null;
  pollCount: number;
  errorCount: number;
  isPolling: boolean;
}

// =============================================================================
// Factory
// =============================================================================

let pollingInstance: PollingService | null = null;

/**
 * Get or create polling service instance
 */
export function getPollingService(
  config?: PollingServiceConfig
): PollingService {
  if (!pollingInstance && config) {
    pollingInstance = new PollingService(config);

    // Add default targets
    for (const target of DEFAULT_POLLING_TARGETS) {
      pollingInstance.addTarget(target);
    }
  }

  if (!pollingInstance) {
    throw new Error("Polling service not initialized. Provide config on first call.");
  }

  return pollingInstance;
}

/**
 * Start polling service with default config
 */
export function startPollingService(
  baseUrl: string = "http://localhost:8000"
): PollingService {
  const service = getPollingService({ baseUrl });
  service.start();
  return service;
}

/**
 * Stop polling service
 */
export function stopPollingService(): void {
  if (pollingInstance) {
    pollingInstance.stop();
    pollingInstance = null;
  }
}
