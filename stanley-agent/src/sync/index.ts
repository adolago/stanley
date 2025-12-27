/**
 * Stanley Sync Module
 *
 * Real-time context synchronization between Stanley agent and GUI components.
 * Provides WebSocket server for bidirectional communication and polling fallback.
 *
 * @module sync
 */

// =============================================================================
// Event System
// =============================================================================

export {
  // Event types and enums
  SyncEventType,
  // Event payloads
  type SyncEventPayload,
  type PortfolioUpdatePayload,
  type NoteSavedPayload,
  type ResearchCompletePayload,
  type AlertTriggeredPayload,
  type ViewOpenedPayload,
  type SymbolSelectedPayload,
  type AgentToolCallPayload,
  type SyncEvent,
  // Event emitter
  SyncEventEmitter,
  // Event factory
  SyncEventFactory,
  // Singleton instances
  syncEvents,
  agentEventFactory,
} from "./events";

// =============================================================================
// WebSocket Server
// =============================================================================

export {
  // Types
  WsMessageType,
  type WsMessage,
  type SubscribePayload,
  type StateSyncPayload,
  type ClientInfo,
  type WsServerConfig,
  // Classes
  MessageQueue,
  SyncWebSocketServer,
  // Factory functions
  getSyncServer,
  startSyncServer,
  stopSyncServer,
} from "./websocket";

// =============================================================================
// Polling Fallback
// =============================================================================

export {
  // Types
  type PollingTarget,
  type PollingServiceConfig,
  type PollingTargetStatus,
  // Constants
  DEFAULT_POLLING_TARGETS,
  // Classes
  PollingService,
  // Factory functions
  getPollingService,
  startPollingService,
  stopPollingService,
} from "./polling";

// =============================================================================
// Unified Sync Manager
// =============================================================================

import {
  syncEvents,
  agentEventFactory,
  SyncEventType,
  type SyncEvent,
} from "./events";
import { startSyncServer, stopSyncServer, type SyncWebSocketServer } from "./websocket";
import {
  startPollingService,
  stopPollingService,
  type PollingService,
} from "./polling";

/**
 * Sync manager configuration
 */
export interface SyncManagerConfig {
  /** WebSocket server port */
  wsPort?: number;
  /** WebSocket server host */
  wsHost?: string;
  /** Stanley API base URL for polling */
  apiBaseUrl?: string;
  /** Whether to enable WebSocket server */
  enableWebSocket?: boolean;
  /** Whether to enable polling fallback */
  enablePolling?: boolean;
  /** Polling interval override */
  pollingInterval?: number;
}

/**
 * Unified sync manager for coordinating WebSocket and polling
 */
export class SyncManager {
  private config: Required<SyncManagerConfig>;
  private wsServer: SyncWebSocketServer | null = null;
  private pollingService: PollingService | null = null;
  private isRunning: boolean = false;

  constructor(config: SyncManagerConfig = {}) {
    this.config = {
      wsPort: config.wsPort ?? 8765,
      wsHost: config.wsHost ?? "127.0.0.1",
      apiBaseUrl: config.apiBaseUrl ?? "http://localhost:8000",
      enableWebSocket: config.enableWebSocket ?? true,
      enablePolling: config.enablePolling ?? true,
      pollingInterval: config.pollingInterval ?? 30000,
    };
  }

  /**
   * Start sync services
   */
  start(): void {
    if (this.isRunning) {
      console.warn("[SyncManager] Already running");
      return;
    }

    console.log("[SyncManager] Starting sync services...");

    // Start WebSocket server
    if (this.config.enableWebSocket) {
      try {
        this.wsServer = startSyncServer(this.config.wsPort);
        console.log(`[SyncManager] WebSocket server started on port ${this.config.wsPort}`);
      } catch (error) {
        console.error("[SyncManager] Failed to start WebSocket server:", error);
      }
    }

    // Start polling service
    if (this.config.enablePolling) {
      try {
        this.pollingService = startPollingService(this.config.apiBaseUrl);
        console.log(`[SyncManager] Polling service started for ${this.config.apiBaseUrl}`);
      } catch (error) {
        console.error("[SyncManager] Failed to start polling service:", error);
      }
    }

    this.isRunning = true;
    console.log("[SyncManager] Sync services started");
  }

  /**
   * Stop sync services
   */
  stop(): void {
    if (!this.isRunning) {
      return;
    }

    console.log("[SyncManager] Stopping sync services...");

    if (this.wsServer) {
      stopSyncServer();
      this.wsServer = null;
    }

    if (this.pollingService) {
      stopPollingService();
      this.pollingService = null;
    }

    this.isRunning = false;
    console.log("[SyncManager] Sync services stopped");
  }

  /**
   * Emit a sync event
   */
  emit(event: SyncEvent): void {
    syncEvents.emitSyncEvent(event);
  }

  /**
   * Subscribe to sync events
   */
  on(eventType: SyncEventType, handler: (event: SyncEvent) => void): void {
    syncEvents.onSyncEvent(eventType, handler);
  }

  /**
   * Subscribe to all sync events
   */
  onAll(handler: (event: SyncEvent) => void): void {
    syncEvents.onAllEvents(handler);
  }

  /**
   * Get event factory for creating events
   */
  get events() {
    return agentEventFactory;
  }

  /**
   * Get sync event emitter
   */
  get emitter() {
    return syncEvents;
  }

  /**
   * Get WebSocket server instance
   */
  get ws(): SyncWebSocketServer | null {
    return this.wsServer;
  }

  /**
   * Get polling service instance
   */
  get polling(): PollingService | null {
    return this.pollingService;
  }

  /**
   * Check if sync manager is running
   */
  isActive(): boolean {
    return this.isRunning;
  }

  /**
   * Get sync status
   */
  getStatus(): SyncStatus {
    return {
      isRunning: this.isRunning,
      webSocket: {
        enabled: this.config.enableWebSocket,
        running: this.wsServer !== null,
        clientCount: this.wsServer?.getClientCount() ?? 0,
        port: this.config.wsPort,
      },
      polling: {
        enabled: this.config.enablePolling,
        running: this.pollingService?.isActive() ?? false,
        targets: this.pollingService?.getStatus() ?? {},
      },
      eventHistory: syncEvents.getEventHistory(10),
    };
  }
}

/**
 * Sync status information
 */
export interface SyncStatus {
  isRunning: boolean;
  webSocket: {
    enabled: boolean;
    running: boolean;
    clientCount: number;
    port: number;
  };
  polling: {
    enabled: boolean;
    running: boolean;
    targets: Record<string, unknown>;
  };
  eventHistory: SyncEvent[];
}

// =============================================================================
// Singleton Instance
// =============================================================================

let syncManagerInstance: SyncManager | null = null;

/**
 * Get or create sync manager instance
 */
export function getSyncManager(config?: SyncManagerConfig): SyncManager {
  if (!syncManagerInstance) {
    syncManagerInstance = new SyncManager(config);
  }
  return syncManagerInstance;
}

/**
 * Start sync manager with default config
 */
export function startSync(config?: SyncManagerConfig): SyncManager {
  const manager = getSyncManager(config);
  manager.start();
  return manager;
}

/**
 * Stop sync manager
 */
export function stopSync(): void {
  if (syncManagerInstance) {
    syncManagerInstance.stop();
    syncManagerInstance = null;
  }
}
