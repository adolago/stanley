/**
 * Stanley WebSocket Server
 *
 * WebSocket server for real-time bidirectional communication between
 * the Stanley agent and GUI clients.
 */

import {
  syncEvents,
  agentEventFactory,
  SyncEventType,
  type SyncEvent,
  type SyncEventPayload,
} from "./events";

// =============================================================================
// Types
// =============================================================================

/**
 * WebSocket message types
 */
export enum WsMessageType {
  // Control messages
  PING = "ping",
  PONG = "pong",
  SUBSCRIBE = "subscribe",
  UNSUBSCRIBE = "unsubscribe",
  ACK = "ack",
  ERROR = "error",

  // Data messages
  EVENT = "event",
  BATCH = "batch",
  STATE_SYNC = "state_sync",
  STATE_REQUEST = "state_request",
}

/**
 * WebSocket message structure
 */
export interface WsMessage {
  type: WsMessageType;
  id: string;
  timestamp: string;
  payload?: unknown;
}

/**
 * Subscribe message payload
 */
export interface SubscribePayload {
  eventTypes: SyncEventType[];
}

/**
 * State sync payload
 */
export interface StateSyncPayload {
  currentView?: string;
  selectedSymbol?: string;
  portfolioSymbols?: string[];
  recentEvents: SyncEvent[];
}

/**
 * Client connection info
 */
export interface ClientInfo {
  id: string;
  connectedAt: string;
  lastPing: string;
  subscriptions: Set<SyncEventType>;
  messageQueue: WsMessage[];
  isAlive: boolean;
}

/**
 * WebSocket server configuration
 */
export interface WsServerConfig {
  port: number;
  host?: string;
  pingInterval?: number;
  maxMessageQueue?: number;
  enableCompression?: boolean;
}

// =============================================================================
// Message Queue
// =============================================================================

/**
 * Message queue for offline periods
 */
export class MessageQueue {
  private queue: WsMessage[] = [];
  private maxSize: number;

  constructor(maxSize: number = 1000) {
    this.maxSize = maxSize;
  }

  /**
   * Add message to queue
   */
  enqueue(message: WsMessage): void {
    this.queue.push(message);
    if (this.queue.length > this.maxSize) {
      this.queue.shift();
    }
  }

  /**
   * Get and clear all queued messages
   */
  drain(): WsMessage[] {
    const messages = [...this.queue];
    this.queue = [];
    return messages;
  }

  /**
   * Get messages since a timestamp
   */
  getSince(timestamp: string): WsMessage[] {
    return this.queue.filter((m) => m.timestamp > timestamp);
  }

  /**
   * Get queue size
   */
  size(): number {
    return this.queue.length;
  }

  /**
   * Clear queue
   */
  clear(): void {
    this.queue = [];
  }
}

// =============================================================================
// WebSocket Server
// =============================================================================

/**
 * WebSocket server for Stanley sync
 */
export class SyncWebSocketServer {
  private server: ReturnType<typeof Bun.serve> | null = null;
  private clients: Map<string, ClientInfo> = new Map();
  private messageQueue: MessageQueue;
  private config: Required<WsServerConfig>;
  private pingIntervalId: ReturnType<typeof setInterval> | null = null;
  private messageCounter: number = 0;

  constructor(config: WsServerConfig) {
    this.config = {
      host: config.host ?? "127.0.0.1",
      port: config.port,
      pingInterval: config.pingInterval ?? 30000,
      maxMessageQueue: config.maxMessageQueue ?? 1000,
      enableCompression: config.enableCompression ?? true,
    };
    this.messageQueue = new MessageQueue(this.config.maxMessageQueue);

    // Subscribe to all sync events and broadcast to clients
    syncEvents.onAllEvents((event) => this.broadcastEvent(event));
  }

  /**
   * Generate unique message ID
   */
  private generateMessageId(): string {
    return `msg-${Date.now()}-${++this.messageCounter}`;
  }

  /**
   * Create WebSocket message
   */
  private createMessage(type: WsMessageType, payload?: unknown): WsMessage {
    return {
      type,
      id: this.generateMessageId(),
      timestamp: new Date().toISOString(),
      payload,
    };
  }

  /**
   * Start the WebSocket server
   */
  start(): void {
    const self = this;

    this.server = Bun.serve({
      hostname: this.config.host,
      port: this.config.port,

      fetch(req, server) {
        const url = new URL(req.url);

        // Handle WebSocket upgrade
        if (url.pathname === "/ws" || url.pathname === "/sync") {
          const upgraded = server.upgrade(req, {
            data: {
              clientId: `client-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
            },
          });

          if (!upgraded) {
            return new Response("WebSocket upgrade failed", { status: 400 });
          }
          return undefined;
        }

        // Health check endpoint
        if (url.pathname === "/health") {
          return new Response(
            JSON.stringify({
              status: "ok",
              clients: self.clients.size,
              uptime: process.uptime(),
            }),
            {
              headers: { "Content-Type": "application/json" },
            }
          );
        }

        return new Response("Not Found", { status: 404 });
      },

      websocket: {
        open(ws) {
          const clientId = (ws.data as { clientId: string }).clientId;

          // Register client
          self.clients.set(clientId, {
            id: clientId,
            connectedAt: new Date().toISOString(),
            lastPing: new Date().toISOString(),
            subscriptions: new Set(),
            messageQueue: [],
            isAlive: true,
          });

          console.log(`[WS] Client connected: ${clientId}`);

          // Emit connection event
          syncEvents.emitSyncEvent({
            ...agentEventFactory.generic(SyncEventType.CLIENT_CONNECTED),
            data: { clientId },
          } as SyncEventPayload & { data: { clientId: string } });

          // Send welcome message with queued events
          const welcomeMsg = self.createMessage(WsMessageType.STATE_SYNC, {
            clientId,
            serverTime: new Date().toISOString(),
            recentEvents: syncEvents.getEventHistory(20),
          });
          ws.send(JSON.stringify(welcomeMsg));
        },

        message(ws, message) {
          const clientId = (ws.data as { clientId: string }).clientId;
          const client = self.clients.get(clientId);

          if (!client) {
            console.error(`[WS] Unknown client: ${clientId}`);
            return;
          }

          try {
            const msg: WsMessage =
              typeof message === "string"
                ? JSON.parse(message)
                : JSON.parse(new TextDecoder().decode(message as Buffer));

            self.handleMessage(ws, clientId, msg);
          } catch (error) {
            console.error(`[WS] Failed to parse message from ${clientId}:`, error);
            ws.send(
              JSON.stringify(
                self.createMessage(WsMessageType.ERROR, {
                  error: "Invalid message format",
                })
              )
            );
          }
        },

        close(ws, code, reason) {
          const clientId = (ws.data as { clientId: string }).clientId;

          console.log(`[WS] Client disconnected: ${clientId} (code: ${code}, reason: ${reason})`);

          // Remove client
          self.clients.delete(clientId);

          // Emit disconnection event
          syncEvents.emitSyncEvent({
            ...agentEventFactory.generic(SyncEventType.CLIENT_DISCONNECTED),
            data: { clientId, code, reason: reason || "unknown" },
          } as SyncEventPayload & { data: { clientId: string; code: number; reason: string } });
        },

        drain(ws) {
          const clientId = (ws.data as { clientId: string }).clientId;
          const client = self.clients.get(clientId);

          if (client && client.messageQueue.length > 0) {
            // Send queued messages
            for (const msg of client.messageQueue) {
              ws.send(JSON.stringify(msg));
            }
            client.messageQueue = [];
          }
        },
      },
    });

    // Start ping interval
    this.pingIntervalId = setInterval(() => this.pingClients(), this.config.pingInterval);

    console.log(`[WS] Server started on ws://${this.config.host}:${this.config.port}`);
  }

  /**
   * Handle incoming WebSocket message
   */
  private handleMessage(
    ws: { send: (data: string) => void },
    clientId: string,
    message: WsMessage
  ): void {
    const client = this.clients.get(clientId);
    if (!client) return;

    switch (message.type) {
      case WsMessageType.PING:
        client.lastPing = new Date().toISOString();
        client.isAlive = true;
        ws.send(
          JSON.stringify(
            this.createMessage(WsMessageType.PONG, { clientTime: message.timestamp })
          )
        );
        break;

      case WsMessageType.PONG:
        client.lastPing = new Date().toISOString();
        client.isAlive = true;
        break;

      case WsMessageType.SUBSCRIBE:
        const subPayload = message.payload as SubscribePayload;
        if (subPayload?.eventTypes) {
          for (const eventType of subPayload.eventTypes) {
            client.subscriptions.add(eventType);
          }
        }
        ws.send(
          JSON.stringify(
            this.createMessage(WsMessageType.ACK, {
              originalId: message.id,
              subscriptions: Array.from(client.subscriptions),
            })
          )
        );
        break;

      case WsMessageType.UNSUBSCRIBE:
        const unsubPayload = message.payload as SubscribePayload;
        if (unsubPayload?.eventTypes) {
          for (const eventType of unsubPayload.eventTypes) {
            client.subscriptions.delete(eventType);
          }
        }
        ws.send(
          JSON.stringify(
            this.createMessage(WsMessageType.ACK, {
              originalId: message.id,
              subscriptions: Array.from(client.subscriptions),
            })
          )
        );
        break;

      case WsMessageType.STATE_REQUEST:
        ws.send(
          JSON.stringify(
            this.createMessage(WsMessageType.STATE_SYNC, {
              recentEvents: syncEvents.getEventHistory(50),
            })
          )
        );
        break;

      case WsMessageType.EVENT:
        // Handle event from GUI
        const eventPayload = message.payload as SyncEvent;
        if (eventPayload) {
          // Re-emit event from GUI source
          syncEvents.emitSyncEvent({
            ...eventPayload,
            source: "gui",
          });
        }
        break;

      default:
        console.warn(`[WS] Unknown message type: ${message.type}`);
    }
  }

  /**
   * Broadcast event to all subscribed clients
   */
  private broadcastEvent(event: SyncEvent): void {
    const message = this.createMessage(WsMessageType.EVENT, event);

    // Queue message for offline clients
    this.messageQueue.enqueue(message);

    // Broadcast to connected clients
    if (this.server) {
      this.server.publish("sync", JSON.stringify(message));
    }
  }

  /**
   * Send message to specific client
   */
  sendToClient(clientId: string, message: WsMessage): boolean {
    const client = this.clients.get(clientId);
    if (!client) {
      return false;
    }

    // Queue if not connected
    client.messageQueue.push(message);
    return true;
  }

  /**
   * Ping all clients to check connectivity
   */
  private pingClients(): void {
    const now = new Date();
    const pingMessage = this.createMessage(WsMessageType.PING);

    for (const [clientId, client] of this.clients) {
      // Check if client is stale (no ping response in 2 intervals)
      const lastPingTime = new Date(client.lastPing).getTime();
      const staleDuration = this.config.pingInterval * 2;

      if (now.getTime() - lastPingTime > staleDuration) {
        if (!client.isAlive) {
          console.log(`[WS] Removing stale client: ${clientId}`);
          this.clients.delete(clientId);
          continue;
        }
        client.isAlive = false;
      }

      // Send ping through server publish
      if (this.server) {
        this.server.publish("sync", JSON.stringify(pingMessage));
      }
    }
  }

  /**
   * Get connected client count
   */
  getClientCount(): number {
    return this.clients.size;
  }

  /**
   * Get all connected client IDs
   */
  getClientIds(): string[] {
    return Array.from(this.clients.keys());
  }

  /**
   * Get client info
   */
  getClientInfo(clientId: string): ClientInfo | undefined {
    return this.clients.get(clientId);
  }

  /**
   * Stop the WebSocket server
   */
  stop(): void {
    if (this.pingIntervalId) {
      clearInterval(this.pingIntervalId);
      this.pingIntervalId = null;
    }

    if (this.server) {
      this.server.stop();
      this.server = null;
    }

    this.clients.clear();
    console.log("[WS] Server stopped");
  }
}

// =============================================================================
// Factory
// =============================================================================

let serverInstance: SyncWebSocketServer | null = null;

/**
 * Get or create WebSocket server instance
 */
export function getSyncServer(config?: WsServerConfig): SyncWebSocketServer {
  if (!serverInstance && config) {
    serverInstance = new SyncWebSocketServer(config);
  }
  if (!serverInstance) {
    throw new Error("WebSocket server not initialized. Provide config on first call.");
  }
  return serverInstance;
}

/**
 * Start WebSocket server with default config
 */
export function startSyncServer(port: number = 8765): SyncWebSocketServer {
  const server = getSyncServer({ port });
  server.start();
  return server;
}

/**
 * Stop WebSocket server
 */
export function stopSyncServer(): void {
  if (serverInstance) {
    serverInstance.stop();
    serverInstance = null;
  }
}
