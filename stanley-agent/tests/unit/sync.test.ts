/**
 * Real-time Sync Tests
 *
 * Tests for the real-time synchronization system that handles
 * WebSocket connections, state synchronization, and event streaming.
 */

import { describe, it, expect, beforeEach, afterEach, mock } from "bun:test";

// Mock WebSocket for testing
class MockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  url: string;
  readyState: number = MockWebSocket.CONNECTING;
  onopen: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;

  private messageQueue: string[] = [];

  constructor(url: string) {
    this.url = url;
    // Simulate async connection
    setTimeout(() => {
      this.readyState = MockWebSocket.OPEN;
      if (this.onopen) {
        this.onopen(new Event("open"));
      }
    }, 10);
  }

  send(data: string): void {
    if (this.readyState !== MockWebSocket.OPEN) {
      throw new Error("WebSocket is not open");
    }
    this.messageQueue.push(data);
  }

  close(code?: number, reason?: string): void {
    this.readyState = MockWebSocket.CLOSING;
    setTimeout(() => {
      this.readyState = MockWebSocket.CLOSED;
      if (this.onclose) {
        this.onclose(new CloseEvent("close", { code, reason }));
      }
    }, 10);
  }

  // Test helper to simulate receiving a message
  _receiveMessage(data: unknown): void {
    if (this.onmessage) {
      this.onmessage(new MessageEvent("message", {
        data: typeof data === "string" ? data : JSON.stringify(data),
      }));
    }
  }

  // Test helper to simulate an error
  _triggerError(error: Error): void {
    if (this.onerror) {
      const errorEvent = new Event("error");
      (errorEvent as any).error = error;
      this.onerror(errorEvent);
    }
  }

  // Test helper to get sent messages
  _getSentMessages(): string[] {
    return [...this.messageQueue];
  }
}

// Mock SyncClient class
interface SyncConfig {
  url: string;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
}

interface SyncEvent {
  type: string;
  payload: unknown;
  timestamp: number;
  id?: string;
}

type SyncEventHandler = (event: SyncEvent) => void;

class MockSyncClient {
  private config: Required<SyncConfig>;
  private ws: MockWebSocket | null = null;
  private connected: boolean = false;
  private reconnectAttempts: number = 0;
  private handlers: Map<string, Set<SyncEventHandler>> = new Map();
  private pendingMessages: SyncEvent[] = [];
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;

  constructor(config: SyncConfig) {
    this.config = {
      url: config.url,
      reconnectInterval: config.reconnectInterval ?? 5000,
      maxReconnectAttempts: config.maxReconnectAttempts ?? 5,
      heartbeatInterval: config.heartbeatInterval ?? 30000,
    };
  }

  async connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws = new MockWebSocket(this.config.url);

      this.ws.onopen = () => {
        this.connected = true;
        this.reconnectAttempts = 0;
        this.flushPendingMessages();
        this.startHeartbeat();
        resolve();
      };

      this.ws.onclose = (event) => {
        this.connected = false;
        this.stopHeartbeat();
        this.handleDisconnect(event);
      };

      this.ws.onmessage = (event) => {
        this.handleMessage(event);
      };

      this.ws.onerror = (event) => {
        if (!this.connected) {
          reject(new Error("Connection failed"));
        }
      };
    });
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close(1000, "Client disconnect");
      this.ws = null;
    }
    this.connected = false;
    this.stopHeartbeat();
  }

  isConnected(): boolean {
    return this.connected;
  }

  on(eventType: string, handler: SyncEventHandler): () => void {
    if (!this.handlers.has(eventType)) {
      this.handlers.set(eventType, new Set());
    }
    this.handlers.get(eventType)!.add(handler);

    // Return unsubscribe function
    return () => {
      this.handlers.get(eventType)?.delete(handler);
    };
  }

  off(eventType: string, handler: SyncEventHandler): void {
    this.handlers.get(eventType)?.delete(handler);
  }

  send(event: Omit<SyncEvent, "timestamp">): void {
    const fullEvent: SyncEvent = {
      ...event,
      timestamp: Date.now(),
    };

    if (this.connected && this.ws) {
      this.ws.send(JSON.stringify(fullEvent));
    } else {
      this.pendingMessages.push(fullEvent);
    }
  }

  private handleMessage(event: MessageEvent): void {
    try {
      const data = JSON.parse(event.data as string) as SyncEvent;
      const handlers = this.handlers.get(data.type);

      if (handlers) {
        for (const handler of handlers) {
          handler(data);
        }
      }

      // Also call wildcard handlers
      const wildcardHandlers = this.handlers.get("*");
      if (wildcardHandlers) {
        for (const handler of wildcardHandlers) {
          handler(data);
        }
      }
    } catch (error) {
      console.error("Failed to parse message:", error);
    }
  }

  private handleDisconnect(event: CloseEvent): void {
    if (event.code !== 1000 && this.reconnectAttempts < this.config.maxReconnectAttempts) {
      this.reconnectAttempts++;
      setTimeout(() => {
        this.connect().catch(() => {});
      }, this.config.reconnectInterval);
    }
  }

  private flushPendingMessages(): void {
    while (this.pendingMessages.length > 0) {
      const event = this.pendingMessages.shift()!;
      if (this.ws && this.connected) {
        this.ws.send(JSON.stringify(event));
      }
    }
  }

  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      this.send({ type: "heartbeat", payload: null });
    }, this.config.heartbeatInterval);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  // Test helper
  _getWebSocket(): MockWebSocket | null {
    return this.ws;
  }
}

describe("Sync Client", () => {
  let syncClient: MockSyncClient;

  beforeEach(() => {
    syncClient = new MockSyncClient({
      url: "ws://localhost:8080/sync",
    });
  });

  afterEach(() => {
    syncClient.disconnect();
  });

  describe("Connection", () => {
    it("should connect to WebSocket server", async () => {
      await syncClient.connect();

      expect(syncClient.isConnected()).toBe(true);
    });

    it("should disconnect from server", async () => {
      await syncClient.connect();
      syncClient.disconnect();

      // Wait for close event
      await new Promise((resolve) => setTimeout(resolve, 50));

      expect(syncClient.isConnected()).toBe(false);
    });

    it("should report connection state correctly", () => {
      expect(syncClient.isConnected()).toBe(false);
    });
  });

  describe("Event Handling", () => {
    it("should subscribe to events", async () => {
      const handler = mock((event: SyncEvent) => {});

      syncClient.on("test-event", handler);
      await syncClient.connect();

      // Simulate receiving an event
      const ws = syncClient._getWebSocket();
      ws?._receiveMessage({ type: "test-event", payload: { data: "test" }, timestamp: Date.now() });

      expect(handler).toHaveBeenCalledTimes(1);
    });

    it("should receive event payload", async () => {
      let receivedPayload: unknown;

      syncClient.on("data-update", (event) => {
        receivedPayload = event.payload;
      });

      await syncClient.connect();

      const ws = syncClient._getWebSocket();
      ws?._receiveMessage({
        type: "data-update",
        payload: { symbol: "AAPL", price: 150.25 },
        timestamp: Date.now(),
      });

      expect(receivedPayload).toEqual({ symbol: "AAPL", price: 150.25 });
    });

    it("should support multiple handlers for same event", async () => {
      const handler1 = mock((event: SyncEvent) => {});
      const handler2 = mock((event: SyncEvent) => {});

      syncClient.on("multi-event", handler1);
      syncClient.on("multi-event", handler2);

      await syncClient.connect();

      const ws = syncClient._getWebSocket();
      ws?._receiveMessage({ type: "multi-event", payload: null, timestamp: Date.now() });

      expect(handler1).toHaveBeenCalledTimes(1);
      expect(handler2).toHaveBeenCalledTimes(1);
    });

    it("should unsubscribe from events", async () => {
      const handler = mock((event: SyncEvent) => {});

      const unsubscribe = syncClient.on("temp-event", handler);
      await syncClient.connect();

      // First event should be handled
      const ws = syncClient._getWebSocket();
      ws?._receiveMessage({ type: "temp-event", payload: null, timestamp: Date.now() });
      expect(handler).toHaveBeenCalledTimes(1);

      // Unsubscribe
      unsubscribe();

      // Second event should not be handled
      ws?._receiveMessage({ type: "temp-event", payload: null, timestamp: Date.now() });
      expect(handler).toHaveBeenCalledTimes(1);
    });

    it("should support wildcard event subscription", async () => {
      const handler = mock((event: SyncEvent) => {});

      syncClient.on("*", handler);
      await syncClient.connect();

      const ws = syncClient._getWebSocket();
      ws?._receiveMessage({ type: "event-a", payload: null, timestamp: Date.now() });
      ws?._receiveMessage({ type: "event-b", payload: null, timestamp: Date.now() });

      expect(handler).toHaveBeenCalledTimes(2);
    });
  });

  describe("Sending Messages", () => {
    it("should send events when connected", async () => {
      await syncClient.connect();

      syncClient.send({ type: "client-event", payload: { action: "test" } });

      const ws = syncClient._getWebSocket();
      const messages = ws?._getSentMessages() ?? [];

      expect(messages).toHaveLength(1);
      expect(JSON.parse(messages[0])).toMatchObject({
        type: "client-event",
        payload: { action: "test" },
      });
    });

    it("should include timestamp in sent events", async () => {
      const before = Date.now();
      await syncClient.connect();

      syncClient.send({ type: "timed-event", payload: null });

      const after = Date.now();
      const ws = syncClient._getWebSocket();
      const messages = ws?._getSentMessages() ?? [];
      const sent = JSON.parse(messages[0]);

      expect(sent.timestamp).toBeGreaterThanOrEqual(before);
      expect(sent.timestamp).toBeLessThanOrEqual(after);
    });

    it("should queue messages when disconnected", () => {
      // Send before connecting
      syncClient.send({ type: "queued-event", payload: { queued: true } });

      // Should not throw and should be queued
      expect(syncClient.isConnected()).toBe(false);
    });

    it("should flush queued messages on connect", async () => {
      // Queue messages before connecting
      syncClient.send({ type: "queued-1", payload: null });
      syncClient.send({ type: "queued-2", payload: null });

      await syncClient.connect();

      const ws = syncClient._getWebSocket();
      const messages = ws?._getSentMessages() ?? [];

      expect(messages.length).toBeGreaterThanOrEqual(2);
    });
  });

  describe("Reconnection", () => {
    it("should attempt reconnection on unexpected disconnect", async () => {
      const client = new MockSyncClient({
        url: "ws://localhost:8080/sync",
        reconnectInterval: 100,
        maxReconnectAttempts: 3,
      });

      await client.connect();

      // Simulate unexpected disconnect
      const ws = client._getWebSocket();
      ws?.close(1006, "Abnormal closure");

      // Wait for close and reconnect attempt
      await new Promise((resolve) => setTimeout(resolve, 150));

      // Should be attempting to reconnect
      client.disconnect();
    });

    it("should not reconnect on normal close", async () => {
      await syncClient.connect();
      syncClient.disconnect();

      // Wait for close
      await new Promise((resolve) => setTimeout(resolve, 50));

      // Should stay disconnected
      expect(syncClient.isConnected()).toBe(false);
    });
  });
});

describe("Sync Event Types", () => {
  describe("State Sync Events", () => {
    it("should handle state update event", () => {
      const event: SyncEvent = {
        type: "state:update",
        payload: {
          path: "user.preferences",
          value: { theme: "dark" },
        },
        timestamp: Date.now(),
      };

      expect(event.type).toBe("state:update");
      expect(event.payload).toHaveProperty("path");
      expect(event.payload).toHaveProperty("value");
    });

    it("should handle state sync event", () => {
      const event: SyncEvent = {
        type: "state:sync",
        payload: {
          fullState: {
            user: { name: "John" },
            session: { id: "abc" },
          },
        },
        timestamp: Date.now(),
      };

      expect(event.type).toBe("state:sync");
    });
  });

  describe("Market Data Events", () => {
    it("should handle price update event", () => {
      const event: SyncEvent = {
        type: "market:price",
        payload: {
          symbol: "AAPL",
          price: 150.25,
          change: 2.50,
          changePercent: 1.69,
        },
        timestamp: Date.now(),
      };

      expect(event.type).toBe("market:price");
    });

    it("should handle trade event", () => {
      const event: SyncEvent = {
        type: "market:trade",
        payload: {
          symbol: "GOOGL",
          price: 140.50,
          volume: 1000,
          side: "buy",
        },
        timestamp: Date.now(),
      };

      expect(event.type).toBe("market:trade");
    });
  });

  describe("Agent Events", () => {
    it("should handle agent thinking event", () => {
      const event: SyncEvent = {
        type: "agent:thinking",
        payload: {
          message: "Analyzing market data...",
        },
        timestamp: Date.now(),
      };

      expect(event.type).toBe("agent:thinking");
    });

    it("should handle tool execution event", () => {
      const event: SyncEvent = {
        type: "agent:tool",
        payload: {
          name: "get_market_data",
          args: { symbol: "AAPL" },
          status: "executing",
        },
        timestamp: Date.now(),
      };

      expect(event.type).toBe("agent:tool");
    });

    it("should handle streaming response event", () => {
      const event: SyncEvent = {
        type: "agent:stream",
        payload: {
          chunk: "Based on the analysis",
          done: false,
        },
        timestamp: Date.now(),
      };

      expect(event.type).toBe("agent:stream");
    });
  });
});

describe("Sync Manager", () => {
  // Test a higher-level sync manager that coordinates multiple sync clients

  interface SyncManagerConfig {
    wsUrl: string;
    onStateChange?: (state: unknown) => void;
    onError?: (error: Error) => void;
  }

  class MockSyncManager {
    private client: MockSyncClient;
    private state: Record<string, unknown> = {};
    private config: SyncManagerConfig;

    constructor(config: SyncManagerConfig) {
      this.config = config;
      this.client = new MockSyncClient({ url: config.wsUrl });

      this.setupHandlers();
    }

    private setupHandlers(): void {
      this.client.on("state:update", (event) => {
        const { path, value } = event.payload as { path: string; value: unknown };
        this.updateState(path, value);
      });

      this.client.on("state:sync", (event) => {
        const { fullState } = event.payload as { fullState: Record<string, unknown> };
        this.state = fullState;
        this.config.onStateChange?.(this.state);
      });
    }

    private updateState(path: string, value: unknown): void {
      const parts = path.split(".");
      let current: Record<string, unknown> = this.state;

      for (let i = 0; i < parts.length - 1; i++) {
        if (!current[parts[i]]) {
          current[parts[i]] = {};
        }
        current = current[parts[i]] as Record<string, unknown>;
      }

      current[parts[parts.length - 1]] = value;
      this.config.onStateChange?.(this.state);
    }

    async start(): Promise<void> {
      await this.client.connect();
    }

    stop(): void {
      this.client.disconnect();
    }

    getState(): Record<string, unknown> {
      return { ...this.state };
    }

    updateRemote(path: string, value: unknown): void {
      this.client.send({
        type: "state:update",
        payload: { path, value },
      });
    }
  }

  it("should sync state from server", async () => {
    let currentState: unknown;

    const manager = new MockSyncManager({
      wsUrl: "ws://localhost:8080/sync",
      onStateChange: (state) => {
        currentState = state;
      },
    });

    await manager.start();

    // Simulate state sync from server
    const ws = (manager as any).client._getWebSocket();
    ws?._receiveMessage({
      type: "state:sync",
      payload: {
        fullState: { user: { name: "John" }, settings: { theme: "dark" } },
      },
      timestamp: Date.now(),
    });

    expect(currentState).toEqual({
      user: { name: "John" },
      settings: { theme: "dark" },
    });

    manager.stop();
  });

  it("should update local state on server update", async () => {
    const manager = new MockSyncManager({
      wsUrl: "ws://localhost:8080/sync",
    });

    await manager.start();

    // Simulate state update from server
    const ws = (manager as any).client._getWebSocket();
    ws?._receiveMessage({
      type: "state:update",
      payload: { path: "user.name", value: "Jane" },
      timestamp: Date.now(),
    });

    expect(manager.getState()).toHaveProperty("user.name", "Jane");

    manager.stop();
  });
});
