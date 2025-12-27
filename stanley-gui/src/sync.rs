//! Stanley Sync Client
//!
//! WebSocket client for real-time synchronization between the Rust GUI
//! and the Stanley agent. Handles bidirectional event communication,
//! reconnection logic, and message queuing.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::mpsc;

// =============================================================================
// Sync Event Types
// =============================================================================

/// Types of sync events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyncEventType {
    // Portfolio events
    PortfolioUpdate,
    PortfolioHoldingAdded,
    PortfolioHoldingRemoved,

    // Note events
    NoteSaved,
    NoteDeleted,
    NoteUpdated,
    ThesisCreated,
    TradeOpened,
    TradeClosed,

    // Research events
    ResearchComplete,
    ResearchStarted,
    ResearchProgress,

    // Alert events
    AlertTriggered,
    AlertAcknowledged,
    PriceAlert,
    FlowAlert,

    // View events (GUI -> Agent)
    ViewOpened,
    ViewClosed,
    SymbolSelected,
    SymbolDeselected,

    // Agent events
    AgentQueryStart,
    AgentQueryComplete,
    AgentToolCall,
    AgentToolResult,
    AgentError,

    // Connection events
    ClientConnected,
    ClientDisconnected,
    SyncError,
}

impl SyncEventType {
    pub fn as_str(&self) -> &'static str {
        match self {
            SyncEventType::PortfolioUpdate => "portfolio_update",
            SyncEventType::PortfolioHoldingAdded => "portfolio_holding_added",
            SyncEventType::PortfolioHoldingRemoved => "portfolio_holding_removed",
            SyncEventType::NoteSaved => "note_saved",
            SyncEventType::NoteDeleted => "note_deleted",
            SyncEventType::NoteUpdated => "note_updated",
            SyncEventType::ThesisCreated => "thesis_created",
            SyncEventType::TradeOpened => "trade_opened",
            SyncEventType::TradeClosed => "trade_closed",
            SyncEventType::ResearchComplete => "research_complete",
            SyncEventType::ResearchStarted => "research_started",
            SyncEventType::ResearchProgress => "research_progress",
            SyncEventType::AlertTriggered => "alert_triggered",
            SyncEventType::AlertAcknowledged => "alert_acknowledged",
            SyncEventType::PriceAlert => "price_alert",
            SyncEventType::FlowAlert => "flow_alert",
            SyncEventType::ViewOpened => "view_opened",
            SyncEventType::ViewClosed => "view_closed",
            SyncEventType::SymbolSelected => "symbol_selected",
            SyncEventType::SymbolDeselected => "symbol_deselected",
            SyncEventType::AgentQueryStart => "agent_query_start",
            SyncEventType::AgentQueryComplete => "agent_query_complete",
            SyncEventType::AgentToolCall => "agent_tool_call",
            SyncEventType::AgentToolResult => "agent_tool_result",
            SyncEventType::AgentError => "agent_error",
            SyncEventType::ClientConnected => "client_connected",
            SyncEventType::ClientDisconnected => "client_disconnected",
            SyncEventType::SyncError => "sync_error",
        }
    }
}

// =============================================================================
// WebSocket Message Types
// =============================================================================

/// WebSocket message types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WsMessageType {
    Ping,
    Pong,
    Subscribe,
    Unsubscribe,
    Ack,
    Error,
    Event,
    Batch,
    StateSync,
    StateRequest,
}

/// WebSocket message structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WsMessage {
    #[serde(rename = "type")]
    pub msg_type: WsMessageType,
    pub id: String,
    pub timestamp: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<serde_json::Value>,
}

impl WsMessage {
    pub fn new(msg_type: WsMessageType, payload: Option<serde_json::Value>) -> Self {
        Self {
            msg_type,
            id: generate_message_id(),
            timestamp: chrono_now(),
            payload,
        }
    }

    pub fn ping() -> Self {
        Self::new(WsMessageType::Ping, None)
    }

    pub fn pong() -> Self {
        Self::new(WsMessageType::Pong, None)
    }

    pub fn subscribe(event_types: Vec<SyncEventType>) -> Self {
        Self::new(
            WsMessageType::Subscribe,
            Some(serde_json::json!({
                "eventTypes": event_types.iter().map(|e| e.as_str()).collect::<Vec<_>>()
            })),
        )
    }

    pub fn event(event: SyncEvent) -> Self {
        Self::new(WsMessageType::Event, Some(serde_json::to_value(event).unwrap_or_default()))
    }

    pub fn state_request() -> Self {
        Self::new(WsMessageType::StateRequest, None)
    }
}

// =============================================================================
// Sync Events
// =============================================================================

/// Base sync event structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncEvent {
    pub id: String,
    #[serde(rename = "type")]
    pub event_type: SyncEventType,
    pub timestamp: String,
    pub source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl SyncEvent {
    pub fn new(event_type: SyncEventType, data: Option<serde_json::Value>) -> Self {
        Self {
            id: generate_event_id(),
            event_type,
            timestamp: chrono_now(),
            source: "gui".to_string(),
            correlation_id: None,
            data,
        }
    }

    /// Create a view opened event
    pub fn view_opened(view_name: &str, view_type: &str, symbol: Option<&str>) -> Self {
        Self::new(
            SyncEventType::ViewOpened,
            Some(serde_json::json!({
                "viewName": view_name,
                "viewType": view_type,
                "context": {
                    "selectedSymbol": symbol
                }
            })),
        )
    }

    /// Create a view closed event
    pub fn view_closed(view_name: &str) -> Self {
        Self::new(
            SyncEventType::ViewClosed,
            Some(serde_json::json!({
                "viewName": view_name
            })),
        )
    }

    /// Create a symbol selected event
    pub fn symbol_selected(symbol: &str, context: &str, previous: Option<&str>) -> Self {
        Self::new(
            SyncEventType::SymbolSelected,
            Some(serde_json::json!({
                "symbol": symbol,
                "viewContext": context,
                "previousSymbol": previous
            })),
        )
    }
}

// =============================================================================
// Connection State
// =============================================================================

/// WebSocket connection status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConnectionStatus {
    #[default]
    Disconnected,
    Connecting,
    Connected,
    Reconnecting,
    Error,
}

impl ConnectionStatus {
    pub fn label(&self) -> &'static str {
        match self {
            ConnectionStatus::Disconnected => "Disconnected",
            ConnectionStatus::Connecting => "Connecting...",
            ConnectionStatus::Connected => "Connected",
            ConnectionStatus::Reconnecting => "Reconnecting...",
            ConnectionStatus::Error => "Error",
        }
    }

    pub fn is_connected(&self) -> bool {
        matches!(self, ConnectionStatus::Connected)
    }
}

// =============================================================================
// Message Queue
// =============================================================================

/// Queue for messages during offline periods
#[derive(Debug)]
pub struct MessageQueue {
    queue: VecDeque<WsMessage>,
    max_size: usize,
}

impl MessageQueue {
    pub fn new(max_size: usize) -> Self {
        Self {
            queue: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    pub fn enqueue(&mut self, message: WsMessage) {
        if self.queue.len() >= self.max_size {
            self.queue.pop_front();
        }
        self.queue.push_back(message);
    }

    pub fn drain(&mut self) -> Vec<WsMessage> {
        self.queue.drain(..).collect()
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    pub fn clear(&mut self) {
        self.queue.clear();
    }
}

impl Default for MessageQueue {
    fn default() -> Self {
        Self::new(100)
    }
}

// =============================================================================
// Sync Client State
// =============================================================================

/// State for the sync client
pub struct SyncClientState {
    /// Current connection status
    pub status: ConnectionStatus,
    /// WebSocket server URL
    pub server_url: String,
    /// Message queue for offline periods
    pub message_queue: MessageQueue,
    /// Subscribed event types
    pub subscriptions: Vec<SyncEventType>,
    /// Last ping timestamp
    pub last_ping: Option<String>,
    /// Last received event
    pub last_event: Option<SyncEvent>,
    /// Recent events cache
    pub event_history: VecDeque<SyncEvent>,
    /// Max event history size
    pub max_history: usize,
    /// Reconnection attempts
    pub reconnect_attempts: u32,
    /// Max reconnection attempts
    pub max_reconnect_attempts: u32,
    /// Reconnection delay (ms)
    pub reconnect_delay_ms: u64,
}

impl Default for SyncClientState {
    fn default() -> Self {
        Self {
            status: ConnectionStatus::Disconnected,
            server_url: "ws://127.0.0.1:8765/ws".to_string(),
            message_queue: MessageQueue::default(),
            subscriptions: vec![
                SyncEventType::PortfolioUpdate,
                SyncEventType::NoteSaved,
                SyncEventType::ResearchComplete,
                SyncEventType::AlertTriggered,
                SyncEventType::AgentQueryComplete,
                SyncEventType::AgentToolCall,
            ],
            last_ping: None,
            last_event: None,
            event_history: VecDeque::with_capacity(50),
            max_history: 50,
            reconnect_attempts: 0,
            max_reconnect_attempts: 5,
            reconnect_delay_ms: 1000,
        }
    }
}

impl SyncClientState {
    pub fn new(server_url: &str) -> Self {
        Self {
            server_url: server_url.to_string(),
            ..Default::default()
        }
    }

    /// Add event to history
    pub fn add_event(&mut self, event: SyncEvent) {
        if self.event_history.len() >= self.max_history {
            self.event_history.pop_front();
        }
        self.last_event = Some(event.clone());
        self.event_history.push_back(event);
    }

    /// Get events by type
    pub fn get_events_by_type(&self, event_type: SyncEventType) -> Vec<&SyncEvent> {
        self.event_history
            .iter()
            .filter(|e| e.event_type == event_type)
            .collect()
    }

    /// Queue message for sending
    pub fn queue_message(&mut self, message: WsMessage) {
        self.message_queue.enqueue(message);
    }

    /// Queue event for sending
    pub fn queue_event(&mut self, event: SyncEvent) {
        self.queue_message(WsMessage::event(event));
    }

    /// Reset reconnection state
    pub fn reset_reconnect(&mut self) {
        self.reconnect_attempts = 0;
    }

    /// Increment reconnection attempts
    pub fn increment_reconnect(&mut self) -> bool {
        self.reconnect_attempts += 1;
        self.reconnect_attempts <= self.max_reconnect_attempts
    }

    /// Get reconnection delay with exponential backoff
    pub fn get_reconnect_delay(&self) -> u64 {
        let base = self.reconnect_delay_ms;
        let factor = 2u64.pow(self.reconnect_attempts.min(5));
        base * factor
    }
}

// =============================================================================
// Sync Client
// =============================================================================

/// Channel messages for sync client
#[derive(Debug)]
pub enum SyncCommand {
    Connect,
    Disconnect,
    Send(WsMessage),
    Subscribe(Vec<SyncEventType>),
    Unsubscribe(Vec<SyncEventType>),
}

/// Sync client for managing WebSocket connection
pub struct SyncClient {
    state: Arc<tokio::sync::RwLock<SyncClientState>>,
    command_tx: Option<mpsc::Sender<SyncCommand>>,
    event_rx: Option<mpsc::Receiver<SyncEvent>>,
}

impl SyncClient {
    pub fn new(server_url: &str) -> Self {
        Self {
            state: Arc::new(tokio::sync::RwLock::new(SyncClientState::new(server_url))),
            command_tx: None,
            event_rx: None,
        }
    }

    /// Get current connection status
    pub async fn status(&self) -> ConnectionStatus {
        self.state.read().await.status
    }

    /// Check if connected
    pub async fn is_connected(&self) -> bool {
        self.state.read().await.status.is_connected()
    }

    /// Send a message (queues if not connected)
    pub async fn send(&self, message: WsMessage) {
        if let Some(tx) = &self.command_tx {
            let _ = tx.send(SyncCommand::Send(message)).await;
        } else {
            // Queue for later if no connection
            self.state.write().await.queue_message(message);
        }
    }

    /// Send a sync event
    pub async fn send_event(&self, event: SyncEvent) {
        self.send(WsMessage::event(event)).await;
    }

    /// Emit view opened event
    pub async fn emit_view_opened(&self, view_name: &str, view_type: &str, symbol: Option<&str>) {
        self.send_event(SyncEvent::view_opened(view_name, view_type, symbol)).await;
    }

    /// Emit view closed event
    pub async fn emit_view_closed(&self, view_name: &str) {
        self.send_event(SyncEvent::view_closed(view_name)).await;
    }

    /// Emit symbol selected event
    pub async fn emit_symbol_selected(&self, symbol: &str, context: &str, previous: Option<&str>) {
        self.send_event(SyncEvent::symbol_selected(symbol, context, previous)).await;
    }

    /// Subscribe to event types
    pub async fn subscribe(&self, event_types: Vec<SyncEventType>) {
        if let Some(tx) = &self.command_tx {
            let _ = tx.send(SyncCommand::Subscribe(event_types)).await;
        }
    }

    /// Get recent events
    pub async fn get_recent_events(&self, limit: usize) -> Vec<SyncEvent> {
        let state = self.state.read().await;
        state
            .event_history
            .iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    /// Get events by type
    pub async fn get_events_by_type(&self, event_type: SyncEventType, limit: usize) -> Vec<SyncEvent> {
        let state = self.state.read().await;
        state
            .event_history
            .iter()
            .filter(|e| e.event_type == event_type)
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    /// Get last event
    pub async fn get_last_event(&self) -> Option<SyncEvent> {
        self.state.read().await.last_event.clone()
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Generate unique message ID
fn generate_message_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("msg-{:x}-{:x}", now.as_secs(), now.subsec_nanos())
}

/// Generate unique event ID
fn generate_event_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("gui-{:x}-{:x}", now.as_secs(), now.subsec_nanos())
}

/// Get current ISO timestamp
fn chrono_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    // Simple ISO format approximation
    let secs = now.as_secs();
    let millis = now.subsec_millis();
    format!(
        "{}Z",
        format_timestamp(secs, millis)
    )
}

/// Format timestamp as ISO 8601
fn format_timestamp(secs: u64, millis: u32) -> String {
    // Approximate date calculation
    let days_since_epoch = secs / 86400;
    let remaining_secs = secs % 86400;
    let hours = remaining_secs / 3600;
    let minutes = (remaining_secs % 3600) / 60;
    let seconds = remaining_secs % 60;

    // Very simplified year/month/day calculation
    let years_since_1970 = days_since_epoch / 365;
    let year = 1970 + years_since_1970;
    let day_of_year = days_since_epoch % 365;
    let month = (day_of_year / 30).min(11) + 1;
    let day = (day_of_year % 30) + 1;

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}",
        year, month, day, hours, minutes, seconds, millis
    )
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_event_creation() {
        let event = SyncEvent::view_opened("dashboard", "dashboard", Some("AAPL"));
        assert_eq!(event.event_type, SyncEventType::ViewOpened);
        assert_eq!(event.source, "gui");
        assert!(event.data.is_some());
    }

    #[test]
    fn test_message_queue() {
        let mut queue = MessageQueue::new(3);
        queue.enqueue(WsMessage::ping());
        queue.enqueue(WsMessage::ping());
        queue.enqueue(WsMessage::ping());
        queue.enqueue(WsMessage::ping()); // Should evict first

        assert_eq!(queue.len(), 3);

        let drained = queue.drain();
        assert_eq!(drained.len(), 3);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_sync_client_state() {
        let mut state = SyncClientState::default();
        assert_eq!(state.status, ConnectionStatus::Disconnected);

        let event = SyncEvent::new(SyncEventType::PortfolioUpdate, None);
        state.add_event(event);

        assert!(state.last_event.is_some());
        assert_eq!(state.event_history.len(), 1);
    }

    #[test]
    fn test_reconnect_delay() {
        let mut state = SyncClientState::default();

        assert_eq!(state.get_reconnect_delay(), 1000); // Base delay

        state.increment_reconnect();
        assert_eq!(state.get_reconnect_delay(), 2000); // 1000 * 2^1

        state.increment_reconnect();
        assert_eq!(state.get_reconnect_delay(), 4000); // 1000 * 2^2
    }

    #[test]
    fn test_ws_message_creation() {
        let ping = WsMessage::ping();
        assert!(matches!(ping.msg_type, WsMessageType::Ping));

        let sub = WsMessage::subscribe(vec![SyncEventType::PortfolioUpdate]);
        assert!(matches!(sub.msg_type, WsMessageType::Subscribe));
        assert!(sub.payload.is_some());
    }
}
