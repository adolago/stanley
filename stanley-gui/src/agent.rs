//! Agent Panel for Stanley GUI
//!
//! Provides an AI agent chat interface with:
//! - Chat interface with streaming responses
//! - Tool call visualization and status
//! - Markdown rendering for responses
//! - Real-time message streaming
//! - WebSocket sync with Stanley agent

use crate::api::StanleyClient;
use crate::sync::{
    ConnectionStatus, SyncClient, SyncClientState, SyncEvent, SyncEventType,
};
use crate::theme::Theme;
use gpui::prelude::*;
use gpui::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// =============================================================================
// Agent Types
// =============================================================================

/// Message role in conversation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Assistant,
    System,
    Tool,
}

impl MessageRole {
    pub fn label(&self) -> &'static str {
        match self {
            MessageRole::User => "You",
            MessageRole::Assistant => "Stanley Agent",
            MessageRole::System => "System",
            MessageRole::Tool => "Tool",
        }
    }
}

/// Tool execution status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolStatus {
    Pending,
    Running,
    Success,
    Error(String),
}

impl ToolStatus {
    pub fn label(&self) -> &'static str {
        match self {
            ToolStatus::Pending => "Pending",
            ToolStatus::Running => "Running...",
            ToolStatus::Success => "Completed",
            ToolStatus::Error(_) => "Error",
        }
    }
}

/// A tool call made by the agent
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
    pub result: Option<String>,
    pub status: ToolStatus,
    pub expanded: bool,
}

impl ToolCall {
    pub fn new(id: String, name: String, arguments: String) -> Self {
        Self {
            id,
            name,
            arguments,
            result: None,
            status: ToolStatus::Pending,
            expanded: false,
        }
    }
}

/// A message in the conversation
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub id: String,
    pub role: MessageRole,
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub timestamp: String,
    pub is_streaming: bool,
}

impl ChatMessage {
    pub fn new_user(content: String) -> Self {
        Self {
            id: uuid_v4(),
            role: MessageRole::User,
            content,
            tool_calls: Vec::new(),
            timestamp: format_now(),
            is_streaming: false,
        }
    }

    pub fn new_assistant(content: String) -> Self {
        Self {
            id: uuid_v4(),
            role: MessageRole::Assistant,
            content,
            tool_calls: Vec::new(),
            timestamp: format_now(),
            is_streaming: false,
        }
    }

    pub fn new_assistant_streaming() -> Self {
        Self {
            id: uuid_v4(),
            role: MessageRole::Assistant,
            content: String::new(),
            tool_calls: Vec::new(),
            timestamp: format_now(),
            is_streaming: true,
        }
    }
}

/// Agent connection status
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum AgentStatus {
    #[default]
    Disconnected,
    Connecting,
    Connected,
    Responding,
    Error(String),
}

impl AgentStatus {
    pub fn label(&self) -> &'static str {
        match self {
            AgentStatus::Disconnected => "Disconnected",
            AgentStatus::Connecting => "Connecting...",
            AgentStatus::Connected => "Ready",
            AgentStatus::Responding => "Thinking...",
            AgentStatus::Error(_) => "Error",
        }
    }

    pub fn is_busy(&self) -> bool {
        matches!(self, AgentStatus::Connecting | AgentStatus::Responding)
    }
}

// =============================================================================
// Agent State
// =============================================================================

/// State for the agent panel
pub struct AgentState {
    /// Current agent status
    pub status: AgentStatus,
    /// Conversation messages
    pub messages: Vec<ChatMessage>,
    /// Current input text
    pub input_text: String,
    /// Whether to show tool calls expanded
    pub show_tools: bool,
    /// System prompt / context
    pub system_prompt: String,
    /// Agent capabilities description
    pub capabilities: Vec<String>,
    /// Sync client state for WebSocket communication
    pub sync_state: SyncClientState,
    /// Recent sync events from agent
    pub sync_events: Vec<SyncEvent>,
}

impl Default for AgentState {
    fn default() -> Self {
        Self {
            status: AgentStatus::Connected,
            messages: vec![
                ChatMessage {
                    id: uuid_v4(),
                    role: MessageRole::Assistant,
                    content: "Hello! I'm the Stanley AI Agent. I can help you analyze investments, \
                        research companies, review portfolio performance, and execute analysis workflows. \
                        What would you like to explore today?".to_string(),
                    tool_calls: Vec::new(),
                    timestamp: format_now(),
                    is_streaming: false,
                },
            ],
            input_text: String::new(),
            show_tools: true,
            system_prompt: "You are Stanley, an AI assistant specialized in institutional \
                investment analysis, money flow tracking, and portfolio management.".to_string(),
            capabilities: vec![
                "Market Analysis".to_string(),
                "Portfolio Review".to_string(),
                "Money Flow Tracking".to_string(),
                "Institutional Holdings".to_string(),
                "Research Reports".to_string(),
                "SEC Filings".to_string(),
            ],
            sync_state: SyncClientState::default(),
            sync_events: Vec::new(),
        }
    }
}

impl AgentState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a user message
    pub fn add_user_message(&mut self, content: String) {
        self.messages.push(ChatMessage::new_user(content));
    }

    /// Add an assistant message
    pub fn add_assistant_message(&mut self, content: String) {
        self.messages.push(ChatMessage::new_assistant(content));
    }

    /// Start streaming an assistant response
    pub fn start_streaming(&mut self) {
        self.status = AgentStatus::Responding;
        self.messages.push(ChatMessage::new_assistant_streaming());
    }

    /// Append content to the current streaming message
    pub fn append_streaming(&mut self, content: &str) {
        if let Some(msg) = self.messages.last_mut() {
            if msg.is_streaming {
                msg.content.push_str(content);
            }
        }
    }

    /// Finish streaming
    pub fn finish_streaming(&mut self) {
        if let Some(msg) = self.messages.last_mut() {
            if msg.is_streaming {
                msg.is_streaming = false;
            }
        }
        self.status = AgentStatus::Connected;
    }

    /// Add a tool call to the current message
    pub fn add_tool_call(&mut self, tool_call: ToolCall) {
        if let Some(msg) = self.messages.last_mut() {
            msg.tool_calls.push(tool_call);
        }
    }

    /// Update tool call status
    pub fn update_tool_status(&mut self, tool_id: &str, status: ToolStatus, result: Option<String>) {
        for msg in self.messages.iter_mut().rev() {
            for tool in msg.tool_calls.iter_mut() {
                if tool.id == tool_id {
                    tool.status = status;
                    tool.result = result;
                    return;
                }
            }
        }
    }

    /// Clear conversation history
    pub fn clear_history(&mut self) {
        self.messages.clear();
        self.add_assistant_message(
            "Conversation cleared. How can I help you?".to_string()
        );
    }

    // =========================================================================
    // Sync Methods
    // =========================================================================

    /// Handle incoming sync event from WebSocket
    pub fn handle_sync_event(&mut self, event: SyncEvent) {
        // Add to recent events
        if self.sync_events.len() >= 50 {
            self.sync_events.remove(0);
        }
        self.sync_events.push(event.clone());

        // Handle specific event types
        match event.event_type {
            SyncEventType::AgentQueryComplete => {
                // Agent finished processing a query
                if let Some(data) = &event.data {
                    if let Some(content) = data.get("content").and_then(|v| v.as_str()) {
                        self.add_assistant_message(content.to_string());
                    }
                }
                self.finish_streaming();
            }
            SyncEventType::AgentToolCall => {
                // Agent is calling a tool
                if let Some(data) = &event.data {
                    let tool_name = data.get("toolName")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();
                    let tool_id = data.get("toolId")
                        .and_then(|v| v.as_str())
                        .unwrap_or(&event.id)
                        .to_string();
                    let arguments = data.get("arguments")
                        .map(|v| v.to_string())
                        .unwrap_or_default();

                    self.add_tool_call(ToolCall::new(tool_id, tool_name, arguments));
                }
            }
            SyncEventType::AgentToolResult => {
                // Tool execution completed
                if let Some(data) = &event.data {
                    let tool_id = data.get("toolId")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let result = data.get("result")
                        .map(|v| v.to_string());
                    let success = data.get("success")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(true);

                    let status = if success {
                        ToolStatus::Success
                    } else {
                        ToolStatus::Error(
                            data.get("error")
                                .and_then(|v| v.as_str())
                                .unwrap_or("Unknown error")
                                .to_string()
                        )
                    };

                    self.update_tool_status(&tool_id, status, result);
                }
            }
            SyncEventType::ResearchComplete => {
                // Research query completed
                if let Some(data) = &event.data {
                    let symbol = data.get("symbol")
                        .and_then(|v| v.as_str())
                        .unwrap_or("N/A");
                    let summary = data.get("summary")
                        .and_then(|v| v.as_str())
                        .unwrap_or("Research completed.");

                    self.add_assistant_message(
                        format!("Research complete for {}:\n\n{}", symbol, summary)
                    );
                }
            }
            SyncEventType::AlertTriggered => {
                // Alert was triggered
                if let Some(data) = &event.data {
                    let message = data.get("message")
                        .and_then(|v| v.as_str())
                        .unwrap_or("Alert triggered");
                    let severity = data.get("severity")
                        .and_then(|v| v.as_str())
                        .unwrap_or("info");

                    self.add_assistant_message(
                        format!("[{}] {}", severity.to_uppercase(), message)
                    );
                }
            }
            _ => {
                // Other events are stored but don't affect chat
            }
        }
    }

    /// Get sync connection status
    pub fn sync_status(&self) -> ConnectionStatus {
        self.sync_state.status
    }

    /// Update sync connection status
    pub fn set_sync_status(&mut self, status: ConnectionStatus) {
        self.sync_state.status = status;
    }

    /// Get recent sync events
    pub fn get_sync_events(&self) -> &[SyncEvent] {
        &self.sync_events
    }

    /// Get sync events by type
    pub fn get_sync_events_by_type(&self, event_type: SyncEventType) -> Vec<&SyncEvent> {
        self.sync_events
            .iter()
            .filter(|e| e.event_type == event_type)
            .collect()
    }

    /// Clear sync events
    pub fn clear_sync_events(&mut self) {
        self.sync_events.clear();
    }

    /// Create a view opened event for syncing
    pub fn create_view_opened_event(&self) -> SyncEvent {
        SyncEvent::view_opened("agent", "agent", None)
    }

    /// Create a symbol selected event for syncing
    pub fn create_symbol_selected_event(&self, symbol: &str, previous: Option<&str>) -> SyncEvent {
        SyncEvent::symbol_selected(symbol, "agent", previous)
    }
}

// =============================================================================
// Agent Panel Rendering
// =============================================================================

/// Render the agent panel view
pub fn render_agent_panel(
    theme: &Theme,
    state: &AgentState,
    input_text: &str,
    _api_client: &Arc<StanleyClient>,
    cx: &mut Context<impl Sized>,
) -> Div {
    div()
        .flex_grow()
        .h_full()
        .flex()
        .flex_col()
        .bg(theme.background)
        // Header
        .child(render_agent_header(theme, state))
        // Messages area
        .child(render_messages_area(theme, state, cx))
        // Input area
        .child(render_input_area(theme, input_text, state.status.is_busy()))
}

/// Render the agent panel header
fn render_agent_header(theme: &Theme, state: &AgentState) -> impl IntoElement {
    let status_color = match &state.status {
        AgentStatus::Connected => theme.positive,
        AgentStatus::Responding => theme.warning,
        AgentStatus::Connecting => theme.accent,
        AgentStatus::Disconnected => theme.text_muted,
        AgentStatus::Error(_) => theme.negative,
    };

    div()
        .h(px(64.0))
        .px(px(24.0))
        .flex()
        .items_center()
        .justify_between()
        .border_b_1()
        .border_color(theme.border_subtle)
        .bg(theme.card_bg)
        // Left: Title and status
        .child(
            div()
                .flex()
                .items_center()
                .gap(px(16.0))
                // Agent icon
                .child(
                    div()
                        .size(px(42.0))
                        .rounded(px(10.0))
                        .bg(theme.accent)
                        .flex()
                        .items_center()
                        .justify_center()
                        .border_1()
                        .border_color(theme.accent_glow)
                        .child(
                            div()
                                .text_size(px(20.0))
                                .font_weight(FontWeight::BLACK)
                                .text_color(hsla(0.0, 0.0, 1.0, 0.95))
                                .child("A")
                        )
                )
                // Title and status
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap(px(2.0))
                        .child(
                            div()
                                .text_size(px(18.0))
                                .font_weight(FontWeight::BOLD)
                                .text_color(theme.text)
                                .child("Stanley Agent")
                        )
                        .child(
                            div()
                                .flex()
                                .items_center()
                                .gap(px(6.0))
                                .child(
                                    div()
                                        .size(px(8.0))
                                        .rounded_full()
                                        .bg(status_color)
                                )
                                .child(
                                    div()
                                        .text_size(px(12.0))
                                        .text_color(theme.text_muted)
                                        .child(state.status.label())
                                )
                        )
                )
        )
        // Right: Action buttons
        .child(
            div()
                .flex()
                .items_center()
                .gap(px(8.0))
                // Capabilities indicator
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(4.0))
                        .children(
                            state.capabilities.iter().take(3).map(|cap| {
                                div()
                                    .px(px(8.0))
                                    .py(px(4.0))
                                    .rounded(px(4.0))
                                    .bg(theme.accent_subtle)
                                    .text_size(px(10.0))
                                    .font_weight(FontWeight::MEDIUM)
                                    .text_color(theme.accent)
                                    .child(cap.clone())
                            }).collect::<Vec<_>>()
                        )
                )
                // Clear button
                .child(
                    div()
                        .id("clear-chat")
                        .px(px(12.0))
                        .py(px(6.0))
                        .rounded(px(6.0))
                        .bg(theme.card_bg_elevated)
                        .border_1()
                        .border_color(theme.border_subtle)
                        .text_size(px(12.0))
                        .text_color(theme.text_muted)
                        .cursor_pointer()
                        .hover(|s| s.bg(theme.hover_bg).text_color(theme.text))
                        .child("Clear")
                )
        )
}

/// Render the messages area
fn render_messages_area(
    theme: &Theme,
    state: &AgentState,
    _cx: &mut Context<impl Sized>,
) -> impl IntoElement {
    div()
        .id("agent-messages")
        .flex_grow()
        .overflow_y_scroll()
        .p(px(20.0))
        .flex()
        .flex_col()
        .gap(px(16.0))
        .children(
            state.messages.iter().map(|msg| {
                render_message(theme, msg, state.show_tools)
            }).collect::<Vec<_>>()
        )
        // Streaming indicator
        .when(state.status == AgentStatus::Responding, |el| {
            el.child(render_streaming_indicator(theme))
        })
}

/// Render a single message
fn render_message(theme: &Theme, message: &ChatMessage, show_tools: bool) -> Div {
    let is_user = message.role == MessageRole::User;
    let is_tool = message.role == MessageRole::Tool;

    let (bg, border, _align) = if is_user {
        (theme.accent_subtle, theme.accent_muted, "flex-end")
    } else if is_tool {
        (theme.warning_subtle, theme.warning, "flex-start")
    } else {
        (theme.card_bg, theme.border_subtle, "flex-start")
    };

    let max_width = if is_user { px(500.0) } else { px(700.0) };

    div()
        .w_full()
        .flex()
        .when(is_user, |el| el.justify_end())
        .child(
            div()
                .max_w(max_width)
                .flex()
                .flex_col()
                .gap(px(8.0))
                // Message header
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(8.0))
                        .when(is_user, |el| el.justify_end())
                        .child(
                            div()
                                .text_size(px(11.0))
                                .font_weight(FontWeight::SEMIBOLD)
                                .text_color(if is_user { theme.accent } else { theme.text_muted })
                                .child(message.role.label())
                        )
                        .child(
                            div()
                                .text_size(px(10.0))
                                .text_color(theme.text_dimmed)
                                .child(message.timestamp.clone())
                        )
                )
                // Message content
                .child(
                    div()
                        .p(px(14.0))
                        .rounded(px(12.0))
                        .bg(bg)
                        .border_1()
                        .border_color(border)
                        .child(
                            render_markdown_content(theme, &message.content, message.is_streaming)
                        )
                )
                // Tool calls (if any)
                .when(show_tools && !message.tool_calls.is_empty(), |el| {
                    el.child(
                        div()
                            .flex()
                            .flex_col()
                            .gap(px(8.0))
                            .children(
                                message.tool_calls.iter().map(|tool| {
                                    render_tool_call(theme, tool)
                                }).collect::<Vec<_>>()
                            )
                    )
                })
        )
}

/// Render markdown content (simplified - just handles basic formatting)
fn render_markdown_content(theme: &Theme, content: &str, is_streaming: bool) -> Div {
    // For now, just render as plain text with code block detection
    // A full markdown parser would be more complex
    let lines: Vec<&str> = content.lines().collect();

    div()
        .flex()
        .flex_col()
        .gap(px(8.0))
        .children(
            lines.iter().map(|line| {
                let line = *line;
                if line.starts_with("```") {
                    // Code block marker
                    div()
                        .text_size(px(12.0))
                        .text_color(theme.text_dimmed)
                        .child(line.to_string())
                } else if line.starts_with('#') {
                    // Heading
                    let level = line.chars().take_while(|c| *c == '#').count();
                    let text = line.trim_start_matches('#').trim();
                    let size = match level {
                        1 => px(18.0),
                        2 => px(16.0),
                        _ => px(14.0),
                    };
                    div()
                        .text_size(size)
                        .font_weight(FontWeight::BOLD)
                        .text_color(theme.text)
                        .child(text.to_string())
                } else if line.starts_with("- ") || line.starts_with("* ") {
                    // List item
                    div()
                        .flex()
                        .items_start()
                        .gap(px(8.0))
                        .child(
                            div()
                                .text_size(px(13.0))
                                .text_color(theme.accent)
                                .child("*")
                        )
                        .child(
                            div()
                                .text_size(px(13.0))
                                .text_color(theme.text_secondary)
                                .child(line[2..].to_string())
                        )
                } else if line.starts_with("> ") {
                    // Blockquote
                    div()
                        .pl(px(12.0))
                        .border_l_2()
                        .border_color(theme.accent)
                        .text_size(px(13.0))
                        .text_color(theme.text_muted)
                        .child(line[2..].to_string())
                } else if line.trim().is_empty() {
                    // Empty line
                    div().h(px(8.0))
                } else {
                    // Regular paragraph
                    div()
                        .text_size(px(13.0))
                        .text_color(theme.text_secondary)
                        .child(line.to_string())
                }
            }).collect::<Vec<_>>()
        )
        // Streaming cursor
        .when(is_streaming, |el| {
            el.child(
                div()
                    .w(px(8.0))
                    .h(px(16.0))
                    .bg(theme.accent)
                    .rounded(px(2.0))
            )
        })
}

/// Render a tool call
fn render_tool_call(theme: &Theme, tool: &ToolCall) -> Div {
    let status_color = match &tool.status {
        ToolStatus::Pending => theme.text_muted,
        ToolStatus::Running => theme.warning,
        ToolStatus::Success => theme.positive,
        ToolStatus::Error(_) => theme.negative,
    };

    div()
        .rounded(px(8.0))
        .bg(theme.card_bg_elevated)
        .border_1()
        .border_color(theme.border_subtle)
        .overflow_hidden()
        // Header
        .child(
            div()
                .px(px(12.0))
                .py(px(10.0))
                .flex()
                .items_center()
                .justify_between()
                .bg(theme.card_bg)
                .border_b_1()
                .border_color(theme.border_subtle)
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(8.0))
                        // Tool icon
                        .child(
                            div()
                                .size(px(20.0))
                                .rounded(px(4.0))
                                .bg(status_color.opacity(0.15))
                                .flex()
                                .items_center()
                                .justify_center()
                                .text_size(px(10.0))
                                .text_color(status_color)
                                .child("T")
                        )
                        // Tool name
                        .child(
                            div()
                                .text_size(px(12.0))
                                .font_weight(FontWeight::SEMIBOLD)
                                .text_color(theme.text)
                                .child(tool.name.clone())
                        )
                )
                // Status badge
                .child(
                    div()
                        .px(px(8.0))
                        .py(px(3.0))
                        .rounded(px(4.0))
                        .bg(status_color.opacity(0.15))
                        .text_size(px(10.0))
                        .font_weight(FontWeight::MEDIUM)
                        .text_color(status_color)
                        .child(tool.status.label())
                )
        )
        // Arguments (collapsed by default)
        .when(tool.expanded || !tool.arguments.is_empty(), |el| {
            el.child(
                div()
                    .px(px(12.0))
                    .py(px(8.0))
                    .child(
                        div()
                            .text_size(px(11.0))
                            .text_color(theme.text_dimmed)
                            .mb(px(4.0))
                            .child("Arguments:")
                    )
                    .child(
                        div()
                            .p(px(8.0))
                            .rounded(px(4.0))
                            .bg(theme.background)
                            .text_size(px(11.0))
                            .font_family("monospace")
                            .text_color(theme.text_muted)
                            .child(truncate_text(&tool.arguments, 200))
                    )
            )
        })
        // Result (if available)
        .when_some(tool.result.clone(), |el, result| {
            el.child(
                div()
                    .px(px(12.0))
                    .py(px(8.0))
                    .border_t_1()
                    .border_color(theme.border_subtle)
                    .child(
                        div()
                            .text_size(px(11.0))
                            .text_color(theme.text_dimmed)
                            .mb(px(4.0))
                            .child("Result:")
                    )
                    .child(
                        div()
                            .p(px(8.0))
                            .rounded(px(4.0))
                            .bg(theme.background)
                            .text_size(px(11.0))
                            .font_family("monospace")
                            .text_color(theme.positive)
                            .child(truncate_text(&result, 300))
                    )
            )
        })
}

/// Render streaming indicator
fn render_streaming_indicator(theme: &Theme) -> Div {
    div()
        .flex()
        .items_center()
        .gap(px(8.0))
        .py(px(8.0))
        .child(
            div()
                .flex()
                .items_center()
                .gap(px(4.0))
                // Animated dots
                .child(div().size(px(6.0)).rounded_full().bg(theme.accent))
                .child(div().size(px(6.0)).rounded_full().bg(theme.accent.opacity(0.7)))
                .child(div().size(px(6.0)).rounded_full().bg(theme.accent.opacity(0.4)))
        )
        .child(
            div()
                .text_size(px(12.0))
                .text_color(theme.text_muted)
                .child("Agent is thinking...")
        )
}

/// Render the input area
fn render_input_area(theme: &Theme, input_text: &str, is_busy: bool) -> Div {
    div()
        .px(px(20.0))
        .py(px(16.0))
        .border_t_1()
        .border_color(theme.border_subtle)
        .bg(theme.card_bg)
        .flex()
        .flex_col()
        .gap(px(12.0))
        // Input row
        .child(
            div()
                .flex()
                .items_end()
                .gap(px(12.0))
                // Text input (using a styled div since GPUI doesn't have native input)
                .child(
                    div()
                        .id("agent-input")
                        .flex_grow()
                        .min_h(px(44.0))
                        .max_h(px(120.0))
                        .px(px(16.0))
                        .py(px(12.0))
                        .rounded(px(10.0))
                        .bg(theme.background)
                        .border_1()
                        .border_color(theme.border)
                        .text_size(px(14.0))
                        .text_color(if input_text.is_empty() { theme.text_muted } else { theme.text })
                        .child(if input_text.is_empty() {
                            "Ask Stanley anything about investments...".to_string()
                        } else {
                            input_text.to_string()
                        })
                )
                // Send button
                .child(
                    div()
                        .id("send-message")
                        .size(px(44.0))
                        .rounded(px(10.0))
                        .bg(if is_busy { theme.text_muted } else { theme.accent })
                        .flex()
                        .items_center()
                        .justify_center()
                        .cursor(if is_busy { CursorStyle::default() } else { CursorStyle::PointingHand })
                        .when(!is_busy, |el| {
                            el.hover(|s| s.bg(theme.accent_hover))
                        })
                        .child(
                            div()
                                .text_size(px(18.0))
                                .font_weight(FontWeight::BOLD)
                                .text_color(hsla(0.0, 0.0, 1.0, 0.95))
                                .child(if is_busy { "..." } else { ">" })
                        )
                )
        )
        // Help text
        .child(
            div()
                .flex()
                .items_center()
                .justify_between()
                .child(
                    div()
                        .text_size(px(11.0))
                        .text_color(theme.text_dimmed)
                        .child("Press Enter to send, Shift+Enter for new line")
                )
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(12.0))
                        .child(
                            div()
                                .text_size(px(11.0))
                                .text_color(theme.text_dimmed)
                                .child("Powered by Claude")
                        )
                )
        )
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Generate a simple UUID v4-like string
fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("msg-{:x}-{:x}", now.as_secs(), now.subsec_nanos())
}

/// Format current time as string
fn format_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let hours = (secs / 3600) % 24;
    let minutes = (secs / 60) % 60;
    format!("{:02}:{:02}", hours, minutes)
}

/// Truncate text with ellipsis
fn truncate_text(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        text.to_string()
    } else {
        format!("{}...", &text[..max_len])
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_state_default() {
        let state = AgentState::default();
        assert_eq!(state.status, AgentStatus::Connected);
        assert!(!state.messages.is_empty());
    }

    #[test]
    fn test_add_user_message() {
        let mut state = AgentState::new();
        let initial_count = state.messages.len();
        state.add_user_message("Test message".to_string());
        assert_eq!(state.messages.len(), initial_count + 1);
        assert_eq!(state.messages.last().unwrap().role, MessageRole::User);
    }

    #[test]
    fn test_streaming_workflow() {
        let mut state = AgentState::new();
        state.start_streaming();
        assert!(state.status.is_busy());
        assert!(state.messages.last().unwrap().is_streaming);

        state.append_streaming("Hello");
        state.append_streaming(" World");
        assert_eq!(state.messages.last().unwrap().content, "Hello World");

        state.finish_streaming();
        assert!(!state.status.is_busy());
        assert!(!state.messages.last().unwrap().is_streaming);
    }

    #[test]
    fn test_tool_call_status() {
        let tool = ToolCall::new(
            "tool-1".to_string(),
            "get_market_data".to_string(),
            r#"{"symbol": "AAPL"}"#.to_string(),
        );
        assert_eq!(tool.status, ToolStatus::Pending);
        assert!(tool.result.is_none());
    }

    #[test]
    fn test_truncate_text() {
        assert_eq!(truncate_text("short", 10), "short");
        assert_eq!(truncate_text("this is a long text", 10), "this is a ...");
    }
}
