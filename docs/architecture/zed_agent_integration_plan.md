# Stanley + Zed + Anthropic Agent SDK Integration Plan

## Executive Summary

This plan outlines the integration of Zed's editor components and Anthropic's Agent SDK into Stanley to create an AI-augmented institutional investment research platform with intelligent "backstage access" - where the AI agent maintains continuous awareness of user context.

**Target Outcome:** A professional-grade notes/research editor powered by Zed's GPUI components, with an AI copilot that understands what the user is seeing, doing, and wanting to accomplish.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STANLEY PLATFORM                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    GPUI Application Layer                            │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│  │  │  Stanley    │  │  Zed Editor │  │   Agent     │  │  Context   │  │   │
│  │  │  Views (42) │  │  Component  │  │  Sidebar    │  │  Bridge    │  │   │
│  │  │             │  │             │  │             │  │            │  │   │
│  │  │ - Dashboard │  │ - Notes     │  │ - Chat UI   │  │ - State    │  │   │
│  │  │ - Portfolio │  │ - Memos     │  │ - Tools     │  │ - Events   │  │   │
│  │  │ - Research  │  │ - Markdown  │  │ - History   │  │ - Intents  │  │   │
│  │  │ - ETF/Macro │  │ - Rich Text │  │ - Suggest   │  │ - Snapshot │  │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬──────┘  │   │
│  └─────────┼────────────────┼────────────────┼───────────────┼─────────┘   │
│            │                │                │               │             │
│            ▼                ▼                ▼               ▼             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Context Orchestration Layer                       │   │
│  │                                                                      │   │
│  │   AgentContext {                                                     │   │
│  │     current_view: View::Portfolio,                                   │   │
│  │     selected_symbols: ["AAPL", "MSFT"],                              │   │
│  │     visible_data: ChartData { ... },                                 │   │
│  │     recent_actions: [FilterApplied, ChartZoomed, NoteCreated],       │   │
│  │     open_notes: [note_id_1, note_id_2],                              │   │
│  │     user_intent: IntentSignal::ResearchMode,                         │   │
│  │   }                                                                  │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                          │
├─────────────────────────────────┼──────────────────────────────────────────┤
│                                 ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Agent Runtime Layer                               │   │
│  │                                                                      │   │
│  │  ┌──────────────────┐    ┌──────────────────┐    ┌────────────────┐ │   │
│  │  │ Anthropic Agent  │    │   MCP Server     │    │  Tool Registry │ │   │
│  │  │ SDK (Python)     │◄──►│   (In-Process)   │◄──►│                │ │   │
│  │  │                  │    │                  │    │ - get_holdings │ │   │
│  │  │ - Claude Opus    │    │ - Context feed   │    │ - analyze_sec  │ │   │
│  │  │ - Agent loop     │    │ - Tool dispatch  │    │ - search_notes │ │   │
│  │  │ - Streaming      │    │ - State sync     │    │ - get_chart    │ │   │
│  │  └────────┬─────────┘    └──────────────────┘    │ - run_screen   │ │   │
│  │           │                                       │ - compare_peer │ │   │
│  │           │                                       └────────────────┘ │   │
│  └───────────┼─────────────────────────────────────────────────────────┘   │
│              │                                                             │
├──────────────┼─────────────────────────────────────────────────────────────┤
│              ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Stanley API Layer (FastAPI)                       │   │
│  │                                                                      │   │
│  │  14 Routers │ 135+ Endpoints │ JWT Auth │ Rate Limiting │ RBAC      │   │
│  │                                                                      │   │
│  │  /api/portfolio/* │ /api/research/* │ /api/etf/* │ /api/macro/*     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase Breakdown

### Phase 1: Foundation - Zed Editor Extraction
**Duration:** 2-3 weeks
**Priority:** Critical
**Hive Minds:** 4 specialized agents

Extract and adapt Zed's editor crates for Stanley's notes system.

#### 1.1 Crate Analysis & Dependency Mapping
- Analyze `editor`, `text`, `rich_text`, `markdown` crate dependencies
- Identify minimum viable crate subset
- Map GPUI version compatibility with Stanley

#### 1.2 Editor Crate Extraction
- Fork/extract required crates
- Remove Zed-specific dependencies (collab, LSP, git integration)
- Adapt for standalone use

#### 1.3 Notes View Integration
- Replace current notes implementation with Zed editor
- Implement note persistence layer
- Add markdown preview support

#### 1.4 Testing & Stabilization
- Unit tests for editor functionality
- Integration tests with Stanley views
- Performance benchmarking

---

### Phase 2: Agent SDK Integration
**Duration:** 2 weeks
**Priority:** High
**Hive Minds:** 3 specialized agents

Integrate Anthropic's Agent SDK as Stanley's AI backbone.

#### 2.1 SDK Setup & Configuration
- Add `claude-agent-sdk` to requirements
- Configure API authentication
- Set up agent runtime environment

#### 2.2 Stanley Tool Implementation
- Implement MCP tools for Stanley operations
- Create tool registry and dispatch system
- Add streaming response handling

#### 2.3 Agent UI Component
- Build agent sidebar in GPUI
- Implement chat interface
- Add tool result visualization

---

### Phase 3: Context Bridge Development
**Duration:** 2-3 weeks
**Priority:** High
**Hive Minds:** 4 specialized agents

Build the "backstage access" context system.

#### 3.1 State Observation System
- Implement view state tracking
- Add selection monitoring
- Create event stream for user actions

#### 3.2 Context Aggregation
- Build context snapshot generator
- Implement efficient diffing for updates
- Add intent signal detection

#### 3.3 Agent Context Feed
- Create context serialization protocol
- Implement streaming context updates
- Add context relevance filtering

#### 3.4 Bidirectional Communication
- Agent suggestions → UI highlights
- UI events → Agent context
- Shared clipboard/selection

---

### Phase 4: Deep Integration
**Duration:** 2-3 weeks
**Priority:** Medium
**Hive Minds:** 5 specialized agents

Create seamless AI-assisted workflows.

#### 4.1 Smart Notes
- AI-suggested annotations
- Auto-linking to securities/research
- Template generation

#### 4.2 Research Copilot
- Proactive insights based on viewed data
- "What am I missing?" analysis
- Peer comparison suggestions

#### 4.3 Portfolio Assistant
- Risk alerts based on positions
- Rebalancing suggestions
- "Explain this movement" feature

#### 4.4 Keyboard & Command Integration
- Agent-aware command palette
- Natural language commands
- Contextual shortcuts

#### 4.5 Memory & Learning
- Session memory persistence
- User preference learning
- Research history indexing

---

## Hive Mind Orchestration

### Activation Instructions

#### Prerequisites

```bash
# Ensure claude-flow is available
cd /home/artur/Repositories/claude-flow
npm install
npm link

# Or use npx
npx claude-flow@alpha --version
```

#### Swarm Initialization

```bash
# Initialize hierarchical swarm for complex multi-phase work
npx claude-flow@alpha mcp start &

# In Claude Code, initialize the coordination topology
mcp__claude-flow__swarm_init {
  topology: "hierarchical",
  maxAgents: 12,
  strategy: "specialized"
}
```

### Phase 1 Hive Minds

Launch these 4 agents in parallel via Claude Code's Task tool:

```javascript
// Single message - spawn all Phase 1 agents concurrently

Task("Crate Analyzer", `
  PHASE 1.1: Zed Crate Analysis

  OBJECTIVE: Analyze Zed's editor-related crates and create dependency map.

  WORKING DIRECTORIES:
  - /home/artur/Repositories/zed/crates/editor
  - /home/artur/Repositories/zed/crates/text
  - /home/artur/Repositories/zed/crates/rich_text
  - /home/artur/Repositories/zed/crates/markdown

  TASKS:
  1. Read each crate's Cargo.toml and map all dependencies
  2. Identify which dependencies are Zed-internal vs external
  3. Create dependency graph showing crate relationships
  4. List minimum crates needed for standalone editor
  5. Check GPUI version compatibility with Stanley

  OUTPUT:
  - Create /home/artur/Repositories/stanley/docs/architecture/zed_crate_analysis.md
  - Include dependency graph (ASCII or mermaid)
  - List recommended extraction strategy

  HOOKS:
  npx claude-flow@alpha hooks pre-task --description "Zed crate analysis"
  npx claude-flow@alpha hooks post-task --task-id "phase1-crate-analysis"
`, "code-analyzer")

Task("Editor Extractor", `
  PHASE 1.2: Editor Crate Extraction

  OBJECTIVE: Extract and adapt Zed's editor crate for Stanley.

  PREREQUISITES: Wait for Crate Analyzer output before deep extraction.

  WORKING DIRECTORIES:
  - Source: /home/artur/Repositories/zed/crates/editor
  - Target: /home/artur/Repositories/stanley/stanley-gui/crates/stanley-editor

  TASKS:
  1. Create stanley-editor crate structure
  2. Copy core editor functionality (editor.rs, element.rs, display_map/)
  3. Remove Zed-specific features (LSP, git blame, collab)
  4. Adapt imports for standalone use
  5. Create minimal Cargo.toml with required dependencies
  6. Ensure GPUI compatibility

  OUTPUT:
  - Working stanley-editor crate
  - Cargo.toml with correct dependencies
  - README.md documenting what was extracted

  IMPORTANT: Preserve licensing headers (Apache 2.0/GPL)

  HOOKS:
  npx claude-flow@alpha hooks pre-task --description "Editor extraction"
  npx claude-flow@alpha hooks post-edit --file "stanley-editor/*"
`, "coder")

Task("Notes Integrator", `
  PHASE 1.3: Notes View Integration

  OBJECTIVE: Integrate extracted editor into Stanley's notes system.

  PREREQUISITES: Editor Extractor must complete first.

  WORKING DIRECTORIES:
  - /home/artur/Repositories/stanley/stanley-gui/src/notes.rs
  - /home/artur/Repositories/stanley/stanley-gui/crates/stanley-editor

  TASKS:
  1. Update stanley-gui/Cargo.toml to include stanley-editor
  2. Refactor notes.rs to use new editor component
  3. Implement NotesEditorView using Zed's editor
  4. Add markdown preview toggle
  5. Implement note save/load with persistence
  6. Add keyboard shortcuts (Ctrl+S save, Ctrl+B bold, etc.)

  DESIGN:
  - Each note is a separate buffer
  - Notes saved to ~/.stanley/notes/*.md
  - Support multiple open notes (tabs)

  HOOKS:
  npx claude-flow@alpha hooks pre-task --description "Notes integration"
  npx claude-flow@alpha hooks post-edit --file "notes.rs"
`, "coder")

Task("Editor Tester", `
  PHASE 1.4: Testing & Stabilization

  OBJECTIVE: Ensure editor integration is stable and performant.

  PREREQUISITES: Notes Integrator must complete.

  WORKING DIRECTORIES:
  - /home/artur/Repositories/stanley/stanley-gui

  TASKS:
  1. Write unit tests for stanley-editor crate
  2. Create integration tests for notes view
  3. Test keyboard shortcuts and commands
  4. Benchmark editor performance (large documents)
  5. Test markdown rendering
  6. Verify persistence layer works correctly

  TEST SCENARIOS:
  - Create/edit/delete notes
  - Large document (10k+ lines) performance
  - Concurrent note editing
  - Markdown preview sync
  - Undo/redo functionality

  OUTPUT:
  - Test files in stanley-gui/src/tests/
  - Performance report in docs/

  HOOKS:
  npx claude-flow@alpha hooks post-task --task-id "phase1-testing"
`, "tester")
```

### Phase 2 Hive Minds

```javascript
// Single message - spawn all Phase 2 agents concurrently

Task("SDK Integrator", `
  PHASE 2.1: Agent SDK Setup

  OBJECTIVE: Integrate Anthropic Agent SDK into Stanley.

  WORKING DIRECTORIES:
  - /home/artur/Repositories/stanley/stanley/agent/

  TASKS:
  1. Create stanley/agent/ module structure
  2. Add claude-agent-sdk to requirements.txt
  3. Create agent configuration system
  4. Implement AgentRuntime class
  5. Set up API key management (env vars)
  6. Create agent initialization in API startup

  FILES TO CREATE:
  - stanley/agent/__init__.py
  - stanley/agent/runtime.py
  - stanley/agent/config.py
  - stanley/agent/tools/__init__.py

  CONFIGURATION:
  - Support multiple model selection (Opus, Sonnet)
  - Configurable system prompts
  - Rate limiting for API calls

  HOOKS:
  npx claude-flow@alpha hooks pre-task --description "Agent SDK setup"
`, "backend-dev")

Task("Tool Builder", `
  PHASE 2.2: Stanley MCP Tools

  OBJECTIVE: Create MCP tools for Stanley operations.

  WORKING DIRECTORIES:
  - /home/artur/Repositories/stanley/stanley/agent/tools/

  TASKS:
  1. Create base tool interface
  2. Implement portfolio tools:
     - get_portfolio_holdings()
     - get_portfolio_performance()
     - analyze_position_risk()
  3. Implement research tools:
     - analyze_security(symbol)
     - get_valuation_metrics(symbol)
     - compare_peers(symbol)
  4. Implement market tools:
     - get_market_data(symbols)
     - get_sector_flows()
     - get_etf_holdings(symbol)
  5. Implement context tools:
     - get_visible_data()
     - get_current_view()
     - get_selected_items()

  EACH TOOL MUST:
  - Have clear docstring for Claude
  - Return structured JSON
  - Handle errors gracefully
  - Log usage for analytics

  OUTPUT:
  - stanley/agent/tools/portfolio.py
  - stanley/agent/tools/research.py
  - stanley/agent/tools/market.py
  - stanley/agent/tools/context.py

  HOOKS:
  npx claude-flow@alpha hooks post-edit --file "tools/*.py"
`, "backend-dev")

Task("Agent UI Builder", `
  PHASE 2.3: Agent UI Component

  OBJECTIVE: Build agent sidebar in Stanley GUI.

  WORKING DIRECTORIES:
  - /home/artur/Repositories/stanley/stanley-gui/src/

  TASKS:
  1. Create agent.rs module for agent UI
  2. Implement AgentSidebar component:
     - Chat message list
     - Input field with send button
     - Streaming response display
     - Tool execution indicators
  3. Add agent toggle to main navigation
  4. Implement keyboard shortcut (Ctrl+Shift+A)
  5. Create message styling (user vs assistant)
  6. Add code block rendering for responses
  7. Implement tool result cards

  UI REQUIREMENTS:
  - Collapsible sidebar (right side)
  - Resizable width
  - Message history scroll
  - Loading indicators during API calls
  - Error state handling

  FILES:
  - stanley-gui/src/agent.rs (new)
  - stanley-gui/src/components/agent_chat.rs (new)
  - Update app.rs for agent integration
  - Update keyboard.rs for shortcuts

  HOOKS:
  npx claude-flow@alpha hooks pre-task --description "Agent UI"
`, "coder")
```

### Phase 3 Hive Minds

```javascript
// Single message - spawn all Phase 3 agents concurrently

Task("State Observer", `
  PHASE 3.1: State Observation System

  OBJECTIVE: Build system to track user state and actions.

  WORKING DIRECTORIES:
  - /home/artur/Repositories/stanley/stanley-gui/src/context/

  TASKS:
  1. Create context/ module in stanley-gui
  2. Implement ViewStateTracker:
     - Current view enum tracking
     - View history (last 10 views)
     - Time spent per view
  3. Implement SelectionTracker:
     - Selected symbols
     - Highlighted chart regions
     - Table row selections
  4. Implement ActionTracker:
     - Recent user actions (enum)
     - Action timestamps
     - Action parameters
  5. Create unified StateObserver that combines all trackers

  EVENTS TO TRACK:
  - view_changed(from, to)
  - symbol_selected(symbol)
  - chart_zoomed(range)
  - filter_applied(filter)
  - note_opened(note_id)
  - search_performed(query)

  FILES:
  - stanley-gui/src/context/mod.rs
  - stanley-gui/src/context/view_state.rs
  - stanley-gui/src/context/selection.rs
  - stanley-gui/src/context/actions.rs
  - stanley-gui/src/context/observer.rs

  HOOKS:
  npx claude-flow@alpha hooks pre-task --description "State observation"
`, "system-architect")

Task("Context Aggregator", `
  PHASE 3.2: Context Aggregation

  OBJECTIVE: Build context snapshot system for agent consumption.

  WORKING DIRECTORIES:
  - /home/artur/Repositories/stanley/stanley-gui/src/context/

  TASKS:
  1. Create AgentContext struct with all relevant state
  2. Implement ContextSnapshot generator
  3. Build efficient diffing system (only send changes)
  4. Add intent signal detection:
     - Research mode (spending time on research views)
     - Monitoring mode (quick dashboard checks)
     - Analysis mode (comparing/filtering)
     - Note-taking mode (editor focused)
  5. Implement context serialization to JSON
  6. Add context relevance scoring

  CONTEXT STRUCT:
  ```rust
  pub struct AgentContext {
      pub timestamp: DateTime<Utc>,
      pub current_view: View,
      pub previous_views: Vec<View>,
      pub selected_symbols: Vec<String>,
      pub visible_data: Option<VisibleDataSnapshot>,
      pub recent_actions: Vec<UserAction>,
      pub open_notes: Vec<NoteId>,
      pub active_filters: Vec<Filter>,
      pub user_intent: IntentSignal,
      pub session_duration: Duration,
  }
  ```

  FILES:
  - stanley-gui/src/context/aggregator.rs
  - stanley-gui/src/context/snapshot.rs
  - stanley-gui/src/context/intent.rs

  HOOKS:
  npx claude-flow@alpha hooks post-edit --file "context/*.rs"
`, "system-architect")

Task("Context Feeder", `
  PHASE 3.3: Agent Context Feed

  OBJECTIVE: Stream context to agent runtime.

  WORKING DIRECTORIES:
  - /home/artur/Repositories/stanley/stanley-gui/src/context/
  - /home/artur/Repositories/stanley/stanley/agent/

  TASKS:
  1. Create context streaming protocol (WebSocket or channel)
  2. Implement Rust → Python context bridge
  3. Add context injection into agent system prompt
  4. Create context update throttling (max 1/sec)
  5. Implement context compression for efficiency
  6. Add context history buffer (last 5 snapshots)

  RUST SIDE:
  - ContextPublisher that sends updates
  - Debouncing logic for frequent changes
  - Binary serialization option

  PYTHON SIDE:
  - ContextReceiver in agent runtime
  - Context formatting for Claude
  - System prompt template with context

  SYSTEM PROMPT TEMPLATE:
  ```
  You are Stanley's AI assistant with awareness of:
  - Current view: {current_view}
  - Selected securities: {selected_symbols}
  - Recent actions: {recent_actions}
  - User intent: {user_intent}

  [Additional context details...]
  ```

  HOOKS:
  npx claude-flow@alpha hooks post-task --task-id "phase3-context-feed"
`, "backend-dev")

Task("Bidirectional Comms", `
  PHASE 3.4: Bidirectional Communication

  OBJECTIVE: Enable agent to influence UI and vice versa.

  TASKS:
  1. Agent → UI signals:
     - Highlight suggested securities
     - Open specific views
     - Create notes with content
     - Apply filters
     - Show alerts/notifications
  2. UI → Agent triggers:
     - "Explain this" on any element
     - "Analyze selection" context menu
     - Quick actions from keyboard
  3. Shared state:
     - Agent can read clipboard
     - Agent can write to clipboard
     - Selection sync

  IMPLEMENTATION:
  - Create AgentCommand enum for agent-to-UI actions
  - Implement command handler in app.rs
  - Add context menu items for agent actions
  - Create notification system for agent alerts

  FILES:
  - stanley-gui/src/context/commands.rs
  - stanley-gui/src/context/bridge.rs
  - Update app.rs with command handling

  HOOKS:
  npx claude-flow@alpha hooks pre-task --description "Bidirectional comms"
`, "system-architect")
```

### Phase 4 Hive Minds

```javascript
// Single message - spawn all Phase 4 agents concurrently

Task("Smart Notes Developer", `
  PHASE 4.1: Smart Notes Features

  OBJECTIVE: AI-enhanced note-taking features.

  TASKS:
  1. Auto-tagging notes with mentioned securities
  2. Suggested links to related research
  3. Template generation based on context
  4. Summarization of long notes
  5. Research note structure suggestions
  6. Auto-formatting of financial data in notes

  FEATURES:
  - @AAPL auto-links to security profile
  - {{valuation}} template insert current metrics
  - "Summarize" command for long notes
  - AI-suggested headings/structure

  HOOKS:
  npx claude-flow@alpha hooks post-edit --file "notes.rs"
`, "coder")

Task("Research Copilot Developer", `
  PHASE 4.2: Research Copilot

  OBJECTIVE: Proactive AI research assistance.

  TASKS:
  1. "What am I missing?" analysis
  2. Automatic peer comparison suggestions
  3. Red flag detection (accounting, governance)
  4. Thesis challenge mode
  5. Source verification prompts
  6. Data staleness warnings

  TRIGGERS:
  - Viewing security for >30s → offer deep dive
  - Comparing peers → suggest metrics
  - Reading 10-K → highlight key sections

  HOOKS:
  npx claude-flow@alpha hooks pre-task --description "Research copilot"
`, "coder")

Task("Portfolio Assistant Developer", `
  PHASE 4.3: Portfolio Assistant

  OBJECTIVE: AI-powered portfolio management assistance.

  TASKS:
  1. Risk alerts based on positions
  2. Correlation warnings
  3. Rebalancing suggestions
  4. Sector drift detection
  5. "Explain this movement" for P&L
  6. What-if scenario analysis

  ALERTS:
  - Position exceeds X% of portfolio
  - Correlation spike detected
  - Sector overweight/underweight
  - Unusual volume in holdings

  HOOKS:
  npx claude-flow@alpha hooks post-task --task-id "phase4-portfolio"
`, "coder")

Task("Command Integrator", `
  PHASE 4.4: Keyboard & Command Integration

  OBJECTIVE: Agent-aware command system.

  TASKS:
  1. Natural language command palette
  2. Agent-suggested commands
  3. Contextual keyboard shortcuts
  4. Voice command preparation (future)
  5. Command history with AI ranking

  COMMANDS:
  - "Show me AAPL valuation" → navigates + populates
  - "Compare to peers" → opens comparison view
  - "Summarize my notes on MSFT" → AI summary
  - "Alert me if VIX > 20" → creates alert

  FILES:
  - Update keyboard.rs with agent commands
  - Create command_palette.rs
  - Add natural language parser

  HOOKS:
  npx claude-flow@alpha hooks post-edit --file "keyboard.rs"
`, "coder")

Task("Memory System Developer", `
  PHASE 4.5: Memory & Learning

  OBJECTIVE: Persistent agent memory system.

  TASKS:
  1. Session memory persistence
  2. User preference learning
  3. Research history indexing
  4. Conversation memory
  5. Tool usage patterns
  6. Personalized suggestions

  STORAGE:
  - ~/.stanley/agent/memory.db (SQLite)
  - Vector embeddings for semantic search
  - Conversation history compression

  LEARNING:
  - Track which suggestions accepted
  - Learn preferred metrics/views
  - Adapt to research style

  FILES:
  - stanley/agent/memory/__init__.py
  - stanley/agent/memory/store.py
  - stanley/agent/memory/embeddings.py

  HOOKS:
  npx claude-flow@alpha hooks post-task --task-id "phase4-memory"
`, "backend-dev")
```

---

## Complete Hive Mind Launch Sequence

### One-Shot Full Deployment

To launch all phases with proper coordination:

```javascript
// In Claude Code - Single message with full orchestration

// 1. Initialize swarm coordination
mcp__claude-flow__swarm_init({
  topology: "hierarchical",
  maxAgents: 16,
  strategy: "specialized"
})

// 2. Create todos for tracking
TodoWrite({ todos: [
  {id: "p1-1", content: "Phase 1.1: Crate Analysis", status: "in_progress", priority: "critical"},
  {id: "p1-2", content: "Phase 1.2: Editor Extraction", status: "pending", priority: "critical"},
  {id: "p1-3", content: "Phase 1.3: Notes Integration", status: "pending", priority: "critical"},
  {id: "p1-4", content: "Phase 1.4: Testing", status: "pending", priority: "high"},
  {id: "p2-1", content: "Phase 2.1: Agent SDK Setup", status: "pending", priority: "high"},
  {id: "p2-2", content: "Phase 2.2: MCP Tools", status: "pending", priority: "high"},
  {id: "p2-3", content: "Phase 2.3: Agent UI", status: "pending", priority: "high"},
  {id: "p3-1", content: "Phase 3.1: State Observer", status: "pending", priority: "medium"},
  {id: "p3-2", content: "Phase 3.2: Context Aggregator", status: "pending", priority: "medium"},
  {id: "p3-3", content: "Phase 3.3: Context Feed", status: "pending", priority: "medium"},
  {id: "p3-4", content: "Phase 3.4: Bidirectional Comms", status: "pending", priority: "medium"},
  {id: "p4-1", content: "Phase 4.1: Smart Notes", status: "pending", priority: "low"},
  {id: "p4-2", content: "Phase 4.2: Research Copilot", status: "pending", priority: "low"},
  {id: "p4-3", content: "Phase 4.3: Portfolio Assistant", status: "pending", priority: "low"},
  {id: "p4-4", content: "Phase 4.4: Command Integration", status: "pending", priority: "low"},
  {id: "p4-5", content: "Phase 4.5: Memory System", status: "pending", priority: "low"},
]})

// 3. Spawn Phase 1 agents (these can run in parallel)
Task("Crate Analyzer", "...", "code-analyzer")
Task("Editor Extractor", "...", "coder")  // Will coordinate with analyzer
Task("Notes Integrator", "...", "coder")  // Will wait for extractor
Task("Editor Tester", "...", "tester")    // Will wait for integrator

// 4. Phase 2-4 agents spawn after Phase 1 completes
// (Use TaskOutput to monitor, then spawn next phase)
```

### Monitoring Progress

```bash
# Check swarm status
npx claude-flow@alpha status

# Monitor specific tasks
npx claude-flow@alpha task status --all

# View memory/context
npx claude-flow@alpha memory list --namespace "stanley-integration"
```

---

## File Structure After Integration

```
stanley/
├── stanley-gui/
│   ├── Cargo.toml (updated with crate deps)
│   ├── crates/
│   │   └── stanley-editor/          # Extracted Zed editor
│   │       ├── Cargo.toml
│   │       └── src/
│   │           ├── lib.rs
│   │           ├── editor.rs
│   │           ├── element.rs
│   │           ├── display_map/
│   │           └── ...
│   └── src/
│       ├── main.rs
│       ├── app.rs (updated)
│       ├── agent.rs (new)
│       ├── notes.rs (refactored)
│       ├── context/                  # New context module
│       │   ├── mod.rs
│       │   ├── observer.rs
│       │   ├── aggregator.rs
│       │   ├── snapshot.rs
│       │   ├── intent.rs
│       │   ├── commands.rs
│       │   └── bridge.rs
│       ├── components/
│       │   ├── agent_chat.rs (new)
│       │   └── ...
│       └── ...
├── stanley/
│   ├── agent/                        # New agent module
│   │   ├── __init__.py
│   │   ├── runtime.py
│   │   ├── config.py
│   │   ├── tools/
│   │   │   ├── __init__.py
│   │   │   ├── portfolio.py
│   │   │   ├── research.py
│   │   │   ├── market.py
│   │   │   └── context.py
│   │   └── memory/
│   │       ├── __init__.py
│   │       ├── store.py
│   │       └── embeddings.py
│   └── ...
└── docs/
    └── architecture/
        ├── zed_agent_integration_plan.md (this file)
        ├── zed_crate_analysis.md (generated)
        └── ...
```

---

## Dependencies to Add

### Python (requirements.txt)
```
# Agent SDK
claude-agent-sdk>=0.1.0

# Memory/Embeddings (optional)
chromadb>=0.4.0
sentence-transformers>=2.2.0
```

### Rust (stanley-gui/Cargo.toml)
```toml
[workspace]
members = [".", "crates/stanley-editor"]

[dependencies]
stanley-editor = { path = "crates/stanley-editor" }

# May need from Zed
text = { git = "https://github.com/zed-industries/zed", package = "text" }
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Zed crate extraction complexity | Start with minimal viable subset, iterate |
| GPUI version incompatibility | Pin versions, test early |
| Agent latency blocking UI | Async runtime, streaming, debouncing |
| Context token overflow | Relevance filtering, compression |
| Memory/performance | Lazy loading, snapshot diffing |
| Licensing concerns | Document Apache 2.0/GPL crate origins |

---

## Success Criteria

### Phase 1 Complete When:
- [ ] Notes view uses Zed editor component
- [ ] Markdown preview works
- [ ] Notes persist to disk
- [ ] All keyboard shortcuts functional
- [ ] Performance: 60fps with 10k line document

### Phase 2 Complete When:
- [ ] Agent SDK initialized on startup
- [ ] 10+ MCP tools functional
- [ ] Agent sidebar shows in UI
- [ ] Streaming responses display correctly
- [ ] Tool results render properly

### Phase 3 Complete When:
- [ ] Context snapshots generated on state change
- [ ] Agent receives context in system prompt
- [ ] Intent detection working
- [ ] Bidirectional commands functional
- [ ] <100ms context update latency

### Phase 4 Complete When:
- [ ] Smart notes features active
- [ ] Research copilot suggestions appear
- [ ] Portfolio alerts trigger
- [ ] Natural language commands work
- [ ] Memory persists across sessions

---

## Quick Start Command

To begin Phase 1 immediately:

```
User prompt to Claude Code:

"Initialize a hierarchical swarm and spawn 4 Opus hive minds to execute
Phase 1 of the Zed + Agent SDK integration plan located at
/home/artur/Repositories/stanley/docs/architecture/zed_agent_integration_plan.md

Start with:
1. Crate Analyzer - analyze Zed editor dependencies
2. Editor Extractor - extract stanley-editor crate
3. Notes Integrator - integrate into Stanley notes view
4. Editor Tester - test and benchmark

Use claude-flow for coordination. Report progress via memory updates."
```

---

*Plan created: 2025-12-27*
*Version: 1.0*
*Author: Claude Opus 4.5 via Hive Mind Orchestration*
