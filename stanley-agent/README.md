# Stanley Agent

AI-powered investment research assistant for the Stanley institutional investment platform. Built with the Vercel AI SDK, supporting 20+ LLM providers through OpenRouter.

## Overview

Stanley Agent provides an AI conversational interface to the Stanley investment research platform. It connects to the Stanley Python API and exposes 20+ MCP (Model Context Protocol) tools for:

- Market data and stock prices
- Institutional holdings (13F filings)
- Money flow analysis
- Portfolio analytics (VaR, beta, sector exposure)
- Research reports and valuations
- Commodities and options data
- Research notes management

## Quick Start

```bash
# Install dependencies
bun install

# Set your API key (choose one)
export OPENROUTER_API_KEY="sk-or-..."  # Recommended
# or
export ANTHROPIC_API_KEY="sk-ant-..."
# or
export OPENAI_API_KEY="sk-..."

# Start Stanley API (in another terminal)
cd .. && python -m stanley.api.main

# Run the agent
bun run start
```

## Configuration

### Provider Selection

Stanley Agent supports multiple LLM providers. Set the appropriate environment variable:

| Provider | Environment Variable | Example Model |
|----------|---------------------|---------------|
| OpenRouter | `OPENROUTER_API_KEY` | anthropic/claude-sonnet-4 |
| Anthropic | `ANTHROPIC_API_KEY` | claude-sonnet-4-20250514 |
| OpenAI | `OPENAI_API_KEY` | gpt-4o |
| Google Vertex | `GOOGLE_CLOUD_PROJECT` | gemini-2.0-flash |
| AWS Bedrock | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | anthropic.claude-3-5-sonnet |
| Groq | `GROQ_API_KEY` | llama-3.3-70b-versatile |
| Ollama | `OLLAMA_HOST` (optional) | llama3.2 |

### Additional Configuration

```bash
# Stanley API URL (default: http://localhost:8000)
export STANLEY_API_URL="http://localhost:8000"

# OpenRouter site info (optional)
export OPENROUTER_SITE_URL="https://your-app.com"
export OPENROUTER_SITE_NAME="Your App Name"

# AWS Bedrock region
export AWS_REGION="us-east-1"

# Google Vertex location
export GOOGLE_CLOUD_LOCATION="us-central1"

# Ollama host (for local LLMs)
export OLLAMA_HOST="http://localhost:11434"
```

## Usage

### Interactive Mode

```bash
# Default provider (OpenRouter)
bun run start

# Specific provider
bun run start --provider anthropic
bun run start --provider openai
bun run start --provider groq

# With config file
bun run start --config ~/.stanley/agent.yaml
```

### Single Query Mode

```bash
bun run start --query "What is AAPL trading at?"
bun run start --query "Analyze the tech sector money flow"
```

### Programmatic Usage

```typescript
import { initAgent, runSingleQuery } from "stanley-agent";

const runtime = await initAgent({
  provider: {
    type: "openrouter",
    defaultModel: "anthropic/claude-sonnet-4",
  },
  stanleyApiUrl: "http://localhost:8000",
});

const response = await runSingleQuery(runtime, "What is AAPL's valuation?");
console.log(response);
```

## Available MCP Tools

### Market Data
- `get_market_data` - Current price, volume, change for a symbol
- `get_equity_flow` - Equity money flow data
- `get_dark_pool` - Dark pool trading activity

### Research
- `get_research` - Comprehensive research report
- `get_valuation` - DCF and valuation analysis
- `get_earnings` - Earnings history and surprises
- `get_peers` - Peer comparison analysis

### Institutional
- `get_institutional_holdings` - 13F institutional holdings
- `analyze_money_flow` - Sector money flow analysis

### Portfolio
- `get_portfolio_analytics` - VaR, beta, sector exposure

### Options
- `get_options_flow` - Options trading flow
- `get_gamma_exposure` - Gamma exposure analysis

### Commodities
- `get_commodities` - Commodity market overview
- `get_commodity` - Specific commodity data

### Notes
- `get_notes` - List research notes
- `get_note` - Get specific note
- `save_note` - Create/update note
- `search_notes` - Search notes
- `get_theses` - Investment theses
- `get_trades` - Trade log
- `get_trade_stats` - Trading statistics

### System
- `health_check` - API health status

## Development

### Project Structure

```
stanley-agent/
├── src/
│   ├── agents/
│   │   └── stanley.ts      # Main agent implementation
│   ├── context/            # Phase 3: Context bridge system
│   │   ├── index.ts        # Context bridge exports
│   │   ├── bridge.ts       # ContextBridge class
│   │   └── sources.ts      # Context source implementations
│   ├── memory/             # Phase 3: Memory persistence
│   │   ├── index.ts        # Memory store exports
│   │   └── store.ts        # MemoryStore class
│   ├── sync/               # Phase 3: Real-time sync
│   │   ├── index.ts        # Sync client exports
│   │   └── client.ts       # SyncClient for WebSocket
│   ├── prompts/            # Phase 3: Prompt building
│   │   ├── index.ts        # Prompt builder exports
│   │   ├── template.ts     # PromptTemplate class
│   │   └── builder.ts      # PromptBuilder class
│   ├── session/            # Phase 3: Session management
│   │   ├── index.ts        # Session manager exports
│   │   └── manager.ts      # SessionManager class
│   ├── mcp/
│   │   └── tools.ts        # MCP tool definitions
│   ├── providers/
│   │   ├── config.ts       # Provider configuration
│   │   └── factory.ts      # Provider factory
│   └── index.ts            # Entry point
├── tests/
│   ├── unit/               # Unit tests
│   │   ├── context.test.ts # Context bridge tests
│   │   ├── memory.test.ts  # Memory persistence tests
│   │   ├── sync.test.ts    # Real-time sync tests
│   │   ├── prompts.test.ts # Prompt building tests
│   │   └── session.test.ts # Session management tests
│   └── integration/        # Integration tests
│       ├── context-bridge.test.ts    # Full context flow
│       └── session-lifecycle.test.ts # Session lifecycle
├── scripts/
│   └── e2e-test.sh         # E2E test script
├── package.json
└── tsconfig.json
```

### Running Tests

```bash
# All tests
bun test

# Unit tests only
bun test:unit

# Integration tests (requires API server)
bun test:integration

# E2E tests
bun run test:e2e

# With coverage
bun test:coverage
```

### Test Environment

For CI environments without API keys, tests automatically skip integration tests:

```bash
export SKIP_INTEGRATION_TESTS=true
bun test
```

### Building

```bash
# Build for production
bun run build

# Type checking
bun run typecheck

# Linting
bun run lint

# Formatting
bun run format
```

## Architecture

### Provider Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Stanley Agent                             │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │  Provider       │  │    Agent        │  │    MCP      │  │
│  │  Factory        │  │    Loop         │  │   Tools     │  │
│  │                 │  │                 │  │             │  │
│  │  - Anthropic    │  │  - Chat         │  │  20+ tools  │  │
│  │  - OpenAI       │  │  - Stream       │  │  for API    │  │
│  │  - OpenRouter   │  │  - Tool calls   │  │  access     │  │
│  │  - Google       │  │  - Callbacks    │  │             │  │
│  │  - AWS          │  │                 │  │             │  │
│  │  - Groq         │  │                 │  │             │  │
│  │  - Ollama       │  │                 │  │             │  │
│  └─────────────────┘  └─────────────────┘  └─────────────┘  │
│                              │                               │
└──────────────────────────────┼───────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Stanley Python API                           │
│                 135+ REST Endpoints                          │
└─────────────────────────────────────────────────────────────┘
```

### Tool Execution Flow

1. User sends message
2. Agent analyzes message and decides on tool calls
3. Tools execute HTTP requests to Stanley API
4. Results are fed back to the model
5. Model generates final response

## Phase 3: Context Bridge System

Phase 3 introduces a comprehensive context management system for maintaining state across conversations and sessions.

### Context Bridge

The Context Bridge gathers context from multiple sources to enrich agent interactions:

```typescript
import { ContextBridge, createContextBridge } from "stanley-agent/context";

// Create context bridge with registered sources
const contextBridge = createContextBridge({
  sources: [
    { name: "memory", priority: 1 },
    { name: "session", priority: 2 },
    { name: "project", priority: 3 },
  ],
});

// Gather all context
const context = await contextBridge.gather();

// Build prompt with context
const prompt = await contextBridge.buildPrompt(systemPrompt, tools);
```

### Session Management

Manage conversation sessions with persistence and resumption:

```typescript
import { SessionManager } from "stanley-agent/session";

// Create session manager with storage
const sessions = new SessionManager({
  storagePath: "~/.stanley/sessions.json",
  maxSessions: 100,
  sessionTimeout: 86400000, // 24 hours
});

// Create new session
const session = sessions.create({
  title: "Investment Research",
  tags: ["research", "tech"],
});

// Add conversation turns
sessions.addMessage(session.id, {
  role: "user",
  content: "Analyze AAPL for me",
});

// Set session context
sessions.setContext(session.id, "currentSymbol", "AAPL");

// Save and restore
await sessions.save();
await sessions.load();

// Resume previous session
const resumed = sessions.resume(previousSessionId);
```

### Memory Persistence

Persist data across sessions with TTL and namespacing:

```typescript
import { MemoryStore } from "stanley-agent/memory";

// Create memory store
const memory = new MemoryStore({
  namespace: "user",
  persistPath: "~/.stanley/memory.json",
  autoSave: true,
});

// Store with TTL
await memory.set("cache:AAPL", marketData, { ttl: 60000 });

// Retrieve
const data = await memory.get<MarketData>("cache:AAPL");

// Query by prefix
const cacheKeys = await memory.keys({ prefix: "cache:" });
```

### Real-time Sync

WebSocket-based real-time synchronization:

```typescript
import { SyncClient } from "stanley-agent/sync";

const sync = new SyncClient({
  url: "ws://localhost:8080/sync",
  reconnectInterval: 5000,
  heartbeatInterval: 30000,
});

// Connect and subscribe to events
await sync.connect();

sync.on("market:price", (event) => {
  console.log("Price update:", event.payload);
});

sync.on("agent:stream", (event) => {
  console.log("Streaming:", event.payload.chunk);
});

// Send events
sync.send({
  type: "state:update",
  payload: { path: "user.preferences", value: { theme: "dark" } },
});
```

### Prompt Building

Dynamic prompt construction with templates:

```typescript
import { PromptTemplate, PromptBuilder } from "stanley-agent/prompts";

// Template-based prompts
const template = new PromptTemplate(`
You are analyzing {{symbol}} for {{timeframe}} outlook.

{{#section:portfolio}}
`);

template.setVariable("symbol", { required: true });
template.setVariable("timeframe", { default: "medium-term" });
template.addSection({
  name: "portfolio",
  content: "Portfolio context here",
  condition: (ctx) => ctx.hasPortfolio,
});

const prompt = template.render({ symbol: "AAPL", hasPortfolio: true });

// Builder-based prompts
const builder = new PromptBuilder()
  .setSystemPrompt("You are Stanley, an investment analyst.")
  .addContext("Current market conditions: bullish")
  .addToolDescription("get_market_data", "Fetch real-time prices")
  .addInstruction("Be concise and data-driven");

const fullPrompt = builder.build();
```

### Configuration Options

```typescript
// Full configuration example
const config = {
  // Context bridge
  context: {
    maxTokens: 8000,
    cacheTtl: 60000,
    enableParallel: true,
  },

  // Session management
  session: {
    storagePath: "~/.stanley/sessions.json",
    maxSessions: 100,
    sessionTimeout: 86400000,
  },

  // Memory store
  memory: {
    namespace: "stanley",
    persistPath: "~/.stanley/memory.json",
    autoSave: true,
  },

  // Real-time sync
  sync: {
    url: "ws://localhost:8080/sync",
    reconnectInterval: 5000,
    maxReconnectAttempts: 5,
    heartbeatInterval: 30000,
  },
};
```

## Troubleshooting

### API Connection Issues

```bash
# Check if API is running
curl http://localhost:8000/api/health

# Start the API
cd .. && python -m stanley.api.main
```

### Provider Authentication

```bash
# Test OpenRouter
curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  https://openrouter.ai/api/v1/models

# Test Anthropic
curl -H "x-api-key: $ANTHROPIC_API_KEY" \
  https://api.anthropic.com/v1/messages
```

### Timeout Issues

Increase timeout for slow connections:

```typescript
const config = createDefaultConfig("openrouter");
config.toolTimeout = 60000; // 60 seconds
```

## License

MIT
