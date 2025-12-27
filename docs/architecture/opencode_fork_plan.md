# OpenCode Fork Plan for Stanley

## Executive Summary

This document provides a comprehensive analysis of the OpenCode repository structure and outlines a detailed plan for forking and adapting OpenCode's architecture to enhance Stanley's AI agent capabilities for institutional investment analysis.

OpenCode is a fully-featured open-source AI coding agent that provides:
- Multi-provider LLM support (Anthropic, OpenAI, Google, OpenRouter, AWS Bedrock, Azure, and 15+ more)
- Dynamic model registry via models.dev integration
- MCP (Model Context Protocol) tools architecture
- Plugin system for extensibility
- Session management with agent orchestration
- Skill-based extensibility

---

## Repository Structure Analysis

### Top-Level Structure

```
opencode/
  packages/           # Monorepo packages
    opencode/         # Main CLI/agent package
    console/          # Web console (app, core, function, mail)
    plugin/           # Plugin SDK
    sdk/              # Client SDK
    app/              # TUI application
    desktop/          # Desktop wrapper
    web/              # Web frontend
    docs/             # Documentation
  sdks/               # Additional SDKs (vscode)
  infra/              # SST infrastructure
  specs/              # API specifications
```

### Core Package: `packages/opencode`

The heart of the system with the following structure:

```
packages/opencode/src/
  agent/              # Agent definitions and configuration
  auth/               # Authentication management
  bus/                # Event bus for inter-component communication
  cli/                # CLI commands and TUI
  config/             # Configuration management (JSONC support)
  env/                # Environment variable handling
  file/               # File operations (ripgrep integration)
  global/             # Global paths and settings
  mcp/                # MCP server integration
  permission/         # Permission management (ask/allow/deny)
  plugin/             # Plugin loader and lifecycle
  project/            # Project instance management
  provider/           # LLM provider management
  session/            # Session and message management
  skill/              # Skill system (markdown-based)
  snapshot/           # Git snapshot tracking
  tool/               # Built-in tools (bash, edit, read, write, etc.)
  util/               # Utilities (log, filesystem, etc.)
```

---

## Key Component Deep-Dive

### 1. Provider System (`src/provider/provider.ts`)

**Architecture:**
- Namespace-based module with `Provider` as the primary export
- Loads models from `models.dev` (external model registry)
- Bundled providers for zero-latency initialization:

```typescript
const BUNDLED_PROVIDERS: Record<string, (options: any) => SDK> = {
  "@ai-sdk/amazon-bedrock": createAmazonBedrock,
  "@ai-sdk/anthropic": createAnthropic,
  "@ai-sdk/azure": createAzure,
  "@ai-sdk/google": createGoogleGenerativeAI,
  "@ai-sdk/google-vertex": createVertex,
  "@ai-sdk/openai": createOpenAI,
  "@ai-sdk/openai-compatible": createOpenAICompatible,
  "@openrouter/ai-sdk-provider": createOpenRouter,
  "@ai-sdk/xai": createXai,
  "@ai-sdk/mistral": createMistral,
  "@ai-sdk/groq": createGroq,
  "@ai-sdk/deepinfra": createDeepInfra,
  "@ai-sdk/cerebras": createCerebras,
  "@ai-sdk/cohere": createCohere,
  "@ai-sdk/gateway": createGateway,
  "@ai-sdk/togetherai": createTogetherAI,
  "@ai-sdk/perplexity": createPerplexity,
  "@ai-sdk/github-copilot": createGitHubCopilotOpenAICompatible,
}
```

**Custom Loaders:**
- Provider-specific initialization (e.g., Anthropic headers, OpenRouter headers)
- AWS Bedrock credential chain integration
- Azure Cognitive Services configuration
- Cloudflare AI Gateway support

**Model Schema:**
```typescript
Model = z.object({
  id: z.string(),
  providerID: z.string(),
  api: z.object({ id, url, npm }),
  name: z.string(),
  capabilities: z.object({
    temperature, reasoning, attachment, toolcall,
    input: { text, audio, image, video, pdf },
    output: { text, audio, image, video, pdf },
    interleaved: boolean | { field: 'reasoning_content' | 'reasoning_details' }
  }),
  cost: z.object({ input, output, cache: { read, write } }),
  limit: z.object({ context, output }),
  status: z.enum(['alpha', 'beta', 'deprecated', 'active']),
  options: z.record(z.string(), z.any()),
  headers: z.record(z.string(), z.string()),
})
```

### 2. Models Registry (`src/provider/models.ts`)

**models.dev Integration:**
- Fetches model definitions from `https://models.dev/api.json`
- Caches locally in `~/.opencode/cache/models.json`
- Auto-refreshes every hour
- Can be disabled via `OPENCODE_DISABLE_MODELS_FETCH` flag

**Key Functions:**
```typescript
export async function get() {
  refresh()
  // Load from cache or bundled data
}

export async function refresh() {
  // Fetch from models.dev and cache
}
```

### 3. Agent System (`src/agent/agent.ts`)

**Agent Types:**
- `build` - Default full-access development agent
- `plan` - Read-only analysis agent (denies edits, asks for bash)
- `general` - Subagent for complex tasks (hidden from user)
- `explore` - Fast codebase exploration subagent
- `compaction`, `title`, `summary` - Internal agents

**Agent Configuration:**
```typescript
Info = z.object({
  name: z.string(),
  description: z.string().optional(),
  mode: z.enum(['subagent', 'primary', 'all']),
  native: z.boolean().optional(),
  hidden: z.boolean().optional(),
  default: z.boolean().optional(),
  topP: z.number().optional(),
  temperature: z.number().optional(),
  color: z.string().optional(),
  permission: z.object({
    edit: Permission,
    bash: z.record(z.string(), Permission),
    skill: z.record(z.string(), Permission),
    webfetch: Permission.optional(),
    doom_loop: Permission.optional(),
    external_directory: Permission.optional(),
  }),
  model: z.object({ modelID, providerID }).optional(),
  prompt: z.string().optional(),
  tools: z.record(z.string(), z.boolean()),
  options: z.record(z.string(), z.any()),
  maxSteps: z.number().int().positive().optional(),
})
```

### 4. MCP Integration (`src/mcp/index.ts`)

**MCP Client Management:**
- Supports local (stdio) and remote (HTTP/SSE) MCP servers
- OAuth support for remote servers (RFC 7591 dynamic registration)
- Tool list change notifications
- Automatic reconnection and status tracking

**MCP Status Types:**
```typescript
Status = z.discriminatedUnion('status', [
  { status: 'connected' },
  { status: 'disabled' },
  { status: 'failed', error: string },
  { status: 'needs_auth' },
  { status: 'needs_client_registration', error: string },
])
```

**MCP Configuration:**
```typescript
// Local MCP server
McpLocal = z.object({
  type: z.literal('local'),
  command: z.string().array(),
  environment: z.record(z.string(), z.string()).optional(),
  enabled: z.boolean().optional(),
  timeout: z.number().int().positive().optional(),
})

// Remote MCP server
McpRemote = z.object({
  type: z.literal('remote'),
  url: z.string(),
  enabled: z.boolean().optional(),
  headers: z.record(z.string(), z.string()).optional(),
  oauth: z.union([McpOAuth, z.literal(false)]).optional(),
  timeout: z.number().int().positive().optional(),
})
```

### 5. Tool Registry (`src/tool/registry.ts`)

**Built-in Tools:**
```typescript
tools = [
  InvalidTool,      // Fallback for invalid tool calls
  BashTool,         // Shell command execution
  ReadTool,         // File reading
  GlobTool,         // File pattern matching
  GrepTool,         // Content search (ripgrep)
  EditTool,         // File editing
  WriteTool,        // File writing
  TaskTool,         // Subagent spawning
  WebFetchTool,     // Web content fetching
  TodoWriteTool,    // Todo list management
  TodoReadTool,     // Todo list reading
  WebSearchTool,    # Exa web search (premium)
  CodeSearchTool,   # Exa code search (premium)
  SkillTool,        // Skill invocation
  LspTool,          // LSP integration (experimental)
  BatchTool,        // Batch operations (experimental)
]
```

**Tool Interface:**
```typescript
Tool.Info = {
  id: string,
  init: (ctx: { agent?: Agent.Info }) => Promise<{
    parameters: z.ZodType,
    description: string,
    execute: (args, ctx) => Promise<{
      title: string,
      output: string,
      metadata: Record<string, any>,
    }>
  }>
}
```

### 6. Session Processor (`src/session/processor.ts`)

**Agent Loop:**
- Processes LLM stream responses
- Handles tool calls with permission checking
- Doom loop detection (3 identical tool calls triggers warning)
- Automatic retry with exponential backoff
- Snapshot tracking for file changes
- Session summarization

**Stream Events:**
- `start`, `reasoning-start/delta/end`
- `tool-input-start/delta/end`, `tool-call`, `tool-result`, `tool-error`
- `text-start/delta/end`
- `start-step`, `finish-step`, `finish`

### 7. Plugin System (`src/plugin/index.ts`, `packages/plugin/`)

**Plugin Interface:**
```typescript
Plugin = (input: PluginInput) => Promise<Hooks>

PluginInput = {
  client: OpencodeClient,
  project: Project,
  directory: string,
  worktree: string,
  $: BunShell,
}

Hooks = {
  event?: (input) => Promise<void>,
  config?: (input) => Promise<void>,
  tool?: { [key: string]: ToolDefinition },
  auth?: AuthHook,
  'chat.message'?: (input, output) => Promise<void>,
  'chat.params'?: (input, output) => Promise<void>,
  'permission.ask'?: (input, output) => Promise<void>,
  'tool.execute.before'?: (input, output) => Promise<void>,
  'tool.execute.after'?: (input, output) => Promise<void>,
  'experimental.chat.messages.transform'?: (input, output) => Promise<void>,
  'experimental.chat.system.transform'?: (input, output) => Promise<void>,
  'experimental.session.compacting'?: (input, output) => Promise<void>,
  'experimental.text.complete'?: (input, output) => Promise<void>,
}
```

**Default Plugins:**
- `opencode-copilot-auth@0.0.9` - GitHub Copilot authentication
- `opencode-anthropic-auth@0.0.5` - Anthropic authentication

### 8. Skill System (`src/skill/skill.ts`)

**Skill Discovery:**
- Scans `skill/**/SKILL.md` in config directories
- Markdown frontmatter with name and description
- Invoked via `SkillTool`

**Skill Schema:**
```typescript
Info = z.object({
  name: z.string(),
  description: z.string(),
  location: z.string(),
})
```

### 9. Configuration System (`src/config/config.ts`)

**Config Loading Order:**
1. Global config: `~/.opencode/{config.json,opencode.json,opencode.jsonc}`
2. `OPENCODE_CONFIG` environment variable
3. Project `.opencode/` directories (walking up from cwd)
4. `OPENCODE_CONFIG_CONTENT` environment variable
5. Well-known URLs for authenticated providers

**Config Features:**
- JSONC support (comments allowed)
- `{env:VAR_NAME}` environment variable interpolation
- `{file:path}` file content inclusion
- Agent/mode definitions via markdown files

---

## Fork Strategy for Stanley

### Phase 1: Provider Layer Adaptation

**Goal:** Enable Stanley to use OpenCode's multi-provider architecture

**Tasks:**
1. Extract `provider/provider.ts` module
2. Adapt for Python/Rust hybrid architecture
3. Focus on OpenRouter integration first
4. Add Stanley-specific providers:
   - Financial data APIs (Alpha Vantage, Polygon, Bloomberg)
   - SEC EDGAR integration
   - DBnomics for macro data

**Files to Fork:**
```
packages/opencode/src/provider/
  provider.ts      -> stanley/providers/opencode_provider.py
  models.ts        -> stanley/providers/models_registry.py
  transform.ts     -> stanley/providers/message_transform.py
```

### Phase 2: MCP Tools Architecture

**Goal:** Enable Stanley to leverage MCP tools for financial analysis

**Tasks:**
1. Implement MCP client in Python (or use existing `mcp` package)
2. Create Stanley-specific MCP servers:
   - `stanley-market-data` - Real-time market data
   - `stanley-research` - Research report generation
   - `stanley-portfolio` - Portfolio analysis
   - `stanley-accounting` - SEC filings analysis

**Files to Fork:**
```
packages/opencode/src/mcp/
  index.ts         -> stanley/mcp/client.py
  auth.ts          -> stanley/mcp/auth.py
  oauth-provider.ts-> stanley/mcp/oauth.py
```

### Phase 3: Agent System

**Goal:** Create Stanley-specific agents for financial analysis

**Stanley Agents:**
```python
agents = {
    "analyst": {
        "mode": "primary",
        "description": "Comprehensive financial analysis agent",
        "tools": ["market_data", "fundamental", "technical", "portfolio"],
    },
    "researcher": {
        "mode": "primary",
        "description": "Deep research and due diligence",
        "tools": ["sec_filings", "earnings", "peer_comparison", "dcf"],
    },
    "quant": {
        "mode": "primary",
        "description": "Quantitative analysis and backtesting",
        "tools": ["backtest", "factor_analysis", "risk_metrics"],
    },
    "macro": {
        "mode": "subagent",
        "description": "Macroeconomic analysis subagent",
        "tools": ["dbnomics", "regime_detection", "commodity_correlation"],
    },
}
```

### Phase 4: Tool Registry

**Goal:** Create Stanley-specific tools

**Stanley Tools:**
```python
tools = [
    # Market Data
    "MarketDataTool",      # Real-time quotes, historical data
    "TechnicalTool",       # Technical indicators
    "OptionChainTool",     # Options analysis

    # Fundamental
    "FinancialsTool",      # Financial statements
    "ValuationTool",       # DCF, multiples
    "EarningsTool",        # Earnings analysis
    "PeerComparisonTool",  # Peer group analysis

    # Institutional
    "ThirteenFTool",       # 13F filings
    "DarkPoolTool",        # Dark pool activity
    "MoneyFlowTool",       # Sector money flow

    # Portfolio
    "PortfolioVaRTool",    # Value at Risk
    "BetaTool",            # Beta calculation
    "SectorExposureTool",  # Sector analysis

    # Research
    "SecFilingsTool",      # SEC EDGAR integration
    "TranscriptTool",      # Earnings call transcripts
    "NewsSentimentTool",   # News analysis
]
```

### Phase 5: Plugin Architecture

**Goal:** Enable third-party extensions for Stanley

**Plugin Types:**
1. **Data Plugins** - Additional data sources
2. **Analysis Plugins** - Custom analysis methods
3. **Strategy Plugins** - Trading strategy implementations
4. **Alert Plugins** - Custom notification systems

### Phase 6: GUI Integration

**Goal:** Connect Rust GUI (stanley-gui) with forked OpenCode backend

**Architecture:**
```
stanley-gui (Rust/GPUI)
    |
    v
stanley-api (Python/FastAPI)
    |
    v
stanley-agent (Python + OpenCode fork)
    |
    v
LLM Providers (OpenRouter/Anthropic/OpenAI)
```

---

## Implementation Roadmap

### Sprint 1: Foundation (Week 1-2)
- [ ] Set up OpenCode as git submodule or extract core modules
- [ ] Create Python bindings for provider system
- [ ] Implement models.dev integration
- [ ] Add OpenRouter provider configuration

### Sprint 2: MCP Layer (Week 3-4)
- [ ] Implement MCP client in Python
- [ ] Create `stanley-market-data` MCP server
- [ ] Create `stanley-research` MCP server
- [ ] Add MCP configuration to Stanley config

### Sprint 3: Agent System (Week 5-6)
- [ ] Port agent configuration system
- [ ] Create analyst, researcher, quant agents
- [ ] Implement permission system
- [ ] Add agent switching in GUI

### Sprint 4: Tools (Week 7-8)
- [ ] Port tool registry architecture
- [ ] Implement Stanley-specific tools
- [ ] Add tool result formatting
- [ ] Integrate with existing Stanley analyzers

### Sprint 5: Integration (Week 9-10)
- [ ] Connect GUI to agent system
- [ ] Add session management
- [ ] Implement streaming responses
- [ ] Add error handling and retry logic

### Sprint 6: Polish (Week 11-12)
- [ ] Plugin system implementation
- [ ] Skill system for common workflows
- [ ] Performance optimization
- [ ] Documentation and examples

---

## Key Integration Points

### 1. OpenRouter Configuration

```python
# stanley/config/opencode.json
{
  "provider": {
    "openrouter": {
      "api": "https://openrouter.ai/api/v1",
      "env": ["OPENROUTER_API_KEY"],
      "options": {
        "headers": {
          "HTTP-Referer": "https://stanley.app/",
          "X-Title": "Stanley Investment Analysis"
        }
      }
    }
  },
  "model": "openrouter/anthropic/claude-sonnet-4"
}
```

### 2. Stanley MCP Servers

```python
# stanley/config/opencode.json
{
  "mcp": {
    "stanley-market": {
      "type": "local",
      "command": ["python", "-m", "stanley.mcp.market_server"]
    },
    "stanley-research": {
      "type": "local",
      "command": ["python", "-m", "stanley.mcp.research_server"]
    },
    "stanley-accounting": {
      "type": "local",
      "command": ["python", "-m", "stanley.mcp.accounting_server"]
    }
  }
}
```

### 3. Agent Configuration

```python
# stanley/config/opencode.json
{
  "agent": {
    "analyst": {
      "model": "openrouter/anthropic/claude-sonnet-4",
      "prompt": "You are a senior institutional investment analyst...",
      "tools": {
        "market_data": true,
        "fundamentals": true,
        "technicals": true,
        "bash": false
      }
    }
  }
}
```

---

## Dependencies

### Required Packages

```python
# Python dependencies
ai-sdk-anthropic     # Vercel AI SDK for Python (when available)
mcp                  # MCP SDK
openrouter           # OpenRouter client
httpx                # HTTP client

# Or use TypeScript bridge
bun                  # Bun runtime for TypeScript interop
```

### TypeScript Bridge Option

If Python implementation is too complex, use TypeScript via subprocess:

```python
import subprocess
import json

def call_opencode(method: str, params: dict) -> dict:
    result = subprocess.run(
        ["bun", "run", "opencode-bridge.ts", method, json.dumps(params)],
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout)
```

---

## Risk Assessment

### Technical Risks

1. **Vercel AI SDK Dependency** - OpenCode is tightly coupled to Vercel AI SDK
   - Mitigation: Abstract SDK layer, implement fallbacks

2. **Bun Runtime** - OpenCode requires Bun for some features
   - Mitigation: Isolate Bun-specific code, use subprocess bridge

3. **TypeScript/Python Interop** - Complex type translation
   - Mitigation: Define clear interfaces, use JSON for data exchange

### Operational Risks

1. **Upstream Changes** - OpenCode is actively developed
   - Mitigation: Fork specific version, track upstream selectively

2. **Provider API Changes** - LLM providers frequently update APIs
   - Mitigation: Use models.dev for automatic updates

---

## Conclusion

OpenCode provides a robust, well-architected foundation for AI agent capabilities. By selectively forking its provider system, MCP integration, and agent architecture, Stanley can gain sophisticated multi-provider LLM support without reinventing the wheel.

The key insight is that OpenCode's modular design allows for incremental adoption:
1. Start with provider system for LLM access
2. Add MCP for tool extensibility
3. Implement agents for specialized analysis
4. Finally add plugins and skills for user customization

This phased approach minimizes risk while delivering value at each stage.

---

## Appendix A: File Mapping

| OpenCode File | Stanley File | Priority |
|--------------|--------------|----------|
| `provider/provider.ts` | `providers/llm_provider.py` | P0 |
| `provider/models.ts` | `providers/model_registry.py` | P0 |
| `mcp/index.ts` | `mcp/client.py` | P1 |
| `agent/agent.ts` | `agents/agent_config.py` | P1 |
| `tool/registry.ts` | `tools/registry.py` | P1 |
| `session/processor.ts` | `session/processor.py` | P2 |
| `plugin/index.ts` | `plugins/loader.py` | P2 |
| `skill/skill.ts` | `skills/skill.py` | P3 |
| `config/config.ts` | `config/opencode_config.py` | P1 |

## Appendix B: API Compatibility Notes

### Vercel AI SDK Functions Used

```typescript
// Provider creation
createAnthropic({ apiKey, headers })
createOpenAI({ apiKey, baseURL })
createOpenRouter({ apiKey, headers })

// Model access
provider.languageModel(modelID)
provider.chat(modelID)
provider.responses(modelID)

// Streaming
generateObject({ model, schema, messages })
streamText({ model, messages, tools })
```

### MCP SDK Functions Used

```typescript
// Client
new Client({ name, version })
client.connect(transport)
client.listTools()
client.callTool({ name, arguments })

// Transports
new StdioClientTransport({ command, args, env })
new StreamableHTTPClientTransport(url, { authProvider })
new SSEClientTransport(url, { authProvider })
```

---

*Document Version: 1.0*
*Last Updated: 2025-12-27*
*Author: Hive Mind Analysis Agent*
