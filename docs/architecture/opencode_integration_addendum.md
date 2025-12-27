# OpenCode Integration Addendum

## Why OpenCode Changes Everything

After analyzing `/home/artur/Repositories/opencode`, we discovered a better foundation for Stanley's AI integration:

### Before: Anthropic Agent SDK Only
- Single provider (Claude)
- Build MCP tools from scratch
- Custom agent loop implementation

### After: OpenCode as Foundation
- **20+ providers** via Vercel AI SDK
- **MCP integration** built-in with OAuth
- **Agent system** ready to use (build, plan, explore)
- **Tool system** identical to Claude Code
- **models.dev** auto-updating model registry

---

## Revised Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STANLEY PLATFORM                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    GPUI Application Layer (Rust)                     │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│  │  │  Stanley    │  │  Zed Editor │  │   Agent     │  │  Context   │  │   │
│  │  │  Views (42) │  │  Component  │  │  Sidebar    │  │  Bridge    │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                        │                                    │
│                                        ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              OpenCode Agent Runtime (TypeScript/Bun)                 │   │
│  │                                                                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │   Provider   │  │    Agent     │  │     MCP      │               │   │
│  │  │   Manager    │  │    System    │  │   Gateway    │               │   │
│  │  │              │  │              │  │              │               │   │
│  │  │ - Anthropic  │  │ - build      │  │ - OAuth      │               │   │
│  │  │ - OpenAI     │  │ - plan       │  │ - Local      │               │   │
│  │  │ - Gemini     │  │ - explore    │  │ - Remote     │               │   │
│  │  │ - Bedrock    │  │ - stanley    │  │ - Tools      │               │   │
│  │  │ - Groq       │  │   (custom)   │  │              │               │   │
│  │  │ - 15+ more   │  │              │  │              │               │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
│  │                                                                      │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │                    Stanley Tools (MCP)                        │   │   │
│  │  │                                                               │   │   │
│  │  │  get_portfolio_holdings  │  analyze_security  │  get_chart   │   │   │
│  │  │  compare_peers          │  search_notes     │  get_flows    │   │   │
│  │  │  run_screen             │  get_valuation    │  alert_create │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                        │                                    │
│                                        ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Stanley API Layer (Python FastAPI)                │   │
│  │                    14 Routers │ 135+ Endpoints                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Provider Configuration

Stanley can now use ANY of these providers based on user subscription:

```yaml
# ~/.stanley/config.yaml
provider:
  # Use Anthropic Claude (default)
  anthropic:
    env: [ANTHROPIC_API_KEY]

  # Or OpenAI
  openai:
    env: [OPENAI_API_KEY]

  # Or Google Cloud (Vertex)
  google-vertex:
    options:
      project: my-gcp-project
      location: us-east5

  # Or AWS Bedrock
  amazon-bedrock:
    options:
      region: us-east-1

  # Or OpenRouter for any model
  openrouter:
    env: [OPENROUTER_API_KEY]

  # Or local LLMs via Groq/Ollama
  groq:
    env: [GROQ_API_KEY]
```

---

## Revised Phase Structure

### Phase 1: Foundation (UNCHANGED)
- Extract Zed editor crates
- Integrate into Stanley notes
- Same 4 hive minds

### Phase 2: OpenCode Integration (REVISED)
**Duration:** 1-2 weeks (faster than before!)
**Hive Minds:** 3 specialized agents

#### 2.1 OpenCode Setup
- Add OpenCode as dependency or submodule
- Configure provider management
- Set up TypeScript/Bun runtime alongside Python

#### 2.2 Stanley Agent Definition
Create custom `stanley` agent in OpenCode style:

```typescript
// stanley-agent/src/agents/stanley.ts
export const stanleyAgent: Agent.Info = {
  name: "stanley",
  description: "Institutional investment research assistant",
  mode: "primary",
  permission: {
    edit: "allow",
    bash: { "*": "ask" },
    skill: { "*": "allow" },
  },
  tools: {
    // Stanley-specific tools
    portfolio: true,
    research: true,
    market: true,
    notes: true,
    // Standard tools
    read: true,
    write: true,
    bash: true,
    websearch: true,
  },
  prompt: `You are Stanley, an AI assistant for institutional investment research.

You have access to:
- Portfolio analytics (VaR, beta, sector exposure)
- Research tools (valuation, earnings, DCF)
- Market data (ETF flows, sector rotation, macro)
- Notes system (markdown research memos)

Current context:
{{context}}

Help the user with their investment research tasks.`,
}
```

#### 2.3 Stanley MCP Server
Create Stanley-specific MCP server with financial tools:

```typescript
// stanley-agent/src/mcp/stanley-server.ts
export const stanleyTools = {
  get_holdings: {
    description: "Get current portfolio holdings",
    parameters: z.object({
      portfolio_id: z.string().optional(),
    }),
    execute: async (args) => {
      // Call Stanley Python API
      return fetch("http://localhost:8000/api/portfolio/holdings")
    },
  },
  analyze_security: {
    description: "Deep analysis of a security",
    parameters: z.object({
      symbol: z.string(),
      include: z.array(z.enum(["valuation", "earnings", "technicals", "peers"])),
    }),
    execute: async (args) => {
      return fetch(`http://localhost:8000/api/research/${args.symbol}`)
    },
  },
  // ... more tools
}
```

### Phase 3: Context Bridge (UNCHANGED)
- State observation
- Context aggregation
- Agent context feed
- Bidirectional communication

### Phase 4: Deep Integration (ENHANCED)
Now with multi-provider support:
- Let users choose their preferred LLM
- Use fast models (Groq) for quick tasks
- Use powerful models (Claude Opus, GPT-4o) for research
- Automatic model selection based on task complexity

---

## New Advantages

| Feature | Before (SDK Only) | After (OpenCode) |
|---------|-------------------|------------------|
| Providers | 1 (Anthropic) | 20+ |
| Model selection | Fixed | Dynamic per-task |
| MCP support | Build from scratch | Built-in |
| Tool system | Build from scratch | Built-in |
| Agent loop | Build from scratch | Built-in |
| OAuth | Build from scratch | Built-in |
| Updates | Manual | Auto (models.dev) |

---

## Integration Steps

### Step 1: Add OpenCode as Git Submodule

```bash
cd /home/artur/Repositories/stanley
git submodule add /home/artur/Repositories/opencode stanley-agent/opencode
# Or link to existing repo
ln -s /home/artur/Repositories/opencode stanley-agent/opencode
```

### Step 2: Create Stanley Agent Package

```bash
mkdir -p stanley-agent/src/{agents,mcp,tools}
cd stanley-agent
bun init
bun add opencode-ai  # Or use local
```

### Step 3: Configure Providers

```bash
# Copy user's existing API keys
cp ~/.opencode/auth.json ~/.stanley/auth.json

# Or configure individually
opencode auth add anthropic
opencode auth add openai
opencode auth add groq
```

### Step 4: Start Stanley Agent Runtime

```bash
# In one terminal: Python API
cd stanley && uvicorn stanley.api.main:app

# In another: OpenCode agent with Stanley tools
cd stanley-agent && bun run start
```

---

## File Structure After Integration

```
stanley/
├── stanley-gui/                    # Rust GPUI frontend
│   ├── crates/stanley-editor/      # Extracted Zed editor
│   └── src/
│       ├── agent.rs                # Agent UI sidebar
│       └── context/                # State observation
├── stanley/                        # Python backend
│   └── api/                        # FastAPI (135+ endpoints)
├── stanley-agent/                  # NEW: OpenCode-based agent
│   ├── opencode/                   # Submodule or symlink
│   ├── src/
│   │   ├── agents/
│   │   │   └── stanley.ts          # Custom Stanley agent
│   │   ├── mcp/
│   │   │   └── stanley-server.ts   # Stanley MCP tools
│   │   └── tools/
│   │       ├── portfolio.ts
│   │       ├── research.ts
│   │       └── market.ts
│   ├── package.json
│   └── tsconfig.json
└── docs/
    └── architecture/
        ├── zed_agent_integration_plan.md
        └── opencode_integration_addendum.md  # This file
```

---

## Quick Start with OpenCode

```bash
# Install opencode globally (if not already)
paru -S opencode-bin  # Arch Linux
# or
npm i -g opencode-ai

# Configure your preferred provider
opencode auth add anthropic  # Or any other

# Start Stanley with AI
cd /home/artur/Repositories/stanley
./start-with-ai.sh
```

---

*Addendum created: 2025-12-27*
*This extends zed_agent_integration_plan.md with OpenCode insights*
