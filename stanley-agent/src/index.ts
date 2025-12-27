/**
 * Stanley Agent - AI-powered investment research assistant
 *
 * This is the main entry point for the Stanley agent runtime.
 * It creates an AI agent with access to Stanley's investment research tools.
 *
 * Features:
 * - Context-aware prompting with automatic intent detection
 * - Multiple personas (analyst, quant, macro, risk, pm, trader)
 * - Dynamic prompt building with token management
 * - Portfolio, research, and market context injection
 */

import {
  createStanleyAgent,
  createContextAwareAgent,
  StanleyAgent,
  Message,
  type ContextData,
  type PromptTemplateKey,
  type PersonaId,
} from "./agents/stanley";
import { createStanleyTools, getToolNames } from "./mcp/tools";
import { createProvider, ProviderInstance } from "./providers/factory";
import { createDefaultConfig, AgentConfig, loadConfigFromFile } from "./providers/config";
import { SessionManager, SessionAnalytics } from "./session";
import type { SessionState } from "./session";
import * as readline from "readline";

// Import prompts module for direct access
import * as prompts from "./prompts";

/**
 * Agent runtime instance
 */
interface AgentRuntime {
  agent: StanleyAgent;
  provider: ProviderInstance;
  config: AgentConfig;
  sessionManager: SessionManager;
}

/**
 * Extended agent config with context-aware options
 */
interface ExtendedAgentConfig extends AgentConfig {
  /** Enable context-aware prompting */
  enableContextAwarePrompting?: boolean;
  /** Initial context data */
  initialContext?: ContextData;
  /** Force a specific template */
  forceTemplate?: PromptTemplateKey;
  /** Force a specific persona */
  forcePersona?: PersonaId;
}

/**
 * Initialize the Stanley agent with configuration
 */
export async function initAgent(config?: ExtendedAgentConfig): Promise<AgentRuntime> {
  const agentConfig = config || createDefaultConfig();

  console.log(`Initializing Stanley agent with provider: ${agentConfig.provider.type}`);
  console.log(`Model: ${agentConfig.provider.defaultModel}`);
  console.log(`Stanley API: ${agentConfig.stanleyApiUrl}`);

  // Create provider instance
  const provider = await createProvider(agentConfig.provider);

  // Create session manager with callbacks
  const sessionManager = new SessionManager({
    persistence: {},
    maxConcurrentSessions: 5,
    autoResumeLastSession: true,
    onSessionStart: (sessionId) => {
      if (agentConfig.enableLogging) {
        console.log(`[Session] Started: ${sessionId}`);
      }
    },
    onSessionEnd: (sessionId, analytics) => {
      if (agentConfig.enableLogging) {
        console.log(`[Session] Ended: ${sessionId}`);
        console.log(`[Session] Duration: ${analytics.duration.durationFormatted}`);
        console.log(`[Session] Total tokens: ${analytics.tokens.totalTokens}`);
      }
    },
    onSessionExpired: (sessionId) => {
      if (agentConfig.enableLogging) {
        console.log(`[Session] Expired: ${sessionId}`);
      }
    },
  });

  // Create Stanley tools
  const tools = createStanleyTools({
    baseUrl: agentConfig.stanleyApiUrl,
    timeout: agentConfig.toolTimeout,
  });

  console.log(`Loaded ${getToolNames(tools).length} tools`);

  // Determine if context-aware prompting should be enabled
  const enableContextAware = config?.enableContextAwarePrompting ?? false;

  // Create agent with optional context-aware prompting
  const agent = enableContextAware
    ? createContextAwareAgent({
        model: provider.model,
        tools,
        maxOutputTokens: 4096,
        initialContext: config?.initialContext,
        promptBuilderOptions: {
          forceTemplate: config?.forceTemplate,
          forcePersona: config?.forcePersona,
        },
        onToolCall: (name, args) => {
          if (agentConfig.enableLogging) {
            console.log(`\n[Tool Call] ${name}:`, JSON.stringify(args, null, 2));
          }
        },
        onToolResult: (name, result) => {
          if (agentConfig.enableLogging) {
            const resultStr = JSON.stringify(result, null, 2);
            const truncated = resultStr.length > 500
              ? resultStr.slice(0, 500) + "..."
              : resultStr;
            console.log(`[Tool Result] ${name}:`, truncated);
          }
        },
        onError: (error) => {
          console.error(`[Error]`, error.message);
        },
      })
    : createStanleyAgent({
        model: provider.model,
        tools,
        maxOutputTokens: 4096,
        onToolCall: (name, args) => {
          if (agentConfig.enableLogging) {
            console.log(`\n[Tool Call] ${name}:`, JSON.stringify(args, null, 2));
          }
        },
        onToolResult: (name, result) => {
          if (agentConfig.enableLogging) {
            const resultStr = JSON.stringify(result, null, 2);
            const truncated = resultStr.length > 500
              ? resultStr.slice(0, 500) + "..."
              : resultStr;
            console.log(`[Tool Result] ${name}:`, truncated);
          }
        },
        onError: (error) => {
          console.error(`[Error]`, error.message);
        },
      });

  if (enableContextAware) {
    console.log("Context-aware prompting: enabled");
  }

  return { agent, provider, config: agentConfig, sessionManager };
}

/**
 * Run interactive chat session
 */
export async function runInteractiveChat(runtime: AgentRuntime): Promise<void> {
  const { agent, provider, config, sessionManager } = runtime;

  // Resume or create session
  const session = await sessionManager.resumeOrCreate(
    provider.provider,
    provider.modelId
  );

  // Get existing messages from session
  const messages = sessionManager.getConversationHistory();

  console.log("\n========================================");
  console.log("Stanley Investment Research Assistant");
  console.log(`Provider: ${provider.provider} | Model: ${provider.modelId}`);
  console.log(`Session: ${session.metadata.id}`);
  if (messages.length > 0) {
    console.log(`Resumed session with ${messages.length} messages`);
  }
  if (agent.isContextAwarePromptingEnabled()) {
    console.log("Mode: Context-Aware Prompting");
  }
  console.log("========================================");
  console.log("Commands:");
  console.log("  exit     - Quit the session");
  console.log("  clear    - Reset conversation");
  console.log("  save     - Save session to disk");
  console.log("  stats    - Show session analytics");
  console.log("  sessions - List saved sessions");
  console.log("  context  - Show current context settings");
  console.log("  intent   - Show detected intent for last query");
  console.log("Available tools:", agent.getToolNames().join(", "));
  console.log("");

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const prompt = (): Promise<string> => {
    return new Promise((resolve) => {
      rl.question("You: ", (answer) => {
        resolve(answer);
      });
    });
  };

  let lastIntent: ReturnType<typeof agent.detectQueryIntent> | undefined;

  while (true) {
    const input = await prompt();
    const command = input.toLowerCase().trim();

    if (command === "exit") {
      console.log("Saving session...");
      await sessionManager.endSession();
      console.log("Goodbye!");
      rl.close();
      break;
    }

    if (command === "clear") {
      sessionManager.clearConversation();
      console.log("Conversation cleared.\n");
      continue;
    }

    if (command === "save") {
      await sessionManager.saveCurrentSession();
      console.log("Session saved.\n");
      continue;
    }

    if (command === "stats") {
      const analytics = sessionManager.getAnalytics();
      if (analytics) {
        console.log("\n" + SessionAnalytics.generateSummary(analytics) + "\n");
      }
      continue;
    }

    if (command === "sessions") {
      const sessions = await sessionManager.listSessions();
      console.log("\nSaved sessions:");
      for (const s of sessions.slice(0, 10)) {
        const date = new Date(s.lastModified).toLocaleString();
        console.log(`  - ${s.sessionId} (${date})`);
      }
      console.log("");
      continue;
    }

    if (command === "context") {
      console.log("\nContext-Aware Prompting:", agent.isContextAwarePromptingEnabled() ? "enabled" : "disabled");
      if (agent.isContextAwarePromptingEnabled()) {
        const ctx = agent.getContext();
        console.log("Current Context:");
        console.log("  Portfolio positions:", ctx.portfolio?.length ?? 0);
        console.log("  Research findings:", ctx.research?.length ?? 0);
        console.log("  Watchlist symbols:", ctx.watchlist?.length ?? 0);
        console.log("  Notes:", ctx.notes?.length ?? 0);
        console.log("  Regime:", ctx.regime ? "set" : "not set");
      }
      console.log("");
      continue;
    }

    if (command === "intent") {
      if (lastIntent) {
        console.log("\nLast Query Intent:");
        console.log(`  Intent: ${lastIntent.intent}`);
        console.log(`  Confidence: ${(lastIntent.confidence * 100).toFixed(0)}%`);
        console.log(`  Keywords: ${lastIntent.keywords.join(", ") || "none"}`);
        console.log(`  Symbols: ${lastIntent.symbols.join(", ") || "none"}`);
      } else {
        console.log("\nNo query processed yet.");
      }
      console.log("");
      continue;
    }

    if (!input.trim()) {
      continue;
    }

    // Detect intent for this query
    if (agent.isContextAwarePromptingEnabled()) {
      lastIntent = agent.detectQueryIntent(input);
    }

    // Add message to session
    const userMessage: Message = { role: "user", content: input };
    sessionManager.addMessage(userMessage);

    try {
      console.log("\nStanley: ");

      const startTime = Date.now();

      // Use streaming for better UX
      const currentHistory = sessionManager.getConversationHistory();
      const stream = agent.chatStream(currentHistory);
      let response: Awaited<ReturnType<typeof agent.chat>> | undefined;

      for await (const chunk of stream) {
        process.stdout.write(chunk);
      }

      // Get final result from generator return value
      const result = await stream.next();
      if (result.done && result.value) {
        response = result.value;
      }

      console.log("\n");

      if (response) {
        // Add assistant message to session
        sessionManager.addMessage({ role: "assistant", content: response.text });

        // Record token usage
        if (response.usage) {
          sessionManager.recordTokens(
            response.usage.promptTokens,
            response.usage.completionTokens
          );
        }

        // Record tool calls
        if (response.toolCalls) {
          for (const call of response.toolCalls) {
            sessionManager.recordTool(
              call.toolName,
              call.args,
              call.result,
              Date.now() - startTime,
              true
            );
          }
        }

        // Show usage and prompt metadata
        const parts: string[] = [];
        if (response.usage) {
          parts.push(`Tokens: ${response.usage.promptTokens} in, ${response.usage.completionTokens} out`);
        }
        if (response.promptMetadata) {
          parts.push(`Intent: ${response.promptMetadata.intent.intent}`);
          parts.push(`Persona: ${response.promptMetadata.persona}`);
        }
        if (parts.length > 0) {
          console.log(`[${parts.join(" | ")}]`);
        }
      }
    } catch (error) {
      console.error("\nError:", error instanceof Error ? error.message : "Unknown error");
      // Remove failed user message from session
      const history = sessionManager.getConversationHistory();
      if (history.length > 0 && history[history.length - 1].role === "user") {
        history.pop();
      }
    }
  }
}

/**
 * Run a single query (non-interactive mode)
 */
export async function runSingleQuery(
  runtime: AgentRuntime,
  query: string
): Promise<string> {
  const { agent, provider, sessionManager } = runtime;

  // Create a session for the single query
  await sessionManager.createSession(provider.provider, provider.modelId);

  const userMessage: Message = { role: "user", content: query };
  sessionManager.addMessage(userMessage);

  const messages = sessionManager.getConversationHistory();
  const response = await agent.chat(messages);

  // Record the response
  sessionManager.addMessage({ role: "assistant", content: response.text });
  if (response.usage) {
    sessionManager.recordTokens(response.usage.promptTokens, response.usage.completionTokens);
  }

  // End the session
  await sessionManager.endSession();

  return response.text;
}

/**
 * Main entry point
 */
async function main() {
  const args = process.argv.slice(2);

  // Parse command line arguments
  let configPath: string | undefined;
  let query: string | undefined;
  let providerArg: string | undefined;
  let contextAware = false;
  let forceTemplate: PromptTemplateKey | undefined;
  let forcePersona: PersonaId | undefined;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--config" && args[i + 1]) {
      configPath = args[++i];
    } else if (args[i] === "--query" && args[i + 1]) {
      query = args[++i];
    } else if (args[i] === "--provider" && args[i + 1]) {
      providerArg = args[++i];
    } else if (args[i] === "--context-aware") {
      contextAware = true;
    } else if (args[i] === "--template" && args[i + 1]) {
      forceTemplate = args[++i] as PromptTemplateKey;
    } else if (args[i] === "--persona" && args[i + 1]) {
      forcePersona = args[++i] as PersonaId;
    } else if (args[i] === "--help") {
      console.log(`
Stanley Agent - AI-powered investment research assistant

Usage:
  bun run src/index.ts [options]

Options:
  --config <path>      Path to configuration file
  --provider <name>    Provider to use (anthropic, openai, openrouter, groq, ollama)
  --query <text>       Run a single query and exit
  --context-aware      Enable context-aware prompting
  --template <name>    Force a specific template (base, research, portfolio, risk, macro, trade, quant, earnings, sector)
  --persona <name>     Force a specific persona (analyst, quant, macro, risk, pm, trader)
  --help               Show this help message

Environment Variables:
  ANTHROPIC_API_KEY     Anthropic API key
  OPENAI_API_KEY        OpenAI API key
  OPENROUTER_API_KEY    OpenRouter API key
  GROQ_API_KEY          Groq API key
  STANLEY_API_URL       Stanley API URL (default: http://localhost:8000)

Examples:
  bun run src/index.ts
  bun run src/index.ts --provider openrouter
  bun run src/index.ts --context-aware --persona risk
  bun run src/index.ts --query "What's the current price of AAPL?"
  bun run src/index.ts --context-aware --template portfolio --query "Analyze my portfolio risk"
`);
      process.exit(0);
    }
  }

  // Load configuration
  let config: ExtendedAgentConfig;
  if (configPath) {
    const baseConfig = await loadConfigFromFile(configPath);
    config = {
      ...baseConfig,
      enableContextAwarePrompting: contextAware,
      forceTemplate,
      forcePersona,
    };
  } else if (providerArg) {
    const baseConfig = createDefaultConfig(providerArg as any);
    config = {
      ...baseConfig,
      enableContextAwarePrompting: contextAware,
      forceTemplate,
      forcePersona,
    };
  } else {
    const baseConfig = createDefaultConfig();
    config = {
      ...baseConfig,
      enableContextAwarePrompting: contextAware,
      forceTemplate,
      forcePersona,
    };
  }

  try {
    const runtime = await initAgent(config);

    if (query) {
      // Single query mode
      const response = await runSingleQuery(runtime, query);
      console.log(response);
    } else {
      // Interactive mode
      await runInteractiveChat(runtime);
    }
  } catch (error) {
    console.error("Failed to initialize agent:", error instanceof Error ? error.message : error);
    process.exit(1);
  }
}

// Run if executed directly
if (import.meta.main) {
  main().catch(console.error);
}

// Export for library usage
export {
  StanleyAgent,
  createStanleyAgent,
  createContextAwareAgent,
  createStanleyTools,
  createProvider,
  createDefaultConfig,
  loadConfigFromFile,
  prompts,
  SessionManager,
  SessionAnalytics,
};

export type { AgentConfig, Message, AgentRuntime, ExtendedAgentConfig, SessionState };

// Re-export prompts types
export type {
  ContextData,
  PromptTemplateKey,
  PersonaId,
  BuiltPrompt,
  BuilderOptions,
  QueryIntent,
  IntentResult,
} from "./agents/stanley";
