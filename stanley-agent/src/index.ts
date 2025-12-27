/**
 * Stanley Agent - AI-powered investment research assistant
 *
 * This is the main entry point for the Stanley agent runtime.
 * It creates an AI agent with access to Stanley's investment research tools.
 */

import { createStanleyAgent, StanleyAgent, Message } from "./agents/stanley";
import { createStanleyTools, getToolNames } from "./mcp/tools";
import { createProvider, ProviderInstance } from "./providers/factory";
import { createDefaultConfig, AgentConfig, loadConfigFromFile } from "./providers/config";
import * as readline from "readline";

/**
 * Agent runtime instance
 */
interface AgentRuntime {
  agent: StanleyAgent;
  provider: ProviderInstance;
  config: AgentConfig;
}

/**
 * Initialize the Stanley agent with configuration
 */
export async function initAgent(config?: AgentConfig): Promise<AgentRuntime> {
  const agentConfig = config || createDefaultConfig();

  console.log(`Initializing Stanley agent with provider: ${agentConfig.provider.type}`);
  console.log(`Model: ${agentConfig.provider.defaultModel}`);
  console.log(`Stanley API: ${agentConfig.stanleyApiUrl}`);

  // Create provider instance
  const provider = await createProvider(agentConfig.provider);

  // Create Stanley tools
  const tools = createStanleyTools({
    baseUrl: agentConfig.stanleyApiUrl,
    timeout: agentConfig.toolTimeout,
  });

  console.log(`Loaded ${getToolNames(tools).length} tools`);

  // Create agent
  const agent = createStanleyAgent({
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

  return { agent, provider, config: agentConfig };
}

/**
 * Run interactive chat session
 */
export async function runInteractiveChat(runtime: AgentRuntime): Promise<void> {
  const { agent, provider, config } = runtime;
  const messages: Message[] = [];

  console.log("\n========================================");
  console.log("Stanley Investment Research Assistant");
  console.log(`Provider: ${provider.provider} | Model: ${provider.modelId}`);
  console.log("========================================");
  console.log("Type 'exit' to quit, 'clear' to reset conversation");
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

  while (true) {
    const input = await prompt();

    if (input.toLowerCase() === "exit") {
      console.log("Goodbye!");
      rl.close();
      break;
    }

    if (input.toLowerCase() === "clear") {
      messages.length = 0;
      console.log("Conversation cleared.\n");
      continue;
    }

    if (!input.trim()) {
      continue;
    }

    messages.push({ role: "user", content: input });

    try {
      console.log("\nStanley: ");

      // Use streaming for better UX
      const stream = agent.chatStream(messages);
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
        messages.push({ role: "assistant", content: response.text });

        if (response.usage) {
          console.log(
            `[Tokens: ${response.usage.promptTokens} in, ${response.usage.completionTokens} out]`
          );
        }
      }
    } catch (error) {
      console.error("\nError:", error instanceof Error ? error.message : "Unknown error");
      // Remove failed user message
      messages.pop();
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
  const { agent } = runtime;
  const messages: Message[] = [{ role: "user", content: query }];

  const response = await agent.chat(messages);
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

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--config" && args[i + 1]) {
      configPath = args[++i];
    } else if (args[i] === "--query" && args[i + 1]) {
      query = args[++i];
    } else if (args[i] === "--provider" && args[i + 1]) {
      providerArg = args[++i];
    } else if (args[i] === "--help") {
      console.log(`
Stanley Agent - AI-powered investment research assistant

Usage:
  bun run src/index.ts [options]

Options:
  --config <path>    Path to configuration file
  --provider <name>  Provider to use (anthropic, openai, openrouter, groq, ollama)
  --query <text>     Run a single query and exit
  --help             Show this help message

Environment Variables:
  ANTHROPIC_API_KEY     Anthropic API key
  OPENAI_API_KEY        OpenAI API key
  OPENROUTER_API_KEY    OpenRouter API key
  GROQ_API_KEY          Groq API key
  STANLEY_API_URL       Stanley API URL (default: http://localhost:8000)

Examples:
  bun run src/index.ts
  bun run src/index.ts --provider openrouter
  bun run src/index.ts --query "What's the current price of AAPL?"
`);
      process.exit(0);
    }
  }

  // Load configuration
  let config: AgentConfig;
  if (configPath) {
    config = await loadConfigFromFile(configPath);
  } else if (providerArg) {
    config = createDefaultConfig(providerArg as any);
  } else {
    config = createDefaultConfig();
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
  createStanleyTools,
  createProvider,
  createDefaultConfig,
  loadConfigFromFile,
};
export type { AgentConfig, Message, AgentRuntime };
