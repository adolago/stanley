/**
 * Stanley Agent Definition
 *
 * The primary AI agent for institutional investment research.
 * Features context-aware prompting for enhanced analytical capabilities.
 */

import { generateText, streamText, type CoreMessage, type LanguageModel, type ToolSet } from "ai";
import {
  PromptBuilder,
  createPromptBuilder,
  buildPrompt,
  detectIntent,
  getPromptTemplate,
  type ContextData,
  type BuiltPrompt,
  type BuilderOptions,
  type PromptTemplateKey,
  type PersonaId,
  type QueryIntent,
  type IntentResult,
} from "../prompts";

/**
 * Agent configuration
 */
export interface AgentOptions {
  model: LanguageModel;
  tools: ToolSet;
  systemPrompt?: string;
  maxOutputTokens?: number;
  onToolCall?: (toolName: string, args: unknown) => void;
  onToolResult?: (toolName: string, result: unknown) => void;
  onError?: (error: Error) => void;
  /** Enable context-aware prompting */
  enableContextAwarePrompting?: boolean;
  /** Initial context data for prompting */
  initialContext?: ContextData;
  /** Default prompt builder options */
  promptBuilderOptions?: BuilderOptions;
}

/**
 * Default system prompt for Stanley agent (fallback)
 */
const DEFAULT_SYSTEM_PROMPT = `You are Stanley, an AI assistant specialized in institutional investment research and analysis.

You have access to tools for:
- Market data and stock prices
- Institutional holdings (13F filings)
- Money flow analysis across sectors
- Portfolio analytics (VaR, beta, sector exposure)
- Dark pool and equity flow data
- Research reports, valuations, and earnings
- Peer comparisons
- Commodity markets
- Options flow and gamma exposure
- Research notes and memos

When helping users:
1. Use tools to gather data before making conclusions
2. Provide clear, data-driven analysis
3. Cite specific metrics and figures
4. Acknowledge uncertainty when data is limited
5. Suggest follow-up analyses when appropriate

Always maintain a professional, institutional-grade perspective in your responses.`;

/**
 * Conversation message type
 */
export type Message = CoreMessage;

/**
 * Agent response type
 */
export interface AgentResponse {
  text: string;
  toolCalls?: Array<{
    toolName: string;
    args: unknown;
    result: unknown;
  }>;
  finishReason: string;
  usage?: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
  /** Prompt metadata when context-aware prompting is enabled */
  promptMetadata?: {
    intent: IntentResult;
    template: PromptTemplateKey;
    persona: PersonaId;
    contextIncluded: string[];
    estimatedPromptTokens: number;
  };
}

/**
 * Stanley Agent class with context-aware prompting
 */
export class StanleyAgent {
  private model: LanguageModel;
  private tools: ToolSet;
  private systemPrompt: string;
  private maxOutputTokens: number;
  private onToolCall?: (toolName: string, args: unknown) => void;
  private onToolResult?: (toolName: string, result: unknown) => void;
  private onError?: (error: Error) => void;

  // Context-aware prompting
  private enableContextAwarePrompting: boolean;
  private promptBuilder: PromptBuilder;
  private contextData: ContextData;
  private promptBuilderOptions: BuilderOptions;

  constructor(options: AgentOptions) {
    this.model = options.model;
    this.tools = options.tools;
    this.systemPrompt = options.systemPrompt || DEFAULT_SYSTEM_PROMPT;
    this.maxOutputTokens = options.maxOutputTokens || 4096;
    this.onToolCall = options.onToolCall;
    this.onToolResult = options.onToolResult;
    this.onError = options.onError;

    // Initialize context-aware prompting
    this.enableContextAwarePrompting = options.enableContextAwarePrompting ?? false;
    this.contextData = options.initialContext || {};
    this.promptBuilderOptions = options.promptBuilderOptions || {};
    this.promptBuilder = createPromptBuilder(this.contextData, this.promptBuilderOptions);
  }

  /**
   * Build system prompt for a query
   * Uses context-aware prompting if enabled, otherwise returns default
   */
  private buildSystemPrompt(query: string): { prompt: string; metadata?: AgentResponse["promptMetadata"] } {
    if (!this.enableContextAwarePrompting) {
      return { prompt: this.systemPrompt };
    }

    const result = this.promptBuilder.build(query, this.promptBuilderOptions);

    return {
      prompt: result.prompt,
      metadata: {
        intent: result.intent,
        template: result.template,
        persona: result.persona,
        contextIncluded: result.context.includedTypes,
        estimatedPromptTokens: result.estimatedTokens,
      },
    };
  }

  /**
   * Generate a response for a single message
   */
  async chat(messages: Message[]): Promise<AgentResponse> {
    const toolCalls: AgentResponse["toolCalls"] = [];

    // Get the last user message for intent detection
    const lastUserMessage = [...messages].reverse().find((m) => m.role === "user");
    const query = lastUserMessage && typeof lastUserMessage.content === "string"
      ? lastUserMessage.content
      : "";

    // Build system prompt
    const { prompt: systemPrompt, metadata: promptMetadata } = this.buildSystemPrompt(query);

    try {
      const result = await generateText({
        model: this.model,
        system: systemPrompt,
        messages,
        tools: this.tools,
        maxOutputTokens: this.maxOutputTokens,
        onStepFinish: (step) => {
          if (step.toolCalls) {
            for (const call of step.toolCalls) {
              const toolName = "toolName" in call ? call.toolName : String(call);
              const args = "args" in call ? call.args : undefined;

              this.onToolCall?.(toolName, args);

              // Find matching result
              const resultEntry = step.toolResults?.find(
                (r) => "toolCallId" in r && "toolCallId" in call && r.toolCallId === call.toolCallId
              );

              if (resultEntry) {
                const resultValue = "result" in resultEntry ? resultEntry.result : undefined;
                this.onToolResult?.(toolName, resultValue);
                toolCalls.push({
                  toolName,
                  args,
                  result: resultValue,
                });
              }
            }
          }
        },
      });

      // Extract usage info safely (v5 uses inputTokens/outputTokens)
      const usage = result.usage ? {
        promptTokens: result.usage.inputTokens ?? 0,
        completionTokens: result.usage.outputTokens ?? 0,
        totalTokens: result.usage.totalTokens ?? 0,
      } : undefined;

      return {
        text: result.text,
        toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
        finishReason: result.finishReason,
        usage,
        promptMetadata,
      };
    } catch (error) {
      if (error instanceof Error) {
        this.onError?.(error);
        throw error;
      }
      throw new Error("Unknown error during chat");
    }
  }

  /**
   * Stream a response for a single message
   */
  async *chatStream(messages: Message[]): AsyncGenerator<string, AgentResponse, unknown> {
    const toolCalls: AgentResponse["toolCalls"] = [];
    let fullText = "";

    // Get the last user message for intent detection
    const lastUserMessage = [...messages].reverse().find((m) => m.role === "user");
    const query = lastUserMessage && typeof lastUserMessage.content === "string"
      ? lastUserMessage.content
      : "";

    // Build system prompt
    const { prompt: systemPrompt, metadata: promptMetadata } = this.buildSystemPrompt(query);

    try {
      const result = streamText({
        model: this.model,
        system: systemPrompt,
        messages,
        tools: this.tools,
        maxOutputTokens: this.maxOutputTokens,
        onStepFinish: (step) => {
          if (step.toolCalls) {
            for (const call of step.toolCalls) {
              const toolName = "toolName" in call ? call.toolName : String(call);
              const args = "args" in call ? call.args : undefined;

              this.onToolCall?.(toolName, args);

              const resultEntry = step.toolResults?.find(
                (r) => "toolCallId" in r && "toolCallId" in call && r.toolCallId === call.toolCallId
              );

              if (resultEntry) {
                const resultValue = "result" in resultEntry ? resultEntry.result : undefined;
                this.onToolResult?.(toolName, resultValue);
                toolCalls.push({
                  toolName,
                  args,
                  result: resultValue,
                });
              }
            }
          }
        },
      });

      for await (const chunk of result.textStream) {
        fullText += chunk;
        yield chunk;
      }

      const finalResult = await result;

      // Extract usage info safely (v5 uses inputTokens/outputTokens)
      const usageData = await finalResult.usage;
      const usage = {
        promptTokens: usageData.inputTokens ?? 0,
        completionTokens: usageData.outputTokens ?? 0,
        totalTokens: usageData.totalTokens ?? 0,
      };

      const finishReason = await finalResult.finishReason;

      return {
        text: fullText,
        toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
        finishReason,
        usage,
        promptMetadata,
      };
    } catch (error) {
      if (error instanceof Error) {
        this.onError?.(error);
        throw error;
      }
      throw new Error("Unknown error during chat stream");
    }
  }

  /**
   * Get available tool names
   */
  getToolNames(): string[] {
    return Object.keys(this.tools);
  }

  /**
   * Update system prompt (disables context-aware prompting)
   */
  setSystemPrompt(prompt: string): void {
    this.systemPrompt = prompt;
    this.enableContextAwarePrompting = false;
  }

  /**
   * Add context to system prompt (legacy method)
   * @deprecated Use updateContext() for context-aware prompting
   */
  addContext(context: string): void {
    this.systemPrompt = `${this.systemPrompt}\n\nCurrent context:\n${context}`;
  }

  // =========================================================================
  // Context-Aware Prompting API
  // =========================================================================

  /**
   * Enable or disable context-aware prompting
   */
  setContextAwarePrompting(enabled: boolean): void {
    this.enableContextAwarePrompting = enabled;
  }

  /**
   * Check if context-aware prompting is enabled
   */
  isContextAwarePromptingEnabled(): boolean {
    return this.enableContextAwarePrompting;
  }

  /**
   * Update context data for prompt building
   */
  updateContext(data: Partial<ContextData>): void {
    this.contextData = { ...this.contextData, ...data };
    this.promptBuilder.setContext(data);
  }

  /**
   * Replace all context data
   */
  setContext(data: ContextData): void {
    this.contextData = data;
    this.promptBuilder = createPromptBuilder(data, this.promptBuilderOptions);
  }

  /**
   * Get current context data
   */
  getContext(): ContextData {
    return { ...this.contextData };
  }

  /**
   * Clear all context data
   */
  clearContext(): void {
    this.contextData = {};
    this.promptBuilder = createPromptBuilder({}, this.promptBuilderOptions);
  }

  /**
   * Update prompt builder options
   */
  setPromptBuilderOptions(options: BuilderOptions): void {
    this.promptBuilderOptions = { ...this.promptBuilderOptions, ...options };
    this.promptBuilder = createPromptBuilder(this.contextData, this.promptBuilderOptions);
  }

  /**
   * Force a specific template for next query
   */
  forceTemplate(template: PromptTemplateKey): void {
    this.promptBuilderOptions = { ...this.promptBuilderOptions, forceTemplate: template };
  }

  /**
   * Force a specific persona for next query
   */
  forcePersona(persona: PersonaId): void {
    this.promptBuilderOptions = { ...this.promptBuilderOptions, forcePersona: persona };
  }

  /**
   * Clear forced template/persona
   */
  clearPromptOverrides(): void {
    delete this.promptBuilderOptions.forceTemplate;
    delete this.promptBuilderOptions.forcePersona;
  }

  /**
   * Detect intent for a query without building full prompt
   */
  detectQueryIntent(query: string): IntentResult {
    return detectIntent(query);
  }

  /**
   * Preview what prompt would be built for a query
   */
  previewPrompt(query: string): BuiltPrompt {
    return this.promptBuilder.build(query, this.promptBuilderOptions);
  }

  /**
   * Get available templates
   */
  static getAvailableTemplates(): PromptTemplateKey[] {
    return PromptBuilder.getTemplates();
  }

  /**
   * Get available personas
   */
  static getAvailablePersonas(): PersonaId[] {
    return PromptBuilder.getPersonas();
  }
}

/**
 * Create a new Stanley agent instance
 */
export function createStanleyAgent(options: AgentOptions): StanleyAgent {
  return new StanleyAgent(options);
}

/**
 * Create a context-aware Stanley agent
 */
export function createContextAwareAgent(
  options: Omit<AgentOptions, "enableContextAwarePrompting"> & {
    initialContext?: ContextData;
    promptBuilderOptions?: BuilderOptions;
  }
): StanleyAgent {
  return new StanleyAgent({
    ...options,
    enableContextAwarePrompting: true,
  });
}

// Re-export types for convenience
export type {
  ContextData,
  BuiltPrompt,
  BuilderOptions,
  PromptTemplateKey,
  PersonaId,
  QueryIntent,
  IntentResult,
};
