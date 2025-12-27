/**
 * Stanley Agent Definition
 *
 * The primary AI agent for institutional investment research.
 */

import { generateText, streamText, type CoreMessage, type LanguageModel, type ToolSet } from "ai";

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
}

/**
 * Default system prompt for Stanley agent
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
}

/**
 * Stanley Agent class
 */
export class StanleyAgent {
  private model: LanguageModel;
  private tools: ToolSet;
  private systemPrompt: string;
  private maxOutputTokens: number;
  private onToolCall?: (toolName: string, args: unknown) => void;
  private onToolResult?: (toolName: string, result: unknown) => void;
  private onError?: (error: Error) => void;

  constructor(options: AgentOptions) {
    this.model = options.model;
    this.tools = options.tools;
    this.systemPrompt = options.systemPrompt || DEFAULT_SYSTEM_PROMPT;
    this.maxOutputTokens = options.maxOutputTokens || 4096;
    this.onToolCall = options.onToolCall;
    this.onToolResult = options.onToolResult;
    this.onError = options.onError;
  }

  /**
   * Generate a response for a single message
   */
  async chat(messages: Message[]): Promise<AgentResponse> {
    const toolCalls: AgentResponse["toolCalls"] = [];

    try {
      const result = await generateText({
        model: this.model,
        system: this.systemPrompt,
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

    try {
      const result = streamText({
        model: this.model,
        system: this.systemPrompt,
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
   * Update system prompt
   */
  setSystemPrompt(prompt: string): void {
    this.systemPrompt = prompt;
  }

  /**
   * Add context to system prompt
   */
  addContext(context: string): void {
    this.systemPrompt = `${this.systemPrompt}\n\nCurrent context:\n${context}`;
  }
}

/**
 * Create a new Stanley agent instance
 */
export function createStanleyAgent(options: AgentOptions): StanleyAgent {
  return new StanleyAgent(options);
}
