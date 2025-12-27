/**
 * Provider Factory
 *
 * Creates AI SDK language model instances for different providers
 */

import type { LanguageModel } from "ai";
import { ProviderConfig, ProviderType, DEFAULT_MODELS } from "./config";

/**
 * Interface for provider creation result
 */
export interface ProviderInstance {
  model: LanguageModel;
  modelId: string;
  provider: ProviderType;
}

/**
 * Create a language model instance for the given provider configuration
 */
export async function createProvider(config: ProviderConfig): Promise<ProviderInstance> {
  const modelId = config.defaultModel || DEFAULT_MODELS[config.type];

  switch (config.type) {
    case "anthropic": {
      const { createAnthropic } = await import("@ai-sdk/anthropic");
      const anthropic = createAnthropic({
        apiKey: config.options?.apiKey,
        baseURL: config.options?.baseUrl,
      });
      return {
        model: anthropic(modelId),
        modelId,
        provider: "anthropic",
      };
    }

    case "openai": {
      const { createOpenAI } = await import("@ai-sdk/openai");
      const openai = createOpenAI({
        apiKey: config.options?.apiKey,
        baseURL: config.options?.baseUrl,
      });
      return {
        model: openai(modelId),
        modelId,
        provider: "openai",
      };
    }

    case "openrouter": {
      const { createOpenRouter } = await import("@openrouter/ai-sdk-provider");
      const openrouter = createOpenRouter({
        apiKey: config.options?.apiKey,
        headers: {
          "HTTP-Referer": config.options?.siteUrl || "https://stanley.ai/",
          "X-Title": config.options?.siteName || "Stanley Agent",
        },
      });
      return {
        model: openrouter(modelId),
        modelId,
        provider: "openrouter",
      };
    }

    case "google-vertex": {
      const { createVertex } = await import("@ai-sdk/google-vertex");
      const vertex = createVertex({
        project: config.options?.project,
        location: config.options?.location,
      });
      return {
        model: vertex(modelId),
        modelId,
        provider: "google-vertex",
      };
    }

    case "amazon-bedrock": {
      const { createAmazonBedrock } = await import("@ai-sdk/amazon-bedrock");
      const bedrock = createAmazonBedrock({
        region: config.options?.region,
        accessKeyId: config.options?.accessKeyId,
        secretAccessKey: config.options?.secretAccessKey,
      });
      return {
        model: bedrock(modelId),
        modelId,
        provider: "amazon-bedrock",
      };
    }

    case "groq": {
      // Groq uses OpenAI-compatible API
      const { createOpenAI } = await import("@ai-sdk/openai");
      const groq = createOpenAI({
        apiKey: config.options?.apiKey,
        baseURL: "https://api.groq.com/openai/v1",
      });
      return {
        model: groq(modelId),
        modelId,
        provider: "groq",
      };
    }

    case "ollama": {
      // Ollama uses OpenAI-compatible API
      const { createOpenAI } = await import("@ai-sdk/openai");
      const ollama = createOpenAI({
        baseURL: config.options?.host || "http://localhost:11434/v1",
        apiKey: "ollama", // Dummy key for local Ollama
      });
      return {
        model: ollama(modelId),
        modelId,
        provider: "ollama",
      };
    }

    default:
      throw new Error(`Unsupported provider: ${config.type}`);
  }
}

/**
 * Create multiple provider instances for fallback or model switching
 */
export async function createProviderPool(configs: ProviderConfig[]): Promise<Map<string, ProviderInstance>> {
  const pool = new Map<string, ProviderInstance>();

  for (const config of configs) {
    try {
      const instance = await createProvider(config);
      pool.set(`${config.type}:${instance.modelId}`, instance);
    } catch (error) {
      console.warn(`Failed to create provider ${config.type}:`, error);
    }
  }

  return pool;
}
