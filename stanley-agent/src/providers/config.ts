/**
 * Provider Configuration Module
 *
 * Supports multiple LLM providers via Vercel AI SDK:
 * - Anthropic (Claude)
 * - OpenAI (GPT-4, GPT-4o)
 * - OpenRouter (any model)
 * - Google Vertex AI
 * - AWS Bedrock
 * - Groq
 */

import { z } from "zod";

/**
 * Supported LLM provider types
 */
export const ProviderType = z.enum([
  "anthropic",
  "openai",
  "openrouter",
  "google-vertex",
  "amazon-bedrock",
  "groq",
  "ollama",
]);

export type ProviderType = z.infer<typeof ProviderType>;

/**
 * Provider-specific configuration options
 */
export const ProviderOptionsSchema = z.object({
  // Common options
  apiKey: z.string().optional(),
  baseUrl: z.string().url().optional(),

  // Google Vertex specific
  project: z.string().optional(),
  location: z.string().optional(),

  // AWS Bedrock specific
  region: z.string().optional(),
  accessKeyId: z.string().optional(),
  secretAccessKey: z.string().optional(),

  // OpenRouter specific
  siteUrl: z.string().optional(),
  siteName: z.string().optional(),

  // Ollama specific
  host: z.string().optional(),
});

export type ProviderOptions = z.infer<typeof ProviderOptionsSchema>;

/**
 * Model configuration
 */
export const ModelConfigSchema = z.object({
  id: z.string(),
  provider: ProviderType,
  displayName: z.string().optional(),
  maxTokens: z.number().int().positive().default(4096),
  contextWindow: z.number().int().positive().optional(),
  supportsTools: z.boolean().default(true),
  supportsVision: z.boolean().default(false),
  costPer1kInput: z.number().optional(),
  costPer1kOutput: z.number().optional(),
});

export type ModelConfig = z.infer<typeof ModelConfigSchema>;

/**
 * Full provider configuration
 */
export const ProviderConfigSchema = z.object({
  type: ProviderType,
  options: ProviderOptionsSchema.optional(),
  models: z.array(ModelConfigSchema).optional(),
  defaultModel: z.string().optional(),
});

export type ProviderConfig = z.infer<typeof ProviderConfigSchema>;

/**
 * Stanley agent configuration
 */
export const AgentConfigSchema = z.object({
  provider: ProviderConfigSchema,
  stanleyApiUrl: z.string().url().default("http://localhost:8000"),
  maxConversationTurns: z.number().int().positive().default(50),
  toolTimeout: z.number().int().positive().default(30000),
  enableLogging: z.boolean().default(true),
});

export type AgentConfig = z.infer<typeof AgentConfigSchema>;

/**
 * Environment variable mappings for each provider
 */
export const PROVIDER_ENV_KEYS: Record<ProviderType, string[]> = {
  anthropic: ["ANTHROPIC_API_KEY"],
  openai: ["OPENAI_API_KEY"],
  openrouter: ["OPENROUTER_API_KEY"],
  "google-vertex": ["GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT"],
  "amazon-bedrock": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
  groq: ["GROQ_API_KEY"],
  ollama: [],
};

/**
 * Default model IDs for each provider
 */
export const DEFAULT_MODELS: Record<ProviderType, string> = {
  anthropic: "claude-sonnet-4-20250514",
  openai: "gpt-4o",
  openrouter: "anthropic/claude-sonnet-4",
  "google-vertex": "gemini-2.0-flash",
  "amazon-bedrock": "anthropic.claude-3-5-sonnet-20241022-v2:0",
  groq: "llama-3.3-70b-versatile",
  ollama: "llama3.2",
};

/**
 * Check if required environment variables are set for a provider
 */
export function validateProviderEnv(provider: ProviderType): { valid: boolean; missing: string[] } {
  const requiredKeys = PROVIDER_ENV_KEYS[provider];
  const missing = requiredKeys.filter((key) => !process.env[key]);
  return { valid: missing.length === 0, missing };
}

/**
 * Load provider configuration from environment variables
 */
export function loadProviderFromEnv(provider: ProviderType): ProviderOptions {
  const options: ProviderOptions = {};

  switch (provider) {
    case "anthropic":
      options.apiKey = process.env.ANTHROPIC_API_KEY;
      break;
    case "openai":
      options.apiKey = process.env.OPENAI_API_KEY;
      break;
    case "openrouter":
      options.apiKey = process.env.OPENROUTER_API_KEY;
      options.siteUrl = process.env.OPENROUTER_SITE_URL;
      options.siteName = process.env.OPENROUTER_SITE_NAME || "Stanley";
      break;
    case "google-vertex":
      options.project = process.env.GOOGLE_CLOUD_PROJECT;
      options.location = process.env.GOOGLE_CLOUD_LOCATION || "us-central1";
      break;
    case "amazon-bedrock":
      options.accessKeyId = process.env.AWS_ACCESS_KEY_ID;
      options.secretAccessKey = process.env.AWS_SECRET_ACCESS_KEY;
      options.region = process.env.AWS_REGION || "us-east-1";
      break;
    case "groq":
      options.apiKey = process.env.GROQ_API_KEY;
      break;
    case "ollama":
      options.host = process.env.OLLAMA_HOST || "http://localhost:11434";
      break;
  }

  return options;
}

/**
 * Create default agent configuration
 */
export function createDefaultConfig(provider: ProviderType = "openrouter"): AgentConfig {
  const validation = validateProviderEnv(provider);
  if (!validation.valid && provider !== "ollama") {
    console.warn(`Warning: Missing environment variables for ${provider}: ${validation.missing.join(", ")}`);
  }

  return {
    provider: {
      type: provider,
      options: loadProviderFromEnv(provider),
      defaultModel: DEFAULT_MODELS[provider],
    },
    stanleyApiUrl: process.env.STANLEY_API_URL || "http://localhost:8000",
    maxConversationTurns: 50,
    toolTimeout: 30000,
    enableLogging: process.env.NODE_ENV !== "test",
  };
}

/**
 * Load configuration from YAML file
 */
export async function loadConfigFromFile(path: string): Promise<AgentConfig> {
  const fs = await import("fs");
  const content = fs.readFileSync(path, "utf-8");

  // Simple YAML parsing for key-value pairs
  const lines = content.split("\n");
  const config: Record<string, unknown> = {};

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;

    const colonIndex = trimmed.indexOf(":");
    if (colonIndex === -1) continue;

    const key = trimmed.slice(0, colonIndex).trim();
    const value = trimmed.slice(colonIndex + 1).trim();

    // Remove quotes if present
    const cleanValue = value.replace(/^["']|["']$/g, "");
    config[key] = cleanValue;
  }

  // Map to AgentConfig structure
  const providerType = ProviderType.parse(config.provider || "openrouter");
  return {
    provider: {
      type: providerType,
      options: loadProviderFromEnv(providerType),
      defaultModel: (config.model as string) || DEFAULT_MODELS[providerType],
    },
    stanleyApiUrl: (config.stanley_api_url as string) || "http://localhost:8000",
    maxConversationTurns: parseInt(config.max_turns as string) || 50,
    toolTimeout: parseInt(config.tool_timeout as string) || 30000,
    enableLogging: config.logging !== "false",
  };
}
