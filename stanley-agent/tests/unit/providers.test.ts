/**
 * Provider Configuration Tests
 *
 * Tests for the provider configuration module including:
 * - Schema validation
 * - Environment variable loading
 * - Default configuration creation
 * - Provider type validation
 */

import { describe, it, expect, beforeEach, afterEach } from "bun:test";
import {
  ProviderType,
  ProviderOptionsSchema,
  ModelConfigSchema,
  ProviderConfigSchema,
  AgentConfigSchema,
  PROVIDER_ENV_KEYS,
  DEFAULT_MODELS,
  validateProviderEnv,
  loadProviderFromEnv,
  createDefaultConfig,
} from "../../src/providers/config";

describe("Provider Configuration", () => {
  // Store original env to restore later
  const originalEnv = { ...process.env };

  beforeEach(() => {
    // Clear relevant env vars before each test
    delete process.env.ANTHROPIC_API_KEY;
    delete process.env.OPENAI_API_KEY;
    delete process.env.OPENROUTER_API_KEY;
    delete process.env.GROQ_API_KEY;
    delete process.env.AWS_ACCESS_KEY_ID;
    delete process.env.AWS_SECRET_ACCESS_KEY;
    delete process.env.AWS_REGION;
    delete process.env.GOOGLE_CLOUD_PROJECT;
    delete process.env.GOOGLE_APPLICATION_CREDENTIALS;
    delete process.env.STANLEY_API_URL;
  });

  afterEach(() => {
    // Restore original environment
    Object.keys(process.env).forEach((key) => {
      if (!(key in originalEnv)) {
        delete process.env[key];
      }
    });
    Object.assign(process.env, originalEnv);
  });

  describe("ProviderType Schema", () => {
    it("should accept valid provider types", () => {
      const validTypes = [
        "anthropic",
        "openai",
        "openrouter",
        "google-vertex",
        "amazon-bedrock",
        "groq",
        "ollama",
      ];

      for (const type of validTypes) {
        const result = ProviderType.safeParse(type);
        expect(result.success).toBe(true);
        if (result.success) {
          expect(result.data).toBe(type);
        }
      }
    });

    it("should reject invalid provider types", () => {
      const invalidTypes = ["invalid", "azure", "cohere", ""];

      for (const type of invalidTypes) {
        const result = ProviderType.safeParse(type);
        expect(result.success).toBe(false);
      }
    });
  });

  describe("ProviderOptionsSchema", () => {
    it("should accept valid options", () => {
      const validOptions = {
        apiKey: "test-key-123",
        baseUrl: "https://api.example.com",
        project: "my-project",
        location: "us-central1",
        region: "us-east-1",
      };

      const result = ProviderOptionsSchema.safeParse(validOptions);
      expect(result.success).toBe(true);
    });

    it("should accept empty options", () => {
      const result = ProviderOptionsSchema.safeParse({});
      expect(result.success).toBe(true);
    });

    it("should reject invalid URL format", () => {
      const invalidOptions = {
        baseUrl: "not-a-valid-url",
      };

      const result = ProviderOptionsSchema.safeParse(invalidOptions);
      expect(result.success).toBe(false);
    });
  });

  describe("ModelConfigSchema", () => {
    it("should accept valid model config with defaults", () => {
      const config = {
        id: "gpt-4o",
        provider: "openai",
      };

      const result = ModelConfigSchema.safeParse(config);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.maxTokens).toBe(4096);
        expect(result.data.supportsTools).toBe(true);
        expect(result.data.supportsVision).toBe(false);
      }
    });

    it("should accept full model config", () => {
      const config = {
        id: "claude-sonnet-4-20250514",
        provider: "anthropic",
        displayName: "Claude Sonnet 4",
        maxTokens: 8192,
        contextWindow: 200000,
        supportsTools: true,
        supportsVision: true,
        costPer1kInput: 0.003,
        costPer1kOutput: 0.015,
      };

      const result = ModelConfigSchema.safeParse(config);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.displayName).toBe("Claude Sonnet 4");
        expect(result.data.maxTokens).toBe(8192);
      }
    });

    it("should reject negative maxTokens", () => {
      const config = {
        id: "model",
        provider: "openai",
        maxTokens: -100,
      };

      const result = ModelConfigSchema.safeParse(config);
      expect(result.success).toBe(false);
    });
  });

  describe("AgentConfigSchema", () => {
    it("should accept minimal valid config", () => {
      const config = {
        provider: {
          type: "openrouter",
        },
      };

      const result = AgentConfigSchema.safeParse(config);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.stanleyApiUrl).toBe("http://localhost:8000");
        expect(result.data.maxConversationTurns).toBe(50);
        expect(result.data.toolTimeout).toBe(30000);
        expect(result.data.enableLogging).toBe(true);
      }
    });

    it("should accept full config", () => {
      const config = {
        provider: {
          type: "anthropic",
          options: {
            apiKey: "test-key",
          },
          defaultModel: "claude-sonnet-4-20250514",
        },
        stanleyApiUrl: "https://api.stanley.io",
        maxConversationTurns: 100,
        toolTimeout: 60000,
        enableLogging: false,
      };

      const result = AgentConfigSchema.safeParse(config);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.stanleyApiUrl).toBe("https://api.stanley.io");
        expect(result.data.enableLogging).toBe(false);
      }
    });
  });

  describe("PROVIDER_ENV_KEYS", () => {
    it("should have env keys for all providers", () => {
      const providers = [
        "anthropic",
        "openai",
        "openrouter",
        "google-vertex",
        "amazon-bedrock",
        "groq",
        "ollama",
      ] as const;

      for (const provider of providers) {
        expect(PROVIDER_ENV_KEYS[provider]).toBeDefined();
        expect(Array.isArray(PROVIDER_ENV_KEYS[provider])).toBe(true);
      }
    });

    it("should have correct keys for anthropic", () => {
      expect(PROVIDER_ENV_KEYS.anthropic).toContain("ANTHROPIC_API_KEY");
    });

    it("should have correct keys for aws bedrock", () => {
      expect(PROVIDER_ENV_KEYS["amazon-bedrock"]).toContain("AWS_ACCESS_KEY_ID");
      expect(PROVIDER_ENV_KEYS["amazon-bedrock"]).toContain("AWS_SECRET_ACCESS_KEY");
    });

    it("should have no required keys for ollama", () => {
      expect(PROVIDER_ENV_KEYS.ollama).toHaveLength(0);
    });
  });

  describe("DEFAULT_MODELS", () => {
    it("should have default model for all providers", () => {
      const providers = [
        "anthropic",
        "openai",
        "openrouter",
        "google-vertex",
        "amazon-bedrock",
        "groq",
        "ollama",
      ] as const;

      for (const provider of providers) {
        expect(DEFAULT_MODELS[provider]).toBeDefined();
        expect(typeof DEFAULT_MODELS[provider]).toBe("string");
        expect(DEFAULT_MODELS[provider].length).toBeGreaterThan(0);
      }
    });
  });

  describe("validateProviderEnv", () => {
    it("should return valid=false when env vars are missing", () => {
      const result = validateProviderEnv("anthropic");
      expect(result.valid).toBe(false);
      expect(result.missing).toContain("ANTHROPIC_API_KEY");
    });

    it("should return valid=true when env vars are set", () => {
      process.env.ANTHROPIC_API_KEY = "test-key";
      const result = validateProviderEnv("anthropic");
      expect(result.valid).toBe(true);
      expect(result.missing).toHaveLength(0);
    });

    it("should always return valid=true for ollama", () => {
      const result = validateProviderEnv("ollama");
      expect(result.valid).toBe(true);
      expect(result.missing).toHaveLength(0);
    });

    it("should handle partial env var setup", () => {
      process.env.AWS_ACCESS_KEY_ID = "test-key";
      // Missing AWS_SECRET_ACCESS_KEY and AWS_REGION
      const result = validateProviderEnv("amazon-bedrock");
      expect(result.valid).toBe(false);
      expect(result.missing).toContain("AWS_SECRET_ACCESS_KEY");
      expect(result.missing).toContain("AWS_REGION");
    });
  });

  describe("loadProviderFromEnv", () => {
    it("should load anthropic config from env", () => {
      process.env.ANTHROPIC_API_KEY = "sk-ant-test-123";
      const options = loadProviderFromEnv("anthropic");
      expect(options.apiKey).toBe("sk-ant-test-123");
    });

    it("should load openai config from env", () => {
      process.env.OPENAI_API_KEY = "sk-test-openai";
      const options = loadProviderFromEnv("openai");
      expect(options.apiKey).toBe("sk-test-openai");
    });

    it("should load openrouter config with site info", () => {
      process.env.OPENROUTER_API_KEY = "sk-or-test";
      process.env.OPENROUTER_SITE_URL = "https://myapp.com";
      process.env.OPENROUTER_SITE_NAME = "MyApp";
      const options = loadProviderFromEnv("openrouter");
      expect(options.apiKey).toBe("sk-or-test");
      expect(options.siteUrl).toBe("https://myapp.com");
      expect(options.siteName).toBe("MyApp");
    });

    it("should use default site name for openrouter", () => {
      process.env.OPENROUTER_API_KEY = "sk-or-test";
      const options = loadProviderFromEnv("openrouter");
      expect(options.siteName).toBe("Stanley");
    });

    it("should load google vertex config from env", () => {
      process.env.GOOGLE_CLOUD_PROJECT = "my-gcp-project";
      process.env.GOOGLE_CLOUD_LOCATION = "us-east4";
      const options = loadProviderFromEnv("google-vertex");
      expect(options.project).toBe("my-gcp-project");
      expect(options.location).toBe("us-east4");
    });

    it("should use default location for google vertex", () => {
      process.env.GOOGLE_CLOUD_PROJECT = "my-gcp-project";
      const options = loadProviderFromEnv("google-vertex");
      expect(options.location).toBe("us-central1");
    });

    it("should load aws bedrock config from env", () => {
      process.env.AWS_ACCESS_KEY_ID = "AKIATEST";
      process.env.AWS_SECRET_ACCESS_KEY = "secret123";
      process.env.AWS_REGION = "us-west-2";
      const options = loadProviderFromEnv("amazon-bedrock");
      expect(options.accessKeyId).toBe("AKIATEST");
      expect(options.secretAccessKey).toBe("secret123");
      expect(options.region).toBe("us-west-2");
    });

    it("should use default region for aws bedrock", () => {
      const options = loadProviderFromEnv("amazon-bedrock");
      expect(options.region).toBe("us-east-1");
    });

    it("should load groq config from env", () => {
      process.env.GROQ_API_KEY = "gsk-test";
      const options = loadProviderFromEnv("groq");
      expect(options.apiKey).toBe("gsk-test");
    });

    it("should load ollama config with default host", () => {
      const options = loadProviderFromEnv("ollama");
      expect(options.host).toBe("http://localhost:11434");
    });

    it("should load ollama config with custom host", () => {
      process.env.OLLAMA_HOST = "http://192.168.1.100:11434";
      const options = loadProviderFromEnv("ollama");
      expect(options.host).toBe("http://192.168.1.100:11434");
    });
  });

  describe("createDefaultConfig", () => {
    it("should create default config with openrouter", () => {
      const config = createDefaultConfig();
      expect(config.provider.type).toBe("openrouter");
      expect(config.stanleyApiUrl).toBe("http://localhost:8000");
      expect(config.maxConversationTurns).toBe(50);
      expect(config.toolTimeout).toBe(30000);
    });

    it("should create config for specified provider", () => {
      const config = createDefaultConfig("anthropic");
      expect(config.provider.type).toBe("anthropic");
      expect(config.provider.defaultModel).toBe(DEFAULT_MODELS.anthropic);
    });

    it("should use STANLEY_API_URL from env", () => {
      process.env.STANLEY_API_URL = "https://api.stanley.io";
      const config = createDefaultConfig();
      expect(config.stanleyApiUrl).toBe("https://api.stanley.io");
    });

    it("should disable logging in test mode", () => {
      process.env.NODE_ENV = "test";
      const config = createDefaultConfig();
      expect(config.enableLogging).toBe(false);
    });

    it("should load api key from env", () => {
      process.env.OPENROUTER_API_KEY = "sk-or-key";
      const config = createDefaultConfig("openrouter");
      expect(config.provider.options?.apiKey).toBe("sk-or-key");
    });
  });
});
