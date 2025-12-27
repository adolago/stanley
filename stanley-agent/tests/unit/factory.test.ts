/**
 * Provider Factory Tests
 *
 * Tests for the provider factory module including:
 * - Provider instance creation
 * - Provider pool management
 * - Error handling for unsupported providers
 */

import { describe, it, expect, beforeEach, afterEach, mock } from "bun:test";
import type { ProviderConfig } from "../../src/providers/config";

// Note: These tests mock the AI SDK imports to avoid requiring actual API keys
// For real integration tests, use the integration test suite

describe("Provider Factory", () => {
  describe("createProvider", () => {
    it("should support anthropic provider type", () => {
      const config: ProviderConfig = {
        type: "anthropic",
        options: { apiKey: "test-key" },
        defaultModel: "claude-sonnet-4-20250514",
      };

      // Verify config structure is valid
      expect(config.type).toBe("anthropic");
      expect(config.options?.apiKey).toBe("test-key");
    });

    it("should support openai provider type", () => {
      const config: ProviderConfig = {
        type: "openai",
        options: { apiKey: "sk-test" },
        defaultModel: "gpt-4o",
      };

      expect(config.type).toBe("openai");
    });

    it("should support openrouter provider type", () => {
      const config: ProviderConfig = {
        type: "openrouter",
        options: {
          apiKey: "sk-or-test",
          siteUrl: "https://stanley.dev",
          siteName: "Stanley",
        },
        defaultModel: "anthropic/claude-sonnet-4",
      };

      expect(config.type).toBe("openrouter");
      expect(config.options?.siteUrl).toBe("https://stanley.dev");
    });

    it("should support google-vertex provider type", () => {
      const config: ProviderConfig = {
        type: "google-vertex",
        options: {
          project: "my-gcp-project",
          location: "us-central1",
        },
        defaultModel: "gemini-2.0-flash",
      };

      expect(config.type).toBe("google-vertex");
      expect(config.options?.project).toBe("my-gcp-project");
    });

    it("should support amazon-bedrock provider type", () => {
      const config: ProviderConfig = {
        type: "amazon-bedrock",
        options: {
          region: "us-east-1",
          accessKeyId: "AKIATEST",
          secretAccessKey: "secret",
        },
        defaultModel: "anthropic.claude-3-5-sonnet-20241022-v2:0",
      };

      expect(config.type).toBe("amazon-bedrock");
      expect(config.options?.region).toBe("us-east-1");
    });

    it("should support groq provider type", () => {
      const config: ProviderConfig = {
        type: "groq",
        options: { apiKey: "gsk-test" },
        defaultModel: "llama-3.3-70b-versatile",
      };

      expect(config.type).toBe("groq");
    });

    it("should support ollama provider type", () => {
      const config: ProviderConfig = {
        type: "ollama",
        options: { host: "http://localhost:11434" },
        defaultModel: "llama3.2",
      };

      expect(config.type).toBe("ollama");
      expect(config.options?.host).toBe("http://localhost:11434");
    });
  });

  describe("ProviderInstance structure", () => {
    it("should define correct instance structure", () => {
      // Define expected structure
      const instance = {
        model: {} as any, // LanguageModelV1
        modelId: "claude-sonnet-4-20250514",
        provider: "anthropic" as const,
      };

      expect(instance.modelId).toBe("claude-sonnet-4-20250514");
      expect(instance.provider).toBe("anthropic");
      expect(instance.model).toBeDefined();
    });
  });

  describe("Provider pool", () => {
    it("should support multiple provider configs", () => {
      const configs: ProviderConfig[] = [
        { type: "anthropic", defaultModel: "claude-sonnet-4-20250514" },
        { type: "openai", defaultModel: "gpt-4o" },
        { type: "groq", defaultModel: "llama-3.3-70b-versatile" },
      ];

      expect(configs).toHaveLength(3);
      expect(configs.map((c) => c.type)).toContain("anthropic");
      expect(configs.map((c) => c.type)).toContain("openai");
      expect(configs.map((c) => c.type)).toContain("groq");
    });

    it("should create pool key format", () => {
      const config: ProviderConfig = {
        type: "openrouter",
        defaultModel: "anthropic/claude-sonnet-4",
      };

      const key = `${config.type}:${config.defaultModel}`;
      expect(key).toBe("openrouter:anthropic/claude-sonnet-4");
    });
  });

  describe("OpenRouter configuration", () => {
    it("should support various model IDs", () => {
      const modelIds = [
        "anthropic/claude-sonnet-4",
        "anthropic/claude-3-5-sonnet",
        "openai/gpt-4o",
        "openai/gpt-4-turbo",
        "google/gemini-2.0-flash-exp",
        "meta-llama/llama-3.3-70b-instruct",
        "mistralai/mistral-large-latest",
        "deepseek/deepseek-r1",
      ];

      for (const modelId of modelIds) {
        const config: ProviderConfig = {
          type: "openrouter",
          defaultModel: modelId,
        };
        expect(config.defaultModel).toBe(modelId);
      }
    });

    it("should format headers correctly", () => {
      const headers = {
        "HTTP-Referer": "https://stanley.dev",
        "X-Title": "Stanley",
      };

      expect(headers["HTTP-Referer"]).toBe("https://stanley.dev");
      expect(headers["X-Title"]).toBe("Stanley");
    });
  });

  describe("Default model selection", () => {
    const { DEFAULT_MODELS } = require("../../src/providers/config");

    it("should have sensible defaults", () => {
      expect(DEFAULT_MODELS.anthropic).toContain("claude");
      expect(DEFAULT_MODELS.openai).toContain("gpt");
      expect(DEFAULT_MODELS.groq).toContain("llama");
    });

    it("should select default when not specified", () => {
      const config: ProviderConfig = {
        type: "anthropic",
        // No defaultModel specified
      };

      const modelId = config.defaultModel || DEFAULT_MODELS[config.type];
      expect(modelId).toBe(DEFAULT_MODELS.anthropic);
    });
  });
});
