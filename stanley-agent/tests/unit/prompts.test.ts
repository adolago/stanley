/**
 * Prompt Building Tests
 *
 * Tests for the prompt template system that builds dynamic prompts
 * with context injection, variable substitution, and formatting.
 */

import { describe, it, expect, beforeEach } from "bun:test";

// Mock PromptTemplate class
interface PromptVariable {
  name: string;
  required?: boolean;
  default?: string;
  description?: string;
}

interface PromptSection {
  name: string;
  content: string;
  enabled?: boolean;
  condition?: (context: Record<string, unknown>) => boolean;
}

class MockPromptTemplate {
  private template: string;
  private variables: Map<string, PromptVariable> = new Map();
  private sections: Map<string, PromptSection> = new Map();

  constructor(template: string) {
    this.template = template;
    this.parseVariables();
  }

  private parseVariables(): void {
    // Find all {{variable}} patterns
    const regex = /\{\{(\w+)\}\}/g;
    let match;

    while ((match = regex.exec(this.template)) !== null) {
      const name = match[1];
      if (!this.variables.has(name)) {
        this.variables.set(name, { name });
      }
    }
  }

  setVariable(name: string, options: Partial<PromptVariable>): void {
    const existing = this.variables.get(name) ?? { name };
    this.variables.set(name, { ...existing, ...options });
  }

  addSection(section: PromptSection): void {
    this.sections.set(section.name, section);
  }

  removeSection(name: string): boolean {
    return this.sections.delete(name);
  }

  render(context: Record<string, unknown> = {}): string {
    let result = this.template;

    // Substitute variables
    for (const [name, variable] of this.variables) {
      const value = context[name] ?? variable.default;
      const placeholder = `{{${name}}}`;

      if (value === undefined && variable.required) {
        throw new Error(`Required variable '${name}' is missing`);
      }

      result = result.replace(new RegExp(placeholder, "g"), String(value ?? ""));
    }

    // Process sections
    for (const [name, section] of this.sections) {
      const sectionPlaceholder = `{{#section:${name}}}`;

      if (result.includes(sectionPlaceholder)) {
        let sectionContent = "";

        const enabled = section.enabled !== false &&
          (!section.condition || section.condition(context));

        if (enabled) {
          sectionContent = section.content;
        }

        result = result.replace(sectionPlaceholder, sectionContent);
      }
    }

    // Clean up empty lines and trim
    result = result.replace(/\n{3,}/g, "\n\n").trim();

    return result;
  }

  getVariables(): string[] {
    return Array.from(this.variables.keys());
  }

  getRequiredVariables(): string[] {
    return Array.from(this.variables.entries())
      .filter(([_, v]) => v.required)
      .map(([name, _]) => name);
  }

  validate(context: Record<string, unknown>): { valid: boolean; missing: string[] } {
    const missing: string[] = [];

    for (const [name, variable] of this.variables) {
      if (variable.required && context[name] === undefined && variable.default === undefined) {
        missing.push(name);
      }
    }

    return { valid: missing.length === 0, missing };
  }
}

// Mock PromptBuilder for complex prompts
class MockPromptBuilder {
  private systemPrompt: string = "";
  private contextSections: string[] = [];
  private toolDescriptions: string[] = [];
  private examples: string[] = [];
  private instructions: string[] = [];

  setSystemPrompt(prompt: string): this {
    this.systemPrompt = prompt;
    return this;
  }

  addContext(context: string): this {
    this.contextSections.push(context);
    return this;
  }

  addToolDescription(name: string, description: string): this {
    this.toolDescriptions.push(`- ${name}: ${description}`);
    return this;
  }

  addExample(input: string, output: string): this {
    this.examples.push(`User: ${input}\nAssistant: ${output}`);
    return this;
  }

  addInstruction(instruction: string): this {
    this.instructions.push(`- ${instruction}`);
    return this;
  }

  build(): string {
    const parts: string[] = [];

    if (this.systemPrompt) {
      parts.push(this.systemPrompt);
    }

    if (this.contextSections.length > 0) {
      parts.push("\n## Context\n");
      parts.push(this.contextSections.join("\n\n"));
    }

    if (this.toolDescriptions.length > 0) {
      parts.push("\n## Available Tools\n");
      parts.push(this.toolDescriptions.join("\n"));
    }

    if (this.instructions.length > 0) {
      parts.push("\n## Instructions\n");
      parts.push(this.instructions.join("\n"));
    }

    if (this.examples.length > 0) {
      parts.push("\n## Examples\n");
      parts.push(this.examples.join("\n\n"));
    }

    return parts.join("\n").trim();
  }

  estimateTokens(): number {
    const text = this.build();
    return Math.ceil(text.length / 4);
  }
}

describe("Prompt Template", () => {
  describe("Basic Rendering", () => {
    it("should render template without variables", () => {
      const template = new MockPromptTemplate("You are an investment analyst.");
      const result = template.render();

      expect(result).toBe("You are an investment analyst.");
    });

    it("should substitute single variable", () => {
      const template = new MockPromptTemplate("Hello, {{name}}!");
      const result = template.render({ name: "Stanley" });

      expect(result).toBe("Hello, Stanley!");
    });

    it("should substitute multiple variables", () => {
      const template = new MockPromptTemplate(
        "Analyze {{symbol}} for {{timeframe}} outlook."
      );
      const result = template.render({ symbol: "AAPL", timeframe: "short-term" });

      expect(result).toBe("Analyze AAPL for short-term outlook.");
    });

    it("should handle repeated variables", () => {
      const template = new MockPromptTemplate(
        "Compare {{symbol}} with {{symbol}}'s historical performance."
      );
      const result = template.render({ symbol: "GOOGL" });

      expect(result).toBe("Compare GOOGL with GOOGL's historical performance.");
    });

    it("should use default value when variable not provided", () => {
      const template = new MockPromptTemplate("Analysis for {{symbol}}.");
      template.setVariable("symbol", { default: "SPY" });

      const result = template.render({});

      expect(result).toBe("Analysis for SPY.");
    });

    it("should override default with provided value", () => {
      const template = new MockPromptTemplate("Analysis for {{symbol}}.");
      template.setVariable("symbol", { default: "SPY" });

      const result = template.render({ symbol: "QQQ" });

      expect(result).toBe("Analysis for QQQ.");
    });
  });

  describe("Required Variables", () => {
    it("should throw when required variable is missing", () => {
      const template = new MockPromptTemplate("Analysis for {{symbol}}.");
      template.setVariable("symbol", { required: true });

      expect(() => template.render({})).toThrow("Required variable 'symbol' is missing");
    });

    it("should not throw when required variable is provided", () => {
      const template = new MockPromptTemplate("Analysis for {{symbol}}.");
      template.setVariable("symbol", { required: true });

      expect(() => template.render({ symbol: "AAPL" })).not.toThrow();
    });

    it("should use default for required variable if provided", () => {
      const template = new MockPromptTemplate("Analysis for {{symbol}}.");
      template.setVariable("symbol", { required: true, default: "SPY" });

      const result = template.render({});
      expect(result).toBe("Analysis for SPY.");
    });
  });

  describe("Sections", () => {
    it("should include enabled section", () => {
      const template = new MockPromptTemplate(
        "Base prompt.\n{{#section:extra}}"
      );
      template.addSection({
        name: "extra",
        content: "Extra context here.",
        enabled: true,
      });

      const result = template.render();

      expect(result).toContain("Extra context here.");
    });

    it("should exclude disabled section", () => {
      const template = new MockPromptTemplate(
        "Base prompt.\n{{#section:extra}}"
      );
      template.addSection({
        name: "extra",
        content: "Extra context here.",
        enabled: false,
      });

      const result = template.render();

      expect(result).not.toContain("Extra context here.");
    });

    it("should conditionally include section based on context", () => {
      const template = new MockPromptTemplate(
        "Prompt.\n{{#section:advanced}}"
      );
      template.addSection({
        name: "advanced",
        content: "Advanced options available.",
        condition: (ctx) => ctx.isAdvanced === true,
      });

      const basicResult = template.render({ isAdvanced: false });
      const advancedResult = template.render({ isAdvanced: true });

      expect(basicResult).not.toContain("Advanced options");
      expect(advancedResult).toContain("Advanced options");
    });

    it("should remove section", () => {
      const template = new MockPromptTemplate("{{#section:removable}}");
      template.addSection({
        name: "removable",
        content: "This will be removed.",
      });

      const removed = template.removeSection("removable");

      expect(removed).toBe(true);
    });
  });

  describe("Variable Discovery", () => {
    it("should list all variables in template", () => {
      const template = new MockPromptTemplate(
        "Analyze {{symbol}} for {{user}} with {{timeframe}}."
      );

      const variables = template.getVariables();

      expect(variables).toContain("symbol");
      expect(variables).toContain("user");
      expect(variables).toContain("timeframe");
      expect(variables).toHaveLength(3);
    });

    it("should list only required variables", () => {
      const template = new MockPromptTemplate("{{a}} {{b}} {{c}}");
      template.setVariable("a", { required: true });
      template.setVariable("c", { required: true });

      const required = template.getRequiredVariables();

      expect(required).toContain("a");
      expect(required).toContain("c");
      expect(required).not.toContain("b");
    });
  });

  describe("Validation", () => {
    it("should validate context has required variables", () => {
      const template = new MockPromptTemplate("{{required1}} {{required2}} {{optional}}");
      template.setVariable("required1", { required: true });
      template.setVariable("required2", { required: true });

      const result = template.validate({ required1: "value" });

      expect(result.valid).toBe(false);
      expect(result.missing).toContain("required2");
    });

    it("should pass validation when all required variables present", () => {
      const template = new MockPromptTemplate("{{a}} {{b}}");
      template.setVariable("a", { required: true });
      template.setVariable("b", { required: true });

      const result = template.validate({ a: "1", b: "2" });

      expect(result.valid).toBe(true);
      expect(result.missing).toHaveLength(0);
    });
  });
});

describe("Prompt Builder", () => {
  let builder: MockPromptBuilder;

  beforeEach(() => {
    builder = new MockPromptBuilder();
  });

  describe("System Prompt", () => {
    it("should set system prompt", () => {
      const result = builder
        .setSystemPrompt("You are Stanley, an investment analyst.")
        .build();

      expect(result).toBe("You are Stanley, an investment analyst.");
    });
  });

  describe("Context", () => {
    it("should add context sections", () => {
      const result = builder
        .setSystemPrompt("Base prompt.")
        .addContext("Current portfolio: AAPL, GOOGL, MSFT")
        .addContext("Market is bullish.")
        .build();

      expect(result).toContain("## Context");
      expect(result).toContain("Current portfolio");
      expect(result).toContain("Market is bullish");
    });
  });

  describe("Tool Descriptions", () => {
    it("should add tool descriptions", () => {
      const result = builder
        .setSystemPrompt("You are an assistant.")
        .addToolDescription("get_market_data", "Get current market data for a symbol")
        .addToolDescription("get_research", "Get research report for a symbol")
        .build();

      expect(result).toContain("## Available Tools");
      expect(result).toContain("- get_market_data: Get current market data");
      expect(result).toContain("- get_research: Get research report");
    });
  });

  describe("Instructions", () => {
    it("should add instructions", () => {
      const result = builder
        .setSystemPrompt("Base.")
        .addInstruction("Always verify data before responding")
        .addInstruction("Use tools when needed")
        .build();

      expect(result).toContain("## Instructions");
      expect(result).toContain("- Always verify data");
      expect(result).toContain("- Use tools when needed");
    });
  });

  describe("Examples", () => {
    it("should add examples", () => {
      const result = builder
        .setSystemPrompt("Base.")
        .addExample("What is AAPL trading at?", "AAPL is currently trading at $150.25.")
        .build();

      expect(result).toContain("## Examples");
      expect(result).toContain("User: What is AAPL trading at?");
      expect(result).toContain("Assistant: AAPL is currently trading at $150.25.");
    });
  });

  describe("Complete Prompt", () => {
    it("should build complete prompt with all sections", () => {
      const result = builder
        .setSystemPrompt("You are Stanley, an AI investment analyst.")
        .addContext("User is analyzing tech stocks.")
        .addToolDescription("get_market_data", "Fetch market data")
        .addInstruction("Be concise and accurate")
        .addExample("Price of AAPL?", "AAPL is at $150.")
        .build();

      expect(result).toContain("You are Stanley");
      expect(result).toContain("## Context");
      expect(result).toContain("## Available Tools");
      expect(result).toContain("## Instructions");
      expect(result).toContain("## Examples");
    });
  });

  describe("Token Estimation", () => {
    it("should estimate token count", () => {
      builder
        .setSystemPrompt("You are an AI assistant.")
        .addContext("Some context here.");

      const tokens = builder.estimateTokens();

      expect(tokens).toBeGreaterThan(0);
      expect(typeof tokens).toBe("number");
    });
  });
});

describe("Stanley Prompts", () => {
  describe("Investment Analysis Prompt", () => {
    it("should create investment analysis prompt", () => {
      const template = new MockPromptTemplate(`
You are Stanley, an institutional investment analysis assistant.

## Current Analysis
Symbol: {{symbol}}
Timeframe: {{timeframe}}

{{#section:portfolio}}

## Instructions
- Provide data-driven insights
- Consider both fundamental and technical factors
- Be objective and balanced
      `.trim());

      template.setVariable("symbol", { required: true });
      template.setVariable("timeframe", { default: "medium-term" });
      template.addSection({
        name: "portfolio",
        content: "Portfolio Context: User holds 100 shares.",
        condition: (ctx) => !!ctx.hasPortfolio,
      });

      const result = template.render({
        symbol: "AAPL",
        hasPortfolio: true,
      });

      expect(result).toContain("Symbol: AAPL");
      expect(result).toContain("Timeframe: medium-term");
      expect(result).toContain("Portfolio Context");
    });
  });

  describe("Research Report Prompt", () => {
    it("should create research report prompt", () => {
      const builder = new MockPromptBuilder()
        .setSystemPrompt(
          "You are a senior equity research analyst at a major investment bank."
        )
        .addContext("Generating comprehensive research report.")
        .addInstruction("Include valuation analysis")
        .addInstruction("Discuss competitive positioning")
        .addInstruction("Provide price target and rating");

      const result = builder.build();

      expect(result).toContain("equity research analyst");
      expect(result).toContain("valuation analysis");
      expect(result).toContain("price target");
    });
  });

  describe("Tool-Augmented Prompt", () => {
    it("should create prompt with tool descriptions", () => {
      const builder = new MockPromptBuilder()
        .setSystemPrompt("You are Stanley with access to investment research tools.")
        .addToolDescription("get_market_data", "Get real-time market data including price, volume, and changes")
        .addToolDescription("get_research", "Get comprehensive research report with valuation and earnings")
        .addToolDescription("get_institutional_holdings", "Get 13F institutional holdings data")
        .addToolDescription("analyze_money_flow", "Analyze sector money flow patterns");

      const result = builder.build();

      expect(result).toContain("get_market_data");
      expect(result).toContain("get_research");
      expect(result).toContain("get_institutional_holdings");
      expect(result).toContain("analyze_money_flow");
    });
  });
});
