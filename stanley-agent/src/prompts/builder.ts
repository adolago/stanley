/**
 * Dynamic Prompt Builder
 *
 * Builds optimized prompts by assessing query intent, selecting
 * appropriate templates/personas, and injecting relevant context.
 */

import { PROMPT_TEMPLATES, PromptTemplateKey, getPromptTemplate } from "./templates";
import { PERSONAS, PersonaId, getPersona, getPersonaForAnalysis, type Persona } from "./personas";
import {
  ContextInjector,
  createContextInjector,
  type ContextData,
  type InjectionOptions,
  type InjectedContext,
} from "./context-injector";

/**
 * Query intent categories
 */
export type QueryIntent =
  | "portfolio_analysis"
  | "risk_assessment"
  | "macro_analysis"
  | "trade_idea"
  | "research"
  | "earnings"
  | "sector_analysis"
  | "quant_analysis"
  | "general";

/**
 * Intent detection result
 */
export interface IntentResult {
  intent: QueryIntent;
  confidence: number;
  keywords: string[];
  symbols: string[];
}

/**
 * Prompt build result
 */
export interface BuiltPrompt {
  /** Final assembled prompt */
  prompt: string;
  /** Detected query intent */
  intent: IntentResult;
  /** Selected template */
  template: PromptTemplateKey;
  /** Selected persona */
  persona: PersonaId;
  /** Context injection result */
  context: InjectedContext;
  /** Total estimated tokens */
  estimatedTokens: number;
  /** Build metadata */
  metadata: {
    buildTimestamp: string;
    templateSelected: string;
    personaSelected: string;
    contextIncluded: string[];
    symbolsFocused: string[];
  };
}

/**
 * Builder options
 */
export interface BuilderOptions {
  /** Maximum total tokens for prompt */
  maxPromptTokens?: number;
  /** Override template selection */
  forceTemplate?: PromptTemplateKey;
  /** Override persona selection */
  forcePersona?: PersonaId;
  /** Additional instructions to append */
  additionalInstructions?: string;
  /** Context injection options */
  contextOptions?: InjectionOptions;
}

/**
 * Intent keywords mapping
 */
const INTENT_KEYWORDS: Record<QueryIntent, string[]> = {
  portfolio_analysis: [
    "portfolio", "holdings", "positions", "allocation", "diversification",
    "rebalance", "weight", "exposure", "benchmark", "performance",
  ],
  risk_assessment: [
    "risk", "var", "value at risk", "volatility", "drawdown", "beta",
    "stress test", "downside", "tail risk", "cvar", "expected shortfall",
  ],
  macro_analysis: [
    "macro", "economy", "economic", "regime", "fed", "rates", "inflation",
    "gdp", "yield curve", "recession", "cycle", "monetary", "fiscal",
  ],
  trade_idea: [
    "trade", "buy", "sell", "long", "short", "entry", "exit", "target",
    "stop loss", "catalyst", "opportunity", "setup", "momentum",
  ],
  research: [
    "research", "analysis", "report", "valuation", "dcf", "fair value",
    "thesis", "fundamentals", "growth", "margins", "competitive",
  ],
  earnings: [
    "earnings", "eps", "revenue", "quarter", "guidance", "beat", "miss",
    "surprise", "call", "forecast", "estimate", "consensus",
  ],
  sector_analysis: [
    "sector", "industry", "rotation", "cyclical", "defensive", "relative strength",
    "technology", "healthcare", "financials", "energy", "materials",
  ],
  quant_analysis: [
    "quantitative", "statistical", "correlation", "regression", "factor",
    "backtest", "alpha", "sharpe", "sortino", "information ratio",
  ],
  general: [],
};

/**
 * Map intents to templates
 */
const INTENT_TEMPLATE_MAP: Record<QueryIntent, PromptTemplateKey> = {
  portfolio_analysis: "portfolio",
  risk_assessment: "risk",
  macro_analysis: "macro",
  trade_idea: "trade",
  research: "research",
  earnings: "earnings",
  sector_analysis: "sector",
  quant_analysis: "quant",
  general: "research",
};

/**
 * Map intents to personas
 */
const INTENT_PERSONA_MAP: Record<QueryIntent, PersonaId> = {
  portfolio_analysis: "pm",
  risk_assessment: "risk",
  macro_analysis: "macro",
  trade_idea: "trader",
  research: "analyst",
  earnings: "analyst",
  sector_analysis: "analyst",
  quant_analysis: "quant",
  general: "analyst",
};

/**
 * Detect query intent from user message
 */
export function detectIntent(query: string): IntentResult {
  const normalizedQuery = query.toLowerCase();
  const foundKeywords: string[] = [];
  let bestIntent: QueryIntent = "general";
  let bestScore = 0;

  // Check each intent's keywords
  for (const [intent, keywords] of Object.entries(INTENT_KEYWORDS)) {
    if (intent === "general") continue;

    let score = 0;
    const matched: string[] = [];

    for (const keyword of keywords) {
      if (normalizedQuery.includes(keyword)) {
        score += keyword.split(" ").length; // Multi-word keywords score higher
        matched.push(keyword);
      }
    }

    if (score > bestScore) {
      bestScore = score;
      bestIntent = intent as QueryIntent;
      foundKeywords.length = 0;
      foundKeywords.push(...matched);
    }
  }

  // Calculate confidence based on keyword matches
  const maxPossibleScore = INTENT_KEYWORDS[bestIntent].reduce(
    (sum, kw) => sum + kw.split(" ").length,
    0
  );
  const confidence = maxPossibleScore > 0
    ? Math.min(bestScore / Math.max(maxPossibleScore / 3, 1), 1)
    : 0;

  // Extract symbols from query
  const symbols = ContextInjector.extractSymbols(query);

  return {
    intent: bestIntent,
    confidence: Math.round(confidence * 100) / 100,
    keywords: foundKeywords,
    symbols,
  };
}

/**
 * Dynamic Prompt Builder class
 */
export class PromptBuilder {
  private contextInjector: ContextInjector;
  private defaultOptions: BuilderOptions;

  constructor(contextData?: ContextData, options?: BuilderOptions) {
    this.contextInjector = createContextInjector(contextData);
    this.defaultOptions = {
      maxPromptTokens: 8000,
      ...options,
    };
  }

  /**
   * Update context data
   */
  setContext(data: Partial<ContextData>): void {
    this.contextInjector.setContext(data);
  }

  /**
   * Build an optimized prompt for a query
   */
  build(query: string, options?: BuilderOptions): BuiltPrompt {
    const opts = { ...this.defaultOptions, ...options };
    const buildTimestamp = new Date().toISOString();

    // Step 1: Detect intent
    const intent = detectIntent(query);

    // Step 2: Select template (override or auto-detect)
    const templateKey = opts.forceTemplate ?? INTENT_TEMPLATE_MAP[intent.intent];
    const template = getPromptTemplate(templateKey);

    // Step 3: Select persona (override or auto-detect)
    const personaId = opts.forcePersona ?? INTENT_PERSONA_MAP[intent.intent];
    const persona = getPersona(personaId);

    // Step 4: Set focus symbols for context prioritization
    if (intent.symbols.length > 0) {
      this.contextInjector.setFocusSymbols(intent.symbols);
    }

    // Step 5: Calculate token budget for context
    const basePromptTokens = this.estimateTokens(template + persona.promptAddendum);
    const additionalTokens = opts.additionalInstructions
      ? this.estimateTokens(opts.additionalInstructions)
      : 0;
    const contextBudget = Math.max(
      opts.maxPromptTokens! - basePromptTokens - additionalTokens - 500, // Buffer
      500
    );

    // Step 6: Inject context
    this.contextInjector.setOptions({
      ...opts.contextOptions,
      maxTokens: contextBudget,
      focusSymbols: intent.symbols,
    });
    const contextResult = this.contextInjector.inject();

    // Step 7: Assemble final prompt
    let prompt = template;
    prompt += persona.promptAddendum;
    prompt += contextResult.contextString;

    if (opts.additionalInstructions) {
      prompt += `\n## Additional Instructions\n\n${opts.additionalInstructions}\n`;
    }

    const totalTokens = this.estimateTokens(prompt);

    return {
      prompt,
      intent,
      template: templateKey,
      persona: personaId,
      context: contextResult,
      estimatedTokens: totalTokens,
      metadata: {
        buildTimestamp,
        templateSelected: templateKey,
        personaSelected: personaId,
        contextIncluded: contextResult.includedTypes,
        symbolsFocused: intent.symbols,
      },
    };
  }

  /**
   * Build a prompt with explicit template and persona
   */
  buildWithConfig(
    query: string,
    template: PromptTemplateKey,
    persona: PersonaId,
    options?: Omit<BuilderOptions, "forceTemplate" | "forcePersona">
  ): BuiltPrompt {
    return this.build(query, {
      ...options,
      forceTemplate: template,
      forcePersona: persona,
    });
  }

  /**
   * Get recommended template for a query
   */
  recommendTemplate(query: string): PromptTemplateKey {
    const intent = detectIntent(query);
    return INTENT_TEMPLATE_MAP[intent.intent];
  }

  /**
   * Get recommended persona for a query
   */
  recommendPersona(query: string): PersonaId {
    const intent = detectIntent(query);
    return INTENT_PERSONA_MAP[intent.intent];
  }

  /**
   * Estimate tokens for text
   */
  private estimateTokens(text: string): number {
    return Math.ceil(text.length / 4);
  }

  /**
   * Get all available templates
   */
  static getTemplates(): PromptTemplateKey[] {
    return Object.keys(PROMPT_TEMPLATES) as PromptTemplateKey[];
  }

  /**
   * Get all available personas
   */
  static getPersonas(): PersonaId[] {
    return Object.keys(PERSONAS) as PersonaId[];
  }
}

/**
 * Create a prompt builder with context
 */
export function createPromptBuilder(
  contextData?: ContextData,
  options?: BuilderOptions
): PromptBuilder {
  return new PromptBuilder(contextData, options);
}

/**
 * Quick helper: build a prompt for a query with auto-detection
 */
export function buildPrompt(
  query: string,
  contextData?: ContextData,
  options?: BuilderOptions
): BuiltPrompt {
  const builder = new PromptBuilder(contextData, options);
  return builder.build(query, options);
}

// detectIntent is already exported above as a function
