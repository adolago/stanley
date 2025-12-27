/**
 * Stanley Agent Prompts Module
 *
 * Context-aware prompting system for enhanced agent intelligence.
 *
 * Features:
 * - Template-based system prompts for different analysis modes
 * - Persona profiles for tailored communication styles
 * - Dynamic context injection with token management
 * - Intent detection for automatic prompt optimization
 *
 * Usage:
 * ```typescript
 * import { createPromptBuilder, buildPrompt } from './prompts';
 *
 * // Quick usage with auto-detection
 * const result = buildPrompt("Analyze my portfolio risk", {
 *   portfolio: myPositions,
 *   regime: currentRegime,
 * });
 *
 * // Full control with builder
 * const builder = createPromptBuilder();
 * builder.setContext({ portfolio: myPositions });
 * const result = builder.build("What's my VaR?", {
 *   forcePersona: "risk",
 * });
 * ```
 */

// Templates
export {
  BASE_TEMPLATE,
  RESEARCH_ASSISTANT_TEMPLATE,
  PORTFOLIO_ANALYSIS_TEMPLATE,
  RISK_ASSESSMENT_TEMPLATE,
  MACRO_ANALYSIS_TEMPLATE,
  TRADE_IDEA_TEMPLATE,
  QUANT_ANALYSIS_TEMPLATE,
  EARNINGS_ANALYSIS_TEMPLATE,
  SECTOR_ANALYSIS_TEMPLATE,
  PROMPT_TEMPLATES,
  getPromptTemplate,
  getTemplateKeys,
  type PromptTemplateKey,
} from "./templates";

// Personas
export {
  ANALYST_PERSONA,
  QUANT_PERSONA,
  MACRO_PERSONA,
  RISK_MANAGER_PERSONA,
  PM_PERSONA,
  TRADER_PERSONA,
  PERSONAS,
  getPersona,
  getPersonaIds,
  getPersonaForAnalysis,
  type Persona,
  type PersonaId,
} from "./personas";

// Context Injector
export {
  ContextInjector,
  createContextInjector,
  type ContextData,
  type PortfolioPosition,
  type ResearchFinding,
  type MarketRegimeSummary,
  type NoteSnippet,
  type ToolResult,
  type InjectionOptions,
  type ContextPriority,
  type InjectedContext,
} from "./context-injector";

// Prompt Builder
export {
  PromptBuilder,
  createPromptBuilder,
  buildPrompt,
  detectIntent,
  type QueryIntent,
  type IntentResult,
  type BuiltPrompt,
  type BuilderOptions,
} from "./builder";

// Re-import for default export convenience object
import {
  createPromptBuilder as _createPromptBuilder,
  buildPrompt as _buildPrompt,
  detectIntent as _detectIntent,
} from "./builder";
import { getPromptTemplate as _getPromptTemplate } from "./templates";
import { getPersona as _getPersona } from "./personas";
import { createContextInjector as _createContextInjector } from "./context-injector";

/**
 * Default export for convenient access
 */
export default {
  createPromptBuilder: _createPromptBuilder,
  buildPrompt: _buildPrompt,
  detectIntent: _detectIntent,
  getPromptTemplate: _getPromptTemplate,
  getPersona: _getPersona,
  createContextInjector: _createContextInjector,
};
