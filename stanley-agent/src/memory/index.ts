/**
 * Memory Persistence Layer
 *
 * This module provides persistent memory capabilities for the Stanley Agent:
 * - Key-value store with TTL and namespacing
 * - Conversation history management
 * - Insights extraction and recall
 * - Vector embeddings for semantic search
 */

// Import all modules for local use
import {
  MemoryStore,
  createMemoryStore,
  getDefaultStore,
  closeDefaultStore,
} from "./store";

import {
  ConversationManager,
  createConversationManager,
  getDefaultConversationManager,
  closeDefaultConversationManager,
} from "./conversation";

import {
  InsightsManager,
  createInsightsManager,
  getDefaultInsightsManager,
  closeDefaultInsightsManager,
} from "./insights";

import {
  EmbeddingsStore,
  SimpleEmbeddingProvider,
  createEmbeddingsStore,
  getDefaultEmbeddingsStore,
  closeDefaultEmbeddingsStore,
  cosineSimilarity,
  euclideanDistance,
} from "./embeddings";

// Re-export everything
export {
  // Memory Store
  MemoryStore,
  createMemoryStore,
  getDefaultStore,
  closeDefaultStore,
  // Conversation Manager
  ConversationManager,
  createConversationManager,
  getDefaultConversationManager,
  closeDefaultConversationManager,
  // Insights Manager
  InsightsManager,
  createInsightsManager,
  getDefaultInsightsManager,
  closeDefaultInsightsManager,
  // Embeddings Store
  EmbeddingsStore,
  SimpleEmbeddingProvider,
  createEmbeddingsStore,
  getDefaultEmbeddingsStore,
  closeDefaultEmbeddingsStore,
  cosineSimilarity,
  euclideanDistance,
};

// Re-export types
export type {
  MemoryNamespace,
  MemoryEntry,
  SetOptions,
  QueryOptions,
  SearchResult,
} from "./store";

export type {
  ToolCallRecord,
  ConversationMessage,
  ConversationSession,
  Conversation,
  SummarizationOptions,
  ConversationExport,
} from "./conversation";

export type {
  InsightCategory,
  InsightImportance,
  Insight,
  CreateInsightOptions,
  SearchInsightOptions,
  InsightRecall,
} from "./insights";

export type {
  EmbeddingVector,
  EmbeddedDocument,
  SimilarityResult,
  EmbeddingProvider,
} from "./embeddings";

/**
 * Memory Manager - Unified interface for all memory operations
 */
export class MemoryManager {
  private _store: MemoryStore | null = null;
  private _conversations: ConversationManager | null = null;
  private _insights: InsightsManager | null = null;
  private _embeddings: EmbeddingsStore | null = null;

  /**
   * Get or create the memory store
   */
  get store(): MemoryStore {
    if (!this._store) {
      this._store = getDefaultStore();
    }
    return this._store;
  }

  /**
   * Get or create the conversation manager
   */
  get conversations(): ConversationManager {
    if (!this._conversations) {
      this._conversations = getDefaultConversationManager();
    }
    return this._conversations;
  }

  /**
   * Get or create the insights manager
   */
  get insights(): InsightsManager {
    if (!this._insights) {
      this._insights = getDefaultInsightsManager();
    }
    return this._insights;
  }

  /**
   * Get or create the embeddings store
   */
  get embeddings(): EmbeddingsStore {
    if (!this._embeddings) {
      this._embeddings = getDefaultEmbeddingsStore();
    }
    return this._embeddings;
  }

  /**
   * Close all database connections
   */
  close(): void {
    if (this._store) {
      this._store.close();
      this._store = null;
    }
    if (this._conversations) {
      this._conversations.close();
      this._conversations = null;
    }
    if (this._insights) {
      this._insights.close();
      this._insights = null;
    }
    if (this._embeddings) {
      this._embeddings.close();
      this._embeddings = null;
    }
  }

  /**
   * Get statistics across all memory stores
   */
  getStats(): {
    store: ReturnType<MemoryStore["getStats"]>;
    conversations: ReturnType<ConversationManager["getStats"]>;
    insights: ReturnType<InsightsManager["getStats"]>;
    embeddings: ReturnType<EmbeddingsStore["getStats"]>;
  } {
    return {
      store: this.store.getStats(),
      conversations: this.conversations.getStats(),
      insights: this.insights.getStats(),
      embeddings: this.embeddings.getStats(),
    };
  }
}

/**
 * Singleton memory manager instance
 */
let defaultMemoryManager: MemoryManager | null = null;

export function getMemoryManager(): MemoryManager {
  if (!defaultMemoryManager) {
    defaultMemoryManager = new MemoryManager();
  }
  return defaultMemoryManager;
}

export function closeMemoryManager(): void {
  if (defaultMemoryManager) {
    defaultMemoryManager.close();
    defaultMemoryManager = null;
  }
}

/**
 * Cleanup function to close all singleton instances
 */
export function cleanupMemory(): void {
  closeDefaultStore();
  closeDefaultConversationManager();
  closeDefaultInsightsManager();
  closeDefaultEmbeddingsStore();
  closeMemoryManager();
}
