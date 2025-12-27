/**
 * Embeddings Module - Vector similarity for semantic search
 *
 * This module provides a stub implementation for semantic search using embeddings.
 * For production use, integrate with an embedding provider (OpenAI, Cohere, etc.)
 * and a vector database (Pinecone, Weaviate, Qdrant, etc.)
 */

import { Database } from "bun:sqlite";
import { existsSync, mkdirSync } from "fs";
import { dirname, join } from "path";

/**
 * Embedding vector type
 */
export type EmbeddingVector = number[];

/**
 * Embedded document
 */
export interface EmbeddedDocument {
  id: string;
  content: string;
  embedding: EmbeddingVector;
  metadata?: Record<string, unknown>;
  createdAt: number;
}

/**
 * Similarity search result
 */
export interface SimilarityResult {
  document: EmbeddedDocument;
  score: number;
}

/**
 * Embedding provider interface
 */
export interface EmbeddingProvider {
  embed(text: string): Promise<EmbeddingVector>;
  embedBatch(texts: string[]): Promise<EmbeddingVector[]>;
  dimensions: number;
}

/**
 * Simple TF-IDF based embedding provider (fallback when no API is available)
 *
 * This is a basic implementation for demonstration and testing.
 * For production, use a proper embedding model.
 */
export class SimpleEmbeddingProvider implements EmbeddingProvider {
  readonly dimensions = 256;
  private vocabulary: Map<string, number> = new Map();
  private idf: Map<string, number> = new Map();

  /**
   * Tokenize text into words
   */
  private tokenize(text: string): string[] {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, " ")
      .split(/\s+/)
      .filter((word) => word.length > 2);
  }

  /**
   * Hash a word to a dimension index
   */
  private hashWord(word: string): number {
    let hash = 0;
    for (let i = 0; i < word.length; i++) {
      hash = (hash * 31 + word.charCodeAt(i)) % this.dimensions;
    }
    return hash;
  }

  /**
   * Generate embedding for text
   */
  async embed(text: string): Promise<EmbeddingVector> {
    const tokens = this.tokenize(text);
    const vector = new Array(this.dimensions).fill(0);

    // Count term frequencies
    const tf: Map<string, number> = new Map();
    for (const token of tokens) {
      tf.set(token, (tf.get(token) || 0) + 1);
    }

    // Build vector using hashing trick
    for (const [word, count] of tf) {
      const idx = this.hashWord(word);
      const freq = count / tokens.length;
      vector[idx] += freq;
    }

    // Normalize
    const magnitude = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0));
    if (magnitude > 0) {
      for (let i = 0; i < vector.length; i++) {
        vector[i] /= magnitude;
      }
    }

    return vector;
  }

  /**
   * Generate embeddings for multiple texts
   */
  async embedBatch(texts: string[]): Promise<EmbeddingVector[]> {
    return Promise.all(texts.map((text) => this.embed(text)));
  }
}

/**
 * Cosine similarity between two vectors
 */
export function cosineSimilarity(a: EmbeddingVector, b: EmbeddingVector): number {
  if (a.length !== b.length) {
    throw new Error("Vectors must have same dimension");
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
  return magnitude === 0 ? 0 : dotProduct / magnitude;
}

/**
 * Euclidean distance between two vectors
 */
export function euclideanDistance(a: EmbeddingVector, b: EmbeddingVector): number {
  if (a.length !== b.length) {
    throw new Error("Vectors must have same dimension");
  }

  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }

  return Math.sqrt(sum);
}

/**
 * Embeddings Store - SQLite-based vector storage
 *
 * Note: For large-scale production use, consider using a dedicated vector database.
 */
export class EmbeddingsStore {
  private db: Database;
  private provider: EmbeddingProvider;

  constructor(provider?: EmbeddingProvider, dbPath?: string) {
    this.provider = provider || new SimpleEmbeddingProvider();
    const path = dbPath || this.getDefaultDbPath();
    this.ensureDirectory(path);
    this.db = new Database(path);
    this.initializeSchema();
  }

  /**
   * Get default database path
   */
  private getDefaultDbPath(): string {
    const xdgData = process.env.XDG_DATA_HOME || join(process.env.HOME || "", ".local", "share");
    return join(xdgData, "stanley-agent", "embeddings.db");
  }

  /**
   * Ensure directory exists
   */
  private ensureDirectory(dbPath: string): void {
    const dir = dirname(dbPath);
    if (!existsSync(dir)) {
      mkdirSync(dir, { recursive: true });
    }
  }

  /**
   * Initialize database schema
   */
  private initializeSchema(): void {
    this.db.run(`
      CREATE TABLE IF NOT EXISTS embeddings (
        id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        embedding BLOB NOT NULL,
        metadata TEXT,
        created_at INTEGER NOT NULL
      )
    `);

    this.db.run(`
      CREATE INDEX IF NOT EXISTS idx_embeddings_created ON embeddings(created_at DESC)
    `);
  }

  /**
   * Generate unique ID
   */
  private generateId(): string {
    return `emb-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
  }

  /**
   * Serialize embedding vector to binary
   */
  private serializeVector(vector: EmbeddingVector): Buffer {
    return Buffer.from(new Float64Array(vector).buffer);
  }

  /**
   * Deserialize embedding vector from binary
   */
  private deserializeVector(buffer: Buffer): EmbeddingVector {
    return Array.from(new Float64Array(buffer.buffer, buffer.byteOffset, buffer.length / 8));
  }

  /**
   * Add a document to the store
   */
  async add(
    content: string,
    options?: { id?: string; metadata?: Record<string, unknown> }
  ): Promise<EmbeddedDocument> {
    const id = options?.id || this.generateId();
    const now = Date.now();

    // Generate embedding
    const embedding = await this.provider.embed(content);

    const doc: EmbeddedDocument = {
      id,
      content,
      embedding,
      metadata: options?.metadata,
      createdAt: now,
    };

    this.db.run(
      `
      INSERT OR REPLACE INTO embeddings (id, content, embedding, metadata, created_at)
      VALUES (?, ?, ?, ?, ?)
      `,
      [
        id,
        content,
        this.serializeVector(embedding),
        options?.metadata ? JSON.stringify(options.metadata) : null,
        now,
      ]
    );

    return doc;
  }

  /**
   * Add multiple documents
   */
  async addBatch(
    documents: Array<{ content: string; id?: string; metadata?: Record<string, unknown> }>
  ): Promise<EmbeddedDocument[]> {
    // Generate embeddings in batch
    const embeddings = await this.provider.embedBatch(documents.map((d) => d.content));
    const now = Date.now();

    const results: EmbeddedDocument[] = [];

    for (let i = 0; i < documents.length; i++) {
      const doc = documents[i];
      const id = doc.id || this.generateId();
      const embedding = embeddings[i];

      this.db.run(
        `
        INSERT OR REPLACE INTO embeddings (id, content, embedding, metadata, created_at)
        VALUES (?, ?, ?, ?, ?)
        `,
        [
          id,
          doc.content,
          this.serializeVector(embedding),
          doc.metadata ? JSON.stringify(doc.metadata) : null,
          now,
        ]
      );

      results.push({
        id,
        content: doc.content,
        embedding,
        metadata: doc.metadata,
        createdAt: now,
      });
    }

    return results;
  }

  /**
   * Get a document by ID
   */
  get(id: string): EmbeddedDocument | null {
    const row = this.db
      .query<
        {
          id: string;
          content: string;
          embedding: Buffer;
          metadata: string | null;
          created_at: number;
        },
        [string]
      >("SELECT * FROM embeddings WHERE id = ?")
      .get(id);

    if (!row) return null;

    return {
      id: row.id,
      content: row.content,
      embedding: this.deserializeVector(row.embedding),
      metadata: row.metadata ? JSON.parse(row.metadata) : undefined,
      createdAt: row.created_at,
    };
  }

  /**
   * Delete a document
   */
  delete(id: string): boolean {
    const result = this.db.run("DELETE FROM embeddings WHERE id = ?", [id]);
    return result.changes > 0;
  }

  /**
   * Search for similar documents
   */
  async search(
    query: string,
    options?: { limit?: number; minScore?: number }
  ): Promise<SimilarityResult[]> {
    const { limit = 10, minScore = 0 } = options || {};

    // Generate query embedding
    const queryEmbedding = await this.provider.embed(query);

    // Get all documents (for small datasets)
    // For larger datasets, use approximate nearest neighbor search
    const rows = this.db
      .query<
        {
          id: string;
          content: string;
          embedding: Buffer;
          metadata: string | null;
          created_at: number;
        },
        []
      >("SELECT * FROM embeddings")
      .all();

    // Calculate similarities
    const results: SimilarityResult[] = rows
      .map((row) => {
        const embedding = this.deserializeVector(row.embedding);
        const score = cosineSimilarity(queryEmbedding, embedding);

        return {
          document: {
            id: row.id,
            content: row.content,
            embedding,
            metadata: row.metadata ? JSON.parse(row.metadata) : undefined,
            createdAt: row.created_at,
          },
          score,
        };
      })
      .filter((r) => r.score >= minScore)
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);

    return results;
  }

  /**
   * Search by embedding vector directly
   */
  searchByVector(
    embedding: EmbeddingVector,
    options?: { limit?: number; minScore?: number }
  ): SimilarityResult[] {
    const { limit = 10, minScore = 0 } = options || {};

    const rows = this.db
      .query<
        {
          id: string;
          content: string;
          embedding: Buffer;
          metadata: string | null;
          created_at: number;
        },
        []
      >("SELECT * FROM embeddings")
      .all();

    const results: SimilarityResult[] = rows
      .map((row) => {
        const docEmbedding = this.deserializeVector(row.embedding);
        const score = cosineSimilarity(embedding, docEmbedding);

        return {
          document: {
            id: row.id,
            content: row.content,
            embedding: docEmbedding,
            metadata: row.metadata ? JSON.parse(row.metadata) : undefined,
            createdAt: row.created_at,
          },
          score,
        };
      })
      .filter((r) => r.score >= minScore)
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);

    return results;
  }

  /**
   * Get statistics
   */
  getStats(): {
    totalDocuments: number;
    dimensions: number;
    totalSize: number;
  } {
    const count = this.db
      .query<{ count: number }, []>("SELECT COUNT(*) as count FROM embeddings")
      .get();

    const size = this.db
      .query<{ total: number }, []>(
        "SELECT SUM(LENGTH(embedding) + LENGTH(content) + COALESCE(LENGTH(metadata), 0)) as total FROM embeddings"
      )
      .get();

    return {
      totalDocuments: count?.count || 0,
      dimensions: this.provider.dimensions,
      totalSize: size?.total || 0,
    };
  }

  /**
   * Clear all documents
   */
  clear(): number {
    const result = this.db.run("DELETE FROM embeddings");
    return result.changes;
  }

  /**
   * Close the database connection
   */
  close(): void {
    this.db.close();
  }
}

/**
 * Create an embeddings store instance
 */
export function createEmbeddingsStore(
  provider?: EmbeddingProvider,
  dbPath?: string
): EmbeddingsStore {
  return new EmbeddingsStore(provider, dbPath);
}

/**
 * Singleton instance for shared access
 */
let defaultStore: EmbeddingsStore | null = null;

export function getDefaultEmbeddingsStore(): EmbeddingsStore {
  if (!defaultStore) {
    defaultStore = createEmbeddingsStore();
  }
  return defaultStore;
}

export function closeDefaultEmbeddingsStore(): void {
  if (defaultStore) {
    defaultStore.close();
    defaultStore = null;
  }
}

/**
 * Integration example with OpenAI embeddings (requires @ai-sdk/openai)
 *
 * Uncomment and adapt for production use:
 *
 * ```typescript
 * import { createOpenAI } from "@ai-sdk/openai";
 * import { embed, embedMany } from "ai";
 *
 * export class OpenAIEmbeddingProvider implements EmbeddingProvider {
 *   readonly dimensions = 1536; // text-embedding-3-small
 *   private model;
 *
 *   constructor(apiKey?: string) {
 *     const openai = createOpenAI({ apiKey });
 *     this.model = openai.embedding("text-embedding-3-small");
 *   }
 *
 *   async embed(text: string): Promise<EmbeddingVector> {
 *     const result = await embed({ model: this.model, value: text });
 *     return result.embedding;
 *   }
 *
 *   async embedBatch(texts: string[]): Promise<EmbeddingVector[]> {
 *     const result = await embedMany({ model: this.model, values: texts });
 *     return result.embeddings;
 *   }
 * }
 * ```
 */
