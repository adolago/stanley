/**
 * Stanley API Client
 *
 * HTTP client for communicating with Stanley's Python backend API.
 */

import type { StanleyApiResponse } from "./types";

const STANLEY_API_BASE_URL =
  process.env.STANLEY_API_URL || "http://localhost:8000";

export interface FetchOptions {
  method?: "GET" | "POST" | "PUT" | "DELETE";
  body?: unknown;
  headers?: Record<string, string>;
  signal?: AbortSignal;
}

/**
 * Make an API request to Stanley's backend
 */
export async function stanleyFetch<T>(
  endpoint: string,
  options: FetchOptions = {}
): Promise<StanleyApiResponse<T>> {
  const { method = "GET", body, headers = {}, signal } = options;

  const url = `${STANLEY_API_BASE_URL}${endpoint}`;

  const fetchOptions: RequestInit = {
    method,
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
      ...headers,
    },
    signal,
  };

  if (body && method !== "GET") {
    fetchOptions.body = JSON.stringify(body);
  }

  try {
    const response = await fetch(url, fetchOptions);

    if (!response.ok) {
      const errorText = await response.text();
      return {
        success: false,
        data: null,
        error: `HTTP ${response.status}: ${errorText}`,
        timestamp: new Date().toISOString(),
      };
    }

    const data = await response.json();
    return data as StanleyApiResponse<T>;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return {
      success: false,
      data: null,
      error: `API request failed: ${message}`,
      timestamp: new Date().toISOString(),
    };
  }
}

/**
 * Format API response for tool output
 */
export function formatToolOutput<T>(
  response: StanleyApiResponse<T>,
  title: string
): string {
  if (!response.success || response.error) {
    return `Error: ${response.error || "Unknown error occurred"}`;
  }

  if (!response.data) {
    return `${title}: No data available`;
  }

  // Format the data as readable JSON
  return `${title}:\n${JSON.stringify(response.data, null, 2)}`;
}

/**
 * Build query string from parameters
 */
export function buildQueryString(
  params: Record<string, string | number | boolean | undefined>
): string {
  const entries = Object.entries(params)
    .filter(([_, value]) => value !== undefined)
    .map(([key, value]) => `${encodeURIComponent(key)}=${encodeURIComponent(String(value))}`);

  return entries.length > 0 ? `?${entries.join("&")}` : "";
}
