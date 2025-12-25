//! API client for communicating with Stanley Python backend
//!
//! Provides async methods for fetching money flow data, institutional
//! holdings, and other analytics from the Stanley backend service.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// API client for Stanley backend
pub struct StanleyClient {
    base_url: String,
    client: reqwest::blocking::Client,
}

impl StanleyClient {
    /// Create a new client with default localhost URL
    pub fn new() -> Self {
        Self::with_url("http://localhost:8000".to_string())
    }

    /// Create a new client with custom base URL
    pub fn with_url(base_url: String) -> Self {
        Self {
            base_url,
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Get sector money flow analysis
    pub fn get_sector_money_flow(&self, sectors: Vec<String>) -> Result<SectorFlowResponse, ApiError> {
        let url = format!("{}/api/money-flow", self.base_url);
        let response = self.client
            .post(&url)
            .json(&SectorFlowRequest { sectors })
            .send()
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response.json().map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Get institutional holdings for a symbol
    pub fn get_institutional_holdings(&self, symbol: &str) -> Result<InstitutionalHoldingsResponse, ApiError> {
        let url = format!("{}/api/institutional/{}", self.base_url, symbol);
        let response = self.client
            .get(&url)
            .send()
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response.json().map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Get equity money flow analysis
    pub fn get_equity_flow(&self, symbol: &str) -> Result<EquityFlowResponse, ApiError> {
        let url = format!("{}/api/equity-flow/{}", self.base_url, symbol);
        let response = self.client
            .get(&url)
            .send()
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response.json().map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Get dark pool activity
    pub fn get_dark_pool_activity(&self, symbol: &str) -> Result<DarkPoolResponse, ApiError> {
        let url = format!("{}/api/dark-pool/{}", self.base_url, symbol);
        let response = self.client
            .get(&url)
            .send()
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response.json().map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Health check
    pub fn health_check(&self) -> Result<HealthResponse, ApiError> {
        let url = format!("{}/api/health", self.base_url);
        let response = self.client
            .get(&url)
            .send()
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response.json().map_err(|e| ApiError::Parse(e.to_string()))
    }
}

impl Default for StanleyClient {
    fn default() -> Self {
        Self::new()
    }
}

/// API error types
#[derive(Debug)]
pub enum ApiError {
    Network(String),
    Parse(String),
    Server(String),
}

// Request/Response types

#[derive(Debug, Serialize)]
pub struct SectorFlowRequest {
    pub sectors: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct SectorFlowResponse {
    pub sectors: Vec<SectorFlow>,
}

#[derive(Debug, Deserialize)]
pub struct SectorFlow {
    pub sector: String,
    pub net_flow_1m: f64,
    pub net_flow_3m: f64,
    pub institutional_change: f64,
    pub smart_money_sentiment: f64,
    pub confidence_score: f64,
}

#[derive(Debug, Deserialize)]
pub struct InstitutionalHoldingsResponse {
    pub symbol: String,
    pub institutional_ownership: f64,
    pub number_of_institutions: u32,
    pub top_holders: Vec<InstitutionalHolder>,
    pub ownership_trend: f64,
    pub concentration_risk: f64,
}

#[derive(Debug, Deserialize)]
pub struct InstitutionalHolder {
    pub manager_name: String,
    pub value_held: f64,
    pub ownership_percentage: f64,
}

#[derive(Debug, Deserialize)]
pub struct EquityFlowResponse {
    pub symbol: String,
    pub money_flow_score: f64,
    pub institutional_sentiment: f64,
    pub smart_money_activity: f64,
    pub short_pressure: f64,
    pub accumulation_distribution: f64,
    pub confidence: f64,
}

#[derive(Debug, Deserialize)]
pub struct DarkPoolResponse {
    pub symbol: String,
    pub data: Vec<DarkPoolData>,
}

#[derive(Debug, Deserialize)]
pub struct DarkPoolData {
    pub date: String,
    pub dark_pool_volume: u64,
    pub total_volume: u64,
    pub dark_pool_percentage: f64,
    pub dark_pool_signal: i8,
}

#[derive(Debug, Deserialize)]
pub struct HealthResponse {
    pub core: bool,
    pub status: String,
}
