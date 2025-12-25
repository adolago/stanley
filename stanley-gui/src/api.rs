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
    pub fn get_sector_money_flow(
        &self,
        sectors: Vec<String>,
    ) -> Result<SectorFlowResponse, ApiError> {
        let url = format!("{}/api/money-flow", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&SectorFlowRequest { sectors })
            .send()
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response.json().map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Get institutional holdings for a symbol
    pub fn get_institutional_holdings(
        &self,
        symbol: &str,
    ) -> Result<InstitutionalHoldingsResponse, ApiError> {
        let url = format!("{}/api/institutional/{}", self.base_url, symbol);
        let response = self
            .client
            .get(&url)
            .send()
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response.json().map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Get equity money flow analysis
    pub fn get_equity_flow(&self, symbol: &str) -> Result<EquityFlowResponse, ApiError> {
        let url = format!("{}/api/equity-flow/{}", self.base_url, symbol);
        let response = self
            .client
            .get(&url)
            .send()
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response.json().map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Get dark pool activity
    pub fn get_dark_pool_activity(&self, symbol: &str) -> Result<DarkPoolResponse, ApiError> {
        let url = format!("{}/api/dark-pool/{}", self.base_url, symbol);
        let response = self
            .client
            .get(&url)
            .send()
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response.json().map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Health check
    pub fn health_check(&self) -> Result<HealthResponse, ApiError> {
        let url = format!("{}/api/health", self.base_url);
        let response = self
            .client
            .get(&url)
            .send()
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response.json().map_err(|e| ApiError::Parse(e.to_string()))
    }

    // Notes API methods

    /// Get list of theses
    pub fn get_theses(
        &self,
        status: Option<&str>,
        symbol: Option<&str>,
    ) -> Result<Vec<NoteResponse>, ApiError> {
        let mut url = format!("{}/api/theses", self.base_url);
        let mut params = Vec::new();
        if let Some(s) = status {
            params.push(format!("status={}", s));
        }
        if let Some(s) = symbol {
            params.push(format!("symbol={}", s));
        }
        if !params.is_empty() {
            url = format!("{}?{}", url, params.join("&"));
        }

        let response = self
            .client
            .get(&url)
            .send()
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response.json().map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Create a new thesis
    pub fn create_thesis(&self, request: CreateThesisRequest) -> Result<NoteResponse, ApiError> {
        let url = format!("{}/api/theses", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response.json().map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Get list of trades
    pub fn get_trades(
        &self,
        status: Option<&str>,
        symbol: Option<&str>,
    ) -> Result<Vec<NoteResponse>, ApiError> {
        let mut url = format!("{}/api/trades", self.base_url);
        let mut params = Vec::new();
        if let Some(s) = status {
            params.push(format!("status={}", s));
        }
        if let Some(s) = symbol {
            params.push(format!("symbol={}", s));
        }
        if !params.is_empty() {
            url = format!("{}?{}", url, params.join("&"));
        }

        let response = self
            .client
            .get(&url)
            .send()
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response.json().map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Create a new trade
    pub fn create_trade(&self, request: CreateTradeRequest) -> Result<NoteResponse, ApiError> {
        let url = format!("{}/api/trades", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response.json().map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Close a trade
    pub fn close_trade(
        &self,
        name: &str,
        request: CloseTradeRequest,
    ) -> Result<NoteResponse, ApiError> {
        let url = format!("{}/api/trades/{}/close", self.base_url, name);
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response.json().map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Get trade statistics
    pub fn get_trade_stats(&self) -> Result<TradeStatsResponse, ApiError> {
        let url = format!("{}/api/trades/stats", self.base_url);
        let response = self
            .client
            .get(&url)
            .send()
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response.json().map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Search notes
    pub fn search_notes(
        &self,
        query: &str,
        limit: Option<u32>,
    ) -> Result<Vec<SearchResult>, ApiError> {
        let limit = limit.unwrap_or(50);
        let url = format!(
            "{}/api/notes/search?query={}&limit={}",
            self.base_url, query, limit
        );
        let response = self
            .client
            .get(&url)
            .send()
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response.json().map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Get notes graph
    pub fn get_notes_graph(&self) -> Result<GraphResponse, ApiError> {
        let url = format!("{}/api/notes/graph", self.base_url);
        let response = self
            .client
            .get(&url)
            .send()
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response.json().map_err(|e| ApiError::Parse(e.to_string()))
    }

    // Events API methods

    /// Get list of events
    pub fn get_events(
        &self,
        event_type: Option<&str>,
        symbol: Option<&str>,
        company: Option<&str>,
    ) -> Result<Vec<NoteResponse>, ApiError> {
        let mut url = format!("{}/api/events", self.base_url);
        let mut params = Vec::new();
        if let Some(t) = event_type {
            params.push(format!("event_type={}", t));
        }
        if let Some(s) = symbol {
            params.push(format!("symbol={}", s));
        }
        if let Some(c) = company {
            params.push(format!("company={}", c));
        }
        if !params.is_empty() {
            url = format!("{}?{}", url, params.join("&"));
        }

        let response = self
            .client
            .get(&url)
            .send()
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response.json().map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Create a new event
    pub fn create_event(&self, request: CreateEventRequest) -> Result<NoteResponse, ApiError> {
        let url = format!("{}/api/events", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response.json().map_err(|e| ApiError::Parse(e.to_string()))
    }

    // People API methods

    /// Get list of people
    pub fn get_people(
        &self,
        company: Option<&str>,
        role: Option<&str>,
    ) -> Result<Vec<NoteResponse>, ApiError> {
        let mut url = format!("{}/api/people", self.base_url);
        let mut params = Vec::new();
        if let Some(c) = company {
            params.push(format!("company={}", c));
        }
        if let Some(r) = role {
            params.push(format!("role={}", r));
        }
        if !params.is_empty() {
            url = format!("{}?{}", url, params.join("&"));
        }

        let response = self
            .client
            .get(&url)
            .send()
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response.json().map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Create a new person profile
    pub fn create_person(&self, request: CreatePersonRequest) -> Result<NoteResponse, ApiError> {
        let url = format!("{}/api/people", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response.json().map_err(|e| ApiError::Parse(e.to_string()))
    }

    // Sectors API methods

    /// Get list of sectors
    pub fn get_sectors(&self) -> Result<Vec<NoteResponse>, ApiError> {
        let url = format!("{}/api/sectors", self.base_url);
        let response = self
            .client
            .get(&url)
            .send()
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response.json().map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Create a new sector overview
    pub fn create_sector(&self, request: CreateSectorRequest) -> Result<NoteResponse, ApiError> {
        let url = format!("{}/api/sectors", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&request)
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

// Notes API types

#[derive(Debug, Serialize)]
pub struct CreateThesisRequest {
    pub symbol: String,
    pub company_name: Option<String>,
    pub sector: Option<String>,
    pub conviction: String,
}

#[derive(Debug, Serialize)]
pub struct CreateTradeRequest {
    pub symbol: String,
    pub direction: String,
    pub entry_price: f64,
    pub shares: f64,
    pub entry_date: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CloseTradeRequest {
    pub exit_price: f64,
    pub exit_date: Option<String>,
    pub exit_reason: Option<String>,
    pub lessons: Option<String>,
    pub grade: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CreateEventRequest {
    pub symbol: String,
    pub company_name: Option<String>,
    pub event_type: String,
    pub event_date: Option<String>,
    pub host: Option<String>,
    pub participants: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct CreatePersonRequest {
    pub full_name: String,
    pub current_role: Option<String>,
    pub current_company: Option<String>,
    pub linkedin_url: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CreateSectorRequest {
    pub sector_name: String,
    pub sub_sectors: Vec<String>,
    pub companies: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct NoteFrontmatter {
    pub title: String,
    #[serde(rename = "type")]
    pub note_type: String,
    pub created: String,
    pub modified: String,
    pub tags: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct ThesisFrontmatter {
    pub title: String,
    pub symbol: String,
    pub company_name: Option<String>,
    pub sector: Option<String>,
    pub status: String,
    pub conviction: String,
    pub entry_price: Option<f64>,
    pub target_price: Option<f64>,
    pub stop_loss: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct TradeFrontmatter {
    pub title: String,
    pub symbol: String,
    pub direction: String,
    pub status: String,
    pub entry_price: f64,
    pub exit_price: Option<f64>,
    pub shares: f64,
    pub pnl: Option<f64>,
    pub pnl_percent: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct NoteResponse {
    pub name: String,
    pub path: String,
    pub frontmatter: serde_json::Value,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct SearchResult {
    pub path: String,
    pub name: String,
    pub title: String,
    #[serde(rename = "type")]
    pub note_type: String,
    pub snippet: String,
}

#[derive(Debug, Deserialize)]
pub struct TradeStatsResponse {
    pub total_trades: u32,
    pub winners: u32,
    pub losers: u32,
    pub win_rate: f64,
    pub total_pnl: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub profit_factor: f64,
}

#[derive(Debug, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
    #[serde(rename = "type")]
    pub node_type: String,
    pub tags: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct GraphEdge {
    pub source: String,
    pub target: String,
}

#[derive(Debug, Deserialize)]
pub struct GraphResponse {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}
