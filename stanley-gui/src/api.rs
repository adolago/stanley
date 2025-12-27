//! API client for communicating with Stanley Python backend
//!
//! Provides async methods for fetching money flow data, institutional
//! holdings, and other analytics from the Stanley backend service.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// API client for Stanley backend (async version)
#[derive(Clone)]
pub struct StanleyClient {
    base_url: String,
    client: reqwest::Client,
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
            client: reqwest::Client::new(),
        }
    }

    /// Create a new client wrapped in Arc for sharing across async tasks
    pub fn new_shared() -> Arc<Self> {
        Arc::new(Self::new())
    }

    /// Get sector money flow analysis
    pub async fn get_sector_money_flow(
        &self,
        sectors: Vec<String>,
    ) -> Result<SectorFlowResponse, ApiError> {
        let url = format!("{}/api/money-flow", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&SectorFlowRequest { sectors })
            .send()
            .await
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response
            .json()
            .await
            .map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Get institutional holdings for a symbol
    pub async fn get_institutional_holdings(
        &self,
        symbol: &str,
    ) -> Result<InstitutionalHoldingsResponse, ApiError> {
        let url = format!("{}/api/institutional/{}", self.base_url, symbol);
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response
            .json()
            .await
            .map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Get equity money flow analysis
    pub async fn get_equity_flow(&self, symbol: &str) -> Result<EquityFlowResponse, ApiError> {
        let url = format!("{}/api/equity-flow/{}", self.base_url, symbol);
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response
            .json()
            .await
            .map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Get dark pool activity
    pub async fn get_dark_pool_activity(&self, symbol: &str) -> Result<DarkPoolResponse, ApiError> {
        let url = format!("{}/api/dark-pool/{}", self.base_url, symbol);
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response
            .json()
            .await
            .map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Health check
    pub async fn health_check(&self) -> Result<HealthResponse, ApiError> {
        let url = format!("{}/api/health", self.base_url);
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response
            .json()
            .await
            .map_err(|e| ApiError::Parse(e.to_string()))
    }

    // Notes API methods

    /// Get list of theses
    pub async fn get_theses(
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
            .await
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response
            .json()
            .await
            .map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Create a new thesis
    pub async fn create_thesis(
        &self,
        request: CreateThesisRequest,
    ) -> Result<NoteResponse, ApiError> {
        let url = format!("{}/api/theses", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response
            .json()
            .await
            .map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Get list of trades
    pub async fn get_trades(
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
            .await
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response
            .json()
            .await
            .map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Create a new trade
    pub async fn create_trade(
        &self,
        request: CreateTradeRequest,
    ) -> Result<NoteResponse, ApiError> {
        let url = format!("{}/api/trades", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response
            .json()
            .await
            .map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Close a trade
    pub async fn close_trade(
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
            .await
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response
            .json()
            .await
            .map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Get trade statistics
    pub async fn get_trade_stats(&self) -> Result<TradeStatsResponse, ApiError> {
        let url = format!("{}/api/trades/stats", self.base_url);
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response
            .json()
            .await
            .map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Search notes
    pub async fn search_notes(
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
            .await
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response
            .json()
            .await
            .map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Get notes graph
    pub async fn get_notes_graph(&self) -> Result<GraphResponse, ApiError> {
        let url = format!("{}/api/notes/graph", self.base_url);
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response
            .json()
            .await
            .map_err(|e| ApiError::Parse(e.to_string()))
    }

    // Events API methods

    /// Get list of events
    pub async fn get_events(
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
            .await
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response
            .json()
            .await
            .map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Create a new event
    pub async fn create_event(
        &self,
        request: CreateEventRequest,
    ) -> Result<NoteResponse, ApiError> {
        let url = format!("{}/api/events", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response
            .json()
            .await
            .map_err(|e| ApiError::Parse(e.to_string()))
    }

    // People API methods

    /// Get list of people
    pub async fn get_people(
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
            .await
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response
            .json()
            .await
            .map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Create a new person profile
    pub async fn create_person(
        &self,
        request: CreatePersonRequest,
    ) -> Result<NoteResponse, ApiError> {
        let url = format!("{}/api/people", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response
            .json()
            .await
            .map_err(|e| ApiError::Parse(e.to_string()))
    }

    // Sectors API methods

    /// Get list of sectors
    pub async fn get_sectors(&self) -> Result<Vec<NoteResponse>, ApiError> {
        let url = format!("{}/api/sectors", self.base_url);
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response
            .json()
            .await
            .map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Create a new sector overview
    pub async fn create_sector(
        &self,
        request: CreateSectorRequest,
    ) -> Result<NoteResponse, ApiError> {
        let url = format!("{}/api/sectors", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response
            .json()
            .await
            .map_err(|e| ApiError::Parse(e.to_string()))
    }

    // Portfolio API methods

    /// Get portfolio analytics (VaR, sector exposure, top holdings)
    pub async fn get_portfolio_analytics(
        &self,
        holdings: Vec<PortfolioHolding>,
        benchmark: Option<&str>,
    ) -> Result<ApiResponse<PortfolioAnalytics>, ApiError> {
        let url = format!("{}/api/portfolio-analytics", self.base_url);
        let request = PortfolioRequest {
            holdings,
            benchmark: benchmark.unwrap_or("SPY").to_string(),
        };
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response
            .json()
            .await
            .map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Get portfolio risk metrics
    pub async fn get_portfolio_risk(
        &self,
        holdings: Vec<PortfolioHolding>,
        confidence_level: Option<f64>,
        method: Option<&str>,
    ) -> Result<ApiResponse<ApiRiskMetrics>, ApiError> {
        let url = format!("{}/api/portfolio/risk", self.base_url);
        let request = RiskRequest {
            holdings,
            confidence_level: confidence_level.unwrap_or(0.95),
            method: method.unwrap_or("historical").to_string(),
            lookback_days: 252,
        };
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response
            .json()
            .await
            .map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Get sector exposure for portfolio
    pub async fn get_sector_exposure(
        &self,
        holdings: Vec<PortfolioHolding>,
    ) -> Result<ApiResponse<SectorExposureResponse>, ApiError> {
        let url = format!("{}/api/portfolio/sector-exposure", self.base_url);
        let request = PortfolioRequest {
            holdings,
            benchmark: "SPY".to_string(),
        };
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| ApiError::Network(e.to_string()))?;

        response
            .json()
            .await
            .map_err(|e| ApiError::Parse(e.to_string()))
    }

    // =========================================================================
    // COMMODITIES ENDPOINTS
    // =========================================================================

    /// Get commodities market overview
    pub async fn get_commodities_overview(&self) -> Result<ApiResponse<CommoditiesOverview>, ApiError> {
        let url = format!("{}/api/commodities", self.base_url);
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ApiError::Network(e.to_string()))?;
        response
            .json()
            .await
            .map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Get commodity detail by symbol
    pub async fn get_commodity_detail(&self, symbol: &str) -> Result<ApiResponse<CommoditySummary>, ApiError> {
        let url = format!("{}/api/commodities/{}", self.base_url, symbol);
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ApiError::Network(e.to_string()))?;
        response
            .json()
            .await
            .map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Get macro-commodity linkage analysis
    pub async fn get_commodity_macro(&self, symbol: &str) -> Result<ApiResponse<MacroAnalysis>, ApiError> {
        let url = format!("{}/api/commodities/{}/macro", self.base_url, symbol);
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ApiError::Network(e.to_string()))?;
        response
            .json()
            .await
            .map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Get commodity correlation matrix
    pub async fn get_commodities_correlations(&self) -> Result<ApiResponse<CorrelationMatrix>, ApiError> {
        let url = format!("{}/api/commodities/correlations", self.base_url);
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ApiError::Network(e.to_string()))?;
        response
            .json()
            .await
            .map_err(|e| ApiError::Parse(e.to_string()))
    }

    // =========================================================================
    // DASHBOARD DATA ENDPOINTS
    // =========================================================================

    /// Get money flow analysis for sectors
    pub async fn get_money_flow(&self) -> Result<ApiResponse<Vec<SectorFlow>>, ApiError> {
        let url = format!("{}/api/money-flow", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&SectorFlowRequest {
                sectors: vec![
                    "XLK".to_string(),
                    "XLF".to_string(),
                    "XLE".to_string(),
                    "XLV".to_string(),
                    "XLI".to_string(),
                ],
            })
            .send()
            .await
            .map_err(|e| ApiError::Network(e.to_string()))?;
        response
            .json()
            .await
            .map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Get market data for a symbol
    pub async fn get_market_data(&self, symbol: &str) -> Result<ApiResponse<MarketData>, ApiError> {
        let url = format!("{}/api/market/{}", self.base_url, symbol);
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ApiError::Network(e.to_string()))?;
        response
            .json()
            .await
            .map_err(|e| ApiError::Parse(e.to_string()))
    }

    /// Get institutional holders for a symbol
    pub async fn get_institutional(
        &self,
        symbol: &str,
    ) -> Result<ApiResponse<Vec<InstitutionalHolder>>, ApiError> {
        let url = format!("{}/api/institutional/{}", self.base_url, symbol);
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ApiError::Network(e.to_string()))?;
        response
            .json()
            .await
            .map_err(|e| ApiError::Parse(e.to_string()))
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

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiError::Network(msg) => write!(f, "Network error: {}", msg),
            ApiError::Parse(msg) => write!(f, "Parse error: {}", msg),
            ApiError::Server(msg) => write!(f, "Server error: {}", msg),
        }
    }
}

// Request/Response types

#[derive(Debug, Serialize)]
pub struct SectorFlowRequest {
    pub sectors: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SectorFlowResponse {
    pub sectors: Vec<SectorFlow>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SectorFlow {
    pub sector: String,
    pub net_flow_1m: f64,
    pub net_flow_3m: f64,
    pub institutional_change: f64,
    pub smart_money_sentiment: f64,
    pub confidence_score: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InstitutionalHoldingsResponse {
    pub symbol: String,
    pub institutional_ownership: f64,
    pub number_of_institutions: u32,
    pub top_holders: Vec<InstitutionalHolder>,
    pub ownership_trend: f64,
    pub concentration_risk: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InstitutionalHolder {
    pub manager_name: String,
    pub value_held: f64,
    pub ownership_percentage: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EquityFlowResponse {
    pub symbol: String,
    pub money_flow_score: f64,
    pub institutional_sentiment: f64,
    pub smart_money_activity: f64,
    pub short_pressure: f64,
    pub accumulation_distribution: f64,
    pub confidence: f64,
}

/// Alias for equity flow data used in comparison views
pub type EquityFlowData = EquityFlowResponse;

/// Market data for a symbol (prices, volume, change)
#[derive(Debug, Clone, Deserialize, Default)]
pub struct MarketData {
    pub symbol: String,
    pub price: f64,
    pub change: f64,
    pub change_percent: f64,
    pub volume: u64,
    pub avg_volume: u64,
    pub market_cap: f64,
    pub pe_ratio: Option<f64>,
    pub dividend_yield: Option<f64>,
    pub high_52w: f64,
    pub low_52w: f64,
}

/// Valuation data for research analysis
#[derive(Debug, Clone, Deserialize, Default)]
pub struct ValuationData {
    pub pe_ratio: f64,
    pub forward_pe: f64,
    pub peg_ratio: f64,
    pub price_to_sales: f64,
    pub pb_ratio: f64,
    pub ps_ratio: f64,
    pub ev_ebitda: f64,
    pub dcf_value: f64,
}

/// Research data for fundamental analysis
#[derive(Debug, Clone, Deserialize, Default)]
pub struct ResearchData {
    pub symbol: String,
    pub company_name: String,
    pub sector: String,
    pub industry: String,
    pub analyst_rating: Option<String>,
    pub price_target: Option<f64>,
    pub eps_estimate: Option<f64>,
    pub revenue_estimate: Option<f64>,
    pub valuation: Option<ValuationData>,
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

// Portfolio API types

/// Portfolio holding for API requests
#[derive(Debug, Clone, Serialize)]
pub struct PortfolioHolding {
    pub symbol: String,
    pub shares: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub average_cost: Option<f64>,
}

/// Request for portfolio analytics
#[derive(Debug, Clone, Serialize)]
pub struct PortfolioRequest {
    pub holdings: Vec<PortfolioHolding>,
    pub benchmark: String,
}

/// Request for portfolio risk analysis
#[derive(Debug, Clone, Serialize)]
pub struct RiskRequest {
    pub holdings: Vec<PortfolioHolding>,
    pub confidence_level: f64,
    pub method: String,
    pub lookback_days: i32,
}

/// Portfolio analytics response data
#[derive(Debug, Clone, Deserialize)]
pub struct PortfolioAnalytics {
    pub total_value: f64,
    pub sector_exposure: std::collections::HashMap<String, f64>,
    pub top_holdings: Vec<HoldingInfo>,
}

/// Individual holding info from API
#[derive(Debug, Clone, Deserialize)]
pub struct HoldingInfo {
    pub symbol: String,
    pub weight: f64,
    pub value: f64,
    pub return_pct: Option<f64>,
}

/// Risk metrics from API
#[derive(Debug, Clone, Deserialize)]
pub struct ApiRiskMetrics {
    pub var_95: f64,
    pub var_99: f64,
    pub cvar_95: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub beta: f64,
}

/// Sector exposure response
#[derive(Debug, Clone, Deserialize)]
pub struct SectorExposureResponse {
    pub portfolio_weights: std::collections::HashMap<String, f64>,
}

/// Generic API response wrapper
#[derive(Debug, Clone, Deserialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
    pub timestamp: String,
}

// Commodities API types

/// Commodity price data from API
#[derive(Debug, Deserialize, Clone)]
pub struct CommodityPrice {
    pub symbol: String,
    pub name: String,
    pub price: f64,
    pub change: f64,
    #[serde(rename = "changePercent")]
    pub change_percent: f64,
    pub high: f64,
    pub low: f64,
    pub volume: i64,
    #[serde(rename = "openInterest")]
    pub open_interest: i64,
    pub timestamp: String,
}

/// Commodity summary with analytics
#[derive(Debug, Deserialize, Clone)]
pub struct CommoditySummary {
    pub symbol: String,
    pub name: String,
    pub category: String,
    pub price: f64,
    #[serde(rename = "change1d")]
    pub change_1d: f64,
    #[serde(rename = "change1w")]
    pub change_1w: f64,
    #[serde(rename = "change1m")]
    pub change_1m: f64,
    #[serde(rename = "changeYtd")]
    pub change_ytd: f64,
    #[serde(rename = "volatility30d")]
    pub volatility_30d: f64,
    pub trend: String,
    #[serde(rename = "relativeStrength")]
    pub relative_strength: f64,
}

/// Category overview data
#[derive(Debug, Deserialize, Clone)]
pub struct CategoryOverview {
    pub category: String,
    pub count: i32,
    #[serde(rename = "avgChange")]
    pub avg_change: f64,
    pub leader: Option<CommodityPrice>,
    pub laggard: Option<CommodityPrice>,
    pub commodities: Vec<CommodityPrice>,
}

/// Market overview response
#[derive(Debug, Deserialize, Clone)]
pub struct CommoditiesOverview {
    pub timestamp: String,
    pub sentiment: String,
    #[serde(rename = "avgChange")]
    pub avg_change: f64,
    pub categories: std::collections::HashMap<String, CategoryOverview>,
}

/// Macro linkage data
#[derive(Debug, Deserialize, Clone)]
pub struct MacroLinkage {
    pub commodity: String,
    #[serde(rename = "macroIndicator")]
    pub macro_indicator: String,
    pub correlation: f64,
    #[serde(rename = "leadLagDays")]
    pub lead_lag_days: i32,
    pub relationship: String,
    pub strength: String,
}

/// Macro linkage analysis response
#[derive(Debug, Deserialize, Clone)]
pub struct MacroAnalysis {
    pub commodity: String,
    pub name: String,
    pub category: String,
    pub linkages: Vec<MacroLinkage>,
    #[serde(rename = "primaryDriver")]
    pub primary_driver: Option<String>,
}

/// Correlation matrix data
#[derive(Debug, Deserialize, Clone)]
pub struct CorrelationMatrix {
    pub symbols: Vec<String>,
    pub matrix: Vec<Vec<f64>>,
}
