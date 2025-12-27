# Stanley REST API Documentation

Stanley provides a comprehensive REST API built with FastAPI for institutional investment analysis.

## Base URL

```
http://localhost:8000/api
```

## Authentication

Currently, the API does not require authentication. API key support is planned for future releases.

## Response Format

All endpoints return a standardized response format:

```json
{
  "success": true,
  "data": { ... },
  "error": null,
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

## Endpoints

### System

#### Health Check
```
GET /api/health
```

Returns the status of all API components.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "components": {
    "api": true,
    "data_manager": true,
    "money_flow_analyzer": true,
    "institutional_analyzer": true,
    "portfolio_analyzer": true,
    "research_analyzer": true,
    "commodities_analyzer": true,
    "options_analyzer": true,
    "etf_analyzer": true,
    "accounting_analyzer": true,
    "signal_generator": true
  },
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

---

### Market Data

#### Get Market Data
```
GET /api/market/{symbol}
```

Returns current market data for a symbol.

**Parameters:**
- `symbol` (path, required): Stock ticker symbol (e.g., AAPL, MSFT)

**Response:**
```json
{
  "success": true,
  "data": {
    "symbol": "AAPL",
    "price": 185.50,
    "change": 2.35,
    "changePercent": 1.28,
    "volume": 45000000,
    "marketCap": null,
    "timestamp": "2024-01-15T10:30:00.000Z"
  }
}
```

---

### Institutional Holdings

#### Get Institutional Holdings
```
GET /api/institutional/{symbol}
```

Returns 13F institutional holdings data.

**Parameters:**
- `symbol` (path, required): Stock ticker symbol

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "managerName": "Vanguard Group Inc",
      "managerCik": "0000102909",
      "sharesHeld": 1250000000,
      "valueHeld": 231250000000,
      "ownershipPercentage": 8.5,
      "changeFromLastQuarter": 0.5
    }
  ]
}
```

---

### Analytics

#### Analyze Money Flow
```
POST /api/money-flow
```

Analyzes money flow across sectors.

**Request Body:**
```json
{
  "sectors": ["XLK", "XLF", "XLE", "XLV"],
  "lookback_days": 63
}
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "symbol": "XLK",
      "netFlow1m": 1500000000,
      "netFlow3m": 4500000000,
      "institutionalChange": 0.15,
      "smartMoneySentiment": 0.72,
      "flowAcceleration": 0.25,
      "confidenceScore": 0.85
    }
  ]
}
```

#### Get Dark Pool Activity
```
GET /api/dark-pool/{symbol}?lookback_days=20
```

Returns dark pool trading activity.

**Parameters:**
- `symbol` (path, required): Stock ticker symbol
- `lookback_days` (query, optional): Number of days (default: 20)

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "symbol": "AAPL",
      "date": "2024-01-15",
      "darkPoolVolume": 15000000,
      "totalVolume": 45000000,
      "darkPoolPercentage": 33.33,
      "largeBlockActivity": 12.5,
      "signal": "bullish"
    }
  ]
}
```

#### Get Equity Flow
```
GET /api/equity-flow/{symbol}?lookback_days=20
```

Returns money flow analysis for an equity.

**Response:**
```json
{
  "success": true,
  "data": {
    "symbol": "AAPL",
    "moneyFlowScore": 0.65,
    "institutionalSentiment": 0.72,
    "smartMoneyActivity": 0.58,
    "shortPressure": -0.15,
    "accumulationDistribution": 0.45,
    "confidence": 0.82
  }
}
```

---

### Portfolio

#### Analyze Portfolio
```
POST /api/portfolio-analytics
```

Analyzes portfolio holdings with risk metrics.

**Request Body:**
```json
{
  "holdings": [
    {"symbol": "AAPL", "shares": 100, "average_cost": 150.00},
    {"symbol": "MSFT", "shares": 50, "average_cost": 300.00},
    {"symbol": "GOOGL", "shares": 25, "average_cost": 140.00}
  ]
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "totalValue": 45000.00,
    "totalReturn": 5000.00,
    "totalReturnPercent": 12.5,
    "beta": 1.15,
    "var95": -1250.00,
    "var99": -2100.00,
    "sectorExposure": {
      "Technology": 0.85,
      "Communication Services": 0.15
    },
    "topHoldings": [...]
  }
}
```

---

### Research

#### Get Research Report
```
GET /api/research/{symbol}
```

Returns comprehensive research analysis.

**Response:**
```json
{
  "success": true,
  "data": {
    "symbol": "AAPL",
    "valuation": {...},
    "earnings": {...},
    "fundamentals": {...},
    "peerComparison": {...}
  }
}
```

#### Get Valuation
```
GET /api/valuation/{symbol}?include_dcf=true
```

Returns valuation analysis with optional DCF model.

#### Get Earnings
```
GET /api/earnings/{symbol}?quarters=12
```

Returns earnings analysis.

#### Get Peer Comparison
```
GET /api/peers/{symbol}
```

Returns peer comparison analysis.

---

### Commodities

#### Get Commodities Overview
```
GET /api/commodities
```

Returns commodity market overview.

#### Get Commodity Detail
```
GET /api/commodities/{symbol}
```

Returns detailed commodity analysis.

**Parameters:**
- `symbol` (path, required): Commodity symbol (e.g., CL, GC, NG)

#### Get Macro-Commodity Linkage
```
GET /api/commodities/{symbol}/macro
```

Returns macro-commodity correlation analysis.

#### Get Commodity Correlations
```
GET /api/commodities/correlations
```

Returns commodity correlation matrix.

---

### Options

#### Get Options Flow
```
GET /api/options/{symbol}/flow
```

Returns options flow analysis.

#### Get Gamma Exposure
```
GET /api/options/{symbol}/gamma
```

Returns gamma exposure analysis.

#### Get Unusual Activity
```
GET /api/options/{symbol}/unusual
```

Returns unusual options activity.

#### Get Put/Call Ratio
```
GET /api/options/{symbol}/put-call
```

Returns put/call ratio analysis.

#### Get Max Pain
```
GET /api/options/{symbol}/max-pain
```

Returns max pain calculation.

---

### ETF Analytics

#### Get ETF Flows
```
GET /api/etf/flows
```

Returns aggregate ETF fund flows.

#### Get Individual ETF Flows
```
GET /api/etf/flows/{symbol}
```

Returns flows for a specific ETF.

#### Get Sector Rotation
```
GET /api/etf/sector-rotation
```

Returns sector rotation signals.

#### Get Sector Heatmap
```
GET /api/etf/sector-heatmap
```

Returns sector performance heatmap data.

#### Get Smart Beta Analysis
```
GET /api/etf/smart-beta
```

Returns smart beta factor analysis.

#### Get Factor Rotation
```
GET /api/etf/factor-rotation
```

Returns factor rotation signals.

#### Get Thematic ETFs
```
GET /api/etf/thematic
```

Returns thematic ETF analysis.

#### Get Institutional ETF Flows
```
GET /api/etf/institutional
```

Returns institutional ETF flows.

#### Get ETF Overview
```
GET /api/etf/overview
```

Returns ETF market overview.

---

### Accounting & SEC Filings

#### Get SEC Filings
```
GET /api/accounting/{symbol}/filings
```

Returns list of SEC filings.

#### Get Financial Statements
```
GET /api/accounting/{symbol}/statements
```

Returns parsed financial statements.

#### Get Earnings Quality
```
GET /api/accounting/{symbol}/earnings-quality
```

Returns earnings quality score.

**Response:**
```json
{
  "success": true,
  "data": {
    "symbol": "AAPL",
    "overallScore": 0.85,
    "accrualQuality": 0.82,
    "revenueQuality": 0.88,
    "cashFlowQuality": 0.90,
    "signals": ["Strong cash conversion", "Low accruals ratio"]
  }
}
```

#### Get Red Flags
```
GET /api/accounting/{symbol}/red-flags
```

Returns accounting red flags.

#### Get Anomalies
```
GET /api/accounting/{symbol}/anomalies
```

Returns detected accounting anomalies.

#### Get Footnotes
```
GET /api/accounting/{symbol}/footnotes
```

Returns key financial statement footnotes.

---

### Macro Analysis

#### Get Economic Indicators
```
GET /api/macro/indicators
```

Returns key economic indicators.

#### Get Market Regime
```
GET /api/macro/regime
```

Returns current market regime.

**Response:**
```json
{
  "success": true,
  "data": {
    "regime": "expansion",
    "confidence": 0.78,
    "volatilityRegime": "low",
    "yieldCurveSignal": "steepening",
    "creditConditions": "tightening"
  }
}
```

#### Get Yield Curve Analysis
```
GET /api/macro/yield-curve
```

Returns yield curve analysis.

#### Get Recession Probability
```
GET /api/macro/recession-probability
```

Returns recession probability model output.

#### Get Credit Spreads
```
GET /api/macro/credit-spreads
```

Returns credit spread analysis.

#### Get Business Cycle
```
GET /api/macro/business-cycle
```

Returns business cycle phase.

#### Get Volatility Regime
```
GET /api/macro/volatility-regime
```

Returns volatility regime.

#### Get Cross-Asset Analysis
```
GET /api/macro/cross-asset
```

Returns cross-asset correlation analysis.

---

### Signals

#### Get Signals for Symbol
```
GET /api/signals/{symbol}
```

Returns generated signals for a symbol.

#### Generate Signals
```
POST /api/signals
```

Generates new signals.

**Request Body:**
```json
{
  "symbols": ["AAPL", "MSFT", "GOOGL"],
  "signal_types": ["momentum", "value", "quality"],
  "lookback_days": 252
}
```

#### Get Backtest Results
```
GET /api/signals/backtest?strategy=momentum&period=1y
```

Returns signal backtest results.

#### Get Performance Stats
```
GET /api/signals/performance/stats
```

Returns signal performance statistics.

---

### Notes (Research Vault)

#### List Notes
```
GET /api/notes
```

Returns list of all research notes.

#### Search Notes
```
GET /api/notes/search?q=apple
```

Searches notes by content.

#### Get Knowledge Graph
```
GET /api/notes/graph
```

Returns note relationship graph.

#### CRUD Operations
```
GET /api/notes/{name}
PUT /api/notes/{name}
DELETE /api/notes/{name}
```

Note management operations.

#### Investment Theses
```
GET /api/theses
POST /api/theses
```

Investment thesis management.

#### Trade Journal
```
GET /api/trades
POST /api/trades
POST /api/trades/{name}/close
GET /api/trades/stats
```

Trade journal operations.

#### Market Events
```
GET /api/events
POST /api/events
```

Market event tracking.

---

## Error Handling

Errors are returned with `success: false` and an error message:

```json
{
  "success": false,
  "data": null,
  "error": "Symbol not found: INVALID",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters |
| 404 | Not Found - Resource not found |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Component not initialized |

## Rate Limiting

Currently no rate limiting is implemented. This is planned for future releases.

## OpenAPI Documentation

When running the API server, interactive documentation is available at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Running the API

```bash
# Using Python module
python -m stanley.api.main

# Using uvicorn
uvicorn stanley.api.main:app --reload --port 8000

# Production
uvicorn stanley.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```
