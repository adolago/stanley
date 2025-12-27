# Stanley REST API Documentation

Stanley provides a comprehensive REST API built with FastAPI for institutional investment analysis.

## Base URL

```
http://localhost:8000/api
```

## Authentication

Stanley implements a comprehensive authentication and authorization system supporting both JWT tokens and API keys.

### JWT Token Authentication (Web Sessions)

JWT tokens are used for web-based authentication with the following characteristics:

- **Access Tokens**: Short-lived (15 minutes by default), stateless
- **Refresh Tokens**: Long-lived (7 days by default), with token rotation
- **Algorithm**: HS256 (configurable via `JWT_ALGORITHM` environment variable)
- **Header**: `Authorization: Bearer <token>`

**Token Claims:**
- `user_id`: User identifier
- `email`: User email
- `roles`: List of user roles
- `exp`: Expiration timestamp
- `iat`: Issued at timestamp
- `jti`: Unique token identifier
- `token_type`: "access" or "refresh"
- `iss`: Issuer (default: "stanley-api")
- `aud`: Audience (default: "stanley-client")

**Creating Tokens:**
```python
from stanley.api.auth import create_token_pair

token_pair = create_token_pair(
    user_id="user_123",
    email="user@example.com",
    roles=["analyst"]
)
# Returns TokenPair with access_token, refresh_token, expires_in
```

### API Key Authentication (Programmatic Access)

API keys are used for programmatic access with the following format:

```
sk_live_<32-char-alphanumeric>  # Production
sk_test_<32-char-alphanumeric>  # Development/Testing
```

**API Key Scopes:**
- `read`: Read-only access
- `write`: Read and write access
- `trade`: Trading operations
- `admin`: Administrative access

**Header**: `X-API-Key: stanley_live_EXAMPLE_KEY_REPLACE_ME_1234`

### Role-Based Access Control (RBAC)

Stanley uses a 6-level role hierarchy:

| Role | Level | Description |
|------|-------|-------------|
| `SUPER_ADMIN` | 6 | Unrestricted access, system configuration |
| `ADMIN` | 5 | User management, API key administration |
| `PORTFOLIO_MANAGER` | 4 | Trader permissions + team management |
| `TRADER` | 3 | Analyst permissions + trading signals, portfolio management |
| `ANALYST` | 2 | Read all analytics + write notes/theses |
| `VIEWER` | 1 | Read-only access to public data |

**Permission Categories:**

Permissions follow the pattern `<domain>:<action>`:

| Permission | Description |
|------------|-------------|
| `market:read` | Access market data |
| `portfolio:read/write` | Portfolio analytics |
| `research:read/write` | Research reports and notes |
| `signals:read/write` | Signal generation |
| `accounting:read` | SEC filings (read-only) |
| `institutional:read` | 13F data (read-only) |
| `options:read/write` | Options analytics |
| `etf:read/write` | ETF analytics |
| `commodities:read/write` | Commodities data |
| `macro:read/write` | Macro indicators |
| `notes:read/write` | Research vault |
| `dark_pool:read` | Dark pool data |
| `admin:read/write` | System settings |

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `JWT_SECRET_KEY` | Yes (production) | - | JWT signing key (min 32 chars) |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | No | 15 | Access token expiry |
| `REFRESH_TOKEN_EXPIRE_DAYS` | No | 7 | Refresh token expiry |
| `JWT_ALGORITHM` | No | HS256 | JWT algorithm |
| `JWT_ISSUER` | No | stanley-api | Token issuer |
| `JWT_AUDIENCE` | No | stanley-client | Token audience |

---

## Rate Limiting

Stanley implements sliding window rate limiting with category-based limits:

| Category | Requests/Minute | Description |
|----------|-----------------|-------------|
| `market_data` | 100 | Market data endpoints |
| `analytics` | 30 | Money flow, dark pool, sector analysis |
| `research` | 20 | Valuation, earnings, peer comparison |
| `accounting` | 10 | SEC EDGAR filings (courtesy limit) |
| `signals` | 50 | Signal generation and backtesting |
| `options` | 30 | Options flow and analytics |
| `etf` | 30 | ETF flows and rotation |
| `macro` | 20 | Economic indicators, regime detection |
| `commodities` | 30 | Commodity data and correlations |
| `portfolio` | 30 | Portfolio analytics |
| `notes` | 50 | Research vault operations |
| `settings` | 30 | User preferences |
| `default` | 60 | Uncategorized endpoints |

**Rate Limit Headers:**

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1704067200
Retry-After: 45  # Only on 429 responses
```

---

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

---

## API Endpoints

### System

#### Health Check
```
GET /api/health
```

Returns the status of all API components.

**Authentication:** None required

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
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

#### Version Info
```
GET /api/version
```

Returns version and build information.

**Response:**
```json
{
  "success": true,
  "data": {
    "version": "2.0.0",
    "api_version": "v1",
    "python_version": "3.11.5",
    "platform": "Linux",
    "build_date": null,
    "git_commit": null
  }
}
```

#### System Status
```
GET /api/status
```

Returns comprehensive system status with metrics.

#### Ping
```
GET /api/ping
```

Simple connectivity test.

**Response:**
```json
{
  "status": "pong",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

---

### Settings

#### Get System Settings (Admin)
```
GET /api/settings
```

**Authentication:** Required (ADMIN role)

Returns system-wide configuration settings.

#### Update System Settings (Admin)
```
PUT /api/settings
```

**Authentication:** Required (ADMIN role)

#### Get User Preferences
```
GET /api/settings/user
```

**Authentication:** Required

#### Update User Preferences
```
PUT /api/settings/user
```

**Authentication:** Required

#### Watchlist Management
```
GET /api/settings/watchlist
PUT /api/settings/watchlist
POST /api/settings/watchlist/{symbol}
DELETE /api/settings/watchlist/{symbol}
```

**Authentication:** Required

#### Alert Management
```
GET /api/settings/alerts
PUT /api/settings/alerts
POST /api/settings/alerts
GET /api/settings/alerts/{alert_id}
PUT /api/settings/alerts/{alert_id}
DELETE /api/settings/alerts/{alert_id}
```

**Authentication:** Required

#### Public Settings Info
```
GET /api/settings/info
```

Returns public configuration information.

---

### Market Data

**Rate Limit:** 100 requests/minute

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

#### Get Real-Time Quote
```
GET /api/market/{symbol}/quote
```

Returns bid/ask, last price, and intraday data.

**Response:**
```json
{
  "success": true,
  "data": {
    "symbol": "AAPL",
    "bid": 185.49,
    "ask": 185.51,
    "bidSize": 100,
    "askSize": 100,
    "last": 185.50,
    "volume": 45000000,
    "open": 183.00,
    "high": 186.00,
    "low": 182.50,
    "close": 185.50,
    "previousClose": 183.15,
    "change": 2.35,
    "changePercent": 1.28,
    "timestamp": "2024-01-15T10:30:00.000Z"
  }
}
```

#### Get Historical Data
```
GET /api/market/{symbol}/history?interval=1d&period=90
```

Returns OHLCV data for the specified period.

**Parameters:**
- `symbol` (path, required): Stock ticker symbol
- `interval` (query, optional): Data interval - `1d`, `1wk`, `1mo` (default: `1d`)
- `period` (query, optional): Days of history, 1-3650 (default: 90)

**Response:**
```json
{
  "success": true,
  "data": {
    "symbol": "AAPL",
    "interval": "1d",
    "dataPoints": [
      {
        "date": "2024-01-15T00:00:00.000Z",
        "open": 183.00,
        "high": 186.00,
        "low": 182.50,
        "close": 185.50,
        "volume": 45000000,
        "adjustedClose": 185.50
      }
    ],
    "startDate": "2024-10-15T00:00:00.000Z",
    "endDate": "2024-01-15T00:00:00.000Z"
  }
}
```

---

### Institutional Holdings

**Rate Limit:** 100 requests/minute

#### Get Institutional Holdings
```
GET /api/institutional/{symbol}
```

Returns 13F institutional holdings data.

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
      "changeFromLastQuarter": null
    }
  ]
}
```

#### Get Ownership Breakdown
```
GET /api/institutional/{symbol}/ownership
```

Returns institutional vs retail ownership metrics.

#### Get 13F Changes
```
GET /api/institutional/{symbol}/changes?conviction_threshold=0.05
```

Enhanced 13F change detection with conviction scoring.

**Parameters:**
- `conviction_threshold` (query, optional): Minimum change to consider significant (default: 0.05)

#### Whale Accumulation Detection
```
GET /api/institutional/{symbol}/whales?min_position_change=0.10&min_aum=1000000000
```

Detect large institutional position changes.

**Parameters:**
- `min_position_change` (query, optional): Minimum position change (default: 0.10)
- `min_aum` (query, optional): Minimum AUM in dollars (default: $1B)

#### Institutional Sentiment Score
```
GET /api/institutional/{symbol}/sentiment
```

Multi-factor institutional sentiment analysis.

#### Position Clusters
```
GET /api/institutional/{symbol}/clusters?n_clusters=4
```

Position clustering analysis for smart money patterns.

#### Cross-Filing Analysis
```
GET /api/institutional/{symbol}/cross-filing?min_filers=3
```

Detect coordinated buying/selling across filers.

#### Smart Money Momentum
```
GET /api/institutional/{symbol}/momentum?window_quarters=4&weight_by_performance=true
```

Track smart money momentum with rolling calculations.

#### Smart Money Flow
```
GET /api/institutional/{symbol}/smart-money-flow
```

Net buying/selling by top-performing managers.

#### New Position Alerts
```
GET /api/institutional/alerts/new-positions?lookback_days=45&min_value=10000000
```

Alert on new institutional positions.

#### Coordinated Buying Alerts
```
GET /api/institutional/alerts/coordinated-buying?min_buyers=3&lookback_days=45
```

Detect stocks being bought by multiple top managers.

#### Conviction Picks
```
GET /api/institutional/conviction-picks?min_weight=0.05&top_n_managers=50
```

High conviction positions across managers.

#### 13F Filing Calendar
```
GET /api/institutional/filing-calendar?quarters_ahead=4
```

Upcoming 13F filing deadlines.

---

### Analytics

**Rate Limit:** 30 requests/minute

#### Analyze Money Flow
```
POST /api/money-flow
```

Analyzes money flow across sectors.

**Request Body:**
```json
{
  "sectors": ["XLK", "XLF", "XLE", "XLV"],
  "lookback_days": 63,
  "period": "1M"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "sectors": {
      "XLK": {
        "symbol": "XLK",
        "netFlow1m": 1500000000,
        "netFlow3m": 4500000000,
        "institutionalChange": 0.15,
        "smartMoneySentiment": 0.72,
        "flowAcceleration": 0.25,
        "confidenceScore": 0.85
      }
    },
    "net_flows": {"XLK": 1500000000},
    "momentum": {"XLK": 0.25},
    "timestamp": "2024-01-15T10:30:00.000Z"
  }
}
```

#### Get Dark Pool Activity
```
GET /api/dark-pool/{symbol}?lookback_days=20
```

Returns dark pool trading activity.

**Response:**
```json
{
  "success": true,
  "data": {
    "symbol": "AAPL",
    "data": [
      {
        "symbol": "AAPL",
        "date": "2024-01-15",
        "darkPoolVolume": 15000000,
        "totalVolume": 45000000,
        "darkPoolPercentage": 0.3333,
        "largeBlockActivity": 0.125,
        "darkPoolSignal": 1
      }
    ],
    "summary": {
      "averageDarkPoolPercentage": 0.35,
      "averageBlockActivity": 0.12,
      "totalDarkPoolVolume": 300000000,
      "signalBias": 5
    },
    "timestamp": "2024-01-15T10:30:00.000Z"
  }
}
```

#### Get Equity Flow
```
GET /api/equity-flow/{symbol}?lookback_days=20
```

Money flow analysis for a specific equity.

#### Get Sector Rotation
```
GET /api/sector-rotation?lookback_days=63
```

Sector rotation analysis and signals.

#### Get Market Breadth
```
GET /api/market-breadth
```

Market breadth indicators.

#### Smart Money Tracking
```
GET /api/smart-money/{symbol}?lookback_days=21
```

Track smart money activity for a symbol.

#### Unusual Volume
```
GET /api/unusual-volume/{symbol}?lookback_days=20
```

Detect unusual volume activity.

#### Flow Momentum
```
GET /api/flow-momentum/{symbol}?lookback_days=21
```

Calculate flow momentum indicators.

#### Comprehensive Analysis
```
GET /api/comprehensive/{symbol}?lookback_days=21
```

All enhanced analytics in a single response.

---

### Portfolio

**Rate Limit:** 30 requests/minute

**Authentication:** Required for all endpoints

#### Analyze Portfolio
```
POST /api/portfolio/analytics
```

Comprehensive portfolio analysis with VaR, beta, sector exposure.

**Request Body:**
```json
{
  "holdings": [
    {"symbol": "AAPL", "shares": 100, "average_cost": 150.00},
    {"symbol": "MSFT", "shares": 50, "average_cost": 300.00},
    {"symbol": "GOOGL", "shares": 25, "average_cost": 140.00}
  ],
  "benchmark": "SPY"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "total_value": 45000.00,
    "total_cost": 40000.00,
    "total_return": 5000.00,
    "total_return_percent": 12.5,
    "beta": 1.15,
    "alpha": 2.5,
    "sharpe_ratio": 1.25,
    "sortino_ratio": 1.45,
    "var_95": 1250.00,
    "var_99": 2100.00,
    "var_95_percent": 2.78,
    "var_99_percent": 4.67,
    "volatility": 18.5,
    "max_drawdown": 8.2,
    "sector_exposure": {
      "Technology": 85.0,
      "Communication": 15.0
    },
    "top_holdings": [...]
  }
}
```

#### Calculate Risk Metrics
```
POST /api/portfolio/risk
```

Comprehensive risk metrics (VaR, CVaR, Sharpe, Sortino).

**Request Body:**
```json
{
  "holdings": [...],
  "confidence_level": 0.95,
  "method": "historical",
  "lookback_days": 252
}
```

#### Performance Attribution
```
POST /api/portfolio/attribution
```

Break down returns by sector and holding.

**Request Body:**
```json
{
  "holdings": [...],
  "period": "1M",
  "benchmark": "SPY"
}
```

#### Optimize Portfolio
```
POST /api/portfolio/optimize
```

Mean-variance portfolio optimization.

**Authentication:** TRADER role or higher required

#### Benchmark Comparison
```
GET /api/portfolio/benchmark/{benchmark}
```

Compare portfolio to a benchmark.

#### Correlation Matrix
```
POST /api/portfolio/correlation
```

Get correlation matrix for portfolio holdings.

#### Sector Exposure
```
POST /api/portfolio/sector-exposure
```

Detailed sector exposure breakdown.

---

### Research

**Rate Limit:** 20 requests/minute

#### Get Research Report
```
GET /api/research/{symbol}
```

Comprehensive research analysis.

**Response:**
```json
{
  "success": true,
  "data": {
    "symbol": "AAPL",
    "companyName": "Apple Inc.",
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "currentPrice": 185.50,
    "marketCap": 2850000000000,
    "valuation": {...},
    "dcf": {...},
    "fairValueRange": {"low": 160, "high": 200},
    "valuationRating": "fairly_valued",
    "earnings": {...},
    "earningsQualityScore": 0.85,
    "revenueGrowth5yr": 12.5,
    "epsGrowth5yr": 15.2,
    "grossMargin": 45.2,
    "operatingMargin": 30.1,
    "netMargin": 25.5,
    "roe": 145.0,
    "roic": 52.0,
    "debtToEquity": 1.95,
    "currentRatio": 0.98,
    "overallScore": 82,
    "strengths": [...],
    "weaknesses": [...],
    "catalysts": [...],
    "risks": [...]
  }
}
```

#### Get Valuation
```
GET /api/valuation/{symbol}?include_dcf=true&method=dcf
```

Valuation analysis with optional DCF model.

**Parameters:**
- `include_dcf` (query, optional): Include DCF analysis (default: true)
- `method` (query, optional): Valuation method - `dcf`, `multiples`, `sum_of_parts`

#### Get Earnings
```
GET /api/earnings/{symbol}?quarters=12&period=quarterly
```

Earnings history, trends, and surprises.

**Parameters:**
- `quarters` (query, optional): Number of quarters (default: 12, max: 40)
- `period` (query, optional): `quarterly`, `annual`, `ttm`

#### Get Peer Comparison
```
GET /api/peers/{symbol}?peers=MSFT,GOOGL,META
```

Relative valuation vs peer group.

**Parameters:**
- `peers` (query, optional): Comma-separated peer symbols (auto-detected if not provided)

#### Get DCF Model
```
GET /api/research/{symbol}/dcf?terminal_growth=0.025&projection_years=5
```

Detailed DCF valuation with sensitivity analysis.

#### Get Trading Multiples
```
GET /api/research/{symbol}/multiples?include_peers=true
```

P/E, EV/EBITDA, P/S, and other valuation multiples.

#### Get Research Summary
```
GET /api/research/{symbol}/summary
```

Quick investment summary with key metrics.

---

### Commodities

**Rate Limit:** 30 requests/minute

#### Get Commodities Overview
```
GET /api/commodities
```

Market overview across all commodity categories.

**Response:**
```json
{
  "success": true,
  "data": {
    "timestamp": "2024-01-15T10:30:00.000Z",
    "sentiment": "bullish",
    "avgChange": 1.25,
    "categories": {
      "energy": {...},
      "precious_metals": {...},
      "base_metals": {...},
      "agriculture": {...},
      "softs": {...}
    }
  }
}
```

#### Get Commodity Detail
```
GET /api/commodities/{symbol}?lookback_days=252
```

Detailed analysis for a specific commodity.

**Parameters:**
- `symbol` (path, required): Commodity symbol (CL, GC, NG, ZC, ZW)
- `lookback_days` (query, optional): Days of history (default: 252)

#### Get Commodity Correlations
```
GET /api/commodities/correlations?commodities=CL,GC,NG&lookback_days=252
```

Correlation matrix for commodities.

#### Get Futures Curve
```
GET /api/commodities/futures-curve/{symbol}?num_contracts=6
```

Futures term structure analysis (contango/backwardation).

#### Get Macro-Commodity Linkage
```
GET /api/commodities/{symbol}/macro?lookback_days=252
```

Analyze commodity-macro relationships.

---

### Options

**Rate Limit:** 30 requests/minute

#### Get Options Flow
```
GET /api/options/{symbol}/flow?lookback_days=5
```

Comprehensive options flow analysis.

**Response:**
```json
{
  "success": true,
  "data": {
    "symbol": "AAPL",
    "totalCallVolume": 125000,
    "totalPutVolume": 85000,
    "totalCallPremium": 15000000,
    "totalPutPremium": 8500000,
    "putCallRatio": 0.68,
    "premiumPutCallRatio": 0.57,
    "netPremiumFlow": 6500000,
    "unusualActivityCount": 12,
    "smartMoneyTrades": 5,
    "sentiment": "bullish",
    "confidence": 0.75,
    "timestamp": "2024-01-15T10:30:00.000Z"
  }
}
```

#### Get Gamma Exposure (GEX)
```
GET /api/options/{symbol}/gamma
```

Aggregate gamma exposure and dealer positioning.

#### Get Unusual Activity
```
GET /api/options/{symbol}/unusual?volume_threshold=2.0&min_premium=50000
```

Detect unusual options activity.

#### Get Put/Call Analysis
```
GET /api/options/{symbol}/put-call
```

Volume and premium-weighted ratios.

#### Get Smart Money Trades
```
GET /api/options/{symbol}/smart-money?min_premium=100000
```

Block trades and sweep orders.

#### Get Max Pain
```
GET /api/options/{symbol}/max-pain?expiration=2024-01-19
```

Max pain calculation by expiration.

#### Get Options Chain
```
GET /api/options/{symbol}/chain?expiration=2024-01-19
```

Full options chain with Greeks.

---

### ETF Analytics

**Rate Limit:** 30 requests/minute

#### Get ETF Flows
```
GET /api/etf/flows?symbols=SPY,QQQ,IWM&lookback_days=90
```

Comprehensive ETF flow analysis.

#### Get ETF Flow Detail
```
GET /api/etf/flows/{symbol}?lookback_days=30
```

Creation/redemption activity for a specific ETF.

#### Get Sector Rotation
```
GET /api/etf/sector-rotation?lookback_days=63
```

Sector ETF rotation signals.

#### Get Sector Heatmap
```
GET /api/etf/sector-heatmap?period=1m
```

Sector performance heatmap data.

**Parameters:**
- `period` (query, optional): `1d`, `1w`, `1m`, `3m`, `ytd`

#### Get Smart Beta Flows
```
GET /api/etf/smart-beta?lookback_days=63
```

Factor ETF flow analysis (value, growth, momentum, etc.).

#### Get Factor Rotation
```
GET /api/etf/factor-rotation
```

Factor rotation signals for tactical allocation.

#### Get Thematic Flows
```
GET /api/etf/thematic?lookback_days=90
```

Thematic ETF analysis (clean energy, AI, crypto, etc.).

#### Get Theme Dashboard
```
GET /api/etf/theme-dashboard
```

Comprehensive thematic overview.

#### Get Institutional ETF Positioning
```
GET /api/etf/institutional?symbols=SPY,QQQ
```

13F institutional holdings in ETFs.

#### Get ETF Overview
```
GET /api/etf/overview
```

Market-wide ETF flow summary.

#### Get ETF Detail
```
GET /api/etf/{symbol}?lookback_days=90
```

Detailed ETF information with flows.

---

### Accounting & SEC Filings

**Rate Limit:** 10 requests/minute (SEC EDGAR courtesy)

#### Get Earnings Quality
```
GET /api/accounting/earnings-quality/{symbol}?manufacturing=false
```

Comprehensive earnings quality analysis including M-Score, F-Score, Z-Score.

**Response:**
```json
{
  "success": true,
  "data": {
    "symbol": "AAPL",
    "overallRating": "good",
    "overallScore": 0.85,
    "mScore": -2.45,
    "mScoreRisk": "low",
    "isLikelyManipulator": false,
    "fScore": 7,
    "fScoreCategory": "strong",
    "zScore": 5.2,
    "zScoreZone": "safe",
    "accrualRatio": 0.05,
    "cashConversion": 1.15,
    "earningsPersistence": 0.92,
    "redFlags": [],
    "timestamp": "2024-01-15T10:30:00.000Z"
  }
}
```

#### Get Beneish M-Score
```
GET /api/accounting/m-score/{symbol}
```

Earnings manipulation detection.

#### Get Piotroski F-Score
```
GET /api/accounting/f-score/{symbol}
```

Financial strength assessment.

#### Get Altman Z-Score
```
GET /api/accounting/z-score/{symbol}?manufacturing=false
```

Bankruptcy risk assessment.

#### Get Red Flags
```
GET /api/accounting/red-flags/{symbol}
```

Comprehensive red flag analysis.

#### Get Red Flag Peer Comparison
```
GET /api/accounting/red-flags/{symbol}/peers?peers=MSFT,GOOGL
```

Compare red flags against peers.

#### Get Anomalies
```
GET /api/accounting/anomalies/{symbol}
```

Detect accounting anomalies including Benford's Law analysis.

#### Get Accrual Analysis
```
GET /api/accounting/accruals/{symbol}
```

Detailed accrual quality analysis.

#### Get Comprehensive Accounting Report
```
GET /api/accounting/comprehensive/{symbol}?manufacturing=false
```

Full accounting quality report.

#### Get Audit Fees
```
GET /api/accounting/audit-fees/{symbol}
```

Audit fee analysis and auditor changes.

#### Get SEC Filings
```
GET /api/accounting/filings/{symbol}?form_type=10-K&limit=20
```

List of SEC filings.

#### Get Financial Statements
```
GET /api/accounting/statements/{symbol}?statement_type=all&periods=4
```

Parsed financial statements.

---

### Macro Analysis

**Rate Limit:** 20 requests/minute

#### Get Economic Indicators
```
GET /api/macro/indicators?country=USA&include_snapshot=true
```

Key economic indicators (GDP, inflation, unemployment).

**Response:**
```json
{
  "success": true,
  "data": {
    "country": "USA",
    "indicators": [
      {
        "code": "GDP_REAL",
        "name": "Real GDP Growth",
        "value": 2.5,
        "unit": "%",
        "frequency": "quarterly",
        "source": "DBnomics"
      }
    ],
    "snapshot": {
      "country": "USA",
      "gdpGrowth": 2.5,
      "inflation": 3.2,
      "unemployment": 3.7,
      "policyRate": 5.25,
      "regime": "expansion"
    },
    "timestamp": "2024-01-15T10:30:00.000Z"
  }
}
```

#### Get Market Regime
```
GET /api/macro/regime?country=USA
```

Multi-signal market regime detection.

**Response:**
```json
{
  "success": true,
  "data": {
    "currentRegime": "goldilocks",
    "confidence": "high",
    "regimeScore": 0.78,
    "components": {...},
    "metrics": {...},
    "risk": {...},
    "positioning": {
      "equity": "overweight",
      "duration": "neutral",
      "credit": "overweight",
      "volatility": "sell"
    },
    "signals": [...],
    "regimeDurationDays": 45,
    "timestamp": "2024-01-15T10:30:00.000Z"
  }
}
```

#### Get Yield Curve Analysis
```
GET /api/macro/yield-curve?country=USA
```

Yield curve shape and recession signals.

#### Get Recession Probability
```
GET /api/macro/recession-probability?country=USA
```

Multi-factor recession risk assessment.

#### Get Fed Watch
```
GET /api/macro/fed-watch
```

Fed meeting probabilities and rate expectations.

#### Get Cross-Asset Correlations
```
GET /api/macro/cross-asset?correlation_window=60&lookback_days=252
```

Stock-bond, stock-commodity, USD correlations.

#### Get Global Overview
```
GET /api/macro/global-overview
```

Economic data across major regions.

#### Compare Countries
```
GET /api/macro/compare-countries?countries=USA,DEU,JPN,GBR,CHN
```

Economic indicator comparison matrix.

---

### Signals

**Rate Limit:** 50 requests/minute

#### Get Signal for Symbol
```
GET /api/signals/{symbol}
```

Generate investment signal for a single symbol.

**Response:**
```json
{
  "success": true,
  "data": {
    "signalId": "sig_abc123",
    "symbol": "AAPL",
    "signalType": "buy",
    "strength": "strong",
    "conviction": 0.82,
    "factors": {
      "money_flow": 0.75,
      "institutional": 0.85,
      "fundamental": 0.80,
      "technical": 0.78,
      "sentiment": 0.70
    },
    "priceAtSignal": 185.50,
    "targetPrice": 210.00,
    "stopLoss": 170.00,
    "holdingPeriodDays": 30,
    "reasoning": "Strong institutional accumulation...",
    "timestamp": "2024-01-15T10:30:00.000Z"
  }
}
```

#### Generate Signals for Multiple Symbols
```
POST /api/signals
```

**Request Body:**
```json
{
  "symbols": ["AAPL", "MSFT", "GOOGL"],
  "min_conviction": 0.3
}
```

#### Get Composite Score
```
GET /api/signals/{symbol}/composite
```

Detailed factor breakdown.

#### Backtest Signals
```
POST /api/signals/backtest
```

**Request Body:**
```json
{
  "symbols": ["AAPL", "MSFT"],
  "start_date": "2023-01-01",
  "end_date": "2024-01-01",
  "holding_period_days": 30,
  "initial_capital": 100000,
  "position_size_pct": 0.10
}
```

#### Quick Backtest
```
GET /api/signals/backtest/quick/{symbol}?days=90
```

Quick single-symbol backtest.

#### Get Performance Stats
```
GET /api/signals/performance/stats
```

Aggregate performance statistics.

#### Get Signal History
```
GET /api/signals/performance/history?symbol=AAPL&limit=100
```

Historical signal outcomes.

#### Record Signal Outcome
```
POST /api/signals/{signal_id}/outcome?exit_price=195.00&exit_reason=target
```

Record actual signal performance.

#### Configure Signals
```
POST /api/signals/configure
```

Customize signal generation parameters.

#### Get Signal Configuration
```
GET /api/signals/configure
```

Current signal generation settings.

#### List Signal Factors
```
GET /api/signals/factors
```

Available signal factors and weights.

#### Get Factor Details
```
GET /api/signals/factors/{factor_name}
```

Detailed factor information.

---

### Notes (Research Vault)

**Rate Limit:** 50 requests/minute

#### List Notes
```
GET /api/notes
```

Returns list of all research notes.

#### Search Notes
```
GET /api/notes/search?q=apple
```

Full-text search across notes.

#### Get Knowledge Graph
```
GET /api/notes/graph
```

Note relationship graph for visualization.

#### CRUD Operations
```
GET /api/notes/{name}
PUT /api/notes/{name}
DELETE /api/notes/{name}
```

Individual note management.

#### Get Backlinks
```
GET /api/notes/{name}/backlinks
```

Notes that link to this note.

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

#### People/Executives
```
GET /api/people
POST /api/people
```

Key person tracking.

#### Sectors
```
GET /api/sectors
POST /api/sectors
```

Sector analysis notes.

#### Daily Notes
```
POST /api/daily
```

Create daily research note.

#### Company Notes
```
POST /api/companies
```

Create company research note.

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
| 401 | Unauthorized - Authentication required |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource not found |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Component not initialized |

---

## OpenAPI Documentation

When running the API server, interactive documentation is available at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Running the API

```bash
# Using Python module
python -m stanley.api.main

# Using uvicorn
uvicorn stanley.api.main:app --reload --port 8000

# Production
uvicorn stanley.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Request Examples

### cURL Examples

**Get market data:**
```bash
curl -X GET "http://localhost:8000/api/market/AAPL"
```

**Authenticated request with JWT:**
```bash
curl -X GET "http://localhost:8000/api/portfolio/analytics" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..."
```

**Authenticated request with API key:**
```bash
curl -X GET "http://localhost:8000/api/portfolio/analytics" \
  -H "X-API-Key: stanley_live_EXAMPLE_KEY_REPLACE_ME_1234"
```

**POST request with JSON body:**
```bash
curl -X POST "http://localhost:8000/api/money-flow" \
  -H "Content-Type: application/json" \
  -d '{"sectors": ["XLK", "XLF", "XLE"], "lookback_days": 63}'
```

### Python Examples

```python
import requests

# Get market data
response = requests.get("http://localhost:8000/api/market/AAPL")
data = response.json()

# Authenticated request
headers = {"Authorization": "Bearer <token>"}
response = requests.post(
    "http://localhost:8000/api/portfolio/analytics",
    headers=headers,
    json={
        "holdings": [
            {"symbol": "AAPL", "shares": 100, "average_cost": 150.00}
        ],
        "benchmark": "SPY"
    }
)
```
