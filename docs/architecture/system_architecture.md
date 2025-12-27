# Stanley System Architecture

## Executive Summary

Stanley is an institutional investment analysis platform built with a hybrid Python/Rust architecture. The system consists of a FastAPI-based Python backend providing analytics and data services, and a GPUI-based Rust GUI for high-performance desktop visualization.

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Component Overview](#component-overview)
3. [Python Backend Architecture](#python-backend-architecture)
4. [Rust GUI Architecture](#rust-gui-architecture)
5. [Authentication and Authorization](#authentication-and-authorization)
6. [Data Flow](#data-flow)
7. [API Endpoint Structure](#api-endpoint-structure)
8. [Security Architecture](#security-architecture)

---

## High-Level Architecture

```
+------------------------------------------------------------------+
|                       STANLEY PLATFORM                            |
+------------------------------------------------------------------+
|                                                                   |
|  +------------------------+        +---------------------------+  |
|  |     RUST GUI           |  HTTP  |     PYTHON BACKEND        |  |
|  |     (stanley-gui)      | <----> |     (FastAPI)             |  |
|  +------------------------+        +---------------------------+  |
|  | GPUI Framework         |        | REST API                  |  |
|  | - Dashboard View       |        | - /api/market/*           |  |
|  | - Portfolio View       |        | - /api/portfolio/*        |  |
|  | - Commodities View     |        | - /api/research/*         |  |
|  | - Notes View           |        | - /api/accounting/*       |  |
|  | - Research View        |        | - /api/signals/*          |  |
|  +------------------------+        +---------------------------+  |
|            |                                   |                  |
|            v                                   v                  |
|  +------------------------+        +---------------------------+  |
|  | Local State            |        | Analyzer Modules          |  |
|  | - LoadingState<T>      |        | - MoneyFlowAnalyzer       |  |
|  | - StanleyClient        |        | - InstitutionalAnalyzer   |  |
|  | - Theme Configuration  |        | - PortfolioAnalyzer       |  |
|  +------------------------+        | - ResearchAnalyzer        |  |
|                                    | - AccountingAnalyzer      |  |
|                                    | - CommoditiesAnalyzer     |  |
|                                    | - MacroAnalyzer           |  |
|                                    +---------------------------+  |
|                                                |                  |
|                                                v                  |
|                                    +---------------------------+  |
|                                    | Data Layer                |  |
|                                    +---------------------------+  |
|                                    | DataManager               |  |
|                                    | - OpenBB Adapter          |  |
|                                    | - Edgar Adapter           |  |
|                                    | - DBnomics Adapter        |  |
|                                    +---------------------------+  |
|                                                                   |
+------------------------------------------------------------------+
```

---

## Component Overview

### Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Desktop GUI | Rust + GPUI (Zed Framework) | High-performance native UI |
| Web API | Python + FastAPI | REST API, analytics engine |
| Data Access | OpenBB, edgartools, DBnomics | Market data, SEC filings, macro data |
| Authentication | JWT + API Keys | Token-based auth with RBAC |
| Rate Limiting | Sliding Window Algorithm | Request throttling by category |

### Python Module Structure

```
stanley/
+-- api/                    # FastAPI REST API
|   +-- auth/               # Authentication layer
|   |   +-- dependencies.py # FastAPI auth dependencies
|   |   +-- jwt.py          # JWT token management
|   |   +-- api_keys.py     # API key management
|   |   +-- rate_limit.py   # Rate limiting middleware
|   |   +-- rbac.py         # Role-based access control
|   +-- routers/            # API route handlers
|   |   +-- market.py       # Market data endpoints
|   |   +-- institutional.py# 13F holdings endpoints
|   |   +-- portfolio.py    # Portfolio analytics
|   |   +-- research.py     # Research/valuation
|   |   +-- accounting.py   # SEC filings
|   |   +-- signals.py      # Trading signals
|   |   +-- commodities.py  # Commodity data
|   |   +-- macro.py        # Macroeconomic data
|   |   +-- notes.py        # Notes/theses management
|   +-- main.py             # Application entry point
+-- analytics/              # Analysis engines
|   +-- money_flow.py       # Money flow analysis
|   +-- institutional.py    # Institutional positioning
|   +-- alerts.py           # Alert generation
+-- accounting/             # SEC filings and accounting
|   +-- edgar_adapter.py    # SEC EDGAR integration
|   +-- anomaly_detection.py# Financial anomaly detection
+-- commodities/            # Commodity analysis
+-- data/                   # Data access layer
|   +-- data_manager.py     # Unified data interface
|   +-- openbb_adapter.py   # OpenBB integration
+-- macro/                  # Macroeconomic analysis
|   +-- regime_detector.py  # Economic regime detection
+-- portfolio/              # Portfolio analytics
|   +-- risk_metrics.py     # VaR, beta, correlations
+-- research/               # Fundamental research
|   +-- dcf.py              # DCF valuation
|   +-- research_analyzer.py# Research aggregation
+-- signals/                # Signal generation
|   +-- signal_generator.py # Multi-factor signals
|   +-- backtester.py       # Signal backtesting
+-- notes/                  # Notes/thesis management
+-- options/                # Options analysis
+-- etf/                    # ETF analysis
+-- integrations/           # External integrations
    +-- nautilus/           # NautilusTrader integration
```

### Rust GUI Structure

```
stanley-gui/src/
+-- main.rs                 # Application entry point
+-- app.rs                  # Main application state (StanleyApp)
+-- api.rs                  # HTTP client for Python backend
+-- theme.rs                # UI theme configuration
+-- portfolio.rs            # Portfolio view components
+-- commodities.rs          # Commodities view components
+-- comparison.rs           # Comparison views
+-- navigation.rs           # Navigation handling
+-- keyboard.rs             # Keyboard shortcuts
+-- settings.rs             # Settings management
+-- components/
    +-- charts.rs           # Chart rendering
    +-- tables.rs           # Data table components
    +-- modals.rs           # Modal dialogs
    +-- forms/              # Form components
```

---

## Python Backend Architecture

### FastAPI Application Structure

The API follows a modular router-based architecture with centralized authentication:

```python
# Application initialization (main.py)
app = FastAPI(
    title="Stanley API",
    description="Institutional Investment Analysis Platform",
    version="1.0.0"
)

# Middleware stack
app.add_middleware(CORSMiddleware, ...)
app.add_middleware(RateLimitMiddleware, ...)

# Router registration
app.include_router(market_router, prefix="/api/market")
app.include_router(institutional_router, prefix="/api/institutional")
app.include_router(portfolio_router, prefix="/api/portfolio")
app.include_router(research_router, prefix="/api/research")
app.include_router(accounting_router, prefix="/api/accounting")
app.include_router(signals_router, prefix="/api/signals")
app.include_router(commodities_router, prefix="/api/commodities")
app.include_router(macro_router, prefix="/api/macro")
app.include_router(notes_router, prefix="/api/notes")
```

### Analyzer Layer

Analyzers are stateless service classes that process data:

```
+-------------------+     +------------------+     +----------------+
| MoneyFlowAnalyzer |     | ResearchAnalyzer |     | PortfolioAnalyzer |
+-------------------+     +------------------+     +----------------+
| analyze_sector_flow()   | get_valuation()  |     | calculate_var()  |
| analyze_equity_flow()   | get_earnings()   |     | get_beta()       |
| get_dark_pool_activity()| get_dcf()        |     | get_sectors()    |
+-------------------+     +------------------+     +----------------+
         |                        |                       |
         +------------------------+-----------------------+
                                  |
                          +---------------+
                          | DataManager   |
                          +---------------+
                          | get_price()   |
                          | get_history() |
                          | get_13f()     |
                          +---------------+
```

### Data Manager Pattern

The DataManager provides a unified async interface to external data sources:

```python
class DataManager:
    """Unified data access layer with caching and fallback."""

    def __init__(self):
        self.openbb = OpenBBAdapter()
        self.edgar = EdgarAdapter()
        self.dbnomics = DBnomicsAdapter()
        self.cache = {}

    async def get_market_data(self, symbol: str) -> MarketData:
        """Fetch market data with caching."""
        ...

    async def get_institutional_holdings(self, symbol: str) -> List[Holding]:
        """Fetch 13F institutional holdings."""
        ...
```

---

## Rust GUI Architecture

### GPUI Application Model

The Rust GUI uses Zed's GPUI framework for GPU-accelerated rendering:

```rust
pub struct StanleyApp {
    // View state
    active_view: ActiveView,
    theme: Theme,

    // Data state with async loading
    market_data: LoadingState<MarketData>,
    sector_flow: LoadingState<Vec<SectorFlow>>,
    equity_flow: LoadingState<EquityFlowResponse>,
    institutional: LoadingState<Vec<InstitutionalHolder>>,

    // Subview states
    commodities_state: CommoditiesState,
    portfolio_holdings: PortfolioLoadState<Vec<Holding>>,

    // API client
    api_client: Arc<StanleyClient>,
}
```

### View Architecture

```
+------------------+
|   StanleyApp     |
+------------------+
        |
        +-------------------+-------------------+
        |                   |                   |
+---------------+   +---------------+   +---------------+
| DashboardView |   | PortfolioView |   | CommoditiesView |
+---------------+   +---------------+   +---------------+
| - MarketData  |   | - Holdings    |   | - Prices      |
| - SectorFlow  |   | - RiskMetrics |   | - Correlations|
| - Institutional|   | - Sectors    |   | - MacroLinks  |
+---------------+   +---------------+   +---------------+
```

### API Client

The Rust GUI communicates with the Python backend via HTTP:

```rust
pub struct StanleyClient {
    base_url: String,         // Default: http://localhost:8000
    client: reqwest::Client,  // Async HTTP client
}

impl StanleyClient {
    // Sector money flow
    pub async fn get_sector_money_flow(&self, sectors: Vec<String>)
        -> Result<SectorFlowResponse, ApiError>;

    // Institutional holdings
    pub async fn get_institutional_holdings(&self, symbol: &str)
        -> Result<InstitutionalHoldingsResponse, ApiError>;

    // Health check
    pub async fn health_check(&self) -> Result<HealthResponse, ApiError>;
}
```

---

## Authentication and Authorization

### Authentication Flow

```
+--------+     +----------------+     +------------------+
| Client | --> | Auth Middleware| --> | Protected Route  |
+--------+     +----------------+     +------------------+
    |               |                        |
    |  JWT Token    |  Validate Token        |  User Context
    |  or API Key   |  Extract User          |  Available
    +---------------+-----------------------+
```

### JWT Token Structure

```python
# Access token payload
{
    "user_id": "uuid",
    "email": "user@example.com",
    "roles": ["analyst", "trader"],
    "jti": "unique-token-id",
    "iat": 1703664000,
    "exp": 1703665800,  # 30 minutes
    "type": "access"
}

# Refresh token (7 days)
{
    "sub": "user-uuid",
    "jti": "unique-token-id",
    "type": "refresh"
}
```

### Role-Based Access Control (RBAC)

```python
class Role(str, Enum):
    VIEWER = "viewer"           # Read-only access
    ANALYST = "analyst"         # Analytics access
    TRADER = "trader"           # Trading signals
    PORTFOLIO_MANAGER = "pm"    # Team management
    ADMIN = "admin"             # Full access
    SUPER_ADMIN = "super_admin" # System config

# Role hierarchy
ROLE_HIERARCHY = {
    Role.VIEWER: 1,
    Role.ANALYST: 2,
    Role.TRADER: 3,
    Role.PORTFOLIO_MANAGER: 4,
    Role.ADMIN: 5,
    Role.SUPER_ADMIN: 6,
}
```

### API Key Authentication

API keys support programmatic access with scoped permissions:

```python
class APIKeyScope(str, Enum):
    READ = "read"       # Maps to VIEWER role
    WRITE = "write"     # Maps to ANALYST role
    TRADE = "trade"     # Maps to TRADER role
    ADMIN = "admin"     # Maps to ADMIN role
```

### FastAPI Dependencies

```python
# Require authentication
@app.get("/api/portfolio")
async def get_portfolio(user: User = Depends(get_current_user)):
    ...

# Require specific role
@app.post("/api/admin/users")
async def create_user(user: User = Depends(require_roles(Role.ADMIN))):
    ...

# Require minimum permission level
@app.get("/api/signals")
async def get_signals(user: User = Depends(require_permission_level(Role.TRADER))):
    ...
```

---

## Data Flow

### Request Flow Diagram

```
                                 RUST GUI
+--------------------------------------------------------------------+
| StanleyApp                                                          |
|   |                                                                 |
|   +-- spawn_async(fetch_market_data())                             |
|         |                                                           |
|         v                                                           |
| +---------------+                                                   |
| | StanleyClient |                                                   |
| +-------+-------+                                                   |
|         |                                                           |
+---------+-----------------------------------------------------------+
          | HTTP GET /api/market/AAPL
          v
+--------------------------------------------------------------------+
|                         PYTHON BACKEND                              |
+--------------------------------------------------------------------+
|         |                                                           |
|  +------v-------+                                                   |
|  | RateLimiter  |  Check rate limits by category                   |
|  +------+-------+                                                   |
|         |                                                           |
|  +------v-------+                                                   |
|  | Auth Layer   |  Validate JWT/API key (optional for some routes) |
|  +------+-------+                                                   |
|         |                                                           |
|  +------v-------+                                                   |
|  | Router       |  Route to appropriate handler                     |
|  +------+-------+                                                   |
|         |                                                           |
|  +------v-------+                                                   |
|  | DataManager  |  Fetch from cache or external source              |
|  +------+-------+                                                   |
|         |                                                           |
|  +------v-------+                                                   |
|  | OpenBB/Edgar |  External data providers                          |
|  +------+-------+                                                   |
|         |                                                           |
+---------+-----------------------------------------------------------+
          | JSON Response
          v
+--------------------------------------------------------------------+
| RUST GUI                                                            |
|   |                                                                 |
|   +-- cx.emit(MarketDataLoaded(data))                              |
|   +-- self.market_data = LoadingState::Loaded(data)                |
|   +-- cx.notify()  // Trigger re-render                             |
+--------------------------------------------------------------------+
```

### Loading State Pattern

Both Rust and Python components use consistent loading state patterns:

```rust
// Rust GUI
pub enum LoadingState<T> {
    NotStarted,
    Loading,
    Loaded(T),
    Error(String),
}
```

```python
# Python (conceptual)
class DataState(Generic[T]):
    status: Literal["pending", "loading", "success", "error"]
    data: Optional[T]
    error: Optional[str]
```

---

## API Endpoint Structure

### Endpoint Categories and Rate Limits

| Category | Prefix | Rate Limit | Description |
|----------|--------|------------|-------------|
| Market Data | `/api/market` | 100/min | Real-time quotes, prices |
| Analytics | `/api/institutional`, `/api/money-flow` | 30/min | Flow analysis |
| Research | `/api/research`, `/api/valuation` | 20/min | Fundamental research |
| Accounting | `/api/accounting` | 10/min | SEC filings (EDGAR courtesy) |
| Signals | `/api/signals` | 50/min | Trading signals |
| Portfolio | `/api/portfolio` | 30/min | Portfolio analytics |
| Commodities | `/api/commodities` | 30/min | Commodity data |
| Macro | `/api/macro` | 20/min | Economic data |
| Notes | `/api/notes`, `/api/theses` | 50/min | Research notes |
| Default | Others | 60/min | General endpoints |

### Core Endpoints

```
# Health & System
GET  /api/health                     # Health check

# Market Data
GET  /api/market/{symbol}            # Quote and market data

# Institutional Analysis
GET  /api/institutional/{symbol}     # 13F holdings
POST /api/money-flow                 # Sector money flow
GET  /api/dark-pool/{symbol}         # Dark pool activity
GET  /api/equity-flow/{symbol}       # Equity flow metrics

# Portfolio
POST /api/portfolio-analytics        # VaR, beta, sectors

# Research
GET  /api/research/{symbol}          # Comprehensive research
GET  /api/valuation/{symbol}         # DCF valuation
GET  /api/earnings/{symbol}          # Earnings analysis
GET  /api/peers/{symbol}             # Peer comparison

# Commodities
GET  /api/commodities                # Market overview
GET  /api/commodities/{symbol}       # Commodity detail
GET  /api/commodities/correlations   # Correlation matrix

# Accounting (SEC Filings)
GET  /api/accounting/{symbol}        # Financial statements
GET  /api/accounting/{symbol}/red-flags  # Accounting red flags

# Signals
GET  /api/signals/{symbol}           # Trading signals
POST /api/signals/backtest           # Signal backtesting

# Notes
GET  /api/theses                     # Investment theses
GET  /api/trades                     # Trade notes
```

---

## Security Architecture

### Defense in Depth

```
+------------------------------------------------------------------+
|                        SECURITY LAYERS                            |
+------------------------------------------------------------------+
|                                                                   |
|  Layer 1: Transport Security                                      |
|  +------------------------------------------------------------+  |
|  | TLS 1.3 | HTTPS | Certificate Validation                    |  |
|  +------------------------------------------------------------+  |
|                                                                   |
|  Layer 2: Rate Limiting                                          |
|  +------------------------------------------------------------+  |
|  | Sliding Window | Category-based | Per-user/IP limits        |  |
|  +------------------------------------------------------------+  |
|                                                                   |
|  Layer 3: Authentication                                         |
|  +------------------------------------------------------------+  |
|  | JWT Tokens | API Keys | OAuth2 (planned)                     |  |
|  +------------------------------------------------------------+  |
|                                                                   |
|  Layer 4: Authorization                                          |
|  +------------------------------------------------------------+  |
|  | RBAC | Role Hierarchy | Permission Checks                    |  |
|  +------------------------------------------------------------+  |
|                                                                   |
|  Layer 5: Input Validation                                       |
|  +------------------------------------------------------------+  |
|  | Pydantic Models | Type Validation | Sanitization             |  |
|  +------------------------------------------------------------+  |
|                                                                   |
+------------------------------------------------------------------+
```

### Rate Limiting Implementation

```python
# Rate limit headers in responses
X-RateLimit-Limit: 100      # Max requests in window
X-RateLimit-Remaining: 87   # Remaining requests
X-RateLimit-Reset: 1703664060  # Window reset timestamp
X-RateLimit-Category: market_data  # Applied category
```

### Security Best Practices

1. **Token Security**
   - Short-lived access tokens (30 minutes)
   - Refresh token rotation
   - Secure token storage guidance

2. **API Key Security**
   - SHA-256 hashed storage
   - Prefix for identification without exposure
   - Optional IP whitelisting
   - Scope-based permissions

3. **Request Security**
   - CORS configuration for allowed origins
   - Request ID tracking for audit
   - Input validation on all endpoints

---

## Deployment Architecture

### Development Environment

```
localhost:8000  <-- Python FastAPI (uvicorn)
     ^
     |
     +-- stanley-gui (GPUI native app)
```

### Production Considerations

```
+------------------+     +--------------------+
|   Load Balancer  |     |   CDN/Edge Cache   |
+--------+---------+     +----------+---------+
         |                          |
         v                          v
+--------+---------+     +----------+---------+
|  API Instance 1  |     |  Static Assets     |
+--------+---------+     +--------------------+
         |
+--------+---------+
|  API Instance N  |
+--------+---------+
         |
         v
+--------+---------+     +----------+---------+
|   Redis Cache    |     |   PostgreSQL       |
+------------------+     +--------------------+
```

---

## Related Documents

- [Multi-User Architecture Design](./multi_user_design.md) - Multi-tenant evolution roadmap
- [Enhanced Money Flow Architecture](./enhanced_money_flow_summary.md) - Money flow module design
- [ML Architecture Roadmap](../ml_architecture_roadmap.md) - Machine learning integration
- [Rust Financial Systems](../rust_financial_systems_architecture.md) - Advanced Rust patterns
- [API Documentation](../API.md) - Detailed API reference
- [Module Documentation](../MODULES.md) - Python module details
