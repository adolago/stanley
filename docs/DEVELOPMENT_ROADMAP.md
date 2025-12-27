# Stanley Development Roadmap

**Generated from 60+ Opus Agent Analysis**
**Date**: December 2024

---

## Executive Summary

Stanley is an institutional investment analysis platform with a Python FastAPI backend (80+ endpoints) and Rust GPUI desktop GUI. Analysis reveals:

- **GUI Coverage**: Only 9.4% of API endpoints connected (8/85)
- **Test Pass Rate**: 914 passed, 273 skipped, 0 failed
- **Security**: Critical gaps - no authentication or rate limiting
- **Architecture**: Monolithic API needs modularization
- **NautilusTrader**: 40% ready for live trading
- **Macro Module**: 70-75% institutional standard completeness

---

## Phase 0: Critical Security (Week 1-2)

### Authentication & Authorization
```python
# Priority: CRITICAL
# Location: stanley/api/auth.py (new)

# Required implementations:
- JWT token authentication
- API key management for programmatic access
- Role-based access control (RBAC)
- Session management with secure cookies
- Rate limiting per user/API key
```

### Security Fixes Required
| Issue | Severity | Fix |
|-------|----------|-----|
| No authentication | CRITICAL | Implement JWT + API keys |
| No rate limiting | HIGH | Add per-endpoint limits |
| No input validation | HIGH | Pydantic strict mode |
| Secrets in code | MEDIUM | Environment variables |
| No CORS config | MEDIUM | Restrict origins |
| No audit logging | MEDIUM | Add security events |

### Rate Limiting Strategy
```python
# Recommended limits:
RATE_LIMITS = {
    "market_data": "100/minute",
    "analytics": "30/minute",
    "research": "20/minute",
    "accounting": "10/minute",  # SEC EDGAR courtesy
    "signals": "50/minute",
}
```

---

## Phase 1: Architecture Refactoring (Week 2-4)

### Split Monolithic API
Current `stanley/api/main.py` is 4346 lines. Split into:

```
stanley/api/
├── main.py              # App initialization only (~100 lines)
├── routers/
│   ├── market.py        # Market data endpoints
│   ├── institutional.py # 13F, ownership endpoints
│   ├── analytics.py     # Money flow, dark pool
│   ├── portfolio.py     # Portfolio analytics
│   ├── research.py      # Valuation, earnings, peers
│   ├── commodities.py   # Commodities endpoints
│   ├── options.py       # Options flow, gamma
│   ├── etf.py           # ETF analytics
│   ├── macro.py         # Economic indicators
│   ├── accounting.py    # SEC filings
│   ├── signals.py       # Signal generation
│   └── notes.py         # Research vault
├── middleware/
│   ├── auth.py          # Authentication
│   ├── rate_limit.py    # Rate limiting
│   └── logging.py       # Request logging
├── dependencies.py      # DI container
└── schemas/             # Pydantic models
```

### Dependency Injection Container
```python
# stanley/api/dependencies.py
from functools import lru_cache

class Container:
    def __init__(self):
        self._stanley = None
        self._cache = None

    @property
    def stanley(self) -> Stanley:
        if not self._stanley:
            self._stanley = Stanley()
        return self._stanley

    @property
    def cache(self) -> Redis:
        if not self._cache:
            self._cache = Redis.from_url(settings.REDIS_URL)
        return self._cache

@lru_cache
def get_container() -> Container:
    return Container()
```

### Async Standardization
```python
# Current: Mixed sync/async
def get_market_data(symbol):  # Sync
    return stanley.get_market_data(symbol)

# Target: Consistent async
async def get_market_data(symbol):  # Async
    return await stanley.get_market_data_async(symbol)
```

---

## Phase 2: Data Layer Enhancement (Week 3-5)

### Missing DataManager Methods
```python
# stanley/data/data_manager.py additions:

async def get_13f_holdings(self, symbol: str) -> pd.DataFrame:
    """Fetch institutional 13F holdings."""

async def get_dark_pool_trades(self, symbol: str) -> pd.DataFrame:
    """Fetch dark pool transaction data."""

async def get_options_flow(self, symbol: str) -> pd.DataFrame:
    """Fetch real-time options flow."""

async def stream_market_data(self, symbols: List[str]) -> AsyncIterator:
    """WebSocket streaming for real-time data."""
```

### Caching Strategy
```python
# Cache tiers:
CACHE_CONFIG = {
    "market_data": {"ttl": 60, "backend": "redis"},      # 1 min
    "fundamentals": {"ttl": 3600, "backend": "redis"},   # 1 hour
    "filings": {"ttl": 86400, "backend": "disk"},        # 24 hours
    "static": {"ttl": 604800, "backend": "disk"},        # 7 days
}
```

### Real-time Data Pipeline
```
[Market Data Sources]
        ↓
[WebSocket Aggregator] → [Redis Pub/Sub] → [GUI Subscribers]
        ↓
[Time-Series DB (TimescaleDB)]
        ↓
[Analytics Engine]
```

---

## Phase 3: GUI Expansion (Week 4-10)

### New Views Required (6 Total)

#### 3.1 Portfolio View (Week 4-5)
```rust
// stanley-gui/src/portfolio.rs
pub struct PortfolioView {
    holdings: LoadState<Vec<Holding>>,
    analytics: LoadState<PortfolioAnalytics>,
    risk_metrics: LoadState<RiskMetrics>,
    sector_allocation: LoadState<Vec<SectorWeight>>,
    selected_holding: Option<usize>,
    time_range: TimeRange,
}

// Components:
// - Holdings table with P&L
// - Sector allocation donut chart
// - VaR/CVaR risk metrics
// - Performance attribution
// - Benchmark comparison
```

**API Connections**: 8 endpoints
- `POST /api/portfolio-analytics`
- `GET /api/portfolio/risk`
- `GET /api/portfolio/attribution`
- `GET /api/portfolio/benchmark`

#### 3.2 ETF View (Week 5-6)
```rust
// stanley-gui/src/etf.rs
pub struct EtfView {
    flows: LoadState<EtfFlows>,
    sector_rotation: LoadState<SectorRotation>,
    smart_beta: LoadState<SmartBetaMetrics>,
    thematic: LoadState<Vec<ThematicEtf>>,
}

// Components:
// - Flow heatmap by sector
// - Rotation momentum chart
// - Factor exposure radar
// - Thematic trends
```

**API Connections**: 5 endpoints
- `GET /api/etf/flows`
- `GET /api/etf/sector-rotation`
- `GET /api/etf/smart-beta`
- `GET /api/etf/thematic`
- `GET /api/etf/{symbol}`

#### 3.3 Accounting View (Week 6-7)
```rust
// stanley-gui/src/accounting.rs
pub struct AccountingView {
    filings: LoadState<Vec<Filing>>,
    statements: LoadState<FinancialStatements>,
    quality_score: LoadState<EarningsQuality>,
    red_flags: LoadState<Vec<RedFlag>>,
    selected_filing: Option<Filing>,
}

// Components:
// - Filing timeline
// - Statement comparison
// - Quality score gauge
// - Red flag alerts
// - Beneish M-Score display
```

**API Connections**: 5 endpoints
- `GET /api/accounting/{symbol}/filings`
- `GET /api/accounting/{symbol}/statements`
- `GET /api/accounting/{symbol}/earnings-quality`
- `GET /api/accounting/{symbol}/red-flags`
- `GET /api/accounting/{symbol}/audit-fees`

#### 3.4 Signals View (Week 7-8)
```rust
// stanley-gui/src/signals.rs
pub struct SignalsView {
    active_signals: LoadState<Vec<Signal>>,
    backtest_results: LoadState<BacktestResults>,
    performance: LoadState<PerformanceStats>,
    signal_config: SignalConfiguration,
}

// Components:
// - Signal cards with entry/exit
// - Backtest equity curve
// - Win rate statistics
// - Sharpe/Sortino metrics
// - Signal builder form
```

**API Connections**: 5 endpoints
- `GET /api/signals/{symbol}`
- `POST /api/signals`
- `GET /api/signals/backtest`
- `GET /api/signals/performance/stats`
- `POST /api/signals/configure`

#### 3.5 Notes View (Week 8-9)
```rust
// stanley-gui/src/notes.rs
pub struct NotesView {
    notes: LoadState<Vec<Note>>,
    theses: LoadState<Vec<Thesis>>,
    trades: LoadState<Vec<TradeEntry>>,
    selected_note: Option<Note>,
    editor_content: String,
}

// Components:
// - Note list with search
// - Markdown editor
// - Thesis tracker
// - Trade journal table
// - Tag management
```

**API Connections**: 6 endpoints
- `GET /api/notes`
- `GET /api/notes/{name}`
- `PUT /api/notes/{name}`
- `GET /api/theses`
- `GET /api/trades`
- `POST /api/trades`

#### 3.6 Commodities View (Week 9-10)
```rust
// stanley-gui/src/commodities.rs
pub struct CommoditiesView {
    overview: LoadState<CommoditiesOverview>,
    selected_commodity: Option<String>,
    detail: LoadState<CommodityDetail>,
    correlations: LoadState<CorrelationMatrix>,
    macro_linkages: LoadState<MacroLinkages>,
}

// Components:
// - Price grid with sparklines
// - Correlation heatmap
// - Macro indicator links
// - Futures curve chart
// - Seasonal patterns
```

**API Connections**: 5 endpoints
- `GET /api/commodities`
- `GET /api/commodities/{symbol}`
- `GET /api/commodities/{symbol}/macro`
- `GET /api/commodities/correlations`
- `GET /api/commodities/futures-curve`

### Enhanced Existing Views

#### Dashboard Enhancements
```rust
// Add to existing Dashboard:
- Watchlist widget with alerts
- Quick signals summary
- Portfolio snapshot
- News feed integration
- Market breadth indicators
```

#### Research View Enhancements
```rust
// Add to existing Research:
- DCF calculator with scenarios
- Peer comparison table
- Earnings surprise history
- Analyst revision tracking
- Fair value range chart
```

---

## Phase 4: API Improvements (Week 5-8)

### New Endpoints Required

```python
# Portfolio enhancements
GET  /api/portfolio/risk              # Detailed risk metrics
GET  /api/portfolio/attribution       # Performance attribution
GET  /api/portfolio/rebalance         # Rebalancing suggestions
POST /api/portfolio/optimize          # Portfolio optimization

# Research enhancements
GET  /api/research/{symbol}/dcf       # Detailed DCF model
GET  /api/research/{symbol}/comps     # Comparable analysis
GET  /api/research/{symbol}/sum-parts # Sum-of-parts valuation

# Macro enhancements
GET  /api/macro/fed-watch             # Fed meeting probabilities
GET  /api/macro/cross-asset           # Cross-asset correlations
GET  /api/macro/factor-returns        # Factor performance

# Real-time
WS   /ws/market                       # Market data stream
WS   /ws/signals                      # Signal alerts stream
WS   /ws/portfolio                    # Portfolio updates stream

# Screening
POST /api/screen/fundamental          # Fundamental screener
POST /api/screen/technical            # Technical screener
POST /api/screen/institutional        # Institutional activity screener
```

### Response Standardization
```python
# Standard response envelope
class APIResponse(BaseModel):
    success: bool
    data: Optional[Any]
    error: Optional[ErrorDetail]
    meta: ResponseMeta

class ResponseMeta(BaseModel):
    timestamp: datetime
    request_id: str
    cache_hit: bool
    processing_time_ms: float
```

### Validation Improvements
```python
# Strict Pydantic models
class SymbolRequest(BaseModel):
    symbol: str = Field(..., pattern=r'^[A-Z]{1,5}$')

    class Config:
        strict = True

class DateRangeRequest(BaseModel):
    start_date: date
    end_date: date

    @validator('end_date')
    def end_after_start(cls, v, values):
        if v < values.get('start_date'):
            raise ValueError('end_date must be after start_date')
        return v
```

---

## Phase 5: Testing Expansion (Week 6-10)

### Current Coverage Gaps
```
Module              | Coverage | Target | Gap
--------------------|----------|--------|-----
stanley/api/        | 45%      | 85%    | -40%
stanley/macro/      | 52%      | 80%    | -28%
stanley/accounting/ | 48%      | 80%    | -32%
stanley/options/    | 61%      | 80%    | -19%
stanley/signals/    | 55%      | 80%    | -25%
```

### New Test Categories

#### Integration Tests
```python
# tests/integration/test_api_flows.py
async def test_full_research_workflow():
    """Test complete research flow: market → analysis → signals."""

async def test_portfolio_rebalancing():
    """Test portfolio analysis to rebalancing suggestions."""

async def test_sec_filing_pipeline():
    """Test SEC filing fetch → parse → quality score."""
```

#### Load Tests
```python
# tests/load/test_api_performance.py
@pytest.mark.load
async def test_concurrent_requests():
    """Test 100 concurrent requests to market endpoint."""

@pytest.mark.load
async def test_sustained_load():
    """Test 10 req/sec for 5 minutes."""
```

#### GUI Tests
```rust
// stanley-gui/tests/integration_tests.rs
#[test]
fn test_view_navigation() {
    // Test all view transitions
}

#[test]
fn test_api_error_handling() {
    // Test error states in all views
}

#[test]
fn test_data_refresh() {
    // Test LoadState transitions
}
```

---

## Phase 6: Advanced Features (Week 10-16)

### 6.1 Alerting System
```python
# stanley/alerts/
class AlertEngine:
    """Real-time alert processing."""

    alert_types = [
        "price_threshold",
        "volume_spike",
        "institutional_filing",
        "earnings_surprise",
        "signal_trigger",
        "risk_breach",
    ]
```

### 6.2 Screening Engine
```python
# stanley/screening/
class Screener:
    """Multi-factor stock screener."""

    criteria_types = [
        "fundamental",  # P/E, P/B, ROE, etc.
        "technical",    # RSI, MACD, moving averages
        "institutional", # 13F changes, dark pool
        "quality",      # Piotroski, Beneish
    ]
```

### 6.3 NautilusTrader Production
```python
# Current: 40% production ready
# Target: 90% production ready

# Required:
- [ ] Production broker adapters (IBKR, Alpaca)
- [ ] Live order management
- [ ] Risk controls and circuit breakers
- [ ] Real-time P&L tracking
- [ ] Execution analytics
- [ ] Compliance logging
```

### 6.4 ML Integration
```python
# stanley/ml/
class MLPipeline:
    models = [
        "earnings_predictor",      # Predict earnings surprises
        "flow_classifier",         # Classify institutional flow
        "regime_detector",         # Market regime classification
        "anomaly_detector",        # Unusual activity detection
    ]
```

---

## Phase 7: Infrastructure (Week 8-14)

### Docker Deployment
```yaml
# docker-compose.prod.yml
services:
  api:
    build: ./docker/api
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://...
    deploy:
      replicas: 3

  redis:
    image: redis:7-alpine

  timescaledb:
    image: timescale/timescaledb:latest-pg15

  nginx:
    image: nginx:alpine
    # Load balancing + SSL termination
```

### CI/CD Pipeline
```yaml
# .github/workflows/ci.yml enhancements
jobs:
  test:
    - Unit tests (pytest)
    - Integration tests
    - Load tests (weekly)
    - Security scan (Bandit, Safety)

  build:
    - Python package
    - Rust GUI (Linux, macOS, Windows)
    - Docker images

  deploy:
    - Staging (auto on PR merge)
    - Production (manual approval)
```

### Monitoring Stack
```yaml
# Observability:
- Prometheus: Metrics collection
- Grafana: Dashboards
- Loki: Log aggregation
- Jaeger: Distributed tracing

# Key metrics:
- API latency p50/p95/p99
- Error rates by endpoint
- Cache hit rates
- Data freshness
- GUI render times
```

---

## Implementation Priority Matrix

| Priority | Item | Effort | Impact | Dependencies |
|----------|------|--------|--------|--------------|
| P0 | Authentication | 2 weeks | CRITICAL | None |
| P0 | Rate limiting | 1 week | HIGH | Auth |
| P1 | Split main.py | 2 weeks | HIGH | None |
| P1 | Portfolio View | 2 weeks | HIGH | API split |
| P1 | Test coverage | Ongoing | HIGH | None |
| P2 | ETF View | 1.5 weeks | MEDIUM | Portfolio |
| P2 | Accounting View | 1.5 weeks | MEDIUM | None |
| P2 | Caching layer | 2 weeks | MEDIUM | Redis |
| P3 | Signals View | 1 week | MEDIUM | None |
| P3 | Notes View | 1 week | LOW | None |
| P3 | Commodities View | 1 week | MEDIUM | None |
| P4 | Alerting | 3 weeks | MEDIUM | WebSocket |
| P4 | Screening | 2 weeks | MEDIUM | None |
| P4 | ML Pipeline | 4 weeks | LOW | Data layer |

---

## Resource Requirements

### Development Team
- 1 Senior Python Backend Developer (API, data layer)
- 1 Rust Developer (GPUI GUI)
- 1 DevOps Engineer (CI/CD, Docker, monitoring)
- 0.5 QA Engineer (testing)

### Infrastructure
- Redis cluster (caching)
- PostgreSQL/TimescaleDB (persistence)
- S3-compatible storage (backups, large files)
- Kubernetes cluster (production deployment)

---

## Success Metrics

| Metric | Current | 8-Week Target | 16-Week Target |
|--------|---------|---------------|----------------|
| API Coverage in GUI | 9.4% | 50% | 85% |
| Test Coverage | ~55% | 70% | 85% |
| API Latency p95 | Unknown | <500ms | <200ms |
| Error Rate | Unknown | <1% | <0.1% |
| NautilusTrader Ready | 40% | 60% | 90% |

---

## Appendix: Agent Analysis Sources

This roadmap synthesizes findings from 60+ specialized Opus agents:

**Architecture & Infrastructure**: architecture-analysis, api-completeness, security-analysis, ci-cd-pipeline, docker-deployment, logging-monitoring, error-handling

**GUI & Frontend**: gui-improvements, gui-integration-mapping, portfolio-view-design, accounting-view-design, etf-view, commodities-view, signals-view, notes-view, chart-components, table-components, form-components, modal-dialogs, navigation, keyboard-shortcuts

**Data & Analytics**: data-layer-analysis, money-flow-algorithms, valuation-models, correlation-analysis, options-analytics, etf-analytics, factor-analysis, institutional-data

**Testing & Quality**: test-coverage, code-quality, data-validation, performance-optimization

**Features & Integrations**: feature-roadmap, nautilus-trader, macro-module, openbb-integration, watchlist-management, multi-symbol-comparison, benchmark-comparison, backtesting, position-sizing, portfolio-risk-models, earnings-quality, news-integration, calendar-integration

**Advanced**: ml-integration, plugin-architecture, multi-user-support, mobile-companion, alerting-system, screening-engine

---

*Last Updated: December 2024*
*Next Review: After Phase 2 completion*
