# Stanley Module Documentation

This document provides detailed documentation for each module in the Stanley investment analysis platform, including module status, dependencies, and API integration points.

## Module Overview

| Module | Status | Version | API Router | Description |
|--------|--------|---------|------------|-------------|
| `stanley.core` | Active | 0.1.0 | - | Main Stanley class and coordination |
| `stanley.data` | Active | 0.1.0 | - | Data management layer |
| `stanley.analytics` | Active | 0.1.0 | `/api/analytics` | Money flow and institutional analysis |
| `stanley.portfolio` | Active | 0.1.0 | `/api/portfolio` | Portfolio analytics and risk metrics |
| `stanley.research` | Active | 0.1.0 | `/api/research` | Valuation, earnings, peer analysis |
| `stanley.macro` | Active | 0.1.0 | `/api/macro` | Macroeconomic indicators and regime detection |
| `stanley.accounting` | Active | 0.1.0 | `/api/accounting` | SEC filings and financial statements |
| `stanley.commodities` | Active | 0.1.0 | `/api/commodities` | Commodity prices and correlations |
| `stanley.options` | Active | 0.1.0 | `/api/options` | Options flow and analytics |
| `stanley.etf` | Active | 0.1.0 | `/api/etf` | ETF flows and rotation analysis |
| `stanley.signals` | Active | 0.1.0 | `/api/signals` | Signal generation and backtesting |
| `stanley.notes` | Active | 0.1.0 | `/api/notes` | Research vault and trade journal |
| `stanley.persistence` | Active | 0.1.0 | - | Local SQLite database |
| `stanley.validation` | Active | 0.1.0 | - | Input/output validation |
| `stanley.plugins` | Active | 0.1.0 | - | Plugin system architecture |
| `stanley.api` | Active | 0.2.0 | All routes | FastAPI REST API layer |
| `stanley.api.auth` | Active | 0.2.0 | - | Authentication and authorization |
| `stanley.integrations.nautilus` | Active | 0.1.0 | - | NautilusTrader integration |

---

## Module Dependency Graph

```
                            +----------------+
                            |  stanley.core  |
                            +-------+--------+
                                    |
                +-------------------+-------------------+
                |                   |                   |
                v                   v                   v
        +-------+-------+   +------+------+   +--------+-------+
        | stanley.data  |   | stanley.api |   | stanley.config |
        +-------+-------+   +------+------+   +----------------+
                |                  |
    +-----------+-----------+      |
    |           |           |      |
    v           v           v      v
+---+---+ +-----+----+ +----+----+ +-------+
|OpenBB | |DBnomics  | |EDGAR    | | Auth  |
|Adapter| |Adapter   | |Adapter  | |Module |
+-------+ +----------+ +---------+ +-------+
    |           |           |
    +-----------+-----------+
                |
    +-----------+-----------+-----------+-----------+
    |           |           |           |           |
    v           v           v           v           v
+---+---+ +-----+-----+ +---+---+ +-----+----+ +----+---+
|analyt.| |portfolio  | |macro  | |research  | |account.|
+-------+ +-----------+ +-------+ +----------+ +--------+
    |           |           |           |
    +-----------+-----------+-----------+
                |
                v
        +-------+-------+
        |   signals     |
        +---------------+

Legend:
  analyt. = stanley.analytics
  account.= stanley.accounting
```

---

## Core Modules

### stanley.core

The main Stanley class that coordinates all functionality.

**Status:** Active
**Dependencies:** `stanley.data`, `stanley.analytics`, `stanley.config`
**API Integration:** Core class used by all API routers

```python
from stanley.core import Stanley

# Initialize Stanley
stanley = Stanley(config_path="config/stanley.yaml")

# Health check
status = stanley.health_check()

# Analyze sector money flow
money_flow = stanley.analyze_sector_money_flow(
    sectors=["XLK", "XLF", "XLE"],
    lookback_days=63
)

# Get institutional holdings
holdings = stanley.get_institutional_holdings("AAPL")
```

### stanley.data

Data management layer with OpenBB integration.

**Status:** Active
**Dependencies:** `openbb`, `pandas`, `aiohttp`
**API Integration:** Used by all analytics modules

**Exports:**
- `DataManager` - Central data coordinator
- `OpenBBAdapter` - OpenBB SDK integration
- `DataProvider` - Base provider interface
- `DataProviderError`, `RateLimitError`, `DataNotFoundError`, `AuthenticationError`
- `OpenBBProvider` - OpenBB implementation

#### DataManager

Central data coordinator for all data sources.

```python
from stanley.data import DataManager

# Initialize with real data
dm = DataManager(use_mock=False)
await dm.initialize()

# Get stock data
stock_data = await dm.get_stock_data("AAPL", start_date, end_date)

# Get institutional holders
holders = await dm.get_institutional_holders("AAPL")

# Get options chain
options = await dm.get_options_chain("AAPL")

# Cleanup
await dm.close()
```

#### OpenBBAdapter

Direct OpenBB SDK integration with caching and rate limiting.

```python
from stanley.data.openbb_adapter import OpenBBAdapter

async with OpenBBAdapter(config) as adapter:
    prices = await adapter.get_historical_prices("AAPL", lookback_days=252)
    holders = await adapter.get_institutional_holders("AAPL")
    options = await adapter.get_options_chain("AAPL")
```

---

## Analytics Modules

### stanley.analytics

Money flow analysis, institutional tracking, and market analytics.

**Status:** Active
**Dependencies:** `stanley.data`, `pandas`, `numpy`
**API Integration:** `/api/analytics/*`, `/api/institutional/*`

**Exports:**
- Core Analyzers: `InstitutionalAnalyzer`, `MoneyFlowAnalyzer`, `OptionsFlowAnalyzer`, `WhaleTracker`, `DarkPoolAnalyzer`
- Sector Rotation: `SectorRotationAnalyzer`, `BusinessCyclePhase`, `SECTOR_ETFS`, `CYCLE_SECTOR_MAP`, `RISK_ON_SECTORS`, `RISK_OFF_SECTORS`
- Smart Money: `SmartMoneyIndex`, `ComponentWeight`, `IndexResult`, `DivergenceResult`, `SignalType`
- Alerts: `AlertAggregator`, `AlertSeverity`, `AlertThresholds`, `AlertType`, `BlockTradeEvent`, `BlockTradeSize`, `FlowMomentumIndicator`, `MoneyFlowAlert`, `SectorRotationSignal`, `SmartMoneyMetrics`, `UnusualVolumeSignal`

#### MoneyFlowAnalyzer

```python
from stanley.analytics import MoneyFlowAnalyzer

analyzer = MoneyFlowAnalyzer(data_manager)

# Analyze sector flow
sector_flow = analyzer.analyze_sector_flow(
    sectors=["XLK", "XLF", "XLE"],
    lookback_days=63
)

# Analyze individual equity flow
equity_flow = analyzer.analyze_equity_flow("AAPL", lookback_days=20)

# Get dark pool activity
dark_pool = analyzer.get_dark_pool_activity("AAPL", lookback_days=20)
```

**Output Fields:**
- `net_flow_1m`: Net flow over 1 month
- `net_flow_3m`: Net flow over 3 months
- `institutional_change`: Change in institutional ownership
- `smart_money_sentiment`: Smart money sentiment score (-1 to 1)
- `flow_acceleration`: Rate of flow change
- `confidence_score`: Confidence in the analysis (0 to 1)

#### InstitutionalAnalyzer

13F institutional holdings analysis.

```python
from stanley.analytics import InstitutionalAnalyzer

analyzer = InstitutionalAnalyzer(data_manager)

# Get holdings data
holdings = analyzer.get_holdings("AAPL")

# Get 13F filings
filings = analyzer._get_13f_holdings("AAPL")
```

**Output Fields:**
- `manager_name`: Institution name
- `manager_cik`: SEC CIK number
- `shares_held`: Number of shares
- `value_held`: Dollar value of position
- `ownership_percentage`: Percentage of shares outstanding

---

## Portfolio Module

### stanley.portfolio

Portfolio analytics including risk metrics and performance attribution.

**Status:** Active
**Dependencies:** `stanley.data`, `numpy`, `scipy`
**API Integration:** `/api/portfolio/*`

#### PortfolioAnalyzer

```python
from stanley.portfolio import PortfolioAnalyzer

analyzer = PortfolioAnalyzer(data_manager)

# Analyze portfolio
holdings = [
    {"symbol": "AAPL", "shares": 100, "average_cost": 150.00},
    {"symbol": "MSFT", "shares": 50, "average_cost": 300.00},
]

summary = await analyzer.analyze(holdings)
```

**PortfolioSummary Fields:**
- `total_value`: Total portfolio value
- `total_return`: Dollar return
- `total_return_percent`: Percentage return
- `beta`: Portfolio beta vs market
- `var_95`: 95% Value at Risk
- `var_99`: 99% Value at Risk
- `sector_exposure`: Dict of sector weights
- `top_holdings`: List of holding details

#### Risk Metrics

```python
from stanley.portfolio.risk_metrics import (
    calculate_var,
    calculate_beta,
    calculate_sharpe_ratio,
    calculate_max_drawdown
)
```

---

## Research Module

### stanley.research

Comprehensive research report generation with valuation models.

**Status:** Active
**Dependencies:** `stanley.data`, `stanley.analytics`, `numpy`
**API Integration:** `/api/research/*`

#### ResearchAnalyzer

```python
from stanley.research import ResearchAnalyzer

analyzer = ResearchAnalyzer(data_manager)

# Generate full research report
report = await analyzer.generate_report("AAPL")

# Get valuation analysis
valuation = await analyzer.get_valuation("AAPL", include_dcf=True)

# Analyze earnings
earnings = await analyzer.analyze_earnings("AAPL", quarters=12)

# Get peer comparison
peers = await analyzer.get_peer_comparison("AAPL")
```

#### Valuation Models

```python
from stanley.research.valuation import ValuationAnalyzer

valuation = ValuationAnalyzer(data_manager)

# Get valuation multiples
multiples = await valuation.get_multiples("AAPL")

# Run DCF model
dcf = await valuation.run_dcf("AAPL")
```

#### Earnings Analysis

```python
from stanley.research.earnings import EarningsAnalyzer

earnings = EarningsAnalyzer(data_manager)

# Get earnings history
history = await earnings.get_history("AAPL", quarters=12)

# Get earnings surprises
surprises = await earnings.get_surprises("AAPL")
```

---

## Macro Module

### stanley.macro

Macroeconomic analysis with DBnomics integration.

**Status:** Active
**Dependencies:** `stanley.data`, `dbnomics`, `pandas`
**API Integration:** `/api/macro/*`

**API Endpoints:**
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/macro/indicators` | Key economic indicators |
| GET | `/api/macro/regime` | Current market regime |
| GET | `/api/macro/yield-curve` | Yield curve analysis |
| GET | `/api/macro/recession-probability` | Recession probability |
| GET | `/api/macro/fed-watch` | Fed policy expectations |
| GET | `/api/macro/cross-asset` | Cross-asset correlations |
| GET | `/api/macro/global-overview` | Global macro overview |
| GET | `/api/macro/compare-countries` | Country comparison |

#### MacroAnalyzer

```python
from stanley.macro import MacroAnalyzer

analyzer = MacroAnalyzer(data_manager)

# Get economic indicators
indicators = await analyzer.get_indicators()

# Detect market regime
regime = await analyzer.detect_regime()

# Analyze yield curve
yield_curve = await analyzer.analyze_yield_curve()

# Get recession probability
recession = await analyzer.get_recession_probability()
```

#### DBnomics Adapter

```python
from stanley.macro.dbnomics_adapter import DBnomicsAdapter

adapter = DBnomicsAdapter()

# Get GDP data
gdp = await adapter.get_gdp_data()

# Get inflation data
cpi = await adapter.get_inflation_data()

# Get unemployment data
unemployment = await adapter.get_unemployment_data()
```

**Regime Types:**
- `expansion`: Economic expansion
- `contraction`: Economic contraction
- `recovery`: Post-recession recovery
- `slowdown`: Economic slowdown

**Submodules:**
- `regime_detector`: Market regime detection
- `yield_curve`: Yield curve analysis including inversion signals
- `recession_model`: Recession probability modeling
- `credit_spreads`: Credit spread analysis (IG, HY spreads)
- `business_cycle`: Business cycle phase detection
- `volatility_regime`: Volatility regime classification (low/medium/high)

---

## Accounting Module

### stanley.accounting

SEC filings and financial statement analysis via edgartools.

**Status:** Active
**Dependencies:** `edgartools`, `stanley.data`
**API Integration:** `/api/accounting/*`

**API Endpoints:**
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/accounting/filings/{symbol}` | SEC filings list |
| GET | `/api/accounting/statements/{symbol}` | Financial statements |
| GET | `/api/accounting/quality/{symbol}` | Earnings quality score |
| GET | `/api/accounting/red-flags/{symbol}` | Accounting red flags |
| GET | `/api/accounting/footnotes/{symbol}` | Footnote extraction |

#### AccountingAnalyzer

```python
from stanley.accounting import AccountingAnalyzer

analyzer = AccountingAnalyzer(edgar_identity="your-email@example.com")

# Get SEC filings
filings = analyzer.get_filings("AAPL")

# Get financial statements
statements = analyzer.get_statements("AAPL")
```

#### EdgarAdapter

```python
from stanley.accounting.edgar_adapter import EdgarAdapter

adapter = EdgarAdapter(identity="your-email@example.com")
adapter.initialize()

# Get company filings
filings = adapter.get_company_filings("AAPL")

# Get specific filing
filing = adapter.get_filing("AAPL", "10-K", year=2023)
```

#### Earnings Quality

```python
from stanley.accounting.earnings_quality import EarningsQualityAnalyzer

analyzer = EarningsQualityAnalyzer(financial_statements)

# Get quality score
score = analyzer.analyze("AAPL")
```

**Quality Metrics:**
- `overall_score`: Overall quality (0-1)
- `accrual_quality`: Accruals ratio quality
- `revenue_quality`: Revenue recognition quality
- `cash_flow_quality`: Cash conversion quality

#### Red Flag Detection

```python
from stanley.accounting.red_flags import RedFlagScorer

scorer = RedFlagScorer(edgar_adapter)

# Get red flags
flags = scorer.analyze("AAPL")
```

**Submodules:**
- `financial_statements`: Financial statement parsing and analysis
- `anomaly_detection`: Accounting anomaly aggregation
- `footnotes`: Financial statement footnote extraction

---

## Commodities Module

### stanley.commodities

Commodity market analysis with macro linkages.

**Status:** Active
**Dependencies:** `stanley.data`, `pandas`
**API Integration:** `/api/commodities/*`

**API Endpoints:**
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/commodities/overview` | Market overview |
| GET | `/api/commodities/{symbol}` | Commodity detail |
| GET | `/api/commodities/{symbol}/macro` | Macro-commodity linkages |
| GET | `/api/commodities/correlations` | Correlation matrix |

#### CommoditiesAnalyzer

```python
from stanley.commodities import CommoditiesAnalyzer

analyzer = CommoditiesAnalyzer(data_manager)

# Get market overview
overview = await analyzer.get_market_overview()

# Get commodity summary
summary = await analyzer.get_summary("CL")  # Crude oil

# Analyze macro linkages
linkage = await analyzer.analyze_macro_linkage("GC")  # Gold

# Get correlation matrix
correlations = await analyzer.get_correlations()
```

**Supported Commodities:**
| Symbol | Commodity |
|--------|-----------|
| `CL` | Crude Oil (WTI) |
| `GC` | Gold |
| `SI` | Silver |
| `NG` | Natural Gas |
| `HG` | Copper |
| `ZC` | Corn |
| `ZW` | Wheat |
| `ZS` | Soybeans |

---

## Options Module

### stanley.options

Options flow and gamma exposure analytics.

**Status:** Active
**Dependencies:** `stanley.data`, `numpy`, `scipy`
**API Integration:** `/api/options/*`

**API Endpoints:**
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/options/flow/{symbol}` | Options flow data |
| GET | `/api/options/gamma/{symbol}` | Gamma exposure |
| GET | `/api/options/unusual/{symbol}` | Unusual activity |
| GET | `/api/options/put-call/{symbol}` | Put/call ratio |
| GET | `/api/options/max-pain/{symbol}` | Max pain calculation |

#### OptionsAnalyzer

```python
from stanley.options import OptionsAnalyzer

analyzer = OptionsAnalyzer(data_manager)

# Get options flow
flow = await analyzer.get_flow("AAPL")

# Get gamma exposure
gamma = await analyzer.get_gamma_exposure("AAPL")

# Get unusual activity
unusual = await analyzer.get_unusual_activity("AAPL")

# Get put/call ratio
pc_ratio = await analyzer.get_put_call_ratio("AAPL")

# Get max pain
max_pain = await analyzer.get_max_pain("AAPL")
```

---

## ETF Module

### stanley.etf

ETF flow and sector rotation analysis.

**Status:** Active
**Dependencies:** `stanley.data`, `stanley.analytics`
**API Integration:** `/api/etf/*`

**API Endpoints:**
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/etf/flows` | Aggregate ETF flows |
| GET | `/api/etf/{symbol}` | Individual ETF flow |
| GET | `/api/etf/rotation` | Sector rotation signals |
| GET | `/api/etf/smart-beta` | Smart beta analysis |
| GET | `/api/etf/thematic` | Thematic ETFs |
| GET | `/api/etf/institutional` | Institutional flows |

#### ETFAnalyzer

```python
from stanley.etf import ETFAnalyzer

analyzer = ETFAnalyzer(data_manager)

# Get ETF flows
flows = await analyzer.get_flows()

# Get individual ETF flow
spy_flow = await analyzer.get_etf_flow("SPY")

# Get sector rotation signals
rotation = await analyzer.get_sector_rotation()

# Get smart beta analysis
smart_beta = await analyzer.get_smart_beta()

# Get thematic ETFs
thematic = await analyzer.get_thematic()

# Get institutional flows
institutional = await analyzer.get_institutional_flows()
```

---

## Signals Module

### stanley.signals

Multi-factor signal generation and backtesting engine.

**Status:** Active
**Dependencies:** `stanley.analytics`, `stanley.research`, `stanley.portfolio`, `stanley.data`
**API Integration:** `/api/signals/*`

**API Endpoints:**
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/signals` | List all signals |
| POST | `/api/signals` | Generate new signals |
| GET | `/api/signals/{symbol}` | Symbol signals |
| GET | `/api/signals/{symbol}/composite` | Composite signal |
| POST | `/api/signals/backtest` | Backtest strategy |
| GET | `/api/signals/backtest/quick/{symbol}` | Quick backtest |
| GET | `/api/signals/performance/stats` | Performance stats |
| GET | `/api/signals/performance/history` | Performance history |
| POST | `/api/signals/{signal_id}/outcome` | Record outcome |
| POST | `/api/signals/configure` | Update config |
| GET | `/api/signals/configure` | Get config |
| GET | `/api/signals/factors` | List factors |
| GET | `/api/signals/factors/{factor_name}` | Factor detail |

#### SignalGenerator

```python
from stanley.signals import SignalGenerator

generator = SignalGenerator(
    money_flow_analyzer=money_flow_analyzer,
    institutional_analyzer=institutional_analyzer,
    research_analyzer=research_analyzer,
    portfolio_analyzer=portfolio_analyzer,
    data_manager=data_manager,
)

# Generate signals
signals = await generator.generate("AAPL")
```

#### SignalBacktester

```python
from stanley.signals import SignalBacktester

backtester = SignalBacktester(data_manager)

# Run backtest
results = await backtester.backtest(
    signals=signals,
    start_date=start_date,
    end_date=end_date,
)
```

#### PerformanceTracker

```python
from stanley.signals import PerformanceTracker

tracker = PerformanceTracker(data_manager)

# Get performance stats
stats = await tracker.get_stats()
```

---

## Notes Module (Research Vault)

### stanley.notes

Research note management with Obsidian-like linking.

**Status:** Active
**Dependencies:** `markdown`, file system
**API Integration:** `/api/notes/*`

**API Endpoints:**
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/notes` | List notes |
| GET | `/api/notes/{id}` | Get note |
| POST | `/api/notes` | Create note |
| PUT | `/api/notes/{id}` | Update note |
| DELETE | `/api/notes/{id}` | Delete note |
| GET | `/api/notes/search` | Search notes |
| GET | `/api/notes/graph` | Knowledge graph |
| GET | `/api/notes/tags` | List tags |

#### NoteManager

```python
from stanley.notes import NoteManager

manager = NoteManager()

# List all notes
notes = manager.list_notes()

# Get note
note = manager.get_note("AAPL-analysis")

# Create/update note
manager.save_note("AAPL-analysis", content)

# Search notes
results = manager.search("earnings")

# Get knowledge graph
graph = manager.get_graph()
```

**Note Types:**
- `note`: General research note
- `thesis`: Investment thesis
- `trade`: Trade journal entry
- `event`: Market event
- `person`: Key person/executive
- `sector`: Sector analysis

---

## Infrastructure Modules

### stanley.persistence

Local SQLite database for storing user data.

**Status:** Active
**Dependencies:** `sqlite3`, `sqlalchemy`
**API Integration:** Used by Notes, Settings, and Portfolio modules

**Stored Data:**
- Watchlists
- User settings
- Alert configurations
- Cached market data
- Historical analysis
- Portfolio holdings
- Trade journal

**Exports:**
- `StanleyDatabase` - Main database interface
- `Watchlist`, `WatchlistItem` - Watchlist models
- `UserSettings` - User preferences
- `Alert`, `AlertConfiguration` - Alert system
- `CachedMarketData` - Data cache
- `AnalysisRecord` - Historical analysis
- `PortfolioHolding` - Portfolio data
- `TradeEntry` - Trade journal
- `DataEncryptor` - Encryption utilities
- `MigrationManager` - Database migrations
- `BackupManager` - Backup/restore
- `SyncManager` - Multi-device sync

### stanley.validation

Comprehensive input/output validation for the Stanley API.

**Status:** Active
**Dependencies:** `pydantic`, `numpy`
**API Integration:** Middleware for all API endpoints

**Exports:**
- **Base validators:** `StanleyBaseModel`, `SymbolValidator`, `SymbolField`
- **Financial validators:** `PositiveFloat`, `NonNegativeFloat`, `Percentage`, `PriceField`, `VolumeField`, `SharesField`, `RatioField`
- **Date validators:** `DateRangeValidator`, `TradingDateField`
- **Request models:** `ValidatedMoneyFlowRequest`, `ValidatedPortfolioRequest`, `ValidatedPortfolioHolding`, `ValidatedResearchRequest`, `ValidatedCommoditiesRequest`, `ValidatedOptionsRequest`
- **Response validators:** `ValidatedMarketData`, `ValidatedPortfolioAnalytics`, `ValidatedValuationMetrics`
- **Data quality:** `DataQualityChecker`, `DataQualityReport`, `DataQualityLevel`, `check_market_data_quality`, `check_ohlc_integrity`, `check_returns_quality`
- **Outliers:** `OutlierDetector`, `OutlierResult`, `detect_price_outliers`, `detect_volume_outliers`, `detect_return_outliers`
- **Sanity checks:** `SanityChecker`, `check_var_sanity`, `check_beta_sanity`, `check_valuation_sanity`, `check_portfolio_weights`
- **Temporal:** `TemporalValidator`, `check_data_freshness`, `validate_date_range`, `validate_trading_hours`
- **Middleware:** `ValidationMiddleware`, `RequestValidator`, `ResponseValidator`
- **Errors:** `ValidationError`, `DataQualityError`, `OutlierError`, `SanityCheckError`, `TemporalValidationError`

### stanley.plugins

Extensible plugin system for custom functionality.

**Status:** Active
**Dependencies:** None (self-contained)
**API Integration:** Plugin-defined endpoints

**Architecture:**
1. **Plugin Base Classes** (`interfaces.py`)
   - `BasePlugin`: Core interface all plugins implement
   - `IndicatorPlugin`: Custom technical/fundamental indicators
   - `DataSourcePlugin`: Custom data providers
   - `AnalyzerPlugin`: Custom analysis modules
   - `ViewPlugin`: Custom visualization/output plugins

2. **Plugin Manager** (`manager.py`)
   - Plugin discovery and registration
   - Lifecycle management (load, enable, disable, unload)
   - Dependency resolution
   - Hot reload support

3. **Plugin Security** (`security.py`)
   - Sandboxed execution
   - Permission system
   - Resource limits

4. **Plugin Marketplace** (`marketplace.py`)
   - Plugin registry and discovery
   - Version management
   - Installation/update mechanism

**Usage:**
```python
from stanley.plugins import PluginManager, IndicatorPlugin

# Create a custom indicator
class MyIndicator(IndicatorPlugin):
    name = "my_indicator"
    version = "1.0.0"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        return data['close'].rolling(20).mean()

# Register with the plugin manager
manager = PluginManager()
manager.register(MyIndicator)

# Or auto-discover plugins
manager.discover_plugins()
```

---

## API Module

### stanley.api

FastAPI REST API layer with comprehensive authentication.

**Status:** Active
**Version:** 0.2.0
**Dependencies:** `fastapi`, `uvicorn`, `pydantic`
**Documentation:** See [API.md](API.md)

**Registered Routers (14 total):**
1. `system` - Health checks, version info
2. `settings` - User preferences
3. `market` - Market data
4. `portfolio` - Portfolio analytics
5. `analytics` - Money flow, dark pool, sector rotation
6. `research` - Valuation, earnings, peer analysis
7. `options` - Options flow and analytics
8. `etf` - ETF flows and analysis
9. `notes` - Research vault
10. `commodities` - Commodities data
11. `macro` - Economic indicators
12. `accounting` - SEC filings
13. `signals` - Signal generation
14. `institutional` - 13F holdings

### stanley.api.auth

Comprehensive authentication and authorization system.

**Status:** Active
**Version:** 0.2.0
**Dependencies:** `pyjwt`, `passlib`, `bcrypt`

**Authentication Methods:**
1. **JWT Token Authentication** (Web Sessions)
   - Access tokens: 15 minutes (configurable)
   - Refresh tokens: 7 days (configurable)
   - Token blacklisting for revocation

2. **API Key Authentication** (Programmatic Access)
   - Format: `sk_live_<32-char>` or `sk_test_<32-char>`
   - Scope-based permissions: read, write, trade, admin

**RBAC Hierarchy:**
| Level | Role | Description |
|-------|------|-------------|
| 6 | SUPER_ADMIN | Unrestricted access |
| 5 | ADMIN | User management, API key admin |
| 4 | PORTFOLIO_MANAGER | Trader + team management |
| 3 | TRADER | Analyst + trading signals |
| 2 | ANALYST | Read all analytics + write notes |
| 1 | VIEWER | Read-only access |

**Rate Limiting:**
| Category | Requests/min |
|----------|--------------|
| market_data | 100 |
| signals | 50 |
| analytics | 30 |
| options | 30 |
| etf | 30 |
| commodities | 30 |
| portfolio | 30 |
| research | 20 |
| macro | 20 |
| accounting | 10 |
| default | 60 |

---

## Integrations

### stanley.integrations.nautilus

NautilusTrader integration for algorithmic trading.

**Status:** Active
**Dependencies:** `nautilus_trader`
**Documentation:** See [NAUTILUS_INTEGRATION.md](NAUTILUS_INTEGRATION.md)

**Components:**
- `OpenBBDataClient`: NautilusTrader data client
- `MoneyFlowActor`: Money flow analysis actor
- `InstitutionalActor`: Institutional holdings actor
- `SmartMoneyIndicator`: Custom smart money indicator
- `InstitutionalMomentumIndicator`: Institutional momentum indicator

---

## Configuration

### stanley.config

Application configuration management.

**Status:** Active
**Dependencies:** `pydantic-settings`

**Key Configuration Files:**
- `config/stanley.yaml` - Main configuration
- `config/logging.py` - Logging configuration
- `config/metrics.py` - Metrics configuration

**Environment Variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `JWT_SECRET_KEY` | - | JWT signing key (required) |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | 15 | Access token TTL |
| `REFRESH_TOKEN_EXPIRE_DAYS` | 7 | Refresh token TTL |
| `JWT_ALGORITHM` | HS256 | JWT algorithm |
| `OPENBB_TOKEN` | - | OpenBB API token |
| `SEC_EDGAR_IDENTITY` | - | SEC EDGAR identity email |

---

## Cross-Module Integration Points

### Data Flow

```
External APIs (OpenBB, DBnomics, EDGAR)
              |
              v
    +------------------+
    |  stanley.data    |  <-- Caching, rate limiting
    +--------+---------+
             |
    +--------+---------+--------+---------+--------+
    |        |         |        |         |        |
    v        v         v        v         v        v
analytics portfolio research macro accounting commodities
    |        |         |        |         |        |
    +--------+---------+--------+---------+--------+
                       |
                       v
              +--------+--------+
              |    signals      |  <-- Multi-factor aggregation
              +--------+--------+
                       |
                       v
              +--------+--------+
              |  stanley.api    |  <-- REST endpoints
              +-----------------+
```

### Module Dependencies Matrix

| Module | data | analytics | portfolio | research | macro | accounting | commodities | signals |
|--------|------|-----------|-----------|----------|-------|------------|-------------|---------|
| analytics | X | - | - | - | - | - | - | - |
| portfolio | X | - | - | - | - | - | - | - |
| research | X | X | - | - | - | X | - | - |
| macro | X | - | - | - | - | - | - | - |
| accounting | - | - | - | - | - | - | - | - |
| commodities | X | - | - | - | X | - | - | - |
| signals | X | X | X | X | - | - | - | - |
| notes | - | - | - | - | - | - | - | - |

---

## Future Modules (Planned)

| Module | Status | Description |
|--------|--------|-------------|
| `stanley.ml` | Planned | Machine learning models |
| `stanley.alerts` | Planned | Real-time alerting system |
| `stanley.streaming` | Planned | WebSocket streaming |
| `stanley.backtesting` | Planned | Historical backtesting engine |
