# Stanley GUI Integration Map

## Executive Summary

This document maps all 80+ API endpoints to GUI views and components, identifying current coverage, gaps, and implementation priorities for the Rust GPUI-based stanley-gui.

**Updated**: The GUI now implements 40+ views with comprehensive navigation and keyboard support.

---

## 1. Current GUI Coverage Analysis

### 1.1 Implemented Views (40+ total)

The navigation system (`navigation.rs`) defines the following view categories:

| Category | Views | Status |
|----------|-------|--------|
| Markets & Overview | Dashboard, Watchlist | IMPLEMENTED |
| Flow Analytics | MoneyFlow, EquityFlow, DarkPool | IMPLEMENTED |
| Institutional | Institutional, ThirteenF | IMPLEMENTED |
| Options | OptionsFlow, OptionsGamma, OptionsUnusual, MaxPain | IMPLEMENTED |
| Research | Research, Valuation, Earnings, Peers | IMPLEMENTED |
| Portfolio | Portfolio, PortfolioRisk, Sectors | IMPLEMENTED |
| Macro | Macro, Commodities, CommodityCorrelations | IMPLEMENTED |
| ETF Analytics | ETFOverview, ETFFlows, SectorRotation, FactorRotation, Thematic | IMPLEMENTED |
| Accounting | Accounting, FinancialStatements, EarningsQuality, RedFlags | IMPLEMENTED |
| Signals | Signals, SignalBacktest, SignalPerformance | IMPLEMENTED |
| Notes | Notes, Theses, Trades, Events | IMPLEMENTED |
| Comparison | Comparison | IMPLEMENTED |
| Settings | Settings, Preferences, ApiConfig, Appearance | IMPLEMENTED |

### 1.2 Current API Client Methods (api.rs)

```rust
// Currently implemented:
health_check()           -> /api/health
get_market_data()        -> /api/market/{symbol}
get_sector_money_flow()  -> /api/money-flow
get_equity_flow()        -> /api/equity-flow/{symbol}
get_institutional()      -> /api/institutional/{symbol}
get_dark_pool()          -> /api/dark-pool/{symbol}
get_research()           -> /api/research/{symbol}
get_options_flow()       -> /api/options/{symbol}/flow
get_theses()             -> /api/theses
```

### 1.3 Keyboard Navigation (keyboard.rs)

Comprehensive keyboard system with 1348 lines of implementation:

| Category | Shortcuts |
|----------|-----------|
| View switching | 1-9 keys for main views |
| Symbol search | Ctrl+K, Cmd+K, Ctrl+P |
| Vim navigation | j/k/h/l, Ctrl+u/d |
| Table navigation | Arrows, PageUp/Down, Home/End, Space, Enter |
| Data actions | Ctrl+R refresh, Ctrl+E export |
| UI controls | Ctrl+T theme, Ctrl+B sidebar, F11 fullscreen |

---

## 2. Complete API Endpoint to GUI Mapping

### 2.1 System Endpoints

| Endpoint | GUI Location | Status |
|----------|-------------|--------|
| `GET /api/health` | Dashboard (status indicator) | IMPLEMENTED |

### 2.2 Market Data Endpoints

| Endpoint | GUI Location | Status |
|----------|-------------|--------|
| `GET /api/market/{symbol}` | Header bar (price display) | IMPLEMENTED |

### 2.3 Analytics Endpoints

| Endpoint | GUI Location | Status |
|----------|-------------|--------|
| `POST /api/money-flow` | MoneyFlow view | IMPLEMENTED |
| `GET /api/dark-pool/{symbol}` | DarkPool view | IMPLEMENTED |
| `GET /api/equity-flow/{symbol}` | Dashboard + MoneyFlow | IMPLEMENTED |

### 2.4 Portfolio Endpoints

| Endpoint | GUI Location | Status |
|----------|-------------|--------|
| `POST /api/portfolio-analytics` | Portfolio view | VIEW READY - API PENDING |

**Components available:**
- Holdings table with symbol, shares, cost, current price, market value, weight
- Portfolio metrics panel (total value, return, return %)
- Risk metrics panel (beta, VaR 95%, VaR 99%)
- Sector exposure visualization
- Performance attribution breakdown

### 2.5 Research Endpoints

| Endpoint | GUI Location | Status |
|----------|-------------|--------|
| `GET /api/research/{symbol}` | Research view | IMPLEMENTED |
| `GET /api/valuation/{symbol}` | Valuation view | VIEW READY |
| `GET /api/earnings/{symbol}` | Earnings view | VIEW READY |
| `GET /api/peers/{symbol}` | Peers view | VIEW READY |

### 2.6 Commodities Endpoints

| Endpoint | GUI Location | Status |
|----------|-------------|--------|
| `GET /api/commodities` | Commodities view | VIEW READY |
| `GET /api/commodities/{symbol}` | Commodities view | VIEW READY |
| `GET /api/commodities/{symbol}/macro` | Commodities view | VIEW READY |
| `GET /api/commodities/correlations` | CommodityCorrelations view | VIEW READY |

**Components available (commodities.rs):**
- Commodity market overview grid (energy, metals, agriculture)
- Individual commodity detail panel
- Macro linkage analysis panel
- Correlation matrix heatmap

### 2.7 Options Endpoints

| Endpoint | GUI Location | Status |
|----------|-------------|--------|
| `GET /api/options/{symbol}/flow` | OptionsFlow view | IMPLEMENTED |
| `GET /api/options/{symbol}/gamma` | OptionsGamma view | VIEW READY |
| `GET /api/options/{symbol}/unusual` | OptionsUnusual view | VIEW READY |
| `GET /api/options/{symbol}/put-call` | Options view | VIEW READY |
| `GET /api/options/{symbol}/smart-money` | Options view | VIEW READY |
| `GET /api/options/{symbol}/max-pain` | MaxPain view | VIEW READY |

### 2.8 ETF Analytics Endpoints

| Endpoint | GUI Location | Status |
|----------|-------------|--------|
| `GET /api/etf/flows` | ETFFlows view | VIEW READY |
| `GET /api/etf/flows/{symbol}` | ETFFlows view | VIEW READY |
| `GET /api/etf/sector-rotation` | SectorRotation view | VIEW READY |
| `GET /api/etf/sector-heatmap` | SectorRotation view | VIEW READY |
| `GET /api/etf/smart-beta` | ETFOverview view | VIEW READY |
| `GET /api/etf/factor-rotation` | FactorRotation view | VIEW READY |
| `GET /api/etf/thematic` | Thematic view | VIEW READY |
| `GET /api/etf/theme-dashboard` | Thematic view | VIEW READY |
| `GET /api/etf/institutional` | ETFOverview view | VIEW READY |
| `GET /api/etf/overview` | ETFOverview view | VIEW READY |

### 2.9 Enhanced Institutional Analytics

| Endpoint | GUI Location | Status |
|----------|-------------|--------|
| `GET /api/institutional/{symbol}` | Institutional view | IMPLEMENTED |
| `GET /api/institutional/{symbol}/changes` | Institutional view | API PENDING |
| `GET /api/institutional/{symbol}/whales` | Institutional view | API PENDING |
| `GET /api/institutional/{symbol}/sentiment` | Institutional view | API PENDING |
| `GET /api/institutional/{symbol}/clusters` | Institutional view | API PENDING |
| `GET /api/institutional/{symbol}/cross-filing` | ThirteenF view | API PENDING |
| `GET /api/institutional/{symbol}/momentum` | Institutional view | API PENDING |
| `GET /api/institutional/{symbol}/smart-money-flow` | Institutional view | API PENDING |
| `GET /api/institutional/alerts/new-positions` | Institutional view | API PENDING |
| `GET /api/institutional/alerts/coordinated-buying` | Institutional view | API PENDING |
| `GET /api/institutional/conviction-picks` | Institutional view | API PENDING |
| `GET /api/institutional/filing-calendar` | ThirteenF view | API PENDING |

### 2.10 Enhanced Money Flow Analytics

| Endpoint | GUI Location | Status |
|----------|-------------|--------|
| `GET /api/money-flow/{symbol}/alerts` | MoneyFlow/Dashboard | API PENDING |
| `GET /api/money-flow/{symbol}/block-trades` | MoneyFlow view | API PENDING |
| `GET /api/money-flow/sector-rotation` | MoneyFlow view | API PENDING |
| `GET /api/money-flow/{symbol}/smart-money` | MoneyFlow view | API PENDING |
| `GET /api/money-flow/{symbol}/unusual-volume` | MoneyFlow view | API PENDING |
| `GET /api/money-flow/{symbol}/momentum` | MoneyFlow view | API PENDING |
| `GET /api/money-flow/{symbol}/comprehensive` | MoneyFlow view | API PENDING |
| `GET /api/money-flow/alerts` | Dashboard (alerts panel) | API PENDING |
| `GET /api/money-flow/alerts/summary` | Dashboard | API PENDING |

### 2.11 Accounting Quality Endpoints

| Endpoint | GUI Location | Status |
|----------|-------------|--------|
| `GET /api/accounting/earnings-quality/{symbol}` | EarningsQuality view | VIEW READY |
| `GET /api/accounting/m-score/{symbol}` | EarningsQuality view | VIEW READY |
| `GET /api/accounting/f-score/{symbol}` | EarningsQuality view | VIEW READY |
| `GET /api/accounting/z-score/{symbol}` | EarningsQuality view | VIEW READY |
| `GET /api/accounting/red-flags/{symbol}` | RedFlags view | VIEW READY |
| `GET /api/accounting/red-flags/{symbol}/peers` | RedFlags view | VIEW READY |
| `GET /api/accounting/anomalies/{symbol}` | RedFlags view | VIEW READY |
| `GET /api/accounting/accruals/{symbol}` | EarningsQuality view | VIEW READY |
| `GET /api/accounting/comprehensive/{symbol}` | Accounting view | VIEW READY |

### 2.12 Notes Endpoints

| Endpoint | GUI Location | Status |
|----------|-------------|--------|
| `GET /api/notes` | Notes view | VIEW READY |
| `GET /api/notes/search` | Notes view | VIEW READY |
| `GET /api/notes/graph` | Notes view | VIEW READY |
| `GET /api/notes/{name}` | Notes view | VIEW READY |
| `GET /api/notes/{name}/backlinks` | Notes view | VIEW READY |
| `PUT /api/notes/{name}` | Notes view | VIEW READY |
| `DELETE /api/notes/{name}` | Notes view | VIEW READY |
| `GET /api/theses` | Theses view | IMPLEMENTED |
| `POST /api/theses` | Theses view | VIEW READY |
| `GET /api/trades` | Trades view | VIEW READY |
| `POST /api/trades` | Trades view | VIEW READY |
| `POST /api/trades/{name}/close` | Trades view | VIEW READY |
| `GET /api/trades/stats` | Trades view | VIEW READY |
| `GET /api/events` | Events view | VIEW READY |
| `POST /api/events` | Events view | VIEW READY |
| `GET /api/people` | Notes view | VIEW READY |
| `POST /api/people` | Notes view | VIEW READY |
| `GET /api/sectors` | Notes view | VIEW READY |
| `POST /api/sectors` | Notes view | VIEW READY |

### 2.13 Signals Endpoints

| Endpoint | GUI Location | Status |
|----------|-------------|--------|
| `GET /api/signals/{symbol}` | Signals view | VIEW READY |
| `POST /api/signals` | Signals view | VIEW READY |
| `GET /api/signals/{signal_id}/backtest` | SignalBacktest view | VIEW READY |
| `GET /api/signals/performance/stats` | SignalPerformance view | VIEW READY |
| `GET /api/signals/performance/history` | SignalPerformance view | VIEW READY |
| `POST /api/signals/batch` | Signals view | VIEW READY |

---

## 3. Navigation Architecture

### 3.1 View Enum (navigation.rs)

```rust
pub enum View {
    // Markets & Overview
    Dashboard,
    Watchlist,

    // Flow Analytics
    MoneyFlow,
    EquityFlow,
    DarkPool,

    // Institutional
    Institutional,
    ThirteenF,

    // Options
    OptionsFlow,
    OptionsGamma,
    OptionsUnusual,
    MaxPain,

    // Research
    Research,
    Valuation,
    Earnings,
    Peers,

    // Portfolio
    Portfolio,
    PortfolioRisk,
    Sectors,

    // Macro & Commodities
    Macro,
    Commodities,
    CommodityCorrelations,

    // ETF Analytics
    ETFOverview,
    ETFFlows,
    SectorRotation,
    FactorRotation,
    Thematic,

    // Accounting & Filings
    Accounting,
    FinancialStatements,
    EarningsQuality,
    RedFlags,

    // Signals & Alerts
    Signals,
    SignalBacktest,
    SignalPerformance,

    // Notes & Research
    Notes,
    Theses,
    Trades,
    Events,

    // Comparison
    Comparison,

    // Settings
    Settings,
    Preferences,
    ApiConfig,
    Appearance,
}
```

### 3.2 Navigation Sections (NavSection)

```rust
pub enum NavSection {
    Markets,
    FlowAnalytics,
    Institutional,
    Options,
    Research,
    Portfolio,
    Macro,
    ETF,
    Accounting,
    Signals,
    Notes,
    Settings,
}
```

---

## 4. Keyboard System (keyboard.rs)

### 4.1 Key Bindings

```rust
// View Navigation
"1" => View::Dashboard
"2" => View::MoneyFlow
"3" => View::Institutional
"4" => View::DarkPool
"5" => View::Options
"6" => View::Research
"7" => View::Portfolio
"8" => View::Commodities
"9" => View::Macro

// Symbol Search
"ctrl+k" | "cmd+k" => OpenSymbolSearch
"ctrl+p" => QuickSymbolLookup

// Vim Navigation
"j" => MoveDown
"k" => MoveUp
"h" => MoveLeft
"l" => MoveRight
"ctrl+u" => HalfPageUp
"ctrl+d" => HalfPageDown
"g g" => GoToTop
"shift+g" => GoToBottom

// Table Navigation
"up" | "down" | "left" | "right" => Navigate
"pageup" | "pagedown" => PageScroll
"home" => FirstRow
"end" => LastRow
"space" => SelectToggle
"enter" => OpenDetail

// Data Actions
"ctrl+r" => Refresh
"ctrl+shift+r" => RefreshAll
"ctrl+e" => Export
"ctrl+s" => Save

// UI Controls
"ctrl+t" => ToggleTheme
"ctrl+b" => ToggleSidebar
"f11" => ToggleFullscreen
"escape" => CloseModal

// Help
"shift+?" => ShowShortcuts
"f1" => OpenHelp
```

---

## 5. Component Architecture

### 5.1 Core Components (components/)

| Component | File | Purpose |
|-----------|------|---------|
| Sidebar | sidebar.rs | Navigation menu |
| Header | header.rs | Top bar with search, symbol display |
| Tables | tables.rs | Data tables with sorting, filtering |
| Charts | charts.rs | Visualization components |
| Modals | modals.rs | Dialog windows |
| Dashboard | dashboard.rs | Dashboard-specific widgets |

### 5.2 Form Components (components/forms/)

| Component | File | Purpose |
|-----------|------|---------|
| TextInput | text_input.rs | Text field input |
| NumberInput | number_input.rs | Numeric input with validation |
| Validation | validation.rs | Form validation utilities |

---

## 6. API Client Extensions Needed

```rust
// Portfolio
get_portfolio_analytics() -> /api/portfolio-analytics

// Options
get_gamma_exposure() -> /api/options/{symbol}/gamma
get_unusual_options() -> /api/options/{symbol}/unusual
get_max_pain() -> /api/options/{symbol}/max-pain

// ETF
get_etf_flows() -> /api/etf/flows
get_sector_rotation() -> /api/etf/sector-rotation
get_etf_overview() -> /api/etf/overview

// Accounting
get_earnings_quality() -> /api/accounting/earnings-quality/{symbol}
get_red_flags() -> /api/accounting/red-flags/{symbol}

// Commodities
get_commodities_overview() -> /api/commodities
get_commodity_detail() -> /api/commodities/{symbol}
get_commodity_correlations() -> /api/commodities/correlations

// Signals
get_signals() -> /api/signals/{symbol}
generate_signal() -> /api/signals
get_signal_backtest() -> /api/signals/{signal_id}/backtest

// Notes (extend existing)
get_notes() -> /api/notes
search_notes() -> /api/notes/search
get_trades() -> /api/trades
get_events() -> /api/events
```

---

## 7. Summary Statistics

| Category | Views | API Endpoints | Connected |
|----------|-------|---------------|-----------|
| Markets | 2 | 2 | 2 |
| Flow Analytics | 3 | 12 | 3 |
| Institutional | 2 | 12 | 1 |
| Options | 4 | 6 | 1 |
| Research | 4 | 4 | 1 |
| Portfolio | 3 | 1 | 0 |
| Macro | 3 | 4 | 0 |
| ETF | 5 | 10 | 0 |
| Accounting | 4 | 9 | 0 |
| Signals | 3 | 6 | 0 |
| Notes | 4 | 19 | 1 |
| Comparison | 1 | - | - |
| Settings | 4 | - | - |
| **TOTAL** | **42** | **85** | **9** |

**Current Status:**
- 42 views implemented in navigation
- 9 API endpoints connected (10.6% API coverage)
- Comprehensive keyboard navigation system
- Full view routing infrastructure ready

**Priority for API Connection:**
1. Portfolio endpoints (risk metrics, holdings)
2. Commodities endpoints (already have view)
3. Options gamma/unusual/max-pain
4. ETF flows and rotation
5. Accounting quality scores

---

## 8. Build Requirements

- **Platform**: Linux with Wayland only (X11 removed)
- **Rust**: 1.70+
- **Backend**: Stanley Python API on localhost:8000

```bash
cd stanley-gui
cargo build --release
./target/release/stanley-gui
```
