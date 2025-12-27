# Stanley GUI Integration Map

## Executive Summary

This document maps all 80+ API endpoints to GUI views and components, identifying current coverage, gaps, and implementation priorities for the Rust GPUI-based stanley-gui.

---

## 1. Current GUI Coverage Analysis

### 1.1 Existing Views (6 total)

| View | Status | API Endpoints Used |
|------|--------|-------------------|
| Dashboard | Implemented | `/api/health`, `/api/market/{symbol}`, `/api/equity-flow/{symbol}`, `/api/money-flow`, `/api/institutional/{symbol}` |
| MoneyFlow | Implemented | `/api/equity-flow/{symbol}`, `/api/money-flow` |
| Institutional | Implemented | `/api/institutional/{symbol}` |
| DarkPool | Implemented | `/api/dark-pool/{symbol}` |
| Options | Implemented | `/api/options/{symbol}/flow` |
| Research | Implemented | `/api/research/{symbol}` (partial - valuation & DCF only) |

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
```

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

### 2.4 Portfolio Endpoints - NEW VIEW NEEDED

| Endpoint | Proposed GUI Location | Status |
|----------|----------------------|--------|
| `POST /api/portfolio-analytics` | **Portfolio view** (new) | NOT IMPLEMENTED |

**Components needed:**
- Holdings table with symbol, shares, cost, current price, market value, weight
- Portfolio metrics panel (total value, return, return %)
- Risk metrics panel (beta, VaR 95%, VaR 99%)
- Sector exposure pie chart or bar chart
- Performance attribution breakdown

### 2.5 Research Endpoints

| Endpoint | GUI Location | Status |
|----------|-------------|--------|
| `GET /api/research/{symbol}` | Research view | PARTIAL (valuation + DCF) |
| `GET /api/valuation/{symbol}` | Research view | Missing dedicated component |
| `GET /api/earnings/{symbol}` | Research view | NOT IMPLEMENTED |
| `GET /api/peers/{symbol}` | Research view | NOT IMPLEMENTED |

**Components needed in Research view:**
- Earnings history chart (quarters with beats/misses)
- Earnings surprise histogram
- Peer comparison table
- Growth metrics panel

### 2.6 Commodities Endpoints - NEW VIEW NEEDED

| Endpoint | Proposed GUI Location | Status |
|----------|----------------------|--------|
| `GET /api/commodities` | **Commodities view** (new) | NOT IMPLEMENTED |
| `GET /api/commodities/{symbol}` | Commodities view | NOT IMPLEMENTED |
| `GET /api/commodities/{symbol}/macro` | Commodities view | NOT IMPLEMENTED |
| `GET /api/commodities/correlations` | Commodities view | NOT IMPLEMENTED |

**Components needed:**
- Commodity market overview grid (energy, metals, agriculture)
- Individual commodity detail panel
- Macro linkage analysis panel
- Correlation matrix heatmap

### 2.7 Options Endpoints

| Endpoint | GUI Location | Status |
|----------|-------------|--------|
| `GET /api/options/{symbol}/flow` | Options view | IMPLEMENTED |
| `GET /api/options/{symbol}/gamma` | Options view | NOT IMPLEMENTED |
| `GET /api/options/{symbol}/unusual` | Options view | NOT IMPLEMENTED |
| `GET /api/options/{symbol}/put-call` | Options view | NOT IMPLEMENTED |
| `GET /api/options/{symbol}/smart-money` | Options view | NOT IMPLEMENTED |
| `GET /api/options/{symbol}/max-pain` | Options view | NOT IMPLEMENTED |

**Components needed in Options view:**
- Gamma exposure chart (strike vs GEX)
- Unusual activity table
- Put/Call analysis panel
- Smart money trades table
- Max pain indicator with expiration selector

### 2.8 ETF Analytics Endpoints - NEW VIEW NEEDED

| Endpoint | Proposed GUI Location | Status |
|----------|----------------------|--------|
| `GET /api/etf/flows` | **ETF view** (new) | NOT IMPLEMENTED |
| `GET /api/etf/flows/{symbol}` | ETF view | NOT IMPLEMENTED |
| `GET /api/etf/sector-rotation` | ETF view | NOT IMPLEMENTED |
| `GET /api/etf/sector-heatmap` | ETF view | NOT IMPLEMENTED |
| `GET /api/etf/smart-beta` | ETF view | NOT IMPLEMENTED |
| `GET /api/etf/factor-rotation` | ETF view | NOT IMPLEMENTED |
| `GET /api/etf/thematic` | ETF view | NOT IMPLEMENTED |
| `GET /api/etf/theme-dashboard` | ETF view | NOT IMPLEMENTED |
| `GET /api/etf/institutional` | ETF view | NOT IMPLEMENTED |
| `GET /api/etf/overview` | ETF view | NOT IMPLEMENTED |

**Components needed:**
- ETF flow overview dashboard
- Sector rotation diagram
- Sector heatmap (color-coded performance)
- Smart beta factor comparison
- Thematic flows leaderboard
- Creation/redemption activity chart

### 2.9 Enhanced Institutional Analytics

| Endpoint | GUI Location | Status |
|----------|-------------|--------|
| `GET /api/institutional/{symbol}` | Institutional view | IMPLEMENTED |
| `GET /api/institutional/{symbol}/changes` | Institutional view | NOT IMPLEMENTED |
| `GET /api/institutional/{symbol}/whales` | Institutional view | NOT IMPLEMENTED |
| `GET /api/institutional/{symbol}/sentiment` | Institutional view | NOT IMPLEMENTED |
| `GET /api/institutional/{symbol}/clusters` | Institutional view | NOT IMPLEMENTED |
| `GET /api/institutional/{symbol}/cross-filing` | Institutional view | NOT IMPLEMENTED |
| `GET /api/institutional/{symbol}/momentum` | Institutional view | NOT IMPLEMENTED |
| `GET /api/institutional/{symbol}/smart-money-flow` | Institutional view | NOT IMPLEMENTED |
| `GET /api/institutional/alerts/new-positions` | Institutional view | NOT IMPLEMENTED |
| `GET /api/institutional/alerts/coordinated-buying` | Institutional view | NOT IMPLEMENTED |
| `GET /api/institutional/conviction-picks` | Institutional view | NOT IMPLEMENTED |
| `GET /api/institutional/filing-calendar` | Institutional view | NOT IMPLEMENTED |

**Components needed in Institutional view:**
- Quarter-over-quarter changes table
- Whale accumulation alerts panel
- Sentiment score gauge
- Position clusters visualization
- Cross-filing consensus indicator
- Smart money momentum chart
- Conviction picks leaderboard
- 13F filing calendar

### 2.10 Enhanced Money Flow Analytics

| Endpoint | GUI Location | Status |
|----------|-------------|--------|
| `GET /api/money-flow/{symbol}/alerts` | MoneyFlow/Dashboard | NOT IMPLEMENTED |
| `GET /api/money-flow/{symbol}/block-trades` | MoneyFlow view | NOT IMPLEMENTED |
| `GET /api/money-flow/sector-rotation` | MoneyFlow view | NOT IMPLEMENTED |
| `GET /api/money-flow/{symbol}/smart-money` | MoneyFlow view | NOT IMPLEMENTED |
| `GET /api/money-flow/{symbol}/unusual-volume` | MoneyFlow view | NOT IMPLEMENTED |
| `GET /api/money-flow/{symbol}/momentum` | MoneyFlow view | NOT IMPLEMENTED |
| `GET /api/money-flow/{symbol}/comprehensive` | MoneyFlow view | NOT IMPLEMENTED |
| `GET /api/money-flow/alerts` | Dashboard (alerts panel) | NOT IMPLEMENTED |
| `GET /api/money-flow/alerts/summary` | Dashboard | NOT IMPLEMENTED |

**Components needed:**
- Dark pool alerts panel
- Block trades table
- Sector rotation diagram
- Smart money tracking indicator
- Unusual volume indicator
- Flow momentum chart
- Global alerts summary widget

### 2.11 Accounting Quality Endpoints - NEW VIEW NEEDED

| Endpoint | Proposed GUI Location | Status |
|----------|----------------------|--------|
| `GET /api/accounting/earnings-quality/{symbol}` | **Accounting view** (new) | NOT IMPLEMENTED |
| `GET /api/accounting/m-score/{symbol}` | Accounting view | NOT IMPLEMENTED |
| `GET /api/accounting/f-score/{symbol}` | Accounting view | NOT IMPLEMENTED |
| `GET /api/accounting/z-score/{symbol}` | Accounting view | NOT IMPLEMENTED |
| `GET /api/accounting/red-flags/{symbol}` | Accounting view | NOT IMPLEMENTED |
| `GET /api/accounting/red-flags/{symbol}/peers` | Accounting view | NOT IMPLEMENTED |
| `GET /api/accounting/anomalies/{symbol}` | Accounting view | NOT IMPLEMENTED |
| `GET /api/accounting/accruals/{symbol}` | Accounting view | NOT IMPLEMENTED |
| `GET /api/accounting/comprehensive/{symbol}` | Accounting view | NOT IMPLEMENTED |

**Components needed:**
- Earnings quality scorecard (M/F/Z scores)
- Manipulation risk gauge
- Red flags list with severity indicators
- Anomaly detection panel
- Accrual quality indicator
- Peer comparison table
- Overall quality rating badge

### 2.12 Notes Endpoints - NEW VIEW NEEDED

| Endpoint | Proposed GUI Location | Status |
|----------|----------------------|--------|
| `GET /api/notes` | **Notes view** (new) | NOT IMPLEMENTED |
| `GET /api/notes/search` | Notes view | NOT IMPLEMENTED |
| `GET /api/notes/graph` | Notes view | NOT IMPLEMENTED |
| `GET /api/notes/{name}` | Notes view | NOT IMPLEMENTED |
| `GET /api/notes/{name}/backlinks` | Notes view | NOT IMPLEMENTED |
| `PUT /api/notes/{name}` | Notes view | NOT IMPLEMENTED |
| `DELETE /api/notes/{name}` | Notes view | NOT IMPLEMENTED |
| `GET /api/theses` | Notes view | NOT IMPLEMENTED |
| `POST /api/theses` | Notes view | NOT IMPLEMENTED |
| `GET /api/trades` | Notes view | NOT IMPLEMENTED |
| `POST /api/trades` | Notes view | NOT IMPLEMENTED |
| `POST /api/trades/{name}/close` | Notes view | NOT IMPLEMENTED |
| `GET /api/trades/stats` | Notes view | NOT IMPLEMENTED |
| `GET /api/events` | Notes view | NOT IMPLEMENTED |
| `POST /api/events` | Notes view | NOT IMPLEMENTED |
| `GET /api/people` | Notes view | NOT IMPLEMENTED |
| `POST /api/people` | Notes view | NOT IMPLEMENTED |
| `GET /api/sectors` | Notes view | NOT IMPLEMENTED |
| `POST /api/sectors` | Notes view | NOT IMPLEMENTED |

**Components needed:**
- Notes list with type filters
- Full-text search bar
- Note graph visualization (nodes + edges)
- Note detail/editor panel
- Investment thesis tracker
- Trade journal with statistics
- Events calendar
- People directory
- Sector overview cards

### 2.13 Signals Endpoints - NEW VIEW NEEDED

| Endpoint | Proposed GUI Location | Status |
|----------|----------------------|--------|
| `GET /api/signals/{symbol}` | **Signals view** (new) | NOT IMPLEMENTED |
| `POST /api/signals` | Signals view | NOT IMPLEMENTED |
| `GET /api/signals/{signal_id}/backtest` | Signals view | NOT IMPLEMENTED |
| `GET /api/signals/performance/stats` | Signals view | NOT IMPLEMENTED |
| `GET /api/signals/performance/history` | Signals view | NOT IMPLEMENTED |
| `POST /api/signals/batch` | Signals view | NOT IMPLEMENTED |

**Components needed:**
- Active signals list
- Signal generation controls
- Backtest results panel
- Performance statistics dashboard
- Historical performance chart
- Batch signal generator

---

## 3. Missing Views Summary

| View | Priority | Endpoints | Complexity |
|------|----------|-----------|------------|
| Portfolio | HIGH | 1 | Medium |
| ETF | HIGH | 10 | High |
| Accounting | HIGH | 9 | Medium |
| Notes | MEDIUM | 19 | High |
| Signals | MEDIUM | 6 | Medium |
| Commodities | LOW | 4 | Medium |

---

## 4. Component Specifications

### 4.1 Portfolio View

```
+------------------------------------------+
|  PORTFOLIO ANALYTICS                      |
+------------------------------------------+
| [Holdings Input Form - symbols/shares]   |
+------------------------------------------+
| Total Value: $XXX,XXX | Return: +X.XX%   |
+------------------------------------------+
| RISK METRICS        | SECTOR EXPOSURE    |
| Beta: X.XX          | [Pie Chart]        |
| VaR 95%: -X.XX%     |                    |
| VaR 99%: -X.XX%     |                    |
+------------------------------------------+
| TOP HOLDINGS TABLE                       |
| Symbol | Shares | Cost | Price | Weight  |
+------------------------------------------+
```

### 4.2 ETF View

```
+------------------------------------------+
|  ETF FLOW ANALYTICS                       |
+------------------------------------------+
| [Sector Heatmap Grid]                    |
+------------------------------------------+
| ROTATION SIGNALS    | THEMATIC FLOWS     |
| Leading: XLK, XLC   | AI: +$500M         |
| Lagging: XLE, XLU   | Clean: -$200M      |
+------------------------------------------+
| FACTOR ROTATION     | SMART BETA         |
| Value -> Growth     | Mom: Overweight    |
+------------------------------------------+
| [Creation/Redemption Activity Chart]     |
+------------------------------------------+
```

### 4.3 Accounting View

```
+------------------------------------------+
|  ACCOUNTING QUALITY - {SYMBOL}           |
+------------------------------------------+
| OVERALL RATING: [BADGE] Score: X.XX      |
+------------------------------------------+
| M-SCORE    | F-SCORE    | Z-SCORE        |
| X.XX       | X/9        | X.XX           |
| [Risk]     | [Strong]   | [Safe Zone]    |
+------------------------------------------+
| RED FLAGS                                |
| [!] Revenue recognition...  HIGH         |
| [!] Accrual anomaly...      MEDIUM       |
+------------------------------------------+
| ANOMALIES DETECTED: X                    |
| Benford Score: X.XX | Disclosure: X.XX   |
+------------------------------------------+
```

### 4.4 Notes View

```
+------------------------------------------+
|  INVESTMENT NOTES                         |
+------------------------------------------+
| [Search] [+Thesis] [+Trade] [+Event]     |
+------------------------------------------+
| SIDEBAR          | NOTE DETAIL           |
| - Theses (5)     | # AAPL Investment...  |
| - Trades (12)    |                       |
| - Events (3)     | ## Key Points         |
| - People (8)     | - ...                 |
| - Sectors (4)    |                       |
+------------------------------------------+
| [Note Graph Visualization]               |
+------------------------------------------+
```

### 4.5 Enhanced Options View

```
+------------------------------------------+
|  OPTIONS FLOW - {SYMBOL}                  |
+------------------------------------------+
| Call Vol: XXX,XXX | Put Vol: XXX,XXX     |
| P/C Ratio: X.XX   | Sentiment: BULLISH   |
+------------------------------------------+
| GAMMA EXPOSURE    | MAX PAIN             |
| [Strike Chart]    | Strike: $XXX         |
|                   | Expiry: YYYY-MM-DD   |
+------------------------------------------+
| UNUSUAL ACTIVITY                         |
| Strike | Exp | Type | Vol | OI | Premium |
+------------------------------------------+
| SMART MONEY TRADES                       |
| [Block trades table with sentiment]      |
+------------------------------------------+
```

### 4.6 Enhanced Institutional View

```
+------------------------------------------+
|  INSTITUTIONAL ANALYTICS - {SYMBOL}       |
+------------------------------------------+
| Sentiment: [Gauge] | Momentum: +X.XX     |
+------------------------------------------+
| TOP HOLDERS        | Q/Q CHANGES         |
| [Existing table]   | [New/Increased/Sold]|
+------------------------------------------+
| WHALE ALERTS                             |
| [List of large position changes]         |
+------------------------------------------+
| CONVICTION PICKS   | FILING CALENDAR     |
| [Top weighted]     | [Upcoming deadlines]|
+------------------------------------------+
```

### 4.7 Signals View

```
+------------------------------------------+
|  TRADING SIGNALS                          |
+------------------------------------------+
| [Generate Signal Button] [Backtest]      |
+------------------------------------------+
| ACTIVE SIGNALS                           |
| Symbol | Type | Signal | Confidence      |
+------------------------------------------+
| PERFORMANCE STATS                        |
| Win Rate: XX% | Avg Return: X.XX%        |
| Sharpe: X.XX  | Max DD: -X.XX%           |
+------------------------------------------+
| [Historical Performance Chart]           |
+------------------------------------------+
```

---

## 5. Implementation Priority Order

### Phase 1: High Value (Week 1-2)
1. **Portfolio View** - Core investment tracking
2. **Enhanced Options View** - Add gamma, unusual, max-pain
3. **Enhanced Institutional View** - Add changes, whales, sentiment

### Phase 2: Analytics Expansion (Week 3-4)
4. **ETF View** - Sector rotation and flows
5. **Accounting View** - Quality scoring
6. **Enhanced MoneyFlow View** - Alerts and comprehensive analysis

### Phase 3: Research & Notes (Week 5-6)
7. **Enhanced Research View** - Add earnings, peers
8. **Signals View** - Signal generation and backtesting
9. **Notes View** - Investment thesis management

### Phase 4: Completion (Week 7-8)
10. **Commodities View** - Commodity analysis
11. **Dashboard Enhancement** - Global alerts integration
12. **Cross-view Linking** - Navigate between related data

---

## 6. API Client Extensions Needed

```rust
// Phase 1
get_portfolio_analytics() -> /api/portfolio-analytics
get_gamma_exposure() -> /api/options/{symbol}/gamma
get_unusual_options() -> /api/options/{symbol}/unusual
get_max_pain() -> /api/options/{symbol}/max-pain
get_institutional_changes() -> /api/institutional/{symbol}/changes
get_whale_accumulation() -> /api/institutional/{symbol}/whales
get_institutional_sentiment() -> /api/institutional/{symbol}/sentiment

// Phase 2
get_etf_flows() -> /api/etf/flows
get_sector_rotation() -> /api/etf/sector-rotation
get_sector_heatmap() -> /api/etf/sector-heatmap
get_earnings_quality() -> /api/accounting/earnings-quality/{symbol}
get_red_flags() -> /api/accounting/red-flags/{symbol}
get_comprehensive_money_flow() -> /api/money-flow/{symbol}/comprehensive
get_money_flow_alerts() -> /api/money-flow/alerts

// Phase 3
get_earnings() -> /api/earnings/{symbol}
get_peers() -> /api/peers/{symbol}
get_signals() -> /api/signals/{symbol}
generate_signals() -> /api/signals
get_notes() -> /api/notes
search_notes() -> /api/notes/search

// Phase 4
get_commodities_overview() -> /api/commodities
get_commodity_detail() -> /api/commodities/{symbol}
```

---

## 7. Data Types to Add (api.rs)

```rust
// Portfolio
struct PortfolioAnalytics { total_value, return_pct, beta, var_95, var_99, sector_exposure, holdings }
struct PortfolioHolding { symbol, shares, avg_cost, current_price, market_value, weight }

// Options Extensions
struct GammaExposure { total_gex, call_gex, put_gex, flip_point, max_gamma_strike }
struct UnusualActivity { strike, expiration, option_type, volume, oi, vol_oi_ratio, premium, sentiment }
struct MaxPainData { max_pain, total_call_oi, total_put_oi, gamma_concentration, pin_risk }

// Institutional Extensions
struct InstitutionalChanges { manager_name, change_type, shares_current, shares_previous, conviction_score }
struct WhaleAlert { institution_name, change_type, magnitude, estimated_aum, alert_level }
struct InstitutionalSentiment { score, classification, confidence, contributing_factors }

// ETF
struct EtfFlow { symbol, net_flow, momentum, institutional_activity }
struct SectorRotation { sector, relative_strength, momentum_score, rotation_phase, recommendation }
struct SectorHeatmapData { sector, performance, color }

// Accounting
struct EarningsQuality { m_score, f_score, z_score, overall_rating, red_flags }
struct RedFlag { category, description, severity, metric, value, threshold }

// Notes
struct Note { name, note_type, tags, created, modified, content }
struct TradeJournal { symbol, direction, entry_price, exit_price, pnl, grade }

// Signals
struct Signal { symbol, signal_type, direction, confidence, timestamp }
struct PerformanceStats { win_rate, avg_return, sharpe_ratio, max_drawdown }
```

---

## 8. View Registration (app.rs)

```rust
pub enum ActiveView {
    Dashboard,      // Existing
    MoneyFlow,      // Existing
    Institutional,  // Existing
    DarkPool,       // Existing
    Options,        // Existing
    Research,       // Existing
    Portfolio,      // NEW
    ETF,           // NEW
    Accounting,    // NEW
    Notes,         // NEW
    Signals,       // NEW
    Commodities,   // NEW
}
```

---

## Summary Statistics

| Category | Total Endpoints | Implemented | Coverage |
|----------|-----------------|-------------|----------|
| System | 1 | 1 | 100% |
| Market Data | 1 | 1 | 100% |
| Analytics | 3 | 3 | 100% |
| Portfolio | 1 | 0 | 0% |
| Research | 4 | 1 | 25% |
| Commodities | 4 | 0 | 0% |
| Options | 6 | 1 | 17% |
| ETF Analytics | 10 | 0 | 0% |
| Institutional (Enhanced) | 12 | 1 | 8% |
| Money Flow (Enhanced) | 9 | 0 | 0% |
| Accounting Quality | 9 | 0 | 0% |
| Notes | 19 | 0 | 0% |
| Signals | 6 | 0 | 0% |
| **TOTAL** | **85** | **8** | **9.4%** |

**Current Implementation: 8 endpoints connected to GUI out of 85 total (9.4% coverage)**
