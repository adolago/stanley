# Bloomberg Terminal Institutional Analytics Research

## Research Summary

This document analyzes Bloomberg Terminal's institutional analytics features to identify key capabilities that Stanley should match or exceed. Research conducted by the Stanley Hive Mind Research Agent.

---

## 1. Bloomberg Terminal Key Features Analysis

### 1.1 Fund Flow Data (FLOD)

**Bloomberg Capabilities:**
- Global ETP Flows Data covering 80,000+ active ETP tickers across 90+ exchanges
- 18,000+ primary share classes representing $19.7+ trillion in AUM
- 18 years of historical data for backtesting (since 2007)
- Real-time flow tracking for gauging sentiment
- Capital inflow/outflow tracking
- Potential delisting/issuer pivot identification
- Product development and sales strategy inputs
- Quantitative model and systematic signal inputs
- Regulatory risk oversight

**Data Delivery:**
- Bloomberg Terminal
- Data License (SFTP, REST API, cloud delivery)
- Enterprise Products

**Integration Points:**
- Bloomberg Dividend Forecasts (BDVD) for ETF distribution estimates
- Bloomberg Fund Risk & Sustainability Analytics for portfolio-level views
- Duration, liquidity, credit, and sustainability metrics

### 1.2 13F Holdings Analysis (DES/FA/OWN/PHDC)

**Bloomberg Functions:**
- `OWN` - Institutional ownership data
- `PHDC` - Portfolio holdings data
- `DES` - Security description with institutional ownership percentage
- `FLNG` - Fund manager holdings lookup

**Key Metrics Provided:**
- Percent of institutional holding
- Business breakdown and geographic breakdown
- Non-cash working capital and net debt issued
- Mutual fund holdings (13F data)
- Change in holding since last filing date
- Largest stockholders in company
- Market cap information
- 100,000+ fund coverage across all asset classes

### 1.3 Options Analytics (OPTN/OPX/BTM/OVME)

**Bloomberg Functions:**
- `OPX` - Real-time options trading activity and volume analytics
- `BTM` - Block trade monitoring
- `OVME` - Options trade structuring
- `GN` - News activity chart

**Key Capabilities:**
- Open interest distribution analysis by expiration date
- Block trade activity detection
- News sentiment integration
- Technical level analysis
- Turnover ratio analysis (open interest / average volumes)
- Day trader/HFT flow identification (low turnover ratio)
- Institutional flow identification (high turnover ratio)
- Pre-trade analytics for directional strategies

### 1.4 Quantitative Risk Management (MARS/QRM)

**Bloomberg MARS (Multi-Asset Risk System):**
- Value-at-Risk (VaR) calculation
- Portfolio stress testing
- Custom risk model creation
- Monte Carlo simulations
- Scenario analysis
- Regression analysis and time-series modeling
- Real-time data integration with predictive models

### 1.5 Institutional Movers/Dark Pool

**Bloomberg Tradebook Features:**
- Multi-venue liquidity sourcing including dark pools
- Latency and toxicity monitoring per venue/stock
- Tradebook Cross negotiating platform (WUD)
- Auto execution functionality (WUDx)

**Market Reality:**
- More shares now trade in dark pools than on NYSE
- 30-40% of heavily traded stock volume occurs in dark pools
- Dark pools designed for large institutional block trades
- No public order book visibility

---

## 2. Available Data Sources (Free and Paid)

### 2.1 SEC EDGAR (Free)

**13F Holdings Data:**
- Source: https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets
- Coverage: All institutional managers with $100M+ AUM
- Update Frequency: Quarterly (45 days after quarter end)
- Data Points: Issuer name, CUSIP, share count, value, voting authority
- Historical Data: Available since 1999

**Implementation Options:**
- Direct SEC EDGAR parsing (free, requires development)
- sec-api.io (commercial, $49-499/month)
- Kaleidoscope kscope.io (commercial)
- Earningsfeed (free tier: 15 requests/minute)

**Parsing Considerations:**
- Post-2013Q2: XML format (straightforward parsing)
- Pre-2013Q2: Heterogeneous formats (complex parsing)

### 2.2 FINRA ATS Dark Pool Data (Free)

**Source:** https://www.finra.org/filing-reporting/otc-transparency

**Data Available:**
- Aggregate trade data by ATS and member firms
- Total shares traded, number of trades, average trade size
- OTC transparency for ATSs and non-ATS market makers

**Update Delays:**
- Tier 1 NMS stocks (S&P 500, Russell 1000): 2-week delay
- Tier 2 NMS stocks and OTC equities: 4-week delay

**Data Since:** 2014 (ATSs), 2016 (OTC market makers)

### 2.3 Options Flow Data Sources

**Free/Freemium:**
- OptionStrat Flow: 15-minute delayed, 10% of total flow (free tier)
- InsiderFinance: Real-time options flow dashboard
- Barchart Options Flow: Top 100 trades per underlying (limited free)

**Paid Services:**
- FlowAlgo: $149/month (2-week trial $37)
- CheddarFlow: AI-driven unusual activity alerts
- Unusual Whales: Dark pool + options flow API
- WhaleStream: Real-time options flow + dark pool monitoring

**API Access:**
- OptionData.io Smart Option Flow Data API
- Unusual Whales Public API

### 2.4 Short Interest Data

**FINRA Data:**
- Bi-monthly short interest reports
- Settlement date data

**Commercial Sources:**
- S3 Partners
- Ortex
- OpenBB (via various providers)

### 2.5 Insider Trading (SEC Form 4)

**Source:** SEC EDGAR
- Real-time filing updates
- Transaction details, ownership changes
- Officer/director identifications

---

## 3. Specific Metrics to Implement

### 3.1 Fund Flow Metrics (Priority: HIGH)

| Metric | Description | Data Source |
|--------|-------------|-------------|
| Net Fund Flow | Daily/weekly ETF creation/redemption | ETF issuers, OpenBB |
| Flow Momentum | Rate of change in flows | Calculated |
| Flow vs. Price Divergence | Flow trend vs. price trend | Calculated |
| Sector Rotation Signal | Cross-sector flow comparison | Calculated |
| AUM-Weighted Flow | Size-adjusted flow impact | Calculated |

### 3.2 13F Analytics Metrics (Priority: HIGH)

| Metric | Description | Data Source |
|--------|-------------|-------------|
| Institutional Ownership % | Total institutional stake | SEC EDGAR 13F |
| QoQ Ownership Change | Quarter-over-quarter delta | SEC EDGAR 13F |
| New Position Detection | First-time institutional buys | SEC EDGAR 13F |
| Exit Position Detection | Complete position liquidations | SEC EDGAR 13F |
| Concentration Risk (HHI) | Ownership concentration index | Calculated |
| Smart Money Score | Weighted manager performance | Calculated |
| Conviction Picks | High-confidence positions | Calculated |
| Portfolio Overlap | Cross-manager position analysis | SEC EDGAR 13F |
| Filing Calendar | Expected filing dates | Calculated |
| Manager Performance | Historical return tracking | Calculated |

### 3.3 Options Flow Metrics (Priority: HIGH)

| Metric | Description | Data Source |
|--------|-------------|-------------|
| Unusual Options Activity | Volume vs. open interest anomaly | Options providers |
| Put/Call Ratio | Sentiment indicator | Options providers |
| Options Volume Surge | Significant volume spikes | Options providers |
| Block Trade Detection | Large institutional orders | Options providers |
| Sweep Detection | Multi-exchange rapid fills | Options providers |
| Max Pain Analysis | Options expiration price target | Calculated |
| Gamma Exposure | Market maker hedging pressure | Calculated |
| Implied Volatility Skew | Put vs. call IV differential | Calculated |

### 3.4 Dark Pool Metrics (Priority: MEDIUM)

| Metric | Description | Data Source |
|--------|-------------|-------------|
| Dark Pool % of Volume | Off-exchange trading ratio | FINRA ATS |
| Large Block Activity | Block trade frequency | FINRA ATS |
| Dark Pool Levels | Price levels with high activity | Calculated |
| Institutional Accumulation Signal | Sustained dark pool buying | Calculated |
| ATS Venue Distribution | Volume by dark pool | FINRA ATS |

### 3.5 Risk Metrics (Priority: MEDIUM)

| Metric | Description | Data Source |
|--------|-------------|-------------|
| Value-at-Risk (VaR) | Portfolio loss probability | Calculated |
| Conditional VaR (CVaR) | Expected loss beyond VaR | Calculated |
| Portfolio Beta | Market sensitivity | Calculated |
| Sector Exposure | Portfolio sector weights | Calculated |
| Factor Exposure | Multi-factor risk decomposition | Calculated |
| Stress Test Scenarios | Historical/hypothetical shocks | Calculated |

---

## 4. Priority Ranking for Implementation

### Phase 1: Core Institutional Analytics (Immediate)

1. **13F Holdings Analysis** (HIGH)
   - Already partially implemented in `stanley/analytics/institutional.py`
   - Need: SEC EDGAR API integration for real data
   - Need: Manager performance tracking
   - Need: Position clustering detection

2. **Money Flow Analysis** (HIGH)
   - Already partially implemented in `stanley/analytics/money_flow.py`
   - Need: Real ETF flow data integration
   - Need: Sector rotation detection

3. **Dark Pool Activity** (HIGH)
   - Placeholder exists in `money_flow.py`
   - Need: FINRA ATS data integration
   - Need: Real-time large block detection

### Phase 2: Options Intelligence (Short-term)

4. **Options Flow Analytics** (HIGH)
   - Need: Options data provider integration
   - Need: Unusual activity detection algorithm
   - Need: Block trade and sweep identification

5. **Put/Call Analysis** (MEDIUM)
   - Need: Open interest tracking
   - Need: Sentiment indicators

### Phase 3: Advanced Analytics (Medium-term)

6. **Smart Money Tracking** (MEDIUM)
   - Need: Manager performance database
   - Need: Conviction pick identification
   - Need: Cross-manager correlation

7. **Risk Analytics** (MEDIUM)
   - Already partially in `stanley/portfolio/`
   - Need: Enhanced stress testing
   - Need: Factor exposure analysis

### Phase 4: Competitive Advantage (Long-term)

8. **AI-Powered Research** (LOW)
   - Document analysis and summarization
   - Natural language queries
   - Pattern recognition across filings

9. **Real-time Integration** (LOW)
   - Live streaming data
   - Automated alerting
   - API for external consumption

---

## 5. Recommended Data Source Strategy

### Tier 1: Free Sources (Implement First)

1. **SEC EDGAR Direct**
   - 13F filings parsing
   - Form 4 insider trading
   - 10-K/10-Q fundamentals

2. **FINRA ATS Transparency**
   - Dark pool volume data
   - OTC market maker data

3. **OpenBB Integration**
   - Already implemented in `stanley/data/openbb_adapter.py`
   - Extend for additional data types

### Tier 2: Low-Cost Commercial (Consider)

1. **Earningsfeed** (Free tier available)
   - Real-time SEC filings
   - Institutional holdings

2. **OptionStrat Flow** (Free tier)
   - Delayed options flow
   - Unusual activity

### Tier 3: Premium Commercial (Future)

1. **Unusual Whales API** ($$$)
   - Real-time options flow
   - Dark pool prints
   - Comprehensive coverage

2. **FlowAlgo** ($149/month)
   - Real-time unusual activity
   - Dark pool tracking

---

## 6. Stanley Current State vs. Bloomberg

| Feature | Stanley Status | Bloomberg | Gap |
|---------|---------------|-----------|-----|
| 13F Holdings | Partial (mock data) | Full | Need real SEC data |
| Dark Pool | Placeholder | Full | Need FINRA integration |
| Options Flow | Not implemented | Full | Need provider |
| ETF Flows | Partial (mock) | Full | Need real flow data |
| Risk Analytics | Partial | Full | Need enhancement |
| Real-time | Not implemented | Full | Future phase |
| AI Research | Not implemented | Full | Future phase |

---

## 7. Key Takeaways

1. **SEC EDGAR is the foundation** - Free access to 13F, Form 4, and company filings
2. **FINRA ATS data is underutilized** - Free dark pool data with 2-4 week delay
3. **Options flow requires commercial data** - No viable free source for real-time
4. **Current Stanley implementation has good structure** - Need real data integration
5. **Prioritize 13F + Dark Pool** - Highest value with lowest cost
6. **Consider tiered approach** - Free sources first, add commercial as needed

---

## Research Agent Notes

This research was conducted as part of the Stanley Hive Mind swarm (swarm_1766744091969_87thyx31j).

**Files Analyzed:**
- `/home/artur/Repositories/stanley/stanley/analytics/institutional.py`
- `/home/artur/Repositories/stanley/stanley/analytics/money_flow.py`
- `/home/artur/Repositories/stanley/stanley/data/data_manager.py`

**Web Sources Consulted:**
- Bloomberg Professional Services documentation
- SEC EDGAR data resources
- FINRA OTC Transparency data
- Various options flow providers

**Next Steps for Other Agents:**
1. Implement SEC EDGAR 13F API integration
2. Implement FINRA ATS data fetching
3. Evaluate options data providers
4. Enhance existing analyzers with real data hooks
