# Stanley Module Documentation

This document provides detailed documentation for each module in the Stanley investment analysis platform.

## Core Modules

### stanley.core

The main Stanley class that coordinates all functionality.

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

### stanley.analytics.money_flow

Money flow analysis for sectors and individual equities.

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

### stanley.analytics.institutional

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

### stanley.analytics.dark_pool

Dark pool and off-exchange trading analysis.

### stanley.analytics.sector_rotation

Sector rotation signals and analysis.

### stanley.analytics.smart_money_index

Smart money tracking and institutional activity signals.

---

## Portfolio Module

### stanley.portfolio.portfolio_analyzer

Portfolio analytics including risk metrics.

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

### stanley.portfolio.risk_metrics

Risk calculation utilities.

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

### stanley.research.research_analyzer

Comprehensive research report generation.

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

### stanley.research.valuation

Valuation models including DCF.

```python
from stanley.research.valuation import ValuationAnalyzer

valuation = ValuationAnalyzer(data_manager)

# Get valuation multiples
multiples = await valuation.get_multiples("AAPL")

# Run DCF model
dcf = await valuation.run_dcf("AAPL")
```

### stanley.research.earnings

Earnings analysis and quality assessment.

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

### stanley.macro.macro_analyzer

Main macroeconomic analysis coordinator.

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

### stanley.macro.dbnomics_adapter

DBnomics integration for economic data.

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

### stanley.macro.regime_detector

Market regime detection using multiple signals.

**Regime Types:**
- `expansion`: Economic expansion
- `contraction`: Economic contraction
- `recovery`: Post-recession recovery
- `slowdown`: Economic slowdown

### stanley.macro.yield_curve

Yield curve analysis including inversion signals.

### stanley.macro.recession_model

Recession probability modeling.

### stanley.macro.credit_spreads

Credit spread analysis (IG, HY spreads).

### stanley.macro.business_cycle

Business cycle phase detection.

### stanley.macro.volatility_regime

Volatility regime classification (low/medium/high).

---

## Accounting Module

### stanley.accounting.accounting_analyzer

Main accounting analysis coordinator.

```python
from stanley.accounting import AccountingAnalyzer

analyzer = AccountingAnalyzer(edgar_identity="your-email@example.com")

# Get SEC filings
filings = analyzer.get_filings("AAPL")

# Get financial statements
statements = analyzer.get_statements("AAPL")
```

### stanley.accounting.edgar_adapter

SEC EDGAR integration via edgartools.

```python
from stanley.accounting.edgar_adapter import EdgarAdapter

adapter = EdgarAdapter(identity="your-email@example.com")
adapter.initialize()

# Get company filings
filings = adapter.get_company_filings("AAPL")

# Get specific filing
filing = adapter.get_filing("AAPL", "10-K", year=2023)
```

### stanley.accounting.financial_statements

Financial statement parsing and analysis.

### stanley.accounting.earnings_quality

Earnings quality scoring.

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

### stanley.accounting.red_flags

Accounting red flag detection.

```python
from stanley.accounting.red_flags import RedFlagScorer

scorer = RedFlagScorer(edgar_adapter)

# Get red flags
flags = scorer.analyze("AAPL")
```

### stanley.accounting.anomaly_detection

Accounting anomaly aggregation.

### stanley.accounting.footnotes

Financial statement footnote extraction.

---

## Commodities Module

### stanley.commodities.commodities_analyzer

Commodity market analysis.

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
- `CL`: Crude Oil (WTI)
- `GC`: Gold
- `SI`: Silver
- `NG`: Natural Gas
- `HG`: Copper
- `ZC`: Corn
- `ZW`: Wheat
- `ZS`: Soybeans

---

## Options Module

### stanley.options.options_analyzer

Options flow and analytics.

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

### stanley.etf.etf_analyzer

ETF flow and rotation analysis.

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

### stanley.signals.signal_generator

Multi-factor signal generation.

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

### stanley.signals.backtester

Signal backtesting engine.

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

### stanley.signals.performance_tracker

Signal performance monitoring.

```python
from stanley.signals import PerformanceTracker

tracker = PerformanceTracker(data_manager)

# Get performance stats
stats = await tracker.get_stats()
```

---

## Notes Module (Research Vault)

### stanley.notes.vault

Research note management with Obsidian-like linking.

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

### stanley.notes.models

Note data models.

**Note Types:**
- `note`: General research note
- `thesis`: Investment thesis
- `trade`: Trade journal entry
- `event`: Market event
- `person`: Key person/executive
- `sector`: Sector analysis

### stanley.notes.templates

Note templates for different types.

---

## Integrations

### stanley.integrations.nautilus

NautilusTrader integration for algorithmic trading.

See [docs/NAUTILUS_INTEGRATION.md](NAUTILUS_INTEGRATION.md) for detailed documentation.

**Components:**
- `OpenBBDataClient`: NautilusTrader data client
- `MoneyFlowActor`: Money flow analysis actor
- `InstitutionalActor`: Institutional holdings actor
- `SmartMoneyIndicator`: Custom smart money indicator
- `InstitutionalMomentumIndicator`: Institutional momentum indicator
