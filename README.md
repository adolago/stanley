# Stanley - Institutional Investment Analysis Platform

Stanley is an institutional investment analysis platform with a Python FastAPI backend and Rust GPUI desktop GUI. It focuses on money flow analysis, institutional positioning, fundamental research, macro analysis, and commodities.

## Philosophy

Stanley is built for investors who value:
- Money flow and institutional positioning data
- Fundamental and macro analysis
- Risk management and portfolio analytics
- Data quality over technical indicators

## Architecture

```
stanley/
├── stanley/              # Python backend (FastAPI)
│   ├── api/             # REST API with 80+ endpoints
│   │   ├── auth/        # JWT, API keys, rate limiting, RBAC
│   │   └── routers/     # Modular endpoint routers
│   ├── core/            # Main Stanley class
│   ├── analytics/       # Money flow and institutional analysis
│   ├── data/            # DataManager and OpenBB adapter
│   ├── portfolio/       # VaR, beta, sector exposure
│   ├── research/        # Valuation, earnings, DCF
│   ├── commodities/     # Commodity prices and macro linkages
│   ├── accounting/      # SEC filings via edgartools
│   ├── macro/           # Economic data via DBnomics
│   ├── etf/             # ETF analytics
│   ├── options/         # Options flow analysis
│   ├── signals/         # Signal generation and backtesting
│   ├── notes/           # Research vault
│   └── integrations/    # NautilusTrader integration
│
└── stanley-gui/         # Rust GPUI desktop application
    └── src/
        ├── app.rs       # Main application state
        ├── api.rs       # HTTP client for backend
        ├── dashboard.rs # Market overview
        ├── portfolio.rs # Portfolio analytics view
        ├── etf.rs       # ETF analytics view
        ├── signals.rs   # Signal management
        ├── accounting.rs # SEC filings view
        ├── macro_view.rs # Economic indicators
        ├── commodities.rs # Commodity markets
        ├── notes.rs     # Research notes
        └── settings.rs  # Application settings
```

## Features

### Core Analytics
- **Money Flow Analysis**: Track institutional money movement across sectors
- **Institutional Holdings**: 13F filings and position changes
- **Dark Pool Data**: Off-exchange trading patterns
- **Options Flow**: Smart money positioning through options
- **Short Interest**: Institutional short positioning

### Research & Valuation
- **DCF Models**: Discounted cash flow valuation
- **Peer Comparison**: Comparable company analysis
- **Earnings Analysis**: Earnings surprises and trends
- **Quality Scores**: Piotroski F-Score, Beneish M-Score

### Portfolio Analytics
- **Risk Metrics**: VaR, CVaR, beta
- **Sector Exposure**: Allocation breakdown
- **Performance Attribution**: Factor-based analysis
- **Benchmark Comparison**: Alpha and tracking error

### Macro & Commodities
- **Economic Indicators**: GDP, inflation, employment (DBnomics)
- **Regime Detection**: Market regime classification
- **Commodity Prices**: Futures and spot prices
- **Cross-Asset Correlations**: Macro-commodity linkages

### SEC Filings
- **Financial Statements**: Income, balance sheet, cash flow
- **Earnings Quality**: Accrual analysis and red flags
- **Filing History**: 10-K, 10-Q, 8-K tracking

## Installation

### Python Backend

```bash
git clone <repository-url>
cd stanley

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment (optional)
cp config/stanley.yaml.example config/stanley.yaml
```

### Rust GUI (Linux - Wayland)

```bash
cd stanley-gui
cargo build --release
```

Note: The GUI is built for Wayland-only on Linux. X11 is not supported.

## Quick Start

### Start the API Server

```bash
python -m stanley.api.main
```

The API will be available at `http://localhost:8000`.

### Run the GUI

```bash
cd stanley-gui
cargo run --release
```

### Python Library Usage

```python
from stanley.core import Stanley
from stanley.analytics import MoneyFlowAnalyzer

# Initialize Stanley
stanley = Stanley()

# Analyze sector money flow
analyzer = MoneyFlowAnalyzer(stanley)
money_flow = analyzer.get_sector_money_flow(sectors=["XLK", "XLF", "XLE"])

# Get institutional holdings
institutional_data = stanley.get_institutional_holdings("AAPL")
```

## API Endpoints

### Market Data
```
GET  /api/health                     # Health check
GET  /api/market/{symbol}            # Stock market data
GET  /api/market/{symbol}/quote      # Real-time quote
```

### Institutional & Money Flow
```
GET  /api/institutional/{symbol}     # 13F holdings
POST /api/money-flow                 # Sector money flow
GET  /api/dark-pool/{symbol}         # Dark pool activity
GET  /api/equity-flow/{symbol}       # Equity money flow
```

### Portfolio Analytics
```
POST /api/portfolio-analytics        # VaR, beta, exposure
GET  /api/portfolio/risk             # Risk metrics
GET  /api/portfolio/attribution      # Performance attribution
```

### Research & Valuation
```
GET  /api/research/{symbol}          # Comprehensive report
GET  /api/valuation/{symbol}         # Valuation with DCF
GET  /api/earnings/{symbol}          # Earnings analysis
GET  /api/peers/{symbol}             # Peer comparison
```

### SEC Accounting
```
GET  /api/accounting/{symbol}/filings    # Filing history
GET  /api/accounting/{symbol}/statements # Financial statements
GET  /api/accounting/{symbol}/quality    # Earnings quality
```

### Commodities
```
GET  /api/commodities                # Market overview
GET  /api/commodities/{symbol}       # Commodity detail
GET  /api/commodities/correlations   # Correlation matrix
GET  /api/commodities/{symbol}/macro # Macro linkages
```

### ETF Analytics
```
GET  /api/etf/flows                  # ETF fund flows
GET  /api/etf/sector-rotation        # Sector rotation
GET  /api/etf/{symbol}               # ETF detail
```

### Signals
```
GET  /api/signals/{symbol}           # Active signals
POST /api/signals                    # Generate signal
GET  /api/signals/backtest           # Backtest results
```

### Research Notes
```
GET  /api/notes                      # List notes
GET  /api/notes/{name}               # Get note
PUT  /api/notes/{name}               # Update note
```

## Configuration

### Environment Variables

```bash
# API Configuration
STANLEY_API_HOST=0.0.0.0
STANLEY_API_PORT=8000

# OpenBB API Key (optional)
OPENBB_API_KEY=your_key

# SEC EDGAR Identity (required for accounting)
SEC_EDGAR_USER_AGENT="Your Name your@email.com"

# Redis (optional, for caching)
REDIS_URL=redis://localhost:6379
```

### YAML Configuration

```yaml
# config/stanley.yaml
openbb:
  api_key: "${OPENBB_API_KEY}"

sec:
  enabled: true
  user_agent: "${SEC_EDGAR_USER_AGENT}"

risk:
  var_confidence: 0.95
  max_position_size: 0.1
  max_sector_exposure: 0.3
```

## Development

### Run Tests

```bash
# All tests
pytest tests/

# With coverage
pytest tests/ --cov=stanley

# Specific module
pytest tests/test_analytics.py
```

### Code Quality

```bash
# Linting
flake8 stanley/

# Formatting
black stanley/

# Type checking
mypy stanley/
```

### Build Rust GUI

```bash
cd stanley-gui

# Development
cargo run

# Release build
cargo build --release
```

## Data Sources

- **OpenBB**: Market data, fundamentals, options
- **SEC EDGAR**: 13F filings, financial statements (via edgartools)
- **DBnomics**: Economic indicators, macro data
- **Dark Pool**: Off-exchange volume data

## Module Status

| Module | Status | Description |
|--------|--------|-------------|
| `core/` | Active | Main Stanley class |
| `analytics/` | Active | Money flow, institutional analysis |
| `data/` | Active | DataManager, OpenBB adapter |
| `accounting/` | Active | SEC filings, financial statements |
| `macro/` | Active | Economic data, regime detection |
| `portfolio/` | Active | VaR, beta, sector exposure |
| `research/` | Active | Valuation, earnings, DCF |
| `commodities/` | Active | Commodity prices, correlations |
| `etf/` | Active | ETF analytics |
| `options/` | Active | Options flow analysis |
| `signals/` | Active | Signal generation, backtesting |
| `notes/` | Active | Research vault |
| `api/` | Active | FastAPI REST endpoints |
| `api/auth/` | Active | JWT, API keys, rate limiting |
| `integrations/nautilus/` | Partial | NautilusTrader (40% ready) |

## Security

The API includes authentication and rate limiting:

- **JWT Authentication**: Token-based user authentication
- **API Keys**: Programmatic access with scoped permissions
- **RBAC**: Role-based access control
- **Rate Limiting**: Per-endpoint request limits

## Contributing

1. Focus on institutional-grade analysis
2. Emphasize data quality and reliability
3. Include comprehensive tests
4. Document assumptions and methodologies

## License

GNU Affero General Public License v3.0 (AGPL-3.0) - See LICENSE file for details.

This project uses OpenBB (AGPL-3.0) and dbnomics (AGPL-3.0), which require the entire project to be licensed under AGPL-3.0. When running this software as a network service, you must provide source code to users of that service.
