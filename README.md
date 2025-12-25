# Stanley - Institutional Investment Analysis Platform

Stanley is a sophisticated investment analysis platform focused on institutional-grade metrics, money flow analysis, and fundamental research. No moon phases, no Fibonacci retracements, no crystal balls - just real financial analysis.

## Philosophy

Stanley is built for serious investors who understand that:
- Money flow drives markets, not chart patterns
- Fundamental analysis trumps technical indicators
- Institutional positioning matters more than retail sentiment
- Risk management is paramount
- Data quality is everything

## Features

### Core Analytics
- **Money Flow Analysis**: Track institutional money movement across sectors and asset classes
- **Options Flow**: Analyze smart money positioning through options activity
- **Dark Pool Data**: Monitor off-exchange trading patterns
- **Short Interest Analytics**: Track institutional short positioning
- **Insider Trading**: Monitor corporate insider transactions
- **Institutional Holdings**: Track 13F filings and position changes

### Data Sources
- OpenBB for comprehensive market data
- SEC filings (13F, 13D, 13G, Forms 4)
- Options flow data providers
- Dark pool volume tracking
- Institutional research platforms
- Economic data feeds

### Portfolio Analytics
- Risk-adjusted performance metrics
- Factor-based attribution analysis
- Liquidity risk assessment
- Concentration risk monitoring
- Scenario analysis and stress testing

## Installation

```bash
git clone <repository-url>
cd stanley
pip install -r requirements.txt
```

## Quick Start

```python
from stanley.core import Stanley
from stanley.analytics import MoneyFlowAnalyzer

# Initialize Stanley
stanley = Stanley(config_path="config/stanley.yaml")

# Analyze money flow
analyzer = MoneyFlowAnalyzer(stanley)
money_flow = analyzer.get_sector_money_flow(sectors=["XLK", "XLF", "XLE"])

# Get institutional positioning
institutional_data = stanley.get_institutional_holdings("AAPL")
```

## Configuration

Stanley requires API keys for various data sources. Configure in `config/stanley.yaml`:

```yaml
# Required API keys
openbb:
  api_key: "your_openbb_api_key"

# Optional data sources
sec:
  enabled: true
  
options_flow:
  provider: "your_provider"
  api_key: "your_api_key"

# Risk parameters
risk:
  max_position_size: 0.1
  max_sector_exposure: 0.3
  var_confidence: 0.95
```

## Project Structure

```
stanley/
├── core/                 # Core Stanley functionality
├── analytics/           # Money flow and institutional analysis
├── data/               # Data sources and adapters
├── portfolio/          # Portfolio analytics and risk management
├── research/           # Fundamental research tools
├── api/                # API endpoints and interfaces
├── config/             # Configuration files
└── tests/              # Test suite
```

## Development

```bash
# Run tests
pytest tests/

# Run linting
flake8 stanley/

# Format code
black stanley/
```

## Contributing

1. Focus on institutional-grade analysis
2. No technical indicators or chart patterns
3. Emphasize data quality and reliability
4. Include comprehensive tests
5. Document all assumptions and methodologies

## License

MIT License - See LICENSE file for details