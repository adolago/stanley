# Stanley - Institutional Investment Platform

Real financial analysis. No technical indicators, no moon phases, no BS.

## What It Does
- Money flow analysis across sectors
- Institutional holdings tracking (13F filings)
- Options flow and dark pool monitoring
- Short interest and insider trading data
- Portfolio risk analytics

## Philosophy
Money flow drives markets, not chart patterns. We analyze:
- Where institutional money is moving
- What smart money is buying/selling
- Real regulatory filings and data
- Quantitative risk metrics

## Quick Start
```bash
pip install -r requirements.txt
python -c "from stanley.core import Stanley; s = Stanley(); print(s.health_check())"
```

## Usage
```python
from stanley.core import Stanley

stanley = Stanley()

# Analyze sector money flow
flow = stanley.analyze_sector_money_flow(['XLK', 'XLF', 'XLE'])

# Get institutional holdings
institutional = stanley.get_institutional_holdings('AAPL')
```

## Data Sources
- OpenBB for market data
- SEC filings (13F, Forms 4)
- Options flow providers
- Dark pool volume data

## No Technical Indicators
We explicitly avoid:
- RSI, MACD, Bollinger Bands
- Fibonacci retracements
- Chart patterns
- Astrological analysis

Focus on real money movement, institutional positioning, and regulatory data.