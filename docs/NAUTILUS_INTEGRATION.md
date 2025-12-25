# Stanley-NautilusTrader Integration

This document provides comprehensive documentation for integrating Stanley's institutional analytics with the NautilusTrader algorithmic trading framework.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Components Reference](#components-reference)
5. [Configuration](#configuration)
6. [Example Strategies](#example-strategies)
7. [API Reference](#api-reference)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### What the Integration Provides

The Stanley-NautilusTrader integration bridges institutional investment analytics with algorithmic trading infrastructure. It enables:

- **Real-time money flow analysis** during backtests and live trading
- **Institutional positioning signals** based on 13F filings and smart money tracking
- **Custom indicators** for smart money and institutional momentum
- **OpenBB data integration** for fetching market data into NautilusTrader

### Architecture Diagram

```
+------------------+     +-------------------+     +----------------------+
|                  |     |                   |     |                      |
|   OpenBB SDK     +---->+  OpenBBAdapter    +---->+  OpenBBDataClient    |
|   (Data Source)  |     |  (Stanley Data)   |     |  (NautilusTrader)    |
|                  |     |                   |     |                      |
+------------------+     +-------------------+     +----------+-----------+
                                                              |
                                                              v
+------------------+     +-------------------+     +----------------------+
|                  |     |                   |     |                      |
| MoneyFlowAnalyzer+---->+ MoneyFlowActor    +---->+  Trading Strategy    |
| InstitutionalAna.|     | InstitutionalActor|     |  (User-Defined)      |
|                  |     |                   |     |                      |
+------------------+     +-------------------+     +----------------------+
                                |
                                v
                    +-------------------+
                    |                   |
                    | SmartMoneyIndicator
                    | InstitutionalMomentum
                    |   Indicator       |
                    +-------------------+
```

### Key Components

| Component | Description |
|-----------|-------------|
| `OpenBBDataClient` | NautilusTrader DataClient that fetches data from OpenBB |
| `OpenBBAdapter` | High-level adapter for OpenBB SDK with caching and rate limiting |
| `MoneyFlowActor` | Actor that wraps Stanley's MoneyFlowAnalyzer for event-driven analysis |
| `InstitutionalActor` | Actor for 13F filing analysis and institutional positioning |
| `SmartMoneyIndicator` | Custom indicator tracking dark pool and block trade activity |
| `InstitutionalMomentumIndicator` | Custom indicator for institutional ownership trends |

---

## Installation

### Requirements

- Python 3.10+
- NautilusTrader 1.180.0+
- OpenBB SDK 4.0+
- Stanley core package

### pip Install Commands

```bash
# Install NautilusTrader
pip install nautilus_trader

# Install OpenBB SDK
pip install openbb

# Install Stanley (from source)
cd /path/to/stanley
pip install -e .

# Install all dependencies
pip install -r requirements.txt
```

### OpenBB Configuration

1. Create an OpenBB account at [https://my.openbb.co](https://my.openbb.co)
2. Generate a Personal Access Token (PAT)
3. Configure OpenBB with your token:

```python
from openbb import obb

# Option 1: Login with token
obb.account.login(pat="your-openbb-token")

# Option 2: Set in environment variable
# export OPENBB_PAT=your-openbb-token
```

### NautilusTrader Setup

NautilusTrader requires Rust for compilation. Ensure you have:

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Verify installation
cargo --version
```

---

## Quick Start

### Basic Example: Fetching Data

```python
import asyncio
from datetime import datetime, timedelta

from stanley.integrations.nautilus import (
    OpenBBDataClient,
    OpenBBDataClientConfig,
    create_instrument_id,
    create_bar_type,
)
from nautilus_trader.model.enums import BarAggregation

async def fetch_data_example():
    # Configure the data client
    config = OpenBBDataClientConfig(
        venue="OPENBB",
        openbb_token="your-token",
        provider="yfinance",
        rate_limit_per_minute=60,
    )

    # Create mock NautilusTrader components (use real ones in production)
    from unittest.mock import Mock
    msgbus = Mock()
    cache = Mock()
    clock = Mock()

    # Initialize client
    client = OpenBBDataClient(
        msgbus=msgbus,
        cache=cache,
        clock=clock,
        config=config,
    )

    # Create instrument
    instrument = client.create_instrument(
        symbol="AAPL",
        exchange="NASDAQ",
        currency="USD",
    )

    # Create bar type
    instrument_id = create_instrument_id("AAPL", "OPENBB")
    bar_type = create_bar_type(
        instrument_id=instrument_id,
        bar_aggregation=BarAggregation.DAY,
        step=1,
    )

    # Request historical bars
    start = datetime.now() - timedelta(days=365)
    end = datetime.now()

    bars = await client.request_bars(
        bar_type=bar_type,
        start=start,
        end=end,
    )

    print(f"Fetched {len(bars)} bars for AAPL")
    return bars

# Run the example
asyncio.run(fetch_data_example())
```

### Running a Simple Backtest

```python
from datetime import datetime
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import Venue

from stanley.integrations.nautilus import (
    OpenBBDataClient,
    OpenBBDataClientConfig,
    MoneyFlowActor,
    MoneyFlowActorConfig,
)

def run_backtest():
    # Create backtest engine
    engine = BacktestEngine()

    # Add venue
    engine.add_venue(
        venue=Venue("OPENBB"),
        oms_type=OmsType.HEDGING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money(100_000, USD)],
    )

    # Configure and add data client
    data_config = OpenBBDataClientConfig(
        venue="OPENBB",
        openbb_token="your-token",
    )

    # Configure and add MoneyFlowActor
    actor_config = MoneyFlowActorConfig(
        symbols=["AAPL.OPENBB", "MSFT.OPENBB", "GOOGL.OPENBB"],
        lookback_bars=20,
        signal_threshold=0.3,
        enable_dark_pool=True,
    )

    actor = MoneyFlowActor(config=actor_config)
    engine.add_actor(actor)

    # Add your strategy here
    # engine.add_strategy(YourStrategy(config))

    # Run backtest
    engine.run()

    return engine

engine = run_backtest()
```

### Getting Signals from Actors

```python
from stanley.integrations.nautilus import (
    MoneyFlowActor,
    MoneyFlowActorConfig,
    InstitutionalActor,
    InstitutionalActorConfig,
)

# Create MoneyFlowActor
mf_config = MoneyFlowActorConfig(
    symbols=["AAPL.OPENBB"],
    lookback_bars=20,
    signal_threshold=0.3,
    confidence_threshold=0.5,
)

money_flow_actor = MoneyFlowActor(config=mf_config)

# After the actor processes bars, get signals:
flow_analysis = money_flow_actor.get_flow_analysis("AAPL")
signal_strength = money_flow_actor.get_signal_strength("AAPL")
confidence = money_flow_actor.get_confidence("AAPL")

print(f"Signal Strength: {signal_strength:.2f}")
print(f"Confidence: {confidence:.2f}")

# Create InstitutionalActor
inst_config = InstitutionalActorConfig(
    universe=["AAPL.OPENBB", "MSFT.OPENBB"],
    tracked_managers=["0000102909", "0001390777"],  # Vanguard, BlackRock
    minimum_aum=1e9,
)

institutional_actor = InstitutionalActor(config=inst_config)

# Get institutional holdings
holdings = institutional_actor.get_holdings("AAPL")
ownership = institutional_actor.get_institutional_ownership("AAPL")
```

---

## Components Reference

### OpenBBAdapter

High-level adapter for fetching data from OpenBB SDK.

#### Usage

```python
from stanley.data.openbb_adapter import OpenBBAdapter

# Initialize with configuration
adapter = OpenBBAdapter(
    config={
        "openbb": {
            "api_key": "your-openbb-token",
            "max_retries": 3,
            "timeout": 30,
            "rate_limit": 5.0,
            "cache_ttl": 300,
        }
    }
)

# Use as async context manager
async with adapter as obb:
    # Get historical prices
    prices = await obb.get_historical_prices(
        symbol="AAPL",
        lookback_days=252,
    )

    # Get institutional holders
    holders = await obb.get_institutional_holders("AAPL")

    # Get insider activity
    insiders = await obb.get_insider_activity(
        symbol="AAPL",
        lookback_days=90,
    )

    # Get options chain
    options = await obb.get_options_chain("AAPL")

    # Get ETF holdings
    etf_holdings = await obb.get_etf_holdings("SPY", top_n=10)
```

#### Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `get_historical_prices(symbol, start_date, end_date, lookback_days)` | Fetch OHLCV data | `pd.DataFrame` |
| `get_multiple_prices(symbols, ...)` | Fetch prices for multiple symbols | `Dict[str, pd.DataFrame]` |
| `get_institutional_holders(symbol, min_value)` | Get institutional holders | `pd.DataFrame` |
| `get_institutional_summary(symbol)` | Get ownership summary | `Dict` |
| `get_insider_activity(symbol, lookback_days, transaction_type)` | Get insider trades | `pd.DataFrame` |
| `get_options_chain(symbol, expiration)` | Get options chain | `pd.DataFrame` |
| `get_unusual_options_activity(symbol, volume_threshold)` | Get unusual options | `pd.DataFrame` |
| `get_etf_holdings(symbol, top_n)` | Get ETF holdings | `pd.DataFrame` |
| `get_short_interest(symbol)` | Get short interest data | `pd.DataFrame` |

### OpenBBDataClient

NautilusTrader DataClient implementation using OpenBB.

#### Usage

```python
from stanley.integrations.nautilus import (
    OpenBBDataClient,
    OpenBBDataClientConfig,
)

config = OpenBBDataClientConfig(
    venue="OPENBB",
    openbb_token="your-token",
    provider="yfinance",
    base_currency="USD",
    request_timeout_secs=30.0,
    max_retries=3,
    cache_enabled=True,
    cache_ttl_secs=300,
    rate_limit_per_minute=60,
)

client = OpenBBDataClient(
    msgbus=msgbus,
    cache=cache,
    clock=clock,
    config=config,
)

# Create instruments
instrument = client.create_instrument(
    symbol="AAPL",
    exchange="NASDAQ",
    currency="USD",
    tick_size=0.01,
    lot_size=1.0,
)

# Request historical bars
bars = await client.request_bars(
    bar_type=bar_type,
    start=start_date,
    end=end_date,
    limit=1000,
)

# Subscribe to bar updates (polling-based)
await client.subscribe_bars(bar_type)

# Request quote ticks
quotes = await client.request_quote_ticks(instrument_id)

# Check connection status
print(f"Connected: {client.is_connected}")
print(f"Subscriptions: {client.subscriptions}")
```

### MoneyFlowActor

Actor that wraps Stanley's MoneyFlowAnalyzer for event-driven analysis.

#### Configuration

```python
from stanley.integrations.nautilus import MoneyFlowActor, MoneyFlowActorConfig

config = MoneyFlowActorConfig(
    # Symbols to monitor
    symbols=["AAPL.OPENBB", "MSFT.OPENBB", "GOOGL.OPENBB"],

    # Analysis parameters
    lookback_bars=20,           # Number of bars for analysis
    update_frequency=1,         # Update every N bars

    # Dark pool analysis
    enable_dark_pool=True,
    dark_pool_lookback_days=20,

    # Signal thresholds
    signal_threshold=0.3,       # Min signal strength to emit event
    confidence_threshold=0.5,   # Min confidence to emit event

    # Sector ETFs for sector flow analysis
    sector_etfs=['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLB', 'XLU', 'XLRE', 'XLC'],

    # Async data refresh interval (seconds)
    data_refresh_interval=300,
)

actor = MoneyFlowActor(config=config)
```

#### Signals Emitted

The actor emits `MoneyFlowSignalEvent` when thresholds are met:

```python
@dataclass
class MoneyFlowSignalEvent(Event):
    symbol: str                    # Stock symbol
    signal_type: str               # 'accumulation', 'distribution', 'dark_pool', 'smart_money'
    signal_strength: float         # -1.0 to 1.0
    confidence: float              # 0.0 to 1.0
    money_flow_score: float
    institutional_sentiment: float
    smart_money_activity: float
    dark_pool_signal: Optional[int]  # -1, 0, 1
    timestamp: datetime
    metadata: Dict[str, Any]
```

#### Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `get_flow_analysis(symbol)` | Get latest money flow analysis | `Optional[Dict]` |
| `get_dark_pool_analysis(symbol)` | Get dark pool analysis | `Optional[pd.DataFrame]` |
| `get_sector_flow()` | Get sector flow analysis | `Optional[pd.DataFrame]` |
| `get_signal_strength(symbol)` | Get current signal strength | `float` |
| `get_confidence(symbol)` | Get confidence score | `float` |

### InstitutionalActor

Actor for 13F filing analysis and institutional positioning tracking.

#### Configuration

```python
from stanley.integrations.nautilus import InstitutionalActor, InstitutionalActorConfig

config = InstitutionalActorConfig(
    # Universe of symbols to monitor
    universe=["AAPL.OPENBB", "MSFT.OPENBB", "GOOGL.OPENBB"],

    # Institutional managers to track (by SEC CIK)
    tracked_managers=[
        "0000102909",  # Vanguard Group
        "0001390777",  # BlackRock
        "0000093751",  # State Street
        "0000315066",  # Fidelity
        "0000080227",  # T. Rowe Price
    ],

    # Minimum AUM for smart money tracking
    minimum_aum=1e9,  # $1 billion

    # Analysis parameters
    update_frequency=5,              # Update every N bars
    lookback_bars=20,

    # Signal thresholds
    signal_threshold=0.3,
    confidence_threshold=0.5,
    ownership_change_threshold=0.05,  # 5% change is significant

    # 13F filing check interval (seconds)
    filing_check_interval=3600,

    # Holdings refresh interval (seconds)
    holdings_refresh_interval=900,
)

actor = InstitutionalActor(config=config)
```

#### Signals Emitted

**InstitutionalSignalEvent:**
```python
@dataclass
class InstitutionalSignalEvent(Event):
    symbol: str
    signal_type: str              # 'accumulation', 'distribution', '13f_change', 'smart_money'
    signal_strength: float        # -1.0 to 1.0
    confidence: float             # 0.0 to 1.0
    institutional_ownership: float
    ownership_trend: float
    smart_money_score: float
    concentration_risk: float
    timestamp: datetime
    metadata: Dict[str, Any]
```

**Filing13FEvent:**
```python
@dataclass
class Filing13FEvent(Event):
    manager_cik: str
    manager_name: str
    filing_date: datetime
    new_positions: List[str]
    closed_positions: List[str]
    significant_increases: List[Dict[str, Any]]
    significant_decreases: List[Dict[str, Any]]
    timestamp: datetime
```

#### Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `get_holdings(symbol)` | Get holdings data | `Optional[Dict]` |
| `get_universe_sentiment()` | Get universe sentiment | `Optional[Dict]` |
| `get_smart_money_activity()` | Get smart money tracking | `Optional[pd.DataFrame]` |
| `get_13f_changes(manager_cik)` | Get 13F changes | `Optional[pd.DataFrame]` |
| `get_signal_strength(symbol)` | Get signal strength | `float` |
| `get_confidence(symbol)` | Get confidence score | `float` |
| `get_institutional_ownership(symbol)` | Get ownership % | `float` |

### SmartMoneyIndicator

Custom indicator tracking smart money activity patterns.

#### Usage

```python
from stanley.integrations.nautilus import SmartMoneyIndicator

indicator = SmartMoneyIndicator(
    period=20,                      # Lookback period
    dark_pool_weight=0.30,          # Weight for dark pool signal
    block_trade_weight=0.25,        # Weight for block trade signal
    flow_imbalance_weight=0.25,     # Weight for flow imbalance
    volume_weight=0.20,             # Weight for volume signal
    dark_pool_threshold=0.25,       # Threshold for significant dark pool
    block_trade_threshold=0.10,     # Threshold for significant blocks
)

# Feed bars to the indicator
for bar in bars:
    indicator.handle_bar(bar)

# Get indicator values
print(f"Value: {indicator.value}")                     # -1.0 to 1.0
print(f"Signal Strength: {indicator.signal_strength}") # 0.0 to 1.0
print(f"Confidence: {indicator.confidence}")           # 0.0 to 1.0

# Component signals
print(f"Dark Pool Signal: {indicator.dark_pool_signal}")
print(f"Block Trade Signal: {indicator.block_trade_signal}")
print(f"Flow Imbalance: {indicator.flow_imbalance_signal}")
print(f"Volume Signal: {indicator.volume_signal}")

# Helper methods
if indicator.is_bullish(threshold=0.3):
    print("Bullish smart money signal")
elif indicator.is_bearish(threshold=-0.3):
    print("Bearish smart money signal")
else:
    print("Neutral")
```

#### Properties

| Property | Description | Range |
|----------|-------------|-------|
| `value` | Main indicator value | -1.0 to 1.0 |
| `signal_strength` | Absolute signal strength | 0.0 to 1.0 |
| `confidence` | Signal confidence | 0.0 to 1.0 |
| `dark_pool_signal` | Dark pool component | -1.0 to 1.0 |
| `block_trade_signal` | Block trade component | -1.0 to 1.0 |
| `flow_imbalance_signal` | Order flow component | -1.0 to 1.0 |
| `volume_signal` | Volume component | -1.0 to 1.0 |
| `dark_pool_percentage` | Estimated dark pool % | 0.0 to 1.0 |
| `block_trade_ratio` | Estimated block ratio | 0.0 to 1.0 |

### InstitutionalMomentumIndicator

Custom indicator measuring institutional momentum.

#### Usage

```python
from stanley.integrations.nautilus import InstitutionalMomentumIndicator

indicator = InstitutionalMomentumIndicator(
    period=20,
    ownership_weight=0.35,       # Weight for ownership trend
    smart_money_weight=0.30,     # Weight for smart money signal
    concentration_weight=0.15,   # Weight for concentration changes
    momentum_weight=0.20,        # Weight for price momentum
    symbol="AAPL",               # Symbol for institutional data
    data_manager=data_manager,   # Optional Stanley data manager
)

# Feed bars to the indicator
for bar in bars:
    indicator.handle_bar(bar)

# Get indicator values
print(f"Value: {indicator.value}")
print(f"Sentiment: {indicator.get_sentiment()}")  # 'bullish', 'bearish', 'neutral'

# Component signals
components = indicator.get_component_signals()
print(f"Ownership Signal: {components['ownership']}")
print(f"Smart Money Signal: {components['smart_money']}")
print(f"Concentration Signal: {components['concentration']}")
print(f"Momentum Signal: {components['momentum']}")

# Check institutional ownership
ownership = indicator.institutional_ownership
if ownership:
    print(f"Institutional Ownership: {ownership:.1%}")
```

---

## Configuration

### stanley.yaml nautilus Section

```yaml
# NautilusTrader Integration
nautilus:
  # Data client configuration
  data_client:
    default_provider: "openbb"    # Data source for Nautilus
    cache_ttl: 3600               # Cache duration in seconds
    rate_limit: 5                 # Requests per second

  # Venue configuration
  venues:
    - name: "NASDAQ"
      venue_id: "XNAS"
    - name: "NYSE"
      venue_id: "XNYS"
    - name: "IEX"
      venue_id: "IEXG"

  # Actor configuration
  actors:
    money_flow:
      lookback_periods: [20, 60, 252]  # Days
      signal_threshold: 0.7

    institutional:
      update_frequency: "daily"
      min_confidence: 0.5
      smart_money_weight: 0.6

  # Indicator configuration
  indicators:
    smart_money:
      period: 20
      sensitivity: 1.0

    institutional_momentum:
      short_period: 20
      long_period: 60
```

### Environment Variables

```bash
# OpenBB Authentication
export OPENBB_PAT="your-openbb-personal-access-token"

# Optional: Override provider settings
export STANLEY_OPENBB_PROVIDER="polygon"
export STANLEY_RATE_LIMIT="10"

# Database (if using persistent storage)
export STANLEY_DB_HOST="localhost"
export STANLEY_DB_PORT="5432"
export STANLEY_DB_NAME="stanley"
export STANLEY_DB_USER="stanley"
export STANLEY_DB_PASSWORD="your-password"

# Redis cache (if using)
export STANLEY_REDIS_HOST="localhost"
export STANLEY_REDIS_PORT="6379"
```

### API Keys

Different data providers require different API keys:

| Provider | API Key Required | Notes |
|----------|-----------------|-------|
| yfinance | No | Free, but rate limited |
| polygon | Yes | Real-time data available |
| fmp | Yes | Fundamentals and institutional data |
| intrinio | Yes | Options data |
| fred | No | Economic data |

Configure in `stanley.yaml` or via environment:

```yaml
openbb:
  api_key: "your-openbb-token"

# Provider-specific keys
providers:
  polygon:
    api_key: "your-polygon-key"
  fmp:
    api_key: "your-fmp-key"
  intrinio:
    api_key: "your-intrinio-key"
```

---

## Example Strategies

### Complete Backtest Example

```python
"""
Complete backtest example using Stanley-NautilusTrader integration.

This strategy combines money flow and institutional signals to generate trades.
"""

from datetime import datetime, timedelta
from decimal import Decimal

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide, TimeInForce
from nautilus_trader.model.identifiers import TraderId, Venue
from nautilus_trader.model.objects import Money, Quantity
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.config import StrategyConfig

from stanley.integrations.nautilus import (
    OpenBBDataClient,
    OpenBBDataClientConfig,
    MoneyFlowActor,
    MoneyFlowActorConfig,
    InstitutionalActor,
    InstitutionalActorConfig,
    SmartMoneyIndicator,
)


class InstitutionalMomentumStrategyConfig(StrategyConfig):
    symbols: list[str]
    signal_threshold: float = 0.5
    position_size_pct: float = 0.10
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.15


class InstitutionalMomentumStrategy(Strategy):
    """
    Strategy that trades based on institutional money flow signals.
    """

    def __init__(self, config: InstitutionalMomentumStrategyConfig):
        super().__init__(config)
        self.config = config
        self.indicators = {}

    def on_start(self):
        # Initialize indicators for each symbol
        for symbol in self.config.symbols:
            instrument_id = self.cache.instrument_id(symbol)
            self.indicators[symbol] = SmartMoneyIndicator(period=20)

            # Subscribe to bar data
            bar_type = self._create_bar_type(instrument_id)
            self.subscribe_bars(bar_type)

    def on_bar(self, bar):
        symbol = str(bar.bar_type.instrument_id.symbol)

        # Update indicator
        if symbol in self.indicators:
            self.indicators[symbol].handle_bar(bar)

        # Check for signals
        indicator = self.indicators.get(symbol)
        if indicator and indicator.initialized:
            self._evaluate_signal(bar, indicator)

    def _evaluate_signal(self, bar, indicator):
        symbol = str(bar.bar_type.instrument_id.symbol)
        instrument_id = bar.bar_type.instrument_id

        # Get current position
        position = self.portfolio.positions.get(instrument_id)

        # Strong bullish signal and no position
        if indicator.is_bullish(self.config.signal_threshold) and position is None:
            self._open_long(instrument_id, bar.close)

        # Strong bearish signal and have long position
        elif indicator.is_bearish(-self.config.signal_threshold) and position:
            if position.is_long:
                self._close_position(instrument_id)

    def _open_long(self, instrument_id, price):
        account = self.portfolio.account(self.venue)
        balance = account.balance_total(USD)

        # Calculate position size
        position_value = float(balance) * self.config.position_size_pct
        quantity = int(position_value / float(price))

        if quantity > 0:
            order = self.order_factory.market(
                instrument_id=instrument_id,
                order_side=OrderSide.BUY,
                quantity=Quantity.from_int(quantity),
                time_in_force=TimeInForce.GTC,
            )
            self.submit_order(order)

    def _close_position(self, instrument_id):
        position = self.portfolio.positions.get(instrument_id)
        if position:
            order = self.order_factory.market(
                instrument_id=instrument_id,
                order_side=OrderSide.SELL,
                quantity=position.quantity,
                time_in_force=TimeInForce.GTC,
            )
            self.submit_order(order)


def run_backtest():
    # Create engine
    engine = BacktestEngine(
        trader_id=TraderId("STANLEY-001"),
    )

    # Add venue
    OPENBB = Venue("OPENBB")
    engine.add_venue(
        venue=OPENBB,
        oms_type=OmsType.HEDGING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money(100_000, USD)],
    )

    # Add actors for signal generation
    mf_config = MoneyFlowActorConfig(
        symbols=["AAPL.OPENBB", "MSFT.OPENBB", "GOOGL.OPENBB"],
        signal_threshold=0.3,
    )
    engine.add_actor(MoneyFlowActor(config=mf_config))

    inst_config = InstitutionalActorConfig(
        universe=["AAPL.OPENBB", "MSFT.OPENBB", "GOOGL.OPENBB"],
    )
    engine.add_actor(InstitutionalActor(config=inst_config))

    # Add strategy
    strategy_config = InstitutionalMomentumStrategyConfig(
        strategy_id="INST-MOM-001",
        symbols=["AAPL.OPENBB", "MSFT.OPENBB", "GOOGL.OPENBB"],
        signal_threshold=0.5,
    )
    engine.add_strategy(InstitutionalMomentumStrategy(config=strategy_config))

    # Run backtest
    engine.run()

    # Generate report
    engine.trader.generate_order_fills_report()
    engine.trader.generate_positions_report()
    engine.trader.generate_account_report()

    return engine


if __name__ == "__main__":
    engine = run_backtest()
    print("Backtest completed!")
```

### Signal-Based Trading Example

```python
"""
Example showing how to react to actor signals in a strategy.
"""

from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.events import Event

from stanley.integrations.nautilus.actors.money_flow_actor import MoneyFlowSignalEvent
from stanley.integrations.nautilus.actors.institutional_actor import (
    InstitutionalSignalEvent,
    Filing13FEvent,
)


class SignalReactiveStrategy(Strategy):
    """Strategy that reacts to Stanley actor signals."""

    def on_event(self, event: Event):
        """Handle incoming events from actors."""

        if isinstance(event, MoneyFlowSignalEvent):
            self._handle_money_flow_signal(event)

        elif isinstance(event, InstitutionalSignalEvent):
            self._handle_institutional_signal(event)

        elif isinstance(event, Filing13FEvent):
            self._handle_13f_filing(event)

    def _handle_money_flow_signal(self, event: MoneyFlowSignalEvent):
        """React to money flow signals."""
        self.log.info(
            f"Money Flow Signal: {event.symbol} - "
            f"Type: {event.signal_type}, "
            f"Strength: {event.signal_strength:.2f}, "
            f"Confidence: {event.confidence:.2f}"
        )

        # Strong accumulation signal
        if event.signal_type == "accumulation" and event.signal_strength > 0.5:
            self._consider_long(event.symbol)

        # Strong distribution signal
        elif event.signal_type == "distribution" and event.signal_strength < -0.5:
            self._consider_exit(event.symbol)

        # Dark pool activity
        elif event.signal_type == "dark_pool" and event.dark_pool_signal == 1:
            self.log.info(f"Bullish dark pool activity detected for {event.symbol}")

    def _handle_institutional_signal(self, event: InstitutionalSignalEvent):
        """React to institutional signals."""
        self.log.info(
            f"Institutional Signal: {event.symbol} - "
            f"Ownership: {event.institutional_ownership:.1%}, "
            f"Smart Money: {event.smart_money_score:.2f}"
        )

        # Smart money accumulation
        if event.signal_type == "smart_money" and event.signal_strength > 0.5:
            self._consider_long(event.symbol)

    def _handle_13f_filing(self, event: Filing13FEvent):
        """React to 13F filing updates."""
        self.log.info(
            f"13F Filing: {event.manager_name} - "
            f"New Positions: {len(event.new_positions)}, "
            f"Closed: {len(event.closed_positions)}"
        )

        # React to new positions from tracked managers
        for symbol in event.new_positions:
            self.log.info(f"{event.manager_name} opened new position in {symbol}")

    def _consider_long(self, symbol: str):
        """Placeholder for long entry logic."""
        pass

    def _consider_exit(self, symbol: str):
        """Placeholder for exit logic."""
        pass
```

### Multi-Actor Strategy

```python
"""
Strategy combining signals from multiple actors.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.config import StrategyConfig


class MultiActorStrategyConfig(StrategyConfig):
    symbols: list[str]
    money_flow_weight: float = 0.4
    institutional_weight: float = 0.4
    momentum_weight: float = 0.2
    combined_threshold: float = 0.6


@dataclass
class CombinedSignal:
    symbol: str
    money_flow_score: float = 0.0
    institutional_score: float = 0.0
    momentum_score: float = 0.0

    def combined_score(self, mf_weight, inst_weight, mom_weight) -> float:
        return (
            self.money_flow_score * mf_weight +
            self.institutional_score * inst_weight +
            self.momentum_score * mom_weight
        )


class MultiActorStrategy(Strategy):
    """Strategy that combines signals from multiple actors."""

    def __init__(self, config: MultiActorStrategyConfig):
        super().__init__(config)
        self.signals: Dict[str, CombinedSignal] = {}

        for symbol in config.symbols:
            self.signals[symbol] = CombinedSignal(symbol=symbol)

    def on_event(self, event):
        """Aggregate signals from different actors."""
        from stanley.integrations.nautilus.actors.money_flow_actor import MoneyFlowSignalEvent
        from stanley.integrations.nautilus.actors.institutional_actor import InstitutionalSignalEvent

        if isinstance(event, MoneyFlowSignalEvent):
            if event.symbol in self.signals:
                self.signals[event.symbol].money_flow_score = event.signal_strength

        elif isinstance(event, InstitutionalSignalEvent):
            if event.symbol in self.signals:
                self.signals[event.symbol].institutional_score = event.signal_strength

        # Evaluate combined signals
        self._evaluate_combined_signals()

    def on_bar(self, bar):
        """Update momentum from price data."""
        symbol = str(bar.bar_type.instrument_id.symbol)

        if symbol in self.signals:
            # Simple momentum calculation
            # In production, use a proper momentum indicator
            pass

    def _evaluate_combined_signals(self):
        """Evaluate combined signals and generate trades."""
        for symbol, signal in self.signals.items():
            combined = signal.combined_score(
                self.config.money_flow_weight,
                self.config.institutional_weight,
                self.config.momentum_weight,
            )

            if combined > self.config.combined_threshold:
                self.log.info(
                    f"Strong BUY signal for {symbol}: {combined:.2f} "
                    f"(MF: {signal.money_flow_score:.2f}, "
                    f"Inst: {signal.institutional_score:.2f}, "
                    f"Mom: {signal.momentum_score:.2f})"
                )
                # Execute buy logic

            elif combined < -self.config.combined_threshold:
                self.log.info(
                    f"Strong SELL signal for {symbol}: {combined:.2f}"
                )
                # Execute sell logic
```

---

## API Reference

### Key Classes and Methods

#### OpenBBDataClientConfig

```python
@dataclass
class OpenBBDataClientConfig:
    venue: str = "OPENBB"              # Venue identifier
    openbb_token: Optional[str] = None # OpenBB API token
    provider: str = "yfinance"         # Default data provider
    base_currency: str = "USD"         # Base currency
    request_timeout_secs: float = 30.0 # Request timeout
    max_retries: int = 3               # Max retry attempts
    retry_delay_secs: float = 1.0      # Delay between retries
    cache_enabled: bool = True         # Enable caching
    cache_ttl_secs: int = 300          # Cache TTL (5 minutes)
    rate_limit_per_minute: int = 60    # Rate limit
```

#### MoneyFlowActorConfig

```python
class MoneyFlowActorConfig(ActorConfig):
    symbols: List[str]                      # Symbols to monitor
    lookback_bars: int = 20                 # Analysis lookback
    update_frequency: int = 1               # Update frequency
    enable_dark_pool: bool = True           # Enable dark pool analysis
    dark_pool_lookback_days: int = 20       # Dark pool lookback
    signal_threshold: float = 0.3           # Min signal to emit
    confidence_threshold: float = 0.5       # Min confidence to emit
    sector_etfs: List[str]                  # Sector ETFs for analysis
    data_refresh_interval: int = 300        # Refresh interval (seconds)
```

#### InstitutionalActorConfig

```python
class InstitutionalActorConfig(ActorConfig):
    universe: List[str]                     # Symbols to monitor
    tracked_managers: List[str]             # Manager CIKs to track
    minimum_aum: float = 1e9                # Min AUM for smart money
    update_frequency: int = 5               # Update frequency
    lookback_bars: int = 20                 # Analysis lookback
    signal_threshold: float = 0.3           # Min signal to emit
    confidence_threshold: float = 0.5       # Min confidence to emit
    ownership_change_threshold: float = 0.05# Significant change threshold
    filing_check_interval: int = 3600       # 13F check interval (seconds)
    holdings_refresh_interval: int = 900    # Holdings refresh (seconds)
```

### Signal Format Documentation

#### MoneyFlowSignalEvent Fields

| Field | Type | Description |
|-------|------|-------------|
| `symbol` | `str` | Stock symbol (e.g., "AAPL") |
| `signal_type` | `str` | One of: 'accumulation', 'distribution', 'dark_pool', 'smart_money', 'neutral' |
| `signal_strength` | `float` | Value from -1.0 (bearish) to 1.0 (bullish) |
| `confidence` | `float` | Value from 0.0 (low) to 1.0 (high) |
| `money_flow_score` | `float` | Overall money flow score |
| `institutional_sentiment` | `float` | Institutional sentiment score |
| `smart_money_activity` | `float` | Smart money activity level |
| `dark_pool_signal` | `int` | -1 (bearish), 0 (neutral), 1 (bullish) |
| `timestamp` | `datetime` | Signal timestamp |
| `metadata` | `Dict` | Additional data (accumulation_distribution, short_pressure) |

#### InstitutionalSignalEvent Fields

| Field | Type | Description |
|-------|------|-------------|
| `symbol` | `str` | Stock symbol |
| `signal_type` | `str` | One of: 'accumulation', 'distribution', '13f_change', 'smart_money', 'neutral' |
| `signal_strength` | `float` | Value from -1.0 to 1.0 |
| `confidence` | `float` | Value from 0.0 to 1.0 |
| `institutional_ownership` | `float` | % owned by institutions (0.0 to 1.0) |
| `ownership_trend` | `float` | Trend direction and magnitude |
| `smart_money_score` | `float` | Smart money positioning score |
| `concentration_risk` | `float` | Ownership concentration risk (0.0 to 1.0) |
| `timestamp` | `datetime` | Signal timestamp |
| `metadata` | `Dict` | Additional data (number_of_institutions, top_holders) |

### Data Type Mappings

#### OpenBB to NautilusTrader Bar Conversion

| OpenBB Column | NautilusTrader Field | Notes |
|---------------|---------------------|-------|
| `date` / `datetime` | `ts_event` | Converted to nanoseconds |
| `open` / `Open` | `open` | Converted to `Price` |
| `high` / `High` | `high` | Converted to `Price` |
| `low` / `Low` | `low` | Converted to `Price` |
| `close` / `Close` | `close` | Converted to `Price` |
| `volume` / `Volume` | `volume` | Converted to `Quantity` |

#### Interval String Mapping

| OpenBB Interval | NautilusTrader BarAggregation |
|-----------------|------------------------------|
| `1m`, `min`, `minute` | `BarAggregation.MINUTE` |
| `1h`, `hr`, `hour` | `BarAggregation.HOUR` |
| `1d`, `day` | `BarAggregation.DAY` |
| `1w`, `wk`, `week` | `BarAggregation.WEEK` |
| `1mo`, `month` | `BarAggregation.MONTH` |

---

## Troubleshooting

### Common Errors

#### OpenBB Not Installed

```
ImportError: OpenBB is not installed. Please install with: pip install openbb
```

**Solution:**
```bash
pip install openbb
```

#### Authentication Failed

```
Error: Failed to connect to OpenBB: Authentication failed
```

**Solution:**
1. Check your OpenBB token is valid
2. Ensure token is correctly set:
```python
from openbb import obb
obb.account.login(pat="your-token")
```

#### No Data Returned

```
Warning: No data returned for AAPL
```

**Possible causes:**
- Market is closed (weekends, holidays)
- Invalid date range
- Rate limit exceeded
- Provider issue

**Solution:**
- Check date range includes trading days
- Wait and retry for rate limits
- Try a different provider

#### Bar Conversion Failed

```
Warning: Failed to convert row X: ...
```

**Possible causes:**
- Missing OHLCV columns
- Invalid data types
- NaN values

**Solution:**
- Verify DataFrame has required columns
- Check for missing data

### Rate Limiting

The integration implements automatic rate limiting:

```python
config = OpenBBDataClientConfig(
    rate_limit_per_minute=60,  # Adjust based on your tier
)
```

If you encounter rate limit errors:
1. Reduce `rate_limit_per_minute`
2. Enable caching (`cache_enabled=True`)
3. Increase `cache_ttl_secs`
4. Use batch requests with `get_multiple_prices()`

### Data Format Issues

#### Column Name Mismatches

OpenBB providers may return different column names. The converters handle common variations:

- `date` / `datetime` / `timestamp` / `Date` / `Datetime`
- `open` / `Open`
- `high` / `High`
- `low` / `Low`
- `close` / `Close`
- `volume` / `Volume`

#### Timezone Issues

Timestamps are handled as follows:
- OpenBB returns market time (usually US Eastern)
- Converted to UTC nanoseconds for NautilusTrader
- Use `pandas.Timestamp` for consistency

```python
import pandas as pd

# Convert to UTC
df['date'] = pd.to_datetime(df['date']).dt.tz_localize('US/Eastern').dt.tz_convert('UTC')
```

### Debugging Tips

1. **Enable debug logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **Check data client status:**
```python
print(f"Connected: {client.is_connected}")
print(f"Subscriptions: {client.subscriptions}")
print(f"Instruments: {client.get_available_symbols()}")
```

3. **Verify indicator initialization:**
```python
print(f"Indicator initialized: {indicator.initialized}")
print(f"Value: {indicator.value}")
```

4. **Monitor actor signals:**
```python
# In your strategy
def on_event(self, event):
    self.log.info(f"Received event: {type(event).__name__}")
```

### Getting Help

- Check the [NautilusTrader documentation](https://nautilustrader.io/docs/)
- Review [OpenBB documentation](https://docs.openbb.co/)
- Open an issue on the Stanley repository
