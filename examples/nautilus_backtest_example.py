#!/usr/bin/env python3
"""
Stanley-NautilusTrader Backtest Example

This example demonstrates how to integrate Stanley's institutional analytics
with NautilusTrader for backtesting trading strategies that incorporate
money flow analysis, institutional positioning, and smart money tracking.

The strategy:
- Uses Stanley's MoneyFlowActor to track institutional money flow signals
- Uses Stanley's InstitutionalActor to monitor 13F filing changes
- Makes trading decisions based on institutional accumulation/distribution patterns
- Implements basic risk management with position sizing and stop losses

Requirements:
    pip install nautilus_trader openbb pandas numpy

Usage:
    python examples/nautilus_backtest_example.py
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

import pandas as pd

# NautilusTrader imports
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.models import FillModel, LatencyModel
from nautilus_trader.config import LoggingConfig, StrategyConfig
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import (
    AccountType,
    AggregationSource,
    BarAggregation,
    OmsType,
    OrderSide,
    PriceType,
    TimeInForce,
)
from nautilus_trader.model.identifiers import InstrumentId, TraderId, Venue
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.trading.strategy import Strategy

# Stanley integration imports
from stanley.integrations.nautilus import (
    MoneyFlowActor,
    MoneyFlowActorConfig,
    InstitutionalActor,
    InstitutionalActorConfig,
    OpenBBDataClientConfig,
    OpenBBBarConverter,
    OpenBBInstrumentProvider,
    InstrumentConfig,
    create_instrument_id,
    create_bar_type,
)
from stanley.integrations.nautilus.actors.money_flow_actor import MoneyFlowSignalEvent
from stanley.integrations.nautilus.actors.institutional_actor import InstitutionalSignalEvent
from stanley.data.providers.openbb_provider import OpenBBAdapter


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class InstitutionalBacktestConfig:
    """Configuration for the institutional backtest."""

    # Symbols to trade
    SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

    # Backtest period
    START_DATE = datetime(2024, 1, 1)
    END_DATE = datetime(2024, 6, 30)

    # Initial capital
    INITIAL_CAPITAL = 1_000_000.0

    # Risk parameters
    MAX_POSITION_SIZE_PCT = 0.10  # 10% max position size
    STOP_LOSS_PCT = 0.05  # 5% stop loss
    TAKE_PROFIT_PCT = 0.15  # 15% take profit

    # Signal thresholds
    MONEY_FLOW_THRESHOLD = 0.4  # Minimum money flow score for entry
    INSTITUTIONAL_THRESHOLD = 0.3  # Minimum institutional signal for entry
    CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for entry

    # Venue for instruments
    VENUE = "OPENBB"


# =============================================================================
# STRATEGY IMPLEMENTATION
# =============================================================================

class InstitutionalMomentumStrategyConfig(StrategyConfig, frozen=True):
    """
    Configuration for the Institutional Momentum Strategy.
    """
    instrument_ids: tuple[str, ...] = ()
    money_flow_threshold: float = 0.4
    institutional_threshold: float = 0.3
    confidence_threshold: float = 0.6
    max_position_size_pct: float = 0.10
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.15


class InstitutionalMomentumStrategy(Strategy):
    """
    A trading strategy that combines Stanley's institutional analytics
    with NautilusTrader's execution capabilities.

    The strategy:
    1. Monitors money flow signals from MoneyFlowActor
    2. Tracks institutional positioning from InstitutionalActor
    3. Enters long positions on strong accumulation signals
    4. Enters short positions on strong distribution signals
    5. Manages risk with stop losses and position sizing
    """

    def __init__(self, config: InstitutionalMomentumStrategyConfig):
        """
        Initialize the strategy.

        Args:
            config: Strategy configuration
        """
        super().__init__(config)

        self.config = config

        # Track signals from Stanley actors
        self._money_flow_signals: dict[str, MoneyFlowSignalEvent] = {}
        self._institutional_signals: dict[str, InstitutionalSignalEvent] = {}

        # Position tracking
        self._positions: dict[str, str] = {}  # symbol -> position_id
        self._entry_prices: dict[str, float] = {}

        # Bar subscriptions
        self._bar_types: dict[str, BarType] = {}

    def on_start(self) -> None:
        """
        Called when the strategy starts.
        Subscribe to bar data for all configured instruments.
        """
        logger.info("InstitutionalMomentumStrategy starting...")

        for instrument_id_str in self.config.instrument_ids:
            instrument_id = InstrumentId.from_str(instrument_id_str)

            # Get the instrument from cache
            instrument = self.cache.instrument(instrument_id)
            if instrument is None:
                logger.warning(f"Instrument not found: {instrument_id}")
                continue

            # Create bar type for daily bars
            bar_type = BarType(
                instrument_id=instrument_id,
                bar_spec=BarType.standard(
                    1,
                    BarAggregation.DAY,
                    PriceType.LAST,
                ).bar_spec,
                aggregation_source=AggregationSource.EXTERNAL,
            )

            self._bar_types[str(instrument_id.symbol)] = bar_type

            # Subscribe to bar data
            self.subscribe_bars(bar_type)

            logger.info(f"Subscribed to bars for {instrument_id}")

        logger.info("InstitutionalMomentumStrategy started successfully")

    def on_stop(self) -> None:
        """
        Called when the strategy stops.
        Close all open positions.
        """
        logger.info("InstitutionalMomentumStrategy stopping...")

        # Close any remaining positions
        for position in self.cache.positions_open():
            self.close_position(position)

        logger.info("InstitutionalMomentumStrategy stopped")

    def on_bar(self, bar: Bar) -> None:
        """
        Process incoming bar data.

        Args:
            bar: The received bar data
        """
        symbol = str(bar.bar_type.instrument_id.symbol)

        # Get current signals
        money_flow = self._money_flow_signals.get(symbol)
        institutional = self._institutional_signals.get(symbol)

        # Check for trading signals
        self._evaluate_entry(symbol, bar, money_flow, institutional)
        self._evaluate_exit(symbol, bar)

    def on_event(self, event) -> None:
        """
        Handle custom events from Stanley actors.

        Args:
            event: The received event
        """
        if isinstance(event, MoneyFlowSignalEvent):
            self._handle_money_flow_signal(event)
        elif isinstance(event, InstitutionalSignalEvent):
            self._handle_institutional_signal(event)

    def _handle_money_flow_signal(self, event: MoneyFlowSignalEvent) -> None:
        """
        Process a money flow signal event.

        Args:
            event: Money flow signal from MoneyFlowActor
        """
        self._money_flow_signals[event.symbol] = event

        logger.debug(
            f"Money flow signal for {event.symbol}: "
            f"type={event.signal_type}, strength={event.signal_strength:.2f}, "
            f"confidence={event.confidence:.2f}"
        )

    def _handle_institutional_signal(self, event: InstitutionalSignalEvent) -> None:
        """
        Process an institutional signal event.

        Args:
            event: Institutional signal from InstitutionalActor
        """
        self._institutional_signals[event.symbol] = event

        logger.debug(
            f"Institutional signal for {event.symbol}: "
            f"type={event.signal_type}, strength={event.signal_strength:.2f}, "
            f"ownership={event.institutional_ownership:.2%}"
        )

    def _evaluate_entry(
        self,
        symbol: str,
        bar: Bar,
        money_flow: Optional[MoneyFlowSignalEvent],
        institutional: Optional[InstitutionalSignalEvent],
    ) -> None:
        """
        Evaluate potential entry signals.

        Args:
            symbol: The symbol being evaluated
            bar: Current bar data
            money_flow: Latest money flow signal
            institutional: Latest institutional signal
        """
        # Skip if we already have a position
        if symbol in self._positions:
            return

        # Skip if we don't have signals
        if money_flow is None or institutional is None:
            return

        # Check confidence threshold
        avg_confidence = (money_flow.confidence + institutional.confidence) / 2
        if avg_confidence < self.config.confidence_threshold:
            return

        # Calculate combined signal
        combined_signal = (
            0.6 * money_flow.signal_strength +
            0.4 * institutional.signal_strength
        )

        # Determine trade direction
        if combined_signal >= self.config.money_flow_threshold:
            # Strong accumulation - go long
            self._enter_long(symbol, bar)
        elif combined_signal <= -self.config.money_flow_threshold:
            # Strong distribution - go short
            self._enter_short(symbol, bar)

    def _enter_long(self, symbol: str, bar: Bar) -> None:
        """
        Enter a long position.

        Args:
            symbol: Symbol to trade
            bar: Current bar data
        """
        instrument_id = bar.bar_type.instrument_id
        instrument = self.cache.instrument(instrument_id)

        if instrument is None:
            return

        # Calculate position size
        account = self.cache.account_for_venue(instrument_id.venue)
        if account is None:
            return

        equity = float(account.balance_total(USD))
        max_position_value = equity * self.config.max_position_size_pct
        current_price = float(bar.close)
        quantity = int(max_position_value / current_price)

        if quantity < 1:
            return

        # Create and submit market order
        order = self.order_factory.market(
            instrument_id=instrument_id,
            order_side=OrderSide.BUY,
            quantity=Quantity.from_int(quantity),
            time_in_force=TimeInForce.GTC,
        )

        self.submit_order(order)
        self._entry_prices[symbol] = current_price

        logger.info(
            f"LONG ENTRY: {symbol} @ {current_price:.2f}, "
            f"quantity={quantity}"
        )

    def _enter_short(self, symbol: str, bar: Bar) -> None:
        """
        Enter a short position.

        Args:
            symbol: Symbol to trade
            bar: Current bar data
        """
        instrument_id = bar.bar_type.instrument_id
        instrument = self.cache.instrument(instrument_id)

        if instrument is None:
            return

        # Calculate position size
        account = self.cache.account_for_venue(instrument_id.venue)
        if account is None:
            return

        equity = float(account.balance_total(USD))
        max_position_value = equity * self.config.max_position_size_pct
        current_price = float(bar.close)
        quantity = int(max_position_value / current_price)

        if quantity < 1:
            return

        # Create and submit market order
        order = self.order_factory.market(
            instrument_id=instrument_id,
            order_side=OrderSide.SELL,
            quantity=Quantity.from_int(quantity),
            time_in_force=TimeInForce.GTC,
        )

        self.submit_order(order)
        self._entry_prices[symbol] = current_price

        logger.info(
            f"SHORT ENTRY: {symbol} @ {current_price:.2f}, "
            f"quantity={quantity}"
        )

    def _evaluate_exit(self, symbol: str, bar: Bar) -> None:
        """
        Evaluate exit conditions for existing positions.

        Args:
            symbol: Symbol to evaluate
            bar: Current bar data
        """
        if symbol not in self._entry_prices:
            return

        entry_price = self._entry_prices[symbol]
        current_price = float(bar.close)

        # Get position for this symbol
        instrument_id = bar.bar_type.instrument_id
        positions = self.cache.positions_open(instrument_id=instrument_id)

        if not positions:
            # Position was closed, clean up tracking
            self._entry_prices.pop(symbol, None)
            self._positions.pop(symbol, None)
            return

        position = positions[0]

        # Check stop loss and take profit
        if position.is_long:
            pnl_pct = (current_price - entry_price) / entry_price

            if pnl_pct <= -self.config.stop_loss_pct:
                logger.info(f"STOP LOSS triggered for {symbol} at {pnl_pct:.2%}")
                self.close_position(position)
            elif pnl_pct >= self.config.take_profit_pct:
                logger.info(f"TAKE PROFIT triggered for {symbol} at {pnl_pct:.2%}")
                self.close_position(position)

        elif position.is_short:
            pnl_pct = (entry_price - current_price) / entry_price

            if pnl_pct <= -self.config.stop_loss_pct:
                logger.info(f"STOP LOSS triggered for {symbol} at {pnl_pct:.2%}")
                self.close_position(position)
            elif pnl_pct >= self.config.take_profit_pct:
                logger.info(f"TAKE PROFIT triggered for {symbol} at {pnl_pct:.2%}")
                self.close_position(position)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_historical_data(
    symbols: list[str],
    start_date: datetime,
    end_date: datetime,
    venue: str = "OPENBB",
) -> tuple[dict[str, Equity], dict[str, list[Bar]]]:
    """
    Load historical data from OpenBB for backtesting.

    Args:
        symbols: List of symbols to load
        start_date: Start date for historical data
        end_date: End date for historical data
        venue: Venue identifier for instruments

    Returns:
        Tuple of (instruments dict, bars dict)
    """
    logger.info(f"Loading historical data for {len(symbols)} symbols...")

    # Initialize OpenBB adapter
    openbb_adapter = OpenBBAdapter(config={
        'provider': 'yfinance',
        'fallback_provider': 'fmp',
    })

    # Create instrument provider
    instrument_provider = OpenBBInstrumentProvider(venue=venue)

    instruments = {}
    all_bars = {}

    for symbol in symbols:
        logger.info(f"Loading data for {symbol}...")

        try:
            # Create instrument definition
            instrument_config = InstrumentConfig(
                symbol=symbol,
                asset_class="EQUITY",
                currency="USD",
                exchange="NASDAQ",
                tick_size=0.01,
                lot_size=1.0,
            )
            instrument = instrument_provider.create_equity(instrument_config)
            instruments[symbol] = instrument

            # Fetch historical data
            df = openbb_adapter.get_historical_data(
                symbol=symbol,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
            )

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                continue

            # Create bar type
            bar_type = BarType(
                instrument_id=instrument.id,
                bar_spec=BarType.standard(
                    1,
                    BarAggregation.DAY,
                    PriceType.LAST,
                ).bar_spec,
                aggregation_source=AggregationSource.EXTERNAL,
            )

            # Convert to NautilusTrader bars
            converter = OpenBBBarConverter(
                instrument_id=instrument.id,
                bar_type=bar_type,
                price_precision=2,
                size_precision=0,
            )
            bars = converter.convert_dataframe(df)

            if bars:
                all_bars[symbol] = bars
                logger.info(f"Loaded {len(bars)} bars for {symbol}")

        except Exception as e:
            logger.error(f"Failed to load data for {symbol}: {e}")
            continue

    return instruments, all_bars


# =============================================================================
# BACKTEST ENGINE SETUP
# =============================================================================

def create_backtest_engine(
    instruments: dict[str, Equity],
    bars: dict[str, list[Bar]],
    initial_capital: float,
    venue: str = "OPENBB",
) -> BacktestEngine:
    """
    Create and configure the NautilusTrader backtest engine.

    Args:
        instruments: Dictionary of instrument definitions
        bars: Dictionary of historical bars per symbol
        initial_capital: Starting capital
        venue: Venue identifier

    Returns:
        Configured BacktestEngine
    """
    logger.info("Creating backtest engine...")

    # Configure the backtest engine
    config = BacktestEngineConfig(
        trader_id=TraderId("BACKTEST-001"),
        logging=LoggingConfig(
            log_level="INFO",
            bypass_logging=False,
        ),
    )

    # Create the engine
    engine = BacktestEngine(config=config)

    # Add venue
    venue_obj = Venue(venue)
    engine.add_venue(
        venue=venue_obj,
        oms_type=OmsType.HEDGING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money(initial_capital, USD)],
        fill_model=FillModel(),
        latency_model=LatencyModel(),
    )

    # Add instruments
    for symbol, instrument in instruments.items():
        engine.add_instrument(instrument)
        logger.info(f"Added instrument: {instrument.id}")

    # Add bar data
    for symbol, bar_list in bars.items():
        for bar in bar_list:
            engine.add_data([bar])

    logger.info(f"Added data for {len(bars)} symbols")

    return engine


def run_backtest(
    engine: BacktestEngine,
    instruments: dict[str, Equity],
    config: InstitutionalBacktestConfig,
) -> None:
    """
    Run the backtest with Stanley actors and the trading strategy.

    Args:
        engine: Configured backtest engine
        instruments: Dictionary of instruments
        config: Backtest configuration
    """
    logger.info("Setting up backtest components...")

    venue = Venue(config.VENUE)

    # Create instrument ID strings for strategy config
    instrument_ids = tuple(
        f"{symbol}.{config.VENUE}" for symbol in instruments.keys()
    )

    # Configure and create MoneyFlowActor
    money_flow_config = MoneyFlowActorConfig(
        component_id="MoneyFlowActor-001",
        symbols=list(instruments.keys()),
        lookback_bars=20,
        update_frequency=1,
        enable_dark_pool=True,
        dark_pool_lookback_days=20,
        signal_threshold=0.3,
        confidence_threshold=0.5,
    )
    money_flow_actor = MoneyFlowActor(config=money_flow_config)

    # Configure and create InstitutionalActor
    institutional_config = InstitutionalActorConfig(
        component_id="InstitutionalActor-001",
        universe=list(instruments.keys()),
        tracked_managers=[
            "0000102909",  # Vanguard
            "0001390777",  # BlackRock
            "0000093751",  # State Street
        ],
        minimum_aum=1e9,
        update_frequency=5,
        lookback_bars=20,
        signal_threshold=0.3,
        confidence_threshold=0.5,
    )
    institutional_actor = InstitutionalActor(config=institutional_config)

    # Configure and create the trading strategy
    strategy_config = InstitutionalMomentumStrategyConfig(
        strategy_id="InstitutionalMomentum-001",
        instrument_ids=instrument_ids,
        money_flow_threshold=config.MONEY_FLOW_THRESHOLD,
        institutional_threshold=config.INSTITUTIONAL_THRESHOLD,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
        max_position_size_pct=config.MAX_POSITION_SIZE_PCT,
        stop_loss_pct=config.STOP_LOSS_PCT,
        take_profit_pct=config.TAKE_PROFIT_PCT,
    )
    strategy = InstitutionalMomentumStrategy(config=strategy_config)

    # Add actors to the engine
    engine.add_actor(money_flow_actor)
    engine.add_actor(institutional_actor)

    # Add strategy to the engine
    engine.add_strategy(strategy)

    logger.info("Running backtest...")

    # Run the backtest
    engine.run()

    logger.info("Backtest completed!")


def display_results(engine: BacktestEngine) -> None:
    """
    Display backtest results and performance metrics.

    Args:
        engine: Completed backtest engine
    """
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)

    # Get account performance
    for account in engine.trader.accounts():
        print(f"\nAccount: {account.id}")
        print("-" * 40)

        # Get balances
        for currency, balance in account.balances().items():
            print(f"  {currency}: {balance}")

        # Get realized P&L
        for currency, pnl in account.realized_pnls().items():
            print(f"  Realized P&L ({currency}): {pnl}")

    # Get order statistics
    orders = engine.cache.orders()
    filled_orders = [o for o in orders if o.is_filled]
    rejected_orders = [o for o in orders if o.is_rejected]

    print(f"\nOrder Statistics:")
    print("-" * 40)
    print(f"  Total Orders: {len(orders)}")
    print(f"  Filled Orders: {len(filled_orders)}")
    print(f"  Rejected Orders: {len(rejected_orders)}")

    # Get position statistics
    positions = engine.cache.positions()
    open_positions = [p for p in positions if p.is_open]
    closed_positions = [p for p in positions if p.is_closed]

    print(f"\nPosition Statistics:")
    print("-" * 40)
    print(f"  Total Positions: {len(positions)}")
    print(f"  Open Positions: {len(open_positions)}")
    print(f"  Closed Positions: {len(closed_positions)}")

    # Calculate win rate for closed positions
    if closed_positions:
        winning = sum(1 for p in closed_positions if p.realized_pnl > 0)
        win_rate = winning / len(closed_positions) * 100
        print(f"  Win Rate: {win_rate:.1f}%")

    # Get trading summary
    print("\n" + "=" * 80)
    print("TRADING SUMMARY BY SYMBOL")
    print("=" * 80)

    for position in positions:
        symbol = str(position.instrument_id.symbol)
        print(f"\n{symbol}:")
        print(f"  Side: {position.side}")
        print(f"  Quantity: {position.quantity}")
        print(f"  Avg Price: {position.avg_px_open}")
        print(f"  Realized P&L: {position.realized_pnl}")
        print(f"  Status: {'Open' if position.is_open else 'Closed'}")

    print("\n" + "=" * 80)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main entry point for the backtest example.
    """
    print("=" * 80)
    print("Stanley-NautilusTrader Institutional Backtest Example")
    print("=" * 80)
    print()

    config = InstitutionalBacktestConfig()

    print(f"Configuration:")
    print(f"  Symbols: {config.SYMBOLS}")
    print(f"  Period: {config.START_DATE.date()} to {config.END_DATE.date()}")
    print(f"  Initial Capital: ${config.INITIAL_CAPITAL:,.2f}")
    print(f"  Max Position Size: {config.MAX_POSITION_SIZE_PCT:.0%}")
    print(f"  Stop Loss: {config.STOP_LOSS_PCT:.0%}")
    print(f"  Take Profit: {config.TAKE_PROFIT_PCT:.0%}")
    print()

    # Load historical data
    instruments, bars = load_historical_data(
        symbols=config.SYMBOLS,
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        venue=config.VENUE,
    )

    if not instruments:
        logger.error("No instruments loaded. Exiting.")
        return

    if not bars:
        logger.error("No bar data loaded. Exiting.")
        return

    # Create backtest engine
    engine = create_backtest_engine(
        instruments=instruments,
        bars=bars,
        initial_capital=config.INITIAL_CAPITAL,
        venue=config.VENUE,
    )

    # Run the backtest
    run_backtest(engine, instruments, config)

    # Display results
    display_results(engine)

    # Dispose of the engine
    engine.dispose()

    print("\nBacktest completed successfully!")


if __name__ == "__main__":
    main()
