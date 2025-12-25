#!/usr/bin/env python3
"""
Stanley-NautilusTrader Integration Demo

This script demonstrates the integration between Stanley's institutional
analytics platform and NautilusTrader's algorithmic trading framework.

The demo showcases:
1. Data fetching from OpenBB and conversion to NautilusTrader format
2. Money flow and institutional signal generation
3. Custom indicator calculations (SmartMoneyIndicator, InstitutionalMomentumIndicator)

Usage Examples:
    # Run all demos with default settings
    python demo_integration.py

    # Analyze a specific symbol
    python demo_integration.py --symbol MSFT

    # Custom date range
    python demo_integration.py --symbol AAPL --start-date 2024-01-01 --end-date 2024-06-30

    # Run only specific mode
    python demo_integration.py --mode signals
    python demo_integration.py --mode indicators
    python demo_integration.py --mode backtest

Requirements:
    - openbb
    - nautilus_trader
    - pandas
    - numpy
    - stanley (this project)

Note: This demo uses simulated data when live data sources are unavailable,
making it suitable for learning and testing the integration architecture.
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Console Output Utilities
# =============================================================================

class Console:
    """Simple console output formatting utilities."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"

    @staticmethod
    def header(text: str) -> None:
        """Print a section header."""
        print(f"\n{Console.BOLD}{Console.HEADER}{'='*60}{Console.ENDC}")
        print(f"{Console.BOLD}{Console.HEADER}{text:^60}{Console.ENDC}")
        print(f"{Console.BOLD}{Console.HEADER}{'='*60}{Console.ENDC}\n")

    @staticmethod
    def subheader(text: str) -> None:
        """Print a subsection header."""
        print(f"\n{Console.BOLD}{Console.CYAN}{'-'*40}{Console.ENDC}")
        print(f"{Console.BOLD}{Console.CYAN}{text}{Console.ENDC}")
        print(f"{Console.BOLD}{Console.CYAN}{'-'*40}{Console.ENDC}")

    @staticmethod
    def info(text: str) -> None:
        """Print info message."""
        print(f"{Console.BLUE}[INFO]{Console.ENDC} {text}")

    @staticmethod
    def success(text: str) -> None:
        """Print success message."""
        print(f"{Console.GREEN}[OK]{Console.ENDC} {text}")

    @staticmethod
    def warning(text: str) -> None:
        """Print warning message."""
        print(f"{Console.YELLOW}[WARN]{Console.ENDC} {text}")

    @staticmethod
    def error(text: str) -> None:
        """Print error message."""
        print(f"{Console.RED}[ERROR]{Console.ENDC} {text}")

    @staticmethod
    def signal(direction: str, strength: float, label: str = "") -> None:
        """Print a trading signal with color coding."""
        if strength > 0.3:
            color = Console.GREEN
            arrow = ">>>"
        elif strength < -0.3:
            color = Console.RED
            arrow = "<<<"
        else:
            color = Console.YELLOW
            arrow = "---"

        prefix = f"[{label}] " if label else ""
        print(f"  {color}{arrow} {prefix}{direction}: {strength:+.4f}{Console.ENDC}")

    @staticmethod
    def table(headers: List[str], rows: List[List[Any]], title: str = "") -> None:
        """Print a formatted table."""
        if title:
            print(f"\n{Console.BOLD}{title}{Console.ENDC}")

        # Calculate column widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))

        # Print header
        header_row = " | ".join(f"{h:<{widths[i]}}" for i, h in enumerate(headers))
        print(f"  {Console.BOLD}{header_row}{Console.ENDC}")
        print(f"  {'-' * len(header_row)}")

        # Print rows
        for row in rows:
            formatted_row = " | ".join(f"{str(c):<{widths[i]}}" for i, c in enumerate(row))
            print(f"  {formatted_row}")


# =============================================================================
# Demo Data Generation
# =============================================================================

def generate_demo_ohlcv(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic demo OHLCV data for testing.

    This function creates synthetic market data that mimics real price
    movements, including trends, volatility clustering, and realistic
    volume patterns.

    Args:
        symbol: Stock ticker symbol
        start_date: Start date for data generation
        end_date: End date for data generation
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: date, open, high, low, close, volume
    """
    np.random.seed(seed)

    # Generate trading days
    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    n_days = len(dates)

    if n_days == 0:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    # Base price parameters (varies by symbol hash for consistency)
    symbol_hash = sum(ord(c) for c in symbol)
    base_price = 100 + (symbol_hash % 400)
    volatility = 0.015 + (symbol_hash % 10) * 0.002

    # Generate returns with mean reversion and volatility clustering
    returns = np.zeros(n_days)
    vol_state = volatility

    for i in range(n_days):
        # GARCH-like volatility
        vol_state = 0.9 * vol_state + 0.1 * volatility * (1 + abs(np.random.randn()))
        returns[i] = np.random.randn() * vol_state + 0.0003  # Slight upward drift

    # Generate prices
    prices = base_price * np.cumprod(1 + returns)

    # Generate OHLC from close prices
    opens = np.roll(prices, 1)
    opens[0] = base_price

    # Intraday ranges
    intraday_vol = np.abs(np.random.randn(n_days)) * volatility * 0.5
    highs = np.maximum(opens, prices) * (1 + intraday_vol)
    lows = np.minimum(opens, prices) * (1 - intraday_vol)

    # Ensure OHLC consistency
    highs = np.maximum(highs, np.maximum(opens, prices))
    lows = np.minimum(lows, np.minimum(opens, prices))

    # Generate volume with patterns
    base_volume = 1_000_000 + (symbol_hash % 5_000_000)
    volume = base_volume * (1 + 0.5 * np.abs(np.random.randn(n_days)))
    volume = volume * (1 + 0.3 * np.abs(returns) / volatility)  # Higher volume on big moves

    df = pd.DataFrame({
        "date": dates,
        "open": np.round(opens, 2),
        "high": np.round(highs, 2),
        "low": np.round(lows, 2),
        "close": np.round(prices, 2),
        "volume": volume.astype(int),
    })

    return df


def generate_demo_institutional_holdings(symbol: str) -> Dict[str, Any]:
    """
    Generate demo institutional holdings data.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Dictionary with institutional holding metrics
    """
    np.random.seed(sum(ord(c) for c in symbol))

    return {
        "symbol": symbol,
        "institutional_ownership": round(0.4 + np.random.random() * 0.4, 4),
        "ownership_trend": round(np.random.randn() * 0.1, 4),
        "smart_money_score": round(np.random.randn() * 0.3, 4),
        "concentration_risk": round(0.2 + np.random.random() * 0.3, 4),
        "number_of_institutions": int(100 + np.random.randint(0, 500)),
        "top_holders": [
            {"name": "Vanguard Group", "shares": 50_000_000, "pct": 0.08},
            {"name": "BlackRock", "shares": 45_000_000, "pct": 0.07},
            {"name": "State Street", "shares": 30_000_000, "pct": 0.05},
        ],
    }


# =============================================================================
# Data Demo
# =============================================================================

async def run_data_demo(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    use_live_data: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Demonstrate data fetching and NautilusTrader conversion.

    This function shows how to:
    1. Fetch OHLCV data from OpenBB
    2. Display summary statistics
    3. Convert data to NautilusTrader Bar format

    Args:
        symbol: Stock ticker symbol
        start_date: Start date for data
        end_date: End date for data
        use_live_data: If True, attempt to fetch live data from OpenBB

    Returns:
        DataFrame with OHLCV data
    """
    Console.header("DATA DEMO: OpenBB to NautilusTrader Conversion")

    df = None

    # Attempt to fetch live data
    if use_live_data:
        Console.info(f"Fetching live data for {symbol} from OpenBB...")
        try:
            from stanley.data.providers.openbb_provider import OpenBBAdapter

            adapter = OpenBBAdapter(config={"provider": "yfinance"})
            df = adapter.get_historical_data(
                symbol=symbol,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
            )
            Console.success(f"Fetched {len(df)} bars from OpenBB")
        except Exception as e:
            Console.warning(f"Could not fetch live data: {e}")
            Console.info("Falling back to demo data...")

    # Use demo data if live data unavailable
    if df is None or df.empty:
        Console.info(f"Generating demo OHLCV data for {symbol}...")
        df = generate_demo_ohlcv(symbol, start_date, end_date)
        Console.success(f"Generated {len(df)} demo bars")

    if df.empty:
        Console.error("No data available")
        return None

    # Display summary statistics
    Console.subheader("OHLCV Summary Statistics")

    stats_headers = ["Metric", "Value"]
    stats_rows = [
        ["Symbol", symbol],
        ["Date Range", f"{df['date'].min().date()} to {df['date'].max().date()}"],
        ["Total Bars", len(df)],
        ["Price Range", f"${df['low'].min():.2f} - ${df['high'].max():.2f}"],
        ["Avg Close", f"${df['close'].mean():.2f}"],
        ["Avg Volume", f"{df['volume'].mean():,.0f}"],
        ["Total Volume", f"{df['volume'].sum():,.0f}"],
        ["Volatility (StdDev)", f"{df['close'].pct_change().std()*100:.2f}%"],
    ]
    Console.table(stats_headers, stats_rows, "Price Data Summary")

    # Show sample data
    Console.subheader("Sample OHLCV Data (First 5 bars)")
    sample_headers = ["Date", "Open", "High", "Low", "Close", "Volume"]
    sample_rows = []
    for _, row in df.head().iterrows():
        sample_rows.append([
            str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"])[:10],
            f"${row['open']:.2f}",
            f"${row['high']:.2f}",
            f"${row['low']:.2f}",
            f"${row['close']:.2f}",
            f"{row['volume']:,}",
        ])
    Console.table(sample_headers, sample_rows)

    # Demonstrate NautilusTrader conversion
    Console.subheader("NautilusTrader Conversion")

    try:
        from stanley.integrations.nautilus import (
            OpenBBBarConverter,
            create_instrument_id,
            create_bar_type,
        )
        from nautilus_trader.model.enums import BarAggregation

        # Create instrument and bar type
        instrument_id = create_instrument_id(symbol, "OPENBB")
        bar_type = create_bar_type(instrument_id, BarAggregation.DAY, 1)

        Console.info(f"Created InstrumentId: {instrument_id}")
        Console.info(f"Created BarType: {bar_type}")

        # Convert DataFrame to Nautilus Bars
        converter = OpenBBBarConverter(
            instrument_id=instrument_id,
            bar_type=bar_type,
            price_precision=2,
            size_precision=0,
        )

        bars = converter.convert_dataframe(df)
        Console.success(f"Converted {len(bars)} bars to NautilusTrader format")

        # Show sample converted bar
        if bars:
            Console.info("\nSample NautilusTrader Bar:")
            sample_bar = bars[0]
            print(f"  Bar Type: {sample_bar.bar_type}")
            print(f"  Open:     {sample_bar.open}")
            print(f"  High:     {sample_bar.high}")
            print(f"  Low:      {sample_bar.low}")
            print(f"  Close:    {sample_bar.close}")
            print(f"  Volume:   {sample_bar.volume}")
            print(f"  ts_event: {sample_bar.ts_event}")

    except ImportError as e:
        Console.warning(f"NautilusTrader not available: {e}")
        Console.info("Showing conversion structure without actual Nautilus types...")

        print("\n  Conversion would create:")
        print(f"    InstrumentId: {symbol}.OPENBB")
        print(f"    BarType: {symbol}.OPENBB-1-DAY-LAST-EXTERNAL")
        print(f"    Bars: {len(df)} Bar objects with Price/Quantity types")

    return df


# =============================================================================
# Signal Demo
# =============================================================================

async def run_signal_demo(
    symbol: str,
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Demonstrate Stanley's signal generation capabilities.

    This function shows how the MoneyFlowActor and InstitutionalActor
    generate trading signals that can be used in NautilusTrader strategies.

    Args:
        symbol: Stock ticker symbol
        df: Optional OHLCV DataFrame (demo data generated if not provided)

    Returns:
        Dictionary containing generated signals
    """
    Console.header("SIGNAL DEMO: Money Flow & Institutional Analysis")

    signals = {
        "symbol": symbol,
        "timestamp": datetime.now().isoformat(),
        "money_flow": {},
        "institutional": {},
    }

    # Generate or use provided data
    if df is None or df.empty:
        Console.info("Generating demo data for signal analysis...")
        df = generate_demo_ohlcv(
            symbol,
            datetime.now() - timedelta(days=60),
            datetime.now(),
        )

    # Money Flow Signals
    Console.subheader("Money Flow Signals")

    Console.info("Analyzing money flow patterns...")
    Console.info("(In production, this uses Stanley's MoneyFlowAnalyzer)")

    # Calculate money flow metrics from price data
    # These calculations simulate what MoneyFlowAnalyzer would return

    # Accumulation/Distribution based on close location value
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"] + 0.0001)
    mf_multiplier = clv * df["volume"]
    ad_line = mf_multiplier.cumsum()
    ad_trend = (ad_line.iloc[-1] - ad_line.iloc[-20]) / (ad_line.std() + 1) if len(ad_line) >= 20 else 0

    # Volume-price trend
    recent_price_change = (df["close"].iloc[-1] - df["close"].iloc[-20]) / df["close"].iloc[-20] if len(df) >= 20 else 0
    recent_volume_avg = df["volume"].iloc[-5:].mean() / df["volume"].mean() if len(df) >= 5 else 1

    # Smart money proxy (volume spikes with narrow ranges)
    ranges = (df["high"] - df["low"]) / df["close"]
    vol_spikes = df["volume"] / df["volume"].rolling(20).mean()
    smart_money_indicator = ((vol_spikes > 1.5) & (ranges < ranges.quantile(0.3))).sum() / len(df)

    # Calculate money flow score
    money_flow_score = 0.4 * np.tanh(ad_trend) + 0.3 * np.tanh(recent_price_change * 10) + 0.3 * smart_money_indicator
    money_flow_score = round(float(money_flow_score), 4)

    # Dark pool signal (simulated)
    dark_pool_signal = int(np.sign(ad_trend)) if abs(ad_trend) > 0.5 else 0

    money_flow_signals = {
        "money_flow_score": money_flow_score,
        "accumulation_distribution": round(float(np.tanh(ad_trend)), 4),
        "smart_money_activity": round(float(smart_money_indicator * 2 - 0.5), 4),
        "dark_pool_signal": dark_pool_signal,
        "volume_trend": round(float(recent_volume_avg - 1), 4),
        "confidence": round(min(0.9, 0.5 + abs(money_flow_score)), 4),
    }
    signals["money_flow"] = money_flow_signals

    # Display money flow signals
    Console.info("\nMoney Flow Analysis Results:")
    Console.signal(
        "Money Flow Score",
        money_flow_signals["money_flow_score"],
        "Overall"
    )
    Console.signal(
        "Accumulation/Distribution",
        money_flow_signals["accumulation_distribution"],
        "A/D"
    )
    Console.signal(
        "Smart Money Activity",
        money_flow_signals["smart_money_activity"],
        "Smart"
    )

    print(f"\n  Dark Pool Signal: {['Selling', 'Neutral', 'Buying'][dark_pool_signal + 1]}")
    print(f"  Volume Trend: {money_flow_signals['volume_trend']:+.2%} vs average")
    print(f"  Confidence: {money_flow_signals['confidence']:.1%}")

    # Institutional Signals
    Console.subheader("Institutional Signals")

    Console.info("Analyzing institutional positioning...")
    Console.info("(In production, this uses Stanley's InstitutionalAnalyzer)")

    # Get demo institutional holdings
    holdings = generate_demo_institutional_holdings(symbol)

    # Calculate institutional signals
    ownership = holdings["institutional_ownership"]
    ownership_trend = holdings["ownership_trend"]
    smart_money = holdings["smart_money_score"]
    concentration = holdings["concentration_risk"]

    # Combined institutional score
    institutional_score = (
        0.35 * ownership_trend * 5 +  # Scale up the trend
        0.30 * smart_money +
        0.20 * (ownership - 0.5) +  # Deviation from 50% ownership
        0.15 * (0.5 - concentration)  # Lower concentration is bullish
    )
    institutional_score = round(float(np.clip(institutional_score, -1, 1)), 4)

    institutional_signals = {
        "institutional_score": institutional_score,
        "ownership_pct": round(ownership, 4),
        "ownership_trend": round(ownership_trend, 4),
        "smart_money_score": round(smart_money, 4),
        "concentration_risk": round(concentration, 4),
        "num_institutions": holdings["number_of_institutions"],
        "signal_type": "accumulation" if institutional_score > 0.3 else ("distribution" if institutional_score < -0.3 else "neutral"),
    }
    signals["institutional"] = institutional_signals

    # Display institutional signals
    Console.info("\nInstitutional Analysis Results:")
    Console.signal(
        "Institutional Score",
        institutional_signals["institutional_score"],
        "Overall"
    )
    Console.signal(
        "Ownership Trend",
        institutional_signals["ownership_trend"] * 10,  # Scale for visibility
        "Trend"
    )
    Console.signal(
        "Smart Money Score",
        institutional_signals["smart_money_score"],
        "Smart"
    )

    print(f"\n  Institutional Ownership: {ownership:.1%}")
    print(f"  Number of Institutions: {holdings['number_of_institutions']}")
    print(f"  Concentration Risk: {concentration:.1%}")
    print(f"  Signal Type: {institutional_signals['signal_type'].upper()}")

    # Top holders
    print(f"\n  Top Institutional Holders:")
    for holder in holdings["top_holders"]:
        print(f"    - {holder['name']}: {holder['shares']:,} shares ({holder['pct']:.1%})")

    # Combined signal summary
    Console.subheader("Combined Signal Summary")

    # What a strategy would receive
    combined_score = (money_flow_score + institutional_score) / 2
    combined_confidence = (money_flow_signals["confidence"] + 0.7) / 2  # Assume 0.7 inst confidence

    print(f"\n  Combined Signal Score: {combined_score:+.4f}")
    print(f"  Combined Confidence: {combined_confidence:.1%}")

    if combined_score > 0.3 and combined_confidence > 0.6:
        Console.success("BULLISH signal - Both money flow and institutional indicators positive")
    elif combined_score < -0.3 and combined_confidence > 0.6:
        Console.warning("BEARISH signal - Both money flow and institutional indicators negative")
    else:
        Console.info("NEUTRAL signal - Mixed or weak indicators")

    return signals


# =============================================================================
# Indicator Demo
# =============================================================================

async def run_indicator_demo(
    symbol: str,
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Demonstrate Stanley's custom NautilusTrader indicators.

    This function shows how the SmartMoneyIndicator and
    InstitutionalMomentumIndicator process bar data and generate signals.

    Args:
        symbol: Stock ticker symbol
        df: Optional OHLCV DataFrame (demo data generated if not provided)

    Returns:
        Dictionary containing indicator values
    """
    Console.header("INDICATOR DEMO: Custom NautilusTrader Indicators")

    # Generate or use provided data
    if df is None or df.empty:
        Console.info("Generating demo data for indicator calculations...")
        df = generate_demo_ohlcv(
            symbol,
            datetime.now() - timedelta(days=60),
            datetime.now(),
        )

    results = {
        "symbol": symbol,
        "smart_money": {},
        "institutional_momentum": {},
    }

    # Smart Money Indicator
    Console.subheader("SmartMoneyIndicator")

    Console.info("The SmartMoneyIndicator tracks institutional activity by analyzing:")
    Console.info("  - Dark pool activity (estimated from price/volume patterns)")
    Console.info("  - Block trade frequency")
    Console.info("  - Order flow imbalance")
    Console.info("  - Volume anomalies")

    try:
        from stanley.integrations.nautilus import SmartMoneyIndicator
        from stanley.integrations.nautilus import create_instrument_id, create_bar_type, OpenBBBarConverter
        from nautilus_trader.model.enums import BarAggregation

        Console.info("\nInitializing SmartMoneyIndicator...")

        # Create indicator
        smart_money = SmartMoneyIndicator(
            period=20,
            dark_pool_weight=0.3,
            block_trade_weight=0.25,
            flow_imbalance_weight=0.25,
            volume_weight=0.2,
        )

        # Convert data to Nautilus bars
        instrument_id = create_instrument_id(symbol, "OPENBB")
        bar_type = create_bar_type(instrument_id, BarAggregation.DAY, 1)
        converter = OpenBBBarConverter(instrument_id, bar_type)
        bars = converter.convert_dataframe(df)

        # Process bars through indicator
        Console.info(f"Processing {len(bars)} bars through indicator...")

        indicator_history = []
        for bar in bars:
            smart_money.handle_bar(bar)
            if smart_money.initialized:
                indicator_history.append({
                    "value": smart_money.value,
                    "strength": smart_money.signal_strength,
                    "confidence": smart_money.confidence,
                    "dark_pool": smart_money.dark_pool_signal,
                    "block_trade": smart_money.block_trade_signal,
                    "flow_imbalance": smart_money.flow_imbalance_signal,
                    "volume": smart_money.volume_signal,
                })

        Console.success(f"Indicator initialized after {smart_money.period} bars")

        # Display current values
        print(f"\n  Current Smart Money Indicator Values:")
        print(f"    Value:           {smart_money.value:+.4f}")
        print(f"    Signal Strength: {smart_money.signal_strength:.4f}")
        print(f"    Confidence:      {smart_money.confidence:.4f}")
        print(f"\n  Component Signals:")
        print(f"    Dark Pool:       {smart_money.dark_pool_signal:+.4f}")
        print(f"    Block Trade:     {smart_money.block_trade_signal:+.4f}")
        print(f"    Flow Imbalance:  {smart_money.flow_imbalance_signal:+.4f}")
        print(f"    Volume:          {smart_money.volume_signal:+.4f}")

        # Signal interpretation
        print(f"\n  Signal Interpretation:")
        if smart_money.is_bullish():
            Console.success("  BULLISH - Smart money accumulating")
        elif smart_money.is_bearish():
            Console.warning("  BEARISH - Smart money distributing")
        else:
            Console.info("  NEUTRAL - No strong smart money signal")

        results["smart_money"] = {
            "value": round(smart_money.value, 4),
            "signal_strength": round(smart_money.signal_strength, 4),
            "confidence": round(smart_money.confidence, 4),
            "components": {
                "dark_pool": round(smart_money.dark_pool_signal, 4),
                "block_trade": round(smart_money.block_trade_signal, 4),
                "flow_imbalance": round(smart_money.flow_imbalance_signal, 4),
                "volume": round(smart_money.volume_signal, 4),
            },
            "is_bullish": smart_money.is_bullish(),
            "is_bearish": smart_money.is_bearish(),
        }

    except ImportError as e:
        Console.warning(f"SmartMoneyIndicator not available: {e}")
        Console.info("Showing simulated indicator output...")

        # Simulate indicator values
        np.random.seed(sum(ord(c) for c in symbol))
        value = np.random.randn() * 0.3

        results["smart_money"] = {
            "value": round(value, 4),
            "signal_strength": round(abs(value), 4),
            "confidence": round(0.5 + abs(value) * 0.3, 4),
            "components": {
                "dark_pool": round(np.random.randn() * 0.2, 4),
                "block_trade": round(np.random.randn() * 0.2, 4),
                "flow_imbalance": round(np.random.randn() * 0.3, 4),
                "volume": round(np.random.randn() * 0.2, 4),
            },
            "is_bullish": value > 0.3,
            "is_bearish": value < -0.3,
        }

        print(f"\n  [Simulated] Smart Money Indicator Values:")
        print(f"    Value:           {results['smart_money']['value']:+.4f}")
        print(f"    Signal Strength: {results['smart_money']['signal_strength']:.4f}")

    # Institutional Momentum Indicator
    Console.subheader("InstitutionalMomentumIndicator")

    Console.info("The InstitutionalMomentumIndicator tracks positioning trends by:")
    Console.info("  - Ownership trend (increasing/decreasing institutional ownership)")
    Console.info("  - Smart money score from Stanley's InstitutionalAnalyzer")
    Console.info("  - Concentration changes (diversification vs concentration)")
    Console.info("  - Price momentum correlation with institutional activity")

    try:
        from stanley.integrations.nautilus import InstitutionalMomentumIndicator
        from stanley.integrations.nautilus import create_instrument_id, create_bar_type, OpenBBBarConverter
        from nautilus_trader.model.enums import BarAggregation

        Console.info("\nInitializing InstitutionalMomentumIndicator...")

        # Create indicator
        inst_momentum = InstitutionalMomentumIndicator(
            period=20,
            ownership_weight=0.35,
            smart_money_weight=0.30,
            concentration_weight=0.15,
            momentum_weight=0.20,
            symbol=symbol,
        )

        # Convert data to Nautilus bars if not already done
        if 'bars' not in dir():
            instrument_id = create_instrument_id(symbol, "OPENBB")
            bar_type = create_bar_type(instrument_id, BarAggregation.DAY, 1)
            converter = OpenBBBarConverter(instrument_id, bar_type)
            bars = converter.convert_dataframe(df)

        # Process bars
        Console.info(f"Processing {len(bars)} bars through indicator...")

        for bar in bars:
            inst_momentum.handle_bar(bar)

        Console.success(f"Indicator initialized after {inst_momentum.period} bars")

        # Display current values
        print(f"\n  Current Institutional Momentum Values:")
        print(f"    Value:           {inst_momentum.value:+.4f}")
        print(f"    Signal Strength: {inst_momentum.signal_strength:.4f}")
        print(f"    Confidence:      {inst_momentum.confidence:.4f}")

        components = inst_momentum.get_component_signals()
        print(f"\n  Component Signals:")
        print(f"    Ownership:       {components['ownership']:+.4f}")
        print(f"    Smart Money:     {components['smart_money']:+.4f}")
        print(f"    Concentration:   {components['concentration']:+.4f}")
        print(f"    Momentum:        {components['momentum']:+.4f}")

        # Signal interpretation
        print(f"\n  Signal Interpretation:")
        sentiment = inst_momentum.get_sentiment()
        if sentiment == "bullish":
            Console.success(f"  BULLISH - Institutions building positions")
        elif sentiment == "bearish":
            Console.warning(f"  BEARISH - Institutions reducing positions")
        else:
            Console.info(f"  NEUTRAL - Mixed institutional activity")

        results["institutional_momentum"] = {
            "value": round(inst_momentum.value, 4),
            "signal_strength": round(inst_momentum.signal_strength, 4),
            "confidence": round(inst_momentum.confidence, 4),
            "components": {k: round(v, 4) for k, v in components.items()},
            "sentiment": sentiment,
        }

    except ImportError as e:
        Console.warning(f"InstitutionalMomentumIndicator not available: {e}")
        Console.info("Showing simulated indicator output...")

        # Simulate indicator values
        np.random.seed(sum(ord(c) for c in symbol) + 1)
        value = np.random.randn() * 0.25

        results["institutional_momentum"] = {
            "value": round(value, 4),
            "signal_strength": round(abs(value), 4),
            "confidence": round(0.4 + abs(value) * 0.4, 4),
            "components": {
                "ownership": round(np.random.randn() * 0.2, 4),
                "smart_money": round(np.random.randn() * 0.2, 4),
                "concentration": round(np.random.randn() * 0.15, 4),
                "momentum": round(np.random.randn() * 0.25, 4),
            },
            "sentiment": "bullish" if value > 0.3 else ("bearish" if value < -0.3 else "neutral"),
        }

        print(f"\n  [Simulated] Institutional Momentum Values:")
        print(f"    Value:           {results['institutional_momentum']['value']:+.4f}")
        print(f"    Signal Strength: {results['institutional_momentum']['signal_strength']:.4f}")

    # Combined Analysis
    Console.subheader("Strategy Integration Example")

    Console.info("In a NautilusTrader strategy, you would use these indicators like this:\n")

    print("""  class InstitutionalStrategy(Strategy):
      def __init__(self, config):
          super().__init__(config)
          self.smart_money = SmartMoneyIndicator(period=20)
          self.inst_momentum = InstitutionalMomentumIndicator(period=20)

      def on_bar(self, bar: Bar):
          self.smart_money.handle_bar(bar)
          self.inst_momentum.handle_bar(bar)

          if not self.smart_money.initialized:
              return

          # Generate trading signal
          if (self.smart_money.is_bullish() and
              self.inst_momentum.get_sentiment() == 'bullish'):
              # Strong buy signal
              self.buy(...)
          elif (self.smart_money.is_bearish() and
                self.inst_momentum.get_sentiment() == 'bearish'):
              # Strong sell signal
              self.sell(...)
  """)

    return results


# =============================================================================
# Backtest Demo
# =============================================================================

async def run_backtest_demo(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
) -> Dict[str, Any]:
    """
    Demonstrate backtest setup with Stanley integration.

    This shows the structure and configuration needed to run
    a backtest using Stanley's indicators in NautilusTrader.

    Args:
        symbol: Stock ticker symbol
        start_date: Backtest start date
        end_date: Backtest end date

    Returns:
        Dictionary with backtest configuration details
    """
    Console.header("BACKTEST DEMO: NautilusTrader Engine Setup")

    Console.info("This demo shows how to configure a NautilusTrader backtest")
    Console.info("with Stanley's institutional analysis integration.\n")

    # Configuration overview
    Console.subheader("Backtest Configuration")

    config = {
        "symbol": symbol,
        "venue": "OPENBB",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "data_client": "OpenBBDataClient",
        "indicators": [
            "SmartMoneyIndicator",
            "InstitutionalMomentumIndicator",
        ],
        "actors": [
            "MoneyFlowActor",
            "InstitutionalActor",
        ],
    }

    Console.table(
        ["Setting", "Value"],
        [[k, str(v)] for k, v in config.items()],
        "Backtest Parameters"
    )

    # Show configuration code
    Console.subheader("Engine Configuration Code")

    print(f"""
  from nautilus_trader.backtest.engine import BacktestEngine
  from nautilus_trader.backtest.config import BacktestEngineConfig, BacktestRunConfig
  from stanley.integrations.nautilus import (
      OpenBBDataClientConfig,
      MoneyFlowActorConfig,
      InstitutionalActorConfig,
  )

  # Configure the backtest engine
  engine_config = BacktestEngineConfig(
      trader_id="BACKTEST-001",
      logging=LoggingConfig(log_level="INFO"),
  )

  # Create the engine
  engine = BacktestEngine(config=engine_config)

  # Add venue
  engine.add_venue(
      venue=Venue("{config['venue']}"),
      oms_type=OmsType.HEDGING,
      account_type=AccountType.MARGIN,
  )

  # Configure data client
  data_client_config = OpenBBDataClientConfig(
      venue="{config['venue']}",
      openbb_token="your-token",  # Optional for yfinance
      provider="yfinance",
  )

  # Add instruments
  instrument = create_equity("{symbol}", venue="{config['venue']}")
  engine.add_instrument(instrument)

  # Add actors for signal generation
  money_flow_config = MoneyFlowActorConfig(
      symbols=["{symbol}.{config['venue']}"],
      lookback_bars=20,
      enable_dark_pool=True,
  )

  institutional_config = InstitutionalActorConfig(
      universe=["{symbol}.{config['venue']}"],
      minimum_aum=1e9,
  )
  """)

    # Strategy template
    Console.subheader("Strategy Template")

    print(f"""
  from nautilus_trader.trading.strategy import Strategy
  from stanley.integrations.nautilus import (
      SmartMoneyIndicator,
      InstitutionalMomentumIndicator,
  )

  class InstitutionalFlowStrategy(Strategy):
      '''
      Strategy using Stanley's institutional indicators.

      Entry Logic:
      - Buy when both SmartMoney and Institutional indicators are bullish
      - Sell when both indicators are bearish

      Risk Management:
      - Position size based on indicator confidence
      - Stop loss at 2% below entry
      '''

      def __init__(self, config):
          super().__init__(config)

          # Initialize indicators
          self.smart_money = SmartMoneyIndicator(period=20)
          self.inst_momentum = InstitutionalMomentumIndicator(
              period=20,
              symbol="{symbol}",
          )

          # Track position
          self.position = None

      def on_start(self):
          # Subscribe to bars
          self.subscribe_bars(BarType.from_str("{symbol}.{config['venue']}-1-DAY-LAST-EXTERNAL"))

      def on_bar(self, bar: Bar):
          # Update indicators
          self.smart_money.handle_bar(bar)
          self.inst_momentum.handle_bar(bar)

          if not self.smart_money.initialized:
              return

          # Calculate position size based on confidence
          confidence = (self.smart_money.confidence + self.inst_momentum.confidence) / 2
          position_size = self.calculate_position_size(confidence)

          # Entry logic
          if self.position is None:
              if self.smart_money.is_bullish() and self.inst_momentum.is_bullish():
                  self.enter_long(position_size)
              elif self.smart_money.is_bearish() and self.inst_momentum.is_bearish():
                  self.enter_short(position_size)

          # Exit logic
          elif self.position.is_long:
              if self.smart_money.is_bearish() or self.inst_momentum.is_bearish():
                  self.close_position()
          elif self.position.is_short:
              if self.smart_money.is_bullish() or self.inst_momentum.is_bullish():
                  self.close_position()
  """)

    # Running the backtest
    Console.subheader("Running the Backtest")

    print("""
  # Load historical data
  df = adapter.get_historical_data(symbol, start_date, end_date)
  bars = converter.convert_dataframe(df)

  # Add data to engine
  engine.add_data(bars)

  # Add strategy
  strategy_config = InstitutionalFlowStrategyConfig(
      instrument_id=instrument.id,
      bar_type=bar_type,
  )
  strategy = InstitutionalFlowStrategy(config=strategy_config)
  engine.add_strategy(strategy)

  # Run backtest
  engine.run()

  # Get results
  results = engine.get_result()
  print(f"Total Return: {results.total_return:.2%}")
  print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
  print(f"Max Drawdown: {results.max_drawdown:.2%}")
  """)

    Console.success("\nBacktest configuration complete!")
    Console.info("This template can be customized based on your specific requirements.")

    return config


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Stanley-NautilusTrader Integration Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_integration.py
  python demo_integration.py --symbol MSFT
  python demo_integration.py --symbol AAPL --start-date 2024-01-01 --end-date 2024-06-30
  python demo_integration.py --mode signals
  python demo_integration.py --mode indicators
  python demo_integration.py --mode backtest
  python demo_integration.py --live-data  # Attempt to fetch real data from OpenBB
        """,
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default="AAPL",
        help="Stock ticker symbol to analyze (default: AAPL)",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date in YYYY-MM-DD format (default: 60 days ago)",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date in YYYY-MM-DD format (default: today)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "data", "signals", "indicators", "backtest"],
        default="all",
        help="Demo mode to run (default: all)",
    )

    parser.add_argument(
        "--live-data",
        action="store_true",
        help="Attempt to fetch live data from OpenBB",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


async def main() -> None:
    """Main entry point for the demo."""
    args = parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse dates
    end_date = (
        datetime.strptime(args.end_date, "%Y-%m-%d")
        if args.end_date
        else datetime.now()
    )
    start_date = (
        datetime.strptime(args.start_date, "%Y-%m-%d")
        if args.start_date
        else end_date - timedelta(days=60)
    )

    # Print header
    Console.header("STANLEY-NAUTILUSTRADER INTEGRATION DEMO")

    print(f"  Symbol:     {args.symbol}")
    print(f"  Date Range: {start_date.date()} to {end_date.date()}")
    print(f"  Mode:       {args.mode}")
    print(f"  Live Data:  {'Yes' if args.live_data else 'No (using demo data)'}")

    # Shared data frame
    df = None

    # Run selected demos
    if args.mode in ("all", "data"):
        df = await run_data_demo(
            args.symbol,
            start_date,
            end_date,
            use_live_data=args.live_data,
        )

    if args.mode in ("all", "signals"):
        await run_signal_demo(args.symbol, df)

    if args.mode in ("all", "indicators"):
        await run_indicator_demo(args.symbol, df)

    if args.mode in ("all", "backtest"):
        await run_backtest_demo(args.symbol, start_date, end_date)

    # Final summary
    Console.header("DEMO COMPLETE")

    Console.info("Key Takeaways:")
    print("  1. OpenBB provides flexible data sourcing with multiple providers")
    print("  2. Stanley's indicators track institutional and smart money activity")
    print("  3. The integration enables strategy development based on money flow analysis")
    print("  4. NautilusTrader provides the execution and backtesting infrastructure")

    print(f"\n  For more information, see:")
    print("    - Stanley docs: CLAUDE.md")
    print("    - NautilusTrader: https://nautilustrader.io")
    print("    - OpenBB: https://openbb.co")


if __name__ == "__main__":
    asyncio.run(main())
