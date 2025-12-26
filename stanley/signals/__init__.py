"""
Signals Module

Provides signal generation, backtesting, and performance tracking
for institutional investment analysis.
"""

from .backtester import (
    BacktestConfig,
    BacktestResult,
    SignalBacktester,
    TradeResult,
)
from .performance_tracker import (
    PerformanceStats,
    PerformanceTracker,
    TradeRecord,
)
from .signal_generator import (
    CompositeSignal,
    Signal,
    SignalGenerator,
    SignalStrength,
    SignalType,
)

__all__ = [
    # Signal generation
    "SignalType",
    "SignalStrength",
    "Signal",
    "CompositeSignal",
    "SignalGenerator",
    # Backtesting
    "BacktestConfig",
    "BacktestResult",
    "TradeResult",
    "SignalBacktester",
    # Performance tracking
    "TradeRecord",
    "PerformanceStats",
    "PerformanceTracker",
]
