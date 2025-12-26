"""
Signal Backtester Module

Backtest investment signals against historical data
to evaluate signal quality and performance.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .signal_generator import Signal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    # Time period
    start_date: datetime
    end_date: datetime

    # Position sizing
    initial_capital: float = 100_000.0
    position_size_pct: float = 0.10  # 10% per position
    max_positions: int = 10
    max_sector_exposure: float = 0.30  # 30% max per sector

    # Trade execution
    slippage_bps: float = 10.0  # 10 basis points
    commission_per_trade: float = 1.0

    # Risk management
    use_stop_loss: bool = True
    use_target_price: bool = True
    trailing_stop_pct: Optional[float] = None
    max_holding_days: int = 90

    # Benchmark
    benchmark: str = "SPY"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "startDate": self.start_date.isoformat(),
            "endDate": self.end_date.isoformat(),
            "initialCapital": self.initial_capital,
            "positionSizePct": self.position_size_pct,
            "maxPositions": self.max_positions,
            "maxSectorExposure": self.max_sector_exposure,
            "slippageBps": self.slippage_bps,
            "commissionPerTrade": self.commission_per_trade,
            "useStopLoss": self.use_stop_loss,
            "useTargetPrice": self.use_target_price,
            "trailingStopPct": self.trailing_stop_pct,
            "maxHoldingDays": self.max_holding_days,
            "benchmark": self.benchmark,
        }


@dataclass
class TradeResult:
    """Result of a single trade."""

    symbol: str
    signal_id: str
    signal_type: SignalType
    entry_date: datetime
    entry_price: float
    exit_date: datetime
    exit_price: float
    shares: float
    pnl: float
    pnl_percent: float
    holding_days: int
    exit_reason: (
        str  # "target", "stop_loss", "max_holding", "reversal", "end_of_period"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "signalId": self.signal_id,
            "signalType": self.signal_type.value,
            "entryDate": self.entry_date.isoformat(),
            "entryPrice": round(self.entry_price, 2),
            "exitDate": self.exit_date.isoformat(),
            "exitPrice": round(self.exit_price, 2),
            "shares": round(self.shares, 2),
            "pnl": round(self.pnl, 2),
            "pnlPercent": round(self.pnl_percent, 4),
            "holdingDays": self.holding_days,
            "exitReason": self.exit_reason,
        }


@dataclass
class BacktestResult:
    """Complete backtest result with metrics."""

    config: BacktestConfig
    trades: List[TradeResult]

    # Performance metrics
    total_return: float
    total_return_percent: float
    annualized_return: float
    benchmark_return: float
    alpha: float
    beta: float

    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: int
    volatility: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_holding_days: float

    # Signal breakdown
    buy_signal_performance: Dict[str, float]
    sell_signal_performance: Dict[str, float]

    # Time series
    equity_curve: pd.Series = field(default_factory=pd.Series)
    drawdown_series: pd.Series = field(default_factory=pd.Series)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config": self.config.to_dict(),
            "trades": [t.to_dict() for t in self.trades],
            "performance": {
                "totalReturn": round(self.total_return, 2),
                "totalReturnPercent": round(self.total_return_percent, 4),
                "annualizedReturn": round(self.annualized_return, 4),
                "benchmarkReturn": round(self.benchmark_return, 4),
                "alpha": round(self.alpha, 4),
                "beta": round(self.beta, 4),
            },
            "risk": {
                "sharpeRatio": round(self.sharpe_ratio, 4),
                "sortinoRatio": round(self.sortino_ratio, 4),
                "maxDrawdown": round(self.max_drawdown, 4),
                "maxDrawdownDurationDays": self.max_drawdown_duration_days,
                "volatility": round(self.volatility, 4),
            },
            "tradeStats": {
                "totalTrades": self.total_trades,
                "winningTrades": self.winning_trades,
                "losingTrades": self.losing_trades,
                "winRate": round(self.win_rate, 4),
                "avgWin": round(self.avg_win, 4),
                "avgLoss": round(self.avg_loss, 4),
                "profitFactor": round(self.profit_factor, 4),
                "avgHoldingDays": round(self.avg_holding_days, 1),
            },
            "signalBreakdown": {
                "buySignals": {
                    k: round(v, 4) for k, v in self.buy_signal_performance.items()
                },
                "sellSignals": {
                    k: round(v, 4) for k, v in self.sell_signal_performance.items()
                },
            },
        }


class SignalBacktester:
    """
    Backtest investment signals against historical data.

    Simulates trade execution with realistic assumptions
    including slippage, commissions, and position sizing.
    """

    def __init__(self, data_manager=None):
        """
        Initialize backtester.

        Args:
            data_manager: DataManager instance for price data
        """
        self.data_manager = data_manager
        logger.info("SignalBacktester initialized")

    async def run_backtest(
        self,
        signals: List[Signal],
        price_data: Optional[Dict[str, pd.DataFrame]] = None,
        config: Optional[BacktestConfig] = None,
    ) -> BacktestResult:
        """
        Run backtest on a list of signals.

        Args:
            signals: List of Signal objects to backtest
            price_data: Dict of symbol to OHLCV DataFrame (optional, will fetch if not provided)
            config: BacktestConfig (optional, uses defaults if not provided)

        Returns:
            BacktestResult with performance metrics
        """
        if not signals:
            return self._empty_result(config or self._default_config())

        if config is None:
            # Infer config from signals
            timestamps = [s.timestamp for s in signals]
            config = BacktestConfig(
                start_date=min(timestamps),
                end_date=max(timestamps) + timedelta(days=90),
            )

        logger.info(
            f"Running backtest on {len(signals)} signals from "
            f"{config.start_date.date()} to {config.end_date.date()}"
        )

        # Get price data if not provided
        if price_data is None:
            price_data = await self._fetch_price_data(signals, config)

        # Sort signals by timestamp
        sorted_signals = sorted(signals, key=lambda s: s.timestamp)

        # Simulate trades
        trades = await self._simulate_trades(sorted_signals, price_data, config)

        # Calculate performance metrics
        result = self.calculate_performance_metrics(trades, config, price_data)

        logger.info(
            f"Backtest complete: {result.total_trades} trades, "
            f"{result.win_rate:.1%} win rate, "
            f"{result.total_return_percent:.2%} total return"
        )

        return result

    async def _fetch_price_data(
        self,
        signals: List[Signal],
        config: BacktestConfig,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch price data for all symbols in signals."""
        symbols = list(set(s.symbol for s in signals))
        symbols.append(config.benchmark)

        price_data = {}

        for symbol in symbols:
            try:
                if self.data_manager:
                    data = await self.data_manager.get_stock_data(
                        symbol,
                        config.start_date - timedelta(days=30),
                        config.end_date + timedelta(days=30),
                    )
                    if not data.empty:
                        price_data[symbol] = data
                else:
                    # Generate mock price data
                    price_data[symbol] = self._generate_mock_prices(
                        symbol, config.start_date, config.end_date
                    )
            except Exception as e:
                logger.warning(f"Failed to fetch price data for {symbol}: {e}")
                price_data[symbol] = self._generate_mock_prices(
                    symbol, config.start_date, config.end_date
                )

        return price_data

    async def _simulate_trades(
        self,
        signals: List[Signal],
        price_data: Dict[str, pd.DataFrame],
        config: BacktestConfig,
    ) -> List[TradeResult]:
        """Simulate trade execution for signals."""
        trades = []
        open_positions = {}
        cash = config.initial_capital

        for signal in signals:
            symbol = signal.symbol

            if symbol not in price_data or price_data[symbol].empty:
                continue

            prices = price_data[symbol]

            # Check if we can take this position
            if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                if len(open_positions) >= config.max_positions:
                    continue

                # Get entry price
                entry_date = signal.timestamp
                entry_price = self._get_price_at_date(prices, entry_date)

                if entry_price is None:
                    continue

                # Apply slippage
                slippage = entry_price * config.slippage_bps / 10000
                if signal.signal_type == SignalType.BUY:
                    entry_price += slippage
                else:
                    entry_price -= slippage

                # Calculate position size
                position_value = cash * config.position_size_pct
                shares = position_value / entry_price

                # Store open position
                open_positions[signal.signal_id] = {
                    "signal": signal,
                    "symbol": symbol,
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "shares": shares,
                    "target_price": signal.target_price,
                    "stop_loss": signal.stop_loss,
                    "max_holding_date": entry_date
                    + timedelta(days=config.max_holding_days),
                }

                # Deduct from cash
                cash -= position_value + config.commission_per_trade

        # Close all open positions at end of period or based on exit conditions
        for signal_id, position in open_positions.items():
            symbol = position["symbol"]
            prices = price_data.get(symbol)

            if prices is None or prices.empty:
                continue

            # Determine exit
            exit_date, exit_price, exit_reason = self._determine_exit(
                position, prices, config
            )

            # Apply slippage
            slippage = exit_price * config.slippage_bps / 10000
            if position["signal"].signal_type == SignalType.BUY:
                exit_price -= slippage  # Selling at lower price
            else:
                exit_price += slippage  # Covering short at higher price

            # Calculate PnL
            if position["signal"].signal_type == SignalType.BUY:
                pnl = (exit_price - position["entry_price"]) * position["shares"]
            else:
                pnl = (position["entry_price"] - exit_price) * position["shares"]

            pnl -= config.commission_per_trade
            pnl_percent = pnl / (position["entry_price"] * position["shares"])

            holding_days = (exit_date - position["entry_date"]).days

            trades.append(
                TradeResult(
                    symbol=symbol,
                    signal_id=signal_id,
                    signal_type=position["signal"].signal_type,
                    entry_date=position["entry_date"],
                    entry_price=position["entry_price"],
                    exit_date=exit_date,
                    exit_price=exit_price,
                    shares=position["shares"],
                    pnl=pnl,
                    pnl_percent=pnl_percent,
                    holding_days=holding_days,
                    exit_reason=exit_reason,
                )
            )

        return trades

    def _determine_exit(
        self,
        position: Dict[str, Any],
        prices: pd.DataFrame,
        config: BacktestConfig,
    ) -> tuple[datetime, float, str]:
        """Determine when and why to exit a position."""
        entry_date = position["entry_date"]
        max_date = min(position["max_holding_date"], config.end_date)
        signal_type = position["signal"].signal_type

        # Get prices from entry to max holding date
        mask = (prices.index >= entry_date) & (prices.index <= max_date)
        period_prices = prices[mask]

        if period_prices.empty:
            # No data, use entry price
            return max_date, position["entry_price"], "no_data"

        # Check for stop loss and target hits
        for idx, row in period_prices.iterrows():
            high = row.get("high", row.get("close", 0))
            low = row.get("low", row.get("close", 0))
            # close = row.get("close", 0)  # Reserved for future use

            if signal_type == SignalType.BUY:
                # Check stop loss
                if config.use_stop_loss and position["stop_loss"]:
                    if low <= position["stop_loss"]:
                        return idx, position["stop_loss"], "stop_loss"

                # Check target
                if config.use_target_price and position["target_price"]:
                    if high >= position["target_price"]:
                        return idx, position["target_price"], "target"

            elif signal_type == SignalType.SELL:
                # Check stop loss (for short)
                if config.use_stop_loss and position["stop_loss"]:
                    if high >= position["stop_loss"]:
                        return idx, position["stop_loss"], "stop_loss"

                # Check target (for short)
                if config.use_target_price and position["target_price"]:
                    if low <= position["target_price"]:
                        return idx, position["target_price"], "target"

        # Exit at end of period
        last_price = period_prices["close"].iloc[-1]
        return period_prices.index[-1], last_price, "max_holding"

    def calculate_performance_metrics(
        self,
        trades: List[TradeResult],
        config: BacktestConfig,
        price_data: Dict[str, pd.DataFrame],
    ) -> BacktestResult:
        """Calculate comprehensive performance metrics from trades."""
        if not trades:
            return self._empty_result(config)

        # Basic trade statistics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.pnl > 0)
        losing_trades = sum(1 for t in trades if t.pnl < 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # PnL statistics
        wins = [t.pnl_percent for t in trades if t.pnl > 0]
        losses = [t.pnl_percent for t in trades if t.pnl < 0]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        total_wins = sum(t.pnl for t in trades if t.pnl > 0)
        total_losses = abs(sum(t.pnl for t in trades if t.pnl < 0))
        profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

        avg_holding_days = np.mean([t.holding_days for t in trades])

        # Total return
        total_pnl = sum(t.pnl for t in trades)
        total_return_percent = total_pnl / config.initial_capital

        # Calculate equity curve
        equity_curve = self._calculate_equity_curve(trades, config)

        # Annualized return
        days = (config.end_date - config.start_date).days
        years = days / 365.25
        annualized_return = (
            (1 + total_return_percent) ** (1 / years) - 1 if years > 0 else 0
        )

        # Benchmark return
        benchmark_return = self._calculate_benchmark_return(config, price_data)

        # Alpha and beta
        alpha = annualized_return - benchmark_return
        beta = 1.0  # Simplified, would need proper regression

        # Risk metrics
        if not equity_curve.empty:
            returns = equity_curve.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

            downside_returns = returns[returns < 0]
            downside_vol = (
                downside_returns.std() * np.sqrt(252)
                if len(downside_returns) > 0
                else 0
            )
            sortino_ratio = annualized_return / downside_vol if downside_vol > 0 else 0

            drawdown_series = self._calculate_drawdown(equity_curve)
            max_drawdown = drawdown_series.min() if not drawdown_series.empty else 0
            max_drawdown_duration = self._calculate_max_drawdown_duration(
                drawdown_series
            )
        else:
            volatility = 0
            sharpe_ratio = 0
            sortino_ratio = 0
            max_drawdown = 0
            max_drawdown_duration = 0
            drawdown_series = pd.Series()

        # Signal breakdown
        buy_trades = [t for t in trades if t.signal_type == SignalType.BUY]
        sell_trades = [t for t in trades if t.signal_type == SignalType.SELL]

        buy_signal_performance = {
            "count": len(buy_trades),
            "win_rate": (
                sum(1 for t in buy_trades if t.pnl > 0) / len(buy_trades)
                if buy_trades
                else 0
            ),
            "avg_return": (
                np.mean([t.pnl_percent for t in buy_trades]) if buy_trades else 0
            ),
        }

        sell_signal_performance = {
            "count": len(sell_trades),
            "win_rate": (
                sum(1 for t in sell_trades if t.pnl > 0) / len(sell_trades)
                if sell_trades
                else 0
            ),
            "avg_return": (
                np.mean([t.pnl_percent for t in sell_trades]) if sell_trades else 0
            ),
        }

        return BacktestResult(
            config=config,
            trades=trades,
            total_return=total_pnl,
            total_return_percent=total_return_percent,
            annualized_return=annualized_return,
            benchmark_return=benchmark_return,
            alpha=alpha,
            beta=beta,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration_days=max_drawdown_duration,
            volatility=volatility,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_holding_days=avg_holding_days,
            buy_signal_performance=buy_signal_performance,
            sell_signal_performance=sell_signal_performance,
            equity_curve=equity_curve,
            drawdown_series=drawdown_series,
        )

    def generate_attribution_report(
        self,
        result: BacktestResult,
    ) -> Dict[str, Any]:
        """
        Generate performance attribution report.

        Args:
            result: BacktestResult from run_backtest

        Returns:
            Dict with attribution breakdown
        """
        trades = result.trades

        if not trades:
            return {
                "summary": "No trades to analyze",
                "by_exit_reason": {},
                "by_holding_period": {},
                "by_symbol": {},
                "monthly_performance": {},
            }

        # Attribution by exit reason
        exit_reasons = {}
        for trade in trades:
            reason = trade.exit_reason
            if reason not in exit_reasons:
                exit_reasons[reason] = {"count": 0, "total_pnl": 0, "win_rate": 0}
            exit_reasons[reason]["count"] += 1
            exit_reasons[reason]["total_pnl"] += trade.pnl

        for reason in exit_reasons:
            reason_trades = [t for t in trades if t.exit_reason == reason]
            wins = sum(1 for t in reason_trades if t.pnl > 0)
            exit_reasons[reason]["win_rate"] = (
                wins / len(reason_trades) if reason_trades else 0
            )

        # Attribution by holding period bucket
        holding_buckets = {
            "< 7 days": {"count": 0, "total_pnl": 0, "avg_return": 0},
            "7-30 days": {"count": 0, "total_pnl": 0, "avg_return": 0},
            "30-60 days": {"count": 0, "total_pnl": 0, "avg_return": 0},
            "> 60 days": {"count": 0, "total_pnl": 0, "avg_return": 0},
        }

        for trade in trades:
            if trade.holding_days < 7:
                bucket = "< 7 days"
            elif trade.holding_days < 30:
                bucket = "7-30 days"
            elif trade.holding_days < 60:
                bucket = "30-60 days"
            else:
                bucket = "> 60 days"

            holding_buckets[bucket]["count"] += 1
            holding_buckets[bucket]["total_pnl"] += trade.pnl

        for bucket in holding_buckets:
            if holding_buckets[bucket]["count"] > 0:
                bucket_trades = [
                    t for t in trades if self._get_bucket(t.holding_days) == bucket
                ]
                holding_buckets[bucket]["avg_return"] = np.mean(
                    [t.pnl_percent for t in bucket_trades]
                )

        # Attribution by symbol
        symbol_perf = {}
        for trade in trades:
            if trade.symbol not in symbol_perf:
                symbol_perf[trade.symbol] = {"count": 0, "total_pnl": 0, "win_rate": 0}
            symbol_perf[trade.symbol]["count"] += 1
            symbol_perf[trade.symbol]["total_pnl"] += trade.pnl

        for symbol in symbol_perf:
            symbol_trades = [t for t in trades if t.symbol == symbol]
            wins = sum(1 for t in symbol_trades if t.pnl > 0)
            symbol_perf[symbol]["win_rate"] = (
                wins / len(symbol_trades) if symbol_trades else 0
            )

        # Sort by total PnL
        symbol_perf = dict(
            sorted(symbol_perf.items(), key=lambda x: x[1]["total_pnl"], reverse=True)
        )

        # Monthly performance
        monthly = {}
        for trade in trades:
            month_key = trade.exit_date.strftime("%Y-%m")
            if month_key not in monthly:
                monthly[month_key] = {"count": 0, "total_pnl": 0, "win_rate": 0}
            monthly[month_key]["count"] += 1
            monthly[month_key]["total_pnl"] += trade.pnl

        for month in monthly:
            month_trades = [t for t in trades if t.exit_date.strftime("%Y-%m") == month]
            wins = sum(1 for t in month_trades if t.pnl > 0)
            monthly[month]["win_rate"] = wins / len(month_trades) if month_trades else 0

        return {
            "summary": {
                "total_trades": len(trades),
                "total_pnl": sum(t.pnl for t in trades),
                "best_trade": max(t.pnl for t in trades),
                "worst_trade": min(t.pnl for t in trades),
            },
            "by_exit_reason": exit_reasons,
            "by_holding_period": holding_buckets,
            "by_symbol": symbol_perf,
            "monthly_performance": monthly,
        }

    def _get_bucket(self, holding_days: int) -> str:
        """Get holding period bucket for a trade."""
        if holding_days < 7:
            return "< 7 days"
        elif holding_days < 30:
            return "7-30 days"
        elif holding_days < 60:
            return "30-60 days"
        else:
            return "> 60 days"

    def _get_price_at_date(
        self,
        prices: pd.DataFrame,
        date: datetime,
    ) -> Optional[float]:
        """Get price at or near a specific date."""
        if prices.empty:
            return None

        # Find closest date
        if date in prices.index:
            return prices.loc[date, "close"]

        # Find next available date
        future_dates = prices.index[prices.index >= date]
        if len(future_dates) > 0:
            return prices.loc[future_dates[0], "close"]

        # Find previous date
        past_dates = prices.index[prices.index < date]
        if len(past_dates) > 0:
            return prices.loc[past_dates[-1], "close"]

        return None

    def _calculate_equity_curve(
        self,
        trades: List[TradeResult],
        config: BacktestConfig,
    ) -> pd.Series:
        """Calculate equity curve from trades."""
        if not trades:
            return pd.Series()

        # Create daily equity curve
        dates = pd.date_range(config.start_date, config.end_date, freq="D")
        equity = pd.Series(index=dates, data=float(config.initial_capital), dtype=float)

        # Apply trade results
        cumulative_pnl = 0
        for trade in sorted(trades, key=lambda t: t.exit_date):
            cumulative_pnl += trade.pnl
            # Update equity from exit date forward
            equity[trade.exit_date :] = config.initial_capital + cumulative_pnl

        return equity

    def _calculate_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series from equity curve."""
        if equity_curve.empty:
            return pd.Series()

        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        return drawdown

    def _calculate_max_drawdown_duration(self, drawdown_series: pd.Series) -> int:
        """Calculate maximum drawdown duration in days."""
        if drawdown_series.empty:
            return 0

        in_drawdown = drawdown_series < 0
        if not in_drawdown.any():
            return 0

        # Find consecutive drawdown periods
        drawdown_starts = in_drawdown.ne(in_drawdown.shift()).cumsum()
        drawdown_groups = in_drawdown.groupby(drawdown_starts)

        max_duration = 0
        for group_id, group in drawdown_groups:
            if group.any():
                duration = len(group)
                max_duration = max(max_duration, duration)

        return max_duration

    def _calculate_benchmark_return(
        self,
        config: BacktestConfig,
        price_data: Dict[str, pd.DataFrame],
    ) -> float:
        """Calculate benchmark return over the backtest period."""
        benchmark = config.benchmark

        if benchmark not in price_data or price_data[benchmark].empty:
            return 0.10  # Default 10% annual

        prices = price_data[benchmark]

        start_price = self._get_price_at_date(prices, config.start_date)
        end_price = self._get_price_at_date(prices, config.end_date)

        if start_price and end_price and start_price > 0:
            total_return = (end_price / start_price) - 1
            days = (config.end_date - config.start_date).days
            years = days / 365.25
            annualized = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
            return annualized

        return 0.10

    def _generate_mock_prices(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Generate mock price data for testing."""
        dates = pd.date_range(start_date, end_date, freq="D")
        n = len(dates)

        # Random walk with drift
        returns = np.random.normal(0.0005, 0.02, n)
        prices = 100 * np.cumprod(1 + returns)

        df = pd.DataFrame(
            {
                "open": prices * (1 + np.random.uniform(-0.01, 0.01, n)),
                "high": prices * (1 + np.random.uniform(0, 0.02, n)),
                "low": prices * (1 - np.random.uniform(0, 0.02, n)),
                "close": prices,
                "volume": np.random.randint(1000000, 10000000, n),
            },
            index=dates,
        )

        return df

    def _default_config(self) -> BacktestConfig:
        """Return default backtest configuration."""
        return BacktestConfig(
            start_date=datetime.now() - timedelta(days=365),
            end_date=datetime.now(),
        )

    def _empty_result(self, config: BacktestConfig) -> BacktestResult:
        """Return empty backtest result."""
        return BacktestResult(
            config=config,
            trades=[],
            total_return=0,
            total_return_percent=0,
            annualized_return=0,
            benchmark_return=0,
            alpha=0,
            beta=1,
            sharpe_ratio=0,
            sortino_ratio=0,
            max_drawdown=0,
            max_drawdown_duration_days=0,
            volatility=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            avg_win=0,
            avg_loss=0,
            profit_factor=0,
            avg_holding_days=0,
            buy_signal_performance={},
            sell_signal_performance={},
            equity_curve=pd.Series(),
            drawdown_series=pd.Series(),
        )

    def health_check(self) -> bool:
        """Check if backtester is operational."""
        return True
