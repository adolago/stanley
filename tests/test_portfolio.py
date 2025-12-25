"""Tests for the Portfolio module."""

import pytest
import numpy as np
import pandas as pd

from stanley.portfolio import PortfolioAnalyzer
from stanley.portfolio.position import Position, Holdings
from stanley.portfolio.risk_metrics import (
    calculate_portfolio_var,
    calculate_beta,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    VaRResult,
    BetaResult,
)


class TestPosition:
    """Tests for Position dataclass."""

    def test_position_creation(self):
        pos = Position(
            symbol="AAPL",
            shares=100,
            average_cost=150.0,
            current_price=175.0,
            sector="Technology",
        )
        assert pos.symbol == "AAPL"
        assert pos.shares == 100
        assert pos.average_cost == 150.0
        assert pos.current_price == 175.0
        assert pos.sector == "Technology"

    def test_position_defaults(self):
        pos = Position(
            symbol="AAPL",
            shares=100,
            average_cost=150.0,
        )
        assert pos.current_price == 0.0
        assert pos.sector == "Unknown"


class TestHoldings:
    """Tests for Holdings dataclass."""

    def test_holdings_empty(self):
        holdings = Holdings()
        assert len(holdings.positions) == 0
        assert holdings.cash == 0.0

    def test_holdings_with_positions(self):
        positions = [
            Position("AAPL", 100, 150.0, 175.0),
            Position("GOOGL", 50, 2800.0, 2900.0),
        ]
        holdings = Holdings(positions=positions, cash=10000.0)
        assert len(holdings.positions) == 2
        assert holdings.cash == 10000.0


class TestVaRCalculation:
    """Tests for VaR calculation."""

    def test_historical_var(self):
        # Create sample returns data
        np.random.seed(42)
        returns = pd.DataFrame({
            "AAPL": np.random.normal(0.001, 0.02, 252),
            "GOOGL": np.random.normal(0.0015, 0.025, 252),
        })
        weights = np.array([0.6, 0.4])
        portfolio_value = 100000.0

        result = calculate_portfolio_var(
            returns, weights, portfolio_value, method="historical"
        )

        assert isinstance(result, VaRResult)
        assert result.var_95 > 0
        assert result.var_99 > result.var_95
        assert result.method == "historical"

    def test_parametric_var(self):
        np.random.seed(42)
        returns = pd.DataFrame({
            "AAPL": np.random.normal(0.001, 0.02, 252),
        })
        weights = np.array([1.0])
        portfolio_value = 100000.0

        result = calculate_portfolio_var(
            returns, weights, portfolio_value, method="parametric"
        )

        assert isinstance(result, VaRResult)
        assert result.method == "parametric"


class TestBetaCalculation:
    """Tests for beta calculation."""

    def test_beta_calculation(self):
        np.random.seed(42)
        # Create correlated returns
        benchmark_returns = pd.Series(np.random.normal(0.001, 0.01, 100))
        portfolio_returns = pd.Series(
            1.2 * benchmark_returns + np.random.normal(0, 0.005, 100)
        )

        result = calculate_beta(
            portfolio_returns,
            benchmark_returns,
        )

        assert isinstance(result, BetaResult)
        # Beta should be close to 1.2 given our simulation
        assert result.beta == pytest.approx(1.2, abs=0.3)


class TestRatioCalculations:
    """Tests for Sharpe and Sortino ratios."""

    def test_sharpe_ratio(self):
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)
        assert isinstance(sharpe, float)

    def test_sortino_ratio(self):
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        sortino = calculate_sortino_ratio(returns, risk_free_rate=0.02)
        assert isinstance(sortino, float)


class TestPortfolioAnalyzer:
    """Tests for PortfolioAnalyzer."""

    def test_analyzer_creation(self):
        analyzer = PortfolioAnalyzer()
        assert analyzer is not None
        assert analyzer.health_check() is True
